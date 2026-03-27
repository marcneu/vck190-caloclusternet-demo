import atexit
import os
import queue
import threading
import time
import logging
from typing import Any

import numpy as np
from flask import Flask, Response, render_template, request
import polars as pl
import pprint

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')

DEMO_MODE   = False  # Set to False to enable accelerator

if not DEMO_MODE:
    from accelerator import Accelerator
    from display.detector import Detector
    from display.render import render_frame

NUM_EVENTS  = 1024
INPUT_WIDTH = 5
N           = 128  # number of TCs per event

INPUT_FRACTIONAL_BITS  = 12  # number of fractional bits for fixed-point representation
OUTPUT_FRACTIONAL_BITS = 11  # number of fractional bits for accelerator output features

INPUT_FEATURES = ["tc_energy", "tc_time_cal", "x", "y", "z"]

PRESCALE_SCALE = {
    "tc_energy": 8,
    "tc_time_cal": 0.25,
    "x": 1.27,
    "y": 1.27,
    "z": 1.52
}
PRESCALE_OFFSET = {
    "tc_energy": 0.0,
    "tc_time_cal": 0.0,
    "x": 0.0,
    "y": 0.0,
    "z": 0.46
}

def apply_prescaling(x, feature_name):
    """Apply prescaling to a feature value based on its name."""
    # logger.debug(f"Applying prescaling to feature '{feature_name}'")
    if feature_name in PRESCALE_SCALE and feature_name in PRESCALE_OFFSET:
        scale = PRESCALE_SCALE[feature_name]
        offset = PRESCALE_OFFSET[feature_name]
        return (x - offset) / scale
    else:
        raise ValueError(f"Unknown feature name: {feature_name}")


app = Flask(__name__)

# ── Sentinel for clean worker shutdown ────────────────────────────────────────

_STOP = object()

# ── Shared state ──────────────────────────────────────────────────────────────

run_queue = queue.Queue(maxsize=1)

stop_event  = threading.Event()

result_lock   = threading.Lock()
latest_result = {"features": None, "cps": None}
latest_input  = {
    "num": None,
    "cell": None,
    "meta_data": None,
    "offline_targets": None,
    "physics_process": None,
    "physics_process_label": None,
    "beam_background_level": None,
    "beam_background_label": None,
}

frame_lock   = threading.Lock()
latest_frame = None

event_display_list = []  # list of dicts with data for each event, to be shown in the three.js display
current_event_display_index = 0  # which event is currently shown in the three.js display


def drain_run_queue() -> None:
    """Remove all pending items from the run queue without blocking."""
    while True:
        try:
            run_queue.get_nowait()
        except queue.Empty:
            break

# start with empty event
event_display_list.append({
    "uni_event_id": "No event data",
    "tc_ids": [],
    "energies": [],
    "offline_targets": [],
    "predicted_clusters": [],
    "physics_process": None,
    "physics_process_label": None,
    "beam_background_level": None,
    "beam_background_label": None,
})

# # for debugging the display:
# event_display_list.append({
#     "uni_event_id": "1_2_2",
#     "tc_ids": [
#         2800, 2801, 2802, 2803, 2804, 2805,
#         2898, 2899, 2900, 2901, 2902, 2903,
#         2996, 2997, 2998, 2999, 3000, 3001,
#         3094, 3095, 3096, 3097, 3098, 3099,
#         3192, 3193, 3194, 3195, 3196, 3197,
#         3290, 3291, 3292, 3293, 3294, 3295,
#         3388, 3389, 3390, 3391, 3392, 3393
#     ],
#     "energies": [
#         2.4, 3.1, 4.2, 5.3, 4.1, 2.8,
#         3.0, 4.5, 6.9, 8.2, 6.8, 4.3,
#         3.6, 5.9, 9.8, 12.4, 10.9, 6.2,
#         3.2, 5.1, 8.6, 11.7, 9.9, 5.8,
#         2.7, 4.3, 6.5, 8.1, 7.2, 4.7,
#         2.1, 3.4, 4.9, 5.6, 5.0, 3.3,
#         1.8, 2.6, 3.9, 4.4, 3.7, 2.4
#     ],
#     "offline_targets": [
#         {"objidx": 0, "train_E": 24.0, "train_x": -0.6, "train_y": 0.5, "train_z": -0.8},
#         {"objidx": 1, "train_E": 18.0, "train_x": -0.2, "train_y": -0.3, "train_z": 0.1},
#         {"objidx": 2, "train_E": 28.0, "train_x": 0.3, "train_y": 0.4, "train_z": 0.7},
#         {"objidx": 3, "train_E": 14.0, "train_x": 0.8, "train_y": -0.4, "train_z": -0.2}
#     ],
#     "predicted_clusters": [
#         {"p_energy": 23.5, "p_x": -0.55, "p_y": 0.48, "p_z": -0.76, "p_signal": 0.93},
#         {"p_energy": 16.9, "p_x": -0.18, "p_y": -0.28, "p_z": 0.09, "p_signal": 0.89},
#         {"p_energy": 27.2, "p_x": 0.34, "p_y": 0.41, "p_z": 0.72, "p_signal": 0.95},
#         {"p_energy": 12.6, "p_x": 0.78, "p_y": -0.36, "p_z": -0.18, "p_signal": 0.82},
#         {"p_energy": 9.4, "p_x": 0.05, "p_y": 0.12, "p_z": -0.05, "p_signal": 0.74}
#     ]
# })
# event_display_list.append({
#     "uni_event_id": "1_2_3",
#     "tc_ids": [3000, 3100, 3200],
#     "energies": [8.5, 9.0, 9.5],
#     "offline_targets": [
#         {"objidx": 0, "train_E": 10.0, "train_x": 1.0, "train_y": 1.0, "train_z": -1.0},
#         {"objidx": 1, "train_E": 20.0, "train_x": 1.0, "train_y": 1.0, "train_z": 1.0},
#     ],
#     "predicted_clusters": [
#         {"p_energy": 9.0, "p_x": 0.5, "p_y": 0.5, "p_z": -0.5, "p_signal": 0.9},
#         {"p_energy": 19.0, "p_x": 0.5, "p_y": 0.5, "p_z": 0.5, "p_signal": 0.8},
#     ]
# })
# event_display_list.append({
#     "uni_event_id": "1_2_4",
#     "tc_ids": [3003, 3103, 3203, 3004, 3104, 3204, 3005, 3105, 3205],
#     "energies": [8.5, 9.0, 9.5, 9.5, 9.0, 8.5, 9.0, 9.0, 9.0],
#     "offline_targets": [
#         {"objidx": 0, "train_E": 10.0, "train_x": 1.0, "train_y": 1.0, "train_z": -1.0},
#         {"objidx": 1, "train_E": 20.0, "train_x": 1.0, "train_y": 1.0, "train_z": 1.0},
#     ],
#     "predicted_clusters": [
#         {"p_energy": 8.5, "p_x": 0.5, "p_y": 0.5, "p_z": -0.5, "p_signal": 0.9},
#         {"p_energy": 19.0, "p_x": 0.5, "p_y": 0.5, "p_z": 0.5, "p_signal": 0.8},
#     ]
# })

def accelerator_worker():
    logger.info("Accelerator worker started.")
    with Accelerator("binaries/accelerator.xclbin", num_events=NUM_EVENTS) as acc:
        logger.debug("Accelerator created successfully.")
        logger.debug("Setting up accelerator...")
        acc.setup()
        logger.debug("Accelerator setup complete. Waiting for run requests...")
        while True:
            params = run_queue.get() 
            if params is _STOP:
                break
            
            logger.debug(f"Received run request.")
            logger.debug(f"Loading data into accelerator: num shape {params['num'].shape}, cell shape {params['cell'].shape}")
            logger.debug(f"  num dtype: {params['num'].dtype}, cell dtype: {params['cell'].dtype}")
            logger.debug(f"  num sample: {params['num'][:5]}, cell sample: {params['cell'][0, :5, :]}") 

            acc.load(params["num"], params["cell"],
                     beta_threshold=params.get("beta", int(0.01*(2**11))),
                     distance_threshold=params.get("dist", int(1.0985098509850986*(2**10))))
            logger.debug("Starting accelerator run...")
            features, cps = acc.run()
            
            logger.debug(f"Accelerator run complete. Features shape: {features.shape}, CPs shape: {cps.shape}")

            with result_lock:
                latest_result["features"]       = features
                latest_result["cps"]            = cps
                latest_input["num"]             = params["num"]
                latest_input["cell"]            = params["cell"]
                latest_input["meta_data"]       = params.get("meta_data", None)
                latest_input["offline_targets"] = params.get("offline_targets", None)
                latest_input["physics_process"] = params.get("physics_process", None)
                latest_input["physics_process_label"] = params.get("physics_process_label", None)
                latest_input["beam_background_level"] = params.get("beam_background_level", None)
                latest_input["beam_background_label"] = params.get("beam_background_label", None)
            
            fill_event_display_list()  # prepare data for three.js event display

    print("[accelerator] worker stopped.")


def visualization_worker(fps: int = 5):
    global latest_frame
    interval = 1.0 / fps

    detector = Detector("/tmp/detector_mesh.vtmb")

    while not stop_event.is_set():
        t0 = time.monotonic()

        with result_lock:
            features = latest_result["features"]
            cps      = latest_result["cps"]

        # if features is not None:
        with frame_lock:
            latest_frame = get_frame(detector, features, cps)

        elapsed = time.monotonic() - t0
        stop_event.wait(timeout=max(0, interval - elapsed))

    print("[visualizer] worker stopped.")

def get_frame(detector: Any, features: np.ndarray, cps: np.ndarray) -> bytes:
    return render_frame(detector, event_index=0, input_file="data/input_test_data.h5", camera_azimuth=45.0)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def trigger_run():
    params = request.get_json() or {}

    if DEMO_MODE:
        # In demo mode no accelerator worker consumes the queue; keep it empty.
        drain_run_queue()
        return {"status": "demo", "message": "Demo mode: accelerator run not queued"}

    if "parquet_path" in params:
        parquet_path = params["parquet_path"]
        meta_data, num_data, cell_data, offline_targets = load_parquet_file(parquet_path, num_events=NUM_EVENTS)
        params["num"] = num_data
        params["cell"] = cell_data
        params["meta_data"] = meta_data
        params["offline_targets"] = offline_targets

    try:
        run_queue.put_nowait(params)
    except queue.Full:
        return {"status": "busy"}, 503
    else:
        return {"status": "dispatched"}


@app.route("/stream")
def stream():
    def generate():
        while not stop_event.is_set():
            with frame_lock:
                frame = latest_frame
            if frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/bmp\r\n\r\n"
                    + frame +
                    b"\r\n"
                )
            time.sleep(1 / 30)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/threejs_event_data")
def threejs_event_data():
    """Provide the three js event display with information about the current event
    
    TODO: Not yet debugged or fully implemented"""
    
    logger.debug(f"Serving three.js event data for event index {current_event_display_index}")
    
    event = event_display_list[current_event_display_index] if current_event_display_index < len(event_display_list) else None
    
    logger.debug(f"Event data: {pprint.pformat(event)}")
    
    if event is None:
        return {"status": "no data"}, 404
    else:
        energies = np.zeros(8736)
        # the energies are zero padded to have a fixed length,
        # so we need to only use the first len(event["tc_ids"]) entries, which correspond to the actual TCs in the event:
        energies[event["tc_ids"]] = event["energies"][:len(event["tc_ids"])]
        event_id = event["uni_event_id"]
        return {
            "uni_event_id": event_id,
            "energies": energies.tolist(),
            "predicted_clusters": event.get("predicted_clusters", []),
            "offline_targets": event.get("offline_targets", []),
            "physics_process": event.get("physics_process", None),
            "physics_process_label": event.get("physics_process_label", None),
            "beam_background_level": event.get("beam_background_level", None),
            "beam_background_label": event.get("beam_background_label", None),
        }
        
@app.route("/threejs_event_data/next")
def threejs_event_data_next():
    global current_event_display_index
    if current_event_display_index < len(event_display_list) - 1:
        current_event_display_index += 1
        return {"status": "ok", "current_index": current_event_display_index}
    else:
        current_event_display_index = 0
        return {"status": "end of list", "current_index": current_event_display_index}
    
@app.route("/threejs_event_data/prev")
def threejs_event_data_prev():
    global current_event_display_index
    if current_event_display_index > 0:
        current_event_display_index -= 1
        return {"status": "ok", "current_index": current_event_display_index}
    else:
        current_event_display_index = len(event_display_list) - 1
        return {"status": "start of list", "current_index": current_event_display_index}
    
@app.route("/event_display")
def serve_event_display():
    """
    Render a simple 3D scene using three.js. The HTML file should be located in the 'templates' directory.
    """
    return render_template('event_display.html')

@app.route("/diagram")
def serve_diagram():
    """
    Diagram explaining the accelerator processing steps
    """
    return render_template('diagram.html')

# Routes to serve GLB files for three.js visualization
@app.route("/crystals.glb")
def serve_merged_glb():
    """Serve the merged GLB file containing all crystals.
    The file should be located in the 'crystal_meshes' directory."""
    glb_path = os.path.join('crystal_meshes', 'crystals.glb')
    if os.path.exists(glb_path):
        with open(glb_path, 'rb') as f:
            glb_data = f.read()
        return Response(glb_data, mimetype='model/gltf-binary')
    else:
        return Response("Merged GLB not found", status=404)


# ── Data loading and processing ────────────────────────────────────────────────────────────────

def load_parquet_file(
        file_path: str,
        num_events: int = NUM_EVENTS,
        time_units: float = 1000.0,
        space_units: float = 100.0
    ) -> tuple:
    """Load input data from a parquet file and prepare it for accelerator processing.
    
    The parquet file is expected to have the following columns:
        - uni_event_id: unique identifier for each event
        - tc_id: identifier for each trigger cell
        - tc_energy, tc_time_cal, x, y, z: features for each trigger cell
        - objidx, train_E, train_x, train_y, train_z: offline targets for each event
    
    Returns:
        - meta_data: list of tuples (uni_event_id, tc_ids) for each event (not zero-padded)
        - num_data: int16 array of shape (num_events,) with the number of TCs per event (zero-padded)
        - cell_data: int16 array of shape (num_events, N, INPUT_WIDTH) with TC features (zero-padded)
        - offline_targets: polars DataFrame with one row per offline cluster
    
    TODO: Add code for loading tracks
    """
    
    default_return = (
        [],
        np.zeros((num_events,),dtype=np.int16),
        np.zeros((num_events, N, INPUT_WIDTH), dtype=np.int16),
        None
    )
    
    logger.debug(f"Loading parquet file: {file_path}")
    
    try:
        df = pl.scan_parquet(file_path)
    except Exception as e:
        logger.error(f"[load_parquet_file] Error opening parquet file: {e}")
        return default_return
    
    try:
        inputs = df.select(
            pl.col("uni_event_id"),
            pl.col("tc_id"),
            apply_prescaling(pl.col("tc_energy"), "tc_energy"),
            apply_prescaling(pl.col("tc_time_cal")/time_units, "tc_time_cal"),
            apply_prescaling(pl.col("x")/space_units, "x"),
            apply_prescaling(pl.col("y")/space_units, "y"),
            apply_prescaling(pl.col("z")/space_units, "z")
        ).collect()
        logger.debug(f"Parquet file loaded successfully. Number of rows: {inputs.height}")
    except Exception as e:
        logger.error(f"[load_parquet_file] Error reading parquet file: {e}")
        return default_return
    
    num_data = np.zeros((num_events,), dtype=np.int16)
    cell_data = np.zeros((num_events, N, INPUT_WIDTH), dtype=np.int16)
    meta_data = []
    
    for i, (name, data) in enumerate(inputs.group_by("uni_event_id")):
        uni_event_id = name[0]
        event_cells = data.select("tc_energy", "tc_time_cal", "x", "y", "z").to_numpy().reshape(-1, INPUT_WIDTH)[:N]
        event_cells = np.trunc(event_cells*2**INPUT_FRACTIONAL_BITS).astype(np.int16)
        tc_ids = data.select("tc_id").to_numpy().flatten()[:N]
        num_data[i] = len(event_cells)
        cell_data[i, :len(event_cells), :] = event_cells
        meta_data.append((uni_event_id, tc_ids))

    try:
        offline_targets = df.select(
            pl.col("uni_event_id"),
            pl.col("objidx"),
            pl.col("train_E"),
            pl.col("train_x"),
            pl.col("train_y"),
            pl.col("train_z"),
        ).unique(
            subset=["uni_event_id", "objidx"],
            keep="first"
        ).collect()
    except Exception as e:
        logger.error(f"[load_parquet_file] Error reading offline targets from parquet file: {e}")
        offline_targets = None
    
    return meta_data, num_data, cell_data, offline_targets

def fill_event_display_list(max_length=100, clear_existing=True):
    """Fill the event display list with data from the last accelerator run,
    so that the three.js display can show it.
    
    TODO: This function is not yet debugged or fully implemented. 
    """

    logger.debug("Filling event display list with latest input data.")
    
    if clear_existing:
        event_display_list.clear()
        logger.debug("Cleared existing event display list.")
    
    with result_lock:
        meta_data = latest_input["meta_data"]
        offline_targets = latest_input["offline_targets"]
        physics_process = latest_input.get("physics_process")
        physics_process_label = latest_input.get("physics_process_label")
        beam_background_level = latest_input.get("beam_background_level")
        beam_background_label = latest_input.get("beam_background_label")
        # objidx 0 is a placeholder for "no cluster", so we filter it out here:
        if offline_targets is not None:
            offline_targets = offline_targets.filter(pl.col("objidx") != 0)
        energies = latest_input["cell"][..., 0] if latest_input["cell"] is not None else None
        energies = energies.astype(np.float32) / (2**INPUT_FRACTIONAL_BITS) if energies is not None else None
        features = latest_result["features"]
        cps = latest_result["cps"]

    if meta_data is None or offline_targets is None or energies is None:
        logger.debug("One or more required data components are missing.")
        return
    
    for event_idx, (uni_event_id, tc_ids) in enumerate(meta_data):
        if event_idx < max_length:
            masked_features = features[event_idx][:len(tc_ids)][cps[event_idx][:len(tc_ids)] == 1]
            # masked_features = features[event_idx][:len(tc_ids)]
            predicted_clusters = [
                {
                    "p_energy": f[0]/(2**OUTPUT_FRACTIONAL_BITS),
                    "p_x": f[1]/(2**OUTPUT_FRACTIONAL_BITS),
                    "p_y": f[2]/(2**OUTPUT_FRACTIONAL_BITS),
                    "p_z": f[3]/(2**OUTPUT_FRACTIONAL_BITS),
                    "p_signal": f[4]/(2**OUTPUT_FRACTIONAL_BITS)
                    # TODO: check feature order and scaling factor
                }
                for f in masked_features
            ]
            event_data = {
                "uni_event_id": uni_event_id,
                "tc_ids": tc_ids.tolist(),
                "offline_targets": offline_targets.filter(pl.col("uni_event_id") == uni_event_id).to_dicts(),
                "energies": energies[event_idx].tolist(),
                "predicted_clusters": predicted_clusters,
                "physics_process": physics_process,
                "physics_process_label": physics_process_label,
                "beam_background_level": beam_background_level,
                "beam_background_label": beam_background_label,
            }
            event_display_list.append(event_data)
            if event_idx == 0:
                logger.debug(f"Example event data: {event_data}")
            if event_idx % 100 == 0:
                logger.debug(f"Added event {event_idx} to display list.")
        else:
            break
    
    logger.debug(f"Event display list filled with {len(event_display_list)} events.")


# ── Graceful shutdown (registered once, called by Flask or Ctrl-C) ────────────

@atexit.register
def shutdown():
    print("[main] shutting down...")
    stop_event.set()

    if DEMO_MODE:
        # Demo mode has no worker thread consuming this queue.
        drain_run_queue()
    else:
        try:
            run_queue.put_nowait(_STOP)
        except queue.Full:
            drain_run_queue()
            try:
                run_queue.put_nowait(_STOP)
            except queue.Full:
                logger.warning("Could not enqueue stop sentinel; worker may already be stopping.")

    for t in _workers:
        t.join(timeout=5)
    print("[main] shutdown complete.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _workers = []
    if not DEMO_MODE:
        _workers.append(threading.Thread(target=accelerator_worker, daemon=True))
        # _workers.append(threading.Thread(target=visualization_worker, daemon=True))
    for t in _workers:
        t.start()

    app.run(host="0.0.0.0", port=8081, threaded=True, use_reloader=False)
