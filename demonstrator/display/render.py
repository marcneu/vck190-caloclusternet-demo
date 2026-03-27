import argparse
import time
import numpy as np
import pyvista as pv
import cv2

from display.preprocessing import (
    load_crystal_data,
    load_offline_cluster_data,
    load_track_data,
    generate_helix_track,
)

from display.detector import Detector

# Plasma colormap: 5 key RGB control points, interpolated on demand
_PLASMA_KEYS = np.array([
    [0.050, 0.030, 0.527],  # 0.00
    [0.499, 0.012, 0.657],  # 0.25
    [0.799, 0.280, 0.470],  # 0.50
    [0.973, 0.585, 0.240],  # 0.75
    [0.940, 0.975, 0.131],  # 1.00
])
_PLASMA_T = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

def plasma(t):
    """Map scalar t in [0,1] to an RGB tuple via the plasma colormap."""
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(float(np.interp(t, _PLASMA_T, _PLASMA_KEYS[:, c])) for c in range(3))

def log_norm(x, vmin, vmax):
    """Log-normalize x into [0, 1]."""
    return float(np.clip((np.log(x) - np.log(vmin)) / (np.log(vmax) - np.log(vmin)), 0.0, 1.0))

# Color settings (dark theme)
BACKGROUND_COLOR = [0.08, 0.1, 0.13]
DETECTOR_COLOR = [0.7, 0.7, 0.7]
TRACK_COLOR = "yellow"
CLUSTER_COLOR = "lime"
CRYSTAL_COLOR = lambda e: plasma(log_norm(e, 0.015, 1.0))
CLUSTER_RADIUS = lambda e: 10 * log_norm(e, 0.01, 8.0)


def load_scene(event_index, input_file, detector_mesh, crystals_mesh):
    """
    Load all data and build PyVista meshes for one event.
    Returns (crystal_meshes_mb, detector_mesh, scene_meshes) where
    scene_meshes is a list of (mesh, kwargs) pairs ready to add to a plotter.
    """

    # Event data
    crystal_ids_all, crystal_energies_all = load_crystal_data(input_file)
    offline_clst_energies_all, offline_clst_positions_all = load_offline_cluster_data(input_file)
    track_data_raw = load_track_data(input_file)

    num_events = len(crystal_ids_all)
    event_index = event_index % num_events
    print(f"Loading event {event_index} / {num_events - 1}")

    crystal_ids = crystal_ids_all[event_index]
    crystal_energies = crystal_energies_all[event_index]
    offline_clst_energies = offline_clst_energies_all[event_index]
    offline_clst_positions = offline_clst_positions_all[event_index]

    # Collect (mesh, add_mesh kwargs) for everything in the scene
    actors = []

    # Detector geometry
    actors.append((detector_mesh, dict(color=DETECTOR_COLOR, opacity=0.01, show_edges=False)))

    # Active crystals
    for idx, crystal_id in enumerate(crystal_ids):
        mesh = crystals_mesh.get(int(crystal_id))
        if mesh is None:
            continue
        color = CRYSTAL_COLOR(crystal_energies[idx])
        actors.append((mesh, dict(color=color, opacity=1.0, show_edges=False)))

    # Offline cluster spheres
    if len(offline_clst_positions) > 0:
        spheres = [
            pv.Sphere(center=pos, radius=CLUSTER_RADIUS(eng))
            for pos, eng in zip(offline_clst_positions, offline_clst_energies)
            if eng > 0
        ]
        if spheres:
            actors.append((pv.merge(spheres), dict(color=CLUSTER_COLOR, opacity=1.0)))

    # Helix tracks
    if event_index < len(track_data_raw) and len(track_data_raw[event_index]) > 0:
        track_params_raw = track_data_raw[event_index]
        track_params = np.transpose(track_params_raw[track_params_raw.any(axis=1)])
        if track_params.size > 0:
            track_points = generate_helix_track(track_params).swapaxes(0, 1).swapaxes(1, 2)
            for tx, ty, tz in track_points:
                mask = ~np.isnan(tx)
                if mask.sum() > 1:
                    pts = np.column_stack([tx[mask], ty[mask], tz[mask]])
                    tube = pv.PolyDataFilters.tube(pv.MultipleLines(pts), radius=1)
                    actors.append((tube, dict(color=TRACK_COLOR, opacity=1.0)))

    return actors


def build_plotter(actors, image_size):
    """Create an offscreen plotter with all scene actors added."""
    plotter = pv.Plotter(off_screen=True, window_size=list(image_size))
    plotter.background_color = BACKGROUND_COLOR
    for mesh, kwargs in actors:
        plotter.add_mesh(mesh, **kwargs)
    plotter.view_vector((1, 0, 0), viewup=(0, 1, 0))
    plotter.reset_camera()
    return plotter


def set_camera_azimuth(plotter, azimuth_deg):
    """
    Position the camera at the given azimuth angle (degrees) around the Y axis.
    Azimuth=0 looks along +X, azimuth=90 looks along +Z.
    """
    focal = np.array(plotter.camera.focal_point)
    pos = np.array(plotter.camera.position)
    distance = np.linalg.norm(pos - focal)

    angle = np.radians(azimuth_deg)
    new_pos = focal + distance * np.array([np.cos(angle), 0, -np.sin(angle)])
    plotter.camera.position = new_pos.tolist()
    plotter.camera.focal_point = focal.tolist()
    plotter.camera.up = (0, 1, 0)


def render_sweep(
    event_index: int,
    angles: list,
    input_file: str,
    output_dir: str = ".",
    image_size: tuple = (1920, 1080),
):
    """
    Render one event at multiple azimuth angles, reusing the same plotter.
    Prints per-image and total timing.
    """
    pv.OFF_SCREEN = True

    t_load_start = time.perf_counter()
    detector = Detector("/tmp/detector_mesh.vtmb")
    actors = load_scene(event_index, input_file, detector.mesh, detector.crystals)
    plotter = build_plotter(actors, image_size)
    t_load_end = time.perf_counter()
    print(f"Scene load + plotter build: {t_load_end - t_load_start:.3f}s\n")

    timings = []
    for azimuth in angles:
        t0 = time.perf_counter()

        set_camera_azimuth(plotter, azimuth)
        plotter.render()
        img_rgba = plotter.screenshot(filename=None, return_img=True)

        t1 = time.perf_counter()

        path = f"{output_dir}/event_{event_index:04d}_az{azimuth:03.0f}.bmp"
        cv2.imwrite(path, cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR))

        elapsed = t1 - t0
        timings.append(elapsed)
        print(f"  az={azimuth:6.1f}°  ->  {path}  ({elapsed:.3f}s)")

    plotter.close()

    total = sum(timings)
    print(f"\n{len(angles)} images in {total:.3f}s  (avg {total/len(angles):.3f}s/image)")


def render_frame(
    detector: "Detector",
    event_index: int,
    input_file: str,
    camera_azimuth: float,
    image_size: tuple = (1920, 1080),
) -> np.ndarray:
    """
    Render a single event and return a BGR numpy array (OpenCV-compatible).
    The caller is responsible for constructing and reusing the Detector instance.
    """
    actors = load_scene(event_index, input_file, detector.mesh, detector.crystals)
    plotter = build_plotter(actors, image_size)
    set_camera_azimuth(plotter, camera_azimuth)
    plotter.render()
    img_rgba = plotter.screenshot(filename=None, return_img=True)
    plotter.close()
    _, bmp = cv2.imencode(".bmp", cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR))
    return bmp.tobytes()


def render_single(
    event_index: int,
    camera_azimuth: float,
    input_file: str,
    output_path: str,
    image_size: tuple = (1920, 1080),
):
    """Render a single event at one azimuth angle."""
    pv.OFF_SCREEN = True
    detector = Detector("/tmp/detector_mesh.vtmb")
    actors = load_scene(event_index, input_file, detector.mesh, detector.crystals)
    plotter = build_plotter(actors, image_size)
    set_camera_azimuth(plotter, camera_azimuth)
    plotter.render()

    t0 = time.perf_counter()
    img_rgba = plotter.screenshot(filename=None, return_img=True)
    t1 = time.perf_counter()

    cv2.imwrite(output_path, cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR))
    plotter.close()
    print(f"Saved: {output_path}  ({t1 - t0:.3f}s)")
