"""
accelerator/wrapper.py

ctypes bindings for libraries/libaccelerator.so.

Array shapes:
    num_data:     (num_events,)                       int16
    cell_data:    (num_events, N, MODEL_INPUT_WIDTH)  int16
    out_features: (num_events, N, MODEL_OUTPUT_WIDTH) int16  -- returned by run()
    out_cps:      (num_events, N)                     int16  -- returned by run()

Constants (must match host.h):
    N                  = 128
    MODEL_INPUT_WIDTH  = 5
    MODEL_OUTPUT_WIDTH = 5
"""

import ctypes
from pathlib import Path

import numpy as np

# ── Model constants (must stay in sync with host.h) ──────────────────────────

N                  = 128
MODEL_INPUT_WIDTH  = 5
MODEL_OUTPUT_WIDTH = 5

# ── Load shared library ───────────────────────────────────────────────────────

_LIB_PATH = Path(__file__).parent / "libaccelerator.so"
_lib = ctypes.CDLL(str(_LIB_PATH))

_int16_p = ctypes.POINTER(ctypes.c_int16)

# ── C function signatures ─────────────────────────────────────────────────────

_lib.accelerator_create.argtypes = [
    ctypes.c_char_p,  # xclbin_path
    ctypes.c_int,     # num_events
    ctypes.c_int,     # verbose
    ctypes.c_int16,   # beta_threshold
    ctypes.c_int16,   # distance_threshold
]
_lib.accelerator_create.restype = ctypes.c_void_p

_lib.accelerator_setup.argtypes = [ctypes.c_void_p]
_lib.accelerator_setup.restype  = ctypes.c_int

_lib.accelerator_load.argtypes = [
    ctypes.c_void_p,  # handle
    _int16_p,         # num_data      [num_events]
    _int16_p,         # cell_data     [num_events * N * MODEL_INPUT_WIDTH]
    ctypes.c_int,     # num_events
    ctypes.c_int16,   # new_beta_threshold
    ctypes.c_int16,   # new_distance_threshold
]
_lib.accelerator_load.restype = ctypes.c_int

_lib.accelerator_run.argtypes = [
    ctypes.c_void_p,  # handle
    _int16_p,         # out_features  [num_events * N * MODEL_OUTPUT_WIDTH]
    _int16_p,         # out_cps       [num_events * N]
]
_lib.accelerator_run.restype = ctypes.c_int

_lib.accelerator_cleanup.argtypes = [ctypes.c_void_p]
_lib.accelerator_cleanup.restype  = None

# ── Helper ────────────────────────────────────────────────────────────────────

def _ptr(arr: np.ndarray):
    """Return a ctypes int16 pointer to a contiguous numpy array."""
    return arr.ctypes.data_as(_int16_p)

# ── Python wrapper class ──────────────────────────────────────────────────────

class Accelerator:
    """
    Manages the lifecycle of one AIE inference pipeline instance.

    Usage:
        acc = Accelerator("design.xclbin", num_events=100)
        acc.setup()
        acc.load(num_array, cell_array, beta_threshold=512, distance_threshold=256)
        features, cps = acc.run()
        acc.cleanup()

    Or as a context manager:
        with Accelerator("design.xclbin", num_events=100) as acc:
            acc.setup()
            acc.load(num_array, cell_array)
            features, cps = acc.run()
    """

    def __init__(
        self,
        xclbin_path: str,
        num_events: int,
        beta_threshold: int = 0,
        distance_threshold: int = 0,
        verbose: bool = False,
    ):
        self._num_events = num_events
        self._handle = _lib.accelerator_create(
            xclbin_path.encode(),
            num_events,
            int(verbose),
            ctypes.c_int16(beta_threshold),
            ctypes.c_int16(distance_threshold),
        )
        if self._handle is None:
            raise RuntimeError("accelerator_create returned NULL")

    def setup(self) -> None:
        rc = _lib.accelerator_setup(self._handle)
        if rc != 0:
            raise RuntimeError(f"accelerator_setup failed (rc={rc})")

    def load(
        self,
        num_data: np.ndarray,
        cell_data: np.ndarray,
        beta_threshold: int = 0,
        distance_threshold: int = 0,
    ) -> None:
        """
        Copy input data to device buffers.

        Args:
            num_data:           int16 array, shape (num_events,)
            cell_data:          int16 array, shape (num_events, N, MODEL_INPUT_WIDTH)
            beta_threshold:     int16 threshold passed to the g-kernel
            distance_threshold: int16 threshold passed to the g-kernel
        """
        num_data  = np.ascontiguousarray(num_data,  dtype=np.int16)
        cell_data = np.ascontiguousarray(cell_data, dtype=np.int16)

        self._num_events = len(num_data)

        rc = _lib.accelerator_load(
            self._handle,
            _ptr(num_data),
            _ptr(cell_data),
            self._num_events,
            ctypes.c_int16(beta_threshold),
            ctypes.c_int16(distance_threshold),
        )
        if rc != 0:
            raise RuntimeError(f"accelerator_load failed (rc={rc})")

    def run(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Run inference and return output arrays.

        Returns:
            features: int16 array, shape (num_events, N, MODEL_OUTPUT_WIDTH)
            cps:      int16 array, shape (num_events, N)
        """
        out_features = np.empty(self._num_events * N * MODEL_OUTPUT_WIDTH, dtype=np.int16)
        out_cps      = np.empty(self._num_events * N,                      dtype=np.int16)

        rc = _lib.accelerator_run(self._handle, _ptr(out_features), _ptr(out_cps))
        if rc != 0:
            raise RuntimeError(f"accelerator_run failed (rc={rc})")

        return (
            out_features.reshape(self._num_events, N, MODEL_OUTPUT_WIDTH),
            out_cps.reshape(self._num_events, N),
        )

    def cleanup(self) -> None:
        if self._handle is not None:
            _lib.accelerator_cleanup(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cleanup()
