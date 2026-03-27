import numpy as np
from copy import deepcopy

def fuse_dense_op(layer_a: dict, layer_b: dict) -> dict | None:
    """
    Attempt to fuse two dense layers by concatenating their weight/bias arrays.
    Returns a new fused layer dict, or None if the layers are not compatible.
    
    Compatibility requires identical layer_type, partition, and all configuration
    parameters except output_dimension.
    """
    if layer_a["layer_type"] != "dense" and layer_a["layer_type"] != "linear":
        return None
    if layer_b["layer_type"] != "dense" and layer_b["layer_type"] != "linear":
        return None
    if layer_a["partition"] != layer_b["partition"]:
        return None
    
    cfg_a = layer_a["configuration"]
    cfg_b = layer_b["configuration"]

    ignored = {"output_dimension"}
    comparable_keys = set(cfg_a.keys()) | set(cfg_b.keys()) - ignored
    if any(cfg_a.get(k) != cfg_b.get(k) for k in comparable_keys):
        return None

    in_dim  = cfg_a["input_dimension"]
    out_a   = cfg_a["output_dimension"]
    out_b   = cfg_b["output_dimension"]

    def pad_weights(w: np.ndarray, in_dim: int, out_dim: int) -> np.ndarray:
        """Allocate (in_dim, out_dim) and copy w into the top-left block."""
        buf = np.zeros((in_dim, out_dim), dtype=w.dtype)
        buf[:w.shape[0], :w.shape[1]] = w
        return buf

    def pad_biases(b: np.ndarray, out_dim: int) -> np.ndarray:
        """Allocate (out_dim,) and copy b into the front."""
        buf = np.zeros((out_dim,), dtype=b.dtype)
        buf[:b.shape[0]] = b
        return buf

    w_a = pad_weights(layer_a["parameters"]["weights"], in_dim, out_a)
    w_b = pad_weights(layer_b["parameters"]["weights"], in_dim, out_b)
    b_a = pad_biases(layer_a["parameters"]["biases"], out_a)
    b_b = pad_biases(layer_b["parameters"]["biases"], out_b)

    fused = deepcopy(layer_a)
    fused["configuration"] = deepcopy(cfg_a)
    fused["configuration"]["output_dimension"] = out_a + out_b
    fused["parameters"] = {
        "weights": np.concatenate([w_a, w_b], axis=1),  # (in_dim, out_a + out_b)
        "biases":  np.concatenate([b_a, b_b],  axis=0),  # (out_a + out_b,)
    }
    return fused