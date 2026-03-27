import json
import numpy as np
from pathlib import Path

class ModelConfig:
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return {"__ndarray__": True, "data": obj.tolist(), "dtype": str(obj.dtype)}
            return super().default(obj)

    @staticmethod
    def _numpy_decoder(obj):
        if "__ndarray__" in obj:
            return np.array(obj["data"], dtype=obj["dtype"])
        return obj

    @classmethod
    def save(cls, layer_cfg: dict, path: Path):
        serializable = {str(k): v for k, v in layer_cfg.items()}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=4, cls=cls.NumpyEncoder)

    @staticmethod
    def load(path: Path) -> dict:
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f, object_hook=ModelConfig._numpy_decoder)
        return {int(k): v for k, v in raw.items()}

    @staticmethod
    def equal(cfg_a: dict, cfg_b: dict) -> bool:
        if cfg_a.keys() != cfg_b.keys():
            print("Mismatch: different layer keys")
            return False
        
        for idx in cfg_a:
            layer_a, layer_b = cfg_a[idx], cfg_b[idx]
            
            if layer_a.keys() != layer_b.keys():
                print(f"Layer {idx}: different field keys")
                return False
            
            for field in layer_a:
                if field == "parameters":
                    params_a, params_b = layer_a["parameters"], layer_b["parameters"]
                    if params_a.keys() != params_b.keys():
                        print(f"Layer {idx}: different parameter keys")
                        return False
                    for param in params_a:
                        if not np.array_equal(params_a[param], params_b[param]):
                            print(f"Layer {idx}, parameter '{param}': arrays not equal")
                            return False
                else:
                    if layer_a[field] != layer_b[field]:
                        print(f"Layer {idx}, field '{field}': {layer_a[field]} != {layer_b[field]}")
                        return False
        
        return True