from pathlib import Path

from deeploy.util.types import QFormat

def emit_cps(id:int, layer: dict) -> str:
    lines = ""

    lines += f"const int LAYER_{id:d}_INPUT_WIDTH = {layer['configuration']['input_dimension']:d};\n"

    input_type = QFormat(layer["configuration"]["input_quantization"]).to_fixed()
    lines += f"typedef {input_type} layer_{id:d}_input_t;\n"

    distance_type = QFormat(layer["configuration"]["distance_quantization"]).to_fixed()
    lines += f"typedef {distance_type} layer_{id:d}_distance_t;\n"

    beta_type = QFormat(layer["configuration"]["beta_quantization"]).to_fixed()
    lines += f"typedef {beta_type} layer_{id:d}_beta_t;\n"

    return lines

def emit_gravnetconv(id:int, layer: dict) -> str:
    lines = ""

    lines += f"const int LAYER_{id:d}_COORDINATE_WIDTH = {layer['configuration']['coordinate_dimension']:d};\n"
    lines += f"const int LAYER_{id:d}_FEATURE_WIDTH = {layer['configuration']['feature_dimension']:d};\n"
    lines += f"const int LAYER_{id:d}_OUTPUT_WIDTH = {layer['configuration']['output_dimension']:d};\n"
    lines += f"const int LAYER_{id:d}_K = {layer['configuration']['k']:d};\n"

    coordinate_type = QFormat(layer["configuration"]["coordinate_quantization"]).to_fixed()
    lines += f"typedef {coordinate_type} layer_{id:d}_coordinate_t;\n"

    feature_type = QFormat(layer["configuration"]["feature_quantization"]).to_fixed()
    lines += f"typedef {feature_type} layer_{id:d}_feature_t;\n"

    distance_type = QFormat(layer["configuration"]["distance_quantization"]).to_fixed()
    lines += f"typedef {distance_type} layer_{id:d}_distance_t;\n"

    exponential_type = QFormat(layer["configuration"]["exponential_quantization"]).to_fixed()
    lines += f"typedef {exponential_type} layer_{id:d}_exponential_t;\n"

    accum_type = QFormat(layer["configuration"]["accum_quantization"]).to_fixed()
    lines += f"typedef {accum_type} layer_{id:d}_accum_t;\n"

    output_type = QFormat(layer["configuration"]["output_quantization"]).to_fixed()
    lines += f"typedef {output_type} layer_{id:d}_output_t;\n"

    return lines

def emit_dense(id:int, layer: dict) -> str:
    lines = ""

    lines += f"const int LAYER_{id:d}_INPUT_WIDTH = {layer['configuration']['input_dimension']:d};\n"

    lines += f"const int LAYER_{id:d}_OUTPUT_WIDTH = {layer['configuration']['output_dimension']:d};\n"

    input_type = QFormat(layer["configuration"]["input_quantization"]).to_fixed()
    lines += f"typedef {input_type} layer_{id:d}_input_t;\n"

    weight_type = QFormat(layer["configuration"]["weight_quantization"]).to_fixed()
    lines += f"typedef {weight_type} layer_{id:d}_weights_t;\n"

    bias_type = QFormat(layer["configuration"]["bias_quantization"]).to_fixed()
    lines += f"typedef {bias_type} layer_{id:d}_biases_t;\n"

    output_type = QFormat(layer["configuration"]["output_quantization"]).to_fixed()
    lines += f"typedef {output_type} layer_{id:d}_output_t;\n"

    accum_type = QFormat(layer["configuration"]["accum_quantization"]).to_fixed()
    lines += f"typedef {accum_type} layer_{id:d}_accum_t;\n"
    len = layer["configuration"]["input_dimension"] * layer["configuration"]["output_dimension"]
    weights = ",".join(f"{x:.32g}" for x in layer["parameters"]["weights"].flatten())
    lines += f"static layer_{id:d}_weights_t layer_{id:d}_weights[{len}] {{" + weights + f"}};\n"

    len = layer["configuration"]["output_dimension"]
    biases = ",".join(f"{x:.32g}" for x in layer["parameters"]["biases"].flatten())
    lines += f"static layer_{id:d}_biases_t layer_{id:d}_biases[{len}] {{" + biases + f"}};\n"
    
    return lines

def emit_concat(id:int, layer: dict) -> str:
    lines = ""

    for i, dim in  enumerate(layer["configuration"]["input_dimension"]):
        lines += f"const int LAYER_{id:d}_INPUT_{i:d}_WIDTH = {dim};\n"
    
    lines += f"const int LAYER_{id:d}_OUTPUT_WIDTH = {layer["configuration"]["output_dimension"]};\n"

    for i, quant_str in  enumerate(layer["configuration"]["input_quantization"]):
        input_type = QFormat(quant_str).to_fixed()
        lines += f"typedef {input_type} layer_{id:d}_input_{i:d}_t;\n"
    
    output_type = QFormat(layer["configuration"]["output_quantization"]).to_fixed()
    lines += f"typedef {output_type} layer_{id:d}_output_t;\n"

    return lines

def emit_multiply_and_scale(id:int, layer: dict) -> str:
    lines = ""

    return lines

def emit(cfg: dict,path:Path):

    lines = """#ifndef WEIGHTS_H_ \
                \n#define WEIGHTS_H_ \
                \n#include "global.h"\n\n"""

    for layer_idx, layer in cfg.items():
        if layer.get("partition") == "PL":
            if layer.get("layer_type") == "dense" or layer.get("layer_type") == "linear":
                lines += emit_dense(layer_idx, layer)
            elif layer.get("layer_type") == "concat":
                lines += emit_concat(layer_idx, layer)
            elif layer.get("layer_type") == "multiply_and_scale":
                lines += emit_multiply_and_scale(layer_idx, layer)
            elif layer.get("layer_type") == "gravnetconv":
                lines += emit_gravnetconv(layer_idx, layer)
            elif layer.get("layer_type") == "cps":
                lines += emit_cps(layer_idx, layer)
            else:
                raise NotImplementedError(f"Layer type {layer.get('layer_type')} not supported in PL backend.")
          
    lines += "\n#endif // WEIGHTS_H_\n"

    with open(path, "w") as file:
        file.write(lines)