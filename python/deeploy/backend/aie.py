


from pathlib import Path
import pandas as pd

from deeploy.util.types import QFormat,Quantizer
from deeploy.util.tiling import pad_and_tile,pad_and_repeat_and_tile

def lower_dense_op_to_dense_pe(layer: dict) -> dict:
    iq = QFormat(layer["configuration"]["input_quantization"])
    bq = QFormat(layer["configuration"]["bias_quantization"])
    wq = QFormat(layer["configuration"]["weight_quantization"])
    oq = QFormat(layer["configuration"]["output_quantization"])

    m = layer["configuration"]["batch_size"]
    k = layer["configuration"]["input_dimension"]
    n = layer["configuration"]["output_dimension"]

    so = iq.fraction + wq.fraction - oq.fraction
    sb = iq.fraction + wq.fraction - bq.fraction

    optlvl = layer.get("optlvl", 0)
    
    result = {
        "pe_type": "dense",
        "partition": layer["partition"],
        "par": 1,
        "m": m,
        "k": k,
        "n": n,
        "so": so,
        "sb": sb,
        "input_type_cstr": iq.to_aie(),
        "weight_type_cstr": wq.to_aie(),
        "output_type_cstr": oq.to_aie(),
        "optlvl": optlvl
    }
    return result

def lower_concat_op_to_concat_pe(layer: dict) -> dict:
    iqs = [QFormat(q) for q in layer["configuration"]["input_quantization"]]
    oq = QFormat(layer["configuration"]["output_quantization"])
    #Find the largest fractional bits among inputs
    max_fraction = max([iq.fraction for iq in iqs])
    #Check if outout fraction is even larger. 
    #The rational is that the shift from input to accumulator is in left direction (up) and the shift from accumulator to output is in right direction (down).
    max_fraction = max (max_fraction, oq.fraction)
    #Compute input scales to align to max_fraction

    input_len = [layer["configuration"]["batch_size"] * dim for dim in layer["configuration"]["input_dimension"]]
    input_parallelism = [32] * len(input_len) #Defined by AI architecture, fixed for now -> Is one tile for MMUL
    input_dims = [dim // 8 for dim in layer["configuration"]["input_dimension"]] # -> one row has 8 elements for MMUL
    input_scales = [max_fraction - iq.fraction for iq in iqs]
    input_type_cstrs = [iq.to_aie() for iq in iqs]

    output_len = layer["configuration"]["output_dimension"] * layer["configuration"]["batch_size"]
    output_scale = max_fraction - oq.fraction
    output_type_cstr = oq.to_aie()

    result = {
        "pe_type": "concat",
        "partition": layer["partition"],
        "par": 1,
        "order": layer["configuration"]["order"],
        "input_len": input_len,
        "input_parallelism":input_parallelism,
        "input_dimension": input_dims,
        "input_scale": input_scales,
        "input_type_cstrs": input_type_cstrs,
        "output_len" : output_len,
        "output_scale": output_scale,
        "output_type_cstr": output_type_cstr,
    }
    return result

def lower_op_to_pe(cfg: dict) -> dict:
    result = {}
    for layer_idx, layer in cfg.items():
        if layer.get("partition") == "AIE":
            if layer.get("layer_type") == "dense" or layer.get("layer_type") == "linear":
                result[layer_idx] = lower_dense_op_to_dense_pe(layer)
            elif(layer.get("layer_type") == "concat"):
                result[layer_idx] = lower_concat_op_to_concat_pe(layer)
    return result
    
def emit_dense_pe(id: int, cfg: dict):

    template = ""
    #template = f"dense_MxK_with_KxN<{cfg["input_type_cstr"]},uint8,int8, LAYER_2_M, LAYER_2_K, LAYER_2_N, LAYER_2_S>"

    params = ""

    params += f"typedef {cfg["input_type_cstr"]} LAYER_{id:d}_INPUT_T;\n"
    params += f"typedef {cfg["weight_type_cstr"]} LAYER_{id:d}_WEIGHT_T;\n"
    params += f"typedef {cfg["output_type_cstr"]} LAYER_{id:d}_OUTPUT_T;\n"
    params += f"const int LAYER_{id:d}_M = {cfg["m"]};\n"
    params += f"const int LAYER_{id:d}_K = {cfg["k"]};\n"
    params += f"const int LAYER_{id:d}_N = {cfg["n"]};\n"
    params += f"const int LAYER_{id:d}_SO = {cfg["so"]};\n"
    params += f"const int LAYER_{id:d}_SB = {cfg["sb"]};\n"
    params += f"const int LAYER_{id:d}_INPUT_LEN = {cfg["m"]*cfg["k"]};\n"
    params += f"const int LAYER_{id:d}_OUTPUT_LEN = {cfg["m"]*cfg["n"]};\n"
    params += f"const int LAYER_{id:d}_OPTLVL = {cfg["optlvl"]};\n"

    return template, params

def emit_concat_pe(id: int, cfg: dict):
    
    template = ""
    #template = f"concat_inner_srs<{','.join(cfg["input_type_cstrs"])},{cfg["output_type_cstr"]}, LAYER_6_INPUT_0_LEN, LAYER_6_INPUT_1_LEN,LAYER_6_PA,LAYER_6_PB,LAYER_6_SA,LAYER_6_SB,LAYER_6_SC>"

    params = ""

    for i, dim in  enumerate(cfg["input_len"]):
        params += f"const int LAYER_{id:d}_INPUT_{i:d}_LEN = {dim};\n"
    for i, dim in  enumerate(cfg["input_dimension"]):
        params += f"const int LAYER_{id:d}_INPUT_{i:d}_D = {dim};\n"
    for i, dim in  enumerate(cfg["input_parallelism"]):
        params += f"const int LAYER_{id:d}_INPUT_{i:d}_P = {dim};\n"
    for i, dim in  enumerate(cfg["input_scale"]):
        params += f"const int LAYER_{id:d}_INPUT_{i:d}_S = {dim};\n"
    params += f"const int LAYER_{id:d}_OUTPUT_LEN = {cfg["output_len"]};\n"
    params += f"const int LAYER_{id:d}_OUTPUT_S   = {cfg["output_scale"]};\n"

    return template, params

def emit_params(cfg: dict,path:Path):
    lines = """#ifndef PARAMS_H_ \
                \n#define PARAMS_H_\n\n"""

    for pe_idx, pe in cfg.items():
        if pe.get("pe_type") == "dense" or pe.get("pe_type") == "linear":
            _,new_lines = emit_dense_pe(pe_idx,pe)
            lines += new_lines
        elif pe.get("pe_type") == "concat":
            _,new_lines = emit_concat_pe(pe_idx,pe)
            lines += new_lines

    lines += "\n#endif // PARAMS_H_\n"

    with open(path, "w") as file:
        file.write(lines)

def emit_dense_weights(id: int,layer: dict) -> list[str]:
    lines = ""

    weights_type = QFormat(layer["configuration"]["weight_quantization"])
    weights_quantizer = Quantizer(weights_type,clip=True)
    weights_tiled = pad_and_tile(layer["parameters"]["weights"], 8, 4)
    weights_as_int = weights_quantizer.quantize_as_int(weights_tiled.flatten())
    weights_len = len(weights_as_int)
    weights_string = ",".join(f"{x:d}" for x in weights_as_int)
    lines += f"std::vector<{weights_type.to_c()}> layer_{id:d}_weights = {{" + weights_string + f"}};\n"

    biases_type = QFormat(layer["configuration"]["bias_quantization"])
    biases_quantizer = Quantizer(weights_type,clip=True)
    biases_tiled = pad_and_repeat_and_tile(layer["parameters"]["biases"], 4, 4)
    biases_as_int = biases_quantizer.quantize_as_int(biases_tiled.flatten())
    biases_len = len(biases_as_int)
    biases_string = ",".join(f"{x:d}" for x in biases_as_int)
    lines += f"std::vector<{biases_type.to_c()}> layer_{id:d}_biases = {{" + biases_string + f"}};\n"

    return lines

def emit_weights(cfg: dict,path:Path):

    lines = """#ifndef WEIGHTS_H_ \
                \n#define WEIGHTS_H_\n\n"""

    for layer_idx, layer in cfg.items():
        if layer.get("partition") == "AIE":
            if layer.get("layer_type") == "dense" or layer.get("layer_type") == "linear":
                lines += emit_dense_weights(layer_idx, layer)

    lines += "\n#endif // WEIGHTS_H_\n"

    with open(path, "w") as file:
        file.write(lines)

