import numpy as np
from deeploy.util.types import QFormat,Quantizer

def quantize_dense_layer(layer: dict) -> dict:
    if (layer.get("layer_type") == "dense" or layer["layer_type"] == "linear"):
        weights_type = QFormat(layer["configuration"]["weight_quantization"])
        weights_unquantized = layer["parameters"]["weights"]
        weights_quantizer = Quantizer(weights_type,clip=True)
        quantized_weights = weights_quantizer.quantize_as_float(weights_unquantized)
        layer["parameters"]["weights"] = quantized_weights

        bias_type = QFormat(layer["configuration"]["bias_quantization"])
        bias_unquantized = layer["parameters"]["biases"]
        bias_quantizer = Quantizer(bias_type,clip=True)
        quantized_bias = bias_quantizer.quantize_as_float(bias_unquantized)
        layer["parameters"]["biases"] = quantized_bias
    return layer
    