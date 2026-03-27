from dataclasses import dataclass
from typing import Union
import numpy as np

@dataclass
class QFormat:
    """
    Represents a Q-format fixed-point number format.
    """
    signed: bool
    bitwidth: int
    integer: int
    fraction: int
    
    def __init__(self, q_string_or_signed: Union[str, bool], 
                 bitwidth: int = None, integer: int = None, fraction: int = None):
        """
        Create QFormat from string or from individual components.
        
        Usage:
            QFormat("Q0.7")  # from string
            QFormat(True, 8, 0, 7)  # from components
        """
        if isinstance(q_string_or_signed, str):
            # Parse from string
            q_string = q_string_or_signed
            if q_string.startswith("UQ"):
                self.signed = False
                parts = q_string[2:].split(".")
            elif q_string.startswith("Q"):
                self.signed = True
                parts = q_string[1:].split(".")
            else:
                raise ValueError(f"Invalid Q-format string: {q_string}")
            
            self.integer = int(parts[0])
            self.fraction = int(parts[1])
            
            if self.signed:
                self.bitwidth = 1 + self.integer + self.fraction
            else:
                self.bitwidth = self.integer + self.fraction
        else:
            # Direct construction from components
            self.signed = q_string_or_signed
            self.bitwidth = bitwidth
            self.integer = integer
            self.fraction = fraction
    
    def __str__(self):
        """Return Q-format string representation"""
        prefix = "Q" if self.signed else "UQ"
        return f"{prefix}{self.integer}.{self.fraction}"

    def to_c(self):
        """
        Return the next best fitting C integer type.
        """
        if self.bitwidth <= 8:
            bits = 8
        elif self.bitwidth <= 16:
            bits = 16
        elif self.bitwidth <= 32:
            bits = 32
        elif self.bitwidth <= 64:
            bits = 64
        else:
            raise ValueError(f"Bitwidth {self.bitwidth} exceeds standard C types (max 32)")
        
        sign_prefix = "int" if self.signed else "uint"
        return f"{sign_prefix}{bits}_t"
    
    def to_aie(self):
        """
        Return the next best fitting AIE integer type.
        """
        if self.bitwidth <= 8:
            bits = 8
        elif self.bitwidth <= 16:
            bits = 16
        elif self.bitwidth <= 32:
            bits = 32
        elif self.bitwidth <= 64:
            bits = 64
        else:
            raise ValueError(f"Bitwidth {self.bitwidth} exceeds standard C types (max 32)")
        
        sign_prefix = "int" if self.signed else "uint"
        return f"{sign_prefix}{bits}"

    def to_fixed(self):
        """
        Return Vitis HLS ap_fixed/ap_ufixed type string.
        Format: ap_fixed<W, I> or ap_ufixed<W, I>
        where W = total bitwidth, I = integer bits (including sign bit for signed)
        """
        type_name = "ap_fixed" if self.signed else "ap_ufixed"
        integer = self.integer + 1 if self.signed else self.integer
        return f"{type_name}<{self.bitwidth}, {integer}>"
    
class Quantizer:

    def __init__(self, qformat: QFormat,clip: bool):
        self.qformat = qformat
        self.clip = clip
    
    def quantize_as_float(self, value: np.array) -> np.array:
        """
        Quantize a floating-point numpy array to the Q-format and return as float.
        """
        quantized = self.quantize_as_int(value)

        scale = 2 ** self.qformat.fraction
        requantized = quantized / scale

        return requantized
    
    def quantize_as_int(self, value: np.array) -> np.array:
        """
        Quantize a floating-point numpy array to the Q-format and return as integer.
        """
        scale = 2 ** self.qformat.fraction
        quantized = np.floor(value * scale)

        if self.clip:
            bitwidth = self.qformat.bitwidth-1 if self.qformat.signed else self.qformat.bitwidth
            clip_max = 2 ** bitwidth -1 
            clip_min = -clip_max
            quantized = np.clip(quantized,clip_min,clip_max)

        return quantized.astype(np.int64)
    
    def requantize(self,value: np.array) -> np.array:
        requantized = value.astype(np.float64)
        scale = 2 ** self.qformat.fraction
        requantized = requantized / scale
        return requantized
     