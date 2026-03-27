import numpy as np
from pathlib import Path

"""
Calulcate plio chunk size based on plio width and datatypes
"""
def plio_chunk_size(pliowidth: int, bitwidth: int):
    if pliowidth not in [32, 64, 128]:
        raise ValueError(f"pliowidth must be in [32, 64, 128], got {pliowidth}")
    
    if bitwidth not in [4, 8, 16, 32]:
        raise ValueError(f"bitwidth must be in [4, 8, 16, 32], got {bitwidth}")
    
    return pliowidth // bitwidth

"""
Store data in chunks to a text file for use with AMD Vitis AIE Simulator.
"""
def store(data: np.array, path: Path, pliowidth :int, bitwidth: int):
    chunk_size = plio_chunk_size(pliowidth,bitwidth)
    chunks = data.flatten().reshape(-1, chunk_size)
    np.savetxt(path, chunks, fmt='%s', delimiter=' ')

"""
Load data from AIE simulator text file into a 2D numpy array.
"""
def load(path: Path, pliowidth :int, bitwidth: int) -> np.array:
    chunk_size = plio_chunk_size(pliowidth,bitwidth)
    data = np.genfromtxt(path, comments='T', dtype=np.int64)
    reshaped = data.flatten().reshape((-1,chunk_size))
    return reshaped