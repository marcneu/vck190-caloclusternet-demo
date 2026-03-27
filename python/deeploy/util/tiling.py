#!/usr/bin/env python3
import numpy as np

"""
Tile a matrix into blocks of size (m x n).
If required, the matrix is first padded to a size consisting of (m x n) tiles.
"""
def pad_and_tile(matrix: np.array, m: int, n: int,order: str = 'C') -> np.array:
  # Calculate padding needed
  rows, cols = matrix.shape
  pad_rows = (m - rows % m) % m  # Extra rows needed
  pad_cols = (n - cols % n) % n  # Extra cols needed
  if pad_rows > 0 or pad_cols > 0:
        matrix = np.pad(matrix, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)
  rows, cols = matrix.shape
  blocks = []
  if order == 'F':  # Column-major
    for c in range(0, cols, n):
      for r in range(0, rows, m):
        block = matrix[r:r+m, c:c+n]
        blocks.append(block)
  else:  # Row-major - default
    for r in range(0, rows, m):
      for c in range(0, cols, n):
        block = matrix[r:r+m, c:c+n]
        blocks.append(block)
  return np.array(blocks)

"""
If required matrix is padded first to match m columns.
Repeat each row of the matrix n times after reshaping it to have m columns.
"""
def pad_and_repeat_and_tile(matrix: np.array, m: int, n: int) -> np.array:
  len = matrix.shape[0]
  pad_len = (m - len % m) % m 

  if pad_len > 0:
        matrix = np.pad(matrix, (0, pad_len), mode='constant', constant_values=0)

  reshaped = np.array(matrix).reshape(-1, m)
  repeated = np.repeat(reshaped, n, axis=0)
  return repeated.flatten()

def detile_from_HxW_to_MxN(matrix: np.array,tile_h: int, tile_w:int, m: int, n: int) -> np.array:
    tiles_vertical = m // tile_h
    tiles_horizontal = n // tile_w
    tiled = matrix.reshape(-1,tiles_vertical, tiles_horizontal, tile_h, tile_w)
    tiled = tiled.transpose(0, 1, 3, 2, 4)
    return tiled.reshape(-1, m, n)
