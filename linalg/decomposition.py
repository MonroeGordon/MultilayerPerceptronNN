from scipy.linalg import lu as slu
from cupyx.scipy.linalg import lu as clu

import cupy as cp
import numpy as np

def lu(matrix: np.ndarray | cp.ndarray,
       permute_l: bool=False,
       overwrite_matrix: bool=False,
       check_finite: bool=False,
       device: str="cpu") -> tuple:
    '''
    Compute the LU decomposition of a square matrix.
    :param matrix: A square matrix (2-dimensional ndarray).
    :param permute_l: If set, permutes L by multiplying it with P
    :param overwrite_matrix: If set, overwrites matrix.
    :param check_finite: If set, checks that all matrix values are finite.
    :param device: CPU or GPU device.
    :return: If not permuting, Permuted (P), Lower (L), and upper (U) triangular matrices, otherwise Permuted Lower
    (PL), and upper (U) matrices.
    '''
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("@ lu: parameter 'matrix' must be a square matrix (2-dimensional ndarray).")

    if device == "cpu":
        nmtx = matrix

        if isinstance(nmtx, cp.ndarray):
            nmtx = cp.asnumpy(nmtx)

        return slu(nmtx, permute_l, overwrite_matrix, check_finite)
    else:
        cmtx = matrix

        if isinstance(cmtx, np.ndarray):
            cmtx = cp.asarray(cmtx)

        return clu(cmtx, permute_l, overwrite_matrix, check_finite)

def svd(matrix: np.ndarray | cp.ndarray,
        device: str="cpu") -> tuple[np.ndarray | cp.ndarray]:
    '''
    Perform Singular Value Decomposition (SVD) on a matrix.
    :param matrix: Input matrix (2-dimensional ndarray).
    :param device: CPU or GPU device.
    :return: U, S, V matrices.
    '''
    if device == "cpu":
        return np.linalg.svd(matrix)
    else:
        return cp.linalg.svd(matrix)