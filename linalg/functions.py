import cupy as cp
import numpy as np

def euclidean_dist(a: np.ndarray | cp.ndarray,
                   b: np.ndarray | cp.ndarray,
                   device: str="cpu") -> np.ndarray | cp.ndarray:
    '''
    Calcualte the Euclidean distance between all points a and b.
    :param a: First data points.
    :param b: Second data points.
    :param device: CPU or GPU device.
    :return: Euclidean distance between all points a and b.
    '''
    if device == "cpu":
        na = a
        nb = b

        if isinstance(na, cp.ndarray):
            na = cp.asnumpy(na)

        if isinstance(nb, cp.ndarray):
            nb = cp.asnumpy(nb)

        return np.sqrt(np.sum((na - nb)**2, axis=1))
    else:
        ca = a
        cb = b

        if isinstance(ca, np.ndarray):
            ca = cp.asarray(ca)

        if isinstance(cb, np.ndarray):
            cb = cp.asarray(cb)

        return cp.sqrt(cp.sum((ca - cb)**2, axis=1))