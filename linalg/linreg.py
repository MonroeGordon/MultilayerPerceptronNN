import cupy as cp
import numpy as np

def ols(x: np.ndarray | cp.ndarray,
        y: np.ndarray | cp.ndarray,
        device: str="cpu") -> float | np.ndarray | cp.ndarray:
    '''
    Calculate ordinary least squares (OLS) coefficients using the normal equation.
    :param x: Design matrix including the intercept term.
    :param y: Response variable vector.
    :param device: CPU or GPU device.
    :return: Coefficients of the regression model.
    '''
    if x.ndim != 2:
        raise ValueError("@ ols: parameter 'x' must be a matrix (2-dimensional ndarray).")

    if y.ndim != 1:
        raise ValueError("@ ols: parameter 'y' must be a vector (1-dimensional ndarray).")

    xt = x.T

    if device == "cpu":
        nx = x
        nxt = xt
        ny = y

        if isinstance(nx, cp.ndarray):
            nx = cp.asnumpy(nx)

        if isinstance(nxt, cp.ndarray):
            nxt = cp.asnumpy(nxt)

        if isinstance(ny, cp.ndarray):
            ny = cp.asnumpy(ny)

        beta_hat = np.linalg.inv(nxt @ nx) @ nxt @ ny
    else:
        cx = x
        cxt = xt
        cy = y

        if isinstance(cx, np.ndarray):
            cx = cp.asarray(cx)

        if isinstance(cxt, np.ndarray):
            cxt = cp.asarray(cxt)

        if isinstance(cy, np.ndarray):
            cy = cp.asarray(cy)

        beta_hat = cp.linalg.inv(cxt @ cx) @ cxt @ cy

    return beta_hat

def ols_predict(x: np.ndarray | cp.ndarray,
                beta: float | np.ndarray | cp.ndarray,
                device: str="cpu") -> np.ndarray | cp.ndarray:
    '''
    Make predictions using the ordinary least squares (OLS) coefficients.
    :param x: Design matrix including the intercept term.
    :param beta: Coefficients of the regression model.
    :param device: CPU or GPU device.
    :return: Predicted values.
    '''
    if x.ndim != 2:
        raise ValueError("@ ols_predict: parameter 'x' must be a matrix (2-dimensional ndarray).")

    return x @ beta