from cupyx.scipy.special import softmax as softmax_gpu
from scipy.special import softmax

import cupy as cp
import numpy as np

class Activation:
    @staticmethod
    def leaky_relu(z: np.ndarray, hyper_param: dict=None) -> np.ndarray:
        '''
        Compute the leaky rectified linear unit activation function.
        :param z: Input value(s).
        :param hyper_param: Hyperparameters. Uses leaky_relu_alpha.
        :return: Leaky ReLU activation.
        '''
        h = 0.0

        if hyper_param is not None and "leaky_relu_alpha" in hyper_param.keys():
            h = hyper_param["leaky_relu_alpha"]

        return np.maximum(0, z) + h

    @staticmethod
    def leaky_relu_der(z: np.ndarray, a: np.ndarray, hyper_param: dict=None) -> np.ndarray:
        '''
        Compute the derivative of the leaky rectified linear unit function.
        :param z: Input value(s).
        :param a: Activation value(s).
        :param hyper_param: Hyperparameters. Uses leaky_relu_alpha.
        :return: Derivative of the leaky rectified linear unit function.
        '''
        h = 0.0

        if hyper_param is not None and "leaky_relu_alpha" in hyper_param.keys():
            h = hyper_param["leaky_relu_alpha"]

        dz = np.ones_like(z)
        dz[z < 0] = h
        return dz

    @staticmethod
    def leaky_relu_gpu(z: cp.ndarray, hyper_param: dict=None) -> cp.ndarray:
        '''
        Compute the leaky rectified linear unit activation function on the GPU.
        :param z: Input value(s).
        :param hyper_param: Hyperparameters. Uses leaky_relu_alpha.
        :return: Leaky ReLU activation.
        '''
        h = 0.0

        if hyper_param is not None and "leaky_relu_alpha" in hyper_param.keys():
            h = hyper_param["leaky_relu_alpha"]

        return cp.maximum(0, z) + h

    @staticmethod
    def leaky_relu_der_gpu(z: cp.ndarray, a: cp.ndarray, hyper_param: dict=None) -> cp.ndarray:
        '''
        Compute the derivative of the leaky rectified linear unit function on the GPU.
        :param z: Input value(s).
        :param a: Activation value(s).
        :param hyper_param: Hyperparameters. Uses leaky_relu_alpha.
        :return: Derivative of the leaky rectified linear unit function.
        '''
        h = 0.0

        if hyper_param is not None and "leaky_relu_alpha" in hyper_param.keys():
            h = hyper_param["leaky_relu_alpha"]

        dz = cp.ones_like(z)
        dz[z < 0] = h
        return dz

    @staticmethod
    def linear(z: np.ndarray, hyper_param: dict=None) -> np.ndarray:
        '''
        Compute the linear activation function.
        :param z: Input value(s).
        :param hyper_param: Hyperparameters.
        :return: Linear activation function.
        '''
        return z

    @staticmethod
    def linear_der(z: np.ndarray, a: np.ndarray, hyper_param: dict=None) -> np.ndarray:
        '''
        Compute the derivative of the linear function.
        :param z: Input value(s).
        :param a: Activation value(s).
        :param hyper_param: Hyperparameters.
        :return: Derivative of the linear function.
        '''
        return np.ones_like(z)

    @staticmethod
    def linear_gpu(z: cp.ndarray, hyper_param: dict=None) -> cp.ndarray:
        '''
        Compute the linear activation function on the GPU.
        :param z: Input value(s).
        :param hyper_param: Hyperparameters.
        :return: Linear activation function.
        '''
        return z

    @staticmethod
    def linear_der_gpu(z: cp.ndarray, a: cp.ndarray, hyper_param: dict=None) -> cp.ndarray:
        '''
        Compute the derivative of the linear function on the GPU.
        :param z: Input value(s).
        :param a: Activation value(s).
        :param hyper_param: Hyperparameters.
        :return: Derivative of the linear function.
        '''
        return cp.ones_like(z)

    @staticmethod
    def relu(z: np.ndarray, hyper_param: dict=None) -> np.ndarray:
        '''
        Compute the rectified linear unit activation function.
        :param z: Input value(s).
        :param hyper_param: Hyperparameters.
        :return: ReLU activation.
        '''
        return np.maximum(0, z)

    @staticmethod
    def relu_der(z: np.ndarray, a: np.ndarray, hyper_param: dict=None) -> np.ndarray:
        '''
        Compute the derivative of the rectified linear unit function.
        :param z: Input value(s).
        :param a: Activation value(s).
        :param hyper_param: Hyperparameters.
        :return: Derivative of the rectified linear unit function.
        '''
        dz = np.ones_like(z)
        dz[z < 0] = 0
        return dz

    @staticmethod
    def relu_gpu(z: cp.ndarray, hyper_param: dict=None) -> cp.ndarray:
        '''
        Compute the rectified linear unit activation function on the GPU.
        :param z: Input value(s).
        :param hyper_param: Hyperparameters.
        :return: ReLU activation.
        '''
        return cp.maximum(0, z)

    @staticmethod
    def relu_der_gpu(z: cp.ndarray, a: cp.ndarray, hyper_param: dict=None) -> cp.ndarray:
        '''
        Compute the derivative of the rectified linear unit function on the GPU.
        :param z: Input value(s).
        :param a: Activation value(s).
        :param hyper_param: Hyperparameters.
        :return: Derivative of the rectified linear unit function.
        '''
        dz = cp.ones_like(z)
        dz[z < 0] = 0
        return dz

    @staticmethod
    def sigmoid(z: np.ndarray, hyper_param: dict=None) -> np.ndarray:
        '''
        Compute the sigmoid activation function.
        :param z: Input value(s).
        :param hyper_param: Hyperparameters.
        :return: Sigmoid activation.
        '''
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_der(z: np.ndarray, a: np.ndarray, hyper_param: dict=None) -> np.ndarray:
        '''
        Compute the derivative of the sigmoid function.
        :param z: Input value(s).
        :param a: Activation value(s).
        :param hyper_param: Hyperparameters.
        :return: Derivative of the sigmoid function.
        '''
        return a * (1 - a)

    @staticmethod
    def sigmoid_gpu(z: cp.ndarray, hyper_param: dict=None) -> cp.ndarray:
        '''
        Compute the sigmoid activation function on the GPU.
        :param z: Input value(s).
        :param hyper_param: Hyperparameters.
        :return: Sigmoid activation.
        '''
        return 1 / (1 + cp.exp(-z))

    @staticmethod
    def sigmoid_der_gpu(z: cp.ndarray, a: cp.ndarray, hyper_param: dict=None) -> cp.ndarray:
        '''
        Compute the derivative of the sigmoid function.
        :param z: Input value(s).
        :param a: Activation value(s).
        :param hyper_param: Hyperparameters.
        :return: Derivative of the sigmoid function.
        '''
        return a * (1 - a)

    @staticmethod
    def softmax(z: np.ndarray, hyper_param: dict=None) -> np.ndarray:
        '''
        Compute the softmax activation function.
        :param z: Input value(s).
        :param hyper_param: Hyperparameters.
        :return: Softmax activation.
        '''
        return softmax(z, axis=1)

    @staticmethod
    def softmax_der(z: np.ndarray, a: np.ndarray, hyper_param: dict=None) -> np.ndarray:
        '''
        Compute the derivative of the softmax function.
        :param z: Input value(s).
        :param a: Activation value(s).
        :param hyper_param: Hyperparameters.
        :return: Derivative of the softmax function.
        '''
        batch_size, num_classes = a.shape if a.ndim > 1 else (1, a.shape)
        jacobian_batch = np.zeros((batch_size, num_classes, num_classes))

        for i in range(batch_size):
            s = a[i, :].reshape(-1, 1)
            jacobian_batch[i, :, :] = np.diagflat(s) - np.dot(s, s.T)

        return jacobian_batch

    @staticmethod
    def softmax_gpu(z: cp.ndarray, hyper_param: dict=None) -> cp.ndarray:
        '''
        Compute the softmax activation function on the GPU.
        :param z: Input value(s).
        :param hyper_param: Hyperparameters.
        :return: Softmax activation.
        '''
        return softmax_gpu(z, axis=1)

    @staticmethod
    def softmax_der_gpu(z: cp.ndarray, a: cp.ndarray, hyper_param: dict=None) -> cp.ndarray:
        '''
        Compute the derivative of the softmax function on the GPU.
        :param z: Input value(s).
        :param a: Activation value(s).
        :param hyper_param: Hyperparameters.
        :return: Derivative of the softmax function.
        '''
        batch_size, num_classes = a.shape if a.ndim > 1 else (1, a.shape)
        jacobian_batch = np.zeros((batch_size, num_classes, num_classes))

        for i in range(batch_size):
            s = a[i, :].reshape(-1, 1)
            jacobian_batch[i, :, :] = np.diagflat(s) - np.dot(s, s.T)

        return jacobian_batch

    @staticmethod
    def tanh(z: np.ndarray, hyper_param: dict=None) -> np.ndarray:
        '''
        Compute the hyperbolic tangent activation function.
        :param z: Input value(s).
        :param hyper_param: Hyperparameters.
        :return: Hyperbolic tangent function.
        '''
        return np.tanh(z)

    @staticmethod
    def tanh_der(z: np.ndarray, a: np.ndarray, hyper_param: dict=None) -> np.ndarray:
        '''
        Compute the derivative of the hyperbolic tangent function.
        :param z: Input value(s).
        :param a: Activation value(s).
        :param hyper_param: Hyperparameters.
        :return: Derivative of the hyperbolic tangent function.
        '''
        return 1 - np.tanh(z)**2

    @staticmethod
    def tanh_gpu(z: cp.ndarray, hyper_param: dict=None) -> cp.ndarray:
        '''
        Compute the hyperbolic tangent activation function on the GPU.
        :param z: Input value(s).
        :param hyper_param: Hyperparameters.
        :return: Hyperbolic tangent function.
        '''
        return cp.tanh(z)

    @staticmethod
    def tanh_der_gpu(z: cp.ndarray, a: cp.ndarray, hyper_param: dict=None) -> cp.ndarray:
        '''
        Compute the derivative of the hyperbolic tangent function on the GPU.
        :param z: Input value(s).
        :param a: Activation value(s).
        :param hyper_param: Hyperparameters.
        :return: Derivative of the hyperbolic tangent function.
        '''
        return 1 - cp.tanh(z)**2