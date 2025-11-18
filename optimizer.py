import cupy as cp
import numpy as np

class Optimizer:
    @staticmethod
    def none(cycle: int,
             samples: int,
             prev_update: np.ndarray,
             learning_rate: float,
             delta: np.ndarray,
             w: np.ndarray,
             a: np.ndarray,
             d: np.ndarray,
             hyper_param: dict=None) -> tuple[np.ndarray, np.ndarray]:
        '''
        Compute weight updates without optimization.
        :param cycle: Training cycle.
        :param samples: Number of training samples.
        :param prev_update: The update at the previous time step.
        :param learning_rate: Learning rate.
        :param delta: Error value(s).
        :param w: Weights.
        :param a: Activations.
        :param d: Activation derivatives.
        :param hyper_param: Hyperparameters.
        :return: No optimization.
        '''
        update = (1 / samples) * learning_rate * np.dot(np.dot(w.T, delta) * d, a.T)

        return update, w - update

    @staticmethod
    def none_gpu(cycle: int,
                 samples: int,
                 prev_update: cp.ndarray,
                 learning_rate: float,
                 delta: cp.ndarray,
                 w: cp.ndarray,
                 a: cp.ndarray,
                 d: cp.ndarray,
                 hyper_param: dict=None) -> tuple[cp.ndarray, cp.ndarray]:
        '''
        Compute weight updates without optimization on the GPU.
        :param cycle: Training cycle.
        :param samples: Number of training samples.
        :param prev_update: The update at the previous time step.
        :param learning_rate: Learning rate.
        :param delta: Error value(s).
        :param w: Weights.
        :param a: Activations.
        :param d: Activation derivatives.
        :param hyper_param: Hyperparameters.
        :return: No optimization.
        '''
        update = (1 / samples) * learning_rate * cp.dot(cp.dot(w.T, delta) * d, a.T)

        return update, w - update

    @staticmethod
    def adadelta(cycle: int,
                 samples: int,
                 prev_update: np.ndarray,
                 learning_rate: float,
                 delta: np.ndarray,
                 w: np.ndarray,
                 a: np.ndarray,
                 d: np.ndarray,
                 hyper_param: dict=None) -> tuple[np.ndarray, np.ndarray]:
        '''
        Compute the AdaDelta optimization function.
        :param cycle: Training cycle.
        :param samples: Number of training samples.
        :param prev_update: The update at the previous time step.
        :param learning_rate: Learning rate.
        :param delta: Error value(s).
        :param w: Weights.
        :param a: Activations.
        :param d: Activation derivatives.
        :param hyper_param: Hyperparameters. Uses adadelta_decay.
        :return: AdaDelta optimization.
        '''
        decay = 0.99

        if hyper_param is not None:
            if "adadelta_decay" in hyper_param.keys():
                decay = hyper_param["adadelta_decay"]

        grad = np.dot(np.dot(w.T, delta) * d, a.T)
        x = decay * prev_update[0, :, :] + (1 - decay) * prev_update[0, :, :]**2
        g = decay * prev_update[1, :, :] + (1 - decay) * grad**2
        epsilon = 1e-8
        update = np.stack(x, g)

        return update, w - (1 / samples) * (np.sqrt(x) + epsilon) / (np.sqrt(g) + epsilon) * learning_rate

    @staticmethod
    def adadelta_gpu(cycle: int,
                     samples: int,
                     prev_update: np.ndarray,
                     learning_rate: float,
                     delta: np.ndarray,
                     w: np.ndarray,
                     a: np.ndarray,
                     d: np.ndarray,
                     hyper_param: dict=None) -> tuple[np.ndarray, np.ndarray]:
        '''
        Compute the AdaDelta optimization function on the GPU.
        :param cycle: Training cycle.
        :param samples: Number of training samples.
        :param prev_update: The update at the previous time step.
        :param learning_rate: Learning rate.
        :param delta: Error value(s).
        :param w: Weights.
        :param a: Activations.
        :param d: Activation derivatives.
        :param hyper_param: Hyperparameters. Uses adadelta_decay.
        :return: AdaDelta optimization.
        '''
        decay = 0.99

        if hyper_param is not None:
            if "adadelta_decay" in hyper_param.keys():
                decay = hyper_param["adadelta_decay"]

        grad = cp.dot(cp.dot(w.T, delta) * d, a.T)
        x = decay * prev_update[0, :, :] + (1 - decay) * prev_update[0, :, :]**2
        g = decay * prev_update[1, :, :] + (1 - decay) * grad**2
        epsilon = 1e-8
        update = cp.stack(x, g)

        return update, w - (1 / samples) * (cp.sqrt(x) + epsilon) / (cp.sqrt(g) + epsilon) * learning_rate

    @staticmethod
    def adagrad(cycle: int,
                samples: int,
                prev_update: np.ndarray,
                learning_rate: float,
                delta: np.ndarray,
                w: np.ndarray,
                a: np.ndarray,
                d: np.ndarray,
                hyper_param: dict=None) -> tuple[np.ndarray, np.ndarray]:
        '''
        Compute the adaptive gradient (AdaGrad) optimization function.
        :param cycle: Training cycle.
        :param samples: Number of training samples.
        :param prev_update: The update at the previous time step.
        :param learning_rate: Learning rate.
        :param delta: Error value(s).
        :param w: Weights.
        :param a: Activations.
        :param d: Activation derivatives.
        :param hyper_param: Hyperparameters.
        :return: Adaptive gradient (AdaGrad) optimization.
        '''
        grad = np.dot(np.dot(w.T, delta) * d, a.T)
        update = prev_update + grad**2
        epsilon = 1e-8

        return update, w - (1 / samples) * grad * learning_rate / np.sqrt(update + epsilon)

    @staticmethod
    def adagrad_gpu(cycle: int,
                    samples: int,
                    prev_update: cp.ndarray,
                    learning_rate: float,
                    delta: cp.ndarray,
                    w: cp.ndarray,
                    a: cp.ndarray,
                    d: cp.ndarray,
                    hyper_param: dict = None) -> tuple[cp.ndarray, cp.ndarray]:
        '''
        Compute the adaptive gradient (AdaGrad) optimization function.
        :param cycle: Training cycle.
        :param samples: Number of training samples.
        :param prev_update: The update at the previous time step.
        :param learning_rate: Learning rate.
        :param delta: Error value(s).
        :param w: Weights.
        :param a: Activations.
        :param d: Activation derivatives.
        :param hyper_param: Hyperparameters.
        :return: Adaptive gradient (AdaGrad) optimization.
        '''
        grad = cp.dot(cp.dot(w.T, delta) * d, a.T)
        update = prev_update + grad ** 2
        epsilon = 1e-8

        return update, w - (1 / samples) * grad * learning_rate / cp.sqrt(update + epsilon)

    @staticmethod
    def adam(cycle: int,
             samples: int,
             prev_update: np.ndarray,
             learning_rate: float,
             delta: np.ndarray,
             w: np.ndarray,
             a: np.ndarray,
             d: np.ndarray,
             hyper_param: dict=None) -> tuple[np.ndarray, np.ndarray]:
        '''
        Compute the adaptive moment estimation (Adam) optimization function.
        :param cycle: Training cycle.
        :param samples: Number of training samples.
        :param prev_update: The update at the previous time step.
        :param learning_rate: Learning rate.
        :param delta: Error value(s).
        :param w: Weights.
        :param a: Activations.
        :param d: Activation derivatives.
        :param hyper_param: Hyperparameters. Uses adam_beta_1 and adam_beta_2.
        :return: Adaptive moment estimation (Adam) optimization.
        '''
        beta1 = 0.9
        beta2 = 0.999

        if hyper_param is not None:
            if "adam_beta_1" in hyper_param.keys():
                beta1 = hyper_param["adam_beta_1"]
            if "adam_beta_2" in hyper_param.keys():
                beta2 = hyper_param["adam_beta_2"]

        grad = np.dot(np.dot(w.T, delta) * d, a.T)
        m = beta1 * prev_update[0, :, :] + (1 - beta1) * grad
        v = beta2 * prev_update[1, :, :] + (1 - beta2) * grad**2
        _m = m / (1 - beta1**cycle)
        _v = v / (1 - beta2**cycle)
        epsilon = 1e-8
        update = np.stack(m, v)

        return update, w - (1 / samples) * _m / (np.sqrt(_v) + epsilon) * learning_rate

    @staticmethod
    def adam_gpu(cycle: int,
                 samples: int,
                 prev_update: cp.ndarray,
                 learning_rate: float,
                 delta: cp.ndarray,
                 w: cp.ndarray,
                 a: cp.ndarray,
                 d: cp.ndarray,
                 hyper_param: dict=None) -> tuple[cp.ndarray, cp.ndarray]:
        '''
        Compute the adaptive moment estimation (Adam) optimization function on the GPU.
        :param cycle: Training cycle.
        :param samples: Number of training samples.
        :param prev_update: The update at the previous time step.
        :param learning_rate: Learning rate.
        :param delta: Error value(s).
        :param w: Weights.
        :param a: Activations.
        :param d: Activation derivatives.
        :param hyper_param: Hyperparameters. Uses adam_beta_1 and adam_beta_2.
        :return: Adaptive moment estimation (Adam) optimization.
        '''
        beta1 = 0.9
        beta2 = 0.999

        if hyper_param is not None:
            if "adam_beta_1" in hyper_param.keys():
                beta1 = hyper_param["adam_beta_1"]
            if "adam_beta_2" in hyper_param.keys():
                beta2 = hyper_param["adam_beta_2"]

        grad = cp.dot(cp.dot(w.T, delta) * d, a.T)
        m = beta1 * prev_update[0, :, :] + (1 - beta1) * grad
        v = beta2 * prev_update[1, :, :] + (1 - beta2) * grad**2
        _m = m / (1 - beta1**cycle)
        _v = v / (1 - beta2**cycle)
        epsilon = 1e-8
        update = cp.stack(m, v)

        return update, w - (1 / samples) * _m / (cp.sqrt(_v) + epsilon) * learning_rate

    @staticmethod
    def momentum(cycle: int,
                 samples: int,
                 prev_update: np.ndarray,
                 learning_rate: float,
                 delta: np.ndarray,
                 w: np.ndarray,
                 a: np.ndarray,
                 d: np.ndarray,
                 hyper_param: dict=None) -> tuple[np.ndarray, np.ndarray]:
        '''
        Compute the momentum optimization function.
        :param cycle: Training cycle.
        :param samples: Number of training samples.
        :param prev_update: The update at the previous time step.
        :param learning_rate: Learning rate.
        :param delta: Error value(s).
        :param w: Weights.
        :param a: Activations.
        :param d: Activation derivatives.
        :param hyper_param: Hyperparameters. Uses momentum_lambda.
        :return: Momentum optimization.
        '''
        lamda = 0.9

        if hyper_param is not None and "momentum_lambda" in hyper_param.keys():
            lamda = hyper_param["momentum_lambda"]

        update = (1 / samples) * lamda * prev_update + learning_rate * np.dot(np.dot(w.T, delta) * d, a.T)

        return update, w - update

    @staticmethod
    def momentum_gpu(cycle: int,
                     samples: int,
                     prev_update: cp.ndarray,
                     learning_rate: float,
                     delta: cp.ndarray,
                     w: cp.ndarray,
                     a: cp.ndarray,
                     d: cp.ndarray,
                     hyper_param: dict=None) -> tuple[cp.ndarray, cp.ndarray]:
        '''
        Compute the momentum optimization function on the GPU.
        :param cycle: Training cycle.
        :param samples: Number of training samples.
        :param prev_update: The update at the previous time step.
        :param learning_rate: Learning rate.
        :param delta: Error value(s).
        :param w: Weights.
        :param a: Activations.
        :param d: Activation derivatives.
        :param hyper_param: Hyperparameters. Uses momentum_lambda.
        :return: Momentum optimization.
        '''
        lamda = 0.9

        if hyper_param is not None and "momentum_lambda" in hyper_param.keys():
            lamda = hyper_param["momentum_lambda"]

        update = (1 / samples) * lamda * prev_update + learning_rate * cp.dot(cp.dot(w.T, delta) * d, a.T)

        return update, w - update

    @staticmethod
    def nesterov(cycle: int,
                 samples: int,
                 prev_update: np.ndarray,
                 learning_rate: float,
                 delta: np.ndarray,
                 w: np.ndarray,
                 a: np.ndarray,
                 d: np.ndarray,
                 hyper_param: dict=None) -> tuple[np.ndarray, np.ndarray]:
        '''
        Compute the Nesterov accelerated gradient (NAG) optimization function.
        :param cycle: Training cycle.
        :param samples: Number of training samples.
        :param prev_update: The update at the previous time step.
        :param learning_rate: Learning rate.
        :param delta: Error value(s).
        :param w: Weights.
        :param a: Activations.
        :param d: Activation derivatives.
        :param hyper_param: Hyperparameters. Uses nesterov_lambda.
        :return: Nesterov accelerated gradient (NAG) optimization.
        '''
        lamda = 0.9

        if hyper_param is not None and "nesterov_lambda" in hyper_param.keys():
            lamda = hyper_param["nesterov_lambda"]

        w_look_ahead = w - lamda * prev_update
        wla_grad = np.dot(np.dot(w_look_ahead.T, delta) * d, a.T)

        update = (1 / samples) * lamda * prev_update + learning_rate * wla_grad

        return update, w - update

    @staticmethod
    def nesterov_gpu(cycle: int,
                     samples: int,
                     prev_update: cp.ndarray,
                     learning_rate: float,
                     delta: cp.ndarray,
                     w: cp.ndarray,
                     a: cp.ndarray,
                     d: cp.ndarray,
                     hyper_param: dict = None) -> tuple[cp.ndarray, cp.ndarray]:
        '''
        Compute the Nesterov accelerated gradient (NAG) optimization function on the GPU.
        :param cycle: Training cycle.
        :param samples: Number of training samples.
        :param prev_update: The update at the previous time step.
        :param learning_rate: Learning rate.
        :param delta: Error value(s).
        :param w: Weights.
        :param a: Activations.
        :param d: Activation derivatives.
        :param hyper_param: Hyperparameters. Uses nesterov_lambda.
        :return: Nesterov accelerated gradient (NAG) optimization.
        '''
        lamda = 0.9

        if hyper_param is not None and "nesterov_lambda" in hyper_param.keys():
            lamda = hyper_param["nesterov_lambda"]

        w_look_ahead = w - lamda * prev_update
        wla_grad = cp.dot(cp.dot(w_look_ahead.T, delta) * d, a.T)

        update = (1 / samples) * lamda * prev_update + learning_rate * wla_grad

        return update, w - update

    @staticmethod
    def rmsprop(cycle: int,
                samples: int,
                prev_update: np.ndarray,
                learning_rate: float,
                delta: np.ndarray,
                w: np.ndarray,
                a: np.ndarray,
                d: np.ndarray,
                hyper_param: dict = None) -> tuple[np.ndarray, np.ndarray]:
        '''
        Compute the root-mean-square propagation (RMSProp) optimization function.
        :param cycle: Training cycle.
        :param samples: Number of training samples.
        :param prev_update: The update at the previous time step.
        :param learning_rate: Learning rate.
        :param delta: Error value(s).
        :param w: Weights.
        :param a: Activations.
        :param d: Activation derivatives.
        :param hyper_param: Hyperparameters. Uses rmsprop_beta.
        :return: Adaptive gradient (RMSProp) optimization.
        '''
        beta = 0.9

        if hyper_param is not None and "rmsprop_beta" in hyper_param.keys():
            beta = hyper_param["rmsprop_beta"]

        grad = np.dot(np.dot(w.T, delta) * d, a.T)
        update = beta * prev_update + (1 - beta) * grad**2
        epsilon = 1e-8

        return update, w - (1 / samples) * grad * learning_rate / np.sqrt(update + epsilon)

    @staticmethod
    def rmsprop_gpu(cycle: int,
                    samples: int,
                    prev_update: cp.ndarray,
                    learning_rate: float,
                    delta: cp.ndarray,
                    w: cp.ndarray,
                    a: cp.ndarray,
                    d: cp.ndarray,
                    hyper_param: dict = None) -> tuple[cp.ndarray, cp.ndarray]:
        '''
        Compute the root-mean-square propagation  (RMSProp) optimization function.
        :param cycle: Training cycle.
        :param samples: Number of training samples.
        :param prev_update: The update at the previous time step.
        :param learning_rate: Learning rate.
        :param delta: Error value(s).
        :param w: Weights.
        :param a: Activations.
        :param d: Activation derivatives.
        :param hyper_param: Hyperparameters. Uses rmsprop_beta.
        :return: Adaptive gradient (RMSProp) optimization.
        '''
        beta = 0.9

        if hyper_param is not None and "rmsprop_beta" in hyper_param.keys():
            beta = hyper_param["rmsprop_beta"]

        grad = cp.dot(cp.dot(w.T, delta) * d, a.T)
        update = beta * prev_update + (1 - beta) * grad**2
        epsilon = 1e-8

        return update, w - (1 / samples) * grad * learning_rate / cp.sqrt(update + epsilon)