import cupy as cp
import numpy as np

class Regularizer:
    @staticmethod
    def none(w: np.ndarray, hyper_param: dict=None) -> float:
        '''
        Return 0.
        :param w: Weights.
        :param hyper_param: Hyperparameters.
        :return: 0
        '''
        return 0

    @staticmethod
    def none_gpu(w: cp.ndarray, hyper_param: dict=None) -> float:
        '''
        Return 0.
        :param w: Weights.
        :param hyper_param: Hyperparameters.
        :return: 0
        '''
        return 0

    @staticmethod
    def elastic_net(w: np.ndarray, hyper_param: dict=None) -> float:
        '''
        Compute the Elastic Net regression (L1 + L2) regularization function.
        :param w: Weights.
        :param hyper_param: Hyperparameters. Uses elastic_net_lambda and elastic_net_alpha.
        :return: Elastic Net regression (L1 + L2) regularization.
        '''
        lamda = 1.0
        alpha = 0.5

        if hyper_param is not None:
            if "elastic_net_lambda" in hyper_param.keys():
                lamda = hyper_param["elastic_net_lambda"]
            if "elastic_net_alpha" in hyper_param.keys():
                alpha = hyper_param["elastic_net_alpha"]

        return lamda * ((1 - alpha) * np.sum(np.abs(w)) + alpha * np.sum(np.square(w)))

    @staticmethod
    def elastic_net_gpu(w: cp.ndarray, hyper_param: dict=None) -> float:
        '''
        Compute the Elastic Net regression (L1 + L2) regularization function on the GPU.
        :param w: Weights.
        :param hyper_param: Hyperparameters. Uses elastic_net_lambda and elastic_net_alpha.
        :return: Elastic Net regression (L1 + L2) regularization.
        '''
        lamda = 1.0
        alpha = 0.5

        if hyper_param is not None:
            if "elastic_net_lambda" in hyper_param.keys():
                lamda = hyper_param["elastic_net_lambda"]
            if "elastic_net_alpha" in hyper_param.keys():
                alpha = hyper_param["elastic_net_alpha"]

        return lamda * ((1 - alpha) * cp.sum(cp.abs(w)) + alpha * cp.sum(cp.square(w)))

    @staticmethod
    def lasso_l1(w: np.ndarray, hyper_param: dict=None) -> float:
        '''
        Compute the LASSO (Least Absolute Shrinkage and Selection Operator) (L1) regularization function.
        :param w: Weights.
        :param hyper_param: Hyperparameters. Uses lasso_l1_lambda.
        :return: LASSO (L1) regularization.
        '''
        lamda = 0.1

        if hyper_param is not None:
            if "lasso_l1_lambda" in hyper_param.keys():
                lamda = hyper_param["lasso_l1_lambda"]

        return lamda * np.sum(np.abs(w))

    @staticmethod
    def lasso_l1_gpu(w: cp.ndarray, hyper_param: dict=None) -> float:
        '''
        Compute the LASSO (Least Absolute Shrinkage and Selection Operator) (L1) regularization function on the GPU.
        :param w: Weights.
        :param hyper_param: Hyperparameters. Uses lasso_l1_lambda.
        :return: LASSO (L1) regularization.
        '''
        lamda = 0.1

        if hyper_param is not None:
            if "lasso_l1_lambda" in hyper_param.keys():
                lamda = hyper_param["lasso_l1_lambda"]

        return lamda * cp.sum(cp.abs(w))

    @staticmethod
    def ridge_l2(w: np.ndarray, hyper_param: dict=None) -> float:
        '''
        Compute the Ridge regression (L2) regularization function.
        :param w: Weights.
        :param hyper_param: Hyperparameters. Uses ridge_l2_lambda.
        :return: Ridge regression (L2) regularization.
        '''
        lamda = 1.0

        if hyper_param is not None:
            if "ridge_l2_lambda" in hyper_param.keys():
                lamda = hyper_param["ridge_l2_lambda"]

        return lamda * np.sum(np.square(w))

    @staticmethod
    def ridge_l2_gpu(w: cp.ndarray, hyper_param: dict=None) -> float:
        '''
        Compute the Ridge regression (L2) regularization function on the GPU.
        :param w: Weights.
        :param hyper_param: Hyperparameters. Uses ridge_l2_lambda.
        :return: Ridge regression (L2) regularization.
        '''
        lamda = 1.0

        if hyper_param is not None:
            if "ridge_l2_lambda" in hyper_param.keys():
                lamda = hyper_param["ridge_l2_lambda"]

        return lamda * cp.sum(cp.square(w))