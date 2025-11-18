import cupy as cp
import numpy as np

class Initializer:
    @staticmethod
    def glorot_normal(shape: tuple[int, int], fan_in: int, fan_out: int) -> np.ndarray:
        '''
        Compute the normal Glorot/Xavier weight initialization function.
        :param shape: Shape of the weight array.
        :param fan_in: Input size.
        :param fan_out: Output size.
        :return: Initialized weight array.
        '''
        return np.random.normal(loc=0, scale=np.sqrt(2.0 / (fan_in + fan_out)), size=shape)

    @staticmethod
    def glorot_normal_gpu(shape: tuple[int, int], fan_in: int, fan_out: int) -> cp.ndarray:
        '''
        Compute the normal Glorot/Xavier weight initialization function on the GPU.
        :param shape: Shape of the weight array.
        :param fan_in: Input size.
        :param fan_out: Output size.
        :return: Initialized weight array.
        '''
        return cp.random.normal(loc=0, scale=cp.sqrt(2.0 / (fan_in + fan_out)), size=shape)

    @staticmethod
    def glorot_uniform(shape: tuple[int, int], fan_in: int, fan_out: int) -> np.ndarray:
        '''
        Compute the uniform Glorot/Xavier weight initialization function.
        :param shape: Shape of the weight array.
        :param fan_in: Input size.
        :param fan_out: Output size.
        :return: Initialized weight array.
        '''
        return (np.random.rand(shape[0], shape[1]) * 2.0 - 1.0) * np.sqrt(6.0 / (fan_in + fan_out))

    @staticmethod
    def glorot_uniform_gpu(shape: tuple[int, int], fan_in: int, fan_out: int) -> cp.ndarray:
        '''
        Compute the uniform Glorot/Xavier weight initialization function on the GPU.
        :param shape: Shape of the weight array.
        :param fan_in: Input size.
        :param fan_out: Output size.
        :return: Initialized weight array.
        '''
        return (cp.random.rand(shape[0], shape[1]) * 2.0 - 1.0) * cp.sqrt(6.0 / (fan_in + fan_out))

    @staticmethod
    def he_normal(shape: tuple[int, int], fan_in: int, fan_out: int) -> np.ndarray:
        '''
        Compute the normal He/Kaiming weight initialization function.
        :param shape: Shape of the weight array.
        :param fan_in: Input size.
        :param fan_out: Output size.
        :return: Initialized weight array.
        '''
        return np.random.normal(loc=0, scale=np.sqrt(2.0 / fan_in), size=shape)

    @staticmethod
    def he_normal_gpu(shape: tuple[int, int], fan_in: int, fan_out: int) -> cp.ndarray:
        '''
        Compute the normal He/Kaiming weight initialization function on the GPU.
        :param shape: Shape of the weight array.
        :param fan_in: Input size.
        :param fan_out: Output size.
        :return: Initialized weight array.
        '''
        return cp.random.normal(loc=0, scale=cp.sqrt(2.0 / fan_in), size=shape)

    @staticmethod
    def he_uniform(shape: tuple[int, int], fan_in: int, fan_out: int) -> np.ndarray:
        '''
        Compute the uniform He/Kaiming weight initialization function.
        :param shape: Shape of the weight array.
        :param fan_in: Input size.
        :param fan_out: Output size.
        :return: Initialized weight array.
        '''
        return (np.random.rand(shape[0], shape[1]) * 2.0 - 1.0) * np.sqrt(6.0 / fan_in)

    @staticmethod
    def he_uniform_gpu(shape: tuple[int, int], fan_in: int, fan_out: int) -> cp.ndarray:
        '''
        Compute the uniform He/Kaiming weight initialization function on the GPU.
        :param shape: Shape of the weight array.
        :param fan_in: Input size.
        :param fan_out: Output size.
        :return: Initialized weight array.
        '''
        return (cp.random.rand(shape[0], shape[1]) * 2.0 - 1.0) * cp.sqrt(6.0 / fan_in)

    @staticmethod
    def lecun_normal(shape: tuple[int, int], fan_in: int, fan_out: int) -> np.ndarray:
        '''
        Compute the normal LeCun weight initialization function.
        :param shape: Shape of the weight array.
        :param fan_in: Input size.
        :param fan_out: Output size.
        :return: Initialized weight array.
        '''
        return np.random.normal(loc=0, scale=np.sqrt(1.0 / fan_in), size=shape)

    @staticmethod
    def lecun_normal_gpu(shape: tuple[int, int], fan_in: int, fan_out: int) -> cp.ndarray:
        '''
        Compute the normal LeCun weight initialization function on the GPU.
        :param shape: Shape of the weight array.
        :param fan_in: Input size.
        :param fan_out: Output size.
        :return: Initialized weight array.
        '''
        return cp.random.normal(loc=0, scale=cp.sqrt(1.0 / fan_in), size=shape)

    @staticmethod
    def lecun_uniform(shape: tuple[int, int], fan_in: int, fan_out: int) -> np.ndarray:
        '''
        Compute the uniform LeCun weight initialization function.
        :param shape: Shape of the weight array.
        :param fan_in: Input size.
        :param fan_out: Output size.
        :return: Initialized weight array.
        '''
        return (np.random.rand(shape[0], shape[1]) * 2.0 - 1.0) * np.sqrt(3.0 / fan_in)

    @staticmethod
    def lecun_uniform_gpu(shape: tuple[int, int], fan_in: int, fan_out: int) -> cp.ndarray:
        '''
        Compute the uniform LeCun weight initialization function on the GPU.
        :param shape: Shape of the weight array.
        :param fan_in: Input size.
        :param fan_out: Output size.
        :return: Initialized weight array.
        '''
        return (cp.random.rand(shape[0], shape[1]) * 2.0 - 1.0) * cp.sqrt(3.0 / fan_in)