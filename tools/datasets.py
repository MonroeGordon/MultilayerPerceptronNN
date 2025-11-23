import cupy as cp
import numpy as np
import pandas
import pandas as pd

class Data:
    '''
    Data class. Stores features and target data from a dataset and provides methods for accessing and using the data.
    '''

    def __init__(self,
                 x_names: np.ndarray,
                 y_names: np.ndarray,
                 x: np.ndarray,
                 y: np.ndarray):
        '''
        Initializes a Data class with the specified feature and target data.
        :param x_names: Feature names (number of features).
        :param y_names: Class names (number of classes).
        :param x: Input feature matrix (number samples, number features).
        :param y: Target value(s) (target names, indices, or one hot encoded).
        '''
        if x.ndim != 2:
            raise ValueError("Data: parameter 'x' must be a matrix (2-dimensional ndarray).")

        if y.shape[0] != x.shape[0]:
            raise ValueError("Data: parameter 'y' shape[0] must equal parameter 'x' shape[0].")

        if x_names.ndim != 1:
            raise ValueError("Data: parameter 'x_names' must be a vector (1-dimensional ndarray).")

        if x_names.shape[0] != x.shape[1]:
            raise ValueError("Data: parameter 'x_names' shape[0] must equal parameter 'x' shape[1].")

        if y_names.ndim != 1:
            raise ValueError("Data: parameter 'y_names' must be a vector (1-dimensional ndarray).")

        if y.ndim == 1:
            if len(np.unique(y)) != y_names.shape[0]:
                raise ValueError("Data: parameter 'y' unique value count must equal parameter 'y_names' shape[0].")
        else:
            if y.shape[1] != y_names.shape[0]:
                raise ValueError("Data: parameter 'y' shape[1] must equal parameter 'y_names' shape[0].")

        self._feature_names = x_names
        self._class_names = y_names
        self._features = x
        self._targets = y

    @property
    def class_names(self):
        return self._class_names

    @ property
    def feature_names(self):
        return self._feature_names

    @property
    def features(self):
        return self._features

    @property
    def features_gpu(self):
        return cp.asarray(self._features)

    @property
    def targets(self):
        return self._targets

    @property
    def targets_gpu(self):
        return cp.asarray(self._targets)

class Datasets:
    '''
    Datasets class. Provides methods for accessing dataset files and converting them into data that a machine learning/
    AI model can process.
    '''

    DATASET_PATH = "C:/BlueSkyAI/Projects/Python/BlueSkyLib/datasets/"

    @staticmethod
    def load_iris(one_hot: bool=False) -> Data:
        '''
        Loads the Iris dataset into a Data class.
        :param one_hot: Perform one hot encoding of target classes.
        :return: A Data class containing the Iris dataset.
        '''
        df = pandas.read_csv(Datasets.DATASET_PATH + "iris.csv")
        fdf = df.iloc[:, :4]
        tdf = df.iloc[:, 4]

        x = fdf.to_numpy()
        y = tdf.to_numpy()

        x_names = np.array(["Sepal Length (cm)", "Sepal Width (cm)", "Petal Length (cm)", "Petal Width (cm)"])

        y_names = np.unique(y)

        if one_hot:
            tdf = pd.get_dummies(tdf, columns=[4])
            y = tdf.to_numpy()
        else:
            for i in range(len(y_names)):
                indices = np.where(y == y_names[i])
                y[indices] = i

        return Data(x_names, y_names, x, y)