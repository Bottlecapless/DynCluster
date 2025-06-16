import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.random_projection import GaussianRandomProjection
from mpl_toolkits.mplot3d import Axes3D  # 用于 3D 绘图
from copy import deepcopy

from constants.filepath_constants import GIST_DIR

class DataReader:
    """
    
    """
    def __init__(self,filename=GIST_DIR, max_samples=None):
        self.__fileName = filename
        self.__maxSample = max_samples
        self.__data = None
        return
    
    @property
    def data(self):
        return deepcopy(self.__data)
    
    def getData(self): return self.__data

    def read_fvecs(self):
        """
        Read a .fvecs file and return an array of vectors.
        :param filename: File name.
        :param max_samples: Maximum number of samples to read.
        :return: Data array.
        """
        data = []
        with open(self.__fileName, 'rb') as f:
            count = 0
            while True:
                dim = np.fromfile(f, dtype=np.int32, count=1)
                if len(dim) == 0 or (self.__maxSample is not None and count >= self.__maxSample):
                    break
                vec = np.fromfile(f, dtype=np.float32, count=dim[0])
                if len(vec) < dim[0]:
                    break
                data.append(vec)
                count += 1
        self.__data = np.array(data)
        return 