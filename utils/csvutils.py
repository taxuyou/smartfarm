import numpy as np
import tensorflow as tf

__all__ = ['tensor2csv']

def tensor2csv(filename, tensor):
    np_tensor = tensor.numpy()
    np_shape = np_tensor.shape

    np_array = np_tensor.reshape((np_shape[1], np_shape[2]))
    np.savetxt(filename, np_array, fmt='%f', delimiter=' ')

