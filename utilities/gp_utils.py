import itertools
import numpy as np
import sklearn.metrics as skmetrics


def calculate_power_set(iterable):
    """ power set([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3) """
    s = list(iterable)
    return list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1)))


def rbf_kernel_w_linear_coefficients(data_1, data_2, coefficients=None):
    """ calculates pairwise kernels according to coefficients, coefficients=all_ones gives Euclidean distance based kernels"""
    coefficients = np.ones(shape=(data_1.shape[1])) if coefficients is None else coefficients
    data_1n, data_2n = data_1*coefficients, data_2*coefficients
    dist_matrix = skmetrics.pairwise_distances(data_1n, data_2n)
    kernel_matrix = np.exp(-0.5 * np.square(dist_matrix))
    return kernel_matrix
