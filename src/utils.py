from itertools import combinations
import numpy as np

def generate_subsets(n):
    subsets = []
    for length in range(n + 1):
        for subset in combinations(range(n), length):
            subsets.append(tuple(subset))
    return subsets

def elementwise_any(list_of_arrays):
    if not len(list_of_arrays):
        return np.False_
    return np.any(np.stack(list_of_arrays), axis=0)


def elementwise_max(list_of_arrays):
    if not len(list_of_arrays):
        return 0
    return np.max(np.stack(list_of_arrays), axis=0)