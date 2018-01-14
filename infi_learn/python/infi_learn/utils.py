"""Various miscellaneous utility classes
"""

import math
import random
import numpy as np
from itertools import product

def shape_data_vec(data):
    """Converts single or array of vector data into shape required for tensorflow

    Parameters
    ----------
    data : 1D or array of 1D arrays/np.array
        If 1D, assumes it is a single vector, not list of 1D vectors (scalars)

    Returns
    -------
    data : numpy array [#data, dim]
    """

    data = np.asarray(data)
    if len(data.shape) < 1 or len(data.shape) > 2:
        raise ValueError(
            'Received data with %d dims, but need 1 or 2' % len(data.shape))

    # If single vector, prepend quantity dimension
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=0)
    return data


def shape_data_2d(data):
    """Converts single or array of 2D data into shape required for tensorflow's
    2D convolutions.

    Parameters
    ----------
    data : 2D or array of 2D arrays/np.array

    Returns
    -------
    data : numpy array [#data, width, height, 1]
    """

    data = np.asarray(data)
    if len(data.shape) < 2 or len(data.shape) > 4:
        raise ValueError(
            'Received data with %d dims, but need 2, 3, or 4' % len(data.shape))

    # If single image, prepend quantity dimension
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)

    if len(data.shape) == 3:
        # If single channeled image, prepend quantity dimension
        if data.shape[-1] == 1:
            data = np.expand_dims(data, axis=0)
        # Else must be array of non-channeled images, prepend channel
        else:
            data = np.expand_dims(data, axis=-1)

    if len(data.shape) == 4:
        if data.shape[-1] != 1:
            raise ValueError(
                'Received data with %d channels, but need 1' % data.shape[-1])

    return data


def shape_data_1d(data):
    """Converts single or array of 1D data into shape required for tensorflow's
    1D convolutions.

    Parameters
    ----------
    data : 1D or array of 1D arrays/np.array

    Returns
    -------
    data : numpy array [#data, len, 1]
    """
    data = np.asarray(data)
    if len(data.shape) < 1 or len(data.shape) > 3:
        raise ValueError(
            'Received data with %d dims, but need 1, 2, or 3' % len(data.shape))

    # If single data, prepend quantity dimension
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=0)

    if len(data.shape) == 2:
        # If single channeled data, prepend quantity dimension
        if data.shape[-1] == 1:
            data = np.expand_dims(data, axis=0)
        # Else must be array of non-channeled data, prepend channel
        else:
            data = np.expand_dims(data, axis=-1)

    if len(data.shape) == 3:
        if data.shape[-1] != 1:
            raise ValueError(
                'Received data with %d channels, but need 1' % data.shape[-1])

    return data

def unique_combos(data, k=1):
    """k=0 corresponds to non-unique
    """
    n = len(data)
    return [(data[i], data[j]) for i in range(n) for j in range(i+k,n)]
