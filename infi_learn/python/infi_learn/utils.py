"""Various miscellaneous utility classes
"""

import math
import numpy as np

class LaserImagePainter(object):
    """Converts planar laser scans to fixed-size 2D byte images.

    Parameters
    ----------
    dim  : int
        Dimensionality of the laser scan
    laser_fov  : 2-tuple or list of float
        Min and max angular sweep of scan in radians
    max_range  : float
        Max valid (or desired) range per beam
    resolution : float
        Image cell size in meters
    """

    def __init__(self, dim, laser_fov, max_range, resolution,
                 dtype=np.uint8, empty_val=0, fill_val=1):
        angles = np.linspace(laser_fov[0], laser_fov[1], dim)
        self._cosses = np.cos(angles)
        self._sines = np.sin(angles)
        self.res = resolution
        self.max_range = max_range

        max_xs = self.max_range * self._cosses
        max_ys = self.max_range * self._sines
        self._offsets = (round(-min(max_xs) / self.res),
                         round(-min(max_ys) / self.res))
        # NOTE In our convention, the origin is in the middle of a square pixel
        self.img_size = (int(math.ceil((max(max_xs) - min(max_xs)) / self.res)) + 1,
                         int(math.ceil((max(max_ys) - min(max_ys)) / self.res)) + 1)
        self.empty_val = empty_val
        self.fill_val = fill_val
        self.dtype = dtype

    def scan_to_image(self, scan):
        """Converts planar laser scan to 2D image. Ranges less than 0,
        exceeding the max range, or non-finite are discarded.

        Parameters
        ----------
        scan : iterable of floats
            The scan ranges, assumed ordered from min angle to max angle

        Returns
        -------
        img  : numpy 2D array [width, height, 1]
            The painted image with 0s in empty space and 1s on hits
        """
        scan = np.array(scan)
        valid = np.logical_and(scan > 0.0, scan <= self.max_range)
        valid = np.logical_and(valid, np.isfinite(scan))
        x = scan * self._cosses
        y = scan * self._sines
        ix = np.round(x[valid] / self.res + self._offsets[0]).astype(int)
        iy = np.round(y[valid] / self.res + self._offsets[1]).astype(int)
        img = np.full(shape=(self.img_size[0], self.img_size[1], 1), dtype=self.dtype,
                      fill_value=self.empty_val)
        img[ix, iy] = self.fill_val
        return img


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
