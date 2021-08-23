# author: Fernando Paolo, 
## author: xin luo; create: 2021.8.8

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.interpolate import InterpolatedUnivariateSpline


def interp1d(x, y, xi, n = 1):
    """ des: 1D interpolation (Spline)
        args:
            x,y are the given pair-wise values
            xi is the interpolation point.
            n: Degree of the smoothing spline. Must be 1 <= n <= 5.
        return:
            interpolated yi value
    """
    idx = np.argsort(x)    # Sort data by x value
    x, y = x[idx], y[idx]  # Sort arrays
    Fi = InterpolatedUnivariateSpline(x, y, k=n)  # Create interpolator
    yi = Fi(xi)     # Interpolated value
    return yi


def interp2d(x, y, z, xi, yi, **kwargs):
    """ des: fast bilinear interpolation from grid.
        arg:
            x, y: 2d array, are the coordinates of data
            z: value corresponding to (coord_x, coord_y)
            xi, yi: scalar/1d array, the coordinates to be interpolated.
        retrun:
            1-d interpolated value
    """
    x = np.flipud(x)   # flip the x-axis, thus the start of the coord convert from bottom-left to up-left
    y = np.flipud(y)
    z = np.flipud(z)
    x = x[0, :]     # x-axis
    y = y[:, 0]     # y-axis
    nx, ny = x.size, y.size
    assert (ny, nx) == z.shape
    assert (x[-1] > x[0]) and (y[-1] > y[0])
    if np.size(xi) == 1 and np.size(yi) > 1:
        xi = xi * np.ones(yi.size)
    elif np.size(yi) == 1 and np.size(xi) > 1:
        yq = yi * np.ones(xi.size)
    xp = (xi - xi[0]) * (nx - 1) / (xi[-1] - xi[0])    # obtain interpolated coords(in the data grid)
    yp = (yq - yi[0]) * (ny - 1) / (yi[-1] - yi[0])

    coord = np.vstack([yp, xp])    # yp, xp: row, col
    # input: value array, and coord([[row..],[col..]], coord.shape[0]:rows, .shape[1]:cols)
    # return: 1-d interpolated value
    zq = map_coordinates(z, coord, **kwargs) 
    return zq




