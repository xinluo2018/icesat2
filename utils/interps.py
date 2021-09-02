## main reference: https://github.com/fspaolo/captoolkit
## author: xin luo; 
## create: 2021.8.8

import numpy as np
from utils.make_grid import make_grid
from utils.spatial_filter import spatial_filter
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from scipy.interpolate import InterpolatedUnivariateSpline


### ------------------------------ ###
###       1-d interpolation       ###
### ------------------------------ ###

def interp1d(x, y, xi, n = 1):
    """ des: 1D interpolation (spline)
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


### ------------------------------ ###
###       2-d interpolation       ###
### ------------------------------ ###

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


### ------------------------------ ###
###       3-d interpolation       ###
### ------------------------------ ###

def interp3d(x, y, t, z, \
                xi, yi, ti, alpha_d, alpha_t):
    '''
    des: 3-d interpolation by using gaussian method
    '''
    d_time = np.abs(t - ti)    # time difference from all the points.            
    d_spat = np.sqrt((x - xi)**2 + (y - yi)**2)  # distance from interpolated point.
    # --- Compute the weighting factors, larger dist,dt, smaller ed,et
    # !!!alpha_d, alpha_t are actually the sigma in gaussian distribution function
    ed = np.exp(-(d_spat ** 2)/(2 * alpha_d ** 2))
    et = np.exp(-(d_time ** 2)/(2 * alpha_t ** 2))
    # Combine weights and scale with error, similar to the joint probability density function
    w = ed * et            
    w += 1e-6    # avoid division of zero
    zi = np.nansum(w*z)/np.nansum(w)   #  weighted mean height
    print(zi)
    sigma_r = np.nansum(w*(z-zi)**2)/np.nansum(w)  # Compute weighted height std 
    ei = np.sqrt(sigma_r ** 2)   # prediction error at grid node (interpolated point)
    ni = len(z)                  # Number of obs. in solution
    
    return zi, ei, ni





