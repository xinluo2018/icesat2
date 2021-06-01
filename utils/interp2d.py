import numpy as np
from scipy.ndimage import map_coordinates

def interp2d(x, y, z, xi, yi, **kwargs):
    """
    x,y,z: given coordinates and values
    xi,yi: target coordinates of the inperpolation map
    Raster to point interpolation based on
    scipy.ndimage import map_coordinates

    :param x: x-coord. in 2D (m)
    :param y: y-coord. in 2D (m)
    :param z: values in 2D
    :param xi: interp. point in x (m)
    :param yi: interp. point in y (m)
    :param kwargs: see map_coordinates
    :return: array of interp. values
    """

    x = np.flipud(x)
    y = np.flipud(y)
    z = np.flipud(z)
    
    x = x[0,:]
    y = y[:,0]
    
    nx, ny = x.size, y.size
    
    x_s, y_s = x[1] - x[0], y[1] - y[0]
    
    if np.size(xi) == 1 and np.size(yi) > 1:
        xi = xi * np.ones(yi.size)
    elif np.size(yi) == 1 and np.size(xi) > 1:
        yi = yi * np.ones(xi.size)
    
    xp = (xi - x[0]) * (nx - 1) / (x[-1] - x[0])
    yp = (yi - y[0]) * (ny - 1) / (y[-1] - y[0])

    coord = np.vstack([yp, xp])
    
    zi = map_coordinates(z, coord, **kwargs)
    
    return zi
