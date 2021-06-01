import numpy as np
from scipy.spatial import cKDTree

def interpmed(x, y, z, Xi, Yi, n, d):
    """
    2D median interpolation of scattered data

    :param x: x-coord (m)
    :param y: y-coord (m)
    :param z: values
    :param Xi: x-coord. grid (2D)
    :param Yi: y-coord. grid (2D)
    :param n: number of nearest neighbours
    :param d: maximum distance allowed (m)
    :return: 1D array of interpolated values
    """

    xi = Xi.ravel()
    yi = Yi.ravel()

    zi = np.zeros(len(xi)) * np.nan

    tree = cKDTree(np.c_[x, y])

    for i in range(len(xi)):

        (dxy, idx) = tree.query((xi[i], yi[i]), k=n)

        if n == 1:
            pass
        elif dxy.min() > d:
            continue
        else:
            pass

        zc = z[idx]

        zi[i] = np.median(zc)

    return zi
