import numpy as np
from scipy import stats

def spatial_filter(x, y, z, dx, dy, n_sigma=3.0):
    """
    Spatial outlier editing filter

    :param x: x-coord (m)
    :param y: y-coord (m)
    :param z: values
    :param dx: filter res. in x (m)
    :param dy: filter res. in y (m)
    :param n_sigma: cutt-off value
    :param thres: max absolute value of data
    :return: filtered array containing nan-values
    """

    Nn = int((np.abs(y.max() - y.min())) / dy) + 1
    Ne = int((np.abs(x.max() - x.min())) / dx) + 1

    f_bin = stats.binned_statistic_2d(x, y, x, bins=(Ne, Nn))

    index = f_bin.binnumber

    ind = np.unique(index)

    zo = z.copy()

    for i in range(len(ind)):
        
        # index for each bin
        idx, = np.where(index == ind[i])

        zb = z[idx]

        if len(zb[~np.isnan(zb)]) == 0:
            continue

        dh = zb - np.nanmedian(zb)

        foo = np.abs(dh) > n_sigma * np.nanstd(dh)

        zb[foo] = np.nan

        zo[idx] = zb

    return zo

