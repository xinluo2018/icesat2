# author: Fernando Paolo, 
# modify: xin luo, 2021.8.15
# des: filtering data within 2d space

import numpy as np
from scipy import stats

def spatial_filter(x, y, z, dx, dy, sigma=3.0):
    """
    des: outlier filtering within the defined spatial region (dx * dy). 
    arg:
        x, y: coord_x and coord_y (m)
        z: value
        dx, dy: resolution in x (m) and y (m)
        n_sigma: cut-off value
        thres: max absolute value of data
    return: 
        zo: filtered z, containing nan-values
    """

    Nn = int((np.abs(y.max() - y.min())) / dy) + 1
    Ne = int((np.abs(x.max() - x.min())) / dx) + 1

    f_bin = stats.binned_statistic_2d(x, y, z, bins=(Ne, Nn))
    index = f_bin.binnumber   # the bin index of each (x,y)
    ind = np.unique(index)
    zo = z.copy()
    # loop for each bin (valid data exit)
    for i in range(len(ind)):
        # index: bin index corresponding to each data point
        idx, = np.where(index == ind[i])   # idx:  data points indices in specific bin
        zb = z[idx]
        if len(zb[~np.isnan(zb)]) == 0:
            continue
        dh = zb - np.nanmedian(zb)
        foo = np.abs(dh) > sigma * np.nanstd(dh)
        zb[foo] = np.nan
        zo[idx] = zb

    return zo
