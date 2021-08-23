import numpy as np
from scipy.spatial import cKDTree

def interpgaus(x, y, z, s, xi, yi, n, d_max, a):

    """
    des:2D interpolation using a gaussian kernel, weighted by distance and error.
    arg: 
        x, y: x-coord (m) and y-coord (m) corresponding to all the data points, 
        z: values
        s: obs. errors
        xi, yi: x-coord (m) and y-coord (m) corresponding to the interpolated points.
        n: the nearest n neighbours for searching.
        d_max: maximum distance allowed (m)
        a: correlation length in distance (m)
    return: 
        zi, ei: interpolated z and the corresponding to error
        ni: number of objects for interpolation
    """

    xi = xi.ravel()
    yi = yi.ravel()

    zi = np.zeros(len(xi)) * np.nan   # 
    ei = np.zeros(len(xi)) * np.nan
    ni = np.zeros(len(xi)) * np.nan

    tree = cKDTree(np.c_[x, y])

    if np.all(np.isnan(s)): 
        s = np.ones(s.shape)   # initial obs. errors

    # loops for all target points
    for i in range(len(xi)):
        # return distance and index
        (dxy, idx) = tree.query((xi[i], yi[i]), k=n)  # find the nearest n obs.
        if n == 1:
            pass
        elif dxy.min() > d_max:
            continue
        else:
            pass
        zc = z[idx]
        sc = s[idx]
        if len(zc[~np.isnan(zc)]) == 0: 
            continue
        wc = (1./sc**2) * np.exp(-(dxy**2)/(2*a**2))     # gaussian weight
        wc += 1e-6  # avoid singularity
        zi[i] = np.nansum(wc * zc) / np.nansum(wc)   # weighted height
        sigma_r = np.nansum(wc * (zc - zi[i])**2) / np.nansum(wc)   # Weighted rmse of height
        sigma_s = 0 if np.all(s == 1) else np.nanmean(sc)        # Obs. error
        ei[i] = np.sqrt(sigma_r ** 2 + sigma_s ** 2)   # Prediction error
        ni[i] = 1 if n == 1 else len(zc)               # Number of points in prediction

    return zi, ei, ni

