import numpy as np
from scipy.spatial import cKDTree

def interpgaus(x, y, z, s, Xi, Yi, n, d, a):
    """
    2D interpolation using a gaussian kernel
    weighted by distance and error

    :param x: x-coord (m)
    :param y: y-coord (m)
    :param z: values
    :param s: obs. errors
    :param Xi: x-coord. interp. point(s) (m)
    :param Yi: y-coord. interp. point(s) (m)
    :param n: number of nearest neighbours
    :param d: maximum distance allowed (m)
    :param a: correlation length in distance (m)
    :return: 1D vec. of prediction, sigma and nobs
    """

    xi = Xi.ravel()
    yi = Yi.ravel()

    zi = np.zeros(len(xi)) * np.nan
    ei = np.zeros(len(xi)) * np.nan
    ni = np.zeros(len(xi)) * np.nan

    tree = cKDTree(np.c_[x, y])

    if np.all(np.isnan(s)): s = np.ones(s.shape)

    for i in range(len(xi)):

        (dxy, idx) = tree.query((xi[i], yi[i]), k=n)

        if n == 1:
            pass
        elif dxy.min() > d:
            continue
        else:
            pass

        zc = z[idx]
        sc = s[idx]
        
        if len(zc[~np.isnan(zc)]) == 0: continue
        
        # Weights
        wc = (1./sc**2) * np.exp(-(dxy**2)/(2*a**2))
        
        # Avoid singularity
        wc += 1e-6
        
        # Predicted value
        zi[i] = np.nansum(wc * zc) / np.nansum(wc)

        # Weighted rmse
        sigma_r = np.nansum(wc * (zc - zi[i])**2) / np.nansum(wc)

        # Obs. error
        sigma_s = 0 if np.all(s == 1) else np.nanmean(sc)

        # Prediction error
        ei[i] = np.sqrt(sigma_r ** 2 + sigma_s ** 2)

        # Number of points in prediction
        ni[i] = 1 if n == 1 else len(zc)

    return zi, ei, ni
