### !! still not full understand

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

def interpkrig(x, y, z, s, Xi, Yi, n, d_max, a):
    """
    2D interpolation using ordinary kriging/collocation
    with second-order markov covariance model.

    :param x: x-coord (m)
    :param y: y-coord (m)
    :param z: values
    :param s: obs. error added to diagonal
    :param Xi: x-coord. interp. point(s) (m)
    :param Yi: y-coord. interp. point(s) (m)
    :param d: maximum distance allowed (m)
    :param a: correlation length in distance (m)
    :param n: number of nearest neighbours, should > 1
    :return: 1D vec. of prediction, sigma and nobs
    """

    n = int(n)

    # Check
    if n == 1: 
        print('n > 1 needed!')
        return

    xi = Xi.ravel()
    yi = Yi.ravel()

    zi = np.zeros(len(xi)) * np.nan
    ei = np.zeros(len(xi)) * np.nan
    ni = np.zeros(len(xi)) * np.nan

    tree = cKDTree(np.c_[x, y])
    
    # Convert to meters
    a *= 0.595 * 1e3
    d_max *= 1e3

    for i in range(len(xi)):
        
        # dxy is the distance between center point to neighboring points
        (dxy, idx) = tree.query((xi[i], yi[i]), k=n) 

        if dxy.min() > d_max:
            continue

        xc = x[idx]
        yc = y[idx]
        zc = z[idx]
        sc = s[idx]

        if len(zc) < 2: 
            continue
        
        m0 = np.median(zc)
        c0 = np.var(zc)
        
        # Covariance function for Dxy  
        Cxy = c0 * (1 + (dxy / a)) * np.exp(-dxy / a)
        
        # Compute pair-wise distance (neighboring points to neighboring points)
        dxx = cdist(np.c_[xc, yc], np.c_[xc, yc], "euclidean")
        
        # Covariance function Dxx
        Cxx = c0 * (1 + (dxx / a)) * np.exp(-dxx / a)
        
        # Measurement noise matrix
        N = np.eye(len(Cxx)) * sc * sc
        
        # Solve for the inverse
        CxyCxxi = np.linalg.solve((Cxx + N).T, Cxy.T)
        
        # Predicted value
        zi[i] = np.dot(CxyCxxi, zc) + (1 - np.sum(CxyCxxi)) * m0
        
        # Predicted error
        ei[i] = np.sqrt(np.abs(c0 - np.dot(CxyCxxi, Cxy.T)))
        
        # Number of points in prediction
        ni[i] = len(zc)

    return zi, ei, ni
