## main reference: https://github.com/fspaolo/captoolkit
## author: xin luo; 
## create: 2021.8.8; 


import numpy as np
from utils.make_grid import make_grid
from utils.spatial_filter import spatial_filter
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.interpolate import InterpolatedUnivariateSpline


### ------------------------------ ###
###       1-d interpolation       ###
### ------------------------------ ###

def interp1d(x, y, xi, n = 1):
    """ des: 1D interpolation (spline)
        args:
            x,y: coord_x and coord_y of the given points.
            xi: the interpolation point.
            n: degree of the smoothing spline. must be 1 <= n <= 5.
        return:
            interpolated yi value.
    """
    idx = np.argsort(x)    # Sort data by x value
    x, y = x[idx], y[idx]  # Sort arrays
    Fi = InterpolatedUnivariateSpline(x, y, k=n)  # Create interpolator
    yi = Fi(xi)     # Interpolated value
    return yi


### ------------------------------ ###
###       2-d interpolation        ###
### ------------------------------ ###

def interp2d_spline(x, y, z, xi, yi, n=None, d_max=2000, order=3):
    """ des: fast bilinear interpolation by using spline method.
        arg:
            x, y: 1d array_like data, are the image coordinates of data
            z: value corresponding to (coord_x, coord_y)
            xi, yi: 1d array_like data, the image coordinates to be interpolated.
            n: the nearest n neighbours for interpolation.
            d_max: allowed distance from the interpolated point.
            order: 1 for linear interpolation, and 3 for cubic interpolation.
        retrun:
            zi, ei: interpolated z and the corresponding to error
            ni: number of objects for interpolation
    """
    tree = cKDTree(np.c_[x, y])
    zi = np.ones((len(xi)))*np.nan   # initial interpolated height
    ei = np.ones((len(xi)))*np.nan
    ni = np.ones((len(xi)))*np.nan   

    ### TODO: convert for-loop to matrix computation.
    for i in range(len(xi)):
        if n:
            idx_1 = tree.query_ball_point([xi[i], yi[i]], r=d_max)   # select neighboring points
            (_, idx_2) = tree.query((xi[i], yi[i]), k=n)    # the nearest n points.
            idx = [id for id in idx_1 if id in idx_2]  
        else:
            idx = tree.query_ball_point([xi[i], yi[i]], r=d_max)   # select neighboring points
        if len(idx) < 6: 
            continue
        x_neigh = x[idx]
        y_neigh = y[idx]
        z_neigh = z[idx]
        d0_spat = np.sqrt((x_neigh - x_neigh[0])**2 + (y_neigh - y_neigh[0])**2)  # dist. between given point.
        di_spat = np.sqrt((x_neigh - xi[i])**2 + (y_neigh - yi[i])**2)  # dist. between given points and interp. point.
        di0_spat = np.sqrt((x_neigh[0] - xi[i])**2 + (y_neigh[0] - yi[i])**2)
        idsort = np.argsort(di_spat)  # ids is the index of points from nearest to farest. 
        ## ------ interpolation points and weights
        d0_spat = d0_spat[idsort]
        z_neigh = z_neigh[idsort]            # from nearest to farest
        w = 1/di_spat[idsort]                # dist. weights

        ## ------ sorted by dist. to neigh[0]    
        idxsort = np.argsort(d0_spat)                
        d0_spat, z_neigh = d0_spat[idxsort], z_neigh[idxsort]  # Sort arrays
        w = w[idxsort]
        ## ------ ensure the neighbor points distributed at two side of the interpolated point.
        if di0_spat <= d0_spat[0] or di0_spat >= d0_spat[-1]:
            continue
        ## ------ interpolation
        Fi = InterpolatedUnivariateSpline(d0_spat, z_neigh, k=order)  # Create interpolator
        zi[i] = Fi(di0_spat)                 # Interpolated value
        ei[i] = np.nansum(w*(z_neigh-zi[i])**2)/np.nansum(w)  # distance weighted height std 
        ni[i] = len(z_neigh)                 # Number of obs. in solution

    return zi, ei, ni



def interp2d_gaus(x, y, z, xi, yi, n=None, d_max=2000, alpha_d=2000):
    """
    des:2D interpolation using a gaussian kernel, weighted by distance.
    arg: 
        x, y: x-coord (m) and y-coord (m) corresponding to all the data points, 
        z: values
        xi, yi: x-coord (m) and y-coord (m) corresponding to the interpolated points.
        n: the nearest n neighbours for interpolation.
        d_max: maximum distance allowed (m)
        alpha_d: correlation length in distance (m)
    return: 
        zi, ei: interpolated z and the corresponding to error
        ni: number of objects for interpolation
    """

    zi = np.zeros(len(xi)) * np.nan     # 
    ei = np.zeros(len(xi)) * np.nan
    ni = np.zeros(len(xi)) * np.nan

    tree = cKDTree(np.c_[x, y])

    # loops for all target points
    for i in range(len(xi)):
        if n:
            idx_1 = tree.query_ball_point([xi[i], yi[i]], r=d_max)   # select neighboring points
            (_, idx_2) = tree.query((xi[i], yi[i]), k=n)    # the nearest n points.
            idx = [id for id in idx_1 if id in idx_2]  
        else:
            idx = tree.query_ball_point([xi[i], yi[i]], r=d_max)   # select neighboring points

        if len(idx) == 0:
            continue
        ## TODO: the neighbor points should be around the interpolated point
        x_neigh = x[idx]
        y_neigh = y[idx]
        z_neigh = z[idx]

        dxy = np.sqrt((x_neigh - xi[i])**2 + (y_neigh - yi[i])**2)
        w = np.exp(-(dxy**2)/(2*alpha_d**2))        # gaussian weight
        w += 1e-6                                   # avoid singularity
        zi[i] = np.nansum(w * z_neigh) / np.nansum(w)   # weighted height
        ei[i] = np.nansum(w * (z_neigh - zi[i])**2) / np.nansum(w)   # Weighted rmse of height
        ni[i] = len(z_neigh)                              # Number of points in prediction

    return zi, ei, ni



def interp2d_krig(x, y, z, xi, yi, n=None, d_max=2000, alpha_d=2000):
    """
    des:2D interpolation by using ordinary kriging/collocation method
    arg: 
        x, y: x-coord (m) and y-coord (m) corresponding to all the data points, 
        z: values
        xi, yi: x-coord (m) and y-coord (m) corresponding to the interpolated points.
        n: the nearest n neighbours for interpolation.
        d_max: maximum distance allowed (m)
        alpha_d: correlation length in distance (m)
    return: 
        zi, ei: interpolated z and the corresponding to error
        ni: number of objects for interpolation
    """

    zi = np.zeros(len(xi)) * np.nan
    ei = np.zeros(len(xi)) * np.nan
    ni = np.zeros(len(xi)) * np.nan
    tree = cKDTree(np.c_[x, y])

    for i in range(len(xi)):        
        if n:
            idx_1 = tree.query_ball_point([xi[i], yi[i]], r=d_max)   # select neighboring points
            (_, idx_2) = tree.query((xi[i], yi[i]), k=n)    # the nearest n points.
            idx = [id for id in idx_1 if id in idx_2]  
        else:
            idx = tree.query_ball_point([xi[i], yi[i]], r=d_max)   # select neighboring points

        if len(idx) < 2:
            continue
        ## TODO: the neighbor points should be around the interpolated point
        x_neigh = x[idx]
        y_neigh = y[idx]
        z_neigh = z[idx]
        dxy = np.sqrt((x_neigh - xi[i])**2 + (y_neigh - yi[i])**2)
        m0 = np.median(z_neigh)
        c0 = np.var(z_neigh)
        # Covariance function for Dxy  
        Cxy = c0 * (1 + (dxy / alpha_d)) * np.exp(-dxy / alpha_d)
        # Compute pair-wise distance (neighboring points to neighboring points)
        dxx = cdist(np.c_[x_neigh, y_neigh], np.c_[x_neigh, y_neigh], "euclidean")
        # Covariance function Dxx
        Cxx = c0 * (1 + (dxx / alpha_d)) * np.exp(-dxx / alpha_d)
        # Solve for the inverse
        CxyCxxi = np.linalg.solve(Cxx.T, Cxy.T)        
        # Predicted value
        zi[i] = np.dot(CxyCxxi, z_neigh) + (1 - np.sum(CxyCxxi)) * m0
        # Predicted error
        ei[i] = np.sqrt(np.abs(c0 - np.dot(CxyCxxi, Cxy.T)))
        # Number of points in prediction
        ni[i] = len(z_neigh)

    return zi, ei, ni



### ------------------------------ ###
###       3-d interpolation       ###
### ------------------------------ ###
def interp3d(x, y, t, z, xi, yi, ti, \
                    alpha_d, alpha_t, d_max=3000):
    '''
    des: 3-d interpolation by using gaussian method
    args:
        x,y,t,z: 1d array_like data, the coord_x,coord_x,time and height of the existing 4-d points
        xi,yi,ti: array_like, the coord_x,coord_y and time of the interpolation points.
        alpha_d, alpha_t: spatial and temporal corr. length (km and months)
        radius: the spatial radius for neighboring points selecting.
    ruturn:
        zi,ei: the height and weighted std of the interpolation point.
        ni: the number of points used for interpolation. 
    '''
    tree = cKDTree(np.c_[x, y])
    zi = np.ones((len(xi)))*np.nan  # initial interpolated height
    ei = np.ones((len(xi)))*np.nan
    ni = np.ones((len(xi)))*np.nan   
    
    ### TODO: convert for-loop to matrix computation.
    for i in range(len(xi)):

        idx = tree.query_ball_point([xi[i], yi[i]], r=d_max)   # select neighboring points
        if len(idx) == 0: 
            continue
        x_neigh = x[idx]
        y_neigh = y[idx]
        z_neigh = z[idx]
        t_neigh = t[idx]

        d_time = np.abs(t_neigh - ti[i])    # time difference from all the points.            
        d_spat = np.sqrt((x_neigh - xi[i])**2 + (y_neigh - yi[i])**2)  # distance from interpolated point.
        # --- Compute the weighting factors, larger dist,dt, smaller ed,et
        # !!!alpha_d, alpha_t are actually the sigma in gaussian distribution function
        ed = np.exp(-(d_spat ** 2)/(2 * alpha_d ** 2))
        et = np.exp(-(d_time ** 2)/(2 * alpha_t ** 2))
        # Combine weights and scale with error, similar to the joint probability density function
        w = ed * et            
        w += 1e-6    # avoid division of zero
        zi[i] = np.nansum(w*z_neigh)/np.nansum(w)   #  weighted mean height
        ei[i] = np.nansum(w*(z_neigh-zi[i])**2)/np.nansum(w)  # Compute weighted height std 
        ni[i] = len(z_neigh)                  # Number of obs. in solution

    return zi, ei, ni



