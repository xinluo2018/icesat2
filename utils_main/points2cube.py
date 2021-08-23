## author: Fernando Paolo
## modify: xin luo, 2021.8.3

"""
des: convert 3-d points to data cube by using spatial-temporal gaussian
    interpolation.
"""


import warnings
warnings.filterwarnings("ignore")
import sys 
sys.path.append("..")
import h5py
import numpy as np
import argparse
import pyproj
from scipy import stats
from scipy.spatial import cKDTree
from numba import jit
from helper import make_grid, spatial_filter


def coor2coor(srs_from, srs_to, x, y):
    """
    from utils/transform_xy.py
    """
    srs_from = pyproj.Proj(int(srs_from))
    srs_to = pyproj.Proj(int(srs_to))
    return pyproj.transform(srs_from, srs_to, x, y, always_xy=True)


@jit(nopython=True)
def fwavg(w, z):
    '''des: weighted mean z'''
    return np.nansum(w*z)/np.nansum(w)

@jit(nopython=True)
def fwstd(w, z, zm):
    return np.nansum(w*(z-zm)**2)/np.nansum(w)

@jit(nopython=True)
def make_weights(dr, dt, ad, at):
    '''
    des: generated key params for gaussian weighting.
    arg: 
        dr, dt: i.e. r-mean(r) and d-mean(d)
        ad, at: are the std corresponding to each dimension. actually sigma in the gaussian distribution. 
    return:
         exponential term in the gaussian function part
    '''
    ed = np.exp(-(dr ** 2)/(2 * ad ** 2))
    et = np.exp(-(dt ** 2)/(2 * at ** 2))
    return ed, et

@jit(nopython=True)
def square_dist(x, y, xi, yi):
    '''arg: x,y are the coords of the neighboring points, and xi, yi are the center point'''
    return np.sqrt((x - xi)**2 + (y - yi)**2)

@jit(nopython=True)
def fast_sort(x):
    return np.argsort(x)     # return the index 

# Description of algorithm
des = 'Spatio-temporal interpolation of irregular data'

# Define command-line arguments
parser = argparse.ArgumentParser(description=des)

parser.add_argument(
        'ifile', metavar='ifile', type=str, nargs='+',
        help='name of input file (h5-format)')

parser.add_argument(
        'ofile', metavar='ofile', type=str, nargs='+',
        help='name of ouput file (h5-format)')

parser.add_argument(
        '-d', metavar=('dx','dy', 'dt'), dest='dxyt', type=float, nargs=3,
        help=('resolution for 3-d grid (km, and month)'),
        default=[1, 1, 1],)

parser.add_argument(
        '-r', metavar='radius', dest='radius', type=float, nargs=1,
        help=('search radius (km)'),
        default=[None],)

parser.add_argument(
        '-a', metavar=('alpha_d','alpha_t'), dest='alpha', type=float, nargs=2,
        help=('spatial and temporal corr. length (km and months)'),
        default=[None, None],)

parser.add_argument(
        '-p', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
        help=('EPSG proj number (AnIS=3031, GrIS=3413)'), 
        default=['3031'],)

parser.add_argument(
        '-s', metavar=('n_sample'), dest='n_sample', type=int, nargs=1,
        help=('sample every n:th point in dataset'),
        default=[1],)

parser.add_argument(    
        '-c', metavar=('filter_dxy', 'thres', 'max'), dest='filter', type=float, nargs=3,
        help=('dim. of filter in km, sigma thres and max-value'),
        default=[0, 0, 9999],)

parser.add_argument(
        '-v', metavar=('x','y','z','t','s'), dest='vnames', type=str, nargs=5,
        help=('name of varibales in the HDF5-file'),
        default=['lon','lat','h_cor','t_year','h_rms'],)


def interpgaus3d(xp, yp, time, zp, dx, dy, dt, sp,radius):

    # Find all NaNs and do not select them
    no_nan = ~np.isnan(zp)
    
    # Remove data wiht NaN's
    xp, yp, zp, tp, sp = xp[no_nan], yp[no_nan], zp[no_nan],\
                            time[no_nan], sp[no_nan]

    xmin, xmax, ymin, ymax = xp.min(), xp.max(), yp.min(), yp.max()
    t_start, t_end = time.min(), time.max()

    # Time vector
    Ti = np.arange(t_start, t_end + dt, dt)

    # Construct the spatial grid: the cross point (Xi, Yi) can be regarded as interpolated point
    Xi, Yi = make_grid(xmin, xmax, ymin, ymax, dx, dy)

    # Shape of grid
    Nx, Ny = Xi.shape

    # Length of time vector
    Nt = len(Ti)

    # Output 3-D vectors: (time, row, col)
    Zi = np.ones((Nx, Ny, Nt)) * np.nan    # predicted height through the neiboring points in the temporal bin 
    Ei = np.ones((Nx, Ny, Nt)) * np.nan    # error of the neiboring points in the temporal bin
    Ni = np.ones((Nx, Ny, Nt)) * np.nan    # number (all time) of the neiboring points

    print("-> creating kdtree ...")
    # Construct cKDTree
    tree = cKDTree(np.c_[xp, yp])
    print('-> interpolating data ...')

    # --- prediction (spatial --> temporal)
    # Loop trough up-left coords of the generated grid (spatial dimension)
    for i in range(int(Nx)):   # loop through rows
        for j in range(int(Ny)):    # loop through cols
            idx = tree.query_ball_point([Xi[i,j], Yi[i,j]], r=radius)
            if len(idx) == 0: 
                continue
            xt = xp[idx]
            yt = yp[idx]
            zt = zp[idx]
            tt = tp[idx]
            st = sp[idx]

            for k in range(int(Nt)):   # loop through time 
                dt = np.abs(tt - Ti[k])   # time interval from the interpolated point.            
                dist = square_dist(xt, yt, Xi[i,j], Yi[i,j])   # distance from interpolated point.
                # Compute the weighting factors, larger dist,dt, smaller ed,et
                # !!!alpha_d, alpha_t are actually the sigma in gaussian distribution function
                ed, et = make_weights(dist, dt, alpha_d, alpha_t)
                # Combine weights and scale with error, similar to the joint probability density function
                w = (1. / st ** 2) * ed * et            
                w += 1e-6    # avoid division of zero
                zi = fwavg(w, zt)     #  weighted mean height
                # Test for signular values 
                if np.abs(zi) < 1e-6: 
                    continue
                sigma_r = fwstd(w, zt, zi)    # Compute weighted height std 
                sigma_s = 0 if np.all(st == 1) else np.nanmean(st)  # Compute systematic error
                ei = np.sqrt(sigma_r ** 2 + sigma_s ** 2)   # prediction error at grid node (interpolated point)
                ni = len(zt)        #  Number of obs. in solution
                ## Save data to output (interpolated point: (k,i,j) )
                Zi[i,j,k] = zi    # interpolated height
                Ei[i,j,k] = ei    # interpolated error
                Ni[i,j,k] = ni    # number of points in the interpolation.
    return Xi, Yi, Ti, Zi, Ei, Ni


if __name__ == '__main__':

    # Parser argument to variable
    args = parser.parse_args()
    # Read input from terminal
    ifile   = args.ifile[0]
    ofile   =  args.ofile[0]
    dx      = args.dxyt[0] * 1e3        # convert km to m
    dy      = args.dxyt[1] * 1e3
    dt      = args.dxyt[2]/12.
    proj    = args.proj[0]
    radius  = args.radius[0] * 1e3     # convert km to m
    alpha_d = args.alpha[0] * 1e3      # sigma in distance dimension
    alpha_t = args.alpha[1] / 12.      # sigma in temporal dimension
    vicol   = args.vnames[:]
    filter_dxy     = args.filter[0] * 1e3      #  spatial distance for filtering
    filter_thres   = args.filter[1]           #  sigma for the outliers filter in a spatial grid
    filter_vmax    = args.filter[2]           #  max height threshold
    nsam           = args.n_sample[0]

    # Print parameters to screen
    print('parameters:')
    for p in vars(args).items(): 
        print(p)

    print("-> reading data ...")
    # Get variable names
    xvar, yvar, zvar, tvar, svar = vicol
    # Load all 1d variables needed
    with h5py.File(ifile, 'r') as fi:

        # Get variables and sub-sample if needed
        lon = fi[xvar][::nsam]
        lat = fi[yvar][::nsam]
        zp  = fi[zvar][::nsam]
        tp  = fi[tvar][::nsam]
        sp  = fi[svar][::nsam] if svar in fi else np.ones(lon.shape)

    xp, yp = coor2coor('4326', proj, lon, lat) # from geo-coords to projection-coords

    # Check if we should filter
    if filter_dxy != 0:
        print('-> filtering data ...')
        # Global filtering before cleaning 
        i_o = np.abs(zp) < filter_vmax
        # Remove all NaNs 
        xp, yp, zp, tp, sp = xp[i_o], yp[i_o], zp[i_o], tp[i_o], sp[i_o]
        # Filter the data in the spatial domain
        zp = spatial_filter(xp.copy(), yp.copy(), zp.copy(), filter_dxy, filter_dxy, sigma=filter_thres)


    Xi, Yi, Ti, Zi, Ei, Ni = interpgaus3d(xp, yp, time=tp, zp=zp, dx=dx, 
                                            dy=dy, dt=dt, sp=sp, radius=radius)

    print('-> saving predictions to file...')

    # Save data to file
    with h5py.File(ofile, 'w') as foo:
        foo['X'] = Xi        #  X, 2-d array, is the projected coord_x with resolution dx
        foo['Y'] = Yi        #  Y, 2-d array, is the projected coord_y with resolution dy
        foo['time'] = Ti     # ti, 1-d array, times series with resolution of tres
        foo['Z_interp'] = Zi
        foo['Z_rmse'] = Ei
        foo['Z_nobs'] = Ni
        foo['epsg'] = int(proj)

