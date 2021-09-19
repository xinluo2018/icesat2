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
        '-v', metavar=('x','y','z','t'), dest='vnames', type=str, nargs=4,
        help=('name of varibales in the HDF5-file'),
        default=['lon','lat','h_cor','t_year'],)


def make_grid(xmin, xmax, ymin, ymax, dx, dy):
    """des: construct output grid-coordinates.
        arg:
            xmin,xmax,ymin,ymax: the range of x and y.
            dx,dy: the resolution in terms of x and y.
        return:
            x, y: np.array, shape equals the generated grid shape
    """
    # Setup grid dimensions
    Nn = int((np.abs(ymax - ymin)) / dy) + 1    # row
    Ne = int((np.abs(xmax - xmin)) / dx) + 1    # col
    # Initiate x/y vectors for grid
    x_i = np.linspace(xmin, xmax, num=Ne)       # image coordinate
    y_i = np.linspace(ymin, ymax, num=Nn)
    return np.meshgrid(x_i, y_i, indexing='xy')


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

def interpgaus3d(x, y, t, z, \
                xi, yi, ti, alpha_d, alpha_t):
    '''
    des: 3-d interpolation by using gaussian method
    '''
    d_time = np.abs(t - ti)    # time difference from all the points.            
    d_spat = square_dist(x, y, xi, yi)   # distance from interpolated point.
    # Compute the weighting factors, larger dist,dt, smaller ed,et
    # !!!alpha_d, alpha_t are actually the sigma in gaussian distribution function
    ed, et = make_weights(d_spat, d_time, alpha_d, alpha_t)
    # Combine weights and scale with error, similar to the joint probability density function
    w = ed * et            
    w += 1e-6    # avoid division of zero
    zi = fwavg(w, z)     #  weighted mean height
    sigma_r = fwstd(w, z, zi)    # Compute weighted height std 
    ei = np.sqrt(sigma_r ** 2)   # prediction error at grid node (interpolated point)
    ni = len(z)                  # Number of obs. in solution
    return zi, ei, ni


if __name__ == '__main__':
    '''
    des: 
        1. spatial filtering for the points;
        2.remote the nan values; 
        3. interpolation of the generated grid cross points.
    '''

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
    xvar, yvar, zvar, tvar = vicol
    # Load all 1d variables needed
    with h5py.File(ifile, 'r') as fi:
        # Get variables and sub-sample if needed
        lon = fi[xvar][::nsam]
        lat = fi[yvar][::nsam]
        zp  = fi[zvar][::nsam]
        tp  = fi[tvar][::nsam]
    xp, yp = coor2coor('4326', proj, lon, lat) # from geo-coords to projection-coords

    # Check if we should filter
    if filter_dxy != 0:
        print('-> filtering data ...')
        # Global filtering before cleaning 
        i_o = np.abs(zp) < filter_vmax
        # Remove all NaNs 
        xp, yp, zp, tp = xp[i_o], yp[i_o], zp[i_o], tp[i_o]
        # Filter the data in the spatial domain
        zp = spatial_filter(xp.copy(), yp.copy(), zp.copy(), filter_dxy, filter_dxy, sigma=filter_thres)

    # Find all NaNs and do not select them
    no_nan = ~np.isnan(zp)
    # Remove data wiht NaN's
    xp, yp, zp, tp = xp[no_nan], yp[no_nan], zp[no_nan], tp[no_nan]

    xmin, xmax, ymin, ymax = xp.min(), xp.max(), yp.min(), yp.max()
    t_start, t_end = tp.min(), tp.max()

    # Time vector
    Ti = np.arange(t_start, t_end + dt, dt)
    # Construct the spatial grid: the cross point (Xi, Yi) can be regarded as interpolated point
    Xi, Yi = make_grid(xmin, xmax, ymin, ymax, dx, dy)
    # Shape of grid
    Nt = len(Ti)
    # Shape of grid
    Nx, Ny = Xi.shape

    # Output 3-D vectors: (row, col, time)
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
            x_local = xp[idx]
            y_local = yp[idx]
            z_local = zp[idx]
            t_local = tp[idx]
            for k in range(int(Nt)):   # loop through time 
                zi, ei, ni = interpgaus3d(x=x_local, y=y_local, t=t_local, z=z_local, \
                                          xi=Xi[i,j], yi=Yi[i,j], ti=Ti[k], alpha_d=alpha_d, alpha_t=alpha_t)
                # Test for signular values 
                if np.abs(zi) < 1e-6: 
                    continue
                ## Save data to output (interpolated point: (k,i,j) )
                Zi[i,j,k] = zi    # interpolated height
                Ei[i,j,k] = ei    # interpolated error
                Ni[i,j,k] = ni    # number of points in the interpolation.

    print('-> saving predictions to file...')

    # Save data to file
    with h5py.File(ofile, 'w') as foo:
        foo['X'] = Xi        #  X, 2-d array, is the projected coord_x with resolution dx
        foo['Y'] = Yi        #  Y, 2-d array, is the projected coord_y with resolution dy
        foo['time'] = Ti     #  ti, 1-d array, times series with resolution of tres
        foo['Z_interp'] = Zi
        foo['Z_rmse'] = Ei
        foo['Z_nobs'] = Ni
        foo['epsg'] = int(proj)

