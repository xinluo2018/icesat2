## author: xin luo; created: 2021.8.3
## this code is original built based on captoolkit: 
## https://github.com/fspaolo/captoolkit/blob/master/captoolkit/fittopo.py
"""
spatial-temporal smooth
"""

import warnings
warnings.filterwarnings("ignore")
import os
import h5py
import pyproj
import argparse
import numpy as np
import statsmodels.api as sm
from datetime import datetime
from scipy.spatial import cKDTree
from statsmodels.robust.scale import mad

# Default grid spacing in x and y (km)
DXY = [1, 1]

# Defaul min and max search radius (km)
RADIUS = [1]

# Default min obs within search radius to compute solution
MINOBS = 10

# Default number of iterations for solution
NITER = 5

# Default ref time for solution: 'year' | 'fixed'=full mean t | 'variable'=cap mean t
TREF = 'fixed'  # 

# Default projection EPSG for solution (AnIS=3031, GrIS=3413(Arctic))
PROJ = 3031

# Default data columns (lon,lat,time,height,error,id)
COLS = ['lon', 'lat', 't_sec', 'h_cor', 'h_rms']

# Default expression to transform time variable
EXPR = None   # ??

# Default order of the surface fit model 
ORDER = 2

# Default number of obs. to change to mean solution
MLIM = 10

# Default njobs for parallel processing of *tiles*
NJOBS = 1

# Maximum slope allowed from the solution, replaced by SLOPE
SLOPE = 1.0

# Output description of solution
description = ('Compute surface elevation residuals '
               'from satellite/airborne altimetry.')

# Define command-line arguments
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
        'files', 
        metavar='file', 
        type=str, 
        nargs='+',
        help='file(s) to process (HDF5)')

parser.add_argument(
        '-d', 
        metavar=('dx','dy'), 
        dest='dxy', 
        type=float, 
        nargs=2,
        help=('spatial resolution for grid-solution (deg or km)'),
        default=DXY,)

parser.add_argument(
        '-r', 
        metavar=('radius'), 
        dest='radius', 
        type=float, 
        nargs=1,
        help=('min and max search radius (km)'),
        default=RADIUS,)

parser.add_argument(
        '-q', 
        metavar=('n_reloc'), 
        dest='nreloc', 
        type=int, 
        nargs=1,
        help=('number of relocations for search radius'),
        default=[0],)

parser.add_argument(
        '-i', 
        metavar='n_iter', 
        dest='niter', 
        type=int, 
        nargs=1,
        help=('maximum number of iterations for model solution'),
        default=[NITER],)

parser.add_argument(
        '-z', 
        metavar='min_obs', 
        dest='minobs', 
        type=int, 
        nargs=1,
        help=('minimum obs to compute solution'),
        default=[MINOBS],)

parser.add_argument(
        '-m', 
        metavar=('mod_lim'), 
        dest='mlim', 
        type=int, 
        nargs=1,
        help=('minimum obs for higher order models'),
        default=[MLIM],)

parser.add_argument(
        '-k', 
        metavar=('mod_order'), 
        dest='order', 
        type=int, 
        nargs=1,
        help=('order of the surface fit model: 1=lin or 2=quad'),
        default=[ORDER],)

parser.add_argument(
        '-t', 
        metavar=('ref_time'), 
        dest='tref', 
        type=str, 
        nargs=1,
        help=('time to reference the solution to: year|fixed|variable'),
        default=[TREF],)

parser.add_argument(
        '-j', 
        metavar=('epsg_num'), 
        dest='proj', 
        type=str, 
        nargs=1,
        help=('projection: EPSG number (AnIS=3031, GrIS=3413)'),
        default=[str(PROJ)],)

parser.add_argument(
        '-v', 
        metavar=('x','y','t','h'), 
        dest='vnames', 
        type=str, 
        nargs=4,
        help=('name of lon/lat/t/h in the HDF5'),
        default=COLS,)

parser.add_argument(
        '-x', 
        metavar=('expr'), 
        dest='expr',  
        type=str, 
        nargs=1,
        help="expression to apply to time (e.g. 't + 2000'), optional",
        default=[EXPR],)

parser.add_argument(
        '-n', 
        metavar=('n_jobs'), 
        dest='njobs', 
        type=int, 
        nargs=1,
        help="for parallel processing of multiple tiles, optional",
        default=[NJOBS],)

parser.add_argument(
        '-s', 
        metavar=('slope_lim'), 
        dest='slplim', 
        type=float, 
        nargs=1,
        help="slope limit for x/y direction (deg)",
        default=[SLOPE],)

parser.add_argument(
        '-p', 
        dest='pshow', 
        action='store_true',
        help=('print diagnostic information to terminal'),
        default=False)

args = parser.parse_args()

# Pass arguments
files  = args.files                  # input file(s)
dx     = args.dxy[0] * 1e3           # grid spacing in x (km -> m)
dy     = args.dxy[1] * 1e3           # grid spacing in y (km -> m)
dmax   = args.radius[0] * 1e3        # min search radius (km -> m)
nreloc = args.nreloc[0]              # number of relocations 
nlim   = args.minobs[0]              # min obs for solution (KDTree searched pothon points)
mlim   = args.mlim[0]                # minimum values for parametric model
niter  = args.niter[0]               # number of iterations for solution
tref_  = args.tref[0]                # ref time for solution (d.yr)
proj   = args.proj[0]                # EPSG number (GrIS=3413, AnIS=3031)
icol   = args.vnames[:]              # data input cols (x,y,t,h)
expr   = args.expr[0]                # expression to transform time, if the data are in one specific year (e.g.,2000), 
                                     # and the input time are doy, this var can be set 2000.
njobs  = args.njobs[0]               # for parallel processing of tiles
order  = args.order[0]               # max order of the surface fit model
slplim = args.slplim[0]              # max allowed surface slope in deg.
diag   = args.pshow                  # print diagnostics to terminal.

print('parameters:')
for p in list(vars(args).items()):
    print(p)

def make_grid(xmin, xmax, ymin, ymax, dx, dy):
    """Construct output grid-coordinates.
        aug:
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


def transform_coord(proj1, proj2, x, y):
    """Transform coordinates from proj1 to proj2 (EPSG num)."""

    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+proj1)
    proj2 = pyproj.Proj("+init=EPSG:"+proj2)

    # Convert coordinates
    return pyproj.transform(proj1, proj2, x, y)


def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). 
        mad: median absolutely deviation"""
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)

def binning_1d(x, y, xmin=None, xmax=None, dx=1 / 12.,
                        window=3 / 12., interp=False, median=False):
    """ des: 1-dimentional binning, e.g., time-series data binning.
        args:
            x, y: independent and dependent variable, e.g., time and value of the time series data.
            xmin, xmax: range of the variable x.
            dx: resolution of x. e.g., 1/12 represents 1 month if the unit of x is year.
            window: size of binning window. 3/12 represents 3 month if the unit of x is year
            interp: interpolate binned values to original x points.
            median: median value of the bin values, if not set, the mean value is calculated.
        return: 
            xb, yb: x, y corresponding to in each bin center, or interpolation point
            eb, nb, sb: error, number, sum statics of each bin
    """
    if xmin is None:
        xmin = np.nanmin(x)
    if xmax is None:
        xmax = np.nanmax(x)

    steps = np.arange(xmin, xmax, dx)   # time steps
    bins = [(ti, ti + window) for ti in steps]    #s

    N = len(bins)  
    xb = np.full(N, np.nan)         # times corresponding to bins
    yb = np.full(N, np.nan)         # values corresponding to bins
    eb = np.full(N, np.nan)         # mads corresponding to bins
    nb = np.full(N, np.nan)         # counts of valid values in bins
    sb = np.full(N, np.nan)         # sum of values in bins

    ## loops for each bin
    for i in range(N):
        t1, t2 = bins[i]
        idx, = np.where((x >= t1) & (x <= t2))

        if len(idx) == 0:
            xb[i] = 0.5 * (t1 + t2)     # determine the time of bin as the center of the time window
            continue                    # finish this loop       

        ybv = y[idx]                    # values in specific bin

        if median:
            yb[i] = np.nanmedian(ybv)
        else:
            yb[i] = np.nanmean(ybv)

        xb[i] = 0.5 * (t1 + t2)         # determine the time of the bin
        eb[i] = mad_std(ybv)            # mad of the bin, represent the error of the bine
        nb[i] = np.sum(~np.isnan(ybv))  # counts of the valid values in bin
        sb[i] = np.sum(ybv)             # sum of the values in bin

    if interp:  
        try:
            yb = np.interp(x, xb, yb)     ## interpolate the values to the given time in the bin
            eb = np.interp(x, xb, eb)     ## interpolate the mad ... 
            sb = np.interp(x, xb, sb)     ## interpolate the sum of the values ...
            xb = x
        except:
            pass

    return xb, yb, eb, nb, sb

def get_radius_idx(x, y, x0, y0, r, Tree, n_reloc=0, min_months=12, 
                    max_reloc=3, time=None, height=None, 
                    dtime=1/12, window_time=1/12):

    """ des: get indices of data points inside radius. 
             search order: spatial --> temporal
             if time is not None, the selected points should be binned within specific time intervals.
        input:
            x, y: coordinates of the data points
            x0, y0: center coordinates for searching
            r: radius
            Tree: spatial.KDTree(points)
            n_reloc: relocation times
            min_months: count threshold for data points of the monthly bins 
            max_reloc: maximum iteration for the new center determination, make more slections of photon points
            time, height (optional): time and height data for photon points corresponding to x,y
            dtime, window_time: time interval and time window, work when time,height are given.
        return:
            idx: index of the selected data points
    """

    # Query the Tree from the center of cell, return the selected points
    idx = Tree.query_ball_point((x0, y0), r)

    if len(idx) < 2:
        return idx  

    if time is not None:   # 
        n_reloc = max_reloc

    if n_reloc < 1:
        return idx   
    
    # relocation: obtain neighbor photon points as many as possible, however:
    # !! the new obtained photon points may farther than given radius relative to (x0,y0)
    for k in range(n_reloc):

        x0_new, y0_new = np.median(x[idx]), np.median(y[idx])
        reloc_dist = np.hypot(x0_new-x0, y0_new-y0)      # distance from relocated points
        if reloc_dist > r:
            break       # skip this loop
        idx = Tree.query_ball_point((x0_new, y0_new), r)    # update the idx of neighbors
        if n_reloc == k+1:
            break

        # If time provided, keep relocating until time-coverage is sufficient 
        if time is not None:
            t_b, height_b = binning_1d(time[idx], height[idx], dx=dtime, window=window_time)[:2]
            # ensure the number of months that have valid points.
            if np.sum(~np.isnan(height_b)) >= min_months: 
                break

    return idx


def rlsq(x, y, n=1):
    """ Fit a robust polynomial
        input:
            x: independent variable
            y: dependent variable
            n: the degree, in y = b0*x0+b1*x1+b2*x2...*bn*xn, the n is the count of x
        return:
            p: fiting parameters, if n=1:
               p[0] is the first-order derivative(i.e. slope of the data points) of the fit function.
            s: mad (median absolutely deviation) of fiting resiuals 
    """
    # Test solution
    if len(x[~np.isnan(y)]) <= (n + 1):

        if n == 0:   # 
            p = np.nan
            s = np.nan
        else:
            p = np.zeros((1, n)) * np.nan
            s = np.nan

        return p, s

    # Empty array
    A = np.empty((0, len(x)))  #
    i = 0     # Create counter for order of x, 0 <= i <= n 

    if n > 1: 
        x -= np.nanmean(x)    # Center x-axis

    # Special case 
    if n == 0:   
        # Mean offset, constant value in the polynomial function
        A = np.ones(len(x)) 

    else:     # i.e. n == 1 or n > 1 
        while i <= n:
            # Stack coefficients: x0->1 (constant coeff), x1->x^2, x2->x^3..., xn->x^n
            A = np.vstack((A, x ** i))
            i += 1        # Update counter

    # Test to see if we can solve the system
    try:
        # Robust least squares fit, 'drop':drop the nan values in observations,
        # RLM.fit use iteratively reweighted least squares.
        fit = sm.RLM(y, A.T, missing='drop').fit(maxiter=5, tol=0.001)
        p = fit.params
        s = mad_std(fit.resid)   # The residuals of the fitted model. y - fittedvalues
    except:  # can't perform fitting by sm.rlm()
        if n == 0:
            p = np.nan
            s = np.nan
        else:
            p = np.zeros((1, n)) * np.nan
            s = np.nan
    ## return inverted p: [b_n, b_n-1,..., b_0]
    return p[::-1], s


# Main function for computing parameters
def main(ifile, n=''):
        
    # Check for empty file
    if os.stat(ifile).st_size == 0:
        print('input file is empty!')
        return
    
    # Start timing of script
    startTime = datetime.now()

    print('loading data ...')

    # Determine input file type
    if not ifile.endswith(('.h5', '.H5', '.hdf', '.hdf5')):
        print("Input file must be in hdf5-format")
        return
    
    # Input variables
    xvar, yvar, tvar, zvar = icol
    
    # Load all 1d variables needed
    with h5py.File(ifile, 'r') as fi:

        lon = fi[xvar][:]
        lat = fi[yvar][:]
        time = fi[tvar][:]
        height = fi[zvar][:]

    # EPSG number for lon/lat proj
    projGeo = '4326'

    # EPSG number for grid proj
    projGrd = proj

    print('converting lon/lat to projected x/y ...')

    # Convert into stereographic coordinates
    (x, y) = transform_coord(projGeo, projGrd, lon, lat)

    # Get bbox from data
    (xmin, xmax, ymin, ymax) = x.min(), x.max(), y.min(), y.max()

    # Apply transformation (according give trans value, e.g.,+2000) to time; 
    if expr: 
        # function eval('1+1') return 2
        time = eval(expr.replace('t', 'time'))

    # Overall (fixed) mean time
    t_mean = np.round(np.nanmean(time), 2)

    # Grid solution - defined by nodes
    (Xi, Yi) = make_grid(xmin, xmax, ymin, ymax, dx, dy)

    # Flatten prediction grid
    xi = Xi.ravel()
    yi = Yi.ravel()

    # Zip data to vector: [(x0,y0),(x1,y1)...(xn,yn)]
    coord_proj = list(zip(x.ravel(), y.ravel()))

    # Construct cKDTree
    print('building the k-d tree ...')
    Tree = cKDTree(coord_proj)

    # Create output containers
    dh_topo = np.full(height.shape, np.nan)   # 
    de_topo = np.full(height.shape, 999999.)  
    mi_topo = np.full(height.shape, np.nan)
    hm_topo = np.full(height.shape, np.nan)
    sx_topo = np.full(height.shape, np.nan)
    sy_topo = np.full(height.shape, np.nan)
    tr_topo = np.full(height.shape, np.nan)
    
    # Set slope limit
    slp_lim = np.tan(np.deg2rad(slplim))   # unit: meter
    
    # Enter prediction loop, looping for locations (for up-left) in the generated grids
    print('predicting values ...')
    for i in range(len(xi)):
        x0, y0 = xi[i], yi[i]   # (x0,y0) is the points corresponding to up-left coords of the grid

        # Get indexes of data within search radius or cell bbox
        idx = get_radius_idx(
                    x, y, x0, y0, dmax, Tree, n_reloc = nreloc,
                    min_months = 12, max_reloc = 3, time = None, height = None)

        # Length of data in search cap
        nobs = len(x[idx])
            
        # Check data density
        if (nobs < nlim): 
            continue

        # Parameters for model-solution, cap corresponding to KDTree searching region
        xcap = x[idx]
        ycap = y[idx]
        tcap = time[idx]
        hcap = height[idx]

        # Copy original height vector
        h_org = hcap.copy()

        # Centroid node
        xc = np.median(xcap)
        yc = np.median(ycap)

        if tref_ == 'fixed':   # determined by all the points in the file
            tref = t_mean
        elif tref_ == 'variable':
            tref = np.nanmean(tcap)  # determined by the selected nearest points
        else:
            tref = np.float(tref_)

        # Design matrix elements: multivariate fitting, var: x,y,time
        c0 = np.ones(len(xcap))   # constant coefficient
        c1 = xcap - xc      # coords_x centralization, why not distance?
        c2 = ycap - yc      # coords_y centralization
        c3 = c1 * c2    # second order var
        c4 = c1 * c1    # second order var
        c5 = c2 * c2    # second order var

        c6 = tcap - tref    # time centrolization

        # Length before editing
        nb = len(hcap)

        # ----- Determine model order
        # 1. Biquadratic (order == 2, var: x,y,time), prefered
        if order == 2 and nb >= mlim * 2:
            # Biquadratic surface and linear trend
            Acap = np.vstack((c0, c1, c2, c3, c4, c5, c6)).T
            mi = 1           #  Model identifier

        # 2. Bilinear (order == 1, var: x,y,time), second choice
        elif nb >= mlim: 
            # Bilinear surface and linear trend
            Acap = np.vstack((c0, c1, c2, c6)).T
            mi = 2       

        # 3. linear for x and y respectively (order == 0, var: x,y)
        else:
            mi = 3  
        
        ## --------- Modelled topography: fitting --> prediction
        ## 1. Biquadratic prediction
        if mi == 1:
            # Construct model object: Robust linear models
            linear_model = sm.RLM(hcap, Acap, M=sm.robust.norms.HuberT(), missing='drop')
            linear_model_fit = linear_model.fit(maxiter=niter, tol=0.001)           
            # Coefficients
            Cm = linear_model_fit.params
            # Prediction
            h_model = np.dot(np.vstack((c0, c1, c2, c3, c4, c5)).T, Cm[[0, 1, 2, 3, 4, 5]])
            # Compute along x and y slopes
            # Cm[1]: the slope ratio along the x; 
            slope_x = np.sign(Cm[1]) * slp_lim if np.abs(Cm[1]) > slp_lim else Cm[1]
            # Cm[2]: the slope ratio along the y
            slope_y = np.sign(Cm[2]) * slp_lim if np.abs(Cm[2]) > slp_lim else Cm[2]
            h_avg = Cm[0]           #  Mean height, due to the centralization of coords and time

        ## 2. linear prediction        
        elif mi == 2:            
            linear_model = sm.RLM(hcap, Acap, M=sm.robust.norms.HuberT(), missing='drop')
            linear_model_fit = linear_model.fit(maxiter=niter, tol=0.001)   # fit           
            Cm = linear_model_fit.params       # Coefficients            
            h_model = np.dot(np.vstack((c0, c1, c2)).T, Cm[[0, 1, 2]])  # Prediction

            # Compute along and across track slope
            slope_x = np.sign(Cm[1]) * slp_lim if np.abs(Cm[1]) > slp_lim else Cm[1]
            slope_y = np.sign(Cm[2]) * slp_lim if np.abs(Cm[2]) > slp_lim else Cm[2]

            # Mean height
            h_avg = Cm[0]

        # 3. linear fitting for coord_x and coord_y respectively. not consider time
        else:            
            # Mean surface from median
            h_avg = np.median(hcap)
            ## Compute centroid coord            
            x_centr = (xcap - xc) + 1e-3      # ?? why + 1e-3
            y_centr = (ycap - yc) + 1e-3
            h_centr = h_org - h_avg           # Center surface height
            # Compute along-x and -y slope, p: estimated slope
            px, rms_x = rlsq(x_centr, h_centr, 1)    # fitting between dh and dx
            py, rms_x = rlsq(y_centr, h_centr, 1)   
            # Set along-x and -y slope
            slope_x = 0 if np.isnan(px[0]) else px[0]
            # Set across-track slope
            slope_y = 0 if np.isnan(py[0]) else py[0]
            # Compute along-x and -y slope, filtering by using slp_lim
            slope_x = np.sign(slope_x) * slp_lim if np.abs(slope_x) > slp_lim else slope_x
            slope_y = np.sign(slope_y) * slp_lim if np.abs(slope_y) > slp_lim else slope_y            
            # Compute the surface height correction
            h_model = h_avg + (slope_x * x_centr) + (slope_y * y_centr)

        # Compute full slope (along track)
        slope = np.arctan(np.sqrt(slope_x**2 + slope_y**2)) * (180 / np.pi)   # unit: degree (0-360)
        # Compute residual (measured height and fitted height)
        dh = h_org - h_model
        # RMSE of the residuals
        MAD_dh = mad_std(dh)
        # Overwrite errors
        iup = MAD_dh < de_topo[idx]      # mask: obtain the predicted point with the smallest error

        # Create temporary variables
        dh_cap = dh_topo[idx].copy()
        de_cap = de_topo[idx].copy()
        hm_cap = hm_topo[idx].copy()
        mi_cap = mi_topo[idx].copy()
        tr_cap = tr_topo[idx].copy()
        
        # Update variables
        dh_cap[iup] = dh[iup]
        de_cap[iup] = MAD_dh
        hm_cap[iup] = h_avg 
        mi_cap[iup] = mi
        tr_cap[iup] = tref
      
        # Update with current solution
        dh_topo[idx] = dh_cap
        de_topo[idx] = de_cap
        hm_topo[idx] = hm_cap
        mi_topo[idx] = mi_cap
        tr_topo[idx] = tr_cap
        sx_topo[idx] = np.arctan(slope_x) * (180 / np.pi)   # slope_x
        sy_topo[idx] = np.arctan(slope_y) * (180 / np.pi)
       
        # Print progress (every N iterations)
        if (i % 1000) == 0 and diag is True:
            # Print message every i:th solution
            print(('%s %i %s %2i %s %i %s %03d %s %.3f %s %.3f' % \
                    ('#', i, '/',len(xi),'Model:',mi,'Nobs:',nb,'Slope:',\
                    np.around(slope,3),'Residual:',np.around(mad_std(dh),3))))

    # Print percentage of not filled
    print(('Total NaNs (percent): %.2f' % \
            (100 * float(len(dh_topo[np.isnan(dh_topo)])) / float(len(dh_topo)))))

    # Print percentage of each model
    one = np.sum(mi_topo == 1)
    two = np.sum(mi_topo == 2)
    tre = np.sum(mi_topo == 3)
    N = float(len(mi_topo))

    print(('Model types (percent): 1 = %.2f, 2 = %.2f, 3 = %.2f' % \
            (100 * one/N, 100 * two/N, 100 * tre/N)))
  
    # Append new columns to original file
    with h5py.File(ifile, 'a') as fi:

        # Check if we have variables in file
        try:
            
            # Save variables
            fi['h_res'] = dh_topo
            fi['h_mod'] = hm_topo
            fi['e_res'] = de_topo
            fi['m_i'] = mi_topo
            fi['t_ref'] = tr_topo
            fi['slp_x'] = sx_topo
            fi['slp_y'] = sy_topo

        except:
            
            # Update variables
            fi['h_res'][:] = dh_topo
            fi['h_mod'][:] = hm_topo
            fi['e_res'][:] = de_topo
            fi['m_i'][:] = mi_topo
            fi['t_ref'][:] = tr_topo
            fi['slp_x'][:] = sx_topo
            fi['slp_y'][:] = sy_topo

    # Rename file
    if ifile.find('TOPO') < 0:
        os.rename(ifile, ifile.replace('.h5', '_TOPO.h5'))
    
    # Print some statistics
    print(('*' * 75))
    print(('%s %s %.5f %s %.2f %s %.2f %s %.2f %s %.2f' % \
        ('Statistics',
         'Mean:', np.nanmedian(dh_topo),
         'Std:', mad_std(dh_topo),
         'Min:', np.nanmin(dh_topo),
         'Max:', np.nanmax(dh_topo),
         'Error:', np.nanmedian(de_topo[dh_topo!=999999]),)))
    print(('*' * 75))
    print('')

    # Print execution time of algorithm
    print(('Execution time: '+ str(datetime.now()-startTime)))

if njobs == 1:
    print('running sequential code ...')
    [main(f, n) for n,f in enumerate(files)]

else:
    print(('running parallel code (%d jobs) ...' % njobs))
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f, n) for n, f in enumerate(files))

    '''
    from dask import compute, delayed
    from distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=8, threads_per_worker=None,
                          scheduler_port=8002, diagnostics_port=8003)
    client = Client(cluster)  # connect to cluster
    print client
    #values = [delayed(main)(f) for f in files]
    #results = compute(*values, get=client.get)
    values = [client.submit(main, f) for f in files]
    results = client.gather(values)
    '''
