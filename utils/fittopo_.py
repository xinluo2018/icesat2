## author: Fernando Paolo
## modify: xin luo, 2021.8.3

"""
des: topograpy detrending (spatial smooth) through a robust polynomial fitting.
"""

import warnings
warnings.filterwarnings("ignore")
import os
import h5py
import argparse
import numpy as np
import statsmodels.api as sm
from datetime import datetime
from scipy.spatial import cKDTree
from helper import make_grid, mad_std
from transform_xy import coor2coor

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
COLS = ['lon', 'lat', 'h_elv', 't_year', 'flag']
# Default order of the surface fit model 
ORDER = 2
# Default number of obs. to change to mean solution
MLIM = 10
# Default njobs for parallel processing of *tiles*
NJOBS = 1
# Maximum slope allowed from the solution, replaced by SLOPE
SLOPE = 10.0

# Output description of solution
description = ('Compute surface elevation residuals '
               'from satellite/airborne altimetry.')

# Define command-line arguments
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
        'files', metavar='file', type=str, nargs='+',
        help='file(s) to process (HDF5)')

parser.add_argument(
        '-d', metavar=('dx','dy'), dest='dxy', type=float, nargs=2,
        help=('spatial resolution for grid-solution (deg or km)'),
        default=DXY)

parser.add_argument(
        '-r', metavar=('radius'), dest='radius', type=float, nargs=1, 
        help=('min and max search radius (km)'),
        default=RADIUS)

parser.add_argument(
        '-i', metavar='n_iter', dest='niter', type=int, nargs=1,
        help=('maximum number of iterations for model solution'),
        default=[NITER],)

parser.add_argument(
        '-z', metavar='min_obs', dest='minobs', type=int, nargs=1,
        help=('minimum obs to compute solution'),
        default=[MINOBS],)

parser.add_argument(
        '-m', metavar=('mod_lim'), dest='mlim', type=int, nargs=1,
        help=('minimum obs for higher order models'),
        default=[MLIM],)

parser.add_argument(
        '-k', metavar=('mod_order'), dest='order', type=int, nargs=1,
        help=('order of the surface fit model: 1=lin or 2=quad'),
        default=[ORDER],)

parser.add_argument(
        '-t', metavar=('ref_time'), dest='tref', type=str, nargs=1,
        help=('time to reference the solution to: year|fixed|variable'),
        default=[TREF],)

parser.add_argument(
        '-j', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
        help=('projection: EPSG number (AnIS=3031, GrIS=3413)'),
        default=[str(PROJ)],)

parser.add_argument(
        '-v', metavar=('x','y','h','t'), dest='vnames', type=str, nargs=4,
        help=('name of lon/lat/h/t in the HDF5'),
        default=COLS,)

parser.add_argument(
        '-n', metavar=('n_jobs'), dest='njobs', type=int, nargs=1,
        help="for parallel processing of multiple tiles, optional",
        default=[NJOBS],)

parser.add_argument(
        '-s', metavar=('slope_lim'), dest='slplim', type=float, nargs=1,
        help="slope limit for x/y direction (deg)",
        default=[SLOPE],)

parser.add_argument(
        '-p', dest='pshow', action='store_true',
        help=('print diagnostic information to terminal'),
        default=False)


# get arguments
args = parser.parse_args()
files  = args.files                  # input file(s)
dx     = args.dxy[0] * 1e3           # grid spacing in x (km -> m)
dy     = args.dxy[1] * 1e3           # grid spacing in y (km -> m)
dmax   = args.radius[0] * 1e3        # min search radius (km -> m)
nlim   = args.minobs[0]              # min obs for solution (KDTree searched pothon points)
mlim   = args.mlim[0]                # minimum values for parametric model
niter  = args.niter[0]               # number of iterations for solution
tref_  = args.tref[0]                # ref time for solution (d.yr)
proj   = args.proj[0]                # EPSG number (GrIS=3413, AnIS=3031)
icol   = args.vnames[:]              # data input cols (x,y,h,t)
njobs  = args.njobs[0]               # for parallel processing of tiles
order  = args.order[0]               # max order of the surface fit model
slplim = args.slplim[0]              # max allowed surface slope in deg.
diag   = args.pshow                  # print diagnostics to terminal.

print('parameters:')
for p in list(vars(args).items()):
    print(p)


def rlsq(x, y, n=1):
    """ des: robust polynomial fitting by using statsmodels.api.sm.RLM()
        input:
            x: independent variable
            y: dependent variable
            n: the degree, in y=b0*x0+b1*x1+b2*x2...*bn*xn, the n is the count of x (exclusion of x0)
        return:
            p: fiting parameters, if n=1:
               p[0] is the first-order derivative (i.e. slope) of the fit curve. 
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

    # Empty array, A is independent vars
    A = np.empty((0, len(x)))   #
    i = 0       # Create counter for n-th order of x, 0 <= i <= n 

    if n > 1: 
        x -= np.nanmean(x)    # center x-axis

    # Special case 
    if n == 0:   
        # fitting order is 0, only have constant coeffs in the fitting function.
        A = np.ones(len(x)) 

    else:      # i.e. n == 1 or n > 1 
        while i <= n:
            # Stack coefficients: x0->1 (constant coeff), x1->x^2, x2->x^3..., xn->x^n
            A = np.vstack((A, x ** i))   # i=0 -> A is constant coeffs.
            i += 1        # Update counter

    # Test to see if we can solve the system
    try:
        # Robust least squares fit, 'drop': drop the nan values in observations,
        # RLM.fit() use iteratively reweighted least squares.
        fit = sm.RLM(y, A.T, missing='drop').fit(maxiter=5, tol=0.0001)
        p = fit.params
        s = mad_std(fit.resid)   # The residuals of the fitted model. y - fittedvalues
    except:  # can't perform fitting by sm.rlm()
        if n == 0:
            p = np.nan
            s = np.nan
        else:
            p = np.zeros((1, n)) * np.nan
            s = np.nan
    ## inverted p: [b_n, b_n-1,..., b_0]
    return p[::-1], s



# Main function for computing parameters
def fittopo(ifile):            
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
    xvar, yvar, zvar, tvar = icol    
    # Load all 1d variables needed
    with h5py.File(ifile, 'r') as fi:
        lon = fi[xvar][:]
        lat = fi[yvar][:]
        height = fi[zvar][:]
        time = fi[tvar][:]

    # EPSG number for lon/lat proj
    projGeo = '4326'
    # EPSG number for grid proj
    projGrd = proj
    print('converting lon/lat to projected x/y ...')
    # Convert into stereographic coordinates
    (x, y) = coor2coor(projGeo, projGrd, lon, lat)
    # Get bbox from data
    (xmin, xmax, ymin, ymax) = x.min(), x.max(), y.min(), y.max()
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
    dh_topo = np.full(height.shape, np.nan)    # residual between predicted and original height
    de_topo = np.full(height.shape, 999999.)   # mad of dh
    mi_topo = np.full(height.shape, np.nan)    # model for the data fitting
    hm_topo = np.full(height.shape, np.nan)    # mean (median) value in the searching space.
    sx_topo = np.full(height.shape, np.nan)    # fitting slope in x-axis
    sy_topo = np.full(height.shape, np.nan)    # fitting slope in y-axis
    tr_topo = np.full(height.shape, np.nan)    # time corresponding to fitting data
    
    # Set slope limit
    slp_lim = np.tan(np.deg2rad(slplim))   # unit: meter
    # Enter prediction loop, looping for locations (for up-left) in the generated grids
    print('predicting values ...')
    for i in range(len(xi)):
        x0, y0 = xi[i], yi[i]   # (x0,y0) is the points corresponding to up-left coords of the grid
        # Get indexes of data within search radius or cell bbox
        idx = Tree.query_ball_point((x0, y0), dmax)
        # Length of data in search cap
        nobs = len(x[idx])            
        # Check data density
        if (nobs < nlim): 
            continue
        # Parameters for model-solution, cap corresponding to KDTree searching region
        xcap = x[idx]
        ycap = y[idx]
        hcap = height[idx]
        tcap = time[idx]
        # Copy original height vector
        h_org = hcap.copy()
        # Centroid node (median value)
        xc = np.median(xcap)
        yc = np.median(ycap)
        if tref_ == 'fixed':         # determined by all the points in the file
            tref = t_mean
        elif tref_ == 'variable':
            tref = np.nanmean(tcap)  # determined by the searching region points
        else:
            tref = np.float(tref_)   # given time
        # Design matrix elements: multivariate fitting, var: x,y,time
        c0 = np.ones(len(xcap))   # constant coefficient
        c1 = xcap - xc      # coords_x centralization
        c2 = ycap - yc      # coords_y centralization
        c3 = c1 * c2        # second order var: coord_x * coord-y
        c4 = c1 * c1        # second order var: coord_x * coord_x
        c5 = c2 * c2        # second order var: coord_y * coord_y
        c6 = tcap - tref    # time centrolization
        # Length before editing
        nb = len(hcap)
        ## ----- Determine model (order)

        # 1. Biquadratic (order == 2, var: x, y, time), prefered
        if order == 2 and nb >= mlim * 2:
            # Biquadratic surface and linear trend
            Acap = np.vstack((c0, c1, c2, c3, c4, c5, c6)).T
            mi = 1              #  Model identifier

        # 2. Bilinear (order == 1, var: x,y,time), second choice
        elif nb >= mlim: 
            # Bilinear surface and linear trend
            Acap = np.vstack((c0, c1, c2, c6)).T
            mi = 2       

        # 3. linear for x and y respectively (order == 0, var: x,y)
        else:
            mi = 3  
        
        ## ------- Modelled topography: 1) fitting --> 2) prediction
        ## 1. Biquadratic prediction
        if mi == 1:
            # Construct model object: Robust linear models
            model = sm.RLM(hcap, Acap, M=sm.robust.norms.HuberT(), missing='drop')
            model_fit = model.fit(maxiter=niter, tol=0.0001)           
            # Coefficients
            Cm = model_fit.params
            # Prediction by fitted model
            h_model = np.dot(np.vstack((c0, c1, c2, c3, c4, c5)).T, Cm[[0, 1, 2, 3, 4, 5]])
            # Compute along x and y slopes
            # Cm[1]: the slope ratio along the x; 
            slope_x = np.sign(Cm[1]) * slp_lim if np.abs(Cm[1]) > slp_lim else Cm[1]
            # Cm[2]: the slope ratio along the y
            slope_y = np.sign(Cm[2]) * slp_lim if np.abs(Cm[2]) > slp_lim else Cm[2]
            h_avg = Cm[0]           #  Mean height, due to the centralization of coords and time

        ## 2. linear prediction        
        elif mi == 2:            
            model = sm.RLM(hcap, Acap, M=sm.robust.norms.HuberT(), missing='drop')
            model_fit = model.fit(maxiter=niter, tol=0.0001)   # fitting           
            Cm = model_fit.params           # Coefficients            
            h_model = np.dot(np.vstack((c0, c1, c2)).T, Cm[[0, 1, 2]])  # Prediction
            # Compute along and across track slope
            slope_x = np.sign(Cm[1]) * slp_lim if np.abs(Cm[1]) > slp_lim else Cm[1]
            slope_y = np.sign(Cm[2]) * slp_lim if np.abs(Cm[2]) > slp_lim else Cm[2]
            # Mean height
            h_avg = Cm[0]

        ## 3. searched points is very less. Then only fit in term of coord_x and coord_y,
        ##    and not consider time, if the searched points still can't support fitting,
        ##    the fitting params is set Nan by rlsq function.
        else:            
            # Mean surface from median
            h_avg = np.median(hcap)
            # Compute along-x and -y slope, p: estimated slope
            px, rms_x = rlsq(c1, hcap, 1)    # fitting between dh and dx
            py, rms_x = rlsq(c2, hcap, 1)   
            # Set along-x and -y slope
            slope_x = 0 if np.isnan(px[0]) else px[0]
            # Set across-track slope
            slope_y = 0 if np.isnan(py[0]) else py[0]
            # Compute along-x and -y slope, filtering by using slp_lim
            slope_x = np.sign(slope_x) * slp_lim if np.abs(slope_x) > slp_lim else slope_x
            slope_y = np.sign(slope_y) * slp_lim if np.abs(slope_y) > slp_lim else slope_y            
            # Compute the surface height correction
            h_model = h_avg + (slope_x * c1) + (slope_y * c2)

        # Compute full slope (along track)
        slope = np.arctan(np.sqrt(slope_x**2 + slope_y**2)) * (180 / np.pi)   # unit: degree (0-360)
        # Compute residual (measured height and fitted height)
        dh = h_org - h_model
        # MAD of the residuals
        MAD_dh = mad_std(dh)
        # Overwrite errors, iup: i_update
        iup = MAD_dh < de_topo[idx]      # mask: update the point with the smaller error

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
        sx_topo[idx] = np.arctan(slope_x) * (180 / np.pi)     # slope_x (degree)
        sy_topo[idx] = np.arctan(slope_y) * (180 / np.pi)     # slope_y (degree)
       
        # Print progress (every N iterations)
        if (i % 2000) == 0 and diag is True:
            # Print message every i:th solution
            print(('%s %i %s %2i %s %i %s %03d %s %.3f %s %.3f' % \
                    ('#', i, '/',len(xi),'Model:',mi,'Nobs:',nb,'Slope:',\
                    np.around(slope,3),'Residual:',np.around(mad_std(dh),3))))

    # # Print percentage of not filled
    print(('Total NaNs (percent): %.2f' % \
            (100 * float(len(dh_topo[np.isnan(dh_topo)])) / float(len(dh_topo)))))

    # Print percentage of each model
    one = np.sum(mi_topo == 1)
    two = np.sum(mi_topo == 2)
    tre = np.sum(mi_topo == 3)
    N = float(len(mi_topo))

    print(('Model types (percent): 1 = %.2f, 2 = %.2f, 3 = %.2f' % \
            (100 * one/N, 100 * two/N, 100 * tre/N)))

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

    with h5py.File(ifile.replace('.h5', '_topo.h5'), 'w') as fi:
        fi['lat'] = lat
        fi['lon'] = lon
        fi['h_org'] = height
        fi['t_year'] = time
        fi['h_fit'] = hm_topo
        fi['h_res'] = dh_topo
        fi['e_res'] = de_topo
        fi['m_i'] = mi_topo
        fi['t_ref'] = tr_topo
        fi['slp_x'] = sx_topo
        fi['slp_y'] = sy_topo


if __name__ == '__main__':

    if njobs == 1:
        print('running sequential code ...')
        [fittopo(f) for f in files]

    else:
        print(('running parallel code (%d jobs) ...' % njobs))
        from joblib import Parallel, delayed
        Parallel(n_jobs=njobs, verbose=5)(delayed(fittopo)(f) for f in files)


