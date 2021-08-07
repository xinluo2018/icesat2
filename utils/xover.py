

"""
Program for computing satellite crossovers from ascending and 
descending orbits using linear or cubic interpolation. 
The program takes as input two files separated into asc. and des. orbits 
with a minimum number of variables: orbit number, lon ,lat, time and 
height. The user can provide three extra variables if needed. For radar 
this might be waveform parameters needed to perform account for changed in 
the scattering regime.
The software uses internal tiling to speed up computation of the crossovers,
with dimensions specified by the user. It can further process data divided 
into external tiles provided by "tile.py". To further speed up processing the 
user can downsample the tracks. This allows for a faster computation of the
crossing point, but the difference is computed using the native sampling. 
Notes:
    For external tile processing please use "tile.py" with the same extent 
    for the A and D files. This as the program uses the tile numbering to 
    determine which of the tiles should be crossed together.
    
    When running in external tile-mode the saved file with crossovers
    will be appended with "_XOVERS_AD/DA". Please use "_A" or "_D" in the
    filename to indicate Asc or Des tracks when running in tile mode. 
    
Example:
    python xover.py a.h5 d.h5 -o xover.h5 -r 350 -p 3031 -d 100 -k 1 1\
    -m linear -v orb lon lat time height dum dum dum -b 10
    
    python xover.py ./tiles/*_a.h5 ./tiles/*_d.h5 -r 350 -p 3031 -d 100 -k\
    1 1 -m linear -v orb lon lat time height dum dum dum -f -b 10
    
Credits:
    captoolkit - JPL Cryosphere Altimetry Processing Toolkit
    Johan Nilsson (johan.nilsson@jpl.nasa.gov)
    Fernando Paolo (paolofer@jpl.nasa.gov)
    Alex Gardner (alex.s.gardner@jpl.nasa.gov)
    Jet Propulsion Laboratory, California Institute of Technology
"""


import os
import sys
import glob
import numpy as np
import pyproj
import h5py
import argparse
import warnings
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from datetime import datetime
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Ignore all warnings
warnings.filterwarnings("ignore")

def get_args():
    """ Get command-line arguments. """
    parser = argparse.ArgumentParser(
            description='Program for computing satellite/airborne crossovers.')

    parser.add_argument(
            'input', metavar='ifile', type=str, nargs=2,
            help='name of two input files to cross (HDF5)')
    parser.add_argument(
            '-o', metavar='ofile', dest='output', type=str, nargs=1,
            help='name of output file (HDF5)',
            default=[None])
    parser.add_argument(
            '-r', metavar=('radius'), dest='radius', type=float, nargs=1,
            help='maximum interpolation distance from crossing location (m)',
            default=[350],)
    parser.add_argument(
            '-p', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
            help=('projection: EPSG number (AnIS=3031, GrIS=3413)'),
            default=['4326'],)
    parser.add_argument(
            '-d', metavar=('tile_size'), dest='dxy', type=int, nargs=1,
            help='tile size (km)',
            default=[None],)
    parser.add_argument(
            '-k', metavar=('na','nd'), dest='nres', type=int, nargs=2,
            help='along-track subsampling every k:th pnt for each file',
            default=[1],)
    parser.add_argument(
            '-b', metavar=('buffer'), dest='buff', type=int, nargs=1,
            help=('tile buffer (km)'),
            default=[0],)
    parser.add_argument(
            '-m', metavar=None, dest='mode', type=str, nargs=1,
            help='interpolation method, "linear" or "cubic"',
            choices=('linear', 'cubic'), default=['linear'],)
    parser.add_argument(
            '-v', metavar=('o','x','y','t','h','b','l','t'), dest='vnames', type=str, nargs=8,
            help=('main vars: names if HDF5, orbit/lon/lat/time/height/bs/lew/tes'),
            default=[None],)
    parser.add_argument(
            '-t', metavar=('t1','t2'), dest='tspan', type=float, nargs=2,
            help='only compute crossovers for given time span',
            default=[None,None],)
    parser.add_argument(     # ???
            '-f', dest='tile', action='store_true',
            help=('run in tile mode'),
            default=False)
    parser.add_argument(    
            '-q', dest='plot', action='store_true',
            help=('plot for inspection'),
            default=False)
    parser.add_argument(
            '-i', dest='diff', action='store_true',
            help=('do not interpolate vars just take diff'),
            default=False)
            
    return parser.parse_args()


def intersect(x_down, y_down, x_up, y_up):
    """  reference: https://stackoverflow.com/questions/17928452/
         find-all-intersections-of-xy-data-point-graph-with-numpy
    des: Find orbit crossover locations through solving the equation: 
         p0 + s*(p1-p0) = q0 + t*(q1-q0); p and q are descending and ascending data respectively.
         if s and t belong to [0,1], p and a actually do intersect.
         !! in order to speed up calculation, this code vectorizing solution of the 2x2 linear systems
    input:
        coordinates(x,y) of the points of descending and ascending orbit 
    retrun:
        np.array(shape: (n,2)), coordinates (x,y) of the intersection points. 
        n is number of intersection points
    """
    p = np.column_stack((x_down, y_down))   # coords of the descending points
    q = np.column_stack((x_up, y_up))       # coords of the ascending points

    (p0, p1, q0, q1) = p[:-1], p[1:], q[:-1], q[1:]   # remove first/last row respectively
    # up(x,y)-down(x,y), array broadcast, dim: (pos_down, pos_up, pos_dif(x,y))
    rhs = q0 - p0[:, np.newaxis, :]    

    mat = np.empty((len(p0), len(q0), 2, 2))  # dim: (p_num, q_num, dif((x, y)), orbit(down,up))
    mat[..., 0] = (p1 - p0)[:, np.newaxis]  # dif (x_down,y_down) between point_down and previous point_down
    mat[..., 1] = q0 - q1      #  dif (x_up, y_up) between point_up and previous point_up
    mat_inv = -mat.copy()
    mat_inv[..., 0, 0] = mat[..., 1, 1]   # exchange between x_dif and y_dif, down and up
    mat_inv[..., 1, 1] = mat[..., 0, 0]

    det = mat[..., 0, 0] * mat[..., 1, 1] - mat[..., 0, 1] * mat[..., 1, 0]
    mat_inv /= det[..., np.newaxis, np.newaxis]    # ???
    params = mat_inv @ rhs[..., np.newaxis]   # 
    intersection = np.all((params >= 0) & (params <= 1), axis=(-1, -2)) #
    p0_s = params[intersection, 0, :] * mat[intersection, :, 0]

    return p0_s + p0[np.where(intersection)[0]]


def transform_coord(proj1, proj2, x, y):
    """ Transform coordinates from proj1 to proj2 (EPSG num). """

    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+str(proj1))
    proj2 = pyproj.Proj("+init=EPSG:"+str(proj2))

    # Convert coordinates
    return pyproj.transform(proj1, proj2, x, y)

def get_bboxs_coor(xmin, xmax, ymin, ymax, dxy):
    """
    des: Define binned boxes (bbox) for speeding up the processing. 
    Args:
        xmin/xmax/ymin/ymax: must be in projection: stereographic (m).
        dxy: size of each box.
    retrun:
        list, consists of corner coordinates of each box.
    """
    # Number of tile edges on each dimension 
    Nns = int(np.abs(ymax - ymin) / dxy) + 1   # row
    New = int(np.abs(xmax - xmin) / dxy) + 1   # col
    # Coord of tile edges for each dimension
    xg = np.linspace(xmin, xmax, New)
    yg = np.linspace(ymin, ymax, Nns)
    # Vector of bbox for each cell, coordinates of corners of each pixel.
    bboxs_coor = [(w,e,s,n) for w,e in zip(xg[:-1], xg[1:]) 
                       for s,n in zip(yg[:-1], yg[1:])]
    del xg, yg
    return bboxs_coor

def get_bboxs_id(x, y, xmin, xmax, ymin, ymax, dxy, buff):
    """
    des: Define binned boxes (bbox) for speeding up the processing.
    arg:
        x,y: coordinates of the photon points
        xmin/xmax/ymin/ymax: must be in grid projection: stereographic (m).
        dxy: grid-cell size.
        buff: buffer region
    return:
        the index of each points corresponding to the generated bins.  
    """
    # Number of tile edges on each dimension
    Nns = int(np.abs(ymax - ymin) / dxy) + 1
    New = int(np.abs(xmax - xmin) / dxy) + 1
    # Coord of tile edges for each dimension
    xg = np.linspace(xmin-buff, xmax+buff, New)
    yg = np.linspace(ymin-buff, ymax+buff, Nns)
    # Indicies for each points
    bboxs_id = stats.binned_statistic_2d(x, y, np.ones(x.shape), 'count', bins=[xg, yg]).binnumber
    return bboxs_id

def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)

def interp1D(x, y, xi, n = 1):
    """ des: 1D interpolation (Spline)
        args:
            x,y are the given pair-wise values
            xi is the interpolation point.
            n: Degree of the smoothing spline. Must be 1 <= n <= 5.
        return:
            interpolated yi value
    """
    idx = np.argsort(x)    # Sort data by x value
    x, y = x[idx], y[idx]  # Sort arrays
    Fi = InterpolatedUnivariateSpline(x, y, k=n)  # Create interpolator
    yi = Fi(xi)     # Interpolated value
    return yi

## ?? what is tile
def tile_num(fname):
    """ Extract tile number from file name. """
    l = os.path.splitext(fname)[0].split('_')   # fname -> list
    i = l.index('tile')   # i is the index in the list
    return int(l[i+1])

def match_tiles(str1, str2):
    """ 
    des: Matches tile indices 
    args:
        str1, str2:  Unix style pathname, e.g., 'root/dir/*'
    return:
        pair-wise files (have the same tile number).
    """
    # Get file names
    files1 = glob.glob(str1)
    files2 = glob.glob(str2)
    # Create output list
    f1out = []
    f2out = []
    # Loop trough files-1
    for file1 in files1:
        f1 = tile_num(file1)  # Get tile index
        # Loop trough files-2
        for file2 in files2:
            f2 = tile_num(file2)   # Get tile index
            # Check if tiles have same index
            if f1 == f2:
                # Save if true
                f1out.append(file1)
                f2out.append(file2)
                break    # break loop: 'for file2 in files2'
    return f1out, f2out

# Read in parameters
args   = get_args()
ifiles = args.input[:]
ofile_ = args.output[0]
radius = args.radius[0]
proj   = args.proj[0]
dxy    = args.dxy[0]
nres_a = args.nres[0]
nres_d = args.nres[1]
buff   = args.buff[0]
mode   = args.mode[0]
vnames = args.vnames[:]
tspan  = args.tspan[:]
tile   = args.tile
plot   = args.plot
diff   = args.diff

print('parameters:')
for arg in vars(args).items(): 
    print(arg)

# Get variable names, 
# !!! o (orbit) in ovar is the ground track number (0-5) of the var,not ascending/descending
# ? bvar,lvar,svar
ovar, xvar, yvar, tvar, zvar, bvar, lvar, svar = vnames

# Create output names, _a and _b represent ascending and descending respectively.
# _x represent crossover point
oovar_a = ovar+'_1'   # orbit of file1 (ascending/descending), orbit is the ground track number(0-5)
oovar_d = ovar+'_2'   # orbit of file2 (descending/ascending)
oxvar_x = xvar        # coord-x name of crossover point 
oyvar_x = yvar        # coord-y name of crossover point 
otvar_a = tvar+'_1'
otvar_d = tvar+'_2'
otvar_x = 'd'+tvar    # 
ozvar_a = zvar+'_1'
ozvar_d = zvar+'_2'
ozvar_x = 'd'+zvar
obvar_a = bvar+'_1'
obvar_d = bvar+'_2'
obvar_x = 'd'+bvar
olvar_a = lvar+'_1'
olvar_d = lvar+'_2'
olvar_x = 'd'+lvar
osvar_a = svar+'_1'
osvar_d = svar+'_2'
osvar_x = 'd'+svar

# Test names
if ozvar_x == osvar_x or ozvar_x == obvar_x or ozvar_x == olvar_x:
    print("****************************************************************")
    print("Extra parameters can't have the same name as the main parameter!")
    print("****************************************************************")
    sys.exit()

# Test for stereographic
if proj != "4326":
    # Convert to meters
    dxy *= 1e3

def main(ifile1, ifile2):
    """ 
    des: Find and compute crossover values. 
    arg:
        ifile1, ifile2: ascending and descending data respectively.
    """
    startTime = datetime.now()
    print('crossing files:', ifile1, ifile2, '...')
    # Load all 1d variables needed
    with h5py.File(ifile1, 'r') as f1, \
         h5py.File(ifile2, 'r') as f2:
        #### ------ f1 (ascending/descending): data reading
        orbit1  = f1[ovar][:]
        lon1    = f1[xvar][:]
        lat1    = f1[yvar][:]
        time1   = f1[tvar][:]
        height1 = f1[zvar][:]
        bs1     = f1[bvar][:] if bvar in f1 else np.zeros(lon1.shape)
        lew1    = f1[lvar][:] if lvar in f1 else np.zeros(lon1.shape)
        tes1    = f1[svar][:] if svar in f1 else np.zeros(lon1.shape)
        ### remove invalid values
        orbit1  = orbit1[np.isfinite(height1)]
        lon1    = lon1[np.isfinite(height1)]
        lat1    = lat1[np.isfinite(height1)]
        time1   = time1[np.isfinite(height1)]
        bs1     = bs1[np.isfinite(height1)]
        lew1    = lew1[np.isfinite(height1)]
        tes1    = tes1[np.isfinite(height1)]
        height1 = height1[np.isfinite(height1)]        
        
        #### ------ f2 (descending/ascending): data reading
        orbit2  = f2[ovar][:]
        lon2    = f2[xvar][:]
        lat2    = f2[yvar][:]
        time2   = f2[tvar][:]
        height2 = f2[zvar][:]
        bs2     = f2[bvar][:] if bvar in f2 else np.zeros(lon2.shape)
        lew2    = f2[lvar][:] if lvar in f2 else np.zeros(lon2.shape)
        tes2    = f2[svar][:] if svar in f2 else np.zeros(lon2.shape)        
        ### remove invalid values
        orbit2  = orbit2[np.isfinite(height2)]
        lon2    = lon2[np.isfinite(height2)]
        lat2    = lat2[np.isfinite(height2)]
        time2   = time2[np.isfinite(height2)]
        bs2     = bs2[np.isfinite(height2)]
        lew2    = lew2[np.isfinite(height2)]
        tes2    = tes2[np.isfinite(height2)]
        height2 = height2[np.isfinite(height2)]
        
        # Set flags for extra parameters
        if np.all(bs1 == 0):   # 
            flag_bs = False
        else:
            flag_bs = True
        if np.all(lew1 == 0):
            flag_le = False
        else:
            flag_le = True
        if np.all(tes1 == 0):
            flag_ts = False
        else:
            flag_ts = True

    # If time span given, filter out invalid data that out the given time span
    if tspan[0] != None:

        t1, t2 = tspan
        ## file 1
        idx, = np.where((time1 >= t1) & (time1 <= t2))
        orbit1 = orbit1[idx]
        lon1 = lon1[idx]
        lat1 = lat1[idx]
        time1 = time1[idx]
        height1 = height1[idx]
        bs1 = bs1[idx]
        lew1 = lew1[idx]
        tes1 = tes1[idx]

        ## file 2
        idx, = np.where((time2 >= t1) & (time2 <= t2))
        orbit2 = orbit2[idx]
        lon2 = lon2[idx]
        lat2 = lat2[idx]
        time2 = time2[idx]
        height2 = height2[idx]
        bs2 = bs2[idx]
        lew2 = lew2[idx]
        tes2 = tes2[idx]
        
        if len(time1) < 3 or len(time2) < 3:   #
            print('no points within time-span!')
            sys.exit()

    # Transform to wanted coordinate system
    (xp1, yp1) = transform_coord(4326, proj, lon1, lat1)
    (xp2, yp2) = transform_coord(4326, proj, lon2, lat2)
    
    # Time limits: the largest time span (yr)
    tmin = min(np.nanmin(time1), np.nanmin(time2))
    tmax = max(np.nanmax(time1), np.nanmax(time2))

    # Region limits: the smallest spatial domain (m)
    xmin = max(np.nanmin(xp1), np.nanmin(xp2))
    xmax = min(np.nanmax(xp1), np.nanmax(xp2))
    ymin = max(np.nanmin(yp1), np.nanmin(yp2))
    ymax = min(np.nanmax(yp1), np.nanmax(yp2))

    # Interpolation type and number of needed points
    if mode == "linear":
        # Linear interpolation
        nobs  = 2
        order = 1
    else:
        # Cubic interpolation
        nobs  = 6
        order = 3

    # in fact, dxy is a mandatory parameter
    if dxy:
        print('tileing asc/des data...')        
        # Get binned boxes index (bin indices corresponding to each (x,y))
        bboxs1 = get_bboxs_id(xp1, yp1, xmin, xmax, ymin, ymax, dxy, buff*1e3)
        bboxs2 = get_bboxs_id(xp2, yp2, xmin, xmax, ymin, ymax, dxy, buff*1e3)
        # Copy box for convenience
        bboxs = bboxs1
    else:        
        # !!!Get bounding box coord from full domain
        bboxs = [(xmin, xmax, ymin, ymax)]

    startTime = datetime.now()
    # Initiate output container (dictionary)
    out = []    
    # Counter: count of the bins valid
    ki = 0

    # Unique boxes, if setting dxy. when not setting dxy??
    ibox = np.unique(bboxs)

    print('computing crossovers ...')
    
    # Loop through each sub-tile
    # k is the unique bin index of (x,y)
    for k in ibox:        
        idx1, = np.where(bboxs1 == k)    # idx is the data points index
        idx2, = np.where(bboxs2 == k)
        # Extract binned data from each set
        orbits1 = orbit1[idx1]
        lons1 = lon1[idx1]
        lats1 = lat1[idx1]
        x1 = xp1[idx1]          # projection coord_x
        y1 = yp1[idx1]          # projection coord_y
        h1 = height1[idx1]
        t1 = time1[idx1]
        b1 = bs1[idx1]
        l1 = lew1[idx1]
        s1 = tes1[idx1]
        
        orbits2 = orbit2[idx2]
        lons2 = lon2[idx2]
        lats2 = lat2[idx2]
        x2 = xp2[idx2]
        y2 = yp2[idx2]
        h2 = height2[idx2]
        t2 = time2[idx2]
        b2 = bs2[idx2]
        l2 = lew2[idx2]
        s2 = tes2[idx2]

        # Get unique orbits (in fact, are the ground track number)
        orb_ids1 = np.unique(orbits1)
        orb_ids2 = np.unique(orbits2)

        # Test if tile has no data
        if len(orbits1) == 0 or len(orbits2) == 0:
            # Go to next bin
            continue

        # ---- Loop through orbits (ground track in the specific bin) 
        # --> file #1
        for orb_id1 in orb_ids1:
            i_trk1 = orbits1 == orb_id1   ## i_trak1: data point index of specific orb_id1 orbit.
            ## Extract points from specific orbit (a specific track)
            xa = x1[i_trk1]
            ya = y1[i_trk1]
            ta = t1[i_trk1]
            ha = h1[i_trk1]
            ba = b1[i_trk1]
            la = l1[i_trk1]
            sa = s1[i_trk1]
            
            # Loop through groud tracks (0-5) in specific bin --> file #2
            for orb_id2 in orb_ids2:

                # Index for single descending orbit
                i_trk2 = orbits2 == orb_id2

                # Extract points from a specific orbit (groud track)
                xb = x2[i_trk2]
                yb = y2[i_trk2]
                tb = t2[i_trk2]
                hb = h2[i_trk2]
                bb = b2[i_trk2]
                lb = l2[i_trk2]
                sb = s2[i_trk2]
                
                # Test length of vector
                if len(xa) < 3 or len(xb) < 3: 
                    continue

                # Compute exact crossing points
                # Between specific track of file1 and specific track of file2
                cxy_main = intersect(xa[::nres_a], ya[::nres_a], \
                                     xb[::nres_d], yb[::nres_d])

                # Test again for crossing
                if len(cxy_main) == 0: continue
                """
                    SUPPORT SHOULD BE ADDED FOR MULTIPLE CROSSOVERS FOR SAME TRACK!
                    below are only consider the first crossover point in this program.
                """
                # Extract crossing coordinates
                xi = cxy_main[0][0]
                yi = cxy_main[0][1]
                
                # Get start coordinates of the specific track in the specific bin
                # a and b are ascending/descending respectively.
                xa0 = xa[0]
                ya0 = ya[0]
                xb0 = xb[0]
                yb0 = yb[0]

                # Compute distance between each (x,y) and crossing point in the bin
                da = (xa - xi) * (xa - xi) + (ya - yi) * (ya - yi)
                db = (xb - xi) * (xb - xi) + (yb - yi) * (yb - yi)

                # Sort by distance (small->large)
                Ida = np.argsort(da)  # Ida is the point index 
                Idb = np.argsort(db)

                # Sort arrays - A (small->large)
                xa = xa[Ida]
                ya = ya[Ida]
                ta = ta[Ida]
                ha = ha[Ida]
                da = da[Ida]
                ba = ba[Ida]
                la = la[Ida]
                sa = sa[Ida]
                
                # Sort arrays - B
                xb = xb[Idb]
                yb = yb[Idb]
                tb = tb[Idb]
                hb = hb[Idb]
                db = db[Idb]
                bb = bb[Idb]
                lb = lb[Idb]
                sb = sb[Idb]
                
                # Get two nearest point from ascending and descending track, respectively.
                # da[[0,1]] return list: [da[0],da[1]]
                dab = np.vstack((da[[0, 1]], db[[0, 1]]))

                ## Test if any point is too farther than the given radius
                if np.any(np.sqrt(dab) > radius):
                    continue
                # Test if enough obs. are available for interpolation
                elif (len(xa) < nobs) or (len(xb) < nobs):
                    continue
                else:
                    # Accepted
                    pass

                # Compute distance（between track start point and all another points
                da0 = (xa - xa0) * (xa - xa0) + (ya - ya0) * (ya - ya0)
                db0 = (xb - xb0) * (xb - xb0) + (yb - yb0) * (yb - yb0)

                # Compute distance (between track start and cross point) 
                dai = (xi - xa0) * (xi - xa0) + (yi - ya0) * (yi - ya0)
                dbi = (xi - xb0) * (xi - xb0) + (yi - yb0) * (yi - yb0)
                
                # Interpolate height to crossover location through the nearest nobs(number) points.
                # x: distance, y: height -> given interpolated distance
                hai = interp1D(da0[0:nobs], ha[0:nobs], dai, order)
                hbi = interp1D(db0[0:nobs], hb[0:nobs], dbi, order)
                
                # Interpolate time to crossover location through the nearest nobs(number) points.
                # x: distance, y: time -> given interploated distance
                tai = interp1D(da0[0:nobs], ta[0:nobs], dai, order)
                tbi = interp1D(db0[0:nobs], tb[0:nobs], dbi, order)
                
                # Test interpolate time values
                if (tai > tmax) or (tai < tmin) or \
                    (tbi > tmax) or (tbi < tmin):
                    continue
                
                # Create output array
                out_i = np.full(20, np.nan)
                
                # Compute differences and save parameters
                out_i[0]  = xi      # crossover points x
                out_i[1]  = yi      # ... y
                out_i[2]  = hai - hbi   ## height difference between ascending and descending interpolations
                out_i[3]  = tai - tbi   ## time difference between ...
                out_i[4]  = tai     # interpolated time by ascending track
                out_i[5]  = tbi     # interpolated time by descending track
                out_i[6]  = hai     # interpolated height by ascending track
                out_i[7]  = hbi     # interpolated height by descending track
                out_i[8]  = (hai - hbi) / (tai - tbi)  # speed of elevation change
                out_i[18] = orb_id1  # groud track of ascending file
                out_i[19] = orb_id2  # groud track of decending file

                # Test for more parameters to difference
                if flag_bs:
                    
                    if diff is True:
                        
                        # Save paramters, the nearest (to cross point) values of ascend/descend track
                        bai = ba[0]
                        bbi = bb[0]
                        
                        # Save difference
                        out_i[9]  = bai - bbi
                        out_i[10] = bai
                        out_i[11] = bbi

                    else:
                        
                        # Interpolate sigma0 to crossover location
                        bai = interp1D(da0[0:nobs], ba[0:nobs], order)
                        bbi = interp1D(db0[0:nobs], bb[0:nobs], order)
                        
                        # Save difference
                        out_i[9]  = bai - bbi
                        out_i[10] = bai
                        out_i[11] = bbi

                if flag_le:   # 
                    
                    if diff is True:

                        # Get paramters
                        lai = la[0]
                        lbi = lb[0]
                        
                        # Save difference
                        out_i[12]  = lai - lbi
                        out_i[13] = lai
                        out_i[14] = lbi
               
                    else:
                        
                        # Interpolate leading edge width to crossover location
                        lai = interp1D(da0[0:nobs], la[0:nobs], order)
                        lbi = interp1D(db0[0:nobs], lb[0:nobs], order)
                        
                        # Save difference
                        out_i[12] = lai - lbi
                        out_i[13] = lai
                        out_i[14] = lbi

                if flag_ts:   ## 
                    
                    if diff is True:
                        
                        # Get parameters
                        sai = sa[0]
                        sbi = sb[0]
                        
                        # Save difference
                        out_i[15] = sai - sbi
                        out_i[16] = sai
                        out_i[17] = sbi

                    else:
                        
                        # Interpolate trailing edge slope to crossover location
                        sai = interp1D(da0[0:nobs], sa[0:nobs], order)
                        sbi = interp1D(db0[0:nobs], sb[0:nobs], order)
                        
                        # Save difference
                        out_i[15] = sai - sbi
                        out_i[16] = sai
                        out_i[17] = sbi
                        
                # Add to list
                out.append(out_i)
                
        # Operating on current tile
        # print('tile:', ki, len(ibox))

        # Update counter 
        ki += 1

    # Change back to numpy array
    out = np.asarray(out)

    # Remove invalid rows
    out = out[~np.isnan(out[:,2]),:]     # out[:,2]: height difference

    # Test if output container is empty 
    if len(out) == 0:
        print('no crossovers found!')
        return

    # Remove the two id columns if they are empty, out[:,-1]: orb_id2，out[:,-2]: orb_id1
    out = out[:,:-2] if np.isnan(out[:,-1]).all() else out

    # Copy ouput arrays 
    xs, ys = out[:,0].copy(), out[:,1].copy()

    # Transform coords back to lat/lon
    out[:,0], out[:,1] = transform_coord(proj, '4326', out[:,0], out[:,1])

    # Create output file name if not given
    if ofile_ is None:
        path, ext = os.path.splitext(ifile1)
        if tile:
            tilenum = str(tile_num(ifile1))
        else:
            tilenum = '' 
        if ifile1.find('_A_') > 0:   # "_A_" in the ifile1 string
            fnam = '_XOVERS_AD_' 
        else:
            fnam = '_XOVERS_DA_'
        ofile = path + fnam + tilenum + ext
    else:
        ofile = ofile_

    # Create h5 file
    with h5py.File(ofile, 'w') as f:
        
        # Add standard parameters corresponding to crossover point.
        # _x: interpolation by ascending and descending points; 
        # _a: interpolation by ascending points; 
        # _d: interpolation by descending points.
        f[oxvar_x] = out[:,0]
        f[oyvar_x] = out[:,1]
        f[ozvar_x] = out[:,2]
        f[otvar_x] = out[:,3]
        f[otvar_a] = out[:,4]
        f[otvar_d] = out[:,5]
        f[ozvar_a] = out[:,6]
        f[ozvar_d] = out[:,7]
        f[oovar_a] = out[:,18]
        f[oovar_d] = out[:,19]
        f['dhdt']  = out[:,8]

        # Add extra parameters
        if flag_bs:
            f[obvar_x] = out[:,9]
            f[obvar_a] = out[:,10]
            f[obvar_d] = out[:,11]
        if flag_le:
            f[olvar_x] = out[:,12]
            f[olvar_a] = out[:,13]
            f[olvar_d] = out[:,14]
        if flag_ts:
            f[osvar_x] = out[:,15]
            f[osvar_a] = out[:,16]
            f[osvar_d] = out[:,17]
                
    # Stat. variables
    dh   = out[:,2]
    dt   = out[:,3]
    dhdt = out[:,8]

    # Compute statistics
    med_dhdt = np.around(np.median(dhdt),3)
    std_dhdt = np.around(mad_std(dhdt),3)
    
    # Print some statistics to screen
    print('')
    print('execution time: ' + str(datetime.now() - startTime))
    print('number of crossovers found:',str(len(out)))
    print('statistics -> median_dhdt:',med_dhdt,'std_dhdt:',std_dhdt, '(dvar/yr)')
    print('ofile name ->', ofile)

    if plot:

        # Some light filtering for plotting
        io = np.abs(dh) < 3.0*mad_std(dh)
        
        gridsize = (3, 2)
        fig = plt.figure(figsize=(12, 8))
        ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2) # gridsize: shape, (0,0):location
        ax2 = plt.subplot2grid(gridsize, (2, 0))
        ax3 = plt.subplot2grid(gridsize, (2, 1))
        # Plot main difference variable
        ax1.scatter(xs*1e-3, ys*1e-3, s=10, c=dh, cmap='jet', vmin=-20, vmax=20)
        #plt.subplot(211)
        ax2.plot(dt[io], dh[io], '.')
        ax2.set_ylabel(ozvar_x)
        ax2.set_xlabel(otvar_x)
        #plt.subplot(212)
        ax3.hist(dh[io],50)
        ax3.set_ylabel('Frequency')
        ax3.set_xlabel(ozvar_x)
        #plt.subplots_adjust(hspace=.5)
        plt.show()

#
# Running main program!
#

# Read file names
str1, str2 = ifiles

# Check for tile mode
if tile:
        
    # Get matching tiles
    files1, files2 = match_tiles(str1, str2)
    # Loop through tiles
    for i in range(len(files1)):            
        # Run main program
        main(files1[i], files2[i])

# Run as single files
else:
        
    # File names
    file1, file2 = str1, str2
            
    # Run main program
    main(file1, file2)
