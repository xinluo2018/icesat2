## author: Fernando Paolo
## modify: xin luo, 2021.8.14

"""
notes:
    For external tile processing please use "tile.py" with the same extent 
    for the A and D files. This as the program uses the tile numbering to 
    determine which of the tiles should be crossed together.
    
    When running in external tile-mode the saved file with crossovers
    will be appended with "_XOVERS_AD/DA". Please use "_A" or "_D" in the
    filename to indicate Asc or Des tracks when running in tile mode. 
    
example:
    python xover.py a.h5 d.h5 -o xover.h5 -r 350 -p 3031 -d 100 -k 1 1\
    -m linear -v orb lon lat time height dum dum dum -b 10
    
    python xover.py ./tiles/*_a.h5 ./tiles/*_d.h5 -r 350 -p 3031 -d 100 -k\
    1 1 -m linear -v orb lon lat time height dum dum dum -f -b 10
    
"""

import os
import sys
import glob
import numpy as np
import pyproj
import h5py
import argparse
import warnings
from datetime import datetime
from helper import intersect, get_bboxs_id
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import InterpolatedUnivariateSpline

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
            '-d', metavar=('tile_size'), dest='tile_dxy', type=int, nargs=1,
            help='tile size (km)',
            default=[None],)
    parser.add_argument(
            '-k', metavar=('n_res'), dest='nres', type=int, nargs=1,
            help='along-track subsampling every k:th pnt for each file',
            default=[1],)
    parser.add_argument(
            '-bf', metavar=('buffer'), dest='buff', type=int, nargs=1,
            help=('region buffer (km)'),
            default=[0],)
    parser.add_argument(
            '-m', metavar=None, dest='mode', type=str, nargs=1,
            help='interpolation method, "linear" or "cubic"',
            choices=('linear', 'cubic'), default=['linear'],)
    parser.add_argument(
            '-v', metavar=('spot','x','y','t','h'), dest='vnames', type=str, nargs=5,
            help=('main vars: names if HDF5, spot/lon/lat/time/height'),
            default=[None],)
    parser.add_argument(         # 
            '-f', dest='tile', action='store_true',
            help=('run in tile mode'),
            default=False)
            
    return parser.parse_args()

def interp1d(x, y, xi, n = 1):
    """ des: 
            1D interpolation (spline)
        args:
            x,y are the given pair-wise values
            xi is the interpolation point.
            n: degree of the smoothing spline. must be 1 <= n <= 5.
        return:
            interpolated yi value
    """
    idx = np.argsort(x)    # Sort data by x value
    x, y = x[idx], y[idx]  # Sort arrays
    Fi = InterpolatedUnivariateSpline(x, y, k=n)  # create interpolator
    yi = Fi(xi)     # interpolated value
    return yi


def tile_num(fname):
    """ Extract tile number from file name. """
    l = os.path.splitext(fname)[0].split('_')   # fname -> list
    i = l.index('tile')   # i is the index in the list
    return int(l[i+1])

def match_tiles(str1, str2):
    """ 
    des: matches tile indices 
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

def coor2coor(srs_from, srs_to, x, y):
    """
    Transform coordinates from srs_from to srs_to
    input:
        srs_from and srs_to are EPSG number (e.g., 4326, 3031)
        x and y are x-coord and y-coord corresponding to srs_from and srs_to    
    return:
        x-coord and y-coord in srs_to 
    """
    srs_from = pyproj.Proj(int(srs_from))
    srs_to = pyproj.Proj(int(srs_to))
    return pyproj.transform(srs_from, srs_to, x, y, always_xy=True)


def xover(lon_as, lat_as, time_as, height_as, spot_as, 
            lon_des, lat_des, time_des, height_des, spot_des, 
            nres, proj, radius_interp, tile_dxy=10, buff=1, mode_interp = 'linear'):
    """ 
    des: find and compute crossover values. 
    arg:
        lon_as, lat_as: coordinates of ascending points.
        time_as, height_as: time, height of ascending points
        orbit_as: groud track (0-5) of the ascending points.
        lon_des, lat_des: coordinates of descending points.
        time_des, height_des: time, height of descending points
        orbit_des: groud track (0-5) of the descending points.
        nres: subsampling along track, every nres points, get one point.
        proj: projection espg number.
        radius_interp: threshold of distance of the nearest points for interpolation
        tile_dxy: width/height of the generated tile. For speeding up processing.
        mode_interp: interpolation method, 'linear' or 'cubic
    return:
        out: 
    """

    # Transform to wanted coordinate system
    (xp_as, yp_as) = coor2coor(4326, proj, lon_as, lat_as)
    (xp_des, yp_des) = coor2coor(4326, proj, lon_des, lat_des)
    
    # Time limits: the largest time span (yr)
    tmin = min(np.nanmin(time_as), np.nanmin(time_as))
    tmax = max(np.nanmax(time_as), np.nanmax(time_des))

    # Region limits: the smallest spatial domain (m)
    xmin = max(np.nanmin(xp_as), np.nanmin(xp_des))
    xmax = min(np.nanmax(xp_as), np.nanmax(xp_des))
    ymin = max(np.nanmin(yp_as), np.nanmin(yp_des))
    ymax = min(np.nanmax(yp_as), np.nanmax(yp_des))

    # Interpolation type and number of needed points
    if mode_interp == "linear":
        # Linear interpolation
        nobs  = 2
        order = 1
    else:
        # Cubic interpolation
        nobs  = 6
        order = 3

    print('tileing asc/des data...') 
    # get binned boxes index (bin indices corresponding to each (x,y))
    # hear bin is tile.
    bboxs_as = get_bboxs_id(xp_as, yp_as, xmin, xmax, ymin, ymax, tile_dxy, buff)     # ascending file
    bboxs_des = get_bboxs_id(xp_des, yp_des, xmin, xmax, ymin, ymax, tile_dxy, buff)  # descending file
    # Copy box for convenience
    bboxs = bboxs_as
    # Initiate output container (dictionary)
    out = []    
    ibox = np.unique(bboxs)    
    num_box = 0              #  counter: count of the bins valid

    print('computing crossovers ...')
    #######   for bin in bins:
    #######       for track_as in tracks_as: 
    #######           for track_des in tracks_des: 
    #######               find the xover_points.
    # loop through each unique bin (tile)
    # k is the unique bin index.
    for k in ibox:        
        idx_as, = np.where(bboxs_as == k)    # idx_ is the data points index
        idx_des, = np.where(bboxs_des == k)
        # Extract points in the bin
        spot_as_ibox = spot_as[idx_as]
        x_as_ibox = xp_as[idx_as]                 #  projection coord_x
        y_as_ibox = yp_as[idx_as]                 #  projection coord_y
        h_as_ibox = height_as[idx_as]
        t_as_ibox = time_as[idx_as]
        
        spot_des_ibox = spot_des[idx_des]         #  
        x_des_ibox = xp_des[idx_des]
        y_des_ibox = yp_des[idx_des]
        h_des_ibox = height_des[idx_des]
        t_des_ibox = time_des[idx_des]

        # Get unique orbits (in fact, are the ground track number)
        spot_as_ibox_unique = np.unique(spot_as_ibox)
        spot_des_ibox_unique = np.unique(spot_des_ibox)

        # Test if tile has no data 
        # cause bboxs = bboxs1, len(orbits1) !=1, len(orbits2) could be 0.
        if len(spot_as_ibox) == 0 or len(spot_des_ibox) == 0:
            continue

        # ---- loop through tracks (ground track in the specific bin) 
        # --> ascending tracks
        for spot_as_ibox_i in spot_as_ibox_unique:
            ## i_trk_: point index of the specific groud track.
            i_as_trk = spot_as_ibox == spot_as_ibox_i 
            ## extract points from the specific orbit (a specific track)
            xa = x_as_ibox[i_as_trk]
            ya = y_as_ibox[i_as_trk]
            ta = t_as_ibox[i_as_trk]
            ha = h_as_ibox[i_as_trk]
            
            # Loop through groud tracks (1-6) in specific bin 
            # --> descending tracks
            for spot_des_ibox_i in spot_des_ibox_unique:

                # index of data points of specific track in file 2
                i_des_trk = spot_des_ibox == spot_des_ibox_i

                # extract points from a specific orbit (groud track)
                xd = x_des_ibox[i_des_trk]
                yd = y_des_ibox[i_des_trk]
                td = t_des_ibox[i_des_trk]
                hd = h_des_ibox[i_des_trk]
                
                # Test length of vector
                if len(xa) < 3 or len(xd) < 3: 
                    continue

                # exact crossing points between two tracks of ascending/descending files.
                cxy_main = intersect(xa[::nres], ya[::nres], \
                                     xd[::nres], yd[::nres])

                # test again for crossing
                if len(cxy_main) == 0: continue
                """
                    SUPPORT SHOULD BE ADDED FOR MULTIPLE CROSSOVERS FOR SAME TRACK!
                    below are only consider the first crossover point in this program.
                """
                # coordinates of crossing points, !!!only one crossover point is extracted.
                xi = cxy_main[0][0]
                yi = cxy_main[0][1]
                
                # Get start coordinates of the specific track in the specific bin
                # a and b are ascending/descending respectively.
                xa0 = xa[0]
                ya0 = ya[0]
                xd0 = xd[0]
                yd0 = yd[0]

                ## -- sort the points by distance from xover point.
                # distance between points (x,y) and crossing point in the bin
                d_as = (xa - xi) * (xa - xi) + (ya - yi) * (ya - yi)
                d_des = (xd - xi) * (xd - xi) + (yd - yi) * (yd - yi)

                # sorted by distance (small->large)
                i_da = np.argsort(d_as)    # i_da is the index corresponding to sorted points.
                i_dd = np.argsort(d_des)

                # sort arrays - A (small->large)
                xa = xa[i_da]
                ya = ya[i_da]
                ta = ta[i_da]
                ha = ha[i_da]
                d_as = d_as[i_da]

                # sort arrays - B
                xd = xd[i_dd]
                yd = yd[i_dd]
                td = td[i_dd]
                hd = hd[i_dd]
                d_des = d_des[i_dd]
                
                # Get distance^2 for two nearest points from ascending and descending track, respectively.
                # d_as[[0,1]] return list: [d_as[0],d_as[1]]
                da_dd = np.vstack((d_as[[0, 1]], d_des[[0, 1]]))

                ## Test if any point is too farther than the given radius
                if np.any(np.sqrt(da_dd) > radius_interp):
                    continue
                
                # Test if enough obs. are available for interpolation
                elif (len(xa) < nobs) or (len(xd) < nobs):
                    continue
                else:
                    pass
                
                ### -- interpolation of the xover point by nearest points.
                # distances between sorted points (x,y) and track start point
                d_a0 = (xa - xa0) * (xa - xa0) + (ya - ya0) * (ya - ya0)
                d_d0 = (xd - xd0) * (xd - xd0) + (yd - yd0) * (yd - yd0)

                # distance between track start point and cross point
                d_ai = (xi - xa0) * (xi - xa0) + (yi - ya0) * (yi - ya0)
                d_di = (xi - xd0) * (xi - xd0) + (yi - yd0) * (yi - yd0)
                
                ##### ----- height interpolation: using the nearest nobs(number) points.
                # params: distances, heights, interpolated distance, interpolation order.
                h_ai = interp1d(d_a0[0:nobs], ha[0:nobs], d_ai, order)
                h_di = interp1d(d_d0[0:nobs], hd[0:nobs], d_di, order)
                
                ##### ----- time interpolation: using the nearest nobs(number) points.
                # x: distance, y: time -> given interploated distance
                t_ai = interp1d(d_a0[0:nobs], ta[0:nobs], d_ai, order)
                t_di = interp1d(d_d0[0:nobs], td[0:nobs], d_di, order)
                
                # Test interpolate time values
                if (t_ai > tmax) or (t_ai < tmin) or \
                    (t_di > tmax) or (t_di < tmin):
                    continue
                
                # Create output array
                out_i = np.full(10, np.nan)
                
                # Compute differences and save parameters
                out_i[0]  = xi             # crossover points coord_x
                out_i[1]  = yi             # ... coord_y
                out_i[2]  = h_ai           # interpolated height by ascending track
                out_i[3]  = h_di           # interpolated height by descending track
                out_i[4]  = t_ai           # interpolated time by ascending track
                out_i[5]  = t_di           # interpolated time by descending track
                out_i[6] = spot_as_ibox_i      # groud track of ascending file
                out_i[7] = spot_des_ibox_i     # groud track of decending file
                out_i[8]  = h_ai - h_di    ## height difference between ascending and descending interpolations
                out_i[9]  = t_ai - t_di    ## time difference between ...
                        
                # Add to list
                out.append(out_i)

        num_box += 1


    # Change back to numpy array
    out = np.asarray(out)

    # Remove invalid rows
    out = out[~np.isnan(out[:,2]), :]     # out[:,2]: height difference

    # Test if output container is empty 
    if len(out) == 0:
        print('no crossovers found!')
        return 
    # remove the two id columns if they are empty, out[:,-1]: orb_id2，out[:,-2]: orb_id1
    out = out[:,:-2] if np.isnan(out[:,-1]).all() else out
    # Transform coords back to lat/lon
    out[:,0], out[:,1] = coor2coor(proj, '4326', out[:,0], out[:,1])
    return out

def xover_main(ifile_as, ifile_des, radius_interp, ofile_,
                    vnames, tile_dxy, tile=False):
    """ 
    des: 
        find and compute crossover values with input file paths. 
    arg:
        ifile1, ifile2: ascending and descending data respectively.
        tspan:  time span, a list contians mininum time (e.g., 2018.22) and maximun time. 
        vnames: list of strings, [ovar, xvar, yvar, tvar, zvar], representing the name of groud track,
                coord_x, coord_y, time, height in the .h5 file.
    """

    startTime = datetime.now()
    # get variable names, 
    spotvar, xvar, yvar, tvar, zvar = vnames

    print('crossing files:', ifile_as, ifile_des, '...')
    # Load all 1d variables needed
    with h5py.File(ifile_as, 'r') as f_as, \
         h5py.File(ifile_des, 'r') as f_des:

        #### ------ ascending file reading and remove invalid values
        spot_as  = f_as[spotvar][:]
        lon_as    = f_as[xvar][:]
        lat_as    = f_as[yvar][:]
        time_as   = f_as[tvar][:]
        height_as = f_as[zvar][:]
        spot_as  = spot_as[np.isfinite(height_as)]
        lon_as    = lon_as[np.isfinite(height_as)]
        lat_as    = lat_as[np.isfinite(height_as)]
        time_as   = time_as[np.isfinite(height_as)]
        height_as = height_as[np.isfinite(height_as)]    

        #### ------ descending file reading and remove invalid values
        spot_des  = f_des[spotvar][:]
        lon_des    = f_des[xvar][:]
        lat_des    = f_des[yvar][:]
        time_des   = f_des[tvar][:]
        height_des = f_des[zvar][:]
        spot_des  = spot_des[np.isfinite(height_des)]
        lon_des    = lon_des[np.isfinite(height_des)]
        lat_des    = lat_des[np.isfinite(height_des)]
        time_des   = time_des[np.isfinite(height_des)]
        height_des = height_des[np.isfinite(height_des)]
        

    out = xover(lon_as, lat_as, time_as, height_as, spot_as, 
                lon_des, lat_des, time_des, height_des, spot_des, 
                nres, proj, radius_interp, tile_dxy=tile_dxy, buff=1, mode_interp = 'linear')

    # create output file name if not given
    # if no given path_ofile, path_ofile is determined by path_ifile.
    if ofile_ is None:
        path, ext = os.path.splitext(ifile_as)
        if tile:
            tilenum = '_' + str(tile_num(ifile_as))
        else:
            tilenum = '' 
        fnam = '_xovers_ad' 
        ofile = path + fnam + tilenum + ext
    else:
        ofile = ofile_

    # create output names
    # _x represent crossover point
    oxvar = xvar                # coord-x name of crossover point 
    oyvar = yvar                # coord-y name of crossover point 
    ozvar_as = zvar+'_as'
    ozvar_des = zvar+'_des'
    otvar_as = tvar+'_as'
    otvar_des = tvar+'_des'
    ospotvar_as = spotvar+'_as'       # orbit of file1 (ascending/descending), orbit is the ground track number(0-5)
    ospotvar_des = spotvar+'_des'     # orbit of file2 (descending/ascending)
    ozvar_dif = zvar + '_dif'
    otvar_dif = tvar+'_dif'     # 

    # Create h5 file
    with h5py.File(ofile, 'w') as f:
        
        # add standard parameters corresponding to crossover point.
        f[oxvar] = out[:,0]
        f[oyvar] = out[:,1]
        f[ozvar_as] = out[:,2]
        f[ozvar_des] = out[:,3]
        f[otvar_as] = out[:,4]
        f[otvar_des] = out[:,5]
        f[ospotvar_as] = out[:,6]
        f[ospotvar_des] = out[:,7]
        f[ozvar_dif] = out[:,8]
        f[otvar_dif] = out[:,9]

if __name__ == '__main__':

    # Read in parameters
    args   = get_args()
    ifiles = args.input[:]
    ofile_ = args.output[0]
    radius = args.radius[0]
    proj   = args.proj[0]
    tile_dxy = args.tile_dxy[0]  # if the input data will be tiled in the processing.
    nres = args.nres[0]
    buff   = args.buff[0]
    mode   = args.mode[0]
    vnames = args.vnames[:]
    tile   = args.tile           # if the input file is tiled 

    print('parameters:')
    for arg in vars(args).items(): 
        print(arg)

    if proj == "4326":
        raise ValueError("proj can't be 4326")

    tile_dxy *= 1e3
    buff *= 1e3
    
    # Read file names
    str_as, str_des = ifiles

    # Check for tile mode
    if tile:
        # Get matching tiles
        files_as, files_des = match_tiles(str_as, str_des)
        # Loop through tiles
        for i in range(len(files_as)):            
            # Run main program
            xover_main(files_as[i], files_des[i], radius_interp=radius, 
                        ofile_=ofile_,vnames=vnames, tile_dxy=tile_dxy, tile=tile)

    # Run as single files
    else:
        # File names
        file_as, file_des = str_as, str_des
        # Run main program
        xover_main(file_as, file_des, radius_interp=radius, 
                    ofile_=ofile_, vnames=vnames, tile_dxy=tile_dxy, tile=False)



