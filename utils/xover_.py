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
import numpy as np
import pyproj
import h5py
import argparse
import warnings
import matplotlib.pyplot as plt
from interp import interp1d
from datetime import datetime
from tile import tile_num, match_tiles
from scipy import stats
from helper import mad_std, intersect,get_bboxs_id
from transform_xy import coor2coor
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
            '-v', metavar=('o','x','y','t','h'), dest='vnames', type=str, nargs=5,
            help=('main vars: names if HDF5, orbit/lon/lat/time/height'),
            default=[None],)
    parser.add_argument(
            '-t', metavar=('t1','t2'), dest='tspan', type=float, nargs=2,
            help='only compute crossovers for given time span',
            default=[None,None],)
    parser.add_argument(       # 
            '-f', dest='tile', action='store_true',
            help=('run in tile mode'),
            default=False)
            
    return parser.parse_args()



def xover(lon_as, lat_as, time_as, height_as, orbit_as, 
            lon_des, lat_des, time_des, height_des, orbit_des, 
            nres, proj, radius_interp, tile_dxy=10, mode_interp = 'linear'):
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
    bboxs_as = get_bboxs_id(xp_as, yp_as, xmin, xmax, ymin, ymax, tile_dxy, buff*1e3) # ascending file
    bboxs_des = get_bboxs_id(xp_des, yp_des, xmin, xmax, ymin, ymax, tile_dxy, buff*1e3) # descending file
    # Copy box for convenience
    bboxs = bboxs_as

    # Initiate output container (dictionary)
    out = []    

    # Unique boxes, if setting dxy. ??when not setting dxy
    ibox = np.unique(bboxs)
    # counter: count of the bins valid
    ki = 0

    print('computing crossovers ...')
    #######   for bin in bins:
    #######       for track_as in tracks_as: 
    #######           for track_des in tracks_des: 
    #######               find the xover_points.
    # loop through each unique bin (tile)
    # k is the unique bin index.
    for k in ibox:        
        idx_as, = np.where(bboxs_as == k)    # idx is the data points index
        idx_des, = np.where(bboxs_des == k)
        # extract binned data from each set
        orbits_as = orbit_as[idx_as]
        x_as = xp_as[idx_as]          # projection coord_x
        y_as = yp_as[idx_as]          # projection coord_y
        h_as = height_as[idx_as]
        t_as = time_as[idx_as]
        
        orbits_des = orbit_des[idx_des]
        x_des = xp_des[idx_des]
        y_des = yp_des[idx_des]
        h_des = height_des[idx_des]
        t_des = time_des[idx_des]

        # Get unique orbits (in fact, are the ground track number)
        orb_ids_as = np.unique(orbits_as)
        orb_ids_des = np.unique(orbits_des)

        # Test if tile has no data, 
        # cause bboxs = bboxs1, len(orbits1) !=1, len(orbits2) could be 0.
        if len(orbits_as) == 0 or len(orbits_des) == 0:
            continue

        # ---- loop through tracks (ground track in the specific bin) 
        # --> ascending tracks
        for orb_id_as in orb_ids_as:
            ## i_trak1: data point index of the specific track in file 1.
            i_trk_as = orbits_as == orb_id_as  
            ## extract points from the specific orbit (a specific track)
            xa = x_as[i_trk_as]
            ya = y_as[i_trk_as]
            ta = t_as[i_trk_as]
            ha = h_as[i_trk_as]
            
            # Loop through groud tracks (0-5) in specific bin 
            # --> descending tracks
            for orb_id_des in orb_ids_des:

                # index of data points of specific track in file 2
                i_trk_des = orbits_des == orb_id_des

                # extract points from a specific orbit (groud track)
                xd = x_des[i_trk_des]
                yd = y_des[i_trk_des]
                td = t_des[i_trk_des]
                hd = h_des[i_trk_des]
                
                # Test length of vector
                if len(xa) < 3 or len(xd) < 3: 
                    continue

                # compute exact crossing points
                # between specific track of file1 and specific track of file2
                cxy_main = intersect(xa[::nres], ya[::nres], \
                                     xd[::nres], yd[::nres])

                # test again for crossing
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
                xd0 = xd[0]
                yd0 = yd[0]

                ## -- sort the points by distance from xover point.
                # Compute distance between each (x,y) and crossing point in the bin
                d_as = (xa - xi) * (xa - xi) + (ya - yi) * (ya - yi)
                d_des = (xd - xi) * (xd - xi) + (yd - yi) * (yd - yi)

                # Sort by distance (small->large)
                i_da = np.argsort(d_as)  # Ida is the point index 
                i_dd = np.argsort(d_des)

                # Sort arrays - A (small->large)
                xa = xa[i_da]
                ya = ya[i_da]
                ta = ta[i_da]
                ha = ha[i_da]
                d_as = d_as[i_da]

                # Sort arrays - B
                xd = xd[i_dd]
                yd = yd[i_dd]
                td = td[i_dd]
                hd = hd[i_dd]
                d_des = d_des[i_dd]
                
                # Get distance^2 of two nearest point from ascending and descending track, respectively.
                # d_as[[0,1]] return list: [d_as[0],d_as[1]]
                da_dd = np.vstack((d_as[[0, 1]], d_des[[0, 1]]))

                ## Test if any point is too farther than the given radius
                if np.any(np.sqrt(da_dd) > radius_interp):
                    continue
                # Test if enough obs. are available for interpolation
                elif (len(xa) < nobs) or (len(xd) < nobs):
                    continue
                else:
                    # Accepted
                    pass
                
                ### -- interpolation of the xover point by nearest points.
                # compute distances（between track start point and all another points)
                # 'all points' are sorted from near to far from the cross point. 
                d_a0 = (xa - xa0) * (xa - xa0) + (ya - ya0) * (ya - ya0)
                d_d0 = (xd - xd0) * (xd - xd0) + (yd - yd0) * (yd - yd0)

                # compute distance (between track start and cross point) 
                d_ai = (xi - xa0) * (xi - xa0) + (yi - ya0) * (yi - ya0)
                d_di = (xi - xd0) * (xi - xd0) + (yi - yd0) * (yi - yd0)
                
                # Interpolate height to crossover location through the nearest nobs(number) points.
                # params: distances, heights, interpolated distance, interpolation order.
                h_ai = interp1d(d_a0[0:nobs], ha[0:nobs], d_ai, order)
                h_di = interp1d(d_d0[0:nobs], hd[0:nobs], d_di, order)
                
                # interpolate time to crossover location through the nearest nobs(number) points.
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
                out_i[0]  = xi            # crossover points x
                out_i[1]  = yi            # ... y
                out_i[2]  = h_ai          # interpolated height by ascending track
                out_i[3]  = h_di          # interpolated height by descending track
                out_i[4]  = t_ai          # interpolated time by ascending track
                out_i[5]  = t_di          # interpolated time by descending track
                out_i[6] = orb_id_as      # groud track of ascending file
                out_i[7] = orb_id_des     # groud track of decending file
                out_i[8]  = h_ai - h_di   ## height difference between ascending and descending interpolations
                out_i[9]  = t_ai - t_di   ## time difference between ...
                        
                # Add to list
                out.append(out_i)


        # operating on current tile
        # print('tile:', ki, len(ibox))
        # update counter 
        ki += 1


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



def xover_main(ifile_as, ifile_des, tspan, radius_interp, ofile_,
                vnames, tile_dxy, tile=False):
    """ 
    des: find and compute crossover values with input file paths. 
    arg:
        ifile1, ifile2: ascending and descending data respectively.
        tspan: time span, a list contians mininum time (e.g., 2018.22) and maximun time. 
        vnames: list of strings, [ovar, xvar, yvar, tvar, zvar], representing the name of groud track,
                coord_x, coord_y, time, height in the .h5 file.
    """

    startTime = datetime.now()
    # get variable names, 
    ovar, xvar, yvar, tvar, zvar = vnames

    print('crossing files:', ifile_as, ifile_des, '...')
    # Load all 1d variables needed
    with h5py.File(ifile_as, 'r') as f_as, \
         h5py.File(ifile_des, 'r') as f_des:
        #### ------ ascending file reading remove invalid values
        orbit_as  = f_as[ovar][:]
        lon_as    = f_as[xvar][:]
        lat_as    = f_as[yvar][:]
        time_as   = f_as[tvar][:]
        height_as = f_as[zvar][:]
        orbit_as  = orbit_as[np.isfinite(height_as)]
        lon_as    = lon_as[np.isfinite(height_as)]
        lat_as    = lat_as[np.isfinite(height_as)]
        time_as   = time_as[np.isfinite(height_as)]
        height_as = height_as[np.isfinite(height_as)]        
        #### ------ descending file reading and remove invalid values
        orbit_des  = f_des[ovar][:]
        lon_des    = f_des[xvar][:]
        lat_des    = f_des[yvar][:]
        time_des   = f_des[tvar][:]
        height_des = f_des[zvar][:]
        orbit_des  = orbit_des[np.isfinite(height_des)]
        lon_des    = lon_des[np.isfinite(height_des)]
        lat_des    = lat_des[np.isfinite(height_des)]
        time_des   = time_des[np.isfinite(height_des)]
        height_des = height_des[np.isfinite(height_des)]
        
    # If time span given, filter out invalid data that out the given time span
    if tspan[0] != None:
        t_as, t_des = tspan
        ## ascending file
        idx, = np.where((time_as >= t_as) & (time_as <= t_des))
        orbit_as = orbit_as[idx]
        lon_as = lon_as[idx]
        lat_as = lat_as[idx]
        time_as = time_as[idx]
        height_as = height_as[idx]
        ## descening file 
        idx, = np.where((time_des >= t_as) & (time_des <= t_des))
        orbit_des = orbit_des[idx]
        lon_des = lon_des[idx]
        lat_des = lat_des[idx]
        time_des = time_des[idx]
        height_des = height_des[idx]
        
        if len(time_as) < 3 or len(time_des) < 3:   #
            print('no points within time-span!')
            sys.exit()

    out = xover(lon_as, lat_as, time_as, height_as, orbit_as, 
            lon_des, lat_des, time_des, height_des, orbit_des, 
            nres, proj, radius_interp, tile_dxy=tile_dxy, mode_interp = 'linear')

    # create output file name if not given
    if ofile_ is None:
        path, ext = os.path.splitext(ifile_as)
        if tile:
            tilenum = '_'+str(tile_num(ifile_as))
        else:
            tilenum = '' 
        fnam = '_xovers_ad' 
        ofile = path + fnam + tilenum + ext
    else:
        ofile = ofile_

    # create output names,
    # _x represent crossover point
    oxvar = xvar                # coord-x name of crossover point 
    oyvar = yvar                # coord-y name of crossover point 
    ozvar_as = zvar+'_as'
    ozvar_des = zvar+'_des'
    otvar_as = tvar+'_as'
    otvar_des = tvar+'_des'
    oovar_as = ovar+'_as'       # orbit of file1 (ascending/descending), orbit is the ground track number(0-5)
    oovar_des = ovar+'_des'     # orbit of file2 (descending/ascending)
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
        f[oovar_as] = out[:,6]
        f[oovar_des] = out[:,7]
        f[ozvar_dif] = out[:,8]
        f[otvar_dif] = out[:,9]
                
    # Stat. variables
    dh   = out[:,8]
    dt   = out[:,9]
    dhdt = dh/dt
    print('dhdt', dhdt)
    # Compute statistics
    med_dhdt = np.around(np.median(dhdt), 3)
    std_dhdt = np.around(mad_std(dhdt), 3)
    
    # Print some statistics to screen
    print('')
    print('execution time: ' + str(datetime.now() - startTime))
    print('number of crossovers found:',str(len(out)))
    print('statistics -> median_dhdt:',med_dhdt,'std_dhdt:',std_dhdt, '(dvar/yr)')
    print('ofile name ->', ofile)


if __name__ == '__main__':

    # Read in parameters
    args   = get_args()
    ifiles = args.input[:]
    ofile_ = args.output[0]
    radius = args.radius[0]
    proj   = args.proj[0]
    tile_dxy = args.tile_dxy[0]  # it the input data will be farther tiled in the processing.
    nres = args.nres[0]
    buff   = args.buff[0]
    mode   = args.mode[0]
    vnames = args.vnames[:]
    tspan  = args.tspan[:]
    tile   = args.tile          # if the input file is tiled 

    print('parameters:')
    for arg in vars(args).items(): 
        print(arg)

    if proj == "4326":
        raise ValueError("proj can't be 4326")

    tile_dxy *= 1e3

    # Read file names
    str_as, str_des = ifiles

    # Check for tile mode
    if tile:
        # Get matching tiles
        files_as, files_des = match_tiles(str_as, str_des)
        # Loop through tiles
        for i in range(len(files_as)):            
            # Run main program
            xover_main(files_as[i], files_des[i], tspan=tspan, radius_interp=radius, 
                        ofile_=ofile_,vnames=vnames, tile_dxy=tile_dxy, tile=tile)

    # Run as single files
    else:
        # File names
        file_as, file_des = str_as, str_des
        # Run main program
        xover_main(file_as, file_des, tspan=tspan, radius_interp=radius, 
                    ofile_=ofile_,vnames=vnames, tile_dxy=tile_dxy, tile=False)



