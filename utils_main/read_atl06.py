# author: Fernando Paolo, 
# modify: xin luo, 2021.8.10.   

'''
des: read and split icesat2 alt06 data by beams 
     and orientations (ascending/descending).
example:
    python readatl06.py ./input/path/*.h5 -o /output/path/dir -n 4
'''

import os
import h5py
import numpy as np
import argparse
from helper import gps2dyr, orbit_type


def get_args():

    description = "read ICESat-2 ATL06 data files by groud track and orbit."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(         
            "ifiles", metavar="ifiles", type=str, nargs="+",
            help="input files to read (.h5).")
    parser.add_argument(           
            '-o', metavar=('outdir'), dest='outdir', type=str, nargs=1,
            help='path to output folder', 
            default=[""])
    parser.add_argument(
            "-n", metavar=("njobs"), dest="njobs", type=int, nargs=1,
            help="number of cores to use for parallel processing", 
            default=[1])
    return parser.parse_args()

def read_atl06(file_in, dir_out):
    ''' 
    des: 
        split icesat2 atl06 data by ground tracks/spots and orbits.
        spot 1,2,3,4,5,6 are always corresponding to beam 1,2,3,4,5,6 
        and spot 1,3,5 are strong beams, spot 2, 4, 6 are weak beams.
        users can add the interested variables by themself.
    arg: 
        file_in: atl06 file, .h5 format
        path_out: path to save the splitted atl06 data
        group: list, beam name to be read, e.g., ['./gt1l', './gt1r']
        orbit: list, orbit name to be read, e.g., ['ascending']
    return:
        splitted icesat2 atl06 data: splitted by spots/beams and orbits (ascending/descending) 
    '''
 
    group = ["./gt1l", "./gt1r", "./gt2l", "./gt2r", "./gt3l", "./gt3r"]

    ## loop for groups
    for k in range(len(group)):

        #-----------------------------------#
        # 1) Read in data for a single beam #
        #-----------------------------------#

        with h5py.File(file_in, "r") as fi:
            try:
                ## group varibales:
                lat = fi[group[k] + "/land_ice_segments/latitude"][:]
                lon = fi[group[k] + "/land_ice_segments/longitude"][:]
                h_li = fi[group[k] + "/land_ice_segments/h_li"][:]
                t_dt = fi[group[k] + "/land_ice_segments/delta_time"][:]
                quality = fi[group[k] + "/land_ice_segments/atl06_quality_summary"][:]
                ## dset varibales
                tref = fi["/ancillary_data/atlas_sdp_gps_epoch"][:]
                cycle = fi["/orbit_info/cycle_number"][:] * np.ones(len(lat))
                rgt = fi["/orbit_info/rgt"][:] * np.ones(len(lat))
                ## group attributes
                beam_type = fi[group[k]].attrs["atlas_beam_type"].decode()
                spot_number = fi[group[k]].attrs["atlas_spot_number"].decode()   #  ??
            except:
                print(("missing group:", group[k]))
                print(("in file:", file_in))
                continue

        ## set beam type: 1 -> strong, 0 -> weak
        if beam_type == "strong":
            beam_types = np.ones(lat.shape)
        else:
            beam_types = np.zeros(lat.shape)

        #----------------------------------------------------#
        # 3) Convert time, separate orbits                   #
        #----------------------------------------------------#

        ### --- creating array of spot numbers
        spot = float(spot_number) * np.ones(lat.shape)   
        t_gps = t_dt + tref
        t_li = gps2dyr(t_gps)    #  time in decimal years
        ### --- obtain orbit type
        (i_asc, i_des) = orbit_type(t_li, lat)    #  track type (asc/des)        
        name, ext = os.path.splitext(os.path.basename(file_in))
        file_out = os.path.join(
            dir_out, name + "_" + group[k][2:] + "_spot" + spot_number + ext
            )
        #------------------------------------------#
        # 3) Writting out the selected data        #
        #------------------------------------------#
        # save ascending data
        if len(lat[i_asc]) > 1:
            with h5py.File(file_out.replace(".h5", "_A.h5"), "w") as fa:
                fa["lon"] = lon[i_asc][:]
                fa["lat"] = lat[i_asc][:]
                fa["h_elv"] = h_li[i_asc][:]
                fa["t_year"] = t_li[i_asc][:]
                fa["cycle"] = cycle[i_asc][:]
                fa["beam_type"] = beam_types[i_asc][:]
                fa["spot"] = spot[i_asc][:]     #  corresponding to each beam.
                fa["rgt"] = rgt[i_asc][:]
                fa["quality_summary"] = quality[i_asc][:]
                ostr = "_A.h5"
            print('written file:', (file_out.replace(".h5", ostr)))

        # save desending data
        if len(lat[i_des]) > 1:
            with h5py.File(file_out.replace(".h5", "_D.h5"), "w") as fd:
                fd["lon"] = lon[i_des][:]
                fd["lat"] = lat[i_des][:]
                fd["h_elv"] = h_li[i_des][:]
                fd["t_year"] = t_li[i_des][:]
                fd["cycle"] = cycle[i_des][:]
                fd["beam_type"] = beam_types[i_des][:]
                fd["spot"] = spot[i_des][:]
                fd["rgt"] = rgt[i_des][:]
                fd["quality_summary"] = quality[i_des][:]
                ostr = "_D.h5"
            print('written file:', (file_out.replace(".h5", ostr)))
        # Update orbit number
    return


if __name__ == '__main__':

    ### ---- read input from command line
    args = get_args()
    ifiles = args.ifiles
    dir_out = args.outdir[0]
    njobs = args.njobs[0]

    if njobs == 1:
        print("running in serial ...")
        [read_atl06(f, dir_out) for f in ifiles]
    else:
        print(("running in parallel (%d jobs) ..." % njobs))
        from joblib import Parallel, delayed
        Parallel(n_jobs=njobs, verbose=5)(
                delayed(read_atl06)(f, dir_out) for f in ifiles)




