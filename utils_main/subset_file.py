
# create: xin luo, 2021.8.30.   


'''
des: subset the icesat2 data by given region and time.

example:
    python subset.py ./input/path/*.h5 -r 
'''


import os
import h5py 
import pyproj
import argparse
import numpy as np
from glob import glob


def get_args():

    """ Get command-line arguments. """
    parser = argparse.ArgumentParser(
            description='subset icesat2 data by region and time')
    parser.add_argument(
            'ifile', metavar='ifile', type=str, nargs='+',
            help='single or multiple file(s) (HDF5)')
    parser.add_argument(
            '-r', metavar=('w','e','s','n'), dest='region', type=float, nargs=4,
            help=('region for data subset (km)'),
            default=[None,None,None,None])
    parser.add_argument(
            '-t', metavar=('time'), dest='time', type=float, nargs=2,
            help=('time for data subset'),
            default=[None, None])
    parser.add_argument(
            '-c', metavar=('lon','lat'), dest='coord_name', type=str, nargs=2,
            help=('name of x/y variables'),
            default=['lon', 'lat'])
    parser.add_argument(
            '-tn', metavar=('time'), dest='time_name', type=str, nargs=1,
            help=('name of time variables'),
            default=['t_year'])

    return parser.parse_args()


def subset_file(ifile, region, time_range, coord_name, time_name):

    print(('input -> ', ifile))
    lon_name, lat_name = coord_name
    time_start, time_end = time_range
    lonmin, lonmax, latmin, latmax = region   # given region

    path, ext = os.path.splitext(ifile)
    ofile = path+'_subs'+ ext

    with h5py.File(ifile, 'r') as fi:
        vnames = list(fi.keys())
        vars = [fi[vname][:] for vname in vnames ]

    vars_dict = [[] for vname in vnames]
    vars_dict = dict(zip(vnames, vars))

    idx_region = np.where( (vars_dict[lon_name] >= lonmin) & (vars_dict[lon_name] <= lonmax) & 
                        (vars_dict[lat_name] >= latmin) & (vars_dict[lat_name] <= latmax) )

    for vname in vnames:
        #### -- subset by specific region 
        vars_dict[vname] = vars_dict[vname][idx_region]

    if vars_dict[lon_name].any():        
        #### -- subset by specific time 
        idx_time = np.where( (vars_dict[time_name] >= time_start) & (vars_dict[time_name] <= time_end) )
        for vname in vnames:
            vars_dict[vname] = vars_dict[vname][idx_time]

    #### ----- write out file
    if vars_dict[lon_name].any():
        with h5py.File(ofile, 'w') as fo:
            for vname in vnames:
                fo[vname] = vars_dict[vname]
            print(('output ->', ofile))
    else:
        print('output -> None')


if __name__ == '__main__':

    # Pass arguments 
    args = get_args()
    ifiles = args.ifile[:]           # input file(s)
    region = args.region[:]         # lon/lat variable names
    coord_name = args.coord_name[:]
    time_range = args.time                # bounding box EPSG (m) or geographical (deg)
    time_name = args.time_name[0]   

    print('Input arguments:')
    for arg in list(vars(args).items()):
        print(arg)

    [subset_file(f, region, time_range, coord_name, time_name) for f in ifiles]




