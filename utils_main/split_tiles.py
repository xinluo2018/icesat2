## author: Fernando Paolo
## modify: xin luo, 2021.8.28


"""
des: split the very-large files into multiple tiles.
"""

import os
import h5py 
import pyproj
import argparse
import numpy as np
from glob import glob

# Optimal chunk size of points.
chunks = 100000

def get_args():
    """ Get command-line arguments. """
    parser = argparse.ArgumentParser(
            description='Split geographical data into (overlapping) tiles')
    parser.add_argument(
            'file', metavar='file', type=str, nargs='+',
            help='single or multiple file(s) to split in tiles (HDF5)')
    parser.add_argument(
            '-b', metavar=('w','e','s','n'), dest='bbox', type=float, nargs=4,
            help=('bounding box for geographic region (deg or m)'),
            default=[None,None,None,None],)
    parser.add_argument(
            '-d', metavar=('dxy'), dest='dxy', type=float, nargs=1,
            help=('block size of tile (km)'),
            default=[], required=True)
    parser.add_argument(
            '-r', metavar=('buffer'), dest='bf', type=float, nargs=1,
            help=("overlap between tiles (km)"),
            default=[0],)
    parser.add_argument(
            '-c', metavar=('lon','lat'), dest='coord_name', type=str, nargs=2,
            help=('name of x/y variables'),
            default=[None, None],)
    parser.add_argument(
            '-p', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
            help=('EPSG proj number (AnIS=3031, GrIS=3413)'),
            default=['3031'],)
    parser.add_argument(
            '-n', metavar=('njobs'), dest='njobs', type=int, nargs=1,
            help="for parallel writing of multiple tiles, optional",
            default=[1],)
    return parser.parse_args()


def print_args(args):
    print('Input arguments:')
    for arg in list(vars(args).items()):
        print(arg)

def coor2coor(srs_from, srs_to, x, y):
    """
    Transform coordinates from srs_from to srs_to
    input:
        srs_from and srs_to are EPSG number (e.g., 4326, 3031)
        x and y are x-coord and y-coord corresponding to srs_from    
    return:
        x-coord and y-coord in srs_to 
    """
    srs_from = pyproj.Proj(int(srs_from))
    srs_to = pyproj.Proj(int(srs_to))
    return pyproj.transform(srs_from, srs_to, x, y, always_xy=True)


def get_xy(ifile, coord_name=['lon', 'lat'], proj='3031'):
    """ Get lon/lat from input file and convert to x/y. """
    lonvar, latvar = coord_name
    with h5py.File(ifile,'r') as fi:
        lon = fi[lonvar][:]
        lat = fi[latvar][:]
    return coor2coor(4326, proj, lon, lat)


def add_suffix(fname, suffix=''):
    path, ext = os.path.splitext(fname)
    return path + suffix + ext


def get_tile_bboxs(grid_bbox, dxy):
    """
    des:    
        Define bbox of tiles given bbox of grid and tile size. 
    """
    xmin, xmax, ymin, ymax = grid_bbox

    # Number of tile edges on each dimension    
    Nns = int(np.abs(ymax - ymin) / dxy) + 1   # row
    New = int(np.abs(xmax - xmin) / dxy) + 1   # col
    
    # Coord of tile edges for each dimension
    xg = np.linspace(xmin, xmax, New)   
    yg = np.linspace(ymin, ymax, Nns)
    
    # Vector of bbox for each tile   ##NOTE: Nested loop!
    bboxs = [(w,e,s,n) for w, e in zip(xg[:-1], xg[1:]) 
                       for s, n in zip(yg[:-1], yg[1:])]
    del xg, yg
    return bboxs

def get_tile_data(ifile, x, y, bbox, buff=1, proj='3031', tile_num=0):
    """ 
    des:
        Extract data within bbox and save to individual file. 
    args:
        buff: unit is meter. 
    """
    xmin, xmax, ymin, ymax = bbox   # given region

    suffix = ('_buff_%g_epsg_%s_tile_%03d' % (buff, proj, tile_num))
    ofile = add_suffix(ifile, suffix)

    with h5py.File(ifile, 'r') as fi:
        vnames = list(fi.keys())
        vars = [fi[vname][:] for vname in vnames ]

    out = [[] for vname in vnames]
    out = dict(zip(vnames, out))
    npts = 0             # conunt of the written points
    nrow = x.shape[0]

    # read and write in chunks, i: start of each chunk
    for i in range(0, nrow, chunks):
        k = min(i+chunks, nrow)   ## avoid the k exceeds the index range of x.
        x_chunk = x[i:k]
        y_chunk = y[i:k]

        # get the tile indices, 
        idx, = np.where( (x_chunk >= xmin-buff*1e3) & (x_chunk <= xmax+buff*1e3) & 
                         (y_chunk >= ymin-buff*1e3) & (y_chunk <= ymax+buff*1e3) )

        # Leave chunk if outside tile
        if len(idx) == 0: continue

        # get chunk of data of the tile
        vars_chunk = [d[i:k] for d in vars]         # -> obtain the points in the chunk
        vars_chunk_region = [d[idx] for d in vars_chunk]   # -> obtain the points in the given region
        # Save chunk
        for (i, vname) in enumerate(vnames):
            out[vname] = out[vname] + list(vars_chunk_region[i])
        npts += vars_chunk_region[0].shape[0]

    if npts != 0: 
        with h5py.File(ofile, 'w') as fo:
            [fo.create_dataset(name = vname, shape = (npts,), dtype='float64')
                        for vname in vnames]
            for vname in vnames:
                fo[vname][:] = out[vname]
        print(('tile %03d: #points' % tile_num, npts, '...'))

def count_files(ifiles, key='*tile*'):
    saved = []
    for f in ifiles: 
        path, ext = os.path.splitext(f)
        saved.extend(glob(path + key))
    return len(saved)

if __name__ == '__main__':

    # Pass arguments 
    args = get_args()
    ifiles = args.file[:]       # input file(s)
    coord_name = args.coord_name[:]     # lon/lat variable names
    bbox_ = args.bbox           # bounding box EPSG (m) or geographical (deg)
    bf = args.bf[0] * 1e3       # buffer (km -> m)
    dxy = args.dxy[0] * 1e3     # tile length (km -> m)
    proj = args.proj[0]         # EPSG proj number
    njobs = args.njobs[0]       # parallel writing

    print_args(args)
    if len(ifiles) == 1: 
        ifiles = glob(ifiles[0])   # pass str if "Argument list too long"

    # Generate list of items (only once!) for parallel proc
    print('generating list of tasks (files x tiles) ...')

    # NOTE: To avoid reprojecting coords (lon/lat -> x/y) several
    # times for each file (one conversion per tile), coordinates are
    # loaded into mem, converted once per file and passed to each worker.

    xys = [get_xy(f, coord_name, proj) for f in ifiles]                          # [(x1,y1), (x2,y2)..]

    if not bbox_[0]:
        xy = np.vstack(xys)
        xmin, ymin = np.min(xy, axis=1)
        xmax, ymax = np.max(xy, axis=1)
        bbox_ = [xmin, xmax, ymin, ymax]

    bboxs = get_tile_bboxs(bbox_, dxy)                                        # [b1, b2..]
    fxys = [(f,x,y) for f, (x,y) in zip(ifiles, xys)]                         # [(f1,x1,y1), (f2,x2,y2)..]
    fxybs = [(f,x,y,b,n+1) for (f,x,y) in fxys for n, b in enumerate(bboxs)]  # [(f1,x1,y1,b1,1), (f1,x1,y1,b2,2)..]

    print(('number of files:', len(ifiles)))
    print(('number of tiles:', len(bboxs)))
    print(('number of tasks:', len(fxybs)))

    # NOTE: For each bbox scan full file

    if njobs == 1:
        print('running sequential code ...')
        [get_tile_data(f, x, y, b, bf, proj, n) for f,x,y,b,n in fxybs]

    else:
        print(('running parallel code (%d jobs) ...' % njobs))
        from joblib import Parallel, delayed
        Parallel(n_jobs=njobs, verbose=5)(
                delayed(get_tile_data)(f, x, y, b, bf, proj, n) for f,x,y,b,n in fxybs)  

    print(('number of tiles with data:', count_files(ifiles)))