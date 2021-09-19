## author: xin luo
## create: 2021.9.19
## des: griding-related functions.


import numpy as np
from scipy import stats


def make_grid(xmin, xmax, ymin, ymax, 
                        dx, dy, return_2d=True):
    """
    des: construct 2D-grid given input boundaries and resolution;
         get grid coordinates (up-left corner of each grid)
    args:
        xmin,xmax, ymin,ymax: x-coord. min, x-coord. max, y-coors. min, y-coord. max
        dx,dy: x-resolution, y-resolution        
        return_2d: if true return 2d coords (x,y are 2d array) 
                   otherwise 1d coords (x,y are 1d vector).
    return: 
        2D grid or 1D vector
    """
    row = int((np.abs(ymax - ymin)) / dy) + 1
    col = int((np.abs(xmax - xmin)) / dx) + 1
    xi = np.linspace(xmin, xmax, num=col)
    yi = np.linspace(ymin, ymax, num=row)    
    if return_2d:
        return np.meshgrid(xi, yi)
    else:
        return xi, yi



def get_grid_coor(xmin, xmax, ymin, ymax, dxy):
    """
    des: get grid coordinates (four corners of each grid). 
    Args:
        xmin/xmax/ymin/ymax: must be in projection: stereographic (m).
        dxy: size of each box.
    retrun:
        list, consists of corner coordinates of each grid.
    """
    # Number of tile edges on each dimension 
    row = int(np.abs(ymax - ymin) / dxy) + 1   # row
    col = int(np.abs(xmax - xmin) / dxy) + 1   # col
    # Coord of tile edges for each dimension
    xg = np.linspace(xmin, xmax, col)
    yg = np.linspace(ymin, ymax, row)
    # Vector of bbox for each cell, coordinates of corners of each pixel.
    grids_coor = [(w,e,s,n) for w,e in zip(xg[:-1], xg[1:]) 
                                  for s,n in zip(yg[:-1], yg[1:])]
    del xg, yg
    return grids_coor


def get_grid_id(x, y, xmin, xmax, ymin, ymax, dxy, buff):
    """
    des: get grid id for each given points (x,y).
    arg:
        x,y: coordinates of the photon points
        xmin/xmax/ymin/ymax: must be in grid projection: stereographic (m).
        dxy: grid-cell size.
        buff: buffer region, unit is same to x, y
    return:
        the index of each points corresponding to the generated bins.  
    """
    # Number of tile edges on each dimension
    row = int(np.abs(ymax - ymin) / dxy) + 1
    col = int(np.abs(xmax - xmin) / dxy) + 1
    # Coord of tile edges for each dimension
    xg = np.linspace(xmin-buff, xmax+buff, col)
    yg = np.linspace(ymin-buff, ymax+buff, row)
    # Indicies for each points
    grids_id = stats.binned_statistic_2d(x, y, np.ones(x.shape), 'count', bins=[xg, yg]).binnumber
    return grids_id

