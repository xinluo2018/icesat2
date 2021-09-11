# author: xin luo, create: 2021.8.8

'''
des: some functions used multiple times in another programs
'''


import numpy as np
from astropy.time import Time
import statsmodels.api as sm
from scipy import stats

def gps2dyr(time):
    """ Converte from GPS time to decimal years. """
    time = Time(time, format="gps")
    time = Time(time, format="decimalyear").value
    return time

def orbit_type(time, lat):
    """
    des: determines ascending and descending tracks
         through testing whether lat increases when time increases.
    arg:
        time, lat: time and latitute of the pohton points.
    return:
        i_asc, i_des: track of the photon points, 1-d data consist of True/Talse. 
    """
    tracks = np.zeros(lat.shape)
    # set track values, !!argmax: the potential turn point of the track
    tracks[0: np.argmax(np.abs(lat))] = 1
    i_asc = np.zeros(tracks.shape, dtype=bool)

    # loop through unique tracks: [0]/[1]/[0,1]
    for track in np.unique(tracks):
        (i_track,) = np.where(track == tracks)
        if len(i_track) < 2:  # number of photon points of specific track i 
            continue
        i_time_min, i_time_max  = time[i_track].argmin(), time[i_track].argmax()
        lat_diff = lat[i_track][i_time_max] - lat[i_track][i_time_min]
        # Determine track type
        if lat_diff > 0:
            i_asc[i_track] = True
    return i_asc, np.invert(i_asc)

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


def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). 
        mad: median absolutely deviation"""
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)


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

# def intersect(x_down, y_down, x_up, y_up):
#     ''' 
#     reference: 
#         https://stackoverflow.com/questions/17928452/
#         find-all-intersections-of-xy-data-point-graph-with-numpy
#     des: 
#         Find orbit crossover locations through solving the equation: 
#         p0 + s*(p1-p0) = q0 + t*(q1-q0) ---> s*(p1-p0)-t*(q1-q0) = q0-p0; 
#         p and q are descending and ascending points respectively.
#         if s and t belong to [0,1], p and q actually do intersect.
#     '''
#     xover_points = []
#     for i in range(len(x_down)-1):
#         p0 = np.array([x_down[i], y_down[i]])
#         p1 = np.array([x_down[i+1], y_down[i+1]])
#         for j in range(len(x_up)-1):
#             q0 = np.array([x_up[j], y_up[j]])
#             q1 = np.array([x_up[j+1], y_up[j+1]])
#             params = np.linalg.solve(np.column_stack((p1-p0, q0-q1)), q0-p0)
#             if np.all((params >= 0) & (params <= 1)):
#                 xover_point = p0 + params[0]*(p1 - p0)
#                 xover_points.append(xover_point)
#     return np.array(xover_points)


# def intersect_(x_down, y_down, x_up, y_up):
#     ### more fast, but difficult to understand
#     """  reference: https://stackoverflow.com/questions/17928452/
#          find-all-intersections-of-xy-data-point-graph-with-numpy
#     des: Find orbit crossover locations through solving the equation: 
#          p0 + s*(p1-p0) = q0 + t*(q1-q0); p and q are descending and ascending points respectively.
#          ---> s*(p1-p0)-t*(q1-q0) = q0-p0
#          if s and t belong to [0,1], p and q actually do intersect.
#          !! in order to speed up calculation, this code vectorizing solution of the 2x2 linear systems
#     input:
#          x_down, y_down: coord_x and coord_y of the descending points
#          x_up, y_up: coord_x, coord_y of the ascending points.
#     retrun:
#          np.array(shape: (n,2)), coordinates (x,y) of the intersection points. 
#          n is number of intersection points
#     """
#     ## TODO: add the time attribute 
#     p = np.column_stack((x_down, y_down))   # coords of the descending points
#     q = np.column_stack((x_up, y_up))       # coords of the ascending points

#     (p0, p1, q0, q1) = p[:-1], p[1:], q[:-1], q[1:]   # remove first/last row respectively
#     # (num_uppoints, 2) - (num_dpoints, 1, 2), array broadcast, dim: (num_dpoints, num_uppoints, 2)
#     rhs = q0 - p0[:, np.newaxis, :]    

#     mat = np.empty((len(p0), len(q0), 2, 2))  # dim: (p_num, q_num, dif((x, y)), orbit(down,up))
#     mat[..., 0] = (p1 - p0)[:, np.newaxis]  # dif (x_down,y_down) between point_down and previous point_down
#     mat[..., 1] = q0 - q1      #  dif (x_up, y_up) between point_up and previous point_up
#     mat_inv = -mat.copy()
#     mat_inv[..., 0, 0] = mat[..., 1, 1]   # exchange between x_dif and y_dif, down and up
#     mat_inv[..., 1, 1] = mat[..., 0, 0]

#     det = mat[..., 0, 0] * mat[..., 1, 1] - mat[..., 0, 1] * mat[..., 1, 0]
#     mat_inv /= det[..., np.newaxis, np.newaxis]    # ???
#     params = mat_inv @ rhs[..., np.newaxis]        # 
#     intersection = np.all((params >= 0) & (params <= 1), axis=(-1, -2)) #
#     p0_s = params[intersection, 0, :] * mat[intersection, :, 0]

#     return p0_s + p0[np.where(intersection)[0]]


def intersect(x_up, y_up, x_down, y_down, t_up, \
                                t_down, z_up =None, z_down=None):
    """
    !!! more fast, but difficult to understand, moreover, we add the time and height input
    reference: 
        https://stackoverflow.com/questions/17928452/
        find-all-intersections-of-xy-data-point-graph-with-numpy
    des: Find orbit crossover locations through solving the equation: 
        p0 + s*(p1-p0) = q0 + t*(q1-q0); p and q are descending and ascending points respectively.
        ---> s*(p1-p0)-t*(q1-q0) = q0-p0
        if s and t belong to [0,1], p and q actually do intersect.
        !! in order to speed up calculation, this code vectorizing solution of the 2x2 linear systems
    input:
        x_down, y_down: coord_x and coord_y of the descending points
        x_up, y_up: coord_x, coord_y of the ascending points.
        t_down, t_up: time of down track and up track, respectively.
        z_down, z_up: height of down track and up track, respectively.
    retrun:
          np.array(shape: (n,2)), coordinates (x,y) of the intersection points. 
          n is number of intersection points
    """
    p = np.column_stack((x_down, y_down))   # coords of the descending points
    q = np.column_stack((x_up, y_up))       # coords of the ascending points

    (p0, p1, q0, q1) = p[:-1], p[1:], q[:-1], q[1:]   # remove first/last row respectively
    # (num_uppoints, 2) - (num_dpoints, 1, 2), array broadcast, dim: (num_dpoints, num_uppoints, 2)
    rhs = q0 - p0[:, np.newaxis, :]    

    mat = np.empty((len(p0), len(q0), 2, 2))  # dim: (p_num, q_num, dif((x, y)), orbit(down,up))
    mat[..., 0] = (p1 - p0)[:, np.newaxis]  # dif (x_down,y_down) between point_down and previous point_down
    mat[..., 1] = q0 - q1      #  dif (x_up, y_up) between point_up and previous point_up
    mat_inv = -mat.copy()
    mat_inv[..., 0, 0] = mat[..., 1, 1]   # exchange between x_dif and y_dif, down and up
    mat_inv[..., 1, 1] = mat[..., 0, 0]

    det = mat[..., 0, 0] * mat[..., 1, 1] - mat[..., 0, 1] * mat[..., 1, 0]
    mat_inv /= det[..., np.newaxis, np.newaxis]    # ???
    params = mat_inv @ rhs[..., np.newaxis]        # 
    intersection = np.all((params >= 0) & (params <= 1), axis=(-1, -2)) #
    p0_s = params[intersection, 0, :] * mat[intersection, :, 0]
    xover_coords = p0_s + p0[np.where(intersection)[0]]
    ## interplate the xover time of down and up tracks respectively.
    ## -- get the previous point of xover point (down and up, respectively)
    p_start_idx = np.where(intersection)[0]   # down track
    q_start_idx = np.where(intersection)[1]   # up track
    ## -- calculate the distance from xover
    d_p = np.sqrt(np.sum((p[p_start_idx+1]-p[p_start_idx])*(p[p_start_idx+1]-p[p_start_idx]), axis=1)) 
    d_pi = np.sqrt(np.sum((xover_coords-p[p_start_idx])*(xover_coords-p[p_start_idx]), axis=1))
    d_q = np.sqrt(np.sum((q[q_start_idx+1]-q[q_start_idx])*(q[q_start_idx+1]-q[q_start_idx]), axis=1)) 
    d_qi = np.sqrt(np.sum((xover_coords-q[q_start_idx])*(xover_coords-q[q_start_idx]), axis=1))
    ## -- get the interpolated time
    dt_down, dt_up = t_down[p_start_idx+1]-t_down[p_start_idx], t_up[q_start_idx+1]-t_up[q_start_idx]
    xover_t_down = t_down[p_start_idx] + (dt_down)*(d_pi/d_p)
    xover_t_up = t_up[q_start_idx] + (dt_up)*(d_qi/d_q)
    ## remove unreasonable xover points.
    idx_save = np.argwhere((dt_down<0.0001) & (dt_up<0.0001)).flatten()
    xover_coords, xover_t_down, xover_t_up = xover_coords[idx_save,:], xover_t_down[idx_save], xover_t_up[idx_save]

    ## -- get the interpolated height
    if z_down is None:
        return xover_coords[:,0], xover_coords[:,1], xover_t_up, xover_t_down
    else:
        xover_z_down = z_down[p_start_idx] + (z_down[p_start_idx+1]-z_down[p_start_idx])*(d_pi/d_p)
        xover_z_up = z_up[q_start_idx] + (z_up[q_start_idx+1]-z_up[q_start_idx])*(d_qi/d_q)
        return xover_coords[:,0], xover_coords[:,1], xover_t_up, xover_t_down, xover_z_up, xover_z_down




def get_bboxs_coor(xmin, xmax, ymin, ymax, dxy):
    """
    des: get bin boxes (bbox) coordinates (four corners of each box). 
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
    des: get binn box (bbox) id of each points (x,y).
    arg:
        x,y: coordinates of the photon points
        xmin/xmax/ymin/ymax: must be in grid projection: stereographic (m).
        dxy: grid-cell size.
        buff: buffer region, unit is same to x, y
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

