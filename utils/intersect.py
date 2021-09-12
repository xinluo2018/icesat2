# author: xin luo
# create: 2021.8.8
# des: get the cross points between ascending and descending orbit points.



import numpy as np


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
    ## interplate the xover time corresponding to down and up tracks, respectively.
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




