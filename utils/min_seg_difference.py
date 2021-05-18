import numpy as np


def min_seg_difference(D6):
    """
    seg_difference_filter: Use elevations and slopes to find bad ATL06 segments
    Inputs: 
        D6: a granule of ATL06 data, in dictionary format.  Must have entries:
            x_atc, h_li, dh_fit_dx
    Returns:
        delta_h_seg: the minimum absolute difference between each segment's endpoints and those of its two neighbors
    """
    h_ep=np.zeros([2, D6['h_li'].size])+np.NaN
    h_ep[0, :]=D6['h_li']-D6['dh_fit_dx']*20  # segment head
    h_ep[1, :]=D6['h_li']+D6['dh_fit_dx']*20  # segment tail
    delta_h_seg=np.zeros_like(D6['h_li'])     # 
    # (segment center) - (previous segment tail), should be equal in theory.
    delta_h_seg[1:]=np.abs(D6['h_li'][1: ]-h_ep[1, :-1])  
    # compare with (segment center) - (next segment head)
    delta_h_seg[:-1]=np.minimum(delta_h_seg[:-1], \
                                    np.abs(D6['h_li'][:-1]-h_ep[0, 1:]))
    return delta_h_seg
