import numpy as np

def mad_std(x, axis=None):
    """
    Robust std.dev using median absolute deviation

    :param x: data values
    :param axis: target axis for computation
    :return: std.dev (MAD)
    """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)
