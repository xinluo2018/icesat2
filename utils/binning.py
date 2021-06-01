import numpy as np

def mad_std(x, axis=None):
    """
    Robust std.dev using median absolute deviation
    :param x: data values
    :param axis: target axis for computation
    :return: std.dev (MAD)
    """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)

def binning(x, y, xmin=None, xmax=None, dx=1 / 12.,
             window=3 / 12., interp=False, median=False):
    """Time-series binning (w/overlapping windows).

        Args:
        x,y: time and value of time series.
        xmin,xmax: time span of returned binned series.
        dx: time step of binning.
        window: size of binning window.
        interp: interpolate binned values to original x points.
    """
    if xmin is None:
        xmin = np.nanmin(x)
    if xmax is None:
        xmax = np.nanmax(x)

    steps = np.arange(xmin, xmax, dx)  # time steps
    bins = [(ti, ti + window) for ti in steps]  # bin limits

    N = len(bins)
    yb = np.full(N, np.nan)
    xb = np.full(N, np.nan)
    eb = np.full(N, np.nan)
    nb = np.full(N, np.nan)
    sb = np.full(N, np.nan)

    for i in range(N):

        t1, t2 = bins[i]
        idx, = np.where((x >= t1) & (x <= t2))

        if len(idx) == 0:
            xb[i] = 0.5 * (t1 + t2)
            continue

        ybv = y[idx]

        if median:
            yb[i] = np.nanmedian(ybv)
        else:
            yb[i] = np.nanmean(ybv)

        xb[i] = 0.5 * (t1 + t2)
        eb[i] = mad_std(ybv)
        nb[i] = np.sum(~np.isnan(ybv))
        sb[i] = np.sum(ybv)

    if interp:
        try:
            yb = np.interp(x, xb, yb)
            eb = np.interp(x, xb, eb)
            sb = np.interp(x, xb, sb)
            xb = x
        except:
            pass

    return xb, yb, eb, nb, sb

