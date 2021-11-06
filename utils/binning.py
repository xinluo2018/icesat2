## author: xin luo; 
## create: 2021.9.19; 


import numpy as np

def binning_1d(x, y, xmin=None, xmax=None, dx=1 / 12.,
                            window=3 / 12., interp=False, median=False):
    """ des: 1-dimentional binning, e.g., time-series data binning.
        args:
            x, y: independent and dependent variable, e.g., time and value of the time series data.
            xmin, xmax: range of the variable x.
            dx: resolution of x. e.g., 1/12 represents 1 month if the unit of x is year.
            window: size of binning window. 3/12 represents 3 month if the unit of x is year
            interp: interpolate bin values to x points. if none, one bin has one value.
            median: median value of the bin values, if not set, the mean value is calculated.
        return: 
            xb, yb: x, y corresponding to in each bin center, or interpolation point
            eb, nb, sb: error, number, sum statics of each bin
    """
    if xmin is None:
        xmin = np.nanmin(x)
    if xmax is None:
        xmax = np.nanmax(x)

    steps = np.arange(xmin, xmax, dx)   # time steps
    bins = [(ti, ti + window) for ti in steps]    #s

    N = len(bins)  
    xb = np.full(N, np.nan)         # times corresponding to bins
    yb = np.full(N, np.nan)         # values corresponding to bins
    eb = np.full(N, np.nan)         # mads corresponding to bins
    nb = np.full(N, np.nan)         # counts of valid values in bins
    sb = np.full(N, np.nan)         # sum of values in bins

    ## loops for each bin
    for i in range(N):
        t1, t2 = bins[i]
        idx, = np.where((x >= t1) & (x <= t2))

        if len(idx) == 0:
            xb[i] = 0.5 * (t1 + t2)     # determine the time of bin as the center of the time window
            continue                    # finish this loop       

        ybv = y[idx]                    # values in specific bin

        if median:
            yb[i] = np.nanmedian(ybv)
        else:
            yb[i] = np.nanmean(ybv)

        xb[i] = 0.5 * (t1 + t2)         # determine the time of the bin
        eb[i] = 1.4826 * np.nanmedian(np.abs(ybv - np.nanmedian(ybv)))  # mad-based std
        nb[i] = np.sum(~np.isnan(ybv))  # counts of the valid values in bin
        sb[i] = np.sum(ybv)             # sum of the values in bin

    if interp:  
        try:
            yb = np.interp(x, xb, yb)     ## interpolate the values to the given time in the bin
            eb = np.interp(x, xb, eb)     ## interpolate the mad ... 
            sb = np.interp(x, xb, sb)     ## interpolate the sum of the values ...
            xb = x
        except:
            pass
    return xb, yb, eb, nb, sb
