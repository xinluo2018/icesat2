import numpy as np

def make_grid(xmin, xmax, ymin, ymax, dx, dy, return_2d=True):
    """
    Construct 2D-grid given input boundaries
    :param xmin: x-coord. min
    :param xmax: x-coord. max
    :param ymin: y-coors. min
    :param ymax: y-coord. max
    :param dx: x-resolution
    :param dy: y-resolution
    :param return_2d: if true return grid otherwise vector
    :return: 2D grid or 1D vector
    """

    Nn = int((np.abs(ymax - ymin)) / dy) + 1
    Ne = int((np.abs(xmax - xmin)) / dx) + 1

    xi = np.linspace(xmin, xmax, num=Ne)
    yi = np.linspace(ymin, ymax, num=Nn)
    
    if return_2d:
        return np.meshgrid(xi, yi)
    else:
        return xi, yi
