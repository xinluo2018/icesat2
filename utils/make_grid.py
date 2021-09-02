import numpy as np

def make_grid(xmin, xmax, ymin, ymax, 
                        dx, dy, return_2d=True):
    """
    des: Construct 2D-grid given input boundaries
    args:
        xmin,xmax, ymin,ymax: x-coord. min, x-coord. max, y-coors. min, y-coord. max
        dx,dy: x-resolution, y-resolution        
        return_2d: if true return grid otherwise vector
    return: 
        2D grid or 1D vector
    """
    Nn = int((np.abs(ymax - ymin)) / dy) + 1
    Ne = int((np.abs(xmax - xmin)) / dx) + 1
    xi = np.linspace(xmin, xmax, num=Ne)
    yi = np.linspace(ymin, ymax, num=Nn)    
    if return_2d:
        return np.meshgrid(xi, yi)
    else:
        return xi, yi
