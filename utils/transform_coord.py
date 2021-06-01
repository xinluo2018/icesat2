import pyproj

def transform_coord(proj1, proj2, x, y):
    """
    Transform coordinates from proj1 to proj2
    usgin EPSG number
    :param proj1: current projection (4326)
    :param proj2: target projection (3031)
    :param x: x-coord in current proj1
    :param y: y-coord in current proj1
    :return: x and y now in proj2
    """
    proj1 = pyproj.Proj("+init=EPSG:" + str(proj1))
    proj2 = pyproj.Proj("+init=EPSG:" + str(proj2))
    return pyproj.transform(proj1, proj2, x, y)