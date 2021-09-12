from osgeo import gdal, osr
import numpy as np

def tiffread(ifile):
    """
    Reading tif-file to memory

    :param ifile: path+name of tif file
    :return: X, Y, Z, dx, dy and proj
    """
    
    file = gdal.Open(ifile, gdal.GA_ReadOnly)
    metaData = file.GetMetadata()
    projection = file.GetProjection()
    src = osr.SpatialReference()
    src.ImportFromWkt(projection)
    proj = src.ExportToWkt()
    
    Nx = file.RasterXSize
    Ny = file.RasterYSize
    
    trans = file.GetGeoTransform()
    
    dx = trans[1]
    dy = trans[5]
    
    Xp = np.arange(Nx)
    Yp = np.arange(Ny)
    
    (Xp, Yp) = np.meshgrid(Xp, Yp)
    
    X = trans[0] + (Xp + 0.5) * trans[1] + (Yp + 0.5) * trans[2]
    Y = trans[3] + (Xp + 0.5) * trans[4] + (Yp + 0.5) * trans[5]
    
    band = file.GetRasterBand(1)
    
    Z = band.ReadAsArray()
    
    dx = np.abs(dx)
    dy = np.abs(dy)
    
    return X, Y, Z, dx, dy, proj
