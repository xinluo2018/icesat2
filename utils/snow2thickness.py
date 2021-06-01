import numpy as np
import pandas as pd

def freeboard_to_thickness(freeboardT, snow_depthT, snow_densityT):
    """
    Hydrostatic equilibrium equation to calculate sea ice thickness 
    from freeboard and snow depth/density data
    Args:
        freeboardT (var): ice freeboard
        snow_depthT (var): snow depth
        snow_densityT (var): final snow density
    Returns:
        ice_thicknessT (var): ice thickness dereived using hydrostatic equilibrium
    """

    # Define density values
    rho_w=1024.
    rho_i=925.
    #rho_s=300.

    # set snow to freeboard where it's bigger than freeboard.
    snow_depthT[snow_depthT>freeboardT]=freeboardT[snow_depthT>freeboardT]

    ice_thicknessT = (rho_w/(rho_w-rho_i))*freeboardT - ((rho_w-snow_densityT)/(rho_w-rho_i))*snow_depthT

    return ice_thicknessT

def getSnowandConverttoThickness(dF, snowDepthVar='snowDepth', 
                                 snowDensityVar='snowDensity',
                                 outVar='iceThickness'):
    """ Grid using nearest neighbour the NESOSIM snow depths to the 
    high-res ICESat-1 freeboard locations
    """
    
    # Convert freeboard to thickness
    # Need to copy arrays or it will overwrite the pandas column!
    freeboardT=np.copy(dF['freeboard'].values)
    snowDepthT=np.copy(dF[snowDepthVar].values)
    snowDensityT=np.copy(dF[snowDensityVar].values)
    ice_thickness = freeboard_to_thickness(freeboardT, snowDepthT, snowDensityT)
    #print(ice_thickness)
    dF[outVar] = pd.Series(np.array(ice_thickness), index=dF.index)
   
    return dF
