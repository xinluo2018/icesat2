import pandas as pd
import numpy as np
import xarray as xr

def getDate(year, month, day):
    """ Get date string from year month and day"""

    return str(year)+'%02d' %month+'%02d' %day

def getNESOSIM(dF, fileSnow, outSnowVar='snow_depth_N', outDensityVar='snow_density_N'):
    """
    Load relevant NESOSIM snow data file and assign to freeboard values
    Args:
        dF (data frame): Pandas dataframe
        fileSnow (string): NESOSIM file path
        outSnowVar (string): Name of snow depth column
        outDensityVar (string): Name of snow density column
    Returns:
        dF (data frame): dataframe updated to include colocated NESOSIM (and dsitributed) snow data
    
    Versions:
        v2: Dropped basemap support and simplified
    """
    
    dateStrStart= getDate(dF.datetime.iloc[0].year, dF.datetime.iloc[0].month, dF.datetime.iloc[0].day)
    dateStrEnd= getDate(dF.datetime.iloc[-1].year, dF.datetime.iloc[-1].month, dF.datetime.iloc[-1].day)
    print('Check dates (should be within a day):', dateStrStart, dateStrEnd)
    
    dN = xr.open_dataset(fileSnow)

    # Get NESOSIM snow depth and density data for the date in the granule
    dNday = dN.sel(day=int(dateStrStart))
    
    # Get NESOSIM data for that data
    lonsN = np.array(dNday.longitude).flatten()
    latsN = np.array(dNday.latitude).flatten()

    # Get dates at start and end of freeboard file
    snowDepthNDay = np.array(dNday.snowDepth).flatten()
    snowDensityNDay = np.array(dNday.density).flatten()
    iceConcNDay = np.array(dNday.iceConc).flatten()
    
    # Remove data where snow depths less than 0 (masked).
    # Might need to chek if I need to apply any other masks here.
    mask=np.where((snowDepthNDay<0.01)|(snowDepthNDay>1)|(iceConcNDay<0.01)|np.isnan(snowDensityNDay))

    snowDepthNDay[mask]=np.nan
    snowDensityNDay[mask]=np.nan
    
    snowDepthNDay=snowDepthNDay
    snowDensityNDay=snowDensityNDay
    
    # I think it's better to declare array now so memory is allocated before the loop?
    snowDepthGISs=np.zeros((dF.shape[0]))
    snowDensityGISs=np.zeros((dF.shape[0]))
    
    # Should change this to an apply or lamda function 
    for x in range(dF.shape[0]):
        
        # Use nearest neighbor to find snow depth at IS2 point
        #snowDepthGISs[x] = griddata((xptsDay, yptsDay), snowDepthDay, (dF['xpts'].iloc[x], dF['ypts'].iloc[x]), method='nearest') 
        #snowDensityGISs[x] = griddata((xptsDay, yptsDay), densityDay, (dF['xpts'].iloc[x], dF['ypts'].iloc[x]), method='nearest')

        # Think this is the much faster way to find nearest neighbor!
        dist = np.sqrt((latsN-dF['lat'].iloc[x])**2+(lonsN-dF['lon'].iloc[x])**2)
        index_min = np.argmin(dist)
        snowDepthGISs[x]=snowDepthNDay[index_min]
        snowDensityGISs[x]=snowDensityNDay[index_min]
   
        
    dF[outSnowVar] = pd.Series(snowDepthGISs, index=dF.index)
    dF[outDensityVar] = pd.Series(snowDensityGISs, index=dF.index)

    return dF
    