import numpy as np
import pandas as pd
import numpy.ma as ma


def WarrenClimatology(lonT, latT, monthT):
    """
    Get Warren1999 snow depth climatology
    Args:
        lonT (var): longitude
        latT (var): latitude
        monthT (var): month with the index starting at 0
        
    Returns:
        Hs (var): Snow depth (m)
        rho_s (var): Snow density (kg/m^3)
        
    """

    H_0 = [28.01, 30.28, 33.89, 36.8, 36.93, 36.59, 11.02, 4.64, 15.81, 22.66, 25.57, 26.67]
    a = [.127, .1056, .5486, .4046, .0214, .7021, .3008, .31, .2119, .3594, .1496, -0.1876]
    b = [-1.1833, -0.5908, -0.1996, -0.4005, -1.1795, -1.4819, -1.2591, -0.635, -1.0292, -1.3483, -1.4643, -1.4229]
    c = [-0.1164, -0.0263, 0.0280, 0.0256, -0.1076, -0.1195, -0.0811, -0.0655, -0.0868, -0.1063, -0.1409, -0.1413]
    d = [-0.0051, -0.0049, 0.0216, 0.0024, -0.0244, -0.0009, -0.0043, 0.0059, -0.0177, 0.0051, -0.0079, -0.0316]
    e = [0.0243, 0.0044, -0.0176, -0.0641, -0.0142, -0.0603, -0.0959, -0.0005, -0.0723, -0.0577, -0.0258, -0.0029]

    # Convert lat and lon into degrees of arc, +x axis along 0 degrees longitude and +y axis along 90E longitude
    x = (90.0 - latT)*np.cos(lonT * np.pi/180.0)  
    y = (90.0 - latT)*np.sin(lonT*np.pi/180.0) 

    Hs = H_0[monthT] + a[monthT]*x + b[monthT]*y + c[monthT]*x*y + (d[monthT]*x*x) + (e[monthT]*y*y)
    

    # Now get SWE, although this is not returned by the function

    H_0swe = [8.37, 9.43,10.74,11.67,11.8,12.48,4.01,1.08,3.84,6.24,7.54,8.0]
    aswe = [-0.027,0.0058,0.1618,0.0841,-0.0043,0.2084,0.097,0.0712,0.0393,0.1158,0.0567,-0.054]
    bswe = [-0.34,-0.1309,0.0276,-0.1328,-0.4284,-0.5739,-0.493,-0.145,-0.2107,-0.2803,-0.3201,-0.365]
    cswe = [-0.0319,0.0017,0.0213,0.0081,-0.038,-0.0468,-0.0333,-0.0155,-0.0182,-0.0215,-0.0284,-0.0362]
    dswe = [-0.0056,-0.0021,0.0076,-0.0003,-0.0071,-0.0023,-0.0026,0.0014,-0.0053,0.0015,-0.0032,-0.0112]
    eswe = [-0.0005,-0.0072,-0.0125,-0.0301,-0.0063,-0.0253,-0.0343,0,-0.019,-0.0176,-0.0129,-0.0035]


    swe = H_0swe[monthT] + aswe[monthT]*x + bswe[monthT]*y + cswe[monthT]*x*y + dswe[monthT]*x*x + eswe[monthT]*y*y

    # Density in kg/m^3
    rho_s = 1000.*(swe/Hs)  
    #print(ma.mean(rho_s))

    # Could mask out bad regions (i.e. land) here if desired.
    # Hsw[where(region_maskG<9.6)]=np.nan
    # Hsw[where(region_maskG==14)]=np.nan
    # Hsw[where(region_maskG>15.5)]=np.nan

    # Could mask out bad regions (i.e. land) here if desired.
    #rho_s[where(region_maskG<9.6)]=np.nan
    #rho_s[where(region_maskG==14)]=np.nan
    #rho_s[where(region_maskG>15.5)]=np.nan

    # Convert snow depth to meters
    Hs=Hs/100.

    return Hs, rho_s

def getWarrenData(dF, outSnowVar, outDensityVar='None'):
    """
    Assign Warren1999 snow dept/density climatology to dataframe
    Added 
    Args:
        dF (data frame): Pandas dataframe
        outSnowVar (string): name of Warren snow depth variable
        outDensityVar (string): name of Warren snow density variable
        
    Returns:
        dF (data frame): Pandas dataframe updated to include colocated Warren snow depth and density
        
    """
    
    # Generate empty lists
    snowDepthW99s=ma.masked_all(np.size(dF['freeboard'].values))
    if (outDensityVar!='None'):
        snowDensityW99s=ma.masked_all(np.size(dF['freeboard'].values))

    # Loop over all freeboard values (rows)
    for x in range(np.size(dF['freeboard'].values)):
        #print(x, dF['lon'].iloc[x], dF['lat'].iloc[x], dF['month'].iloc[x]-1)
        # SUbtract 1 from month as warren index in fucntion starts at 0
        snowDepthDayW99T, snowDensityW99T=WarrenClimatology(dF['lon'].iloc[x], dF['lat'].iloc[x], dF['datetime'].iloc[x].month-1)
        

        # Append values to list
        snowDepthW99s[x]=snowDepthDayW99T
        if (outDensityVar!='None'):
            snowDensityW99s[x]=snowDensityW99T

    # Assign list to dataframe as a series
    dF[outSnowVar] = pd.Series(snowDepthW99s, index=dF.index)
    if (outDensityVar!='None'):
        dF[outDensityVar] = pd.Series(snowDensityW99s, index=dF.index)
    

    return dF