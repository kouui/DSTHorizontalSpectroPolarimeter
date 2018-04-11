"""
author: kouui
date: 2018/04/07

Function library to plot DSTHSSP calibration result
"""

import matplotlib.pyplot as plt


def plotResidual(residual, ha, key, ylim_I=0.003):
    """
    residual : fitting residual array (1D, length of time series x3)
    ha       : hour angle array (1D, length of time series)
    key      : an array to identify input and output stokes vector (1D, length of time series x3)
    """
    
    nTime = ha.shape[0]
    dataDict = makeDataDict(key, nTime, residual, ha)
    
    fig, axs = plt.subplots(3,5, figsize=(20,10), dpi=100)
    FontSize = 20
    
    for i,output in enumerate(["Q","U","V"]):
        for j,incident in enumerate(["I","+Q","-Q","+U","-U"]):
            
            ax = axs[i,j]
            data = dataDict[incident+"2"+output]
            ax.plot(data["ha"], data["data"], '*', markersize=6)
            ax.axhline(y=0, linewidth=0.7)
            
            #--- title
            if i == 0:
                ax.set_title(incident, fontsize=FontSize)
            else:
                pass
            #--- ylabel
            if j == 0:
                ax.set_ylabel("$\Delta ({}/I)$".format(output), fontsize=FontSize)
            else:
                pass
                
            #--- xlabel
            if i == 2 and j == 0:
                ax.set_xlabel("Hour angle", fontsize=FontSize)
            else:
                pass
                
            #--- ylimit, fill_between
            if j == 0:
                ax.set_ylim(-ylim_I, ylim_I)
                #ax.fill_between(ha, -0.0003,0.0003, facecolor='gray', alpha=0.3, interpolate=True)
            else:
                ax.set_ylim(-0.1, 0.1)
                #ax.fill_between(ha, -0.001,0.001, facecolor='gray', alpha=0.3, interpolate=True)
    
    plt.show()
    
    
    return None

def plotFittingResult(result, ha, key):
    
    return None

def makeDataDict(key, nTime, dataArray, ha):
    
    assert nTime==int(len(key)/3), "bad length of key or nTime."
    assert nTime==int(len(dataArray)/3), "bad length of dataArray or nTime."
    
    dataDict = {
        "I2Q"  : {"key":0 ,"data":[],"ha":[]},
        "I2U"  : {"key":1 ,"data":[],"ha":[]},
        "I2V"  : {"key":2 ,"data":[],"ha":[]},
        "+Q2Q" : {"key":3 ,"data":[],"ha":[]},
        "+Q2U" : {"key":4 ,"data":[],"ha":[]},
        "+Q2V" : {"key":5 ,"data":[],"ha":[]},
        "-Q2Q" : {"key":6 ,"data":[],"ha":[]},
        "-Q2U" : {"key":7 ,"data":[],"ha":[]},
        "-Q2V" : {"key":8 ,"data":[],"ha":[]},
        "+U2Q" : {"key":9 ,"data":[],"ha":[]},
        "+U2U" : {"key":10,"data":[],"ha":[]},
        "+U2V" : {"key":11,"data":[],"ha":[]},
        "-U2Q" : {"key":12,"data":[],"ha":[]},
        "-U2U" : {"key":13,"data":[],"ha":[]},
        "-U2V" : {"key":14,"data":[],"ha":[]},
    }
    
    ha3 = ha.tolist() + ha.tolist() + ha.tolist()
    
    for k,d,h in zip(key, dataArray, ha3):
        
        for dictkey, value in dataDict.items():
            if value["key"] == k:
                value["data"].append(d)
                value["ha"].append(h)
                break
                
    return dataDict

def key2IncidentStokes(key):
    """
    Input:
        key : a single integer, 0,1,2,...,14
    
    to use this function, for example you have a key array called keyArray,
    then the incident stokes array is
    
    >>>incidentStokesArray = list(map(key2IncidentStokes,keyArray))
    
    """
    if key in (0,1,2):
        return "I"
    elif key in (3,4,5):
        return "+Q"
    elif key in (6,7,8):
        return "-Q"
    elif key in (9,10,11):
        return "+U"
    elif key in (12,13,14):
        return "-U"
    else:
        assert False, "bad value of key"
    
def key2FittingStokes(key):
    """
    Input:
        key : a single integer, 0,1,2,...,14
    
    to use this function, for example you have a key array called keyArray,
    then the fitting stokes array is
    
    >>>fittingStokesArray = list(map(key2FittingStokes,keyArray))
    
    """
    if key in (0,3,6,9,12):
        return "Q"
    elif key in (1,4,7,10,13):
        return "U"
    elif key in (2,5,8,11,14):
        return "V"
    else:
        assert False, "bad value of key"
        