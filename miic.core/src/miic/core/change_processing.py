# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:11:53 2016

@author: chris
"""
import numpy as np
from copy import deepcopy
from datetime import datetime

from miic.core.miic_utils import serial_date_from_datetime, convert_time




def time_select(dv_dict,starttime=None,endtime=None):
    """ Select time period from change data
    """
    
    dvc = deepcopy(dv_dict)
    
    time = convert_time(dvc['time'])
    # convert starttime and endtime input.
    # if they are None take the first or last values of the time vector
    if starttime == None:
        starttime = time[0]
    else:
        if not isinstance(starttime, datetime):
            starttime = convert_time([starttime])[0]
    if endtime == None:
        endtime = time[-1]
    else:
        if not isinstance(endtime, datetime):
            endtime = convert_time([endtime])[0]
                    
    # select period
    ind = np.nonzero((time >= starttime) * (time < endtime))[0]  # ind is
                                                #  a list(tuple) for dimensions
    dvc['value'] = dv_dict['value'][ind]
    dvc['time'] = dv_dict['time'][ind]
    return dvc


def estimate_trend(dv_dict):
    """Estimates a linear trend in the change measurements contained in 
    dv_dict.
    """
    
    # create time vector
    xi = []
    for t in convert_time(dv_dict['time']):
        xi.append(serial_date_from_datetime(t))
    xi = np.array(xi)
    
    # create matrix for inversion
    A = np.array([xi, np.ones(len(xi))])
        
    # invertion
    w = np.linalg.lstsq(A.T,-dv_dict['value'])[0] # obtaining the parameters
    
    y = w[0]*xi + w[1]
    std = np.std(y-(-dv_dict['value']))
    w = np.append(w, std)

    return w