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


def dt_baseline(dt_dict):
    """Find best baseline of time shift measurement

    In a time shift measurement on a set of noise correlation functions the
    baseline is undefined as the reference is arbitrary. However, if two
    stations recieve GPS signal the time difference will be constant. This
    means that shifts resulting from wrong timing will be random and those from
    correct times are constant. Here we estimate the most common time shift and
    assume that it characterises periods with correctly working clocks.
    """
    dt_bl = dt_dict['second_axis'][np.argmax(np.nansum(dt_dict['sim_mat'],axis=0))][0]

    return dt_bl


def dv_combine(dv_dict_l, method='average_sim_mat'):
    """Combine a set of change measuements
    
    A list of dv dictionaries from e.g. different channels of the same station
    combination is combined into a single dv dictionary.

    If method is 'average_sim_mat' the similarity matrices are averaged. The
    value along the crest of the averated matrix is used as new value.
    """
    assert type(dv_dict_l) == type([]), "dv_dict_l is not a list"
    
    if method == 'average_sim_mat':
        res_dv = deepcopy(dv_dict_l[0])
        for dv in dv_dict_l[1:]:
            res_dv['sim_mat'] += dv['sim_mat']
        res_dv['sim_mat'] /= len(dv_dict_l)
        res_dv['value'] = res_dv['second_axis'][np.argmax(res_dv['sim_mat'],axis=1)]
        res_dv['corr'] = np.max(res_dv['sim_mat'],axis=1)
    
    return res_dv

