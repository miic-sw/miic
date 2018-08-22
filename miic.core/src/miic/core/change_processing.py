# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:11:53 2016

@author: chris
"""
import numpy as np
from copy import deepcopy
from datetime import datetime
from scipy.optimize import fmin
import pdb

from miic.core.miic_utils import corr_mat_check, dv_check, zerotime, serial_date_from_datetime, convert_time
import miic.core.corr_mat_processing as cm
from miic.core.plot_fun import plot_dv
from miic.core.stream import corr_trace_to_obspy
import miic.core.dtype_mod as dm
from obspy.signal.util import next_pow_2
from obspy import Trace, UTCDateTime




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
    ind = ~np.isnan(dt_dict['value'])
    hh = np.histogram(dt_dict['value'][ind],bins=np.squeeze(dt_dict['second_axis']))
    dt_bl = hh[1][np.argmax(hh[0])]
    return dt_bl


def dv_combine(dv_list, method='average_sim_mat'):
    """Combine a set of change measuements
    
    A list of dv dictionaries from e.g. different channels of the same station
    combination is combined into a single dv dictionary.

    If method is 'average_sim_mat' the similarity matrices are averaged. The
    value along the crest of the averated matrix is used as new value.
    """
    assert type(dv_list) == type([]), "dv_list is not a list"
    
    if method == 'average_sim_mat':
        res_dv = deepcopy(dv_list[0])
        for dv in dv_list[1:]:
            res_dv['sim_mat'] += dv['sim_mat']
        res_dv['sim_mat'] /= len(dv_list)
        res_dv['value'] = res_dv['second_axis'][np.argmax(res_dv['sim_mat'],axis=1)]
        res_dv['corr'] = np.max(res_dv['sim_mat'],axis=1)
    
    return res_dv
    
    
def dv_combine_multi_ref(dv_list,max_shift=0.01,method='shift',offset=[]):
    """Combine a list of change measuments with different reference traces
    
    Combine a list of change dictionaries obtained from different references
    into a single one. The similarity matricies are simply averaged after they
    have been shifted to account for the offset between the different
    references. The offset between the references is estimated as the shift
    between the two correlation matrices with two possible methods: maximum of
    the summed product of shifted similarity matrices (`shift`) or the median
    of the difference between the estimated changes (`diff`). The first
    measurement in the list is not shifted. If the input list of measurements
    is longer than two, the individual shifts (with respect to the unshifted
    first measurment) is estimated from least squares inversion of the shifts
    between all measurements.
    
    :type dv_list: list dict
    :param dv_list: list of velociy change dictionaries to be combined
    :type max_shift: float
    :param max_shift: maximum shift to be tested beween different measuremants
    :type method: str
    :param method: method to estimate the offset between two measurements
    :type offset: list
    :param offset: pre-estimated velocity offset between dv measurements in
        units of setps in the dv_measurements. The first element in the dv_list
        will be set to zero offset.
    
    :type: dict
    :return: combined dv_dict
    """
    
    assert type(dv_list) == type([]), "dv_list is not a list"
    assert method in ['shift','diff'], "method has to be either 'shift' or "\
                "'diff'."
    if offset:
        assert len(offset) == len(dv_list), "if given offset must be of the "\
                "same length as dv_list"
    #stps should be at mostas large as the lagest second axis maller than max_shftp    
    
    if offset:
        offset = np.array(offset)
        offset -= offset[0]
    else:
        steps = max_shift/(dv_list[0]['second_axis'][1]-dv_list[0]['second_axis'][0])
        steps = int(np.floor(steps))     
        shift = []
        G = np.zeros(((len(dv_list)**2-len(dv_list))/2,len(dv_list)-1),dtype=float)
        cnt = 0    
        for ind1,dv1 in enumerate(dv_list):
            for ind2,dv2 in enumerate(dv_list):
                if ind2 > ind1:
                   shift.append(_dv_shift(dv1,dv2,steps,method))
                   if ind1 > 0:   # assume hat first refrence is not shifed
                       G[cnt,ind1-1] = 1
                   G[cnt,ind2-1] = -1
                   cnt += 1
        offset  = np.linalg.lstsq(G,shift)[0]
        offset = np.concatenate(([0],(np.round(offset)).astype(int)))
    cdv = deepcopy(dv_list[0])
    ns = int(len(cdv['second_axis']))
    for ind in range(1,len(dv_list)):
        cdv['sim_mat'][:,np.max([0,offset[ind]]):np.min([ns,ns+offset[ind]])] += \
            dv_list[ind]['sim_mat'][:,-np.min([0.,offset[ind]]):np.min([ns,ns-offset[ind]])]
    cdv['sim_mat'] /= len(dv_list)
    cdv['value'] = np.argmax(cdv['sim_mat'],axis=1)
    cdv['corr'] = np.max(cdv['sim_mat'],axis=1)
    cdv['value']= cdv['second_axis'][cdv['value']]

    return cdv


def _dv_shift(dv1,dv2,steps,method):
    if method == 'shift':
        c = []
        shi_range = np.arange(-steps,steps)
        for shi in shi_range:
            c.append(np.nansum(dv2['sim_mat'][:,steps+shi:-steps+shi]*dv1['sim_mat'][:,steps:-steps]))
        shift = shi_range[np.argmax(c)]
    elif method == 'diff':
        shift_val = np.median(dv2['value'] - dv1['value'])
        shiftn = np.argmin(np.abs(shift_val-dv1['second_axis']))
        shiftz = np.argmin(np.abs(-dv1['second_axis']))
        shift = shiftn - shiftz

    return shift
    


def create_stretch_mat(ref,stretch_range=0.03,stretch_steps=300,tw=None,sides='both'):
    assert isinstance(ref,Trace), 'ref must be an obspy.Trace object'
    stretch_values = dm.Spaced_values(-stretch_range,stretch_range/stretch_steps,\
                                    length=2*stretch_steps+1, \
                   s_type='stretch_values', name='stretch_values')
    
    # create time axis of original trace
    time_values = dm.Spaced_values(start=ref.stats.starttime-zerotime, \
                                   delta=ref.stats.delta, \
                                   length=ref.stats.npts,\
                                   s_type='lag_times',name='lag_times',unit=dm.Unit(['s']))
    times = time_values.get_data()
    # create spline object
    ref_tr_spline = UnivariateSpline(times, ref.data, s=0)
    # create time indices
    if type(tw) == type(None):
        twind = np.arange(time_values.length)
    else:
        twind = create_time_indices(ref.stats.starttime, ref.stats.sampling_rate, tw, sides)
    # times for interpolation
    str_times = times[twind]
    str_time_ser = dm.Series(str_times,s_type='lag_times',name='lag_times',unit=time_values.unit)
    # create space for stretching matrix
    mat = np.zeros([stretch_values.length,len(str_times)],dtype=np.float64)
    # populate stretching matrix
    for (k, this_fac) in enumerate(np.exp(stretch_values.get_data())):
        mat[k, :] = ref_tr_spline(str_times * this_fac)
    # put it in a matrix structure
    dmmat = dm.Matrix(data=mat,m_type='amplitude',first_axis=stretch_values,\
                     second_axis=str_time_ser,name='stretch_matrix')
    return dmmat


def create_time_indices(starttime, sampling_rate, tw, sides='both'):
    assert len(tw) == 2, 'time window tw must be a two item list: %s' % tw
    assert (tw[0]>=0) or (tw[1]>=0), 'time window must be non-negative: %s' % tw
    tind = (np.arange(tw[0]*sampling_rate,tw[1]*sampling_rate,1)).astype(int)
    zind = int(np.round((zerotime - starttime) * sampling_rate))
    if sides == 'left':
        twind = zind - tind[::-1]
    elif sides == 'right':
        twind = zind + tind
    elif sides == 'both':
        if tind[0] == 0:
            # zero lag time contanined
            twind = np.concatenate((zind - tind[::-1],zind + tind[1:]))
        else:
            twind = np.concatenate((zind - tind[::-1],zind + tind))
    else:
        raise ValueError, 'Value for sides not recognized: %s' % sides
    return twind


def measure_change(mat,ref,normalize=False):
    """Measure the similarity between rows in two matrices"""
    
    smat = np.dot(mat,ref.T)
    if normalize:
        m_sq = np.sum(mat ** 2, axis=1)
        r_sq = np.sum(ref ** 2, axis=1)
        norm = np.sqrt(np.dot(np.atleast_2d(m_sq).T,np.atleast_2d(r_sq)))
        smat /= norm
    
    bfind = np.argmax(smat,axis=1)
    bfval = smat[np.arange(len(bfind)),bfind]
    return smat, bfind, bfval


def change_time_select(dv, starttime=None, endtime=None, include_end=False):
    """ Select time period from a change measurement.

    Select a portion of a change measurement that falls into the
    time period `starttime`<= selected times <= `endtime` and return it.

    :type dv: dictionary
    :param dv: change dictionary as produced e.g. by
        `velocity_change`
    :type starttime: datetime.datetime object or time string
    :param starttime: beginning of the selected time period
    :type endtime: datetime.datetime object or time string
    :param endtime: end of the selected time period

    :rtype: dictionary
    :return: **dv**: velocity change dictionary restricted to the
        selected time period.
    """
    # check input
    if not isinstance(dv, dict):
        raise TypeError("dv needs to be change dictionary.")

    if dv_check(dv)['is_incomplete']:
        raise ValueError("Error: dv is not a valid change \
            dictionary.")
    
    sdv = deepcopy(dv)

    # convert time vector
    time = convert_time(sdv['time'])

    # convert starttime and endtime input.
    # if they are None take the first or last values of the time vector
    if type(starttime) == type(None):
        starttime = time[0]
    else:
        if not isinstance(starttime, datetime):
            starttime = convert_time([starttime])[0]
    if type(endtime) == type(None):
        endtime = time[-1]
    else:
        if not isinstance(endtime, datetime):
            endtime = convert_time([endtime])[0]

    # select period
    if include_end:
        ind = np.nonzero((time >= starttime) * (time <= endtime))[0]
    else:
        ind = np.nonzero((time >= starttime) * (time < endtime))[0]

    # trim the matrix
    sdv['value'] = dv['value'][ind]
    sdv['corr'] = dv['corr'][ind]
    if 'sim_mat' in dv.keys():
        sdv['sim_mat'] = dv['sim_mat'][ind, :]
    # adopt time vector
    sdv['time'] = dv['time'][ind]

    return sdv
   

def change_average(dv, n_av):
    """ Moving average of changes measurements.
    
    Calculate a moving average of the similarity matrix in a change dictionary
    and resetimate the change.
    """
    # check input
    if not isinstance(dv, dict):
        raise TypeError("dv needs to be change dictionary.")

    if dv_check(dv)['is_incomplete']:
        raise ValueError("Error: dv is not a valid change \
            dictionary.")
    assert type(n_av), 'n_av must be an integer'
    assert 1 == n_av % 2, 'n_av must be an odd number'
    assert 'sim_mat' in dv.keys(), 'sim_mat must be contained in dv'
    
    dv['sim_mat'][np.isnan(dv['sim_mat'])] = 0.
    adv = deepcopy(dv)
    for shift in np.arange(-np.floor(float(n_av)/2.),0).astype(int):
        adv['sim_mat'][:-shift,:] += dv['sim_mat'][shift:,:]
    for shift in np.arange(1,np.ceil(float(n_av)/2.)).astype(int):
        adv['sim_mat'][shift:,:] += dv['sim_mat'][:-shift,:]
    adv['sim_mat'] /= n_av
    bfind = np.argmax(adv['sim_mat'],axis=1)
    adv['corr'] = adv['sim_mat'][np.arange(len(bfind)),bfind]
    adv['value'] = adv['second_axis'][bfind]
    return adv
    
    
def change_interpolate(dv,search_range=None, fit_range=7):
    """ Interpolate the similarity matrix to artificially decrease the
    stretching increment
    
    fit_range: number of stretch values on each side of the maximum to be
        included in the polynomial fitting to obtian the maximum
    """
    if not isinstance(dv, dict):
        raise TypeError("dv needs to be change dictionary.")

    if dv_check(dv)['is_incomplete']:
        raise ValueError("Error: dv is not a valid change \
            dictionary.")
    idv = deepcopy(dv)
    if type(search_range) == type(None):
        bfind = np.argmax(dv['sim_mat'],axis=1)
    else:
        assert len(search_range) == 2, 'search_range must be a two element list'
        smin = np.argmin(np.abs(dv['second_axis']-search_range[0]))
        smax = np.argmin(np.abs(dv['second_axis']-search_range[1]))
        bfind = np.argmax(dv['sim_mat'][:,smin:smax],axis=1) + smin
    
    #import pdb
    #import matplotlib.pyplot as plt
    #pdb.set_trace()
    # interpolate over a larger range around the maximum
    for ii in range(len(dv['time'])):
        fitind = range(np.max((0,bfind[ii]-fit_range)),np.min((bfind[ii]+fit_range+1,len(dv['second_axis']))))
        if len(fitind) == 2.*fit_range + 1:  # otherwise the range for fitting is to short
            if not np.any(np.isnan(dv['sim_mat'][ii,fitind])):
                p = np.polyfit(dv['second_axis'][fitind],dv['sim_mat'][ii,fitind],deg=2)
                loc = -p[1]/(2.*p[0])
                if p[0]<1 and loc>dv['second_axis'][fitind[0]] and loc<dv['second_axis'][fitind[-1]]: # otherwise it is not a maximum or extrapolation
                    idv['value'][ii] = loc
                    idv['corr'][ii] = p[0]*idv['value'][ii]**2 + p[1]*idv['value'][ii] + p[2]
    #pdb.set_trace()
                
                
    # handle the maxima at edges
    """
    maind = bfind==(dv['sim_mat'].shape[1]-1)
    bfind[maind] -= 1
    miind = bfind==0
    bfind[miind] += 1    
    lint = np.arange(len(dv['time']))
    smval = dv['sim_mat'][lint,bfind-1]
    val = dv['sim_mat'][lint,bfind]
    gtval = dv['sim_mat'][lint,bfind+1]
    c = val
    a = val - (gtval+smval)/2.
    b = gtval - c - a
    iloc = -b/(2.*a)
    import pdb
    pdb.set_trace()
    ind = iloc>1 or iloc<-1
    idv['corr'] = a*(iloc**2) + b*iloc + c
    idv['value'] = dv['second_axis'][bfind] + (dv['second_axis'][bfind+1] - dv['second_axis'][bfind])*iloc
    # handle the maxima at edges
    idv['value'][maind] = dv['value'][maind]
    idv['value'][miind] = dv['value'][miind]
    idv['corr'][maind] = dv['corr'][maind]
    idv['corr'][miind] = dv['corr'][miind]
    """
    return idv


def velocity_change(cmat,ref=None,tw=None,sides='both',stretch_range=0.03,\
                    stretch_steps=100,return_simmat=True,stretch_mat=None):
    """ Measure the velocity change in a correlation matrix
    
    cmat: correlation matrix dictionary
    ref: obspy trace
    tw: list with start and end time of time window in seconds
    sides: 'both', 'left' or ''right'
    stretch_range: range of stretching to be tested [-stretch_range, +stretch_range]"""

    assert corr_mat_check(cmat), 'cmat must be a correlation matrix dictionary'
    if type(ref) == type(None):
        reftr = cm.corr_mat_extract_trace(cmat,method='norm_mean')
        ref = corr_trace_to_obspy(reftr)
    else:
        assert isinstance(ref,Trace), 'refrence must be an obspy.trace.'
        assert (ref.stats.sampling_rate == cmat['stats']['sampling_rate']),\
                'sampling rates of cmat and ref must match'
    if type(tw) != type(None):
        assert len(tw) == 2, 'tw must be a 2 item list: %s' % tw
    else:
        tw = [0.,UTCDateTime(convert_time([cmat['stats']['endtime']])[0])-zerotime]

    # create stretching matrix by stretching reference
    if type(stretch_mat) == type(None):
        stretch_mat = create_stretch_mat(ref,stretch_range=stretch_range,stretch_steps=stretch_steps,tw=tw, sides=sides)
    # select required part from correlation matrix
    twind = create_time_indices(UTCDateTime(convert_time([cmat['stats']['starttime']])[0]),
                                    cmat['stats']['sampling_rate'],tw=tw, sides=sides)
    tcmat = cmat['corr_data'][:,twind]
    # calculate similarity
    sim_mat, bfind, bfval = measure_change(tcmat,stretch_mat.get_data(),normalize=True)
    # translate index into measurement
    strvec = stretch_mat.first_axis.get_data()
    dt = strvec[bfind]
    # create dv structure
    dv = {'time': cmat['time'],
          'stats': cmat['stats'],
          'corr': np.squeeze(bfval),
          'value': -np.squeeze(dt),
          'second_axis': strvec,
          'value_type': np.array(['stretch']),
          'method': np.array(['single_ref'])}
    if return_simmat:
        dv.update({'sim_mat': np.fliplr(sim_mat).astype(np.float16)})
    return dv


# Functions for modeling velocity changes



def model_dv(dv,model_type,param=()):
    """Model velocity change measurements
    
    :type model: str
    :param model: type of model to be fitted to the data
    :type param: dict
    :param param: dictionary with parameters for the model to be fitted
    
    :rtype: dict
    :return: dictionary with model parameters and modelled velocity changes
    """
    
    if model_type == "seasonal":
        model_res = _model_seasonal(dv,param)
    elif model_type == "trend":
        model_res = _model_trend(dv,param)
    elif model_type == "const_exp_recovery":
        model_res = _model_const_exp_recovery(dv,param)
    elif model_type == 'trend_const_exp_recovery':
        model_res = _model_trend_const_exp_recovery(dv,param)
    
    model = {'type':model_type,'model_res':model_res,'param':param}
    model_dat = model_dv_forward(dv,model_type,model_res,param)
    model.update({'model_dat':model_dat})
    return model


def model_dv_forward(dv,model_type,model_res,param=()):
    if model_type == "seasonal":
        model = _model_seasonal_forward(dv,model_res)
    elif model_type == "trend":
        model = _model_trend_forward(dv,model_res)
    elif model_type == "const_exp_recovery":
        model = _model_const_exp_recovery_forward(dv,model_res,param[0])
    elif model_type == 'trend_const_exp_recovery':
        xdata = [(float(t.toordinal())*86400+float(t.hour)*3600+float(t.minute)*60\
                 +float(t.second)) for t in convert_time(dv['time'])]
        model = _model_trend_const_exp_recovery_forward(dv,param.append(xdata))
    return model        


#def _model_trend_const_exp_recovery_forward(dv,param)
    
    


def _model_seasonal_forward(dv,x):
    pfac = 1.991021277657232e-07  # 2*pi/length_of_year_in_second
    xdata = [(float(t.toordinal())*86400+float(t.hour)*3600+float(t.minute)*60\
             +float(t.second)) for t in convert_time(dv['time'])]
    xdata = np.array(xdata)
    model = x[0] * np.cos(xdata*pfac) + x[1] * np.sin(xdata*pfac) + x[2]
    # construct a dv dictionary to return the model
    ret_model = deepcopy(dv)
    ret_model['value'] = model
    # find correlation values along model
    tmp = np.tile(dv['second_axis'],(dv['sim_mat'].shape[0],1))
    tmp -= np.tile(np.atleast_2d(model).T,(1,dv['sim_mat'].shape[1]))
    ind = np.argmin(np.abs(tmp),axis=1)
    ret_model['corr'] = [dv['sim_mat'][ii,ind[ii]] \
                         for ii in range(dv['sim_mat'].shape[0])]       
    return ret_model
        

def _model_seasonal(dv,param):
    """Model a seasonal cycle
    
    """
    if not param:
        param = (0., 0., 0.)
    xdata = [(float(t.toordinal())*86400+float(t.hour)*3600+float(t.minute)*60\
             +float(t.second)) for t in convert_time(dv['time'])]
    xdata = np.array(xdata)
    model_res = fmin(_sine_fit, [param[0], param[1], param[2]], \
                     args=(xdata,dv), xtol=0.00001)
 
    return model_res
    
    
def _sine_fit(x,xdata,dv):
    pfac = 1.991021277657232e-07  # 2*pi/length_of_year_in_second
    model = x[0] * np.cos(xdata*pfac) + x[1] * np.sin(xdata*pfac) + x[2]
    scor = _calc_misfit(dv,model)
    return np.nansum(np.max(dv['sim_mat'],axis=1)) - scor 


def _model_trend(dv,param):
    """Model a linead trend
    """
    if not param:
        print "param not given"
        param = (0., 0.,1)
    xdata = [(float(t.toordinal())*86400+float(t.hour)*3600+float(t.minute)*60\
             +float(t.second)) for t in convert_time(dv['time'])] 
    xdata = np.array(xdata)
    #model_res = minimize(_trend_fit,(param[0], param[1]),args=(xdata,dv),tol=1e-20,bounds=((-0.03,0.03),(-2,2)))
    #model_res = fmin(_trend_fit, [param[0], param[1]], \
    #                 args=(xdata,dv),xtol=1e-14)
    import pdb
    pdb.set_trace()                 
                     
    # alternative grid serch
    c = dv['sim_mat'].ravel()
    lsa = len(dv['second_axis'])
    lt = len(dv['time'])
    inc = param[2]
    slval = np.arange(-lsa+1,lsa-1)
    csum = np.zeros((len(slval),lsa))
    x = np.zeros((2))
    for sl in slval[::inc]:
        ind = np.round(np.linspace(0,(lt-1)*lsa+sl,lt)).astype(int)
        for st in (range(np.max((-sl,0)),np.min((lsa,lsa-sl))))[::inc]:
            csum[int(sl+lsa-1),int(st)] = np.nansum(c[ind+st])
            
    st = np.argmax(np.max(csum,axis=0))
    sl = slval[np.argmax(csum[:,st])]
    t = convert_time(dv['time'])
    x[0] = dv['second_axis'][st]
    x[1] = sl * (dv['second_axis'][-1] -dv['second_axis'][0])/(lsa-1) /\
            (t[-1]-t[0]).days
    model_res = x
    
    return model_res
    

def _trend_fit(x,xdata,dv):
    model = x[0] + 1e-10*x[1]*(xdata-xdata[0])
    scor = _calc_misfit(dv,model)
    return np.nansum(np.max(dv['sim_mat'],axis=1)) - scor


def _model_trend_forward(dv,x):
    xdata = [(float(t.toordinal())*86400+float(t.hour)*3600+float(t.minute)*60\
             +float(t.second)) for t in convert_time(dv['time'])]
    xdata = np.array(xdata)
    model = x[0] + 1e-10*x[1]*(xdata-xdata[0])
    # construct a dv dictionary to return the model
    ret_model = deepcopy(dv)
    ret_model['value'] = model
    # find correlation values along model
    tmp = np.tile(dv['second_axis'],(dv['sim_mat'].shape[0],1))
    tmp -= np.tile(np.atleast_2d(model).T,(1,dv['sim_mat'].shape[1]))
    ind = np.argmin(np.abs(tmp),axis=1)
    ret_model['corr'] = [dv['sim_mat'][ii,ind[ii]] \
                         for ii in range(dv['sim_mat'].shape[0])]
    return ret_model


def _model_const_exp_recovery(dv,param):
    """Model with exponentially recovering perturbations
    
    :type param[0]: obspy.stream
    :param param[0]: timeseries with excitation
    :type param[1]: float
    :param param[1]: amplitude scaling of the excitation
    :type param[2]: float
    :param param[2]: time constant of recovery in seconds
    """
    xdata = [(float(t.toordinal())*86400+float(t.hour)*3600+float(t.minute)*60\
             +float(t.second)) for t in convert_time(dv['time'])] 
    xdata = np.array(xdata)
    exc = param[0]
    
    
    
    #model_res = minimize(_const_exp_recovery_fit, [1, 1], \
    #                 args=(xdata,dv,exc,param[1],param[2]),bounds=())
   
    model_res = fmin(_const_exp_recovery_fit, [1,1], \
                     args=(xdata,dv,exc,param[1], param[2]),xtol=0.001,maxiter=100)
    model_res[0] *= param[1]
    model_res[1] *= param[2]
    
    return model_res

def _const_exp_recovery_fit(x,xdata,dv,exc,amp_scale,time_scale):
    print x
    a = x[0] * amp_scale
    tau = x[1] * time_scale
    t = exc[0].stats['starttime']
    st = float(t.toordinal())*86400+float(t.hour)*3600+float(t.minute)*60\
             +float(t.second)
    dt = exc[0].stats['delta']
    tv = [st+dt*ind for ind in range(exc[0].stats['npts'])]
    tv = np.array(tv)
    rec = a * np.exp(-(tv-tv[0])/tau)
    model = np.convolve(exc[0].data,rec,"full")
    model = model[:len(exc[0].data)]
    model = np.interp(xdata,tv,model)
    scor = _calc_misfit(dv,model)
    return np.nansum(np.max(dv['sim_mat'],axis=1)) - scor
    
    
def _model_const_exp_recovery_forward(dv,x,exc):  

    xdata = [(float(t.toordinal())*86400+float(t.hour)*3600+float(t.minute)*60\
             +float(t.second)) for t in convert_time(dv['time'])]
    xdata = np.array(xdata)
    t = exc[0].stats['starttime']
    st = float(t.toordinal())*86400+float(t.hour)*3600+float(t.minute)*60\
             +float(t.second)
    dt = exc[0].stats['delta']
    tv = np.arange(st,st+dt*exc[0].stats['npts'],dt)
    tv = [st+dt*ind for ind in range(exc[0].stats['npts'])]
    tv = np.array(tv)
    rec = x[0] * np.exp(-(tv-tv[0])/x[1])
    model = np.convolve(exc[0].data,rec,"full")
    model = model[:len(exc[0].data)]
    model = np.interp(xdata,tv,model)
    # construct a dv dictionary to return the model
    ret_model = deepcopy(dv)
    ret_model['value'] = model
    # find correlation values along model
    tmp = np.tile(dv['second_axis'],(dv['sim_mat'].shape[0],1))
    tmp -= np.tile(np.atleast_2d(model).T,(1,dv['sim_mat'].shape[1]))
    ind = np.argmin(np.abs(tmp),axis=1)
    ret_model['corr'] = [dv['sim_mat'][ii,ind[ii]] \
                         for ii in range(dv['sim_mat'].shape[0])]
    return ret_model

    
    
def _calc_misfit(dv,model):
    """Calculate the sum of the correlation values along the model curve
    """
    tmp = np.tile(dv['second_axis'],(dv['sim_mat'].shape[0],1))
    tmp -= np.tile(np.atleast_2d(model).T,(1,dv['sim_mat'].shape[1]))
    ind = np.argmin(np.abs(tmp),axis=1)
    scor = np.nansum([dv['sim_mat'][ii,ind[ii]] \
                      for ii in range(dv['sim_mat'].shape[0])])
    return scor    
    
    
    
    
def _modelvec_from_dict(model_par):
    """ create a vector of parameters to be inverted from the model dictionary
    """
    x0 = []
    for model_type in sorted(model_par['type']):
        for param in sorted(model_par[model_type]):
            if param in model_par[model_type]['to_vari']:
                x0.append(model_par[model_type][param])
    return x0
    
def _modeldict_from_vec(start_model,x):
    model_par = deepcopy(start_model)
    xcnt = 0
    for model_type in sorted(start_model['type']):
        for param in sorted(start_model[model_type]['to_vari']):
            model_par[model_type][param] = x[xcnt]
            xcnt += 1
    return model_par
    
    
    
def _misfit_int_corr(x, dv, dv_time, exc_time, exc, start_model):
    """Calculate  misfit between modeled velocity and measured streatching.
    
    Stretching is -1 * velocity change
    """
    # construct model_par
    model_par = _modeldict_from_vec(start_model,x)
    model = _forward(dv_time, exc_time, exc, model_par)
    """
    scale = dv['second_axis'][1] - dv['second_axis'][0]
    offset = dv['second_axis'][0]
    mod_ind = (np.round((model-offset) / scale).astype(int))
    ser_ind = mod_ind + np.arange(0,dv['sim_mat'].shape[0]*
                                  dv['sim_mat'].shape[1],
                                  dv['sim_mat'].shape[1])
    off_ind = np.stack((mod_ind>0,mod_ind<len(dv['second_axis'])),axis=0)
    off_ind = np.all(off_ind, axis=0)
    scorr = np.nansum(dv['sim_mat'].ravel()[ser_ind[off_ind]])
    scorr -= (len(off_ind) - np.sum(off_ind))   # values off sclae get -1
    mf = -scorr/len(model)  # -1 would be a perfekt fit
    print x, mf
    """
    corr = _model_correlation(dv, model)
    mf = -np.nanmean(corr)
    print mf,
    return mf
    
def _model_correlation(dv, model):
    """
    return the correlation value for velocity given by model
    """
    scale = dv['second_axis'][1] - dv['second_axis'][0]
    offset = dv['second_axis'][0]
    mod_ind = (np.round((model-offset) / scale).astype(int))
    ser_ind = mod_ind + np.arange(0,dv['sim_mat'].shape[0]*
                                  dv['sim_mat'].shape[1],
                                  dv['sim_mat'].shape[1])
    off_ind = np.stack((mod_ind>0,mod_ind<len(dv['second_axis'])),axis=0)
    off_ind = np.all(off_ind, axis=0)
    ser_ind[~off_ind] = 0
    corr = dv['sim_mat'].ravel()[ser_ind]
    corr[~off_ind] = -1
    return corr
    

def _misfit_L2(x, dv, dv_time, exc_time, exc, start_model):
    # construct model_par
    model_par = _modeldict_from_vec(start_model,x)
    #print model_par
    model = _forward(dv_time, exc_time, exc, model_par)
    mf = np.nanmean((model - dv['value'])**2)
    print mf,
    return mf
    
def _print_model_par(model_par):
    for type in sorted(model_par['type']):
        print '%s:' % type
        for par in model_par[type].keys():
            if par not in ['to_vari','units']:
                print '\t%s = %s %s' % (par,str(model_par[type][par]),
                                        model_par[type]['units'][par])
        print '\tfitted parameters: %s\n' % str(model_par[type]['to_vari'])

def substract_model(dv):
    """ Substract the modelled change from the data.
    """
    assert 'model_value' in dv, "You need to calculate a model first"
    if 'sim_mat' in dv:
        tmp_sim_mat = np.zeros_like(dv['sim_mat'])
        la = len(dv['second_axis'])
        cent = np.argmin(np.abs(dv['second_axis']))
        for ind in range(len(dv['time'])):
            shift = np.argmin(np.abs(dv['second_axis']-dv['model_value'][ind])) - cent
            tmp_sim_mat[ind,max((0,-shift)):min((la,la-shift))] = \
            dv['sim_mat'][ind,max((0,shift)):min((la,la+shift))]
        dv['sim_mat'] = tmp_sim_mat
    dv['value'] -= dv['model_value']
    dv['model_value'] *= 0.

    
class Change:
    def __init__(self):
        print 'init'
        self.dv = {}

    def copy(self):
        return deepcopy(self)
    
    def set_dv(self, dv):
        self.dv = dv

    def set_excitation(self, excitation):
        t = excitation.stats['starttime']
        st = float(t.toordinal())*86400+float(t.hour)*3600+ \
                    float(t.minute)*60 + float(t.second)
        dt = excitation.stats['delta']
        exc_time = np.arange(st,st+dt*excitation.stats['npts'],dt)
        exc_time = [st+dt*ind for ind in range(excitation.stats['npts'])]
        # set time of first measurement to zero
        exc_time = np.array(exc_time)# - dv_time[0]
        self.excitation = excitation
        self.excitation_time = exc_time
            
    def model_forward(self, misfit_function=_misfit_int_corr):
        self.dv = model_forward(self.dv, self.model_par, self.excitation,
                                misfit_function)
    
    def substract_model(self):
        substract_model(self.dv)
    
    def set_start_model(self, type):
        """
        'type' should be a list of types of changes the shall be included in
        the model possibilities are 'offset', 'trend', 'cyclic', 'const_exp'
        """
        self.start_model = {'type':type}
        for this_type in type:
            if this_type == 'offset':
                self.start_model.update({'offset':{'offset':0.,
                                                   'to_vari':['offset'],
                                                   'units':{'offset':'1'}}})
            elif this_type == 'trend':
                self.start_model.update({'trend':{'trend':0.,
                                                  'to_vari':['trend'],
                                                  'units':{'trend':'1/a'}}})
            elif this_type == 'cyclic':
                self.start_model.update({'cyclic':{'sine':0.,'cosine':0.,
                                                   'period':365,
                                                   'to_vari':['sine',
                                                              'cosine'],
                                                   'units':{'cosine':'1',
                                                            'sine':'1',
                                                            'period':'d'}}})
            elif this_type == 'const_exp':
                self.start_model.update({'const_exp':{'sensitivity':1.,
                                                      'tau':1.,
                                                      'to_vari':['sensitivity',
                                                                 'tau'],
                                                      'units':{'sensitivity':'1/input',
                                                               'tau':'a'}}})
            elif this_type == 'var_exp':
                self.start_model.update({'var_exp':{'sensitivity':1.,
                                                      'tau_scale':1.,
                                                      'to_vari':['sensitivity',
                                                                 'tau_scale'],
                                                      'units':{'sensitivity':'1/input',
                                                               'tau_scale':'a/input'}}})
            elif this_type == 'const_log':
                self.start_model.update({'const_log':{'sensitivity':1.,
                                                      'tau_min':-1,
                                                      'tau_max':2.,
                                                      'to_vari':['sensitivity',
                                                                 'tau_max'],
                                                      'units':{'sensitivity':'1/input',
                                                               'tau_min':'log10(d)',
                                                               'tau_max':'log10(d)'}}})
            elif this_type == 'var_log':
                self.start_model.update({'var_log':{'sensitivity':1.,
                                                      'tau_min':-1,
                                                      'tau_max_scale':2.,
                                                      'to_vari':['sensitivity',
                                                                 'tau_max_scale'],
                                                      'units':{'sensitivity':'1/input',
                                                               'tau_min':'log10(d)',
                                                               'tau_max_scale':'log10(d)/input'}}})
            elif this_type == 'state_model':
                self.start_model.update({'state_model':{'sensitivity':1.,
                                                        'tau_min':-2,
                                                        'tau_max':3,
                                                        'Ntau':100,
                                                        'to_vari':['sensitivity','tau_max'],
                                                        'units':{'sensitivity':'1/input',
                                                                 'tau_min':'log10(d)',
                                                                 'tau_max':'log10(d)',
                                                                 'Ntau':'1'}}})
            elif this_type == 'flat_state_model':
                self.start_model.update({'flat_state_model':{'sensitivity':1.,
                                                        'tau_min':-2,
                                                        'tau_max':3,
                                                        'Ntau':100,
                                                        'to_vari':['sensitivity','tau_max'],
                                                        'units':{'sensitivity':'1/input',
                                                                 'tau_min':'log10(d)',
                                                                 'tau_max':'log10(d)',
                                                                 'Ntau':'1'}}})
            elif this_type == 'strain_rate_model':
                self.start_model.update({'strain_rate_model':{'sensitivity':1.,
                                                        'tau_min':-2,
                                                        'tau_max':3,
                                                        'Ntau':100,
                                                        'to_vari':['sensitivity','tau_max'],
                                                        'units':{'sensitivity':'1/input',
                                                                 'tau_min':'log10(d)',
                                                                 'tau_max':'log10(d)',
                                                                 'Ntau':'1'}}})
                                                                 
    def fit_model(self, misfit_function=_misfit_int_corr):
        """
        'excitation' is supposed to be a obspy.trace
        """
    
        # generate vector for starting model
        x0 = _modelvec_from_dict(self.start_model)

            
        # create time vector for dv measurements
        dv_time = [(float(t.toordinal())*86400+float(t.hour)*3600+float(t.minute)*60\
                 +float(t.second)) for t in convert_time(self.dv['time'])]
        dv_time = np.array(dv_time)
        
        # create time vector for excitation
        mod_with_exc = ['const_exp','var_exp','const_log','var_log',
                        'state_model','flat_state_model','strain_rate_model']
        if any(x in self.start_model['type'] for x in mod_with_exc):
            assert hasattr(self,'excitation'), 'Attach an excitation first'
            """
            t = excitation.stats['starttime']
            st = float(t.toordinal())*86400+float(t.hour)*3600+ \
                    float(t.minute)*60 + float(t.second)
            dt = excitation.stats['delta']
            exc_time = np.arange(st,st+dt*excitation.stats['npts'],dt)
            exc_time = [st+dt*ind for ind in range(excitation.stats['npts'])]
            # set time of first measurement to zero
            """
            exc_time = self.excitation_time - dv_time[0]
            exc = self.excitation.data
        else:
            exc_time = []
            exc = []
        # set time of first measurement to zero
        dv_time -= dv_time[0]
            
        ret = fmin(misfit_function, x0, args=(self.dv, dv_time,
                                             exc_time, exc, self.start_model),
                                             xtol=1e-8,maxiter=1000,
                                             full_output=True)
        x = ret[0]
        mf = ret[1]                          
        self.model_par = _modeldict_from_vec(self.start_model,x)
        self.dv['model_value'] = _forward(dv_time, exc_time, exc,
                                          self.model_par)
        self.dv['model_corr'] = _model_correlation(self.dv,
                                                   self.dv['model_value'])
        self.dv['misfit_function'] = '%s' % misfit_function
        self.dv['model_misfit'] = mf
    
    def plot(self):
        plot_dv(self.dv)
    
    def print_start_model(self):
        if not hasattr(self,'start_model'):
            print 'No start_model defined.'
            return
        _print_model_par(self.start_model)
            
    def print_model(self):
        if not hasattr(self,'model_par'):
            print 'No model defined.'
            return
        _print_model_par(self.model_par)
        print 'Model misfit: %e\n' % self.dv['model_misfit'] 
    

def model_forward(dv,model_par,excitation=[],misfit_function=_misfit_int_corr):
    # create time vector for dv measurements
    #pdb.set_trace()
    tdv = deepcopy(dv)
    dv_time = [(float(t.toordinal())*86400+float(t.hour)*3600+float(t.minute)*60\
              +float(t.second)) for t in convert_time(tdv['time'])]
    dv_time = np.array(dv_time)
    
    if excitation:
        t = excitation.stats['starttime']
        st = float(t.toordinal())*86400+float(t.hour)*3600+ \
                   float(t.minute)*60 + float(t.second)
        dt = excitation.stats['delta']
        exc_time = np.arange(st,st+dt*excitation.stats['npts'],dt)
        exc_time = [st+dt*ind for ind in range(excitation.stats['npts'])]
        # set time of first measurement to zero
        exc_time = np.array(exc_time) - dv_time[0]
        exc = excitation.data  
    else:
        exc = []
        exc_time = []
    dv_time -= dv_time[0]
    tdv['model_value'] = _forward(dv_time, exc_time, exc, model_par)
    tdv['model_corr'] = _model_correlation(tdv, tdv['model_value'])
    tdv['misfit_function'] = '%s' % misfit_function
    mf = (_modelvec_from_dict(model_par), tdv, dv_time, exc_time,
          exc,model_par)
    tdv['model_misfit'] = mf
    return tdv


def _forward(dv_time, exc_time, exc, model_par):
    """ Model stretching due to changes using different components
    
    Assume all components are additive.
    """
    mdv = np.zeros(len(dv_time),dtype=float)
    if 'offset' in model_par['type']:
        mdv += model_par['offset']['offset']
    if 'trend' in model_par['type']:
        mdv += model_par['trend']['trend'] * dv_time/31536000
    if 'cyclic' in model_par['type']:
        phase = dv_time*2*np.pi/(model_par['cyclic']['period']*86400)
        mdv += model_par['cyclic']['sine']*np.sin(phase) + \
               model_par['cyclic']['cosine']*np.cos(phase)
    if 'const_exp' in model_par['type']:
        mdv += _forward_const_exp(dv_time, exc_time, exc,
                                  model_par['const_exp']['sensitivity'],
                                  model_par['const_exp']['tau'])
    if 'var_exp' in model_par['type']:
        mdv += _forward_var_exp(dv_time, exc_time, exc,
                                model_par['var_exp']['sensitivity'],
                                model_par['var_exp']['tau_scale'])
    if 'const_log' in model_par['type']:
        mdv += _forward_const_log(dv_time, exc_time, exc,
                                  model_par['const_log']['sensitivity'],
                                  model_par['const_log']['tau_min'],
                                  model_par['const_log']['tau_max'])
    if 'var_log' in model_par['type']:
        mdv += _forward_var_log(dv_time, exc_time, exc,
                                model_par['var_log']['sensitivity'],
                                model_par['var_log']['tau_min'],
                                model_par['var_log']['tau_max_scale'])
    if 'state_model' in model_par['type']:
        #pdb.set_trace()
        mdv += _forward_state_model_fd(dv_time, exc_time, exc,
                                       model_par['state_model']['exc_scale'],
                                       model_par['state_model']['sensitivity'],
                                       np.logspace(model_par['state_model']['tau_min'],
                                                   model_par['state_model']['tau_max'],
                                                   model_par['state_model']['Ntau']),
                                       model_par['state_model']['p0'])
    if 'flat_state_model' in model_par['type']:
        #pdb.set_trace()
        mdv += _forward_flat_state_model_fd(dv_time, exc_time, exc,
                                       model_par['flat_state_model']['exc_scale'],
                                       model_par['flat_state_model']['sensitivity'],
                                       np.logspace(model_par['flat_state_model']['tau_min'],
                                                   model_par['flat_state_model']['tau_max'],
                                                   model_par['flat_state_model']['Ntau']),
                                       model_par['flat_state_model']['p0'])
    if 'strain_rate_model' in model_par['type']:
        #pdb.set_trace()        
        mdv += _forward_rate_model(dv_time, exc_time, exc,
                                       model_par['strain_rate_model']['exc_scale'],
                                       model_par['strain_rate_model']['sensitivity'],
                                       np.logspace(model_par['strain_rate_model']['tau_min'],
                                                   model_par['strain_rate_model']['tau_max'],
                                                   model_par['strain_rate_model']['Ntau']),
                                       model_par['strain_rate_model']['p0'])
    return mdv


def _forward_state_model(dv_time, exc_time, exc, sensitivity, taus, p0):
    """Roel's and Jack's analytic solution to the state model.
    
    units: exc_time in seconds
    tau in seconds
    """
    # int 2 - alpha t
    #pdb.set_trace()
    tim = exc_time - exc_time[0]
    t_scale = exc_time[0]
    dt = np.diff(exc_time)/t_scale
    dt = np.concatenate((dt,np.array([dt[-1]])))
    A = np.cumsum(exc*dt)
    eA = np.exp(A)
    
    p = np.zeros((len(tim),len(taus)))
    for ind,tau in enumerate(taus):
        if p0[ind] < 0:
            p0[ind] = exc[0]/(1./tau/t_scale + exc[0])
        emFt = np.exp(tim/tau) * eA
        p[:,ind] = 1 - 1./emFt * (1 - p0[ind] + 1/tau/t_scale * np.cumsum(emFt*dt))

"""
def _forward_flat_state_model_fd(dv_time, exc_time, exc, exc_scale, sensitivity, taus, p0):
    texc = deepcopy(exc)*exc_scale
    p = np.zeros((len(exc_time),len(taus)))
    alp = 1./taus
    for ind,al in enumerate(alp):
        if p0[ind] < 0:
            p0[ind] = texc[0]/(1+texc[0])
    p[0,:] = p0
    dt = np.diff(exc_time)
    pequi = texc/(1+texc)
    for ind,tdt in enumerate(dt):
        p[ind+1,:] = pequi[ind] + (p[ind,:] - pequi[ind])*np.exp(-alp*(1.+texc[ind])*tdt)
    #p /= np.tile(np.atleast_2d(alp),(len(exc),1))    
    import pdb
    import matplotlib.pyplot as plt
    #p *= np.tile(alp,())
    #plt.imshow(np.log(p),aspect='auto')
    #plt.colorbar()
    #plt.show()
    #pdb.set_trace()
    #p /= np.tile(alp,(len(texc),1))
    model = np.sum(p,axis=1)
    model = np.interp(dv_time-dv_time[0],exc_time,model)
    model -= model[0]
    model /= np.max(np.abs(model))
    model *= sensitivity
    #plt.imshow(p,aspect='auto')
    #plt.show()
    return model
"""


def _forward_flat_state_model_fd(dv_time, exc_time, exc, exc_scale, sensitivity, taus, p0):
    texc = deepcopy(exc)*exc_scale
    p = np.zeros((len(exc_time),len(taus)))
    alp = 1./taus
    for ind,al in enumerate(alp):
        if p0[ind] < 0:
            p0[ind] = texc[0]/(1+texc[0])
    p[0,:] = p0
    dt = np.diff(exc_time)
    pequi = texc/(1+texc)
    for ind,tdt in enumerate(dt):
        p[ind+1,:] = pequi[ind] + (p[ind,:] - pequi[ind])*np.exp(-alp*tdt)
        
    p /= np.tile(alp**1.,(len(texc),1))

    #import pdb
    #import matplotlib.pyplot as plt
    #plt.imshow(np.log(p),aspect='auto')
    #plt.colorbar()
    #plt.colorbar()
    #plt.show()
    #pdb.set_trace()
    model = np.sum(p,axis=1)
    model = np.interp(dv_time-dv_time[0],exc_time,model)
    model -= model[0]
    model /= np.max(np.abs(model))
    model *= sensitivity
    #plt.imshow(p,aspect='auto')
    #plt.show()
    return model
    
    
def _forward_rate_model(dv_time, exc_time, exc, exc_scale, sensitivity, taus, p0):
    texc = deepcopy(exc)*exc_scale
    p = np.zeros((len(exc_time),len(taus)))
    alp = 1./taus
    for ind,al in enumerate(alp):
        if p0[ind] < 0:
            p0[ind] = texc[0]/(1+texc[0])
    p[0,:] = p0
    dt = np.diff(exc_time)
    pequi = texc/(1+texc)
    import pdb
    pdb.set_trace()
    for ind,tdt in enumerate(dt):
        p[ind+1,:] = p[ind,:] - p[ind,:]*(1.-np.exp(-texc[ind]*alp*tdt)) + (p[ind,:] - p0)*(1.-np.exp(-alp*tdt))
    import matplotlib.pyplot as plt
    plt.imshow(np.log(p))
    plt.show()    
    #p /= np.tile(alp**0.7,(len(texc),1))

    #import pdb
    #import matplotlib.pyplot as plt
    #plt.imshow(np.log(p),aspect='auto')
    #plt.colorbar()
    #plt.colorbar()
    #plt.show()
    #pdb.set_trace()
    model = np.sum(p,axis=1)
    model = np.interp(dv_time-dv_time[0],exc_time,model)
    model -= model[0]
    model /= np.max(np.abs(model))
    model *= sensitivity
    #plt.imshow(p,aspect='auto')
    #plt.show()
    return model    


def _forward_state_model_fd(dv_time, exc_time, exc, exc_scale, sensitivity, taus, p0):
    """numerical fd solution to the state model.
    
    units: exc_time in seconds
    tau in seconds
    """
    # int 2 - alpha t
    #pdb.set_trace()
    texc = deepcopy(exc)*exc_scale
    p = np.zeros((len(exc_time),len(taus)))
    alp = 1./taus
    for ind,al in enumerate(alp):
        if p0[ind] < 0:
            p0[ind] = texc[0]/(al+texc[0])
    p[0,:] = p0
    dt = np.diff(exc_time)
    for ind,tdt in enumerate(dt):
        pequi = texc[ind]/(alp+texc[ind])
        p[ind+1,:] = pequi + (p[ind,:] - pequi)*np.exp(-(alp+texc[ind])*tdt)
    model = np.sum(p,axis=1)
    model = np.interp(dv_time-dv_time[0],exc_time,model)
    model -= model[0]
    model /= np.abs(np.max(model))
    model *= sensitivity
    #plt.imshow(p,aspect='auto')
    #plt.show()
    return model
    

def _forward_const_exp(dv_time, exc_time, exc, sensitivity, tau):
    """Expoential recovery with a constant time scales.
    
    Exponential with constant time constant is convolved with excitation.
    """
    rec = sensitivity * np.exp(-exc_time/(tau*31536000))
    model = np.convolve(exc,rec,"full")
    model = model[:len(exc)]
    model = np.interp(dv_time,exc_time,model)
    return model


def _forward_var_exp(dv_time, exc_time, exc, sensitivity, tau_scale):
    """Expoential recovery with variable time scales.
    
    Recovery time of each excitation sample is scaled with the amplitude of
    the excitation.
    """
    model = np.zeros(len(dv_time))
    for ii in np.arange(len(dv_time)):
        t = dv_time[ii] - exc_time[exc_time<=dv_time[ii]] 
        texc = exc[:len(t)]
        model[ii] = sensitivity * np.sum(texc * np.exp(-t/(tau_scale*31536000*texc)))
    return model


def _forward_const_log(dv_time, exc_time, exc, sensitivity, tau_min, tau_max):
    """Logarithmic relaxation function with constant time scales
    
    Excitation time series is convolved with a relaxation function generated 
    by integration of scaled exponential with time constats between tau_min 
    and tau_max
    """
    tscales = np.logspace(tau_min,tau_max, num=100,endpoint=True,base=10)
    rec = np.zeros(len(exc_time))  
    t = (exc_time - exc_time[0])/86400
    for ts in tscales:    
        rec += np.exp(-t/ts)
    rec *= sensitivity
    model = np.convolve(exc,rec,"full")
    model = model[:len(exc)]
    model = np.interp(dv_time,exc_time,model)
    model -= np.min(model)
    return model


def _forward_var_log(dv_time, exc_time, exc, sensitivity, tau_min, tau_max_scale):
    print sensitivity, tau_min, tau_max_scale
    n_scales  = 100
    t = (exc_time - exc_time[0])/86400
    tscales = np.logspace(tau_min, tau_max_scale+np.log10(np.max(exc)), num=n_scales,
                          endpoint=True,base=10)
    exc_scales = tscales/np.power(10,tau_max_scale)
    exc1 = np.zeros((1, next_pow_2(2*len(exc))))
    exc1[0,:len(exc)] = exc
    excm = np.tile(exc1,(n_scales,1))
    expm = np.zeros_like(excm)
    eexc = np.zeros(exc1.shape[1])
    eexc[:len(exc)] = exc
    for ind,ts in enumerate(tscales):
        expm[ind,:len(exc)] = np.exp(-t/ts)
        excm[ind,eexc < exc_scales[ind]] = 0
    expm = np.fft.irfft(np.fft.rfft(expm,axis=1)*np.fft.rfft(excm,axis=1))
    model = np.sum(expm,axis=0)
    model = model[:len(exc_time)]
    model = np.interp(dv_time,exc_time,model)
    model *= np.power(10,sensitivity)
    model -= np.min(model)
    return model
