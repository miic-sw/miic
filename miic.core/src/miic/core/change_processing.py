# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:11:53 2016

@author: chris
"""
import numpy as np
from copy import deepcopy
from datetime import datetime, timedelta
from scipy.optimize import fmin, minimize
import pdb

from miic.core.miic_utils import serial_date_from_datetime, convert_time
from miic.core.plot_fun import plot_dv

from obspy.signal.util import next_pow_2




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
    
    
def dv_combine_multi_ref(dv_list,max_shift=0.01):
    """Combine a list of change measuments with different reference traces
    
    Combine a list of change dictionaries obtained from different references
    into a single one. The similarity matricies are simply averaged after they
    have been shifted to account for the offset between the different
    references. The offset between the references is estimated as the shift
    between the two correlation matrices that results in the maximum sum of
    element wise products (correlation). The first measurement in the list is
    not shifted. If the input list of measurements is longer than two, the 
    individual shifts (with respect to the anshifted first measurment) is
    estimated from least squares inversion of the shifts between all
    measurements.
    
    :type dv_list: list dict
    :param dv_list: list of velociy change dictionaries to be combined
    :type max_shift: float
    :param max_shift: maximum shift to be tested beween different measuremants
    
    :type: dict
    :return: combined dv_dict
    """
    
    assert type(dv_list) == type([]), "dv_list is not a list"    
    #stps should be at mostas large as the lagest second axis maller than max_shftp    
    
    
    steps = max_shift/(dv_list[0]['second_axis'][1]-dv_list[0]['second_axis'][0])
    steps = int(np.floor(steps))     
    shift = []
    G = np.zeros(((len(dv_list)**2-len(dv_list))/2,len(dv_list)-1),dtype=float)
    cnt = 0    
    for ind1,dv1 in enumerate(dv_list):
        for ind2,dv2 in enumerate(dv_list):
            if ind2 > ind1:
                shift.append(_dv_shift(dv1,dv2,steps))
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
    cdv['corr'] = [cdv['sim_mat'][ind,cdv['value'][ind]] for ind in range(len(cdv['sim_mat'][:,cdv['value']]))]
    cdv['value']= cdv['second_axis'][cdv['value']]

    return cdv


def _dv_shift(dv1,dv2,steps):
    c = []
    shi_range = np.arange(-steps,steps)
    for shi in shi_range:
        c.append(np.nansum(dv2['sim_mat'][:,steps+shi:-steps+shi]*dv1['sim_mat'][:,steps:-steps]))
    shift = shi_range[np.argmax(c)]
    return shift
    

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
                print '\t%s = %e %s' % (par,model_par[type][par],
                                        model_par[type]['units'][par])
        print '\tfitted parameters: %s\n' % str(model_par[type]['to_vari'])
    
class Change:
    def __init__(self):
        print 'init'
        self.dv = {}
    
    def set_dv(self, dv):
        self.dv = dv
    
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
                                                        'tau_min':0.01,
                                                        'tau_max':1000,
                                                        'Ntau':100,
                                                        'to_vari':['sensitivity','tau_max'],
                                                        'units':{'sensitivity':'1/input',
                                                                 'tau_min':'log10(d)',
                                                                 'tau_max':'log10(d)',
                                                                 'Ntau':'1'}}})
    def fit_model(self, excitation=[],
                  misfit_function=_misfit_int_corr):
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
        mod_with_exc = ['const_exp','var_exp','const_log','var_log']
        if any(x in self.start_model['type'] for x in mod_with_exc):
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
        mdv += _forward_state_model(dv_time, exc_time, exc,
                                    model_par['state_model']['sensitivity'],
                                    model_par['state_model']['tau_min'],
                                    model_par['state_model']['tau_max'],
                                    model_par['state_model']['Ntau'])
    return mdv

import matplotlib.pyplot as plt
def _forward_state_model(dv_time, exc_time, exc, sensitivity, taus, p0):
    """Roel's and Jack's analytic solution to the state model.
    
    units: exc_time in seconds
    tau in seconds
    """
    # int 2 - alpha t
    pdb.set_trace()
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
    pdb.set_trace()   
    print 'sfewrfqF'


def _forward_state_model_fd(dv_time, exc_time, exc, sensitivity, taus, p0):
    """numerical fd solution to the state model.
    
    units: exc_time in seconds
    tau in seconds
    """
    # int 2 - alpha t
    p = np.zeros((len(exc_time),len(taus)))
    alp = 1./taus
    for ind,tau in enumerate(taus):
        if p0[ind] < 0:
            p0[ind] = exc[0]*tau/(1 + tau * exc[0])
    p[0,:] = p0
    dt = np.diff(exc_time)
    #dt = np.concatenate((dt,np.array([dt[-1]])))
    for ind,tdt in enumerate(dt):
        if ind == 3530:
            print 'sdf'
            pdb.set_trace()
        pequi = exc[ind]/(alp+exc[ind])
        p[ind+1,:] = pequi + (p[ind,:] - pequi)*np.exp(-(alp+exc[ind])*tdt)
    model = np.sum(p,axis=1)
    model = np.interp(dv_time,exc_time,model)

    pdb.set_trace()   
    print 'sfewrfqF'
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
