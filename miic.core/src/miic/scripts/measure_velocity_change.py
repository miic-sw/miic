# coding: utf-8

import os
import logging
import datetime
import sys
import numpy as np
from copy import deepcopy

from miic.core.miic_utils import save_dv, dir_read, mat_to_ndarray, datetime_list
from miic.core.script_utils import ini_project, read_parameter_file, create_path
from miic.core.corr_mat_processing import corr_mat_filter, corr_mat_correct_stretch, corr_mat_stretch, corr_mat_normalize, corr_mat_extract_trace, corr_mat_resample, corr_mat_trim
import miic.core.plot_fun as pf





def measure_velocity_change(par):
    """Estimate velocity changes.
    
    This function estimates the velocity changes from noise correlation functions.
    
    :type par: dict
    :param par: project parameters
    """
    
    # set up the logger
    logger = logging.getLogger('measure_velocity_change')
    hdlr = logging.FileHandler(os.path.join(par['log_dir'],'%s_measure_velocity_change.log' % (
                        par['execution_start'])))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.DEBUG)
    
    # create lists of times windows for analyzing changes
    start_time_list = datetime_list(par['dv']['start_date'], par['dv']['end_date'],
                                    par['dv']['date_inc'])
    end_time_list = deepcopy(start_time_list)
    wl = datetime.timedelta(seconds=par['dv']['win_len'])
    end_time_list = [x+wl for x in end_time_list]

    # available correlations
    fnames = dir_read(par['co']['res_dir'],'mat__*.mat')

    for fname in fnames:
        try:
            logging.info('Working on combination %s' % fname)
            mat = mat_to_ndarray(fname)
            # normalize the matrix to maxima
            mat = corr_mat_normalize(mat,normtype='absmax')
            # resample the correlation matrix
            mat = corr_mat_resample(mat, start_time_list,end_time_list)
            # filter the matrices
            mat = corr_mat_filter(mat,[par['dv']['freq_min'], par['dv']['freq_max']])
            # make a trimmed copy for plotting
            tmat = corr_mat_trim(mat,-(par['dv']['tw_start']+par['dv']['tw_len']),
                                 (par['dv']['tw_start']+par['dv']['tw_len']))

            tw = [np.arange(par['dv']['tw_start']*tmat['stats']['sampling_rate'],(par['dv']['tw_start']+par['dv']['tw_len'])*tmat['stats']['sampling_rate'],1)]

            # extract initial reference trace (mean excluding very different traces)
            tr = corr_mat_extract_trace(tmat,method='mean')
            tw = [np.arange(par['dv']['tw_start']*tmat['stats']['sampling_rate'],(par['dv']['tw_start']+par['dv']['tw_len'])*tmat['stats']['sampling_rate'],1)]
            # initial time shift estimation            
            dv = corr_mat_stretch(tmat,ref_trc=tr['corr_trace'],return_sim_mat=True,stretch_steps=par['dv']['stretch_steps'],stretch_range=par['dv']['stretch_range'],tw=tw)
            # correct the traces for the shift and re-create a better reference
            cmat = corr_mat_correct_stretch(mat,dv)
            # make a trimmed copy
            tcmat = corr_mat_trim(cmat,-(par['dv']['tw_start']+par['dv']['tw_len']),
                                 (par['dv']['tw_start']+par['dv']['tw_len']))
            # extract the final reference trace (mean excluding very different traces)
            tr = corr_mat_extract_trace(tcmat,method='mean')
            # obtain an improved time shift measurement
            dv = corr_mat_stretch(tmat,ref_trc=tr['corr_trace'],return_sim_mat=True,stretch_steps=par['dv']['stretch_steps'],stretch_range=par['dv']['stretch_range'],tw=tw)

            if par['dv']['plot_vel_change']:
                filename = mat['stats']['station']+'_'+mat['stats']['channel']
                pf.plot_dv(dv,save_dir=par['dv']['fig_dir'],figure_file_name=filename,normalize_simmat=True,sim_mat_Clim=[-1,1.])
            save_dv(dv, '', save_dir=par['dv']['res_dir'])
        except:
            logging.warning('Error processing %s: %s' % (fname,sys.exc_info()[0]))
    return 0
    
 
if __name__=="__main__":
    if len(sys.argv) < 2:
        print 'Specify the project parameter file as first and second arguments.'
        sys.exit()
    par_file = sys.argv[1]
    # initialize the project, create folders and set derived parameters
    par = ini_project(par_file)
    # create output directories
    create_path(par['dv']['fig_dir'])
    create_path(par['dv']['res_dir'])

    measure_velocity_change(par)





