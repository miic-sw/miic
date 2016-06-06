# -*- coding: utf-8 -*-
"""
Estimate potential errors in the timing of seismic data. The script estimates
time differences between station pairs and performes a minimization to obtain 
consistent estimates of individual clock errors.

The operation is controlled by a parameter file that is passed a command line
argument.
"""
from copy import deepcopy
import datetime
import logging
import os
import sys
import numpy as np

from miic.core.miic_utils import dir_read, mat_to_ndarray, datetime_list, \
    create_path, save_dv
import miic.core.plot_fun as pl
from miic.core.corr_mat_processing import corr_mat_shift, corr_mat_normalize, \
    corr_mat_extract_trace, corr_mat_resample, corr_mat_filter, corr_mat_trim
    




def time_difference_estimation(par_file):
    """Estimate clock differences between pairs of stations
    
    This function estimates the time differences between the clocks of pairs
    of stations by measuring the shift of noise correlation functions.
    """
    par = __import__(par_file)
    logging.basicConfig(filename=os.path.join(par.log_dir,'time_difference_\
        estimation.log'), level=logging.DEBUG, format='%(asctime)s %(message)s')
    logger = logging.getLogger('time_difference_estimation')
    logger.info('Hello')
    create_path(par.dt_res_dir)
    create_path(par.dt_fig_dir)
    # create lists of times windows for analyzing changes
    start_time_list = datetime_list(par.dt_start_date, par.dt_end_date,
                                    par.dt_date_inc)
    end_time_list = deepcopy(start_time_list)
    wl = datetime.timedelta(seconds=par.dt_win_len)
    end_time_list = [x+wl for x in end_time_list]

    # available correlations
    flist = dir_read(par.corr_res_dir,'mat*mat')
    for filename in flist:
        try:
            logging.info('Working on %s' % filename)
            mat = mat_to_ndarray(filename)
            # normalize the matrix to maxima
            mat = corr_mat_normalize(mat,normtype='absmax')
            # extract a reference trace (mean excluding very different traces)
            tr = corr_mat_extract_trace(mat,method='mean')
            # resample the correlation matrix
            mat = corr_mat_resample(mat, start_time_list,end_time_list)
            # filter the matrices
            mat = corr_mat_filter(mat,[par.dt_freq_min, par.dt_freq_max])
            # make a trimmed copy for plotting
            tmat = corr_mat_trim(mat,-(par.dt_tw_start+par.dt_tw_len),
                                 (par.dt_tw_start+par.dt_tw_len))
            # plot correlation matrix
            if par.plot_corr_matrix:
                pl.plot_single_corr_matrix(tmat,filename=os.path.join(par.dt_fig_dir,mat['stats']['station']+'_'+mat['stats']['channel']),clim=[-1,1])
            tw = [np.arange(par.dt_tw_start*mat['stats']['sampling_rate'],(par.dt_tw_start+par.dt_tw_len)*mat['stats']['sampling_rate'],1)]
            dt = corr_mat_shift(mat,ref_trc=tr['corr_trace'],return_sim_mat=True,shift_range=20,tw=tw)
            if par.plot_time_shifts:
                filename = mat['stats']['station']+'_'+mat['stats']['channel']
                pl.plot_dv(dt,save_dir=par.dt_fig_dir,figure_file_name=filename,normalize_simmat=True,sim_mat_Clim=[-1,1.])
            save_dv(dt, '', save_dir=par.dt_res_dir)
        except:
            logging.warning('Error processing %s' % filename)
    return 0


if __name__=="__main__":
    if len(sys.argv) < 2:
        print 'Specify the parameter file name as first argument.'
        sys.exit()
    par_file = sys.argv[1]
    time_difference_estimation(par_file)
    
    
    
    
