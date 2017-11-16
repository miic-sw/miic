import os
import sys
import logging
from miic.core.miic_utils import dir_read, mat_to_ndarray
from miic.core.script_utils import read_parameter_file, ini_project, create_path
from miic.core import corr_mat_processing as cm
from miic.core import plot_fun as pf
from miic.core.stream import corr_trace_to_obspy, stream_stack_distance_intervals



def plot_correlation_matrices(par,plot_par):
    """Plot the correlation matrices.
    """
    # set up the logger
    logger = logging.getLogger('paracorr')
    hdlr = logging.FileHandler(os.path.join(par['log_dir'],'%s_plot_correlation_matrices.log' % (
                        par['execution_start'])))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.DEBUG)
    
    fnames = dir_read(par['co']['res_dir'],'mat__*.mat')
    for fname in fnames:
        mat = mat_to_ndarray(fname)
        fmat = cm.corr_mat_filter(mat,[plot_par['co_mat']['freqmin'],
                                       plot_par['co_mat']['freqmax']])
        filename = os.path.split(fname)[1].replace('.mat','.png')
        filename = os.path.join(par['co']['fig_dir'],filename)
        pf.plot_single_corr_matrix(fmat,seconds=plot_par['co_mat']['corr_len'],
                                  filename=filename)


def plot_trace_distance_sections(par,plot_par):
    """Plot trace distance sections in different frequency bands.
    """
    # set up the logger
    logger = logging.getLogger('paracorr')
    hdlr = logging.FileHandler(os.path.join(par['log_dir'],'%s_plot_correlation_matrices.log' % (
                        par['execution_start'])))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.DEBUG)
    
    trl = []
    fnames = dir_read(par['co']['res_dir'],'ctr__*.mat')
    for fname in fnames:
        trl.append(mat_to_ndarray(fname))
    st = corr_trace_to_obspy(trl)
    st = st.select(channel=plot_par['co_dist_section']['channel'])
    freqmax = plot_par['co_dist_section']['freqmin']
    while freqmax <= plot_par['co_dist_section']['freqmax']:
        cst = st.copy()
        freqmin = freqmax
        freqmax *= plot_par['co_dist_section']['freq_inc_fac']
        cst.filter(type='bandpass',freqmin=freqmin,freqmax=freqmax,zerophase=True)
        if plot_par['co_dist_section']['stack']:
            cst = stream_stack_distance_intervals(cst,
                            plot_par['co_dist_section']['stack_interval'],norm_type='rms')
        fname = 'trace_distance_section_%0.3f-%0.3f.png' % (freqmin, freqmax)
        fname = os.path.join(par['co']['fig_dir'],fname)
        pf.plot_trace_distance_section(cst,scale=0.2,outfile=fname)



if __name__=="__main__":
    if len(sys.argv) < 3:
        print 'Specify the project parameter file and the plot parameter file' \
                'names as first and second arguments.'
        sys.exit()
    par_file = sys.argv[1]
    plot_par_file = sys.argv[2]
    # initialize the project, create folders and set derived parameters
    par = ini_project(par_file)
    plot_par = read_parameter_file(plot_par_file)
    # create output directory
    create_path(par['co']['fig_dir'])

    plot_correlation_matrices(par,plot_par)
    plot_trace_distance_sections(par,plot_par)

