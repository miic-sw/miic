# -*- coding: utf-8 -*-
""" calculate cross or autocorrelations
"""
import sys
import glob
import os
import datetime
import logging

from mpi4py import MPI

import numpy as np
from copy import deepcopy

from obspy import Stream, Trace
from obspy.core import UTCDateTime

from miic.core.miic_utils import create_path, dir_read, lat_lon_ele_load, \
        convert_to_matlab, mat_to_ndarray, \
        save_dict_to_matlab_file, datetime_list, correlation_subdir_name, \
        get_valid_traces
from miic.core.stream import stream_add_lat_lon_ele, corr_trace_to_obspy
from miic.core.stream import read_from_filesystem
from miic.core.corr_mat_processing import corr_mat_create_from_traces, \
        corr_mat_extract_trace, corr_mat_merge, corr_mat_extract_trace, \
        corr_mat_resample, corr_mat_from_corr_stream, corr_mat_from_h5
import miic.core.pxcorr_func as px

from miic.core.script_utils import ini_project, combine_station_channels, \
        select_available_combinations



def paracorr(par):
    """Computation of noise correlation functions
    
    Compute noise correlation functions according to specifications parameter
    dictionary ``par``. This function is most conviniently used as a python
    program passing the parameter file as argument. This use is explained in
    the tutorial on correlation.
    
    The processing is performed in the following sequence
    
     * Data is read in typically day-long chunks ideally contained in a single 
       file to speed up reading time
     * Preprocessing on the long sequences to avoid dominating influence of 
       perturbing signals if processed in shorter chunks.
     * dividing these long sequences into shorter ones (typically an hour)
     * time domain preprocessing
     * frequency domain preprocessing
     * correlation
     * if ``direct_output`` is present data is directly writen by individual
       processes
     * optionally rotation of correlation tensor into ZRT system (not possible
       in combination with direct output
     * combination of correlation traces of subsequent time segments in 
       correlation matrices
     * optionally delete traces of individual time segments
    
    :type par: dict
    :param par: processing parameters
    """


    # initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    psize = comm.Get_size()
    
    # set up the logger
    logger = logging.getLogger('paracorr')
    hdlr = logging.FileHandler(os.path.join(par['log_dir'],'%s_paracorr_%d.log' % (
                        par['execution_start'],rank)))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.DEBUG)

    lle_df = lat_lon_ele_load(par['net']['coordinate_file'])
    comb_list = combine_station_channels(par['net']['stations'],
                            par['net']['channels'],par['co'],lle_df)

    sttimes = datetime_list(par['co']['read_start'], par['co']['read_end'], inc=par['co']['read_inc'])	## loop over 24hrs/whole days
    time_inc = datetime.timedelta(seconds=par['co']['read_len'])
    res_dir = par['co']['res_dir']
    station_list = par['net']['stations']
    channel_list = par['net']['channels']

    channel_dict={}
    for station in station_list :
        channel_dict[station]=map(lambda x: str.replace(x,x,x[-3:]),
                   filter(lambda x:station+'.' in x, set(comb_list[0]+comb_list[1])))


    program_start = UTCDateTime()

    # bulid a dictionary that caches the streams that reside in the same file
    stream_cache = {}
    for station in par['net']['stations']:
        stream_cache.update({station:{}})
        #for channel in par['net']['channels']:
        for channel in channel_dict[station] :
            stream_cache[station].update({channel:Stream().append(Trace())})


    # mapping of stations to processes
    pmap = (np.arange(len(station_list))*psize)/len(station_list)
    # indecies for stations to be worked on by each process
    st_ind = np.where(pmap == rank)[0]

    # number of subdivision of read length
    if 'subdivision' in par['co'].keys():
        nsub = int(np.ceil((float(par['co']['read_len']) - par['co']['subdivision']['corr_len'])
                                      /par['co']['subdivision']['corr_inc'])+1)
    else:
        nsub = 1

    # loop over times
    pathname = os.path.join(res_dir, correlation_subdir_name(sttimes[0]))
    print '\nrank %d of %d'  % (rank,psize)
    logger.debug('Rank %d of %d Beginning execution.'  % (rank,psize))
    for sttime in sttimes:
        if rank == 0:
            print "\n>>> Working on %s at %s:" % (sttime,UTCDateTime())
            logger.debug("\n>>> Working on %s at %s:" % (sttime,UTCDateTime()))
        usttime = UTCDateTime(sttime)
        # fill cache and extract current stream: only done by process 0
        cst = Stream()
        # loop over stations different stations for every process
        for this_ind in st_ind:
            station = station_list[this_ind]
            tst = Stream()
            #for channel in channel_list:
            for channel in channel_dict[station] :
                print channel, station
                try:
                    if ((len(stream_cache[station][channel])==0) or 
                            (not ((stream_cache[station][channel][0].stats['starttime']<=usttime) &
                            (stream_cache[station][channel][0].stats['endtime']>=(usttime+par['co']['read_len']))))):
                        stream_cache[station][channel] = read_from_filesystem('%s.*.%s' %(station, channel), sttime, sttime+time_inc, par['net']['fss'], trim=False)
                        if not stream_cache[station][channel]:
                            logger.warning("%s %s at %s: No trace read." % (station, channel, sttime))
                            continue
                        samp_flag = False
                        for tr in stream_cache[station][channel]:
                            if tr.stats.sampling_rate != par['co']['sampling_rate']:
                                samp_flag = True
                        if samp_flag:
                            logger.warning("%s %s at %s: Mismatching sampling rate." % (station, channel, sttime))
                            stream_cache[station][channel] = Stream()
                            continue
                        if par['co']['decimation'] > 1:
                            sst = stream_cache[station][channel].split()
                            sst.decimate(par['co']['decimation'])
                            stream_cache[station][channel] = deepcopy(sst.merge())
                    ttst = stream_cache[station][channel].copy().trim(starttime=usttime,
                                                 endtime=usttime+par['co']['read_len'])
                    get_valid_traces(ttst)
                    tst +=ttst
                except:
                    logger.warning("%s %s at %s: %s" % (station, channel, sttime, sys.exc_info()[0]))
            cst += tst
        cst = stream_add_lat_lon_ele(cst,lle_df)
        # initial preprocessing on long time series
        if 'preProcessing' in par['co'].keys():
            for procStep in par['co']['preProcessing']:
                cst = procStep['function'](cst,**procStep['args'])


        # create output path
        pathname = os.path.join(par['co']['res_dir'],correlation_subdir_name(sttime))
        if rank == 0:
            create_path(pathname)
            print "\tCorrelating  %s combinations at %s:" % (str(len(comb_list[0])),UTCDateTime())
            logger.debug("\tCorrelating  %s combinations at %s:" % (str(len(comb_list[0])),UTCDateTime()))
                
        # broadcast every station to every process    
        st = Stream()
        for pind in range(psize):
            pst = Stream()
            if rank == pind:
                pst = deepcopy(cst)
            pst = comm.bcast(pst, root=pind)
            st += pst


        ## do correlations
        if len(st) == 0:
            logger.warning("%s: No traces to correlate." % (sttime))
        else:
            targs = deepcopy(par['co']['corr_args'])
            if 'direct_output' in targs.keys():
                if targs['direct_output']['function'] == 'convert_to_matlab':
                    targs['direct_output']['base_dir'] = pathname
                if  targs['direct_output']['function'] == 'corr_to_hdf5':
                    targs['direct_output']['base_dir'] = par['co']['res_dir']

            # loop over subdivisions
            for subn in range(nsub):
                if nsub > 1:
                    sub_st = st.copy().trim(starttime=UTCDateTime(sttime)+
                                    subn*par['co']['subdivision']['corr_inc'],
                                    endtime=UTCDateTime(sttime)+subn*par['co']['subdivision']['corr_inc']+
                                    par['co']['subdivision']['corr_len'])
                    get_valid_traces(sub_st)
                else:
                    sub_st = st
                targs.update({'combinations': select_available_combinations(sub_st,comb_list,targs)})
                if len(targs['combinations']) == 0:
                    continue
                cst = px.stream_pxcorr(sub_st,targs,comm=comm)


    program_end = UTCDateTime()

    print 'rank %d execution time' % rank, program_end-program_start
    logger.debug('Rank %d of %d End execution.'  % (rank,psize))




if __name__=="__main__":
    if len(sys.argv) < 2:
        print 'Specify the parameter file name as first argument.'
        sys.exit()
    par_file = sys.argv[1]
    # initialize the project, create folders and set derived parameters
    par = ini_project(par_file)
    
    # create output directory
    create_path(par['co']['res_dir'])
    # do correlations
    paracorr(par)
    
