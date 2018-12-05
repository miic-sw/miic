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
        convert_to_matlab, mat_to_ndarray, convert_time_to_string, \
        save_dict_to_matlab_file, datetime_list, correlation_subdir_name, \
        get_valid_traces, convert_time
from miic.core.stream import stream_add_lat_lon_ele, corr_trace_to_obspy
from miic.core.stream import read_from_filesystem
from miic.core.corr_mat_processing import corr_mat_create_from_traces, \
        corr_mat_extract_trace, corr_mat_merge, corr_mat_extract_trace, \
        corr_mat_resample, corr_mat_from_corr_stream, corr_mat_from_h5, \
        corr_mat_time_select, corr_mat_rotate_horiz, corr_mat_to_h5
import miic.core.pxcorr_func as px

from miic.core.script_utils import ini_project, combine_station_channels, \
        select_available_combinations


def paracorr(par):
    """Computation of noise correlation functions
    
    Compute noise correlation functions according to specifications parameter
    dictionary ``par``. This function is most conveniently used as a python
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
     * function version expects ``direct_output`` and ``corr_to_hdf5`` 
     * data is directly written by individual processes sequentially to hdf5
     * format files. 

     * Options in previous paracorr() version (not using h5 direct-output) which
     * are no longer relevant
     * if ``direct_output`` is present data is directly written by individual
       processes
     * combination of correlation traces of subsequent time segments in 
       correlation matrices
     * optionally rotation of correlation tensor into ZRT system (not possible
       in combination with direct output
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

    # Determine expected channels (defined in coordinate_file)
    channel_dict={}
    for station in station_list :
        channel_dict[station]=map(lambda x: str.replace(x,x,x[-3:]),
                   filter(lambda x:station+'.' in x, set(comb_list[0]+comb_list[1])))


    program_start = UTCDateTime()

    # build a dictionary that caches the streams that reside in the same file
    stream_cache = {}
    for station in par['net']['stations']:
        stream_cache.update({station:{}})
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


        if rank == 0 :
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
                    pathname = os.path.join(par['co']['res_dir'],correlation_subdir_name(sttime))
                    targs['direct_output']['base_dir'] = pathname
                if  targs['direct_output']['function'] == 'corr_to_hdf5':
                    targs['direct_output']['base_dir'] = par['co']['res_dir']

            # loop over subdivisions
            for subn in range(nsub):
                if nsub > 1:
                    sub_st = st.copy().trim(starttime=UTCDateTime(sttime)+
                                    subn*par['co']['subdivision']['corr_inc'],
                                    endtime=UTCDateTime(sttime)+subn*par['co']['subdivision']['corr_inc']+
                                    par['co']['subdivision']['corr_len']-
                                    1./(par['co']['sampling_rate']/par['co']['decimation']))
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




def resample_h5_results(par):
    """Resample h5 files.
    
    Resample the matrices stored in h5 to increments par['co']['read_inc']
    and save as matlab format files. Also write a single mean corr trace.
    
    Function performs a similar function to merge_corr_results in the matlab
    format version of calculate_noise_correlations.py
    """
    
    # initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    psize = comm.Get_size()
    
    # set up the logger
    logger = logging.getLogger('resample_h5_corrs')
    hdlr = logging.FileHandler(os.path.join(par['log_dir'],'%s_resample_h5_results_%d.log' % (
                        par['execution_start'],rank)))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.DEBUG)
    
    program_start = UTCDateTime()
    
    print '\nresample_h5_results - Rank %d of %d'  % (rank,psize)
    logger.debug('Rank %d of %d Beginning execution.'  % (rank,psize))

    lle_df = lat_lon_ele_load(par['net']['coordinate_file'])
    comb_list = combine_station_channels(par['net']['stations'],
                            par['net']['channels'],par['co'],lle_df)

    if ('subdivision' in par['co']) and not par['co']['subdivision']['recombine_subdivision']:
        sttimes = datetime_list(par['co']['read_start'], par['co']['read_end'], 
                                inc=par['co']['subdivision']['corr_inc'])
    else:
        sttimes = datetime_list(par['co']['read_start'], par['co']['read_end'], inc=par['co']['read_inc'])


    # mapping of combinations to processes
    pmap = (np.arange(len(comb_list[0]))*psize)/len(comb_list[0])
    # indices for stations to be worked on by each process
    comb_ind = np.where(pmap == rank)[0]

    for cnt,cind in enumerate(comb_ind):

        if par['co']['corr_args']['direct_output']['function'] == 'corr_to_hdf5':
            logger.debug("\n>>> Merging combination %s-%s (%s/%s) at %s:" % \
                    (comb_list[0][cind],comb_list[1][cind],str(cnt+1),str(len(comb_ind)),UTCDateTime()))
            print("\n>>> Merging combination %s-%s (%s/%s) at %s:" % \
                    (comb_list[0][cind],comb_list[1][cind],str(cnt+1),str(len(comb_ind)),UTCDateTime()))
            ID0 = comb_list[0][cind].split('.')
            ID1 = comb_list[1][cind].split('.')
            comb_str = '%s_%s%s.%s%s.*.%s%s.h5' % (par['co']['corr_args']['direct_output']['base_name'],
                ID0[0],ID1[0],ID0[1],ID1[1],ID0[3],ID1[3])
            h5lst=glob.glob(os.path.join(par['co']['res_dir'],"".join([ID0[3],ID1[3]]),comb_str))
            if not len(h5lst) == 1 :
                logger.debug("\n>>> Combination %s-%s  not found " % \
                        (comb_list[0][cind],comb_list[1][cind],))
            else :
                mat = corr_mat_from_h5(h5lst[0])
                # Save normalised mean trace for combination
                tr = corr_mat_extract_trace(mat,method='norm_mean')
                save_dict_to_matlab_file(h5lst[0].replace(par['co']['corr_args']['direct_output']['base_name'],'ctr').replace('h5','mat'),tr)

                # Generate matrix resampled to the read increments (e.g. daily averaged CCFs) and save
                # Resample time range defined by first to last full day of correlations.
                start=convert_time_to_string([datetime.datetime.combine(convert_time(mat['time'])[0].date()+datetime.timedelta(days=1),
                                datetime.time.min)])[0]

                read_time_period=False
                if not read_time_period :
                    resample_times=datetime_list(start, mat['time'][-1], inc=par['co']['read_inc'])
                else :
                    resample_times=datetime_list(par['co']['read_start'], par['co']['read_end'], inc=par['co']['read_inc'])

                resampled_mat=corr_mat_resample(deepcopy(mat),resample_times) # Resample to read increments
                save_dict_to_matlab_file(h5lst[0].replace(
                    par['co']['corr_args']['direct_output']['base_name'],
                    'resamp_'+par['co']['corr_args']['direct_output']['base_name']).replace('h5','mat'),resampled_mat)
        else :
            logger.debug("\n>>> Correlation was not run with h5 direct ouput")

    program_end = UTCDateTime()
    print 'merge_corr_results rank %d execution time' % rank, program_end-program_start
    logger.debug('Rank %d of %d End execution.'  % (rank,psize))
    comm.barrier() # all processes must be complete




def rotate_horizontals(par,file_type="mat") :
    '''
    Rotate horizontal correlations (NE,NN,EN,EE) to RR,TT,RT,TR 
    Can be used to rotate either the direct output .h5 files, or 
    the .mat resampled matrices.

    :type par: dict
    :param par: processing parameters
    :type file_type: str
    :param file_type: method. File type of target matrices files
            "mat" for matlab format resampled matrices (resamp_)
            "h5" for h5-direct-output files
    '''

    # initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    psize = comm.Get_size()

    # set up the logger
    logger = logging.getLogger('rotate_corr')
    hdlr = logging.FileHandler(os.path.join(par['log_dir'],'%s_rotate_corr_%d.log' % (
                        par['execution_start'],rank)))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.DEBUG)

    program_start = UTCDateTime()

    print('Rank %d of %d Beginning execution.'  % (rank,psize))
    logger.debug('Rank %d of %d Beginning execution.'  % (rank,psize))

    # Get list of available channel combinations
    lle_df = lat_lon_ele_load(par['net']['coordinate_file'])
    comb_list = combine_station_channels(par['net']['stations'],
                            par['net']['channels'],par['co'],lle_df)

    # Create a list of the selected station combinations
    sta1 = [".".join(element.split('.')[:2]) for element in comb_list[0]]
    sta2 = [".".join(element.split('.')[:2]) for element in comb_list[1]]
    selected_sta_comb = sorted(set([sta1[ind]+'-'+sta2[ind] for ind in range(0,len(sta1)) ]))

    # Build a dictionary of what channels are present at stations
    station_list=sorted(set(sta1+sta2))
    channel_dict={}
    for station in station_list :
        channel_dict[station]=map(lambda x: str.replace(x,x,x[-3:]),
                   filter(lambda x:station+'.' in x, set(comb_list[0]+comb_list[1])))

    # Build list of station combinations (different to channel combinations) called comb_list (odd i know)
    comb_list=[[],[]]
    for ii in range(0,len(selected_sta_comb)) :
        comb_list[0].append(selected_sta_comb[ii].split("-")[0])
        comb_list[1].append(selected_sta_comb[ii].split("-")[1])

    # mapping of combinations to processes
    pmap = (np.arange(len(comb_list[0]))*psize)/len(comb_list[0])
    # indecies for stations to be worked on by each process
    comb_ind = np.where(pmap == rank)[0]

    for cnt,cind in enumerate(comb_ind):
        ID0 = comb_list[0][cind].split('.')
        ID1 = comb_list[1][cind].split('.')
        stem1=channel_dict[comb_list[0][cind]][0][:2]
        stem2=channel_dict[comb_list[1][cind]][0][:2]
        # Rotate the resampled files (second line) or the original h5 files (first line)
        if file_type == 'h5' :
            comb_str = '%s_%s%s.%s%s.*.%s[NE]%s[NE].h5' % (par['co']['corr_args']['direct_output']['base_name'],
                        ID0[0],ID1[0],ID0[1],ID1[1],stem1,stem2)
        elif file_type == 'mat' :
            comb_str = '%s_%s%s.%s%s.*.%s[NE]%s[NE].mat' % ('resamp_'+par['co']['corr_args']['direct_output']['base_name'],
                        ID0[0],ID1[0],ID0[1],ID1[1],stem1,stem2)

        # If direct output files were separated
        if 'separate' in par['co']['corr_args']['direct_output'].keys():
            fnames=glob.glob(os.path.join(par['co']['res_dir'],'*',comb_str))
        else :
            fnames=glob.glob(os.path.join(par['co']['res_dir'],comb_str))


        mat_list,ncf_chnls=[],[]
        for fname in fnames :
            # Rotate the resampled files or the original h5 files
            if file_type == 'mat' :
                mat_list.append(mat_to_ndarray(fname)) # For use if rotating .mat files
            elif file_type == 'h5' :
                mat_list.append(corr_mat_from_h5(fname)) # For use if rotating h5 files from direct output.
            ncf_chnls.append(os.path.basename(os.path.dirname(fname)))
        if not len(ncf_chnls) == 4 :
            print("Unexpected num of channels (%s) for %s" % (str(len(ncf_chnls)),comb_str))
            logger.warning("Unexpected num of channels (%s) for %s" % (str(len(ncf_chnls)),comb_str) )
            continue
        # Do rotation
        print("Rank %s doing rotation of channels (%s) for %s at %s" % (str(rank),str(len(ncf_chnls)),comb_str,UTCDateTime.now()))
        rotated=corr_mat_rotate_horiz(mat_list)

        # Write matlab or h5 files for the RR,TT,RT,TR channels, find mean trace and write out.
        for rot_corr_mat in rotated :
            # Define output subdirectory
            if 'separate' in par['co']['corr_args']['direct_output'].keys():
                allowed = ['network','station','location','channel']
                if par['co']['corr_args']['direct_output']['separate'] not in allowed:
                    raise InputError("'direct_output['separate']' must be in %s" % allowed)
                subdir=rot_corr_mat['stats'][par['co']['corr_args']['direct_output']['separate']].replace('-','')
            else :
                subdir=""
            corr_dir=par['co']['res_dir']
            create_path(os.path.join(corr_dir,subdir))


            if file_type == 'h5' :
                comb_str = '%s_%s%s.%s%s.%s%s.%s%s.h5' % (par['co']['corr_args']['direct_output']['base_name'],
                    rot_corr_mat['stats_tr1']['network'],rot_corr_mat['stats_tr2']['network'],
                    rot_corr_mat['stats_tr1']['station'],rot_corr_mat['stats_tr2']['station'],
                    rot_corr_mat['stats_tr1']['location'],rot_corr_mat['stats_tr2']['location'],
                    rot_corr_mat['stats_tr1']['channel'],rot_corr_mat['stats_tr2']['channel'])
                corr_mat_to_h5(rot_corr_mat,os.path.join(corr_dir,subdir,comb_str)) # when rotating h5 files
                tr = corr_mat_extract_trace(rot_corr_mat,method='norm_mean')
                save_dict_to_matlab_file(os.path.join(corr_dir,subdir,
                    comb_str.replace(par['co']['corr_args']['direct_output']['base_name'],'ctr').replace('h5','mat')),tr)
            elif file_type == 'mat' :
                comb_str = '%s_%s%s.%s%s.%s%s.%s%s.mat' % ('resamp_'+par['co']['corr_args']['direct_output']['base_name'],
                    rot_corr_mat['stats_tr1']['network'],rot_corr_mat['stats_tr2']['network'],
                    rot_corr_mat['stats_tr1']['station'],rot_corr_mat['stats_tr2']['station'],
                    rot_corr_mat['stats_tr1']['location'],rot_corr_mat['stats_tr2']['location'],
                    rot_corr_mat['stats_tr1']['channel'],rot_corr_mat['stats_tr2']['channel'])
                save_dict_to_matlab_file(os.path.join(corr_dir,subdir,comb_str),rot_corr_mat) # when rotating mat files
                tr = corr_mat_extract_trace(rot_corr_mat,method='norm_mean')
                save_dict_to_matlab_file(os.path.join(corr_dir,subdir,
                    comb_str.replace(par['co']['corr_args']['direct_output']['base_name'],'ctr')),tr)

    print('Rank %d of %d finishing execution.'  % (rank,psize))
    logger.debug('Rank %d of %d finishing execution.'  % (rank,psize))
    comm.barrier() # all processes must be complete




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

    # # resample correlation results
    # resample_h5_results(par)

    # # Rotate horizontals in resampled corr_matrix to RT system
    # rotate_horizontals(par,file_type="mat")
    # # Rotate direct-output corr_matrix to RT system
    # rotate_horizontals(par,file_type="h5")




