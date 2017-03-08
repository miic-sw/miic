# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:33:04 2017

@author: chris
"""

import os
import yaml
from miic.core.miic_utils import create_path
import miic.core.pxcorr_func as px
from copy import deepcopy


def ini_project(par_file):
    """Initialize a project of network synchrinization
    
    Read the yaml parameter file, and complete its content, i.e. combine
    subdirectory names. Some project relevant directories are created and
    directory names completed. The list of combinations is filled if not given
    explicitly. The parameter dictionary is returned.
    
    :type par_file: str
    :param par_file: path of the yaml parameter file
    :rtype: dict
    :return: parameters
    """

    with open(par_file,'rb') as f:
        try:
            par = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise(exc)
    create_path(par['proj_dir'])
    par.update({'log_dir':os.path.join(par['proj_dir'],par['log_subdir'])})
    par.update({'fig_dir':os.path.join(par['proj_dir'],par['fig_subdir'])})
    create_path(par['log_dir'])
    create_path(par['fig_dir'])
    
    par['co'].update({'res_dir': os.path.join(par['proj_dir'],
                                             par['co']['subdir'])})
    create_path(par['co']['res_dir'])
    
    # create corr_args
    # replace function name by function itself
    for func in par['co']['corr_args']['TDpreProcessing']:
        func['function'] = getattr(px, func['function'])
    for func in par['co']['corr_args']['FDpreProcessing']:
        func['function'] = getattr(px, func['function'])
    if 'direct_output' in par['co']['corr_args']:
        par['co']['corr_args']['direct_output'].update({'base_dir': par['co']['res_dir']})
    return par



def correlation_subdir_name(date):
    """Create the path to a sub folder for the correlation traces.

    The path will have the following structure:
    YEAR/JDAY
    """

    if isinstance(date,UTCDateTime):
        date = date.datetime

    subpath = join(str(date.year),"%03d" % date.timetuple().tm_yday)

    return subpath


def combine_station_channels(stations,channels,method):
    """Create a list of combination

    Combine stations given in a list of NET.STATION parts of the seedID
    and a list of channel names using a given method.
    
    :type stations: list with items of the form 'NET.STATION'
    :param stations: stations to be used if present
    :type channel: list with the channel names
    :param channels: channels to be used if present
    :type method: string
    :param method: Determines which traces of the strem are combined.
    
        ``'betweenStations'``: Traces are combined if either their station or
            their network names are different including all possible channel
            combinations.
        ``'betweenComponents'``: Traces are combined if their components (last
            letter of channel name) names are different and their station and
            network names are identical (single station cross-correlation).
        ``'autoComponents'``: Traces are combined only with themselves.
        ``'allSimpleCombinations'``: All Traces are combined once (onle one of
            (0,1) and (1,0))
        ``'allCombinations'``: All traces are combined in both orders ((0,1)
            and (1,0))
    :rtype: list
    :return: list containing two list [[IDs of first trace],[IDs of second trace]]
    """
    stations = sorted(deepcopy(stations))
    channels = sorted(deepcopy(channels))
    first = []
    second = []
    if method == 'betweenStations':
        for ii in range(len(stations)):
            for jj in range(ii+1,len(stations)):
                for k in range(len(channels)):
                    for l in range(len(channels)):
                        first.append('%s..%s' % (stations[ii],channels[k]))
                        second.append('%s..%s' % (stations[jj],channels[l]))
    elif method == 'betweenComponents':
        for ii in range(len(stations)):
            for k in range(len(channels)):
                for l in range(k+1,len(channels)):
                    first.append('%s..%s' % (stations[ii],channels[k]))
                    second.append('%s..%s' % (stations[ii],channels[l]))
    elif method == 'autoComponents':
        for ii in range(len(stations)):
            for k in range(len(channels)):
                first.append('%s..%s' % (stations[ii],channels[k]))
                second.append('%s..%s' % (stations[ii],channels[k]))
    elif method == 'allSimpleCombinations':
        for ii in range(len(stations)):
            for jj in range(ii,len(stations)):
                for k in range(len(channels)):
                    for l in range(len(channels)):
                        first.append('%s..%s' % (stations[ii],channels[k]))
                        second.append('%s..%s' % (stations[jj],channels[l]))
    else:
        raise ValueError("Method has to be one of ('betweenStations', "
                         "'betweenComponents', 'autoComponents',"
                         "'allSimpleCombinations').")
    return [first, second]


def select_available_combinations(st,comb_list):
   """Estimate available subset of combinations
   """
   stations = []
   combis = []
   for tr in st:
       stations.append('%s.%s..%s' %(tr.stats.network,tr.stats.station,tr.stats.channel))

   for ind in range(len(st)):
       # find all occurrences of a trace in combinations
       findex = [ii for ii,x in enumerate(comb_list[0]) if x == stations[ind]]
       # for every occurrence ...
       for find in findex:
           try:
               # ... check whether the trace that is to be combined is present in the stream
               sind = stations.index(comb_list[1][find])
               combis.append((ind,sind))
           except:
               pass

   return combis


