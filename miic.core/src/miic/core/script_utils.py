# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:33:04 2017

@author: chris
"""

import os
import yaml
from miic.core.miic_utils import create_path
import miic.core.pxcorr_func as px



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