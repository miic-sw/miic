# -*- coding: utf-8 -*-
"""
@author:
Eraldo Pomponi

@copyright:
The MIIC Development Team (eraldo.pomponi@uni-leipzig.de)

@license:
GNU Lesser General Public License, Version 3
(http://www.gnu.org/copyleft/lesser.html)

Created on Dec 15, 2011
"""

# Main imports
import os
import datetime

import numpy as np
import scipy.io as sio
from multiprocessing import Pool

import matplotlib.pyplot as plt
import matplotlib.delaunay as delaunay

# Configuration storage
from configobj import ConfigObj

# ETS import
try:
    BC_UI = True
    from traits.api import HasTraits, Int, Float, List, Directory, Str, \
        Bool, Date, Enum, File
    from traitsui.api import Item, VGroup, View, Tabbed
except ImportError:
    BC_UI = False
    pass

# Local imports
from miic.core.ndarray_sigproc import ndarray_smooth, ndarray_filter
from miic.core.stretch_mod import time_windows_creation, multi_ref_creation, \
    multi_ref_vchange_and_align, time_stretch_estimate
from miic.core.miic_utils import dir_read, find_comb, \
    nd_mat_center_part, mat_to_ndarray, collapse_to_single_vect, \
    convert_time, flatten_recarray, dv_check, lat_lon_ele_load, \
    load_pickled_Series_DataFrame_Panel, from_single_pattern_to_panel

from miic.core.plot_fun import plot_single_corr_matrix, plot_dv

#==============================================================================
# For backward compatibility reasons
#==============================================================================

from miic.core.inversion import \
    inversion_with_missed_data as inversion_with_missed_data_macro

from miic.core.miic_utils import create_date_obj

if BC_UI:
    from miic.core.miic_utils import _from_single_pattern_to_panel_view, \
        _create_date_obj_view

###################################################################
# HELP FUNCTIONS
###################################################################


def _stack(X, vect, axis):
    """ Stacks together arrays of compatible size.

    This function stacks (pile up) array with compatible size to create a
    matrix. This stack can be done row-wise (``axis=0``) or column-wise
    (``axis=1``). To achieve this behaviour in a function based environment
    like blockcanvas, it uses a ``global`` variable that persist between
    successive call to this routine.
    It must be taken into account that this global variable needs to be
    reseted by hand if it is necessary to restart the piling procedure
    (see also the function ``clear_global_X``).

    Parameters
    ----------
    vect : ndarray
        1D Array to stack with the current global variable ``X``
    axis : int
        Stacking axis. 0: row-wise
                       1: column-wise

    Returns
    -------
    X : ndarray
        Global variable that holds the stacked data
    """

    if X is None:
        if axis == 0:
            X = np.reshape(vect, (1, vect.size))
        elif axis == 1:
            X = np.reshape(vect, (vect.size, 1))
        return X
    else:
        try:
            if axis == 0:
                X = np.vstack([X, vect.reshape((1, vect.size))])
            elif axis == 1:
                X = np.hstack([X, vect.reshape((vect.size, 1))])

        except Exception, e:
            print "Exception occurred stacking data!!!"
            print "Exception: %s" % e

            if axis == 0:
                fv = np.ones((1, X.shape[1])) * np.NaN
                X = np.vstack([X, fv])
            elif axis == 1:
                fv = np.ones((X.shape[0], 1)) * np.NaN
                X = np.hstack([X, fv])
            pass

    return X

###################################################################
# SPECIAL CALLABLE CLASS                                          #
###################################################################


class _RecombineCorrData:

    def __init__(self, base_name, base_dir, save_dir, channels_pair, \
                 center_win_len, suffix):

        self.base_name = base_name
        self.base_dir = base_dir
        self.save_dir = save_dir
        # self.time_dir = time_dir
        self.channels_pair = channels_pair
        self.center_win_len = center_win_len
        self.suffix = suffix

    def __call__(self, pattern):

        print "#### Start sequence for pattern %s ####" % pattern

        f_pattern = \
            '*' + '_' + self.base_name + \
            '_' + pattern

        # To handle old style filenames we need to add the _.mat later

        if self.suffix != '':
            cf_pattern = f_pattern + '_' + self.suffix

            files_list1 = dir_read(base_dir=self.base_dir,
                       pattern=cf_pattern + '.mat',
                       sort_flag=True)
        else:
            files_list1 = dir_read(base_dir=self.base_dir,
                                   pattern=f_pattern + '.mat',
                                   sort_flag=True)

        if files_list1 == []:
            # Try the old style names with channel pairs in the end
            f_pattern = f_pattern + '_' + self.channels_pair

            if self.suffix != '':
                cf_pattern = f_pattern + '_' + self.suffix

                files_list1 = dir_read(base_dir=self.base_dir,
                           pattern=cf_pattern + '.mat',
                           sort_flag=True)
            else:
                files_list1 = dir_read(base_dir=self.base_dir,
                                       pattern=f_pattern + '.mat',
                                       sort_flag=True)

        default_var_name = 'corr_trace'
        time_format = "%Y-%m-%dT%H:%M:%S.%fZ"

        stats = None
        stats_tr1 = None
        stats_tr2 = None

        first_stats_tr1 = None
        first_stats_tr2 = None

        try:
            X = None
            time_vect = []
            for celem1 in files_list1:

                load_var1 = mat_to_ndarray(celem1)

                # Get stats the first time it is available
                if stats is None:
                    if 'stats' in load_var1:
                        stats = flatten_recarray(load_var1['stats'])
                # Take the first stats_tr1 and stats_tr2 obj as a sample
                # of the specific meta-information of the combined traces
                if stats_tr1 is None:
                    if 'stats_tr1' in load_var1:
                        first_stats_tr1 = \
                            flatten_recarray(load_var1['stats_tr1'])
                if stats_tr2 is None:
                    if 'stats_tr2' in load_var1:
                        first_stats_tr2 = \
                            flatten_recarray(load_var1['stats_tr2'])

                # Take the timing from the stats_tr1 and stats_tr2 obj when
                # available
                if 'stats_tr1' in load_var1 and 'stats_tr2' in load_var1:
                    stats_tr1 = flatten_recarray(load_var1['stats_tr1'])
                    stats_tr2 = flatten_recarray(load_var1['stats_tr2'])

                    time_tr1 = \
                        datetime.datetime.strptime(stats_tr1['starttime'],
                                                          time_format)

                    time_tr2 = \
                        datetime.datetime.strptime(stats_tr2['starttime'],
                                                          time_format)

                    mtime = '%s' % (max(time_tr1, time_tr2))

                else:
                    mtime = \
                        datetime.datetime.strptime(\
                                        flatten_recarray(stats)['starttime'],
                                        time_format)
                    print "Warning: stats_tr1 or stats_tr2 not available: use \
                        default time %s" % mtime

                time_vect.append(mtime)

                tr_pattern = 'trace_' + pattern  # + '_' + self.channels_pair
                if tr_pattern in load_var1:
                    result1 = load_var1[tr_pattern]
                elif default_var_name in load_var1:
                    result1 = load_var1[default_var_name]

                X = _stack(X, result1, 0)

        except Exception, e:
            print "#### Sequence ERROR: EXIT for pattern %s ####" % pattern
            print "Exception: %s" % e
            print "#" * 30
            return 'FAILED_' + pattern

        try:
            if (X is not None) and (self.center_win_len < stats['npts']):

                cent_mat1 = nd_mat_center_part(X, self.center_win_len, axis=1)

                # adjust starttime endtime and npts in stats

                # Trace length in seconds

#                trace_length = self.center_win_len / stats['sampling_rate']
#                trace_length_delta = datetime.timedelta(seconds=trace_length)
#                zerotime = datetime.datetime(1971, 1, 1, 0, 0, 0)
#
#                new_starttime = zerotime - (trace_length_delta / 2)
#                new_endtime = new_starttime + trace_length_delta
#
#                stats['starttime'] = new_starttime.isoformat()
#                stats['endtime'] = new_endtime.isoformat()
#                stats['npts'] = self.center_win_len

                # adjust starttime endtime and npts in stats
                # trace_length = int(self.center_win_len * \
                # stats['sampling_rate'])
                # time from old to new start
                timeOffset = \
                    datetime.timedelta(
                            np.ceil(float(stats['npts'] -
                                          self.center_win_len) / 2) /
                                       stats['sampling_rate'] / 86400)
                stats['starttime'] = datetime.datetime.strftime(timeOffset
                                  + convert_time([stats['starttime']])[0],
                                  time_format)
                stats['npts'] = self.center_win_len
                stats['endtime'] = \
                    datetime.datetime.strftime(\
                        convert_time([stats['starttime']])[0] +
                        datetime.timedelta(float(self.center_win_len - 1) /
                        stats['sampling_rate'] / 86400), time_format)
                if self.suffix != '':
                    file_name = 'mat_' + pattern + '_' + self.suffix + '.mat'
                else:
                    file_name = 'mat_' + pattern + '.mat'

                if (first_stats_tr1 is not None) and \
                    (first_stats_tr2 is not None):
                    sio.savemat(os.path.join(self.save_dir, file_name),
                                {'corr_data': cent_mat1,
                                 'time': time_vect,
                                 'stats': stats,
                                 'stats_tr1': first_stats_tr1,
                                 'stats_tr2': first_stats_tr2},
                                oned_as='column')
                else:
                    sio.savemat(os.path.join(self.save_dir, file_name),
                                {'corr_data': cent_mat1,
                                 'time': time_vect,
                                 'stats': stats},
                                oned_as='column')
            else:
                print "#### Sequence ERROR: No matrix created \
                    for pattern %s" % pattern
                return 'FAILED_' + pattern

            print "#### SUCCESSFULLY END sequence for \
                pattern %s ####\n\n" % pattern

            return 'SUCCESS_' + pattern

        except Exception, e:
            print "#### Sequence ERROR: Saving result\
                for pattern %s ####" % pattern
            print "Exception: %s" % e
            print "#" * 30
            return 'FAILED_' + pattern


###################################################################
# MAIN
###################################################################

def recombine_corr_data(base_name, suffix, base_dir, save_dir, \
                        center_win_len, \
                        channels_pair, fs, old_style=False,
                        parallel=True):
    """ Correlation matrix creation.

    This function creates the whole set of correlation matrix starting from
    a set of corr functions (one for each day and for each stations pair
     - cross - or station - auto - ). The mat files that are created contains
    two variables: ``corr_data`` that stores the correlation matrix and
    ``time`` that stores timing informations

    :type base_name: string
    :param base_name: Common "root" for every generated filename. It must not
        include underscores
    :param suffix: Optional suffix for the filename (it must be the same used
        with the :class:`~miic.core.miic_utils.convert_to_matlab` function
        through with the data were saved)
    :type base_dir: string
    :param base_dir: Where the corr functions are stored
    :type save_dir: string
    :param save_dir: Where all the corr matrix will be saved
    :type channels_pair: string
    :param channels_pair: The channels pair that we are interested with
        (e.g. 'HHZ-HHZ')
    :type patters_list_filename: string
    :param patters_list_filename: File where the results obtained with this
        procedure will be logged. It is necessary for the DataFrame
        construction in the function `from_single_pattern_to_dataframe`
    :type fs: float
    :param fs: Pseudo Sampling frequency for the corr curve
    :type old_style: bool
    :param old_style: If true, the suffic is generated using the `fs`
        ( "_<fs>Hz") instead of being passed as a parameter)
    :type parallel: bool
    :param parallel: if true it tries to use as much cores as are available to
        do the computation
    """

    if not os.path.isdir(save_dir):
        print "`save_dir` doesn't exist ..... creating"
        os.mkdir(save_dir)

    if old_style:
        suffix = "%sHz" % fs

    base_patterns = find_comb(base_dir, suffix=suffix)

    if parallel:
        # Create a pool of process
        pool = Pool()
        successfull = pool.map(_RecombineCorrData(base_name, \
                                                  base_dir, \
                                                  save_dir, \
                                                  channels_pair, \
                                                  center_win_len, \
                                                  suffix), \
                               base_patterns)

        pool.close()
        pool.join()

    else:
        successfull = map(_RecombineCorrData(base_name, \
                                             base_dir, \
                                             save_dir, \
                                             channels_pair, \
                                             center_win_len, \
                                             suffix), \
                          base_patterns)

    np.save(os.path.join(save_dir, 'success_patterns.npy'), \
            successfull)


if BC_UI:
    class _recombine_corr_data_view(HasTraits):
        base_name = Str('trace')
        suffix = Str('')
        base_dir = Directory()
        save_dir = Directory()
        channels_pair = Str('Z-Z')
        center_win_len = Int(3000)
        fs = Float(10.0)
        old_style = Bool(False)
        parallel = Bool(True)
    
        trait_view = View(VGroup(Item('base_name'),
                                 Item('suffix'),
                                 Item('base_dir'),
                                 Item('save_dir'),
                                 Item('center_win_len'),
                                 Item('old_style'),
                                 Item('parallel'),
                                 label='Config'
                                 ),
                          VGroup(Item('channels_pair',
                                      label='Channel pair [e.g.HHZ-HHZ]'),
                                 Item('fs'),
                                 label='Old setup extras',
                                 enabled_when='old_style'))

###################################################################
# SPECIAL CALLABLE CLASS                                          #
###################################################################


class _VChangeEstimate:

    def __init__(self, base_dir, save_dir, center_win_len, fmin, fmax, order, \
                 fs, smooth_win, tw_start_list, tw_len, st_range, st_steps, \
                 suffix):

        self.base_dir = base_dir
        self.save_dir = save_dir
        self.center_win_len = center_win_len
        self.fmin = fmin
        self.fmax = fmax
        self.order = order
        self.fs = fs
        self.smooth_win = smooth_win
        self.tw_start_list = tw_start_list
        self.tw_len = tw_len
        self.st_range = st_range
        self.st_steps = st_steps

        self.suffix = suffix

    def __call__(self, pattern):

        print "#### Start sequence for pattern %s ####" % pattern

        try:
            if self.suffix != '':
                f_pattern = 'mat_' + pattern + '_' + self.suffix + '.mat'
            else:
                f_pattern = 'mat_' + pattern + '.mat'

            filename = os.path.join(self.base_dir, f_pattern)
            load_var1 = mat_to_ndarray(filename)

            X = load_var1['corr_data']

        except Exception, e:
            print "#### Sequence ERROR: Loading mat for\
                pattern %s ####" % pattern
            print "Exception: %s" % e
            print "#" * 30
            return 'FAILED_' + pattern

        try:
            cent_mat1 = nd_mat_center_part(X, self.center_win_len, axis=1)
            X3 = ndarray_filter(cent_mat1, 'bandpass', self.fs, \
                                {'corners': self.order, 'zerophase': True, \
                                 'freqmax': self.fmax, 'freqmin': self.fmin})
            X5 = ndarray_smooth(X3, self.smooth_win, 'flat', axis=1)

        except Exception, e:
            print "#### Sequence ERROR: Filtering for\
                pattern %s ####" % pattern
            print "Exception: %s" % e
            print "#" * 30
            return 'FAILED_' + pattern

        try:

            rowd1 = collapse_to_single_vect(X5, 0)
            tw_mat2 = time_windows_creation(self.tw_start_list, self.tw_len)

            vdict = time_stretch_estimate(X5, ref_trc=rowd1,
                                          tw=tw_mat2,
                                          stretch_range=self.st_range,
                                          stretch_steps=self.st_steps,
                                          sides='both')

#            new_m1, deltas1 = stretch_mat_creation(rowd1, \
#                                                   str_range=self.st_range, \
#                                                   nstr=self.st_steps)
#            vdict = velocity_change_estimete(X5, tw_mat2,
#                                             new_m1, deltas1,
#                                             return_sim_mat=True)

        except Exception, e:
            print "#### Sequence ERROR: Velocity change estimate\
                for pattern %s ####" % pattern
            print "Exception: %s" % e
            print "#" * 30
            return 'FAILED_' + pattern

        try:

            if self.suffix != '':
                vchange_file_name = 'vchange_full_' + \
                    pattern + '_' + self.suffix + '.mat'
            else:
                vchange_file_name = 'vchange_full_' + pattern + '.mat'

            vdict.update({'time': load_var1['time']})

            if 'stats' in load_var1:
                vdict.update({'stats': load_var1['stats']})

            sio.savemat(os.path.join(self.save_dir, vchange_file_name), \
                        vdict,
                        oned_as='column')

            print "#### SUCCESSFULLY END sequence for\
                pattern %s ####\n\n" % pattern

            return 'SUCCESS_' + pattern

        except Exception, e:
            print "#### Sequence ERROR: Saving result for\
                pattern %s ####" % pattern
            print "Exception: %s" % e
            print "#" * 30
            return 'FAILED_' + pattern


def vchange_estimate(base_dir, \
                     suffix, \
                     save_dir, \
                     # corr_dir, \
                     center_win_len, \
                     fmax, \
                     fmin, \
                     order, \
                     fs, \
                     smooth_win, \
                     tw_start_list, \
                     tw_len, \
                     st_range, \
                     st_steps, \
                     list_of_combinations, \
                     all, old_style=False,
                     parallel=True):
    """ Velocity change estimation macro.

    This function, starting from the whole set of correlation matrix created
    for a defined dataset, evaluate the velocity change doing a simple
    pre-processing:
       bandpass filtering and smoothing.

    :type base_dir: string
    :param base_dir: Where the corr matrix are stored
    :type suffix: string
    :param suffix: Optional suffix for the filename (it must be the same used
        with the :class:`~miic.core.macro.recombine_corr_data` function
        through with the corr matrix were saved)
    :type save_dir: string
    :param save_dir: Where all the velocity change traces and the corresponding
        corr traces will be saved.
    :type center_win_len: int
    :param center_win_len: How many samples will be taken from the central part
        of the corr func
    :type fmax: float
    :param fmax: Maximum freq for the bandpass filter
    :type fmin: float
    :param fmin: Minimum freq for the bandpass filter
    :type order: int
    :param order: Filter order (i.e. corners in obspy terminology)
    :type fs: float
    :param fs: Sampling frequency for the corr trace
    :type smooth_win: int
    :param smooth_win: Width of the smoothing window
    :type tw_start_list: list of int
    :param tw_start_list: List of time lag (in samples) that we want to adopt
        in the velocity change evaluation
    :type tw_len: int
    :param tw_len: Time window length in samples for each time lag
    :type st_range: float
    :param st_range: Stretching range (i.e. 0.01 -> 1%)
    :type st_steps: int
    :param st_steps: How many stretching steps in the specified range
    :type list_of_combinations: string
    :param list_of_combinations: The list of combinations that we want to plot
        (e.g. ['QC01-QC02','QC05-QC11']
    :type all: bool
    :param all: If true all possible combinations will be calculated
    :type old_style: bool
    :param old_style: If true, the suffix is generated using the `fs`
        ( "_<fs>Hz") instead of being passed as a parameter)
    :type parallel: bool
    :param parallel: if true it tries to use as much cores as are available to
        do the computation
    """

    if not os.path.isdir(save_dir):
        print "`save_dir` doesn't exist ..... creating"
        os.mkdir(save_dir)

    if old_style:
        suffix = "%sHz" % fs

    if all:
        # base_patterns = find_comb(corr_dir, suffix=suffix)
        base_patterns = find_comb(base_dir, suffix=suffix, is_mat=True)
    else:
        base_patterns = list_of_combinations

    if parallel:
        pool = Pool()
        successfull = pool.map(_VChangeEstimate(base_dir, \
                                                save_dir, \
                                                center_win_len, \
                                                fmin, fmax, order, fs, \
                                                smooth_win, \
                                                tw_start_list, tw_len, \
                                                st_range, st_steps, suffix), \
                               base_patterns)
        pool.close()
        pool.join()
    else:
        successfull = map(_VChangeEstimate(base_dir, \
                                           save_dir, \
                                           center_win_len, \
                                           fmin, fmax, order, fs, \
                                           smooth_win, \
                                           tw_start_list, tw_len, \
                                           st_range, st_steps, suffix), \
                          base_patterns)

    np.save(os.path.join(save_dir, 'success_patterns.npy'), \
            successfull)


if BC_UI:
    class _vchange_estimate_view(HasTraits):
        base_dir = Directory()
        save_dir = Directory()
        # corr_dir = Directory()
        fmax = Float(1.0)
        fmin = Float(0.5)
        order = Int(4)
        fs = Float(10.0)
        smooth_win = Int(5)
        center_win_len = Int(500)
        tw_start_list = List(Int)
        tw_len = List(Int(100))
        st_range = Float(0.01)
        st_steps = Int(1000)
        list_of_combinations = List(Str)
        all = Bool(False)
        suffix = Str('')
        old_style = Bool(False)
        parallel = Bool(True)
    
        trait_view = View(Tabbed(VGroup(Item('base_dir'),
                                        Item('save_dir'),
                                        # Item('corr_dir'),
                                        Item('old_style'),
                                        Item('suffix',
                                             label='filename suffix',
                                             enabled_when='not old_style'),
                                        Item('parallel'),
                                        label='Directories'),
                                 VGroup(Item('fmax'),
                                        Item('fmin'),
                                        Item('order'),
                                        Item('fs'),
                                        Item('smooth_win'),
                                        label='Processing'),
                                 Tabbed(Item('tw_start_list', height=100, \
                                             width=90),
                                        Item('tw_len', height=100, width=90),
                                        Item('center_win_len'),
                                        label='Time lag'),
                                 VGroup(Item('st_range'),
                                        Item('st_steps'),
                                        # Item('patters_list_filename', \
                                        #     label='patters_list_filename\
                                        #     (no ext)'),
                                        label='Stretch'),
                                 VGroup(Item('all'),
                                        Item('list_of_combinations', \
                                             height=100, \
                                             enabled_when='all==False'),
                                        label='combinations')))
    
##############################################################################
# MULTI-REFERENCE VELOCITY CHANGE ESTIMATE                                   #
##############################################################################


class _VChangeEstimateMultiRef:

    def __init__(self, base_dir, save_dir, center_win_len, fmin, fmax, order, \
                 fs, smooth_win, tw_start, tw_len, st_range, st_steps, \
                 suffix, multi_ref_freq, use_breakpoint, breakpoint):

        self.base_dir = base_dir
        self.save_dir = save_dir
        self.center_win_len = center_win_len
        self.fmin = fmin
        self.fmax = fmax
        self.order = order
        self.fs = fs
        self.smooth_win = smooth_win
        self.tw_start = tw_start
        self.tw_len = tw_len
        self.st_range = st_range
        self.st_steps = st_steps

        self.suffix = suffix
        self.multi_ref_freq = multi_ref_freq
        self.use_breakpoint = use_breakpoint
        self.breakpoint = breakpoint

    def __call__(self, pattern):

        print "#### Start sequence for pattern %s ####" % pattern

        try:
            if self.suffix != '':
                f_pattern = 'mat_' + pattern + '_' + self.suffix + '.mat'
            else:
                f_pattern = 'mat_' + pattern + '.mat'
            filename = os.path.join(self.base_dir, f_pattern)
            load_var1 = mat_to_ndarray(filename)

            X = load_var1['corr_data']

        except Exception, e:
            print "#### Sequence ERROR: Loading mat for\
                pattern %s ####" % pattern
            print "Exception: %s" % e
            print "#" * 30
            return 'FAILED_' + pattern

        try:
            cent_mat1 = nd_mat_center_part(X, self.center_win_len, axis=1)
            X3 = ndarray_filter(cent_mat1, 'bandpass', self.fs, \
                                {'corners': self.order, 'zerophase': True, \
                                 'freqmax': self.fmax, 'freqmin': self.fmin})
            corr_mat = ndarray_smooth(X3, self.smooth_win, 'flat', axis=1)

        except Exception, e:
            print "#### Sequence ERROR: Filtering for\
                pattern %s ####" % pattern
            print "Exception: %s" % e
            print "#" * 30
            return 'FAILED_' + pattern

        try:

            rtime = convert_time(load_var1['time'])
            ref_mat = multi_ref_creation(corr_mat,
                                         rtime,
                                         freq=self.multi_ref_freq,
                                         use_break_point=self.use_breakpoint,
                                         break_point=self.breakpoint)

            tw = time_windows_creation(self.tw_start, self.tw_len)

            dv = multi_ref_vchange_and_align(corr_mat,
                                             ref_mat,
                                             tw=tw,
                                             stretch_range=self.st_range,
                                             stretch_steps=self.st_steps,
                                             return_sim_mat=True)

        except Exception, e:
            print "#### Sequence ERROR: Velocity change estimate\
                for pattern %s ####" % pattern
            print "Exception: %s" % e
            print "#" * 30
            return 'FAILED_' + pattern

        try:

            if self.suffix != '':
                vchange_file_name = 'vchange_full_' + \
                    pattern + '_' + self.suffix + '.mat'
            else:
                vchange_file_name = 'vchange_full_' + pattern + '.mat'

            # ADD meta-information to the returned dictionary

            if 'stats' in load_var1:
                dv.update({'stats': load_var1['stats']})
            if 'time' in load_var1:
                dv.update({'time': load_var1['time']})

            sio.savemat(os.path.join(self.save_dir, vchange_file_name),
                        dv,
                        oned_as='column')

            print "#### SUCCESSFULLY END sequence for\
                pattern %s ####\n\n" % pattern

            return 'SUCCESS_' + pattern

        except Exception, e:
            print "#### Sequence ERROR: Saving result for\
                pattern %s ####" % pattern
            print "Exception: %s" % e
            print "#" * 30
            return 'FAILED_' + pattern


def vchange_estimate_multi_ref(base_dir, \
                               suffix, \
                               save_dir, \
                               # corr_dir, \
                               center_win_len, \
                               fmax, \
                               fmin, \
                               order, \
                               fs, \
                               smooth_win, \
                               tw_start, \
                               tw_len, \
                               st_range, \
                               st_steps, \
                               list_of_combinations, \
                               all, \
                               multi_ref_freq, \
                               use_breakpoint, \
                               breakpoint, \
                               old_style=False,
                               parallel=True):
    """ Velocity change estimation macro.

    This function, starting from the whole set of correlation matrix created
    for a defined dataset, evaluate the velocity change doing a simple
    pre-processing:
       bandpass filtering and smoothing.

    :type base_dir: string
    :param base_dir: Where the corr matrix are stored
    :type suffix: string
    :param suffix: Optional suffix for the filename (it must be the same used
        with the :class:`~miic.core.macro.recombine_corr_data` function
        through with the corr matrix were saved)
    :type save_dir: string
    :param save_dir: Where all the velocity change traces and the corresponding
        corr traces will be saved.
    :type center_win_len: int
    :param center_win_len: How many samples will be taken from the central part
        of the corr func
    :type fmax: float
    :param fmax: Maximum freq for the bandpass filter
    :type fmin: float
    :param fmin: Minimum freq for the bandpass filter
    :type order: int
    :param order: Filter order (i.e. corners in obspy terminology)
    :type fs: float
    :param fs: Sampling frequency for the corr trace
    :type smooth_win: int
    :param smooth_win: Width of the smoothing window
    :type tw_start_list: list of int
    :param tw_start_list: List of time lag (in samples) that we want to adopt
        in the velocity change evaluation
    :type tw_len: int
    :param tw_len: Time window length in samples for each time lag
    :type st_range: float
    :param st_range: Stretching range (i.e. 0.01 -> 1%)
    :type st_steps: int
    :param st_steps: How many stretching steps in the specified range
    :type list_of_combinations: string
    :param list_of_combinations: The list of combinations that we want to plot
        (e.g. ['QC01-QC02','QC05-QC11']
    :type all: bool
    :param all: If true all possible combinations will be calculated
    :type old_style: bool
    :param old_style: If true, the suffic is generated using the `fs`
        ( "_<fs>Hz") instead of being passed as a parameter)
    :type parallel: bool
    :param parallel: if true it tries to use as much cores as are available to
        do the computation
    """

    if not os.path.isdir(save_dir):
        print "`save_dir` doesn't exist ..... creating"
        os.mkdir(save_dir)

    pool = Pool()

    if old_style:
        suffix = "%sHz" % fs

    if all:
        base_patterns = find_comb(base_dir, suffix, is_mat=True)
    else:
        base_patterns = list_of_combinations

    if parallel:
        pool = Pool()
        successfull = pool.map(_VChangeEstimateMultiRef(base_dir, \
                                                        save_dir, \
                                                        center_win_len, \
                                                        fmin, \
                                                        fmax, \
                                                        order, \
                                                        fs, \
                                                        smooth_win, \
                                                        tw_start, \
                                                        tw_len, \
                                                        st_range, \
                                                        st_steps, \
                                                        suffix, \
                                                        multi_ref_freq, \
                                                        use_breakpoint, \
                                                        breakpoint), \
                               base_patterns)
        pool.close()
        pool.join()
    else:
        successfull = map(_VChangeEstimateMultiRef(base_dir, \
                                                   save_dir, \
                                                   center_win_len, \
                                                   fmin, \
                                                   fmax, \
                                                   order, \
                                                   fs, \
                                                   smooth_win, \
                                                   tw_start, \
                                                   tw_len, \
                                                   st_range, \
                                                   st_steps, \
                                                   suffix, \
                                                   multi_ref_freq, \
                                                   use_breakpoint, \
                                                   breakpoint), \
                          base_patterns)

    np.save(os.path.join(save_dir, 'success_patterns.npy'), \
            successfull)


if BC_UI:
    class _vchange_estimate_multi_ref_view(HasTraits):
        base_dir = Directory()
        save_dir = Directory()
        # corr_dir = Directory()
        fmax = Float(1.0)
        fmin = Float(0.5)
        order = Int(4)
        fs = Float(10.0)
        smooth_win = Int(5)
        center_win_len = Int(500)
        tw_start = Int(80)
        tw_len = Int(300)
        st_range = Float(0.01)
        st_steps = Int(1000)
        list_of_combinations = List(Str)
        all = Bool(False)
        multi_ref_freq = Int(30)
        use_breakpoint = Bool(False)
        breakpoint = Date()
        suffix = Str('')
        old_style = Bool(False)
        parallel = Bool(True)
    
        trait_view = View(Tabbed(VGroup(Item('base_dir'),
                                        Item('save_dir'),
                                        # Item('corr_dir'),
                                        Item('old_style'),
                                        Item('suffix',
                                             label='filename suffix',
                                             enabled_when='not old_style'),
                                        Item('parallel'),
                                        label='Directories'),
                                 VGroup(Item('fmax'),
                                        Item('fmin'),
                                        Item('order'),
                                        Item('fs'),
                                        Item('smooth_win'),
                                        label='Processing'),
                                 Tabbed(Item('tw_start'),
                                        Item('tw_len'),
                                        Item('center_win_len'),
                                        label='Time lag'),
                                 VGroup(Item('st_range'),
                                        Item('st_steps'),
                                        label='Stretch'),
                                 VGroup(Item('all'),
                                        Item('list_of_combinations', \
                                             height=100, \
                                             enabled_when='all==False'),
                                        label='combinations')),
                                 VGroup(Item('multi_ref_freq'),
                                        Item('use_breakpoint'),
                                        Item('breakpoint',
                                             enabled_when='use_breakpoint'),
                                        label='Multi-ref'))
    
#----------------------------------------------------------------------------#


def create_plots_macro(panel_load_dir,
                       save_dir,
                       list_of_combinations,
                       start_day,
                       stop_day,
                       all=False,
                       marker_day=None):
    """ Velocity change and related correlation curve images creation.

    This is a basic tool to create figures for the velocity change and the
    related correlation curve. It is general enough but in case of need must
    be modified by hand.

    :type panel_load_dir: string
    :param panel_load_dir: Where the :class:`pandas.Panel' objects are located
        (those one created with the function `from_single_pattern_to_panel`)
    :type save_dir: string
    :param save_dir: Directory where to save the resulting images
    :type list_of_combinations: list of str
    :param list_of_combinations: The list of combinations that we want to plot
        (e.g. ['QC01-QC02','QC05-QC11']
    :type all: bool
    :param all: If true all possible combinations are used
    :type start_day: :class:`~datetime.Date`
    :param start_day: Starting day
    :type stop_day: :class:`~datetime.Date`
    :param stop_day: Stopping day
    :type marker_day: :class:`~datetime.Date`
    :param marker_day: If passed, a vertical red line will be plotted
        at this date.
    """

    if not os.path.isdir(save_dir):
        print "`save_dir` doesn't exist ..... creating"
        os.mkdir(save_dir)

    # Load the data
    dcs_dict = load_pickled_Series_DataFrame_Panel(
                        os.path.join(panel_load_dir, 'dcs_dict.pickle'))

    pp = dcs_dict['dvP']
    qq = dcs_dict['corrP']

    # pp = load(os.path.join(panel_load_dir, 'dv.pickle'))
    # qq = load(os.path.join(panel_load_dir, 'corr.pickle'))

    # Create the plots

    for (labv, labc) in zip(pp, qq):

        dfv = pp[labv].ix[start_day:stop_day]
        dfc = qq[labc].ix[start_day:stop_day]

        plt.figure(figsize=(8, 6), dpi=150)
        # Summit
        if all:
            dfv.plot(lw=2)
        else:
            dfv.ix[:, list_of_combinations].plot(lw=2)
        # plt.ylim([-0.01, 0.01])
        plt.legend(loc='lower left')
        plt.ylabel('dv/v')
        plt.title('Velocity change')
        if marker_day is not None:
            plt.axvline(x=marker_day, lw=2, color='red')
        plt.savefig(os.path.join(save_dir, 'dv_' + labv + '.png'))

        plt.figure(figsize=(8, 6), dpi=150)
        if all:
            dfv.mean(axis=1).plot(lw=2)
        else:
            dfv.ix[:, list_of_combinations].mean(axis=1).plot(lw=2)
        # plt.ylim([-0.01, 0.01])
        plt.legend(loc='lower left')
        plt.ylabel('dv/v')
        plt.title('Velocity change')
        if marker_day is not None:
            plt.axvline(x=marker_day, lw=2, color='red')
        plt.savefig(os.path.join(save_dir, 'dv_' + labv + '_mean' + '.png'))

        plt.figure(figsize=(8, 6), dpi=150)
        if all:
            dfc.plot(lw=2)
        else:
            dfc.ix[:, list_of_combinations].plot(lw=2)
        plt.legend(loc='lower left')
        plt.ylabel('Corr.')
        plt.title('Cross-Correlation')
        if marker_day is not None:
            plt.axvline(x=marker_day, lw=2, color='red')
        plt.savefig(os.path.join(save_dir, 'corr_' + labc + '.png'))

        plt.figure(figsize=(8, 6), dpi=150)
        if all:
            dfc.mean(axis=1).plot(lw=2)
        else:
            dfc.ix[:, list_of_combinations].mean(axis=1).plot(lw=2)
        plt.legend(loc='lower left')
        plt.ylabel('Corr.')
        plt.title('Cross-Correlation')
        if marker_day is not None:
            plt.axvline(x=marker_day, lw=2, color='red')
        plt.savefig(os.path.join(save_dir, 'corr_' + labc + '_mean' + '.png'))


if BC_UI:
    class _create_plots_macro_view(HasTraits):
        panel_load_dir = Directory()
        save_dir = Directory()
        list_of_combinations = List(Str)
        all = Bool(False)
        marker_day = None
    
        trait_view = View(Item('panel_load_dir'),
                          Item('save_dir'),
                          Item('all'),
                          Item('list_of_combinations', height=100, \
                               enabled_when='all==False'))
    
        def _all_changed(self, old, new):
            if old == True:
                self.list_of_combinations = []
    
    
def one_step_vchange_estimate_and_plot(base_dir, \
                     suffix, \
                     save_dir, \
                     # corr_dir, \
                     center_win_len=1500, \
                     fmax=1.0, \
                     fmin=0.5, \
                     order=4, \
                     fs=10.0, \
                     smooth_win=5, \
                     tw_start_list=[40, 60, 80], \
                     tw_len=[100, 100, 100], \
                     st_range=0.01, \
                     st_steps=1000, \
                     list_of_combinations=[], \
                     start_day='03/04/2010', \
                     stop_day='03/04/2011', \
                     all=False, \
                     marker_day=None, \
                     old_style=False):
    """ One step function to do the velocity change estimation and plot

    This function wants to group together everything is necessary to move
    from the single auto-cross correlation functions to the velocity change
    estimation and plot in one step.
    It collates together the three ``macro`` functions ``vchange_estimate``,
    ``from_single_pattern_to_panel`` and ``create_plots_macro`` plus the
    creation of a standard directory structure to hold all the produced
    results in a consistent way plus a configuration file that stores all
    the parameters that have been adopted to produce them.

    :type base_dir: directory
    :param base_dir: Where the auto-cross correlation matrix are stored
    :type save_dir: directory
    :param save_dir: Where all results will be saved (see Notes for details)
    :type center_win_len: int
    :param center_win_len: How many samples will be taken from the central part
        of the corr func
    :type fmax: float
    :param fmax: Maximum freq for the bandpass filter
    :type fmin: float
    :param fmin: Minimum freq for the bandpass filter
    :type order: int
    :param order: Filter order (i.e. corners in obspy terminology)
    :type fs: float
    :param fs: Sampling frequency for the corr trace
    :type smooth_win: int
    :param smooth_win: Width of the smoothing window
    :type tw_start_list: list of int
    :param tw_start_list: List of time lag (in samples) that we want to adopt
        in the velocity change evaluation
    :type tw_len: list of int
    :param tw_len: Time window length in samples for each time lag
    :type st_range: float
    :param st_range: Stretching range (i.e. 0.01 -> 1%)
    :type st_steps: int
    :param st_steps: How many stretching steps in the specified range
    :type list_of_combinations: list of string
    :param list_of_combinations: The list of combinations that we want to plot
        (e.g. ['QC01-QC02','QC05-QC11']
    :type all: bool
    :param all: If true all available combinations are ploted
    :type start_day: `~datetime.date` object
    :param start_day: Starting day
    :type stop_day: `~datetime.date` object
    :param stop_day: Stoplting day
    :type marker_day: `~datetime.date` object
    :param marker_day: If passed a vertical red line will be plotted
        at this date.
    :type suffix: string
    :param suffix: Optional suffix for the filename to look for (it must be the
        same as used with the :class:`~miic.core.macro.vchange_estimate`
        function through with the corr ad dv/v curves were created)
    :type old_style: bool
    :param old_style: If true, the suffic is generated using the `fs`
        ( "_<fs>Hz") instead of being passed as a parameter)

    .. rubric:: Notes

    The directories structure that will be created under ``save_dir`` has this
    layout:

    ``save_dir``
        |------> panels
        |------> img
        |``vchange_files``

    ``panels`` stores the :class:`~pandas.Panel` objects
    ``img`` stores all produced images
    In ``save_dir`` is also stored a configuration.txt file that contains all
    the parameters in the currunt setup in the for of key->value pairs.
    """

    CONFIG_FILENAME = 'configuration.txt'

    # Do some error checking
    if save_dir == "":
        raise IOError("Empty save path.")
    elif not os.path.isdir(save_dir):
        raise IOError("Save path %s doesn't point to a directory" % save_dir)

    # Create the directory structure
    panels_dir = os.path.join(save_dir, 'panels')
    if not os.path.isdir(panels_dir):
        os.mkdir(panels_dir)
    else:
        print("Warning: `panels` directory already exist")

    img_dir = os.path.join(save_dir, 'img')
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    else:
        print("Warning: `img` directory already exist")

    dataset = {}
    dataset['corr_matrix_folder'] = base_dir
    # dataset['correlation_functions_folder'] = corr_dir

    # Save the parameters values
    vchange_filtering = {}
    vchange_filtering['fmax'] = fmax
    vchange_filtering['fmin'] = fmin
    vchange_filtering['order'] = order
    vchange_filtering['fs'] = fs
    vchange_filtering['smooth_win'] = smooth_win

    vchange_stretch = {}
    vchange_stretch['time_windows_start_list'] = tw_start_list
    vchange_stretch['time_windows_len_list'] = tw_len
    vchange_stretch['stretching_range'] = st_range
    vchange_stretch['stretching_steps'] = st_steps

    vchange_combinations = {}
    vchange_combinations['all_combinations'] = all
    vchange_combinations['list_of_combinations'] = list_of_combinations

    vchange_plot = {}
    vchange_plot['start_day'] = start_day
    vchange_plot['stop_day'] = stop_day
    vchange_plot['marker_day'] = marker_day

    config = ConfigObj()

    config.filename = os.path.join(save_dir, CONFIG_FILENAME)
    config['dataset'] = dataset
    config['combinations'] = vchange_combinations
    config['filtering'] = vchange_filtering
    config['stretching'] = vchange_stretch
    config['plot'] = vchange_plot

    # Write the configuration to file
    config.write()

    vchange_estimate(base_dir, \
                     suffix, \
                     save_dir, \
                     # corr_dir, \
                     center_win_len, \
                     fmax, \
                     fmin, \
                     order, \
                     fs, \
                     smooth_win, \
                     tw_start_list, \
                     tw_len, \
                     st_range, \
                     st_steps, \
                     list_of_combinations, \
                     all, \
                     old_style)

    from_single_pattern_to_panel(load_dir=save_dir, \
                                 save_dir=panels_dir, \
                                 fs=fs, suffix=suffix, old_style=old_style)

    create_plots_macro(panels_dir, \
                       img_dir, \
                       list_of_combinations, \
                       start_day, stop_day, \
                       all=all, \
                       marker_day=marker_day)


if BC_UI:
    class _one_step_vchange_estimate_and_plot_view(HasTraits):
    
        base_dir = Directory()
        save_dir = Directory()
        # corr_dir = Directory()
        old_style = Bool(False)
        suffix = Str('')
        center_win_len = Int(500)
        fmax = Float(1.0)
        fmin = Float(0.5)
        order = Int(4)
        fs = Float(10.0)
        smooth_win = Int(5)
        tw_start_list = List(Int)
        tw_len = List(Int(100))
        st_range = Float(0.01)
        st_steps = Int(1000)
        list_of_combinations = List(Str)
        all = Bool(False)
        start_day = Date()
        stop_day = Date()
        marker_day = Date()
    
        trait_view = View(Tabbed(VGroup(Item('base_dir'),
                                        Item('save_dir'),
                                        # Item('corr_dir'),
                                        Item('old_style'),
                                        Item('suffix',
                                             label='filename suffix',
                                             enabled_when='not old_style'),
                                        label='Directories'),
                                 VGroup(Item('all'),
                                        Item('list_of_combinations', height=100, \
                                             enabled_when='all==False'),
                                        label='Combinations'),
                                 VGroup(Item('fmax'),
                                        Item('fmin'),
                                        Item('order'),
                                        Item('fs'),
                                        Item('smooth_win'),
                                        label='Processing'),
                                 Tabbed(Item('tw_start_list', height=100, \
                                             width=90),
                                        Item('tw_len', height=100, width=90),
                                        Item('center_win_len'),
                                        label='Time lag'),
                                 VGroup(Item('st_range'),
                                        Item('st_steps'),
                                        label='Stretch'),
                                 VGroup(Item('start_day'),
                                        Item('stop_day'),
                                        Item('marker_day'),
                                        label='Plot')))
    
        def _all_changed(self, old, new):
            if old == True:
                self.list_of_combinations = []
    
    
def plot_corr_matrix(base_dir='.', \
                     save_dir='./img', \
                     center_win_len=500, \
                     axis=1, \
                     plot_selected_curves=False, \
                     selected_curves_idx_list=[]):
    """ Plot all correlation matrixes stored in a folder as .mat file

    This is a basic function to plot the correlation matrixes stored in a
    folder as .mat files.
    It loads each matrix, check if time information are available in the same
    file and then creates a ``contourf`` plot with 20 levels.
    It also add simple lables to each axis and a generic title "Correlation
    Matrix".
    It is also possible to generate, at the same time, plots of a list of
    interesting correlation functions just setting
    ``plot_selected_curves==True`` and passing the list of indexes in the
    parameter ``selected_curves_idx_list``.

    :type base_dir: directory
    :param base_dir: Where the auto-cross correlation matrix are stored
    :type save_dir: directory
    :param save_dir: Directory where to save the resulting images
    :type center_win_len: int
    :param center_win_len: How many samples will be taken from the central part
        of the correlation function
    :type axis: int
    :param axis: On which axis apltly the selection
        (axis==1 cols, axis==1 rows)
    :type plot_selected_curves: bool
    :param plot_selected_curves: If it is requested to plot single correlation
        functions
    :type selected_curves_idx_list: list of int
    :param selected_curves_idx_list: The list of index (in the selected axis)
        of the curves that will be ploted
    """

    f_pattern = '*.mat'
    files_list1 = dir_read(base_dir=base_dir, pattern=f_pattern, \
                           sort_flag=True)

    # create the location for saving the figures
    if files_list1 and not os.path.isdir(save_dir):
            os.mkdir(save_dir)

    for filename in files_list1:
        # load the correlation matrix
        aa = mat_to_ndarray(os.path.join(base_dir, filename), flatten=True)

        # create the name of the file to save
        res = os.path.basename(filename).split(os.path.extsep)
        base_new_name = os.path.extsep.join([spiece for (i, spiece) in\
                                             enumerate(res)\
                                             if i < len(res) - 1])
        figure_file_name = os.path.join(save_dir, base_new_name)

        # call the function to create and save the figure
        plot_single_corr_matrix(aa, center_win_len=center_win_len, \
                                filename=figure_file_name)

        # go ahead with plotting selected correlation functions
        if plot_selected_curves:
            X = aa['corr_data']

            if 'time' in aa:
                time = convert_time(aa['time'])

            X = nd_mat_center_part(X, center_win_len, axis=1)

            tlag = np.arange(-np.floor(X.shape[1] / 2), X.shape[1] - \
                         np.floor(X.shape[1] / 2))
            if axis == 1:
                if X.shape[0] < max(selected_curves_idx_list):
                    print('One or more indexs out of range. Skip')
                    continue
            else:
                if X.shape[1] < max(selected_curves_idx_list):
                    print('One or more indexs out of range. Skip')
                    continue
            for idx in selected_curves_idx_list:
                plt.figure(figsize=(7, 9), dpi=150)
                if axis == 1:
                    plt.plot(tlag, X[idx])
                    plt.ylabel('Corr. value')
                    plt.xlabel('Time lag')
                else:
                    plt.plot(tlag, X[:, idx])
                    plt.xlabel('Corr. value')
                    plt.ylabel('Time lag')
                if 'time' in aa:
                    plt.legend(['Day ' + time[idx].isoformat()])
                    sidx = time[idx].isoformat()
                else:
                    plt.legend(['Day ' + idx])
                    sidx = str(idx)

                plt.title('Correlation Function')
                ax = plt.gca()
                ax.grid(True)
                plt.savefig(os.path.join(save_dir, base_new_name + \
                                         '_corrFunctDay_' + sidx + '.png'))


if BC_UI:
    class _plot_corr_matrix_view(HasTraits):
    
        base_dir = Directory()
        save_dir = Directory()
        center_win_len = Int(500)
        axis = Enum(1, 0)
        plot_selected_curves = Bool(False)
        selected_curves_idx_list = List(Int)
    
        trait_view = View(Tabbed(VGroup(Item('base_dir'),
                                        Item('save_dir'),
                                        label='Dirs'),
                                 VGroup(Item('center_win_len'),
                                        Item('axis'),
                                        label='Selection'),
                                 VGroup(Item('plot_selected_curves'),
                                        Item('selected_curves_idx_list', \
                                             height=100, \
                                             width=90, \
                                             label='Selected idxs'),
                                        label='Single curves')))
    

def plot_dvs_macro(base_dir='.', save_dir='./img'):
    """ Plot all correlation matrixes stored in a folder as .mat file

    This is a basic function to plot the velocity change dv dictionaries
    stored in a folder as .mat files.
    It loads each dictionary, check if their content is in agreement with the
    presumed structure and the plot it.

    :type base_dir: directory
    :param base_dir: Where the auto-cross correlation matrix are stored
    :type save_dir: directory
    :param save_dir: Directory where to save the resulting images
    """

    f_pattern = '*.mat'
    files_list1 = dir_read(base_dir=base_dir, pattern=f_pattern, \
                           sort_flag=True)

    # create the location for saving the figures
    if files_list1 and not os.path.isdir(save_dir):
            os.mkdir(save_dir)

    for filename in files_list1:
        # load the correlation matrix
        dv = mat_to_ndarray(os.path.join(base_dir, filename))

        check_staus = dv_check(dv)

        if check_staus['is_incomplete']:
            continue

        # create the name of the file to save
        res = os.path.basename(filename).split(os.path.extsep)
        base_new_name = os.path.extsep.join([spiece for (i, spiece) in\
                                             enumerate(res)\
                                             if i < len(res) - 1])
        figure_file_name = os.path.join(save_dir, base_new_name)

        # call the function to create and save the figure
        plot_dv(dv, save_dir=save_dir, figure_file_name=figure_file_name)


class _SpeedUpPlot:

    def __init__(self, save_dir, R, time, xpt, ypt, xgrid, ygrid, edges, \
                 the_levels, m):
        self.save_dir = save_dir
        self.R = R
        self.time = time
        self.xpt = xpt
        self.ypt = ypt
        self.xgrid = xgrid
        self.ygrid = ygrid
        self.edges = edges
        self.the_levels = the_levels
        self.m = m

    def __call__(self, day):
        strday = self.time[day].isoformat()

        fig = plt.figure(figsize=(8, 6), dpi=150)
        zpt = self.R[day, :]
        z_s = plt.mlab.griddata(self.xpt, self.ypt, zpt, self.xgrid, \
                                self.ygrid, interp='nn')
        self.m.plot(self.xpt[self.edges.T], self.ypt[self.edges.T], 'w--', \
                    marker='o', markerfacecolor='r', markeredgecolor='k')
        self.m.contourf(self.xgrid, self.ygrid, z_s, self.the_levels, \
                        zorder=10)

        ax = plt.gca()
        ax.annotate(strday, xy=(0.05, 0.04), xycoords='figure fraction')

        # plt.axis('equal')
        plt.title("Velocity Change")
        plt.jet()
        # plt.gca().set_xlim(55.65, 55.782)
        # plt.setp(plt.gca(), 'yticklabels', [])
        # plt.setp(plt.gca(), 'xticklabels', [])
        # plt.gcf().canvas.draw()
        plt.colorbar(format='%2.3e')

        filename = strday + '_dv_tomo.png'
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close(fig)


def real_vchanges_tomograpy(save_dir, lat_lon_alt_filename, \
                            real_velocity_change_filename, inter_points):
    """Simple tomography obtained using also matplotlib BaseMap (experimental)
    """

    from mpl_toolkits.basemap import Basemap
    import re

    df_geo_data = lat_lon_ele_load(lat_lon_alt_filename)

    R = load_pickled_Series_DataFrame_Panel(real_velocity_change_filename)
    R[R > 0.2] = np.nan

    R = -1 * R  # To conform to the conventional colors adopted for seismic
                # tomography

    # Select just the geo data related to the statons that are included in
    # the R DataFrame
    pattern = '|'.join(R.columns)
    df_geo_data = df_geo_data.select(lambda x: re.search(pattern, str(x)) \
                                     is not None)

    stations_in_order = [x.split('.')[1:2][0] \
                         for x in df_geo_data.index.values]

    # Rearrange R to the same order as the geo informations
    R = R.reindex_axis(stations_in_order, axis=1, copy=False)

    time = R.index

    R_min = R.min().min()
    R_max = R.max().max()

    # Plot parameters

    if R_max > 0 and R_min > 0:

        the_levels = np.logspace(np.log10(R_min), np.log10(R_max), 200)

    elif R_max > 0 and R_min < 0:

        the_levels1 = np.logspace(-4, np.log10(R_max), 200)
        the_levels2 = np.logspace(-4, np.log10(abs(R_min)), 200)

        the_levels = np.append(-1 * the_levels2[::-1][:-1], the_levels1)

    elif R_max < 0 and R_min < 0:

        the_levels = np.logspace(np.log10(abs(R_max)), \
                                 np.log10(abs(R_min)), 200)
        the_levels = -1 * the_levels[:-1:]

    # the_levels = np.log10(np.logspace(R_min, R_max,100))
    #    the_levels=np.log10(np.logspace(R_min, R_max,100))
    #    the_colors = np.linspace(.21, 1.0, 20)
    #    the_colors = str(the_colors).split()[1:-1]

    # Create the Basemap object
    m = Basemap(llcrnrlon=55.6, llcrnrlat= -21.4, urcrnrlon=55.85, \
                urcrnrlat= -21.18, projection='lcc', resolution='f', \
                lat_1= -21., lat_2= -21.5, lat_0= -21.25, lon_0=55.7)

    parallels = np.linspace(-22., -21., 10)
    meridians = np.linspace(55.45, 55.9, 10)

    xpt, ypt = m(df_geo_data['latitude'], df_geo_data['longitude'])
    _, edges, _, _ = delaunay.delaunay(xpt, ypt)

    xg = np.linspace(np.nanmin(xpt), np.nanmax(xpt), inter_points)
    yg = np.linspace(np.nanmin(ypt), np.nanmax(ypt), inter_points)
    xgrid, ygrid = np.meshgrid(xg, yg)

    for day in time:

        zpt = R.xs(day).values.astype('float64')

        mask = ~np.isnan(zpt)

        if not np.all(mask == False):

            fig = plt.figure(figsize=(12, 8), dpi=150)

            ax = fig.add_subplot(111)
            m.ax = ax

            strday = day.isoformat()

            m.drawcoastlines(linewidth=2, color='red')
            m.drawparallels(parallels, labels=[1, 0, 0, 0])
            m.drawmeridians(meridians, labels=[0, 0, 0, 1])

            z_s = plt.mlab.griddata(xpt[mask], ypt[mask], zpt[mask],
                                    xgrid, ygrid, interp='nn')

            m.plot(xpt[edges.T], ypt[edges.T], '--', color='0.8',
                   marker='v', markerfacecolor='r', markeredgewidth=1,
                   markeredgecolor='r', markersize=3, zorder=20)

            xx = m.contourf(xgrid, ygrid, z_s, the_levels, zorder=10)

            ax = plt.gca()
            ax.annotate(strday, xy=(0.05, 0.04), xycoords='figure fraction',
                        fontsize='xx-large', fontweight='bold')

            plt.title("Velocity Change")
            plt.jet()
            fig.colorbar(xx, format='%5.4f',
                         ticks=np.linspace(R_min, R_max, 10),
                         shrink=0.8)

            filename = strday + '_dv_tomo.png'
            fig.savefig(os.path.join(save_dir, filename))
            plt.close(fig)


if BC_UI:
    class _real_vchanges_tomograpy_view(HasTraits):
        save_dir = Directory()
        lat_lon_alt_filename = File()
        real_velocity_change_filename = File()
        inter_points = Int(1500)
    
        traits_view = View(Item('save_dir'),
                           Item('lat_lon_alt_filename'),
                           Item('real_velocity_change_filename'),
                           Item('inter_points'))
        

# EOF
