# -*- coding: utf-8 -*-
"""
@author:
Eraldo Pomponi

@copyright:
The MIIC Development Team (eraldo.pomponi@uni-leipzig.de)

@license:
GNU Lesser General Public License, Version 3
(http://www.gnu.org/copyleft/lesser.html)

Created on Apr 2, 2012
"""
# Main imports
import numpy as np
from multiprocessing import Pool
from copy import deepcopy
import datetime
import glob
import re
import os


# ETS imports
#try:
#    BC_UI = True
#    from traits.api import HasTraits, File, Int, Bool, Enum, Button, \
#        Float, Dict
#    from traitsui.api import View, Item, Include, HGroup, VGroup, Tabbed
#except ImportError:
BC_UI = False
#    pass
    
# Obspy import
from obspy import read
from obspy.core.stream import Stream
from obspy.core.trace import Trace
from obspy.core import Stats
from obspy.core import UTCDateTime
from obspy.signal.filter import envelope

# local imports
from miic.core.miic_utils import convert_time


class InputError(Exception):
    """
    Exception for Input errors.
    """
    def __init__(self, msg):
        Exception.__init__(self, msg)


def stream_combine(st1, st2):
    """ Combins two streams into a single one """

    if not isinstance(st1, Stream) or not isinstance(st2, Stream):
        raise InputError("'st1' ans 'st2' must be a 'obspy.core.stream.Stream'\
             objects")

    for tr in st2:
        st1.append(tr)

    return st1


if BC_UI:
    class _stream_combine_view(HasTraits):
        trait_view = View()


def stream_add_lat_lon_ele(st, df):
    """ Add lat lon ele information to each trace if of a stream, if available.

    This function checks all the traces in the stream ``st`` and if their name
    is available in the DataFrame structure ``df``, the corresponding lat, lon
    and ele are included in the tr.stats dictionary.

    :type st: :class:`~obspy.core.stream.Stream`
    :param st: The Stream to which add geo informations.
    :type df: :class:`~pandas.DataFrame`
    :param df: The dataframe that has as index the stations name and 3 columns
        lat, lon and ele.

    :rtype: :class:`~obspy.core.stream.Stream`
    :return: **st_with_geo**: Stream with trace' geo information updated.
    """

    if not isinstance(st, Stream):
        raise InputError("'st' must be a 'obspy.core.stream.Stream' objects")

    for tr in st:
        selector = _Selector(tr.id)
        tr_geo_info = df.select(selector, axis=0)
        if tr_geo_info.index.size > 0:
            if 'sac' not in tr.stats:
                tr.stats['sac'] = {}
                tr.stats.sac['stla'] = tr_geo_info['latitude'][0]
                tr.stats.sac['stlo'] = tr_geo_info['longitude'][0]
                tr.stats.sac['stel'] = tr_geo_info['elevation'][0]
            else:
                tr.stats.sac.stla = tr_geo_info['latitude'][0]
                tr.stats.sac.stlo = tr_geo_info['longitude'][0]
                tr.stats.sac.stel = tr_geo_info['elevation'][0]

    st_with_geo = st
    return st_with_geo


class _Selector:

    def __init__(self, sid):
        self.tr_network, self.tr_station, self.tr_location, self.tr_channel = \
            sid.split('.', 4)

    def __call__(self, x):
        network, station, location, channel = x.split('.', 4)

        if not (network == '*' or network == self.tr_network):
            return False

        if not (station == '*' or station == self.tr_station):
            return False

        if not (location == '*' or location == self.tr_location):
            return False

        if not(channel == '*' or channel == self.tr_channel):
            return False

        return True


if BC_UI:
    class _stream_add_lat_lon_ele_view(HasTraits):
        trait_view = View()


def stream_save_sac(st, filename):
    """ Save a stream in sac format """

    if not isinstance(st, Stream):
        raise InputError("'st' must be a 'obspy.core.stream.Stream' objects")

    st.write(filename=filename, format='sac')


if BC_UI:
    class _stream_save_sac_view(HasTraits):
        filename = File
        trait_view = View(Item('filename'))


def corr_trace_to_obspy(corr_trace):
    """ Convert a correlation trace dictionary to an obspy trace.

    Convert a single correlation trace (or a list of) in an
    :class:`~obspy.core.trace.Trace` (or :class:`~obspy.core.stream.Stream`)
    object.

    :type corr_trace: dictionary of type correlation trace or list of these
    :param corr_trace: input date to be converted

    :rtype: :class:`~obspy.core.trace.Trace` if ``corr_trace`` is a dict
        and :class:`~obspy.core.stream.Stream` if ``corr_trace`` is a list of
        dicts
    :return: **st**: the obspy object containing the data
    """

    if isinstance(corr_trace, list):
        st = Stream()
        for tr in corr_trace:
            st.append(_single_corr_trace_to_obspy_trace(tr))
    else:
        st = _single_corr_trace_to_obspy_trace(corr_trace)
    return st


if BC_UI:
    class _corr_trace_to_obspy_view(HasTraits):
        trait_view = View()


def _single_corr_trace_to_obspy_trace(trace):
    """ Convert a correlation trace dictionary to an obspy trace.

    Convert a single correlation trace in an
    :class:`~obspy.core.trace.Trace` object.

    :type corr_trace: dictionary of type correlation trace
    :param corr_trace: input date to be converted

    :rtype: :class:`~obspy.core.trace.Trace`
    :return: **tr**: the obspy object containing the data
    """

    tr = Trace(data=np.squeeze(trace['corr_trace']))
    stats_keys = ['network', 'station', 'location',
                  'channel', 'npts', 'sampling_rate']

    sac_keys = ['baz', 'az', 'stla', 'stlo', 'stel',
                'evla', 'evlo', 'evel', 'dist']

    # copy stats
    for key in stats_keys:
        try:
            tr.stats[key] = trace['stats'][key]
        except:
            print 'Error copying key: %s' % key
            raise
    
    # special keys
    tr.stats['starttime'] = UTCDateTime(
                              convert_time([trace['stats']['starttime']])[0])

    # test for presence of geo information
    flag = 0
    for key in sac_keys:
        if not key in trace['stats']:
            flag += 1
    if flag == 0:  # geo information present
        tr.stats['sac'] = {}
        for key in sac_keys:
            tr.stats['sac'][key] = trace['stats'][key]
            
    # copy stats_tr1
    if 'stats_tr1' in trace:
        tr.stats_tr1 = Stats()
        tr.stats_tr1['starttime'] = UTCDateTime(
                              convert_time([trace['stats_tr1']['starttime']])[0])
        for key in stats_keys:
            try:
                tr.stats_tr1[key] = trace['stats_tr1'][key]
            except:
                print 'Error copying key: %s' % key
                raise
        for key in sac_keys:
            try:
                tr.stats_tr1[key] = trace['stats_tr1'][key]
            except:
                pass

    # copy stats_tr2
    if 'stats_tr2' in trace:
        tr.stats_tr2 = Stats()
        tr.stats_tr2['starttime'] = UTCDateTime(
                              convert_time([trace['stats_tr2']['starttime']])[0])
        for key in stats_keys:
            try:
                tr.stats_tr2[key] = trace['stats_tr2'][key]
            except:
                print 'Error copying key: %s' % key
                raise
        for key in sac_keys:
            try:
                tr.stats_tr2[key] = trace['stats_tr2'][key]
            except:
                pass

    return tr


def _tr_gen(st):
    """ A generator that extract the traces form a Straem object

    :type st: :py:class:`~obspy.core.stream.Stream`
    :param st: Stream where to take the traces
    """
    for _ in st:
        yield st.pop(0)


class _AppendST:

    def __init__(self, st):
        self.st = st

    def __call__(self, trs):
        for tr in trs:
            self.st.append(tr)


class _StDownsample:

    def __init__(self, final_freq, no_filter=False,
                      strict_length=False):
        self.final_freq = final_freq
        self.no_filter = no_filter
        self.strict_length = strict_length

    def __call__(self, tr):

        sampling_rate = tr.stats.sampling_rate
        res = np.mod(sampling_rate, self.final_freq)
        decimation_factor = int(sampling_rate // self.final_freq)
        if res != 0:
            print "downsample factor rounded to integer: %4.2f -> %i !!!!" % \
                (decimation_factor + res, res)

        if decimation_factor > 16:
            print "downsample factor > 16. Bound to 16"
            decimation_factor = 16

        if hasattr(tr, 'decimate'):
            tr.decimate(decimation_factor,
                        self.no_filter,
                        self.strict_length)
        else:
            tr.downsample(decimation_factor,
                          self.no_filter,
                          self.strict_length)
        return tr


def stack_stream(st):
    """Calculate mean of all traces in stream

    Calculate the mean of all overlapping segments of the trace in the input
    stream provided the traces match the sampling rate of the first trace.

    :type st: :class:`~obspy.core.stream.Stream`
    :param st: The stream with the traces to be stacked

    :rtype: :class:`~obspy.core.stream.Stream`
    :return: **rst**: Stream with stacked data in the first trace and number of
        stacked traces for each sample in the second trace
    """
    sampling_rate = st[0].stats.sampling_rate
    starttime = st[0].stats.starttime
    endtime = st[0].stats.endtime
    for tr in st:
        if tr.stats.starttime < starttime:
            starttime = tr.stats.starttime
        if tr.stats.endtime > endtime:
            endtime = tr.stats.endtime
    npts = (endtime-starttime)*sampling_rate + 1
    sum_tr = deepcopy(st[0])
    sum_tr.stats.starttime = starttime
    sum_tr.data = np.zeros(npts,dtype=float)
    sum_tr.stats.station = 'MEAN'
    num_tr = deepcopy(sum_tr)
    num_tr.stats.station= 'NUM'
    for tr in st[1:]:
        if tr.stats.sampling_rate == sampling_rate:
            sind = int(np.floor((tr.stats.starttime-starttime) * sampling_rate))
            tnpts = tr.stats.npts
            sum_tr.data[sind:sind+tnpts] += tr.data
            num_tr.data[sind:sind+tnpts] += 1
    rst = stream.Stream()
    sum_tr.data /= num_tr.data
    rst.append(sum_tr)
    rst.append(num_tr)
    return rst
                   
    



def stream_downsample(st, final_freq, no_filter=False, \
                      strict_length=False, parallel=True, processes=None):
    """Downsample all the traces in the input stream to a desired frequency.

    This function downsample all the traces in the input stream ``st`` to a
    desired frequency. The decimation_factor is rounded to the closest integer
    in case of a rational value.
    It is also possible to decide if the anti-aliasing filter is applied or not
    before downsampling through the ``no_filter`` flag.
    If the length of the data array modulo decimation_factor is not zero then
    the endtime of the trace is changing on sub-sample scale. To abort
    downsampling in case of changing endtimes set strict_length=True.

    .. Note::

    This operation is performed in place on the actual data arrays. The
    raw data is not accessible anymore afterwards. To keep your
    original data, use :func:`~miic.core.alpha_mod.stream_copy` to create
    a copy of your stream object.
    This function can also work in parallel an all or a specified number of
    cores available in the hosting machine. It must also be noticed that a
    decimation_factor grater than 16 is not allowed so, in case it is
    effectively higher, it is bounded to 16.

    :type st: :class:`~obspy.core.stream.Stream`
    :param st: The container for the Traces that we want to downsample
    :type final_freq: float
    :param final_freq: Desired final frequency
    :type no_filter: bool, optional
    :param no_filter: If ``True`` no anti-aliasing low-pass filter is applied.
        Defaults to False.
    :type strict_length: bool, optional
    :param strict_length: Leave traces unchanged for which endtime of trace
        would change. Defaults to False. (for more details see:
        :class:`~obspy.core.trace.Trace.decimate`)
    :type parallel: bool (Default: True)
    :pram parallel: If the filtering will be run in parallel or not
    :type processes: int
    :pram processes: Number of processes to start (if None it will be equal
        to the number of cores available in the hosting machine)

    :rtype: :class:`~obspy.core.stream.Stream`
    :return: **st_down**: Downsampled Stream.
    """
    if not isinstance(st, Stream):
        raise InputError("'st' must be a 'obspy.core.stream.Stream' object")

    if not parallel:
        trs = map(_StDownsample(final_freq, no_filter, strict_length), \
                  _tr_gen(st))
        st.extend(trs)
    else:
        if processes == 0:
            processes = None

        p = Pool(processes=processes)

        p.map_async(_StDownsample(final_freq, no_filter, strict_length),
                    _tr_gen(st),
                    callback=_AppendST(st))

        p.close()
        p.join()

    st_down = st
    return st_down


if BC_UI:
    class _stream_downsample_view(HasTraits):
        final_freq = Int(2)
        no_filter = Bool(False)
        strict_length = Bool(False)
        parallel = Bool(True)
        processes = Int(0)
    
        trait_view = View(Item('final_freq'),
                          Item('no_filter', style='custom'),
                          Item('strict_length', style='custom'),
                          Item('parallel'),
                          Item('processes'))



def stream_envelope(st):
    """ Calculate the envelopes of all traces in a stream.

    It works on a copy of the stream.

    :type st: ::class:`~obspy.core.stream.Stream`
    :param st: Stream fo be used for calculating the envelopes.

    :rtype sst: :class:`~obspy.core.stream.Stream`
    :return: **sst**: envelopes of stream
    """
    for tr in st:
        tr.data = envelope(tr.data)
    return st


def stream_stack_distance_intervals(st, interval):
    """ Stack average traces in a stream if their distance difference is
    smaller than interval.

    The stream containing a number of traces with given distance (e.g. from source)
    is used to create a number of equally spaced traces by averaging traces that fall
    into the same distance bin. If interval is a scalar the bins are equally spaced with
    a width of interval. If interval is a sequence its elements define the lower distance
    limit of the bins.

    :type st: :class:`~obspy.core.stream.Stream`
    :param st: Stream fo be used for stacking.
    :type interval: scalar os array like
    :param interval: width of bins in case of scalar or smaller edge of
        bins if interval is a sequence.
    :rtype sst: :class:`~obspy.core.stream.Stream`
    :return: **sst**: stacked stream
    """
    dist = []
    npts = []
    for tr in st:
        dist.append(tr.stats.sac['dist'])
        npts.append(tr.stats['npts'])
    
    if not hasattr(interval, "__len__"):
        bins = np.arange(min(dist), max(dist),interval)
    else:
        bins = np.array(interval)
    
    sst = Stream()
    for ii in bins:
        sst.append(Trace(data=np.zeros(max(npts),dtype=np.float64),header={
            'network':'stack','station':str(ii),'location':'',
            'channel':st[0].stats['channel'],'starttime':st[0].stats['starttime'],'sampling_rate':st[0].stats['sampling_rate'],
            'sac':{'dist':ii,'az':0,'evla':0.,'evlo':0.,'stla':ii/(np.pi*6371000)*180.,'stlo':0.}}))
    count = np.zeros_like(bins)
    for tr in st:
        ind = sum((tr.stats.sac['dist'] - bins)>=0)-1
        sst[ind].data[0:tr.stats['npts']] += tr.data
        count[ind] += 1
    for ind, tr in enumerate(sst):
        tr.data /= count[ind]
    
    return sst


def stream_mute(st, filter=(), mute_method='std_factor', mute_value=3,
                taper_len=5,extend_gaps=True):
    """
    Mute parts of data that exceed a threshold

    To completely surpress the effect of data with high amplitudes e.g. after
    aftershocks these parts of the data are muted (set to zero). The respective
    parts of the signal are identified as those where the envelope in a given
    frequency exceeds a threshold given directly as absolute number or as a
    multiple of the data's standard deviation. A taper of length `taper_len` is
    applied to smooth the edges of muted segments. Setting `extend_gaps` to
    Ture will ensure that the taper is applied outside the segments and data
    inside these segments will all zero. Edges of the data will be tapered too
    in this case.
    The function returns a muted copy of the data.

    :Example:
    ``args={'filter':{'type':'bandpass', 'freqmin':1., 'freqmax':6.},'taper_len':1., 'threshold':1000, 'std_factor':1, 'extend_gaps':True}``

    :type st: obspy.Stream
    :param st: stream with data to be muted
    :type filter: dict
    :param filter: Necessary arguments for the respective filter
        that will be passed on. (e.g. ``freqmin=1.0``, ``freqmax=20.0`` for
        ``"bandpass"``)
    :type mute_method: str
    :param mute_method: either 'std_factor' or 'threshold' that lead to muting
        when the (filtered) envelope exceeds the standard deviation of the data
        by a certain factor (mute_value) or when the envelope exceeds a fixed
        value, respectively.
    :type mute_value: float
    :param mute_value: numerical value corresponding to 'mute_method'
    :type taper_len: float
    :param taper_len: length of taper in seconds
    :type extend_gaps: Bool
    :param extend_gaps: make sure tapering is done outside the segments to be
        muted. This step involves an additional convolution.

    :rtype: obspy.stream
    :return: muted data
    """

    assert type(st) == Stream, "st is not an obspy stream"
    assert mute_method in ['std_factor', 'threshold'], "unsupported mute_type"
    assert type(mute_value) is float or type(mute_value) is int, \
                                    "mute_value is not a number"
    assert type(taper_len) is float or type(taper_len) is int, \
                                    "taper_len is not a number"
    assert type(extend_gaps) is bool, "extend_gaps is not a boolean."
    # copy the data
    cst = st.copy()
    cst.merge()
    cst.detrend('linear')
    cst.taper(max_length=taper_len,max_percentage=0.1)


    for ind,tr in enumerate(cst):
        # return zeros if length of traces is shorter than taper
        ntap = int(taper_len*tr.stats.sampling_rate)
        if tr.stats.npts<=ntap:
            cst[ind].data = np.zeros_like(tr.data)
            continue

        # filter if asked to
        ftr = tr.copy()
        if filter:
            ftr.filter('bandpass', freqmin=filter[0], freqmax=filter[1])

        # calculate envelope
        #D = np.abs(signal.hilbert(C,axis=0))
        D = np.abs(ftr.data)

        # calculate threshold
        if mute_method == 'threshold':
            thres = np.zeros_like(ftr.data) + mute_value
        elif mute_method == 'std_factor':
            thres = np.std(ftr.data) * mute_value

        # calculate mask
        mask = np.ones_like(tr.data)
        mask[D>thres]=0
        # extend the muted segments to make sure the whole segment is zero after
        if extend_gaps:
            assert ntap != 0, "length of taper cannot be zero if extend_gaps is True"
            tap = np.ones(ntap)/ntap
            mask = np.convolve(mask,tap, mode='same')
            nmask = np.ones_like(D)
            nmask[mask<0.999999999] = 0
        else:
            nmask = mask

        # apply taper
        if ntap > 0:
            tap = 2. - (np.cos(np.arange(ntap,dtype=float)/ntap*2.*np.pi) + 1.)
            tap /= ntap
            nmask = np.convolve(nmask,tap, mode='same')

        # mute date with tapered mask
        cst[ind].data *= nmask
    return cst


class _StFilter:

    def __init__(self, ftype, param):
        self.ftype = ftype
        self.param = param

    def __call__(self, tr):
        tr.filter(self.ftype, **self.param)
        return tr


def stream_filter(st, ftype, filter_option, parallel=True, processes=None):
    """ Filter each trace of a Stream according to the given parameters

    This faction apply the specified filter function to all the traces in the
    present in the input :py:class:`~obspy.core.stream.Stream`.

    .. Note::

    This operation is performed in place on the actual data arrays. The
    raw data is not accessible anymore afterwards. To keep your
    original data, use :func:`~miic.core.alpha_mod.stream_copy` to create
    a copy of your stream object.
    This function can also work in parallel an all or a specified number of
    cores available in the hosting machine.

    :type ftype: str
    :param ftype: String that specifies which filter is applied (e.g.
            ``"bandpass"``). See the `Supported Filter`_ section below for
            further details.
    :type filter_option: dict
    :param filter_option: Necessary arguments for the respective filter
        that will be passed on. (e.g. ``freqmin=1.0``, ``freqmax=20.0`` for
        ``"bandpass"``)
    :type parallel: bool (Default: True)
    :pram parallel: If the filtering will be run in parallel or not
    :type processes: int
    :pram processes: Number of processes to start (if None it will be equal
        to the number of cores available in the hosting machine)

    .. rubric:: _`Supported Filter`

    ``'bandpass'``
        Butterworth-Bandpass (uses :func:`obspy.signal.filter.bandpass`).

    ``'bandstop'``
        Butterworth-Bandstop (uses :func:`obspy.signal.filter.bandstop`).

    ``'lowpass'``
        Butterworth-Lowpass (uses :func:`obspy.signal.filter.lowpass`).

    ``'highpass'``
        Butterworth-Highpass (uses :func:`obspy.signal.filter.highpass`).

    ``'lowpassCheby2'``
        Cheby2-Lowpass (uses :func:`obspy.signal.filter.lowpassCheby2`).

    ``'lowpassFIR'`` (experimental)
        FIR-Lowpass (uses :func:`obspy.signal.filter.lowpassFIR`).

    ``'remezFIR'`` (experimental)
        Minimax optimal bandpass using Remez algorithm (uses
        :func:`obspy.signal.filter.remezFIR`).

    """
    if not isinstance(st, Stream):
        raise InputError("'st' must be a 'obspy.core.stream.Stream' object")

    fparam = dict([(kw_filed, filter_option[kw_filed]) \
                for kw_filed in filter_option])

    if not parallel:
        st.filter(ftype, **fparam)
    else:
        if processes == 0:
            processes = None

        p = Pool(processes=processes)

        p.map_async(_StFilter(ftype, fparam),
                    _tr_gen(st),
                    callback=_AppendST(st))
        p.close()
        p.join()

    # Change the name to help blockcanvas readability
    st_filtered = st
    return st_filtered


if BC_UI:
    class _stream_filter_view(HasTraits):
    
        freq_l = Float(1.0)
        df_l = Float(1.0)
        corners_l = Int(4)
        zerophase_l = Bool(False)
    
        freq_h = Float(1.0)
        df_h = Float(1.0)
        corners_h = Int(4)
        zerophase_h = Bool(False)
    
        freqmin_bp = Float(1.0)
        freqmax_bp = Float(10.0)
        df_bp = Float(1.0)
        corners_bp = Int(4)
        zerophase_bp = Bool(False)
    
        freqmin_bs = Float(1.0)
        freqmax_bs = Float(10.0)
        df_bs = Float(1.0)
        corners_bs = Int(4)
        zerophase_bs = Bool(False)
    
        filter_option = Dict
    
        save_b = Button('Save')
    
        ftype = Enum('lowpass', 'highpass', 'bandpass', 'bandstop')
    
        parallel = Bool(True)
        processes = Int(0)
    
        trait_view = View(Tabbed(VGroup(
                          Item('ftype'),
                          Item('parallel'),
                          Item('processes'),
                          Item('filter_option', style='readonly')),
                          Include('lowpass_view'),
                          Include('highpass_view'),
                          Include('bandpass_view'),
                          Include('bandstop_view')))
    
        lowpass_view = HGroup(Item('freq_l', label='f max'),
                              Item('corners_l', label='corners'),
                              Item('zerophase_l', label='zerophase'),
                              Item('save_b', show_label=False),
                              label='lowpass',
                              enabled_when="ftype='lowpass'")
    
        highpass_view = HGroup(Item('freq_h', label='f min'),
                              Item('corners_h', label='corners'),
                              Item('zerophase_h', label='zerophase'),
                              Item('save_b', show_label=False),
                              label='highpass',
                              enabled_when="ftype='highpass'")
    
        bandpass_view = HGroup(Item('freqmin_bp', label='f min'),
                              Item('freqmax_bp', label='f max'),
                              Item('corners_bp', label='corners'),
                              Item('zerophase_bp', label='zerophase'),
                              Item('save_b', show_label=False),
                              label='bandpass',
                              enabled_when="ftype='bandpass'")
    
        bandstop_view = HGroup(Item('freqmin_bs', label='f min'),
                              Item('freqmax_bs', label='f max'),
                              Item('corners_bs', label='corners'),
                              Item('zerophase_bs', label='zerophase'),
                              Item('save_b', show_label=False),
                              label='bandstop',
                              enabled_when="ftype='bandstop'")
    
        def _save_b_fired(self):
    
            if self.ftype == 'lowpass':
                self.filter_option = {'freq': self.freq_l,
                                      'corners': self.corners_l,
                                      'zerophase': self.zerophase_l}
            elif self.ftype == 'highpass':
                self.filter_option = {'freq': self.freq_h,
                                      'corners': self.corners_h,
                                      'zerophase': self.zerophase_h}
            elif self.ftype == 'bandpass':
                self.filter_option = {'freqmin': self.freqmin_bp,
                                      'freqmax': self.freqmax_bp,
                                      'corners': self.corners_bp,
                                      'zerophase': self.zerophase_bp}
            elif self.ftype == 'bandstop':
                self.filter_option = {'freqmin': self.freqmin_bs,
                                      'freqmax': self.freqmax_bs,
                                      'corners': self.corners_bs,
                                      'zerophase': self.zerophase_bs}
    
    
def trace_sym_pad_shrink_to_npts(tr, npts):
    """ Pads with 0 or shrink the trace data symmetrically to npts points.

    This functions pads or shrinks, respectively if the orignal verctor is
    shorten or longer repsect to the request npts, the trace.data vector
    to npts points respect to its center. The padding value is 0 and both the
    operation of padding shrinking are done symmetrically respect to the
    central point.

    :type tr: :class:`~obspy.core.trace.Trace`
    :param tr: Trace to be padded
    :type npts: int
    :param npts: Length at which pad

    :rtype tr_padded: :class:`~obspy.core.trace.Trace`
    :return: **tr_padded**: Pointer to the original trace but with data padded
    """
    lp = npts - tr.stats.npts
    # print "lp: %d" % lp

    if lp == 0:
        pass
    elif lp > 0:
        left = np.ceil(lp / 2)
        right = lp - left
        cdata = np.append(np.zeros(left, dtype=tr.data.dtype), tr.data)
        tr.data = np.append(cdata, np.zeros(right, dtype=tr.data.dtype))
        tr.stats['starttime'] -= float(left) * tr.stats['delta']
    else:
        lp = -lp
        left = np.ceil(lp / 2)
        right = lp - left
        tr.data = tr.data[left:-right]
        tr.stats['starttime'] += float(left) * tr.stats['delta']

    tr.stats['npts'] = npts
    tr_padded = tr
    return tr_padded


IDformat = ['%NET','%STA','%LOC','%CHA']

def read_from_filesystem(ID,starttime,endtime,fs,trim=True):
    """Function to read data from a filesystem with a give file structure
    just by specifying ID and time interval.

    :param ID: seedID of the channel to read (NET.STA.LOC.CHA)
    :type ID: string
    :param starttime: start time
    :type starttime: datetime.datetime
    :param endtime: end time
    :type endtime: datetime.datetime
    :param fs: file structure descriptor
    :type fs: list
    :param trim: switch for trimming of the stream
    :type trim: bool

    :rtype: `obspy.stream`
    :return: data stream of requested data

    If the switch for trimming is False the whole stream in the files is
    returned.

    **File structure descriptor**
    fs is a list of strings or other lists indicating the elements in the
    file structure. Each item of this list is translated in one level of the
    directory structure. The first element is a string indicating the
    base_directory. The following elements can be strings to indicate
    one of the following
     - %X as defined by datetime.strftime indicating an element of
       the time. e.g. %H
     - %NET: network name
     - %STA: station name
     - %CHA: channel name
     - %LOC: location
     - string with out %
    The format strings are replaced either with an element of the starttime
    if they correspond to a datetime specifyer or with the respective part
    of the seedID. A string withouta % sign is not changed. If more than one
    format string is required in one directory level the need to be
    separated within a sublist.

    A format string for the ID can be followed by a pair of braces including
    two strings that will be used to replace the first string with the
    second. This cna be used if the filename contains part of the ID in a
    different form.

    :Exapmple:

    Example for a station 'GJE' in network 'HEJSL' with channel 'BHZ' and
    location '00' with the start time 2010-12-24_11:36:30 and \\
    ``fs = ['base_dir','%Y','%b','%NET,['%j','_','%STA'','_T_',"%CHA('BH','')",'.mseed']]``
    will be translated in a linux filename
    ``base_dir/2010/Nov/HEJSL/317_GJE_T_Z.mseed``

    :Note:

    If the data contain traces of different channels in the same file with
    different start and endtimes the routine will not work properly when the
    period spans multiple files.
    """

    #check input
    assert type(starttime) is datetime.datetime, \
        'starttime is not a datetime.datetime object: %s is type %s' % \
        (starttime, type(starttime))
    assert type(endtime) is datetime.datetime, \
        'endtime is not a datetime.datetime object: %s is type' % \
        (endtime, type(endtime))
    # translate file structure string
    fpattern = _current_filepattern(ID,starttime,fs)
    st = _read_filepattern(fpattern, starttime, endtime,trim)

    # if trace starts too late have a look in the previous section
    if (len(st)==0) or ((st[0].stats.starttime-st[0].stats.delta).datetime > starttime):
        fpattern, _ = _adjacent_filepattern(ID,starttime,fs,-1)
        st += _read_filepattern(fpattern, starttime, endtime,trim)
        st.merge()
    thistime = starttime
    while ((len(st)==0) or (st[0].stats.endtime.datetime < endtime)) & (thistime < endtime):
        fpattern, thistime = _adjacent_filepattern(ID,thistime,fs,1)
        if thistime == starttime:
            break
        st += _read_filepattern(fpattern, starttime, endtime,trim)
        st.merge()
    if trim:
        st.trim(starttime=UTCDateTime(starttime),endtime=UTCDateTime(endtime))
    st = st.select(id=ID)
    return st


def _read_filepattern(fpattern, starttime, endtime, trim):
    """Read a stream from files whose names match a given pattern.
    """
    flist = glob.glob(fpattern)
    starttimes = []
    endtimes = []
    # first only read the header information
    for fname in flist:
        st = read(fname,headonly=True)
        starttimes.append(st[0].stats.starttime.datetime)
        endtimes.append(st[-1].stats.endtime.datetime)
    # now read the stream from the files that contain the period
    st = Stream()
    for ind,fname in enumerate(flist):
        if (starttimes[ind] < endtime) and (endtimes[ind] > starttime):
            if trim:
                st += read(fname,starttime=UTCDateTime(starttime),endtime=UTCDateTime(endtime))
            else:
                st += read(fname)
    try:
        st.merge()
    except:
        print "Error merging traces for requested period!"
        st = Stream()
    return st


def _adjacent_filepattern(ID,starttime,fs,inc):
    """Return the file name that contains the data sequence prior to the one
    that contains the given time for a given file structure and ID.

    :param inc: either 1 for following of -1 for previous period
    :type inc: int
    """
    assert ((inc == 1) or (inc == -1)), " inc must either be 1 or -1"
    fname = ''
    flag = 0
    # find earlier time by turning back the time by one increment of the
    # last time indicator in the fs
    for part in fs[-1::-1]:
        if not isinstance(part,list):
            part = [part]
        for tpart in part[-1::-1]:
            if (not ((('(' in tpart) and (')' in tpart)) or (tpart in IDformat))
                                        and ('%' in tpart)
                                        and (flag == 0)):
                flag = 1
                if tpart in ['%H','%I']:
                    thistime = starttime + inc * datetime.timedelta(hours=1)
                elif tpart == '%p':
                    thistime = starttime + inc * datetime.timedelta(hours=12)
                elif tpart in ['%a','%A','%w','%d','%j']:
                    thistime = starttime + inc * datetime.timedelta(days=1)
                elif tpart in ['%U','%W']:
                    thistime = starttime + inc * datetime.timedelta(days=7)
                elif tpart in ['%b','%B','%m']:
                    if starttime.month + inc == 0:
                        thistime = datetime.datetime(starttime.year-1,
                                                     12,
                                                     starttime.day,
                                                     starttime.hour,
                                                     starttime.minute,
                                                     starttime.second,
                                                     starttime.microsecond)
                    elif starttime.month + inc == 13:
                        thistime = datetime.datetime(starttime.year+1,
                                                     1,
                                                     starttime.day,
                                                     starttime.hour,
                                                     starttime.minute,
                                                     starttime.second,
                                                     starttime.microsecond)
                    else:
                        thistime = datetime.datetime(starttime.year,
                                                     starttime.month + inc,
                                                     starttime.day,
                                                     starttime.hour,
                                                     starttime.minute,
                                                     starttime.second,
                                                     starttime.microsecond)
                elif tpart in ['%y','%Y']:
                    thistime = datetime.datetime(starttime.year - inc,
                                                     starttime.month,
                                                     starttime.day,
                                                     starttime.hour,
                                                     starttime.minute,
                                                     starttime.second,
                                                     starttime.microsecond)
    fname = _current_filepattern(ID,thistime,fs)
    return fname, thistime


def _current_filepattern(ID,starttime,fs):
    """Return the file name that contains the data sequence that contains
    the given time for a given file structure and ID.
    """
    fname = ''
    for part in fs:
        if not isinstance(part,list):
            part = [part]
        fpartname = ''
        for tpart in part:
            fpartname += _fs_translate(tpart,ID,starttime)
        fname = os.path.join(fname,fpartname)
    return fname


def _fs_translate(part,ID,starttime):
    """Translate part of the file structure descriptor.
    """
    IDlist = ID.split('.')
    if ('(' in part) and (')' in part):
        trans = re.search('\(.*?\)',part).group(0)
    else:
        trans = None
    # in case there is something to translate remove from the filepart
    if trans:
        part = part.replace(trans,'')
    # if there is no %-sign it is a fixed string
    if not '%' in part:
        res = part
    # in case it belongs to the ID replace it with the respective ID-part
    if part in IDformat:
        res = IDlist[IDformat.index(part)]
    # otherwise it must be part of the date string
    else:
        res = starttime.strftime(part)
    # replace if nesseccary
    if trans:
        transl = trans[1:-1].split(',')
        assert len(transl) == 2, "%s is not valid for replacement" % trans
        res = res.replace(transl[0].replace("'",""),transl[1].replace("'",""))
    return res



#EOF
