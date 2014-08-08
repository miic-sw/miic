# -*- coding: utf-8 -*-
"""
This modules collect all those functions that are necessary to correlate
seismic traces. It is possible to fully exploit multi-core CPU capability
working in parallel. The base structure that collects the traces must
be :class:`obspy.core.Stream'.

@author:
Eraldo Pomponi

@copyright:
The MIIC Development Team (eraldo.pomponi@uni-leipzig.de)

@license:
GNU Lesser General Public License, Version 3
(http://www.gnu.org/copyleft/lesser.html)

Created on Nov 25, 2010
"""

# Main imports
import numpy as np
from scipy.fftpack import fftn, ifftn, fftfreq
from multiprocessing import Pool

# ETS imports
try:
    BC_UI = True
    from traits.api import HasTraits, Int, Bool
    from traitsui.api import View, Item
except ImportError:
    BC_UI = False
    pass

# Obspy import
from obspy.core import Stream, Trace, Stats, UTCDateTime
from obspy.signal.invsim import cosTaper

# Local import
from miic.core.miic_utils import trace_calc_az_baz_dist


class InputError(Exception):
    """ Exception for Input errors """
    def __init__(self, msg):
        Exception.__init__(self, msg)


if BC_UI:
    class _AppendST:
    
        def __init__(self, st):
            self.st = st

        def __call__(self, trs):
            for tr in trs:
                self.st.append(tr)


# From MLPY #
def extend(x, method='reflection', length='powerof2', size=0):
    """ Extend the 1D numpy array x beyond its original length

    This function comes from MLPY package with just a minor change.

    :type x: :class:`~numpy.ndarray`
    :param x: 1D ndarray.

    :type method: string, optional, `default`: ``reflection``
    :param method: Indicates which extension method to use. Possible choice
        are ``reflection``, ``periodic`` and ``zeros``.
    :type length: string,  optional, `default`:``powerof2``
    :param length: Indicates how to determinate the length of the extended
        data. Possible choices are ``powerof2`` and ``double`` and ``fixed``

    :rtype: :class:`~numpy.ndarray`
    :return: **xe**: Extended version of x

    .. rubric:: Basic Usage

    >>> import numpy as np
    >>> from corr_fun import extend
    >>> a = np.array([1,2,3,4,5])
    >>> xe = extend(a, method='periodic', length='powerof2')
    >>> xe
    array([1, 2, 3, 4, 5, 1, 2, 3])
    """

    if length == 'powerof2':
        lt = pow(2, np.ceil(np.log2(x.shape[0])))
        lp = lt - x.shape[0]

    elif length == 'double':
        lp = x.shape[0]

    elif length == 'fixed':
        if size == 0:
            raise ValueError
        else:
            lp = size - x.shape[0]
    else:
        ValueError("length %s is not available" % length)

    if method == 'reflection':
        xret = np.append(x, x[::-1][:lp])

    elif method == 'periodic':
        xret = np.append(x, x[:lp])

    elif method == 'zeros':
        xret = np.append(x, np.zeros(lp, dtype=x.dtype))

    else:
        ValueError("method %s is not available" % method)

    return xret


def merge_id(tr1, tr2):
    """ Create a SEED like `id` for an ObsPy Trace object that stores corr data

    It creates a SEED like `id` for the :class:`~obspy.core.trace.Trace`
    objects that stores the obtained correlation data between the two input
    Traces. It is created as a ``-`` separated combination of the
    ['network','station','location','channel'] informations of each one
    of them.

    :type tr1: :class:`~obspy.core.trace.Trace`
    :param tr1: First Trace
    :type tr2: :class:`~obspy.core.trace.Trace`
    :param tr2: Second Trace

    :rtype: string
    :return: **id**: SEED like `id` for the given combination of traces.
    """

    if not isinstance(tr1, Trace):
        raise TypeError("tr1 must be an obspy Trace object.")

    if not isinstance(tr2, Trace):
        raise TypeError("tr2 must be an obspy Trace object.")

    keywords = ['network', 'station', 'location', 'channel']

    m_id = '.'.join([tr1.stats[key] + '-' + tr2.stats[key] \
                     for key in keywords])

    return m_id


def combine_stats(tr1, tr2):
    """ Combine the meta-information of two ObsPy Trace objects

    This function returns a ObsPy :class:`~obspy.core.trace.Stats` object
    obtained combining the two associated with the input Traces.
    Namely ``tr1.stats`` and ``tr2.stats``.

    The fields ['network','station','location','channel'] are combined in
    a ``-`` separated fashion to create a "pseudo" SEED like ``id``.

    For all the others fields, only "common" information are retained: This
    means that only keywords that exist in both dictionaries will be included
    in the resulting one.

    :type tr1: :class:`~obspy.core.trace.Trace`
    :param tr1: First Trace
    :type tr2: :class:`~obspy.core.trace.Trace`
    :param tr2: Second Trace

    :rtype: :class:`~obspy.core.trace.Stats`
    :return: **stats**: combined Stats object
    """

    if not isinstance(tr1, Trace):
        raise TypeError("tr1 must be an obspy Trace object.")

    if not isinstance(tr2, Trace):
        raise TypeError("tr2 must be an obspy Trace object.")

    tr1_keys = tr1.stats.keys()
    tr2_keys = tr2.stats.keys()

    stats = Stats()

    # Adjust the information to create a new SEED like id
    keywords = ['network', 'station', 'location', 'channel']
    sac_keywords = ['sac']

    for key in keywords:
        if key in tr1_keys and key in tr2_keys:
            stats[key] = tr1.stats[key] + '-' + tr2.stats[key]

    for key in tr1_keys:
        if key not in keywords and key not in sac_keywords:
            if key in tr2_keys:
                if tr1.stats[key] == tr2.stats[key]:
                    # in the stats object there are read only objects
                    try:
                        stats[key] = tr1.stats[key]
                    except AttributeError:
                        pass

    try:
        stats['sac'] = {}
        stats.sac['stla'] = tr1.stats.sac.stla
        stats.sac['stlo'] = tr1.stats.sac.stlo
        stats.sac['stel'] = tr1.stats.sac.stel
        stats.sac['evla'] = tr2.stats.sac.stla
        stats.sac['evlo'] = tr2.stats.sac.stlo
        stats.sac['evel'] = tr2.stats.sac.stel

        az, baz, dist = trace_calc_az_baz_dist(tr1, tr2)

        stats.sac['dist'] = dist / 1000
        stats.sac['az'] = az
        stats.sac['baz'] = baz
    except KeyError:
        stats.pop('sac')
        print "Problem processing the geo information. Stats dictionary \
            not extended"

    return stats


def conv_traces(tr1, tr2, normal=True):
    """ Convolve two Traces and merge their meta-information

    It convolves the data stored in two :class:`~obspy.core.trace.Trace`
    Objects in frequency domain. If ``normal==True`` the resulting correlation
    data are normalized by a factor of
    :func:`sqrt(||tr1.data||^2 x ||tr2.data||^2)`

    Meta-informations associated to the resulting Trace are obtained through:

        - Merging the original meta-informations of the two input traces
          according to the :func:`~miic.core.corr_fun.combine_stats` function.

        - Adding the original two `Stats` objects to the newly
          created :class:`~obspy.core.trace.Trace` object as:
          >>> conv_tr.stats_tr1 = tr1.stats
          >>> conv_tr.stats_tr2 = tr2.stats
        - Fixing:
          >>> conv_tr.stats['npts'] = '...number of correlation points...'
          >>> conv_tr.stats['starttime'] = tr2.stats['starttime'] -
              tr1.stats['starttime']

    :type tr1: :class:`~obspy.core.trace.Trace`
    :param tr1: First Trace
    :type tr2: :class:`~obspy.core.trace.Trace`
    :param tr2: Second Trace
    :type normal: bool
    :param normal: Normalization flag

    :rtype: :class:`~obspy.core.trace.Trace`
    :return: **conv_tr**: Trace that stores convolved data and meta-information

    """

    if not isinstance(tr1, Trace):
        raise TypeError("tr1 must be an obspy Trace object.")

    if not isinstance(tr2, Trace):
        raise TypeError("tr2 must be an obspy Trace object.")
    
    zerotime = UTCDateTime(1971, 1, 1, 0, 0, 0)
    conv_tr = Trace()

    # extend traces to the next power of 2 of the longest trace
    lt = pow(2, np.ceil(np.log2(np.max([tr1.stats['npts'],
             tr2.stats['npts']]))))
    s1 = extend(tr1.data, method='zeros', length='fixed',size=lt)
    s2 = extend(tr2.data, method='zeros', length='fixed',size=lt)

    # create the combined stats
    conv_tr.stats = combine_stats(tr1, tr2)
    conv_tr.stats_tr1 = tr1.stats
    conv_tr.stats_tr2 = tr2.stats

    conv_tr.stats_tr1.npts = min(tr1.stats.npts, tr2.stats.npts)
    conv_tr.stats_tr2.npts = min(tr1.stats.npts, tr2.stats.npts)

    if normal:
        denom = np.sqrt(np.dot(s1.astype(np.float64), s1.T) *
                        np.dot(s2.astype(np.float64), s2.T))
    else:
        denom = 1.

    # remaining offset in samples (just remove fractions of samples)
    roffset = np.round((tr2.stats.starttime - tr1.stats.starttime) *
                        tr1.stats.sampling_rate)
    offset = (tr2.stats.starttime - tr1.stats.starttime) * \
        tr1.stats.sampling_rate - roffset
    # remaining offset in seconds
    roffset /= tr1.stats.sampling_rate


    convData = _fftconvolve(s1[::-1], s2, offset)
    convData = np.multiply(convData, (1 / denom))
    
    # set number of samples
    conv_tr.stats['npts'] = convData.shape[0]

    # time lag of the zero position, i.e. lag time of alignent
    t_offset_zeroleg = (float(convData.shape[0]) - 1.) / \
        (2. * tr1.stats.sampling_rate)

    # set starttime
    conv_tr.stats['starttime'] = zerotime - t_offset_zeroleg + \
        roffset

    conv_tr.data = convData

    return conv_tr
    
    
def deconvolve_traces(signal, divisor, eps, freq=[], residual=False):
    """ Deconvolve a time series from a set of time series.

    The function is a wrapper for the :class:`scipy.signal.convolve`
    function.
    
    :type signal: :class:`~obspy.core.stream.Stream`
    :param signal: signal from which the divisor is to be deconvolved
    :type divisor: :class:`~obspy.core.trace.Trace`
    :param divisor: time series that is to be deconvolved from signal
    :type eps: float
    :param eps: fraction of spectral mean used as a water level to 
        avoid spectral holes in the deconvolution.
    :type freq: two element array-like
    :param freq: frequency range for the estimation of the mean power
        that is scaled with ``eps`` to obtian the water level  
    :type residual: bool
    :param residual: return residual if True, defaults to False

    :rtype: obspy.core.stream
    :return: **(dcst, rst)**: decorrelated stream and residual (only
        if ``residual=True``

    """

    
    zerotime = UTCDateTime(1971,1,1)
    # trace length is taken from signal (must be even to use real fft)
    if signal[0].stats['npts'] % 2:
        trlen = signal[0].stats['npts']+1
    else:
        trlen = signal[0].stats['npts']
    delta = divisor.stats['delta']
    
    
    # prepare divisor
    divisor.detrend(type='constant')
    taper = cosTaper(divisor.stats['npts'],p=0.05)
    divisor.data *= taper

    divisor.trim(starttime=divisor.stats['starttime'],endtime=divisor.stats['starttime']+
                 (trlen-1)*delta,pad=True,fill_value=0,nearest_sample=False)
    # FFT divisor
    fd = np.fft.fftpack.rfft(divisor.data)
    # estimate the waterlevel to stabilize deconvolution
    if freq:
        f = np.linspace(-signal[0].stats['sampling_rate']/2., signal[0].stats['sampling_rate']/2.,len(fd))
        ind = np.nonzero(np.all([f>freq[0],f<freq[1]],axis=0))
        wl = eps * np.mean((fd*fd.conj())[ind])
    else:
        wl = eps * np.mean((fd*fd.conj()))
    
    # create the output stream
    dcst = Stream()
    rst = Stream()
    for tr in signal:
        if tr.stats['sampling_rate'] != divisor.stats['sampling_rate']:
            print "Sampling rates don't match for \n %s" % tr
            continue
        
        # prepare nuerator
        tr.detrend('constant')
        taper = cosTaper(tr.stats['npts'])
        tr.trim(starttime=tr.stats['starttime'], endtime=tr.stats['starttime']+
                (trlen-1)*delta,pad=True,fill_value=0,nearest_sample=False)
        tr.data *= taper
        # fft numerator
        sf = np.fft.fftpack.rfft(tr.data)
        
        # calculate deconvolution
        fdc = sf*fd/(fd**2+wl)
        dc = np.fft.fftpack.irfft(fdc)

        # template to hold results
        dctr = tr.copy()
        # propagate metadata
        dctr.stats = combine_stats(tr,divisor)
        dctr.data = dc
        dctr.stats['npts'] = len(dc)
        dctr.stats['starttime'] = zerotime - (divisor.stats['starttime']-tr.stats['starttime'])
        dctr.stats_tr1 = tr.stats
        dctr.stats_tr2 = divisor.stats
        
        # append to output stream
        dcst.append(dctr)
        
        if residual:
            # residual
            rtr = dctr.copy()
            rtr.data = tr.data - np.fft.fftpack.irfft(fdc * fd)
            # append to output stream
            rst.append(rtr)
            return (dcst, rst)
        
        return dcst


def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) / 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def _fftconvolve(in1, in2, offset, mode="full"):
    """Convolve two N-dimensional arrays using FFT. See convolve.
    """
    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)
    complex_result = (np.issubdtype(in1.dtype, np.complex) or
                      np.issubdtype(in2.dtype, np.complex))
    size = s1 + s2 - 1

    # Always use 2**n-sized FFT
    fsize = 2 ** np.ceil(np.log2(size))
    freq = fftfreq(int(fsize), 1)
    IN1 = fftn(in1, fsize) * np.exp(-1j * freq * offset * 2 * np.pi)
    IN1 *= fftn(in2, fsize)
    fslice = tuple([slice(0, int(sz)) for sz in size])
    ret = ifftn(IN1)[fslice].copy()
    del IN1
    if not complex_result:
        ret = ret.real
    if mode == "full":
        return ret
    elif mode == "same":
        if np.product(s1, axis=0) > np.product(s2, axis=0):
            osize = s1
        else:
            osize = s2
        return _centered(ret, osize)
    elif mode == "valid":
        return _centered(ret, abs(s2 - s1) + 1)


class _doCorr:

    def __init__(self, st, normal):
        self.st = st
        self.normal = normal

    def __call__(self, scomb):
        (k, i) = scomb
        tr1 = self.st[k]
        tr2 = self.st[i]
        conv_tr = conv_traces(tr1, tr2, self.normal)
        return conv_tr


def corr_trace_fun(signals, comb=[], normal=True,
                   parallel=True, processes=None):
    """ Correlate Traces according to the given combinations

    The `corr_trace_fun` correlates the Traces contained in the passed
    :class:`~obspy.core.trace.Stream` object according to the list of
    combinations `tuple` given in input. It does the job asynchronously
    instantiating as many process as cores available in the hosting machine.
    If traces do not share the same starttime the correlation trace is shifted
    by fractions of a sample such that time alignment is obtained precisely at
    the sample 1971-01-01T00:00:00Z. If there is no overlap between the
    traces this time might not be in the stream.

    :type signals: :class:`~obspy.core.stream.Stream`
    :param signals: The container for the Traces that we want to correlate
    :type comb: list, optional
    :param comb: List of combinations that must be calculated
    :type normal: bool, otional
    :param normal: Normalization flag (See
        :func:`~miic.core.corr_fun.conv_traces` for details)
    :type parallel: bool (Default: True)
    :pram parallel: If the filtering will be run in parallel or not
    :type processes: int
    :pram processes: Number of processes to start (if None it will be equal
        to the number of cores available in the hosting machine)

    :rtype: :class:`~obspy.core.stream.Stream`
    :return: **corrData**: The resulting object containing the correlation data
        and their meta-informations obtained as described
        in thr function :func:`~miic.core.corr_fun.conv_traces`
    """

    if not isinstance(signals, Stream):
        raise TypeError("signal must be an obspy Stream object.")

    corrData = Stream()

    nSignal = signals.count()

    if nSignal == 0:
        print "Empty stream!!"
        return corrData

    if (nSignal == 1) and not (comb == [(1, 1)]):
        print "Single trace. No cross correlation"
        return corrData

    if comb == []:
        comb = [(k, i) for k in range(nSignal) for i in range(k + 1, nSignal)]

    if not parallel:
        dc = _doCorr(signals, normal)
        corrData.extend(map(dc, comb))
    else:
        if processes == 0:
            processes = None

        p = Pool(processes=processes)

        p.map_async(_doCorr(signals, normal),
                    comb,
                    callback=_AppendST(corrData))

        p.close()
        p.join()

    return corrData


if BC_UI:
    class _corr_trace_fun_view(HasTraits):
        """ View Class for corr_trace_fun_parallel function
        """
        normal = Bool(True)
        parallel = Bool(True)
        processes = Int(0)
    
        trait_view = View(Item('normal', label='normalized'),
                          Item('parallel'),
                          Item('processes', enabled_when="parallel"))
    

def _norm_corr_trace(tr, unit_measure=3600):

    factor = (tr.stats_tr1.npts / tr.stats.sampling_rate) / unit_measure

    tr.data /= factor

    return tr


def norm_corr_stream(st, unit_measure=3600):
    """ Normalization of the correlation traces to the selected unit_measure

    This function normalizes the correlation traces in the input stream by a
    factor that takes into account exactly how long was the noise record that
    has been used to generate the correlation traces. The traces are dived by
    this length and multiplied to correspond to a length unit_measure that is,
    by default, equal to one hour.

    :type st: :class: `~obspy.core.stream.Stream`
    :param st: Stream that contains the traces that must be normalized
    :type unit_measure: int
    :param unit_measure: Unit measure respect which normalize the traces
        (in sec)

    :rtype: :class: `~obspy.core.stream.Stream`
    :return: **st_norm** : Normalized stream
    """

    if not isinstance(st, Stream):
        raise TypeError("st must be an obspy Stream object.")

    st_norm = st.copy()

    for tr in st_norm:
        _norm_corr_trace(tr, unit_measure)

    return st_norm


if BC_UI:
    class _norm_corr_stream_view(HasTraits):
        unit_measure = Int(3600)
        trait_view = View(Item('unit_measure', label='Unit measure (sec)'))
    
# EOF
