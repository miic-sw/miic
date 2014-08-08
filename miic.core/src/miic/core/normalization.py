"""
@author:
Eraldo Pomponi
Mohammad Javad Fallahi

@copyright:
The MIIC Development Team (eraldo.pomponi@uni-leipzig.de)

@license:
GNU Lesser General Public License, Version 3
(http://www.gnu.org/copyleft/lesser.html)

Created on Nov 1, 2011False
"""

# Main imports
import numpy as np
from numpy.fft import fft, ifft
from multiprocessing import Pool


# ETS imports
try:
    BC_UI = True
    from traits.api import HasTraits, Int, Str, Bool
    from traitsui.api import View, Item
except ImportError:
    BC_UI = False
    pass

# Obspy imports
from obspy.core import Stream
from obspy.signal.invsim import cosTaper

##############################################################################
# Exceptions                                                                 #
##############################################################################


class InputError(Exception):
    """
    Exception for Input errors.
    """
    def __init__(self, msg):
        Exception.__init__(self, msg)


def _tr_gen(st):
    """ A generator that extract the traces form a Strem object

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


# Temporal normalization
def temp_norm(x, N):
    """ Computes the running-absolute-mean normalization of 1D numpy array x

    This function computes the running average of the absolute value of the
    data (array x) in a normalization time window of fixed length (2N+1) and
    weights the data at the center of the window by the inverse of this
    average.
    A one-sample window (N = 0) is equivalent to one-bit normalization.

    :type x: :class:`~numpy.ndarray`
    :param x: 1d array
    :type N: int
    :param N: 2N+1 : width of the normalization window

    :rtype: :class:`~numpy.ndarray`
    :return: **nx**: Running-absolute-mean normalization of x
    """
    w = np.zeros(len(x))
    nx = np.zeros(len(x))

    for n in range(len(x)):
        if n < N:
            w[n] = np.mean(np.abs(x[:n + N + 1]))
            if w[n] != 0.:
                nx[n] = x[n] / float(w[n])
        else:
            w[n] = np.mean(np.abs(x[n - N:n + N + 1]))
            if w[n] != 0.:
                nx[n] = x[n] / float(w[n])

    return nx


if BC_UI:
    class _temp_norm_view(HasTraits):
    
        N = Int(10)
    
        trait_view = View(Item('N', label='Norm. Win size'))


class _StTempNorm:

    def __init__(self, N):
        self.N = N

    def __call__(self, tr):
        tr.data = temp_norm(tr.data, self.N)
        return tr


def stream_temp_norm(st, N, parallel=True, processes=0):
    """ Computes the running-absolute-mean normaliz. of all Traces in st

    .. note::

    This operation is performed in place on the actual data arrays. The
    raw data is not accessible anymore afterwards. To keep your
    original data, use :func:`~miic.core.alpha_mod.stream_copy` to create
    a copy of your stream object.
    This function can also work in parallel an all or a specified number of
    cores available in the hosting machine.

    :type st: :class:`~obspy.core.trace.Stream`
    :param st: The container for the Traces that we want to normalize them
        in time domain
    :type N: int
    :param N: 2N+1 : width of the normalization window
    :type parallel: bool (Default: True)
    :pram parallel: If the filtering will be run in parallel or not
    :type processes: int
    :pram processes: Number of processes to start (if None it will be equal
        to the number of cores available in the hosting machine)

    :rtype: :class:`~obspy.core.trace.Stream`
    :return: **st_tnorm**: The resulting object containing the normalized
        version of all Traces in signals
    """

    if not isinstance(st, Stream):
        raise InputError("'signals' must be an \
            'obspy.core.stream.Stream' object")

    if not parallel:
        trs = map(_StTempNorm(N), _tr_gen(st))
        st.extend(trs)
    else:
        if processes == 0:
            processes = None

        p = Pool(processes=processes)

        p.map_async(_StTempNorm(N),
                    _tr_gen(st),
                    callback=_AppendST(st))

        p.close()
        p.join()

    st_tnorm = st

    return st_tnorm


if BC_UI:
    class _stream_temp_norm_view(HasTraits):
    
        msg = Str("""The 1-bit normalization is done in place on the actual
                   stream""")
        N = Int(10)
        paralllel = Bool(True)
        processes = Int(0)
    
        trait_view = View(Item('N', label='Norm. Win size'),
                          Item('parallel'),
                          Item('processes'),
                          Item('msg', style='readonly'))
    

class _St1BitNorm:

    def __call__(self, tr):
        tr.data = np.sign(tr.data)
        return tr


def stream_1bit_norm(st, parallel=True, processes=0):
    """ Compute the 1-bit normalization of all Traces in the Stream object

    .. note::

    This operation is performed in place on the actual data arrays. The
    raw data is not accessible anymore afterwards. To keep your
    original data, use :func:`~miic.core.alpha_mod.stream_copy` to create
    a copy of your stream object.
    This function can also work in parallel an all or a specified number of
    cores available in the hosting machine.

    :type st: :class:`~obspy.core.trace.Stream`
    :param st: Stream to be normalized
    :type parallel: bool (Default: True)
    :pram parallel: If the filtering will be run in parallel or not
    :type processes: int
    :pram processes: Number of processes to start (if None it will be equal
        to the number of cores available in the hosting machine)

    :rtype: :class:`~obspy.core.trace.Stream`
    :return: **st_1norm**: Copy of the original Stream obj with normalized
        traces
    """
    if not isinstance(st, Stream):
        raise InputError("'st' must be a 'obspy.core.stream.Stream' object")

    if not parallel:
        trs = map(_St1BitNorm(), _tr_gen(st))
        st.extend(trs)
    else:
        if processes == 0:
            processes = None

        p = Pool(processes=processes)

        p.map_async(_St1BitNorm(),
                    _tr_gen(st),
                    callback=_AppendST(st))

        p.close()
        p.join()

    st_1norm = st

    return st_1norm


if BC_UI:
    class _stream_1bit_norm_view(HasTraits):
        msg = Str("""The 1-bit normalization is done in place on the actual
                   stream""")
        parallel = Bool(True)
        processes = Int(0)
    
        trait_view = View(Item('parallel'),
                          Item('processes'),
                          Item('msg', style='readonly'))
    

def spect_norm(x):
    """ Computes the spectral normalization of  1D numpy array x

    This function divides the amplitude of the x spectrum by its absolute value

    :type x: :class:`~numpy.ndarray`
    :param x: 1d array

    :rtype: :class:`~numpy.ndarray`
    :return: **x_copy**: Whitened version of x
    """

    x_copy = x.copy()

    x_copy *= cosTaper(len(x_copy), 0.01)

    FFT = fft(x_copy)

    x_copy = ifft(FFT / abs(FFT)).real

    return x_copy


if BC_UI:
    class _spect_norm_view(HasTraits):
        msg = Str('The spectral normalization is done on a copy of the array')
        trait_view = View(Item('msg', style='readonly'))


class _StSpectNorm:

    def __call__(self, tr):
        try:
            if tr.stats.npts > 1:
                tr.data *= cosTaper(tr.stats.npts, 0.01)
                FFT = fft(tr.data)
                tr.data = ifft(FFT / abs(FFT)).real
        except ValueError:
            print "ERROR"
            print tr.stats.npts
        return tr


def stream_spect_norm(st, parallel=True, processes=0):
    """ Computes the spectral normalization of all Traces in st

    .. note::

    This operation is performed in place on the actual data arrays. The
    raw data is not accessible anymore afterwards. To keep your
    original data, use :func:`~miic.core.alpha_mod.stream_copy` to create
    a copy of your stream object.
    This function can also work in parallel an all or a specified number of
    cores available in the hosting machine.

    :type st: :class:`~obspy.core.trace.Stream`
    :param st: The container for the Traces that we want to normalize
        their spectrum
    :type parallel: bool (Default: True)
    :pram parallel: If the filtering will be run in parallel or not
    :type processes: int
    :pram processes: Number of processes to start (if None it will be equal
        to the number of cores available in the hosting machine)

    :rtype: :class:`~obspy.core.trace.Stream`
    :return: **st_spnorm**: The resulting object containing the whitened
        version of all Traces in st
    """
    if not isinstance(st, Stream):
        raise InputError("'signals' must be an \
            'obspy.core.stream.Stream' object")

    if not parallel:
        trs = map(_StSpectNorm(), _tr_gen(st))
        st.extend(trs)
    else:
        if processes == 0:
            processes = None

        p = Pool(processes=processes)

        p.map_async(_StSpectNorm(),
                    _tr_gen(st),
                    callback=_AppendST(st))
        p.close()
        p.join()

    st_spnorm = st
    return st_spnorm


if BC_UI:
    class _stream_spect_norm_view(HasTraits):
        msg = Str("""The spectral normalization is done in place on the actual
                   stream""")
        paralllel = Bool(True)
        processes = Int(0)
    
        trait_view = View(Item('parallel'),
                          Item('processes'),
                          Item('msg', style='readonly'))
