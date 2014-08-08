"""
@author:
Eraldo Pomponi

@copyright:
The MIIC Development Team (eraldo.pomponi@uni-leipzig.de)

@license:
GNU Lesser General Public License, Version 3
(http://www.gnu.org/copyleft/lesser.html)

Created on Oct 5, 2011
"""

# Main imports
import numpy as np

# ETS imports
try:
    BC_UI = True
    from traits.api import HasTraits, Int, Dict, \
        Float, Bool, Enum, Button
    from traitsui.api import View, Item, HGroup, Tabbed, VGroup, Include
except ImportError:
    BC_UI = False
    pass

# Local Imports
from miic.core.corr_mat_processing import corr_mat_smooth

# Obspy imports
import obspy.signal as signal


# FIX: This function is marked to be removed. It is still in the library but it
# will be removed in the next release
def ndarray_smooth(X, wsize, wtype, axis=1):
    """ Moving average filter using a window with requested size.

    This method is based on the convolution of a scaled window with the
    signal. It is applied along the specified ``axis``.
    Each row/col (i.e. depending on the selected ``axis``) is "prepared" by
    introducing reflected copies of it (with the window size) in both ends so
    that transient parts are minimized in the beginning and end part of the
    resulting array.

    :type X: :class:`~numpy.ndarray`
    :param X: Matrix to be filtered
    :type wsize: int
    :param wsize: Window size
    :type wtype: string
    :param wtype: Window type. It can be one of:
            ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    :type axis: int
    :param axis: Axis along with apply the filter. O: row by row
                                                   1: col by col

    :rtype: :class:`~numpy.ndarray`
    :return: **X**: Filtered matrix
    """

    return corr_mat_smooth(X, wsize, wtype, axis=axis)


if BC_UI:
    class _ndarray_smooth_view(HasTraits):
    
        wsize = Int(10)
        wtype = Enum(['flat', 'hanning', 'hamming', 'bartlett', 'blackman'])
        axis = Enum([0, 1])
    
        trait_view = View(Item('wsize'),
                          Item('wtype'),
                          Item('axis'))


def ndarray_filter(X, ftype, sampling_rate, filter_option):
    """ Filters data on X row-wise.

    This is performed "in-place" on each row array. The original data
    are not accessible anymore afterwards.

    :type X: :class:`~numpy.ndarray`
    :param X: Matrix to be filtered
    :type ftype: string
    :param ftype: Filter type. It can be one of:
            ['bandpass','bandstop','lowpass','highpass']
    :type sampling_rate: float
    :param sampling_rate: Sampling rate
    :type filter_option: dict
    :param filter_option: Option specific for each type of filter. They are
        described in the corresponding View

    :rtype: :class:`~numpy.ndarray`
    :return: **X_fil**: Filtered matrix
    """
    kw = dict([(kw_filed, filter_option[kw_filed])\
                for kw_filed in filter_option])

    # dictionary to map given type-strings to filter functions
    filter_functions = {"bandpass": signal.bandpass,
                        "bandstop": signal.bandstop,
                        "lowpass": signal.lowpass,
                        "highpass": signal.highpass}

    # make type string comparison case insensitive
    ftype = ftype.lower()

    if ftype not in filter_functions:
        msg = "Filter type \"%s\" not recognized. " % ftype + \
              "Filter type must be one of: %s." % filter_functions.keys()
        raise ValueError(msg)

    row, _ = X.shape

    for idx in np.arange(row):
        # do the actual filtering. the options dictionary is passed as
        # kwargs to the function that is mapped according to the
        # filter_functions dictionary.
        X[idx] = filter_functions[ftype](X[idx],
                                           df=sampling_rate, **kw)

    X_fil = X
    return X_fil


if BC_UI:
    class _ndarray_filter_view(HasTraits):
    
        sampling_rate = Float
    
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
    
        trait_view = View(Item('sampling_rate'),
                          Tabbed(VGroup(Item('ftype'),
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
                              enabled_when="type='lowpass'")
    
        highpass_view = HGroup(Item('freq_h', label='f min'),
                              Item('corners_h', label='corners'),
                              Item('zerophase_h', label='zerophase'),
                              Item('save_b', show_label=False),
                              label='highpass',
                              enabled_when="type='highpass'")
    
        bandpass_view = HGroup(Item('freqmin_bp', label='f min'),
                              Item('freqmax_bp', label='f max'),
                              Item('corners_bp', label='corners'),
                              Item('zerophase_bp', label='zerophase'),
                              Item('save_b', show_label=False),
                              label='bandpass',
                              enabled_when="type='bandpass'")
    
        bandstop_view = HGroup(Item('freqmin_bs', label='f min'),
                              Item('freqmax_bs', label='f max'),
                              Item('corners_bs', label='corners'),
                              Item('zerophase_bs', label='zerophase'),
                              Item('save_b', show_label=False),
                              label='bandstop',
                              enabled_when="type='bandstop'")
    
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
    
    
# def ndarray_wt_denoise(X, family, order, level, mode='soft', axis=0):
#    """ Wavelet base ndarray denoising.
#
#    This function denoise each row/column, depending on the ``axis`` selected,
#    of the matrix ``X`` in wavelet domain.
#    Two approach are available: ``Soft`` and ``Hard`` threscholding.
#
#    :type X: :class:`~numpy.ndarray`
#    :param X: Matrix to be denoised
#    :type family: string
#    :param family: Wavelet type. It can be one of:
#            ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey']
#    :type order: int
#    :param order: Wavelt order (e.g wavelet=sym, order=4 -> sym4)
#    :type level: int
#    :param level: Decomposition level
#    :type mode: string
#    :param mode: Denoising strategy. It can be one of:
#            ['soft', 'hard']
#    :type axis: int
#    :param axis: Axis along with apply the denoising algorithm.
#
#    :rtype: :class:`~numpy.ndarray`
#    :return: **X_den**: Denoised matrix
#
#    """
#    from wt_fun import WT_Denoise
#
#    wt_c = WT_Denoise(family=family, order=order, level=level)
#
#    if axis == 1:
#        X = X.T
#
#    row, _ = X.shape
#
#    for idx in np.arange(row):
#        wt_c.sig = X[idx]
#        wt_c.filter(mode=mode)
#        X[idx] = wt_c.sig
#
#    if axis == 1:
#        X = X.T
#
#    X_den = X
#    return X_den
#
#
# class _ndarray_wt_denoise_view(HasTraits):
#
#    family = Enum(['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey'])
#    order = Int(2)
#    level = Int(3)
#    mode = Enum('soft', 'hard')
#    axis = Enum(0, 1)
#
#    trait_view = View(Item('family'),
#                      Item('order'),
#                      Item('level'),
#                      Item('mode', label='thresh. mode'),
#                      Item('axis', label='0:by row - 1:by col'))
