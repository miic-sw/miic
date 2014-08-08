"""
@author:
Eraldo Pomponi

@copyright:
The MIIC Development Team (eraldo.pomponi@uni-leipzig.de)

@license:
GNU Lesser General Public License, Version 3
(http://www.gnu.org/copyleft/lesser.html)

Created on -- Mar 22, 2011 --
"""
# PyWavelets import
import pywt

from scipy import median, diff, sqrt, log, power
from numpy import sign

try:
    from traits.api import HasTraits, Array, Int, Str, Property, Float
except ImportError:
    pass

class WT_Denoise(HasTraits):
    """ WT Based event removal.
    """

    # Signal to "clean"
    sig = Array()

    # Decomposition level
    level = Int

    # Wavelet family (['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey'])
    family = Str('sym')

    # Order
    order = Int(4)

    # Wavelet
    wt = Property(depends_on=['family', 'order'])

    # Extension mode (['zpd', 'cpd', 'sym', 'ppd', 'sp1', 'per'])
    mode = Str('per')

    # noise level
    sigma = Property(depends_on=['sig'])
    _sigma = Float

    ##### Public interface #####

    def filter(self, mode='soft'):

        if self.level > self.max_dec_level():
            clevel = self.max_dec_level()
        else:
            clevel = self.level

        # decompose
        coeffs = pywt.wavedec(self.sig, pywt.Wavelet(self.wt), \
                              mode=self.mode, \
                              level=clevel)

        # threshold evaluation
        th = sqrt(2 * log(len(self.sig)) * power(self.sigma, 2))

        # thresholding
        for (i, cAD) in enumerate(coeffs):
            if mode == 'soft':
                coeffs[i] = pywt.thresholding.soft(cAD, th)
            elif mode == 'hard':
                coeffs[i] = pywt.thresholding.hard(cAD, th)

        # reconstruct
        rec_sig = pywt.waverec(coeffs, pywt.Wavelet(self.wt), mode=self.mode)
        if len(rec_sig) == (len(self.sig) + 1):
            self.sig = rec_sig[:-1]

    def max_dec_level(self):
        return pywt.dwt_max_level(len(self.sig), pywt.Wavelet(self.wt))

    # Getter/Setter

    def _get_wt(self):
        return ('%s%d') % (self.family, self.order)

    def _get_sigma(self):

        self._sigma = median(abs(diff(self.sig) - median(diff(self.sig)))) \
            / (sqrt(2) * 0.6745)
        return self._sigma


class WT_Event_Filter(HasTraits):
    """ WT Based event removal.
    """

    # Signal to "clean"
    sig = Array()

    # Decomposition level
    level = Int

    # Wavelet family (['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey'])
    family = Str('sym')

    # Order
    order = Int(4)

    # Wavelet
    wt = Property(depends_on=['family', 'order'])

    # Extension mode (['zpd', 'cpd', 'sym', 'ppd', 'sp1', 'per'])
    mode = Str('per')

    # noise level
    sigma = Property(depends_on=['sig'])
    _sigma = Float

    ##### Public interface #####

    def filter(self):

        if self.level > self.max_dec_level():
            clevel = self.max_dec_level()
        else:
            clevel = self.level

        # decompose
        coeffs = pywt.wavedec(self.sig, pywt.Wavelet(self.wt), \
                              mode=self.mode, \
                              level=clevel)

        # threshold evaluation
        th = sqrt(2 * log(len(self.sig)) * power(self.sigma, 2))

        # thresholding
        for (i, cAD) in enumerate(coeffs):
            if i == 0:
                continue
            coeffs[i] = sign(cAD) * pywt.thresholding.less(abs(cAD), th)

        # reconstruct
        rec_sig = pywt.waverec(coeffs, pywt.Wavelet(self.wt), mode=self.mode)
        if len(rec_sig) == (len(self.sig) + 1):
            self.sig = rec_sig[:-1]

    def max_dec_level(self):
        return pywt.dwt_max_level(len(self.sig), pywt.Wavelet(self.wt))

    # Getter/Setter

    def _get_wt(self):
        return ('%s%d') % (self.family, self.order)

    def _get_sigma(self):

        self._sigma = median(abs(diff(self.sig) - median(diff(self.sig)))) \
            / (sqrt(2) * 0.6745)
        return self._sigma


if __name__ == '__main__':

    import numpy as np

    sig = np.array([1, 2, 3, 3, 5, 5, 6, 7, 8, 9, 10, 9, -1, -5, 3, 3])
    sig = sig.T
    family = 'db'
    order = 2
    level = 2
    WT_C = WT_Event_Filter(sig=sig, family=family, order=order, level=level)

    print WT_C.wt
    print WT_C.sigma

    WT_C.filter()

    print WT_C.sig
    print WT_C.sig.shape

    family = 'sym'
    order = 8
    level = 10
    n_points = 645
    x = np.linspace(0, 20 * np.pi, n_points)
    y_orig = np.sin(x)
    y = y_orig   # np.random.normal(size=y_orig.shape)*0.1

    WT_DEN = WT_Denoise(sig=y, family=family, order=order, level=level)

    WT_DEN.filter(mode='soft')

    s_den = WT_DEN.sig

    print s_den.shape
    print y_orig
    print s_den
    print np.sum(abs(y_orig - s_den), 0) / n_points
