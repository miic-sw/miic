import os
import numpy as np
import ctypes as C

from obspy.signal.headers import clibevresp
from obspy.core.util.base import NamedTemporaryFile
from obspy.signal.invsim import cornFreq2Paz, pazToFreqResp, c_sac_taper
from miic.core.miic_utils import nextpow2



def evalresp(t_samp, nfft, filename, date, station='*', channel='*',
             network='*', locid='*', units="VEL", start_stage=-1, stop_stage=0, freq=False,
             debug=False):
    """
    Use the evalresp library to extract instrument response
    information from a SEED RESP-file. To restrict the response to the 
    instrument the start and stop stages can be specified here.

    :type t_samp: float
    :param t_samp: Sampling interval in seconds
    :type nfft: int
    :param nfft: Number of FFT points of signal which needs correction
    :type filename: str (or open file like object)
    :param filename: SEED RESP-filename or open file like object with RESP
        information. Any object that provides a read() method will be
        considered to be a file like object.
    :type date: UTCDateTime
    :param date: Date of interest
    :type station: str
    :param station: Station id
    :type channel: str
    :param channel: Channel id
    :type network: str
    :param network: Network id
    :type locid: str
    :param locid: Location id
    :type units: str
    :param units: Units to return response in. Can be either DIS, VEL or ACC
    :type start_stage: int
    :param start_stage: integer stage numbers of start stage (<0 causes
        default evalresp bahaviour).
    :type stop_stage: int
    :param stop_stage: integer stage numbers of stop stage
    :type debug: bool
    :param debug: Verbose output to stdout. Disabled by default.
    :rtype: numpy.ndarray complex128
    :return: Frequency response from SEED RESP-file of length nfft
    """
    if isinstance(filename, basestring):
        with open(filename, 'rb') as fh:
            data = fh.read()
    elif hasattr(filename, 'read'):
        data = filename.read()
    # evalresp needs files with correct line separators depending on OS
    fh = NamedTemporaryFile()
    #with NamedTemporaryFile() as fh:
    if 1:
        tempfile = fh.name
        fh.write(os.linesep.join(data.splitlines()))
        fh.close()

        fy = 1 / (t_samp * 2.0)
        # start at zero to get zero for offset/ DC of fft
        freqs = np.linspace(0, fy, nfft // 2 + 1)
        start_stage_c = C.c_int(start_stage)
        stop_stage_c = C.c_int(stop_stage)
        stdio_flag = C.c_int(0)
        sta = C.create_string_buffer(station)
        cha = C.create_string_buffer(channel)
        net = C.create_string_buffer(network)
        locid = C.create_string_buffer(locid)
        unts = C.create_string_buffer(units)
        if debug:
            vbs = C.create_string_buffer("-v")
        else:
            vbs = C.create_string_buffer("")
        rtyp = C.create_string_buffer("CS")
        datime = C.create_string_buffer(date.formatSEED())
        fn = C.create_string_buffer(tempfile)
        nfreqs = C.c_int(freqs.shape[0])
        res = clibevresp.evresp(sta, cha, net, locid, datime, unts, fn,
                                freqs, nfreqs, rtyp, vbs, start_stage_c,
                                stop_stage_c, stdio_flag, C.c_int(0))
        # optimizing performance, see
        # http://wiki.python.org/moin/PythonSpeed/PerformanceTips
        nfreqs, rfreqs, rvec = res[0].nfreqs, res[0].freqs, res[0].rvec
        h = np.empty(nfreqs, dtype='complex128')
        f = np.empty(nfreqs, dtype='float64')
        for i in xrange(nfreqs):
            h[i] = rvec[i].real + rvec[i].imag * 1j
            f[i] = rfreqs[i]
        clibevresp.free_response(res)
        del nfreqs, rfreqs, rvec, res
    if freq:
        return h, f
    return h


def correct_response(st, removeResp=False, removePAZ=False, simPAZ=False, pre_filt=None, cornFreq=0.0083):
    """
    Correct the seismometer response.
    
    Seismometer response is given in either a dictionary ``removeResp''
    or a dictionary ``removePAZ''. ``removeResp has precedence. The
    dictionaries have the following structure

    removeResp: dictionary with Response information to be removed
        has the following keys:
        respfile: (str) filename of evalresp response file.
        units: (str) Units to return response in. Can be either DIS, VEL or ACC
        start_stage: (int) integer stage numbers of start stage (<0 causes
            default evalresp bahaviour).
        stop_stage: (int) integer stage numbers of stop stage
    removePAZ: dictionary with poles and zeros to be removed has the following
        keys:
            poles: (list of complex numbers) location of poles
            zeros: (list of complex numbers) location of zeros
            gain: (float) gain
            sensitivity: (float) sensitivity
        It can easily be retrieved with obspy.arclink.client.Client.getPAZ

    if ``removeResp'' is given the response of each trace must be present in
    the respfile. If ``removePAZ'' is used the response is assumed to be the
    same for all traces in the stream.
    A filter specified in pre_filt can be applied in to avoid amplification of
    noise.
    The instrument to be simulated is either described in the dictionary simPAZ
    or if simPAZ is False by the corner frequency ``cornFreq''. Response
    correction is done in place and original data is overwritten.
    
    The input stream ``st'' should be demeaned and tapered.
    
    :type st: obspy.core.stream.Stream
    :param st: data stream to be corrected
    :type removeResp: dict
    :param removeResp: Response information to be removed
    :type removePAZ: dict
    :param removePAZ: Response information to be removed
    :type simPAZ: dict
    :param simPAZ: Response information to be simulated
    :type cornFreq: float
    :param cornFreq: corner frequency of instrument to be simulated
    :type pre_filt: list
    :param pre_filt: 4 corners of the filter
    """
    
    for tr in st:
        starttime = tr.stats['starttime']
        endtime = tr.stats['endtime']
        network = tr.stats['network']
        station = tr.stats['station']
        channel = tr.stats['channel']
        location = tr.stats['location']
        length = tr.stats['npts']
        sampling_rate = tr.stats['sampling_rate']
        np2l = nextpow2(2.*length)
        
        if not simPAZ:
            simPAZ = cornFreq2Paz(cornFreq, damp=0.70716)
        simresp, freqs = np.conj(pazToFreqResp(simPAZ['poles'], simPAZ['zeros'],
                          scale_fac=simPAZ['gain']*simPAZ['sensitivity'],
                          t_samp=1./sampling_rate,
                          nfft=np2l, freq=True)) #see Doc of pazToFreqResp for reason of conj()
        
        if removeResp:
            freqresp, freqs = evalresp(1./sampling_rate,np2l,removeResp['respfile'],
                                starttime, network=network, station=station,
                                channel=channel, locid=location,
                                start_stage=removeResp['start_stage'],
                                stop_stage=removeResp['stop_stage'],
                                units=removeResp['units'], freq=True)
        else:
            freqresp, freqs = np.conj(pazToFreqResp(removePAZ['poles'], removePAZ['zeros'],
                          scale_fac=removePAZ['gain']*removePAZ['sensitivity'],
                          t_samp=1./sampling_rate,
                          nfft=np2l, freq=True)) #see Doc of pazToFreqResp for reason of conj()
        
        ftr = np.fft.rfft(tr.data,n=np2l)
        ftr /= freqresp
        ftr[0] = 0.j  # correct the NaN in the DC component
        ftr *= simresp
        
        if pre_filt:
            ftr *= c_sac_taper(freqs, flimit=pre_filt)

        tr.data = np.fft.irfft(ftr)
        tr.trim(starttime,endtime)
    
    return   




        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
