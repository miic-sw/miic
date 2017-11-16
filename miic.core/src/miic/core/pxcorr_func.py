#!/usr/bin/python

import numpy as np
import scipy.signal as signal
from mpi4py import MPI
from copy import deepcopy

from obspy.core import UTCDateTime, stream, trace
import obspy.signal as osignal

from miic.core.corr_fun import combine_stats
from miic.core.miic_utils import convert_to_matlab

from numpy import (expand_dims, nanmean,reshape, transpose, take,
             sort, ones, arange, dot, cast, asarray)
from scipy import linalg


zerotime = UTCDateTime(1971,1,1)



def pxcorr_write(comm,A,st,**kwargs):
    """ A is an array with time along going the first dimension.
    """
    global zerotime

    t0 = MPI.Wtime()
    msize = A.shape
    ntrc = msize[1]

    psize = comm.Get_size()
    rank = comm.Get_rank()
    # time domain processing
    # map of traces on precesses
    pmap = (np.arange(ntrc)*psize)/ntrc

    # indecies for traces to be worked on by each process
    ind = pmap == rank

    ######################################
    ## time domain pre-processing
    params = {}
    for key in kwargs.keys():
        if not 'Processing' in key:
            params.update({key:kwargs[key]})
    for proc in kwargs['TDpreProcessing']:
        A[:,ind] = proc['function'](A[:,ind],proc['args'],params)
    # zero-padding
    A = zeroPadding(A,{'type':'avoidWrapPowerTwo'},params)

    ######################################
    ## FFT
    # Allocate space for rfft of data
    zmsize = A.shape
    fftsize = zmsize[0]//2+1
    B = np.zeros((fftsize,ntrc),dtype=complex)

    B[:,ind] = np.fft.rfft(A[:,ind],axis=0)
    freqs = rfftfreq(zmsize[0],1./kwargs['sampling_rate'])

    ######################################
    ## frequency domain pre-processing
    params.update({'freqs':freqs})
    for proc in kwargs['FDpreProcessing']:
        B[:,ind] = proc['function'](B[:,ind],proc['args'],params)

    ######################################
    ## collect results
    comm.barrier()
    comm.Allreduce(MPI.IN_PLACE,[B,MPI.DOUBLE],op=MPI.SUM)

    t1 = MPI.Wtime()

    ######################################
    ## correlation
    csize = len(kwargs['combinations'])
    irfftsize = (fftsize-1)*2
    sampleToSave = int(np.ceil(kwargs['lengthToSave'] *
                               kwargs['sampling_rate']))
    C = np.zeros((sampleToSave*2+1,csize),dtype=np.float64)

    center = irfftsize // 2

    pmap = (np.arange(csize)*psize)/csize
    ind = pmap == rank
    ind = np.arange(csize)[ind]
    for ii in ind:
        #print rank, ii, kwargs['combinations'][ii]
        # offset of starttimes in samples(just remove fractions of samples)
        offset = (kwargs['starttime'][kwargs['combinations'][ii][0]] -
                  kwargs['starttime'][kwargs['combinations'][ii][1]])
        if kwargs['center_correlation']:
            roffset = 0.
        else:
            # offset exceeding a fraction of integer
            roffset = np.fix(offset * kwargs['sampling_rate']) / kwargs['sampling_rate']
        # faction of samples to be compenasated by shifting
        offset -= roffset
        # normalization factor of fft correlation
        if kwargs['normalize_correlation']:
            norm = (np.sqrt(2.*np.sum(B[:,kwargs['combinations'][ii][0]] *
                                      B[:,kwargs['combinations'][ii][0]].conj()) -
                                      B[0,kwargs['combinations'][ii][0]]**2) *
                    np.sqrt(2.*np.sum(B[:,kwargs['combinations'][ii][1]] *
                                      B[:,kwargs['combinations'][ii][1]].conj()) -
                                      B[0,kwargs['combinations'][ii][1]]**2) /
                    irfftsize).real
        else:
            norm = 1.
        M = (B[:,kwargs['combinations'][ii][0]].conj() *
                               B[:,kwargs['combinations'][ii][1]] *
                               np.exp(1j * freqs * offset * 2 * np.pi))
                               #np.exp(-1j * freqs/kwargs['sampling_rate'] * offset/kwargs['sampling_rate'] * 2 * np.pi))

        tmp = np.fft.irfft(M,axis=0).real
        # cut the center and do fftshift
        out = np.concatenate((tmp[-sampleToSave:],tmp[:sampleToSave+1]))/norm
        starttime = zerotime -sampleToSave/kwargs['sampling_rate'] - roffset


        # put trace into a stream
        cst = stream.Stream()
        cstats = combine_stats(st[kwargs['combinations'][ii][0]],
                               st[kwargs['combinations'][ii][1]])
        cstats['starttime'] = starttime
        cstats['npts'] = len(out)
        cst.append(trace.Trace(data=out, header=cstats))
        cst[0].stats_tr1 = st[kwargs['combinations'][ii][0]].stats
        cst[0].stats_tr2 = st[kwargs['combinations'][ii][1]].stats
        if kwargs['direct_output']['function'] == 'convert_to_matlab':
            convert_to_matlab(cst,kwargs['direct_output']['base_name'],
                              kwargs['direct_output']['base_dir'])
    return 0


def pxcorr(comm,A,**kwargs):
    """ A is an array with time along going the first dimension.
    """
    global zerotime    
    
    t0 = MPI.Wtime()
    msize = A.shape
    ntrc = msize[1]
    
    psize = comm.Get_size()
    rank = comm.Get_rank()
    # time domain processing
    # map of traces on precesses
    pmap = (np.arange(ntrc)*psize)/ntrc
    
    # indecies for traces to be worked on by each process
    ind = pmap == rank
    
    ######################################
    ## time domain pre-processing
    params = {}
    for key in kwargs.keys():
        if not 'Processing' in key:
            params.update({key:kwargs[key]})
    for proc in kwargs['TDpreProcessing']:
        A[:,ind] = proc['function'](A[:,ind],proc['args'],params)
    # zero-padding
    A = zeroPadding(A,{'type':'avoidWrapPowerTwo'},params)

    ######################################
    ## FFT        
    # Allocate space for rfft of data
    zmsize = A.shape
    fftsize = zmsize[0]//2+1
    B = np.zeros((fftsize,ntrc),dtype=complex)
    
    B[:,ind] = np.fft.rfft(A[:,ind],axis=0)
    freqs = rfftfreq(zmsize[0],1./kwargs['sampling_rate'])
    
    ######################################
    ## frequency domain pre-processing
    params.update({'freqs':freqs})
    for proc in kwargs['FDpreProcessing']:
        B[:,ind] = proc['function'](B[:,ind],proc['args'],params)
        
    ######################################
    ## collect results
    comm.barrier()
    comm.Allreduce(MPI.IN_PLACE,[B,MPI.DOUBLE],op=MPI.SUM)
    
    t1 = MPI.Wtime()

    ######################################
    ## correlation        
    csize = len(kwargs['combinations'])
    irfftsize = (fftsize-1)*2
    sampleToSave = int(np.ceil(kwargs['lengthToSave'] *
                               kwargs['sampling_rate']))
    C = np.zeros((sampleToSave*2+1,csize),dtype=np.float64)

    center = irfftsize // 2

    pmap = (np.arange(csize)*psize)/csize
    ind = pmap == rank
    ind = np.arange(csize)[ind]
    starttimes = np.zeros(csize,dtype=np.float64)
    for ii in ind:
        #print rank, ii, kwargs['combinations'][ii]
        # offset of starttimes in samples(just remove fractions of samples)
        offset = (kwargs['starttime'][kwargs['combinations'][ii][0]] -
                  kwargs['starttime'][kwargs['combinations'][ii][1]])
        if kwargs['center_correlation']:
            roffset = 0.
        else:
            # offset exceeding a fraction of integer
            roffset = np.fix(offset * kwargs['sampling_rate']) / kwargs['sampling_rate']
        # faction of samples to be compenasated by shifting 
        offset -= roffset
        # normalization factor of fft correlation
        if kwargs['normalize_correlation']:
            norm = (np.sqrt(2.*np.sum(B[:,kwargs['combinations'][ii][0]] * 
                                      B[:,kwargs['combinations'][ii][0]].conj()) -
                                      B[0,kwargs['combinations'][ii][0]]**2) *
                    np.sqrt(2.*np.sum(B[:,kwargs['combinations'][ii][1]] * 
                                      B[:,kwargs['combinations'][ii][1]].conj()) -
                                      B[0,kwargs['combinations'][ii][1]]**2) /
                    irfftsize).real
        else:
            norm = 1.
        M = (B[:,kwargs['combinations'][ii][0]].conj() * 
                               B[:,kwargs['combinations'][ii][1]] * 
                               np.exp(1j * freqs * offset * 2 * np.pi))
                               #np.exp(-1j * freqs/kwargs['sampling_rate'] * offset/kwargs['sampling_rate'] * 2 * np.pi))
        ######################################
        ## frequency domain postProcessing
        #
        
        tmp = np.fft.irfft(M,axis=0).real
        # cut the center and do fftshift
        C[:,ii] = np.concatenate((tmp[-sampleToSave:],tmp[:sampleToSave+1]))/norm
        starttimes[ii] = zerotime -sampleToSave/kwargs['sampling_rate'] - roffset

    ######################################
    ## time domain postProcessing

    ######################################
    ## collect results 
    comm.barrier()
    comm.Allreduce(MPI.IN_PLACE,[C,MPI.DOUBLE],op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE,[starttimes,MPI.DOUBLE],op=MPI.SUM)
    #print 'werqwer'
    return (C,starttimes)


def detrend(A,args,params):
    """
    Remove trend from data
    
    Remove the trend from the time series data in `A`. Several methods are \\
    possible. The method is specified as the value of the `type` keyword in
    the argument dictionary `args`.
    
    Possible `types` of detrending:
       -`constant` or `demean`: substract mean of traces
       
       -`linear`: substract a least squares fitted linear trend form the data
    
    :type A: numpy.ndarray
    :param A: time series data with time oriented along the first \\
        dimension (columns)
    :type args: dictionary
    :param args: the only used keyword is `type`
    :type params: dictionary
    :param params: not used here
    
    :rtype: numpy.ndarray
    :return: detrended time series data
    """
    #for compatibility with obspy
    if args['type'] == 'demean':
        args['type'] = 'constant'
    if args['type'] == 'detrend':
        args['type'] = 'linear'

    # Detrend function taken from scipy and modified to account for nan
    type=args['type']
    axis=0
    data=A
    if type not in ['linear', 'l', 'constant', 'c']:
        raise ValueError("Trend type must be 'linear' or 'constant'.")
    data = asarray(data)
    dtype = data.dtype.char
    if dtype not in 'dfDF':
        dtype = 'd'
    if type in ['constant', 'c']:
        ret = data - expand_dims(nanmean(data, axis), axis)
        return ret
    else:
        dshape = data.shape
        N = dshape[axis]
        bp = sort(unique(r_[0, bp, N]))
        if np.any(bp > N):
            raise ValueError("Breakpoints must be less than length "
                             "of data along given axis.")
        Nreg = len(bp) - 1
        # Restructure data so that axis is along first dimension and
        #  all other dimensions are collapsed into second dimension
        rnk = len(dshape)
        if axis < 0:
            axis = axis + rnk
        newdims = r_[axis, 0:axis, axis + 1:rnk]
        newdata = reshape(transpose(data, tuple(newdims)),
                          (N, _prod(dshape) // N))
        newdata = newdata.copy()  # make sure we have a copy
        if newdata.dtype.char not in 'dfDF':
            newdata = newdata.astype(dtype)
        # Find leastsq fit and remove it for each piece
        for m in range(Nreg):
            Npts = bp[m + 1] - bp[m]
            A = ones((Npts, 2), dtype)
            A[:, 0] = cast[dtype](arange(1, Npts + 1) * 1.0 / Npts)
            sl = slice(bp[m], bp[m + 1])
            coef, resids, rank, s = linalg.lstsq(A, newdata[sl])
            newdata[sl] = newdata[sl] - dot(A, coef)
        # Put data back in original shape.
        tdshape = take(dshape, newdims, 0)
        ret = reshape(newdata, tuple(tdshape))
        vals = list(range(1, rnk))
        olddims = vals[:axis] + [0] + vals[axis:]
        ret = transpose(ret, tuple(olddims))
        return ret

def TDnormalization(A,args,params):
    """
    Amplitude dependent time domain normalization
    
    Calculate the envelope of the filtered trace, smooth it in a window of
    length `windowLength` and normalize the waveform by this trace. The two
    used keywords in `args` are `filter and `windowLength` that describe the
    filter and the length of the envelope smoothing window, respectively.
    
    `args` has the following structure:
    
        args = {'windowLength':`length of the envelope smoothing window in \\
        [s]`,'filter':{'type':`filterType`, fargs}}``
            
        `type` may be `bandpass` with the corresponding fargs `freqmin` and \\ 
        `freqmax` or `highpass`/`lowpass` with the `fargs` `freqmin`/`freqmax`
            
        :Example:  
        ``args = {'windowLength':5,'filter':{'type':'bandpass','freqmin':0.5,
        'freqmax':2.}}``
        
    :type A: numpy.ndarray
    :param A: time series data with time oriented along the first \\
        dimension (columns)
    :type args: dictionary
    :param args: arguments dictionary as described above
    :type params: dictionary
    :param params: not used here
    
    :rtype: numpy.ndarray
    :return: normalized time series data
    """
    # filter if args['filter']
    B = deepcopy(A)
    if args['filter']:
        func = getattr(osignal,args['filter']['type'])
        fargs = deepcopy(args['filter'])
        fargs.pop('type')
        B = func(A.T,df=params['sampling_rate'],**fargs).T
    else:
        B = deepcopy(A)
    # simple calculation of envelope
    B = B**2
    #print 'shape B', B.shape
    # smoothing of envelepe in both directions to avoid a shift
    window = (np.ones(np.ceil(args['windowLength']*params['sampling_rate']))/
              np.ceil(args['windowLength']*params['sampling_rate']))
    #print 'shape window', window.shape
    for ind in range(B.shape[1]):
        B[:,ind] = np.convolve(B[:,ind],window, mode='same')        
        B[:,ind] = np.convolve(B[::-1,ind],window, mode='same')[::-1]
        B[:,ind] += np.max(B[:,ind])*1e-6
    # normalization
    A /= np.sqrt(B)
    return A


def taper(A,args,params):
    """
    Taper to the time series data
    
    Apply a simple taper to the time series data.
    
    `args` has the following structure:
    
        args = {'type':`type of taper`,taper_args}``
            
        `type` may be `cosine_taper` with the corresponding taper_args `p` the 
        percentage of the traces to taper. Possibilities of `type` are \\
        given by `obspy.signal`.
            
        :Example:  
        ``args = {'type':'cosine_taper','p':0.1}``
    
    :type A: numpy.ndarray
    :param A: time series data with time oriented along the first \\
        dimension (columns)
    :type args: dictionary
    :param args: arguments dictionary as described above
    :type params: dictionary
    :param params: not used here
    
    :rtype: numpy.ndarray
    :return: tapered time series data
    """
    if args['type'] == 'cosine_taper':
        func = osignal.invsim.cosine_taper
    else:
        func = getattr(signal,args['type'])
    args = deepcopy(args)
    args.pop('type')
    tap = func(A.shape[0],**args)
    A *= np.tile(np.atleast_2d(tap).T,(1,A.shape[1]))
    return A


def clip(A,args,params):
    """
    Clip time series data at a multiple of the standard deviation
    
    Set amplitudes exeeding a certatin threshold to this threshold.
    The threshold for clipping is estimated as the standard deviation of each
    trace times a factor specified in `args`.
    
    :Note: Traces should be demeaned before clipping.
        
    :type A: numpy.ndarray
    :param A: time series data with time oriented along the first \\
        dimension (columns)
    :type args: dictionary
    :param args: the only keyword allowed is `std_factor` describing the \\
        scaling of the standard deviation for the clipping threshold
    :type params: dictionary
    :param params: not used here
    
    :rtype: numpy.ndarray
    :return: clipped time series data
    """
    stds = np.nanstd(A,axis=0)
    for ind in range(A.shape[1]):
        ts = args['std_factor']*stds[ind]
        A[A[:,ind]>ts,ind] = ts
        A[A[:,ind]<-ts,ind] = -ts
    return A


def mute(A,args,params):
    """
    Mute parts of data that exceed a threshold
    
    To completely surpress the effect of data with high amplitudes e.g. after
    aftershocks these parts of the data are muted (set to zero). The respective
    parts of the signal are identified as those where the envelope in a given
    frequency exceeds a threshold given directly as absolute numer or as a 
    multiple of the data's standard deviation. A taper of length `taper_len` is
    applied to smooth the edges of muted segments. Setting `extend_gaps` to
    Ture will ensure that the taper is applied outside the segments and data
    inside these segments will all zero. Edges of the data will be tapered too
    in this case.
    
    :Example:
    ``args={'filter':{'type':'bandpass', 'freqmin':1., 'freqmax':6.},'taper_len':1., 'threshold':1000, 'std_factor':1, 'extend_gaps':True}``
              
    :type A: numpy.ndarray
    :param A: time series data with time oriented along the first \\
        dimension (columns)
    :type args: dictionary
    :param args: the following keywords are allowed:
    
        * `filter`: (dictionary) description of filter to be applied befor calculation of \\
            the signal envelope. If not given the envelope is calculated \\
            from raw data. The value of the keyword filter is the same as \\
            the `args` for the function `TDfilter`.
        * `threshold`: (float) absolute amplitude of threhold for muting
        * `std_factor`: (float) alternativly to an absolute number the threhold can be\\
            estimated as a multiple of the standard deviation if the scaling\\
            is given in as value of the keyword `std_factor`. If neither\\
            `threshold` nor `std_factor` are given `std_factor`=1 is assumed.\\
        * `extend_gaps` (boolean) if True date abive the threshold is \\
            guaranteed to be muted, otherwise tapering will leak into these\\
            parts. This step involves an additional convolution.
        * `taper_len`: (float) length of taper for muted segments in seconds
    :type params: dictionary
    :param params: filled automatically by `pxcorr`
    
    :rtype: numpy.ndarray
    :return: clipped time series data
    """

    # return zeros if length of traces is shorter than taper
    ntap = int(args['taper_len']*params['sampling_rate'])
    if A.shape[0]<=ntap:
        return np.zeros_like(A)

    # filter if asked to
    if 'filter' in args.keys():
        C = TDfilter(A,args['filter'],params)
    else:
        C = deepcopy(A)
    
    # calculate envelope
    #D = np.abs(signal.hilbert(C,axis=0))
    D = np.abs(C)
    
    # calculate threshold
    if 'threshold' in args.keys():
        thres = np.zeros(A.shape[1]) + args['threshold']
    elif 'std_factor' in args.keys():
        thres = np.std(C,axis=0) * args['std_factor']
    else:
        thres = np.std(C,axis=0)
    
    # calculate mask
    mask = np.ones_like(D)
    mask[D>np.tile(np.atleast_2d(thres),(A.shape[0],1))]=0
    # extend the muted segments to make sure the whole segment is zero after
    if args['extend_gaps']:
        tap = np.ones(ntap)/ntap
        for ind in range(A.shape[1]):
            mask[:,ind] = np.convolve(mask[:,ind],tap, mode='same')
        nmask = np.ones_like(D)
        nmask[mask<1.] = 0
    else:
        nmask = mask
    
    # apply taper
    tap = 2. - (np.cos(np.arange(ntap,dtype=float)/ntap*2.*np.pi) + 1.)
    tap /= ntap
    for ind in range(A.shape[1]):
        nmask[:,ind] = np.convolve(nmask[:,ind],tap, mode='same')
    
    # mute date with tapered mask
    A *= nmask
    return A


def TDfilter(A,args,params):
    """
    Filter time series data
    
    Filter in time domain. Types of filters are defined by `obspy.signal`.
    
    `args` has the following structure:
    
        args = {'type':`filterType`, fargs}
            
        `type` may be `bandpass` with the corresponding fargs `freqmin` and \\ 
        `freqmax` or `highpass`/`lowpass` with the `fargs` `freqmin`/`freqmax`
            
        :Example:  
        ``args = {'type':'bandpass','freqmin':0.5,'freqmax':2.}``
        
    :type A: numpy.ndarray
    :param A: time series data with time oriented along the first \\
        dimension (columns)
    :type args: dictionary
    :param args: arguments dictionary as described above
    :type params: dictionary
    :param params: not used here
    
    :rtype: numpy.ndarray
    :return: filtered time series data
    """
    func = getattr(osignal.filter,args['type'])
    args = deepcopy(args)
    args.pop('type')
    # filtering in obspy.signal is done along the last dimension that why .T
    A = func(A.T,df=params['sampling_rate'],**args).T
    return A


def normalizeStandardDeviation(A,args,params):
    """
    Divide the time series by their standard deviation
    
    Divide the amplitudes of each trace by its standard deviation.
    
    :type A: numpy.ndarray
    :param A: time series data with time oriented along the first \\
        dimension (columns)
    :type args: dictionary
    :param args: not used here
    :type params: dictionary
    :param params: not used here
    
    :rtype: numpy.ndarray
    :return: normalized time series data
    """
    std = np.std(A,axis=0)
    A /= np.tile(std,(A.shape[0],1))
    return A
    
    
def signBitNormalization(A,args,params):
    """
    One bit normalization of time series data
    
    Return the sign of the samples (-1, 0, 1).
    
    :type A: numpy.ndarray
    :param A: time series data with time oriented along the first \\
        dimension (columns)
    :type args: dictionary
    :param args: not used here
    :type params: dictionary
    :param params: not used here
    
    :rtype: numpy.ndarray
    :return: 1-bit normalized time series data
    """
    return np.sign(A)


def zeroPadding(A,args,params):
    """
    Append zeros to the traces 
    
    Pad traces with zeros to increase the speed of the Fourier transforms and
    to avoid wrap around effects. Three possibilities for the length of the 
    padding can be set in `args['type']`
    
        -`nextPowerOfTwo`: traces are padded to a length that is the next power \\
            of two from the original length
        -`avoidWrapAround`: depending on length of the trace that is to be used \\
            the padded part is just long enough to avoid wrap around
        -`avoidWrapPowerTwo`: use the next power of two that avoids wrap around
        
        :Example: ``args = {'type':'avoidWrapPowerTwo'}``
    
    :type A: numpy.ndarray
    :param A: time series data with time oriented along the first \\
        dimension (columns)
    :type args: dictionary
    :param args: arguments dictionary as described above
    :type params: dictionary
    :param params: not used here
    
    :rtype: numpy.ndarray
    :return: zero padded time series data
    """
    npts,ntrc = A.shape
    if args['type'] == 'nextPowerOfTwo':
        N = osignal.util.next_pow_2(npts)
    elif args['type'] == 'avoidWrapAround':
        N = npts + params['sampling_rate'] * params['lengthToSave']
    elif args['type'] == 'avoidWrapPowerTwo':
        N = osignal.util.next_pow_2(npts + params['sampling_rate'] *
                                  params['lengthToSave'])
    else:
        raise ValueError("type '%s' of zero padding not implemented" %
                         args['type'])
    A = np.concatenate((A,np.zeros((N-npts,ntrc),dtype=np.float64)),axis=0)
    return A
    

def spectralWhitening(B,args,params):
    """
    Spectal whitening of Fourier-transformed date

    Normalize the amplitude spectrum of the complex spectra in `B`. The
    `args` dictionary may contain the keyword `joint_norm`. If its value is
    True the normalization of sets of three traces are normalized jointly by
    the mean of their amplitude spectra. This is useful for later rotation of
    correlated traces in the ZNE system into the ZRT system.
    
    :type B: numpy.ndarray
    :param B: Fourier transformed time series data with frequency oriented\\
        along the first dimension (columns)
    :type args: dictionary
    :param args: arguments dictionary as described above
    :type params: dictionary
    :param params: not used here
    
    :rtype: numpy.ndarray
    :return: whitened spectal data
    """
    absB = np.absolute(B)
    if 'joint_norm' in args.keys():
        if args['joint_norm'] == True:
            assert B.shape[1] % 3 == 0, "for joint normalization the number\
                      of traces needs to the multiple of 3: %d" % B.shape[0]
            for ii in np.arange(0,B.shape[1],3):
                print 'ii', ii
                absB[:,ii:ii+3] = np.tile(np.atleast_2d(
                                    np.mean(absB[:,ii:ii+3],axis=1)).T,[1,3])
    B /= absB
    # remove zero freq component 
    #B[0,:] = 0.j
    return B
    
    
def FDfilter(B,args,params):
    """
    Filter Fourier-transformed data

    Filter Fourier tranformed data by tapering in frequency domain. The `args`
    dictionary is supposed to contain the key `flimit` with a value that is a 
    four element list or tuple defines the corner frequencies (f1, f2, f3, f4)
    in Hz of the cosine taper which is one between f2 and f3 and tapers to zero
    for f1 < f < f2 and f3 < f < f4.

    :type B: numpy.ndarray
    :param B: Fourier transformed time series data with frequency oriented\\
        along the first dimension (columns)
    :type args: dictionary
    :param args: arguments dictionary as described above
    :type params: dictionary
    :param params: params['freqs'] contains an array with the freqency values
        of the samples in `B`
    
    :rtype: numpy.ndarray
    :return: filtered spectal data
    """
    args = deepcopy(args)
    args.update({'freqs':params['freqs']})
    tap = osignal.invsim.cosine_taper(B.shape[0],**args)
    B *= np.tile(np.atleast_2d(tap).T,(1,B.shape[1]))
    return B
    

def FDsignBitNormalization(B,args,params):
    """
    Sign bit normalization of frequency transformed data
    
    Divides each sample by its amplitude resulting in trace with amplidues of
    (-1, 0, 1). As this operation is done in frequency domain it requires two
    Fourier transforms and is thus quite costly but alows to be performed 
    after other steps of frequency domain procesing e.g. whitening.
    

    :type B: numpy.ndarray
    :param B: Fourier transformed time series data with frequency oriented\\
        along the first dimension (columns)
    :type args: dictionary
    :param args: not used in this function
    :type params: dictionary
    :param params: not used in this function
      
    :rtype: numpy.ndarray
    :return: frequency transform of the 1-bit normalized data
    """
    B = np.fft.irfft(B,axis=0)
    C = B.real
    C = np.sign(B)
    return np.fft.rfft(C,axis=0)
     
    
def rfftfreq(n, d=1.0):
    """
    Return the Discrete Fourier Transform sample frequencies
    (for usage with rfft, irfft).

    The returned float array `f` contains the frequency bin centers in cycles
    per unit of the sample spacing (with zero at the start).  For instance, if
    the sample spacing is in seconds, then the frequency unit is cycles/second.

    Given a window length `n` and a sample spacing `d`::

      f = [0, 1, ...,     n/2-1,     n/2] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n)   if n is odd

    Unlike `fftfreq` (but like `scipy.fftpack.rfftfreq`)
    the Nyquist frequency component is considered to be positive.

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.

    Returns
    -------
    f : ndarray
        Array of length ``n//2 + 1`` containing the sample frequencies.

    Examples
    --------
    >>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5, -3, 4], dtype=float)
    >>> fourier = np.fft.rfft(signal)
    >>> n = signal.size
    >>> sample_rate = 100
    >>> freq = np.fft.fftfreq(n, d=1./sample_rate)
    >>> freq
    array([  0.,  10.,  20.,  30.,  40., -50., -40., -30., -20., -10.])
    >>> freq = np.fft.rfftfreq(n, d=1./sample_rate)
    >>> freq
    array([  0.,  10.,  20.,  30.,  40.,  50.])

    """
    if not isinstance(n, int):
        raise ValueError("n should be an integer")
    val = 1.0/(n*d)
    N = n//2 + 1
    results = np.arange(0, N, dtype=int)
    return results * val
    

def stream_pxcorr(st,options,comm=None):
    """ 
    Preprocess and correlate traces in a stream

    This is the central function of this module. It takes an obspy.stram input
    stream applies and:
        - applies time domain preprocesing
        - Fourier transforms the data
        - applies frequency domain preprocessing
        - multiplies the conjugate spectra (correlation)
        - transforms back into time domain
        - returns a stream with the correlated data

    All this can be done in parallel on different CPU communicating via
    the `mpi4py` implementation of MPI. Control the different processing steps
    are controlled with the dictionary `options`. The following keys are
    required in `options`:
        - combinations: list of tuples that identify the combinations of the\\
            traces in `st` to be correlated
        - lengthToSave: length of the correlated traces in s to return
        - normalize_correlation: Boolean. If True the correaltion is\\
            normalized. If False the pure product of the spectra is returned
        - center_correlation: Boolean. If True th location of zero lag time\\
            is in the center of the returned trace. If False the position of\\
            zero lag time is determined by the start times of the traces. If\\
            they are identical it is in the center anyway. If the difference \\
            in start times is larger the zero lag time is offset.
        - TDpreProcessing: list controlling the time domain preprocessing
        - FDpreProcessing: list controlling the frequency domain preprocessing

    The item in the list `TDpreProcessing` and `FDpreProcessing` are 
    dictionaries with two keys: `function` containing the function to apply and
    `args` being a dictionary with the arguments for this function. The
    functions in `TDpreProcessing` are applied in their order before the
    Fourier transformation and those in FDpreProcessing` are applied in their
    order Fourier domain.

    :Example:
        ``options = {'TDpreProcessing':[{'function':detrend,
                                'args':{'type':'linear'}},
                               {'function':taper,
                                'args':{'type':'cosine_taper',
                                        'p':0.01}},
                               {'function':TDfilter,
                                'args':{'type':'bandpass',
                                        'freqmin':1.,
                                        'freqmax':3.}},
                               {'function':TDnormalization,
                                'args':{'filter':{'type':'bandpass',
                                                 'freqmin':0.5,
                                                 'freqmax':2.},
                                        'windowLength':1.}},
                               {'function':signBitNormalization,
                                'args':{}}
                                 ],
            'FDpreProcessing':[{'function':spectralWhitening,
                                'args':{}},
                               {'function':FDfilter,
                                'args':{'flimit':[0.5, 1., 5., 7.]}}],
            'lengthToSave':20,
            'center_correlation':True,
            'normalize_correlation':True,
            'combinations':[(0,0),(0,1),(0,2),(1,2)]}``
    
    `comm` is a mpi4py communicator that can be passed if already initialized
    otherwise it is created here.

    :type st: obspy.stream
    :param st: stream with traces to be correlated
    :type options: dictionary
    :param options: controll dictionary as described above
    :type comm: mpi4py communicator
    :param comm: communicator if initialized externally
    """
    
    # initialize MPI
    if not comm:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()    

    # get parameters of the data
    if rank == 0:
        starttime = []
        npts = []
        for tr in st:
            starttime.append(tr.stats['starttime'])
            npts.append(tr.stats['npts'])
        npts = np.max(np.array(npts))
    else:
        starttime = None
        npts = None
    starttime = comm.bcast(starttime, root=0)
    npts = comm.bcast(npts, root=0)
    # fill matrix with noise data
    A = np.zeros([npts,len(st)])
    if rank == 0:    
        for ii in range(len(st)):
            A[0:st[ii].stats['npts'],ii] = st[ii].data
    comm.Bcast([A,MPI.DOUBLE],root=0)
    options.update({'starttime':starttime,
                    'sampling_rate':st[0].stats['sampling_rate']})

    # call pxcorr for correlation with direct output
    if 'direct_output' in options.keys():
        pxcorr_write(comm,A,st,**options)
        return stream.Stream()

    # call pxcorr for correlation
    A, starttime = pxcorr(comm,A,**options)
    npts = A.shape[0]
    
    # put trace into a stream
    cst = stream.Stream()
    for ii in range(len(options['combinations'])):
        cstats = combine_stats(st[options['combinations'][ii][0]],
                               st[options['combinations'][ii][1]])
        cstats['starttime'] = starttime[ii]
        cstats['npts'] = npts
        cst.append(trace.Trace(data=A[:,ii], header=cstats))
        cst[-1].stats_tr1 = st[options['combinations'][ii][0]].stats
        cst[-1].stats_tr2 = st[options['combinations'][ii][1]].stats
        
    return cst
     



def calc_cross_combis(st, method='betweenStations'):
    """
    Calculate a list of all cross correlation combination
    of traces in the stream: i.e. all combination with two different
    stations involved.

    :type method: string
    :param method: Determines which traces of the strem are combined.
    
        ``'betweenStations'``: Traces are combined if either their station or
            their network names are different.
        ``'betweenComponents'``: Traces are combined if their components (last
            letter of channel name) names are different and their station and
            network names are identical (single station cross-correlation).
        ``'autoComponents'``: Traces are combined only with themselves.
        ``'allSimpleCombinations'``: All Traces are combined once (onle one of
            (0,1) and (1,0))
        ``'allCombinations'``: All traces are combined in both orders ((0,1)
            and (1,0))
    """

    combis = []
    if method == 'betweenStations':
        for ii in range(len(st)):
            for jj in range(ii+1,len(st)):
                if ((st[ii].stats['network'] != st[jj].stats['network']) or 
                    (st[ii].stats['station'] != st[jj].stats['station'])):
                    combis.append((ii,jj))
    elif method == 'betweenComponents':
        for ii in range(len(st)):
            for jj in range(ii+1,len(st)):
                if ((st[ii].stats['network'] == st[jj].stats['network']) and 
                    (st[ii].stats['station'] == st[jj].stats['station']) and
                    (st[ii].stats['channel'][-1] != st[jj].stats['channel'][-1])):
                    combis.append((ii,jj))
    elif method == 'autoComponents':
        for ii in range(len(st)):
            combis.append((ii,ii))
    elif method == 'allSimpleCombinations':
        for ii in range(len(st)):
            for jj in range(ii,len(st)):
                    combis.append((ii,jj))
    elif method == 'allCombinations':
        for ii in range(len(st)):
            for jj in range(len(st)):
                combis.append((ii,jj))
    else:
        raise ValueError("Method has to be one of ('betweenStations', "
                         "'betweenComponents', 'autoComponents', "
                         "'allSimpleCombinations' or 'allCombinations').")
    return combis



def rotate_multi_corr_stream(st):
    """Rotate a stream with full Greens tensor from ENZ to RTZ

    Take a stream with numerous correlation traces and rotate the 
    combinations of ENZ components into combinations of RTZ components in case all
    nine components of the Green's tensor are present. If not all nine components
    are present no trace for this station combination is returned.
    
    :type st: obspy.stream
    :param st: stream with data in ENZ system
    :rtype: obspy.stream
    :return: stream in the RTZ system
    """
    
    out_st = stream.Stream()
    while st:
        tl = range(9)
        tst = st.select(network=st[0].stats['network'],station=st[0].stats['station'])
        cnt = 0
        for ttr in tst:
            if ttr.stats['channel'][2] == 'E':
                if ttr.stats['channel'][6] == 'E':
                    tl[0] = ttr
                    cnt += 1
                elif ttr.stats['channel'][6] == 'N':
                    tl[1] = ttr
                    cnt += 2
                elif ttr.stats['channel'][6] == 'Z':
                    tl[2] = ttr
                    cnt += 4
            elif ttr.stats['channel'][2] == 'N':
                if ttr.stats['channel'][6] == 'E':
                    tl[3] = ttr
                    cnt += 8
                elif ttr.stats['channel'][6] == 'N':
                    tl[4] = ttr
                    cnt += 16
                elif ttr.stats['channel'][6] == 'Z':
                    tl[5] = ttr
                    cnt += 32
            elif ttr.stats['channel'][2] == 'Z':
                if ttr.stats['channel'][6] == 'E':
                    tl[6] = ttr
                    cnt += 64
                elif ttr.stats['channel'][6] == 'N':
                    tl[7] = ttr
                    cnt += 128
                elif ttr.stats['channel'][6] == 'Z':
                    tl[8] = ttr                   
                    cnt += 256
        if cnt == 2**9-1:
            st0 = stream.Stream()
            for t in tl:
                st0.append(t)
            st1 = _rotate_corr_stream(st0)
            out_st += st1
        elif cnt == 27: # only horizontal component combinations present
            st0 = stream.Stream()
            for t in [0, 1, 3, 4]:
                st0.append(tl[t])
            st1 = _rotate_corr_stream_horizontal(st0)
            out_st += st1
        elif cnt == 283: # horizontal combinations + ZZ
            st0 = stream.Stream()
            for t in [0, 1, 3, 4]:
                st0.append(tl[t])
            st1 = _rotate_corr_stream_horizontal(st0)
            out_st += st1
            out_st.append(tl[8])
        for ttr in tst:
            for ind,tr in enumerate(st):
                if ttr.id == tr.id:
                    st.pop(ind)
    
    return out_st


def _rotate_corr_stream_horizontal(st):
    """ Rotate traces in stream from the EE-EN-NE-NN system to
    the RR-RT-TR-TT system. The letters give the component order
    in the input and output streams. Input traces are assumed to be of same length
    and simultaneously sampled.
    """

    # rotation angles
    # phi1 : counter clockwise angle between E and R(towards second station)
    # the leading -1 accounts fact that we rotate the coordinate system, not a vector
    phi1 = - np.pi/180*(90-st[0].stats['sac']['az'])
    # phi2 : counter clockwise angle between E and R(away from first station)
    phi2 = - np.pi/180*(90-st[0].stats['sac']['baz']+180)

    c1 = np.cos(phi1)
    s1 = np.sin(phi1)
    c2 = np.cos(phi2)
    s2 = np.sin(phi2)

    rt = stream.Stream()
    RR = st[0].copy()
    RR.data = c1*c2*st[0].data - c1*s2*st[1].data - s1*c2*st[2].data + s1*s2*st[3].data
    tcha = list(RR.stats['channel'])
    tcha[2] = 'R'
    tcha[6] = 'R'
    RR.stats['channel'] = ''.join(tcha)
    rt.append(RR)

    RT = st[0].copy()
    RT.data = c1*s2*st[0].data + c1*c2*st[1].data - s1*s2*st[2].data - s1*c2*st[3].data
    tcha = list(RT.stats['channel'])
    tcha[2] = 'R'
    tcha[6] = 'T'
    RT.stats['channel'] = ''.join(tcha)
    rt.append(RT)

    TR = st[0].copy()
    TR.data = s1*c2*st[0].data - s1*s2*st[1].data + c1*c2*st[2].data - c1*s2*st[3].data
    tcha = list(TR.stats['channel'])
    tcha[2] = 'T'
    tcha[6] = 'R'
    TR.stats['channel'] = ''.join(tcha)
    rt.append(TR)

    TT = st[0].copy()
    TT.data = s1*s2*st[0].data + s1*c2*st[1].data + c1*s2*st[2].data + c1*c2*st[3].data
    tcha = list(TT.stats['channel'])
    tcha[2] = 'T'
    tcha[6] = 'T'
    TT.stats['channel'] = ''.join(tcha)
    rt.append(TT)

    return rt


def _rotate_corr_stream(st):
    """ Rotate traces in stream from the EE-EN-EZ-NE-NN-NZ-ZE-ZN-ZZ system to
    the RR-RT-RZ-TR-TT-TZ-ZR-ZT-ZZ system. The letters give the component order
    in the input and output streams. Input traces are assumed to be of same length
    and simultaneously sampled.
    """
    
    # rotation angles
    # phi1 : counter clockwise angle between E and R(towards second station)
    # the leading -1 accounts fact that we rotate the coordinate system, not a vector
    phi1 = - np.pi/180*(90-st[0].stats['sac']['az'])
    # phi2 : counter clockwise angle between E and R(away from first station)
    phi2 = - np.pi/180*(90-st[0].stats['sac']['baz']+180)
    
    c1 = np.cos(phi1)
    s1 = np.sin(phi1)
    c2 = np.cos(phi2)
    s2 = np.sin(phi2)
    
    rtz = stream.Stream()
    RR = st[0].copy()
    RR.data = c1*c2*st[0].data - c1*s2*st[1].data - s1*c2*st[3].data + s1*s2*st[4].data
    tcha = list(RR.stats['channel'])
    tcha[2] = 'R'
    tcha[6] = 'R'
    RR.stats['channel'] = ''.join(tcha)
    rtz.append(RR)
    
    RT = st[0].copy()
    RT.data = c1*s2*st[0].data + c1*c2*st[1].data - s1*s2*st[3].data - s1*c2*st[4].data
    tcha = list(RT.stats['channel'])
    tcha[2] = 'R'
    tcha[6] = 'T'
    RT.stats['channel'] = ''.join(tcha)
    rtz.append(RT)
    
    RZ = st[0].copy()
    RZ.data = c1*st[2].data - s1*st[5].data
    tcha = list(RZ.stats['channel'])
    tcha[2] = 'R'
    tcha[6] = 'Z'
    RZ.stats['channel'] = ''.join(tcha)
    rtz.append(RZ)
    
    TR = st[0].copy()
    TR.data = s1*c2*st[0].data - s1*s2*st[1].data + c1*c2*st[3].data - c1*s2*st[4].data
    tcha = list(TR.stats['channel'])
    tcha[2] = 'T'
    tcha[6] = 'R'
    TR.stats['channel'] = ''.join(tcha)
    rtz.append(TR)
    
    TT = st[0].copy()
    TT.data = s1*s2*st[0].data + s1*c2*st[1].data + c1*s2*st[3].data + c1*c2*st[4].data
    tcha = list(TT.stats['channel'])
    tcha[2] = 'T'
    tcha[6] = 'T'
    TT.stats['channel'] = ''.join(tcha)
    rtz.append(TT)
    
    TZ = st[0].copy()
    TZ.data = s1*st[2].data + c1*st[5].data
    tcha = list(TZ.stats['channel'])
    tcha[2] = 'T'
    tcha[6] = 'Z'
    TZ.stats['channel'] = ''.join(tcha)
    rtz.append(TZ)
    
    ZR = st[0].copy()
    ZR.data = c2*st[6].data - s2*st[7].data
    tcha = list(ZR.stats['channel'])
    tcha[2] = 'Z'
    tcha[6] = 'R'
    ZR.stats['channel'] = ''.join(tcha)
    rtz.append(ZR)
    
    ZT = st[0].copy()
    ZT.data = s2*st[6].data + c2*st[7].data
    tcha = list(ZT.stats['channel'])
    tcha[2] = 'Z'
    tcha[6] = 'T'
    ZT.stats['channel'] = ''.join(tcha)
    rtz.append(ZT)
    
    rtz.append(st[8].copy())
    
    return rtz


def set_sample_options():
    args = {'TDpreProcessing':[{'function':detrend,
                                'args':{'type':'linear'}},
                               {'function':taper,
                                'args':{'type':'cosine_taper',
                                        'p':0.01}},
                               {'function':mute,
                                'args':{'filter':{'type':'bandpass',
                                                 'freqmin':1.,
                                                 'freqmax':6.},
                                       'taper_len':1.,
                                       # 'threshold':1000, absolute threshold
                                       'std_factor':1,
                                       'extend_gaps':True}},
                               {'function':TDfilter,
                                'args':{'type':'bandpass',
                                        'freqmin':1.,
                                        'freqmax':3.}},
                               {'function':TDnormalization,
                                'args':{'filter':{'type':'bandpass',
                                                 'freqmin':0.5,
                                                 'freqmax':2.},
                                        'windowLength':1.}},
                               {'function':signBitNormalization,
                                'args':{}}
                                 ],
            'FDpreProcessing':[{'function':spectralWhitening,
                                'args':{}},
                               {'function':FDfilter,
                                'args':{'flimit':[0.5, 1., 5., 7.]}}],
            'lengthToSave':20,
            'center_correlation':True,      # make sure zero correlation time is in the center
            'normalize_correlation':True,
            'combinations':[(0,0),(0,1),(0,2),(1,2)]}
              
    return args
