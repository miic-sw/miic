Correlation
===========
This modules collect all those functions that are necessary to correlate
seismic traces. It is possible to fully exploit multicore CPU capability
working in parallel. The base structure that collects the traces that must
be correlated can be chosen between :class:`~obspy.core.stream.Stream` or 
classic numpy :class:`~numpy.ndarray` array.

@author:
Eraldo Pomponi

@copyright:
The MIIC Development Team (eraldo.pomponi@uni-leipzig.de)

@license:
GNU Lesser General Public License, Version 3
(http://www.gnu.org/copyleft/lesser.html)

Created on Nov 25, 2010

.. currentmodule:: miic.core.corr_fun
.. automodule:: miic.core.corr_fun
        
    .. comment to end block
        
    BlockCanvas oriented
    --------------------
    .. autosummary::
       :toctree: autogen
       :nosignatures:
       
       ~corr_trace_fun
       ~norm_corr_stream
       
    .. comment to end block
       
       
    Basic
    -----
    .. autosummary::
       :toctree: autogen
       :nosignatures:  

       ~conv_traces
       
    .. comment to end block
       
       
    Helper functions
    ----------------
    .. autosummary::
       :toctree: autogen
       :nosignatures:    
       
       ~extend
       ~merge_id
       ~combine_stats
    
    .. comment to end block
