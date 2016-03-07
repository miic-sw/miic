Inversion
=========
This module collects the functions necessary to solve the inverse problem
that arise when detecting the real velocity change in a duffusive medium. 

@author:
Eraldo Pomponi

@copyright:
The MIIC Development Team (eraldo.pomponi@uni-leipzig.de)

@license:
GNU Lesser General Public License, Version 3
(http://www.gnu.org/copyleft/lesser.html)

.. currentmodule:: miic.core.inversion
.. automodule:: miic.core.inversion

    .. comment to end block
    
    BlockCanvas oriented
    --------------------
    .. autosummary::
       :toctree: autogen
       :nosignatures:
       
       ~resolution_matrix
       ~resolution_matrix_reduced
       ~invert
       ~quantify_vchange_drop
       ~reconstruction_error
       ~inversion_with_missed_data
       
    .. comment to end block
    
          
    Helper functions
    ----------------
    .. autosummary::
       :toctree: autogen
       :nosignatures:
       
       ~hard_threshold_dv
        
