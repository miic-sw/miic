from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize("miic/core/pxcorr_func.pyx"))


## module purge
## module load gcc
## intel creates an error.
## python setup.py build_ext --inplace
