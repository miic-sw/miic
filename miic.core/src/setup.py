#!/usr/bin/env python
"""
@author:
Eraldo Pomponi

@copyright:
2011 - Eraldo Pomponi 
eraldo.pomponi@gmail.com

@license:
All rights reserved

Created on Oct 8, 2011
"""

from os.path import join
from setuptools import setup, find_packages

info = {}
execfile(join('miic', '__init__.py'), info)

setup(name='MIIC App',
      version=info['__version__'],
      description='MIIC project application',
      author='Eraldo Pomponi',
      author_email='eraldo.pomponi@gmail.com',
      url='http://theo1.geo.uni-leipzig.de/',
      classifiers=[c.strip() for c in """\
        Development Status :: 5 - Production/Stable
        Intended Audience :: Developers
        Intended Audience :: Science/Research
        License :: OSI Approved :: BSD License
        Operating System :: MacOS
        Operating System :: Microsoft :: Windows
        Operating System :: OS Independent
        Operating System :: POSIX
        Operating System :: Unix
        Programming Language :: Python
        Topic :: Scientific/Engineering
        Topic :: Software Development
        Topic :: Software Development :: Libraries
        """.splitlines() if len(c.strip()) > 0],
      packages=find_packages(),
      platforms=["Windows", "Linux", "Mac OS-X", "Unix", "Solaris"],
      zip_safe=False,
     )

