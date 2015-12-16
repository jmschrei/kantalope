# setup.py
# Contact: Jacob Schreiber <jmschr@cs.washington.edu>

import numpy
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[
    Extension("kantalope",
              ["kantalope/model.pyx"],
              extra_compile_args = ["-O3", "-fopenmp" ],
              extra_link_args=['-fopenmp'],
              include_dirs=[numpy.get_include()]
              ) 
]

setup( 
  name = "kantalope",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules
)