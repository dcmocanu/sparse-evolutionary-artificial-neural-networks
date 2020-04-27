# compile this file with: "cythonize -a -i sparseoperations.pyx"
# I have tested this method in Linux (Ubuntu). If you compile it in Windows you may need some work around.
# For widows users run: "python setup.py install"

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(["sparse_operations/sparseoperations.pyx"], annotate=True),
    include_dirs=[numpy.get_include()]
)