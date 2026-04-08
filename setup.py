"""
Build script for the hamr_read.pp_c Cython extension.
All project metadata lives in pyproject.toml.
Install with: pip install .
"""

import sys
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

_is_mac = sys.platform == "darwin"

ext = Extension(
    name="hamr_read.pp_c",
    sources=["hamr_read/pp_c.pyx", "hamr_read/functions.c"],
    include_dirs=[np.get_include()],
    extra_compile_args=[] if _is_mac else ["-fopenmp"],
    extra_link_args=["-O2"] if _is_mac else ["-O2", "-fopenmp"],
)

setup(
    ext_modules=cythonize(
        [ext],
        compiler_directives={"language_level": "3"},
    ),
)
