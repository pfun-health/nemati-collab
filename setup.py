"""Build configuration for the compiled model library.

The ODE model is plain C with no external dependencies. It is compiled as a
shared library that ``pycardio.model`` loads through :mod:`ctypes`; it is
*not* a CPython extension module and is never imported directly.
"""

import sys

from setuptools import Extension, setup

extra_compile_args = [] if sys.platform == "win32" else ["-O3", "-std=c11", "-fvisibility=hidden"]
libraries = [] if sys.platform == "win32" else ["m"]

setup(
    ext_modules=[
        Extension(
            "pycardio._libmodel",
            sources=["pycardio/_c/model.c"],
            depends=["pycardio/_c/model_params.h"],
            extra_compile_args=extra_compile_args,
            libraries=libraries,
        )
    ]
)
