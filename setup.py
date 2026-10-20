import os
import sys
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

# HiGHS headers and library: $HIGHS_PREFIX if set, else the active conda
# environment, else the Python prefix.
conda_prefix = os.environ.get("HIGHS_PREFIX") or os.environ.get("CONDA_PREFIX", sys.prefix)

include_dirs = [
    os.path.join(conda_prefix, "include"),
    os.path.join(conda_prefix, "include", "highs")
]

library_dirs = [
    os.path.join(conda_prefix, "lib")
]

extra_compile_args = ["-O3", "-std=c++14", "-fPIC"]
extra_link_args = []
libraries = ["highs"]

if sys.platform == "darwin":
    extra_compile_args.extend(["-Xpreprocessor", "-fopenmp"])
    extra_link_args.extend(["-L" + os.path.join(conda_prefix, "lib"), "-lomp"])
elif sys.platform == "win32":
    extra_compile_args = ["/O2", "/openmp", "/std:c++14"]
else:  # Linux
    extra_compile_args.extend(["-fopenmp"])
    extra_link_args.extend(["-fopenmp"])

if sys.platform != "win32":
    # record the HiGHS library folder in the extension (rpath), so that
    # LD_LIBRARY_PATH / DYLD_LIBRARY_PATH need not be set at run time
    extra_link_args.append("-Wl,-rpath," + os.path.join(conda_prefix, "lib"))

ext_modules = [
    Pybind11Extension(
        "maclp._fast_aabb_cpp",  # pack into maclp
        ["src/bindings.cpp", "src/aabb_solver.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="maclp",
    version="0.1.0",
    description="Matrix constraints for Label Shift, including BBSE, LP & GE Estimators",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "numba",
    ],
)