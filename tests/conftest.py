"""Shared pytest configuration for the siggi test suite.

Limit the per-process BLAS/OpenMP thread pools *before* numpy (and the
libraries it pulls in) are imported. The optimization tests in
``test_siggi.py`` fan out across several worker processes via joblib's
multiprocessing backend; without this, each worker spawns a full set of BLAS
threads, massively oversubscribing the CPU. That oversubscription makes the
tests crawl and, because the workers are forked from an already multi-threaded
process, can occasionally deadlock. Pinning the thread pools to one thread per
process removes both problems.
"""

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")
