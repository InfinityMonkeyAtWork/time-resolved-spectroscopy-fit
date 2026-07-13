# Pin BLAS/OpenMP to one thread per process BEFORE numpy is imported
# anywhere: the suite runs parallel by default (pytest-xdist, see addopts
# in pyproject.toml), and per-worker BLAS thread pools oversubscribe the
# machine (measured 2x wall-time cost). setdefault keeps explicit user
# overrides working.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
