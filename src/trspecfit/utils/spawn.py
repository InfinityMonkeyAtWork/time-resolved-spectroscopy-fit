"""
Shared multiprocessing helpers for spawn-backed worker pools.

Both the Slice-by-Slice executor (``File.fit_slice_by_slice``) and the
MCMC worker pool (``fitlib.fit_wrapper``) start workers with the ``spawn``
method and need the same ``__main__`` protection, so the helper lives in
this neutral module rather than in either caller.
"""

from __future__ import annotations

import contextlib
import sys


#
@contextlib.contextmanager
def sanitized_spawn_main():
    """Hide a non-importable ``__main__.__file__`` from spawn workers.

    When a notebook is executed via IPython's ``%run example.ipynb``,
    ``__main__.__file__`` points at the notebook JSON; multiprocessing's
    spawn ``prepare()`` would re-run that path via ``runpy`` in every
    worker and crash (the JSON is not Python). Spawn workers never need
    ``__main__`` content here — everything they use is installed from
    trspecfit modules (SbS via ``sbs_worker_init``, MCMC via the pickled
    objective) — so drop the attribute for the pool's lifetime and restore
    it afterwards. A regular ``python script.py`` main keeps its ``.py``
    ``__file__`` untouched.
    """

    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        yield
        return
    main_file = getattr(main_mod, "__file__", None)
    sanitize = main_file is not None and not str(main_file).endswith(".py")
    if sanitize:
        del main_mod.__file__
    try:
        yield
    finally:
        if sanitize:
            main_mod.__file__ = main_file
