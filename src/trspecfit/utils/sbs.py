"""
Slice-by-Slice (SbS) multiprocessing helpers.

Worker-process plumbing for ``File.fit_slice_by_slice()``. Workers are
spawned by ``ProcessPoolExecutor``; ``sbs_worker_init`` runs once per
worker to install a shared model and dispatch args as worker-local
globals, and ``sbs_fit_one_slice`` consumes them to fit a single slice.

The module also exposes the seed-handling helpers used in the serial
path (``extract_sbs_seed_template``, ``prepare_sbs_model_for_slice``).
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from trspecfit import fitlib
from trspecfit.config.plot import PlotConfig
from trspecfit.utils import lmfit as ulmfit

if TYPE_CHECKING:
    from trspecfit import mcp

# These globals are populated only inside ProcessPoolExecutor worker
# processes by ``sbs_worker_init``. They let workers reuse a single
# pickled Model and dispatch_args across all slices they process,
# avoiding per-task pickle overhead.
_WORKER_MODEL: mcp.Model | None = None
_WORKER_DISPATCH_ARGS: tuple[Any, ...] | None = None
_WORKER_SEED_TEMPLATE: list[float] | None = None


#
def extract_sbs_seed_template(
    seed_values: Any, parameter_names: Sequence[str]
) -> list[float]:
    """Normalize explicit SbS seed values to a full ordered parameter list."""

    if isinstance(seed_values, dict):
        missing = [name for name in parameter_names if name not in seed_values]
        extra = [name for name in seed_values if name not in parameter_names]
        if missing or extra:
            raise ValueError(
                "Explicit SbS seed dict must define exactly the model parameters.\n"
                f"Missing: {missing or 'none'}\n"
                f"Extra: {extra or 'none'}"
            )
        out: list[float] = []
        for name in parameter_names:
            value = seed_values[name]
            if isinstance(value, (list, tuple, np.ndarray)):
                if len(value) == 0:
                    raise ValueError(
                        f"Explicit SbS seed for parameter '{name}' is empty."
                    )
                value = value[0]
            out.append(float(value))
        return out

    out = ulmfit.par_extract(seed_values, return_type="list")
    if len(out) != len(parameter_names):
        raise ValueError(
            "Explicit SbS seed must provide one value per model parameter.\n"
            f"Expected {len(parameter_names)} values, got {len(out)}."
        )
    return out


#
def prepare_sbs_model_for_slice(
    model: mcp.Model,
    dispatch_args: tuple[Any, ...],
    seed_template: list[float],
    *,
    seed_adapt: Literal["argmax_shift"] | None,
    s: np.ndarray,
    energy: np.ndarray,
    e_lim: list[int] | None,
    e_pos_pars: list[str],
    e_pos_vals: pd.Series | None,
    data_base_argmax_energy: float | None,
    fit_fun_str: str,
) -> list[float]:
    """Reset the shared SbS model to the seed template for one slice."""

    model.update_value(new_par_values=seed_template, par_select="all")

    if seed_adapt == "argmax_shift":
        if e_pos_vals is None or data_base_argmax_energy is None:
            raise ValueError(
                "argmax_shift seed adaptation requires baseline-derived x0 values."
            )
        delta_max = energy[np.argmax(s)] - data_base_argmax_energy
        new_e_vals = list(e_pos_vals.add(delta_max))
        model.update_value(new_par_values=new_e_vals, par_select=e_pos_pars)

    initial_guess = ulmfit.par_extract(model.lmfit_pars, return_type="list")
    model.const = (energy, s, fit_fun_str, 0, e_lim, [])
    model.args = dispatch_args
    return initial_guess


#
def sbs_worker_init(
    model: mcp.Model,
    dispatch_args: tuple[Any, ...],
    seed_template: list[float],
) -> None:
    """Executor initializer: install per-worker model and Agg backend.

    Runs once per worker process before any task. Stashes the deep-pickled
    model and GIR/MCP dispatch args as worker-local globals so individual
    slice tasks don't have to pay the pickle cost on every submission.
    Forces matplotlib to the non-interactive Agg backend so the per-slice
    plot calls inside workers don't try to open a display.
    """

    global _WORKER_MODEL, _WORKER_DISPATCH_ARGS, _WORKER_SEED_TEMPLATE
    import matplotlib

    matplotlib.use("Agg", force=True)
    _WORKER_MODEL = model
    _WORKER_DISPATCH_ARGS = dispatch_args
    _WORKER_SEED_TEMPLATE = seed_template


#
def sbs_fit_one_slice(
    s_i: int,
    s: np.ndarray,
    *,
    energy: np.ndarray,
    e_lim: list[int] | None,
    seed_adapt: Literal["argmax_shift"] | None,
    e_pos_pars: list[str],
    e_pos_vals: pd.Series | None,
    data_base_argmax_energy: float | None,
    fit_fun_str: str,
    stages: int,
    path_slice: pathlib.Path,
    plot_config: PlotConfig,
    fit_wrapper_kwargs: dict[str, Any],
) -> tuple[int, list[Any]]:
    """Fit one energy slice in a worker process.

    Uses worker-local ``_WORKER_MODEL`` and ``_WORKER_DISPATCH_ARGS``
    installed by :func:`sbs_worker_init`. Mutates the worker's model
    state (x0 params, ``const``, ``args``) — this is safe because tasks
    within a single worker run sequentially and each slice overwrites
    the relevant fields.

    Returns
    -------
    tuple[int, list]
        (slice_index, fit_wrapper_result) so the caller can reassemble
        out-of-order completions back into slice order.
    """

    assert _WORKER_MODEL is not None, "sbs_worker_init must run first"
    assert _WORKER_DISPATCH_ARGS is not None
    assert _WORKER_SEED_TEMPLATE is not None
    model = _WORKER_MODEL
    dispatch_args = _WORKER_DISPATCH_ARGS
    seed_template = _WORKER_SEED_TEMPLATE

    initial_guess = prepare_sbs_model_for_slice(
        model,
        dispatch_args,
        seed_template,
        seed_adapt=seed_adapt,
        s=s,
        energy=energy,
        e_lim=e_lim,
        e_pos_pars=e_pos_pars,
        e_pos_vals=e_pos_vals,
        data_base_argmax_energy=data_base_argmax_energy,
        fit_fun_str=fit_fun_str,
    )
    const = model.const
    args = model.args
    assert const is not None
    assert args is not None

    result_sbs = fitlib.fit_wrapper(
        const=const,
        args=args,
        par_names=model.parameter_names,
        par=model.lmfit_pars,
        stages=stages,
        show_output=0,
        save_output=1,
        save_path=path_slice,
        **fit_wrapper_kwargs,
    )

    fitlib.plt_fit_res_1d(
        x=const[0],
        y=const[1],
        fit_fun_str=fit_fun_str,
        par_init=initial_guess,
        par_fin=result_sbs[1],
        args=args,
        plot_sum=False,
        show_init=True,
        fit_lim=e_lim,
        config=plot_config,
        save_img=-1,
        save_path=path_slice.with_suffix(".png"),
    )

    return s_i, result_sbs
