"""Benchmark: GIR compiled evaluator vs interpreter (MCP).

Measures per-call evaluation time and full-fit time for both paths
using an example fitting workflow.

Usage:
    .venv/bin/python benchmark_gir.py                # per-call only, example 02
    .venv/bin/python benchmark_gir.py --example 2    # explicit example number
    .venv/bin/python benchmark_gir.py --fit          # include full-fit benchmark
    .venv/bin/python benchmark_gir.py --fit -n 5     # 5 fit repetitions
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path

# Keep matplotlib cache writes out of unwritable home config dirs.
if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path(tempfile.gettempdir()) / "trspecfit-matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from trspecfit import File, Project, spectra
from trspecfit.graph_ir import build_graph, can_lower_2d, schedule_2d

EXAMPLES_ROOT = Path("examples/fitting_workflows")
_STRING_PATTERN = re.compile(r"""(['"])(.*?)\1""")


# ------------------------------------------------------------------
# Notebook parser
# ------------------------------------------------------------------


#
def _extract_str(src, key):
    """Extract a single quoted string keyword argument from source text."""

    match = re.search(rf'{key}\s*=\s*([\'"])(.*?)\1', src)
    return match.group(2) if match else None


#
def _extract_str_or_list(src, key):
    """Extract either a single string or a list of strings from source text."""

    match = re.search(rf"{key}\s*=\s*\[([^\]]+)\]", src)
    if match:
        return [
            item_match.group(2)
            for item_match in _STRING_PATTERN.finditer(match.group(1))
        ]
    return _extract_str(src, key)


#
def _extract_float(src, key):
    """Extract a float keyword argument from source text."""

    match = re.search(rf"{key}\s*=\s*([0-9.eE+-]+)", src)
    return float(match.group(1)) if match else None


#
def _parse_dynamics_from_notebook(notebook_path):
    """Extract add_time_dependence kwargs from example.ipynb.

    Returns
    -------
    list of dict
        Each dict has keys: target_model, target_parameter,
        dynamics_yaml, dynamics_model, and optionally frequency.
    """

    with notebook_path.open() as f:
        nb = json.load(f)

    calls = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "add_time_dependence" not in src:
            continue

        call = {
            "target_model": _extract_str(src, "target_model"),
            "target_parameter": _extract_str(src, "target_parameter"),
            "dynamics_yaml": _extract_str(src, "dynamics_yaml"),
            "dynamics_model": _extract_str_or_list(src, "dynamics_model"),
        }
        freq = _extract_float(src, "frequency")
        if freq is not None:
            call["frequency"] = freq
        calls.append(call)

    return calls


# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------


#
def _find_example_folder(example_num):
    """Find the example folder matching the given number."""

    prefix = f"{example_num:02d}_"
    for d in sorted(EXAMPLES_ROOT.iterdir()):
        if d.is_dir() and d.name.startswith(prefix) and "_fits" not in d.name:
            return d
    raise FileNotFoundError(f"No example folder matching {prefix}* in {EXAMPLES_ROOT}")


#
def load_example(example_num, *, add_dynamics=True):
    """Load an example fitting workflow as a 2D model.

    Parameters
    ----------
    example_num : int
        Example number (1-5).
    add_dynamics : bool
        If True, add time dependence from the notebook.

    Returns
    -------
    file : File
    dynamics_calls : list of dict
        Parsed add_time_dependence kwargs from the notebook.
    """

    folder = _find_example_folder(example_num)
    data_dir = folder / "data"

    project = Project(path=str(folder), name="bench")
    project.show_output = 0

    energy = np.loadtxt(data_dir / "energy.csv")
    time_ax = np.loadtxt(data_dir / "time.csv")
    data = np.loadtxt(data_dir / "data.csv", delimiter=",")

    file = File(
        parent_project=project,
        name="bench",
        data=data,
        energy=energy,
        time=time_ax,
    )
    file.load_model(model_yaml="models_energy.yaml", model_info="2D")

    dynamics_calls = _parse_dynamics_from_notebook(folder / "example.ipynb")

    if add_dynamics:
        for call in dynamics_calls:
            file.add_time_dependence(**call)

    return file, dynamics_calls


#
def prepare_paths(file):
    """Build GIR plan and interpreter args from a 2D file."""

    model = file.model_active
    assert model is not None

    graph = build_graph(model)
    lowerable = can_lower_2d(graph)
    if not lowerable:
        return model, None, None, None, False

    plan = schedule_2d(graph)
    name_to_idx = {n: i for i, n in enumerate(model.parameter_names)}
    theta_indices = np.array(
        [name_to_idx[n] for n in plan.opt_param_names], dtype=np.intp
    )
    par = [model.lmfit_pars[n].value for n in model.parameter_names]

    return model, par, plan, theta_indices, True


# ------------------------------------------------------------------
# Per-call benchmark
# ------------------------------------------------------------------


#
def bench_per_call(file, *, n_warmup=5, n_calls=200):
    """Time a single evaluate call for both paths."""

    model, par, plan, theta_indices, lowerable = prepare_paths(file)
    if not lowerable:
        print("Model is not lowerable by GIR — skipping per-call benchmark.")
        return None, None

    energy = file.energy

    # --- GIR path ---
    for _ in range(n_warmup):
        spectra.fit_model_gir(energy, par, True, plan, theta_indices)

    t0 = time.perf_counter()
    for _ in range(n_calls):
        spectra.fit_model_gir(energy, par, True, plan, theta_indices)
    gir_total = time.perf_counter() - t0
    gir_per_call = gir_total / n_calls

    # --- Interpreter path ---
    for _ in range(n_warmup):
        spectra.fit_model_mcp(energy, par, True, model, 2)

    t0 = time.perf_counter()
    for _ in range(n_calls):
        spectra.fit_model_mcp(energy, par, True, model, 2)
    mcp_total = time.perf_counter() - t0
    mcp_per_call = mcp_total / n_calls

    # --- Correctness check ---
    fast = spectra.fit_model_gir(energy, par, True, plan, theta_indices)
    slow = spectra.fit_model_mcp(energy, par, True, model, 2)
    max_diff = np.max(np.abs(fast - slow))

    print("=" * 60)
    print("PER-CALL EVALUATION BENCHMARK")
    print(f"  Grid:       {len(file.time)} time x {len(file.energy)} energy")
    print(f"  Calls:      {n_calls} (+ {n_warmup} warmup)")
    print(f"  GIR:        {gir_per_call * 1e3:8.2f} ms/call")
    print(f"  Interpreter:{mcp_per_call * 1e3:8.2f} ms/call")
    print(f"  Speedup:    {mcp_per_call / gir_per_call:8.2f}x")
    print(f"  Max |diff|: {max_diff:.2e}")
    print("=" * 60)

    return gir_per_call, mcp_per_call


# ------------------------------------------------------------------
# GIR-only loop (for external profilers like py-spy)
# ------------------------------------------------------------------


#
def bench_gir_only(file, *, n_warmup=50, n_calls=5000):
    """Run the GIR evaluator in a tight loop with no interpreter path.

    Purpose is to give a sampling profiler a clean window in which only
    the GIR code path executes -- no MCP calls, no correctness check,
    no prints inside the hot loop. The per-call timing is still
    reported at the end as a sanity anchor, but is not the primary
    output: the flamegraph from py-spy is.

    Parameters
    ----------
    file
        Loaded 2D file (see ``load_example``).
    n_warmup
        Warmup calls before the timed window.
    n_calls
        Timed calls. At ~2-3 ms/call and 500 Hz sampling, 5000 calls
        gives ~5000 samples -- enough to resolve buckets above ~1%.
    """

    model, par, plan, theta_indices, lowerable = prepare_paths(file)
    if not lowerable:
        raise RuntimeError("Model is not lowerable -- GIR profile has no target.")

    energy = file.energy

    for _ in range(n_warmup):
        spectra.fit_model_gir(energy, par, True, plan, theta_indices)

    t0 = time.perf_counter()
    for _ in range(n_calls):
        spectra.fit_model_gir(energy, par, True, plan, theta_indices)
    total = time.perf_counter() - t0

    print("=" * 60)
    print("GIR-ONLY PROFILE LOOP")
    print(f"  Grid:       {len(file.time)} time x {len(file.energy)} energy")
    print(f"  Warmup:     {n_warmup}")
    print(f"  Calls:      {n_calls}")
    print(f"  Total:      {total:8.2f} s")
    print(f"  Per-call:   {total / n_calls * 1e3:8.2f} ms")
    print("=" * 60)


# ------------------------------------------------------------------
# nfev capture (decision input for Jacobian payoff)
# ------------------------------------------------------------------


#
def capture_nfev(example_num):
    """Run the standard baseline + 2D pipeline and capture residual counts.

    Wraps ``fitlib.residual_fun`` with a counter. Each lmfit
    ``minimize`` call is one optimization stage; the counter partitions
    evaluations per stage. Totals are the ceiling on any Jacobian
    payoff: if a fit converges in N evals, a 3x nfev cut from Dfun
    caps Jacobian win at ~3x at fit level.
    """

    from trspecfit import fitlib

    counts: list[int] = []
    call_count = [0]
    orig_residual = fitlib.residual_fun

    #
    def counting_residual(*args, **kwargs):
        call_count[0] += 1
        return orig_residual(*args, **kwargs)

    fitlib.residual_fun = counting_residual

    #
    def _snapshot(label):
        prev = counts[-1] if counts else 0
        counts.append(call_count[0])
        print(f"  {label:32s}{call_count[0]:6d}  (+{call_count[0] - prev})")

    try:
        file, dynamics_calls = load_example(example_num, add_dynamics=False)
        file.define_baseline(
            time_start=0, time_stop=10, time_type="ind", show_plot=False
        )

        print(f"  {'stage':32s}{'total':>6s}  (+delta)")
        file.fit_baseline(model_name="2D", stages=2, try_ci=0)
        _snapshot("after fit_baseline (stages=2)")

        for call in dynamics_calls:
            file.add_time_dependence(**call)

        file.fit_2d(model_name="2D", stages=2, try_ci=0)
        _snapshot("after fit_2d (stages=2)")
    finally:
        fitlib.residual_fun = orig_residual

    print()
    print(f"  Total residual evaluations: {call_count[0]}")
    return call_count[0]


# ------------------------------------------------------------------
# Plan-build timing (is the plan builder itself a bottleneck?)
# ------------------------------------------------------------------


#
def capture_plan_time(example_num):
    """Measure ``build_graph`` + ``schedule_2d`` cost vs total fit_2d time.

    Wraps both functions on the ``trspecfit.graph_ir`` module so that
    ``File.fit_2d``'s local ``from trspecfit.graph_ir import ...`` picks
    up the timed versions on its next import. Reports plan-build time
    as an absolute number and as a fraction of total fit_2d wall time.
    A plan / fit ratio > ~10%% means evaluator-only optimization leaves
    material work unoptimized.
    """

    from trspecfit import graph_ir

    orig_build = graph_ir.build_graph
    orig_sched = graph_ir.schedule_2d
    times = {"build_graph": 0.0, "schedule_2d": 0.0, "build_calls": 0}

    #
    def timed_build(*args, **kwargs):
        t0 = time.perf_counter()
        r = orig_build(*args, **kwargs)
        times["build_graph"] += time.perf_counter() - t0
        times["build_calls"] += 1
        return r

    #
    def timed_sched(*args, **kwargs):
        t0 = time.perf_counter()
        r = orig_sched(*args, **kwargs)
        times["schedule_2d"] += time.perf_counter() - t0
        return r

    graph_ir.build_graph = timed_build
    graph_ir.schedule_2d = timed_sched

    try:
        file, dynamics_calls = load_example(example_num, add_dynamics=False)
        file.define_baseline(
            time_start=0, time_stop=10, time_type="ind", show_plot=False
        )
        file.fit_baseline(model_name="2D", stages=2, try_ci=0)
        for call in dynamics_calls:
            file.add_time_dependence(**call)

        t0 = time.perf_counter()
        file.fit_2d(model_name="2D", stages=2, try_ci=0)
        fit_time = time.perf_counter() - t0
    finally:
        graph_ir.build_graph = orig_build
        graph_ir.schedule_2d = orig_sched

    plan_total = times["build_graph"] + times["schedule_2d"]
    ratio = plan_total / fit_time if fit_time > 0 else 0.0

    print(
        f"  build_graph:  {times['build_graph'] * 1000:8.2f} ms "
        f"({times['build_calls']} call(s))"
    )
    print(f"  schedule_2d:  {times['schedule_2d'] * 1000:8.2f} ms")
    print(f"  plan total:   {plan_total * 1000:8.2f} ms")
    print(f"  fit_2d wall:  {fit_time:8.2f} s")
    print(f"  plan / fit:   {ratio * 100:8.3f} %")

    return plan_total, fit_time


# ------------------------------------------------------------------
# Full-fit benchmark
# ------------------------------------------------------------------


#
def bench_fit(example_num, dynamics_calls, *, n_reps=3):
    """Time a complete fit_2d for both paths."""

    gir_times = []
    mcp_times = []

    for i in range(n_reps):
        for fun_str, times_list in [
            ("fit_model_gir", gir_times),
            ("fit_model_mcp", mcp_times),
        ]:
            file, _ = load_example(example_num, add_dynamics=False)
            file.define_baseline(
                time_start=0, time_stop=10, time_type="ind", show_plot=False
            )
            file.fit_baseline(model_name="2D", stages=2, try_ci=0)
            for call in dynamics_calls:
                file.add_time_dependence(**call)
            file.p.spec_fun_str = fun_str
            t0 = time.perf_counter()
            file.fit_2d(model_name="2D", stages=2, try_ci=0)
            times_list.append(time.perf_counter() - t0)

        print(
            f"  Rep {i + 1}/{n_reps}: "
            f"GIR={gir_times[-1]:.2f}s  MCP={mcp_times[-1]:.2f}s"
        )

    gir_med = np.median(gir_times)
    mcp_med = np.median(mcp_times)
    formatted_gir_times = [f"{t:.2f}" for t in gir_times]
    formatted_mcp_times = [f"{t:.2f}" for t in mcp_times]

    print()
    print("=" * 60)
    print("FULL FIT BENCHMARK")
    print(f"  Reps:       {n_reps}")
    print(f"  GIR:        {gir_med:8.2f} s (median)")
    print(f"  Interpreter:{mcp_med:8.2f} s (median)")
    print(f"  Speedup:    {mcp_med / gir_med:8.2f}x")
    print(f"  GIR all:    {formatted_gir_times}")
    print(f"  MCP all:    {formatted_mcp_times}")
    print("=" * 60)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GIR vs interpreter benchmark",
    )
    parser.add_argument(
        "--example",
        type=int,
        default=2,
        help="Example number (default: 2 = dependent_parameters)",
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help="Run full-fit benchmark",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=3,
        help="Fit repetitions (default: 3)",
    )
    parser.add_argument(
        "--calls",
        type=int,
        default=200,
        help="Per-call iterations (default: 200)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Run GIR-only loop for external profilers (e.g. py-spy). "
            "Skips the interpreter path and correctness check. Use "
            "--calls to control loop length (default 5000 when "
            "--profile is set)."
        ),
    )
    parser.add_argument(
        "--nfev",
        action="store_true",
        help=(
            "Run standard baseline + fit_2d pipeline and report "
            "total residual evaluations. If --example is 0, runs "
            "all examples and reports a summary."
        ),
    )
    parser.add_argument(
        "--plan-time",
        action="store_true",
        help=(
            "Measure build_graph + schedule_2d cost vs total fit_2d "
            "wall time. If --example is 0, runs all examples."
        ),
    )
    args = parser.parse_args()

    # --nfev / --plan-time with example 0 means "all examples"; skip the
    # single-example preamble so the capture fn can load each one itself.
    skip_preamble = (args.nfev or args.plan_time) and args.example == 0
    if not skip_preamble:
        folder = _find_example_folder(args.example)
        print(f"Example: {folder.name}")

        file, dynamics_calls = load_example(args.example)
        model = file.model_active
        assert model is not None
        graph = build_graph(model)
        print(f"  lowerable: {can_lower_2d(graph)}")
        for call in dynamics_calls:
            target_parameter = call["target_parameter"]
            dynamics_model = call["dynamics_model"]
            print(f"  dynamics:  {target_parameter} <- {dynamics_model}")
        print()

    if args.profile:
        # Profile-mode default is larger than the per-call default so
        # py-spy has enough samples; honor --calls if the user set it.
        profile_calls = args.calls if args.calls != 200 else 5000
        bench_gir_only(file, n_calls=profile_calls)
    elif args.nfev:
        if args.example == 0:
            totals: dict[int, int | str] = {}
            for n in range(1, 6):
                print()
                print("=" * 60)
                print(f"NFEV CAPTURE -- example {n:02d}")
                print("=" * 60)
                try:
                    totals[n] = capture_nfev(n)
                except FileNotFoundError as e:
                    print(f"  skipped: {e}")
                    totals[n] = "skipped"
            print()
            print("=" * 60)
            print("NFEV SUMMARY")
            for n, total in totals.items():
                total_str = f"{total:6d}" if isinstance(total, int) else f"{total:>6s}"
                print(f"  example {n:02d}:  {total_str} residual evals")
            print("=" * 60)
        else:
            print("=" * 60)
            print(f"NFEV CAPTURE -- example {args.example:02d}")
            print("=" * 60)
            capture_nfev(args.example)
    elif args.plan_time:
        if args.example == 0:
            ratios: dict[int, str] = {}
            for n in range(1, 6):
                print()
                print("=" * 60)
                print(f"PLAN-TIME CAPTURE -- example {n:02d}")
                print("=" * 60)
                try:
                    plan_t, fit_t = capture_plan_time(n)
                    ratios[n] = (
                        f"{plan_t * 1000:7.2f} ms plan / "
                        f"{fit_t:5.2f} s fit "
                        f"= {plan_t / fit_t * 100:6.3f}%"
                    )
                except FileNotFoundError as e:
                    print(f"  skipped: {e}")
                    ratios[n] = "skipped"
            print()
            print("=" * 60)
            print("PLAN-TIME SUMMARY")
            for n, ratio_str in ratios.items():
                print(f"  example {n:02d}:  {ratio_str}")
            print("=" * 60)
        else:
            print("=" * 60)
            print(f"PLAN-TIME CAPTURE -- example {args.example:02d}")
            print("=" * 60)
            capture_plan_time(args.example)
    else:
        bench_per_call(file, n_calls=args.calls)

        if args.fit:
            print()
            bench_fit(args.example, dynamics_calls, n_reps=args.n)
