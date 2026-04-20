# Execution Plan: Lowered Evaluator

## Motivation

A 2D fit (400 energy x 440 time, 4 free params, 556 optimizer iterations)
takes ~10 seconds. Profiling shows:

| Category | Self-time | % of fit | Cause |
|---|---|---|---|
| Math (GLP + LinBack) | 3.2 s | 32% | The actual numeric work |
| Parameter overhead | 2.8 s | 28% | `par_extract`, `Par.value`, `valuesdict` -- called 2M times |
| Python dispatch | 1.6 s | 16% | `Component.value`, `combine`, `create_value_1d`, OOP calls |
| Other | 2.6 s | 24% | lmfit internals, array ops, etc. |

**Root cause:** The code acts as an *interpreter* inside the optimizer loop.
`create_value_2d` (mcp.py:866) loops over time steps; at each step
`Component.value` (mcp.py:1518) loops over parameters; each `Par.value`
(mcp.py:1979) dispatches through 5 branches and calls `par_extract` which
calls `valuesdict()`. All of this repeats 440 x 556 = 245k times.

The irony: for time-dependent parameters, the full time trace *already
exists* in `Dynamics.value_1d`. But `Par.value(t_ind)` consumes it one
scalar at a time via `base + self.t_model.value_1d[t_ind]`.

**Goal:** Stop interpreting the model on every residual call. Lower the
model once into a flat, array-oriented execution plan, then evaluate it as
a pure function.


## Architecture overview

```
User API (unchanged)                   Future model-builder UI
    |                                          |
    v                                          v
Model / Component / Par         <-- OOP tree (parsing, validation, user interaction)
    |                                          |
    |  build_graph(model)                      |  (UI will target GraphIR directly)
    v                                          v
GraphIR                         <-- NEW: semantic IR (DAG), axis-agnostic
    |                                   works for 1D and 2D models
    |  can_lower_2d(graph)       <-- gate: can the 2D backend compile this graph?
    |  schedule_2d(graph)        <-- NEW: compile DAG to flat 2D execution schedule
    v
ScheduledPlan2D                 <-- NEW: packed arrays, (n_params, n_time) traces
    |
    |  evaluate_2d(plan, theta)  <-- NEW: pure function, array-oriented
    v
spectrum (n_time, n_energy)
```

Three-layer design:

1. **OOP tree** -- the user-facing `Model`/`Component`/`Par` objects.
   Handles parsing, validation, user interaction. Unchanged.
2. **GraphIR** -- a directed acyclic graph of typed nodes with explicit
   data-dependency edges. This is the *semantic* representation of the
   model. Both the OOP tree and a future drag-and-drop model-builder UI
   can target it. **The graph is axis-agnostic** -- it works for 1D
   energy models, 1D time models, and 2D time+energy models. A user
   builds a 1D energy model first (valid graph, no time axis), then
   adds dynamics (graph gains time-dependent nodes and a time axis),
   making it compilable by the 2D backend. Standalone ``TIME_1D`` graphs
   are still valuable even without a lowered backend because they allow
   a future model-builder UI to author, validate, and save reusable
   dynamics-only YAML models.
3. **ScheduledPlan2D** -- a flat, packed-array execution schedule compiled
   from the graph by the 2D backend. No Python objects, no strings, no
   dicts in the hot path. This is what the evaluator actually runs.
   Backend-specific: a future `ScheduledPlan1D` / `can_lower_1d` /
   `evaluate_1d` can target the same GraphIR with a simpler storage
   model (no `(n_params, n_time)` trace matrix needed for 1D) for
   `ENERGY_1D` models (`fit_baseline`, `fit_spectrum`). `TIME_1D`
   standalone dynamics graphs remain graph-valid but are out of lowered
   backend scope for now.

Key principle: **the OOP tree is for humans; the GraphIR is for semantics;
the ScheduledPlan2D is for the optimizer.**


---
## Spec


### 1. GraphIR -- the semantic intermediate representation

The GraphIR is a DAG (directed acyclic graph) of typed nodes connected by
explicit dependency edges. It captures the full semantics of the model
without prescribing an evaluation strategy.

#### 1.1 Node types

```python
class NodeKind(IntEnum):
    """Node types in the model graph."""

    # --- Parameter nodes (leaves) ---
    STATIC_PARAM = 0        # fixed value, never changes during fit
    OPT_PARAM = 1           # optimizer-visible parameter (lmfit varies it)

    # --- Computed parameter nodes ---
    DYNAMICS_TRACE = 2      # time-dependent trace: evaluates a dynamics
                            # function over the full time axis, producing
                            # an (n_time,) array
    PARAM_PLUS_TRACE = 3    # base_param + dynamics_trace -> (n_time,) resolved value
    EXPRESSION = 4          # arithmetic expression referencing other params

    # --- Component evaluation nodes ---
    COMPONENT_EVAL = 5      # evaluates a component function over its domain axis:
                            #   energy functions -> (n_energy,) or (n_time, n_energy)
                            #   time functions -> (n_time,)
                            #   profile functions -> (n_aux,)
                            # the domain is determined by the component's package
                            # (fcts_energy, fcts_time, fcts_profile), not by the
                            # graph's DomainKind

    # --- Reduction / combination nodes ---
    SUM = 6                 # element-wise sum of multiple inputs
    SPECTRUM_FED_OP = 7     # component that consumes accumulated spectrum
                            # as input (Shirley only in v1)

    # --- Convolution and profile nodes ---
    # These are part of the IR spec and build_graph emits them when the
    # model uses these features. They cannot be compiled by the v1 2D
    # backend (can_lower_2d returns False), but they are fully representable
    # in the graph -- a future backend or the model-builder UI can work
    # with them.
    CONVOLUTION = 100       # convolves accumulated signal with a kernel
                            # component (e.g. gaussCONV). Edges: ADDEND from
                            # the accumulated signal, PARAM_INPUT from kernel
                            # parameters. Output replaces the accumulated signal.
    PROFILE_SAMPLE = 101    # resolves one profiled parameter at one
                            # aux_axis point:
                            #   base + profile.value_1d[aux_ind]
                            # One PROFILE_SAMPLE node per profiled parameter
                            # per aux_axis point.
    PROFILE_AVERAGE = 102   # uniform average over per-sample COMPONENT_EVAL
                            # outputs for one component. Edges: ADDEND from
                            # each per-sample COMPONENT_EVAL node.
    SUBCYCLE_MASK = 103     # element-wise multiply by time_n_sub mask array.
                            # Applied to a DYNAMICS_TRACE to zero out inactive
                            # subcycle regions.
    SUBCYCLE_REMAP = 104    # remaps a DYNAMICS_TRACE to use time_norm instead
                            # of the raw time axis (resets to 0 each subcycle).
                            # Precedes the dynamics function evaluation.
```

#### 1.2 Edge semantics

Edges are typed and carry dependency information:

```python
class EdgeKind(IntEnum):
    """Edge types in the model graph."""

    PARAM_INPUT = 0      # parameter flows into a component or expression
    TRACE_INPUT = 1      # dynamics trace flows into PARAM_PLUS_TRACE
    BASE_INPUT = 2       # base param flows into PARAM_PLUS_TRACE
    ADDEND = 3           # component output flows into SUM
    SPECTRUM_INPUT = 4   # accumulated spectrum flows into SPECTRUM_FED_OP
    EXPR_REF = 5         # parameter reference within an expression
```

#### 1.3 Graph structure

```python
@dataclass
class GraphNode:
    """One node in the model graph."""

    id: int                         # unique node ID
    kind: NodeKind
    name: str                       # human-readable name (e.g. "GLP_01_A", "GLP_01")
    source_order: int               # stable ordering key for deterministic scheduling.
                                    # When built from YAML: component definition order.
                                    # When built from UI: insertion order in the canvas.
                                    # schedule_2d uses this as tie-breaker when
                                    # topological sort has multiple valid orderings.

    # Payload (interpretation depends on kind):
    value: float | None             # for STATIC_PARAM, OPT_PARAM: initial value
    function_name: str | None       # for COMPONENT_EVAL, SPECTRUM_FED_OP, DYNAMICS_TRACE,
                                    # CONVOLUTION: the function registry name
                                    # (e.g. "GLP", "Shirley", "expFun", "gaussCONV").
                                    # This is the graph-level function identity -- it
                                    # works across all domains (energy, time, profile).
                                    # Backend-specific compilers map this to their own
                                    # op enums (e.g. schedule_2d maps "GLP" -> OpKind.GLP).
    package: str | None             # which function module: "energy", "time", "profile".
                                    # Together with function_name, uniquely identifies
                                    # the callable. Needed because function names could
                                    # theoretically collide across packages.
    expr_string: str | None         # for EXPRESSION: the expression source
    vary: bool                      # for OPT_PARAM: whether optimizer can change it
    bounds: tuple[float, float] | None  # for OPT_PARAM: (min, max) bounds
    arrays: dict[str, np.ndarray]   # node-local array data. Examples:
                                    #   SUBCYCLE_MASK: {"time_n_sub": array}
                                    #   SUBCYCLE_REMAP: {"time_norm": array}
                                    #   PROFILE_SAMPLE: {"aux_axis": array}
                                    #   CONVOLUTION: {"kernel_time": array}
                                    # Empty dict for nodes that need no array payload.
                                    # This is graph-level data, not backend-specific --
                                    # the scheduler copies what it needs into the plan.


@dataclass
class GraphEdge:
    """One edge in the model graph."""

    source: int                     # source node ID
    target: int                     # target node ID
    kind: EdgeKind
    position: int | None            # for PARAM_INPUT: which positional arg (0, 1, 2, ...)


class DomainKind(IntEnum):
    """Model domain classification.

    Determined by which axes the model operates on:
    - ENERGY_1D: model has energy axis only. This is the starting state
      for all energy models (fit_baseline, fit_spectrum). COMPONENT_EVAL
      nodes evaluate energy functions over (n_energy,).
    - TIME_1D: model has time axis only. Used for standalone Dynamics
      models. COMPONENT_EVAL nodes evaluate time functions over
      (n_time,). This domain is valid in the GraphIR for semantic
      completeness, graph tooling, and future model-builder/UI workflows,
      but has no compiled backend in the current scope.
    - ENERGY_TIME_2D: model has both axes. Created when dynamics are
      added to an ENERGY_1D model. COMPONENT_EVAL nodes evaluate energy
      functions over (n_time, n_energy) via broadcasting.
    """

    ENERGY_1D = 0       # energy-resolved only (e.g. fit_baseline, fit_spectrum)
    TIME_1D = 1         # time-resolved only (e.g. standalone dynamics fit)
    ENERGY_TIME_2D = 2  # energy + time resolved (the 2D fit case)


@dataclass
class GraphIR:
    """Directed acyclic graph representing a model.

    Axis-agnostic: works for 1D and 2D models. A 1D energy model has
    time=None; adding dynamics populates time and promotes domain to
    ENERGY_TIME_2D. Backend-specific compilers (schedule_2d, etc.)
    check the domain before compiling.
    """

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    domain: DomainKind              # what axes this model operates on
    energy: np.ndarray | None       # (n_energy,) or None
    time: np.ndarray | None         # (n_time,) or None
    # Reverse map for quick lookup
    node_by_name: dict[str, int]    # name -> node ID
```

#### 1.4 Example: `simple_energy` model as a graph

The YAML model:
```yaml
simple_energy:
    Offset:
      y0: [2, True, 0, 5]
    Shirley:
      pShirley: [4.0E-4, False]
    GLP:
      A: [20, True, 5, 25]
      x0: [84.5, True, 82, 88]
      F: [1.0, True, 0.75, 2.5]
      m: [0.3, True, 0, 1]
    GLP:
      A: [17, True, 5, 25]
      x0: [88.1, True]
      F: [1.0, True, 0.75, 2.5]
      m: [0.3, True, 0, 1]
```

Becomes:
```
Nodes:
  0: OPT_PARAM     "Offset_y0"        value=2.0
  1: STATIC_PARAM   "Shirley_pShirley" value=4e-4
  2: OPT_PARAM     "GLP_01_A"         value=20.0
  3: OPT_PARAM     "GLP_01_x0"        value=84.5
  4: OPT_PARAM     "GLP_01_F"         value=1.0
  5: OPT_PARAM     "GLP_01_m"         value=0.3
  6: OPT_PARAM     "GLP_02_A"         value=17.0
  7: OPT_PARAM     "GLP_02_x0"        value=88.1
  8: OPT_PARAM     "GLP_02_F"         value=1.0
  9: OPT_PARAM     "GLP_02_m"         value=0.3
 10: COMPONENT_EVAL "GLP_01"           function_name="GLP"     package="energy"
 11: COMPONENT_EVAL "GLP_02"           function_name="GLP"     package="energy"
 12: SUM            "peak_sum"
 13: COMPONENT_EVAL "Offset"           function_name="Offset"  package="energy"
 14: SPECTRUM_FED_OP "Shirley"         function_name="Shirley" package="energy"
 15: SUM            "total"

Edges:
  2 -> 10  PARAM_INPUT(pos=0)    # A -> GLP_01
  3 -> 10  PARAM_INPUT(pos=1)    # x0 -> GLP_01
  4 -> 10  PARAM_INPUT(pos=2)    # F -> GLP_01
  5 -> 10  PARAM_INPUT(pos=3)    # m -> GLP_01
  6 -> 11  PARAM_INPUT(pos=0)    # A -> GLP_02
  7 -> 11  PARAM_INPUT(pos=1)    # x0 -> GLP_02
  8 -> 11  PARAM_INPUT(pos=2)    # F -> GLP_02
  9 -> 11  PARAM_INPUT(pos=3)    # m -> GLP_02
 10 -> 12  ADDEND               # GLP_01 -> peak_sum
 11 -> 12  ADDEND               # GLP_02 -> peak_sum
  0 -> 13  PARAM_INPUT(pos=0)    # y0 -> Offset
 12 -> 14  SPECTRUM_INPUT        # peak_sum -> Shirley (Shirley consumes spectrum)
  1 -> 14  PARAM_INPUT(pos=0)    # pShirley -> Shirley
 12 -> 15  ADDEND               # peak_sum -> total
 13 -> 15  ADDEND               # Offset -> total
 14 -> 15  ADDEND               # Shirley -> total
```

Notice: Offset has no SPECTRUM_INPUT edge because it does not consume
the accumulated spectrum. LinBack also does not -- it receives `spectrum`
in the current Python signature for interface uniformity, but ignores it.
Only Shirley has a true data dependency on the peak sum.

#### 1.5 Time-dependent model as a graph

When `GLP_01_A` has a dynamics model (`expFun` with params `A`, `tau`,
`t0`, `y0`):

```
Additional nodes:
 20: OPT_PARAM     "GLP_01_A_expFun_01_A"     value=5.0
 21: OPT_PARAM     "GLP_01_A_expFun_01_tau"   value=100.0
 22: OPT_PARAM     "GLP_01_A_expFun_01_t0"    value=0.0
 23: STATIC_PARAM   "GLP_01_A_expFun_01_y0"   value=0.0
 24: DYNAMICS_TRACE "GLP_01_A_dynamics"         function_name="expFun"  package="time"
 25: PARAM_PLUS_TRACE "GLP_01_A_resolved"

Additional edges:
 20 -> 24  PARAM_INPUT(pos=0)    # A -> dynamics
 21 -> 24  PARAM_INPUT(pos=1)    # tau -> dynamics
 22 -> 24  PARAM_INPUT(pos=2)    # t0 -> dynamics
 23 -> 24  PARAM_INPUT(pos=3)    # y0 -> dynamics
  2 -> 25  BASE_INPUT            # GLP_01_A base value
 24 -> 25  TRACE_INPUT           # dynamics trace
 25 -> 10  PARAM_INPUT(pos=0)    # resolved A -> GLP_01 (replaces edge 2->10)
```

#### 1.6 Expression model as a graph

When `GLP_02_A = "3/4*GLP_01_A"`:

```
Node 6 becomes:
  6: EXPRESSION "GLP_02_A"  expr_string="3/4*GLP_01_A"

Additional edge:
  2 -> 6  EXPR_REF   # GLP_01_A referenced in expression
```

The expression node's output feeds into GLP_02 as before via
`6 -> 11 PARAM_INPUT(pos=0)`.

If `GLP_01_A` is time-dependent (has a PARAM_PLUS_TRACE), then node 6
references the resolved value (node 25), and the expression output is
also `(n_time,)`.

If `GLP_01_A` is profiled, the interpreter semantics are different:
the expression must be re-evaluated at each aux-axis point before the
component is evaluated and averaged. In the graph, this is represented
by per-sample `EXPRESSION` nodes (for example
`GLP_02_A_profile_expr_0`, `GLP_02_A_profile_expr_1`, ...) whose
`EXPR_REF` edges point to the matching per-sample `PROFILE_SAMPLE`
nodes for `GLP_01_A`. Those per-sample expression nodes feed per-sample
`COMPONENT_EVAL` nodes, and the component-level `PROFILE_AVERAGE`
averages the resulting traces.

#### 1.7 Why a graph, not a two-bucket sort

The previous version of this doc sorted components into "independent"
and "spectrum-dependent" buckets. This is wrong for two reasons:

1. **It conflates dependency class with evaluation order.** Offset and
   LinBack are currently classified as `SPECTRUM_DEP` because they have
   `spectrum` in their Python signature, but neither actually reads it.
   Only Shirley does. The graph makes this explicit: only Shirley has a
   `SPECTRUM_INPUT` edge.

2. **Reordering risks changing semantics.** The current interpreter
   combines components in LIFO order (mcp.py:806). Background
   components are defined first in YAML, peaks last, so the reverse
   iteration evaluates peaks before backgrounds. This ordering is
   validated: `background_last` in the test YAML is expected to fail.
   The graph preserves the original model semantics via edges, and the
   scheduler derives a safe topological order.


### 2. `can_lower_2d(graph) -> bool`

Note the input: `can_lower_2d` operates on the GraphIR, not the OOP model.
This means "can the 2D NumPy backend compile this graph," not "can this
model be represented as a graph at all." Every model can be a graph; not
every graph can be compiled to the 2D fast evaluator yet. A 1D-only graph
(``ENERGY_1D`` or ``TIME_1D``) returns False here. A future
`can_lower_1d` would target ``ENERGY_1D`` models; ``TIME_1D`` standalone
dynamics graphs remain graph-valid but backend-out-of-scope for now.

**Compilable in v1:**
- All `COMPONENT_EVAL` nodes have a `function_name` that `schedule_2d`
  can map to a supported `OpKind` with verified broadcast semantics.
  v1 list: `Gauss`, `GaussAsym`, `Lorentz`, `Voigt`, `GLS`, `GLP`,
  `DS`, `Offset`, `LinBack`.
  Offset and LinBack are `COMPONENT_EVAL`, not `SPECTRUM_FED_OP` --
  they do not consume the accumulated spectrum (see section 4).
- `SPECTRUM_FED_OP` nodes: `Shirley` only (the sole function that
  truly reads the accumulated peak sum)
- All `EXPRESSION` nodes contain only arithmetic operations
  (add, sub, mul, div, neg, pow, literal constants, parameter references)
- `CONVOLUTION`, `PROFILE_*`, and `SUBCYCLE_*` nodes are all now
  compilable (Phases 6.2-6.4).  Convolution is gated per-node by
  ``_is_lowerable_convolution_2d`` (time-domain kernel, resolved-trace
  topology); subcycle nodes are compiled away in ``schedule_2d`` into
  per-substep ``dyn_sub_time_axes`` / ``dyn_sub_masks`` schedule arrays.
- `graph.domain == ENERGY_TIME_2D` (has both energy and time axes)

**Falls back to interpreter when `can_lower_2d` returns False:**
- 1D-only models (domain != ENERGY_TIME_2D; future `can_lower_1d` would
  target `ENERGY_1D`, not `TIME_1D`, in the current roadmap)
- Non-arithmetic expressions
- Convolution shapes outside the resolved-trace contract encoded in
  ``_is_lowerable_convolution_2d`` (e.g. spectrum-level ``comp_type="conv"``,
  non-time kernels)

Phases 6.2-6.4 folded `CONVOLUTION`, `PROFILE_*`, and `SUBCYCLE_*` into
the 2D backend.  A resolved-trace (``subcycle=0``) IRF coexists with
subcycle-aware dynamics in the same model; only ``subcycle>0`` substeps
carry ``time_norm`` / ``time_n_sub`` schedule arrays.


### 3. ScheduledPlan2D -- the compiled 2D execution schedule

The scheduler takes a GraphIR and produces a flat, packed-array execution
plan. No Python objects in the hot path.

#### 3.1 Storage model: uniform `(n_params, n_time)` trace matrix

The previous version of this doc stored some params as scalars and some as
`(n_time,)` arrays, which forces `dtype=object` or runtime branching.
Instead, we use a uniform storage model:

**All parameters are stored as `(n_time,)` traces in a dense matrix.**

- Static params: broadcast to `(n_time,)` by repeating the scalar value.
  This wastes ~8 bytes * n_time per static param (e.g. 3.5 KB for 440
  time points) but eliminates all scalar-vs-array branching.
- Optimizer params: same -- broadcast to `(n_time,)`.
- Time-dependent params: naturally `(n_time,)` already.
- Expression results: `(n_time,)` regardless of whether inputs are
  time-varying (if all inputs are constant, the output is constant
  across time, but stored as a repeated vector anyway).

The scratch matrix has shape `(n_params, n_time)` and dtype `float64`.
Component evaluation gathers rows from this matrix and reshapes them to
`(n_time, 1)` for broadcasting against `(1, n_energy)`.

Memory cost: for a model with 20 parameters and 440 time points,
the scratch matrix is 20 * 440 * 8 = 69 KB. Negligible.

#### 3.2 Data structures

```python
class OpKind(IntEnum):
    """2D backend component function op codes.

    This is the *backend-specific* lowered enum, not the graph-level
    function identity. schedule_2d maps GraphNode.function_name to
    OpKind during compilation (e.g. "GLP" -> GLP, "Shirley" -> SHIRLEY).
    A future 1D backend would have its own enum with time-domain ops.
    """

    GAUSS = 0
    GAUSS_ASYM = 1
    LORENTZ = 2
    GLS = 3
    GLP = 4
    DS = 5
    OFFSET = 10
    LINBACK = 11
    SHIRLEY = 12


@dataclass(frozen=True)
class ScheduledPlan2D:
    """Compiled 2D execution schedule.

    No Python objects in the hot path (except ``expr_programs``).
    """

    energy: np.ndarray              # (n_energy,)
    time: np.ndarray                # (n_time,)
    n_params: int                   # total parameter count (all types)
    n_time: int                     # len(time)

    # --- Parameter mapping ---
    # Initial trace matrix: param_traces[i, :] is the base trace for param i.
    # For static/opt params, all n_time values are identical.
    # For time-dep params, the trace is base + dynamics_trace.
    param_traces_init: np.ndarray   # (n_params, n_time) initial values

    # Which rows are optimizer-visible (overwritten from theta each call):
    opt_indices: np.ndarray         # (n_opt,) int -- row indices into trace matrix
    opt_param_names: list[str]      # (n_opt,) canonical optimizer parameter names
                                    # defines the order theta must follow

    # --- Dynamics subgraphs (grouped by target PARAM_PLUS_TRACE) ---
    # Each dynamics group corresponds to one time-dependent parameter.
    # Multiple dynamics components (e.g. bi-exponential terms) feeding
    # the same PARAM_PLUS_TRACE are grouped together. Substeps within a
    # group are indexed via the CSR-style dyn_group_indptr.
    n_dyn_groups: int
    dyn_group_target_row: np.ndarray  # (n_dyn_groups,) int
    dyn_group_base_row: np.ndarray    # (n_dyn_groups,) int
    dyn_group_indptr: np.ndarray      # (n_dyn_groups + 1,) int -- CSR into substep arrays
    dyn_sub_func_id: np.ndarray       # (n_substeps,) int
    dyn_sub_param_rows: np.ndarray    # (n_substeps, max_dyn_params) int, -1 padded
    dyn_sub_n_params: np.ndarray      # (n_substeps,) int

    # --- Expression evaluation ---
    n_expressions: int
    expr_target_rows: np.ndarray    # (n_expressions,) int -- which row to write
    expr_programs: list["ExprProgram"]  # compiled RPN programs

    # --- Interleaved parameter resolution schedule ---
    # Dynamics groups and expressions may depend on each other
    # (for example, a dynamics parameter expressed in terms of another
    # dynamics parameter). These arrays encode the correct topological
    # execution order:
    #   kind=0 -> dynamics group step, index into dyn_group_* arrays
    #   kind=1 -> expression step, index into expr_* arrays / expr_programs
    resolution_kinds: np.ndarray    # (n_dyn_groups + n_expressions,) int8
    resolution_indices: np.ndarray  # (n_dyn_groups + n_expressions,) int

    # --- Scheduled component ops ---
    # Components in topologically-sorted execution order (derived from graph edges).
    # NOT a naive "peaks then backgrounds" reorder -- preserves model semantics.
    n_ops: int
    op_schedule: np.ndarray         # (n_ops,) int -- execution order indices
    op_kinds: np.ndarray            # (n_ops,) OpKind int codes
    # For each op: which rows in trace matrix are its parameter inputs.
    # Stored as CSR (compressed sparse row) for variable-length param lists:
    op_param_indptr: np.ndarray     # (n_ops + 1,) int -- CSR row pointers
    op_param_indices: np.ndarray    # (total_op_params,) int -- row indices
    # Which ops need the accumulated spectrum as input:
    op_needs_spectrum: np.ndarray   # (n_ops,) bool
    # Accumulation targets: which ops contribute to the "peak sum" that
    # spectrum-fed ops consume. Derived from graph SUM/SPECTRUM_INPUT edges.
    op_is_pre_spectrum: np.ndarray  # (n_ops,) bool -- contributes to peak_sum
```

#### 3.3 Expression programs

```python
class ExprNodeKind(IntEnum):
    """RPN instruction types."""

    CONST = 0       # push literal float
    PARAM_REF = 1   # push trace matrix row (by index)
    ADD = 2         # pop 2, push sum
    SUB = 3         # pop 2, push difference
    MUL = 4         # pop 2, push product
    DIV = 5         # pop 2, push quotient
    NEG = 6         # pop 1, push negation
    POW = 7         # pop 2, push power


@dataclass(frozen=True)
class ExprProgram:
    """Compiled expression: flat int array encoding an RPN program."""

    # Encoding: pairs of (node_kind, operand).
    # CONST: operand is float bits (np.float64.view(np.int64))
    # PARAM_REF: operand is row index into trace matrix
    # Operators: operand is 0 (unused)
    instructions: np.ndarray        # (2 * n_instructions,) int64
```

All values flowing through the RPN evaluator are `(n_time,)` arrays
(rows from the trace matrix). Constants are broadcast to `(n_time,)`.
This means expression evaluation is uniform -- no scalar/array branching.

**What's out of scope (v1):** Function calls (`np.exp`, `np.log`),
conditionals, string operations. These would extend `ExprNodeKind`.
If encountered during graph construction, those nodes stay in the graph
but `can_lower_2d()` returns False.


### 4. `build_graph(model) -> GraphIR`

Walks the OOP tree once and emits the graph.

#### Algorithm

```
build_graph(model):
    1. Create parameter nodes
       For each component in model.components:
         For each Par in component.pars:
           - If expression: create EXPRESSION node
           - If par.t_vary: create OPT_PARAM (base), DYNAMICS_TRACE,
             PARAM_PLUS_TRACE nodes, plus OPT/STATIC nodes for dynamics params
           - If par.vary == False: create STATIC_PARAM node
           - Else: create OPT_PARAM node

    2. Create expression edges
       For each EXPRESSION node:
         - Parse expr_string to extract referenced parameter names
         - Add EXPR_REF edge from each referenced param node to this node
         - If any referenced param has a PARAM_PLUS_TRACE, reference that
           instead (resolved time-dependent value)

    3. Create component nodes
       For each component in model.components:
         - Create COMPONENT_EVAL or SPECTRUM_FED_OP node based on:
           * Shirley: SPECTRUM_FED_OP (it actually reads spectrum)
           * Offset, LinBack: COMPONENT_EVAL (they don't read spectrum
             despite Python signature; LinBack uses x/params only,
             Offset is pure constant)
           * Others: COMPONENT_EVAL
         - Add PARAM_INPUT edges from resolved param nodes to component,
           with position matching function signature order

    4. Create convolution, profile, and subcycle nodes (when present)
       For each component with comp_type == "conv":
         - Create CONVOLUTION node
         - Add ADDEND edge from accumulated signal (the SUM being built)
         - Add PARAM_INPUT edges from kernel parameters
       For each component with a profiled parameter (`p_vary`) or an
       expression parameter that references a profiled parameter
       (`expr_refs_profile_dep`):
         - For each direct `p_vary` parameter:
           create PROFILE_SAMPLE nodes for each aux_axis point
         - Add PARAM_INPUT edges into each PROFILE_SAMPLE
           (base + profile-model params for that aux point)
         - For each `expr_refs_profile_dep` parameter:
           create per-sample EXPRESSION nodes, with EXPR_REF edges to
           the matching per-sample PROFILE_SAMPLE nodes of the
           referenced profiled parameter
         - Create a per-sample COMPONENT_EVAL node for each aux_axis
           point, using PROFILE_SAMPLE / per-sample EXPRESSION inputs
           where needed
         - Create one component-level PROFILE_AVERAGE node with ADDEND
           edges from the per-sample COMPONENT_EVAL nodes
         - Replace the original component in the combination graph with
           this PROFILE_AVERAGE node
       For each dynamics model with subcycle != 0:
         - Create SUBCYCLE_REMAP node before the DYNAMICS_TRACE
         - Create SUBCYCLE_MASK node after the DYNAMICS_TRACE
         - Store time_norm and time_n_sub in node.arrays
           (e.g. SUBCYCLE_REMAP.arrays["time_norm"],
            SUBCYCLE_MASK.arrays["time_n_sub"])

       These nodes are fully represented in the graph. can_lower_2d()
       will return False if any are present, but the graph is still
       valid and can be interpreted or compiled by a future backend.

    5. Create combination nodes
       - Analyze the model's combine order (LIFO from mcp.py:806)
       - Create SUM node for peak accumulation
       - For SPECTRUM_FED_OP nodes (Shirley): add SPECTRUM_INPUT edge
         from the peak SUM node
       - Create final SUM node that adds everything together
       - Edges preserve the original combine semantics

    6. Assign source_order to all nodes
       - Monotonically increasing, following the order nodes were created
         during the walk. For YAML-built models this reflects component
         definition order. For UI-built graphs, the UI sets source_order
         at node creation time.

    7. Return GraphIR
```

#### Offset and LinBack reclassification

The current interpreter treats all background functions uniformly
(`comp_type == "back"` -> pass `spectrum=value`). But examining the
actual functions:

- `Offset(x, y0, spectrum)`: returns `np.full_like(spectrum, y0)`.
  Uses `spectrum` only for shape. With uniform `(n_time, n_energy)`
  arrays, this is just a broadcast scalar -- no spectrum dependency.
- `LinBack(x, m, b, xStart, xStop, spectrum)`: returns a piecewise
  function of `x`, `m`, `b`, `xStart`, `xStop`. The `spectrum` argument
  is never read. Pure function of its own params + energy axis.
- `Shirley(x, pShirley, spectrum)`: returns
  `pShirley * cumsum(spectrum[::-1])[::-1]`. Truly reads `spectrum`.

In the graph, only Shirley gets a `SPECTRUM_INPUT` edge. Offset and
LinBack become regular `COMPONENT_EVAL` nodes (type `INDEPENDENT` in
the old terminology). This is semantically correct and allows them to
be evaluated in the first pass alongside peaks.


### 5. `schedule_2d(graph) -> ScheduledPlan2D`

Compiles the graph into a flat execution schedule. The v1 backend is a
**specialized lowering target**, not a generic DAG executor. The GraphIR
is general-purpose (and a future UI will build arbitrary graphs), but the
current scheduler and evaluator collapse the graph to a fixed-shape
pipeline: parameter traces -> component ops -> peak_sum -> optional
spectrum-fed ops -> final sum. This is intentional -- it keeps the
evaluator simple and fast. As the compiler supports more node types,
the pipeline shape may grow, but it should always be a concrete schedule,
never a runtime graph walker.

#### Algorithm

```
schedule_2d(graph):
    1. Topological sort of all nodes
       Tie-breaker: when two nodes have no dependency ordering between
       them, sort by node.source_order (lower first). This makes the
       schedule deterministic regardless of whether the graph came from
       YAML parsing or a UI drag-and-drop canvas.

    2. Assign trace matrix rows
       - One row per parameter (STATIC_PARAM, OPT_PARAM, resolved
         PARAM_PLUS_TRACE, EXPRESSION output)
       - Contiguous: opt params first (so theta maps to a slice),
         then static, then computed

    3. Compile grouped dynamics subgraphs
       - For each DYNAMICS_TRACE node:
         * Record function ID (expFun -> 0, sinFun -> 1, etc.)
         * Record param row indices (from PARAM_INPUT edges)
         * Record the PARAM_PLUS_TRACE target row and its base row
       - Group all DYNAMICS_TRACE nodes feeding the same
         PARAM_PLUS_TRACE into one dynamics group
       - Pack groups as CSR-style arrays:
         dyn_group_target_row / dyn_group_base_row / dyn_group_indptr,
         plus flat dyn_sub_* arrays for the grouped substeps

    4. Compile expressions
       - For each EXPRESSION node (in topological order):
         * Parse expr_string into RPN using Python ast module
         * Reject unsupported AST nodes (function calls, etc.)
         * Resolve param references to trace matrix row indices
           (using EXPR_REF edges as the source of truth)
         * Encode as flat int64 array

    5. Build interleaved parameter-resolution schedule
       - Walk topo order and emit:
         * expression steps when EXPRESSION nodes appear
         * one dynamics-group step at the last DYNAMICS_TRACE node in
           each group, so all prerequisite expressions are already
           resolved before that group is evaluated
       - Store as resolution_kinds / resolution_indices

    6. Schedule component ops
       - Topological order from graph edges determines execution order
       - For each COMPONENT_EVAL / SPECTRUM_FED_OP node:
         * Record OpKind
         * Record param row indices (from PARAM_INPUT edges, CSR-encoded)
         * Record whether it needs spectrum input (has SPECTRUM_INPUT edge)
       - Derive op_is_pre_spectrum: which ops contribute to the
         accumulated peak sum (inputs to the SUM node that feeds
         SPECTRUM_FED_OP nodes)

    7. Initialize trace matrix
       - For static params: fill row with repeated scalar
       - For opt params: fill row with repeated initial value
       - Replay the same interleaved resolution schedule used by the
         runtime evaluator:
         * dynamics group -> base + sum(substep traces)
         * expression -> evaluate RPN program against current trace matrix

    8. Pack into ScheduledPlan2D, return
```

#### Critical invariants

- `schedule_2d()` is called **once** before the optimizer starts. The plan
  is immutable during optimization (except the trace matrix scratch
  space, which is a working copy).
- The plan does not hold references to any `Model`, `Component`, or `Par`
  objects. It is pure data.
- Optimizer parameter ordering is defined by `opt_param_names` stored
  in the plan (see below). This is the canonical order -- `extract_theta`
  must return values in this order, and result writeback must use it.
  At construction time, `opt_param_names` is derived from
  `model.parameter_names` / `model.lmfit_pars` (only the vary=True
  subset), so the contract is explicit rather than implicit.
- The execution order preserves model semantics: the scheduler derives
  it from graph edges with stable tie-breaking on original definition
  order, not from a coarse bucket sort.


### 6. `evaluate_2d(plan, theta) -> ndarray`

Pure function. Takes the scheduled plan and optimizer parameter vector,
returns `(n_time, n_energy)` spectrum.

#### Algorithm

```
evaluate_2d(plan, theta):
    1. PARAMETER RESOLUTION (once per call)
       a. Copy plan.param_traces_init -> traces  (n_params, n_time) scratch
       b. Broadcast optimizer params into trace matrix:
          traces[plan.opt_indices, :] = theta[:, np.newaxis]
       c. Execute the interleaved resolution schedule:
          For each step in (resolution_kinds, resolution_indices):
            - If kind == dynamics group:
              * target = dyn_group_target_row[idx]
              * start from traces[target, :] = traces[base_row, :]
              * for each grouped substep:
                - Gather params from traces[dyn_sub_param_rows[s], 0]
                - Call dynamics function(plan.time, *p) -> (n_time,) trace
                - Accumulate into the target row
            - If kind == expression:
              * Execute RPN against traces matrix
              * All operands are (n_time,) rows; arithmetic broadcasts
              * Write result row: traces[expr_target_rows[idx], :] = result

    2. COMPONENT EVALUATION (in scheduled order)
       result = zeros(n_time, n_energy)
       peak_sum = zeros(n_time, n_energy)

       For each op in plan.op_schedule:
         - Gather params: rows from traces matrix via CSR indices
           -> reshape each from (n_time,) to (n_time, 1) for broadcasting
         - If NOT op_needs_spectrum[op]:
             component = eval_op(plan.energy, params, plan.op_kinds[op])
             result += component
             if op_is_pre_spectrum[op]:
                 peak_sum += component
         - If op_needs_spectrum[op]:
             component = eval_op_with_spectrum(
                 plan.energy, params, peak_sum, plan.op_kinds[op])
             result += component

    3. Return result  # (n_time, n_energy)
```

#### Component evaluation dispatch

Each `OpKind` maps directly to the shared kernels in
`src/trspecfit/functions/energy.py`, which are the single source of
truth for component math. Peak functions broadcast naturally with
`(n_time, 1)` params and `(1, n_energy)` energy; `Offset`, `LinBack`,
and `Shirley` are written to support both 1D and broadcasted 2D inputs:

```python
_OP_DISPATCH = {
    OpKind.GAUSS: (fcts_energy.Gauss, 3, False),
    OpKind.GLP: (fcts_energy.GLP, 4, False),
    OpKind.OFFSET: (fcts_energy.Offset, 1, False),
    OpKind.LINBACK: (fcts_energy.LinBack, 4, False),
    OpKind.SHIRLEY: (fcts_energy.Shirley, 1, True),
}
```

Shirley should use axis-agnostic last-axis operations (`axis=-1`) so the
same implementation works for both 1D and batched inputs.

#### Dynamics function dispatch

Dynamics functions are called once per residual call on the full time
axis. The evaluator maps `dyn_sub_func_id` to the existing functions
from `functions/time.py`:

```python
DYNAMICS_DISPATCH = {
    0: fcts_time.expFun,      # (t, A, tau, t0, y0) -> (n_time,)
    1: fcts_time.sinFun,      # (t, A, f, phi, t0, y0) -> (n_time,)
    2: fcts_time.linFun,      # (t, m, t0, y0) -> (n_time,)
    ...
}
```

These are the same functions, called once with full arrays. No per-step
loop.


### 7. Integration with lmfit

#### Current call chain (2D fits)
```
lmfit.minimize(residual_fun, params, ...)
  -> residual_fun(params, x, data, ..., args=(model, 2))
    -> par_extract(params) -> list of values
    -> fit_model_mcp(x, par_values, True, model, 2)
      -> model.update_value(par_values)     # write theta into lmfit.Parameters
      -> model.create_value_2d()            # THE HOT PATH (interpreter loop)
        -> for each time step:
            create_value_1d(t_ind=ti)
              -> for each component:
                  Component.value(t_ind)
                    -> for each par: Par.value(t_ind)  <-- 2M calls
      -> return model.value_2d
    -> residual = data - fit
    -> return residual.flatten()
```

#### New call chain (implemented)
```
# Default project setting:
project.spec_fun_str = "fit_model_gir"

# Before a lowerable fit (1D or 2D):
graph = build_graph(model)
plan = schedule_2d(graph)   # or schedule_1d(graph) for ENERGY_1D
theta_indices = precompute_indices(model.parameter_names, plan.opt_param_names)

# During fitting:
lmfit.minimize(residual_fun, params, ..., args=(plan, theta_indices, model, dim))
  -> residual_fun(params, x, data, ..., fit_fun_str="fit_model_gir")
    -> par_extract(params) -> full par_values list
    -> fit_model_gir(x, par_values, True, plan, theta_indices, model, dim)
      -> theta = par_values[theta_indices]
      -> spectrum = evaluate_2d(plan, theta)   # or evaluate_1d for 1D
    -> residual = data - spectrum
    -> return residual.flatten()
```

**Compiled-path args contract:** the compiled path always passes
``(plan, theta_indices, model, dim)`` as *args*, regardless of
whether the plan is ``ScheduledPlan1D`` or ``ScheduledPlan2D``.
The *model* and *dim* are carried so that ``fit_model_gir`` can
fall back to the interpreter when needed (e.g. ``plot_sum=False``
for 1D component extraction).  The non-lowerable fallback passes
``(model, dim)`` only.

Implemented behavior:
- `fit_model_gir` is a dispatcher. When `args[0]` is a compiled
  plan it destructures `(plan, theta_indices, model, dim)` and
  uses the fast evaluator.  For 1D plans with `plot_sum=False`
  it falls back to `fit_model_mcp` for component extraction.
  When `args[0]` is not a plan it forwards to `fit_model_mcp`.
- `fit_model_mcp` remains the explicit interpreter mode.
- `fit_model_compare` is the validation mode. For lowerable fits
  (1D or 2D) it runs GIR and interpreter, compares with
  `assert_allclose`, and returns the GIR result.
- `File.fit_2d`, `File.fit_baseline`, and `File.fit_spectrum`
  build a graph / plan when `spec_fun_str` is `fit_model_gir` or
  `fit_model_compare`.
- After fitting, all three methods write `result[1].params` back
  into `model.lmfit_pars` via `par_extract` + `update_value`,
  because `fit_wrapper` optimizes a deepcopy and the GIR path
  does not mutate model state on every residual call.


### 8. Scope boundaries -- what compiles in v1 vs later

| Feature | v1 compiler | Graph representable | Notes |
|---|---|---|---|
| Additive peaks (Gauss, GLP, GLS, DS, Voigt, etc.) | Yes | Yes | Core use case |
| Offset, LinBack | Yes | Yes | Reclassified as COMPONENT_EVAL (no spectrum dep) |
| Shirley | Yes | Yes | SPECTRUM_FED_OP using last-axis cumulative sum |
| Arithmetic expressions | Yes | Yes | Compiled to RPN |
| Time-dependent params (Dynamics) | Yes | Yes | Dynamics subgraph compiled |
| Voigt | Yes | Yes | Broadcast-safe on the current evaluator path |
| Convolution components | Yes (Phase 6.3) | Yes (CONVOLUTION node) | Resolved-trace time-domain IRF; gated by ``_is_lowerable_convolution_2d`` |
| Profile-varying params | Yes (Phase 6.2) | Yes (PROFILE_* nodes) | 1D and 2D backends |
| Subcycle dynamics | Yes (Phase 6.4) | Yes (SUBCYCLE_* nodes) | Compiled into per-substep ``dyn_sub_time_axes`` / ``dyn_sub_masks`` |
| Non-arithmetic expressions | No | Partial | Would extend ExprNodeKind |
| Project-level fits | No | Deferred | Multi-graph coordination |

`can_lower_2d` and `can_lower_1d` are the gatekeepers for the 2D and 1D
backends respectively. The graph is broader than any single compiler --
the UI can build graphs that no backend can compile yet.
`can_lower_1d` / `schedule_1d` / `evaluate_1d` target `ENERGY_1D`
models on the same GraphIR (implemented in Phase 6.1).


### 9. Validation strategy

The compiled evaluator must produce results that agree with the
interpreter to within reasonable floating-point tolerance. Because the
compiled path sums components in a different order (graph-scheduled vs
the interpreter's LIFO combine loop), and broadcasts 2D operations
instead of accumulating per-time-step, floating-point summation order
may differ. This means bitwise identity is not guaranteed -- `allclose`
with a practical tolerance is the contract.

```python
def validate_plan(model, plan):
    """Compare interpreter vs compiled evaluator output."""

    # Evaluate via interpreter
    model.create_value_2d()
    interp_result = model.value_2d.copy()

    # Evaluate via plan
    theta = extract_theta(model.lmfit_pars)
    plan_result = evaluate_2d(plan, theta)

    # Compare -- rtol=1e-10 accounts for summation order differences.
    # For typical spectroscopy data (values 0-100), this means agreement
    # to ~10 significant digits.
    assert np.allclose(interp_result, plan_result, atol=1e-10, rtol=1e-10)
```

This validation runs:
- In the test suite, for every test model that `can_lower_2d()` accepts
- In explicit compare mode during fitting via
  `spec_fun_str="fit_model_compare"`
- On fallback cases, compare mode delegates to the interpreter so
  behavior matches the non-GIR path


### 10. Future: backends and Jacobians

Once `evaluate_2d(plan, theta) -> spectrum` exists as a pure function:

**Numba:** `@njit` on the component eval dispatch loop. Most of the
plan (int index arrays, CSR param maps, dense trace matrix) is
Numba-compatible. The expression programs (`list[ExprProgram]`) would
need flattening to a CSR-style encoding first -- this is noted in the
plan as a v1 simplification that can be tightened later.

**JAX:** Replace `np` with `jnp` in the evaluator. The plan's array
structure maps directly to JAX arrays. Key wins:
- `jax.jit` compiles the full evaluator (including parameter resolution)
- `jax.jacfwd` / `jax.jacrev` gives analytic Jacobians for free
- GPU acceleration for large grids

**Analytic Jacobians:** With a differentiable evaluator, lmfit's
Levenberg-Marquardt can use exact Jacobians instead of finite differences.
This replaces `2 * n_free_params + 1` evaluator calls per iteration with
1 evaluator call + 1 Jacobian call. For 4 free params, that's 9 -> 2
calls, ~4.5x fewer evaluations per iteration.

**Variable projection (VARPRO):** Linear parameters (amplitudes `A`,
offset `y0`, slope `m`) can be solved in closed form given the nonlinear
parameters. Reduces optimizer dimensionality. The graph makes identifying
linear params straightforward: any param that appears as a linear factor
in its component's function.


---
## Implementation tracking

This document is the long-lived design/spec for the lowered evaluator.

- Implementation status, completed phases, and active follow-up work live in
  `PLAN.md` at the repo root.
- The closed backend-decision benchmark write-up lives in
  [archive/numba_vs_jax.md](archive/numba_vs_jax.md).
