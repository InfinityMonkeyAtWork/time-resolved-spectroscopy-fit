"""Graph-based intermediate representation for model fitting.

The GraphIR is a DAG (directed acyclic graph) of typed nodes connected
by explicit dependency edges.  It captures the full semantics of a
model without prescribing an evaluation strategy.

Three-layer design:

1. **OOP tree** -- the user-facing Model/Component/Par objects.
   Handles parsing, validation, user interaction.  Unchanged.
2. **GraphIR** -- a directed acyclic graph of typed nodes with explicit
   data-dependency edges.  Axis-agnostic: works for 1D and 2D models.
3. **ScheduledPlan2D** -- a flat, packed-array execution schedule
   compiled from the graph by the 2D backend.  No Python objects, no
   strings, no dicts in the hot path.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from trspecfit.functions import energy as fcts_energy
from trspecfit.functions import profile as fcts_profile

if TYPE_CHECKING:
    from trspecfit.mcp import Component, Model, Par


#
#
class NodeKind(IntEnum):
    """Node types in the model graph."""

    # --- Parameter nodes (leaves) ---
    STATIC_PARAM = 0
    OPT_PARAM = 1

    # --- Computed parameter nodes ---
    DYNAMICS_TRACE = 2
    PARAM_PLUS_TRACE = 3
    EXPRESSION = 4

    # --- Component evaluation nodes ---
    COMPONENT_EVAL = 5

    # --- Reduction / combination nodes ---
    SUM = 6
    SPECTRUM_FED_OP = 7

    # --- Convolution and profile nodes (representable, not v1-compilable) ---
    CONVOLUTION = 100
    PROFILE_SAMPLE = 101
    PROFILE_AVERAGE = 102
    SUBCYCLE_MASK = 103
    SUBCYCLE_REMAP = 104


#
#
class EdgeKind(IntEnum):
    """Edge types in the model graph."""

    PARAM_INPUT = 0
    TRACE_INPUT = 1
    BASE_INPUT = 2
    ADDEND = 3
    SPECTRUM_INPUT = 4
    EXPR_REF = 5


#
#
class DomainKind(IntEnum):
    """Model domain classification.

    Determined by which axes the model operates on:

    - ``ENERGY_1D``: model has energy axis only.
    - ``TIME_1D``: model has time axis only.
    - ``ENERGY_TIME_2D``: model has both axes.
    """

    ENERGY_1D = 0
    TIME_1D = 1
    ENERGY_TIME_2D = 2


#
#
class OpKind(IntEnum):
    """2D backend component function op codes.

    Backend-specific lowered enum.  ``schedule_2d`` maps
    ``GraphNode.function_name`` to ``OpKind`` during compilation.
    """

    GAUSS = 0
    GAUSS_ASYM = 1
    LORENTZ = 2
    VOIGT = 3
    GLS = 4
    GLP = 5
    DS = 6
    OFFSET = 10
    LINBACK = 11
    SHIRLEY = 12


#
#
class DynFuncKind(IntEnum):
    """Dynamics function op codes for the 2D backend.

    Backend-specific lowered enum.  ``schedule_2d`` maps
    ``GraphNode.function_name`` (for ``DYNAMICS_TRACE`` nodes) to
    ``DynFuncKind`` during compilation.
    """

    EXPFUN = 0
    SINFUN = 1
    LINFUN = 2
    SINDIVX = 3
    ERFFUN = 4
    SQRTFUN = 5


#
#
class ProfileFuncKind(IntEnum):
    """Profile-function op codes for lowered 1D profile evaluation."""

    PEXPDECAY = 0
    PLINEAR = 1
    PGAUSS = 2


#
#
class ParamSourceKind(IntEnum):
    """Parameter source kinds for profiled component evaluation (1D and 2D)."""

    SCALAR = 0
    PROFILE_SAMPLE = 1
    PROFILE_EXPR = 2


_FUNCTION_NAME_TO_DYN_FUNC: dict[str, DynFuncKind] = {
    "expFun": DynFuncKind.EXPFUN,
    "sinFun": DynFuncKind.SINFUN,
    "linFun": DynFuncKind.LINFUN,
    "sinDivX": DynFuncKind.SINDIVX,
    "erfFun": DynFuncKind.ERFFUN,
    "sqrtFun": DynFuncKind.SQRTFUN,
}

_FUNCTION_NAME_TO_PROFILE_FUNC: dict[str, ProfileFuncKind] = {
    "pExpDecay": ProfileFuncKind.PEXPDECAY,
    "pLinear": ProfileFuncKind.PLINEAR,
    "pGauss": ProfileFuncKind.PGAUSS,
}

_FUNCTION_NAME_TO_OP: dict[str, OpKind] = {
    "Gauss": OpKind.GAUSS,
    "GaussAsym": OpKind.GAUSS_ASYM,
    "Lorentz": OpKind.LORENTZ,
    "Voigt": OpKind.VOIGT,
    "GLS": OpKind.GLS,
    "GLP": OpKind.GLP,
    "DS": OpKind.DS,
    "Offset": OpKind.OFFSET,
    "LinBack": OpKind.LINBACK,
    "Shirley": OpKind.SHIRLEY,
}

#: Maps ``OpKind`` → ``(eval_function, needs_spectrum)``.
#: Single source of truth for component dispatch -- used by both the
#: evaluator hot path and constant-component precomputation.
OP_DISPATCH: dict[int, tuple[Callable[..., Any], bool]] = {
    int(OpKind.GAUSS): (fcts_energy.Gauss, False),
    int(OpKind.GAUSS_ASYM): (fcts_energy.GaussAsym, False),
    int(OpKind.LORENTZ): (fcts_energy.Lorentz, False),
    int(OpKind.VOIGT): (fcts_energy.Voigt, False),
    int(OpKind.GLS): (fcts_energy.GLS, False),
    int(OpKind.GLP): (fcts_energy.GLP, False),
    int(OpKind.DS): (fcts_energy.DS, False),
    int(OpKind.OFFSET): (fcts_energy.Offset, False),
    int(OpKind.LINBACK): (fcts_energy.LinBack, False),
    int(OpKind.SHIRLEY): (fcts_energy.Shirley, True),
}


PROFILE_DISPATCH: dict[int, Callable[..., Any]] = {
    int(ProfileFuncKind.PEXPDECAY): fcts_profile.pExpDecay,
    int(ProfileFuncKind.PLINEAR): fcts_profile.pLinear,
    int(ProfileFuncKind.PGAUSS): fcts_profile.pGauss,
}


#
#
class ExprNodeKind(IntEnum):
    """RPN instruction types for compiled expressions."""

    CONST = 0
    PARAM_REF = 1
    ADD = 2
    SUB = 3
    MUL = 4
    DIV = 5
    NEG = 6
    POW = 7


#
#
@dataclass
class GraphNode:
    """One node in the model graph."""

    id: int
    kind: NodeKind
    name: str
    source_order: int

    # Payload (interpretation depends on kind):
    value: float | None = None
    function_name: str | None = None
    package: str | None = None
    expr_string: str | None = None
    vary: bool = False
    bounds: tuple[float, float] | None = None
    arrays: dict[str, np.ndarray] = field(default_factory=dict)


#
#
@dataclass
class GraphEdge:
    """One edge in the model graph."""

    source: int
    target: int
    kind: EdgeKind
    position: int | None = None


#
#
@dataclass
class GraphIR:
    """Directed acyclic graph representing a model.

    Axis-agnostic: works for 1D and 2D models.  A 1D energy model has
    ``time=None``; adding dynamics populates ``time`` and promotes
    ``domain`` to ``ENERGY_TIME_2D``.
    """

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    domain: DomainKind
    energy: np.ndarray | None = None
    time: np.ndarray | None = None
    node_by_name: dict[str, int] = field(default_factory=dict)

    #
    def to_dot(self, *, collapse_profiles: bool = True) -> str:
        """Return a Graphviz DOT string for this graph.

        Node shapes and colours encode ``NodeKind``; edge labels encode
        ``EdgeKind``.  The output can be rendered with ``dot -Tpng`` or
        any Graphviz viewer.

        Parameters
        ----------
        collapse_profiles : bool, default=True
            When True, per-sample profile nodes (``PROFILE_SAMPLE``,
            per-sample ``COMPONENT_EVAL``, per-sample ``EXPRESSION``)
            are collapsed into single representative nodes showing the
            sample count.  This keeps profile models readable.
        """

        _NODE_STYLE: dict[NodeKind, dict[str, str]] = {
            NodeKind.STATIC_PARAM: dict(
                shape="ellipse", style="filled", fillcolor="#d3d3d3"
            ),
            NodeKind.OPT_PARAM: dict(
                shape="ellipse", style="filled", fillcolor="#87ceeb"
            ),
            NodeKind.DYNAMICS_TRACE: dict(
                shape="box", style="filled", fillcolor="#ffa07a"
            ),
            NodeKind.PARAM_PLUS_TRACE: dict(
                shape="box", style="filled", fillcolor="#ffcc80"
            ),
            NodeKind.EXPRESSION: dict(
                shape="hexagon", style="filled", fillcolor="#dda0dd"
            ),
            NodeKind.COMPONENT_EVAL: dict(
                shape="box", style="filled,bold", fillcolor="#90ee90"
            ),
            NodeKind.SUM: dict(shape="diamond", style="filled", fillcolor="#fffacd"),
            NodeKind.SPECTRUM_FED_OP: dict(
                shape="box", style="filled,bold", fillcolor="#f08080"
            ),
            NodeKind.CONVOLUTION: dict(
                shape="octagon", style="filled", fillcolor="#e0e0ff"
            ),
            NodeKind.PROFILE_SAMPLE: dict(
                shape="parallelogram", style="filled", fillcolor="#c8e6c9"
            ),
            NodeKind.PROFILE_AVERAGE: dict(
                shape="parallelogram", style="filled", fillcolor="#a5d6a7"
            ),
            NodeKind.SUBCYCLE_MASK: dict(
                shape="trapezium", style="filled", fillcolor="#ffe0b2"
            ),
            NodeKind.SUBCYCLE_REMAP: dict(
                shape="trapezium", style="filled", fillcolor="#ffcc80"
            ),
        }

        _EDGE_STYLE: dict[EdgeKind, dict[str, str]] = {
            EdgeKind.PARAM_INPUT: dict(color="#333333"),
            EdgeKind.TRACE_INPUT: dict(color="#ff6600", style="dashed"),
            EdgeKind.BASE_INPUT: dict(color="#0066cc", style="dashed"),
            EdgeKind.ADDEND: dict(color="#009933", style="bold"),
            EdgeKind.SPECTRUM_INPUT: dict(color="#cc0000", style="bold"),
            EdgeKind.EXPR_REF: dict(color="#9933cc", style="dotted"),
        }

        # --- Profile collapsing ---
        # Maps each per-sample node id to the representative node id for
        # its group.  Nodes not in this dict are emitted as-is.
        collapsed: dict[int, int] = {}  # sample_nid -> representative_nid
        # Groups: representative_nid -> (base_name, kind, count, first_node)
        _profile_groups: dict[int, tuple[str, NodeKind, int, GraphNode]] = {}

        if collapse_profiles:
            _sample_re = re.compile(
                r"^(.+?)_(profile_sample|profile_expr|sample)_(\d+)$"
            )
            # Group nodes by (base_name, kind)
            groups: dict[tuple[str, NodeKind], list[GraphNode]] = {}
            for node in self.nodes:
                m = _sample_re.match(node.name)
                if m is not None:
                    base = m.group(1)
                    groups.setdefault((base, node.kind), []).append(node)

            for (base, kind), members in groups.items():
                if len(members) < 2:
                    continue
                rep = members[0]
                for member in members:
                    collapsed[member.id] = rep.id
                _profile_groups[rep.id] = (base, kind, len(members), rep)

        hidden_nodes = {nid for nid in collapsed if collapsed[nid] != nid}

        lines: list[str] = [
            "digraph ModelGraph {",
            "  rankdir=BT;",
            '  node [fontname="Helvetica", fontsize=10];',
            '  edge [fontname="Helvetica", fontsize=8];',
        ]

        for node in self.nodes:
            if node.id in hidden_nodes:
                continue
            attrs = dict(_NODE_STYLE.get(node.kind, {}))

            if node.id in _profile_groups:
                base, kind, count, _ = _profile_groups[node.id]
                label_parts = [f"{base} (\u00d7{count})", kind.name]
            else:
                label_parts = [node.name, node.kind.name]
                if node.function_name:
                    label_parts.append(f"fn={node.function_name}")
                if node.value is not None:
                    label_parts.append(f"val={node.value:g}")
                if node.expr_string:
                    label_parts.append(f"expr={node.expr_string}")

            attrs["label"] = "\\n".join(label_parts)
            attr_str = ", ".join(f'{k}="{v}"' for k, v in attrs.items())
            lines.append(f"  n{node.id} [{attr_str}];")

        seen_edges: set[tuple[int, int, EdgeKind, int | None]] = set()
        for edge in self.edges:
            src = collapsed.get(edge.source, edge.source)
            tgt = collapsed.get(edge.target, edge.target)
            key = (src, tgt, edge.kind, edge.position)
            if key in seen_edges:
                continue
            seen_edges.add(key)

            attrs = dict(_EDGE_STYLE.get(edge.kind, {}))
            label = edge.kind.name
            if edge.position is not None:
                label += f"[{edge.position}]"
            attrs["label"] = label
            attr_str = ", ".join(f'{k}="{v}"' for k, v in attrs.items())
            lines.append(f"  n{src} -> n{tgt} [{attr_str}];")

        lines.append("}")
        return "\n".join(lines)


#
#
@dataclass(frozen=True)
class ExprProgram:
    """Compiled expression: flat int array encoding an RPN program.

    Encoding: pairs of ``(node_kind, operand)``.

    - ``CONST``: operand is float bits (``np.float64.view(np.int64)``)
    - ``PARAM_REF``: operand is row index into trace matrix
    - Operators: operand is 0 (unused)
    """

    instructions: np.ndarray  # (2 * n_instructions,) int64


#
#
@dataclass(frozen=True)
class ScheduledPlan2D:
    """Compiled 2D execution schedule.

    No Python objects in the hot path (except ``expr_programs``).
    """

    energy: np.ndarray  # (n_energy,)
    time: np.ndarray  # (n_time,)
    n_params: int
    n_time: int

    # --- Parameter mapping ---
    param_traces_init: np.ndarray  # (n_params, n_time)
    opt_indices: np.ndarray  # (n_opt,) int
    opt_param_names: list[str]  # (n_opt,) canonical optimizer param names

    # --- Dynamics subgraphs (grouped by target PARAM_PLUS_TRACE) ---
    # Each "dynamics group" corresponds to one time-dependent parameter.
    # Multiple dynamics components (e.g. bi-exponential) targeting the
    # same PARAM_PLUS_TRACE are grouped together.  Substeps within a
    # group are indexed via the CSR-style dyn_group_indptr.
    n_dyn_groups: int
    dyn_group_target_row: np.ndarray  # (n_dyn_groups,) int
    dyn_group_base_row: np.ndarray  # (n_dyn_groups,) int
    dyn_group_indptr: np.ndarray  # (n_dyn_groups + 1,) int -- CSR into substep arrays
    dyn_sub_func_id: np.ndarray  # (n_substeps,) int
    dyn_sub_param_rows: np.ndarray  # (n_substeps, max_dyn_params) int, -1 padded
    dyn_sub_n_params: np.ndarray  # (n_substeps,) int

    # --- Expression evaluation ---
    n_expressions: int
    expr_target_rows: np.ndarray  # (n_expressions,) int
    expr_programs: list[ExprProgram]

    # --- Interleaved parameter resolution schedule ---
    # Dynamics groups and expressions may depend on each other (e.g. a
    # dynamics param that is an expression of another dynamics param).
    # The resolution_kinds / resolution_indices arrays encode the correct
    # topological execution order:
    #   kind=0 -> dynamics group step, index into dyn_group_* arrays
    #   kind=1 -> expression step, index into expr_* arrays / expr_programs
    resolution_kinds: np.ndarray  # (n_dyn_groups + n_expressions,) int8
    resolution_indices: np.ndarray  # (n_dyn_groups + n_expressions,) int

    # --- Profile-varying parameter groups (fixed aux_axis shape) ---
    n_aux: int
    aux_axis: np.ndarray  # (n_aux,)
    n_profile_samples: int
    profile_sample_base_rows: np.ndarray  # (n_profile_samples,) int
    profile_sample_component_indptr: np.ndarray  # (n_profile_samples + 1,) int
    profile_component_func_ids: np.ndarray  # (n_profile_components,) int
    profile_component_param_indptr: np.ndarray  # (n_profile_components + 1,) int
    profile_component_param_rows: np.ndarray  # (total_profile_component_params,) int
    n_profile_exprs: int
    profile_expr_programs: list[ExprProgram]

    # --- Scheduled component ops ---
    n_ops: int
    op_schedule: np.ndarray  # (n_ops,) int
    op_kinds: np.ndarray  # (n_ops,) OpKind int codes
    op_param_indptr: np.ndarray  # (n_ops + 1,) int -- CSR row pointers
    op_param_source_kinds: np.ndarray  # (total_op_params,) ParamSourceKind int codes
    op_param_indices: np.ndarray  # (total_op_params,) int -- row/group indices
    op_needs_spectrum: np.ndarray  # (n_ops,) bool
    op_is_pre_spectrum: np.ndarray  # (n_ops,) bool
    op_is_profiled: np.ndarray  # (n_ops,) bool
    op_is_constant: np.ndarray  # (n_ops,) bool
    cached_result: np.ndarray  # (n_time, n_energy)
    cached_peak_sum: np.ndarray  # (n_time, n_energy)


#
#
@dataclass(frozen=True)
class ScheduledPlan1D:
    """Compiled 1D execution schedule for ENERGY_1D models.

    Simpler than ``ScheduledPlan2D``: no time axis, no dynamics, no
    trace matrix.  Parameters are stored as a flat ``(n_params,)``
    scalar vector.
    """

    energy: np.ndarray  # (n_energy,)
    n_params: int

    # --- Parameter mapping ---
    param_values_init: np.ndarray  # (n_params,) initial scalar values
    opt_indices: np.ndarray  # (n_opt,) int -- indices into param_values
    opt_param_names: list[str]  # (n_opt,) canonical optimizer param names

    # --- Expression evaluation (topological order, no dynamics) ---
    n_expressions: int
    expr_target_indices: np.ndarray  # (n_expressions,) int
    expr_programs: list[ExprProgram]

    # --- Profile-varying parameter groups (fixed aux_axis shape) ---
    n_aux: int
    aux_axis: np.ndarray  # (n_aux,)
    n_profile_samples: int
    profile_sample_base_indices: np.ndarray  # (n_profile_samples,) int
    profile_sample_component_indptr: np.ndarray  # (n_profile_samples + 1,) int
    profile_component_func_ids: np.ndarray  # (n_profile_components,) int
    profile_component_param_indptr: np.ndarray  # (n_profile_components + 1,) int
    profile_component_param_indices: np.ndarray  # (total_profile_component_params,) int
    n_profile_exprs: int
    profile_expr_programs: list[ExprProgram]

    # --- Scheduled component ops ---
    n_ops: int
    op_kinds: np.ndarray  # (n_ops,) OpKind int codes
    op_param_indptr: np.ndarray  # (n_ops + 1,) int -- CSR row pointers
    op_param_source_kinds: np.ndarray  # (total_op_params,) ParamSourceKind int codes
    op_param_indices: np.ndarray  # (total_op_params,) int -- source indices
    op_needs_spectrum: np.ndarray  # (n_ops,) bool
    op_is_pre_spectrum: np.ndarray  # (n_ops,) bool
    op_is_profiled: np.ndarray  # (n_ops,) bool
    op_is_constant: np.ndarray  # (n_ops,) bool
    cached_result: np.ndarray  # (n_energy,)
    cached_peak_sum: np.ndarray  # (n_energy,)


_PACKAGE_SHORT_NAMES: dict[str, str] = {
    "fcts_energy": "energy",
    "fcts_time": "time",
    "fcts_profile": "profile",
}


#
def _package_short_name(comp: Component) -> str:
    """Return ``"energy"``, ``"time"``, or ``"profile"`` for a component."""

    mod_name = comp.package_name.rsplit(".", maxsplit=1)[-1]
    return _PACKAGE_SHORT_NAMES.get(mod_name, mod_name)


#
def _par_initial_value(par: Par) -> float:
    """Extract the current scalar value from a Par's lmfit_par."""

    vals = list(par.lmfit_par.valuesdict().values())
    return float(vals[0]) if vals else 0.0


#
def _par_bounds(par: Par) -> tuple[float, float] | None:
    """Extract bounds from a Par's lmfit_par, or None."""

    for p in par.lmfit_par.values():
        mn = p.min if p.min is not None else -np.inf
        mx = p.max if p.max is not None else np.inf
        return (float(mn), float(mx))
    return None


#
def _is_expression_par(par: Par) -> bool:
    """True if this Par is defined by an expression string."""

    return len(par.info) == 1 and isinstance(par.info[0], str)


#
def _par_expression_string(par: Par) -> str | None:
    """Return the expression string to use for graph wiring.

    Prefer the lmfit expression when available because Dynamics models
    auto-prefix references there (e.g. ``expFun_01_A`` ->
    ``GLP_01_A_expFun_01_A`` or ``parTEST_expFun_01_A``). Fall back to
    the raw ``par.info[0]`` expression for plain energy-model expressions.
    """

    if not _is_expression_par(par):
        return None

    for lmfit_par in par.lmfit_par.values():
        if lmfit_par.expr:
            return str(lmfit_par.expr)

    return str(par.info[0])


#
def _extract_expression_references(expr_string: str) -> list[str]:
    """Extract identifier-like references from an expression string.

    This is intentionally generic: it collects all Python-identifier-like
    tokens in lexical order and leaves semantic filtering to the caller
    (for example, by checking whether the token is present in
    ``resolved_param`` or ``node_by_name``).
    """

    pattern = r"\b[A-Za-z_][A-Za-z0-9_]*\b"
    refs: list[str] = []
    seen: set[str] = set()
    for token in re.findall(pattern, expr_string):
        if token in seen:
            continue
        refs.append(token)
        seen.add(token)
    return refs


#
def _component_param_names(comp: Component) -> list[str]:
    """Return function parameter names, excluding axis arg and ``spectrum``."""

    args = comp.fct_args[1:]  # drop first (axis: x or t)
    if args and args[-1] == "spectrum":
        args = args[:-1]
    return args


#
def _par_is_vary(par: Par) -> bool:
    """True if the Par has ``vary=True`` in its lmfit parameter."""

    return bool(
        par.lmfit_par.valuesdict() and any(p.vary for p in par.lmfit_par.values())
    )


#
#
class _GraphBuilder:
    """Mutable state used while building a GraphIR from a Model."""

    def __init__(self) -> None:
        self.nodes: list[GraphNode] = []
        self.edges: list[GraphEdge] = []
        self.node_by_name: dict[str, int] = {}
        self._next_id: int = 0
        self._source_order: int = 0
        self._removed: set[int] = set()

    #
    def add_node(self, kind: NodeKind, name: str, **kwargs) -> int:
        """Create a node, assign it an id and source_order, return id."""

        nid = self._next_id
        self._next_id += 1
        order = self._source_order
        self._source_order += 1
        node = GraphNode(id=nid, kind=kind, name=name, source_order=order, **kwargs)
        self.nodes.append(node)
        self.node_by_name[name] = nid
        return nid

    #
    def add_edge(
        self, source: int, target: int, kind: EdgeKind, *, position: int | None = None
    ) -> None:
        self.edges.append(
            GraphEdge(source=source, target=target, kind=kind, position=position)
        )

    #
    def mark_removed(self, nid: int) -> None:
        """Mark a node for removal during finalization."""

        self._removed.add(nid)

    #
    def finalize(self) -> tuple[list[GraphNode], list[GraphEdge], dict[str, int]]:
        """Remove marked nodes/edges and re-index so node id == list position."""

        if not self._removed:
            return self.nodes, self.edges, dict(self.node_by_name)

        kept_nodes = [n for n in self.nodes if n.id not in self._removed]
        kept_edges = [
            e
            for e in self.edges
            if e.source not in self._removed and e.target not in self._removed
        ]

        # Re-index: old id -> new dense id
        id_map = {old.id: new_id for new_id, old in enumerate(kept_nodes)}
        for node in kept_nodes:
            node.id = id_map[node.id]
        for edge in kept_edges:
            edge.source = id_map[edge.source]
            edge.target = id_map[edge.target]

        node_by_name = {n.name: n.id for n in kept_nodes}
        return kept_nodes, kept_edges, node_by_name


#
def build_graph(model: Model) -> GraphIR:
    """Walk the OOP Model tree and emit a GraphIR.

    Parameters
    ----------
    model : Model
        A fully-constructed model (components, pars, and any dynamics /
        profile models already attached).

    Returns
    -------
    GraphIR
        The semantic DAG representation.
    """

    b = _GraphBuilder()

    # ----- determine domain -----
    has_energy = model.energy is not None
    has_time = model.time is not None
    if has_energy and has_time:
        domain = DomainKind.ENERGY_TIME_2D
    elif has_time:
        domain = DomainKind.TIME_1D
    else:
        domain = DomainKind.ENERGY_1D

    # ------------------------------------------------------------------ #
    # 1. Create parameter nodes for every component                       #
    # ------------------------------------------------------------------ #
    # Maps par.name -> node id of the *resolved* value (which might be a
    # PARAM_PLUS_TRACE node if time-dependent, or an EXPRESSION node).
    resolved_param: dict[str, int] = {}

    for comp in model.components:
        for par in comp.pars:
            _emit_par_nodes(b, par, resolved_param)

    # ------------------------------------------------------------------ #
    # 2. Create component nodes and wire PARAM_INPUT edges                #
    # ------------------------------------------------------------------ #
    # Collect component node ids for combination wiring.
    # comp_nodes: list of (comp, node_id, is_spectrum_fed)
    comp_nodes: list[tuple[Component, int, bool]] = []

    for comp in model.components:
        if comp.comp_type == "none":
            continue

        pkg_name = _package_short_name(comp)
        is_shirley = comp.fct_str == "Shirley"

        # Convolution components
        if comp.comp_type == "conv":
            nid = b.add_node(
                NodeKind.CONVOLUTION,
                comp.comp_name,
                function_name=comp.fct_str,
                package=pkg_name,
            )
            # Store kernel-related arrays if available
            if comp.time is not None:
                b.nodes[nid].arrays["kernel_time"] = comp.time
        elif is_shirley:
            nid = b.add_node(
                NodeKind.SPECTRUM_FED_OP,
                comp.comp_name,
                function_name=comp.fct_str,
                package=pkg_name,
            )
        else:
            nid = b.add_node(
                NodeKind.COMPONENT_EVAL,
                comp.comp_name,
                function_name=comp.fct_str,
                package=pkg_name,
            )

        # Wire PARAM_INPUT edges from resolved params to component
        param_names = _component_param_names(comp)
        for pos, _pname in enumerate(param_names):
            par = comp.pars[pos]
            src = resolved_param[par.name]
            b.add_edge(src, nid, EdgeKind.PARAM_INPUT, position=pos)

        comp_nodes.append((comp, nid, is_shirley))

    # ------------------------------------------------------------------ #
    # 3. Emit PROFILE_*, SUBCYCLE_* nodes for components that use them     #
    # ------------------------------------------------------------------ #
    # profile_samples maps par_name -> [sample_nid_0, ..., sample_nid_n].
    # Populated by p_vary components, consumed by expr_refs_profile_dep
    # components.  This preserves per-sample context across components so
    # that expression params referencing a profiled param get per-sample
    # EXPRESSION nodes (not a single averaged replacement).
    profile_samples: dict[str, list[int]] = {}
    for i, (comp, nid, is_shirley) in enumerate(comp_nodes):
        new_nid = _emit_profile_nodes(b, comp, nid, resolved_param, profile_samples)
        if new_nid != nid:
            b.mark_removed(nid)
            comp_nodes[i] = (comp, new_nid, is_shirley)
    for comp in model.components:
        if comp.comp_type == "none":
            continue
        _emit_subcycle_nodes(b, comp, resolved_param)

    # ------------------------------------------------------------------ #
    # 4. Create expression edges                                          #
    # ------------------------------------------------------------------ #
    # Runs after profile nodes.  Expression params that are
    # expr_refs_profile_dep are already fully wired per-sample inside
    # _emit_profile_nodes, so skip them here.
    for comp in model.components:
        for par in comp.pars:
            if not _is_expression_par(par):
                continue
            if par.expr_refs_profile_dep:
                continue  # handled per-sample in _emit_profile_nodes
            expr_nid = b.node_by_name[par.name]
            expr_str = _par_expression_string(par)
            if expr_str is None:
                continue
            refs = _extract_expression_references(expr_str)
            for ref_name in refs:
                if ref_name in resolved_param:
                    ref_nid = resolved_param[ref_name]
                    b.add_edge(ref_nid, expr_nid, EdgeKind.EXPR_REF)

    # ------------------------------------------------------------------ #
    # 5. Create SUM / combination nodes                                   #
    # ------------------------------------------------------------------ #
    _emit_combination_nodes(b, comp_nodes)

    nodes, edges, node_by_name = b.finalize()
    return GraphIR(
        nodes=nodes,
        edges=edges,
        domain=domain,
        energy=model.energy,
        time=model.time,
        node_by_name=node_by_name,
    )


#
def _emit_par_nodes(
    b: _GraphBuilder,
    par: Par,
    resolved_param: dict[str, int],
) -> None:
    """Create graph nodes for one Par (and its dynamics/profile subgraph)."""

    # Expression parameter
    if _is_expression_par(par):
        nid = b.add_node(
            NodeKind.EXPRESSION,
            par.name,
            expr_string=par.info[0],
            value=_par_initial_value(par),
        )
        resolved_param[par.name] = nid
        return

    # Base parameter node
    if _par_is_vary(par):
        base_nid = b.add_node(
            NodeKind.OPT_PARAM,
            par.name,
            value=_par_initial_value(par),
            vary=True,
            bounds=_par_bounds(par),
        )
    else:
        base_nid = b.add_node(
            NodeKind.STATIC_PARAM,
            par.name,
            value=_par_initial_value(par),
        )

    # Default: resolved value is the base node itself
    resolved_param[par.name] = base_nid

    # Time-dependent parameter (Dynamics subgraph)
    if par.t_vary and par.t_model is not None:
        _emit_dynamics_subgraph(b, par, base_nid, resolved_param)


#
def _emit_dynamics_subgraph(
    b: _GraphBuilder,
    par: Par,
    base_nid: int,
    resolved_param: dict[str, int],
) -> None:
    """Emit DYNAMICS_TRACE + PARAM_PLUS_TRACE nodes for a time-dep par."""

    t_model = par.t_model
    assert t_model is not None

    # Create parameter nodes for each dynamics component's parameters.
    # Dynamics pars can be expressions (e.g. multi-cycle subcycle models
    # where expFun_02_A = "-expFun_01_A").
    dyn_param_nids: list[list[int]] = []
    dyn_expr_pars: list[tuple[Par, int]] = []  # (par, node_id) for deferred wiring
    for dyn_comp in t_model.components:
        if dyn_comp.comp_type == "none":
            dyn_param_nids.append([])
            continue
        comp_nids: list[int] = []
        for dyn_par in dyn_comp.pars:
            if _is_expression_par(dyn_par):
                # Store the canonical (lmfit-prefixed) expression, not
                # the raw YAML text, so expr_string matches the EXPR_REF
                # edges wired below.
                canonical_expr = _par_expression_string(dyn_par) or dyn_par.info[0]
                dnid = b.add_node(
                    NodeKind.EXPRESSION,
                    dyn_par.name,
                    expr_string=canonical_expr,
                    value=_par_initial_value(dyn_par),
                )
                dyn_expr_pars.append((dyn_par, dnid))
            elif _par_is_vary(dyn_par):
                dnid = b.add_node(
                    NodeKind.OPT_PARAM,
                    dyn_par.name,
                    value=_par_initial_value(dyn_par),
                    vary=True,
                    bounds=_par_bounds(dyn_par),
                )
            else:
                dnid = b.add_node(
                    NodeKind.STATIC_PARAM,
                    dyn_par.name,
                    value=_par_initial_value(dyn_par),
                )
            comp_nids.append(dnid)
        dyn_param_nids.append(comp_nids)

    # Wire EXPR_REF edges for dynamics expression pars.
    # Dynamics expressions may be auto-prefixed by lmfit (e.g.
    # "expFun_01_A" -> "GLP_01_A_expFun_01_A" or "parTEST_expFun_01_A").
    # Use the canonical expression string and then match identifier refs
    # against graph node names.
    for dyn_par, expr_nid in dyn_expr_pars:
        expr_str = _par_expression_string(dyn_par)
        if expr_str is None:
            continue
        refs = _extract_expression_references(expr_str)
        for ref_name in refs:
            if ref_name in b.node_by_name:
                b.add_edge(b.node_by_name[ref_name], expr_nid, EdgeKind.EXPR_REF)

    # Create DYNAMICS_TRACE or CONVOLUTION nodes per dynamics component.
    # Convolution components (gaussCONV, etc.) are CONVOLUTION nodes that
    # wrap the resolved trace (conv(trace, kernel)), not addends.
    trace_nids: list[int] = []
    conv_nids: list[int] = []
    for i, dyn_comp in enumerate(t_model.components):
        if dyn_comp.comp_type == "none":
            continue

        node_name = f"{par.name}_dynamics"
        if len(t_model.components) > 1:
            node_name = f"{par.name}_{dyn_comp.comp_name}_dynamics"

        if dyn_comp.comp_type == "conv":
            nid = b.add_node(
                NodeKind.CONVOLUTION,
                node_name,
                function_name=dyn_comp.fct_str,
                package="time",
            )
            # Wire dynamics params -> conv node
            for pos, dnid in enumerate(dyn_param_nids[i]):
                b.add_edge(dnid, nid, EdgeKind.PARAM_INPUT, position=pos)
            conv_nids.append(nid)
        else:
            nid = b.add_node(
                NodeKind.DYNAMICS_TRACE,
                node_name,
                function_name=dyn_comp.fct_str,
                package="time",
            )
            # Wire dynamics params -> trace node
            for pos, dnid in enumerate(dyn_param_nids[i]):
                b.add_edge(dnid, nid, EdgeKind.PARAM_INPUT, position=pos)
            trace_nids.append(nid)

    # Create PARAM_PLUS_TRACE node (base + sum of traces)
    resolved_name = f"{par.name}_resolved"
    resolved_nid = b.add_node(
        NodeKind.PARAM_PLUS_TRACE,
        resolved_name,
    )
    b.add_edge(base_nid, resolved_nid, EdgeKind.BASE_INPUT)
    for trace_nid in trace_nids:
        b.add_edge(trace_nid, resolved_nid, EdgeKind.TRACE_INPUT)

    # Convolution nodes wrap the resolved trace: conv(resolved, kernel).
    # Each CONVOLUTION takes TRACE_INPUT from the current resolved node
    # and produces the new resolved value.
    for conv_nid in conv_nids:
        b.add_edge(resolved_nid, conv_nid, EdgeKind.TRACE_INPUT)
        resolved_nid = conv_nid

    resolved_param[par.name] = resolved_nid


#
def _rewire_param_input(
    b: _GraphBuilder, target: int, *, old_source: int, new_source: int
) -> None:
    """Replace the source of a PARAM_INPUT edge targeting *target*."""

    for edge in b.edges:
        if (
            edge.target == target
            and edge.kind == EdgeKind.PARAM_INPUT
            and edge.source == old_source
        ):
            edge.source = new_source
            return


#
def _rewire_trace_input(
    b: _GraphBuilder, target: int, *, old_source: int, new_source: int
) -> None:
    """Replace the source of a TRACE_INPUT edge targeting *target*."""

    for edge in b.edges:
        if (
            edge.target == target
            and edge.kind == EdgeKind.TRACE_INPUT
            and edge.source == old_source
        ):
            edge.source = new_source
            return


#
def _emit_profile_nodes(
    b: _GraphBuilder,
    comp: Component,
    comp_nid: int,
    resolved_param: dict[str, int],
    profile_samples: dict[str, list[int]],
) -> int:
    """Emit per-sample evaluation subgraph for profiled components.

    A component needs profile treatment when any of its params is
    ``p_vary`` (directly profiled) or ``expr_refs_profile_dep`` (an
    expression referencing a profiled param on another component).
    The interpreter evaluates the full component at each aux-axis point
    and averages the resulting traces: ``mean_i(f(p_0, ..., p_i, ...))``.

    Graph structure per aux point *i*:

    - For ``p_vary`` params: a PROFILE_SAMPLE node computes the
      parameter value at aux point *i* from the base value and the
      profile function.
    - For ``expr_refs_profile_dep`` params: a per-sample EXPRESSION
      node evaluates the expression using the PROFILE_SAMPLE of the
      referenced profiled param at the same aux point.
    - A per-sample COMPONENT_EVAL evaluates the component function
      with these per-sample inputs.

    After all aux points:

    - A component-level PROFILE_AVERAGE averages the per-sample
      COMPONENT_EVAL traces and replaces the original component in
      the combination graph.

    ``profile_samples`` is shared across components so that
    ``expr_refs_profile_dep`` params on one component can reference
    the PROFILE_SAMPLE nodes created for a ``p_vary`` param on
    a different component.

    Returns
    -------
    int
        Node id that replaces *comp_nid* in the combination graph,
        or *comp_nid* unchanged if no profile treatment is needed.
    """

    p_vary_pars = [p for p in comp.pars if p.p_vary and p.p_model is not None]
    expr_dep_pars = [p for p in comp.pars if p.expr_refs_profile_dep]
    if not p_vary_pars and not expr_dep_pars:
        return comp_nid

    # --- Determine aux_axis length ---
    n_aux: int | None = None
    if p_vary_pars:
        aux_axis = p_vary_pars[0].p_model.aux_axis  # type: ignore[union-attr]
        if aux_axis is None:
            return comp_nid
        n_aux = len(aux_axis)
    if n_aux is None:
        # expr_dep_pars only — infer n_aux from an already-populated
        # profile_samples entry.
        for ref_name in _expr_dep_profile_refs(expr_dep_pars):
            if ref_name in profile_samples:
                n_aux = len(profile_samples[ref_name])
                aux_axis = None  # not needed for expr-dep-only components
                break
    if n_aux is None:
        return comp_nid

    pkg_name = _package_short_name(comp)

    # --- Profile function parameter nodes for p_vary pars ---
    prof_param_nids_by_par: dict[str, list[int]] = {}
    for par in p_vary_pars:
        p_model = par.p_model
        assert p_model is not None

        prof_param_nids: list[int] = []
        prof_resolved: dict[str, int] = {}
        for prof_comp in p_model.components:
            if prof_comp.comp_type == "none":
                continue
            for prof_par in prof_comp.pars:
                _emit_par_nodes(b, prof_par, prof_resolved)
                prof_param_nids.append(prof_resolved[prof_par.name])

        prof_param_nids_by_par[par.name] = prof_param_nids

    # --- Per-sample nodes ---
    sample_comp_nids: list[int] = []

    for aux_i in range(n_aux):
        # PROFILE_SAMPLE for each p_vary param
        sample_nids: dict[str, int] = {}  # par.name -> sample nid
        for par in p_vary_pars:
            base_nid = resolved_param[par.name]
            p_aux = p_vary_pars[0].p_model.aux_axis  # type: ignore[union-attr]
            sample_nid = b.add_node(
                NodeKind.PROFILE_SAMPLE,
                f"{par.name}_profile_sample_{aux_i}",
                arrays={"aux_axis": p_aux},
            )
            b.add_edge(base_nid, sample_nid, EdgeKind.PARAM_INPUT, position=0)
            for pi, pnid in enumerate(prof_param_nids_by_par[par.name]):
                b.add_edge(pnid, sample_nid, EdgeKind.PARAM_INPUT, position=pi + 1)
            sample_nids[par.name] = sample_nid
            profile_samples.setdefault(par.name, []).append(sample_nid)

        # Per-sample EXPRESSION for each expr_refs_profile_dep param
        expr_sample_nids: dict[str, int] = {}  # par.name -> expr nid
        for par in expr_dep_pars:
            expr_str = _par_expression_string(par)
            if expr_str is None:
                continue
            expr_nid = b.add_node(
                NodeKind.EXPRESSION,
                f"{par.name}_profile_expr_{aux_i}",
                expr_string=expr_str,
                value=_par_initial_value(par),
            )
            # Wire EXPR_REF to the per-sample PROFILE_SAMPLE of the
            # referenced profiled param (not the averaged value).
            refs = _extract_expression_references(expr_str)
            for ref_name in refs:
                if ref_name in profile_samples:
                    b.add_edge(
                        profile_samples[ref_name][aux_i],
                        expr_nid,
                        EdgeKind.EXPR_REF,
                    )
                elif ref_name in resolved_param:
                    b.add_edge(
                        resolved_param[ref_name],
                        expr_nid,
                        EdgeKind.EXPR_REF,
                    )
            expr_sample_nids[par.name] = expr_nid

        # Per-sample COMPONENT_EVAL
        sample_eval_nid = b.add_node(
            NodeKind.COMPONENT_EVAL,
            f"{comp.comp_name}_sample_{aux_i}",
            function_name=comp.fct_str,
            package=pkg_name,
        )

        # Wire params into the sample component eval
        param_names = _component_param_names(comp)
        for pos, _pname in enumerate(param_names):
            par = comp.pars[pos]
            if par.name in sample_nids:
                src = sample_nids[par.name]
            elif par.name in expr_sample_nids:
                src = expr_sample_nids[par.name]
            else:
                src = resolved_param[par.name]
            b.add_edge(src, sample_eval_nid, EdgeKind.PARAM_INPUT, position=pos)

        sample_comp_nids.append(sample_eval_nid)

    # --- Component-level PROFILE_AVERAGE over sample traces ---
    comp_avg_nid = b.add_node(
        NodeKind.PROFILE_AVERAGE,
        f"{comp.comp_name}_profile_avg",
    )
    for sc_nid in sample_comp_nids:
        b.add_edge(sc_nid, comp_avg_nid, EdgeKind.ADDEND)

    return comp_avg_nid


#
def _expr_dep_profile_refs(expr_dep_pars: list[Par]) -> list[str]:
    """Collect profiled param names referenced by expr_refs_profile_dep pars."""

    refs: list[str] = []
    seen: set[str] = set()
    for par in expr_dep_pars:
        expr_str = _par_expression_string(par)
        if expr_str is None:
            continue
        for ref in _extract_expression_references(expr_str):
            if ref not in seen:
                refs.append(ref)
                seen.add(ref)
    return refs


#
def _emit_subcycle_nodes(
    b: _GraphBuilder,
    comp: Component,
    resolved_param: dict[str, int],
) -> None:
    """Emit SUBCYCLE_REMAP and SUBCYCLE_MASK nodes for subcycle dynamics.

    SUBCYCLE_REMAP feeds into the DYNAMICS_TRACE (provides the remapped
    time axis).  SUBCYCLE_MASK consumes the DYNAMICS_TRACE output
    (zeroes inactive regions).
    """

    # Standalone TIME_1D dynamics model: subcycle info lives directly on the
    # time component itself rather than on a parent Par.t_model.
    comp_nid = b.node_by_name.get(comp.comp_name)
    if (
        comp_nid is not None
        and comp.subcycle != 0
        and comp.time_norm is not None
        and comp.time_n_sub is not None
    ):
        remap_nid = b.add_node(
            NodeKind.SUBCYCLE_REMAP,
            f"{comp.comp_name}_remap",
            arrays={"time_norm": comp.time_norm},
        )
        b.add_edge(remap_nid, comp_nid, EdgeKind.TRACE_INPUT)

        mask_nid = b.add_node(
            NodeKind.SUBCYCLE_MASK,
            f"{comp.comp_name}_mask",
            arrays={"time_n_sub": comp.time_n_sub},
        )
        b.add_edge(comp_nid, mask_nid, EdgeKind.TRACE_INPUT)

    for par in comp.pars:
        if not par.t_vary or par.t_model is None:
            continue
        t_model = par.t_model
        for dyn_comp in t_model.components:
            if dyn_comp.subcycle == 0:
                continue
            if dyn_comp.time_norm is None or dyn_comp.time_n_sub is None:
                continue

            # Find the DYNAMICS_TRACE node for this dynamics component
            trace_name = f"{par.name}_dynamics"
            if len(t_model.components) > 1:
                trace_name = f"{par.name}_{dyn_comp.comp_name}_dynamics"
            trace_nid = b.node_by_name.get(trace_name)
            if trace_nid is None:
                continue

            # SUBCYCLE_REMAP before the dynamics trace
            remap_nid = b.add_node(
                NodeKind.SUBCYCLE_REMAP,
                f"{par.name}_{dyn_comp.comp_name}_remap",
                arrays={"time_norm": dyn_comp.time_norm},
            )
            b.add_edge(remap_nid, trace_nid, EdgeKind.TRACE_INPUT)

            # SUBCYCLE_MASK after the dynamics trace
            mask_nid = b.add_node(
                NodeKind.SUBCYCLE_MASK,
                f"{par.name}_{dyn_comp.comp_name}_mask",
                arrays={"time_n_sub": dyn_comp.time_n_sub},
            )
            b.add_edge(trace_nid, mask_nid, EdgeKind.TRACE_INPUT)

            # Rewire PARAM_PLUS_TRACE to consume the masked trace
            # instead of the raw DYNAMICS_TRACE.
            resolved_nid = resolved_param.get(par.name)
            if resolved_nid is not None:
                _rewire_trace_input(
                    b, resolved_nid, old_source=trace_nid, new_source=mask_nid
                )


#
def _emit_combination_nodes(
    b: _GraphBuilder,
    comp_nodes: list[tuple[Component, int, bool]],
) -> None:
    """Create SUM nodes that mirror the model's LIFO combine semantics.

    Classification:

    - **peaks**: ``comp_type == "add"`` — feed ``peak_sum``
    - **backgrounds**: ``comp_type == "back"`` but *not* spectrum-fed
      (Offset, LinBack) — feed ``total`` directly, *not* ``peak_sum``
    - **spectrum-fed**: Shirley — receives ``SPECTRUM_INPUT`` from
      ``peak_sum``, feeds ``total``
    - **convolution**: ``comp_type == "conv"`` — receives ``ADDEND``
      from ``peak_sum``, feeds ``total``

    This matches the spec example (lowered_evaluator.md lines 298-305):
    only peaks contribute to ``peak_sum``; Offset/LinBack are added
    at the ``total`` level.
    """

    if not comp_nodes:
        return

    # Classify nodes
    peaks: list[int] = []  # comp_type == "add"
    backgrounds: list[int] = []  # Offset, LinBack (comp_type == "back", not Shirley)
    spectrum_fed: list[int] = []  # Shirley
    convolution: list[int] = []  # conv components

    for comp, nid, is_shirley in comp_nodes:
        if comp.comp_type == "conv":
            convolution.append(nid)
        elif is_shirley:
            spectrum_fed.append(nid)
        elif comp.comp_type == "add":
            peaks.append(nid)
        else:
            # comp_type == "back" but not Shirley -> Offset, LinBack
            backgrounds.append(nid)

    # peak_sum: accumulates only peak (comp_type == "add") components
    peak_sum_nid: int | None = None
    if peaks:
        peak_sum_nid = b.add_node(NodeKind.SUM, "peak_sum")
        for nid in peaks:
            b.add_edge(nid, peak_sum_nid, EdgeKind.ADDEND)

    # Wire SPECTRUM_INPUT edges from peak_sum to spectrum-fed ops
    if peak_sum_nid is not None:
        for nid in spectrum_fed:
            b.add_edge(peak_sum_nid, nid, EdgeKind.SPECTRUM_INPUT)

    # Convolution nodes get ADDEND edge from peak_sum (signal to convolve)
    if peak_sum_nid is not None:
        for nid in convolution:
            b.add_edge(peak_sum_nid, nid, EdgeKind.ADDEND)

    # total: final sum of everything
    # Collect all addends for total.  Use peak_sum (not individual peaks)
    # so the graph reflects the semantic grouping.
    total_addends: list[int] = []
    if peak_sum_nid is not None:
        total_addends.append(peak_sum_nid)
    total_addends.extend(backgrounds)
    total_addends.extend(spectrum_fed)
    total_addends.extend(convolution)

    if len(total_addends) > 1:
        total_nid = b.add_node(NodeKind.SUM, "total")
        for nid in total_addends:
            b.add_edge(nid, total_nid, EdgeKind.ADDEND)


# ---------------------------------------------------------------------------
# can_lower_2d
# ---------------------------------------------------------------------------

_LOWERABLE_2D_FUNCTIONS: frozenset[str] = frozenset(
    {
        "Gauss",
        "GaussAsym",
        "Lorentz",
        "Voigt",
        "GLS",
        "GLP",
        "DS",
        "Offset",
        "LinBack",
        "Shirley",
    }
)

# Base set of non-lowerable node kinds shared across backends.
# Backend-specific sets below carve out nodes as each backend gains
# support (e.g. profile nodes for 1D and 2D).
_NON_LOWERABLE_NODE_KINDS_BASE: frozenset[NodeKind] = frozenset(
    {
        NodeKind.CONVOLUTION,
        NodeKind.PROFILE_SAMPLE,
        NodeKind.PROFILE_AVERAGE,
        NodeKind.SUBCYCLE_MASK,
        NodeKind.SUBCYCLE_REMAP,
    }
)

# 2D backend: profiles are now compilable (Phase 6.2b).
_NON_LOWERABLE_2D_NODE_KINDS: frozenset[NodeKind] = (
    _NON_LOWERABLE_NODE_KINDS_BASE
    - frozenset({NodeKind.PROFILE_SAMPLE, NodeKind.PROFILE_AVERAGE})
)


#
def can_lower_2d(graph: GraphIR) -> bool:
    """Check whether the 2D NumPy backend can compile this graph.

    Parameters
    ----------
    graph : GraphIR
        The model graph to check.

    Returns
    -------
    bool
        True if ``schedule_2d`` can compile this graph.
    """

    if graph.domain != DomainKind.ENERGY_TIME_2D:
        return False

    for node in graph.nodes:
        # Reject future node types not yet compilable
        if node.kind in _NON_LOWERABLE_2D_NODE_KINDS:
            return False

        # Check component functions are supported
        if node.kind in (NodeKind.COMPONENT_EVAL, NodeKind.SPECTRUM_FED_OP):
            if node.function_name not in _LOWERABLE_2D_FUNCTIONS:
                return False

        # Check dynamics functions are supported
        if node.kind == NodeKind.DYNAMICS_TRACE:
            if node.function_name not in _FUNCTION_NAME_TO_DYN_FUNC:
                return False

        # Check profile nodes have required aux_axis array
        if node.kind == NodeKind.PROFILE_SAMPLE:
            if "aux_axis" not in node.arrays:
                return False

        # Check expressions are arithmetic-only (defer full AST check
        # to the expression compiler; here just reject obvious non-starters)
        if node.kind == NodeKind.EXPRESSION and node.expr_string is not None:
            if not _is_arithmetic_expression(node.expr_string):
                return False

    return True


# Node kinds that are never valid in 1D energy models.  1D models have
# no time axis, so DYNAMICS_TRACE / PARAM_PLUS_TRACE should not appear.
# Start from the 2D blocklist so future unsupported node kinds propagate
# automatically, then carve out the Phase 6.2 profile nodes that 1D now
# lowers explicitly.
_NON_LOWERABLE_1D_NODE_KINDS: frozenset[NodeKind] = (
    _NON_LOWERABLE_NODE_KINDS_BASE
    - frozenset({NodeKind.PROFILE_SAMPLE, NodeKind.PROFILE_AVERAGE})
    | frozenset(
        {
            NodeKind.DYNAMICS_TRACE,
            NodeKind.PARAM_PLUS_TRACE,
        }
    )
)


#
def can_lower_1d(graph: GraphIR) -> bool:
    """Check whether the 1D NumPy backend can compile this graph.

    Parameters
    ----------
    graph : GraphIR
        The model graph to check.

    Returns
    -------
    bool
        True if ``schedule_1d`` can compile this graph.
    """

    if graph.domain != DomainKind.ENERGY_1D:
        return False

    for node in graph.nodes:
        if node.kind in _NON_LOWERABLE_1D_NODE_KINDS:
            return False

        if node.kind in (NodeKind.COMPONENT_EVAL, NodeKind.SPECTRUM_FED_OP):
            if node.function_name not in _LOWERABLE_2D_FUNCTIONS:
                return False

        if node.kind == NodeKind.PROFILE_SAMPLE:
            if "aux_axis" not in node.arrays:
                return False

        if node.kind == NodeKind.EXPRESSION and node.expr_string is not None:
            if not _is_arithmetic_expression(node.expr_string):
                return False

    return True


#
def _is_arithmetic_expression(expr_string: str) -> bool:
    """Return True if the expression uses only arithmetic ops and param refs.

    Does a lightweight check via the ``ast`` module.  Rejects function
    calls, attribute access, subscripts, etc.
    """

    import ast

    try:
        tree = ast.parse(expr_string, mode="eval")
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.Expression,
                ast.BinOp,
                ast.UnaryOp,
                ast.Constant,
                ast.Name,
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.Pow,
                ast.USub,
                ast.UAdd,
                ast.Load,
            ),
        ):
            continue
        return False
    return True


# ---------------------------------------------------------------------------
# Symbolic expression compiler (Phase 2.1)
# ---------------------------------------------------------------------------


#
#
@dataclass(frozen=True)
class SymbolicRPN:
    """Symbolic RPN program with parameter references by name.

    This is the *frontend* output of the expression compiler.
    ``schedule_2d`` binds names to trace-matrix row indices and
    produces the final ``ExprProgram``.

    Each instruction is a ``(ExprNodeKind, operand)`` pair:

    - ``CONST``: operand is the float value itself
    - ``PARAM_REF``: operand is the parameter name (str)
    - Operators: operand is ``None``
    """

    instructions: list[tuple[ExprNodeKind, float | str | None]]
    referenced_names: list[str]  # unique param names in order of first appearance


#
def compile_expr_symbolic(expr_string: str) -> SymbolicRPN:
    """Parse an arithmetic expression string into symbolic RPN.

    Uses the Python ``ast`` module to walk the expression tree and
    emit a postfix instruction sequence.  Parameter references are
    kept as name strings; the scheduler resolves them to row indices.

    Parameters
    ----------
    expr_string : str
        Arithmetic expression (e.g. ``"3/4*GLP_01_A"``).

    Returns
    -------
    SymbolicRPN
        The symbolic RPN program.

    Raises
    ------
    ValueError
        If the expression contains unsupported AST nodes.
    """

    import ast

    tree = ast.parse(expr_string, mode="eval")

    instructions: list[tuple[ExprNodeKind, float | str | None]] = []
    names_seen: dict[str, None] = {}  # ordered set via dict

    _OP_MAP: dict[type, ExprNodeKind] = {
        ast.Add: ExprNodeKind.ADD,
        ast.Sub: ExprNodeKind.SUB,
        ast.Mult: ExprNodeKind.MUL,
        ast.Div: ExprNodeKind.DIV,
        ast.Pow: ExprNodeKind.POW,
    }

    #
    def _walk(node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            _walk(node.body)

        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError(
                    f"Unsupported constant {node.value!r}"
                    f" in expression: {expr_string!r}"
                )
            instructions.append((ExprNodeKind.CONST, float(node.value)))

        elif isinstance(node, ast.Name):
            instructions.append((ExprNodeKind.PARAM_REF, node.id))
            names_seen.setdefault(node.id, None)

        elif isinstance(node, ast.BinOp):
            _walk(node.left)
            _walk(node.right)
            op_kind = _OP_MAP.get(type(node.op))
            if op_kind is None:
                raise ValueError(
                    f"Unsupported binary op {type(node.op).__name__!r}"
                    f" in expression: {expr_string!r}"
                )
            instructions.append((op_kind, None))

        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                _walk(node.operand)
                instructions.append((ExprNodeKind.NEG, None))
            elif isinstance(node.op, ast.UAdd):
                _walk(node.operand)
                # UAdd is a no-op
            else:
                raise ValueError(
                    f"Unsupported unary op {type(node.op).__name__!r}"
                    f" in expression: {expr_string!r}"
                )

        else:
            raise ValueError(
                f"Unsupported AST node {type(node).__name__!r}"
                f" in expression: {expr_string!r}"
            )

    _walk(tree)
    return SymbolicRPN(
        instructions=instructions,
        referenced_names=list(names_seen),
    )


#
def _bind_expr_to_rows(
    symbolic: SymbolicRPN,
    name_to_row: dict[str, int],
) -> ExprProgram:
    """Convert a symbolic RPN program to a row-bound ExprProgram.

    Parameters
    ----------
    symbolic : SymbolicRPN
        The symbolic RPN from ``compile_expr_symbolic``.
    name_to_row : dict[str, int]
        Maps parameter names to trace-matrix row indices.

    Returns
    -------
    ExprProgram
        Row-bound RPN program ready for the evaluator.
    """

    flat: list[int] = []
    for kind, operand in symbolic.instructions:
        flat.append(int(kind))
        if kind == ExprNodeKind.CONST:
            assert isinstance(operand, (int, float))
            flat.append(int(np.float64(operand).view(np.int64)))
        elif kind == ExprNodeKind.PARAM_REF:
            assert isinstance(operand, str)
            flat.append(name_to_row[operand])
        else:
            flat.append(0)
    return ExprProgram(instructions=np.array(flat, dtype=np.int64))


# ---------------------------------------------------------------------------
# schedule_2d (Phase 2.2)
# ---------------------------------------------------------------------------


#
def _topological_sort(graph: GraphIR) -> list[int]:
    """Topological sort of graph node IDs.

    Tie-breaker: when two nodes have no dependency ordering between
    them, sort by ``node.source_order`` (lower first).  This makes the
    schedule deterministic.

    Does NOT assume ``node.id == list index``.
    """

    import heapq

    id_to_node: dict[int, GraphNode] = {n.id: n for n in graph.nodes}
    in_degree: dict[int, int] = {n.id: 0 for n in graph.nodes}
    children: dict[int, list[int]] = {n.id: [] for n in graph.nodes}
    for edge in graph.edges:
        children[edge.source].append(edge.target)
        in_degree[edge.target] += 1

    # Priority queue: (source_order, node_id) — lower source_order first
    heap: list[tuple[int, int]] = []
    for node in graph.nodes:
        if in_degree[node.id] == 0:
            heapq.heappush(heap, (node.source_order, node.id))

    result: list[int] = []
    while heap:
        _order, nid = heapq.heappop(heap)
        result.append(nid)
        for child in children[nid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                heapq.heappush(
                    heap,
                    (id_to_node[child].source_order, child),
                )

    n = len(graph.nodes)
    if len(result) != n:
        raise ValueError(f"Graph has a cycle: sorted {len(result)} of {n} nodes")
    return result


#
def schedule_2d(graph: GraphIR) -> ScheduledPlan2D:
    """Compile a GraphIR into a flat 2D execution schedule.

    Parameters
    ----------
    graph : GraphIR
        Must pass ``can_lower_2d(graph)``.

    Returns
    -------
    ScheduledPlan2D
        Packed-array execution schedule for ``evaluate_2d``.

    Raises
    ------
    ValueError
        If the graph cannot be lowered (domain, unsupported nodes, etc.).
    """

    if not can_lower_2d(graph):
        raise ValueError("Graph cannot be lowered to 2D backend")

    assert graph.energy is not None
    assert graph.time is not None
    n_time = len(graph.time)

    # ------------------------------------------------------------------ #
    # 1. Topological sort                                                  #
    # ------------------------------------------------------------------ #
    topo_order = _topological_sort(graph)

    # Build id -> node lookup.  Do NOT assume node.id == list index;
    # external graph producers may use arbitrary ids.
    id_to_node: dict[int, GraphNode] = {n.id: n for n in graph.nodes}

    # ------------------------------------------------------------------ #
    # 2. Assign trace-matrix rows                                          #
    # ------------------------------------------------------------------ #
    # Nodes that occupy a row in the trace matrix: all parameter-like
    # nodes (STATIC_PARAM, OPT_PARAM, PARAM_PLUS_TRACE, EXPRESSION).
    _ROW_KINDS = frozenset(
        {
            NodeKind.STATIC_PARAM,
            NodeKind.OPT_PARAM,
            NodeKind.PARAM_PLUS_TRACE,
            NodeKind.EXPRESSION,
        }
    )

    # Collect nodes that need rows, bucketed for ordering:
    #   opt params first, then static, then computed (PPT + EXPR)
    opt_nodes: list[GraphNode] = []
    static_nodes: list[GraphNode] = []
    computed_nodes: list[GraphNode] = []

    for nid in topo_order:
        node = id_to_node[nid]
        if node.kind not in _ROW_KINDS or _is_profile_expr_node(node):
            continue
        if node.kind == NodeKind.OPT_PARAM and node.vary:
            opt_nodes.append(node)
        elif node.kind in (NodeKind.STATIC_PARAM, NodeKind.OPT_PARAM):
            # OPT_PARAM with vary=False is treated like static
            static_nodes.append(node)
        else:
            computed_nodes.append(node)

    all_row_nodes = opt_nodes + static_nodes + computed_nodes
    n_params = len(all_row_nodes)
    name_to_row: dict[str, int] = {}
    for row, node in enumerate(all_row_nodes):
        name_to_row[node.name] = row
    row_is_constant = np.zeros(n_params, dtype=np.bool_)
    for node in static_nodes:
        row_is_constant[name_to_row[node.name]] = True

    # opt_indices and opt_param_names
    n_opt = len(opt_nodes)
    opt_indices = np.arange(n_opt, dtype=np.intp)
    opt_param_names = [n.name for n in opt_nodes]

    # ------------------------------------------------------------------ #
    # 3. Compile dynamics subgraphs (grouped by PARAM_PLUS_TRACE)          #
    # ------------------------------------------------------------------ #
    # First pass: extract per-DYNAMICS_TRACE info and find which PPT
    # each trace feeds.
    dyn_trace_nodes = [
        id_to_node[nid]
        for nid in topo_order
        if id_to_node[nid].kind == NodeKind.DYNAMICS_TRACE
    ]

    # Per-trace (substep) info, indexed by position in dyn_trace_nodes.
    sub_func_ids: list[int] = []
    sub_param_row_lists: list[list[int]] = []
    sub_ppt_row: list[int] = []  # which PPT row this trace targets
    sub_base_row: list[int] = []

    for dyn_node in dyn_trace_nodes:
        assert dyn_node.function_name is not None
        func_kind = _FUNCTION_NAME_TO_DYN_FUNC.get(dyn_node.function_name)
        if func_kind is None:
            raise ValueError(f"Unknown dynamics function: {dyn_node.function_name!r}")
        sub_func_ids.append(int(func_kind))

        # Param rows (from PARAM_INPUT edges, ordered by position)
        param_edges = sorted(
            (
                e
                for e in graph.edges
                if e.target == dyn_node.id and e.kind == EdgeKind.PARAM_INPUT
            ),
            key=lambda e: e.position or 0,
        )
        sub_param_row_lists.append(
            [name_to_row[id_to_node[e.source].name] for e in param_edges]
        )

        # Target: the PARAM_PLUS_TRACE node this trace feeds
        ppt_edges = [
            e
            for e in graph.edges
            if e.source == dyn_node.id and e.kind == EdgeKind.TRACE_INPUT
        ]
        assert len(ppt_edges) == 1, (
            f"DYNAMICS_TRACE '{dyn_node.name}' must feed exactly one"
            f" PARAM_PLUS_TRACE, found {len(ppt_edges)}"
        )
        ppt_node = id_to_node[ppt_edges[0].target]
        sub_ppt_row.append(name_to_row[ppt_node.name])

        # Base row: the BASE_INPUT to the PARAM_PLUS_TRACE
        base_edges = [
            e
            for e in graph.edges
            if e.target == ppt_node.id and e.kind == EdgeKind.BASE_INPUT
        ]
        assert len(base_edges) == 1
        sub_base_row.append(name_to_row[id_to_node[base_edges[0].source].name])

    # Second pass: group substeps by target PPT row, preserving topo order.
    # Each unique PPT row becomes one dynamics group.
    seen_ppt: dict[int, int] = {}  # ppt_row -> group index
    group_target_rows: list[int] = []
    group_base_rows: list[int] = []
    group_substeps: list[list[int]] = []  # group_idx -> [substep indices]

    for sub_idx, ppt_row in enumerate(sub_ppt_row):
        if ppt_row not in seen_ppt:
            gid = len(group_target_rows)
            seen_ppt[ppt_row] = gid
            group_target_rows.append(ppt_row)
            group_base_rows.append(sub_base_row[sub_idx])
            group_substeps.append([])
        group_substeps[seen_ppt[ppt_row]].append(sub_idx)

    n_dyn_groups = len(group_target_rows)
    n_substeps = len(dyn_trace_nodes)

    # Pack substep arrays (flat, ordered by group then topo within group)
    flat_sub_indices: list[int] = []
    indptr: list[int] = [0]
    for subs in group_substeps:
        flat_sub_indices.extend(subs)
        indptr.append(indptr[-1] + len(subs))

    max_dyn_params = (
        max(len(r) for r in sub_param_row_lists) if sub_param_row_lists else 0
    )
    dyn_sub_param_rows = np.full((n_substeps, max_dyn_params), -1, dtype=np.intp)
    dyn_sub_n_params_list: list[int] = []
    dyn_sub_func_id_list: list[int] = []
    for flat_i, orig_i in enumerate(flat_sub_indices):
        rows = sub_param_row_lists[orig_i]
        dyn_sub_param_rows[flat_i, : len(rows)] = rows
        dyn_sub_n_params_list.append(len(rows))
        dyn_sub_func_id_list.append(sub_func_ids[orig_i])

    dyn_group_target_row = np.array(group_target_rows, dtype=np.intp)
    dyn_group_base_row = np.array(group_base_rows, dtype=np.intp)
    dyn_group_indptr = np.array(indptr, dtype=np.intp)
    dyn_sub_func_id = np.array(dyn_sub_func_id_list, dtype=np.intp)
    dyn_sub_n_params = np.array(dyn_sub_n_params_list, dtype=np.intp)

    # ------------------------------------------------------------------ #
    # 4. Compile expressions (topological order)                           #
    # ------------------------------------------------------------------ #
    expr_nodes_topo = [
        id_to_node[nid]
        for nid in topo_order
        if id_to_node[nid].kind == NodeKind.EXPRESSION
        and not _is_profile_expr_node(id_to_node[nid])
    ]
    n_expressions = len(expr_nodes_topo)

    # Build per-expression name→row maps from EXPR_REF edges.
    # The graph's EXPR_REF edges are the single source of truth for
    # which node each name in the expression resolves to.  We walk
    # those edges to build a per-node override map, then compile the
    # symbolic RPN (for operator structure) and bind names to rows
    # using the edge-derived map.
    #
    # Index: expr_node.id -> {identifier_in_expr_string -> row}
    expr_ref_maps: dict[int, dict[str, int]] = {}
    for expr_node in expr_nodes_topo:
        ref_map: dict[str, int] = {}
        for edge in graph.edges:
            if edge.target != expr_node.id or edge.kind != EdgeKind.EXPR_REF:
                continue
            src_node = id_to_node[edge.source]
            src_row = name_to_row[src_node.name]
            # The expr_string references names that appear as identifiers.
            # The EXPR_REF source node's name is the canonical form (may
            # include "_resolved" suffix or prefixed dynamics names).
            # Walk the expression's identifier tokens to find which one
            # this edge satisfies.
            assert expr_node.expr_string is not None
            for token in _extract_expression_references(expr_node.expr_string):
                if token == src_node.name:
                    ref_map[token] = src_row
                    break
            else:
                # The identifier in the expression doesn't match the
                # source node name literally.  This shouldn't happen if
                # build_graph stored the canonical expr_string, but
                # fall back: try matching any identifier that names a
                # parameter whose resolved row IS this source row.
                for token in _extract_expression_references(expr_node.expr_string):
                    if token in name_to_row and name_to_row[token] == src_row:
                        ref_map[token] = src_row
                        break
                    # Also try: source is a resolved node, token is the
                    # base param name
                    if token not in ref_map and src_node.name.endswith("_resolved"):
                        base_name = src_node.name.removesuffix("_resolved")
                        if token == base_name:
                            ref_map[token] = src_row
                            break
        expr_ref_maps[expr_node.id] = ref_map

    # Compile and bind expressions
    expr_programs: list[ExprProgram] = []
    expr_target_rows_list: list[int] = []
    for expr_node in expr_nodes_topo:
        assert expr_node.expr_string is not None
        symbolic = compile_expr_symbolic(expr_node.expr_string)

        # Build the binding map: start with the base name_to_row, then
        # overlay the edge-derived overrides for this expression.
        binding = dict(name_to_row)
        binding.update(expr_ref_maps.get(expr_node.id, {}))

        program = _bind_expr_to_rows(symbolic, binding)
        expr_programs.append(program)
        target_row = name_to_row[expr_node.name]
        expr_target_rows_list.append(target_row)
        row_is_constant[target_row] = all(
            row_is_constant[binding[name]] for name in symbolic.referenced_names
        )

    expr_target_rows = np.array(expr_target_rows_list, dtype=np.intp)

    # Build resolution schedule: map DYNAMICS_TRACE node ids to their
    # group index, and emit each group exactly once (on the *last* trace
    # in that group, so all expression deps are resolved first).
    _dyn_id_to_ppt = {n.id: sub_ppt_row[i] for i, n in enumerate(dyn_trace_nodes)}
    _ppt_to_group = seen_ppt  # ppt_row -> group index
    _expr_id_to_idx = {n.id: i for i, n in enumerate(expr_nodes_topo)}

    # Find which DYNAMICS_TRACE is the last in each group (in topo order).
    # We emit the group step at that point so all substep expressions are
    # resolved before the group evaluates.
    _last_dyn_in_group: dict[int, int] = {}  # group_idx -> last dyn node id
    for nid in topo_order:
        if nid in _dyn_id_to_ppt:
            gid = _ppt_to_group[_dyn_id_to_ppt[nid]]
            _last_dyn_in_group[gid] = nid

    resolution_kinds_list: list[int] = []
    resolution_indices_list: list[int] = []
    emitted_groups: set[int] = set()
    for nid in topo_order:
        if nid in _dyn_id_to_ppt:
            gid = _ppt_to_group[_dyn_id_to_ppt[nid]]
            if _last_dyn_in_group[gid] == nid and gid not in emitted_groups:
                resolution_kinds_list.append(0)  # dynamics group
                resolution_indices_list.append(gid)
                emitted_groups.add(gid)
        elif nid in _expr_id_to_idx:
            resolution_kinds_list.append(1)  # expression
            resolution_indices_list.append(_expr_id_to_idx[nid])

    resolution_kinds = np.array(resolution_kinds_list, dtype=np.int8)
    resolution_indices = np.array(resolution_indices_list, dtype=np.intp)

    # ------------------------------------------------------------------ #
    # 4b. Compile PROFILE_SAMPLE groups                                    #
    # ------------------------------------------------------------------ #
    # Build edge indexes for profile compilation (mirrors schedule_1d).
    param_edges_by_target: dict[int, list[GraphEdge]] = {}
    expr_ref_edges_by_target: dict[int, list[GraphEdge]] = {}
    addend_edges_by_target: dict[int, list[GraphEdge]] = {}
    spectrum_input_targets: set[int] = set()
    for edge in graph.edges:
        if edge.kind == EdgeKind.PARAM_INPUT:
            param_edges_by_target.setdefault(edge.target, []).append(edge)
        elif edge.kind == EdgeKind.EXPR_REF:
            expr_ref_edges_by_target.setdefault(edge.target, []).append(edge)
        elif edge.kind == EdgeKind.ADDEND:
            addend_edges_by_target.setdefault(edge.target, []).append(edge)
        elif edge.kind == EdgeKind.SPECTRUM_INPUT:
            spectrum_input_targets.add(edge.target)

    profile_sample_groups: dict[str, list[GraphNode]] = {}
    for nid in topo_order:
        node = id_to_node[nid]
        if node.kind != NodeKind.PROFILE_SAMPLE:
            continue
        group_name = _profile_group_name(node.name, "profile_sample")
        profile_sample_groups.setdefault(group_name, []).append(node)

    plan_aux_axis = np.zeros(0, dtype=np.float64)
    n_aux = 0
    profile_sample_base_rows_list: list[int] = []
    profile_sample_component_indptr_list: list[int] = [0]
    profile_component_func_ids_list: list[int] = []
    profile_component_param_indptr_list: list[int] = [0]
    profile_component_param_rows_list: list[int] = []
    profile_sample_is_constant_list: list[bool] = []
    profile_sample_group_idx: dict[str, int] = {}

    for group_name, sample_nodes in profile_sample_groups.items():
        sample_nodes_sorted = sorted(
            sample_nodes,
            key=lambda node: _profile_group_index(node.name, "profile_sample"),
        )
        aux_indices = [
            _profile_group_index(node.name, "profile_sample")
            for node in sample_nodes_sorted
        ]
        if aux_indices != list(range(len(sample_nodes_sorted))):
            raise ValueError(
                f"PROFILE_SAMPLE nodes for {group_name!r} do not cover "
                "a contiguous aux-axis range"
            )

        aux_axis = sample_nodes_sorted[0].arrays.get("aux_axis")
        if aux_axis is None:
            raise ValueError(f"PROFILE_SAMPLE {group_name!r} is missing aux_axis")
        aux_axis = np.asarray(aux_axis, dtype=np.float64)
        if n_aux == 0:
            n_aux = len(aux_axis)
            plan_aux_axis = aux_axis.copy()
        elif len(aux_axis) != n_aux or not np.array_equal(aux_axis, plan_aux_axis):
            raise ValueError("All lowered profile groups must share one fixed aux_axis")
        if len(sample_nodes_sorted) != n_aux:
            raise ValueError(
                f"PROFILE_SAMPLE group {group_name!r} has "
                f"{len(sample_nodes_sorted)} samples but aux_axis length {n_aux}"
            )

        rep_node = sample_nodes_sorted[0]
        rep_param_edges = sorted(
            param_edges_by_target.get(rep_node.id, []),
            key=lambda edge: edge.position or 0,
        )
        if not rep_param_edges:
            raise ValueError(f"PROFILE_SAMPLE {group_name!r} has no PARAM_INPUT edges")

        base_node = id_to_node[rep_param_edges[0].source]
        base_row = name_to_row[base_node.name]
        is_constant = bool(row_is_constant[base_row])
        profile_sample_base_rows_list.append(base_row)

        component_func_by_name: dict[str, int] = {}
        component_param_rows_by_name: dict[str, list[int]] = {}
        component_order: list[str] = []
        for edge in rep_param_edges[1:]:
            src_node = id_to_node[edge.source]
            src_row = name_to_row[src_node.name]
            # PARAM_PLUS_TRACE nodes have a "_resolved" suffix; strip
            # it for profile component parsing but use the resolved row.
            parse_name = src_node.name
            if src_node.kind == NodeKind.PARAM_PLUS_TRACE:
                parse_name = parse_name.removesuffix("_resolved")
            comp_name, func_name = _parse_profile_component_param_name(
                group_name,
                parse_name,
            )
            prof_func_kind = _FUNCTION_NAME_TO_PROFILE_FUNC.get(func_name)
            if prof_func_kind is None:
                raise ValueError(f"Unknown profile function: {func_name!r}")

            if comp_name not in component_func_by_name:
                component_order.append(comp_name)
                component_func_by_name[comp_name] = int(prof_func_kind)
                component_param_rows_by_name[comp_name] = []

            component_param_rows_by_name[comp_name].append(src_row)
            is_constant = is_constant and bool(row_is_constant[src_row])

        for comp_name in component_order:
            profile_component_func_ids_list.append(component_func_by_name[comp_name])
            profile_component_param_rows_list.extend(
                component_param_rows_by_name[comp_name]
            )
            profile_component_param_indptr_list.append(
                len(profile_component_param_rows_list)
            )
        profile_sample_component_indptr_list.append(
            len(profile_component_func_ids_list)
        )

        group_idx = len(profile_sample_base_rows_list) - 1
        profile_sample_group_idx[group_name] = group_idx
        profile_sample_is_constant_list.append(is_constant)

    n_profile_samples = len(profile_sample_base_rows_list)
    profile_sample_base_rows = np.array(profile_sample_base_rows_list, dtype=np.intp)
    profile_sample_component_indptr = np.array(
        profile_sample_component_indptr_list, dtype=np.intp
    )
    profile_component_func_ids = np.array(
        profile_component_func_ids_list, dtype=np.intp
    )
    profile_component_param_indptr = np.array(
        profile_component_param_indptr_list, dtype=np.intp
    )
    profile_component_param_rows = np.array(
        profile_component_param_rows_list, dtype=np.intp
    )
    profile_sample_is_constant = np.array(
        profile_sample_is_constant_list, dtype=np.bool_
    )

    # ------------------------------------------------------------------ #
    # 4c. Compile per-sample profile expressions                           #
    # ------------------------------------------------------------------ #
    profile_expr_groups: dict[str, list[GraphNode]] = {}
    for nid in topo_order:
        node = id_to_node[nid]
        if _is_profile_expr_node(node):
            group_name = _profile_group_name(node.name, "profile_expr")
            profile_expr_groups.setdefault(group_name, []).append(node)

    profile_expr_programs_2d: list[ExprProgram] = []
    profile_expr_is_constant_list: list[bool] = []
    profile_expr_group_idx: dict[str, int] = {}
    for group_name, p_expr_nodes in profile_expr_groups.items():
        p_expr_nodes_sorted = sorted(
            p_expr_nodes,
            key=lambda node: _profile_group_index(node.name, "profile_expr"),
        )
        aux_indices = [
            _profile_group_index(node.name, "profile_expr")
            for node in p_expr_nodes_sorted
        ]
        if aux_indices != list(range(len(p_expr_nodes_sorted))):
            raise ValueError(
                f"Profile expression nodes for {group_name!r} do not cover "
                "a contiguous aux-axis range"
            )
        if len(p_expr_nodes_sorted) != n_aux:
            raise ValueError(
                f"Profile expression group {group_name!r} has "
                f"{len(p_expr_nodes_sorted)} samples but aux_axis length {n_aux}"
            )

        rep_node = p_expr_nodes_sorted[0]
        if rep_node.expr_string is None:
            raise ValueError(
                f"Profile expression {group_name!r} is missing expr_string"
            )
        expr_refs = set(_extract_expression_references(rep_node.expr_string))

        prof_ref_map: dict[str, int] = {}
        for edge in expr_ref_edges_by_target.get(rep_node.id, []):
            src_node = id_to_node[edge.source]
            if src_node.kind == NodeKind.PROFILE_SAMPLE:
                sample_name = _profile_group_name(src_node.name, "profile_sample")
                src_idx = n_params + profile_sample_group_idx[sample_name]
                match_name = sample_name
            else:
                src_idx = name_to_row[src_node.name]
                match_name = src_node.name
                # PARAM_PLUS_TRACE nodes carry a "_resolved" suffix;
                # the expression string uses the bare param name.
                if match_name not in expr_refs and src_node.name.endswith("_resolved"):
                    match_name = src_node.name.removesuffix("_resolved")

            if match_name in expr_refs:
                prof_ref_map[match_name] = src_idx

        symbolic = compile_expr_symbolic(rep_node.expr_string)
        prof_binding: dict[str, int] = dict(name_to_row)
        prof_binding.update(prof_ref_map)
        program = _bind_expr_to_rows(symbolic, prof_binding)
        profile_expr_programs_2d.append(program)

        is_constant = True
        for name in symbolic.referenced_names:
            bound_idx = int(prof_binding[name])
            if bound_idx < n_params:
                is_constant = is_constant and bool(row_is_constant[bound_idx])
            else:
                is_constant = is_constant and bool(
                    profile_sample_is_constant[bound_idx - n_params]
                )
        profile_expr_is_constant_list.append(is_constant)
        profile_expr_group_idx[group_name] = len(profile_expr_programs_2d) - 1

    n_profile_exprs = len(profile_expr_programs_2d)
    profile_expr_is_constant = np.array(profile_expr_is_constant_list, dtype=np.bool_)

    # ------------------------------------------------------------------ #
    # 5. Schedule component ops                                            #
    # ------------------------------------------------------------------ #
    # Identify peak_sum contributors: nodes with ADDEND edges into the
    # "peak_sum" SUM node (if it exists).
    peak_sum_sources: set[int] = set()
    peak_sum_nid = graph.node_by_name.get("peak_sum")
    if peak_sum_nid is not None:
        for e in addend_edges_by_target.get(peak_sum_nid, []):
            peak_sum_sources.add(e.source)

    # Collect per-sample component inputs for PROFILE_AVERAGE nodes.
    profile_avg_sample_inputs: dict[int, list[GraphNode]] = {}
    sample_component_ids: set[int] = set()
    for nid in topo_order:
        node = id_to_node[nid]
        if node.kind != NodeKind.PROFILE_AVERAGE:
            continue
        sample_nodes = [
            id_to_node[edge.source]
            for edge in addend_edges_by_target.get(node.id, [])
            if id_to_node[edge.source].kind
            in (NodeKind.COMPONENT_EVAL, NodeKind.SPECTRUM_FED_OP)
        ]
        profile_avg_sample_inputs[node.id] = sample_nodes
        sample_component_ids.update(sample.id for sample in sample_nodes)

    comp_op_kinds = {NodeKind.COMPONENT_EVAL, NodeKind.SPECTRUM_FED_OP}
    comp_nodes_topo = [
        id_to_node[nid]
        for nid in topo_order
        if (
            (id_to_node[nid].kind in comp_op_kinds and nid not in sample_component_ids)
            or id_to_node[nid].kind == NodeKind.PROFILE_AVERAGE
        )
    ]
    n_ops = len(comp_nodes_topo)

    op_schedule = np.arange(n_ops, dtype=np.intp)
    op_kinds_list: list[int] = []
    op_param_indptr_list: list[int] = [0]
    op_param_source_kinds_list: list[int] = []
    op_param_indices_list: list[int] = []
    op_needs_spectrum_list: list[bool] = []
    op_is_pre_spectrum_list: list[bool] = []
    op_is_profiled_list: list[bool] = []
    op_is_constant_list: list[bool] = []

    for comp_node in comp_nodes_topo:
        if comp_node.kind == NodeKind.PROFILE_AVERAGE:
            # Profiled component: gather from per-sample COMPONENT_EVAL inputs.
            sample_nodes = sorted(
                profile_avg_sample_inputs.get(comp_node.id, []),
                key=lambda node: _profile_component_sample_index(node.name),
            )
            if not sample_nodes:
                raise ValueError(
                    f"PROFILE_AVERAGE {comp_node.name!r} has no sample component inputs"
                )
            if len(sample_nodes) != n_aux:
                raise ValueError(
                    f"PROFILE_AVERAGE {comp_node.name!r} has "
                    f"{len(sample_nodes)} samples "
                    f"but aux_axis length {n_aux}"
                )

            rep_node = sample_nodes[0]
            assert rep_node.function_name is not None
            op = _FUNCTION_NAME_TO_OP.get(rep_node.function_name)
            if op is None:
                raise ValueError(
                    f"Unknown component function: {rep_node.function_name!r}"
                )
            op_kinds_list.append(int(op))
            op_is_profiled_list.append(True)

            rep_param_edges = sorted(
                param_edges_by_target.get(rep_node.id, []),
                key=lambda edge: edge.position or 0,
            )
            sample_param_edges = [
                sorted(
                    param_edges_by_target.get(sample_node.id, []),
                    key=lambda edge: edge.position or 0,
                )
                for sample_node in sample_nodes
            ]
            is_constant = True
            for pos, rep_edge in enumerate(rep_param_edges):
                src_node = id_to_node[rep_edge.source]
                if src_node.kind == NodeKind.PROFILE_SAMPLE:
                    group_name = _profile_group_name(src_node.name, "profile_sample")
                    source_kind = int(ParamSourceKind.PROFILE_SAMPLE)
                    source_idx = profile_sample_group_idx[group_name]
                    is_constant = is_constant and bool(
                        profile_sample_is_constant[source_idx]
                    )
                    for aux_i, edges in enumerate(sample_param_edges):
                        sample_src = id_to_node[edges[pos].source]
                        if sample_src.kind != NodeKind.PROFILE_SAMPLE:
                            raise ValueError(
                                "Mixed parameter source kinds "
                                f"in profiled op {comp_node.name!r}"
                            )
                        if (
                            _profile_group_name(sample_src.name, "profile_sample")
                            != group_name
                            or _profile_group_index(sample_src.name, "profile_sample")
                            != aux_i
                        ):
                            raise ValueError(
                                "Inconsistent PROFILE_SAMPLE wiring "
                                f"in {comp_node.name!r}"
                            )
                elif _is_profile_expr_node(src_node):
                    group_name = _profile_group_name(src_node.name, "profile_expr")
                    source_kind = int(ParamSourceKind.PROFILE_EXPR)
                    source_idx = profile_expr_group_idx[group_name]
                    is_constant = is_constant and bool(
                        profile_expr_is_constant[source_idx]
                    )
                    for aux_i, edges in enumerate(sample_param_edges):
                        sample_src = id_to_node[edges[pos].source]
                        if not _is_profile_expr_node(sample_src):
                            raise ValueError(
                                "Mixed expression source kinds "
                                f"in profiled op {comp_node.name!r}"
                            )
                        if (
                            _profile_group_name(sample_src.name, "profile_expr")
                            != group_name
                            or _profile_group_index(sample_src.name, "profile_expr")
                            != aux_i
                        ):
                            raise ValueError(
                                "Inconsistent profile-expression "
                                f"wiring in {comp_node.name!r}"
                            )
                else:
                    source_kind = int(ParamSourceKind.SCALAR)
                    source_idx = name_to_row[src_node.name]
                    is_constant = is_constant and bool(row_is_constant[source_idx])
                    for edges in sample_param_edges[1:]:
                        if id_to_node[edges[pos].source].id != src_node.id:
                            raise ValueError(
                                "Scalar parameter source changed "
                                f"across samples in {comp_node.name!r}"
                            )

                op_param_source_kinds_list.append(source_kind)
                op_param_indices_list.append(source_idx)

            op_param_indptr_list.append(len(op_param_indices_list))
        else:
            # Non-profiled component: standard param row wiring.
            assert comp_node.function_name is not None
            op = _FUNCTION_NAME_TO_OP.get(comp_node.function_name)
            if op is None:
                raise ValueError(
                    f"Unknown component function: {comp_node.function_name!r}"
                )
            op_kinds_list.append(int(op))
            op_is_profiled_list.append(False)

            param_edges = sorted(
                param_edges_by_target.get(comp_node.id, []),
                key=lambda edge: edge.position or 0,
            )
            is_constant = True
            for pe in param_edges:
                src_node = id_to_node[pe.source]
                src_row = name_to_row[src_node.name]
                op_param_source_kinds_list.append(int(ParamSourceKind.SCALAR))
                op_param_indices_list.append(src_row)
                is_constant = is_constant and bool(row_is_constant[src_row])

            op_param_indptr_list.append(len(op_param_indices_list))

        has_spec_input = comp_node.id in spectrum_input_targets
        op_needs_spectrum_list.append(has_spec_input)
        op_is_pre_spectrum_list.append(comp_node.id in peak_sum_sources)
        op_is_constant_list.append((not has_spec_input) and is_constant)

    op_kinds = np.array(op_kinds_list, dtype=np.intp)
    op_param_indptr = np.array(op_param_indptr_list, dtype=np.intp)
    op_param_source_kinds = np.array(op_param_source_kinds_list, dtype=np.int8)
    op_param_indices = np.array(op_param_indices_list, dtype=np.intp)
    op_needs_spectrum = np.array(op_needs_spectrum_list, dtype=np.bool_)
    op_is_pre_spectrum = np.array(op_is_pre_spectrum_list, dtype=np.bool_)
    op_is_profiled = np.array(op_is_profiled_list, dtype=np.bool_)
    op_is_constant = np.array(op_is_constant_list, dtype=np.bool_)

    # ------------------------------------------------------------------ #
    # 6. Initialize trace matrix                                           #
    # ------------------------------------------------------------------ #
    param_traces_init = np.zeros((n_params, n_time), dtype=np.float64)

    # Static and opt params: broadcast initial value
    for node in opt_nodes + static_nodes:
        row = name_to_row[node.name]
        param_traces_init[row, :] = node.value if node.value is not None else 0.0

    # PARAM_PLUS_TRACE: base + dynamics trace at initial values.
    # We need to evaluate dynamics functions at initial parameter values
    # to populate these rows.
    from trspecfit.functions import time as fcts_time

    _DYN_DISPATCH: dict[int, Callable[..., Any]] = {
        int(DynFuncKind.EXPFUN): fcts_time.expFun,
        int(DynFuncKind.SINFUN): fcts_time.sinFun,
        int(DynFuncKind.LINFUN): fcts_time.linFun,
        int(DynFuncKind.SINDIVX): fcts_time.sinDivX,
        int(DynFuncKind.ERFFUN): fcts_time.erfFun,
        int(DynFuncKind.SQRTFUN): fcts_time.sqrtFun,
    }

    # Dynamics groups and expressions are interleaved in topological order
    # so that expression-valued dynamics params (e.g. expFun_02_t0 =
    # "expFun_01_t0") are resolved before the group that consumes them.
    for step in range(len(resolution_kinds)):
        kind = int(resolution_kinds[step])
        idx = int(resolution_indices[step])
        if kind == 0:  # dynamics group
            target = int(dyn_group_target_row[idx])
            base = int(dyn_group_base_row[idx])
            param_traces_init[target, :] = param_traces_init[base, :]
            for s in range(int(dyn_group_indptr[idx]), int(dyn_group_indptr[idx + 1])):
                n_dp = int(dyn_sub_n_params[s])
                p_vals = [
                    float(param_traces_init[dyn_sub_param_rows[s, j], 0])
                    for j in range(n_dp)
                ]
                func = _DYN_DISPATCH[int(dyn_sub_func_id[s])]
                param_traces_init[target, :] += func(graph.time, *p_vals)
        else:  # expression
            target_row = int(expr_target_rows[idx])
            program = expr_programs[idx]
            result = _eval_expr_program_init(program, param_traces_init, n_time)
            param_traces_init[target_row, :] = result

    # ------------------------------------------------------------------ #
    # 6b. Precompute constant component contributions                      #
    # ------------------------------------------------------------------ #
    # Evaluate profile sample/expr values at initial traces for caching.
    from trspecfit.eval_2d import (
        _evaluate_profile_expr_values_2d,
        _evaluate_profile_sample_values_2d,
    )

    profile_sample_values_init = _evaluate_profile_sample_values_2d(
        plan_aux_axis,
        param_traces_init,
        profile_sample_base_rows,
        profile_sample_component_indptr,
        profile_component_func_ids,
        profile_component_param_indptr,
        profile_component_param_rows,
    )
    profile_expr_values_init = _evaluate_profile_expr_values_2d(
        param_traces_init,
        profile_sample_values_init,
        n_params,
        profile_expr_programs_2d,
    )

    energy = graph.energy[np.newaxis, :]
    cached_result = np.zeros((n_time, len(graph.energy)), dtype=np.float64)
    cached_peak_sum = np.zeros_like(cached_result)
    for op_idx in range(n_ops):
        if not op_is_constant[op_idx]:
            continue
        start = int(op_param_indptr[op_idx])
        end = int(op_param_indptr[op_idx + 1])

        if op_is_profiled[op_idx]:
            from trspecfit.eval_2d import _evaluate_profiled_op_2d

            component = _evaluate_profiled_op_2d(
                energy,
                int(op_kinds[op_idx]),
                op_param_source_kinds[start:end],
                op_param_indices[start:end],
                param_traces_init,
                profile_sample_values_init,
                profile_expr_values_init,
                cached_peak_sum,
                needs_spectrum=bool(op_needs_spectrum[op_idx]),
                n_aux=n_aux,
            )
        else:
            param_rows = op_param_indices[start:end]
            params = [
                param_traces_init[int(row), :][:, np.newaxis] for row in param_rows
            ]
            func, _needs = OP_DISPATCH[int(op_kinds[op_idx])]
            if op_needs_spectrum[op_idx]:
                component = func(energy, *params, cached_peak_sum)
            else:
                component = func(energy, *params)

        cached_result += component
        if op_is_pre_spectrum[op_idx]:
            cached_peak_sum += component

    # ------------------------------------------------------------------ #
    # 7. Pack into ScheduledPlan2D                                         #
    # ------------------------------------------------------------------ #
    return ScheduledPlan2D(
        energy=graph.energy,
        time=graph.time,
        n_params=n_params,
        n_time=n_time,
        param_traces_init=param_traces_init,
        opt_indices=opt_indices,
        opt_param_names=opt_param_names,
        n_dyn_groups=n_dyn_groups,
        dyn_group_target_row=dyn_group_target_row,
        dyn_group_base_row=dyn_group_base_row,
        dyn_group_indptr=dyn_group_indptr,
        dyn_sub_func_id=dyn_sub_func_id,
        dyn_sub_param_rows=dyn_sub_param_rows,
        dyn_sub_n_params=dyn_sub_n_params,
        n_expressions=n_expressions,
        expr_target_rows=expr_target_rows,
        expr_programs=expr_programs,
        resolution_kinds=resolution_kinds,
        resolution_indices=resolution_indices,
        n_aux=n_aux,
        aux_axis=plan_aux_axis,
        n_profile_samples=n_profile_samples,
        profile_sample_base_rows=profile_sample_base_rows,
        profile_sample_component_indptr=profile_sample_component_indptr,
        profile_component_func_ids=profile_component_func_ids,
        profile_component_param_indptr=profile_component_param_indptr,
        profile_component_param_rows=profile_component_param_rows,
        n_profile_exprs=n_profile_exprs,
        profile_expr_programs=profile_expr_programs_2d,
        n_ops=n_ops,
        op_schedule=op_schedule,
        op_kinds=op_kinds,
        op_param_indptr=op_param_indptr,
        op_param_source_kinds=op_param_source_kinds,
        op_param_indices=op_param_indices,
        op_needs_spectrum=op_needs_spectrum,
        op_is_pre_spectrum=op_is_pre_spectrum,
        op_is_profiled=op_is_profiled,
        op_is_constant=op_is_constant,
        cached_result=cached_result,
        cached_peak_sum=cached_peak_sum,
    )


#
def _eval_expr_program_init(
    program: ExprProgram,
    traces: np.ndarray,
    n_time: int,
) -> np.ndarray:
    """Evaluate an RPN ExprProgram against the trace matrix.

    Thin wrapper around :func:`trspecfit.eval_2d.eval_expr_program`
    kept for backward compatibility in ``schedule_2d`` init.
    """

    from trspecfit.eval_2d import eval_expr_program

    return eval_expr_program(program, traces)


# ---------------------------------------------------------------------------
# schedule_1d (Phase 6.1)
# ---------------------------------------------------------------------------


#
def _eval_expr_scalar(program: ExprProgram, values: np.ndarray) -> float:
    """Evaluate an RPN ExprProgram against a scalar parameter vector.

    Parameters
    ----------
    program
        Compiled RPN instruction array.
    values
        ``(n_params,)`` scalar parameter vector.

    Returns
    -------
    float
        Scalar result.
    """

    stack: list[float] = []
    instr = program.instructions
    n_instr = len(instr) // 2

    for i in range(n_instr):
        kind = ExprNodeKind(instr[2 * i])
        operand = instr[2 * i + 1]

        if kind == ExprNodeKind.CONST:
            stack.append(float(np.int64(operand).view(np.float64)))
        elif kind == ExprNodeKind.PARAM_REF:
            stack.append(float(values[int(operand)]))
        elif kind == ExprNodeKind.ADD:
            b, a = stack.pop(), stack.pop()
            stack.append(a + b)
        elif kind == ExprNodeKind.SUB:
            b, a = stack.pop(), stack.pop()
            stack.append(a - b)
        elif kind == ExprNodeKind.MUL:
            b, a = stack.pop(), stack.pop()
            stack.append(a * b)
        elif kind == ExprNodeKind.DIV:
            b, a = stack.pop(), stack.pop()
            stack.append(a / b)
        elif kind == ExprNodeKind.NEG:
            stack.append(-stack.pop())
        elif kind == ExprNodeKind.POW:
            b, a = stack.pop(), stack.pop()
            stack.append(a**b)

    assert len(stack) == 1
    return stack[0]


#
def _profile_group_name(node_name: str, label: str) -> str:
    """Return the shared profile group base name from a per-sample node name."""

    match = re.fullmatch(rf"(.+)_{label}_(\d+)", node_name)
    if match is None:
        raise ValueError(f"Malformed profile node name: {node_name!r}")
    return match.group(1)


#
def _profile_group_index(node_name: str, label: str) -> int:
    """Return the aux-axis sample index encoded in a per-sample node name."""

    match = re.fullmatch(rf"(.+)_{label}_(\d+)", node_name)
    if match is None:
        raise ValueError(f"Malformed profile node name: {node_name!r}")
    return int(match.group(2))


#
def _profile_component_sample_index(node_name: str) -> int:
    """Return the aux-axis sample index from ``<component>_sample_<i>``."""

    return _profile_group_index(node_name, "sample")


#
def _is_profile_expr_node(node: GraphNode) -> bool:
    """Return True for per-sample profile-expression nodes."""

    return (
        node.kind == NodeKind.EXPRESSION
        and re.fullmatch(r"(.+)_profile_expr_(\d+)", node.name) is not None
    )


#
def _parse_profile_component_param_name(
    target_param_name: str,
    source_param_name: str,
) -> tuple[str, str]:
    """Parse ``<target>_<profile_comp>_<par>`` into component + function name."""

    prefix = f"{target_param_name}_"
    if not source_param_name.startswith(prefix):
        raise ValueError(
            f"Profile parameter {source_param_name!r} does not match "
            f"target parameter {target_param_name!r}"
        )

    remainder = source_param_name[len(prefix) :]
    comp_name, sep, _par_name = remainder.rpartition("_")
    if not sep or not comp_name:
        raise ValueError(f"Malformed profile parameter name: {source_param_name!r}")

    func_name, sep2, comp_idx = comp_name.rpartition("_")
    if not sep2 or not comp_idx.isdigit():
        raise ValueError(f"Malformed profile component name: {comp_name!r}")

    return comp_name, func_name


#
def _eval_expr_vector(program: ExprProgram, traces: np.ndarray) -> np.ndarray:
    """Evaluate an RPN ExprProgram against an aux-resolved trace matrix."""

    from trspecfit.eval_2d import eval_expr_program

    return eval_expr_program(program, traces)


#
def _evaluate_profile_sample_values(
    aux_axis: np.ndarray,
    scalar_values: np.ndarray,
    profile_sample_base_indices: np.ndarray,
    profile_sample_component_indptr: np.ndarray,
    profile_component_func_ids: np.ndarray,
    profile_component_param_indptr: np.ndarray,
    profile_component_param_indices: np.ndarray,
) -> np.ndarray:
    """Evaluate lowered PROFILE_SAMPLE groups into ``(n_groups, n_aux)`` values."""

    n_groups = len(profile_sample_base_indices)
    n_aux = len(aux_axis)
    if n_groups == 0:
        return np.zeros((0, n_aux), dtype=np.float64)

    sample_values = np.empty((n_groups, n_aux), dtype=np.float64)
    for group_idx in range(n_groups):
        base_idx = int(profile_sample_base_indices[group_idx])
        values = np.full(n_aux, scalar_values[base_idx], dtype=np.float64)

        comp_start = int(profile_sample_component_indptr[group_idx])
        comp_end = int(profile_sample_component_indptr[group_idx + 1])
        for comp_idx in range(comp_start, comp_end):
            func = PROFILE_DISPATCH[int(profile_component_func_ids[comp_idx])]
            param_start = int(profile_component_param_indptr[comp_idx])
            param_end = int(profile_component_param_indptr[comp_idx + 1])
            params = [
                float(scalar_values[int(idx)])
                for idx in profile_component_param_indices[param_start:param_end]
            ]
            values += np.asarray(func(aux_axis, *params), dtype=np.float64)

        sample_values[group_idx, :] = values

    return sample_values


#
def _evaluate_profile_expr_values(
    scalar_values: np.ndarray,
    profile_sample_values: np.ndarray,
    n_params: int,
    profile_expr_programs: list[ExprProgram],
) -> np.ndarray:
    """Evaluate lowered per-sample profile expressions over the aux axis."""

    n_exprs = len(profile_expr_programs)
    if n_exprs == 0:
        n_aux = profile_sample_values.shape[1] if profile_sample_values.size else 0
        return np.zeros((0, n_aux), dtype=np.float64)

    n_aux = profile_sample_values.shape[1]
    traces = np.empty(
        (n_params + profile_sample_values.shape[0], n_aux), dtype=np.float64
    )
    traces[:n_params, :] = scalar_values[:, np.newaxis]
    if profile_sample_values.size:
        traces[n_params:, :] = profile_sample_values

    expr_values = np.empty((n_exprs, n_aux), dtype=np.float64)
    for expr_idx, program in enumerate(profile_expr_programs):
        expr_values[expr_idx, :] = _eval_expr_vector(program, traces)

    return expr_values


#
def _evaluate_scheduled_op_1d(
    energy: np.ndarray,
    kind: int,
    param_source_kinds: np.ndarray,
    param_indices: np.ndarray,
    scalar_values: np.ndarray,
    profile_sample_values: np.ndarray,
    profile_expr_values: np.ndarray,
    peak_sum: np.ndarray,
    *,
    needs_spectrum: bool,
    is_profiled: bool,
    n_aux: int,
) -> np.ndarray:
    """Evaluate one scheduled 1D op in either scalar or profiled mode."""

    func, _needs = OP_DISPATCH[kind]

    if not is_profiled:
        scalar_params = [float(scalar_values[int(row)]) for row in param_indices]
        if needs_spectrum:
            return np.asarray(func(energy, *scalar_params, peak_sum), dtype=np.float64)
        return np.asarray(func(energy, *scalar_params), dtype=np.float64)

    energy_2d = energy[np.newaxis, :]
    params_2d: list[np.ndarray] = []
    for source_kind, source_idx in zip(
        param_source_kinds,
        param_indices,
        strict=True,
    ):
        if int(source_kind) == int(ParamSourceKind.SCALAR):
            param = np.full(
                (n_aux, 1), scalar_values[int(source_idx)], dtype=np.float64
            )
        elif int(source_kind) == int(ParamSourceKind.PROFILE_SAMPLE):
            param = profile_sample_values[int(source_idx), :][:, np.newaxis]
        else:
            param = profile_expr_values[int(source_idx), :][:, np.newaxis]
        params_2d.append(param)

    if needs_spectrum:
        profiled = func(energy_2d, *params_2d, peak_sum[np.newaxis, :])
    else:
        profiled = func(energy_2d, *params_2d)
    result: np.ndarray = np.asarray(profiled, dtype=np.float64).mean(axis=0)
    return result


#
def schedule_1d(graph: GraphIR) -> ScheduledPlan1D:
    """Compile a GraphIR into a flat 1D execution schedule.

    Parameters
    ----------
    graph : GraphIR
        Must pass ``can_lower_1d(graph)``.

    Returns
    -------
    ScheduledPlan1D
        Packed-array execution schedule for ``evaluate_1d``.

    Raises
    ------
    ValueError
        If the graph cannot be lowered (domain, unsupported nodes, etc.).
    """

    if not can_lower_1d(graph):
        raise ValueError("Graph cannot be lowered to 1D backend")

    assert graph.energy is not None

    # ------------------------------------------------------------------ #
    # 1. Topological sort + helper lookups                                 #
    # ------------------------------------------------------------------ #
    topo_order = _topological_sort(graph)
    id_to_node: dict[int, GraphNode] = {n.id: n for n in graph.nodes}
    param_edges_by_target: dict[int, list[GraphEdge]] = {}
    expr_ref_edges_by_target: dict[int, list[GraphEdge]] = {}
    addend_edges_by_target: dict[int, list[GraphEdge]] = {}
    spectrum_input_targets: set[int] = set()
    for edge in graph.edges:
        if edge.kind == EdgeKind.PARAM_INPUT:
            param_edges_by_target.setdefault(edge.target, []).append(edge)
        elif edge.kind == EdgeKind.EXPR_REF:
            expr_ref_edges_by_target.setdefault(edge.target, []).append(edge)
        elif edge.kind == EdgeKind.ADDEND:
            addend_edges_by_target.setdefault(edge.target, []).append(edge)
        elif edge.kind == EdgeKind.SPECTRUM_INPUT:
            spectrum_input_targets.add(edge.target)

    # ------------------------------------------------------------------ #
    # 2. Assign scalar parameter indices                                   #
    # ------------------------------------------------------------------ #
    _ROW_KINDS = frozenset(
        {
            NodeKind.STATIC_PARAM,
            NodeKind.OPT_PARAM,
            NodeKind.EXPRESSION,
        }
    )

    opt_nodes: list[GraphNode] = []
    static_nodes: list[GraphNode] = []
    computed_nodes: list[GraphNode] = []

    for nid in topo_order:
        node = id_to_node[nid]
        if node.kind not in _ROW_KINDS or _is_profile_expr_node(node):
            continue
        if node.kind == NodeKind.OPT_PARAM and node.vary:
            opt_nodes.append(node)
        elif node.kind in (NodeKind.STATIC_PARAM, NodeKind.OPT_PARAM):
            static_nodes.append(node)
        else:
            computed_nodes.append(node)

    all_param_nodes = opt_nodes + static_nodes + computed_nodes
    n_params = len(all_param_nodes)
    name_to_idx = {node.name: idx for idx, node in enumerate(all_param_nodes)}
    idx_is_constant = np.zeros(n_params, dtype=np.bool_)
    for node in static_nodes:
        idx_is_constant[name_to_idx[node.name]] = True

    n_opt = len(opt_nodes)
    opt_indices = np.arange(n_opt, dtype=np.intp)
    opt_param_names = [n.name for n in opt_nodes]

    # ------------------------------------------------------------------ #
    # 3. Compile scalar expressions                                        #
    # ------------------------------------------------------------------ #
    expr_nodes_topo = [
        id_to_node[nid]
        for nid in topo_order
        if id_to_node[nid].kind == NodeKind.EXPRESSION
        and not _is_profile_expr_node(id_to_node[nid])
    ]
    expr_programs: list[ExprProgram] = []
    expr_target_indices_list: list[int] = []
    for expr_node in expr_nodes_topo:
        assert expr_node.expr_string is not None
        symbolic = compile_expr_symbolic(expr_node.expr_string)
        expr_refs = set(_extract_expression_references(expr_node.expr_string))

        ref_map: dict[str, int] = {}
        for edge in expr_ref_edges_by_target.get(expr_node.id, []):
            src_node = id_to_node[edge.source]
            src_idx = name_to_idx[src_node.name]
            if src_node.name in expr_refs:
                ref_map[src_node.name] = src_idx

        binding = dict(name_to_idx)
        binding.update(ref_map)
        program = _bind_expr_to_rows(symbolic, binding)
        expr_programs.append(program)

        target_idx = name_to_idx[expr_node.name]
        expr_target_indices_list.append(target_idx)
        idx_is_constant[target_idx] = all(
            idx_is_constant[int(binding[name])] for name in symbolic.referenced_names
        )

    n_expressions = len(expr_programs)
    expr_target_indices = np.array(expr_target_indices_list, dtype=np.intp)

    # ------------------------------------------------------------------ #
    # 4. Compile PROFILE_SAMPLE groups                                     #
    # ------------------------------------------------------------------ #
    profile_sample_groups: dict[str, list[GraphNode]] = {}
    for nid in topo_order:
        node = id_to_node[nid]
        if node.kind != NodeKind.PROFILE_SAMPLE:
            continue
        group_name = _profile_group_name(node.name, "profile_sample")
        profile_sample_groups.setdefault(group_name, []).append(node)

    plan_aux_axis = np.zeros(0, dtype=np.float64)
    n_aux = 0
    profile_sample_base_indices_list: list[int] = []
    profile_sample_component_indptr_list: list[int] = [0]
    profile_component_func_ids_list: list[int] = []
    profile_component_param_indptr_list: list[int] = [0]
    profile_component_param_indices_list: list[int] = []
    profile_sample_is_constant_list: list[bool] = []
    profile_sample_group_idx: dict[str, int] = {}

    for group_name, sample_nodes in profile_sample_groups.items():
        sample_nodes_sorted = sorted(
            sample_nodes,
            key=lambda node: _profile_group_index(node.name, "profile_sample"),
        )
        aux_indices = [
            _profile_group_index(node.name, "profile_sample")
            for node in sample_nodes_sorted
        ]
        if aux_indices != list(range(len(sample_nodes_sorted))):
            raise ValueError(
                f"PROFILE_SAMPLE nodes for {group_name!r} do not cover "
                "a contiguous aux-axis range"
            )

        aux_axis = sample_nodes_sorted[0].arrays.get("aux_axis")
        if aux_axis is None:
            raise ValueError(f"PROFILE_SAMPLE {group_name!r} is missing aux_axis")
        aux_axis = np.asarray(aux_axis, dtype=np.float64)
        if n_aux == 0:
            n_aux = len(aux_axis)
            plan_aux_axis = aux_axis.copy()
        elif len(aux_axis) != n_aux or not np.array_equal(aux_axis, plan_aux_axis):
            raise ValueError("All lowered profile groups must share one fixed aux_axis")
        if len(sample_nodes_sorted) != n_aux:
            raise ValueError(
                f"PROFILE_SAMPLE group {group_name!r} has {len(sample_nodes_sorted)} "
                f"samples but aux_axis length {n_aux}"
            )

        rep_node = sample_nodes_sorted[0]
        param_edges = sorted(
            param_edges_by_target.get(rep_node.id, []),
            key=lambda edge: edge.position or 0,
        )
        if not param_edges:
            raise ValueError(f"PROFILE_SAMPLE {group_name!r} has no PARAM_INPUT edges")

        base_node = id_to_node[param_edges[0].source]
        if base_node.name not in name_to_idx:
            raise ValueError(
                f"PROFILE_SAMPLE base source {base_node.name!r} is not scalar-lowerable"
            )
        base_idx = name_to_idx[base_node.name]
        is_constant = bool(idx_is_constant[base_idx])
        profile_sample_base_indices_list.append(base_idx)

        component_func_by_name: dict[str, int] = {}
        component_param_indices_by_name: dict[str, list[int]] = {}
        component_order: list[str] = []
        for edge in param_edges[1:]:
            src_node = id_to_node[edge.source]
            if src_node.name not in name_to_idx:
                raise ValueError(
                    f"Profile parameter source {src_node.name!r} "
                    "is not scalar-lowerable"
                )
            comp_name, func_name = _parse_profile_component_param_name(
                group_name,
                src_node.name,
            )
            prof_func_kind = _FUNCTION_NAME_TO_PROFILE_FUNC.get(func_name)
            if prof_func_kind is None:
                raise ValueError(f"Unknown profile function: {func_name!r}")

            if comp_name not in component_func_by_name:
                component_order.append(comp_name)
                component_func_by_name[comp_name] = int(prof_func_kind)
                component_param_indices_by_name[comp_name] = []

            src_idx = name_to_idx[src_node.name]
            component_param_indices_by_name[comp_name].append(src_idx)
            is_constant = is_constant and bool(idx_is_constant[src_idx])

        for comp_name in component_order:
            profile_component_func_ids_list.append(component_func_by_name[comp_name])
            profile_component_param_indices_list.extend(
                component_param_indices_by_name[comp_name]
            )
            profile_component_param_indptr_list.append(
                len(profile_component_param_indices_list)
            )
        profile_sample_component_indptr_list.append(
            len(profile_component_func_ids_list)
        )

        group_idx = len(profile_sample_base_indices_list) - 1
        profile_sample_group_idx[group_name] = group_idx
        profile_sample_is_constant_list.append(is_constant)

    n_profile_samples = len(profile_sample_base_indices_list)
    profile_sample_base_indices = np.array(
        profile_sample_base_indices_list, dtype=np.intp
    )
    profile_sample_component_indptr = np.array(
        profile_sample_component_indptr_list, dtype=np.intp
    )
    profile_component_func_ids = np.array(
        profile_component_func_ids_list, dtype=np.intp
    )
    profile_component_param_indptr = np.array(
        profile_component_param_indptr_list, dtype=np.intp
    )
    profile_component_param_indices = np.array(
        profile_component_param_indices_list, dtype=np.intp
    )
    profile_sample_is_constant = np.array(
        profile_sample_is_constant_list,
        dtype=np.bool_,
    )

    # ------------------------------------------------------------------ #
    # 5. Compile per-sample profile expressions                            #
    # ------------------------------------------------------------------ #
    profile_expr_groups: dict[str, list[GraphNode]] = {}
    for nid in topo_order:
        node = id_to_node[nid]
        if _is_profile_expr_node(node):
            group_name = _profile_group_name(node.name, "profile_expr")
            profile_expr_groups.setdefault(group_name, []).append(node)

    profile_expr_programs: list[ExprProgram] = []
    profile_expr_is_constant_list: list[bool] = []
    profile_expr_group_idx: dict[str, int] = {}
    for group_name, expr_nodes in profile_expr_groups.items():
        expr_nodes_sorted = sorted(
            expr_nodes,
            key=lambda node: _profile_group_index(node.name, "profile_expr"),
        )
        aux_indices = [
            _profile_group_index(node.name, "profile_expr")
            for node in expr_nodes_sorted
        ]
        if aux_indices != list(range(len(expr_nodes_sorted))):
            raise ValueError(
                f"Profile expression nodes for {group_name!r} do not cover "
                "a contiguous aux-axis range"
            )
        if len(expr_nodes_sorted) != n_aux:
            raise ValueError(
                f"Profile expression group {group_name!r} has {len(expr_nodes_sorted)} "
                f"samples but aux_axis length {n_aux}"
            )

        rep_node = expr_nodes_sorted[0]
        if rep_node.expr_string is None:
            raise ValueError(
                f"Profile expression {group_name!r} is missing expr_string"
            )
        expr_refs = set(_extract_expression_references(rep_node.expr_string))

        prof_ref_map: dict[str, int] = {}
        for edge in expr_ref_edges_by_target.get(rep_node.id, []):
            src_node = id_to_node[edge.source]
            if src_node.kind == NodeKind.PROFILE_SAMPLE:
                sample_name = _profile_group_name(src_node.name, "profile_sample")
                src_idx = n_params + profile_sample_group_idx[sample_name]
                match_name = sample_name
            else:
                src_idx = name_to_idx[src_node.name]
                match_name = src_node.name

            if match_name in expr_refs:
                prof_ref_map[match_name] = src_idx

        symbolic = compile_expr_symbolic(rep_node.expr_string)
        binding = dict(name_to_idx)
        binding.update(prof_ref_map)
        program = _bind_expr_to_rows(symbolic, binding)
        profile_expr_programs.append(program)

        is_constant = True
        for name in symbolic.referenced_names:
            bound_idx = int(binding[name])
            if bound_idx < n_params:
                is_constant = is_constant and bool(idx_is_constant[bound_idx])
            else:
                is_constant = is_constant and bool(
                    profile_sample_is_constant[bound_idx - n_params]
                )
        profile_expr_is_constant_list.append(is_constant)
        profile_expr_group_idx[group_name] = len(profile_expr_programs) - 1

    n_profile_exprs = len(profile_expr_programs)
    profile_expr_is_constant = np.array(profile_expr_is_constant_list, dtype=np.bool_)

    # ------------------------------------------------------------------ #
    # 6. Schedule component ops                                            #
    # ------------------------------------------------------------------ #
    peak_sum_sources: set[int] = set()
    peak_sum_nid = graph.node_by_name.get("peak_sum")
    if peak_sum_nid is not None:
        for edge in addend_edges_by_target.get(peak_sum_nid, []):
            peak_sum_sources.add(edge.source)

    profile_avg_sample_inputs: dict[int, list[GraphNode]] = {}
    sample_component_ids: set[int] = set()
    for nid in topo_order:
        node = id_to_node[nid]
        if node.kind != NodeKind.PROFILE_AVERAGE:
            continue
        sample_nodes = [
            id_to_node[edge.source]
            for edge in addend_edges_by_target.get(node.id, [])
            if id_to_node[edge.source].kind
            in (NodeKind.COMPONENT_EVAL, NodeKind.SPECTRUM_FED_OP)
        ]
        profile_avg_sample_inputs[node.id] = sample_nodes
        sample_component_ids.update(sample.id for sample in sample_nodes)

    comp_nodes_topo = [
        id_to_node[nid]
        for nid in topo_order
        if (
            (
                id_to_node[nid].kind
                in (NodeKind.COMPONENT_EVAL, NodeKind.SPECTRUM_FED_OP)
                and nid not in sample_component_ids
            )
            or id_to_node[nid].kind == NodeKind.PROFILE_AVERAGE
        )
    ]

    op_kinds_list: list[int] = []
    op_param_indptr_list: list[int] = [0]
    op_param_source_kinds_list: list[int] = []
    op_param_indices_list: list[int] = []
    op_needs_spectrum_list: list[bool] = []
    op_is_pre_spectrum_list: list[bool] = []
    op_is_profiled_list: list[bool] = []
    op_is_constant_list: list[bool] = []

    for comp_node in comp_nodes_topo:
        if comp_node.kind == NodeKind.PROFILE_AVERAGE:
            sample_nodes = sorted(
                profile_avg_sample_inputs.get(comp_node.id, []),
                key=lambda node: _profile_component_sample_index(node.name),
            )
            if not sample_nodes:
                raise ValueError(
                    f"PROFILE_AVERAGE {comp_node.name!r} has no sample component inputs"
                )
            if len(sample_nodes) != n_aux:
                raise ValueError(
                    f"PROFILE_AVERAGE {comp_node.name!r} has "
                    f"{len(sample_nodes)} samples "
                    f"but aux_axis length {n_aux}"
                )

            rep_node = sample_nodes[0]
            assert rep_node.function_name is not None
            op = _FUNCTION_NAME_TO_OP.get(rep_node.function_name)
            if op is None:
                raise ValueError(
                    f"Unknown component function: {rep_node.function_name!r}"
                )
            op_kinds_list.append(int(op))
            op_is_profiled_list.append(True)

            rep_param_edges = sorted(
                param_edges_by_target.get(rep_node.id, []),
                key=lambda edge: edge.position or 0,
            )
            sample_param_edges = [
                sorted(
                    param_edges_by_target.get(sample_node.id, []),
                    key=lambda edge: edge.position or 0,
                )
                for sample_node in sample_nodes
            ]
            is_constant = True
            for pos, rep_edge in enumerate(rep_param_edges):
                src_node = id_to_node[rep_edge.source]
                if src_node.kind == NodeKind.PROFILE_SAMPLE:
                    group_name = _profile_group_name(src_node.name, "profile_sample")
                    source_kind = int(ParamSourceKind.PROFILE_SAMPLE)
                    source_idx = profile_sample_group_idx[group_name]
                    is_constant = is_constant and bool(
                        profile_sample_is_constant[source_idx]
                    )
                    for aux_i, edges in enumerate(sample_param_edges):
                        sample_src = id_to_node[edges[pos].source]
                        if sample_src.kind != NodeKind.PROFILE_SAMPLE:
                            raise ValueError(
                                "Mixed parameter source kinds "
                                f"in profiled op {comp_node.name!r}"
                            )
                        if (
                            _profile_group_name(sample_src.name, "profile_sample")
                            != group_name
                            or _profile_group_index(sample_src.name, "profile_sample")
                            != aux_i
                        ):
                            raise ValueError(
                                "Inconsistent PROFILE_SAMPLE wiring "
                                f"in {comp_node.name!r}"
                            )
                elif _is_profile_expr_node(src_node):
                    group_name = _profile_group_name(src_node.name, "profile_expr")
                    source_kind = int(ParamSourceKind.PROFILE_EXPR)
                    source_idx = profile_expr_group_idx[group_name]
                    is_constant = is_constant and bool(
                        profile_expr_is_constant[source_idx]
                    )
                    for aux_i, edges in enumerate(sample_param_edges):
                        sample_src = id_to_node[edges[pos].source]
                        if not _is_profile_expr_node(sample_src):
                            raise ValueError(
                                "Mixed expression source kinds "
                                f"in profiled op {comp_node.name!r}"
                            )
                        if (
                            _profile_group_name(sample_src.name, "profile_expr")
                            != group_name
                            or _profile_group_index(sample_src.name, "profile_expr")
                            != aux_i
                        ):
                            raise ValueError(
                                "Inconsistent profile-expression "
                                f"wiring in {comp_node.name!r}"
                            )
                else:
                    if src_node.name not in name_to_idx:
                        raise ValueError(
                            f"Non-scalar parameter source {src_node.name!r} in 1D op"
                        )
                    source_kind = int(ParamSourceKind.SCALAR)
                    source_idx = name_to_idx[src_node.name]
                    is_constant = is_constant and bool(idx_is_constant[source_idx])
                    for edges in sample_param_edges[1:]:
                        if id_to_node[edges[pos].source].id != src_node.id:
                            raise ValueError(
                                "Scalar parameter source changed "
                                f"across samples in {comp_node.name!r}"
                            )

                op_param_source_kinds_list.append(source_kind)
                op_param_indices_list.append(source_idx)

            op_param_indptr_list.append(len(op_param_indices_list))
        else:
            assert comp_node.function_name is not None
            op = _FUNCTION_NAME_TO_OP.get(comp_node.function_name)
            if op is None:
                raise ValueError(
                    f"Unknown component function: {comp_node.function_name!r}"
                )
            op_kinds_list.append(int(op))
            op_is_profiled_list.append(False)

            param_edges = sorted(
                param_edges_by_target.get(comp_node.id, []),
                key=lambda edge: edge.position or 0,
            )
            is_constant = True
            for edge in param_edges:
                src_node = id_to_node[edge.source]
                if src_node.name not in name_to_idx:
                    raise ValueError(
                        f"Non-scalar parameter source {src_node.name!r} in 1D op"
                    )
                src_idx = name_to_idx[src_node.name]
                op_param_source_kinds_list.append(int(ParamSourceKind.SCALAR))
                op_param_indices_list.append(src_idx)
                is_constant = is_constant and bool(idx_is_constant[src_idx])

            op_param_indptr_list.append(len(op_param_indices_list))

        has_spec_input = comp_node.id in spectrum_input_targets
        op_needs_spectrum_list.append(has_spec_input)
        op_is_pre_spectrum_list.append(comp_node.id in peak_sum_sources)
        op_is_constant_list.append((not has_spec_input) and is_constant)

    n_ops = len(comp_nodes_topo)
    op_kinds = np.array(op_kinds_list, dtype=np.intp)
    op_param_indptr = np.array(op_param_indptr_list, dtype=np.intp)
    op_param_source_kinds = np.array(op_param_source_kinds_list, dtype=np.int8)
    op_param_indices = np.array(op_param_indices_list, dtype=np.intp)
    op_needs_spectrum = np.array(op_needs_spectrum_list, dtype=np.bool_)
    op_is_pre_spectrum = np.array(op_is_pre_spectrum_list, dtype=np.bool_)
    op_is_profiled = np.array(op_is_profiled_list, dtype=np.bool_)
    op_is_constant = np.array(op_is_constant_list, dtype=np.bool_)

    # ------------------------------------------------------------------ #
    # 7. Initialize scalar + profile values                                #
    # ------------------------------------------------------------------ #
    param_values_init = np.zeros(n_params, dtype=np.float64)
    for node in opt_nodes + static_nodes:
        param_values_init[name_to_idx[node.name]] = (
            node.value if node.value is not None else 0.0
        )

    for i in range(n_expressions):
        target_idx = int(expr_target_indices[i])
        param_values_init[target_idx] = _eval_expr_scalar(
            expr_programs[i], param_values_init
        )

    profile_sample_values_init = _evaluate_profile_sample_values(
        plan_aux_axis,
        param_values_init,
        profile_sample_base_indices,
        profile_sample_component_indptr,
        profile_component_func_ids,
        profile_component_param_indptr,
        profile_component_param_indices,
    )
    profile_expr_values_init = _evaluate_profile_expr_values(
        param_values_init,
        profile_sample_values_init,
        n_params,
        profile_expr_programs,
    )

    # ------------------------------------------------------------------ #
    # 8. Precompute constant component contributions                       #
    # ------------------------------------------------------------------ #
    energy = graph.energy
    cached_result = np.zeros(len(energy), dtype=np.float64)
    cached_peak_sum = np.zeros_like(cached_result)
    for op_idx in range(n_ops):
        if not op_is_constant[op_idx]:
            continue

        start = int(op_param_indptr[op_idx])
        end = int(op_param_indptr[op_idx + 1])
        component = _evaluate_scheduled_op_1d(
            energy,
            int(op_kinds[op_idx]),
            op_param_source_kinds[start:end],
            op_param_indices[start:end],
            param_values_init,
            profile_sample_values_init,
            profile_expr_values_init,
            cached_peak_sum,
            needs_spectrum=bool(op_needs_spectrum[op_idx]),
            is_profiled=bool(op_is_profiled[op_idx]),
            n_aux=n_aux,
        )

        cached_result += component
        if op_is_pre_spectrum[op_idx]:
            cached_peak_sum += component

    # ------------------------------------------------------------------ #
    # 9. Pack into ScheduledPlan1D                                         #
    # ------------------------------------------------------------------ #
    return ScheduledPlan1D(
        energy=energy,
        n_params=n_params,
        param_values_init=param_values_init,
        opt_indices=opt_indices,
        opt_param_names=opt_param_names,
        n_expressions=n_expressions,
        expr_target_indices=expr_target_indices,
        expr_programs=expr_programs,
        n_aux=n_aux,
        aux_axis=plan_aux_axis,
        n_profile_samples=n_profile_samples,
        profile_sample_base_indices=profile_sample_base_indices,
        profile_sample_component_indptr=profile_sample_component_indptr,
        profile_component_func_ids=profile_component_func_ids,
        profile_component_param_indptr=profile_component_param_indptr,
        profile_component_param_indices=profile_component_param_indices,
        n_profile_exprs=n_profile_exprs,
        profile_expr_programs=profile_expr_programs,
        n_ops=n_ops,
        op_kinds=op_kinds,
        op_param_indptr=op_param_indptr,
        op_param_source_kinds=op_param_source_kinds,
        op_param_indices=op_param_indices,
        op_needs_spectrum=op_needs_spectrum,
        op_is_pre_spectrum=op_is_pre_spectrum,
        op_is_profiled=op_is_profiled,
        op_is_constant=op_is_constant,
        cached_result=cached_result,
        cached_peak_sum=cached_peak_sum,
    )
