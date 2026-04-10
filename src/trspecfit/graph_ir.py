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
    GLS = 3
    GLP = 4
    DS = 5
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


_FUNCTION_NAME_TO_DYN_FUNC: dict[str, DynFuncKind] = {
    "expFun": DynFuncKind.EXPFUN,
    "sinFun": DynFuncKind.SINFUN,
    "linFun": DynFuncKind.LINFUN,
    "sinDivX": DynFuncKind.SINDIVX,
    "erfFun": DynFuncKind.ERFFUN,
    "sqrtFun": DynFuncKind.SQRTFUN,
}

_FUNCTION_NAME_TO_OP: dict[str, OpKind] = {
    "Gauss": OpKind.GAUSS,
    "GaussAsym": OpKind.GAUSS_ASYM,
    "Lorentz": OpKind.LORENTZ,
    "GLS": OpKind.GLS,
    "GLP": OpKind.GLP,
    "DS": OpKind.DS,
    "Offset": OpKind.OFFSET,
    "LinBack": OpKind.LINBACK,
    "Shirley": OpKind.SHIRLEY,
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

    # --- Dynamics subgraphs ---
    n_dynamics: int
    dynamics_func_id: np.ndarray  # (n_dynamics,) int
    dynamics_param_rows: np.ndarray  # (n_dynamics, max_dyn_params) int, -1 padded
    dynamics_n_params: np.ndarray  # (n_dynamics,) int
    dynamics_target_row: np.ndarray  # (n_dynamics,) int
    dynamics_base_row: np.ndarray  # (n_dynamics,) int

    # --- Expression evaluation ---
    n_expressions: int
    expr_target_rows: np.ndarray  # (n_expressions,) int
    expr_programs: list[ExprProgram]

    # --- Scheduled component ops ---
    n_ops: int
    op_schedule: np.ndarray  # (n_ops,) int
    op_kinds: np.ndarray  # (n_ops,) OpKind int codes
    op_param_indptr: np.ndarray  # (n_ops + 1,) int -- CSR row pointers
    op_param_indices: np.ndarray  # (total_op_params,) int -- row indices
    op_needs_spectrum: np.ndarray  # (n_ops,) bool
    op_is_pre_spectrum: np.ndarray  # (n_ops,) bool


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

    This matches the spec example (execution_plan.md lines 298-305):
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
        "GLS",
        "GLP",
        "DS",
        "Offset",
        "LinBack",
        "Shirley",
    }
)

_NON_LOWERABLE_NODE_KINDS: frozenset[NodeKind] = frozenset(
    {
        NodeKind.CONVOLUTION,
        NodeKind.PROFILE_SAMPLE,
        NodeKind.PROFILE_AVERAGE,
        NodeKind.SUBCYCLE_MASK,
        NodeKind.SUBCYCLE_REMAP,
    }
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
        if node.kind in _NON_LOWERABLE_NODE_KINDS:
            return False

        # Check component functions are supported
        if node.kind in (NodeKind.COMPONENT_EVAL, NodeKind.SPECTRUM_FED_OP):
            if node.function_name not in _LOWERABLE_2D_FUNCTIONS:
                return False

        # Check dynamics functions are supported
        if node.kind == NodeKind.DYNAMICS_TRACE:
            if node.function_name not in _FUNCTION_NAME_TO_DYN_FUNC:
                return False

        # Check expressions are arithmetic-only (defer full AST check
        # to the expression compiler; here just reject obvious non-starters)
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
        if node.kind not in _ROW_KINDS:
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

    # opt_indices and opt_param_names
    n_opt = len(opt_nodes)
    opt_indices = np.arange(n_opt, dtype=np.intp)
    opt_param_names = [n.name for n in opt_nodes]

    # ------------------------------------------------------------------ #
    # 3. Compile dynamics subgraphs                                        #
    # ------------------------------------------------------------------ #
    dyn_trace_nodes = [
        id_to_node[nid]
        for nid in topo_order
        if id_to_node[nid].kind == NodeKind.DYNAMICS_TRACE
    ]
    n_dynamics = len(dyn_trace_nodes)

    # Pre-compute max_dyn_params for rectangular array
    dyn_param_counts: list[int] = []
    dyn_param_row_lists: list[list[int]] = []
    dyn_func_ids: list[int] = []
    dyn_target_rows: list[int] = []
    dyn_base_rows: list[int] = []

    for dyn_node in dyn_trace_nodes:
        # Function ID
        assert dyn_node.function_name is not None
        func_kind = _FUNCTION_NAME_TO_DYN_FUNC.get(dyn_node.function_name)
        if func_kind is None:
            raise ValueError(f"Unknown dynamics function: {dyn_node.function_name!r}")
        dyn_func_ids.append(int(func_kind))

        # Param rows (from PARAM_INPUT edges, ordered by position)
        param_edges = sorted(
            (
                e
                for e in graph.edges
                if e.target == dyn_node.id and e.kind == EdgeKind.PARAM_INPUT
            ),
            key=lambda e: e.position or 0,
        )
        param_rows = [name_to_row[id_to_node[e.source].name] for e in param_edges]
        dyn_param_row_lists.append(param_rows)
        dyn_param_counts.append(len(param_rows))

        # Target row: the PARAM_PLUS_TRACE node that this trace feeds into
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
        dyn_target_rows.append(name_to_row[ppt_node.name])

        # Base row: the BASE_INPUT to the PARAM_PLUS_TRACE
        base_edges = [
            e
            for e in graph.edges
            if e.target == ppt_node.id and e.kind == EdgeKind.BASE_INPUT
        ]
        assert len(base_edges) == 1
        base_node = id_to_node[base_edges[0].source]
        dyn_base_rows.append(name_to_row[base_node.name])

    # Pack into arrays
    max_dyn_params = max(dyn_param_counts) if dyn_param_counts else 0
    dynamics_param_rows = np.full((n_dynamics, max_dyn_params), -1, dtype=np.intp)
    for i, rows in enumerate(dyn_param_row_lists):
        dynamics_param_rows[i, : len(rows)] = rows

    dynamics_func_id = np.array(dyn_func_ids, dtype=np.intp)
    dynamics_n_params = np.array(dyn_param_counts, dtype=np.intp)
    dynamics_target_row = np.array(dyn_target_rows, dtype=np.intp)
    dynamics_base_row = np.array(dyn_base_rows, dtype=np.intp)

    # ------------------------------------------------------------------ #
    # 4. Compile expressions (topological order)                           #
    # ------------------------------------------------------------------ #
    expr_nodes_topo = [
        id_to_node[nid]
        for nid in topo_order
        if id_to_node[nid].kind == NodeKind.EXPRESSION
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
        expr_target_rows_list.append(name_to_row[expr_node.name])

    expr_target_rows = np.array(expr_target_rows_list, dtype=np.intp)

    # ------------------------------------------------------------------ #
    # 5. Schedule component ops                                            #
    # ------------------------------------------------------------------ #
    # Identify peak_sum contributors: nodes with ADDEND edges into the
    # "peak_sum" SUM node (if it exists).
    peak_sum_sources: set[int] = set()
    peak_sum_nid = graph.node_by_name.get("peak_sum")
    if peak_sum_nid is not None:
        for e in graph.edges:
            if e.target == peak_sum_nid and e.kind == EdgeKind.ADDEND:
                peak_sum_sources.add(e.source)

    comp_op_kinds = {NodeKind.COMPONENT_EVAL, NodeKind.SPECTRUM_FED_OP}
    comp_nodes_topo = [
        id_to_node[nid] for nid in topo_order if id_to_node[nid].kind in comp_op_kinds
    ]
    n_ops = len(comp_nodes_topo)

    op_schedule = np.arange(n_ops, dtype=np.intp)
    op_kinds_list: list[int] = []
    op_param_indptr_list: list[int] = [0]
    op_param_indices_list: list[int] = []
    op_needs_spectrum_list: list[bool] = []
    op_is_pre_spectrum_list: list[bool] = []

    for comp_node in comp_nodes_topo:
        # OpKind
        assert comp_node.function_name is not None
        op = _FUNCTION_NAME_TO_OP.get(comp_node.function_name)
        if op is None:
            raise ValueError(f"Unknown component function: {comp_node.function_name!r}")
        op_kinds_list.append(int(op))

        # Param rows (from PARAM_INPUT edges, ordered by position)
        param_edges = sorted(
            (
                e
                for e in graph.edges
                if e.target == comp_node.id and e.kind == EdgeKind.PARAM_INPUT
            ),
            key=lambda e: e.position or 0,
        )
        for pe in param_edges:
            src_node = id_to_node[pe.source]
            op_param_indices_list.append(name_to_row[src_node.name])
        op_param_indptr_list.append(len(op_param_indices_list))

        # Needs spectrum?
        has_spec_input = any(
            e.target == comp_node.id and e.kind == EdgeKind.SPECTRUM_INPUT
            for e in graph.edges
        )
        op_needs_spectrum_list.append(has_spec_input)

        # Contributes to peak_sum?
        op_is_pre_spectrum_list.append(comp_node.id in peak_sum_sources)

    op_kinds = np.array(op_kinds_list, dtype=np.intp)
    op_param_indptr = np.array(op_param_indptr_list, dtype=np.intp)
    op_param_indices = np.array(op_param_indices_list, dtype=np.intp)
    op_needs_spectrum = np.array(op_needs_spectrum_list, dtype=np.bool_)
    op_is_pre_spectrum = np.array(op_is_pre_spectrum_list, dtype=np.bool_)

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

    for i in range(n_dynamics):
        n_dp = int(dynamics_n_params[i])
        p_vals = [
            float(param_traces_init[dynamics_param_rows[i, j], 0]) for j in range(n_dp)
        ]
        func = _DYN_DISPATCH[int(dynamics_func_id[i])]
        trace = func(graph.time, *p_vals)
        base_val = param_traces_init[dynamics_base_row[i], 0]
        param_traces_init[dynamics_target_row[i], :] = base_val + trace

    # Expressions: evaluate in topological order using the trace matrix.
    # Simple scalar-broadcast evaluation (take column 0 for constants).
    for idx in range(n_expressions):
        target_row = int(expr_target_rows[idx])
        program = expr_programs[idx]
        result = _eval_expr_program_init(program, param_traces_init, n_time)
        param_traces_init[target_row, :] = result

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
        n_dynamics=n_dynamics,
        dynamics_func_id=dynamics_func_id,
        dynamics_param_rows=dynamics_param_rows,
        dynamics_n_params=dynamics_n_params,
        dynamics_target_row=dynamics_target_row,
        dynamics_base_row=dynamics_base_row,
        n_expressions=n_expressions,
        expr_target_rows=expr_target_rows,
        expr_programs=expr_programs,
        n_ops=n_ops,
        op_schedule=op_schedule,
        op_kinds=op_kinds,
        op_param_indptr=op_param_indptr,
        op_param_indices=op_param_indices,
        op_needs_spectrum=op_needs_spectrum,
        op_is_pre_spectrum=op_is_pre_spectrum,
    )


#
def _eval_expr_program_init(
    program: ExprProgram,
    traces: np.ndarray,
    n_time: int,
) -> np.ndarray:
    """Evaluate an RPN ExprProgram against the trace matrix.

    Used during plan initialization to populate expression rows.
    Returns an ``(n_time,)`` array.
    """

    stack: list[np.ndarray] = []
    instr = program.instructions
    n_instr = len(instr) // 2

    for i in range(n_instr):
        kind = ExprNodeKind(instr[2 * i])
        operand = instr[2 * i + 1]

        if kind == ExprNodeKind.CONST:
            val = np.int64(operand).view(np.float64)
            stack.append(np.full(n_time, val, dtype=np.float64))

        elif kind == ExprNodeKind.PARAM_REF:
            stack.append(traces[int(operand), :].copy())

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
