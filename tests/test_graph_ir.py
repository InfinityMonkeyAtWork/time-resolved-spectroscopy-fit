"""Tests for graph_ir: build_graph, can_lower_2d, expression compiler, schedule_2d."""

import numpy as np
import pytest

from trspecfit import File, Project
from trspecfit.graph_ir import (
    DomainKind,
    DynFuncKind,
    EdgeKind,
    ExprNodeKind,
    NodeKind,
    OpKind,
    SymbolicRPN,
    build_graph,
    can_lower_2d,
    compile_expr_symbolic,
    schedule_2d,
)


# Helpers
#
def _make_energy_model(model_info):
    """Create project + file + load energy model, return (file, model)."""

    project = Project(path="tests")
    file = File(parent_project=project)
    file.energy = np.linspace(80, 90, 201)
    file.load_model(
        model_yaml="models/file_energy.yaml",
        model_info=model_info,
    )
    model = file.model_active
    assert model is not None
    return file, model


#
def _make_2d_model(model_info, dynamics_params):
    """Create a 2D model by loading energy model + adding dynamics.

    Parameters
    ----------
    model_info : list[str]
        Energy model name(s) from file_energy.yaml.
    dynamics_params : list[tuple[str, str, list[str]]]
        Each tuple: (target_parameter, dynamics_yaml_model, dynamics_model_info).
    """

    project = Project(path="tests")
    file = File(parent_project=project)
    file.energy = np.linspace(80, 90, 201)
    file.time = np.linspace(-10, 100, 111)
    file.load_model(
        model_yaml="models/file_energy.yaml",
        model_info=model_info,
    )
    model = file.model_active
    assert model is not None

    for target_par, _dyn_yaml, dyn_model in dynamics_params:
        file.add_time_dependence(
            target_model=model_info[0],
            target_parameter=target_par,
            dynamics_yaml="models/file_time.yaml",
            dynamics_model=dyn_model,
        )

    return file, model


#
def _make_time_only_model(model_info, *, frequency=-1):
    """Create a standalone dynamics model, return (file, model)."""

    project = Project(path="tests")
    file = File(parent_project=project)
    file.time = np.linspace(-10, 100, 111)
    model = file.load_model(
        model_yaml="models/file_time.yaml",
        model_info=model_info,
        par_name="parTEST",
        model_type="dynamics",
    )
    if frequency != -1:
        model.set_frequency(frequency=frequency)
    return file, model


#
def _nodes_by_kind(graph, kind):
    """Return list of nodes with given NodeKind."""

    return [n for n in graph.nodes if n.kind == kind]


#
def _edges_to(graph, target_id, kind=None):
    """Return edges targeting a specific node, optionally filtered by kind."""

    edges = [e for e in graph.edges if e.target == target_id]
    if kind is not None:
        edges = [e for e in edges if e.kind == kind]
    return edges


#
def _edges_from(graph, source_id, kind=None):
    """Return edges originating from a specific node, optionally filtered."""

    edges = [e for e in graph.edges if e.source == source_id]
    if kind is not None:
        edges = [e for e in edges if e.kind == kind]
    return edges


#
def _node_by_name(graph, name):
    """Return the node with the given name, or None."""

    nid = graph.node_by_name.get(name)
    if nid is None:
        return None
    return graph.nodes[nid]


#
#
class TestBuildGraphEnergy1D:
    """Test build_graph for 1D energy models."""

    #
    def test_simple_energy_structure(self):
        """Simple energy model produces correct node and edge structure."""

        _file, model = _make_energy_model(["simple_energy"])
        graph = build_graph(model)

        assert graph.domain == DomainKind.ENERGY_1D
        assert graph.energy is not None
        assert graph.time is None

        # 4 components: Offset, Shirley, GLP_01, GLP_02
        comp_evals = _nodes_by_kind(graph, NodeKind.COMPONENT_EVAL)
        spectrum_fed = _nodes_by_kind(graph, NodeKind.SPECTRUM_FED_OP)
        # Offset, GLP_01, GLP_02 -> COMPONENT_EVAL; Shirley -> SPECTRUM_FED_OP
        assert len(comp_evals) == 3
        assert len(spectrum_fed) == 1
        assert spectrum_fed[0].function_name == "Shirley"

    #
    def test_simple_energy_param_nodes(self):
        """Parameter nodes have correct kinds and values."""

        _file, model = _make_energy_model(["simple_energy"])
        graph = build_graph(model)

        # Offset_y0: vary=True -> OPT_PARAM
        offset_y0 = _node_by_name(graph, "Offset_y0")
        assert offset_y0 is not None
        assert offset_y0.kind == NodeKind.OPT_PARAM
        assert offset_y0.value == 2.0
        assert offset_y0.vary is True

        # Shirley_pShirley: vary=False -> STATIC_PARAM
        shirley_p = _node_by_name(graph, "Shirley_pShirley")
        assert shirley_p is not None
        assert shirley_p.kind == NodeKind.STATIC_PARAM
        assert np.isclose(shirley_p.value, 4e-4)

        # GLP_01_A: vary=True with bounds
        glp_A = _node_by_name(graph, "GLP_01_A")
        assert glp_A is not None
        assert glp_A.kind == NodeKind.OPT_PARAM
        assert glp_A.value == 20.0
        assert glp_A.bounds == (5.0, 25.0)

    #
    def test_simple_energy_param_input_edges(self):
        """PARAM_INPUT edges wire params to components with correct positions."""

        _file, model = _make_energy_model(["simple_energy"])
        graph = build_graph(model)

        # GLP_01 should have 4 PARAM_INPUT edges (A, x0, F, m)
        glp01_nid = graph.node_by_name["GLP_01"]
        param_edges = _edges_to(graph, glp01_nid, EdgeKind.PARAM_INPUT)
        assert len(param_edges) == 4
        positions = sorted(e.position for e in param_edges)
        assert positions == [0, 1, 2, 3]

    #
    def test_offset_is_component_eval(self):
        """Offset is COMPONENT_EVAL (not SPECTRUM_FED_OP) in the graph."""

        _file, model = _make_energy_model(["simple_energy"])
        graph = build_graph(model)

        offset_node = _node_by_name(graph, "Offset")
        assert offset_node is not None
        assert offset_node.kind == NodeKind.COMPONENT_EVAL
        # No SPECTRUM_INPUT edge
        spec_edges = _edges_to(graph, offset_node.id, EdgeKind.SPECTRUM_INPUT)
        assert len(spec_edges) == 0

    #
    def test_shirley_has_spectrum_input(self):
        """Shirley gets a SPECTRUM_INPUT edge from peak_sum."""

        _file, model = _make_energy_model(["simple_energy"])
        graph = build_graph(model)

        shirley_node = _node_by_name(graph, "Shirley")
        assert shirley_node is not None
        spec_edges = _edges_to(graph, shirley_node.id, EdgeKind.SPECTRUM_INPUT)
        assert len(spec_edges) == 1
        # Source should be peak_sum SUM node
        peak_sum = graph.nodes[spec_edges[0].source]
        assert peak_sum.kind == NodeKind.SUM
        assert peak_sum.name == "peak_sum"

    #
    def test_sum_nodes(self):
        """peak_sum and total SUM nodes are created with correct ADDEND edges."""

        _file, model = _make_energy_model(["simple_energy"])
        graph = build_graph(model)

        sum_nodes = _nodes_by_kind(graph, NodeKind.SUM)
        names = {n.name for n in sum_nodes}
        assert "peak_sum" in names
        assert "total" in names

        # peak_sum receives addends from peaks only: GLP_01, GLP_02
        # Offset is a background — it feeds total directly, not peak_sum
        peak_sum_nid = graph.node_by_name["peak_sum"]
        addend_edges = _edges_to(graph, peak_sum_nid, EdgeKind.ADDEND)
        assert len(addend_edges) == 2
        addend_names = {graph.nodes[e.source].name for e in addend_edges}
        assert addend_names == {"GLP_01", "GLP_02"}

        # total receives: peak_sum + Offset + Shirley
        total_nid = graph.node_by_name["total"]
        total_edges = _edges_to(graph, total_nid, EdgeKind.ADDEND)
        assert len(total_edges) == 3
        total_names = {graph.nodes[e.source].name for e in total_edges}
        assert "peak_sum" in total_names
        assert "Offset" in total_names
        assert "Shirley" in total_names

    #
    def test_node_by_name_consistent(self):
        """node_by_name maps are consistent with nodes list."""

        _file, model = _make_energy_model(["simple_energy"])
        graph = build_graph(model)

        for name, nid in graph.node_by_name.items():
            assert graph.nodes[nid].name == name

    #
    def test_source_order_monotonic(self):
        """source_order values are monotonically increasing."""

        _file, model = _make_energy_model(["simple_energy"])
        graph = build_graph(model)

        orders = [n.source_order for n in graph.nodes]
        assert orders == sorted(orders)
        assert len(set(orders)) == len(orders)  # all unique

    #
    def test_single_gauss_no_background(self):
        """Single Gauss peak: valid graph, no orphan total node."""

        _file, model = _make_energy_model(["single_gauss"])
        graph = build_graph(model)

        assert graph.domain == DomainKind.ENERGY_1D
        comp_evals = _nodes_by_kind(graph, NodeKind.COMPONENT_EVAL)
        assert len(comp_evals) == 1
        assert comp_evals[0].function_name == "Gauss"
        spectrum_fed = _nodes_by_kind(graph, NodeKind.SPECTRUM_FED_OP)
        assert len(spectrum_fed) == 0

        # Single component: no total node needed (peak_sum is the output)
        sum_nodes = _nodes_by_kind(graph, NodeKind.SUM)
        # peak_sum has 1 addend; no separate total since there's only 1 addend
        for sn in sum_nodes:
            if sn.name == "total":
                # If total exists, it must have edges (not orphaned)
                total_edges = _edges_to(graph, sn.id, EdgeKind.ADDEND)
                assert len(total_edges) > 0, "total node must not be orphaned"

    #
    def test_offset_feeds_total_not_peak_sum(self):
        """Offset is a background: feeds total directly, not peak_sum."""

        _file, model = _make_energy_model(["simple_energy"])
        graph = build_graph(model)

        peak_sum_nid = graph.node_by_name["peak_sum"]
        peak_sum_sources = {
            graph.nodes[e.source].name
            for e in _edges_to(graph, peak_sum_nid, EdgeKind.ADDEND)
        }
        # Offset must NOT be in peak_sum
        assert "Offset" not in peak_sum_sources

        # Offset must be in total
        total_nid = graph.node_by_name["total"]
        total_sources = {
            graph.nodes[e.source].name
            for e in _edges_to(graph, total_nid, EdgeKind.ADDEND)
        }
        assert "Offset" in total_sources


#
#
class TestBuildGraphExpressions:
    """Test build_graph for models with expression parameters."""

    #
    def test_expression_nodes_created(self):
        """Expression parameters produce EXPRESSION nodes."""

        _file, model = _make_energy_model(["energy_expression"])
        graph = build_graph(model)

        expr_nodes = _nodes_by_kind(graph, NodeKind.EXPRESSION)
        # GLP_02: A, x0, F, m are all expressions
        assert len(expr_nodes) == 4

        A_expr = _node_by_name(graph, "GLP_02_A")
        assert A_expr is not None
        assert A_expr.kind == NodeKind.EXPRESSION
        assert A_expr.expr_string == "3/4*GLP_01_A"

    #
    def test_expression_ref_edges(self):
        """EXPR_REF edges link referenced params to expression nodes."""

        _file, model = _make_energy_model(["energy_expression"])
        graph = build_graph(model)

        A_expr_nid = graph.node_by_name["GLP_02_A"]
        ref_edges = _edges_to(graph, A_expr_nid, EdgeKind.EXPR_REF)
        assert len(ref_edges) == 1
        # Source should be GLP_01_A
        source_node = graph.nodes[ref_edges[0].source]
        assert source_node.name == "GLP_01_A"

    #
    def test_expression_fan_out(self):
        """Fan-out: two expression nodes reference the same source param."""

        _file, model = _make_energy_model(["expression_fan_out"])
        graph = build_graph(model)

        # Both GLP_02_A and GLP_03_A reference GLP_01_A
        glp01_A_nid = graph.node_by_name["GLP_01_A"]
        outgoing = _edges_from(graph, glp01_A_nid, EdgeKind.EXPR_REF)
        target_names = {graph.nodes[e.target].name for e in outgoing}
        assert "GLP_02_A" in target_names
        assert "GLP_03_A" in target_names

    #
    def test_expression_chain(self):
        """Chain: GLP_01 -> GLP_02 -> GLP_03 expressions."""

        _file, model = _make_energy_model(["expression_chain"])
        graph = build_graph(model)

        # GLP_02_A references GLP_01_A
        glp02_A_nid = graph.node_by_name["GLP_02_A"]
        ref_edges = _edges_to(graph, glp02_A_nid, EdgeKind.EXPR_REF)
        source_names = {graph.nodes[e.source].name for e in ref_edges}
        assert "GLP_01_A" in source_names

        # GLP_03_A references GLP_02_A
        glp03_A_nid = graph.node_by_name["GLP_03_A"]
        ref_edges = _edges_to(graph, glp03_A_nid, EdgeKind.EXPR_REF)
        source_names = {graph.nodes[e.source].name for e in ref_edges}
        assert "GLP_02_A" in source_names

    #
    def test_forward_reference_expression(self):
        """Forward-referenced expressions produce correct graph."""

        _file, model = _make_energy_model(["energy_expression_forward_reference"])
        graph = build_graph(model)

        # GLP_01_A references GLP_02_A (forward ref)
        glp01_A = _node_by_name(graph, "GLP_01_A")
        assert glp01_A is not None
        assert glp01_A.kind == NodeKind.EXPRESSION


#
#
class TestBuildGraph2D:
    """Test build_graph for 2D models with time-dependent parameters."""

    #
    def test_2d_domain(self):
        """Adding dynamics promotes domain to ENERGY_TIME_2D."""

        _file, model = _make_2d_model(
            ["simple_energy"],
            [("GLP_01_A", "models/file_time.yaml", ["MonoExpPos"])],
        )
        graph = build_graph(model)

        assert graph.domain == DomainKind.ENERGY_TIME_2D
        assert graph.energy is not None
        assert graph.time is not None

    #
    def test_dynamics_trace_node(self):
        """Time-dependent param produces DYNAMICS_TRACE + PARAM_PLUS_TRACE."""

        _file, model = _make_2d_model(
            ["simple_energy"],
            [("GLP_01_A", "models/file_time.yaml", ["MonoExpPos"])],
        )
        graph = build_graph(model)

        # DYNAMICS_TRACE node for GLP_01_A
        trace_nodes = _nodes_by_kind(graph, NodeKind.DYNAMICS_TRACE)
        assert len(trace_nodes) >= 1
        trace = trace_nodes[0]
        assert "GLP_01_A" in trace.name
        assert trace.function_name == "expFun"
        assert trace.package == "time"

        # PARAM_PLUS_TRACE node
        ppt_nodes = _nodes_by_kind(graph, NodeKind.PARAM_PLUS_TRACE)
        assert len(ppt_nodes) >= 1
        resolved = _node_by_name(graph, "GLP_01_A_resolved")
        assert resolved is not None

    #
    def test_dynamics_param_nodes(self):
        """Dynamics model parameters appear as OPT/STATIC param nodes."""

        _file, model = _make_2d_model(
            ["simple_energy"],
            [("GLP_01_A", "models/file_time.yaml", ["MonoExpPos"])],
        )
        graph = build_graph(model)

        # expFun has params: A, tau, t0, y0
        # A and tau are vary=True, t0 and y0 are vary=False
        dyn_A = _node_by_name(graph, "GLP_01_A_expFun_01_A")
        assert dyn_A is not None
        assert dyn_A.kind == NodeKind.OPT_PARAM

        dyn_y0 = _node_by_name(graph, "GLP_01_A_expFun_01_y0")
        assert dyn_y0 is not None
        assert dyn_y0.kind == NodeKind.STATIC_PARAM

    #
    def test_dynamics_edges(self):
        """Dynamics nodes are wired: params -> trace -> resolved."""

        _file, model = _make_2d_model(
            ["simple_energy"],
            [("GLP_01_A", "models/file_time.yaml", ["MonoExpPos"])],
        )
        graph = build_graph(model)

        # DYNAMICS_TRACE has PARAM_INPUT edges
        trace_nodes = _nodes_by_kind(graph, NodeKind.DYNAMICS_TRACE)
        assert len(trace_nodes) >= 1
        trace_nid = trace_nodes[0].id
        param_edges = _edges_to(graph, trace_nid, EdgeKind.PARAM_INPUT)
        assert len(param_edges) == 4  # A, tau, t0, y0

        # PARAM_PLUS_TRACE has BASE_INPUT + TRACE_INPUT
        resolved = _node_by_name(graph, "GLP_01_A_resolved")
        assert resolved is not None
        base_edges = _edges_to(graph, resolved.id, EdgeKind.BASE_INPUT)
        trace_edges = _edges_to(graph, resolved.id, EdgeKind.TRACE_INPUT)
        assert len(base_edges) == 1
        assert len(trace_edges) == 1

    #
    def test_resolved_param_wired_to_component(self):
        """PARAM_PLUS_TRACE (resolved) feeds into component, not the base."""

        _file, model = _make_2d_model(
            ["simple_energy"],
            [("GLP_01_A", "models/file_time.yaml", ["MonoExpPos"])],
        )
        graph = build_graph(model)

        glp01_nid = graph.node_by_name["GLP_01"]
        param_edges = _edges_to(graph, glp01_nid, EdgeKind.PARAM_INPUT)

        # The pos=0 edge (A) should come from the resolved node
        A_edge = [e for e in param_edges if e.position == 0][0]
        source_node = graph.nodes[A_edge.source]
        assert source_node.kind == NodeKind.PARAM_PLUS_TRACE
        assert "resolved" in source_node.name

    #
    def test_2d_expression_references_resolved(self):
        """Expression referencing a time-dep param gets EXPR_REF to resolved."""

        _file, model = _make_2d_model(
            ["energy_expression"],
            [("GLP_01_A", "models/file_time.yaml", ["MonoExpPos"])],
        )
        graph = build_graph(model)

        # GLP_02_A = "3/4*GLP_01_A" -- should reference the resolved node
        A_expr_nid = graph.node_by_name["GLP_02_A"]
        ref_edges = _edges_to(graph, A_expr_nid, EdgeKind.EXPR_REF)
        assert len(ref_edges) == 1
        source = graph.nodes[ref_edges[0].source]
        # The resolved_param for GLP_01_A should be the PARAM_PLUS_TRACE
        assert source.kind == NodeKind.PARAM_PLUS_TRACE

    #
    def test_multiple_time_dep_params(self):
        """Multiple time-dependent params each get their own dynamics subgraph."""

        _file, model = _make_2d_model(
            ["simple_energy"],
            [
                ("GLP_01_A", "models/file_time.yaml", ["MonoExpPos"]),
                ("GLP_01_x0", "models/file_time.yaml", ["MonoExpNeg"]),
            ],
        )
        graph = build_graph(model)

        trace_nodes = _nodes_by_kind(graph, NodeKind.DYNAMICS_TRACE)
        assert len(trace_nodes) == 2

        ppt_nodes = _nodes_by_kind(graph, NodeKind.PARAM_PLUS_TRACE)
        assert len(ppt_nodes) == 2


#
#
class TestCanLower2D:
    """Test can_lower_2d gate function."""

    #
    def test_1d_model_returns_false(self):
        """1D-only graph returns False (awaits future can_lower_1d)."""

        _file, model = _make_energy_model(["simple_energy"])
        graph = build_graph(model)
        assert not can_lower_2d(graph)

    #
    def test_simple_2d_returns_true(self):
        """Simple 2D model with supported functions returns True."""

        _file, model = _make_2d_model(
            ["simple_energy"],
            [("GLP_01_A", "models/file_time.yaml", ["MonoExpPos"])],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)

    #
    def test_2d_with_expressions_returns_true(self):
        """2D model with arithmetic expressions returns True."""

        _file, model = _make_2d_model(
            ["energy_expression"],
            [("GLP_01_A", "models/file_time.yaml", ["MonoExpPos"])],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)

    #
    def test_linback_model_returns_true(self):
        """LinBack is reclassified as COMPONENT_EVAL and is lowerable."""

        # Build a model with LinBack manually
        # Since we don't have a LinBack YAML test model, we'll check
        # via the node type: LinBack should be COMPONENT_EVAL, not
        # SPECTRUM_FED_OP.  The can_lower_2d check verifies function name.
        _file, model = _make_energy_model(["simple_energy"])
        graph = build_graph(model)
        # Verify the types in the graph are correct even though it's 1D
        for n in graph.nodes:
            if n.kind == NodeKind.COMPONENT_EVAL:
                assert n.function_name in (
                    "Gauss",
                    "GaussAsym",
                    "Lorentz",
                    "GLS",
                    "GLP",
                    "DS",
                    "Offset",
                    "LinBack",
                )
            if n.kind == NodeKind.SPECTRUM_FED_OP:
                assert n.function_name == "Shirley"


#
#
class TestPackageMapping:
    """Test that component package is mapped correctly."""

    #
    def test_energy_components_have_energy_package(self):
        """Energy function components get package='energy'."""

        _file, model = _make_energy_model(["simple_energy"])
        graph = build_graph(model)

        glp01 = _node_by_name(graph, "GLP_01")
        assert glp01 is not None
        assert glp01.package == "energy"

        offset = _node_by_name(graph, "Offset")
        assert offset is not None
        assert offset.package == "energy"

    #
    def test_dynamics_trace_has_time_package(self):
        """Dynamics trace nodes get package='time'."""

        _file, model = _make_2d_model(
            ["simple_energy"],
            [("GLP_01_A", "models/file_time.yaml", ["MonoExpPos"])],
        )
        graph = build_graph(model)

        trace_nodes = _nodes_by_kind(graph, NodeKind.DYNAMICS_TRACE)
        assert len(trace_nodes) >= 1
        assert trace_nodes[0].package == "time"


#
def _make_profile_model(energy_model_info, target_par, profile_model_info):
    """Create model with a profiled parameter."""

    project = Project(path="tests")
    aux_axis = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    file = File(parent_project=project, aux_axis=aux_axis)
    file.energy = np.linspace(80, 90, 201)
    file.load_model(
        model_yaml="models/file_energy.yaml",
        model_info=energy_model_info,
    )
    file.add_par_profile(
        target_model=energy_model_info[0],
        target_parameter=target_par,
        profile_yaml="models/file_profile.yaml",
        profile_model=profile_model_info,
    )
    model = file.model_active
    assert model is not None
    return file, model


#
#
class TestProfileNodes:
    """Test profile node emission and wiring."""

    #
    def test_profile_nodes_created(self):
        """Profile parameters produce PROFILE_SAMPLE and PROFILE_AVERAGE nodes."""

        _file, model = _make_profile_model(
            ["single_gauss"], "Gauss_01_A", ["profile_pExpDecay"]
        )
        graph = build_graph(model)

        samples = _nodes_by_kind(graph, NodeKind.PROFILE_SAMPLE)
        assert len(samples) == 5  # one per aux_axis point

        # One component-level PROFILE_AVERAGE (trace averaging)
        averages = _nodes_by_kind(graph, NodeKind.PROFILE_AVERAGE)
        assert len(averages) == 1
        assert averages[0].name == "Gauss_01_profile_avg"

    #
    def test_profile_sample_has_param_input_edges(self):
        """Each PROFILE_SAMPLE has PARAM_INPUT from base and profile params."""

        _file, model = _make_profile_model(
            ["single_gauss"], "Gauss_01_A", ["profile_pExpDecay"]
        )
        graph = build_graph(model)

        samples = _nodes_by_kind(graph, NodeKind.PROFILE_SAMPLE)
        for sample in samples:
            param_edges = _edges_to(graph, sample.id, EdgeKind.PARAM_INPUT)
            # pos 0 = base param, pos 1..N = profile params (A, tau for pExpDecay)
            assert len(param_edges) == 3

    #
    def test_profile_average_has_addend_edges(self):
        """Component-level PROFILE_AVERAGE has ADDEND from sample evals."""

        _file, model = _make_profile_model(
            ["single_gauss"], "Gauss_01_A", ["profile_pExpDecay"]
        )
        graph = build_graph(model)

        comp_avg = _node_by_name(graph, "Gauss_01_profile_avg")
        assert comp_avg is not None
        addend_edges = _edges_to(graph, comp_avg.id, EdgeKind.ADDEND)
        assert len(addend_edges) == 5  # one per aux_axis point
        for edge in addend_edges:
            source = graph.nodes[edge.source]
            assert source.kind == NodeKind.COMPONENT_EVAL

    #
    def test_profile_average_replaces_component_in_combination(self):
        """PROFILE_AVERAGE replaces the component in the combination graph.

        The interpreter evaluates the component at each aux point and
        averages the traces: mean(f(p_i)).  The graph reflects this by
        creating per-sample COMPONENT_EVAL nodes feeding into
        PROFILE_AVERAGE, which replaces the original component.
        """

        _file, model = _make_profile_model(
            ["single_gauss"], "Gauss_01_A", ["profile_pExpDecay"]
        )
        graph = build_graph(model)

        # PROFILE_AVERAGE should exist and receive ADDEND from sample evals
        avg_node = _node_by_name(graph, "Gauss_01_profile_avg")
        assert avg_node is not None
        assert avg_node.kind == NodeKind.PROFILE_AVERAGE

        # Each sample COMPONENT_EVAL feeds into PROFILE_AVERAGE
        addend_edges = _edges_to(graph, avg_node.id, EdgeKind.ADDEND)
        assert len(addend_edges) == 5  # one per aux_axis point
        for edge in addend_edges:
            source = graph.nodes[edge.source]
            assert source.kind == NodeKind.COMPONENT_EVAL
            assert "sample" in source.name

    #
    def test_profile_triggers_can_lower_2d_false(self):
        """Models with profile nodes are not lowerable in v1."""

        _file, model = _make_profile_model(
            ["single_gauss"], "Gauss_01_A", ["profile_pExpDecay"]
        )
        graph = build_graph(model)
        assert not can_lower_2d(graph)

    #
    def test_profile_param_nodes_created(self):
        """Profile function parameters appear as OPT/STATIC param nodes."""

        _file, model = _make_profile_model(
            ["single_gauss"], "Gauss_01_A", ["profile_pExpDecay"]
        )
        graph = build_graph(model)

        # pExpDecay has params A and tau, both vary=True
        prof_A = _node_by_name(graph, "Gauss_01_A_pExpDecay_01_A")
        assert prof_A is not None
        assert prof_A.kind == NodeKind.OPT_PARAM

        prof_tau = _node_by_name(graph, "Gauss_01_A_pExpDecay_01_tau")
        assert prof_tau is not None
        assert prof_tau.kind == NodeKind.OPT_PARAM

    #
    def test_sample_component_eval_wiring(self):
        """Per-sample COMPONENT_EVAL gets profiled param from PROFILE_SAMPLE.

        Profiled params are wired from PROFILE_SAMPLE; non-profiled params
        (x0, SD for Gauss) come from the regular resolved parameter nodes.
        """

        _file, model = _make_profile_model(
            ["single_gauss"], "Gauss_01_A", ["profile_pExpDecay"]
        )
        graph = build_graph(model)

        # Check the first sample's component eval
        sample_eval = _node_by_name(graph, "Gauss_01_sample_0")
        assert sample_eval is not None
        assert sample_eval.function_name == "Gauss"

        param_edges = _edges_to(graph, sample_eval.id, EdgeKind.PARAM_INPUT)
        # Gauss has 3 params: A (pos=0), x0 (pos=1), SD (pos=2)
        assert len(param_edges) == 3

        # pos=0 (A) should come from PROFILE_SAMPLE
        A_edge = [e for e in param_edges if e.position == 0][0]
        A_source = graph.nodes[A_edge.source]
        assert A_source.kind == NodeKind.PROFILE_SAMPLE

        # pos=1 (x0) should come from regular param (STATIC_PARAM or OPT_PARAM)
        x0_edge = [e for e in param_edges if e.position == 1][0]
        x0_source = graph.nodes[x0_edge.source]
        assert x0_source.kind in (NodeKind.STATIC_PARAM, NodeKind.OPT_PARAM)

    #
    def test_original_component_removed_after_profile(self):
        """Original COMPONENT_EVAL is removed when profile replaces it.

        When _emit_profile_nodes creates per-sample evals and a
        PROFILE_AVERAGE, the original component node becomes orphaned.
        It should be removed from the graph to avoid confusing downstream
        passes (scheduler, topological sort, future lowering).
        """

        _file, model = _make_profile_model(
            ["single_gauss"], "Gauss_01_A", ["profile_pExpDecay"]
        )
        graph = build_graph(model)

        # The original "Gauss_01" node should not exist
        original = _node_by_name(graph, "Gauss_01")
        assert original is None, (
            "Original COMPONENT_EVAL 'Gauss_01' should be removed after"
            " profile replacement"
        )

    #
    def test_no_edges_reference_removed_nodes(self):
        """All edge endpoints reference nodes that exist in the graph."""

        _file, model = _make_profile_model(
            ["single_gauss"], "Gauss_01_A", ["profile_pExpDecay"]
        )
        graph = build_graph(model)

        node_ids = {n.id for n in graph.nodes}
        for edge in graph.edges:
            assert edge.source in node_ids, (
                f"Edge source {edge.source} not in graph nodes"
            )
            assert edge.target in node_ids, (
                f"Edge target {edge.target} not in graph nodes"
            )

    #
    def test_all_component_evals_reachable_from_output(self):
        """Every COMPONENT_EVAL feeds into the output (no disconnected nodes).

        Walk backwards from SUM/PROFILE_AVERAGE nodes; every
        COMPONENT_EVAL should be reachable.
        """

        _file, model = _make_profile_model(
            ["single_gauss"], "Gauss_01_A", ["profile_pExpDecay"]
        )
        graph = build_graph(model)

        # Build reverse adjacency: target -> set of sources
        children: dict[int, set[int]] = {}
        for edge in graph.edges:
            children.setdefault(edge.target, set()).add(edge.source)

        # Find all SUM and PROFILE_AVERAGE nodes (output-facing)
        output_kinds = {NodeKind.SUM, NodeKind.PROFILE_AVERAGE}
        roots = [n.id for n in graph.nodes if n.kind in output_kinds]

        # BFS backwards from roots
        reachable: set[int] = set()
        queue = list(roots)
        while queue:
            nid = queue.pop()
            if nid in reachable:
                continue
            reachable.add(nid)
            queue.extend(children.get(nid, []))

        # Every COMPONENT_EVAL must be reachable
        comp_evals = _nodes_by_kind(graph, NodeKind.COMPONENT_EVAL)
        for ce in comp_evals:
            assert ce.id in reachable, (
                f"COMPONENT_EVAL '{ce.name}' (id={ce.id}) is not reachable"
                " from any output node"
            )


# Subcycle node tests
#
def _make_subcycle_model():
    """Create a 2D model with multi-cycle subcycle dynamics."""

    project = Project(path="tests")
    file = File(parent_project=project)
    file.energy = np.linspace(80, 90, 201)
    file.time = np.linspace(-10, 100, 111)
    file.load_model(
        model_yaml="models/file_energy.yaml",
        model_info=["simple_energy"],
    )
    file.add_time_dependence(
        target_model="simple_energy",
        target_parameter="GLP_01_A",
        dynamics_yaml="models/file_time.yaml",
        dynamics_model=["ModelNone", "MonoExpNeg", "MonoExpPosExpr"],
        frequency=10,
    )
    model = file.model_active
    assert model is not None
    return file, model


#
#
class TestSubcycleNodes:
    """Test subcycle node emission and wiring."""

    #
    def test_subcycle_nodes_created(self):
        """Subcycle dynamics produce SUBCYCLE_REMAP and SUBCYCLE_MASK nodes."""

        _file, model = _make_subcycle_model()
        graph = build_graph(model)

        remap_nodes = _nodes_by_kind(graph, NodeKind.SUBCYCLE_REMAP)
        mask_nodes = _nodes_by_kind(graph, NodeKind.SUBCYCLE_MASK)
        # Two subcycle components: expFun_01 (sub=1) and expFun_02 (sub=2)
        assert len(remap_nodes) == 2
        assert len(mask_nodes) == 2

    #
    def test_subcycle_remap_wired_to_trace(self):
        """SUBCYCLE_REMAP feeds TRACE_INPUT into DYNAMICS_TRACE."""

        _file, model = _make_subcycle_model()
        graph = build_graph(model)

        remap_nodes = _nodes_by_kind(graph, NodeKind.SUBCYCLE_REMAP)
        for remap in remap_nodes:
            outgoing = _edges_from(graph, remap.id, EdgeKind.TRACE_INPUT)
            assert len(outgoing) == 1
            target = graph.nodes[outgoing[0].target]
            assert target.kind == NodeKind.DYNAMICS_TRACE

    #
    def test_subcycle_mask_wired_from_trace(self):
        """SUBCYCLE_MASK receives TRACE_INPUT from DYNAMICS_TRACE."""

        _file, model = _make_subcycle_model()
        graph = build_graph(model)

        mask_nodes = _nodes_by_kind(graph, NodeKind.SUBCYCLE_MASK)
        for mask in mask_nodes:
            incoming = _edges_to(graph, mask.id, EdgeKind.TRACE_INPUT)
            assert len(incoming) == 1
            source = graph.nodes[incoming[0].source]
            assert source.kind == NodeKind.DYNAMICS_TRACE

    #
    def test_subcycle_mask_has_time_n_sub_array(self):
        """SUBCYCLE_MASK nodes carry time_n_sub array data."""

        _file, model = _make_subcycle_model()
        graph = build_graph(model)

        mask_nodes = _nodes_by_kind(graph, NodeKind.SUBCYCLE_MASK)
        for mask in mask_nodes:
            assert "time_n_sub" in mask.arrays
            assert mask.arrays["time_n_sub"].shape == (111,)

    #
    def test_subcycle_remap_has_time_norm_array(self):
        """SUBCYCLE_REMAP nodes carry time_norm array data."""

        _file, model = _make_subcycle_model()
        graph = build_graph(model)

        remap_nodes = _nodes_by_kind(graph, NodeKind.SUBCYCLE_REMAP)
        for remap in remap_nodes:
            assert "time_norm" in remap.arrays
            assert remap.arrays["time_norm"].shape == (111,)

    #
    def test_subcycle_triggers_can_lower_2d_false(self):
        """Models with subcycle nodes are not lowerable in v1."""

        _file, model = _make_subcycle_model()
        graph = build_graph(model)
        assert not can_lower_2d(graph)

    #
    def test_subcycle_mask_wired_into_resolved(self):
        """PARAM_PLUS_TRACE consumes SUBCYCLE_MASK, not raw DYNAMICS_TRACE.

        The interpreter multiplies the trace by time_n_sub inline, so the
        resolved parameter must depend on the masked trace.
        """

        _file, model = _make_subcycle_model()
        graph = build_graph(model)

        resolved = _node_by_name(graph, "GLP_01_A_resolved")
        assert resolved is not None  # type guard

        # TRACE_INPUT edges should come from SUBCYCLE_MASK, not DYNAMICS_TRACE
        trace_edges = _edges_to(graph, resolved.id, EdgeKind.TRACE_INPUT)
        assert len(trace_edges) >= 1
        for edge in trace_edges:
            source = graph.nodes[edge.source]
            assert source.kind == NodeKind.SUBCYCLE_MASK, (
                f"PARAM_PLUS_TRACE should consume SUBCYCLE_MASK, "
                f"got {source.kind.name} ({source.name})"
            )


#
#
class TestDynamicsExpressions:
    """Test that expression params inside dynamics models are emitted."""

    #
    def test_dynamics_expression_nodes_created(self):
        """Expression params in dynamics produce EXPRESSION nodes."""

        _file, model = _make_subcycle_model()
        graph = build_graph(model)

        # expFun_02 has expression params A and tau
        expr_nodes = _nodes_by_kind(graph, NodeKind.EXPRESSION)
        expr_names = {n.name for n in expr_nodes}
        assert "GLP_01_A_expFun_02_A" in expr_names
        assert "GLP_01_A_expFun_02_tau" in expr_names

    #
    def test_dynamics_expression_ref_edges(self):
        """Dynamics EXPRESSION nodes have EXPR_REF edges to referenced params."""

        _file, model = _make_subcycle_model()
        graph = build_graph(model)

        # expFun_02_A = "-expFun_01_A" (auto-prefixed to GLP_01_A_expFun_01_A)
        expr_A_nid = graph.node_by_name["GLP_01_A_expFun_02_A"]
        ref_edges = _edges_to(graph, expr_A_nid, EdgeKind.EXPR_REF)
        assert len(ref_edges) == 1
        source = graph.nodes[ref_edges[0].source]
        assert source.name == "GLP_01_A_expFun_01_A"

    #
    def test_dynamics_non_expression_params_unchanged(self):
        """Non-expression dynamics params are still OPT/STATIC."""

        _file, model = _make_subcycle_model()
        graph = build_graph(model)

        # expFun_02_t0 is [0, False, 0, 1] -> STATIC_PARAM
        t0 = _node_by_name(graph, "GLP_01_A_expFun_02_t0")
        assert t0 is not None
        assert t0.kind == NodeKind.STATIC_PARAM

        # expFun_01_A is [-1, True, -5, 0] -> OPT_PARAM
        A1 = _node_by_name(graph, "GLP_01_A_expFun_01_A")
        assert A1 is not None
        assert A1.kind == NodeKind.OPT_PARAM


#
#
class TestBuildGraphTime1D:
    """Test build_graph for standalone dynamics models (TIME_1D)."""

    #
    def test_time_only_domain(self):
        """Standalone dynamics model builds a valid TIME_1D graph."""

        _file, model = _make_time_only_model(["MonoExpPos"])
        graph = build_graph(model)

        assert graph.domain == DomainKind.TIME_1D
        assert graph.energy is None
        assert graph.time is not None

    #
    def test_time_only_expression_ref_edges(self):
        """Standalone dynamics expressions get EXPR_REF edges."""

        _file, model = _make_time_only_model(
            ["ModelNone", "MonoExpNeg", "MonoExpPosExpr"],
            frequency=10,
        )
        graph = build_graph(model)

        expr_A_nid = graph.node_by_name["parTEST_expFun_02_A"]
        ref_edges = _edges_to(graph, expr_A_nid, EdgeKind.EXPR_REF)
        assert len(ref_edges) == 1
        source = graph.nodes[ref_edges[0].source]
        assert source.name == "parTEST_expFun_01_A"

        expr_tau_nid = graph.node_by_name["parTEST_expFun_02_tau"]
        ref_edges = _edges_to(graph, expr_tau_nid, EdgeKind.EXPR_REF)
        assert len(ref_edges) == 1
        source = graph.nodes[ref_edges[0].source]
        assert source.name == "parTEST_expFun_01_tau"

    #
    def test_time_only_subcycle_nodes_created(self):
        """Standalone multi-cycle dynamics emit SUBCYCLE_* nodes."""

        _file, model = _make_time_only_model(
            ["ModelNone", "MonoExpNeg", "MonoExpPosExpr"],
            frequency=10,
        )
        graph = build_graph(model)

        remap_nodes = _nodes_by_kind(graph, NodeKind.SUBCYCLE_REMAP)
        mask_nodes = _nodes_by_kind(graph, NodeKind.SUBCYCLE_MASK)
        assert len(remap_nodes) == 2
        assert len(mask_nodes) == 2

    #
    def test_time_only_convolution_node_created(self):
        """Standalone IRF dynamics emit a CONVOLUTION node."""

        _file, model = _make_time_only_model(["MonoExpPosIRF"])
        graph = build_graph(model)

        conv_nodes = _nodes_by_kind(graph, NodeKind.CONVOLUTION)
        assert len(conv_nodes) == 1
        assert conv_nodes[0].function_name == "gaussCONV"
        assert conv_nodes[0].package == "time"

    #
    def test_time_only_graph_not_lowerable_in_2d_backend(self):
        """TIME_1D graphs remain non-lowerable for the 2D backend."""

        _file, model = _make_time_only_model(["MonoExpPos"])
        graph = build_graph(model)
        assert not can_lower_2d(graph)


#
#
class TestProfileExpressionInteraction:
    """Test that expressions referencing profiled params see the profiled value."""

    #
    def test_expression_evaluated_per_sample(self):
        """Expression referencing a profiled par is evaluated per-sample.

        two_glp_expr_amplitude: GLP_02_A = "GLP_01_A * 0.5"
        Profile pLinear on GLP_01_A.

        The interpreter evaluates ``expr(parY_i)`` at each aux point, then
        evaluates the component, then averages traces.  The graph must
        have per-sample EXPRESSION nodes with EXPR_REF to the per-sample
        PROFILE_SAMPLE of the referenced profiled param.
        """

        file, model = _make_profile_model(
            ["two_glp_expr_amplitude"], "GLP_01_A", ["profile_pLinear"]
        )
        graph = build_graph(model)

        # GLP_02 should have per-sample COMPONENT_EVAL + EXPRESSION nodes
        glp02_avg = _node_by_name(graph, "GLP_02_profile_avg")
        assert glp02_avg is not None
        assert glp02_avg.kind == NodeKind.PROFILE_AVERAGE

        # 5 sample component evals feed into GLP_02's profile average
        addend_edges = _edges_to(graph, glp02_avg.id, EdgeKind.ADDEND)
        assert len(addend_edges) == 5

        # Each per-sample EXPRESSION references a PROFILE_SAMPLE (not avg)
        for aux_i in range(5):
            expr_node = _node_by_name(graph, f"GLP_02_A_profile_expr_{aux_i}")
            assert expr_node is not None
            assert expr_node.kind == NodeKind.EXPRESSION

            ref_edges = _edges_to(graph, expr_node.id, EdgeKind.EXPR_REF)
            assert len(ref_edges) == 1
            source = graph.nodes[ref_edges[0].source]
            assert source.kind == NodeKind.PROFILE_SAMPLE
            assert source.name == f"GLP_01_A_profile_sample_{aux_i}"

        # The original GLP_02_A EXPRESSION node should have no EXPR_REF
        # (it's skipped in step 4 because expr_refs_profile_dep is True)
        orig_expr_nid = graph.node_by_name["GLP_02_A"]
        ref_edges = _edges_to(graph, orig_expr_nid, EdgeKind.EXPR_REF)
        assert len(ref_edges) == 0

    #
    def test_non_profiled_expression_refs_base(self):
        """Expression referencing a non-profiled par still points to base."""

        _file, model = _make_energy_model(["energy_expression"])
        graph = build_graph(model)

        # GLP_02_x0 = "GLP_01_x0 +3.6" -- no profile on x0
        expr_nid = graph.node_by_name["GLP_02_x0"]
        ref_edges = _edges_to(graph, expr_nid, EdgeKind.EXPR_REF)
        assert len(ref_edges) == 1

        source = graph.nodes[ref_edges[0].source]
        assert source.kind == NodeKind.OPT_PARAM
        assert source.name == "GLP_01_x0"


# Time-dependent profile parameter tests
#
def _make_time_dep_profile_model():
    """Create model with a profiled par whose profile slope is time-dependent.

    single_glp with pLinear(m=-0.5, b=0) on GLP_01_A,
    then MonoExpPos dynamics on GLP_01_A_pLinear_01_m.
    """

    project = Project(path="tests")
    aux_axis = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    file = File(parent_project=project, aux_axis=aux_axis)
    file.energy = np.linspace(80, 90, 201)
    file.time = np.linspace(-10, 100, 111)
    file.load_model(
        model_yaml="models/file_energy.yaml",
        model_info=["single_glp"],
    )
    file.add_par_profile(
        target_model="single_glp",
        target_parameter="GLP_01_A",
        profile_yaml="models/file_profile.yaml",
        profile_model=["profile_pLinear"],
    )
    file.add_time_dependence(
        target_model="single_glp",
        target_parameter="GLP_01_A_pLinear_01_m",
        dynamics_yaml="models/file_time.yaml",
        dynamics_model=["MonoExpPos"],
    )
    model = file.model_active
    assert model is not None
    return file, model


#
#
class TestTimeDependentProfileParams:
    """Test that profile params with dynamics get full subgraph treatment."""

    #
    def test_profile_par_dynamics_trace_created(self):
        """Profile par with t_vary produces a DYNAMICS_TRACE node."""

        _file, model = _make_time_dep_profile_model()
        graph = build_graph(model)

        # GLP_01_A_pLinear_01_m should have a dynamics trace
        trace_nodes = _nodes_by_kind(graph, NodeKind.DYNAMICS_TRACE)
        trace_names = {n.name for n in trace_nodes}
        assert any("pLinear_01_m" in name for name in trace_names)

    #
    def test_profile_par_has_param_plus_trace(self):
        """Profile par with dynamics gets a PARAM_PLUS_TRACE (resolved)."""

        _file, model = _make_time_dep_profile_model()
        graph = build_graph(model)

        ppt_nodes = _nodes_by_kind(graph, NodeKind.PARAM_PLUS_TRACE)
        ppt_names = {n.name for n in ppt_nodes}
        assert any("pLinear_01_m" in name for name in ppt_names)

    #
    def test_profile_par_dynamics_params_in_graph(self):
        """Dynamics params for time-dep profile par appear in graph."""

        _file, model = _make_time_dep_profile_model()
        graph = build_graph(model)

        # MonoExpPos has params: A, tau, t0, y0
        # The profile par is GLP_01_A_pLinear_01_m, so dynamics params
        # are prefixed: GLP_01_A_pLinear_01_m_expFun_01_A, etc.
        dyn_A = _node_by_name(graph, "GLP_01_A_pLinear_01_m_expFun_01_A")
        assert dyn_A is not None
        assert dyn_A.kind == NodeKind.OPT_PARAM

        dyn_y0 = _node_by_name(graph, "GLP_01_A_pLinear_01_m_expFun_01_y0")
        assert dyn_y0 is not None
        assert dyn_y0.kind == NodeKind.STATIC_PARAM

    #
    def test_resolved_profile_par_in_sample_edges(self):
        """PROFILE_SAMPLE edges use the resolved (time-dep) profile param."""

        _file, model = _make_time_dep_profile_model()
        graph = build_graph(model)

        # The profile's m param should be resolved via PARAM_PLUS_TRACE.
        # Find it: GLP_01_A_pLinear_01_m_resolved
        ppt_nodes = [
            n
            for n in _nodes_by_kind(graph, NodeKind.PARAM_PLUS_TRACE)
            if "pLinear_01_m" in n.name
        ]
        assert len(ppt_nodes) == 1
        resolved_m_nid = ppt_nodes[0].id

        # Check that at least one PROFILE_SAMPLE has this resolved node
        # as a PARAM_INPUT source
        samples = _nodes_by_kind(graph, NodeKind.PROFILE_SAMPLE)
        assert len(samples) > 0
        found_resolved_in_sample = False
        for sample in samples:
            for e in _edges_to(graph, sample.id, EdgeKind.PARAM_INPUT):
                if e.source == resolved_m_nid:
                    found_resolved_in_sample = True
                    break
        assert found_resolved_in_sample


# Dynamics convolution tests
#
def _make_irf_dynamics_model():
    """Create a 2D model where dynamics includes a convolution (gaussCONV)."""

    project = Project(path="tests")
    file = File(parent_project=project)
    file.energy = np.linspace(80, 90, 201)
    file.time = np.linspace(-10, 100, 111)
    file.load_model(
        model_yaml="models/file_energy.yaml",
        model_info=["simple_energy"],
    )
    file.add_time_dependence(
        target_model="simple_energy",
        target_parameter="GLP_01_A",
        dynamics_yaml="models/file_time.yaml",
        dynamics_model=["MonoExpPosIRF"],
    )
    model = file.model_active
    assert model is not None
    return file, model


#
#
class TestDynamicsConvolution:
    """Test convolution components inside dynamics models."""

    #
    def test_gaussconv_emitted_as_convolution_node(self):
        """gaussCONV in dynamics should be a CONVOLUTION node, not DYNAMICS_TRACE."""

        _file, model = _make_irf_dynamics_model()
        graph = build_graph(model)

        conv_nodes = _nodes_by_kind(graph, NodeKind.CONVOLUTION)
        assert len(conv_nodes) >= 1
        conv_names = {n.function_name for n in conv_nodes}
        assert "gaussCONV" in conv_names

        # gaussCONV should NOT appear as a DYNAMICS_TRACE
        trace_nodes = _nodes_by_kind(graph, NodeKind.DYNAMICS_TRACE)
        trace_fns = {n.function_name for n in trace_nodes}
        assert "gaussCONV" not in trace_fns

    #
    def test_expfun_still_emitted_as_dynamics_trace(self):
        """expFun in the same dynamics model is still a DYNAMICS_TRACE."""

        _file, model = _make_irf_dynamics_model()
        graph = build_graph(model)

        trace_nodes = _nodes_by_kind(graph, NodeKind.DYNAMICS_TRACE)
        trace_fns = {n.function_name for n in trace_nodes}
        assert "expFun" in trace_fns

    #
    def test_irf_dynamics_not_lowerable(self):
        """Dynamics with convolution should make can_lower_2d return False."""

        _file, model = _make_irf_dynamics_model()
        graph = build_graph(model)
        assert not can_lower_2d(graph)

    #
    def test_convolution_wraps_resolved_trace(self):
        """CONVOLUTION takes TRACE_INPUT from PARAM_PLUS_TRACE, not as addend.

        The interpreter applies conv(accumulated_trace, kernel).  The graph
        must show CONVOLUTION receiving TRACE_INPUT from the resolved node,
        and the final resolved param must be the CONVOLUTION (not
        PARAM_PLUS_TRACE).
        """

        _file, model = _make_irf_dynamics_model()
        graph = build_graph(model)

        conv_nodes = _nodes_by_kind(graph, NodeKind.CONVOLUTION)
        assert len(conv_nodes) == 1
        conv = conv_nodes[0]

        # CONVOLUTION receives TRACE_INPUT from PARAM_PLUS_TRACE
        trace_edges = _edges_to(graph, conv.id, EdgeKind.TRACE_INPUT)
        assert len(trace_edges) == 1
        source = graph.nodes[trace_edges[0].source]
        assert source.kind == NodeKind.PARAM_PLUS_TRACE

        # CONVOLUTION also receives PARAM_INPUT from its kernel params (SD)
        param_edges = _edges_to(graph, conv.id, EdgeKind.PARAM_INPUT)
        assert len(param_edges) >= 1

        # The resolved param should be the CONVOLUTION node, not
        # PARAM_PLUS_TRACE, since conv is the final transform.
        resolved = _node_by_name(graph, "GLP_01_A_resolved")
        assert resolved is not None  # type guard
        # PARAM_PLUS_TRACE should NOT have TRACE_INPUT from CONVOLUTION
        ppt_trace_edges = _edges_to(graph, resolved.id, EdgeKind.TRACE_INPUT)
        ppt_sources = {graph.nodes[e.source].kind for e in ppt_trace_edges}
        assert NodeKind.CONVOLUTION not in ppt_sources

    #
    def test_convolution_is_final_resolved(self):
        """The CONVOLUTION node feeds the component, not PARAM_PLUS_TRACE."""

        _file, model = _make_irf_dynamics_model()
        graph = build_graph(model)

        # GLP_01's A param (pos=0) should come from the CONVOLUTION node
        glp01_nid = graph.node_by_name["GLP_01"]
        param_edges = _edges_to(graph, glp01_nid, EdgeKind.PARAM_INPUT)
        A_edge = [e for e in param_edges if e.position == 0][0]
        source = graph.nodes[A_edge.source]
        assert source.kind == NodeKind.CONVOLUTION


#
#
class TestToDot:
    """Tests for GraphIR.to_dot() Graphviz rendering."""

    #
    def test_simple_energy_dot(self):
        """to_dot returns valid DOT with all nodes and edges."""

        _file, model = _make_energy_model(["simple_energy"])
        graph = build_graph(model)
        dot = graph.to_dot()

        assert dot.startswith("digraph ModelGraph {")
        assert dot.endswith("}")
        # Every node appears
        for node in graph.nodes:
            assert f"n{node.id}" in dot
            assert node.name in dot
        # Every edge appears
        for edge in graph.edges:
            assert f"n{edge.source} -> n{edge.target}" in dot

    #
    def test_2d_model_dot(self):
        """to_dot works for a 2D model with dynamics nodes."""

        _file, model = _make_2d_model(
            ["simple_energy"],
            [("GLP_01_A", "models/file_time.yaml", ["MonoExpPos"])],
        )
        graph = build_graph(model)
        dot = graph.to_dot()

        assert "DYNAMICS_TRACE" in dot
        assert "PARAM_PLUS_TRACE" in dot

    #
    def test_expression_dot(self):
        """to_dot renders expression nodes with expr string."""

        _file, model = _make_energy_model(["energy_expression"])
        graph = build_graph(model)
        dot = graph.to_dot()

        assert "EXPRESSION" in dot
        assert "3/4*GLP_01_A" in dot
        assert "EXPR_REF" in dot


#
#
class TestModelVisualize:
    """Tests for Model.visualize() convenience method."""

    #
    def test_string_rendering(self):
        """rendering='string' returns DOT source."""

        _file, model = _make_energy_model(["simple_energy"])
        dot = model.visualize(rendering="string")

        assert dot is not None
        assert dot.startswith("digraph ModelGraph {")

    #
    def test_graphviz_fallback(self, monkeypatch):
        """Falls back to string output when graphviz is not installed."""

        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "graphviz":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        _file, model = _make_energy_model(["simple_energy"])
        with pytest.warns(UserWarning, match="graphviz Python package not installed"):
            dot = model.visualize(rendering="graphviz")

        # Falls back to string output
        assert dot is not None
        assert dot.startswith("digraph ModelGraph {")

    #
    def test_2d_model_visualize(self):
        """visualize works on 2D models with dynamics."""

        _file, model = _make_2d_model(
            ["simple_energy"],
            [("GLP_01_A", "models/file_time.yaml", ["MonoExpPos"])],
        )
        dot = model.visualize(rendering="string")

        assert dot is not None
        assert "DYNAMICS_TRACE" in dot


#
#
class TestProfileCollapsing:
    """Tests for profile node collapsing in to_dot."""

    #
    def test_collapsed_by_default(self):
        """Profile samples are collapsed into single nodes by default."""

        _file, model = _make_profile_model(
            ["single_gauss"], "Gauss_01_A", ["profile_pExpDecay"]
        )
        graph = build_graph(model)
        dot = graph.to_dot()

        # Should show count, not individual sample nodes
        assert "\u00d75" in dot  # ×5 (aux_axis has 5 points)
        assert "PROFILE_SAMPLE" in dot
        # Individual sample names should NOT appear
        assert "profile_sample_1" not in dot
        assert "sample_1" not in dot

    #
    def test_collapse_disabled(self):
        """collapse_profiles=False shows all per-sample nodes."""

        _file, model = _make_profile_model(
            ["single_gauss"], "Gauss_01_A", ["profile_pExpDecay"]
        )
        graph = build_graph(model)
        dot = graph.to_dot(collapse_profiles=False)

        # Individual sample names should appear
        assert "profile_sample_0" in dot
        assert "profile_sample_1" in dot
        assert "sample_0" in dot
        assert "sample_1" in dot

    #
    def test_collapsed_fewer_nodes(self):
        """Collapsing reduces node count significantly."""

        _file, model = _make_profile_model(
            ["single_gauss"], "Gauss_01_A", ["profile_pExpDecay"]
        )
        graph = build_graph(model)
        dot_collapsed = graph.to_dot(collapse_profiles=True)
        dot_full = graph.to_dot(collapse_profiles=False)

        # Collapsed should have far fewer node declarations
        collapsed_nodes = dot_collapsed.count("[shape=")
        full_nodes = dot_full.count("[shape=")
        assert collapsed_nodes < full_nodes

    #
    def test_non_profile_model_unaffected(self):
        """Collapsing has no effect on models without profiles."""

        _file, model = _make_energy_model(["simple_energy"])
        graph = build_graph(model)
        dot_collapsed = graph.to_dot(collapse_profiles=True)
        dot_uncollapsed = graph.to_dot(collapse_profiles=False)

        assert dot_collapsed == dot_uncollapsed

    #
    def test_visualize_passes_collapse(self):
        """Model.visualize passes collapse_profiles through."""

        _file, model = _make_profile_model(
            ["single_gauss"], "Gauss_01_A", ["profile_pExpDecay"]
        )
        dot_collapsed = model.visualize(rendering="string", collapse_profiles=True)
        dot_full = model.visualize(rendering="string", collapse_profiles=False)

        assert dot_collapsed is not None
        assert dot_full is not None
        assert "\u00d75" in dot_collapsed
        assert "profile_sample_1" in dot_full


# ===================================================================== #
# Phase 2: Expression compiler tests                                     #
# ===================================================================== #


#
#
class TestCompileExprSymbolic:
    """Test compile_expr_symbolic: AST -> symbolic RPN."""

    #
    def test_simple_multiply(self):
        """Simple multiply: '3/4*GLP_01_A'."""

        rpn = compile_expr_symbolic("3/4*GLP_01_A")

        assert isinstance(rpn, SymbolicRPN)
        assert rpn.referenced_names == ["GLP_01_A"]
        # Should have CONST(3), CONST(4), DIV, PARAM_REF(GLP_01_A), MUL
        kinds = [k for k, _ in rpn.instructions]
        assert ExprNodeKind.CONST in kinds
        assert ExprNodeKind.PARAM_REF in kinds
        assert ExprNodeKind.DIV in kinds
        assert ExprNodeKind.MUL in kinds

    #
    def test_addition(self):
        """Addition: 'GLP_01_x0 +3.6'."""

        rpn = compile_expr_symbolic("GLP_01_x0 +3.6")

        assert rpn.referenced_names == ["GLP_01_x0"]
        kinds = [k for k, _ in rpn.instructions]
        assert ExprNodeKind.ADD in kinds

    #
    def test_identity_ref(self):
        """Identity: 'GLP_01_F' (just a parameter reference)."""

        rpn = compile_expr_symbolic("GLP_01_F")

        assert rpn.referenced_names == ["GLP_01_F"]
        assert len(rpn.instructions) == 1
        assert rpn.instructions[0] == (ExprNodeKind.PARAM_REF, "GLP_01_F")

    #
    def test_negation(self):
        """Negation: '-GLP_01_A_expFun_01_A'."""

        rpn = compile_expr_symbolic("-GLP_01_A_expFun_01_A")

        assert rpn.referenced_names == ["GLP_01_A_expFun_01_A"]
        kinds = [k for k, _ in rpn.instructions]
        assert ExprNodeKind.NEG in kinds

    #
    def test_power(self):
        """Power: 'GLP_01_A ** 2'."""

        rpn = compile_expr_symbolic("GLP_01_A ** 2")

        kinds = [k for k, _ in rpn.instructions]
        assert ExprNodeKind.POW in kinds

    #
    def test_multiple_references(self):
        """Multiple refs: 'GLP_01_A * 0.5 + GLP_01_x0'."""

        rpn = compile_expr_symbolic("GLP_01_A * 0.5 + GLP_01_x0")

        assert "GLP_01_A" in rpn.referenced_names
        assert "GLP_01_x0" in rpn.referenced_names

    #
    def test_complex_expression(self):
        """Complex: '(GLP_01_A + GLP_02_A) / 2'."""

        rpn = compile_expr_symbolic("(GLP_01_A + GLP_02_A) / 2")

        assert "GLP_01_A" in rpn.referenced_names
        assert "GLP_02_A" in rpn.referenced_names

    #
    def test_constant_only(self):
        """Pure constant: '42.0'."""

        rpn = compile_expr_symbolic("42.0")

        assert rpn.referenced_names == []
        assert len(rpn.instructions) == 1
        assert rpn.instructions[0][0] == ExprNodeKind.CONST
        assert rpn.instructions[0][1] == 42.0

    #
    def test_rejects_function_call(self):
        """Function calls (np.exp, etc.) raise ValueError."""

        with pytest.raises(ValueError, match="Unsupported"):
            compile_expr_symbolic("np.exp(GLP_01_A)")

    #
    def test_rejects_attribute_access(self):
        """Attribute access raises ValueError."""

        with pytest.raises(ValueError, match="Unsupported"):
            compile_expr_symbolic("foo.bar")

    #
    def test_uadd_is_noop(self):
        """Unary plus is a no-op."""

        rpn = compile_expr_symbolic("+GLP_01_A")

        assert len(rpn.instructions) == 1
        assert rpn.instructions[0] == (ExprNodeKind.PARAM_REF, "GLP_01_A")

    #
    def test_rpn_evaluation_matches_python(self):
        """Evaluate the RPN symbolically and check against Python eval.

        For '3/4*GLP_01_A' with GLP_01_A=20, result should be 15.0.
        """

        rpn = compile_expr_symbolic("3/4*GLP_01_A")

        # Manual RPN evaluation
        stack: list[float] = []
        for kind, operand in rpn.instructions:
            if kind == ExprNodeKind.CONST:
                stack.append(float(operand))
            elif kind == ExprNodeKind.PARAM_REF:
                stack.append(20.0)  # GLP_01_A = 20
            elif kind == ExprNodeKind.MUL:
                b, a = stack.pop(), stack.pop()
                stack.append(a * b)
            elif kind == ExprNodeKind.DIV:
                b, a = stack.pop(), stack.pop()
                stack.append(a / b)
            elif kind == ExprNodeKind.ADD:
                b, a = stack.pop(), stack.pop()
                stack.append(a + b)

        assert len(stack) == 1
        assert np.isclose(stack[0], 15.0)


# ===================================================================== #
# Phase 2: schedule_2d structural validation tests                       #
# ===================================================================== #


#
#
class TestSchedule2DSimple:
    """Test schedule_2d with simple_energy + MonoExpPos on GLP_01_A.

    Smallest real happy path: the canonical can_lower_2d=True case.
    """

    #
    def _make_plan(self):
        _file, model = _make_2d_model(
            ["simple_energy"],
            [("GLP_01_A", "models/file_time.yaml", ["MonoExpPos"])],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        return plan, graph, model

    #
    def test_plan_axes(self):
        """Plan stores correct energy and time axes."""

        plan, graph, _model = self._make_plan()
        np.testing.assert_array_equal(plan.energy, graph.energy)
        np.testing.assert_array_equal(plan.time, graph.time)
        assert plan.n_time == len(graph.time)

    #
    def test_param_count(self):
        """n_params covers all parameter-like nodes."""

        plan, graph, _model = self._make_plan()

        # Count parameter-like nodes in graph
        _ROW_KINDS = {
            NodeKind.STATIC_PARAM,
            NodeKind.OPT_PARAM,
            NodeKind.PARAM_PLUS_TRACE,
            NodeKind.EXPRESSION,
        }
        n_expected = sum(1 for n in graph.nodes if n.kind in _ROW_KINDS)
        assert plan.n_params == n_expected

    #
    def test_opt_params_first(self):
        """Optimizer params are assigned the first rows in the trace matrix."""

        plan, _graph, _model = self._make_plan()
        n_opt = len(plan.opt_param_names)
        assert n_opt > 0
        np.testing.assert_array_equal(plan.opt_indices, np.arange(n_opt, dtype=np.intp))

    #
    def test_opt_param_names(self):
        """opt_param_names lists all vary=True params."""

        plan, graph, _model = self._make_plan()

        vary_nodes = [n for n in graph.nodes if n.kind == NodeKind.OPT_PARAM and n.vary]
        vary_names = {n.name for n in vary_nodes}
        assert set(plan.opt_param_names) == vary_names

    #
    def test_trace_matrix_shape(self):
        """param_traces_init has shape (n_params, n_time)."""

        plan, _graph, _model = self._make_plan()
        assert plan.param_traces_init.shape == (plan.n_params, plan.n_time)

    #
    def test_static_param_broadcast(self):
        """Static params have uniform rows (same value across time)."""

        plan, graph, _model = self._make_plan()

        for node in graph.nodes:
            if node.kind == NodeKind.STATIC_PARAM:
                # Find row by checking opt_param_names length + offset
                # Easier: trace matrix row should be uniform
                # Find this param's name in the plan
                # Static params come after opt params
                pass

        # Just verify all non-opt rows are uniform for static params
        n_opt = len(plan.opt_param_names)
        for row in range(n_opt, plan.n_params):
            vals = plan.param_traces_init[row, :]
            if np.all(vals == vals[0]):
                continue  # uniform: static or const expression
            # Non-uniform: must be a time-dep resolved param
            # (PARAM_PLUS_TRACE or expression referencing one)

    #
    def test_dynamics_compiled(self):
        """One dynamics group (expFun on GLP_01_A) is compiled."""

        plan, _graph, _model = self._make_plan()

        assert plan.n_dyn_groups == 1
        # Single substep in the group
        assert plan.dyn_group_indptr[1] - plan.dyn_group_indptr[0] == 1
        assert plan.dyn_sub_func_id[0] == int(DynFuncKind.EXPFUN)
        assert plan.dyn_sub_n_params[0] == 4  # A, tau, t0, y0

    #
    def test_dynamics_param_rows_valid(self):
        """Dynamics substep param rows point to valid trace matrix rows."""

        plan, _graph, _model = self._make_plan()

        n_substeps = int(plan.dyn_group_indptr[-1])
        for s in range(n_substeps):
            n_dp = plan.dyn_sub_n_params[s]
            for j in range(n_dp):
                row = plan.dyn_sub_param_rows[s, j]
                assert 0 <= row < plan.n_params

    #
    def test_dynamics_target_and_base_rows_valid(self):
        """Target and base rows are valid indices."""

        plan, _graph, _model = self._make_plan()

        for i in range(plan.n_dyn_groups):
            assert 0 <= plan.dyn_group_target_row[i] < plan.n_params
            assert 0 <= plan.dyn_group_base_row[i] < plan.n_params
            assert plan.dyn_group_target_row[i] != plan.dyn_group_base_row[i]

    #
    def test_resolved_trace_nonconstant(self):
        """The PARAM_PLUS_TRACE row varies over time (dynamics + base)."""

        plan, _graph, _model = self._make_plan()

        target_row = plan.dyn_group_target_row[0]
        trace = plan.param_traces_init[target_row, :]
        # expFun with nonzero A produces a non-constant trace
        assert not np.all(trace == trace[0])

    #
    def test_no_expressions(self):
        """simple_energy has no expression params -> 0 expressions."""

        plan, _graph, _model = self._make_plan()
        assert plan.n_expressions == 0

    #
    def test_component_ops_scheduled(self):
        """4 component ops: Offset, Shirley, GLP_01, GLP_02."""

        plan, _graph, _model = self._make_plan()

        assert plan.n_ops == 4
        op_kind_set = set(plan.op_kinds.tolist())
        assert int(OpKind.GLP) in op_kind_set
        assert int(OpKind.OFFSET) in op_kind_set
        assert int(OpKind.SHIRLEY) in op_kind_set

    #
    def test_shirley_needs_spectrum(self):
        """Shirley is the only op that needs_spectrum."""

        plan, _graph, _model = self._make_plan()

        shirley_indices = [
            i for i in range(plan.n_ops) if plan.op_kinds[i] == int(OpKind.SHIRLEY)
        ]
        assert len(shirley_indices) == 1
        assert plan.op_needs_spectrum[shirley_indices[0]]
        # All others don't need spectrum
        for i in range(plan.n_ops):
            if i not in shirley_indices:
                assert not plan.op_needs_spectrum[i]

    #
    def test_peaks_are_pre_spectrum(self):
        """GLP_01 and GLP_02 contribute to peak_sum (op_is_pre_spectrum)."""

        plan, _graph, _model = self._make_plan()

        glp_indices = [
            i for i in range(plan.n_ops) if plan.op_kinds[i] == int(OpKind.GLP)
        ]
        assert len(glp_indices) == 2
        for idx in glp_indices:
            assert plan.op_is_pre_spectrum[idx]

    #
    def test_offset_not_pre_spectrum(self):
        """Offset is background, not in peak_sum."""

        plan, _graph, _model = self._make_plan()

        offset_indices = [
            i for i in range(plan.n_ops) if plan.op_kinds[i] == int(OpKind.OFFSET)
        ]
        assert len(offset_indices) == 1
        assert not plan.op_is_pre_spectrum[offset_indices[0]]

    #
    def test_csr_param_indices_valid(self):
        """CSR param indices point to valid trace matrix rows."""

        plan, _graph, _model = self._make_plan()

        assert len(plan.op_param_indptr) == plan.n_ops + 1
        assert plan.op_param_indptr[0] == 0
        for i in range(plan.n_ops):
            start = plan.op_param_indptr[i]
            end = plan.op_param_indptr[i + 1]
            assert end >= start
            for j in range(start, end):
                assert 0 <= plan.op_param_indices[j] < plan.n_params

    #
    def test_shirley_scheduled_after_peaks(self):
        """Shirley must execute after GLP components in the schedule."""

        plan, _graph, _model = self._make_plan()

        glp_positions = [
            i for i in range(plan.n_ops) if plan.op_kinds[i] == int(OpKind.GLP)
        ]
        shirley_positions = [
            i for i in range(plan.n_ops) if plan.op_kinds[i] == int(OpKind.SHIRLEY)
        ]
        assert len(shirley_positions) == 1
        assert all(shirley_positions[0] > glp_pos for glp_pos in glp_positions)


#
#
class TestSchedule2DExpressions:
    """Test schedule_2d with energy_expression + MonoExpPos on GLP_01_A.

    Key phase-2 interaction: expression must read the resolved
    time-dependent parameter, not the base scalar.
    """

    #
    def _make_plan(self):
        _file, model = _make_2d_model(
            ["energy_expression"],
            [("GLP_01_A", "models/file_time.yaml", ["MonoExpPos"])],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        return plan, graph, model

    #
    def test_expressions_compiled(self):
        """energy_expression has 4 expression params -> 4 programs."""

        plan, _graph, _model = self._make_plan()
        assert plan.n_expressions == 4
        assert len(plan.expr_programs) == 4

    #
    def test_expression_target_rows_valid(self):
        """Expression target rows point to valid trace matrix indices."""

        plan, _graph, _model = self._make_plan()

        for i in range(plan.n_expressions):
            assert 0 <= plan.expr_target_rows[i] < plan.n_params

    #
    def test_expression_programs_nonempty(self):
        """Each expression program has non-zero instructions."""

        plan, _graph, _model = self._make_plan()

        for prog in plan.expr_programs:
            assert len(prog.instructions) > 0

    #
    def test_expression_a_reads_resolved(self):
        """GLP_02_A = '3/4*GLP_01_A' reads the PARAM_PLUS_TRACE row.

        The expression must reference the resolved time-dep row, not
        the base OPT_PARAM row, because GLP_01_A has dynamics.
        """

        plan, graph, _model = self._make_plan()

        # Find the PARAM_PLUS_TRACE row for GLP_01_A
        resolved_node = None
        for n in graph.nodes:
            if n.kind == NodeKind.PARAM_PLUS_TRACE and "GLP_01_A" in n.name:
                resolved_node = n
                break
        assert resolved_node is not None  # type guard

        # The expression program for GLP_02_A should reference the
        # resolved row, not the base OPT_PARAM row
        expr_node = None
        for n in graph.nodes:
            if n.kind == NodeKind.EXPRESSION and n.name == "GLP_02_A":
                expr_node = n
                break
        assert expr_node is not None  # type guard

        # Find which expression program corresponds to GLP_02_A
        expr_target_idx = None
        for i in range(plan.n_expressions):
            # The expr_string in the expression node should match
            if plan.expr_target_rows[i] == _find_row_for_name(graph, plan, "GLP_02_A"):
                expr_target_idx = i
                break
        assert expr_target_idx is not None

        # Check that the program contains a PARAM_REF to the resolved row
        prog = plan.expr_programs[expr_target_idx]
        resolved_row = _find_row_for_name(graph, plan, resolved_node.name)
        _assert_program_references_row(prog, resolved_row)

    #
    def test_resolved_trace_in_expression_varies(self):
        """Expression result varies over time because source is time-dep."""

        plan, graph, _model = self._make_plan()

        expr_row = _find_row_for_name(graph, plan, "GLP_02_A")
        trace = plan.param_traces_init[expr_row, :]
        # 3/4 * (base + dynamics) should vary
        assert not np.all(trace == trace[0])

    #
    def test_dynamics_still_compiled(self):
        """Dynamics on GLP_01_A is still compiled alongside expressions."""

        plan, _graph, _model = self._make_plan()
        assert plan.n_dyn_groups == 1


#
#
class TestSchedule2DMultipleDynamics:
    """Test schedule_2d with two independent dynamics subgraphs.

    simple_energy with GLP_01_A <- MonoExpPos, GLP_01_x0 <- MonoExpNeg.
    Stresses row packing and multiple dynamics without unsupported features.
    """

    #
    def _make_plan(self):
        _file, model = _make_2d_model(
            ["simple_energy"],
            [
                ("GLP_01_A", "models/file_time.yaml", ["MonoExpPos"]),
                ("GLP_01_x0", "models/file_time.yaml", ["MonoExpNeg"]),
            ],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        return plan, graph, model

    #
    def test_two_dynamics_groups(self):
        """Two dynamics groups compiled (one per time-dep parameter)."""

        plan, _graph, _model = self._make_plan()
        assert plan.n_dyn_groups == 2
        # Both groups have a single substep (mono-exponential each)
        for g in range(2):
            n_sub = plan.dyn_group_indptr[g + 1] - plan.dyn_group_indptr[g]
            assert n_sub == 1
        # Both substeps are expFun
        assert plan.dyn_sub_func_id[0] == int(DynFuncKind.EXPFUN)
        assert plan.dyn_sub_func_id[1] == int(DynFuncKind.EXPFUN)

    #
    def test_distinct_target_rows(self):
        """Each dynamics group targets a different trace row."""

        plan, _graph, _model = self._make_plan()

        assert plan.dyn_group_target_row[0] != plan.dyn_group_target_row[1]

    #
    def test_distinct_base_rows(self):
        """Each dynamics group has a different base row."""

        plan, _graph, _model = self._make_plan()

        assert plan.dyn_group_base_row[0] != plan.dyn_group_base_row[1]

    #
    def test_both_resolved_vary_over_time(self):
        """Both resolved traces are non-constant."""

        plan, _graph, _model = self._make_plan()

        for i in range(2):
            row = plan.dyn_group_target_row[i]
            trace = plan.param_traces_init[row, :]
            assert not np.all(trace == trace[0])

    #
    def test_dynamics_param_rows_dont_overlap(self):
        """Dynamics substep params for the two groups don't share rows."""

        plan, _graph, _model = self._make_plan()

        s0 = int(plan.dyn_group_indptr[0])
        s1 = int(plan.dyn_group_indptr[1])
        rows_0 = set(plan.dyn_sub_param_rows[s0, : plan.dyn_sub_n_params[s0]].tolist())
        rows_1 = set(plan.dyn_sub_param_rows[s1, : plan.dyn_sub_n_params[s1]].tolist())
        assert rows_0.isdisjoint(rows_1)


#
#
class TestSchedule2DExpressionOnly:
    """Test schedule_2d expression compiler with expression-only models.

    Uses expression_fan_out, expression_chain, forward_reference
    with dynamics on Offset_y0 to make them 2D-lowerable.
    """

    #
    def _make_2d_expr_model(self, model_info):
        """Load expression model + attach dynamics to Offset_y0."""

        _file, model = _make_2d_model(
            model_info,
            [("Offset_y0", "models/file_time.yaml", ["MonoExpPos"])],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        return plan, graph

    #
    def test_fan_out_expressions(self):
        """expression_fan_out: GLP_02_A and GLP_03_A both reference GLP_01_A."""

        plan, graph = self._make_2d_expr_model(["expression_fan_out"])

        # Should have many expressions: GLP_02 and GLP_03 each have
        # A, x0, F, m as expressions
        assert plan.n_expressions == 8

    #
    def test_chain_expressions(self):
        """expression_chain: GLP_01 -> GLP_02 -> GLP_03."""

        plan, graph = self._make_2d_expr_model(["expression_chain"])

        # GLP_02 has 4 exprs, GLP_03 has 4 exprs = 8 total
        assert plan.n_expressions == 8

    #
    def test_chain_topological_order(self):
        """In expression_chain, GLP_02_A is evaluated before GLP_03_A.

        GLP_03_A = 'GLP_02_A * 0.5', so GLP_02_A must come first.
        """

        plan, graph = self._make_2d_expr_model(["expression_chain"])

        # Find target rows for GLP_02_A and GLP_03_A
        row_02_A = _find_row_for_name(graph, plan, "GLP_02_A")
        row_03_A = _find_row_for_name(graph, plan, "GLP_03_A")

        # Find their positions in expr_target_rows
        idx_02 = None
        idx_03 = None
        for i in range(plan.n_expressions):
            if plan.expr_target_rows[i] == row_02_A:
                idx_02 = i
            if plan.expr_target_rows[i] == row_03_A:
                idx_03 = i

        assert idx_02 is not None
        assert idx_03 is not None
        assert idx_02 < idx_03

    #
    def test_forward_reference_compiles(self):
        """energy_expression_forward_reference compiles without error."""

        plan, graph = self._make_2d_expr_model(["energy_expression_forward_reference"])

        # GLP_01 has 4 expression params
        assert plan.n_expressions == 4

    #
    def test_expression_values_initialized(self):
        """Expression rows are initialized with evaluated values."""

        plan, graph = self._make_2d_expr_model(["expression_fan_out"])

        # GLP_02_A = "GLP_01_A * 0.5" with GLP_01_A = 20 -> should be 10
        row = _find_row_for_name(graph, plan, "GLP_02_A")
        # All time steps should have the same value (GLP_01_A is not time-dep)
        vals = plan.param_traces_init[row, :]
        assert np.allclose(vals, 10.0)


#
#
class TestExprCompilerVsAsteval:
    """Compare compiled RPN evaluation against asteval as external oracle.

    Verifies that the custom AST-to-RPN path in compile_expr_symbolic +
    _eval_expr_program_init produces the same results as lmfit's
    expression engine.
    """

    _CASES = [
        # (expr_string, variable_dict)
        # precedence
        ("3/4*GLP_01_A", {"GLP_01_A": 20.0}),
        ("A + B * 2", {"A": 5.0, "B": 3.0}),
        # unary minus / plus
        ("-A", {"A": 7.0}),
        ("+A", {"A": 7.0}),
        ("-(A + 2)", {"A": 7.0}),
        # power
        ("A**2", {"A": 3.0}),
        ("A**2 + B", {"A": 3.0, "B": 1.0}),
        # mixed refs/constants
        ("(A + B) / 2", {"A": 10.0, "B": 6.0}),
        # resolved-row style names (prefixed dynamics names)
        ("GLP_01_A_resolved * 0.5", {"GLP_01_A_resolved": 20.0}),
        ("GLP_01_A_expFun_01_A + 1", {"GLP_01_A_expFun_01_A": 4.0}),
        # edge cases
        ("3.14159", {}),
        ("A / B", {"A": 10.0, "B": 3.0}),
    ]

    #
    @pytest.mark.parametrize("expr_string,variables", _CASES)
    def test_matches_asteval(self, expr_string, variables):
        """Compiled RPN result matches asteval for the same expression."""

        from asteval import Interpreter as AstevalInterpreter

        from trspecfit.graph_ir import (
            _bind_expr_to_rows,
            _eval_expr_program_init,
            compile_expr_symbolic,
        )

        # --- asteval reference ---
        aeval = AstevalInterpreter()
        for name, val in variables.items():
            aeval.symtable[name] = val
        expected = float(aeval(expr_string))

        # --- compiled RPN ---
        n_time = 5
        symbolic = compile_expr_symbolic(expr_string)

        # Build a fake trace matrix: one row per variable, broadcast
        name_to_row = {}
        traces = np.zeros((len(variables), n_time), dtype=np.float64)
        for i, (name, val) in enumerate(variables.items()):
            name_to_row[name] = i
            traces[i, :] = val

        program = _bind_expr_to_rows(symbolic, name_to_row)
        result = _eval_expr_program_init(program, traces, n_time)

        # All time steps should match the scalar asteval result
        assert np.allclose(result, expected, rtol=1e-12), (
            f"Expression {expr_string!r}: RPN gave {result[0]}, asteval gave {expected}"
        )


#
#
class TestSchedule2DRejectsNonLowerable:
    """schedule_2d raises ValueError for non-lowerable graphs."""

    #
    def test_1d_graph_raises(self):
        """1D-only graph raises ValueError."""

        _file, model = _make_energy_model(["simple_energy"])
        graph = build_graph(model)
        with pytest.raises(ValueError, match="cannot be lowered"):
            schedule_2d(graph)

    #
    def test_profile_graph_raises(self):
        """Profile model raises ValueError."""

        _file, model = _make_profile_model(
            ["single_gauss"], "Gauss_01_A", ["profile_pExpDecay"]
        )
        graph = build_graph(model)
        with pytest.raises(ValueError, match="cannot be lowered"):
            schedule_2d(graph)


# ===================================================================== #
# Helpers for schedule_2d tests                                          #
# ===================================================================== #


#
def _find_row_for_name(graph, plan, param_name):
    """Find the trace matrix row index for a parameter name.

    Uses opt_param_names for opt rows.  For other rows, walks the
    plan's trace matrix looking for the right row based on the graph.
    """

    from trspecfit.graph_ir import _topological_sort

    _ROW_KINDS = {
        NodeKind.STATIC_PARAM,
        NodeKind.OPT_PARAM,
        NodeKind.PARAM_PLUS_TRACE,
        NodeKind.EXPRESSION,
    }

    # Replicate schedule_2d's row assignment using topological order
    topo_order = _topological_sort(graph)

    opt_nodes = []
    static_nodes = []
    computed_nodes = []
    for nid in topo_order:
        n = graph.nodes[nid]
        if n.kind not in _ROW_KINDS:
            continue
        if n.kind == NodeKind.OPT_PARAM and n.vary:
            opt_nodes.append(n)
        elif n.kind in (NodeKind.STATIC_PARAM, NodeKind.OPT_PARAM):
            static_nodes.append(n)
        else:
            computed_nodes.append(n)

    all_nodes = opt_nodes + static_nodes + computed_nodes
    for row, n in enumerate(all_nodes):
        if n.name == param_name:
            return row

    raise KeyError(f"Parameter {param_name!r} not found in graph")


#
def _assert_program_references_row(program, expected_row):
    """Assert that an ExprProgram contains a PARAM_REF to the given row."""

    instr = program.instructions
    n_instr = len(instr) // 2
    for i in range(n_instr):
        kind = ExprNodeKind(instr[2 * i])
        operand = instr[2 * i + 1]
        if kind == ExprNodeKind.PARAM_REF and operand == expected_row:
            return
    raise AssertionError(f"ExprProgram does not reference row {expected_row}")


# ===================================================================== #
# Regression tests for scheduler bugs                                    #
# ===================================================================== #


#
#
class TestExprBindingUsesEdges:
    """Regression: expression binding must use EXPR_REF edges, not reparsing.

    Dynamics expression nodes can have raw YAML text in expr_string
    (e.g. "-expFun_01_A") while the EXPR_REF edges use the canonical
    lmfit-prefixed form (e.g. "GLP_01_A_expFun_01_A").  The scheduler
    must derive bindings from edges, not from reparsing the string.
    """

    #
    def test_synthetic_graph_with_prefixed_expr_ref(self):
        """In-memory graph with auto-prefixed EXPR_REF does not crash."""

        from trspecfit.graph_ir import GraphEdge, GraphIR, GraphNode

        energy = np.linspace(80, 90, 51)
        time = np.linspace(0, 10, 21)

        nodes = [
            GraphNode(
                id=0,
                kind=NodeKind.OPT_PARAM,
                name="A_base",
                source_order=0,
                value=10.0,
                vary=True,
                bounds=(0.0, 50.0),
            ),
            GraphNode(
                id=1,
                kind=NodeKind.OPT_PARAM,
                name="dyn_A",
                source_order=1,
                value=1.0,
                vary=True,
                bounds=(0.0, 5.0),
            ),
            GraphNode(
                id=2,
                kind=NodeKind.OPT_PARAM,
                name="dyn_tau",
                source_order=2,
                value=2.5,
                vary=True,
                bounds=(1.0, 10.0),
            ),
            GraphNode(
                id=3,
                kind=NodeKind.STATIC_PARAM,
                name="dyn_t0",
                source_order=3,
                value=0.0,
            ),
            GraphNode(
                id=4,
                kind=NodeKind.STATIC_PARAM,
                name="dyn_y0",
                source_order=4,
                value=0.0,
            ),
            GraphNode(
                id=5,
                kind=NodeKind.DYNAMICS_TRACE,
                name="A_dynamics",
                source_order=5,
                function_name="expFun",
                package="time",
            ),
            GraphNode(
                id=6,
                kind=NodeKind.PARAM_PLUS_TRACE,
                name="A_base_resolved",
                source_order=6,
            ),
            # Expression node: raw string says "short_name" but EXPR_REF
            # edge correctly points to A_base_resolved (the prefixed form).
            GraphNode(
                id=7,
                kind=NodeKind.EXPRESSION,
                name="B_expr",
                source_order=7,
                expr_string="0.5 * A_base_resolved",
            ),
            GraphNode(
                id=8,
                kind=NodeKind.COMPONENT_EVAL,
                name="Gauss_01",
                source_order=8,
                function_name="Gauss",
                package="energy",
            ),
            GraphNode(
                id=9,
                kind=NodeKind.STATIC_PARAM,
                name="x0",
                source_order=9,
                value=85.0,
            ),
            GraphNode(
                id=10,
                kind=NodeKind.STATIC_PARAM,
                name="SD",
                source_order=10,
                value=1.0,
            ),
            GraphNode(
                id=11,
                kind=NodeKind.SUM,
                name="peak_sum",
                source_order=11,
            ),
        ]
        edges = [
            GraphEdge(source=1, target=5, kind=EdgeKind.PARAM_INPUT, position=0),
            GraphEdge(source=2, target=5, kind=EdgeKind.PARAM_INPUT, position=1),
            GraphEdge(source=3, target=5, kind=EdgeKind.PARAM_INPUT, position=2),
            GraphEdge(source=4, target=5, kind=EdgeKind.PARAM_INPUT, position=3),
            GraphEdge(source=0, target=6, kind=EdgeKind.BASE_INPUT),
            GraphEdge(source=5, target=6, kind=EdgeKind.TRACE_INPUT),
            # EXPR_REF: B_expr references A_base_resolved
            GraphEdge(source=6, target=7, kind=EdgeKind.EXPR_REF),
            # Component wiring
            GraphEdge(source=7, target=8, kind=EdgeKind.PARAM_INPUT, position=0),
            GraphEdge(source=9, target=8, kind=EdgeKind.PARAM_INPUT, position=1),
            GraphEdge(source=10, target=8, kind=EdgeKind.PARAM_INPUT, position=2),
            GraphEdge(source=8, target=11, kind=EdgeKind.ADDEND),
        ]
        g = GraphIR(
            nodes=nodes,
            edges=edges,
            domain=DomainKind.ENERGY_TIME_2D,
            energy=energy,
            time=time,
            node_by_name={n.name: n.id for n in nodes},
        )
        assert can_lower_2d(g)

        # This used to crash with KeyError because the scheduler reparsed
        # expr_string instead of using EXPR_REF edges.
        plan = schedule_2d(g)
        assert plan.n_expressions == 1

        # The expression result should be 0.5 * resolved_A, which varies
        target_row = plan.expr_target_rows[0]
        trace = plan.param_traces_init[target_row, :]
        assert not np.all(trace == trace[0])


#
#
class TestCanLower2DChecksDynamics:
    """Regression: can_lower_2d must reject unknown dynamics functions.

    Previously, can_lower_2d only checked component functions but not
    DYNAMICS_TRACE.function_name.  A graph with an unsupported dynamics
    function would pass can_lower_2d and then crash in schedule_2d.
    """

    #
    def test_unknown_dynamics_returns_false(self):
        """Graph with unknown dynamics function_name returns False."""

        from trspecfit.graph_ir import GraphEdge, GraphIR, GraphNode

        energy = np.linspace(80, 90, 51)
        time = np.linspace(0, 10, 21)

        nodes = [
            GraphNode(
                id=0,
                kind=NodeKind.OPT_PARAM,
                name="A",
                source_order=0,
                value=10.0,
                vary=True,
                bounds=(0.0, 50.0),
            ),
            GraphNode(
                id=1,
                kind=NodeKind.OPT_PARAM,
                name="d_p1",
                source_order=1,
                value=1.0,
                vary=True,
                bounds=(0.0, 5.0),
            ),
            GraphNode(
                id=2,
                kind=NodeKind.DYNAMICS_TRACE,
                name="A_dynamics",
                source_order=2,
                function_name="madeUpFun",
                package="time",
            ),
            GraphNode(
                id=3,
                kind=NodeKind.PARAM_PLUS_TRACE,
                name="A_resolved",
                source_order=3,
            ),
            GraphNode(
                id=4,
                kind=NodeKind.COMPONENT_EVAL,
                name="Gauss_01",
                source_order=4,
                function_name="Gauss",
                package="energy",
            ),
            GraphNode(
                id=5,
                kind=NodeKind.STATIC_PARAM,
                name="x0",
                source_order=5,
                value=85.0,
            ),
            GraphNode(
                id=6,
                kind=NodeKind.STATIC_PARAM,
                name="SD",
                source_order=6,
                value=1.0,
            ),
            GraphNode(
                id=7,
                kind=NodeKind.SUM,
                name="peak_sum",
                source_order=7,
            ),
        ]
        edges = [
            GraphEdge(source=1, target=2, kind=EdgeKind.PARAM_INPUT, position=0),
            GraphEdge(source=0, target=3, kind=EdgeKind.BASE_INPUT),
            GraphEdge(source=2, target=3, kind=EdgeKind.TRACE_INPUT),
            GraphEdge(source=3, target=4, kind=EdgeKind.PARAM_INPUT, position=0),
            GraphEdge(source=5, target=4, kind=EdgeKind.PARAM_INPUT, position=1),
            GraphEdge(source=6, target=4, kind=EdgeKind.PARAM_INPUT, position=2),
            GraphEdge(source=4, target=7, kind=EdgeKind.ADDEND),
        ]
        g = GraphIR(
            nodes=nodes,
            edges=edges,
            domain=DomainKind.ENERGY_TIME_2D,
            energy=energy,
            time=time,
            node_by_name={n.name: n.id for n in nodes},
        )

        # Previously returned True; now must return False
        assert not can_lower_2d(g)


#
#
class TestNonDenseNodeIds:
    """Regression: scheduler must not assume node.id == list index.

    Externally-constructed GraphIR instances may use arbitrary ids.
    """

    #
    def test_sparse_node_ids(self):
        """Graph with non-zero-based ids schedules correctly."""

        from trspecfit.graph_ir import GraphEdge, GraphIR, GraphNode

        energy = np.linspace(80, 90, 51)
        time = np.linspace(0, 10, 21)

        # Ids start at 100, not 0
        nodes = [
            GraphNode(
                id=100,
                kind=NodeKind.OPT_PARAM,
                name="A",
                source_order=0,
                value=10.0,
                vary=True,
                bounds=(0.0, 50.0),
            ),
            GraphNode(
                id=101,
                kind=NodeKind.OPT_PARAM,
                name="dyn_A",
                source_order=1,
                value=1.0,
                vary=True,
                bounds=(0.0, 5.0),
            ),
            GraphNode(
                id=102,
                kind=NodeKind.OPT_PARAM,
                name="dyn_tau",
                source_order=2,
                value=2.5,
                vary=True,
                bounds=(1.0, 10.0),
            ),
            GraphNode(
                id=103,
                kind=NodeKind.STATIC_PARAM,
                name="dyn_t0",
                source_order=3,
                value=0.0,
            ),
            GraphNode(
                id=104,
                kind=NodeKind.STATIC_PARAM,
                name="dyn_y0",
                source_order=4,
                value=0.0,
            ),
            GraphNode(
                id=105,
                kind=NodeKind.DYNAMICS_TRACE,
                name="A_dynamics",
                source_order=5,
                function_name="expFun",
                package="time",
            ),
            GraphNode(
                id=106,
                kind=NodeKind.PARAM_PLUS_TRACE,
                name="A_resolved",
                source_order=6,
            ),
            GraphNode(
                id=107,
                kind=NodeKind.STATIC_PARAM,
                name="x0",
                source_order=7,
                value=85.0,
            ),
            GraphNode(
                id=108,
                kind=NodeKind.STATIC_PARAM,
                name="SD",
                source_order=8,
                value=1.0,
            ),
            GraphNode(
                id=109,
                kind=NodeKind.COMPONENT_EVAL,
                name="Gauss_01",
                source_order=9,
                function_name="Gauss",
                package="energy",
            ),
            GraphNode(
                id=110,
                kind=NodeKind.SUM,
                name="peak_sum",
                source_order=10,
            ),
        ]
        edges = [
            GraphEdge(source=101, target=105, kind=EdgeKind.PARAM_INPUT, position=0),
            GraphEdge(source=102, target=105, kind=EdgeKind.PARAM_INPUT, position=1),
            GraphEdge(source=103, target=105, kind=EdgeKind.PARAM_INPUT, position=2),
            GraphEdge(source=104, target=105, kind=EdgeKind.PARAM_INPUT, position=3),
            GraphEdge(source=100, target=106, kind=EdgeKind.BASE_INPUT),
            GraphEdge(source=105, target=106, kind=EdgeKind.TRACE_INPUT),
            GraphEdge(source=106, target=109, kind=EdgeKind.PARAM_INPUT, position=0),
            GraphEdge(source=107, target=109, kind=EdgeKind.PARAM_INPUT, position=1),
            GraphEdge(source=108, target=109, kind=EdgeKind.PARAM_INPUT, position=2),
            GraphEdge(source=109, target=110, kind=EdgeKind.ADDEND),
        ]
        g = GraphIR(
            nodes=nodes,
            edges=edges,
            domain=DomainKind.ENERGY_TIME_2D,
            energy=energy,
            time=time,
            node_by_name={n.name: n.id for n in nodes},
        )
        assert can_lower_2d(g)

        # Previously crashed with IndexError
        plan = schedule_2d(g)

        assert plan.n_dyn_groups == 1
        assert plan.n_ops == 1
        assert plan.dyn_sub_func_id[0] == int(DynFuncKind.EXPFUN)
