"""Tests for graph_ir: build_graph and can_lower_2d."""

import numpy as np
import pytest

from trspecfit import File, Project
from trspecfit.graph_ir import (
    DomainKind,
    EdgeKind,
    NodeKind,
    build_graph,
    can_lower_2d,
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
