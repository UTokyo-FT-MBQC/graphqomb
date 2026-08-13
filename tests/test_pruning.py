from __future__ import annotations

import math

import pytest

from graphqomb.command import M, N
from graphqomb.common import Axis, AxisMeasBasis, Initialization, Plane, PlannerMeasBasis, Sign, default_meas_basis
from graphqomb.graphstate import GraphState
from graphqomb.pruning import prune_isolated_components, prune_z_nodes
from graphqomb.qompiler import qompile


def test_prune_z_measured_node() -> None:
    """A Z-measured non-input node disappears from the graph, flows, checks, and observables."""
    graph = GraphState()
    node_in = graph.add_node()
    node_z = graph.add_node()
    node_out = graph.add_node()
    graph.add_edge(node_in, node_z)
    graph.add_edge(node_z, node_out)
    graph.register_input(node_in, 0)
    graph.register_output(node_out, 0)
    graph.assign_meas_basis(node_in, default_meas_basis())
    graph.assign_meas_basis(node_z, AxisMeasBasis(Axis.Z, Sign.PLUS))

    result = prune_z_nodes(
        graph,
        xflow={node_in: {node_z, node_out}},
        zflow={node_in: set(), node_z: {node_in, node_out}},
        parity_check_group=[{node_in, node_z}, {node_z}],
        parity_check_tags=["keep", "drop"],
        logical_observables={0: {node_in, node_z}},
    )

    assert result.removed_nodes == {node_z}
    assert result.graph.nodes == {node_in, node_out}
    assert result.graph.edges == set()
    assert result.xflow == {node_in: {node_out}}
    assert result.zflow == {node_in: set()}
    assert result.parity_check_group == [{node_in}]
    assert result.parity_check_tags == ["keep"]
    assert result.logical_observables == {0: {node_in}}


def test_prune_z_measured_node_any_representation() -> None:
    """Sign-flipped and planner Z bases are recognized as Z measurements."""
    graph = GraphState()
    node_minus = graph.add_node()
    node_planner = graph.add_node()
    node_out = graph.add_node()
    graph.add_edge(node_minus, node_out)
    graph.add_edge(node_planner, node_out)
    graph.register_output(node_out, 0)
    graph.assign_meas_basis(node_minus, AxisMeasBasis(Axis.Z, Sign.MINUS))
    graph.assign_meas_basis(node_planner, PlannerMeasBasis(Plane.XZ, math.pi))

    result = prune_z_nodes(graph, xflow={}, zflow={node_minus: {node_out}, node_planner: {node_out}})

    assert result.removed_nodes == {node_minus, node_planner}
    assert result.graph.nodes == {node_out}


def test_prune_z_prep_input() -> None:
    """A Z-prepared input node is pruned regardless of its measurement basis."""
    graph = GraphState()
    node_zprep = graph.add_node()
    node_in = graph.add_node()
    node_out = graph.add_node()
    graph.add_edge(node_zprep, node_in)
    graph.add_edge(node_in, node_out)
    graph.register_input(node_zprep, 0, init=Initialization(axis=Axis.Z))
    graph.register_input(node_in, 1)
    graph.register_output(node_out, 1)
    graph.assign_meas_basis(node_zprep, default_meas_basis())
    graph.assign_meas_basis(node_in, default_meas_basis())

    result = prune_z_nodes(graph, xflow={node_in: {node_out}})

    assert result.removed_nodes == {node_zprep}
    assert result.graph.nodes == {node_in, node_out}
    assert result.graph.input_node_indices == {node_in: 1}
    assert result.graph.edges == {(node_in, node_out)}


def test_uncorrected_z_measurement_is_kept() -> None:
    """A Z-measured node whose byproduct is not corrected by the flows stays."""
    # Without a Z correction on node_out the original output is |+> or |->
    # depending on the measurement outcome; pruning would fix one branch.
    graph = GraphState()
    node_z = graph.add_node()
    node_out = graph.add_node()
    graph.add_edge(node_z, node_out)
    graph.register_output(node_out, 0)
    graph.assign_meas_basis(node_z, AxisMeasBasis(Axis.Z, Sign.PLUS))

    result = prune_z_nodes(graph, xflow={})

    assert result.removed_nodes == frozenset()
    assert result.graph.nodes == {node_z, node_out}


def test_z_neighbors_need_no_correction() -> None:
    """Z byproducts landing on other Z-basis nodes are vacuous, so a Z cluster prunes without flow."""
    graph = GraphState()
    node_za = graph.add_node()
    node_zb = graph.add_node()
    node_out = graph.add_node()
    graph.add_edge(node_za, node_zb)
    graph.register_output(node_out, 0)
    graph.assign_meas_basis(node_za, AxisMeasBasis(Axis.Z, Sign.PLUS))
    graph.assign_meas_basis(node_zb, AxisMeasBasis(Axis.Z, Sign.PLUS))

    result = prune_z_nodes(graph, xflow={})

    assert result.removed_nodes == {node_za, node_zb}
    assert result.graph.nodes == {node_out}


def test_stabilizer_equivalent_correction_is_recognized() -> None:
    """A byproduct correction expressed through an X flow entry qualifies as well."""
    # X^m on node_w times the byproduct Z^m on node_v is the graph stabilizer
    # X_w Z_v of the chain z - v - w, so the correction cancels the byproduct.
    graph = GraphState()
    node_z = graph.add_node()
    node_v = graph.add_node()
    node_w = graph.add_node()
    graph.add_edge(node_z, node_v)
    graph.add_edge(node_v, node_w)
    graph.register_output(node_w, 0)
    graph.assign_meas_basis(node_z, AxisMeasBasis(Axis.Z, Sign.PLUS))
    graph.assign_meas_basis(node_v, default_meas_basis())

    result = prune_z_nodes(graph, xflow={node_z: {node_w}, node_v: {node_w}}, zflow={node_z: set(), node_v: set()})

    assert result.removed_nodes == {node_z}
    assert result.graph.nodes == {node_v, node_w}
    assert result.xflow == {node_v: {node_w}}


def test_y_initialized_correction_target_is_checked() -> None:
    """X corrections onto a Y-initialized node follow its Y-type stabilizer generator."""
    graph = GraphState()
    node_z = graph.add_node()
    node_y = graph.add_node()
    graph.add_edge(node_z, node_y)
    graph.register_input(node_y, 0, init=Initialization(axis=Axis.Y))
    graph.register_output(node_y, 0)
    graph.assign_meas_basis(node_z, AxisMeasBasis(Axis.Z, Sign.PLUS))

    # X^m Z^m on node_y leaves a net X^m after the byproduct Z^m, which maps
    # the Y eigenstate to the orthogonal one; the Y-type generator would need
    # an extra Z on node_y, so node_z must be kept.
    kept = prune_z_nodes(graph, xflow={node_z: {node_y}}, zflow={node_z: {node_y}})
    assert kept.removed_nodes == frozenset()

    # A plain Z correction cancels the byproduct for any initialization.
    pruned = prune_z_nodes(graph, xflow={}, zflow={node_z: {node_y}})
    assert pruned.removed_nodes == {node_z}


def test_z_initialized_output_neighbor() -> None:
    """Z byproducts on a Z-initialized output are vacuous, but X corrections onto it never qualify."""
    graph = GraphState()
    node_z = graph.add_node()
    node_w = graph.add_node()
    graph.add_edge(node_z, node_w)
    graph.register_input(node_w, 0, init=Initialization(axis=Axis.Z))
    graph.register_output(node_w, 0)
    graph.assign_meas_basis(node_z, AxisMeasBasis(Axis.Z, Sign.PLUS))

    pruned = prune_z_nodes(graph, xflow={})
    assert pruned.removed_nodes == {node_z}

    kept = prune_z_nodes(graph, xflow={node_z: {node_w}})
    assert kept.removed_nodes == frozenset()


def test_x_and_y_initialized_inputs_are_kept() -> None:
    """Non-Z initializations do not qualify a non-Z-measured input for pruning."""
    graph = GraphState()
    node_x = graph.add_node()
    node_y = graph.add_node()
    node_out = graph.add_node()
    graph.add_edge(node_x, node_out)
    graph.add_edge(node_y, node_out)
    graph.register_input(node_x, 0, init=Initialization(axis=Axis.X))
    graph.register_input(node_y, 1, init=Initialization(axis=Axis.Y))
    graph.register_output(node_out, 0)
    graph.assign_meas_basis(node_x, default_meas_basis())
    graph.assign_meas_basis(node_y, default_meas_basis())

    result = prune_z_nodes(graph, xflow={})

    assert result.removed_nodes == frozenset()
    assert result.graph.nodes == {node_x, node_y, node_out}


def test_output_nodes_are_kept() -> None:
    """Output nodes stay even when they are Z-prepared inputs."""
    graph = GraphState()
    node = graph.add_node()
    graph.register_input(node, 0, init=Initialization(axis=Axis.Z))
    graph.register_output(node, 0)

    result = prune_z_nodes(graph, xflow={})

    assert result.removed_nodes == frozenset()
    assert result.graph.nodes == {node}
    assert result.graph.output_node_indices == {node: 0}


def test_node_attributes_survive_pruning() -> None:
    """Kept nodes preserve indices, coordinates, initializations, and bases."""
    graph = GraphState()
    node_in = graph.add_node(coordinate=(0.0, 0.0))
    node_z = graph.add_node(coordinate=(1.0, 0.0))
    node_out = graph.add_node(coordinate=(2.0, 0.0))
    graph.add_edge(node_in, node_z)
    graph.add_edge(node_in, node_out)
    graph.register_input(node_in, 3, init=Initialization(axis=Axis.X, tag="tagged"))
    graph.register_output(node_out, 3)
    graph.assign_meas_basis(node_in, PlannerMeasBasis(Plane.XY, 0.25))
    graph.assign_meas_basis(node_z, AxisMeasBasis(Axis.Z, Sign.PLUS))

    result = prune_z_nodes(graph, xflow={node_in: {node_out}}, zflow={node_in: set(), node_z: {node_in}})

    assert result.graph.input_node_indices == {node_in: 3}
    assert result.graph.input_initializations[node_in] == Initialization(axis=Axis.X, tag="tagged")
    assert result.graph.coordinates == {node_in: (0.0, 0.0), node_out: (2.0, 0.0)}
    assert math.isclose(result.graph.meas_bases[node_in].angle, 0.25)


def test_zflow_derived_from_source_graph() -> None:
    """When zflow is omitted it is derived by odd neighbors before pruning."""
    graph = GraphState()
    node_in = graph.add_node()
    node_z = graph.add_node()
    node_out = graph.add_node()
    graph.add_edge(node_in, node_out)
    graph.add_edge(node_z, node_out)
    graph.register_input(node_in, 0)
    graph.register_output(node_out, 0)
    graph.assign_meas_basis(node_in, default_meas_basis())
    graph.assign_meas_basis(node_z, AxisMeasBasis(Axis.Z, Sign.PLUS))

    # The self-referencing xflow entry gives node_z the derived zflow
    # odd_neighbors({node_z}) = {node_out}, which corrects its byproduct.
    result = prune_z_nodes(graph, xflow={node_in: {node_out}, node_z: {node_z}})

    # odd_neighbors({node_out}) = {node_in, node_z}; node_z is pruned afterwards
    assert result.zflow == {node_in: {node_in}}


def test_misaligned_parity_check_tags_raise() -> None:
    graph = GraphState()
    node = graph.add_node()
    graph.register_output(node, 0)

    with pytest.raises(ValueError, match="parity_check_tags"):
        prune_z_nodes(graph, xflow={}, parity_check_group=[{node}], parity_check_tags=["a", "b"])


def test_prune_isolated_component_without_logical_or_output() -> None:
    """A component with neither outputs nor observable seeds disappears with its checks and flows."""
    graph = GraphState()
    node_in = graph.add_node()
    node_out = graph.add_node()
    iso_a = graph.add_node()
    iso_b = graph.add_node()
    graph.add_edge(node_in, node_out)
    graph.add_edge(iso_a, iso_b)
    graph.register_input(node_in, 0)
    graph.register_output(node_out, 0)
    graph.assign_meas_basis(node_in, default_meas_basis())
    graph.assign_meas_basis(iso_a, default_meas_basis())
    graph.assign_meas_basis(iso_b, default_meas_basis())

    result = prune_isolated_components(
        graph,
        xflow={node_in: {node_out}, iso_a: {iso_b}},
        zflow={node_in: set(), iso_a: set()},
        parity_check_group=[{iso_a, iso_b}, {node_in, iso_a}, {node_in}],
        parity_check_tags=["dropped", "mixed", "kept"],
        logical_observables={0: {node_in}},
    )

    assert result.removed_nodes == {iso_a, iso_b}
    assert result.graph.nodes == {node_in, node_out}
    assert result.xflow == {node_in: {node_out}}
    assert result.parity_check_group == [{node_in}, {node_in}]
    assert result.parity_check_tags == ["mixed", "kept"]
    assert result.logical_observables == {0: {node_in}}


def test_components_with_observable_seed_or_output_are_kept() -> None:
    """Observable seeds and output nodes both make a component relevant."""
    graph = GraphState()
    seed_component = graph.add_node()
    output_component = graph.add_node()
    graph.register_output(output_component, 0)
    graph.assign_meas_basis(seed_component, default_meas_basis())

    result = prune_isolated_components(graph, xflow={}, logical_observables={0: {seed_component}})

    assert result.removed_nodes == frozenset()
    assert result.graph.nodes == {seed_component, output_component}


def test_flow_coupled_component_is_kept() -> None:
    """A component feeding a correction into a relevant component is classically coupled and stays."""
    graph = GraphState()
    node_out = graph.add_node()
    node_ctrl = graph.add_node()
    iso_a = graph.add_node()
    iso_b = graph.add_node()
    graph.add_edge(iso_a, iso_b)
    graph.register_output(node_out, 0)
    graph.assign_meas_basis(node_ctrl, default_meas_basis())
    graph.assign_meas_basis(iso_a, default_meas_basis())
    graph.assign_meas_basis(iso_b, default_meas_basis())

    result = prune_isolated_components(graph, xflow={node_ctrl: {node_out}, iso_a: {iso_b}})

    assert result.removed_nodes == {iso_a, iso_b}
    assert result.graph.nodes == {node_out, node_ctrl}
    assert result.xflow == {node_ctrl: {node_out}}


def test_flow_target_component_is_kept() -> None:
    """A component receiving a correction from a relevant component stays as well."""
    graph = GraphState()
    node_in = graph.add_node()
    node_out = graph.add_node()
    stray = graph.add_node()
    graph.add_edge(node_in, node_out)
    graph.register_input(node_in, 0)
    graph.register_output(node_out, 0)
    graph.assign_meas_basis(node_in, default_meas_basis())
    graph.assign_meas_basis(stray, default_meas_basis())

    result = prune_isolated_components(graph, xflow={node_in: {node_out, stray}})

    assert result.removed_nodes == frozenset()
    assert result.graph.nodes == {node_in, node_out, stray}


def test_zflow_coupling_counts_for_relevance() -> None:
    """An explicit zflow entry into a kept component keeps the source component too."""
    graph = GraphState()
    node_out = graph.add_node()
    node_ctrl = graph.add_node()
    graph.register_output(node_out, 0)
    graph.assign_meas_basis(node_ctrl, default_meas_basis())

    result = prune_isolated_components(graph, xflow={}, zflow={node_ctrl: {node_out}})

    assert result.removed_nodes == frozenset()
    assert result.graph.nodes == {node_out, node_ctrl}


def test_prune_everything_without_outputs_and_observables() -> None:
    """With no outputs and no observables, every component is irrelevant by definition."""
    graph = GraphState()
    node_a = graph.add_node()
    node_b = graph.add_node()
    graph.add_edge(node_a, node_b)
    graph.assign_meas_basis(node_a, default_meas_basis())
    graph.assign_meas_basis(node_b, default_meas_basis())

    result = prune_isolated_components(graph, xflow={node_a: {node_b}})

    assert result.removed_nodes == {node_a, node_b}
    assert result.graph.nodes == set()


def test_z_prune_then_isolated_prune_compiles() -> None:
    """Z-pruning may disconnect the graph; chaining both prunes feeds qompile."""
    # node_z bridges the logical wire (node_in - node_out) and a hanging pair
    # (iso_a - iso_b); Z-pruning node_z leaves the pair isolated.
    graph = GraphState()
    node_in = graph.add_node()
    node_out = graph.add_node()
    node_z = graph.add_node()
    iso_a = graph.add_node()
    iso_b = graph.add_node()
    graph.add_edge(node_in, node_out)
    graph.add_edge(node_out, node_z)
    graph.add_edge(node_z, iso_a)
    graph.add_edge(iso_a, iso_b)
    graph.register_input(node_in, 0)
    graph.register_output(node_out, 0)
    graph.assign_meas_basis(node_in, default_meas_basis())
    graph.assign_meas_basis(node_z, AxisMeasBasis(Axis.Z, Sign.PLUS))
    graph.assign_meas_basis(iso_a, default_meas_basis())
    graph.assign_meas_basis(iso_b, default_meas_basis())

    z_pruned = prune_z_nodes(
        graph,
        xflow={node_in: {node_out}, iso_a: {iso_b}},
        zflow={node_in: set(), iso_a: set(), node_z: {node_out, iso_a}},
        parity_check_group=[{node_z, iso_a, iso_b}],
        logical_observables={0: {node_in}},
    )
    assert z_pruned.removed_nodes == {node_z}

    result = prune_isolated_components(
        z_pruned.graph,
        z_pruned.xflow,
        z_pruned.zflow,
        parity_check_group=z_pruned.parity_check_group,
        parity_check_tags=z_pruned.parity_check_tags,
        logical_observables=z_pruned.logical_observables,
    )

    assert result.removed_nodes == {iso_a, iso_b}
    assert result.graph.nodes == {node_in, node_out}
    assert result.parity_check_group == []
    assert result.parity_check_tags == []

    pattern = qompile(
        result.graph,
        result.xflow,
        result.zflow,
        parity_check_group=result.parity_check_group,
        parity_check_tags=result.parity_check_tags,
        logical_observables=result.logical_observables,
    )
    prepared_or_measured = {cmd.node for cmd in pattern.commands if isinstance(cmd, (N, M))}
    assert prepared_or_measured.isdisjoint({node_z, iso_a, iso_b})


def test_pruned_result_compiles() -> None:
    """The pruned pieces feed directly into qompile."""
    graph = GraphState()
    node_in = graph.add_node()
    node_z = graph.add_node()
    node_out = graph.add_node()
    graph.add_edge(node_in, node_out)
    graph.add_edge(node_z, node_out)
    graph.register_input(node_in, 0)
    graph.register_output(node_out, 0)
    graph.assign_meas_basis(node_in, default_meas_basis())
    graph.assign_meas_basis(node_z, AxisMeasBasis(Axis.Z, Sign.PLUS))

    result = prune_z_nodes(
        graph,
        xflow={node_in: {node_out}},
        zflow={node_in: set(), node_z: {node_out}},
        parity_check_group=[{node_z}],
        logical_observables={0: {node_in, node_z}},
    )

    pattern = qompile(
        result.graph,
        result.xflow,
        result.zflow,
        parity_check_group=result.parity_check_group,
        parity_check_tags=result.parity_check_tags,
        logical_observables=result.logical_observables,
    )

    prepared_or_measured = {cmd.node for cmd in pattern.commands if isinstance(cmd, (N, M))}
    assert node_z not in prepared_or_measured
