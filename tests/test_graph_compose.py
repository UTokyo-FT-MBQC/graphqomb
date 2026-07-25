"""Test for compose function."""

from __future__ import annotations

import pytest

from graphqomb.common import Axis, Plane, PlannerMeasBasis
from graphqomb.graphstate import BaseGraphState, GraphState, compose


def create_simple_graph(input_qindices: list[int], output_qindices: list[int]) -> GraphState:
    """Create a simple graph state for testing.

    Parameters
    ----------
    input_qindices : list[int]
        List of qubit indices for input nodes.
    output_qindices : list[int]
        List of qubit indices for output nodes.

    Returns
    -------
    GraphState
        Created graph state with specified input and output qindices.
    """
    graph = GraphState()

    input_nodes: list[int] = []
    for q_index in input_qindices:
        node: int = graph.add_node()
        graph.register_input(node, q_index)
        # All non-output nodes need measurement basis for canonical form
        graph.assign_meas_basis(node, PlannerMeasBasis(Plane.XY, 0.0))
        input_nodes.append(node)

    output_nodes: list[int] = []
    for q_index in output_qindices:
        output_node: int = graph.add_node()
        graph.register_output(output_node, q_index)
        # Output nodes don't need measurement basis
        output_nodes.append(output_node)

    # Add edges between input and output nodes
    for i, input_node in enumerate(input_nodes):
        if i < len(output_nodes):
            graph.add_edge(input_node, output_nodes[i])

    return graph


def test_compose_with_common_qindex() -> None:
    """Test compose function with common qindex between graphs."""
    # Create graph1: input [0, 1] -> output [0, 1]
    graph1: GraphState = create_simple_graph([0, 1], [0, 1])

    # Create graph2: input [1, 2] -> output [1, 2]
    graph2: GraphState = create_simple_graph([1, 2], [1, 2])

    # Compose the graphs
    composed: BaseGraphState
    composed, _, _ = compose(graph1, graph2)

    # Check that qindex 1 is automatically connected
    # graph1 has input [0,1] and output [0,1], graph2 has input [1,2] and output [1,2]
    # Connection: graph1.output[1] -> graph2.input[1]
    # Result: inputs [0,1,2] and outputs [0,1,2]
    expected_input_qindices: set[int] = {0, 1, 2}
    expected_output_qindices: set[int] = {0, 1, 2}

    assert set(composed.input_node_indices.values()) == expected_input_qindices
    assert set(composed.output_node_indices.values()) == expected_output_qindices


def test_compose_no_common_qindex() -> None:
    """Test compose function when graphs have no common qindex."""
    # Create graph1: input [0] -> output [1]
    graph1: GraphState = create_simple_graph([0], [1])

    # Create graph2: input [2] -> output [3]
    graph2: GraphState = create_simple_graph([2], [3])

    # Compose the graphs
    composed: BaseGraphState
    composed, _, _ = compose(graph1, graph2)

    # All qindices should be preserved
    expected_input_qindices: set[int] = {0, 2}
    expected_output_qindices: set[int] = {1, 3}

    assert set(composed.input_node_indices.values()) == expected_input_qindices
    assert set(composed.output_node_indices.values()) == expected_output_qindices


def test_compose_with_non_contiguous_explicit_node_ids() -> None:
    """Test compose remaps sparse source node ids consistently."""
    graph1 = GraphState()
    g1_in = graph1.add_node(10, coordinate=(1.0, 0.0))
    g1_out = graph1.add_node(30, coordinate=(3.0, 0.0))
    graph1.add_edge(g1_in, g1_out)
    graph1.register_input(g1_in, 0)
    graph1.register_output(g1_out, 1)
    graph1.assign_meas_basis(g1_in, PlannerMeasBasis(Plane.XY, 0.0))

    graph2 = GraphState()
    g2_in = graph2.add_node(100, coordinate=(10.0, 0.0))
    g2_mid = graph2.add_node(300, coordinate=(30.0, 0.0))
    g2_out = graph2.add_node(500, coordinate=(50.0, 0.0))
    graph2.add_edge(g2_in, g2_mid)
    graph2.add_edge(g2_mid, g2_out)
    graph2.register_input(g2_in, 1)
    graph2.register_output(g2_out, 2)
    graph2.assign_meas_basis(g2_in, PlannerMeasBasis(Plane.XY, 0.0))
    graph2.assign_meas_basis(g2_mid, PlannerMeasBasis(Plane.XY, 0.0))

    composed, node_map1, node_map2 = compose(graph1, graph2)

    assert node_map1[g1_out] == node_map2[g2_in]
    assert composed.input_node_indices == {node_map1[g1_in]: 0}
    assert composed.output_node_indices == {node_map2[g2_out]: 2}
    assert composed.has_edge(node_map1[g1_in], node_map2[g2_in])
    assert composed.has_edge(node_map2[g2_in], node_map2[g2_mid])
    assert composed.has_edge(node_map2[g2_mid], node_map2[g2_out])
    assert composed.coordinates[node_map1[g1_in]] == (1.0, 0.0)
    assert composed.coordinates[node_map2[g2_in]] == (10.0, 0.0)
    assert composed.coordinates[node_map2[g2_mid]] == (30.0, 0.0)
    assert composed.coordinates[node_map2[g2_out]] == (50.0, 0.0)


def test_compose_qindex_conflict() -> None:
    """Test compose function raises error for qindex conflicts."""
    # Create graph1: input [0] -> output [1]
    graph1: GraphState = create_simple_graph([0], [1])

    # Create graph2: input [2] -> output [0]  # 0 conflicts with graph1's input but no connection
    graph2: GraphState = create_simple_graph([2], [0])

    # Should raise ValueError due to qindex conflict
    with pytest.raises(ValueError, match="Qindex conflicts detected"):
        compose(graph1, graph2)


def test_compose_rejects_connection_through_measured_output() -> None:
    """Test compose function raises error when a connected output is measured."""
    graph1: GraphState = create_simple_graph([0], [1])
    measured_output = next(node for node, q_index in graph1.output_node_indices.items() if q_index == 1)
    graph1.assign_meas_basis(measured_output, PlannerMeasBasis(Plane.XY, 0.0))

    graph2: GraphState = create_simple_graph([1], [2])

    with pytest.raises(ValueError, match="measured output qubit indices"):
        compose(graph1, graph2)


def test_compose_allows_measured_output_outside_connection() -> None:
    """Test compose function keeps measured outputs that are not connected."""
    graph1: GraphState = create_simple_graph([0, 2], [1, 3])
    measured_output = next(node for node, q_index in graph1.output_node_indices.items() if q_index == 3)
    graph1.assign_meas_basis(measured_output, PlannerMeasBasis(Plane.XY, 0.0))

    graph2: GraphState = create_simple_graph([1], [4])

    composed, node_map1, _ = compose(graph1, graph2)

    assert node_map1[measured_output] in composed.meas_bases


def test_compose_preserves_measurement_bases() -> None:
    """Test that measurement bases are preserved during composition."""
    graph1: GraphState = GraphState()
    node1: int = graph1.add_node()
    graph1.register_input(node1, 0)
    # Non-output nodes need measurement basis
    graph1.assign_meas_basis(node1, PlannerMeasBasis(Plane.XY, 0.0))

    node2: int = graph1.add_node()
    graph1.register_output(node2, 1)
    meas_basis: PlannerMeasBasis = PlannerMeasBasis(Plane.XY, 0.5)
    graph1.assign_meas_basis(node2, meas_basis)

    graph2: GraphState = GraphState()
    node3: int = graph2.add_node()
    graph2.register_input(node3, 2)
    # Non-output nodes need measurement basis
    graph2.assign_meas_basis(node3, PlannerMeasBasis(Plane.XY, 0.0))

    node4: int = graph2.add_node()
    graph2.register_output(node4, 3)

    composed: BaseGraphState
    node_map1: dict[int, int]
    composed, node_map1, _ = compose(graph1, graph2)

    # Check that measurement basis is preserved for output node from graph1
    mapped_node2: int = node_map1[node2]
    assert composed.meas_bases[mapped_node2] == meas_basis


def test_compose_preserves_surviving_input_initialization_axes() -> None:
    """Composition preserves initialization axes for inputs that remain inputs."""
    graph1 = GraphState()
    g1_in = graph1.add_node()
    g1_out = graph1.add_node()
    graph1.add_edge(g1_in, g1_out)
    graph1.register_input(g1_in, 0, init_axis=Axis.Y)
    graph1.register_output(g1_out, 1)
    graph1.assign_meas_basis(g1_in, PlannerMeasBasis(Plane.XY, 0.0))

    graph2 = GraphState()
    g2_in_connected = graph2.add_node()
    g2_in_survives = graph2.add_node()
    g2_out = graph2.add_node()
    graph2.add_edge(g2_in_connected, g2_out)
    graph2.add_edge(g2_in_survives, g2_out)
    graph2.register_input(g2_in_connected, 1, init_axis=Axis.Z)
    graph2.register_input(g2_in_survives, 2, init_axis=Axis.Z)
    graph2.register_output(g2_out, 3)
    graph2.assign_meas_basis(g2_in_connected, PlannerMeasBasis(Plane.XY, 0.0))
    graph2.assign_meas_basis(g2_in_survives, PlannerMeasBasis(Plane.XY, 0.0))

    composed, node_map1, node_map2 = compose(graph1, graph2)

    assert composed.input_initialization_axes == {
        node_map1[g1_in]: Axis.Y,
        node_map2[g2_in_survives]: Axis.Z,
    }


def test_compose_full_connection() -> None:
    """Test compose where all outputs of graph1 connect to inputs of graph2."""
    # Create graph1: input [0] -> output [1, 2]
    graph1: GraphState = create_simple_graph([0], [1, 2])

    # Create graph2: input [1, 2] -> output [3]
    graph2: GraphState = create_simple_graph([1, 2], [3])

    # Compose the graphs
    composed: BaseGraphState
    composed, _, _ = compose(graph1, graph2)

    # Should result in: input [0] -> output [3]
    expected_input_qindices: set[int] = {0}
    expected_output_qindices: set[int] = {3}

    assert set(composed.input_node_indices.values()) == expected_input_qindices
    assert set(composed.output_node_indices.values()) == expected_output_qindices


def test_compose_empty_connection() -> None:
    """Test compose when there are no common qindices."""
    # Create graph1: input [0] -> output [1]
    graph1: GraphState = create_simple_graph([0], [1])

    # Create graph2: input [2] -> output [3]
    graph2: GraphState = create_simple_graph([2], [3])

    # Compose the graphs (no connections)
    composed: BaseGraphState
    composed, _, _ = compose(graph1, graph2)

    # All original qindices should be preserved
    expected_input_qindices: set[int] = {0, 2}
    expected_output_qindices: set[int] = {1, 3}

    assert set(composed.input_node_indices.values()) == expected_input_qindices
    assert set(composed.output_node_indices.values()) == expected_output_qindices

    # Check that all nodes and edges are preserved
    total_original_nodes: int = len(graph1.nodes) + len(graph2.nodes)
    # No nodes should be excluded since no connections are made
    assert len(composed.nodes) == total_original_nodes
