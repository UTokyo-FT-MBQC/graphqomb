"""Tests for the GraphState class."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from graphqomb.common import Axis, Plane, PlannerMeasBasis
from graphqomb.euler import LocalClifford
from graphqomb.graphstate import GraphState, bipartite_edges, compose, odd_neighbors


@pytest.fixture
def graph() -> GraphState:
    """Generate an empty GraphState object.

    Returns
    -------
    GraphState: An empty GraphState object.
    """
    return GraphState()


@pytest.fixture
def canonical_graph() -> GraphState:
    graph = GraphState()
    in_node = graph.add_node()
    out_node = graph.add_node()

    q_idx = 0
    graph.register_input(in_node, q_idx)
    graph.register_output(out_node, q_idx)
    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.5 * np.pi))
    return graph


def test_add_node(graph: GraphState) -> None:
    """Test adding a node to the graph."""
    node_index = graph.add_node()
    assert node_index in graph.nodes
    assert len(graph.nodes) == 1


def test_standard_graph_api_adds_nodes_and_edges(graph: GraphState) -> None:
    """Test the standard graph-style node and edge API."""
    node0 = graph.add_node(coordinate=(1.0, 2.0))
    node10 = graph.add_node(10)
    node11 = graph.add_node()

    assert node0 == 0
    assert node10 == 10
    assert node11 == 11
    assert graph.nodes == {0, 10, 11}
    assert graph.number_of_nodes() == 3
    assert graph.has_node(10)
    assert not graph.has_node(9)
    assert graph.coordinates == {0: (1.0, 2.0)}

    graph.add_edge(node0, node10)
    graph.add_edge(node10, node11)

    assert graph.edges == {(0, 10), (10, 11)}
    assert graph.number_of_edges() == 2
    assert graph.has_edge(node10, node0)
    assert not graph.has_edge(node0, node11)


def test_standard_graph_api_rejects_duplicate_node(graph: GraphState) -> None:
    """Test that add_node rejects an existing explicit node id."""
    graph.add_node(5)
    with pytest.raises(ValueError, match="Node already exists node=5"):
        graph.add_node(5)


def test_standard_graph_api_remove_node_cleans_metadata(graph: GraphState) -> None:
    """Test remove_node removes incident edges and node metadata."""
    node0 = graph.add_node(coordinate=(1.0, 2.0))
    node1 = graph.add_node(coordinate=(3.0, 4.0))
    graph.add_edge(node0, node1)
    graph.register_output(node1, 0)
    graph.assign_meas_basis(node0, PlannerMeasBasis(Plane.XY, 0.0))

    graph.remove_node(node0)

    assert graph.nodes == {node1}
    assert graph.edges == set()
    assert node0 not in graph.meas_bases
    assert graph.coordinates == {node1: (3.0, 4.0)}


def test_add_node_input_output(graph: GraphState) -> None:
    """Test adding a node as input and output."""
    node_index = graph.add_node()
    q_index = 0
    graph.register_input(node_index, q_index)
    graph.register_output(node_index, q_index)
    assert node_index in graph.input_node_indices
    assert node_index in graph.output_node_indices
    assert graph.input_node_indices[node_index] == q_index
    assert graph.output_node_indices[node_index] == q_index


def test_register_input_defaults_to_x_initialization_axis(graph: GraphState) -> None:
    """Input nodes default to positive X-basis initialization."""
    node_index = graph.add_node()

    graph.register_input(node_index, 0)

    assert graph.input_initialization_axes == {node_index: Axis.X}


def test_register_input_accepts_pauli_initialization_axis(graph: GraphState) -> None:
    """Input nodes can be initialized in a positive Pauli eigenstate."""
    node_index = graph.add_node()

    graph.register_input(node_index, 0, init_axis=Axis.Y)

    assert graph.input_initialization_axes == {node_index: Axis.Y}


def test_register_input_rejects_non_axis_initialization(graph: GraphState) -> None:
    """Input registration rejects values outside the Axis enum."""
    node_index = graph.add_node()
    invalid_axis: Any = "X"

    with pytest.raises(TypeError, match="Input initialization axis must be one of"):
        graph.register_input(node_index, 0, init_axis=invalid_axis)


def test_register_input_defaults_to_untagged_initialization(graph: GraphState) -> None:
    """Input nodes default to an untagged initialization."""
    node_index = graph.add_node()

    graph.register_input(node_index, 0)

    assert graph.input_initialization_tags == {node_index: ""}


def test_register_input_accepts_initialization_tag(graph: GraphState) -> None:
    """Input nodes can carry a Stim-style initialization tag."""
    node_index = graph.add_node()

    graph.register_input(node_index, 0, init_tag="init_data")

    assert graph.input_initialization_tags == {node_index: "init_data"}


def test_register_input_rejects_non_str_initialization_tag(graph: GraphState) -> None:
    """Input registration rejects non-string initialization tags."""
    node_index = graph.add_node()
    invalid_tag: Any = 7

    with pytest.raises(TypeError, match="Input initialization tag must be a str"):
        graph.register_input(node_index, 0, init_tag=invalid_tag)


def test_ensure_node_exists_raises(graph: GraphState) -> None:
    """Test ensuring a node exists in the graph."""
    with pytest.raises(ValueError, match="Node does not exist node=1"):
        graph._ensure_node_exists(1)


def test_ensure_node_exists(graph: GraphState) -> None:
    """Test ensuring a node exists in the graph."""
    node_index = graph.add_node()
    graph._ensure_node_exists(node_index)


def test_neighbors(graph: GraphState) -> None:
    """Test getting the neighbors of a node in the graph."""
    node_index1 = graph.add_node()
    node_index2 = graph.add_node()
    node_index3 = graph.add_node()
    graph.add_edge(node_index1, node_index2)
    graph.add_edge(node_index2, node_index3)
    assert graph.neighbors(node_index1) == {node_index2}
    assert graph.neighbors(node_index2) == {node_index1, node_index3}
    assert graph.neighbors(node_index3) == {node_index2}


def test_add_edge(graph: GraphState) -> None:
    """Test adding an edge to the graph."""
    node_index1 = graph.add_node()
    node_index2 = graph.add_node()
    graph.add_edge(node_index1, node_index2)
    assert (node_index1, node_index2) in graph.edges or (node_index2, node_index1) in graph.edges
    assert len(graph.edges) == 1


def test_add_duplicate_edge(graph: GraphState) -> None:
    """Test adding a duplicate edge to the graph."""
    node_index1 = graph.add_node()
    node_index2 = graph.add_node()
    graph.add_edge(node_index1, node_index2)
    with pytest.raises(ValueError, match=f"Edge already exists node1={node_index1}, node2={node_index2}"):
        graph.add_edge(node_index1, node_index2)


def test_add_edge_rejects_self_loop(graph: GraphState) -> None:
    """Test that adding a self-loop edge raises an error."""
    node_index = graph.add_node()
    with pytest.raises(ValueError, match="Self-loops are not allowed"):
        graph.add_edge(node_index, node_index)


def test_add_edge_with_nonexistent_node(graph: GraphState) -> None:
    """Test adding an edge with a nonexistent node to the graph."""
    node_index1 = graph.add_node()
    with pytest.raises(ValueError, match="Node does not exist node=2"):
        graph.add_edge(node_index1, 2)


def test_remove_node_with_nonexistent_node(graph: GraphState) -> None:
    """Test removing a nonexistent node from the graph."""
    with pytest.raises(ValueError, match="Node does not exist node=1"):
        graph.remove_node(1)


def test_remove_node_with_input_removal(graph: GraphState) -> None:
    """Test removing an input node from the graph"""
    node_index = graph.add_node()
    graph.register_input(node_index, 0)
    with pytest.raises(ValueError, match="The input node cannot be removed"):
        graph.remove_node(node_index)


def test_remove_node(graph: GraphState) -> None:
    """Test removing a node from the graph."""
    node_index = graph.add_node()
    graph.remove_node(node_index)
    assert node_index not in graph.nodes
    assert len(graph.nodes) == 0


def test_remove_node_from_minimal_graph(graph: GraphState) -> None:
    """Test removing a node from the graph with edges."""
    node_index1 = graph.add_node()
    node_index2 = graph.add_node()
    graph.add_edge(node_index1, node_index2)
    graph.remove_node(node_index1)
    assert node_index1 not in graph.nodes
    assert node_index2 in graph.nodes
    assert len(graph.nodes) == 1
    assert len(graph.edges) == 0


def test_remove_node_from_3_nodes_graph(graph: GraphState) -> None:
    """Test removing a node from the graph with 3 nodes and edges."""
    node_index1 = graph.add_node()
    node_index2 = graph.add_node()
    node_index3 = graph.add_node()
    graph.add_edge(node_index1, node_index2)
    graph.add_edge(node_index2, node_index3)
    q_index = 0
    graph.register_input(node_index1, q_index)
    graph.register_output(node_index3, q_index)
    graph.remove_node(node_index2)
    assert graph.nodes == {node_index1, node_index3}
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 0
    assert graph.input_node_indices == {node_index1: q_index}
    assert graph.output_node_indices == {node_index3: q_index}


def test_remove_edge_with_nonexistent_nodes(graph: GraphState) -> None:
    """Test removing an edge with nonexistent nodes from the graph."""
    with pytest.raises(ValueError, match="Node does not exist"):
        graph.remove_edge(1, 2)


def test_remove_edge_with_nonexistent_edge(graph: GraphState) -> None:
    """Test removing a nonexistent edge from the graph."""
    node_index1 = graph.add_node()
    node_index2 = graph.add_node()
    with pytest.raises(ValueError, match="Edge does not exist"):
        graph.remove_edge(node_index1, node_index2)


def test_remove_edge(graph: GraphState) -> None:
    """Test removing an edge from the graph."""
    node_index1 = graph.add_node()
    node_index2 = graph.add_node()
    graph.add_edge(node_index1, node_index2)
    graph.remove_edge(node_index1, node_index2)
    assert (node_index1, node_index2) not in graph.edges
    assert (node_index2, node_index1) not in graph.edges
    assert len(graph.edges) == 0


def test_register_output_raises_1(graph: GraphState) -> None:
    with pytest.raises(ValueError, match="Node does not exist node=1"):
        graph.register_output(1, 0)


def test_assign_meas_basis(graph: GraphState) -> None:
    """Test setting the measurement basis of a node."""
    node_index = graph.add_node()
    meas_basis = PlannerMeasBasis(Plane.XZ, 0.5 * math.pi)
    graph.assign_meas_basis(node_index, meas_basis)
    assert graph.meas_bases[node_index].plane == Plane.XZ
    assert math.isclose(graph.meas_bases[node_index].angle, 0.5 * math.pi)


def test_check_canonical_form_true(canonical_graph: GraphState) -> None:
    """Test if the graph is in canonical form."""
    canonical_graph.check_canonical_form()


def test_check_canonical_form_input_output_mismatch(canonical_graph: GraphState) -> None:
    """Test if the graph is in canonical form with input-output mismatch."""
    node_index = canonical_graph.add_node()
    canonical_graph.register_input(node_index, 1)
    canonical_graph.assign_meas_basis(node_index, PlannerMeasBasis(Plane.XY, 0.5 * np.pi))
    # The current implementation does not check input-output mismatch, so the test should pass
    canonical_graph.check_canonical_form()


def test_check_canonical_form_with_local_clifford_false(canonical_graph: GraphState) -> None:
    """Test if the graph is in canonical form with local Clifford operator."""
    local_clifford = LocalClifford()
    in_node = next(iter(canonical_graph.input_node_indices))
    canonical_graph.apply_local_clifford(in_node, local_clifford)
    with pytest.raises(ValueError, match="Clifford operators are applied"):
        canonical_graph.check_canonical_form()


def test_check_canonical_form_with_local_clifford_expansion_true(canonical_graph: GraphState) -> None:
    """Test if the graph is in canonical form with local Clifford operator expansion."""
    local_clifford = LocalClifford()
    in_node = next(iter(canonical_graph.input_node_indices))
    canonical_graph.apply_local_clifford(in_node, local_clifford)
    canonical_graph.expand_local_cliffords()
    canonical_graph.check_canonical_form()  # Should not raise an exception


def test_check_canonical_form_missing_meas_basis_false(canonical_graph: GraphState) -> None:
    """Test if the graph is in canonical form with missing measurement basis."""
    _ = canonical_graph.add_node()
    with pytest.raises(ValueError, match="All non-output nodes must have measurement basis"):
        canonical_graph.check_canonical_form()


def test_check_canonical_form_empty_graph_is_true() -> None:
    """Test if an empty graph is in canonical form."""
    graph = GraphState()
    graph.check_canonical_form()  # Should not raise an exception


def test_check_meas_raises_value_error(graph: GraphState) -> None:
    """Test if measurement planes and angles are set improperly."""
    node_index = graph.add_node()
    with pytest.raises(ValueError, match=f"Measurement basis not set for node {node_index}"):
        graph._check_meas_basis()


def test_check_meas_basis_success(graph: GraphState) -> None:
    """Test if measurement planes and angles are set properly."""
    graph._check_meas_basis()
    node_index1 = graph.add_node()
    q_index = 0
    graph.register_input(node_index1, q_index)
    meas_basis = PlannerMeasBasis(Plane.XY, 0.5 * np.pi)
    graph.assign_meas_basis(node_index1, meas_basis)
    graph._check_meas_basis()

    node_index2 = graph.add_node()
    graph.add_edge(node_index1, node_index2)
    graph.register_output(node_index2, q_index)
    graph._check_meas_basis()


def test_bipartite_edges() -> None:
    """Test the function that generate complete bipartite edges"""
    assert bipartite_edges(set(), set()) == set()
    assert bipartite_edges({1, 2}, {3, 4}) == {(1, 3), (1, 4), (2, 3), (2, 4)}


def test_odd_neighbors(graph: GraphState) -> None:
    r"""Test the function that returns odd neighbors of a node.

    node1 --- node2
        \     /
         \   /
         node3
    """
    node1 = graph.add_node()
    node2 = graph.add_node()
    node3 = graph.add_node()

    graph.add_edge(node1, node2)
    graph.add_edge(node2, node3)
    graph.add_edge(node3, node1)

    assert odd_neighbors({node1}, graph) == {node2, node3}
    assert odd_neighbors({node1, node2}, graph) == {node1, node2}


# ---- Coordinate Tests ----


def test_set_coordinate(graph: GraphState) -> None:
    """Test setting coordinates for a node."""
    node = graph.add_node()
    graph.set_coordinate(node, (1.0, 2.0))
    assert graph.coordinates == {node: (1.0, 2.0)}


def test_set_coordinate_invalid_node(graph: GraphState) -> None:
    """Test that set_coordinate raises error for non-existent node."""
    with pytest.raises(ValueError, match="Node does not exist"):
        graph.set_coordinate(999, (1.0, 2.0))


def test_set_coordinate_3d(graph: GraphState) -> None:
    """Test setting 3D coordinates for a node."""
    node = graph.add_node()
    graph.set_coordinate(node, (1.0, 2.0, 3.0))
    assert graph.coordinates == {node: (1.0, 2.0, 3.0)}


def test_add_node_with_coordinate() -> None:
    """Test adding a node with coordinates."""
    graph = GraphState()
    node = graph.add_node(coordinate=(1.5, 2.5))
    assert graph.coordinates == {node: (1.5, 2.5)}


def test_remove_node_removes_coordinate() -> None:
    """Test that removing a node also removes its coordinate."""
    graph = GraphState()
    node1 = graph.add_node(coordinate=(1.0, 2.0))
    node2 = graph.add_node(coordinate=(3.0, 4.0))
    graph.add_edge(node1, node2)
    graph.register_output(node2, 0)
    graph.assign_meas_basis(node1, PlannerMeasBasis(Plane.XY, 0.0))

    graph.remove_node(node1)
    assert node1 not in graph.coordinates
    assert graph.coordinates == {node2: (3.0, 4.0)}


def test_from_graph_with_coordinates() -> None:
    """Test from_graph with coordinates parameter."""
    nodes = ["a", "b", "c"]
    edges = [("a", "b"), ("b", "c")]
    coordinates = {"a": (0.0, 0.0), "b": (1.0, 0.0), "c": (2.0, 0.0)}

    graph, node_map = GraphState.from_graph(nodes, edges, inputs=["a"], outputs=["c"], coordinates=coordinates)

    assert graph.coordinates[node_map["a"]] == (0.0, 0.0)
    assert graph.coordinates[node_map["b"]] == (1.0, 0.0)
    assert graph.coordinates[node_map["c"]] == (2.0, 0.0)


def test_from_base_graph_state_copies_coordinates() -> None:
    """Test that from_base_graph_state copies coordinates."""
    graph1 = GraphState()
    node1 = graph1.add_node(coordinate=(1.0, 2.0))
    node2 = graph1.add_node(coordinate=(3.0, 4.0))
    graph1.add_edge(node1, node2)
    graph1.register_input(node1, 0)
    graph1.register_output(node2, 0)
    graph1.assign_meas_basis(node1, PlannerMeasBasis(Plane.XY, 0.0))

    graph2, node_map = GraphState.from_base_graph_state(graph1)

    assert graph2.coordinates[node_map[node1]] == (1.0, 2.0)
    assert graph2.coordinates[node_map[node2]] == (3.0, 4.0)


def test_compose_copies_coordinates() -> None:
    """Test that compose copies coordinates from both graphs."""
    # Create first graph with coordinates
    graph1 = GraphState()
    g1_in = graph1.add_node(coordinate=(0.0, 0.0))
    g1_out = graph1.add_node(coordinate=(1.0, 0.0))
    graph1.add_edge(g1_in, g1_out)
    graph1.register_input(g1_in, 0)
    graph1.register_output(g1_out, 0)
    graph1.assign_meas_basis(g1_in, PlannerMeasBasis(Plane.XY, 0.0))

    # Create second graph with coordinates
    graph2 = GraphState()
    g2_in = graph2.add_node(coordinate=(2.0, 0.0))
    g2_out = graph2.add_node(coordinate=(3.0, 0.0))
    graph2.add_edge(g2_in, g2_out)
    graph2.register_input(g2_in, 0)
    graph2.register_output(g2_out, 0)
    graph2.assign_meas_basis(g2_in, PlannerMeasBasis(Plane.XY, 0.0))

    # Compose graphs
    composed, node_map1, node_map2 = compose(graph1, graph2)

    # Verify coordinates from graph1 (input node only, output is connected)
    assert composed.coordinates[node_map1[g1_in]] == (0.0, 0.0)

    # Verify coordinates from graph2
    assert composed.coordinates[node_map2[g2_in]] == (2.0, 0.0)
    assert composed.coordinates[node_map2[g2_out]] == (3.0, 0.0)
