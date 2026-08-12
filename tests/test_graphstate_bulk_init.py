"""Test for GraphState bulk initialization methods: from_graph() and from_base_graph_state()."""

from __future__ import annotations

import math

import pytest

from graphqomb.common import Axis, Initialization, Plane, PlannerMeasBasis
from graphqomb.graphstate import GraphState


def test_from_graph_with_string_nodes() -> None:
    """Test from_graph() with string node identifiers."""
    nodes = ["start", "middle", "end"]
    edges = [("start", "middle"), ("middle", "end")]

    gs, node_map = GraphState.from_graph(nodes=nodes, edges=edges)

    assert node_map == {"start": 0, "middle": 1, "end": 2}
    assert gs.nodes == {0, 1, 2}
    assert gs.edges == {(0, 1), (1, 2)}


def test_from_graph_with_tuple_nodes() -> None:
    """Test from_graph() with tuple node identifiers for grid graphs."""
    grid_nodes = [(i, j) for i in range(2) for j in range(2)]
    grid_edges = [
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1)),
        ((1, 0), (1, 1)),
    ]

    gs, node_map = GraphState.from_graph(nodes=grid_nodes, edges=grid_edges)

    assert len(node_map) == 4
    assert gs.nodes == {0, 1, 2, 3}
    assert len(gs.edges) == 4


def test_from_graph_with_int_nodes() -> None:
    """Test from_graph() with integer node identifiers (backward compatibility)."""
    nodes = [10, 20, 30]
    edges = [(10, 20), (20, 30)]

    gs, node_map = GraphState.from_graph(nodes=nodes, edges=edges)

    assert node_map == {10: 0, 20: 1, 30: 2}
    assert gs.nodes == {0, 1, 2}


def test_from_graph_with_inputs_outputs() -> None:
    """Test from_graph() with input and output registration."""
    nodes = ["a", "b", "c"]
    edges = [("a", "b"), ("b", "c")]
    inputs = ["a"]
    outputs = ["c"]

    gs, node_map = GraphState.from_graph(nodes=nodes, edges=edges, inputs=inputs, outputs=outputs)

    assert gs.input_node_indices == {node_map[inputs[0]]: 0}
    assert gs.output_node_indices == {node_map[outputs[0]]: 0}


def test_from_graph_with_multiple_inputs_outputs() -> None:
    """Test from_graph() with multiple inputs and outputs."""
    nodes = ["a", "b", "c", "d"]
    edges = [("a", "b"), ("b", "c"), ("c", "d")]
    inputs = ["a", "b"]
    outputs = ["c", "d"]

    gs, node_map = GraphState.from_graph(nodes=nodes, edges=edges, inputs=inputs, outputs=outputs)

    assert gs.input_node_indices == {node_map[inputs[0]]: 0, node_map[inputs[1]]: 1}
    assert gs.output_node_indices == {node_map[outputs[0]]: 0, node_map[outputs[1]]: 1}


def test_from_graph_with_input_initializations() -> None:
    """Test from_graph() preserves specified input initializations."""
    nodes = ["a", "b", "c"]
    edges = [("a", "c"), ("b", "c")]
    inputs = ["a", "b"]

    gs, node_map = GraphState.from_graph(
        nodes=nodes,
        edges=edges,
        inputs=inputs,
        input_initializations={"a": Initialization(axis=Axis.Y), "b": Initialization(axis=Axis.Z, tag="anc")},
    )

    assert gs.input_initializations == {
        node_map["a"]: Initialization(axis=Axis.Y),
        node_map["b"]: Initialization(axis=Axis.Z, tag="anc"),
    }


def test_from_graph_rejects_initialization_for_non_input_node() -> None:
    """Input initializations cannot silently target non-input nodes."""
    with pytest.raises(ValueError, match="specified for non-input node"):
        GraphState.from_graph(
            nodes=["input", "not-input"],
            edges=[],
            inputs=["input"],
            input_initializations={"not-input": Initialization(axis=Axis.Z)},
        )


def test_from_graph_rejects_initializations_without_inputs() -> None:
    """Initializations require corresponding input registrations."""
    with pytest.raises(ValueError, match="specified for non-input node"):
        GraphState.from_graph(
            nodes=["node"],
            edges=[],
            input_initializations={"node": Initialization(axis=Axis.Y)},
        )


def test_from_graph_with_meas_bases() -> None:
    """Test from_graph() with measurement bases."""
    nodes = ["a", "b", "c"]
    edges = [("a", "b"), ("b", "c")]
    meas_bases = {
        "a": PlannerMeasBasis(Plane.XY, 0.0),
        "b": PlannerMeasBasis(Plane.XY, 1.5708),
    }

    gs, node_map = GraphState.from_graph(nodes=nodes, edges=edges, meas_bases=meas_bases)

    assert node_map[nodes[0]] in gs.meas_bases
    assert node_map[nodes[1]] in gs.meas_bases
    assert node_map[nodes[2]] not in gs.meas_bases
    # Check that the measurement bases have the correct attributes
    assert gs.meas_bases[node_map[nodes[0]]].plane == Plane.XY
    assert math.isclose(gs.meas_bases[node_map[nodes[0]]].angle, 0.0)
    assert gs.meas_bases[node_map[nodes[1]]].plane == Plane.XY
    assert math.isclose(gs.meas_bases[node_map[nodes[1]]].angle, 1.5708)


def test_from_graph_with_partial_meas_bases() -> None:
    """Test from_graph() with partial measurement basis specification."""
    nodes = ["a", "b", "c"]
    edges = [("a", "b")]
    meas_bases = {"a": PlannerMeasBasis(Plane.XY, 0.0)}

    gs, node_map = GraphState.from_graph(nodes=nodes, edges=edges, meas_bases=meas_bases)

    assert node_map[nodes[0]] in gs.meas_bases
    assert node_map[nodes[1]] not in gs.meas_bases


def test_from_graph_errors_duplicate_nodes() -> None:
    """Test from_graph() raises error on duplicate nodes."""
    nodes = ["a", "b", "a"]  # 'a' appears twice
    edges = [("a", "b")]

    with pytest.raises(ValueError, match="Duplicate nodes in input"):
        GraphState.from_graph(nodes=nodes, edges=edges)


def test_from_graph_errors_invalid_input_node() -> None:
    """Test from_graph() raises error when input node not in nodes."""
    nodes = ["a", "b", "c"]
    edges = [("a", "b")]
    inputs = ["d"]  # 'd' not in nodes

    with pytest.raises(ValueError, match=r"Input node .* not in nodes collection"):
        GraphState.from_graph(nodes=nodes, edges=edges, inputs=inputs)


def test_from_graph_errors_invalid_output_node() -> None:
    """Test from_graph() raises error when output node not in nodes."""
    nodes = ["a", "b", "c"]
    edges = [("a", "b")]
    outputs = ["d"]  # 'd' not in nodes

    with pytest.raises(ValueError, match=r"Output node .* not in nodes collection"):
        GraphState.from_graph(nodes=nodes, edges=edges, outputs=outputs)


def test_from_graph_errors_invalid_edges() -> None:
    """Test from_graph() raises error on non-existent node in edge."""
    nodes = ["a", "b"]
    edges = [("a", "c")]  # 'c' not in nodes

    with pytest.raises(ValueError, match="Edge references non-existent node"):
        GraphState.from_graph(nodes=nodes, edges=edges)


def test_from_graph_empty_nodes() -> None:
    """Test from_graph() with empty nodes."""
    nodes: list[int] = []
    edges: list[tuple[int, int]] = []
    gs, node_map = GraphState.from_graph(nodes=nodes, edges=edges)

    assert node_map == {}
    assert gs.nodes == set()
    assert gs.edges == set()


def test_from_graph_single_node() -> None:
    """Test from_graph() with single node."""
    gs, node_map = GraphState.from_graph(nodes=["a"], edges=[])

    assert node_map == {"a": 0}
    assert gs.nodes == {0}


def test_from_base_graph_state_simple() -> None:
    """Test from_base_graph_state() basic copying."""
    # Create source graph
    src = GraphState()
    n0 = src.add_node()
    n1 = src.add_node()
    n2 = src.add_node()
    src.add_edge(n0, n1)
    src.add_edge(n1, n2)
    src.register_input(n0, 0)
    src.register_output(n2, 0)
    src.assign_meas_basis(n0, PlannerMeasBasis(Plane.XY, 0.0))
    src.assign_meas_basis(n1, PlannerMeasBasis(Plane.XY, 1.5708))

    # Copy the graph
    dst, node_map = GraphState.from_base_graph_state(src)

    assert node_map == {0: 0, 1: 1, 2: 2}
    assert dst.nodes == {0, 1, 2}
    assert dst.edges == {(0, 1), (1, 2)}
    assert dst.input_node_indices == {0: 0}
    assert dst.output_node_indices == {2: 0}
    # Check measurement bases
    assert 0 in dst.meas_bases
    assert 1 in dst.meas_bases
    assert dst.meas_bases[0].plane == Plane.XY
    assert math.isclose(dst.meas_bases[0].angle, 0.0)
    assert dst.meas_bases[1].plane == Plane.XY
    assert math.isclose(dst.meas_bases[1].angle, 1.5708)


def test_from_base_graph_state_preserves_indices() -> None:
    """Test from_base_graph_state() preserves qubit indices."""
    # Create source graph with custom qubit indices
    src = GraphState()
    n0 = src.add_node()
    n1 = src.add_node()
    src.register_input(n0, 5)
    src.register_output(n1, 10)

    # Copy the graph
    dst, node_map = GraphState.from_base_graph_state(src)

    assert dst.input_node_indices[node_map[n0]] == 5
    assert dst.output_node_indices[node_map[n1]] == 10


def test_from_base_graph_state_preserves_input_initializations() -> None:
    """Test from_base_graph_state() preserves input initializations."""
    src = GraphState()
    n0 = src.add_node()
    n1 = src.add_node()
    src.register_input(n0, 0, init=Initialization(axis=Axis.Y))
    src.register_input(n1, 1, init=Initialization(axis=Axis.Z, tag="anc"))

    dst, node_map = GraphState.from_base_graph_state(src)

    assert dst.input_initializations == {
        node_map[n0]: Initialization(axis=Axis.Y),
        node_map[n1]: Initialization(axis=Axis.Z, tag="anc"),
    }


def test_from_base_graph_state_independence() -> None:
    """Test that modifications to copied graph don't affect original."""
    # Create source graph
    src = GraphState()
    src.add_node()
    src.add_node()

    # Copy the graph
    dst, _ = GraphState.from_base_graph_state(src)

    # Modify copied graph
    dst.add_node()

    assert len(src.nodes) == 2
    assert len(dst.nodes) == 3


def test_from_base_graph_state_empty() -> None:
    """Test from_base_graph_state() with empty graph."""
    src = GraphState()
    dst, node_map = GraphState.from_base_graph_state(src)

    assert node_map == {}
    assert dst.nodes == set()
    assert dst.edges == set()


def test_from_graph_with_complex_structure() -> None:
    """Test from_graph() with complex graph structure."""
    # Create a more complex graph
    nodes = ["n1", "n2", "n3", "n4", "n5"]
    edges = [
        ("n1", "n2"),
        ("n1", "n3"),
        ("n2", "n4"),
        ("n3", "n4"),
        ("n4", "n5"),
    ]
    inputs = ["n1"]
    outputs = ["n5"]

    gs, node_map = GraphState.from_graph(nodes=nodes, edges=edges, inputs=inputs, outputs=outputs)

    # Verify structure
    assert len(node_map) == 5
    assert gs.input_node_indices == {node_map["n1"]: 0}
    assert gs.output_node_indices == {node_map["n5"]: 0}
    assert len(gs.edges) == 5


def test_from_graph_preserves_node_order() -> None:
    """Test that from_graph() preserves node order for sequential index assignment."""
    nodes = ["z", "a", "m"]  # Out of alphabetical order
    edges: list[tuple[str, str]] = []

    _gs, node_map = GraphState.from_graph(nodes=nodes, edges=edges)

    # Nodes should be mapped in the order they were provided
    assert node_map == {"z": 0, "a": 1, "m": 2}


def test_from_base_graph_state_with_complex_graph() -> None:
    """Test from_base_graph_state() with a complex graph."""
    # Create source graph with multiple components
    src = GraphState()
    nodes = [src.add_node() for _ in range(5)]

    # Add edges to create a specific structure
    src.add_edge(nodes[0], nodes[1])
    src.add_edge(nodes[1], nodes[2])
    src.add_edge(nodes[2], nodes[3])
    src.add_edge(nodes[3], nodes[4])

    # Register some inputs and outputs
    src.register_input(nodes[0], 0)
    src.register_input(nodes[1], 1)
    src.register_output(nodes[3], 0)
    src.register_output(nodes[4], 1)

    # Set measurement bases
    for node in nodes[:-1]:
        src.assign_meas_basis(node, PlannerMeasBasis(Plane.XY, 0.0))

    # Copy the graph
    dst, _node_map = GraphState.from_base_graph_state(src)

    # Verify all structure is preserved
    assert dst.nodes == set(range(5))
    assert len(dst.edges) == 4
    assert len(dst.input_node_indices) == 2
    assert len(dst.output_node_indices) == 2
