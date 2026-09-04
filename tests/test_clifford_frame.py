"""Tests for the Clifford extension of the correction frame."""

from __future__ import annotations

import itertools

import pytest

from graphqomb import clifford_algebra as ca
from graphqomb.common import Axis as MeasAxis
from graphqomb.common import AxisMeasBasis, Sign
from graphqomb.graphstate import GraphState
from graphqomb.pauli_frame import CliffordFrame, PauliFrame


def _chain_graph(length: int) -> GraphState:
    graph = GraphState()
    nodes = [graph.add_node() for _ in range(length)]
    graph.register_input(nodes[0], 0)
    graph.register_output(nodes[-1], 0)
    for node1, node2 in itertools.pairwise(nodes):
        graph.add_edge(node1, node2)
    for node in nodes[:-1]:
        graph.assign_meas_basis(node, AxisMeasBasis(MeasAxis.X, Sign.PLUS))
    return graph


def test_pauli_frame_is_clifford_frame_alias() -> None:
    assert PauliFrame is CliffordFrame


def test_cflow_normalization_splits_pauli_part() -> None:
    graph = _chain_graph(3)
    frame = CliffordFrame(graph, xflow={}, zflow={}, cflow={0: {1: ca.compose(ca.S, ca.X)}})
    assert frame.cflow == {0: {1: ca.S}}
    assert frame.xflow == {0: {1}}
    assert frame.zflow == {}
    assert frame.inv_cflow == {1: {0: ca.S}}


def test_cflow_normalization_cancels_existing_pauli_flow() -> None:
    graph = _chain_graph(3)
    frame = CliffordFrame(graph, xflow={0: {1}}, zflow={}, cflow={0: {1: ca.X}})
    assert frame.cflow == {}
    assert frame.xflow == {0: set()}


def test_meas_flip_accumulates_cosets() -> None:
    graph = _chain_graph(3)
    frame = CliffordFrame(graph, xflow={}, zflow={}, cflow={0: {1: ca.S}})

    frame.meas_flip(0)
    assert frame.coset[1] == ca.S
    assert not frame.x_pauli[1]
    assert not frame.z_pauli[1]

    # S * S = Z modulo phase, so the coset resets and the Z bit toggles.
    frame.meas_flip(0)
    assert frame.coset[1] == ca.IDENTITY
    assert not frame.x_pauli[1]
    assert frame.z_pauli[1]


def test_meas_flip_conjugates_pauli_through_coset() -> None:
    graph = _chain_graph(4)
    frame = CliffordFrame(graph, xflow={0: {1}, 1: {2}}, zflow={}, cflow={0: {2: ca.S}})

    frame.meas_flip(0)
    assert frame.coset[2] == ca.S
    frame.meas_flip(1)
    # X * S = S * (S^-1 X S) = S * (-Y) modulo phase, so both bits toggle.
    assert frame.coset[2] == ca.S
    assert frame.x_pauli[2]
    assert frame.z_pauli[2]


def test_meas_flip_without_cflow_matches_pauli_frame() -> None:
    graph = _chain_graph(3)
    frame = CliffordFrame(graph, xflow={0: {1}}, zflow={0: {2}})
    frame.meas_flip(0)
    assert frame.x_pauli == {0: False, 1: True, 2: False}
    assert frame.z_pauli == {0: False, 1: False, 2: True}
    assert frame.coset == dict.fromkeys(graph.nodes, ca.IDENTITY)


def test_parents_and_children_include_cflow() -> None:
    graph = _chain_graph(3)
    frame = CliffordFrame(graph, xflow={}, zflow={}, cflow={0: {1: ca.S}})
    assert frame.children(0) == {1}
    assert frame.parents(1) == {0}


def test_incomparable_noncommuting_sources_raise() -> None:
    graph = GraphState()
    n0 = graph.add_node()
    n1 = graph.add_node()
    n2 = graph.add_node()
    graph.register_input(n0, 0)
    graph.register_input(n1, 1)
    graph.register_output(n2, 0)
    graph.add_edge(n0, n2)
    graph.add_edge(n1, n2)
    graph.assign_meas_basis(n0, AxisMeasBasis(MeasAxis.X, Sign.PLUS))
    graph.assign_meas_basis(n1, AxisMeasBasis(MeasAxis.X, Sign.PLUS))

    with pytest.raises(ValueError, match="do not commute"):
        CliffordFrame(graph, xflow={n1: {n2}}, zflow={}, cflow={n0: {n2: ca.S}})

    # Ordering the sources through the DAG makes the same corrections valid.
    frame = CliffordFrame(graph, xflow={n0: {n1}, n1: {n2}}, zflow={}, cflow={n0: {n2: ca.S}})
    assert frame.cflow == {n0: {n2: ca.S}}

    # Commuting corrections need no ordering: S and Z commute mod phase.
    frame = CliffordFrame(graph, xflow={}, zflow={n1: {n2}}, cflow={n0: {n2: ca.S}})
    assert frame.cflow == {n0: {n2: ca.S}}


def test_detector_certification_raises_for_cflow_influenced_nodes() -> None:
    graph = _chain_graph(3)
    frame = CliffordFrame(
        graph,
        xflow={0: {1}},
        zflow={},
        cflow={0: {1: ca.S}},
        parity_check_group=[{1}],
    )
    with pytest.raises(NotImplementedError, match="Clifford feedforward"):
        frame.detector_groups()
    with pytest.raises(NotImplementedError, match="Clifford feedforward"):
        frame.detector_stabilizers()
    with pytest.raises(NotImplementedError, match="Clifford feedforward"):
        frame.detector_determinism()
    with pytest.raises(NotImplementedError, match="Clifford feedforward"):
        frame.logical_observables_group({1})


def test_detector_certification_still_works_for_uninfluenced_nodes() -> None:
    graph = _chain_graph(4)
    frame_with_cflow = CliffordFrame(
        graph,
        xflow={0: {1}},
        zflow={},
        cflow={2: {3: ca.S}},
        parity_check_group=[{1}],
    )
    frame_without_cflow = CliffordFrame(graph, xflow={0: {1}}, zflow={}, parity_check_group=[{1}])
    assert frame_with_cflow.detector_groups() == frame_without_cflow.detector_groups()
