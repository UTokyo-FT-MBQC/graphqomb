from __future__ import annotations

import pytest

from graphqomb.common import default_meas_basis
from graphqomb.focus_flow import _update_gflow, focus_gflow, is_focused
from graphqomb.graphstate import GraphState


@pytest.fixture
def graphstate() -> GraphState:
    return GraphState()


def test_update_gflow_selects_minimal_node() -> None:
    """_update_gflow must XOR with the *earliest* node in topo_order."""
    gflow: dict[int, set[int]] = {0: {1}, 1: {2}, 2: set()}
    _update_gflow(0, gflow, s_k=[2, 1], topo_order=[0, 1, 2])
    assert gflow[0] == {1, 2}


def test_is_focused_true_for_focused_flow(graphstate: GraphState) -> None:
    """If a child is the same as its parent, the flow is focused."""
    node1 = graphstate.add_node()
    node2 = graphstate.add_node()
    graphstate.add_edge(node1, node2)
    q_index = 0
    graphstate.register_input(node1, q_index)
    graphstate.register_output(node2, q_index)
    graphstate.assign_meas_basis(node1, default_meas_basis())
    flow = {node1: node2}
    assert is_focused(flow, graphstate)


def test_is_focused_false_for_unfocused_flow(graphstate: GraphState) -> None:
    """If a child differs from its parent, the flow is not focused."""
    node1 = graphstate.add_node()
    node2 = graphstate.add_node()
    node3 = graphstate.add_node()
    graphstate.add_edge(node1, node2)
    graphstate.add_edge(node2, node3)
    graphstate.assign_meas_basis(node1, default_meas_basis())
    graphstate.assign_meas_basis(node2, default_meas_basis())
    flow = {node1: node2, node2: node3}
    assert not is_focused(flow, graphstate)


def test_focus_gflow_raises_typeerror(graphstate: GraphState) -> None:
    """If neither Flow nor GFlow, focus_gflow must raise TypeError."""
    with pytest.raises(TypeError):
        focus_gflow({"bad": object()}, graphstate)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
