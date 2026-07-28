from __future__ import annotations

import numpy as np
import pytest

from graphqomb.common import Axis, AxisMeasBasis, Plane, PlannerMeasBasis, Sign
from graphqomb.feedforward import (
    TOPO_ORDER_CYCLE_ERROR_MSG,
    _is_flow,
    _is_gflow,
    check_dag,
    check_flow,
    dag_from_flow,
    inverse_dag_from_dag,
    pauli_simplification,
    propagate_correction_map,
    signal_shifting,
    topo_order_from_inv_dag,
)
from graphqomb.graphstate import GraphState


def two_node_graph() -> tuple[GraphState, int, int]:
    graphstate = GraphState()
    node1 = graphstate.add_node()
    node2 = graphstate.add_node()
    graphstate.add_edge(node1, node2)
    return graphstate, node1, node2


def test_is_flow_true() -> None:
    flow = {0: 1, 2: 3}
    assert _is_flow(flow)
    assert not _is_gflow(flow)


def test_is_flow_false_if_mixed_types() -> None:
    mixed: dict[int, int | set[int]] = {0: 1, 1: {2}}
    assert not _is_flow(mixed)
    assert not _is_gflow(mixed)


def test_is_gflow_true() -> None:
    gflow: dict[int, set[int]] = {0: {1}, 1: set()}
    assert _is_gflow(gflow)
    assert not _is_flow(gflow)


def test_dag_from_flow_basic_flow() -> None:
    graphstate, node1, node2 = two_node_graph()
    flow = {node1: node2}

    dag = dag_from_flow(graphstate, flow)
    check_dag(dag)

    assert dag[0] == {1}


def test_dag_from_flow_basic_gflow() -> None:
    graphstate, node1, node2 = two_node_graph()
    gflow: dict[int, set[int]] = {node1: {node2}, node2: set()}

    dag = dag_from_flow(graphstate, gflow)
    check_dag(dag)

    assert dag[node1] == {node2}
    assert dag[node2] == set()


def test_dag_from_flow_removes_self_loop_from_xflow() -> None:
    graphstate, node1, node2 = two_node_graph()
    xflow: dict[int, set[int]] = {node1: {node1, node2}, node2: set()}
    zflow: dict[int, set[int]] = {node1: set(), node2: set()}

    dag = dag_from_flow(graphstate, xflow, zflow)
    check_dag(dag)

    assert node1 not in dag[node1]
    assert dag[node1] == {node2}


def test_dag_from_flow_invalid_type_raises() -> None:
    graphstate, node1, node2 = two_node_graph()
    invalid: dict[int, int | set[int]] = {node1: node2, node2: {node2}}  # mixed types
    with pytest.raises(TypeError):
        dag_from_flow(graphstate, invalid)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_dag_from_flow_cycle_detection() -> None:
    graphstate, node1, node2 = two_node_graph()
    cyclic_flow = {node1: node2, node2: node1}

    dag = dag_from_flow(graphstate, cyclic_flow)
    with pytest.raises(ValueError, match="Cycle detected in the graph:"):
        check_dag(dag)


def test_check_dag_detects_cycle_longer_than_two_nodes() -> None:
    dag = {0: {1}, 1: {2}, 2: {0}}

    with pytest.raises(ValueError, match="Cycle detected in the graph:"):
        check_dag(dag)


def test_check_flow_false_for_cycle() -> None:
    graphstate, node1, node2 = two_node_graph()
    cyclic_flow = {node1: node2, node2: node1}
    with pytest.raises(ValueError, match="Cycle detected in the graph:"):
        check_flow(graphstate, cyclic_flow)


def test_check_flow_true_for_acyclic() -> None:
    graphstate, node1, node2 = two_node_graph()
    flow = {node1: node2}
    check_flow(graphstate, flow)


def test_topo_order_from_inv_dag_basic() -> None:
    inv_dag: dict[int, set[int]] = {
        0: set(),
        1: {0},
        2: {1},
    }
    assert topo_order_from_inv_dag(inv_dag) == [0, 1, 2]


def test_inverse_dag_from_dag_basic() -> None:
    dag: dict[int, set[int]] = {
        0: {1, 2},
        1: {2},
        2: set(),
    }
    assert inverse_dag_from_dag(dag) == {
        0: set(),
        1: {0},
        2: {0, 1},
    }


def test_inverse_dag_from_dag_with_all_nodes() -> None:
    dag = {0: {1}}
    assert inverse_dag_from_dag(dag, all_nodes={0, 1, 2}) == {
        0: set(),
        1: {0},
        2: set(),
    }


def test_topo_order_from_inv_dag_cycle_raises() -> None:
    inv_dag = {
        0: {1},
        1: {0},
    }
    with pytest.raises(RuntimeError, match=TOPO_ORDER_CYCLE_ERROR_MSG):
        topo_order_from_inv_dag(inv_dag)


# Tests for propagate_correction_map


def test_propagate_correction_map_xy_plane() -> None:
    """Test propagate_correction_map with XY plane measurement."""
    # Create a simple graph with 3 nodes: parent -> target -> child
    graphstate = GraphState()
    parent = graphstate.add_node()
    target = graphstate.add_node()
    child = graphstate.add_node()
    graphstate.add_edge(parent, target)
    graphstate.add_edge(target, child)

    # Set measurement basis for target node (XY plane)
    graphstate.assign_meas_basis(target, PlannerMeasBasis(Plane.XY, 0.0))
    # Set measurement basis for other nodes to avoid validation errors
    graphstate.assign_meas_basis(parent, PlannerMeasBasis(Plane.XY, 0.0))

    # Register child as output
    graphstate.register_output(child, 0)

    # Define flows
    xflow = {parent: {target}, target: {child}}
    zflow = {parent: {target}, target: {child}}

    # Propagate through target node
    new_xflow, new_zflow = propagate_correction_map(target, graphstate, xflow, zflow)

    # In XY plane, Z correction should be removed from parent -> target
    assert target not in new_zflow[parent]
    # X correction from target (child) should be propagated to parent
    assert child in new_xflow[parent]
    assert target in new_xflow[parent]


def test_propagate_correction_map_yz_plane() -> None:
    """Test propagate_correction_map with YZ plane measurement."""
    # Create a simple graph with 3 nodes: parent -> target -> child
    graphstate = GraphState()
    parent = graphstate.add_node()
    target = graphstate.add_node()
    child = graphstate.add_node()
    graphstate.add_edge(parent, target)
    graphstate.add_edge(target, child)

    # Set measurement basis for target node (YZ plane)
    graphstate.assign_meas_basis(target, PlannerMeasBasis(Plane.YZ, 0.0))
    graphstate.assign_meas_basis(parent, PlannerMeasBasis(Plane.YZ, 0.0))

    # Register child as output
    graphstate.register_output(child, 0)

    # Define flows
    xflow = {parent: {target}, target: {child}}
    zflow = {parent: {target}, target: {child}}

    # Propagate through target node
    new_xflow, new_zflow = propagate_correction_map(target, graphstate, xflow, zflow)

    # In YZ plane, X correction should be removed from parent -> target
    assert target not in new_xflow[parent]
    # Z correction from target (child) should be propagated to parent
    assert child in new_zflow[parent]
    assert target in new_zflow[parent]


def test_propagate_correction_map_xz_plane() -> None:
    """Test propagate_correction_map with XZ plane measurement."""
    # Create a simple graph with 3 nodes: parent -> target -> child
    graphstate = GraphState()
    parent = graphstate.add_node()
    target = graphstate.add_node()
    child = graphstate.add_node()
    graphstate.add_edge(parent, target)
    graphstate.add_edge(target, child)

    # Set measurement basis for target node (XZ plane)
    graphstate.assign_meas_basis(target, PlannerMeasBasis(Plane.XZ, 0.0))
    graphstate.assign_meas_basis(parent, PlannerMeasBasis(Plane.XZ, 0.0))

    # Register child as output
    graphstate.register_output(child, 0)

    # Define flows
    xflow = {parent: {target}, target: {child}}
    zflow = {parent: {target}, target: {child}}

    # Propagate through target node
    new_xflow, new_zflow = propagate_correction_map(target, graphstate, xflow, zflow)

    # In XZ plane, both X and Z corrections should be removed from parent -> target
    assert target not in new_xflow[parent]
    assert target not in new_zflow[parent]
    # Both X and Z corrections from target (child) should be propagated to parent
    assert child in new_xflow[parent]
    assert child in new_zflow[parent]


def test_propagate_correction_map_output_node_error() -> None:
    """Test that propagate_correction_map raises error for output nodes."""
    graphstate = GraphState()
    node = graphstate.add_node()
    graphstate.register_output(node, 0)

    xflow: dict[int, set[int]] = {node: set()}
    zflow: dict[int, set[int]] = {node: set()}

    with pytest.raises(ValueError, match="Cannot propagate flow for output nodes"):
        propagate_correction_map(node, graphstate, xflow, zflow)


def test_propagate_correction_map_zflow_none() -> None:
    """Test propagate_correction_map with zflow=None."""
    # Create a simple graph
    graphstate = GraphState()
    parent = graphstate.add_node()
    target = graphstate.add_node()
    child = graphstate.add_node()
    graphstate.add_edge(parent, target)
    graphstate.add_edge(target, child)

    # Set measurement basis
    graphstate.assign_meas_basis(target, PlannerMeasBasis(Plane.XY, 0.0))
    graphstate.assign_meas_basis(parent, PlannerMeasBasis(Plane.XY, 0.0))

    # Register child as output
    graphstate.register_output(child, 0)

    # Define xflow only
    xflow = {parent: {target}, target: {child}}

    # Should not raise error; zflow will be generated automatically
    new_xflow, new_zflow = propagate_correction_map(target, graphstate, xflow)

    # Check that both xflow and zflow were generated and corrections were propagated
    assert isinstance(new_xflow, dict)
    assert isinstance(new_zflow, dict)
    assert parent in new_xflow
    assert parent in new_zflow


# Tests for signal_shifting


def test_signal_shifting_simple() -> None:
    """Test signal_shifting on a simple graph."""
    # Create a linear graph: node0 -> node1 -> output
    graphstate = GraphState()
    node0 = graphstate.add_node()
    node1 = graphstate.add_node()
    output = graphstate.add_node()
    graphstate.add_edge(node0, node1)
    graphstate.add_edge(node1, output)

    # Set measurement bases
    graphstate.assign_meas_basis(node0, PlannerMeasBasis(Plane.XY, 0.0))
    graphstate.assign_meas_basis(node1, PlannerMeasBasis(Plane.XY, 0.0))

    # Register output
    graphstate.register_output(output, 0)

    # Define flows
    xflow = {node0: {node1}, node1: {output}}
    zflow = {node0: {node1}, node1: {output}}

    # Apply signal shifting
    new_xflow, new_zflow = signal_shifting(graphstate, xflow, zflow)

    # Verify that flows are valid dictionaries
    assert isinstance(new_xflow, dict)
    assert isinstance(new_zflow, dict)


def test_signal_shifting_zflow_none() -> None:
    """Test signal_shifting with zflow=None."""
    # Create a simple graph
    graphstate = GraphState()
    node0 = graphstate.add_node()
    node1 = graphstate.add_node()
    output = graphstate.add_node()
    graphstate.add_edge(node0, node1)
    graphstate.add_edge(node1, output)

    # Set measurement bases
    graphstate.assign_meas_basis(node0, PlannerMeasBasis(Plane.XY, 0.0))
    graphstate.assign_meas_basis(node1, PlannerMeasBasis(Plane.XY, 0.0))

    # Register output
    graphstate.register_output(output, 0)

    # Define xflow only
    xflow = {node0: {node1}, node1: {output}}

    # Apply signal shifting without zflow
    new_xflow, new_zflow = signal_shifting(graphstate, xflow)

    # Should not raise error; zflow will be generated automatically
    assert isinstance(new_xflow, dict)
    assert isinstance(new_zflow, dict)


# Tests for pauli_simplification


def test_pauli_simplification_x_axis_removes_x_correction() -> None:
    """Test that X-axis measurement removes X corrections from the flow."""
    # Create a 3-node graph: parent -> target -> output
    graphstate = GraphState()
    parent = graphstate.add_node()
    target = graphstate.add_node()
    output = graphstate.add_node()
    graphstate.add_edge(parent, target)
    graphstate.add_edge(target, output)

    # Set X-axis measurement basis for target
    graphstate.assign_meas_basis(target, AxisMeasBasis(Axis.X, Sign.PLUS))
    graphstate.assign_meas_basis(parent, AxisMeasBasis(Axis.X, Sign.PLUS))

    graphstate.register_output(output, 0)

    # Define flows where parent's X correction depends on target
    xflow: dict[int, set[int]] = {parent: {target}, target: {output}}
    zflow: dict[int, set[int]] = {parent: {target}, target: {output}}

    new_xflow, new_zflow = pauli_simplification(graphstate, xflow, zflow)

    # X-axis measurement should remove target from parent's X corrections
    assert target not in new_xflow[parent]
    # Z corrections should remain unchanged
    assert target in new_zflow[parent]


def test_pauli_simplification_z_axis_removes_z_correction() -> None:
    """Test that Z-axis measurement removes Z corrections from the flow."""
    # Create a 3-node graph: parent -> target -> output
    graphstate = GraphState()
    parent = graphstate.add_node()
    target = graphstate.add_node()
    output = graphstate.add_node()
    graphstate.add_edge(parent, target)
    graphstate.add_edge(target, output)

    # Set Z-axis measurement basis for target
    graphstate.assign_meas_basis(target, AxisMeasBasis(Axis.Z, Sign.PLUS))
    graphstate.assign_meas_basis(parent, AxisMeasBasis(Axis.Z, Sign.PLUS))

    graphstate.register_output(output, 0)

    # Define flows where parent's Z correction depends on target
    xflow: dict[int, set[int]] = {parent: {target}, target: {output}}
    zflow: dict[int, set[int]] = {parent: {target}, target: {output}}

    new_xflow, new_zflow = pauli_simplification(graphstate, xflow, zflow)

    # Z-axis measurement should remove target from parent's Z corrections
    assert target not in new_zflow[parent]
    # X corrections should remain unchanged
    assert target in new_xflow[parent]


def test_pauli_simplification_y_axis_removes_both_corrections() -> None:
    """Test that Y-axis measurement removes both X and Z corrections from the flow."""
    # Create a 3-node graph: parent -> target -> output
    graphstate = GraphState()
    parent = graphstate.add_node()
    target = graphstate.add_node()
    output = graphstate.add_node()
    graphstate.add_edge(parent, target)
    graphstate.add_edge(target, output)

    # Set Y-axis measurement basis for target
    graphstate.assign_meas_basis(target, AxisMeasBasis(Axis.Y, Sign.PLUS))
    graphstate.assign_meas_basis(parent, AxisMeasBasis(Axis.X, Sign.PLUS))

    graphstate.register_output(output, 0)

    # Define flows where parent's corrections depend on target
    xflow: dict[int, set[int]] = {parent: {target}, target: {output}}
    zflow: dict[int, set[int]] = {parent: {target}, target: {output}}

    new_xflow, new_zflow = pauli_simplification(graphstate, xflow, zflow)

    # Y-axis measurement should remove target from both X and Z corrections
    assert target not in new_xflow[parent]
    assert target not in new_zflow[parent]


def test_pauli_simplification_y_axis_skip() -> None:
    """Test that Y-axis measurement skips if not both corrections are present."""
    # Create a 3-node graph: parent -> target -> output
    graphstate = GraphState()
    parent = graphstate.add_node()
    target = graphstate.add_node()
    output = graphstate.add_node()
    graphstate.add_edge(parent, target)
    graphstate.add_edge(target, output)

    # Set Y-axis measurement basis for target
    graphstate.assign_meas_basis(target, AxisMeasBasis(Axis.Y, Sign.PLUS))
    graphstate.assign_meas_basis(parent, AxisMeasBasis(Axis.X, Sign.PLUS))

    graphstate.register_output(output, 0)

    # Define flows where parent's corrections depend on target
    xflow: dict[int, set[int]] = {parent: {target}, target: {output}}
    zflow: dict[int, set[int]] = {parent: {output}, target: {output}}

    # Skip removing X correction
    new_xflow, _ = pauli_simplification(graphstate, xflow, zflow)

    assert target in new_xflow[parent]  # X correction remains

    xflow = {parent: {output}, target: {output}}
    zflow = {parent: {target}, target: {output}}
    # Skip removing Z correction
    _, new_zflow = pauli_simplification(graphstate, xflow, zflow)

    assert target in new_zflow[parent]  # Z correction remains


def test_pauli_simplification_non_pauli_leaves_unchanged() -> None:
    """Test that non-Pauli measurement angles leave corrections unchanged."""
    # Create a 3-node graph: parent -> target -> output
    graphstate = GraphState()
    parent = graphstate.add_node()
    target = graphstate.add_node()
    output = graphstate.add_node()
    graphstate.add_edge(parent, target)
    graphstate.add_edge(target, output)

    # Set non-Pauli measurement basis for target (XY plane, angle=pi/4)
    graphstate.assign_meas_basis(target, PlannerMeasBasis(Plane.XY, 0.25 * np.pi))
    graphstate.assign_meas_basis(parent, PlannerMeasBasis(Plane.XY, 0.0))

    graphstate.register_output(output, 0)

    # Define flows
    xflow: dict[int, set[int]] = {parent: {target}, target: {output}}
    zflow: dict[int, set[int]] = {parent: {target}, target: {output}}

    new_xflow, new_zflow = pauli_simplification(graphstate, xflow, zflow)

    # Non-Pauli angle should leave corrections unchanged
    assert target in new_xflow[parent]
    assert target in new_zflow[parent]


def test_pauli_simplification_preserves_original_flows() -> None:
    """Test that original xflow and zflow are not modified."""
    # Create a 3-node graph: parent -> target -> output
    graphstate = GraphState()
    parent = graphstate.add_node()
    target = graphstate.add_node()
    output = graphstate.add_node()
    graphstate.add_edge(parent, target)
    graphstate.add_edge(target, output)

    # Set X-axis measurement basis for target
    graphstate.assign_meas_basis(target, AxisMeasBasis(Axis.X, Sign.PLUS))
    graphstate.assign_meas_basis(parent, AxisMeasBasis(Axis.X, Sign.PLUS))

    graphstate.register_output(output, 0)

    # Define flows
    xflow: dict[int, set[int]] = {parent: {target}, target: {output}}
    zflow: dict[int, set[int]] = {parent: {target}, target: {output}}

    # Store original values
    original_xflow_parent = set(xflow[parent])
    original_zflow_parent = set(zflow[parent])

    pauli_simplification(graphstate, xflow, zflow)

    # Original flows should be unchanged
    assert xflow[parent] == original_xflow_parent
    assert zflow[parent] == original_zflow_parent
