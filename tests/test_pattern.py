"""Tests for pattern module."""

from __future__ import annotations

import math

import pytest

from graphqomb.command import TICK, E, M, N
from graphqomb.common import Plane, PlannerMeasBasis
from graphqomb.graphstate import GraphState
from graphqomb.pattern import Pattern
from graphqomb.pauli_frame import PauliFrame


def test_n_command_str_with_coordinate() -> None:
    """Test N command string representation with coordinate."""
    cmd = N(node=5, coordinate=(1.0, 2.0))
    assert str(cmd) == "N: node=5, coord=(1.0, 2.0)"


def test_n_command_str_without_coordinate() -> None:
    """Test N command string representation without coordinate."""
    cmd = N(node=3)
    assert str(cmd) == "N: node=3"


@pytest.fixture
def pattern_components() -> tuple[dict[int, int], dict[int, int], PauliFrame, list[int]]:
    """Create shared components for building Pattern instances.

    Returns
    -------
    tuple[dict[int, int], dict[int, int], PauliFrame, list[int]]
        A tuple containing input node indices, output node indices,
        Pauli frame, and list of nodes.
    """
    graph = GraphState()
    nodes = [graph.add_node() for _ in range(3)]

    graph.register_input(nodes[0], 0)
    graph.register_output(nodes[2], 0)

    pauli_frame = PauliFrame(graph, xflow={}, zflow={})

    return graph.input_node_indices, graph.output_node_indices, pauli_frame, nodes


def test_pattern_depth_counts_tick_commands(
    pattern_components: tuple[dict[int, int], dict[int, int], PauliFrame, list[int]],
) -> None:
    """Test that depth counts the number of TICK commands."""
    input_nodes, output_nodes, pauli_frame, nodes = pattern_components
    meas_basis = PlannerMeasBasis(Plane.XY, 0.0)
    commands = (
        N(node=nodes[0]),
        TICK(),
        E(nodes=(nodes[0], nodes[1])),
        M(node=nodes[1], meas_basis=meas_basis),
        TICK(),
    )

    pattern = Pattern(
        input_node_indices=input_nodes,
        output_node_indices=output_nodes,
        commands=commands,
        clifford_frame=pauli_frame,
    )

    assert pattern.depth == 2


def test_pattern_depth_is_zero_without_ticks(
    pattern_components: tuple[dict[int, int], dict[int, int], PauliFrame, list[int]],
) -> None:
    """Test that depth returns zero when pattern has no TICK commands."""
    input_nodes, output_nodes, pauli_frame, nodes = pattern_components
    meas_basis = PlannerMeasBasis(Plane.XY, 0.5)
    commands = (
        N(node=nodes[0]),
        E(nodes=(nodes[0], nodes[1])),
        M(node=nodes[1], meas_basis=meas_basis),
    )

    pattern = Pattern(
        input_node_indices=input_nodes,
        output_node_indices=output_nodes,
        commands=commands,
        clifford_frame=pauli_frame,
    )

    assert pattern.depth == 0


# Tests for active_volume


def test_active_volume_sums_space_list(
    pattern_components: tuple[dict[int, int], dict[int, int], PauliFrame, list[int]],
) -> None:
    """Test that active_volume equals sum of space list."""
    input_nodes, output_nodes, pauli_frame, nodes = pattern_components
    meas_basis = PlannerMeasBasis(Plane.XY, 0.0)
    commands = (
        N(node=nodes[1]),
        TICK(),
        E(nodes=(nodes[0], nodes[1])),
        M(node=nodes[1], meas_basis=meas_basis),
        TICK(),
    )

    pattern = Pattern(
        input_node_indices=input_nodes,
        output_node_indices=output_nodes,
        commands=commands,
        clifford_frame=pauli_frame,
    )

    assert pattern.active_volume == sum(pattern.space)


def test_active_volume_with_multiple_ticks(
    pattern_components: tuple[dict[int, int], dict[int, int], PauliFrame, list[int]],
) -> None:
    """Test active_volume correctly sums space across multiple time slices."""
    input_nodes, output_nodes, pauli_frame, nodes = pattern_components
    meas_basis = PlannerMeasBasis(Plane.XY, 0.0)
    # Pattern: 1 input, add 2 nodes, measure 1, add 1, measure 2
    # space should be [1, 3, 2, 3, 1]
    commands = (
        N(node=nodes[1]),
        N(node=nodes[2]),
        TICK(),
        M(node=nodes[1], meas_basis=meas_basis),
        TICK(),
        N(node=3),  # New node
        TICK(),
        M(node=nodes[2], meas_basis=meas_basis),
        M(node=3, meas_basis=meas_basis),
        TICK(),
    )

    pattern = Pattern(
        input_node_indices=input_nodes,
        output_node_indices=output_nodes,
        commands=commands,
        clifford_frame=pauli_frame,
    )

    # space = [1, 3, 2, 3, 1] -> active_volume = 10
    assert pattern.active_volume == sum(pattern.space)


# Tests for volume


def test_volume_equals_max_space_times_depth(
    pattern_components: tuple[dict[int, int], dict[int, int], PauliFrame, list[int]],
) -> None:
    """Test that volume equals max_space times depth."""
    input_nodes, output_nodes, pauli_frame, nodes = pattern_components
    meas_basis = PlannerMeasBasis(Plane.XY, 0.0)
    commands = (
        N(node=nodes[1]),
        TICK(),
        M(node=nodes[1], meas_basis=meas_basis),
        TICK(),
    )

    pattern = Pattern(
        input_node_indices=input_nodes,
        output_node_indices=output_nodes,
        commands=commands,
        clifford_frame=pauli_frame,
    )

    assert pattern.volume == pattern.max_space * pattern.depth


def test_volume_is_zero_without_ticks(
    pattern_components: tuple[dict[int, int], dict[int, int], PauliFrame, list[int]],
) -> None:
    """Test that volume is zero when there are no TICK commands."""
    input_nodes, output_nodes, pauli_frame, nodes = pattern_components
    meas_basis = PlannerMeasBasis(Plane.XY, 0.0)
    commands = (
        N(node=nodes[1]),
        M(node=nodes[1], meas_basis=meas_basis),
    )

    pattern = Pattern(
        input_node_indices=input_nodes,
        output_node_indices=output_nodes,
        commands=commands,
        clifford_frame=pauli_frame,
    )

    assert pattern.depth == 0
    assert pattern.volume == 0


# Tests for idle_times


def test_idle_times_returns_dict_for_measured_qubits(
    pattern_components: tuple[dict[int, int], dict[int, int], PauliFrame, list[int]],
) -> None:
    """Test that idle_times returns a dict with entries for measured qubits."""
    input_nodes, output_nodes, pauli_frame, nodes = pattern_components
    meas_basis = PlannerMeasBasis(Plane.XY, 0.0)
    commands = (
        N(node=nodes[1]),
        TICK(),
        M(node=nodes[1], meas_basis=meas_basis),
        TICK(),
    )

    pattern = Pattern(
        input_node_indices=input_nodes,
        output_node_indices=output_nodes,
        commands=commands,
        clifford_frame=pauli_frame,
    )

    idle_times = pattern.idle_times
    # Should include measured node
    assert nodes[1] in idle_times
    # idle_time for nodes[1]: prepared at time 0, measured at time 1 -> idle = 1
    assert idle_times[nodes[1]] == 1


def test_idle_times_input_nodes_use_zero_baseline(
    pattern_components: tuple[dict[int, int], dict[int, int], PauliFrame, list[int]],
) -> None:
    """Test that input nodes have idle time starting from time 0."""
    input_nodes, output_nodes, pauli_frame, nodes = pattern_components
    meas_basis = PlannerMeasBasis(Plane.XY, 0.0)
    # Input node measured at time 1 (after 1 TICK)
    # prepared_time = 0, current_time = 1 -> idle_time = 1
    commands = (
        TICK(),
        M(node=nodes[0], meas_basis=meas_basis),
        TICK(),
    )

    pattern = Pattern(
        input_node_indices=input_nodes,
        output_node_indices=output_nodes,
        commands=commands,
        clifford_frame=pauli_frame,
    )

    idle_times = pattern.idle_times
    # Input node prepared at 0, measured after 1 TICK (time=1)
    # idle_time = 1 - 0 = 1
    assert idle_times[nodes[0]] == 1


def test_idle_times_output_nodes_included_when_prepared() -> None:
    """Test that output nodes are included in idle_times when they are prepared."""
    # Create a graph where output node is also an input node
    graph = GraphState()
    input_node = graph.add_node()
    output_node = graph.add_node()

    graph.register_input(input_node, 0)
    graph.register_input(output_node, 1)  # Output node is also an input
    graph.register_output(output_node, 0)

    pauli_frame = PauliFrame(graph, xflow={}, zflow={})

    meas_basis = PlannerMeasBasis(Plane.XY, 0.0)
    commands = (
        TICK(),
        M(node=input_node, meas_basis=meas_basis),
        TICK(),
        TICK(),
    )

    pattern = Pattern(
        input_node_indices=graph.input_node_indices,
        output_node_indices=graph.output_node_indices,
        commands=commands,
        clifford_frame=pauli_frame,
    )

    idle_times = pattern.idle_times
    # Output node prepared at 0, final time is 3 -> idle = 3
    assert output_node in idle_times
    assert idle_times[output_node] == 3


# Tests for throughput


def test_throughput_calculates_measurements_per_tick(
    pattern_components: tuple[dict[int, int], dict[int, int], PauliFrame, list[int]],
) -> None:
    """Test that throughput correctly calculates measurements per tick."""
    input_nodes, output_nodes, pauli_frame, nodes = pattern_components
    meas_basis = PlannerMeasBasis(Plane.XY, 0.0)
    # 2 measurements, 4 TICKs -> throughput = 0.5
    commands = (
        N(node=nodes[1]),
        TICK(),
        M(node=nodes[1], meas_basis=meas_basis),
        TICK(),
        N(node=3),
        TICK(),
        M(node=3, meas_basis=meas_basis),
        TICK(),
    )

    pattern = Pattern(
        input_node_indices=input_nodes,
        output_node_indices=output_nodes,
        commands=commands,
        clifford_frame=pauli_frame,
    )

    assert math.isclose(pattern.throughput, 2 / 4)


def test_throughput_raises_for_zero_depth(
    pattern_components: tuple[dict[int, int], dict[int, int], PauliFrame, list[int]],
) -> None:
    """Test that throughput raises ValueError when pattern has no TICKs."""
    input_nodes, output_nodes, pauli_frame, nodes = pattern_components
    meas_basis = PlannerMeasBasis(Plane.XY, 0.0)
    commands = (
        N(node=nodes[1]),
        M(node=nodes[1], meas_basis=meas_basis),
    )

    pattern = Pattern(
        input_node_indices=input_nodes,
        output_node_indices=output_nodes,
        commands=commands,
        clifford_frame=pauli_frame,
    )

    with pytest.raises(ValueError, match="Cannot calculate throughput"):
        _ = pattern.throughput
