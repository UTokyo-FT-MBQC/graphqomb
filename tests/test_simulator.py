"""Tests for pattern simulation behavior."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from graphqomb.command import TICK, E, M, N
from graphqomb.common import Axis, AxisMeasBasis, Initialization, Sign
from graphqomb.graphstate import GraphState
from graphqomb.pattern import Pattern
from graphqomb.pauli_frame import PauliFrame
from graphqomb.simulator import PatternSimulator, SimulatorBackend
from graphqomb.statevec import StateVector


def _single_output_pattern(
    *,
    measured: bool,
    axis: Axis = Axis.Z,
    init_axis: Axis = Axis.X,
) -> tuple[Pattern, int]:
    graph = GraphState()
    node = graph.add_node()
    graph.register_input(node, 0, init=Initialization(axis=init_axis))
    graph.register_output(node, 0)

    pauli_frame = PauliFrame(graph, xflow={}, zflow={})
    commands = (M(node, AxisMeasBasis(axis, Sign.PLUS)),) if measured else ()
    pattern = Pattern(
        input_node_indices=graph.input_node_indices,
        output_node_indices=graph.output_node_indices,
        commands=commands,
        pauli_frame=pauli_frame,
        input_initializations=graph.input_initializations,
    )
    return pattern, node


def _deterministic_non_output_measurement_pattern() -> tuple[Pattern, int]:
    """Create a pattern whose Z-initialized input has deterministic Z outcome."""
    graph = GraphState()
    input_node = graph.add_node()
    output_node = graph.add_node()
    graph.add_edge(input_node, output_node)
    graph.register_input(input_node, 0, init=Initialization(axis=Axis.Z))
    graph.register_output(output_node, 0)
    graph.assign_meas_basis(input_node, AxisMeasBasis(Axis.Z, Sign.PLUS))

    pattern = Pattern(
        input_node_indices=graph.input_node_indices,
        output_node_indices=graph.output_node_indices,
        commands=(
            N(output_node),
            E((input_node, output_node)),
            M(input_node, graph.meas_bases[input_node]),
            TICK(),
        ),
        pauli_frame=PauliFrame(graph, xflow={}, zflow={input_node: {output_node}}),
        input_initializations=graph.input_initializations,
    )
    return pattern, input_node


def test_pattern_simulator_rejects_unsupported_command() -> None:
    """Unsupported commands fail explicitly instead of recursing."""
    pattern, _ = _single_output_pattern(measured=False)
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    unsupported_command: Any = None

    with pytest.raises(TypeError, match="Unsupported command for pattern simulation: NoneType"):
        simulator.apply_cmd(unsupported_command, rng=np.random.default_rng(0))


def test_pattern_simulator_initializes_input_in_x_basis() -> None:
    """PatternSimulator initializes X-axis inputs as |+>."""
    pattern, _ = _single_output_pattern(measured=False, init_axis=Axis.X)
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)

    simulator.simulate()

    np.testing.assert_allclose(simulator.state.state(), np.asarray([1.0, 1.0]) / np.sqrt(2))


def test_pattern_simulator_initializes_input_in_y_basis() -> None:
    """PatternSimulator initializes Y-axis inputs as |Y+>."""
    pattern, _ = _single_output_pattern(measured=False, init_axis=Axis.Y)
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)

    simulator.simulate()

    np.testing.assert_allclose(simulator.state.state(), np.asarray([1.0, 1.0j]) / np.sqrt(2))


def test_pattern_simulator_initializes_input_in_z_basis() -> None:
    """PatternSimulator initializes Z-axis inputs as |0>."""
    pattern, _ = _single_output_pattern(measured=False, init_axis=Axis.Z)
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)

    simulator.simulate()

    np.testing.assert_allclose(simulator.state.state(), np.asarray([1.0, 0.0]))


def test_pattern_simulator_reorders_mixed_input_axes_by_logical_qindex() -> None:
    """Mixed input states are returned in logical output-qubit order."""
    graph = GraphState()
    y_input = graph.add_node()
    z_input = graph.add_node()
    graph.register_input(y_input, 1, init=Initialization(axis=Axis.Y))
    graph.register_input(z_input, 0, init=Initialization(axis=Axis.Z))
    graph.register_output(y_input, 1)
    graph.register_output(z_input, 0)
    pattern = Pattern(
        input_node_indices=graph.input_node_indices,
        output_node_indices=graph.output_node_indices,
        commands=(),
        pauli_frame=PauliFrame(graph, xflow={}, zflow={}),
        input_initializations=graph.input_initializations,
    )
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)

    simulator.simulate()

    expected = np.asarray([1.0, 1.0j, 0.0, 0.0], dtype=np.complex128).reshape(2, 2) / np.sqrt(2)
    assert simulator.state.state().shape == (2, 2)
    assert simulator.state.state().dtype == np.complex128
    np.testing.assert_allclose(simulator.state.state(), expected)


def test_pattern_simulator_samples_non_output_from_exact_probability_by_default() -> None:
    """Non-output measurements use the current state instead of a 50/50 assumption."""
    pattern, input_node = _deterministic_non_output_measurement_pattern()
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)

    simulator.simulate(rng=np.random.default_rng(2))

    assert simulator.results == {input_node: False}
    np.testing.assert_allclose(simulator.state.state(), np.asarray([1.0, 1.0]) / np.sqrt(2))


def test_pattern_simulator_samples_y_initialized_non_output_exactly() -> None:
    """A Y-initialized input measured in Y has a deterministic positive outcome."""
    graph = GraphState()
    input_node = graph.add_node()
    graph.register_input(input_node, 0, init=Initialization(axis=Axis.Y))
    graph.assign_meas_basis(input_node, AxisMeasBasis(Axis.Y, Sign.PLUS))
    pattern = Pattern(
        input_node_indices=graph.input_node_indices,
        output_node_indices={},
        commands=(M(input_node, graph.meas_bases[input_node]),),
        pauli_frame=PauliFrame(graph, xflow={}, zflow={}),
        input_initializations=graph.input_initializations,
    )
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)

    simulator.simulate(rng=np.random.default_rng(2))

    assert simulator.results == {input_node: False}
    np.testing.assert_allclose(simulator.state.state(), np.asarray(1.0))


def test_pattern_simulator_can_use_legacy_uniform_non_output_sampling() -> None:
    """calc_prob=False preserves the legacy 50/50 non-output sampling behavior."""
    pattern, input_node = _deterministic_non_output_measurement_pattern()
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector, calc_prob=False)

    simulator.simulate(rng=np.random.default_rng(2))

    assert simulator.results == {input_node: True}
    np.testing.assert_allclose(simulator.state.state(), np.asarray([1.0, -1.0]) / np.sqrt(2))


def test_pattern_simulator_applies_output_x_frame_to_statevector() -> None:
    """An unmeasured output statevector should include pending X frame corrections."""
    pattern, node = _single_output_pattern(measured=False)
    pattern.pauli_frame.x_flip(node)
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    simulator.state = StateVector([1.0, 0.0])

    simulator.simulate()

    np.testing.assert_allclose(simulator.state.state(), np.asarray([0.0, 1.0]))


def test_pattern_simulator_applies_output_z_frame_to_statevector() -> None:
    """An unmeasured output statevector should include pending Z frame corrections."""
    pattern, node = _single_output_pattern(measured=False)
    pattern.pauli_frame.z_flip(node)
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    simulator.state = StateVector([1 / np.sqrt(2), 1 / np.sqrt(2)])

    simulator.simulate()

    np.testing.assert_allclose(simulator.state.state(), np.asarray([1 / np.sqrt(2), -1 / np.sqrt(2)]))


def test_pattern_simulator_measures_output_after_pauli_frame() -> None:
    """Output measurement results should be reported after applying the output Pauli frame."""
    pattern, node = _single_output_pattern(measured=True)
    pattern.pauli_frame.x_flip(node)
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    simulator.state = StateVector([0.0, 1.0])

    simulator.simulate(rng=np.random.default_rng(123))

    assert simulator.results == {node: False}
    assert simulator.output_results == {0: False}


def test_pattern_simulator_measures_output_z_frame_in_x_basis() -> None:
    """A pending output Z frame should flip an X-basis output measurement result."""
    pattern, node = _single_output_pattern(measured=True, axis=Axis.X)
    pattern.pauli_frame.z_flip(node)
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    simulator.state = StateVector([1 / np.sqrt(2), 1 / np.sqrt(2)])

    simulator.simulate(rng=np.random.default_rng(123))

    assert simulator.results == {node: True}
    assert simulator.output_results == {0: True}


def test_pattern_simulator_reorders_remaining_outputs_after_terminal_measurement() -> None:
    """A measured output may leave sparse qindices for quantum outputs."""
    graph = GraphState()
    measured_node = graph.add_node()
    quantum_node = graph.add_node()
    graph.register_input(measured_node, 0)
    graph.register_input(quantum_node, 1)
    graph.register_output(measured_node, 0)
    graph.register_output(quantum_node, 1)
    graph.assign_meas_basis(measured_node, AxisMeasBasis(Axis.Z, Sign.PLUS))
    pattern = Pattern(
        input_node_indices=graph.input_node_indices,
        output_node_indices=graph.output_node_indices,
        commands=(M(measured_node, AxisMeasBasis(Axis.Z, Sign.PLUS)),),
        pauli_frame=PauliFrame(graph, xflow={}, zflow={}),
    )
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)

    simulator.simulate(rng=np.random.default_rng(3))

    assert set(simulator.output_results) == {0}
    np.testing.assert_allclose(simulator.state.state(), np.asarray([1, 1]) / np.sqrt(2))


def test_pattern_simulator_reorders_outputs_by_qindex_rank() -> None:
    """A non-self-inverse output permutation should preserve qindex ordering."""
    graph = GraphState()
    for qindex in (2, 0, 1):
        node = graph.add_node()
        graph.register_input(node, qindex)
        graph.register_output(node, qindex)
    pattern = Pattern(
        input_node_indices=graph.input_node_indices,
        output_node_indices=graph.output_node_indices,
        commands=(),
        pauli_frame=PauliFrame(graph, xflow={}, zflow={}),
    )
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    initial_state = np.zeros((2, 2, 2), dtype=np.complex128)
    initial_state[0, 1, 0] = 1
    simulator.state = StateVector(initial_state)

    simulator.simulate()

    expected_state = np.zeros((2, 2, 2), dtype=np.complex128)
    expected_state[1, 0, 0] = 1
    np.testing.assert_allclose(simulator.state.state(), expected_state)
