"""Test circuit module."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import typing_extensions

from graphqomb.circuit import BaseCircuit, Circuit, CircuitScheduleStrategy, MBQCCircuit, circuit2graph
from graphqomb.common import Plane, PlannerMeasBasis
from graphqomb.feedforward import pauli_simplification, signal_shifting
from graphqomb.gates import (
    CNOT,
    CZ,
    Gate,
    H,
    J,
    PhaseGadget,
    S,
    T,
    UnitGate,
    X,
    Y,
    Z,
)
from graphqomb.qompiler import qompile
from graphqomb.schedule_solver import ScheduleConfig, Strategy
from graphqomb.scheduler import Scheduler
from graphqomb.simulator import CircuitSimulator, PatternSimulator, SimulatorBackend

NON_PAULI_TEST_ANGLES = (0.2, 0.3, 0.7, 1.1, 0.25 * np.pi)


def _assert_circuit_pattern_equivalence(
    circuit: MBQCCircuit,
    *,
    measurement_seeds: tuple[int, ...] = (0, 1, 2),
) -> None:
    circ_simulator = CircuitSimulator(circuit, SimulatorBackend.StateVector)
    circ_simulator.simulate()
    circ_state = circ_simulator.state.state()

    for measurement_seed in measurement_seeds:
        graphstate, xflow, scheduler = circuit2graph(circuit)
        pattern = qompile(graphstate, xflow, scheduler=scheduler)

        simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
        simulator.simulate(rng=np.random.default_rng(measurement_seed))
        pattern_state = simulator.state.state()

        inner_product = np.vdot(pattern_state, circ_state)
        assert np.isclose(np.abs(inner_product), 1.0, atol=1e-9)


def _build_small_random_single_qubit_circuit(seed: int) -> MBQCCircuit:
    rng = np.random.default_rng(seed)
    num_instructions = int(rng.integers(2, 6))
    circuit = MBQCCircuit(1)

    for _ in range(num_instructions):
        circuit.j(0, float(rng.choice(NON_PAULI_TEST_ANGLES)))

    return circuit


# MBQCCircuit tests


def test_mbqc_circuit_init() -> None:
    """Test MBQCCircuit initialization."""
    circuit = MBQCCircuit(num_qubits=3)
    assert circuit.num_qubits == 3
    assert circuit.instructions() == []


def test_mbqc_circuit_j_gate() -> None:
    """Test adding J gate."""
    circuit = MBQCCircuit(num_qubits=2)
    circuit.j(qubit=0, angle=0.5)
    instructions = circuit.unit_instructions()
    assert len(instructions) == 1
    assert isinstance(instructions[0], J)
    assert instructions[0].qubit == 0
    assert math.isclose(instructions[0].angle, 0.5)


def test_mbqc_circuit_cz_gate() -> None:
    """Test adding CZ gate."""
    circuit = MBQCCircuit(num_qubits=3)
    circuit.cz(qubit1=0, qubit2=2)
    instructions = circuit.unit_instructions()
    assert len(instructions) == 1
    assert isinstance(instructions[0], CZ)
    assert instructions[0].qubits == (0, 2)


def test_mbqc_circuit_phase_gadget() -> None:
    """Test adding phase gadget."""
    circuit = MBQCCircuit(num_qubits=4)
    circuit.phase_gadget(qubits=[0, 1, 3], angle=0.25)
    instructions = circuit.unit_instructions()
    assert len(instructions) == 1
    assert isinstance(instructions[0], PhaseGadget)
    assert instructions[0].qubits == [0, 1, 3]
    assert math.isclose(instructions[0].angle, 0.25)


def test_mbqc_circuit_multiple_gates() -> None:
    """Test adding multiple gates."""
    circuit = MBQCCircuit(num_qubits=3)
    circuit.j(qubit=0, angle=0.5)
    circuit.cz(qubit1=0, qubit2=1)
    circuit.j(qubit=1, angle=-0.3)
    circuit.phase_gadget(qubits=[0, 1, 2], angle=0.25)
    circuit.cz(qubit1=1, qubit2=2)

    instructions = circuit.unit_instructions()
    assert len(instructions) == 5
    assert all(isinstance(inst, (J, CZ, PhaseGadget)) for inst in instructions)


def test_mbqc_circuit_instructions_returns_copy() -> None:
    """Test that instructions() returns a copy of the list."""
    circuit = MBQCCircuit(num_qubits=2)
    circuit.j(qubit=0, angle=0.5)
    circuit.cz(qubit1=0, qubit2=1)

    instructions1 = circuit.unit_instructions()
    instructions2 = circuit.unit_instructions()

    # Should be different list objects
    assert instructions1 is not instructions2
    # But with same contents
    assert instructions1 == instructions2


def test_mbqc_circuit_abstract_base_class() -> None:
    """Test that MBQCCircuit is an instance of BaseCircuit."""
    circuit = MBQCCircuit(num_qubits=2)
    assert isinstance(circuit, BaseCircuit)


# Circuit tests


def test_circuit_init() -> None:
    """Test Circuit initialization."""
    circuit = Circuit(num_qubits=3)
    assert circuit.num_qubits == 3
    assert circuit.unit_instructions() == []
    assert circuit.instructions() == []


def test_circuit_single_qubit_gates() -> None:
    """Test single qubit gate applications."""
    circuit = Circuit(num_qubits=2)
    circuit.apply_macro_gate(H(qubit=0))
    circuit.apply_macro_gate(X(qubit=1))
    circuit.apply_macro_gate(Y(qubit=0))
    circuit.apply_macro_gate(Z(qubit=1))
    circuit.apply_macro_gate(S(qubit=0))
    circuit.apply_macro_gate(T(qubit=1))

    macro_gates = circuit.instructions()
    assert len(macro_gates) == 6

    # Test that each macro gate type is preserved
    assert isinstance(macro_gates[0], H)
    assert isinstance(macro_gates[1], X)
    assert isinstance(macro_gates[2], Y)
    assert isinstance(macro_gates[3], Z)
    assert isinstance(macro_gates[4], S)
    assert isinstance(macro_gates[5], T)


def test_circuit_two_qubit_gates() -> None:
    """Test two qubit gate applications."""
    circuit = Circuit(num_qubits=3)
    circuit.apply_macro_gate(CNOT(qubits=(0, 1)))
    circuit.apply_macro_gate(CNOT(qubits=(1, 2)))

    macro_gates = circuit.instructions()
    assert len(macro_gates) == 2
    assert all(isinstance(gate, CNOT) for gate in macro_gates)


def test_circuit_instructions_expansion() -> None:
    """Test that unit_instructions() correctly expands macro gates."""
    circuit = Circuit(num_qubits=3)

    # Add various macro gates
    circuit.apply_macro_gate(H(qubit=0))  # 1 unit gate
    circuit.apply_macro_gate(CNOT(qubits=(0, 1)))  # 3 unit gates
    circuit.apply_macro_gate(X(qubit=2))  # 2 unit gates
    circuit.apply_macro_gate(Y(qubit=1))  # 4 unit gates
    circuit.apply_macro_gate(Z(qubit=0))  # 2 unit gates

    # Get expanded instructions
    instructions = circuit.unit_instructions()

    # Calculate expected total
    expected_count = 1 + 3 + 2 + 4 + 2
    assert len(instructions) == expected_count

    # Verify all are UnitGate instances
    assert all(isinstance(inst, (J, CZ, PhaseGadget)) for inst in instructions)

    # Test that instructions() returns macro gates
    macro_instructions = circuit.instructions()
    assert len(macro_instructions) == 5  # 5 macro gates
    assert all(isinstance(inst, Gate) for inst in macro_instructions)


def test_circuit_instructions_matches_manual_expansion() -> None:
    """Test that instructions() matches manual expansion using chain."""
    circuit = Circuit(num_qubits=2)

    # Add some gates
    circuit.apply_macro_gate(H(qubit=0))
    circuit.apply_macro_gate(T(qubit=0))
    circuit.apply_macro_gate(CNOT(qubits=(0, 1)))
    circuit.apply_macro_gate(S(qubit=1))

    # Get unit instructions from method
    instructions = circuit.unit_instructions()

    # Manual expansion
    macro_gates = circuit.instructions()
    expected_instructions = list(itertools.chain.from_iterable(gate.unit_gates() for gate in macro_gates))

    assert len(instructions) == len(expected_instructions)
    # Compare each instruction
    for inst, expected in zip(instructions, expected_instructions, strict=False):
        assert type(inst) is type(expected)
        if isinstance(inst, J) and isinstance(expected, J):
            assert inst.qubit == expected.qubit
            assert inst.angle == expected.angle
        elif isinstance(inst, CZ) and isinstance(expected, CZ):
            assert inst.qubits == expected.qubits

    # Test that instructions() returns macro gates
    macro_instructions = circuit.instructions()
    assert len(macro_instructions) == 4  # 4 macro gates
    assert macro_instructions == circuit.instructions()


def test_circuit_empty_circuit_instructions() -> None:
    """Test empty circuit returns empty instructions."""
    circuit = Circuit(num_qubits=3)
    assert circuit.instructions() == []
    assert circuit.instructions() == []


def test_circuit_instructions_returns_copy() -> None:
    """Test that instructions() returns a copy."""
    circuit = Circuit(num_qubits=2)
    circuit.apply_macro_gate(H(qubit=0))
    circuit.apply_macro_gate(X(qubit=1))

    macro1 = circuit.instructions()
    macro2 = circuit.instructions()

    # Should be different list objects
    assert macro1 is not macro2
    # But with same contents
    assert len(macro1) == len(macro2)
    for g1, g2 in zip(macro1, macro2, strict=False):
        assert type(g1) is type(g2)


def test_circuit_abstract_base_class() -> None:
    """Test that Circuit is an instance of BaseCircuit."""
    circuit = Circuit(num_qubits=2)
    assert isinstance(circuit, BaseCircuit)


# circuit2graph tests


def test_circuit2graph_simple_circuit() -> None:
    """Test conversion of a simple circuit."""
    circuit = MBQCCircuit(num_qubits=2)
    circuit.j(qubit=0, angle=0.5)
    circuit.cz(qubit1=0, qubit2=1)
    circuit.j(qubit=1, angle=-0.3)

    graph, gflow, scheduler = circuit2graph(circuit)

    # Check graph properties
    assert len(graph.input_node_indices) == 2
    assert len(graph.output_node_indices) == 2
    # 2 input nodes + 2 J gate nodes = 4 total nodes
    assert len(graph.nodes) == 4

    # Check gflow
    assert len(gflow) == 2  # Two J gates should have gflow

    # Check scheduler
    assert isinstance(scheduler, Scheduler)


def test_circuit2graph_phase_gadget_circuit() -> None:
    """Test conversion with phase gadget."""
    circuit = MBQCCircuit(num_qubits=3)
    circuit.phase_gadget(qubits=[0, 1, 2], angle=0.25)

    graph, _, _ = circuit2graph(circuit)

    # Check graph properties
    assert len(graph.input_node_indices) == 3
    assert len(graph.output_node_indices) == 3
    # 3 input nodes + 1 phase gadget node = 4 total nodes
    assert len(graph.nodes) == 4

    # Check phase gadget node has correct measurement basis
    pg_nodes = [n for n in graph.nodes if n not in graph.input_node_indices]
    assert len(pg_nodes) == 1
    pg_node = pg_nodes[0]
    basis = graph.meas_bases[pg_node]
    assert isinstance(basis, PlannerMeasBasis)
    assert basis.plane == Plane.YZ
    assert math.isclose(basis.angle, 0.25)


def test_circuit2graph_empty_circuit() -> None:
    """Test conversion of empty circuit."""
    circuit = MBQCCircuit(num_qubits=2)
    graph, gflow, scheduler = circuit2graph(circuit)

    # Check graph properties
    assert len(graph.input_node_indices) == 2
    assert len(graph.output_node_indices) == 2
    assert len(graph.nodes) == 2  # Only input/output nodes
    assert len(gflow) == 0  # No gflow for empty circuit

    # Check scheduler
    assert isinstance(scheduler, Scheduler)


def test_circuit2graph_invalid_instruction() -> None:
    """Test that invalid instruction raises TypeError."""

    # Create a mock invalid circuit
    class MockCircuit(BaseCircuit):
        @property
        @typing_extensions.override
        def num_qubits(self) -> int:
            return 1

        @typing_extensions.override
        def instructions(self) -> list[Gate]:
            return [X(qubit=0)]

        @typing_extensions.override
        def unit_instructions(self) -> list[UnitGate]:
            # Return a non-UnitGate object to trigger error
            return [X(qubit=0)]  # type: ignore[list-item]  # ty: ignore[invalid-return-type]

    circuit = MockCircuit()
    with pytest.raises(TypeError, match="Invalid instruction"):
        circuit2graph(circuit)


def test_circuit2graph_complex_circuit() -> None:
    """Test conversion of a more complex circuit."""
    circuit = MBQCCircuit(num_qubits=4)

    # Build a circuit with multiple gate types
    circuit.j(qubit=0, angle=np.pi / 4)
    circuit.j(qubit=1, angle=np.pi / 2)
    circuit.cz(qubit1=0, qubit2=2)
    circuit.cz(qubit1=1, qubit2=3)
    circuit.phase_gadget(qubits=[2, 3], angle=0.5)
    circuit.j(qubit=2, angle=-np.pi / 4)
    circuit.j(qubit=3, angle=np.pi)

    graph, gflow, scheduler = circuit2graph(circuit)

    # Check basic properties
    assert len(graph.input_node_indices) == 4
    assert len(graph.output_node_indices) == 4

    # Count nodes: 4 inputs + 4 J nodes + 1 phase gadget = 9
    assert len(graph.nodes) == 9

    # Check gflow: 4 J gates + 1 phase gadget = 5 entries
    assert len(gflow) == 5

    # Check scheduler
    assert isinstance(scheduler, Scheduler)


def test_circuit2graph_measurement_basis_assignment() -> None:
    """Test that measurement bases are correctly assigned."""
    circuit = MBQCCircuit(num_qubits=2)
    circuit.j(qubit=0, angle=0.7)
    circuit.j(qubit=1, angle=-1.2)

    graph, _, _ = circuit2graph(circuit)

    # Find non-output nodes with measurement basis (J gates are applied to input nodes)
    measured_nodes = [
        (n, basis) for n in graph.nodes if n not in graph.output_node_indices and (basis := graph.meas_bases.get(n))
    ]

    assert len(measured_nodes) == 2
    for _node, basis in measured_nodes:
        assert isinstance(basis, PlannerMeasBasis)
        assert basis.plane == Plane.XY
        # Check angles match the J gate angles (negated)
        assert basis.angle in {-0.7, 1.2}


def test_circuit2graph_circuit_with_macro_gates() -> None:
    """Test conversion of Circuit with macro gates."""
    circuit = Circuit(num_qubits=2)
    circuit.apply_macro_gate(H(qubit=0))
    circuit.apply_macro_gate(CNOT(qubits=(0, 1)))

    graph, _, _ = circuit2graph(circuit)

    # Check that macro gates are properly expanded
    assert len(graph.input_node_indices) == 2
    assert len(graph.output_node_indices) == 2
    # H expands to 1 J, CNOT expands to 2 J + 1 CZ = 3 nodes total
    assert len(graph.nodes) == 5  # 2 inputs + 3 new nodes


# circuit2graph scheduling tests


def test_circuit2graph_returns_scheduler() -> None:
    """Test that circuit2graph returns a valid Scheduler object."""
    circuit = MBQCCircuit(num_qubits=2)
    circuit.j(qubit=0, angle=0.5)
    circuit.cz(qubit1=0, qubit2=1)

    graph, _gflow, scheduler = circuit2graph(circuit)

    assert isinstance(scheduler, Scheduler)
    assert scheduler.graph is graph


def test_circuit2graph_j_gate_timing() -> None:
    """Test that J gates are scheduled sequentially on the same qubit."""
    circuit = MBQCCircuit(num_qubits=1)
    circuit.j(qubit=0, angle=0.5)
    circuit.j(qubit=0, angle=0.3)
    circuit.j(qubit=0, angle=0.1)

    _graph, _gflow, scheduler = circuit2graph(circuit)

    # Check that measurement times are unique and ordered
    measure_times = [t for t in scheduler.measure_time.values() if t is not None]
    assert measure_times == sorted(measure_times)
    assert len(set(measure_times)) == len(measure_times)  # All unique


def test_circuit2graph_minimize_qubits_strategy_serializes() -> None:
    """Test that MINIMIZE_SPACE strategy serializes independent J gates."""
    circuit = MBQCCircuit(num_qubits=2)
    circuit.j(qubit=0, angle=0.1)
    circuit.j(qubit=1, angle=0.2)

    graph_parallel, _gflow_parallel, scheduler_parallel = circuit2graph(circuit)
    graph_min, _gflow_min, scheduler_min = circuit2graph(
        circuit,
        schedule_strategy=CircuitScheduleStrategy.MINIMIZE_SPACE,
    )

    parallel_input_nodes = list(graph_parallel.input_node_indices.keys())
    parallel_meas_times = [scheduler_parallel.measure_time[node] for node in parallel_input_nodes]
    assert all(time is not None for time in parallel_meas_times)
    parallel_meas_times_int = [time for time in parallel_meas_times if time is not None]
    assert sorted(parallel_meas_times_int) == [1, 1]

    min_input_nodes = list(graph_min.input_node_indices.keys())
    min_meas_times = [scheduler_min.measure_time[node] for node in min_input_nodes]
    assert all(time is not None for time in min_meas_times)
    min_meas_times_int = [time for time in min_meas_times if time is not None]
    assert sorted(min_meas_times_int) == [1, 2]

    scheduler_min.validate_schedule()


def test_circuit2graph_phase_gadget_timing() -> None:
    """Test that phase gadget has valid timing."""
    circuit = MBQCCircuit(num_qubits=3)
    circuit.j(qubit=0, angle=0.5)  # qubit 0 at timestep 1
    circuit.j(qubit=0, angle=0.3)  # qubit 0 at timestep 2
    circuit.phase_gadget(qubits=[0, 1, 2], angle=0.25)

    graph, _gflow, scheduler = circuit2graph(circuit)

    # Check that phase gadget node has valid timing
    pg_nodes = [n for n in graph.nodes if graph.meas_bases.get(n) and graph.meas_bases[n].plane == Plane.YZ]
    assert len(pg_nodes) == 1
    assert scheduler.prepare_time.get(pg_nodes[0]) is not None
    assert scheduler.measure_time.get(pg_nodes[0]) is not None

    # Phase gadget should be prepared at max timestep of involved qubits
    assert scheduler.prepare_time.get(pg_nodes[0]) == 2  # qubit 0 at timestep 2


def test_circuit2graph_schedule_is_valid() -> None:
    """Test that generated schedule passes validation."""
    circuit = MBQCCircuit(num_qubits=3)
    circuit.j(qubit=0, angle=0.5)
    circuit.cz(qubit1=0, qubit2=1)
    circuit.j(qubit=1, angle=0.3)
    circuit.cz(qubit1=1, qubit2=2)
    circuit.j(qubit=2, angle=0.1)

    _graph, _gflow, scheduler = circuit2graph(circuit)

    # This should not raise any exceptions
    scheduler.validate_schedule()


def test_signal_shifting_circuit_integration() -> None:
    """Test signal_shifting integration with circuit compilation and simulation."""
    # Create a simple quantum circuit
    circuit = MBQCCircuit(3)
    circuit.j(0, 0.5 * np.pi)
    circuit.cz(0, 1)
    circuit.cz(0, 2)
    circuit.j(1, 0.75 * np.pi)
    circuit.j(2, 0.25 * np.pi)
    circuit.cz(0, 2)
    circuit.cz(1, 2)

    # Convert circuit to graph and gflow
    graphstate, gflow, _ = circuit2graph(circuit)

    # Apply signal shifting
    xflow, zflow = signal_shifting(graphstate, gflow)

    # Compile to pattern
    pattern = qompile(graphstate, xflow, zflow)

    # Verify pattern is runnable
    assert pattern is not None
    assert pattern.max_space >= 0
    assert pattern.depth >= 0

    # Simulate the pattern
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    simulator.simulate()
    state = simulator.state
    statevec = state.state()

    # Compare with circuit simulator
    circ_simulator = CircuitSimulator(circuit, SimulatorBackend.StateVector)
    circ_simulator.simulate()
    circ_state = circ_simulator.state.state()
    inner_product = np.vdot(statevec, circ_state)

    # Verify that the results match (inner product should be close to 1)
    assert np.isclose(np.abs(inner_product), 1.0)


def test_pauli_simplification_circuit_integration() -> None:
    """Test pauli_simplification integration with circuit compilation and simulation."""
    # Create a quantum circuit (using j for rotations, cz for entanglement)
    circuit = MBQCCircuit(2)
    circuit.j(0, 0.5 * np.pi)  # Rotation on qubit 0
    circuit.cz(0, 1)
    circuit.j(1, 0.25 * np.pi)  # Rotation on qubit 1

    # Convert circuit to graph and gflow
    graphstate, gflow, _ = circuit2graph(circuit)

    # Apply pauli simplification
    xflow, zflow = pauli_simplification(graphstate, gflow)

    # Compile to pattern
    pattern = qompile(graphstate, xflow, zflow)

    # Verify pattern is runnable
    assert pattern is not None
    assert pattern.max_space >= 0

    # Simulate the pattern
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    simulator.simulate()
    state = simulator.state
    statevec = state.state()

    # Compare with circuit simulator
    circ_simulator = CircuitSimulator(circuit, SimulatorBackend.StateVector)
    circ_simulator.simulate()
    circ_state = circ_simulator.state.state()
    inner_product = np.vdot(statevec, circ_state)

    # Verify that the results match (inner product should be close to 1)
    assert np.isclose(np.abs(inner_product), 1.0)


def test_non_pauli_j_angles_circuit_pattern_equivalence_regression() -> None:
    """Non-Pauli J angles should still compile to a pattern with the same output state."""
    circuit = MBQCCircuit(1)
    circuit.j(0, 0.2)
    circuit.j(0, 0.3)

    _assert_circuit_pattern_equivalence(circuit)


def test_simple_entangling_circuit_pattern_equivalence() -> None:
    """A small entangling circuit should preserve the output state after compilation."""
    circuit = MBQCCircuit(2)
    circuit.j(0, 0.2)
    circuit.cz(0, 1)
    circuit.j(1, 0.3)

    _assert_circuit_pattern_equivalence(circuit)


def test_pure_cz_circuit_pattern_equivalence() -> None:
    """A CZ-only circuit should preserve the output state after compilation."""
    circuit = MBQCCircuit(2)
    circuit.cz(0, 1)

    _assert_circuit_pattern_equivalence(circuit)


@pytest.mark.parametrize("seed", range(6))
def test_small_random_single_qubit_circuit_pattern_equivalence(seed: int) -> None:
    """Small random single-qubit J chains should preserve the circuit output state after compilation."""
    circuit = _build_small_random_single_qubit_circuit(seed)

    _assert_circuit_pattern_equivalence(circuit)


def test_circuit2graph_single_qubit_no_gates() -> None:
    """Test single qubit circuit with no gates."""
    circuit = MBQCCircuit(num_qubits=1)

    graph, gflow, scheduler = circuit2graph(circuit)

    assert len(graph.nodes) == 1
    assert len(gflow) == 0
    assert isinstance(scheduler, Scheduler)


def test_circuit2graph_multiple_parallel_qubits() -> None:
    """Test circuit with operations on multiple independent qubits."""
    circuit = MBQCCircuit(num_qubits=4)
    circuit.j(qubit=0, angle=0.1)
    circuit.j(qubit=1, angle=0.2)
    circuit.j(qubit=2, angle=0.3)
    circuit.j(qubit=3, angle=0.4)

    _graph, _gflow, scheduler = circuit2graph(circuit)

    # All qubits should have valid schedules
    scheduler.validate_schedule()


def test_circuit2graph_deep_circuit() -> None:
    """Test circuit with many sequential operations."""
    circuit = MBQCCircuit(num_qubits=2)
    for i in range(10):
        circuit.j(qubit=0, angle=0.1 * i)
        circuit.cz(qubit1=0, qubit2=1)
        circuit.j(qubit=1, angle=0.1 * i)

    graph, _gflow, scheduler = circuit2graph(circuit)

    # Verify schedule is valid
    scheduler.validate_schedule()

    # Check expected number of nodes: 2 input + 20 J gates
    assert len(graph.nodes) == 22


def test_circuit2graph_scheduler_can_resolve_with_different_strategy() -> None:
    """Test that scheduler can be re-solved with different optimization strategy."""
    circuit = MBQCCircuit(num_qubits=3)
    circuit.j(qubit=0, angle=0.5)
    circuit.cz(qubit1=0, qubit2=1)
    circuit.j(qubit=1, angle=0.3)
    circuit.cz(qubit1=1, qubit2=2)
    circuit.j(qubit=2, angle=0.1)

    _graph, _gflow, scheduler = circuit2graph(circuit)

    # Re-solve with different strategy
    config = ScheduleConfig(Strategy.MINIMIZE_SPACE)
    result = scheduler.solve_schedule(config)

    # solve_schedule should return True on success
    assert result is True

    # The schedule should still have valid prepare/measure times
    assert all(t is not None for t in scheduler.prepare_time.values())
    assert all(t is not None for t in scheduler.measure_time.values())
