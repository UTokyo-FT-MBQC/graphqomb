"""Tests for importing supported Stim circuits into GraphQOMB patterns."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import numpy as np
import pytest

from graphqomb.command import M
from graphqomb.common import Axis, AxisMeasBasis, Sign
from graphqomb.graphstate import odd_neighbors
from graphqomb.pattern import is_runnable
from graphqomb.qec.qeccode import YFoliation
from graphqomb.simulator import PatternSimulator, SimulatorBackend
from graphqomb.statevec import StateVector
from graphqomb.stim_compiler import stim_compile
from graphqomb.stim_importer import stim_circuit_to_pattern, stim_file_to_pattern, stim_text_to_pattern

stim = pytest.importorskip("stim")

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def test_stim_text_to_pattern_imports_unitary_clifford_block() -> None:
    result = stim_text_to_pattern(
        """
        H 10
        CX 10 12
        S_DAG 12
        """
    )

    assert result.stim_to_qubit == {10: 0, 12: 1}
    assert result.qubit_to_stim == {0: 10, 1: 12}
    assert result.mpp_extractions == ()
    assert set(result.pattern.input_node_indices.values()) == {0, 1}
    assert set(result.pattern.output_node_indices.values()) == {0, 1}


@pytest.mark.parametrize(
    "gate_name",
    [name for name in sorted(stim.gate_data()) if stim.gate_data(name).is_unitary and name not in {"SPP", "SPP_DAG"}],
)
def test_stim_circuit_to_pattern_imports_every_fixed_clifford_gate(gate_name: str) -> None:
    """Test every fixed Stim Clifford gate through parser and importer integration."""
    gate_data = stim.gate_data(gate_name)
    targets = [0] if gate_data.is_single_qubit_gate else [0, 1]
    source = stim.Circuit()
    source.append(gate_name, targets)

    result = stim_circuit_to_pattern(source)

    assert set(result.pattern.input_node_indices.values()) == set(targets)
    assert set(result.pattern.output_node_indices.values()) == set(targets)


@pytest.mark.parametrize("gate_name", ["SPP", "SPP_DAG"])
def test_stim_text_to_pattern_imports_pauli_product_rotations(gate_name: str) -> None:
    result = stim_text_to_pattern(f"{gate_name} X0*Y1*!Z2")

    assert set(result.pattern.input_node_indices.values()) == {0, 1, 2}
    assert set(result.pattern.output_node_indices.values()) == {0, 1, 2}


@pytest.mark.parametrize("gate_name", ["SPP", "SPP_DAG"])
def test_stim_text_to_pattern_imports_repeated_qubit_pauli_products(gate_name: str) -> None:
    result = stim_text_to_pattern(f"{gate_name} X0*X0*Z1")

    assert result.stim_to_qubit == {0: 0, 1: 1}
    assert set(result.pattern.input_node_indices.values()) == {0, 1}
    assert set(result.pattern.output_node_indices.values()) == {0, 1}


def test_stim_text_to_pattern_cancels_repeated_cz_in_one_tick_block() -> None:
    result = stim_text_to_pattern(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        CZ 0 1
        CZ 0 1
        """
    )
    graph = result.pattern.pauli_frame.graphstate
    input_coordinates = {qubit: graph.coordinates[node] for node, qubit in graph.input_node_indices.items()}

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 0
    assert input_coordinates == {0: (0.0, 0.0, 0.0), 1: (1.0, 0.0, 0.0)}


def test_stim_text_to_pattern_does_not_advance_z_for_cancelled_single_qubit_block() -> None:
    result = stim_text_to_pattern("QUBIT_COORDS(0, 0) 0\nH 0\nH 0")
    graph = result.pattern.pauli_frame.graphstate
    input_node = next(iter(graph.input_node_indices))

    assert graph.number_of_nodes() == 1
    assert graph.coordinates[input_node] == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("instruction", "has_x_correction", "has_z_correction"),
    [
        ("CNOT rec[-1] 1", True, False),
        ("CY rec[-1] 1", True, True),
        ("CZ rec[-1] 1", False, True),
        ("CZ 1 rec[-1]", False, True),
        ("XCZ 1 rec[-1]", True, False),
        ("YCZ 1 rec[-1]", True, True),
    ],
)
def test_stim_text_to_pattern_imports_classically_controlled_pauli_corrections(
    instruction: str,
    has_x_correction: bool,
    has_z_correction: bool,
) -> None:
    result = stim_text_to_pattern(f"M 0\nTICK\n{instruction}")
    frame = result.pattern.pauli_frame
    source = next(node for node, qubit in result.pattern.output_node_indices.items() if qubit == 0)
    target = next(node for node, qubit in result.pattern.output_node_indices.items() if qubit == 1)

    assert frame.xflow.get(source, set()) == ({target} if has_x_correction else set())
    assert frame.zflow.get(source, set()) == ({target} if has_z_correction else set())


def test_stim_text_to_pattern_feedback_x_adds_deferred_z_on_future_neighbors() -> None:
    result = stim_text_to_pattern(
        """
        MPP X0
        DETECTOR rec[-1]
        TICK
        CNOT rec[-1] 1
        TICK
        H 1
        """
    )
    frame = result.pattern.pauli_frame
    source = next(iter(frame.parity_check_group[0]))
    target = next(iter(frame.xflow[source]))
    output = next(node for node, qubit in result.pattern.output_node_indices.items() if qubit == 1)

    assert target != output
    assert odd_neighbors({target}, frame.graphstate) == {output}
    assert frame.zflow.get(source, set()) == {output}


def test_stim_text_to_pattern_feedback_x_skips_z_on_past_neighbors() -> None:
    result = stim_text_to_pattern(
        """
        MPP X0
        DETECTOR rec[-1]
        TICK
        H 1
        TICK
        CNOT rec[-1] 1
        TICK
        H 1
        """
    )
    frame = result.pattern.pauli_frame
    source = next(iter(frame.parity_check_group[0]))
    target = next(iter(frame.xflow[source]))
    output = next(node for node, qubit in result.pattern.output_node_indices.items() if qubit == 1)
    past_neighbors = frame.graphstate.neighbors(target) - {output}

    assert past_neighbors
    assert frame.zflow.get(source, set()) == {output}


def test_stim_text_to_pattern_feedback_before_entanglement_keeps_detectors_deterministic() -> None:
    result = stim_text_to_pattern(
        """
        R 1
        RX 2
        M 0
        TICK
        CX rec[-1] 1
        TICK
        CZ 1 2
        TICK
        MX 2
        M 1
        DETECTOR rec[-2] rec[-3]
        DETECTOR rec[-1] rec[-3]
        """
    )
    frame = result.pattern.pauli_frame

    assert frame.detector_determinism() == [True, True]
    exported = stim.Circuit(stim_compile(result.pattern))
    # Raises if the exported circuit contains a non-deterministic detector.
    exported.detector_error_model()


def test_stim_text_to_pattern_imports_batched_feedback_pairs_by_parity() -> None:
    result = stim_text_to_pattern(
        """
        MPP X0
        DETECTOR rec[-1]
        TICK
        CX rec[-1] 1 rec[-1] 2 rec[-1] 2
        """
    )
    frame = result.pattern.pauli_frame
    source = next(iter(frame.parity_check_group[0]))
    target = next(node for node, qubit in result.pattern.output_node_indices.items() if qubit == 1)

    assert frame.xflow[source] == {target}
    assert frame.zflow.get(source, set()) == set()


def test_stim_text_to_pattern_schedules_direct_measurement_feedback_source_causally() -> None:
    result = stim_text_to_pattern("M 0\nTICK\nCX rec[-1] 1\nTICK\nH 1")
    pattern = result.pattern

    is_runnable(pattern)
    source = next(node for node, qubit in pattern.output_node_indices.items() if qubit == 0)
    target = next(iter(pattern.pauli_frame.xflow[source]))
    measured_order = [cmd.node for cmd in pattern if isinstance(cmd, M)]

    assert measured_order.index(source) < measured_order.index(target)


def test_stim_text_to_pattern_applies_direct_measurement_feedback_in_simulation() -> None:
    rng = np.random.default_rng(7)
    for _ in range(20):
        result = stim_text_to_pattern("R 1\nM 0\nTICK\nCX rec[-1] 1\nTICK\nM 1")
        simulator = PatternSimulator(result.pattern, SimulatorBackend.StateVector)
        simulator.simulate(rng=rng)
        source = next(node for node, qubit in result.pattern.output_node_indices.items() if qubit == 0)
        target = next(node for node, qubit in result.pattern.output_node_indices.items() if qubit == 1)

        # q1 is reset to |0>, then X^{m0} is applied, so M1 must equal M0.
        assert simulator.results[source] == simulator.results[target]


def test_stim_text_to_pattern_splits_adjacent_feedback_and_unitary_operations() -> None:
    stim_text = "RX 0\nR 1\nM 0\nTICK\nCX rec[-1] 1\nH 1\nTICK\nMX 1"

    for seed in range(8):
        result = stim_text_to_pattern(stim_text)
        source = next(node for node, qubit in result.pattern.output_node_indices.items() if qubit == 0)
        target = next(node for node, qubit in result.pattern.output_node_indices.items() if qubit == 1)
        simulator = PatternSimulator(result.pattern, SimulatorBackend.StateVector)

        simulator.simulate(rng=np.random.default_rng(seed))

        # H maps X^{m0}|0> to the X-basis state with outcome m0.
        assert simulator.results[target] == simulator.results[source]


def test_stim_text_to_pattern_accepts_feedback_fused_with_quantum_pairs() -> None:
    """``CX rec[-1] 1`` fused with a plain ``CX 2 3`` splits into feedback and unitary parts."""
    result = stim_text_to_pattern("M 0\nTICK\nCX rec[-1] 1 2 3")

    assert set(result.pattern.output_node_indices.values()) == {0, 1, 2, 3}
    simulator = PatternSimulator(result.pattern, SimulatorBackend.StateVector)
    simulator.simulate(rng=np.random.default_rng(3))


def test_stim_text_to_pattern_rejects_feedback_before_any_measurement_record() -> None:
    with pytest.raises(ValueError, match="before the beginning of time"):
        stim_text_to_pattern("CX rec[-1] 0")


def test_stim_text_to_pattern_preserves_unitary_semantics_across_ticks() -> None:
    initial = np.asarray([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
    expected = np.asarray([1 + 1j, 1 - 1j], dtype=np.complex128) / 2

    for seed in range(8):
        pattern = stim_text_to_pattern("S 0\nTICK\nH 0\n").pattern
        simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
        simulator.state = StateVector(initial)
        simulator.simulate(rng=np.random.default_rng(seed))

        overlap = np.vdot(expected, simulator.state.state())
        assert np.isclose(abs(overlap), 1.0, atol=1e-9)


@pytest.mark.parametrize(
    ("stim_text", "expected"),
    [
        ("SQRT_Y 0", [0.0, 1.0]),
        ("C_XYZ 0", [(1 - 1j) / 2, (1 + 1j) / 2]),
    ],
)
def test_stim_text_to_pattern_preserves_negative_measurement_gate_semantics(
    stim_text: str, expected: list[complex]
) -> None:
    """Test the X- and Y- measurement basis gates end to end on a |+> input."""
    initial = np.asarray([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
    expected_state = np.asarray(expected, dtype=np.complex128)

    for seed in range(8):
        pattern = stim_text_to_pattern(stim_text).pattern
        simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
        simulator.state = StateVector(initial)
        simulator.simulate(rng=np.random.default_rng(seed))

        overlap = np.vdot(expected_state, simulator.state.state())
        assert np.isclose(abs(overlap), 1.0, atol=1e-9)


def test_stim_text_to_pattern_preserves_sparse_qubit_coordinates() -> None:
    result = stim_text_to_pattern(
        """
        QUBIT_COORDS(1, 2) 10
        QUBIT_COORDS(3) 99
        QUBIT_COORDS(4) 99
        H 99
        """
    )

    assert result.stim_to_qubit == {10: 0, 99: 1}
    assert result.pattern.input_coordinates
    assert set(result.pattern.input_coordinates.values()) == {(1.0, 2.0, 1.0), (3.0, 4.0, 0.0)}


def test_stim_text_to_pattern_aligns_parallel_gate_outputs_with_different_depths() -> None:
    result = stim_text_to_pattern(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        H 0
        S 1
        """
    )
    graph = result.pattern.pauli_frame.graphstate
    output_coordinates = {qubit: graph.coordinates[node] for node, qubit in graph.output_node_indices.items()}
    lane_0_z = sorted(coord[2] for coord in graph.coordinates.values() if np.isclose(coord[0], 0.0))
    lane_1_z = sorted(coord[2] for coord in graph.coordinates.values() if np.isclose(coord[0], 1.0))

    assert output_coordinates == {0: (0.0, 0.0, 2.0), 1: (1.0, 0.0, 2.0)}
    assert lane_0_z == [0.0, 2.0]
    assert lane_1_z == [0.0, 1.0, 2.0]


def test_stim_text_to_pattern_relocates_idle_input_without_adding_a_wire_node() -> None:
    result = stim_text_to_pattern(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        H 0
        """
    )
    graph = result.pattern.pauli_frame.graphstate
    idle_input = next(node for node, qubit in graph.input_node_indices.items() if qubit == 1)
    idle_output = next(node for node, qubit in graph.output_node_indices.items() if qubit == 1)

    assert graph.number_of_nodes() == 3
    assert idle_input == idle_output
    assert graph.coordinates[idle_input] == (1.0, 0.0, 1.0)
    assert graph.neighbors(idle_input) == set()


@pytest.mark.parametrize(
    ("instruction", "initial", "expected"),
    [
        ("H 0", [1, 0], [1 / np.sqrt(2), 1 / np.sqrt(2)]),
        ("S 0", [1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), 1j / np.sqrt(2)]),
        ("S_DAG 0", [1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1j / np.sqrt(2)]),
        ("X 0", [1, 0], [0, 1]),
        ("Y 0", [1, 0], [0, 1j]),
        ("Z 0", [1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]),
    ],
)
def test_stim_text_to_pattern_preserves_supported_single_qubit_gates(
    instruction: str,
    initial: list[complex],
    expected: list[complex],
) -> None:
    pattern = stim_text_to_pattern(instruction).pattern
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    simulator.state = StateVector(initial)

    simulator.simulate(rng=np.random.default_rng(3))

    assert np.isclose(abs(np.vdot(expected, simulator.state.state())), 1.0, atol=1e-9)


@pytest.mark.parametrize(
    ("instruction", "initial", "expected"),
    [
        ("CX 0 1", [0, 0, 1, 0], [0, 0, 0, 1]),
        ("CZ 0 1", [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, -0.5]),
        ("SWAP 0 1", [0, 0, 1, 0], [0, 1, 0, 0]),
    ],
)
def test_stim_text_to_pattern_preserves_supported_two_qubit_gates(
    instruction: str,
    initial: list[complex],
    expected: list[complex],
) -> None:
    pattern = stim_text_to_pattern(instruction).pattern
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    simulator.state = StateVector(initial)

    simulator.simulate(rng=np.random.default_rng(3))

    assert np.isclose(abs(np.vdot(expected, simulator.state.state())), 1.0, atol=1e-9)


def test_stim_text_to_pattern_imports_tick_separated_mpp_block() -> None:
    result = stim_text_to_pattern(
        """
        H 10
        TICK
        MPP X10*Z12
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(3) rec[-1]
        TICK
        CZ 10 12
        """
    )

    assert result.stim_to_qubit == {10: 0, 12: 1}
    assert len(result.mpp_extractions) == 1
    assert len(result.pattern.pauli_frame.parity_check_group) == 1
    assert set(result.pattern.pauli_frame.logical_observables) == {3}
    assert set(result.pattern.output_node_indices.values()) == {0, 1}


def test_stim_text_to_pattern_combines_commuting_mpp_instructions_in_one_tick_block() -> None:
    result = stim_text_to_pattern(
        """
        MPP X0
        DETECTOR rec[-1]
        MPP Z1
        DETECTOR rec[-1]
        """
    )

    assert len(result.mpp_extractions) == 1
    assert result.mpp_extractions[0].supports == (((0, "X"),), ((1, "Z"),))
    assert len(result.pattern.pauli_frame.parity_check_group) == 2


def test_stim_text_to_pattern_rejects_anticommuting_mpp_in_one_tick_block() -> None:
    with pytest.raises(ValueError, match="must commute"):
        stim_text_to_pattern("MPP X0\nMPP Z0")


@pytest.mark.parametrize(
    ("y_foliation", "expected_node_count"),
    [(YFoliation.TYPE_I, 27), (YFoliation.TYPE_II, 28)],
)
def test_stim_text_to_pattern_builds_commuting_mpp_block_at_common_z(
    y_foliation: YFoliation,
    expected_node_count: int,
) -> None:
    result = stim_text_to_pattern(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        QUBIT_COORDS(2, 0) 2
        QUBIT_COORDS(3, 0) 3
        QUBIT_COORDS(4, 0) 4
        QUBIT_COORDS(5, 0) 5
        QUBIT_COORDS(6, 0) 6
        MPP X0*X1*X4*X5
        DETECTOR rec[-1]
        MPP Z0*Z1*Z2*Z3
        DETECTOR rec[-1]
        MPP Y0*X2*Z4*Z6
        DETECTOR rec[-1]
        MPP Z4*Z5
        DETECTOR rec[-1]
        MPP X1*X3
        DETECTOR rec[-1]
        MPP Z2*X6
        DETECTOR rec[-1]
        """,
        y_foliation=y_foliation,
    )
    graph = result.pattern.pauli_frame.graphstate
    z_coordinates = {coordinate[2] for coordinate in graph.coordinates.values()}

    assert len(result.mpp_extractions) == 1
    assert len(result.mpp_extractions[0].supports) == 6
    assert graph.number_of_nodes() == expected_node_count
    assert np.isclose(min(z_coordinates), 0.0)
    assert np.isclose(max(z_coordinates), 2.0)
    assert set(result.pattern.input_node_indices.values()) == set(range(7))
    assert set(result.pattern.output_node_indices.values()) == set(range(7))
    mixed_check_ancilla = next(iter(result.pattern.pauli_frame.parity_check_group[2]))
    mixed_loop_ancilla = next(iter(result.pattern.pauli_frame.parity_check_group[5]))
    assert graph.has_edge(mixed_check_ancilla, mixed_loop_ancilla)


def test_stim_text_to_pattern_advances_z_once_per_mpp_tick_block() -> None:
    result = stim_text_to_pattern(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        MPP X0
        MPP X1
        TICK
        MPP X0*X1
        """
    )
    graph = result.pattern.pauli_frame.graphstate
    z_coordinates = {coordinate[2] for coordinate in graph.coordinates.values()}

    assert len(result.mpp_extractions) == 2
    assert np.isclose(max(z_coordinates), 4.0)


def test_stim_text_to_pattern_relocates_idle_input_to_mpp_output_layer() -> None:
    result = stim_text_to_pattern(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        MPP X0
        """
    )
    graph = result.pattern.pauli_frame.graphstate
    idle_input = next(node for node, qubit in graph.input_node_indices.items() if qubit == 1)
    idle_output = next(node for node, qubit in graph.output_node_indices.items() if qubit == 1)
    active_output = next(node for node, qubit in graph.output_node_indices.items() if qubit == 0)

    assert idle_input == idle_output
    assert graph.coordinates[idle_input] == (1.0, 0.0, 2.0)
    assert graph.coordinates[active_output] == (0.0, 0.0, 2.0)
    assert graph.neighbors(idle_input) == set()


def test_stim_text_to_pattern_derives_complete_zflow_from_xflow() -> None:
    result = stim_text_to_pattern("H 0\nTICK\nMPP X0")
    frame = result.pattern.pauli_frame

    assert frame.zflow == {node: odd_neighbors(targets, frame.graphstate) for node, targets in frame.xflow.items()}


def test_stim_text_to_pattern_excludes_mpp_ancilla_from_xflow() -> None:
    result = stim_text_to_pattern("MPP X0\nDETECTOR rec[-1]")
    frame = result.pattern.pauli_frame

    ancilla_nodes = frame.parity_check_group[0]
    assert len(ancilla_nodes) == 1
    assert set(frame.xflow) == set(frame.graphstate.meas_bases) - ancilla_nodes
    assert all(targets.isdisjoint(ancilla_nodes) for targets in frame.xflow.values())


def test_stim_text_to_pattern_appends_output_after_type_i_mpp_measurements() -> None:
    result = stim_text_to_pattern("MPP X0")
    graph = result.pattern.pauli_frame.graphstate

    assert graph.number_of_nodes() == 4
    assert len(graph.meas_bases) == 3
    assert len(graph.output_node_indices) == 1
    assert next(iter(graph.output_node_indices)) not in graph.meas_bases


def test_stim_text_to_pattern_appends_output_after_type_ii_y_measurements() -> None:
    result = stim_text_to_pattern("MPP Y0", y_foliation=YFoliation.TYPE_II)
    graph = result.pattern.pauli_frame.graphstate
    y_measurements = [
        basis for basis in graph.meas_bases.values() if isinstance(basis, AxisMeasBasis) and basis.axis == Axis.Y
    ]

    assert graph.number_of_nodes() == 5
    assert len(y_measurements) == 3
    assert len(graph.output_node_indices) == 1
    assert next(iter(graph.output_node_indices)) not in graph.meas_bases


def test_stim_import_entry_points_accept_type_ii_foliation(tmp_path: Path) -> None:
    stim_path = tmp_path / "y_measurement.stim"
    stim_path.write_text("MPP Y0", encoding="utf-8")

    circuit_result = stim_circuit_to_pattern(stim.Circuit("MPP Y0"), y_foliation=YFoliation.TYPE_II)
    file_result = stim_file_to_pattern(stim_path, y_foliation=YFoliation.TYPE_II)

    for result in (circuit_result, file_result):
        axes = [
            basis.axis
            for basis in result.pattern.pauli_frame.graphstate.meas_bases.values()
            if isinstance(basis, AxisMeasBasis)
        ]
        assert axes.count(Axis.Y) == 3


@pytest.mark.parametrize(
    ("instruction", "expected_axis"),
    [
        ("M 10", Axis.Z),
        ("MZ 10", Axis.Z),
        ("MX 10", Axis.X),
        ("MY 10", Axis.Y),
    ],
)
def test_stim_text_to_pattern_imports_single_qubit_pauli_measurements(
    instruction: str,
    expected_axis: Axis,
) -> None:
    result = stim_text_to_pattern(instruction)

    measurements = [command for command in result.pattern.commands if isinstance(command, M)]

    assert result.mpp_extractions == ()
    assert result.stim_to_qubit == {10: 0}
    assert result.pattern.input_node_indices == {0: 0}
    assert result.pattern.output_node_indices == {0: 0}
    assert len(measurements) == 1
    assert measurements[0].node == 0
    assert isinstance(measurements[0].meas_basis, AxisMeasBasis)
    assert measurements[0].meas_basis.axis == expected_axis
    assert measurements[0].meas_basis.sign == Sign.PLUS


def test_stim_text_to_pattern_assigns_single_measurement_to_existing_wire_node() -> None:
    result = stim_text_to_pattern("QUBIT_COORDS(1, 2) 10\nH 10\nTICK\nMX 10")
    output_node = next(node for node, qubit in result.pattern.output_node_indices.items() if qubit == 0)
    output_measurements = [
        command for command in result.pattern.commands if isinstance(command, M) and command.node == output_node
    ]

    assert result.mpp_extractions == ()
    assert len(output_measurements) == 1
    assert isinstance(output_measurements[0].meas_basis, AxisMeasBasis)
    assert output_measurements[0].meas_basis.axis == Axis.X
    assert output_measurements[0].meas_basis.sign == Sign.PLUS
    assert result.pattern.pauli_frame.graphstate.coordinates[output_node] == (1.0, 2.0, 1.0)


def test_stim_text_to_pattern_preserves_mpp_lane_coordinate_for_terminal_measurement() -> None:
    result = stim_text_to_pattern(
        """
        QUBIT_COORDS(1, 2) 0
        MPP X0
        TICK
        MX 0
        """
    )
    graph = result.pattern.pauli_frame.graphstate
    output_node = next(node for node, qubit in result.pattern.output_node_indices.items() if qubit == 0)
    output_basis = graph.meas_bases[output_node]

    assert graph.number_of_nodes() == 4
    assert graph.coordinates[output_node] == (1.0, 2.0, 2.0)
    assert isinstance(output_basis, AxisMeasBasis)
    assert output_basis.axis == Axis.X


def test_stim_text_to_pattern_places_gate_after_mpp_at_next_z_layer() -> None:
    result = stim_text_to_pattern(
        """
        QUBIT_COORDS(1, 2) 0
        MPP X0
        TICK
        H 0
        TICK
        MX 0
        """
    )
    graph = result.pattern.pauli_frame.graphstate
    output_node = next(node for node, qubit in graph.output_node_indices.items() if qubit == 0)

    assert graph.coordinates[output_node] == (1.0, 2.0, 3.0)
    assert all(len(coord) == 3 for coord in graph.coordinates.values())
    assert {coord[2] for coord in graph.coordinates.values()} == {0.0, 1.0, 2.0, 3.0}


def test_stim_text_to_pattern_composes_gate_output_with_mpp_input_at_same_z() -> None:
    result = stim_text_to_pattern(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        H 0
        TICK
        MPP X0*Z1
        """
    )
    graph = result.pattern.pauli_frame.graphstate
    input_coordinates = {qubit: graph.coordinates[node] for node, qubit in graph.input_node_indices.items()}
    output_coordinates = {qubit: graph.coordinates[node] for node, qubit in graph.output_node_indices.items()}

    assert input_coordinates == {0: (0.0, 0.0, 0.0), 1: (1.0, 0.0, 1.0)}
    assert output_coordinates == {0: (0.0, 0.0, 3.0), 1: (1.0, 0.0, 3.0)}
    assert {(coord[0], coord[2]) for coord in graph.coordinates.values()} >= {
        (0.0, 1.0),
        (1.0, 1.0),
    }


@pytest.mark.parametrize(
    ("instruction", "expected_axis"),
    [
        ("MXX 10 12", "X"),
        ("MYY 10 12", "Y"),
        ("MZZ 10 12", "Z"),
    ],
)
def test_stim_text_to_pattern_imports_pair_pauli_measurements(
    instruction: str,
    expected_axis: str,
) -> None:
    result = stim_text_to_pattern(instruction)

    assert result.mpp_extractions[0].supports == (((10, expected_axis), (12, expected_axis)),)
    assert result.stim_to_qubit == {10: 0, 12: 1}


def test_stim_text_to_pattern_preserves_multiple_measurement_results_in_target_order() -> None:
    result = stim_text_to_pattern(
        """
        M 0 2
        MXX 1 3 4 5
        DETECTOR rec[-4] rec[-1]
        """
    )

    assert result.mpp_extractions[0].supports == (
        ((1, "X"), (3, "X")),
        ((4, "X"), (5, "X")),
    )
    direct_measurements = [command for command in result.pattern.commands if isinstance(command, M)]
    assert (
        sum(
            isinstance(command.meas_basis, AxisMeasBasis) and command.meas_basis.axis == Axis.Z
            for command in direct_measurements
        )
        == 2
    )
    assert len(result.pattern.pauli_frame.parity_check_group) == 1
    assert len(result.pattern.pauli_frame.parity_check_group[0]) == 2


def test_stim_text_to_pattern_maps_m_and_mpp_records_to_one_detector() -> None:
    result = stim_text_to_pattern(
        """
        MPP X0
        X_ERROR(0.01) 7
        M 7
        DETECTOR rec[-2] rec[-1]
        """
    )

    assert result.mpp_extractions[0].supports == (((0, "X"),),)
    assert any(
        isinstance(command, M) and isinstance(command.meas_basis, AxisMeasBasis) and command.meas_basis.axis == Axis.Z
        for command in result.pattern.commands
    )
    assert len(result.pattern.pauli_frame.parity_check_group) == 1
    assert len(result.pattern.pauli_frame.parity_check_group[0]) == 2


def test_stim_text_to_pattern_omits_noise_and_measurement_error_probabilities() -> None:
    result = stim_text_to_pattern(
        """
        DEPOLARIZE1(0.25) 0
        X_ERROR(0.125) 0
        MX(0.5) 0
        DETECTOR rec[-1]
        """
    )

    assert result.mpp_extractions == ()
    measurements = [command for command in result.pattern.commands if isinstance(command, M)]
    assert len(measurements) == 1
    assert isinstance(measurements[0].meas_basis, AxisMeasBasis)
    assert measurements[0].meas_basis.axis == Axis.X
    assert len(result.pattern.pauli_frame.parity_check_group) == 1


def test_stim_text_to_pattern_preserves_ideal_herald_records_as_zero() -> None:
    result = stim_text_to_pattern(
        """
        MPP X0
        HERALDED_ERASE(0.25) 0
        DETECTOR rec[-2] rec[-1]
        """
    )

    assert result.mpp_extractions[0].supports == (((0, "X"),),)
    assert len(result.pattern.pauli_frame.parity_check_group) == 1
    assert len(result.pattern.pauli_frame.parity_check_group[0]) == 1


def test_stim_text_to_pattern_preserves_cross_block_detector_records() -> None:
    result = stim_text_to_pattern(
        """
        MPP X0
        TICK
        MPP Z0
        DETECTOR rec[-2]
        """
    )

    assert len(result.mpp_extractions) == 2
    assert len(result.pattern.pauli_frame.parity_check_group) == 1
    assert len(result.pattern.pauli_frame.parity_check_group[0]) == 1


def test_stim_text_to_pattern_tracks_all_record_types_with_global_indices() -> None:
    result = stim_text_to_pattern(
        """
        MPP X0
        HERALDED_ERASE(0.25) 2
        M 1
        TICK
        MPP Z0
        DETECTOR rec[-4] rec[-3] rec[-2] rec[-1]
        OBSERVABLE_INCLUDE(5) rec[-4] rec[-1]
        """
    )

    assert result.stim_to_qubit == {0: 0, 1: 1}
    assert len(result.mpp_extractions) == 2
    for extraction in result.mpp_extractions:
        assert extraction.detector_record_indices == (frozenset({0, 1, 2, 3}),)
        assert extraction.logical_observable_record_indices == {5: frozenset({0, 3})}
    assert len(result.pattern.pauli_frame.parity_check_group) == 1
    assert len(result.pattern.pauli_frame.parity_check_group[0]) == 3
    assert len(result.pattern.pauli_frame.logical_observables[5]) == 2


@pytest.mark.parametrize(
    "text",
    [
        "MPP X0\nDETECTOR rec[-1]",
        "MPP X0\nTICK\nMPP X0\nDETECTOR rec[-1] rec[-2]",
    ],
)
def test_stim_text_to_pattern_preserves_deterministic_mpp_detectors(text: str) -> None:
    pattern = stim_text_to_pattern(text).pattern
    compiled = stim.Circuit(stim_compile(pattern, emit_qubit_coords=False))

    compiled.detector_error_model()


@pytest.mark.parametrize("y_foliation", [YFoliation.TYPE_I, YFoliation.TYPE_II])
def test_stim_text_to_pattern_preserves_detectors_for_twisted_stabilizer_orders(
    y_foliation: YFoliation,
) -> None:
    pattern = stim_text_to_pattern(
        """
        MPP X0*Z1
        MPP Z0*X1
        TICK
        MPP X0*Z1
        DETECTOR rec[-1] rec[-3]
        MPP Z0*X1
        DETECTOR rec[-1] rec[-3]
        """,
        y_foliation=y_foliation,
    ).pattern
    compiled = stim.Circuit(stim_compile(pattern, emit_qubit_coords=False))

    assert compiled.detector_error_model().num_detectors == 2


@pytest.mark.parametrize(
    ("text", "expected_ancilla_axis"),
    [
        ("RY 0\nTICK\nMPP Y0\nDETECTOR rec[-1]", Axis.Y),
        ("RY 0 1\nTICK\nMPP Y0*Y1\nDETECTOR rec[-1]", Axis.X),
    ],
)
def test_stim_text_to_pattern_preserves_deterministic_type_i_y_mpp_detector(
    text: str,
    expected_ancilla_axis: Axis,
) -> None:
    pattern = stim_text_to_pattern(text, y_foliation=YFoliation.TYPE_I).pattern
    ancilla_node = next(iter(pattern.pauli_frame.parity_check_group[0]))
    ancilla_basis = pattern.pauli_frame.graphstate.meas_bases[ancilla_node]
    compiled = stim.Circuit(stim_compile(pattern, emit_qubit_coords=False))

    assert isinstance(ancilla_basis, AxisMeasBasis)
    assert ancilla_basis.axis == expected_ancilla_axis
    compiled.detector_error_model()


def test_stim_text_to_pattern_composes_mpp_output_into_next_mpp_input() -> None:
    result = stim_text_to_pattern("MPP X0\nTICK\nMPP X0")
    graph = result.pattern.pauli_frame.graphstate

    assert graph.number_of_nodes() == 7
    assert len(graph.meas_bases) == 6
    assert len(graph.input_node_indices) == 1
    assert len(graph.output_node_indices) == 1


def test_stim_text_to_pattern_accepts_annotation_only_tick_block() -> None:
    result = stim_text_to_pattern(
        """
        MPP X0
        TICK
        DETECTOR rec[-1]
        """
    )

    assert len(result.pattern.pauli_frame.parity_check_group) == 1


@pytest.mark.parametrize(
    ("text", "expected_axis", "expected_sign"),
    [
        ("H 0\nM 0", Axis.X, Sign.PLUS),
        ("H 0\nMX 0", Axis.Z, Sign.PLUS),
        ("H 0\nMY 0", Axis.Y, Sign.MINUS),
    ],
)
def test_stim_text_to_pattern_folds_clifford_into_same_tick_measurement(
    text: str,
    expected_axis: Axis,
    expected_sign: Sign,
) -> None:
    result = stim_text_to_pattern(text)
    graph = result.pattern.pauli_frame.graphstate
    measurements = [command for command in result.pattern.commands if isinstance(command, M)]

    assert graph.number_of_nodes() == 1
    assert len(measurements) == 1
    assert isinstance(measurements[0].meas_basis, AxisMeasBasis)
    assert measurements[0].meas_basis.axis == expected_axis
    assert measurements[0].meas_basis.sign == expected_sign


def test_stim_text_to_pattern_splits_mixed_tick_block_across_qubits() -> None:
    result = stim_text_to_pattern("H 1\nM 0")
    measurements = [command for command in result.pattern.commands if isinstance(command, M)]
    measured_axes = [
        command.meas_basis.axis for command in measurements if isinstance(command.meas_basis, AxisMeasBasis)
    ]

    assert set(result.pattern.output_node_indices.values()) == {0, 1}
    assert Axis.Z in measured_axes
    simulator = PatternSimulator(result.pattern, SimulatorBackend.StateVector)
    simulator.simulate(rng=np.random.default_rng(3))
    assert np.isclose(abs(np.vdot([1.0, 0.0], simulator.state.state())), 1.0, atol=1e-9)


@pytest.mark.parametrize("measurement", ["MPP X0", "MXX 0 1"])
def test_stim_text_to_pattern_accepts_clifford_with_pauli_product_in_one_tick(measurement: str) -> None:
    result = stim_text_to_pattern(f"H 0\n{measurement}\nDETECTOR rec[-1]")

    assert len(result.mpp_extractions) == 1
    assert len(result.pattern.pauli_frame.parity_check_group) == 1


def test_stim_text_to_pattern_orders_clifford_before_same_tick_pauli_product() -> None:
    pattern = stim_text_to_pattern("H 0\nMPP Z0\nDETECTOR rec[-1]").pattern
    compiled = stim.Circuit(stim_compile(pattern, emit_qubit_coords=False))

    compiled.detector_error_model()


def test_stim_text_to_pattern_orders_clifford_before_same_tick_measurement_with_detector() -> None:
    pattern = stim_text_to_pattern("H 0\nM 0\nDETECTOR rec[-1]").pattern
    compiled = stim.Circuit(stim_compile(pattern, emit_qubit_coords=False))

    compiled.detector_error_model()


@pytest.mark.parametrize(
    ("instruction", "expected_axis", "compiled_instruction"),
    [
        ("R 0", Axis.Z, "R 0"),
        ("RZ 0", Axis.Z, "R 0"),
        ("RX 0", Axis.X, "RX 0"),
        ("RY 0", Axis.Y, "RY 0"),
    ],
)
def test_stim_text_to_pattern_imports_initial_reset(
    instruction: str,
    expected_axis: Axis,
    compiled_instruction: str,
) -> None:
    result = stim_text_to_pattern(instruction)
    input_node = next(node for node, q_index in result.pattern.input_node_indices.items() if q_index == 0)

    assert result.pattern.input_initialization_axes[input_node] == expected_axis
    assert compiled_instruction in stim_compile(result.pattern, emit_qubit_coords=False).splitlines()


def test_stim_text_to_pattern_folds_initial_h_into_rx_and_aligns_input_z() -> None:
    result = stim_text_to_pattern(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        R 0 1
        H 1
        """
    )
    graph = result.pattern.pauli_frame.graphstate
    input_nodes = {qubit: node for node, qubit in graph.input_node_indices.items()}
    compiled = stim_compile(result.pattern, emit_qubit_coords=False).splitlines()

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 0
    assert graph.input_initialization_axes[input_nodes[0]] == Axis.Z
    assert graph.input_initialization_axes[input_nodes[1]] == Axis.X
    assert graph.coordinates[input_nodes[0]] == (0.0, 0.0, 0.0)
    assert graph.coordinates[input_nodes[1]] == (1.0, 0.0, 0.0)
    assert f"R {input_nodes[0]}" in compiled
    assert f"RX {input_nodes[1]}" in compiled


def test_stim_text_to_pattern_removes_clifford_preserving_initial_reset_state() -> None:
    result = stim_text_to_pattern("QUBIT_COORDS(0, 0) 0\nR 0\nS 0")
    graph = result.pattern.pauli_frame.graphstate
    input_node = next(iter(graph.input_node_indices))

    assert graph.number_of_nodes() == 1
    assert graph.input_initialization_axes[input_node] == Axis.Z
    assert graph.coordinates[input_node] == (0.0, 0.0, 0.0)


def test_stim_text_to_pattern_uses_last_leading_reset() -> None:
    result = stim_text_to_pattern("R 0\nRY 0\nH 0")
    input_node = next(node for node, q_index in result.pattern.input_node_indices.items() if q_index == 0)

    assert result.pattern.input_initialization_axes[input_node] == Axis.Y


def test_stim_text_to_pattern_allows_initial_reset_after_other_qubit_operation() -> None:
    result = stim_text_to_pattern("H 0\nR 1")
    input_node = next(node for node, q_index in result.pattern.input_node_indices.items() if q_index == 1)

    assert result.pattern.input_initialization_axes[input_node] == Axis.Z


@pytest.mark.parametrize("instruction", ["R", "RX", "RY"])
def test_stim_text_to_pattern_rejects_reset_after_quantum_operation(instruction: str) -> None:
    with pytest.raises(ValueError, match="only initial resets are supported"):
        stim_text_to_pattern(f"H 0\n{instruction} 0")


@pytest.mark.parametrize("instruction", ["MR 0", "MRX 0", "MRY 0"])
def test_stim_text_to_pattern_defers_measurement_reset_instructions(instruction: str) -> None:
    with pytest.raises(ValueError, match="Unsupported Stim instruction"):
        stim_text_to_pattern(instruction)


def test_stim_text_to_pattern_rejects_duplicate_qubit_coordinates() -> None:
    with pytest.raises(ValueError, match="distinct XY projections"):
        stim_text_to_pattern("QUBIT_COORDS(0, 0) 0\nQUBIT_COORDS(0, 0) 1\nCZ 0 1")


def test_stim_text_to_pattern_rejects_coordinates_sharing_an_xy_projection() -> None:
    with pytest.raises(ValueError, match="distinct XY projections"):
        stim_text_to_pattern(
            "QUBIT_COORDS(0, 0, 1) 0\nQUBIT_COORDS(0, 0, 2) 1\nCZ 0 1",
            coord_dims=3,
        )


def test_stim_text_to_pattern_issues_new_qubit_index_for_reused_qubit() -> None:
    result = stim_text_to_pattern("M 0\nTICK\nH 0")

    assert result.stim_to_qubit == {0: 0}
    assert result.qubit_to_stim == {0: 0, 1: 0}
    assert set(result.pattern.output_node_indices.values()) == {1}


def test_stim_text_to_pattern_continues_reused_wire_in_post_measurement_state() -> None:
    for seed in range(8):
        result = stim_text_to_pattern("M 0\nTICK\nH 0")
        simulator = PatternSimulator(result.pattern, SimulatorBackend.StateVector)
        simulator.simulate(rng=np.random.default_rng(seed))

        outcome = next(iter(simulator.results.values()))
        expected = np.asarray([1.0, (-1.0) ** outcome], dtype=np.complex128) / np.sqrt(2)
        assert np.isclose(abs(np.vdot(expected, simulator.state.state())), 1.0, atol=1e-9)


def test_stim_text_to_pattern_places_reused_wire_at_same_xy_and_new_z() -> None:
    result = stim_text_to_pattern("QUBIT_COORDS(1, 2) 0\nM 0\nTICK\nH 0")
    graph = result.pattern.pauli_frame.graphstate
    coordinates = sorted(graph.coordinates.values(), key=operator.itemgetter(2))

    assert len(coordinates) == graph.number_of_nodes()
    assert all(coordinate[:2] == (1.0, 2.0) for coordinate in coordinates)
    assert coordinates[0] == (1.0, 2.0, 0.0)
    assert all(coordinate[2] > 0.0 for coordinate in coordinates[1:])


@pytest.mark.parametrize(
    ("measurement", "expected_axis"),
    [("M", Axis.Z), ("MX", Axis.X), ("MY", Axis.Y)],
)
def test_stim_text_to_pattern_repeated_measurement_outcomes_agree(
    measurement: str,
    expected_axis: Axis,
) -> None:
    for seed in range(8):
        result = stim_text_to_pattern(f"{measurement} 0\nTICK\n{measurement} 0")
        simulator = PatternSimulator(result.pattern, SimulatorBackend.StateVector)
        simulator.simulate(rng=np.random.default_rng(seed))

        assert len(simulator.results) == 2
        assert len(set(simulator.results.values())) == 1

    graph = stim_text_to_pattern(f"{measurement} 0\nTICK\n{measurement} 0").pattern.pauli_frame.graphstate
    continuation_node = next(node for node, qubit in graph.input_node_indices.items() if qubit == 1)
    assert graph.input_initialization_axes[continuation_node] == expected_axis


@pytest.mark.parametrize(
    "text",
    [
        "M 0\nTICK\nM 0\nDETECTOR rec[-1] rec[-2]",
        "MX 0\nTICK\nMX 0\nDETECTOR rec[-1] rec[-2]",
        "MY 0\nTICK\nMY 0\nDETECTOR rec[-1] rec[-2]",
        "M 0\nTICK\nH 0\nTICK\nMX 0\nDETECTOR rec[-1] rec[-2]",
        "MX 0\nTICK\nH 0\nTICK\nM 0\nDETECTOR rec[-1] rec[-2]",
        "MY 0\nTICK\nSQRT_Y 0\nTICK\nMY 0\nDETECTOR rec[-1] rec[-2]",
        "M 0\nTICK\nM 0\nTICK\nM 0\nDETECTOR rec[-1] rec[-3]",
        "M 0\nTICK\nMPP Z0\nDETECTOR rec[-1] rec[-2]",
        "M 0\nTICK\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-2]",
    ],
)
def test_stim_text_to_pattern_preserves_deterministic_annotations_across_reuse(text: str) -> None:
    pattern = stim_text_to_pattern(text).pattern
    compiled = stim.Circuit(stim_compile(pattern, emit_qubit_coords=False))

    compiled.detector_error_model()


def test_stim_text_to_pattern_allows_pauli_product_after_single_measurement() -> None:
    result = stim_text_to_pattern("M 0\nTICK\nMPP X0")

    assert len(result.mpp_extractions) == 1
    assert result.qubit_to_stim == {0: 0, 1: 0}
    simulator = PatternSimulator(result.pattern, SimulatorBackend.StateVector)
    simulator.simulate(rng=np.random.default_rng(3))


def test_stim_text_to_pattern_entangles_reused_wire_after_measurement() -> None:
    for seed in range(8):
        result = stim_text_to_pattern("M 0\nTICK\nCZ 0 1")
        simulator = PatternSimulator(result.pattern, SimulatorBackend.StateVector)
        simulator.simulate(rng=np.random.default_rng(seed))

        outcome = int(next(iter(simulator.results.values())))
        reused_wire = np.zeros(2, dtype=np.complex128)
        reused_wire[outcome] = 1.0
        partner_wire = np.asarray([1.0, (-1.0) ** outcome], dtype=np.complex128) / np.sqrt(2)
        expected = np.kron(partner_wire, reused_wire)
        assert np.isclose(abs(np.vdot(expected, simulator.state.state())), 1.0, atol=1e-9)


def test_stim_text_to_pattern_keeps_measured_state_when_qubit_is_reused() -> None:
    result = stim_text_to_pattern("H 0\nM 0\nTICK\nH 0\nTICK\nMX 0\nDETECTOR rec[-1] rec[-2]")
    graph = result.pattern.pauli_frame.graphstate
    measured_axes = [basis.axis for basis in graph.meas_bases.values() if isinstance(basis, AxisMeasBasis)]
    compiled = stim.Circuit(stim_compile(result.pattern, emit_qubit_coords=False))

    assert Axis.Z in measured_axes
    compiled.detector_error_model()


def test_stim_text_to_pattern_rejects_reuse_after_inverted_measurement() -> None:
    with pytest.raises(ValueError, match="inverted single-qubit measurement"):
        stim_text_to_pattern("M !0\nTICK\nH 0")


@pytest.mark.parametrize("text", ["M 0 0", "M 0\nM 0"])
def test_stim_text_to_pattern_sequences_repeated_measurement_in_one_tick(text: str) -> None:
    for seed in range(4):
        result = stim_text_to_pattern(text)
        simulator = PatternSimulator(result.pattern, SimulatorBackend.StateVector)
        simulator.simulate(rng=np.random.default_rng(seed))

        assert len(simulator.results) == 2
        assert len(set(simulator.results.values())) == 1


_DIFFERENTIAL_GATES = ("H", "S", "S_DAG", "X", "Z", "SQRT_Y", "C_XYZ")
_DIFFERENTIAL_MEASUREMENTS = ("M", "MX", "MY")


def _random_wire_circuit(rng: np.random.Generator) -> str:
    lines: list[str] = []
    for _ in range(int(rng.integers(1, 4))):
        for _ in range(int(rng.integers(0, 3))):
            lines.append(f"{rng.choice(_DIFFERENTIAL_GATES)} 0")
            if rng.random() < 0.4:
                lines.append("TICK")
        lines.append(f"{rng.choice(_DIFFERENTIAL_MEASUREMENTS)} 0")
        if rng.random() < 0.6:
            lines.append("TICK")
    lines.append(f"{rng.choice(_DIFFERENTIAL_GATES)} 0")
    return "\n".join(lines)


def _postselected_stim_state(text: str, outcomes: list[bool]) -> NDArray[np.complex128]:
    circuit = stim.Circuit(text)
    simulator = stim.TableauSimulator()
    simulator.set_num_qubits(circuit.num_qubits)
    record_index = 0
    for instruction in circuit:
        name = instruction.name
        if name in {"M", "MX", "MY"}:
            postselect = {
                "M": simulator.postselect_z,
                "MX": simulator.postselect_x,
                "MY": simulator.postselect_y,
            }[name]
            for target in instruction.targets_copy():
                postselect(target.qubit_value, desired_value=bool(outcomes[record_index]))
                record_index += 1
        elif name in {"CX", "CZ"} and any(t.is_measurement_record_target for t in instruction.targets_copy()):
            targets = instruction.targets_copy()
            for control, target in zip(targets[::2], targets[1::2], strict=True):
                if control.is_measurement_record_target:
                    if outcomes[record_index + control.value]:
                        apply_pauli = simulator.x if name == "CX" else simulator.z
                        apply_pauli(target.qubit_value)
                else:
                    simulator.do(stim.Circuit(f"{name} {control.qubit_value} {target.qubit_value}"))
        elif name != "TICK":
            simulator.do(stim.Circuit(str(instruction)))
    assert record_index == len(outcomes)
    simulator.set_num_qubits(circuit.num_qubits)
    return np.asarray(simulator.state_vector(endian="big"), dtype=np.complex128)


def test_stim_text_to_pattern_matches_stim_for_random_reuse_circuits() -> None:
    """Differential test against stim, postselected on the sampled outcomes.

    Random single-wire circuits mix Clifford gates and measurements inside
    shared TICK blocks and keep using the qubit after measurements. The final
    pattern state must match stim's state for the same measurement record.
    """
    rng = np.random.default_rng(20260724)
    for _ in range(25):
        text = "RX 0\nTICK\n" + _random_wire_circuit(rng)
        total_records = stim.Circuit(text).num_measurements
        markers = "\n".join(f"DETECTOR rec[{index - total_records}]" for index in range(total_records))
        for sim_seed in range(2):
            result = stim_text_to_pattern(text + "\n" + markers)
            simulator = PatternSimulator(result.pattern, SimulatorBackend.StateVector)
            simulator.simulate(rng=np.random.default_rng(sim_seed))

            record_nodes = [next(iter(group)) for group in result.pattern.pauli_frame.parity_check_group]
            outcomes = [simulator.results[node] for node in record_nodes]
            reference = _postselected_stim_state(text, outcomes)
            overlap = abs(np.vdot(reference, simulator.state.state().flatten()))
            assert np.isclose(overlap, 1.0, atol=1e-8), text


def test_stim_text_to_pattern_feedback_resets_reused_wire_deterministically() -> None:
    """``M 0`` then ``CX rec[-1] 0`` leaves the reused wire in |0> for every outcome."""
    for seed in range(8):
        result = stim_text_to_pattern("M 0\nTICK\nCX rec[-1] 0")
        simulator = PatternSimulator(result.pattern, SimulatorBackend.StateVector)
        simulator.simulate(rng=np.random.default_rng(seed))

        expected = np.asarray([1.0, 0.0], dtype=np.complex128)
        assert np.isclose(abs(np.vdot(expected, simulator.state.state())), 1.0, atol=1e-9)


def test_stim_text_to_pattern_defuses_feedback_fused_with_unitary_gates() -> None:
    """Adjacent ``CZ rec[-1] 0`` and ``CZ 0 1`` arrive stim-fused as one instruction."""
    for seed in range(8):
        result = stim_text_to_pattern("M 0\nTICK\nCZ rec[-1] 0\nCZ 0 1")
        simulator = PatternSimulator(result.pattern, SimulatorBackend.StateVector)
        simulator.simulate(rng=np.random.default_rng(seed))

        outcome = int(next(iter(simulator.results.values())))
        reused_wire = np.zeros(2, dtype=np.complex128)
        reused_wire[outcome] = 1.0
        partner_wire = np.asarray([1.0, (-1.0) ** outcome], dtype=np.complex128) / np.sqrt(2)
        expected = np.kron(partner_wire, reused_wire)
        assert np.isclose(abs(np.vdot(expected, simulator.state.state())), 1.0, atol=1e-9)


def _random_two_qubit_circuit(rng: np.random.Generator) -> str:
    lines: list[str] = []
    for _ in range(int(rng.integers(3, 7))):
        roll = rng.random()
        if roll < 0.4:
            lines.append(f"{rng.choice(_DIFFERENTIAL_GATES)} {int(rng.integers(2))}")
        elif roll < 0.6:
            lines.append("CZ 0 1")
        elif roll < 0.85:
            lines.append(f"{rng.choice(_DIFFERENTIAL_MEASUREMENTS)} {int(rng.integers(2))}")
            if rng.random() < 0.4:
                lines.append(f"C{rng.choice(['X', 'Z'])} rec[-1] {int(rng.integers(2))}")
        else:
            lines.append("TICK")
    lines.extend(f"{rng.choice(_DIFFERENTIAL_GATES)} {qubit}" for qubit in range(2))
    return "\n".join(lines)


def test_stim_text_to_pattern_matches_stim_for_random_two_qubit_reuse_circuits() -> None:
    """Differential test with entanglement, reuse, and record-controlled feedback.

    Random two-qubit circuits mix CZ, Clifford gates, single-qubit
    measurements with reuse, and ``rec``-controlled feedback. The final
    pattern state must match stim's state postselected on the same
    measurement record. Trials where trailing Cliffords cancel to the
    identity terminate a wire instead of reusing it; those are skipped, and
    the counter asserts the skips stay rare.
    """
    rng = np.random.default_rng(20260725)
    compared = 0
    for _ in range(20):
        text = "RX 0\nRX 1\nTICK\n" + _random_two_qubit_circuit(rng)
        total_records = stim.Circuit(text).num_measurements
        markers = "\n".join(f"DETECTOR rec[{index - total_records}]" for index in range(total_records))
        for sim_seed in range(2):
            result = stim_text_to_pattern(text + "\n" + markers)
            internal_order = sorted(set(result.pattern.output_node_indices.values()))
            stim_order = [result.qubit_to_stim[qubit] for qubit in internal_order]
            if sorted(stim_order) != [0, 1]:
                break
            simulator = PatternSimulator(result.pattern, SimulatorBackend.StateVector)
            simulator.simulate(rng=np.random.default_rng(sim_seed))

            record_nodes = [next(iter(group)) for group in result.pattern.pauli_frame.parity_check_group]
            outcomes = [simulator.results[node] for node in record_nodes]
            reference = _postselected_stim_state(text, outcomes)
            reference = np.transpose(reference.reshape(2, 2), axes=stim_order).flatten()
            overlap = abs(np.vdot(reference, simulator.state.state().flatten()))
            assert np.isclose(overlap, 1.0, atol=1e-8), text
            compared += 1
    assert compared >= 30


def test_stim_text_to_pattern_allows_disjoint_qubit_after_single_measurement() -> None:
    result = stim_text_to_pattern("M 0\nTICK\nH 1")
    measured_nodes = {command.node for command in result.pattern.commands if isinstance(command, M)}
    simulator = PatternSimulator(result.pattern, SimulatorBackend.StateVector)

    simulator.simulate(rng=np.random.default_rng(3))

    assert result.stim_to_qubit == {0: 0, 1: 1}
    assert set(result.pattern.output_node_indices.values()) == {0, 1}
    assert any(result.pattern.output_node_indices[node] == 0 for node in measured_nodes)
    assert set(simulator.output_results) == {0}
    assert np.isclose(abs(np.vdot([1, 0], simulator.state.state())), 1.0, atol=1e-9)


@pytest.mark.parametrize(
    ("instruction", "expected_axis", "compiled_instruction"),
    [
        ("MX !0", Axis.X, "MX !0"),
        ("MY !0", Axis.Y, "MY !0"),
        ("M !0", Axis.Z, "MZ !0"),
    ],
)
def test_stim_text_to_pattern_preserves_inverted_single_measurement(
    instruction: str,
    expected_axis: Axis,
    compiled_instruction: str,
) -> None:
    result = stim_text_to_pattern(instruction)
    measurements = [command for command in result.pattern.commands if isinstance(command, M)]

    assert result.mpp_extractions == ()
    assert len(measurements) == 1
    assert isinstance(measurements[0].meas_basis, AxisMeasBasis)
    assert measurements[0].meas_basis.axis == expected_axis
    assert measurements[0].meas_basis.sign == Sign.MINUS
    assert compiled_instruction in stim_compile(result.pattern, emit_qubit_coords=False).splitlines()


@pytest.mark.parametrize(
    ("instruction", "name"),
    [
        ("MXX !0 1", "MXX"),
        ("MYY 0 !1", "MYY"),
        ("MZZ !0 1", "MZZ"),
    ],
)
def test_stim_text_to_pattern_rejects_inverted_pair_measurement_result(
    instruction: str,
    name: str,
) -> None:
    with pytest.raises(ValueError, match=rf"Signed {name} products are not supported"):
        stim_text_to_pattern(instruction)


def test_stim_text_to_pattern_rejects_true_mpad_record() -> None:
    with pytest.raises(ValueError, match="MPAD 1 records are not supported"):
        stim_text_to_pattern("MPP X0\nMPAD 1")


def test_stim_text_to_pattern_rejects_record_before_beginning_of_time() -> None:
    with pytest.raises(ValueError, match="before the beginning of time"):
        stim_text_to_pattern("DETECTOR rec[-1]")
