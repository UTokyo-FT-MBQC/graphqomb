"""Tests for Stim Clifford transpilation into the GraphQOMB gate basis."""

from __future__ import annotations

import math
import random
from itertools import product

import pytest
import stim

from graphqomb.stim_glue.transpiler import (
    HS_DAG_STIM_GATE,
    HS_STIM_GATE,
    HZ_STIM_GATE,
    STIM_GATE_J_ANGLES,
    UnsupportedInstructionError,
    optimize_j_cz,
    transpile,
)

PRESERVED_INSTRUCTIONS = {
    "DETECTOR",
    "MPAD",
    "OBSERVABLE_INCLUDE",
    "QUBIT_COORDS",
    "SHIFT_COORDS",
    "TICK",
}
BOUNDARY_OPERATIONS = {"R", "RX", "RY", "M", "MX", "MY"}
J_GATES = ("H", HS_STIM_GATE, HZ_STIM_GATE, HS_DAG_STIM_GATE)
BASIS_GATES = {*J_GATES, "CZ"}


def assert_only_graphqomb_basis(circuit: stim.Circuit) -> None:
    """Assert that every unitary instruction is a Clifford J gate or CZ."""
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            assert_only_graphqomb_basis(instruction.body_copy())
        elif stim.gate_data(instruction.name).is_unitary:
            assert instruction.name in BASIS_GATES
        else:
            assert instruction.name in PRESERVED_INSTRUCTIONS | BOUNDARY_OPERATIONS


def assert_same_tableau(actual: stim.Circuit, expected: stim.Circuit) -> None:
    """Assert unitary equivalence up to global phase."""
    assert stim.Tableau.from_circuit(actual) == stim.Tableau.from_circuit(expected)


def test_hs_stim_gate_is_j_pi_over_two_and_s_decomposition_is_hs_then_h() -> None:
    expected_hs = stim.Circuit("S 0\nH 0")
    native_hs = stim.Circuit(f"{HS_STIM_GATE} 0")

    assert HS_STIM_GATE == "C_XNYZ"
    assert_same_tableau(native_hs, expected_hs)
    assert str(transpile("S 0")) == f"{HS_STIM_GATE} 0\nH 0"
    assert_same_tableau(transpile("S 0"), stim.Circuit("S 0"))


def test_each_basis_gate_is_the_j_gate_named_by_its_angle() -> None:
    """Each single-qubit basis gate must equal J(angle) = H Rz(angle) up to phase."""
    for gate_name, angle in STIM_GATE_J_ANGLES.items():
        quarter_turns = round(angle / (math.pi / 2))
        j_circuit = stim.Circuit()
        for _ in range(quarter_turns % 4):
            j_circuit.append("S", [0])
        j_circuit.append("H", [0])

        assert_same_tableau(stim.Circuit(f"{gate_name} 0"), j_circuit)


@pytest.mark.parametrize(
    "gate_name",
    [name for name in sorted(stim.gate_data()) if stim.gate_data(name).is_unitary and name not in {"SPP", "SPP_DAG"}],
)
def test_every_fixed_arity_stim_clifford_gate(gate_name: str) -> None:
    gate_data = stim.gate_data(gate_name)
    targets = [0] if gate_data.is_single_qubit_gate else [0, 1]
    source = stim.Circuit()
    source.append(gate_name, targets)

    result = transpile(source)

    assert_only_graphqomb_basis(result)
    assert_same_tableau(result, source)


@pytest.mark.parametrize("gate_name", ["SPP", "SPP_DAG"])
def test_pauli_product_rotations(gate_name: str) -> None:
    source = stim.Circuit(f"{gate_name} X5*Y2*!Z9 !Y4 X0*Z7\n")

    result = transpile(source)

    assert_only_graphqomb_basis(result)
    assert_same_tableau(result, source)


@pytest.mark.parametrize("gate_name", ["SPP", "SPP_DAG"])
def test_pauli_product_rotations_support_repeated_qubits(gate_name: str) -> None:
    source = stim.Circuit(f"{gate_name} X5*X5*Z2 X1*Y1*Y2*Z2 X0*!X0")

    result = transpile(source)

    assert_only_graphqomb_basis(result)
    assert_same_tableau(result, source)


@pytest.mark.parametrize("gate_name", ["SPP", "SPP_DAG"])
def test_pauli_product_rotations_reject_anti_hermitian_products(gate_name: str) -> None:
    with pytest.raises(UnsupportedInstructionError, match="invalid Pauli product"):
        transpile(f"{gate_name} X0*Y0")


def test_repeat_blocks_annotations_and_tags_are_preserved() -> None:
    source = stim.Circuit(
        """
        QUBIT_COORDS(1, 2) 0
        REPEAT[outer] 3 {
            TICK[layer]
            S_DAG[phase] 0
            CX[entangle] 0 1
        }
        SHIFT_COORDS(0, 0, 1)
        """
    )

    result = transpile(source)

    assert_only_graphqomb_basis(result)
    assert_same_tableau(result, source)
    repeat = result[1]
    assert isinstance(repeat, stim.CircuitRepeatBlock)
    assert repeat.repeat_count == 3
    assert repeat.tag == "outer"
    assert "[phase]" in str(repeat.body_copy())
    assert "[entangle]" in str(repeat.body_copy())


def test_record_annotations_and_mpad_are_preserved_as_optimization_barriers() -> None:
    source = stim.Circuit(
        """
        H 1
        M 0
        DETECTOR[det](1, 2, 3) rec[-1]
        OBSERVABLE_INCLUDE[obs](7) rec[-1]
        MPAD[pad] 0 1
        H 1
        """
    )

    assert transpile(source, optimize=True) == source
    assert optimize_j_cz(source) == source
    assert transpile(source, optimize=True).num_measurements == source.num_measurements


def test_accepts_circuit_text() -> None:
    result = transpile("X 0\nCY 0 1\n")

    assert isinstance(result, stim.Circuit)
    assert_only_graphqomb_basis(result)


def test_optimizer_removes_h_squared_hs_cubed_and_cz_squared() -> None:
    source = stim.Circuit(
        f"""
        H 0 0
        {HS_STIM_GATE} 1 1 1
        CZ 2 3
        {HS_STIM_GATE} 4
        CZ 3 2
        """
    )

    result = optimize_j_cz(source)

    assert str(result) == f"{HS_STIM_GATE} 4"


def test_optimizer_cancels_basis_gates_across_commuting_operations() -> None:
    source = stim.Circuit(
        f"""
        H 0
        CZ 1 2
        H 0
        {HS_STIM_GATE} 3
        H 4
        {HS_STIM_GATE} 3
        CZ 1 2
        {HS_STIM_GATE} 3
        H 4
        """
    )

    assert len(optimize_j_cz(source)) == 0


def test_optimizer_batch_normalizes_many_disjoint_single_qubit_runs() -> None:
    qubit_count = 2048
    source = stim.Circuit()
    source.append("H", range(qubit_count))
    source.append("H", range(qubit_count))
    source.append(HS_STIM_GATE, [qubit_count])

    assert optimize_j_cz(source) == stim.Circuit(f"{HS_STIM_GATE} {qubit_count}")


def test_optimizer_treats_tick_as_a_barrier_and_recurses_into_repeat() -> None:
    source = stim.Circuit(
        f"""
        H 0
        TICK
        H 0
        REPEAT 2 {{
            {HS_STIM_GATE} 1 1 1
            CZ 1 2 2 1
        }}
        """
    )

    result = optimize_j_cz(source)

    assert str(result).startswith("H 0\nTICK\nH 0\nREPEAT 2")
    repeat = result[3]
    assert isinstance(repeat, stim.CircuitRepeatBlock)
    assert len(repeat.body_copy()) == 0


def test_transpile_optimization_cancels_source_basis_identities() -> None:
    source = stim.Circuit("H 0 0\nS 1 1 1 1\nCZ 2 3 2 3")

    assert len(transpile(source, optimize=True)) == 0


@pytest.mark.parametrize("measurement", ["M", "MX", "MY"])
def test_measurement_is_not_folded_when_post_measurement_state_is_reused(measurement: str) -> None:
    source = stim.Circuit(f"H 0\n{measurement} 0\nH 0")

    assert optimize_j_cz(source) == source


@pytest.mark.parametrize("measurement", ["M", "MX", "MY"])
def test_optimizer_preserves_inverted_measurement_targets(measurement: str) -> None:
    source = stim.Circuit(f"{measurement}(0.125) !0")

    assert optimize_j_cz(source) == source


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("RX 0\nH 0", "R 0"),
        (f"RY 0\n{HS_DAG_STIM_GATE} 0", "R 0"),
        ("H 0\nMX 0", "M 0"),
        ("H 0\nMY(0.125) 0", "MY(0.125) !0"),
        (f"{HS_DAG_STIM_GATE} 0\nMY 0", "MX 0"),
    ],
)
def test_optimizer_folds_all_pauli_reset_and_measurement_boundaries(source: str, expected: str) -> None:
    assert optimize_j_cz(source) == stim.Circuit(expected)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (f"H 0\n{HS_STIM_GATE} 0\nMX 0\nH 0", "MX 0\nH 0"),
        (f"RY 0\nH 0\n{HZ_STIM_GATE} 0", "RY 0\nH 0"),
    ],
)
def test_optimizer_uses_axis_relative_boundary_normal_forms(source: str, expected: str) -> None:
    assert optimize_j_cz(source) == stim.Circuit(expected)


def test_optimizer_preserved_measurement_qubits_block_terminal_folds() -> None:
    source = stim.Circuit("H 0\nM 0\nH 1\nM 1")

    optimized = optimize_j_cz(source, preserved_measurement_qubits={0})

    assert optimized == stim.Circuit("H 0\nM 0\nMX 1")


def test_transpile_optimize_respects_preserved_measurement_qubits() -> None:
    circuit = transpile("H 0\nM 0", optimize=True, preserved_measurement_qubits={0})

    assert circuit == stim.Circuit("H 0\nM 0")


def test_optimizer_emits_shortest_single_qubit_words() -> None:
    """Every J word must optimize to a shortest word with the same Clifford action."""
    words = [word for length in range(5) for word in product(J_GATES, repeat=length)]
    minimal_lengths: dict[tuple[str, str], int] = {}
    for word in words:
        circuit = stim.Circuit()
        for gate in word:
            circuit.append(gate, [0])
        tableau = stim.Tableau.from_circuit(circuit) if word else stim.Tableau(1)
        key = (str(tableau.x_output(0)), str(tableau.z_output(0)))
        minimal_lengths.setdefault(key, len(word))

        optimized = optimize_j_cz(circuit)
        gate_count = 0
        for instruction in optimized:
            assert isinstance(instruction, stim.CircuitInstruction)
            gate_count += len(instruction.targets_copy())
        assert gate_count == minimal_lengths[key]
        assert_same_tableau(optimized + stim.Circuit("I 0"), circuit + stim.Circuit("I 0"))
    assert len(minimal_lengths) == 24
    assert max(minimal_lengths.values()) == 3


def test_all_single_qubit_words_preserve_reset_state_and_measurement_observable() -> None:
    reset_axes = {"R": "Z", "RX": "X", "RY": "Y"}
    measurement_axes = {"M": "Z", "MX": "X", "MY": "Y"}
    for length in range(5):
        for word in product(J_GATES, repeat=length):
            word_circuit = stim.Circuit()
            for gate in word:
                word_circuit.append(gate, [0])
            original_tableau = stim.Tableau.from_circuit(word_circuit) if word else stim.Tableau(1)

            for reset, axis in reset_axes.items():
                optimized_reset = optimize_j_cz(stim.Circuit(f"{reset} 0") + word_circuit)
                assert _prepared_state_key(optimized_reset) == _tableau_axis_output(original_tableau, axis), (
                    f"{reset=} {word=}"
                )

            inverse = original_tableau.inverse()
            for measurement, axis in measurement_axes.items():
                optimized_measurement = optimize_j_cz(word_circuit + stim.Circuit(f"{measurement} !0"))
                assert _measurement_observable_key(optimized_measurement) == -_tableau_axis_output(inverse, axis), (
                    f"{measurement=} {word=}"
                )


def _tableau_axis_output(tableau: stim.Tableau, axis: str) -> stim.PauliString:
    return {
        "X": tableau.x_output(0),
        "Y": tableau.y_output(0),
        "Z": tableau.z_output(0),
    }[axis]


def _unitary_part(circuit: stim.Circuit) -> stim.Circuit:
    result = stim.Circuit()
    for instruction in circuit:
        if not isinstance(instruction, stim.CircuitRepeatBlock) and stim.gate_data(instruction.name).is_unitary:
            result.append(instruction)
    return result


def _prepared_state_key(circuit: stim.Circuit) -> stim.PauliString:
    reset = next(
        instruction
        for instruction in circuit
        if not isinstance(instruction, stim.CircuitRepeatBlock) and instruction.name in {"R", "RX", "RY"}
    )
    preparation = stim.Circuit()
    if reset.name == "RX":
        preparation.append("H", [0])
    elif reset.name == "RY":
        preparation.append("H", [0])
        preparation.append("S", [0])
    preparation += _unitary_part(circuit)
    tableau = stim.Tableau.from_circuit(preparation) if len(preparation) else stim.Tableau(1)
    return tableau.z_output(0)


def _measurement_observable_key(circuit: stim.Circuit) -> stim.PauliString:
    assert len(_unitary_part(circuit)) == 0
    measurement = next(
        instruction
        for instruction in circuit
        if not isinstance(instruction, stim.CircuitRepeatBlock) and instruction.name in {"M", "MX", "MY"}
    )
    target = measurement.targets_copy()[0]
    sign = "-" if target.is_inverted_result_target else "+"
    pauli = {"M": "Z", "MX": "X", "MY": "Y"}[measurement.name]
    return stim.PauliString(sign + pauli)


def test_random_circuits_keep_their_tableau_with_and_without_optimization() -> None:
    single_qubit_gates = [
        name
        for name in sorted(stim.gate_data())
        if stim.gate_data(name).is_unitary and stim.gate_data(name).is_single_qubit_gate
    ]
    two_qubit_gates = [
        name
        for name in sorted(stim.gate_data())
        if stim.gate_data(name).is_unitary and stim.gate_data(name).is_two_qubit_gate
    ]
    rng = random.Random(20260719)  # ruff:ignore[suspicious-non-cryptographic-random-usage] - deterministic test data

    for _ in range(30):
        num_qubits = rng.randint(2, 5)
        source = stim.Circuit()
        for _ in range(rng.randint(1, 20)):
            if rng.random() < 0.5:
                source.append(rng.choice(single_qubit_gates), [rng.randrange(num_qubits)])
            else:
                source.append(rng.choice(two_qubit_gates), rng.sample(range(num_qubits), 2))
        expected = stim.Tableau.from_circuit(source)

        for optimize in (False, True):
            result = transpile(source, optimize=optimize)
            assert_only_graphqomb_basis(result)
            if result.num_qubits < source.num_qubits:
                result = result.copy()
                result.append("I", [source.num_qubits - 1])
            assert stim.Tableau.from_circuit(result) == expected


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("X_ERROR(0.1) 0", "not a unitary Clifford gate"),
        ("CX rec[-1] 0", "non-qubit target"),
    ],
)
def test_rejects_operations_that_cannot_be_represented(source: str, message: str) -> None:
    with pytest.raises(UnsupportedInstructionError, match=message):
        transpile(source)


def test_non_clifford_rejection_lists_the_current_j_cz_basis() -> None:
    with pytest.raises(UnsupportedInstructionError) as exc_info:
        transpile("X_ERROR(0.1) 0")

    message = str(exc_info.value)
    assert "Clifford J/CZ basis" in message
    for gate in BASIS_GATES:
        assert gate in message


def test_rejection_reports_nested_repeat_location() -> None:
    source = stim.Circuit("REPEAT 2 {\nREPEAT 3 {\nCX rec[-1] 0\n}\n}")

    with pytest.raises(
        UnsupportedInstructionError,
        match=r"circuit, instruction 0, REPEAT body, instruction 0, REPEAT body, instruction 0",
    ):
        transpile(source)
