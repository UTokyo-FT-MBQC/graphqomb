"""Tests for the exact pending-Clifford MPP rewriter."""

from __future__ import annotations

import random
from typing import Literal, cast

import numpy as np
import pytest
import stim

from graphqomb.stim_glue.mpp_rewriter import UnsupportedSyndromeCircuitError, rewrite_to_mpp


def _assert_exact_channel(source: stim.Circuit, rewritten: stim.Circuit) -> None:
    """Require the canonical signed stabilizer-flow bases to match exactly."""
    assert rewritten.num_measurements == source.num_measurements
    assert rewritten.flow_generators() == source.flow_generators()


def _assert_same_reference_signs(left: stim.Circuit, right: stim.Circuit) -> None:
    left_detectors, left_observables = left.reference_detector_and_observable_signs()
    right_detectors, right_observables = right.reference_detector_and_observable_signs()
    assert np.array_equal(left_detectors, right_detectors)
    assert np.array_equal(left_observables, right_observables)


def test_x_check_exposes_data_mpp_and_moves_body_behind_it() -> None:
    source = stim.Circuit(
        """
        R 4
        H 4
        CX 4 0 4 1 4 2 4 3
        H 4
        M 4
        """
    )

    result = rewrite_to_mpp(source)

    assert result.circuit == stim.Circuit(
        """
        R 4
        MPP X0*X1*X2*X3
        H 4
        CX 4 0 4 1 4 2 4 3
        H 4
        """
    )
    assert result.checks[0].product == stim.PauliString("+XXXX_")
    _assert_exact_channel(source, result.circuit)


def test_z_check_exposes_data_mpp_and_moves_body_behind_it() -> None:
    source = stim.Circuit("R 4\nCX 0 4 1 4\nM 4")

    result = rewrite_to_mpp(source)

    assert result.circuit == stim.Circuit("R 4\nMPP Z0*Z1\nCX 0 4 1 4")
    _assert_exact_channel(source, result.circuit)


@pytest.mark.parametrize(
    ("source_text", "product"),
    [
        pytest.param("RX 4\nCZ 4 0 4 1\nMX 4", "+ZZ___", id="x-reset"),
        pytest.param("RY 4\nCX 4 0\nMY 4", "+X____", id="y-reset"),
        pytest.param("R 4\nH 4\nCX 4 0 4 1\nH 4\nM !4", "-XX___", id="inverted"),
    ],
)
def test_matching_source_reset_factor_is_removed(source_text: str, product: str) -> None:
    source = stim.Circuit(source_text)

    result = rewrite_to_mpp(source)

    assert str(result.checks[0].product) == product
    assert 4 not in result.checks[0].product.pauli_indices()
    _assert_exact_channel(source, result.circuit)


def test_data_reset_factors_are_not_removed() -> None:
    source = stim.Circuit("R 0 1 2 4\nCX 0 4 1 4\nM 4")

    result = rewrite_to_mpp(source)

    assert result.checks[0].product == stim.PauliString("+ZZ___")
    _assert_exact_channel(source, result.circuit)


def test_mismatched_reset_factor_keeps_full_product_and_body() -> None:
    # This is the minimal counterexample to dropping a restricted residual
    # frame. Keeping the exact body behind the MPP preserves the record-
    # dependent Clifford action on data qubit 0.
    source = stim.Circuit("RX 2\nCZ 0 2\nS 2\nMX 2")

    result = rewrite_to_mpp(source)

    assert result.circuit == stim.Circuit("RX 2\nMPP !Z0*Y2\nCZ 0 2\nS 2")
    assert result.checks[0].product == stim.PauliString("-Z_Y")
    _assert_exact_channel(source, result.circuit)


def test_measure_reset_contracts_replaced_ancilla_body() -> None:
    source = stim.Circuit(
        """
        R 4
        CX 0 4 1 4
        MR 4
        CX 0 4 1 4
        MR 4
        DETECTOR rec[-1] rec[-2]
        """
    )

    result = rewrite_to_mpp(source)

    assert result.circuit == stim.Circuit(
        """
        R 4
        MPP Z0*Z1
        R 4
        MPP Z0*Z1
        R 4
        DETECTOR rec[-1] rec[-2]
        """
    )
    assert result.foliation_circuit == stim.Circuit(
        """
        MPP Z0*Z1
        TICK
        MPP Z0*Z1
        DETECTOR rec[-1] rec[-2]
        """
    )
    assert result.eliminated_qubits == (4,)
    _assert_exact_channel(source, result.circuit)


def test_retained_reset_target_does_not_hide_contracted_round_boundary() -> None:
    source = stim.Circuit(
        """
        R 4
        CX 0 4 1 4
        MR 4
        R 5
        CX 0 4 1 4
        MR 4
        """
    )

    result = rewrite_to_mpp(source)

    assert result.foliation_circuit == stim.Circuit(
        """
        MPP Z0*Z1
        R 5
        TICK
        MPP Z0*Z1
        """
    )
    assert result.eliminated_qubits == (4,)
    _assert_exact_channel(source, result.circuit)


def test_source_tick_supplies_contracted_round_boundary() -> None:
    source = stim.Circuit(
        """
        R 4
        CX 0 4 1 4
        MR 4
        TICK
        CX 0 4 1 4
        MR 4
        """
    )

    result = rewrite_to_mpp(source)

    assert result.foliation_circuit == stim.Circuit(
        """
        MPP Z0*Z1
        TICK
        MPP Z0*Z1
        """
    )
    _assert_exact_channel(source, result.circuit)


def test_duplicate_mpp_support_starts_a_new_internal_layer() -> None:
    source = stim.Circuit("MPP Z0*Z1 X2*X3 !Z1*Z0 X3*X2")

    result = rewrite_to_mpp(source)

    assert result.circuit == source
    assert result.foliation_circuit == stim.Circuit(
        """
        MPP Z0*Z1 X2*X3
        TICK
        MPP !Z1*Z0 X3*X2
        """
    )


def test_measure_reset_keeps_noncontractible_data_clifford() -> None:
    source = stim.Circuit("R 2\nH 0\nMR 2")

    result = rewrite_to_mpp(source)

    assert result.circuit == stim.Circuit("R 2\nMPAD 0\nH 0\nR 2")
    _assert_exact_channel(source, result.circuit)


def test_measure_reset_tag_is_copied_to_measurement_and_reset() -> None:
    result = rewrite_to_mpp("R 2\nCX 0 2\nMR[round=1] 2")

    measurement = result.circuit[1]
    reset = result.circuit[-1]
    assert isinstance(measurement, stim.CircuitInstruction)
    assert isinstance(reset, stim.CircuitInstruction)
    assert measurement.tag == "round=1"
    assert reset.tag == "round=1"


def test_multiple_targets_update_reset_stabilizers_in_record_order() -> None:
    # The early Y measurement on qubit 2 anticommutes with the reset Z on
    # qubit 3 after pull-back. The later qubit-3 measurement must therefore
    # retain its Z3 factor instead of substituting a stabilizer that no longer
    # exists.
    source = stim.Circuit(
        """
        R 3
        SQRT_XX 3 2
        YCZ 1 0
        CZ 0 1
        MY 1 0 2 2 2 3
        """
    )

    result = rewrite_to_mpp(source)

    assert result.checks[-1].product == stim.PauliString("-__XZ")
    _assert_exact_channel(source, result.circuit)


def test_repeated_measure_reset_targets_preserve_correlated_records() -> None:
    source = stim.Circuit("MR 0 0")

    result = rewrite_to_mpp(source)

    assert result.circuit == source
    _assert_exact_channel(source, result.circuit)


def test_negative_identity_stays_a_real_signed_measurement() -> None:
    source = stim.Circuit("R 4\nM !4")

    result = rewrite_to_mpp(source)

    assert result.circuit == stim.Circuit("R 4\nM !4")
    assert result.checks[0].product == stim.PauliString("-____Z")
    assert "MPAD 1" not in str(result.circuit)
    _assert_exact_channel(source, result.circuit)


def test_positive_identity_becomes_mpad() -> None:
    source = stim.Circuit("R 4\nM 4")

    result = rewrite_to_mpp(source)

    assert result.circuit == stim.Circuit("R 4\nMPAD 0")
    assert result.checks[0].product == stim.PauliString("+_____")
    _assert_exact_channel(source, result.circuit)


def test_mixed_products_preserve_record_order() -> None:
    source = stim.Circuit("R 2 3\nCX 0 2\nM 2 3")

    result = rewrite_to_mpp(source)

    assert result.circuit == stim.Circuit("R 2 3\nMPP Z0\nMPAD 0\nCX 0 2")
    assert [check.measurement_index for check in result.checks] == [0, 1]
    assert [str(check.product) for check in result.checks] == ["+Z___", "+____"]
    _assert_exact_channel(source, result.circuit)


def test_mpad_advances_check_mapping_index() -> None:
    result = rewrite_to_mpp("MPAD 0 1\nR 4\nCX 0 4\nM 4")

    assert [check.measurement_index for check in result.checks] == [2]
    assert result.circuit.num_measurements == 3


def test_reusing_unreset_measurement_post_state_is_exact() -> None:
    source = stim.Circuit("R 4\nCX 0 4\nM 4\nCX 0 4\nM 4")

    result = rewrite_to_mpp(source)

    assert result.circuit == stim.Circuit("R 4\nMPP Z0\nMPAD 0\nCX 0 4 0 4")
    assert result.fallback_segments == ()
    _assert_exact_channel(source, result.circuit)


def test_reset_after_entangling_is_a_deterministic_flush_barrier() -> None:
    source = stim.Circuit("R 4\nCX 0 4\nR 4\nM 4")

    result = rewrite_to_mpp(source)

    assert result.circuit == stim.Circuit("R 4\nCX 0 4\nR 4\nMPAD 0")
    _assert_exact_channel(source, result.circuit)


def test_feedback_is_a_deterministic_flush_barrier() -> None:
    source = stim.Circuit("R 0 1\nX 0\nM 0\nCX rec[-1] 1\nM 1")

    result = rewrite_to_mpp(source)

    assert result.circuit == stim.Circuit("R 0 1\nMPP !Z0\nX 0\nCX rec[-1] 1\nM 1")
    assert result.fallback_segments == ()
    _assert_exact_channel(source, result.circuit)


def test_sweep_control_is_rejected() -> None:
    with pytest.raises(UnsupportedSyndromeCircuitError, match="Classical feedback"):
        rewrite_to_mpp("R 0\nM 0\nCX sweep[0] 1\nM 1")


@pytest.mark.parametrize("gate", ["SPP", "SPP_DAG"])
def test_variable_length_spp_is_moved_behind_measurement(gate: str) -> None:
    source = stim.Circuit(f"R 2\n{gate} X0*Z1\nCX 0 2\nM 2")

    result = rewrite_to_mpp(source)

    assert gate in str(result.circuit)
    assert result.circuit[1].name == "MPP"
    _assert_exact_channel(source, result.circuit)


def test_mpp_instruction_passes_through_verbatim() -> None:
    source = stim.Circuit("MPP X0*X1 Z2*Z3")

    result = rewrite_to_mpp(source)

    assert result.circuit == source
    assert all(check.source_qubit is None for check in result.checks)


def test_mpp_repeated_qubit_factors_reduce_in_mapping() -> None:
    source = stim.Circuit("MPP X0*X1*X0 X2*X2")

    result = rewrite_to_mpp(source)

    assert result.circuit == source
    assert [str(check.product) for check in result.checks] == ["+_X_", "+___"]


def test_mpp_high_qubit_indices_parse_exactly() -> None:
    source = stim.Circuit("MPP Z1*Z2 X0*X3 !Z0*Z1*Z3 Y0*X1*Z2")

    result = rewrite_to_mpp(source)

    assert [str(check.product) for check in result.checks] == ["+_ZZ_", "+X__X", "-ZZ_Z", "+YXZ_"]


def test_nonhermitian_repeated_mpp_product_is_rejected() -> None:
    with pytest.raises(UnsupportedSyndromeCircuitError, match="Non-Hermitian"):
        rewrite_to_mpp("MPP X0*Y0")


def test_pair_measurement_is_pulled_and_body_is_retained() -> None:
    source = stim.Circuit("R 2\nCX 0 2\nMZZ 1 2")

    result = rewrite_to_mpp(source)

    assert result.circuit == stim.Circuit("R 2\nMPP Z0*Z1*Z2\nCX 0 2")
    _assert_exact_channel(source, result.circuit)


def test_annotations_and_coordinates_are_preserved() -> None:
    source = stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        R 0 1
        CX 0 1
        M 1
        DETECTOR(0, 0) rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )

    result = rewrite_to_mpp(source)

    assert result.circuit.get_final_qubit_coordinates() == source.get_final_qubit_coordinates()
    assert result.circuit.num_detectors == source.num_detectors
    assert result.circuit.num_observables == source.num_observables
    _assert_same_reference_signs(result.circuit, source)
    _assert_exact_channel(source, result.circuit)


@pytest.mark.parametrize(
    "generator",
    ["surface_code:rotated_memory_z", "repetition_code:memory"],
)
def test_generated_memories_rewrite_exactly(generator: str) -> None:
    source = stim.Circuit.generated(generator, distance=3, rounds=3).flattened()

    result = rewrite_to_mpp(source)

    assert result.circuit.num_detectors == source.num_detectors
    assert result.circuit.num_observables == source.num_observables
    assert result.fallback_segments == ()
    _assert_exact_channel(source, result.circuit)


def test_surface_code_checks_are_data_only() -> None:
    source = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3)

    result = rewrite_to_mpp(source)

    syndrome_checks = [check for check in result.checks if check.segment_index < 3]
    assert len(syndrome_checks) == 24
    assert {len(check.product.pauli_indices()) for check in syndrome_checks} == {2, 4}


def test_noise_instruction_is_rejected() -> None:
    with pytest.raises(UnsupportedSyndromeCircuitError, match="DEPOLARIZE1"):
        rewrite_to_mpp("DEPOLARIZE1(0.01) 0")


def test_noisy_measurement_is_rejected() -> None:
    with pytest.raises(UnsupportedSyndromeCircuitError, match="Noisy measurement"):
        rewrite_to_mpp("M(0.01) 0")


def test_fallback_modes_are_compatibility_aliases() -> None:
    source = stim.Circuit("R 4\nCX 0 4\nM 4")

    circuit_mode = rewrite_to_mpp(source, fallback="circuit")
    segment_mode = rewrite_to_mpp(source, fallback="segment")

    assert circuit_mode == segment_mode
    assert circuit_mode.fallback_segments == ()


def test_unknown_fallback_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="fallback must be"):
        rewrite_to_mpp("M 0", fallback=cast('Literal["circuit", "segment"]', "tick"))


def test_empty_circuit_rewrites_to_empty_circuit() -> None:
    result = rewrite_to_mpp("")

    assert result.circuit == stim.Circuit()
    assert result.checks == ()


def test_rewriter_does_not_call_stim_flow_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> object:
        msg = "the exact rewrite must not consult Stim flow analysis"
        raise AssertionError(msg)

    monkeypatch.setattr(stim.Circuit, "flow_generators", fail)
    monkeypatch.setattr(stim.Circuit, "has_flow", fail)

    result = rewrite_to_mpp("R 4\nCX 0 4\nM 4")

    assert result.checks[0].product == stim.PauliString("+Z____")


def test_random_clifford_reset_measure_circuits_preserve_canonical_flows() -> None:
    rng = random.Random(0xC1FF0AD)  # ruff:ignore[suspicious-non-cryptographic-random-usage]
    one_qubit_gates = ["H", "S", "SQRT_X", "SQRT_Y", "X", "Y", "Z", "H_XY", "H_YZ", "C_XYZ"]
    two_qubit_gates = ["CX", "CY", "CZ", "XCZ", "YCZ", "SWAP", "ISWAP", "SQRT_XX", "SQRT_YY"]

    for _case in range(250):
        num_qubits = rng.randrange(1, 5)
        source = stim.Circuit()
        for _instruction in range(rng.randrange(1, 14)):
            choice = rng.random()
            if choice < 0.2:
                source.append(rng.choice(["R", "RX", "RY"]), [rng.randrange(num_qubits)])
            elif choice < 0.65:
                if num_qubits > 1 and rng.random() < 0.6:
                    source.append(rng.choice(two_qubit_gates), rng.sample(range(num_qubits), 2))
                else:
                    source.append(rng.choice(one_qubit_gates), [rng.randrange(num_qubits)])
            elif choice < 0.88:
                targets = rng.choices(range(num_qubits), k=rng.randrange(1, num_qubits + 1))
                source.append(rng.choice(["M", "MX", "MY"]), targets)
            else:
                targets = rng.choices(range(num_qubits), k=rng.randrange(1, num_qubits + 1))
                source.append(rng.choice(["MR", "MRX", "MRY"]), targets)

        rewritten = rewrite_to_mpp(source).circuit

        assert rewritten.flow_generators() == source.flow_generators(), f"source:\n{source}\nrewritten:\n{rewritten}"
