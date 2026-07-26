"""Tests for rewriting Stim syndrome-extraction circuits into MPP form."""

from __future__ import annotations

import numpy as np
import pytest
import stim

from graphqomb.stim_mpp_rewriter import (
    MppRewriteVerificationError,
    UnsupportedSyndromeCircuitError,
    rewrite_to_mpp,
)


def _assert_same_reference_signs(left: stim.Circuit, right: stim.Circuit) -> None:
    left_detectors, left_observables = left.reference_detector_and_observable_signs()
    right_detectors, right_observables = right.reference_detector_and_observable_signs()
    assert np.array_equal(left_detectors, right_detectors)
    assert np.array_equal(left_observables, right_observables)


def test_x_check_gadget_becomes_single_mpp() -> None:
    result = rewrite_to_mpp(
        """
        R 4
        H 4
        CX 4 0 4 1 4 2 4 3
        H 4
        M 4
        """
    )

    assert result.circuit == stim.Circuit("R 4\nMPP X0*X1*X2*X3")
    assert len(result.checks) == 1
    assert result.checks[0].measurement_index == 0
    assert result.checks[0].source_qubit == 4
    assert result.checks[0].product == stim.PauliString("+XXXX_")


def test_z_check_gadget_becomes_single_mpp() -> None:
    result = rewrite_to_mpp(
        """
        R 4
        CX 0 4 1 4
        M 4
        """
    )

    assert result.circuit == stim.Circuit("R 4\nMPP Z0*Z1")


def test_x_prepared_ancilla_with_cz_body() -> None:
    result = rewrite_to_mpp(
        """
        RX 4
        CZ 4 0 4 1
        MX 4
        """
    )

    assert result.circuit == stim.Circuit("RX 4\nMPP Z0*Z1")


def test_y_prepared_ancilla() -> None:
    result = rewrite_to_mpp(
        """
        RY 4
        CX 4 0
        MY 4
        """
    )

    assert result.circuit == stim.Circuit("RY 4\nMPP X0")


def test_inverted_ancilla_measurement_inverts_the_product() -> None:
    result = rewrite_to_mpp(
        """
        R 4
        H 4
        CX 4 0 4 1
        H 4
        M !4
        """
    )

    assert result.circuit == stim.Circuit("R 4\nMPP !X0*X1")
    assert result.checks[0].product.sign == -1


def test_round_one_data_resets_are_not_stripped_from_products() -> None:
    result = rewrite_to_mpp(
        """
        R 0 1 2 4
        CX 0 4 1 4
        M 4
        """
    )

    assert result.circuit == stim.Circuit("R 0 1 2 4\nMPP Z0*Z1")


def test_trivial_data_measurements_pass_through_verbatim() -> None:
    result = rewrite_to_mpp(
        """
        R 4
        CX 0 4 1 4
        M 4
        DETECTOR rec[-1]
        M 0 1
        """
    )

    assert result.circuit == stim.Circuit("R 4\nMPP Z0*Z1\nDETECTOR rec[-1]\nM 0 1")
    assert [check.source_qubit for check in result.checks] == [4, 0, 1]


def test_measure_reset_splits_into_mpp_and_reset() -> None:
    result = rewrite_to_mpp(
        """
        R 4
        CX 0 4 1 4
        MR 4
        CX 0 4 1 4
        MR 4
        DETECTOR rec[-1] rec[-2]
        """
    )

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


def test_detectors_and_observables_are_copied_verbatim() -> None:
    source = stim.Circuit(
        """
        R 0 1 2 3 4
        CX 0 3 1 3 1 4 2 4
        MR 3 4
        DETECTOR(1, 0) rec[-2]
        DETECTOR(3, 0) rec[-1]
        CX 0 3 1 3 1 4 2 4
        MR 3 4
        DETECTOR(1, 1) rec[-2] rec[-4]
        DETECTOR(3, 1) rec[-1] rec[-3]
        M 0 1 2
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )

    result = rewrite_to_mpp(source)

    assert result.circuit.num_measurements == source.num_measurements
    assert result.circuit.num_detectors == source.num_detectors
    _assert_same_reference_signs(result.circuit, source)


def test_generated_surface_code_memory_rewrites_and_verifies() -> None:
    source = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3)

    result = rewrite_to_mpp(source)

    assert result.circuit.num_measurements == source.num_measurements
    assert result.circuit.num_detectors == source.num_detectors
    _assert_same_reference_signs(result.circuit, source)
    # Middle rounds infer one weight-2 or weight-4 stabilizer per ancilla.
    middle_round = [check for check in result.checks if check.segment_index == 1]
    assert len(middle_round) == 8
    assert {len(check.product.pauli_indices()) for check in middle_round} == {2, 4}


def test_generated_repetition_code_memory_rewrites_and_verifies() -> None:
    source = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=4)

    result = rewrite_to_mpp(source)

    assert result.circuit.num_measurements == source.num_measurements
    _assert_same_reference_signs(result.circuit, source)


def test_mpp_instruction_passes_through_verbatim() -> None:
    result = rewrite_to_mpp("MPP X0*X1 Z2*Z3")

    assert result.circuit == stim.Circuit("MPP X0*X1 Z2*Z3")
    assert result.checks[0].source_qubit is None


def test_mpp_repeated_qubit_factors_are_reduced_in_mapping_and_pass_through() -> None:
    source = stim.Circuit("MPP X0*X1*X0 X2*X2")

    result = rewrite_to_mpp(source)

    assert result.circuit == source
    assert [check.product for check in result.checks] == [
        stim.PauliString("+_X_"),
        stim.PauliString("+___"),
    ]


def test_mpp_repeated_qubit_factors_must_form_a_hermitian_product() -> None:
    with pytest.raises(UnsupportedSyndromeCircuitError, match="Non-Hermitian"):
        rewrite_to_mpp("MPP X0*Y0")


def test_pair_measurement_passes_through_verbatim_without_body() -> None:
    result = rewrite_to_mpp("MZZ 0 1 2 3")

    assert result.circuit == stim.Circuit("MZZ 0 1 2 3")
    assert [check.product for check in result.checks] == [
        stim.PauliString("+ZZ__"),
        stim.PauliString("+__ZZ"),
    ]


def test_pair_measurement_entangling_an_unmeasured_ancilla_fails_verification() -> None:
    # Qubit 2 stays entangled with the data after MZZ, so dropping the CX
    # would lose the 1 -> Z1*Z2 xor rec correlation.
    with pytest.raises(MppRewriteVerificationError):
        rewrite_to_mpp(
            """
            R 2
            CX 0 2
            MZZ 1 2
            """
        )


def test_mpp_tag_is_preserved() -> None:
    result = rewrite_to_mpp(
        """
        R 4
        CX 0 4 1 4
        M[round1] 4
        """
    )

    last_instruction = result.circuit[-1]
    assert isinstance(last_instruction, stim.CircuitInstruction)
    assert last_instruction.tag == "round1"


def test_noise_instruction_is_rejected() -> None:
    with pytest.raises(UnsupportedSyndromeCircuitError, match="DEPOLARIZE1"):
        rewrite_to_mpp("DEPOLARIZE1(0.01) 0")


def test_noisy_measurement_is_rejected() -> None:
    with pytest.raises(UnsupportedSyndromeCircuitError, match="Noisy measurement"):
        rewrite_to_mpp("M(0.01) 0")


def test_classical_feedback_is_rejected() -> None:
    with pytest.raises(UnsupportedSyndromeCircuitError, match="feedback"):
        rewrite_to_mpp("M 0\nCX rec[-1] 1")


@pytest.mark.parametrize(
    ("measurement", "expected_circuit", "expected_product"),
    [
        pytest.param("M 4", "R 4\nMPAD 0", "+_____", id="positive-identity"),
        pytest.param("M !4", "R 4\nMPAD 1", "-_____", id="negative-identity"),
    ],
)
def test_identity_measurement_becomes_mpad(
    measurement: str,
    expected_circuit: str,
    expected_product: str,
) -> None:
    result = rewrite_to_mpp(f"R 4\n{measurement}")

    assert result.circuit == stim.Circuit(expected_circuit)
    assert result.checks[0].product == stim.PauliString(expected_product)


def test_identity_measure_reset_preserves_tag_and_reset() -> None:
    result = rewrite_to_mpp("R 4\nMR[empty-check] 4")

    assert result.circuit == stim.Circuit("R 4\nMPAD[empty-check] 0\nR[empty-check] 4")
    assert result.checks[0].product == stim.PauliString("+_____")


def test_mixed_nontrivial_and_identity_measurements_preserve_record_order() -> None:
    result = rewrite_to_mpp(
        """
        R 2 3
        CX 0 2
        M 2 3
        """
    )

    assert result.circuit == stim.Circuit("R 2 3\nMPP Z0\nMPAD 0")
    assert [check.measurement_index for check in result.checks] == [0, 1]
    assert [check.product for check in result.checks] == [
        stim.PauliString("+Z___"),
        stim.PauliString("+____"),
    ]


def test_mpad_advances_following_check_mapping_index() -> None:
    result = rewrite_to_mpp(
        """
        MPAD 0 1
        R 4
        CX 0 4
        M 4
        """
    )

    assert result.circuit == stim.Circuit("MPAD 0 1\nR 4\nMPP Z0")
    assert result.circuit.num_measurements == 3
    assert [check.measurement_index for check in result.checks] == [2]


def test_matching_flow_generators_skip_per_flow_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_on_per_flow_verification(_circuit: stim.Circuit, _flow: stim.Flow, *, unsigned: bool = False) -> bool:
        del unsigned
        msg = "matching canonical generators should not invoke per-flow verification"
        raise AssertionError(msg)

    monkeypatch.setattr(stim.Circuit, "has_flow", fail_on_per_flow_verification)

    result = rewrite_to_mpp("R 4\nCX 0 4 1 4\nM 4")

    assert result.circuit == stim.Circuit("R 4\nMPP Z0*Z1")


def test_reusing_unreset_measured_qubit_is_rejected() -> None:
    with pytest.raises(UnsupportedSyndromeCircuitError, match="measured-but-not-reset"):
        rewrite_to_mpp(
            """
            R 4
            CX 0 4
            M 4
            CX 0 4
            M 4
            """
        )


def test_residual_data_unitary_fails_verification() -> None:
    with pytest.raises(MppRewriteVerificationError):
        rewrite_to_mpp(
            """
            R 4
            H 4
            CX 4 0
            H 4
            S 0
            M 4
            """
        )


def test_residual_data_unitary_is_dropped_silently_without_verification() -> None:
    result = rewrite_to_mpp(
        """
        R 4
        H 4
        CX 4 0
        H 4
        S 0
        M 4
        """,
        verify=False,
    )

    assert result.circuit == stim.Circuit("R 4\nMPP X0")


def test_basis_mismatched_residual_on_measured_ancilla_is_kept_in_the_product() -> None:
    # The ancilla measurement pulls back to X on the Z-prepared, measured-out
    # qubit 5. The mismatched support cannot be substituted away, so it stays
    # in the product; flow verification confirms the rewrite is still sound.
    result = rewrite_to_mpp(
        """
        R 4 5
        H 4
        CX 4 0 4 5
        H 4 5
        M 4 5
        """
    )

    assert result.circuit == stim.Circuit("R 4 5\nMPP X0*X5 X5")


def test_rewrite_accepts_circuit_objects() -> None:
    source = stim.Circuit("R 4\nCX 0 4 1 4\nM 4")

    assert rewrite_to_mpp(source).circuit == stim.Circuit("R 4\nMPP Z0*Z1")
