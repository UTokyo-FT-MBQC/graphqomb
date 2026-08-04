"""Tests for rewriting Stim syndrome-extraction circuits into MPP form."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pytest
import stim

from graphqomb.stim_glue.mpp_rewriter import (
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


def test_inverted_mpp_target_sign_is_reduced_into_the_mapping() -> None:
    result = rewrite_to_mpp("MPP !X0*X1")

    assert result.circuit == stim.Circuit("MPP !X0*X1")
    assert result.checks[0].product == stim.PauliString("-XX")


def test_inverted_pair_measurement_target_inverts_the_product() -> None:
    result = rewrite_to_mpp("MZZ !0 1")

    assert result.circuit == stim.Circuit("MZZ !0 1")
    assert result.checks[0].product == stim.PauliString("-ZZ")


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

    assert result.circuit == stim.Circuit("R 4\nMPP Z0*Z1\nDETECTOR rec[-1]\nTICK\nM 0 1")
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


def test_data_measurements_after_measure_reset_pass_through_verbatim() -> None:
    result = rewrite_to_mpp(
        """
        R 2
        CX 0 2 1 2
        MR 2
        DETECTOR rec[-1]
        M 0 1
        """
    )

    assert result.circuit == stim.Circuit("R 2\nMPP Z0*Z1\nR 2\nDETECTOR rec[-1]\nM 0 1")
    assert [check.segment_index for check in result.checks] == [0, 1, 1]


def test_consecutive_measure_resets_split_into_segments() -> None:
    result = rewrite_to_mpp("MR 4\nDETECTOR rec[-1]\nMR 4")

    assert result.circuit == stim.Circuit("MR 4\nDETECTOR rec[-1]\nMPAD 0\nR 4")
    assert [check.segment_index for check in result.checks] == [0, 1]


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
    # The final transversal data readout starts its own segment and stays a
    # plain weight-1 Z measurement instead of dragging reset-ancilla factors.
    final_segment = max(check.segment_index for check in result.checks)
    data_checks = [check for check in result.checks if check.segment_index == final_segment]
    assert len(data_checks) == 9
    assert all(
        check.source_qubit is not None
        and check.product.sign == 1
        and check.product.pauli_indices() == [check.source_qubit]
        and check.product[check.source_qubit] == 3
        for check in data_checks
    )


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


def test_pair_measurement_preserves_residual_entangling_frame() -> None:
    source = stim.Circuit(
        """
        R 2
        CX 0 2
        MZZ 1 2
        """
    )

    result = rewrite_to_mpp(source)

    assert result.circuit == stim.Circuit("R 2\nMPP Z0*Z1*Z2\nCX 0 2")
    assert result.circuit.flow_generators() == source.flow_generators()


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


def test_remeasuring_a_late_reset_qubit_starts_a_new_segment() -> None:
    result = rewrite_to_mpp(
        """
        R 4
        CX 0 4
        M 4
        R 4
        M 4
        """
    )

    assert result.circuit == stim.Circuit("R 4\nMPP Z0\nR 4\nMPAD 0")
    assert [check.segment_index for check in result.checks] == [0, 1]


def test_remeasuring_a_dirty_qubit_in_a_product_is_rejected() -> None:
    with pytest.raises(UnsupportedSyndromeCircuitError, match="never reset"):
        rewrite_to_mpp(
            """
            R 4
            CX 0 4
            M 4
            H 0
            MZZ 4 1
            """
        )


def test_reset_after_entangling_in_the_same_segment_is_rejected() -> None:
    with pytest.raises(UnsupportedSyndromeCircuitError, match="after it was entangled"):
        rewrite_to_mpp(
            """
            R 4
            CX 0 4
            R 4
            M 4
            """
        )


def test_residual_data_unitary_is_preserved() -> None:
    result = rewrite_to_mpp(
        """
        R 4
        H 4
        CX 4 0
        H 4
        S 0
        M 4
        """
    )

    assert result.circuit == stim.Circuit("R 4\nMPP X0\nS 0")


def test_residual_frame_is_preserved_across_segments() -> None:
    source = stim.Circuit(
        """
        R 0 1
        CX 0 1
        H 0
        M 1
        H 0
        M 0
        """
    )

    result = rewrite_to_mpp(source)

    assert result.circuit == stim.Circuit(
        """
        R 0 1
        MPP Z0
        H 0
        MPP X0
        """
    )
    assert np.array_equal(result.circuit.reference_sample(), source.reference_sample())


def test_unsupported_residual_channel_falls_back_to_gate_level_source() -> None:
    source = stim.Circuit(
        """
        R 1
        H 1
        CX 1 0
        M 1
        R 1
        H 1
        M 1
        """
    )

    result = rewrite_to_mpp(source)

    assert result.circuit == source.flattened()
    assert result.circuit.num_measurements == source.num_measurements
    assert np.array_equal(result.circuit.reference_sample(), source.reference_sample())
    assert [check.source_qubit for check in result.checks] == [1, 1]


def test_nonlocal_body_rewrites_via_channel_flows() -> None:
    source = stim.Circuit(
        """
        R 0 1
        SWAP 0 1
        M 0
        """
    )

    result = rewrite_to_mpp(source)

    assert result.circuit == stim.Circuit("R 0 1\nMPP Z1")
    assert result.checks[0].product == stim.PauliString("+_Z")
    assert result.circuit.flow_generators() == source.flow_generators()


@pytest.mark.parametrize(
    "residual",
    ["H 0", "S 0", "SQRT_X 0", "H 1", "S 1"],
)
def test_nonlocal_body_with_residual_frame_stays_flow_equivalent(residual: str) -> None:
    # A body that is not local on the surviving qubit is only representable
    # when the residual Clifford frame can be synthesized from the segment's
    # channel flows; otherwise the gate-level fallback must win. Either way the
    # rewrite has to keep every stabilizer flow of its source.
    source = stim.Circuit(f"R 0 1\nH 0\nSWAP 0 1\n{residual}\nM 0")

    result = rewrite_to_mpp(source)

    assert all(result.circuit.has_flow(flow) for flow in source.flow_generators())
    assert all(source.has_flow(flow) for flow in result.circuit.flow_generators())


def test_fallback_numbers_segments_like_the_optimized_rewrite() -> None:
    # A measurement after a late reset opens a new segment. The gate-level
    # fallback has to apply the same boundary rule as the streaming rewriter,
    # otherwise `CheckMapping.segment_index` depends on which path was taken.
    optimized = rewrite_to_mpp("R 0\nM 0\nR 0\nM 0")
    # The leading segment has an unrepresentable residual frame, so the whole
    # circuit takes the gate-level fallback while keeping the same boundary.
    fallback = rewrite_to_mpp("R 0 1\nH 0\nSWAP 0 1\nS 0\nM 0\nR 0\nM 0")

    assert fallback.circuit == stim.Circuit("R 0 1\nH 0\nSWAP 0 1\nS 0\nM 0\nR 0\nM 0")
    assert [check.segment_index for check in optimized.checks] == [0, 1]
    assert [check.segment_index for check in fallback.checks] == [0, 1]


def test_fallback_preserves_mpad_bits_and_measurement_indices() -> None:
    source = stim.Circuit(
        """
        R 1
        H 1
        CX 1 0
        M 1
        R 1
        H 1
        MPAD 0 1
        M 1
        """
    )

    result = rewrite_to_mpp(source)

    assert result.circuit == source.flattened()
    assert [check.measurement_index for check in result.checks] == [0, 3]
    assert [check.source_qubit for check in result.checks] == [1, 1]


def test_fallback_preserves_signed_composite_measurements() -> None:
    source = stim.Circuit(
        """
        QUBIT_COORDS(1, 2) 1
        QUBIT_COORDS(3, 4) 2
        R 1
        H 1
        CX 1 0
        M 1
        R 1
        H 1
        MPP !X1*Y2*Z0
        MXX !1 2
        """
    )

    result = rewrite_to_mpp(source)

    assert result.circuit == source.flattened()
    assert np.array_equal(result.circuit.reference_sample(), source.reference_sample())
    assert [check.measurement_index for check in result.checks] == [0, 1, 2]


def test_fallback_preserves_qubit_coordinates() -> None:
    source = stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        R 0 1
        H 0
        SWAP 0 1
        S 0
        M 0
        R 0
        M 0
        """
    )

    result = rewrite_to_mpp(source)

    assert result.circuit.get_final_qubit_coordinates() == source.get_final_qubit_coordinates()


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


def test_empty_circuit_rewrites_to_empty_circuit() -> None:
    result = rewrite_to_mpp("")

    assert result.circuit == stim.Circuit()
    assert result.checks == ()


def test_segment_fallback_matches_circuit_mode_when_every_segment_rewrites() -> None:
    source = "R 4\nCX 0 4 1 4\nM 4\nR 4\nM 4"

    circuit_mode = rewrite_to_mpp(source)
    segment_mode = rewrite_to_mpp(source, fallback="segment")

    assert segment_mode.circuit == circuit_mode.circuit
    assert segment_mode.checks == circuit_mode.checks
    assert circuit_mode.fallback_segments == ()
    assert segment_mode.fallback_segments == ()


def test_segment_fallback_keeps_only_the_failing_segment_gate_level() -> None:
    # The leading segment has an unrepresentable residual frame; the trailing
    # reset-and-measure segment still rewrites (to a deterministic MPAD).
    source = stim.Circuit("R 0 1\nH 0\nSWAP 0 1\nS 0\nM 0\nR 0\nM 0")

    result = rewrite_to_mpp(source, fallback="segment")

    assert result.circuit == stim.Circuit("R 0 1\nH 0\nSWAP 0 1\nS 0\nM 0\nR 0\nMPAD 0")
    assert result.fallback_segments == (0,)
    assert [check.segment_index for check in result.checks] == [0, 1]
    assert all(result.circuit.has_flow(flow) for flow in source.flow_generators())
    assert all(source.has_flow(flow) for flow in result.circuit.flow_generators())


def test_whole_circuit_fallback_lists_every_segment() -> None:
    result = rewrite_to_mpp("R 0 1\nH 0\nSWAP 0 1\nS 0\nM 0\nR 0\nM 0")

    assert result.fallback_segments == (0, 1)


def test_segment_fallback_forces_reused_post_state_producer_gate_level() -> None:
    # Segment 1 reuses ancilla 4's measurement post-state, which a rewritten
    # segment 0 would trash (its rewrite deletes the entangling body), so
    # segment 0 is forced verbatim; segment 1 then rewrites against the
    # faithful post-state.
    source = stim.Circuit("R 4\nCX 0 4\nM 4\nH 4\nM 4")

    result = rewrite_to_mpp(source, fallback="segment")

    assert result.circuit == stim.Circuit("R 4\nCX 0 4\nM 4\nMPP X4")
    assert result.fallback_segments == (0,)
    assert [str(check.product) for check in result.checks] == ["+____Z", "+____X"]


def test_segment_fallback_keeps_unrewritable_reuse_chain_gate_level() -> None:
    # Both segments depend on ancilla 4's post-state and neither rewrite
    # verifies, so the result is the gate-level source, segment by segment.
    source = stim.Circuit("R 4\nCX 0 4\nM 4\nCX 0 4\nM 4")

    result = rewrite_to_mpp(source, fallback="segment")

    assert result.circuit == source
    assert result.fallback_segments == (0, 1)
    assert all(result.circuit.has_flow(flow) for flow in source.flow_generators())


def test_segment_fallback_keeps_feedback_segment_gate_level() -> None:
    source = stim.Circuit("R 0 1\nM 0\nCX rec[-1] 1\nM 1\nDETECTOR rec[-1]")

    result = rewrite_to_mpp(source, fallback="segment")

    assert result.circuit == stim.Circuit("R 0 1\nMPAD 0\nCX rec[-1] 1\nM 1\nDETECTOR rec[-1]")
    assert result.fallback_segments == (1,)
    _assert_same_reference_signs(result.circuit, source)


def test_segment_fallback_still_rejects_noise() -> None:
    with pytest.raises(UnsupportedSyndromeCircuitError, match="Unsupported instruction"):
        rewrite_to_mpp("X_ERROR(0.1) 0\nM 0", fallback="segment")


def test_segment_fallback_still_rejects_noisy_measurement() -> None:
    with pytest.raises(UnsupportedSyndromeCircuitError, match="Noisy measurement"):
        rewrite_to_mpp("R 0\nM(0.01) 0", fallback="segment")


def test_segment_fallback_still_rejects_sweep_bit_controls() -> None:
    with pytest.raises(UnsupportedSyndromeCircuitError, match="Classical feedback"):
        rewrite_to_mpp("R 0\nM 0\nCX sweep[0] 1\nM 1", fallback="segment")


def test_rewrite_rejects_unknown_fallback_mode() -> None:
    with pytest.raises(ValueError, match="fallback must be"):
        rewrite_to_mpp("M 0", fallback=cast('Literal["circuit", "segment"]', "tick"))


def test_segment_fallback_keeps_negative_deterministic_record_importable() -> None:
    # A deterministic-minus record would become MPAD 1, which the Stim
    # importer rejects; under segment fallback the unsubstituted retry keeps
    # it a signed MPP so the record a verbatim feedback segment consumes
    # survives with its parity.
    source = stim.Circuit("R 0 1\nX 0\nM 0\nCX rec[-1] 1\nM 1")

    result = rewrite_to_mpp(source, fallback="segment")

    assert result.circuit == stim.Circuit("R 0 1\nMPP !Z0\nCX rec[-1] 1\nM 1")
    assert result.fallback_segments == (1,)
    _assert_same_reference_signs(
        result.circuit + stim.Circuit("DETECTOR rec[-1] rec[-2]"),
        source + stim.Circuit("DETECTOR rec[-1] rec[-2]"),
    )


def test_segment_fallback_keeps_inverted_identity_measurement_verbatim() -> None:
    # An inverted identity measurement stays a real measurement instead of the
    # MPAD 1 pad the circuit-level rewrite emits.
    result = rewrite_to_mpp("R 4\nM !4", fallback="segment")

    assert result.circuit == stim.Circuit("R 4\nM !4")
    assert result.fallback_segments == ()
