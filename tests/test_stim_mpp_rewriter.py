"""Record-channel and local-extraction tests for photonic MPP rewriting."""

from __future__ import annotations

import random

import numpy as np
import pytest
import stim

from graphqomb.stim_glue.mpp_rewriter import UnsupportedSyndromeCircuitError, rewrite_to_mpp
from tests.stim_reference import assert_record_channel


def _check_channel(text: str | stim.Circuit) -> None:
    source = stim.Circuit(text) if isinstance(text, str) else text
    result = rewrite_to_mpp(source)
    assert result.circuit is result.foliation_circuit
    assert result.checks is result.foliation_checks
    assert_record_channel(source, result.circuit, result.foliation_record_to_source)


@pytest.mark.parametrize(
    ("text", "product"),
    [
        ("RX 4\nCX 4 0 4 1\nMX 4", "+XX___"),
        ("RX 4\nCZ 4 0 4 1\nMX !4", "-ZZ___"),
        ("R 4\nCX 0 4 1 4\nM 4", "+ZZ___"),
    ],
)
def test_extraction_replaces_body_and_its_preparation(text: str, product: str) -> None:
    result = rewrite_to_mpp(text)
    assert len(result.circuit) == 1
    assert result.circuit[0].name == "MPP"
    assert result.checks[0].product == stim.PauliString(product)
    assert result.checks[0].source_qubit == 4
    assert result.eliminated_qubits == (4,)
    _check_channel(text)


def test_data_preparations_are_not_removed() -> None:
    source = "R 0 1 2 4\nCX 0 4 1 4\nM 4"
    result = rewrite_to_mpp(source)
    assert result.circuit == stim.Circuit("R 0 1 2\nMPP Z0*Z1")
    _check_channel(source)


@pytest.mark.parametrize(
    "text",
    [
        "RX 2\nCZ 0 2\nS 2\nMX 2",  # Non-projector action on surviving data.
        "RX 2\nCZ 2 0\nH 1\nMX 2\nM 1",  # Independent data Clifford.
        "R 2\nH 2\nCX 2 0\nH 2\nM 2",  # Unrecognized local basis wrapper.
        "RY 2\nCX 2 0\nMY 2",  # No special Y-gadget requirement.
        "X 0\nCX 0 1\nR 0\nM 1",  # Reset does not erase a partner's state.
        "H 0\nCX 0 1",  # No demand to remove an unmeasured output circuit.
        "H 0\nMPP X0*X1\nS 0\nM 0",
        "SPP X0*Z1\nM 0 1",
    ],
)
def test_unrecognized_intervals_keep_gates_without_duplicating_measurements(text: str) -> None:
    result = rewrite_to_mpp(text)
    assert result.circuit == stim.Circuit(text)
    _check_channel(text)


@pytest.mark.parametrize("readout", ["M", "MX", "MY", "MR", "MRX", "MRY"])
def test_terminal_readout_does_not_prepare_an_output(readout: str) -> None:
    text = f"H 0\n{readout}[readout] !0"
    result = rewrite_to_mpp(text)
    assert result.circuit.num_measurements == 1
    assert result.circuit[-1].name in {"M", "MX", "MY"}
    assert result.circuit[-1].tag == "readout"
    assert not any(i.name in {"R", "RX", "RY"} for i in result.circuit)
    _check_channel(text)


@pytest.mark.parametrize(
    ("readout", "reset", "measurement"),
    [
        ("M", "R", "M"),
        ("MX", "RX", "MX"),
        ("MY", "RY", "MY"),
        ("MR", "R", "M"),
        ("MRX", "RX", "MX"),
        ("MRY", "RY", "MY"),
    ],
)
def test_repeated_targets_measure_independent_positive_state(readout: str, reset: str, measurement: str) -> None:
    text = f"{readout} !0 0"
    result = rewrite_to_mpp(text)
    assert result.circuit == stim.Circuit(f"{measurement} !0\n{reset} 0\n{measurement} 0")
    assert not result.circuit.compile_sampler(seed=0).sample(16)[:, 1].any()
    _check_channel(text)


def test_plain_measurement_reuse_is_not_nondestructive_stim() -> None:
    result = rewrite_to_mpp("X 0\nM 0 0")
    np.testing.assert_array_equal(result.circuit.compile_sampler(seed=0).sample(1), [[True, False]])
    _check_channel("X 0\nM 0 0")


def test_explicit_reset_overrides_measurement_axis() -> None:
    text = "MY 0\nR[next] 0\nM 0"
    assert rewrite_to_mpp(text).circuit == stim.Circuit(text)
    _check_channel(text)


def test_only_replaced_lifetime_loses_its_preparation() -> None:
    text = "RX 2\nCZ 2 0\nMX 2\nRY[later] 2\nH 2\nMY 2"
    result = rewrite_to_mpp(text)
    assert result.circuit == stim.Circuit("MPP Z0\nRY[later] 2\nH 2\nMY 2")
    assert result.eliminated_qubits == ()
    _check_channel(text)


def test_later_reset_only_output_is_retained() -> None:
    text = "R 2\nCX 0 2\nM 2\nRX[output] 2"
    result = rewrite_to_mpp(text)
    assert result.circuit == stim.Circuit("MPP Z0\nRX[output] 2")
    assert not result.eliminated_qubits
    _check_channel(text)


def test_earlier_noncontracted_lifetime_is_retained() -> None:
    text = "RY 2\nH 2\nMY 2\nRX 2\nCZ 2 0\nMX 2"
    result = rewrite_to_mpp(text)
    assert result.circuit == stim.Circuit("RY 2\nH 2\nMY 2\nTICK\nMPP Z0")
    assert not result.eliminated_qubits
    _check_channel(text)


def test_measure_reset_tag_follows_actual_continuation_only() -> None:
    result = rewrite_to_mpp("MR[round] 0\nH 0")
    assert result.circuit == stim.Circuit("M[round] 0\nR[round] 0\nH 0")


def test_reset_and_measurement_tags_survive_extraction() -> None:
    result = rewrite_to_mpp("RX[prep] 2\nCZ 2 0\nMX[syndrome] 2\nR[next] 2\nM[end] 2")
    assert result.circuit == stim.Circuit("MPP[syndrome] Z0\nR[next] 2\nM[end] 2")


def test_readout_permutation_remaps_records_signs_padding_and_feedback() -> None:
    text = """
        MPAD 1 0
        RX 0 1 2 3
        RX 4 5
        CX 4 0 4 1 4 2 4 3
        CZ 5 0 5 1 5 2 5 3
        MX[readout] !0 1 2 3 4 !5
        DETECTOR[check] rec[-6] rec[-1]
        CX rec[-6] 6
        M 6
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-7]
    """
    result = rewrite_to_mpp(text)
    assert result.foliation_record_to_source == (0, 1, 6, 7, 2, 3, 4, 5, 8)
    assert [c.source_qubit for c in result.checks] == [4, 5, 0, 1, 2, 3, 6]
    assert [c.measurement_index for c in result.checks] == list(range(2, 9))
    assert [i.tag for i in result.circuit if i.name == "DETECTOR"] == ["check"]
    _check_channel(text)


def test_interleaved_anticommuting_factors_keep_their_phase() -> None:
    source = stim.Circuit("RX 2 3\nCX 2 0\nCZ 3 0 3 1\nCX 2 1\nMX 2 3")
    result = rewrite_to_mpp(source)
    assert result.circuit == source
    assert not result.eliminated_qubits
    _check_channel(source)
    rng = random.Random(29)  # ruff:ignore[suspicious-non-cryptographic-random-usage]
    contracted = 0
    for _ in range(40):
        gates = [("CX", [2, 0]), ("CX", [2, 1]), ("CZ", [3, 0]), ("CZ", [3, 1])]
        rng.shuffle(gates)
        source = stim.Circuit("RX 2 3")
        for name, targets in gates:
            source.append(name, targets)
        source.append("MX", rng.sample([2, 3], 2))
        contracted += bool(rewrite_to_mpp(source).eliminated_qubits)
        _check_channel(source)
    assert 0 < contracted < 40


@pytest.mark.parametrize("text", ["R 4\nM 4", "R 4\nM !4"])
def test_known_result_remains_a_physical_readout(text: str) -> None:
    result = rewrite_to_mpp(text)
    assert result.circuit == stim.Circuit(text)
    assert "MPAD" not in str(result.circuit)
    _check_channel(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("MPP Z0*Z1 X2*X3 !Z1*Z0", "MPP Z0*Z1 X2*X3\nTICK\nMPP !Z1*Z0"),
        ("MXX 0 1\nMZZ 1 2", "MPP X0*X1\nTICK\nMPP Z1*Z2"),
        ("MZZ 0 1\nMYY 0 1", "MPP Z0*Z1 Y0*Y1"),
        ("MYY[pair] !0 1 1 !0", "MPP[pair] !Y0*Y1\nTICK\nMPP[pair] Y1*!Y0"),
        ("MPP Z0*Z1 Z1*Z2 X0*X1*X2", "MPP Z0*Z1 Z1*Z2 X0*X1*X2"),
    ],
)
def test_product_layers_preserve_order_and_commutation(text: str, expected: str) -> None:
    assert rewrite_to_mpp(text).circuit == stim.Circuit(expected)
    _check_channel(text)


def test_mpp_repeated_factors_reduce_in_mapping() -> None:
    result = rewrite_to_mpp("MPP X0*X1*X0 X2*X2")
    assert [str(c.product) for c in result.checks] == ["+_X_", "+___"]
    _check_channel("MPP X0*X1*X0 X2*X2")


@pytest.mark.parametrize("text", ["MPP X0*Y0", "DEPOLARIZE1(0.01) 0", "M(0.01) 0", "M 0\nCX sweep[0] 1"])
def test_unsupported_input_is_rejected(text: str) -> None:
    with pytest.raises(UnsupportedSyndromeCircuitError):
        rewrite_to_mpp(text)


def test_empty_circuit() -> None:
    result = rewrite_to_mpp("")
    assert result.circuit == stim.Circuit()
    assert result.checks == ()
    assert result.foliation_record_to_source == result.eliminated_qubits == ()


def test_no_runtime_flow_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> object:
        msg = "production rewriting must not ask a flow oracle"
        raise AssertionError(msg)

    monkeypatch.setattr(stim.Circuit, "flow_generators", fail)
    monkeypatch.setattr(stim.Circuit, "has_flow", fail)
    for text in ["R 4\nCX 0 4\nMR 4\nCX 0 4\nM 4", "RX 2\nCZ 2 0\nS 2\nMX 2"]:
        result = rewrite_to_mpp(text)
        assert result.circuit.num_measurements == stim.Circuit(text).num_measurements


@pytest.mark.parametrize("generator", ["surface_code:rotated_memory_z", "repetition_code:memory"])
def test_generated_memories_preserve_record_channel(generator: str) -> None:
    _check_channel(stim.Circuit.generated(generator, distance=3, rounds=3))


def test_random_clifford_measure_reset_circuits_preserve_record_channel() -> None:
    rng = random.Random(0xC1FF0AD)  # ruff:ignore[suspicious-non-cryptographic-random-usage]
    for _ in range(150):
        source = stim.Circuit()
        for _ in range(rng.randrange(1, 14)):
            name = rng.choice(
                ["R", "RX", "RY", "H", "S", "SQRT_X", "CX", "CZ", "CY", "SWAP", "M", "MX", "MY", "MR", "MRX", "MRY"]
            )
            targets = rng.sample(range(4), 2 if name in {"CX", "CZ", "CY", "SWAP"} else 1)
            source.append(name, targets)
            if source.num_measurements and rng.random() < 0.15:
                source.append("CX", [stim.target_rec(-1), rng.randrange(4)])
        _check_channel(source)


def test_unrecognized_interval_preserves_annotation_and_tick_positions() -> None:
    text = "H 0\nTICK\nCX 0 1\nQUBIT_COORDS(2, 3) 1\nTICK\nM 0 1"
    assert rewrite_to_mpp(text).circuit == stim.Circuit(text)
    _check_channel(text)


def test_z_extraction_reorders_mixed_data_readout() -> None:
    source = "R 0 1\nCX 0 1\nM 0 !1"
    result = rewrite_to_mpp(source)
    assert result.circuit == stim.Circuit("R 0\nMPP !Z0\nTICK\nM 0")
    assert result.foliation_record_to_source == (1, 0)
    assert result.eliminated_qubits == (1,)
    _check_channel(source)
