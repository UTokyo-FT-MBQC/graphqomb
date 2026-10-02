"""Constructive check-gadget contraction and degree-four Foliation regressions."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import stim

from graphqomb.stim_glue import rewrite_to_mpp, stim_circuit_to_pattern
from graphqomb.stim_glue._gadget_contract import contract_disposable_gadgets

if TYPE_CHECKING:
    import pytest


def _assert_classical_channel(source: stim.Circuit) -> None:
    result = rewrite_to_mpp(source)
    assert result.circuit.flow_generators() == source.flow_generators()
    left, right = source.copy(), result.foliation_circuit.copy()
    left.append("R", range(source.num_qubits))
    right.append("R", range(source.num_qubits))
    permutation = result.foliation_record_to_source
    assert sorted(permutation) == list(range(source.num_measurements))
    inverse = {old: new for new, old in enumerate(permutation)}
    # A permutation changes the canonical choice of generators. Test both
    # spans by membership, including signs, instead of comparing their lists.
    for a, b, mapping in ((left, right, inverse), (right, left, dict(enumerate(permutation)))):
        for flow in a.flow_generators():
            output = flow.output_copy()
            # Stim 1.16's canonical generators can omit the sign of MPAD 1.
            # Establish the sign using the circuit before mapping the flow.
            if not a.has_flow(flow):
                output = -output
                assert a.has_flow(
                    stim.Flow(input=flow.input_copy(), output=output, measurements=flow.measurements_copy())
                )
            mapped = stim.Flow(
                input=flow.input_copy(),
                output=output,
                measurements=[mapping[m] for m in flow.measurements_copy()],
            )
            assert b.has_flow(mapped), (source, a, b, flow, mapped)


def _two_round_checks(*, mixed_readout: bool) -> stim.Circuit:
    circuit = stim.Circuit("RX 0 1 2 3")
    for round_index in range(2):
        circuit += stim.Circuit("""
            RX 4 5
            TICK
            CX 4 0 4 1 4 2 4 3
            CZ 5 0 5 1 5 2 5 3
            TICK
        """)
        circuit.append("MX", range(6) if mixed_readout and round_index == 1 else [4, 5])
        circuit.append("TICK", [])
    return circuit


def test_separate_measure_and_reset_contract_to_degree_four() -> None:
    source = _two_round_checks(mixed_readout=False)
    source += stim.Circuit("DETECTOR rec[-1] rec[-3]\nDETECTOR rec[-2] rec[-4]")
    result = rewrite_to_mpp(source)
    graph = stim_circuit_to_pattern(result.foliation_circuit).pattern.clifford_frame.graphstate
    assert result.eliminated_qubits == (4, 5)
    assert max(len(graph.neighbors(n)) for n in graph.nodes) <= 4
    assert not any(stim.gate_data(i.name).is_unitary for i in result.foliation_circuit)
    _assert_classical_channel(source)


def test_mixed_readout_permutation_preserves_annotations_and_degree() -> None:
    source = _two_round_checks(mixed_readout=True)
    source += stim.Circuit("""
        DETECTOR[check_z] rec[-1] rec[-7]
        DETECTOR[check_x] rec[-2] rec[-8]
        OBSERVABLE_INCLUDE(0) rec[-6] rec[-5] rec[-4] rec[-3]
    """)
    result = rewrite_to_mpp(source)
    assert result.foliation_record_to_source == (0, 1, 6, 7, 2, 3, 4, 5)
    assert max(c.product.weight for c in result.foliation_checks) == 4
    pattern = stim_circuit_to_pattern(result.foliation_circuit).pattern
    graph = pattern.clifford_frame.graphstate
    assert max(len(graph.neighbors(n)) for n in graph.nodes) <= 4
    assert all(pattern.clifford_frame.detector_determinism())
    dets, obs = result.foliation_circuit.compile_detector_sampler(seed=0).sample(32, separate_observables=True)
    assert not dets.any()
    assert not obs.any()
    assert [i.tag for i in result.foliation_circuit if i.name == "DETECTOR"] == ["check_z", "check_x"]
    _assert_classical_channel(source)


def test_permutation_remaps_feedback_and_preserves_inversion_and_padding() -> None:
    source = _two_round_checks(mixed_readout=False)
    source += stim.Circuit("""
        MPAD 1 0
        RX 4 5
        CX 4 0 4 1 4 2 4 3
        CZ 5 0 5 1 5 2 5 3
        MX[readout] !0 1 2 3 4 !5
        CX rec[-6] 6
        M 6
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-7]
    """)
    result = rewrite_to_mpp(source)
    assert result.foliation_record_to_source != tuple(range(source.num_measurements))
    assert any(i.tag == "readout" for i in result.foliation_circuit)
    _assert_classical_channel(source)
    for left, right in zip(
        source.reference_detector_and_observable_signs(),
        result.foliation_circuit.reference_detector_and_observable_signs(),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)


def test_retained_ancilla_and_unrelated_data_clifford_are_not_discarded() -> None:
    for text in (
        "RX 2\nCZ 2 0\nMX 2\nCX 2 1\nM 1",
        "RX 2\nCZ 2 0\nH 1\nMX 2\nM 1",
        "RX 2\nCZ 2 0\nS 2\nMX 2",
        "MPAD 1\nRX 2\nCZ rec[-1] 2\nCZ 2 0\nMX 2",
    ):
        source = stim.Circuit(text)
        assert 2 not in rewrite_to_mpp(source).eliminated_qubits
        _assert_classical_channel(source)


def test_random_circuits_preserve_classical_channel_after_contraction() -> None:
    rng = random.Random(17)  # ruff:ignore[suspicious-non-cryptographic-random-usage]
    for _ in range(100):
        source = stim.Circuit()
        for _ in range(12):
            name = rng.choice(["R", "RX", "H", "S", "CX", "CZ", "M", "MX", "MRX"])
            targets = rng.sample(range(4), 2 if name in {"CX", "CZ"} else 1)
            source.append(name, targets)
            source.append("TICK", [])
        _assert_classical_channel(source)


def test_disposable_contraction_does_not_call_flow_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> object:
        message = "the disposable-gadget rule must be constructive"
        raise AssertionError(message)

    monkeypatch.setattr(stim.Circuit, "flow_generators", fail)
    monkeypatch.setattr(stim.Circuit, "has_flow", fail)
    result = rewrite_to_mpp(_two_round_checks(mixed_readout=True))
    assert result.eliminated_qubits == (4, 5)


def test_interleaved_controlled_paulis_account_for_ancilla_phases() -> None:
    # The checks commute as complete Pauli products, but these interleaved
    # gates leave CZ between the controls. Checking support commutation alone
    # would incorrectly drop that phase.
    source = stim.Circuit("RX 2 3\nCX 2 0\nCZ 3 0 3 1\nCX 2 1\nMX 2 3")
    assert not contract_disposable_gadgets(source).discarded_qubits
    _assert_classical_channel(source)
    rng = random.Random(29)  # ruff:ignore[suspicious-non-cryptographic-random-usage]
    contracted = 0
    for _ in range(100):
        gates = [("CX", [2, 0]), ("CX", [2, 1]), ("CZ", [3, 0]), ("CZ", [3, 1])]
        rng.shuffle(gates)
        source = stim.Circuit("RX 2 3")
        for name, targets in gates:
            source.append(name, targets)
        source.append("MX", rng.sample([2, 3], 2))
        contracted += bool(contract_disposable_gadgets(source).discarded_qubits)
        _assert_classical_channel(source)
    assert 0 < contracted < 100
