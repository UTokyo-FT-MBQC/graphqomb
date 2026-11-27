"""Branch Choi comparisons test quantum action, not only record statistics."""

from __future__ import annotations

from itertools import permutations
from typing import TYPE_CHECKING

import numpy as np
import pytest
import stim

from graphqomb.stim_glue import rewrite_syndrome_rounds, stim_circuit_to_pattern, stim_compile
from graphqomb.stim_glue._t_gates import parse_t_circuit

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _embed(matrix: NDArray[np.complex128], targets: list[int], width: int) -> NDArray[np.complex128]:
    result = np.zeros((2**width, 2**width), dtype=np.complex128)
    mask = sum(1 << q for q in targets)
    for column in range(2**width):
        local_column = sum(((column >> q) & 1) << k for k, q in enumerate(targets))
        for local_row in range(2 ** len(targets)):
            row = (column & ~mask) | sum(((local_row >> k) & 1) << q for k, q in enumerate(targets))
            result[row, column] = matrix[local_row, local_column]
    return result


def _instruments(  # ruff: ignore[complex-structure, too-many-branches, too-many-statements, too-many-locals]
    circuit: stim.Circuit,
    width: int,
    data: list[int],
) -> dict[tuple[int, ...], NDArray[np.complex128]]:
    """Independent dense Kraus oracle with ancilla trace, arbitrary data input."""
    initial = np.zeros((2**width, 2 ** len(data)), dtype=np.complex128)
    for column in range(2 ** len(data)):
        initial[sum(((column >> k) & 1) << q for k, q in enumerate(data)), column] = 1
    branches: dict[tuple[int, ...], list[NDArray[np.complex128]]] = {(): [initial]}
    identity = np.eye(2**width, dtype=np.complex128)
    for inst in circuit.flattened():  # ruff: ignore[too-many-nested-blocks]
        assert isinstance(inst, stim.CircuitInstruction)
        name = inst.name
        targets = inst.targets_copy()
        if name in {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS"}:
            continue
        if name in {"R", "RX", "RY", "M", "MX", "MY", "MPP"}:
            observables = []
            if name == "MPP":
                for group in inst.target_groups():
                    p = stim.PauliString(width)
                    for t in group:
                        if t.is_combiner:
                            continue
                        p[t.value] = "X" if t.is_x_target else "Y" if t.is_y_target else "Z"
                        if t.is_inverted_result_target:
                            p *= -1
                    observables.append(p)
            else:
                for t in targets:
                    p = stim.PauliString(width)
                    p[t.value] = name[-1] if len(name) > 1 else "Z"
                    if t.is_inverted_result_target:
                        p *= -1
                    observables.append(p)
            for index, p in enumerate(observables):
                matrix = np.asarray(p.to_unitary_matrix(endian="little"), dtype=np.complex128)
                following: dict[tuple[int, ...], list[NDArray[np.complex128]]] = {}
                for record, kraus in branches.items():
                    for bit in (0, 1):
                        operation = (identity + (-1) ** bit * matrix) / 2
                        if name.startswith("R"):
                            if bit:
                                flip = stim.PauliString(width)
                                flip[targets[index].value] = "Z" if name == "RX" else "X"
                                operation = flip.to_unitary_matrix(endian="little") @ operation
                            key = record
                        else:
                            key = (*record, bit)
                        following.setdefault(key, []).extend(operation @ k for k in kraus)
                branches = following
        else:
            arity = 1 if stim.gate_data(name).is_single_qubit_gate else 2
            for start in range(0, len(targets), arity):
                group = targets[start : start + arity]
                feedback = group[0].is_measurement_record_target
                unitary = stim.Tableau.from_named_gate(name[-1] if feedback else name).to_unitary_matrix(
                    endian="little"
                )
                matrix = _embed(
                    np.asarray(unitary, dtype=np.complex128),
                    [t.value for t in (group[1:] if feedback else group)],
                    width,
                )
                branches = {
                    record: [matrix @ k if not feedback or record[group[0].value] else k for k in kraus]
                    for record, kraus in branches.items()
                }
    ancillas = sorted(set(range(width)) - set(data))
    result = {}
    for record, kraus in branches.items():
        choi = np.zeros((4 ** len(data), 4 ** len(data)), dtype=np.complex128)
        for a in range(2 ** len(ancillas)):
            rows = [
                sum(((a >> k) & 1) << q for k, q in enumerate(ancillas))
                | sum(((d >> k) & 1) << q for k, q in enumerate(data))
                for d in range(2 ** len(data))
            ]
            for k in kraus:
                vector = k[rows].reshape(-1)
                choi += np.outer(vector, vector.conj())
        result[record] = choi
    return result


def _assert_instrument(text: str, data: list[int]) -> None:
    source = stim.Circuit(text)
    result = rewrite_syndrome_rounds(source)
    expected = _instruments(source, source.num_qubits, data)
    actual = _instruments(result.circuit, source.num_qubits, data)
    actual = {tuple(record[i] for i in result.record_map): choi for record, choi in actual.items()}
    assert expected.keys() == actual.keys()
    for record in expected:
        np.testing.assert_allclose(actual[record], expected[record], atol=2e-6, rtol=2e-6, err_msg=str(record))


@pytest.mark.parametrize(
    "text",
    [
        "R 1\nCX 0 1\nM 1",
        "RX 1\nCZ 1 0\nH 0\nMX 1",
        "RY 1\nCX 1 0\nMY !1",
        "R 1\nH 1\nCZ 1 0\nH 1\nX 1\nM 1",
        "RX 1\nCX 1 0\nCZ 1 0\nCX 1 0\nMX 1",  # negative product
        "RX 1\nCZ 1 0\nS 1\nMX 1",  # residual S dagger: must retain
        "R 1\nCX 0 1\nM 1\nH 0\nR 1\nCX 0 1\nM 1",  # noncommuting rounds
        "RX 1\nCZ 1 0\nMX 1\nCZ rec[-1] 0\nH 0",  # retained Pauli feedback
        "RX 1\nCZ 1 0\nMX !0 1\nCX rec[-1] 0\nMX 0",  # record permutation
    ],
)
def test_arbitrary_input_quantum_instrument(text: str) -> None:
    _assert_instrument(text, [0])


@pytest.mark.parametrize("interactions", list(permutations(["CX 2 0", "CX 2 1", "CZ 3 0", "CZ 3 1"])))
def test_interleaved_probe_phases(interactions: tuple[str, ...]) -> None:
    text = "RX 2 3\n" + "\n".join(interactions) + "\nMX 2 3"
    _assert_instrument(text, [0, 1])


@pytest.mark.parametrize(
    "text",
    [
        "RX 2\nCY 2 0\nCX 0 1\nS 1\nMX 2",
        "RX 2\nX 2\nCZ 2 0\nCZ 2 1\nX 2\nMX 2",
        "R 0\nRY 2\nCX 2 0\nH 0\nCZ 0 1\nMY 2",
    ],
)
def test_data_cliffords_and_preparation_survive_factorization(text: str) -> None:
    assert rewrite_syndrome_rounds(stim.Circuit(text)).rewritten_rounds == 1
    _assert_instrument(text, [0, 1])


def test_inter_probe_phase_cannot_be_dropped_even_when_products_commute() -> None:
    ordinary = stim.Circuit("RX 2 3\nCX 2 0\nCX 2 1\nCZ 3 0\nCZ 3 1\nMX 2 3")
    crossed = stim.Circuit("RX 2 3\nCX 2 0\nCZ 3 0\nCZ 3 1\nCX 2 1\nMX 2 3")
    assert rewrite_syndrome_rounds(ordinary).rewritten_rounds == 1
    result = rewrite_syndrome_rounds(crossed)
    assert result.circuit == crossed
    assert result.retained_reasons == ("residual phase between probes",)


def test_ambiguous_fresh_readout_pair_is_retained() -> None:
    source = stim.Circuit("R 0 1\nCX 0 1\nM 0 1")
    result = rewrite_syndrome_rounds(source)
    assert result.circuit == source
    assert result.retained_reasons == ("interaction between probe candidates",)


def test_repeat_is_expanded_without_merging_measurement_layers() -> None:
    result = rewrite_syndrome_rounds(stim.Circuit("REPEAT 3 {\nR 1\nCX 0 1\nM 1\n}"))
    assert result.rewritten_rounds == 3
    assert sum(inst.name == "MPP" for inst in result.circuit) == 3
    assert result.record_map == (0, 1, 2)


def test_rounds_survive_known_initial_state_and_graph_import() -> None:
    result = rewrite_syndrome_rounds(stim.Circuit("R 0 1\nCX 0 1\nM 1\nR 1\nCX 0 1\nM 1\nDETECTOR rec[-1] rec[-2]"))
    assert result.rewritten_rounds == 2
    assert result.eliminated_probes == 2
    assert sum(inst.name == "MPP" for inst in result.circuit) == 2
    assert str(result.circuit).startswith("R 0\n")
    graph = stim_circuit_to_pattern(result.circuit)
    assert len(graph.pattern.frame.graphstate.nodes) > 2
    exported = stim.Circuit(stim_compile(graph.pattern))
    assert exported.num_detectors == 1
    assert not np.asarray(exported.compile_detector_sampler(seed=7).sample(16)).any()


def test_mixed_readout_annotations_and_feedback_remap() -> None:
    source = stim.Circuit("RX 1\nCZ 1 0\nMX 0 1\nDETECTOR[tag] rec[-1]\nOBSERVABLE_INCLUDE(2) rec[-2]\nCX rec[-1] 0")
    result = rewrite_syndrome_rounds(source)
    assert result.record_map == (1, 0)
    assert "DETECTOR[tag] rec[-2]" in str(result.circuit)
    assert "OBSERVABLE_INCLUDE(2) rec[-1]" in str(result.circuit)
    assert "CX rec[-2] 0" in str(result.circuit)
    imported = stim_circuit_to_pattern(result.circuit)
    assert imported.pattern.frame.xflow


@pytest.mark.parametrize("tail", ["H 1", "M 1", "CX 1 0", "OBSERVABLE_INCLUDE(0) Z1"])
def test_unclosed_probe_is_retained(tail: str) -> None:
    source = stim.Circuit("R 1\nCX 0 1\nM 1\n" + tail)
    result = rewrite_syndrome_rounds(source)
    assert result.circuit == source
    assert result.eliminated_probes == 0


def test_native_mpp_layers_are_untouched() -> None:
    source = stim.Circuit("RX 0 1\nMPP X0*X1\nTICK\nMPP Z0*Z1\nDETECTOR rec[-1] rec[-2]")
    assert rewrite_syndrome_rounds(source).circuit == source


def test_eliminated_probe_coordinates_are_removed_across_all_lifetimes() -> None:
    source = stim.Circuit(
        "QUBIT_COORDS[shared](2, 3) 0 1 2\n"
        "REPEAT 2 {\nR 1\nCX 0 1\nM 1\n}\n"
        "MPAD 1\nDETECTOR rec[-2] rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
        "QUBIT_COORDS[probe](9, 9) 1"
    )
    result = rewrite_syndrome_rounds(source)
    assert result.eliminated_probes == 2
    assert result.record_map == (0, 1, 2)
    # Keep both data and the originally coordinate-only wire, including the tag.
    assert result.circuit.get_final_qubit_coordinates() == {0: [2, 3], 2: [2, 3]}
    assert result.circuit[:1] == stim.Circuit("QUBIT_COORDS[shared](2, 3) 0 2")
    assert result.circuit[-3:] == stim.Circuit("MPAD 1\nDETECTOR rec[-2] rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-1]")


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("H 1\nM 1\n", ""),
        ("", "R 1"),
        ("", "R 1\nMPP X1"),
        ("", "R 1\nCX rec[-1] 1"),
        ("", "RX 1\nCZ 1 0\nS 1\nMX 1"),
        ("OBSERVABLE_INCLUDE(0) Z1\nM 2\n", ""),
    ],
)
def test_probe_coordinates_survive_references_elsewhere(before: str, after: str) -> None:
    source = stim.Circuit("QUBIT_COORDS(1, 2) 1\n" + before + "R 1\nCX 0 1\nM 1\n" + after)
    result = rewrite_syndrome_rounds(source)
    assert result.eliminated_probes == 1
    assert result.circuit.get_final_qubit_coordinates() == {1: [1, 2]}


@pytest.mark.parametrize("noise", ["X_ERROR(0.01) 0", "M(0.01) 0", "HERALDED_ERASE(0) 0"])
def test_noise_requires_explicit_caller_decision(noise: str) -> None:
    with pytest.raises(ValueError, match="ideal input"):
        rewrite_syndrome_rounds(stim.Circuit(noise))


def test_data_gate_moves_probe_support_before_phase_check_accepts_commuting_layer() -> None:
    """Support moved onto another data qubit by a data Clifford is seen by the later probe's phase check."""
    source = stim.Circuit("RX 2 3\nCX 2 0\nCZ 0 1\nCZ 1 3\nMX 2 3")
    result = rewrite_syndrome_rounds(source)

    assert result.rewritten_rounds == 1
    assert result.eliminated_probes == 2
    mpps = [str(inst) for inst in result.circuit if inst.name == "MPP"]
    assert mpps == ["MPP X0*Z1 Z1"]


def test_data_gate_moves_probe_support_before_phase_check_rejects_residual_phase() -> None:
    """A later probe anticommuting with the moved support leaves the window unchanged with its reason."""
    source = stim.Circuit("RX 2 3\nCX 2 0\nCZ 0 1\nCX 3 1\nMX 2 3")
    result = rewrite_syndrome_rounds(source)

    assert result.rewritten_rounds == 0
    assert result.retained_reasons == ("residual phase between probes",)
    assert result.circuit == source


def _assert_interacting_instrument(text: str, data: list[int]) -> None:
    source = stim.Circuit(text)
    result = rewrite_syndrome_rounds(source, interacting_probes=True)
    assert result.rewritten_rounds == 1
    expected = _instruments(source, source.num_qubits, data)
    actual = _instruments(result.circuit, source.num_qubits, data)
    actual = {tuple(record[i] for i in result.record_map): choi for record, choi in actual.items()}
    assert expected.keys() == actual.keys()
    for record in expected:
        np.testing.assert_allclose(actual[record], expected[record], atol=2e-6, rtol=2e-6, err_msg=str(record))


# A Bell pair of probes measures an X and a Z product of the same data qubits.
_SUPERDENSE_PAIR = "RX 2\nR 3\nCX 2 3\nCX 0 2 1 3\nCX 2 0 3 1\nCX 2 3\nM 3\nMX 2"
_SUPERDENSE_PLAQUETTE = "RX 4\nR 5\nCX 4 5\nCX 0 4 2 5\nCX 1 4 3 5\nCX 4 0 5 2\nCX 4 1 5 3\nCX 4 5\nM 5\nMX 4"


def test_interacting_probes_are_retained_by_default() -> None:
    source = stim.Circuit(_SUPERDENSE_PAIR)
    result = rewrite_syndrome_rounds(source)
    assert result.circuit == source
    assert result.rewritten_rounds == 0


@pytest.mark.parametrize(
    ("text", "data"),
    [
        (_SUPERDENSE_PAIR, [0, 1]),
        (_SUPERDENSE_PLAQUETTE, [0, 1, 2, 3]),
        ("RX 2\nR 3\nCX 2 3\nCX 2 0 3 1\nCX 0 2 1 3\nCX 2 3\nM 3\nMX 2", [0, 1]),  # X products collected first
        ("H 0\nCX 0 2\n" + _SUPERDENSE_PAIR, [0, 1]),  # probe reused after an earlier lifetime
        ("RX 2\nR 3\nH 0 1\nCX 2 3\nCX 0 2 1 3\nCX 2 0 3 1\nS 1\nCX 2 3\nM !3\nMX !2", [0, 1]),  # data Cliffords, signs
        ("RX 2\nR 3\nCX 2 3\nCX 0 2 1 3\nCX 2 0 3 1\nCZ 0 1\nCX 2 3\nM 3\nMX 2\nM 0", [0, 1]),  # retained readout
    ],
)
def test_interacting_probes_preserve_arbitrary_input_instrument(text: str, data: list[int]) -> None:
    _assert_interacting_instrument(text, data)


def test_superdense_pair_becomes_products_with_pauli_feedback() -> None:
    result = rewrite_syndrome_rounds(stim.Circuit(_SUPERDENSE_PAIR), interacting_probes=True)
    assert result.eliminated_probes == 2
    assert result.retained_reasons == ()
    assert [str(inst) for inst in result.circuit if inst.name == "MPP"] == ["MPP Z0*Z1 X0*X1"]
    # One probe per data qubit: the collected Z parity kicks back an X on one of them.
    assert [str(inst) for inst in result.circuit if inst.name == "CX"] == ["CX rec[-2] 1"]
    # Two data qubits per probe: the same record still kicks back an X on one probe's pair.
    plaquette = rewrite_syndrome_rounds(stim.Circuit(_SUPERDENSE_PLAQUETTE), interacting_probes=True)
    assert [str(inst) for inst in plaquette.circuit if inst.name != "TICK"] == [
        "MPP Z0*Z1*Z2*Z3 X0*X1*X2*X3",
        "CX rec[-2] 2 rec[-2] 3",
    ]


def test_superdense_hexagon_flows_match_source() -> None:
    source = stim.Circuit(
        "RX 6\nR 7\nCX 6 7\nCX 0 6 3 7\nCX 1 6 4 7\nCX 2 6 5 7\nCX 6 0 7 3\nCX 6 1 7 4\nCX 6 2 7 5\nCX 6 7\nM 7\nMX 6"
    )
    result = rewrite_syndrome_rounds(source, interacting_probes=True)
    assert result.rewritten_rounds == 1
    assert result.record_map == (0, 1)
    flows = [
        stim.Flow("ZZZZZZ -> rec[0]"),
        stim.Flow("XXXXXX -> rec[1]"),
        stim.Flow("1 -> XXXXXX xor rec[1]"),
        # The Z record is removed from the output by the X kickback on three qubits.
        stim.Flow("1 -> ZZZZZZ"),
        stim.Flow("Z____Z -> _ZZZZ_ xor rec[0]"),
    ]
    assert source.has_all_flows(flows)
    assert result.circuit.has_all_flows(flows)


def test_interacting_readout_group_remaps_annotations_and_imports() -> None:
    second = _SUPERDENSE_PAIR.replace("M 3\nMX 2", "M 3\nDETECTOR[z] rec[-1]\nMX 2\nDETECTOR[x] rec[-1] rec[-3]")
    source = stim.Circuit(_SUPERDENSE_PAIR + "\n" + second)
    result = rewrite_syndrome_rounds(source, interacting_probes=True)
    assert result.rewritten_rounds == 2
    assert result.record_map == (0, 1, 2, 3)
    # Both annotations follow the two products of their readout group.
    assert "DETECTOR[z] rec[-2]" in str(result.circuit)
    assert "DETECTOR[x] rec[-1] rec[-3]" in str(result.circuit)
    assert not np.asarray(result.circuit.compile_detector_sampler(seed=3).sample(32)).any()
    exported = stim.Circuit(stim_compile(stim_circuit_to_pattern(result.circuit).pattern))
    assert exported.num_detectors == source.num_detectors
    assert exported.detector_error_model().num_errors == 0
    assert not np.asarray(exported.compile_detector_sampler(seed=3).sample(32)).any()


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("R 0 1\nCX 0 1\nM 0 1", "probe has trivial or non-Hermitian data product"),
        # Without the decoding CX each probe reads a random bit of the Bell pair.
        (
            "RX 2\nR 3\nCX 2 3\nCX 0 2 1 3\nCX 2 0 3 1\nM 3\nMX 2",
            "probe record is not a Pauli product of the window input",
        ),
        # Data reset in the same window: its products are not functions of the window input.
        ("RX 0 1\n" + _SUPERDENSE_PAIR, "probe record is not a Pauli product of the window input"),
    ],
)
def test_interacting_probes_outside_the_accepted_form_are_retained(text: str, reason: str) -> None:
    source = stim.Circuit(text)
    result = rewrite_syndrome_rounds(source, interacting_probes=True)
    assert result.circuit == source
    assert result.retained_reasons == (reason,)


@pytest.mark.parametrize("interacting_probes", [False, True])
@pytest.mark.parametrize("text", ["RX 1\nCX 1 0\nT 0\nMX 1", "R 1\nCX 0 1\nT_DAG 0\nM 1"])
def test_t_transport_gate_is_a_barrier(text: str, interacting_probes: bool) -> None:
    source = parse_t_circuit(text)
    result = rewrite_syndrome_rounds(source, interacting_probes=interacting_probes)
    assert result.circuit == source
    assert result.rewritten_rounds == 0
