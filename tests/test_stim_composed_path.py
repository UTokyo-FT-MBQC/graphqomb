"""Round extraction -> ordinary graph importer -> annotated Stim export."""

from __future__ import annotations

import pytest
import stim

from graphqomb.qeccode import YFoliation
from graphqomb.stim_glue import rewrite_syndrome_rounds, stim_circuit_to_pattern, stim_compile
from graphqomb.stim_glue._parse import collect_record_annotations


def _annotation_moments(circuit: stim.Circuit) -> list[int]:
    """Exact joint Fourier moments for a small annotated stabilizer circuit."""
    prepared = stim.Circuit()
    prepared.append("R", range(circuit.num_qubits))
    prepared += circuit
    annotations = collect_record_annotations(circuit)
    groups = [*annotations.detectors, *annotations.logical_observables.values()]
    reference = prepared.reference_sample()
    moments = []
    for subset in range(1 << len(groups)):
        records: set[int] = set()
        for index, group in enumerate(groups):
            if (subset >> index) & 1:
                records.symmetric_difference_update(group)
        # Offline test oracle only; the rewriter never calls flow analysis.
        determined = prepared.has_flow(stim.Flow(measurements=sorted(records)), unsigned=True)
        moments.append((-1 if sum(reference[i] for i in records) % 2 else 1) if determined else 0)
    return moments


@pytest.mark.parametrize("foliation", list(YFoliation))
@pytest.mark.parametrize(
    ("text", "rounds"),
    [
        (
            (
                "R 0 2\nCX 0 2\nM 2\nR 2\nCX 0 2\nM 2\n"
                "DETECTOR[type=flag] rec[-1] rec[-2]\nM 0\nOBSERVABLE_INCLUDE(2) rec[-1]"
            ),
            2,
        ),
        ("RX 1\nCY 1 0\nMX !1\nDETECTOR rec[-1]\nMY 0\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-2]", 1),
        (
            ("RX 1\nCZ 1 0\nMX 0 !1\nDETECTOR rec[-1]\nCX rec[-1] 0\nMX 0\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-3]"),
            1,
        ),
        ("RX 1\nCZ 1 0\nS 1\nMX 1\nMX 0\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-2]", 0),
        ("R 1\nCX 0 1\nM 1\nH 0\nR 1\nCX 0 1\nM 1\nDETECTOR rec[-1] rec[-2]", 2),
        ("R 0 1\nCX 0 1\nMR 1\nCX 0 1\nMR 1\nDETECTOR rec[-1] rec[-2]", 0),
        ("MPP !Y0*Y1\nTICK\nMPP Y0*Y1\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-2]", 0),
    ],
)
def test_round_rewrite_preserves_exported_joint_annotations(text: str, rounds: int, foliation: YFoliation) -> None:
    source = stim.Circuit(text)
    result = rewrite_syndrome_rounds(source)
    assert result.rewritten_rounds == rounds
    assert result.circuit.num_measurements == source.num_measurements
    imported = stim_circuit_to_pattern(result.circuit, y_foliation=foliation)
    exported = stim.Circuit(stim_compile(imported.pattern))
    assert exported.num_detectors == source.num_detectors
    assert exported.num_observables == source.num_observables
    assert collect_record_annotations(exported).detector_tags == collect_record_annotations(source).detector_tags
    assert _annotation_moments(source) == _annotation_moments(exported)


def test_rewrite_preserves_data_coordinates_without_disconnected_probe_output() -> None:
    source = stim.Circuit("QUBIT_COORDS(0, 0) 0\nQUBIT_COORDS(1, 0) 1\nR 1\nCX 0 1\nM 1")
    result = rewrite_syndrome_rounds(source)
    assert result.eliminated_probes == 1
    imported = stim_circuit_to_pattern(result.circuit)
    graph = imported.pattern.clifford_frame.graphstate
    data_node = graph.input_node_indices[imported.stim_to_qubit[0]]
    assert graph.coordinates[data_node][:2] == (0.0, 0.0)
    assert set(imported.stim_to_qubit) == {0}
    assert all(graph.neighbors(node) for node in graph.nodes)
