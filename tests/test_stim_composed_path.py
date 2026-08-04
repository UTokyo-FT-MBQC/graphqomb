"""Contract tests for the composed ``rewrite_to_mpp`` -> ``stim_circuit_to_pattern`` path.

The rewriter's output circuit is a valid importer input, whichever internal
path (optimized rewrite or gate-level fallback) produced it. Each case checks
the full chain: rewrite, import, compile back to Stim, and require that the
detector/observable counts match the source, that the noiseless detector error
model is empty (every detector deterministic), and that no graph node loses
its coordinate relative to a fully coordinated source.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
import stim

from graphqomb.stim_glue import rewrite_to_mpp, stim_circuit_to_pattern, stim_compile

if TYPE_CHECKING:
    from graphqomb.stim_glue import StimImportResult

_FALLBACK_CIRCUIT = """
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(1, 0) 1
R 0 1
H 0
SWAP 0 1
S 0
M 0
R 0
M 0
DETECTOR rec[-1]
DETECTOR rec[-2]
OBSERVABLE_INCLUDE(0) rec[-1]
"""

_SIGNED_Y_MPP_CIRCUIT = """
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(1, 0) 1
QUBIT_COORDS(2, 0) 2
R 0 1 2
TICK
MPP !Y0*Y1 Y1*Y2
TICK
MPP !Y0*Y1 Y1*Y2
DETECTOR rec[-1] rec[-3]
DETECTOR rec[-2] rec[-4]
"""

_MID_CIRCUIT_RESET_CIRCUIT = """
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(1, 0) 1
R 0 1
TICK
CX 0 1
M 1
R 1
TICK
CX 0 1
M 1
DETECTOR rec[-1] rec[-2]
"""


def _uncoordinated_node_count(result: StimImportResult) -> int:
    graph = result.pattern.pauli_frame.graphstate
    return graph.number_of_nodes() - len(graph.coordinates)


def _assert_same_reference_signs(left: stim.Circuit, right: stim.Circuit) -> None:
    left_detectors, left_observables = left.reference_detector_and_observable_signs()
    right_detectors, right_observables = right.reference_detector_and_observable_signs()
    assert np.array_equal(left_detectors, right_detectors)
    assert np.array_equal(left_observables, right_observables)


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3),
            id="surface-code-d3",
        ),
        pytest.param(
            stim.Circuit.generated("repetition_code:memory", distance=3, rounds=4),
            id="repetition-code-d3",
        ),
        pytest.param(stim.Circuit(_FALLBACK_CIRCUIT), id="gate-level-fallback"),
        pytest.param(stim.Circuit(_SIGNED_Y_MPP_CIRCUIT), id="signed-y-mpp"),
        pytest.param(stim.Circuit(_MID_CIRCUIT_RESET_CIRCUIT), id="mid-circuit-reset"),
    ],
)
@pytest.mark.parametrize("fallback", ["circuit", "segment"])
def test_rewrite_output_imports_like_the_source(source: stim.Circuit, fallback: Literal["circuit", "segment"]) -> None:
    rewritten = rewrite_to_mpp(source, fallback=fallback).circuit

    composed = stim_circuit_to_pattern(rewritten)
    direct = stim_circuit_to_pattern(source)
    compiled = stim.Circuit(stim_compile(composed.pattern))

    assert compiled.num_detectors == source.num_detectors
    assert compiled.num_observables == source.num_observables
    assert compiled.detector_error_model(decompose_errors=False).num_errors == 0
    _assert_same_reference_signs(compiled, source)
    # When the source carries QUBIT_COORDS for every qubit (the repetition
    # code generator emits none), neither path may produce a coordinate-less
    # node.
    if len(source.get_final_qubit_coordinates()) == source.num_qubits:
        assert _uncoordinated_node_count(composed) == 0
        assert _uncoordinated_node_count(direct) == 0


def test_segment_fallback_feedback_on_deterministic_one_record_imports() -> None:
    # The rewritten producer of a deterministic-1 record must stay a real
    # measurement (a signed MPP here, never MPAD 1, which the importer
    # rejects), so the verbatim feedback segment consuming it still fires.
    source = stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        R 0 1
        X 0
        M 0
        CX rec[-1] 1
        M 1
        DETECTOR rec[-1]
        """
    )

    rewritten = rewrite_to_mpp(source, fallback="segment")

    assert rewritten.fallback_segments == (1,)
    composed = stim_circuit_to_pattern(rewritten.circuit)
    compiled = stim.Circuit(stim_compile(composed.pattern))

    assert compiled.num_detectors == source.num_detectors
    assert compiled.detector_error_model(decompose_errors=False).num_errors == 0
    _assert_same_reference_signs(compiled, source)
    assert _uncoordinated_node_count(composed) == 0


def test_externally_pre_split_circuit_imports_like_the_original() -> None:
    # A caller may hand the importer a circuit whose reset lifetimes already
    # sit on fresh qubit ids (id 2 continues id 1 below). The importer cannot
    # know the wires' shared origin, but the imported pattern must still agree
    # with the original circuit's detector/observable counts and stay
    # deterministic.
    original = stim.Circuit(_MID_CIRCUIT_RESET_CIRCUIT)
    pre_split = stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        R 0 1
        TICK
        CX 0 1
        M 1
        R 2
        TICK
        CX 0 2
        M 2
        DETECTOR rec[-1] rec[-2]
        """
    )

    original_result = stim_circuit_to_pattern(original)
    # Wire 2 has no QUBIT_COORDS and no recoverable origin, which the importer
    # reports as partial coordinate coverage.
    with pytest.warns(UserWarning, match="have no QUBIT_COORDS"):
        pre_split_result = stim_circuit_to_pattern(pre_split)

    for result in (original_result, pre_split_result):
        compiled = stim.Circuit(stim_compile(result.pattern))

        assert compiled.num_detectors == original.num_detectors
        assert compiled.num_observables == original.num_observables
        assert compiled.detector_error_model(decompose_errors=False).num_errors == 0
