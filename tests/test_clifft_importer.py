"""Behavioral checks for the two Clifford+T import paths."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from graphqomb.command import M, N
from graphqomb.common import Axis, AxisMeasBasis, Plane, PlannerMeasBasis, Sign, determine_pauli_axis
from graphqomb.graphstate import GraphState
from graphqomb.noise_model import DepolarizingNoiseModel
from graphqomb.qompiler import qompile
from graphqomb.stim_glue import clifft_compile, clifft_file_to_pattern, clifft_text_to_pattern, stim_compile

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal


@pytest.mark.parametrize("mode", ["measurement", "injection"])
@pytest.mark.parametrize(
    ("gates", "readout", "expected"),
    [("T 0\nT 0", "MY", False), ("T_DAG 0\nT_DAG 0", "MY", True), ("T 0\nT_DAG 0", "MX", False)],
)
def test_t_signs_and_cancellation_in_every_sampled_branch(
    mode: Literal["measurement", "injection"], gates: str, readout: str, expected: bool
) -> None:
    clifft = pytest.importorskip("clifft")
    result = clifft_text_to_pattern(f"RX 0\n{gates}\n{readout} 0\nOBSERVABLE_INCLUDE(0) rec[-1]", t_mode=mode)
    program = clifft.compile(clifft_compile(result.pattern, preparation_gates=result.preparation_gates))
    sample = clifft.sample(program, shots=512, seed=9)
    assert np.all(np.asarray(sample.observables, dtype=bool)[:, 0] == expected)
    measured = [c.node for c in result.pattern if isinstance(c, M)]
    records = np.asarray(sample.measurements, dtype=bool)
    for magic in result.preparation_gates:
        probe = next(iter(result.pattern.frame.graphstate.neighbors(magic)))
        branches = records[:, [measured.index(probe), measured.index(magic)]]
        assert {tuple(row) for row in branches} == {(False, False), (False, True), (True, False), (True, True)}


@pytest.mark.parametrize("mode", ["measurement", "injection"])
@pytest.mark.parametrize("merge_safe_ticks", [False, True])
def test_joint_records_survive_entanglement_feedback_repeats_and_resets(
    mode: Literal["measurement", "injection"], merge_safe_ticks: bool
) -> None:
    clifft = pytest.importorskip("clifft")
    text = """RX 0
R 1
T 0
CX 0 1
H 0
T_DAG 1
MX 0
CZ rec[-1] 1
TICK
REPEAT 2 {
    T 1
    H 1
}
MR 1
RX 1
T_DAG 1
MX 1
OBSERVABLE_INCLUDE(0) rec[-3] rec[-2] rec[-1]
"""
    imported = clifft_text_to_pattern(text, t_mode=mode, merge_safe_ticks=merge_safe_ticks)
    reference = clifft.sample(clifft.compile(text), shots=8192, seed=1)
    output = clifft_compile(imported.pattern, preparation_gates=imported.preparation_gates)
    sample = clifft.sample(clifft.compile(output), shots=8192, seed=2)
    measured = [c.node for c in imported.pattern if isinstance(c, M)]
    columns = [measured.index(imported.record_to_node[index]) for index in range(3)]
    records = np.asarray(sample.measurements, dtype=bool)[:, columns]
    expected = np.asarray(reference.measurements, dtype=bool)
    weights = 2 ** np.arange(3)
    actual_hist = np.bincount(records @ weights, minlength=8) / len(records)
    expected_hist = np.bincount(expected @ weights, minlength=8) / len(expected)
    # Independent fixed-seed samples: 0.04 is over five worst-case standard errors.
    np.testing.assert_allclose(actual_hist, expected_hist, atol=0.04, rtol=0)
    np.testing.assert_array_equal(np.asarray(sample.observables)[:, 0], np.logical_xor.reduce(records, axis=1))


def test_injection_has_pauli_measurements_and_scheduled_magic_resources() -> None:
    imported = clifft_text_to_pattern("RX 0\nT 0\nH 0\nT_DAG 0\nMX 0", t_mode="injection")
    pattern = imported.pattern
    assert len(imported.preparation_gates) == 2
    assert set(imported.preparation_gates.values()) == {"T", "T_DAG"}
    assert all(determine_pauli_axis(c.meas_basis) is not None for c in pattern if isinstance(c, M))
    assert not imported.preparation_gates.keys() & pattern.input_node_indices.keys()
    prepared = {c.node for c in pattern if isinstance(c, N)}
    assert imported.preparation_gates.keys() <= prepared
    measured = [c.node for c in pattern if isinstance(c, M)]
    for magic in imported.preparation_gates:
        probe = next(iter(pattern.frame.graphstate.neighbors(magic)))
        assert measured.index(probe) < measured.index(magic)


def test_measurement_mode_preserves_t_basis_and_stim_rejects_it() -> None:
    imported = clifft_text_to_pattern("RX 0\nT 0\nT_DAG 0\nMX 0")
    angles = [
        c.meas_basis.angle for c in imported.pattern if isinstance(c, M) and determine_pauli_axis(c.meas_basis) is None
    ]
    assert angles == [-math.pi / 4, math.pi / 4]
    assert not imported.preparation_gates
    with pytest.raises(ValueError, match="Unsupported measurement basis"):
        stim_compile(imported.pattern)


@pytest.mark.parametrize("turns", [1, 3, 5, 7, -1])
def test_clifford_observable_readout_matches_its_analytic_probability(turns: int) -> None:
    clifft = pytest.importorskip("clifft")
    graph = GraphState()
    node = graph.add_node()
    graph.register_input(node, 0)
    graph.assign_meas_basis(node, PlannerMeasBasis(Plane.XY, turns * math.pi / 4))
    pattern = qompile(graph, {}, logical_observables={0: {node}})
    sample = clifft.sample(clifft.compile(clifft_compile(pattern)), shots=4096, seed=3)
    probability = np.asarray(sample.observables, dtype=bool)[:, 0].mean()
    assert abs(probability - (1 - math.cos(turns * math.pi / 4)) / 2) < 0.04


def test_scheduled_magic_rotation_precedes_preparation_noise() -> None:
    imported = clifft_text_to_pattern("RX 0\nT 0\nMX 0", t_mode="injection")
    magic = next(iter(imported.preparation_gates))
    text = clifft_compile(
        imported.pattern, preparation_gates=imported.preparation_gates, noise_models=[DepolarizingNoiseModel(0.01)]
    )
    assert f"RX {magic}\nT {magic}\nDEPOLARIZE1(0.01) {magic}\n" in text


def test_file_import_and_private_tag_rejection(tmp_path: Path) -> None:
    path = tmp_path / "circuit.stim"
    path.write_text("RX 0\nT 0\nMX 0", encoding="utf-8")
    assert len(clifft_file_to_pattern(path).record_to_node) == 1
    with pytest.raises(ValueError, match="transport tags"):
        clifft_text_to_pattern("I[graphqomb.internal.T] 0")


def test_internal_preparation_gates_reject_input_nodes() -> None:
    graph = GraphState()
    node = graph.add_node()
    graph.register_input(node, 0)
    graph.assign_meas_basis(node, AxisMeasBasis(Axis.X, Sign.PLUS))
    with pytest.raises(ValueError, match="not an internal node"):
        clifft_compile(qompile(graph, {}), preparation_gates={node: "T"})
