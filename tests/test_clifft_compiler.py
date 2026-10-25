"""Tests for the Clifft compiler."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from examples.ftqc._support.steane import build_steane_pattern
from examples.ftqc._support.steane_noise import steane_encoder
from graphqomb import clifford_algebra as ca
from graphqomb.command import M
from graphqomb.common import Axis, AxisMeasBasis, Initialization, Sign
from graphqomb.graphstate import GraphState
from graphqomb.noise_model import DepolarizingNoiseModel
from graphqomb.qompiler import qompile
from graphqomb.scheduler import Scheduler
from graphqomb.stim_glue.clifft_compiler import _clifford, _controlled_clifford, clifft_compile
from graphqomb.stim_glue.compiler import stim_compile

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from graphqomb.clifford_algebra import C1Element
    from graphqomb.pattern import Pattern

_ALL_CLIFFORDS = tuple(
    ca.C1Element(x_axis, x_sign, z_axis, z_sign)
    for x_axis, z_axis in itertools.permutations((Axis.X, Axis.Y, Axis.Z), 2)
    for x_sign, z_sign in itertools.product((1, -1), repeat=2)
)
_SQRT_X = np.asarray([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex128) / 2
_ONE_QUBIT: dict[str, NDArray[np.complex128]] = {
    "T": np.diag([1.0, np.exp(1j * math.pi / 4)]).astype(np.complex128),
    "T_DAG": np.diag([1.0, np.exp(-1j * math.pi / 4)]).astype(np.complex128),
    "H": np.asarray([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2),
    "S": np.diag([1, 1j]).astype(np.complex128),
    "X": np.asarray([[0, 1], [1, 0]], dtype=np.complex128),
    "Z": np.diag([1, -1]).astype(np.complex128),
    "SQRT_X": _SQRT_X,
    "SQRT_X_DAG": _SQRT_X.conj().T,
}
_TWO_QUBIT: dict[str, NDArray[np.complex128]] = {
    "CX": np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128),
    "CZ": np.diag([1, 1, 1, -1]).astype(np.complex128),
}


def _gadget_unitary(ops: tuple[str, ...]) -> NDArray[np.complex128]:
    """Multiply gadget instructions on qubits 0 (control) and 1 (target), big endian.

    Returns
    -------
    numpy.ndarray
        The 4x4 unitary in application order.
    """
    unitary = np.eye(4, dtype=np.complex128)
    for op in ops:
        name, *targets = op.split()
        for index in range(0, len(targets), 2 if name in _TWO_QUBIT else 1):
            if name in _TWO_QUBIT:
                assert targets[index : index + 2] == ["0", "1"]
                gate = _TWO_QUBIT[name]
            else:
                factors = [np.eye(2, dtype=np.complex128)] * 2
                factors[int(targets[index])] = _ONE_QUBIT[name]
                gate = np.asarray(np.kron(*factors), dtype=np.complex128)
            unitary = gate @ unitary
    return unitary


def _is_multiple_of(matrix: NDArray[np.complex128], reference: NDArray[np.complex128]) -> bool:
    return bool(np.isclose(abs(np.trace(reference.conj().T @ matrix)), 2.0))


@pytest.mark.parametrize("element", _ALL_CLIFFORDS)
def test_controlled_clifford_gadget_applies_element_only_on_control_one(element: C1Element) -> None:
    unitary = _gadget_unitary(_controlled_clifford(element, 0, 1))
    np.testing.assert_allclose(unitary[:2, 2:], 0.0, atol=1e-12)
    np.testing.assert_allclose(unitary[2:, :2], 0.0, atol=1e-12)
    # Each control branch may carry its own phase: the control is a basis state.
    assert _is_multiple_of(unitary[:2, :2], np.eye(2, dtype=np.complex128))
    assert _is_multiple_of(unitary[2:, 2:], ca.to_matrix(element))


def _t_teleportation(
    readout: Axis | None = Axis.X,
) -> tuple[Pattern, int, int, int]:
    """Build ZZ-probe T teleportation with the S^p correction as cflow.

    Returns
    -------
    tuple
        The pattern and the data, magic, and probe nodes.
    """
    graph = GraphState()
    data, magic, probe = graph.add_node(), graph.add_node(), graph.add_node()
    graph.register_input(data, 0)
    graph.register_input(magic, 1)
    graph.register_output(data, 0)
    for node in (data, magic):
        graph.add_edge(probe, node)
    for node in (probe, magic):
        graph.assign_meas_basis(node, AxisMeasBasis(Axis.X, Sign.PLUS))
    measure_time = {probe: 1, magic: 2}
    observables = None
    if readout is not None:
        graph.assign_meas_basis(data, AxisMeasBasis(readout, Sign.PLUS))
        measure_time[data] = 3
        observables = {0: {data}}
    xflow: dict[int, set[int]] = {}
    zflow = {magic: {data}}
    cflow = {probe: {data: ca.S}}
    scheduler = Scheduler(graph, xflow, zflow, cflow=cflow)
    scheduler.manual_schedule(prepare_time={probe: 0}, measure_time=measure_time)
    pattern = qompile(graph, xflow, zflow, cflow, logical_observables=observables, scheduler=scheduler)
    return pattern, data, magic, probe


@pytest.mark.parametrize("element", _ALL_CLIFFORDS)
def test_fixed_clifford_applies_element(element: C1Element) -> None:
    unitary = _gadget_unitary(_clifford(element, 1))
    assert _is_multiple_of(unitary[:2, :2], ca.to_matrix(element))
    np.testing.assert_allclose(unitary[:2, 2:], 0.0, atol=1e-12)


def test_input_gates_follow_the_reset_before_prepare_noise() -> None:
    pattern, data, magic, _ = _t_teleportation()
    text = clifft_compile(
        pattern, input_gates={magic: "T", data: ("H", "R_Z(0.25)")}, noise_models=[DepolarizingNoiseModel(0.01)]
    )
    assert f"RX {magic}\nT {magic}\nDEPOLARIZE1(0.01) {magic}\n" in text
    assert f"RX {data}\nH {data}\nR_Z(0.25) {data}\nDEPOLARIZE1(0.01) {data}\n" in text


def test_conditional_clifford_uses_a_fresh_ancilla_before_the_target_measurement() -> None:
    pattern, data, _, probe = _t_teleportation()
    text = clifft_compile(pattern, input_circuit="H 7\nCX 0 7")
    lines = text.splitlines()
    ancilla = 8
    load = lines.index(f"CX rec[-2] {ancilla}")
    assert lines[load - 1] == f"R {ancilla}"
    gadget = tuple(lines[load + 1 : load + 5])
    assert gadget == _controlled_clifford(ca.S, ancilla, data)
    # The probe is measured before the magic qubit, so S precedes the Z correction.
    assert lines[load + 5 : load + 7] == ["CZ rec[-1] 0", f"MX {data}"]
    assert lines.index("H 7") < lines.index(f"MX {probe}")


def test_unmeasured_output_receives_conditional_clifford_at_the_end() -> None:
    pattern, data, _, _ = _t_teleportation(readout=None)
    lines = clifft_compile(pattern).splitlines()
    assert lines[-7:] == ["R 3", "CX rec[-2] 3", *_controlled_clifford(ca.S, 3, data), f"CZ rec[-1] {data}"]


@pytest.mark.parametrize("value", [False, True])
def test_fixed_control_replaces_every_feedback_of_its_source(value: bool) -> None:
    pattern, data, _, probe = _t_teleportation()
    fixed = clifft_compile(pattern, fixed_controls={probe: value}).splitlines()
    measure = fixed.index(f"MX {data}")
    # The probe's S is fixed; the magic qubit's Z feedback keeps its record.
    expected = [f"S {data}"] if value else []
    assert fixed[measure - len(expected) - 1 : measure] == [*expected, "CZ rec[-1] 0"]
    assert not any(line.startswith(("R 3", "T ")) for line in fixed)
    unfixed = clifft_compile(pattern).splitlines()
    assert fixed[: unfixed.index("R 3")] == unfixed[: unfixed.index("R 3")]


def test_fixed_controls_reject_nodes_without_clifford_feedforward() -> None:
    pattern, _, magic, _ = _t_teleportation()
    with pytest.raises(ValueError, match="not a source of Clifford feedforward"):
        clifft_compile(pattern, fixed_controls={magic: True})


def test_clifft_text_omits_instruction_tags() -> None:
    graph = GraphState()
    node = graph.add_node()
    graph.register_input(node, 0, init=Initialization(Axis.Z, tag="prep"))
    graph.assign_meas_basis(node, AxisMeasBasis(Axis.Z, Sign.PLUS))
    pattern = qompile(graph, {}, {}, parity_check_group=[{node}], parity_check_tags=["type=flag"])
    assert "R[prep] 0" in stim_compile(pattern, backpropagate=False)
    assert clifft_compile(pattern).splitlines() == ["R 0", "MZ 0", "TICK", "DETECTOR rec[-1]"]


def test_stim_compile_still_rejects_clifford_feedforward() -> None:
    pattern, *_ = _t_teleportation()
    with pytest.raises(ValueError, match="cflow"):
        stim_compile(pattern, backpropagate=False)


@pytest.mark.parametrize(
    ("input_gates", "match"),
    [
        ({5: "T"}, "not an input node"),
        ({0: "M"}, "Unsupported input gate"),
        ({0: "CX"}, "Unsupported input gate"),
        ({0: "T T"}, "Unsupported input gate"),
        ({0: "R_Z"}, "Unsupported input gate"),
        ({0: "T(0.1)"}, "Unsupported input gate"),
    ],
)
def test_invalid_input_gates_are_rejected(input_gates: dict[int, str], match: str) -> None:
    pattern, *_ = _t_teleportation()
    with pytest.raises(ValueError, match=match):
        clifft_compile(pattern, input_gates=input_gates)


@pytest.mark.parametrize(
    "line", ["M 0", "MPP X0*X1", "DETECTOR(0) rec[-1]", "CX rec[-1] 0", "REPEAT 2 {", "HERALDED_ERASE(0.1) 0"]
)
def test_input_circuit_rejects_measurement_records(line: str) -> None:
    pattern, *_ = _t_teleportation()
    with pytest.raises(ValueError, match="measurement records"):
        clifft_compile(pattern, input_circuit=f"H 0\n{line}")


def _measurement_records(pattern: Pattern) -> list[int]:
    return [command.node for command in pattern if isinstance(command, M)]


@pytest.mark.parametrize("expected_bit", [False, True])
def test_clifft_t_teleportation_is_deterministic_in_all_branches(expected_bit: bool) -> None:
    clifft = pytest.importorskip("clifft")
    pattern, data, magic, probe = _t_teleportation()
    # T maps T^dagger |+/-> to |+/->, so the corrected X readout is deterministic.
    data_gates = ("Z", "T_DAG") if expected_bit else ("T_DAG",)
    result = clifft.sample(
        clifft.compile(clifft_compile(pattern, input_gates={data: data_gates, magic: "T"})), shots=400, seed=1
    )
    measurements = np.asarray(result.measurements, dtype=bool)
    records = _measurement_records(pattern)
    branches = {tuple(row) for row in measurements[:, [records.index(probe), records.index(magic)]]}
    assert branches == set(itertools.product((False, True), repeat=2))
    assert np.all(np.asarray(result.observables, dtype=bool)[:, 0] == expected_bit)


@pytest.mark.parametrize("value", [False, True])
def test_clifft_fixed_control_is_correct_when_the_record_agrees(value: bool) -> None:
    clifft = pytest.importorskip("clifft")
    pattern, data, magic, probe = _t_teleportation()
    text = clifft_compile(pattern, input_gates={data: "T_DAG", magic: "T"}, fixed_controls={probe: value})
    result = clifft.sample(clifft.compile(text), shots=400, seed=5)
    agrees = np.asarray(result.measurements, dtype=bool)[:, _measurement_records(pattern).index(probe)] == value
    observables = np.asarray(result.observables, dtype=bool)[:, 0]
    assert 0 < agrees.sum() < len(agrees)
    assert not observables[agrees].any()
    # Disagreeing shots apply the wrong S, so the X readout is random.
    assert observables[~agrees].any()


def test_clifft_noise_flips_the_t_teleportation_readout() -> None:
    clifft = pytest.importorskip("clifft")
    pattern, data, magic, _ = _t_teleportation()
    text = clifft_compile(pattern, input_gates={data: "T_DAG", magic: "T"}, noise_models=[DepolarizingNoiseModel(0.05)])
    result = clifft.sample(clifft.compile(text), shots=2000, seed=2)
    assert 0.0 < np.asarray(result.observables, dtype=bool)[:, 0].mean() < 0.5


@pytest.mark.parametrize("element", _ALL_CLIFFORDS)
@pytest.mark.parametrize("input_axis", [Axis.X, Axis.Y, Axis.Z])
def test_clifft_conditional_clifford_matches_its_axis_action(element: C1Element, input_axis: Axis) -> None:
    clifft = pytest.importorskip("clifft")
    graph = GraphState()
    source, target = graph.add_node(), graph.add_node()
    graph.register_input(source, 0)
    graph.register_input(target, 1, init=Initialization(input_axis))
    # D maps the +1 eigenstate of P to the sign eigenstate of D P D^dagger.
    axis, sign = ca.act_on_axis(element, input_axis)
    # Z readout of the |+> source gives both correction branches.
    graph.assign_meas_basis(source, AxisMeasBasis(Axis.Z, Sign.PLUS))
    graph.assign_meas_basis(target, AxisMeasBasis(axis, Sign.PLUS))
    cflow = {source: {target: element}}
    scheduler = Scheduler(graph, {}, {}, cflow=cflow)
    scheduler.manual_schedule(prepare_time={}, measure_time={source: 0, target: 1})
    pattern = qompile(graph, {}, {}, cflow, scheduler=scheduler)
    result = clifft.sample(clifft.compile(clifft_compile(pattern)), shots=200, seed=3)
    measurements = np.asarray(result.measurements, dtype=bool)
    corrected = measurements[measurements[:, 0]]
    assert len(corrected) > 0
    assert np.all(corrected[:, 1] == (sign == -1))


@pytest.mark.parametrize("readout", [Axis.X, Axis.Z])
@pytest.mark.parametrize("expected_bit", [False, True])
def test_clifft_encoded_steane_t_teleportation_has_silent_detectors(readout: Axis, expected_bit: bool) -> None:
    clifft = pytest.importorskip("clifft")
    steane = build_steane_pattern(readout)
    data_gates: tuple[str, ...]
    if readout == Axis.X:
        data_gates = ("H", "Z", "T_DAG") if expected_bit else ("H", "T_DAG")
    else:
        data_gates = ("X",) if expected_bit else ()
    input_circuit = "\n".join(
        (
            steane_encoder(steane.data_input_nodes, data_gates),
            steane_encoder(steane.magic_input_nodes, ("H", "T")),
        )
    )
    text = clifft_compile(steane.pattern, input_circuit=input_circuit)
    result = clifft.sample(clifft.compile(text), shots=200, seed=4)
    measurements = np.asarray(result.measurements, dtype=bool)
    control = _measurement_records(steane.pattern).index(steane.control_node)
    assert set(measurements[:, control]) == {False, True}
    assert not np.asarray(result.detectors, dtype=bool).any()
    assert np.all(np.asarray(result.observables, dtype=bool)[:, 0] == expected_bit)
