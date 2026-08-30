"""End-to-end tests for classically-controlled Clifford feedforward."""

from __future__ import annotations

import cmath
import itertools
import math
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from graphqomb import clifford_algebra as ca
from graphqomb.command import TICK, E, M, N
from graphqomb.common import Plane, PlannerMeasBasis, is_close_angle, meas_basis
from graphqomb.feedforward import check_flow, pauli_simplification, propagate_correction_map, signal_shifting
from graphqomb.graphstate import GraphState
from graphqomb.pattern import Pattern
from graphqomb.pauli_frame import CliffordFrame
from graphqomb.qompiler import qompile
from graphqomb.scheduler import Scheduler
from graphqomb.simulator import PatternSimulator, SimulatorBackend
from graphqomb.statevec import StateVector

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

_HADAMARD = np.asarray([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
_T_GATE = np.diag([1.0, cmath.exp(1j * math.pi / 4)]).astype(np.complex128)
_Z_GATE = np.diag([1.0, -1.0]).astype(np.complex128)
_T_PLUS = (_T_GATE @ np.asarray([1.0, 1.0], dtype=np.complex128)) / math.sqrt(2)


class _ForcedOutcomeRng:
    """Rng stub forcing measurement outcomes when ``calc_prob=False``."""

    def __init__(self, outcomes: Iterable[bool]) -> None:
        self._outcomes = iter(outcomes)

    def uniform(self) -> float:
        return 0.0 if next(self._outcomes) else 0.9


def _forced_rng(*outcomes: bool) -> np.random.Generator:
    return cast("np.random.Generator", _ForcedOutcomeRng(outcomes))


def _random_state(rng: np.random.Generator) -> NDArray[np.complex128]:
    state = rng.normal(size=2) + 1j * rng.normal(size=2)
    return np.asarray(state / np.linalg.norm(state), dtype=np.complex128)


def _overlap(state1: NDArray[np.complex128], state2: NDArray[np.complex128]) -> float:
    return float(abs(np.vdot(state1, state2)) / (np.linalg.norm(state1) * np.linalg.norm(state2)))


def _t_gadget_graph() -> GraphState:
    graph = GraphState()
    n0 = graph.add_node()
    n1 = graph.add_node()
    n2 = graph.add_node()
    graph.register_input(n0, 0)
    graph.register_input(n1, 1)
    graph.register_output(n2, 0)
    graph.add_edge(n0, n1)
    graph.add_edge(n1, n2)
    graph.assign_meas_basis(n0, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(n1, PlannerMeasBasis(Plane.XY, 0.0))
    return graph


def _t_gadget_pattern() -> Pattern:
    graph = _t_gadget_graph()
    frame = CliffordFrame(graph, xflow={0: {1}, 1: {2}}, zflow={0: {2}}, cflow={0: {1: ca.S}})
    commands = (
        N(2),
        E((0, 1)),
        E((1, 2)),
        M(0, PlannerMeasBasis(Plane.XY, 0.0)),
        M(1, PlannerMeasBasis(Plane.XY, 0.0)),
        TICK(),
    )
    return Pattern(
        input_node_indices=graph.input_node_indices,
        output_node_indices=graph.output_node_indices,
        commands=commands,
        pauli_frame=frame,
        input_initializations=graph.input_initializations,
    )


def _simulate_t_gadget(
    pattern: Pattern,
    psi: NDArray[np.complex128],
    rng: np.random.Generator,
    *,
    calc_prob: bool,
) -> tuple[NDArray[np.complex128], PatternSimulator]:
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector, calc_prob=calc_prob)
    simulator.state = StateVector.from_product_states([psi, _T_PLUS])
    simulator.simulate(rng)
    return np.asarray(simulator.state.state()).ravel(), simulator


@pytest.mark.parametrize("m0", [False, True])
@pytest.mark.parametrize("m1", [False, True])
def test_t_gadget_forced_branches(m0: bool, m1: bool) -> None:
    """All four outcome branches yield H T H |psi> after the frame is undone."""
    rng = np.random.default_rng(7)
    for _ in range(3):
        psi = _random_state(rng)
        pattern = _t_gadget_pattern()
        output, simulator = _simulate_t_gadget(pattern, psi, _forced_rng(m0, m1), calc_prob=False)
        assert simulator.results == {0: m0, 1: m1}
        expected = _HADAMARD @ _T_GATE @ _HADAMARD @ psi
        assert _overlap(output, expected) == pytest.approx(1.0)
        # The recorded output frame on node 2 stays in the Pauli sector.
        frame = pattern.pauli_frame
        assert frame.coset[2] == ca.IDENTITY
        assert frame.x_pauli[2] == m1
        assert frame.z_pauli[2] == m0
        # In the m0=1 branch the frame on node 1 was S * X: coset S plus the X bit.
        if m0:
            assert frame.coset[1] == ca.S


@pytest.mark.parametrize("seed", range(6))
def test_t_gadget_sampled(seed: int) -> None:
    """Born-sampled branches also produce H T H |psi>."""
    rng = np.random.default_rng(seed)
    psi = _random_state(rng)
    output, _simulator = _simulate_t_gadget(_t_gadget_pattern(), psi, rng, calc_prob=True)
    expected = _HADAMARD @ _T_GATE @ _HADAMARD @ psi
    assert _overlap(output, expected) == pytest.approx(1.0)


def test_t_gadget_through_qompile() -> None:
    """Compiling with cflow through qompile realizes H T H |psi>."""
    rng = np.random.default_rng(11)
    for m0, m1 in itertools.product([False, True], repeat=2):
        psi = _random_state(rng)
        graph = _t_gadget_graph()
        pattern = qompile(graph, xflow={0: {1}, 1: {2}}, cflow={0: {1: ca.S}})
        output, simulator = _simulate_t_gadget(pattern, psi, _forced_rng(m0, m1), calc_prob=False)
        assert simulator.results == {0: m0, 1: m1}
        expected = _HADAMARD @ _T_GATE @ _HADAMARD @ psi
        assert _overlap(output, expected) == pytest.approx(1.0)


def test_qompile_folds_cflow_pauli_part() -> None:
    """A cflow value S*X folds its Pauli part into xflow, keeping the S coset."""
    graph = _t_gadget_graph()
    pattern = qompile(graph, xflow={0: {1}, 1: {2}}, cflow={0: {1: ca.compose(ca.S, ca.X)}})
    frame = pattern.pauli_frame
    assert frame.cflow == {0: {1: ca.S}}
    # The folded X cancels the existing xflow correction 0 -> 1.
    assert frame.xflow[0] == set()


def _plane_change_pattern(theta: float) -> Pattern:
    graph = GraphState()
    n0 = graph.add_node()
    n1 = graph.add_node()
    n2 = graph.add_node()
    graph.register_input(n0, 0)
    graph.register_output(n2, 0)
    graph.add_edge(n0, n1)
    graph.add_edge(n1, n2)
    graph.assign_meas_basis(n0, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(n1, PlannerMeasBasis(Plane.YZ, theta))
    frame = CliffordFrame(graph, xflow={0: {1}}, zflow={0: {2}}, cflow={0: {1: ca.S}})
    commands = (
        N(n1),
        N(n2),
        E((0, 1)),
        E((1, 2)),
        M(0, PlannerMeasBasis(Plane.XY, 0.0)),
        M(1, PlannerMeasBasis(Plane.YZ, theta)),
        TICK(),
    )
    return Pattern(
        input_node_indices=graph.input_node_indices,
        output_node_indices=graph.output_node_indices,
        commands=commands,
        pauli_frame=frame,
        input_initializations=graph.input_initializations,
    )


def test_s_coset_moves_yz_measurement_to_xz_plane() -> None:
    """An S coset on a YZ-measured node yields an effective XZ-plane basis."""
    theta = 0.7
    pattern = _plane_change_pattern(theta)
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector, calc_prob=False)

    pattern.pauli_frame.meas_flip(0)  # emulate outcome 1 on node 0
    basis = simulator._updated_measurement_basis(M(1, PlannerMeasBasis(Plane.YZ, theta)))
    assert basis.plane == Plane.XZ
    # Frame S * X on (YZ, theta): the X flip gives theta + pi, then S maps
    # YZ -> XZ with angle negation.
    assert is_close_angle(basis.angle, -(theta + math.pi))
    reference = ca.to_matrix(ca.compose(ca.S, ca.X)) @ meas_basis(Plane.YZ, theta)
    assert _overlap(basis.vector(), reference) == pytest.approx(1.0)


def test_pure_s_coset_yz_to_xz_angle() -> None:
    """A pure S coset (no Pauli bits) maps (YZ, theta) to (XZ, -theta)."""
    theta = 0.7
    graph = GraphState()
    n0 = graph.add_node()
    n1 = graph.add_node()
    graph.register_input(n0, 0)
    graph.register_output(n1, 0)
    graph.add_edge(n0, n1)
    graph.assign_meas_basis(n0, PlannerMeasBasis(Plane.XY, 0.0))
    frame = CliffordFrame(graph, xflow={}, zflow={}, cflow={0: {1: ca.S}})
    pattern = Pattern(
        input_node_indices=graph.input_node_indices,
        output_node_indices=graph.output_node_indices,
        commands=(N(n1), E((0, 1)), M(0, PlannerMeasBasis(Plane.XY, 0.0)), TICK()),
        pauli_frame=frame,
        input_initializations=graph.input_initializations,
    )
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector, calc_prob=False)
    frame.meas_flip(0)
    basis = simulator._updated_measurement_basis(M(1, PlannerMeasBasis(Plane.YZ, theta)))
    assert basis.plane == Plane.XZ
    assert is_close_angle(basis.angle, -theta)


def _cz_diagonal(num_qubits: int, qubit1: int, qubit2: int) -> NDArray[np.complex128]:
    diag = np.ones(1 << num_qubits, dtype=np.complex128)
    for index in range(1 << num_qubits):
        bits = format(index, f"0{num_qubits}b")
        if bits[qubit1] == "1" and bits[qubit2] == "1":
            diag[index] = -1
    return diag


def _project(state: NDArray[np.complex128], bra: NDArray[np.complex128]) -> NDArray[np.complex128]:
    projected = np.tensordot(bra.conj(), state, axes=(0, 0))
    return np.asarray(projected / np.linalg.norm(projected), dtype=np.complex128)


@pytest.mark.parametrize("m0", [False, True])
@pytest.mark.parametrize("m1", [False, True])
def test_plane_change_end_to_end(m0: bool, m1: bool) -> None:
    """The simulated output matches a direct matrix-algebra reference."""
    theta = 0.7
    pattern = _plane_change_pattern(theta)
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector, calc_prob=False)
    simulator.simulate(_forced_rng(m0, m1))
    assert simulator.results == {0: m0, 1: m1}
    output = np.asarray(simulator.state.state()).ravel()

    # Reference: CZ01 CZ12 |+++>, then project node 0 and node 1 onto the
    # effective bases dictated by the frame F = (S X)^{m0} on node 1, and undo
    # the recorded Z^{m0} frame on node 2.
    plus = np.ones(2, dtype=np.complex128) / math.sqrt(2)
    state = np.asarray(np.kron(np.kron(plus, plus), plus), dtype=np.complex128)
    state *= _cz_diagonal(3, 0, 1) * _cz_diagonal(3, 1, 2)
    state = state.reshape((2, 2, 2))

    basis0 = meas_basis(Plane.XY, math.pi if m0 else 0.0)
    state = _project(state, basis0)

    frame_matrix = ca.to_matrix(ca.compose(ca.S, ca.X)) if m0 else np.eye(2, dtype=np.complex128)
    basis1 = frame_matrix @ meas_basis(Plane.YZ, theta + (math.pi if m1 else 0.0))
    reference = _project(state, basis1)
    if m0:
        reference = _Z_GATE @ reference

    assert _overlap(output, reference) == pytest.approx(1.0)


def test_scheduler_dag_includes_cflow_edges() -> None:
    graph = _t_gadget_graph()
    scheduler = Scheduler(graph, xflow={0: {1}}, zflow={0: {2}}, cflow={0: {1: ca.S}, 1: {2: ca.H}})
    assert 1 in scheduler.dag[0]
    assert 2 in scheduler.dag[1]


def test_check_flow_detects_cflow_cycle() -> None:
    graph = _t_gadget_graph()
    with pytest.raises(ValueError, match="Cycle detected"):
        check_flow(graph, xflow={}, zflow={}, cflow={0: {1: ca.S}, 1: {0: ca.S}})


def test_feedforward_rewrites_reject_cflow() -> None:
    graph = _t_gadget_graph()
    xflow = {0: {1}, 1: {2}}
    cflow = {0: {1: ca.S}}
    with pytest.raises(NotImplementedError, match="not supported yet"):
        signal_shifting(graph, xflow, cflow=cflow)
    with pytest.raises(NotImplementedError, match="not supported yet"):
        pauli_simplification(graph, xflow, cflow=cflow)
    with pytest.raises(NotImplementedError, match="not supported yet"):
        propagate_correction_map(0, graph, xflow, cflow=cflow)
