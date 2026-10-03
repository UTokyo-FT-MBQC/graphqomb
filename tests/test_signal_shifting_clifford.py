"""Signal shifting preserves Clifford boundaries and downstream computations."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from graphqomb import clifford_algebra as ca
from graphqomb.common import Plane, PlannerMeasBasis
from graphqomb.feedforward import propagate_correction_map, signal_shifting
from graphqomb.graphstate import GraphState
from graphqomb.qompiler import qompile
from graphqomb.simulator import PatternSimulator, SimulatorBackend
from graphqomb.statevec import StateVector

if TYPE_CHECKING:
    from collections.abc import Iterable

    from graphqomb.common import MeasBasis


class _BornBranchRng:
    """Select either branch of the balanced teleportation measurements."""

    def __init__(self, outcomes: Iterable[bool]) -> None:
        self._outcomes = iter(outcomes)

    def uniform(self) -> float:
        return 0.9 if next(self._outcomes) else 0.0


def _measured_chain(num_nodes: int) -> GraphState:
    graph = GraphState()
    for _ in range(num_nodes):
        graph.add_node()
    graph.register_output(num_nodes - 1, 0)
    for node in range(num_nodes - 1):
        graph.add_edge(node, node + 1)
        graph.assign_meas_basis(node, PlannerMeasBasis(Plane.XY, 0.0))
    return graph


@pytest.mark.parametrize("shifted", [False, True])
@pytest.mark.parametrize("outcomes", list(itertools.product([False, True], repeat=4)))
def test_t_teleportation_all_branches(
    shifted: bool, outcomes: tuple[bool, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete graph realization of T accepts every branch, before and after shifting."""
    graph = _measured_chain(5)
    graph.register_input(0, 0)
    graph.register_input(2, 1)
    xflow = {node: {node + 1} for node in range(4)}
    zflow = {0: {2}, 1: {3}, 2: {4}}
    cflow = {1: {2: ca.S}}
    if shifted:
        xflow, zflow = signal_shifting(graph, xflow, zflow, cflow=cflow)
        # The source and target of S keep their incoming dependencies. The
        # final Pauli-only teleportation shifts its outcome relabeling away.
        assert zflow[0] == {2}
        assert zflow[1] == set()
        assert xflow[1] == {2, 4}

    pattern = qompile(graph, xflow, zflow, cflow=cflow)
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector, calc_prob=True)
    t_gate = np.diag([1.0, np.exp(1j * math.pi / 4)])
    psi = np.asarray([0.3 + 0.1j, 0.7 - 0.2j], dtype=np.complex128)
    psi /= np.linalg.norm(psi)
    magic_state = t_gate @ (np.ones(2, dtype=np.complex128) / math.sqrt(2))
    simulator.state = StateVector.from_product_states([psi, magic_state])

    # Record Born probabilities before each actual sampled measurement. This
    # checks that the forced branches are possible and no postselection hides
    # a failure; all four teleportations have probability 1/2 per outcome.
    branch_probabilities: list[float] = []
    original_sample_measure = StateVector.sample_measure

    def record_sample_measure(state: StateVector, qubit: int, basis: MeasBasis, rng: np.random.Generator) -> bool:
        vector = state.state()
        projected = np.tensordot(basis.vector().conj(), vector, axes=(0, qubit))
        prob_zero = float(np.linalg.norm(projected) ** 2 / np.linalg.norm(vector) ** 2)
        outcome = original_sample_measure(state, qubit, basis, rng)
        branch_probabilities.append(1.0 - prob_zero if outcome else prob_zero)
        return outcome

    monkeypatch.setattr(StateVector, "sample_measure", record_sample_measure)
    simulator.simulate(cast("np.random.Generator", _BornBranchRng(outcomes)))
    assert simulator.results == dict(enumerate(outcomes))
    assert branch_probabilities == pytest.approx([0.5] * 4)
    assert math.prod(branch_probabilities) == pytest.approx(1 / 16)
    output = np.asarray(simulator.state.state()).ravel()
    assert abs(np.vdot(t_gate @ psi, output)) == pytest.approx(1.0)


def test_shift_preserves_control_bit_and_adaptive_target() -> None:
    """A Clifford control must keep its original, already corrected record."""
    graph = _measured_chain(5)
    xflow = {0: {1}, 1: {2}, 2: {3}, 3: {4}}
    zflow = {0: {1}, 1: {2}, 2: {3}}
    cflow = {1: {2: ca.S}}

    shifted_x, shifted_z = signal_shifting(graph, xflow, zflow, cflow=cflow)
    assert shifted_z[0] == {1}
    assert shifted_z[1] == {2}
    assert shifted_z[2] == set()
    assert shifted_x[2] == {3, 4}
    assert cflow == {1: {2: ca.S}}
    assert xflow == {0: {1}, 1: {2}, 2: {3}, 3: {4}}
    assert zflow == {0: {1}, 1: {2}, 2: {3}}


def test_shift_preserves_remote_clifford_event_order() -> None:
    """A copied Pauli correction may not move across an earlier Clifford event."""
    graph = _measured_chain(5)
    xflow = {0: {1}, 1: {2}, 2: {3}, 3: {4}}
    zflow = {0: {2}, 1: {3}}
    cflow = {1: {4: ca.S}}
    # In particular, node 3 sends X to node 4 after the S from node 1.
    # Shifting the Z from node 1 through node 3 would move a second X to
    # source 1, where the fixed X-then-S order changes the corrected state.
    assert signal_shifting(graph, xflow, zflow, cflow=cflow) == (xflow, zflow)
    for node in (0, 1, 2, 3):
        new_x, new_z = propagate_correction_map(node, graph, xflow, zflow, cflow=cflow)
        assert (new_x, new_z) == (xflow, zflow)
        assert new_x is not xflow
        assert new_z is not zflow
        assert new_x[0] is not xflow[0]
        assert new_z[0] is not zflow[0]


def test_shift_unrelated_region_with_non_pauli_measurement() -> None:
    """Clifford dependencies elsewhere do not disable valid plane-based shifts."""
    graph = _measured_chain(5)
    graph.assign_meas_basis(3, PlannerMeasBasis(Plane.XY, 0.37))
    xflow = {0: {1}, 2: {3}, 3: {4}}
    zflow = {2: {3}}
    cflow = {0: {1: ca.H}}
    shifted_x, shifted_z = signal_shifting(graph, xflow, zflow, cflow=cflow)
    assert shifted_x == {0: {1}, 2: {3, 4}, 3: {4}}
    assert shifted_z == {2: set()}


def test_shift_sparse_maps_ignores_self_correction_bookkeeping() -> None:
    """Propagated maps add missing sources without copying ignored self-edges."""
    graph = _measured_chain(5)
    xflow = {3: {3, 4}}
    zflow = {2: {3}, 3: {3}}
    cflow = {0: {1: ca.S}}
    shifted_x, shifted_z = signal_shifting(graph, xflow, zflow, cflow=cflow)
    assert shifted_x == {2: {4}, 3: {3, 4}}
    assert shifted_z == {2: set(), 3: {3}}


@pytest.mark.parametrize("gate", [ca.X, ca.Y, ca.Z, ca.S, ca.H])
def test_supplied_cflow_gates_all_define_boundaries(gate: ca.C1Element) -> None:
    """Preserving the supplied cflow requires preserving its record controls."""
    graph = _measured_chain(4)
    xflow = {0: {1}, 1: {2}, 2: {3}}
    zflow = {0: {1}}
    cflow = {1: {2: gate}}
    assert signal_shifting(graph, xflow, zflow, cflow=cflow) == (xflow, zflow)


@pytest.mark.parametrize("cflow", [{0: {2: ca.IDENTITY}}, {1: {1: ca.S}}, {0: {}}])
def test_inactive_cflow_entries_do_not_block_shifting(cflow: dict[int, dict[int, ca.C1Element]]) -> None:
    graph = _measured_chain(3)
    xflow = {0: {1}, 1: {2}}
    zflow = {0: {1}}
    expected = signal_shifting(graph, xflow, zflow)
    assert signal_shifting(graph, xflow, zflow, cflow=cflow) == expected


def test_identity_cflow_does_not_create_a_dependency_cycle() -> None:
    graph = _measured_chain(3)
    xflow = {0: {1}, 1: {2}}
    zflow = {0: {1}}
    cflow = {1: {0: ca.IDENTITY}}
    assert signal_shifting(graph, xflow, zflow, cflow=cflow) == signal_shifting(graph, xflow, zflow)


def test_cancelled_pauli_cflow_edge_does_not_create_a_dependency_cycle() -> None:
    """The executable DAG normalizes the Pauli correction before detecting cycles."""
    graph = _measured_chain(3)
    xflow = {0: {1}, 1: {0}}
    zflow: dict[int, set[int]] = {}
    cflow = {0: {1: ca.X}}
    pattern = qompile(graph, xflow, zflow, cflow=cflow)
    assert pattern.clifford_frame.cflow == {}
    assert pattern.clifford_frame.xflow == {0: set(), 1: {0}}
    assert signal_shifting(graph, xflow, zflow, cflow=cflow) == (xflow, zflow)


def test_cancelled_pauli_cflow_still_protects_its_record_control() -> None:
    """Returned maps retain cflow verbatim, so paired cancellations cannot be split."""
    graph = _measured_chain(4)
    xflow = {1: {2}, 2: {3}}
    zflow = {0: {1}}
    cflow = {1: {2: ca.X}}
    pattern = qompile(graph, xflow, zflow, cflow=cflow)
    assert pattern.clifford_frame.cflow == {}
    assert pattern.clifford_frame.xflow == {1: set(), 2: {3}}
    assert signal_shifting(graph, xflow, zflow, cflow=cflow) == (xflow, zflow)
    assert propagate_correction_map(1, graph, xflow, zflow, cflow=cflow) == (xflow, zflow)


def test_true_clifford_dependency_cycle_is_rejected() -> None:
    graph = _measured_chain(3)
    xflow = {0: {1}, 1: {2}}
    cflow = {1: {0: ca.S}}
    with pytest.raises(ValueError, match="cycle"):
        signal_shifting(graph, xflow, zflow={}, cflow=cflow)
