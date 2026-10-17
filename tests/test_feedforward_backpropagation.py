"""Offline parity compilation preserves retained feedback and quantum outputs."""

from __future__ import annotations

import itertools
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest
import stim

from graphqomb import clifford_algebra as ca
from graphqomb.common import Axis, AxisMeasBasis, Initialization, Plane, PlannerMeasBasis, Sign
from graphqomb.feedforward import Feedforward
from graphqomb.graphstate import GraphState
from graphqomb.pauli_frame import CliffordFrame, PauliFrame
from graphqomb.ptn_format import dumps, loads
from graphqomb.qompiler import qompile
from graphqomb.scheduler import Scheduler
from graphqomb.simulator import PatternSimulator, SimulatorBackend
from graphqomb.statevec import StateVector
from graphqomb.stim_glue.compiler import stim_compile

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from graphqomb.common import MeasBasis
    from graphqomb.pattern import Pattern


def _graph(*axes: Axis | None) -> GraphState:
    graph = GraphState()
    for index, axis in enumerate(axes):
        node = graph.add_node()
        graph.register_input(node, index)
        if axis is None:
            graph.register_output(node, 0)
        else:
            graph.assign_meas_basis(node, AxisMeasBasis(axis, Sign.PLUS))
    return graph


@pytest.mark.parametrize(
    ("axis", "x", "z", "expected"),
    [
        (Axis.X, True, False, {1}),
        (Axis.X, False, True, {0, 1}),
        (Axis.Y, True, False, {0, 1}),
        (Axis.Y, False, True, {0, 1}),
        (Axis.Y, True, True, {1}),
        (Axis.Z, True, False, {0, 1}),
        (Axis.Z, False, True, {1}),
    ],
)
def test_pauli_relabelings_move_into_both_parity_types(axis: Axis, x: bool, z: bool, expected: set[int]) -> None:
    graph = _graph(Axis.Z, axis)
    original = Feedforward(
        {0: {1}} if x else {},
        {0: {1}} if z else {},
        parity_check_group=[{1}],
        logical_observables={7: {1}},
        parity_check_tags=["type=flag"],
    )

    compiled = original.backpropagate(graph)

    assert compiled.xflow == {}
    assert compiled.zflow == {}
    assert compiled.cflow == {}
    assert compiled.parity_check_group == [expected]
    assert compiled.logical_observables == {7: expected}
    assert compiled.parity_check_tags == ["type=flag"]
    pattern = qompile(graph, compiled)
    assert pattern.frame.parity_check_group == [expected]
    assert pattern.frame.logical_observables == {7: expected}


def test_diamond_paths_cancel_without_recursion_or_shared_mutation() -> None:
    graph = _graph(*(Axis.Z for _ in range(4)))
    xflow = {0: {1, 2}, 1: {3}, 2: {3}}
    seeds = [{3}, {0, 3}]
    original = Feedforward(xflow, {}, parity_check_group=seeds, logical_observables={7: {1, 2, 3}, 3: {1}})

    compiled = original.backpropagate(graph)

    assert compiled.parity_check_group == [{1, 2, 3}, {0, 1, 2, 3}]
    assert compiled.logical_observables == {7: {3}, 3: {0, 1}}
    assert xflow == original.xflow == {0: {1, 2}, 1: {3}, 2: {3}}
    assert seeds == original.parity_check_group == [{3}, {0, 3}]
    compiled.parity_check_group[0].clear()
    assert original.parity_check_group == seeds


def test_clifford_boundary_keeps_control_ancestors_and_avoids_double_counting() -> None:
    graph = _graph(Axis.Z, Axis.Z, Axis.X, Axis.Y, Axis.Z)
    original = Feedforward(
        {0: {1}, 1: {4}, 2: {3}, 3: {4}},
        {0: {2}, 1: {3}},
        {1: {2: ca.S}},
        parity_check_group=[{1}, {2}, {3}, {4}],
        logical_observables={0: {1, 4}},
    )

    compiled = original.backpropagate(graph)

    assert compiled.xflow == {0: {1}}
    assert compiled.zflow == {0: {2}}
    assert compiled.cflow == {1: {2: ca.S}}
    assert compiled.parity_check_group == [{1}, {2}, {1, 2, 3}, {2, 3, 4}]
    assert compiled.logical_observables == {0: {1, 2, 3, 4}}
    # Repeating the transform must not absorb retained feedback into the parities.
    repeated = compiled.backpropagate(graph)
    assert repeated.xflow == compiled.xflow
    assert repeated.zflow == compiled.zflow
    assert repeated.cflow == compiled.cflow
    assert repeated.parity_check_group == compiled.parity_check_group
    assert repeated.logical_observables == compiled.logical_observables
    pattern = qompile(graph, compiled)
    assert isinstance(pattern.frame, CliffordFrame)
    assert pattern.frame.parity_check_group == compiled.parity_check_group


@pytest.mark.parametrize("quantum_output", [False, True])
def test_dependencies_of_quantum_outputs_and_non_pauli_measurements_are_retained(quantum_output: bool) -> None:
    graph = _graph(Axis.Z, Axis.Z, None if quantum_output else Axis.X)
    if not quantum_output:
        graph.assign_meas_basis(2, PlannerMeasBasis(Plane.XY, np.pi / 4))
    original = Feedforward({0: {1}, 1: {2}}, {}, parity_check_group=[{1}])

    compiled = original.backpropagate(graph)

    assert compiled.xflow == {0: {1}, 1: {2}}
    assert compiled.parity_check_group == [{1}]


def test_normalization_precedes_boundary_selection() -> None:
    graph = _graph(Axis.Z, Axis.Z)
    original = Feedforward({0: {1}}, {}, {0: {1: ca.X}}, parity_check_group=[{1}])

    compiled = original.backpropagate(graph)

    assert compiled.xflow == {}
    assert compiled.zflow == {}
    assert compiled.cflow == {}
    assert compiled.parity_check_group == [{1}]
    assert original.cflow == {0: {1: ca.X}}


def test_long_dependency_chain_does_not_use_recursive_full_closure_caches() -> None:
    length = 1500
    graph = _graph(*(Axis.Z for _ in range(length)))
    original = Feedforward({node: {node + 1} for node in range(length - 1)}, {}, parity_check_group=[{length - 1}])
    compiled = original.backpropagate(graph)
    assert compiled.parity_check_group == [set(range(length))]
    assert compiled.xflow == {}
    frame = PauliFrame(graph, original.xflow, {}, parity_check_group=original.parity_check_group)
    assert frame.detector_groups() == compiled.parity_check_group


@pytest.mark.parametrize("support", [{2}, {9}])
def test_unmeasured_or_unknown_parity_references_are_rejected(support: set[int]) -> None:
    graph = _graph(Axis.Z, Axis.Z, None)
    with pytest.raises(ValueError, match="unmeasured"):
        Feedforward({}, {}, parity_check_group=[support]).backpropagate(graph)


def test_invalid_dependency_is_rejected_before_it_could_be_removed() -> None:
    graph = _graph(Axis.Z, Axis.Z)
    with pytest.raises(ValueError, match="Cycle detected"):
        Feedforward({0: {1}, 1: {0}}, {}).backpropagate(graph)


def test_qompile_rejects_ambiguous_ir_and_separate_map_arguments() -> None:
    graph = _graph(Axis.Z, Axis.Z)
    with pytest.raises(TypeError, match="do not mix"):
        qompile(graph, Feedforward({}, {}), parity_check_group=[{0}])


def test_qompile_validates_a_supplied_schedule_against_the_residual_flows() -> None:
    graph = _graph(Axis.Z, Axis.Z)
    original = Feedforward({0: {1}}, {}, parity_check_group=[{1}])
    scheduler = Scheduler(graph, {}, {})
    scheduler.manual_schedule(prepare_time={}, measure_time={0: 0, 1: 0})
    with pytest.raises(ValueError, match="DAG violation"):
        qompile(graph, original, scheduler=scheduler)
    pattern = qompile(graph, original.backpropagate(graph), scheduler=scheduler)
    assert pattern.frame.parity_check_group == [{0, 1}]


def test_ptn_roundtrip_and_stim_export_do_not_backpropagate_again() -> None:
    graph = _graph(Axis.Z, Axis.Z, Axis.Z)
    original = Feedforward({0: {1}, 1: {2}}, {}, parity_check_group=[{1}, {2}], logical_observables={5: {2}})
    pattern = qompile(graph, original.backpropagate(graph))
    restored = loads(dumps(pattern))

    assert restored.frame.parity_check_group == [{0, 1}, {0, 1, 2}]
    assert restored.frame.logical_observables == {5: {0, 1, 2}}
    circuit = stim.Circuit(stim_compile(restored))
    assert circuit.num_detectors == 2
    assert circuit.num_observables == 6
    assert all("rec[" not in str(op) for op in circuit if op.name in {"CX", "CZ"})
    assert circuit.get_detector_coordinates().keys() == {0, 1}


def test_retained_pauli_feedback_is_executed_by_stim_instead_of_expanding_annotations() -> None:
    graph = GraphState()
    for index in range(3):
        node = graph.add_node()
        graph.register_input(node, index, init=Initialization(Axis.X if index == 0 else Axis.Z))
        graph.assign_meas_basis(node, AxisMeasBasis(Axis.Z, Sign.PLUS))
    original = Feedforward({0: {1}, 1: {2}}, {}, parity_check_group=[{0, 2}], logical_observables={0: {0, 2}})
    circuit = stim.Circuit(stim_compile(qompile(graph, original)))

    assert str(circuit).count("CX rec[-1]") == 2
    detector = next(op for op in circuit if op.name == "DETECTOR")
    assert isinstance(detector, stim.CircuitInstruction)
    assert {target.value for target in detector.targets_copy()} == {-3, -1}
    compiled_circuit = stim.Circuit(stim_compile(qompile(graph, original.backpropagate(graph))))
    # A random Z readout on qubit 0 is copied through the two |0> targets.
    assert circuit.has_flow(stim.Flow("1 -> rec[-3] xor rec[-1]"))
    assert compiled_circuit.has_flow(stim.Flow("1 -> rec[-2] xor rec[-1]"))
    assert circuit.detector_error_model().num_detectors == 1
    assert compiled_circuit.detector_error_model().num_detectors == 1


def test_stim_executes_retained_feedback_on_a_quantum_output() -> None:
    graph = GraphState()
    source, output = graph.add_node(), graph.add_node()
    graph.register_input(source, 0)
    graph.register_input(output, 1, init=Initialization(Axis.Z))
    graph.register_output(output, 0)
    graph.assign_meas_basis(source, AxisMeasBasis(Axis.Z, Sign.PLUS))
    compiled = Feedforward({source: {output}}, {}).backpropagate(graph)

    circuit = stim.Circuit(stim_compile(qompile(graph, compiled)))

    assert compiled.xflow == {source: {output}}
    assert circuit.has_flow(stim.Flow("1 -> Z1 xor rec[-1]"))


def test_clifford_boundary_preserves_the_full_born_weighted_instrument(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare all branch Choi operators, control records, parities, and quantum output."""
    graph = _graph(Axis.Z, Axis.Z, Axis.X, Axis.Y, Axis.Z, None)
    for node in range(5):
        graph.add_edge(node, node + 1)
    # Six untouched reference qubits certify the channel on arbitrary inputs.
    for index in range(6):
        reference = graph.add_node()
        graph.register_input(reference, 6 + index)
        graph.register_output(reference, index + 1)
    original = Feedforward(
        {0: {1}, 1: {4}, 2: {3}, 3: {4}},
        {0: {2}, 1: {3, 5}},
        {1: {2: ca.S}},
        parity_check_group=[{1}, {3}, {4}],
        logical_observables={0: {1, 4}},
    )
    compiled = original.backpropagate(graph)
    initial = np.eye(64, dtype=np.complex128).ravel() / 8

    def instrument(pattern: Pattern) -> dict[tuple[bool, ...], NDArray[np.complex128]]:
        instrument_states: dict[tuple[bool, ...], NDArray[np.complex128]] = {}
        outcomes = iter((False,))
        probability = 1.0

        def force_measure(state: StateVector, qubit: int, basis: MeasBasis, rng: np.random.Generator) -> bool:
            nonlocal probability
            del rng
            outcome = next(outcomes)
            projected = np.tensordot(basis.vector().conj(), state.state(), axes=(0, qubit))
            prob_zero = float(np.linalg.norm(projected) ** 2)
            probability *= 1 - prob_zero if outcome else prob_zero
            state.measure(qubit, basis, int(outcome))
            return outcome

        monkeypatch.setattr(StateVector, "sample_measure", force_measure)
        for branch in itertools.product((False, True), repeat=5):
            outcomes = iter(branch)
            probability = 1.0
            simulator = PatternSimulator(deepcopy(pattern), SimulatorBackend.StateVector)
            simulator.state = StateVector(initial)
            simulator.simulate()
            records = simulator.results
            parities = tuple(
                bool(sum(records[node] for node in group) % 2)
                for group in (*pattern.frame.parity_check_group, *pattern.frame.logical_observables.values())
            )
            key = (*parities, records[0], records[1], records[2])
            output = simulator.state.state().ravel()
            weighted_state = probability * np.outer(output, output.conj())
            instrument_states[key] = instrument_states.get(key, np.zeros_like(weighted_state)) + weighted_state
        return instrument_states

    before = instrument(qompile(graph, original))
    after = instrument(qompile(graph, compiled))

    assert before.keys() == after.keys()
    assert sum(float(np.trace(state).real) for state in before.values()) == pytest.approx(1.0, abs=1e-12)
    assert sum(float(np.trace(state).real) for state in after.values()) == pytest.approx(1.0, abs=1e-12)
    for key, state in before.items():
        np.testing.assert_allclose(after[key], state, atol=1e-12, rtol=0, err_msg=f"records/parities {key}")
