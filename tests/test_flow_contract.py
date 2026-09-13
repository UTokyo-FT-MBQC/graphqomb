"""Flow sources, shared scheduling dependencies, and ignored self-targets."""

from __future__ import annotations

import itertools

import pytest

from graphqomb import clifford_algebra as ca
from graphqomb.common import Plane, PlannerMeasBasis
from graphqomb.feedforward import check_flow, dag_from_flow
from graphqomb.graphstate import GraphState
from graphqomb.pattern import is_runnable
from graphqomb.pauli_frame import CliffordFrame
from graphqomb.qompiler import qompile
from graphqomb.scheduler import Scheduler


def _graph() -> GraphState:
    graph = GraphState()
    for node in range(3):
        graph.add_node()
        graph.register_input(node, node)
    graph.register_output(2, 0)
    for node in (0, 1):
        graph.assign_meas_basis(node, PlannerMeasBasis(Plane.XY, 0))
    return graph


def _flows(
    kind: str, source: int, target: int
) -> tuple[dict[int, set[int]], dict[int, set[int]], dict[int, dict[int, ca.C1Element]]]:
    return (
        {source: {target}} if kind == "x" else {},
        {source: {target}} if kind == "z" else {},
        {source: {target: ca.S}} if kind == "c" else {},
    )


@pytest.mark.parametrize("kind", ["x", "z", "c"])
def test_unmeasured_output_source_is_rejected(kind: str) -> None:
    graph = _graph()
    xflow, zflow, cflow = _flows(kind, 2, 1)
    with pytest.raises(ValueError, match="Flow source 2 is not measured"):
        check_flow(graph, xflow, zflow, cflow)
    with pytest.raises(ValueError, match="Flow source 2 is not measured"):
        Scheduler(graph, xflow, zflow, cflow)
    with pytest.raises(ValueError, match="Flow source 2 is not measured"):
        CliffordFrame(graph, xflow, zflow, cflow=cflow)
    with pytest.raises(ValueError, match="Flow source 2 is not measured"):
        qompile(graph, xflow, zflow, cflow)


def test_integer_flow_with_zero_target_still_checks_source() -> None:
    with pytest.raises(ValueError, match="Flow source 2 is not measured"):
        check_flow(_graph(), {2: 0})


@pytest.mark.parametrize("kind", ["x", "z", "c"])
def test_measured_output_can_control_corrections(kind: str) -> None:
    graph = _graph()
    graph.assign_meas_basis(2, PlannerMeasBasis(Plane.XY, 0))
    xflow, zflow, cflow = _flows(kind, 2, 1)
    check_flow(graph, xflow, zflow, cflow)
    scheduler = Scheduler(graph, xflow, zflow, cflow)
    assert scheduler.dag[2] == {1}
    assert scheduler.solve_schedule()
    is_runnable(qompile(graph, xflow, zflow, cflow, scheduler=scheduler))


@pytest.mark.parametrize("kind", ["x", "z", "c"])
@pytest.mark.parametrize("source_time", [0, 1])
def test_supplied_scheduler_checks_all_current_flows(kind: str, source_time: int) -> None:
    graph = _graph()
    xflow, zflow, cflow = _flows(kind, 0, 1)
    scheduler = Scheduler(graph, {}, {})
    scheduler.manual_schedule(prepare_time={}, measure_time={0: source_time, 1: 0})
    scheduler.validate_schedule()
    with pytest.raises(ValueError, match="DAG violation"):
        qompile(graph, xflow, zflow, cflow, scheduler=scheduler)
    assert scheduler.dag == {0: set(), 1: set(), 2: set()}


@pytest.mark.parametrize("kind", ["x", "z", "c"])
def test_supplied_scheduler_accepts_compatible_timing(kind: str) -> None:
    graph = _graph()
    xflow, zflow, cflow = _flows(kind, 0, 1)
    scheduler = Scheduler(graph, {}, {})
    scheduler.manual_schedule(prepare_time={}, measure_time={0: 0, 1: 1})
    is_runnable(qompile(graph, xflow, zflow, cflow, scheduler=scheduler))
    assert scheduler.dag[0] == set()  # Validation must not mutate a shared scheduler.
    current = Scheduler(graph, xflow, zflow, cflow)
    assert current.dag == dag_from_flow(graph, xflow, zflow, cflow)
    assert current.dag[0] == {1}
    assert current.solve_schedule()
    source_time, target_time = current.measure_time[0], current.measure_time[1]
    assert source_time is not None
    assert target_time is not None
    assert source_time < target_time


def test_scheduler_validation_uses_normalized_cflow() -> None:
    graph = _graph()
    scheduler = Scheduler(graph, {}, {})
    scheduler.manual_schedule(prepare_time={}, measure_time={0: 1, 1: 0})
    # Pure-Pauli cflow adds a dependency even though its coset is trivial.
    with pytest.raises(ValueError, match="DAG violation"):
        qompile(graph, {}, {}, {0: {1: ca.X}}, scheduler=scheduler)
    # The same correction cancels an explicit X; no dependency remains.
    is_runnable(qompile(graph, {0: {1}}, {}, {0: {1: ca.X}}, scheduler=scheduler))


@pytest.mark.parametrize(
    "gate",
    [
        ca.compose(ca.compose(d, x), z)
        for d, x, z in itertools.product(ca.TRANSVERSAL, (ca.IDENTITY, ca.X), (ca.IDENTITY, ca.Z))
    ],
)
def test_self_cflow_is_allowed_but_not_applied(gate: ca.C1Element) -> None:
    graph = _graph()
    frame = CliffordFrame(graph, {0: {0}}, {0: {0}}, cflow={0: {0: gate, 1: ca.S}})
    assert frame.parents(0) == set()
    assert frame.children(0) == {1}
    frame.meas_flip(0)
    assert frame.coset[0] == ca.IDENTITY
    assert not frame.x_pauli[0]
    assert not frame.z_pauli[0]
    assert frame.coset[1] == ca.S
    assert frame.z_pauli[1]
    assert all(target != 0 for target, _gate in frame.correction_events[0])
    is_runnable(qompile(graph, {0: {0}}, {0: {0}}, {0: {0: gate, 1: ca.S}}))
