"""Steane logical T injection, compiled parities, and logical/fault checks."""

from __future__ import annotations

import itertools
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_array

from graphqomb import clifford_algebra as ca
from graphqomb.command import M
from graphqomb.common import Axis, AxisMeasBasis, Initialization, Sign
from graphqomb.feedforward import Feedforward
from graphqomb.graphstate import compose_into, odd_neighbors
from graphqomb.qeccode import StabilizerCode, build_graph_state
from graphqomb.qompiler import qompile
from graphqomb.scheduler import ScheduleConfig, Scheduler, Strategy
from graphqomb.simulator import PatternSimulator, SimulatorBackend
from graphqomb.statevec import StateVector

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from graphqomb.pattern import Pattern
    from graphqomb.qeccode import StabilizerGraphStateBuildResult

STEANE_SUPPORTS = ((0, 1, 2, 3), (0, 1, 4, 5), (0, 2, 4, 6))
LOGICAL_SUPPORT = (4, 5, 6)
_INJECTION_LAYER = 2
# Three quadrilateral plaquettes meet at q0; q4/q5/q6 form the lower boundary.
PATCH_VERTICES = ((0.0, 0.0), (-0.9, 0.52), (0.9, 0.52), (0.0, 2.08), (0.0, -1.04), (-1.8, -1.04), (1.8, -1.04))
DATA_LAYER_Z = (0.0, 1.0, 2.0, 3.5, 5.0)


@dataclass(frozen=True)
class SteanePattern:
    """Resource node identities for encoded ZZ-based T injection."""

    pattern: Pattern
    data_nodes: tuple[int, ...]
    data_layers: tuple[tuple[int, ...], ...]
    before_nodes: tuple[int, ...]
    after_nodes: tuple[int, ...]
    magic_nodes: tuple[int, ...]
    probe_node: int
    control_node: int
    readout_axis: Axis | None


@dataclass(frozen=True)
class SteaneShot:
    """Injection bits, both syndrome rounds, final detectors, and logical bit."""

    injection: tuple[bool, bool]
    correction_control: bool
    before: tuple[bool, ...]
    after: tuple[bool, ...]
    detectors: tuple[bool, ...]
    observable: bool


def steane_basis() -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Build logical zero from the simplex row space, and its complement coset.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The normalized 128-component logical zero and one states.
    """
    zero = np.zeros(128, dtype=np.complex128)
    for coefficients in itertools.product((False, True), repeat=len(STEANE_SUPPORTS)):
        support: set[int] = set()
        for coefficient, row in zip(coefficients, STEANE_SUPPORTS, strict=True):
            if coefficient:
                support.symmetric_difference_update(row)
        index = sum(1 << (6 - qubit) for qubit in support)
        zero[index] = 1 / np.sqrt(8)
    return zero, zero[::-1].copy()


def steane_input(logical_state: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Encode the data, append an encoded T magic state, and a zero control.

    Returns
    -------
    numpy.ndarray
        A normalized 15-qubit state in the pattern's input-qubit order.

    Raises
    ------
    ValueError
        If the logical input is not a normalized two-component vector.
    """
    if logical_state.shape != (2,) or not np.isclose(np.linalg.norm(logical_state), 1.0):
        msg = "The logical input must be a normalized two-component state."
        raise ValueError(msg)
    zero, one = steane_basis()
    data = logical_state[0] * zero + logical_state[1] * one
    magic = (zero + np.exp(1j * np.pi / 4) * one) / np.sqrt(2)
    return np.asarray(np.kron(np.kron(data, magic), [1.0, 0.0]), dtype=np.complex128)


def _syndrome_round(z_base: int) -> tuple[StabilizerGraphStateBuildResult, Scheduler, dict[int, set[int]]]:
    """Build and schedule one Steane round with at most eight live qubits.

    Returns
    -------
    tuple
        The graph builder result, sequential schedule, and data teleportation flow.
    """
    rows = [[int(qubit in support) for qubit in range(7)] for support in STEANE_SUPPORTS]
    matrix = [row + [0] * 7 for row in rows] + [[0] * 7 + row for row in rows]
    resource = build_graph_state(
        StabilizerCode(csr_array(np.asarray(matrix, dtype=np.uint8))), z_base=z_base, data_as_io=True
    )
    for (qubit, layer), node in resource.data_nodes.items():
        vertex_x, vertex_y = PATCH_VERTICES[qubit]
        resource.graph.set_coordinate(node, (vertex_x, vertex_y, DATA_LAYER_Z[layer]))
    for row, node in resource.ancilla_nodes.items():
        # X checks attach to the middle data layer; Z checks attach to its input.
        layer = z_base + (1 if row < len(STEANE_SUPPORTS) else 0)
        support = STEANE_SUPPORTS[row % len(STEANE_SUPPORTS)]
        center_x = sum(PATCH_VERTICES[q][0] for q in support) / len(support)
        center_y = sum(PATCH_VERTICES[q][1] for q in support) / len(support)
        # Separate extraction from the data plane, and place round 2 after injection.
        offset_z = 0.75 if layer == _INJECTION_LAYER else 0.25
        resource.graph.set_coordinate(node, (center_x, center_y, DATA_LAYER_Z[layer] + offset_z))
    xflow = {
        resource.data_nodes[qubit, layer]: {resource.data_nodes[qubit, layer + 1]}
        for qubit in range(7)
        for layer in (z_base, z_base + 1)
    }
    scheduler = Scheduler(resource.graph, xflow)
    scheduler.solve_schedule(ScheduleConfig(strategy=Strategy.MINIMIZE_SPACE))
    return resource, scheduler, xflow


def build_steane_pattern(readout_axis: Axis | None) -> SteanePattern:  # ruff:ignore[too-many-statements, too-many-locals]
    """Compile two syndrome rounds, logical T injection, and optional readout.

    Transform the feedforward IR before lowering while retaining the
    sequential schedule.

    Returns
    -------
    SteanePattern
        A 56-node pattern with encoded data/magic inputs and a delayed S control.

    Raises
    ------
    ValueError
        If the readout is neither X, Z, nor an unmeasured quantum output.
    """
    if readout_axis not in {Axis.X, Axis.Z, None}:
        msg = "The Steane demo supports X, Z, or quantum output."
        raise ValueError(msg)
    before, before_schedule, xflow = _syndrome_round(0)
    graph = before.graph
    injection_data = tuple(before.data_nodes[qubit, 2] for qubit in range(7))
    # Rotate the second 2D patch so the logical-Z boundaries face one another.
    magic_nodes = tuple(graph.add_node(coordinate=(-x, -4.5 - y, 2.5)) for x, y in PATCH_VERTICES)
    for qubit, node in enumerate(magic_nodes):
        graph.register_input(node, 7 + qubit)
        graph.assign_meas_basis(node, AxisMeasBasis(Axis.X, Sign.PLUS))
    probe = graph.add_node(coordinate=(0.0, -2.25, 2.25))
    graph.assign_meas_basis(probe, AxisMeasBasis(Axis.X, Sign.PLUS))
    for qubit in LOGICAL_SUPPORT:
        graph.add_edge(probe, injection_data[qubit])
        graph.add_edge(probe, magic_nodes[qubit])
    control = graph.add_node(coordinate=(3.8, -2.25, 4.5))
    graph.register_input(control, 14, init=Initialization(Axis.Z))
    graph.assign_meas_basis(control, AxisMeasBasis(Axis.Z, Sign.PLUS))

    after, after_schedule, after_xflow = _syndrome_round(2)
    node_map = compose_into(graph, after.graph)
    xflow.update(
        {node_map[source]: {node_map[target] for target in targets} for source, targets in after_xflow.items()}
    )
    data = tuple(node_map[after.data_nodes[qubit, 4]] for qubit in range(7))
    before_nodes = tuple(before.ancilla_nodes[row] for row in range(6))
    after_nodes = tuple(node_map[after.ancilla_nodes[row]] for row in range(6))
    if readout_axis is not None:
        for node in data:
            graph.assign_meas_basis(node, AxisMeasBasis(readout_axis, Sign.PLUS))

    # Copy the logical ZZ bit into a |0> control with X feedforward. Its Z
    # measurement can be delayed until every syndrome-round Pauli event is over.
    xflow[probe] = {control}
    zflow = {source: odd_neighbors(targets, graph) for source, targets in xflow.items()}
    for qubit in LOGICAL_SUPPORT:
        zflow.setdefault(magic_nodes[qubit], set()).update(data[index] for index in LOGICAL_SUPPORT)
    earlier = (set(graph.meas_bases) - set(data)) - {control}
    for source in earlier:
        zflow.setdefault(source, set()).add(control)
    # Physical S^dagger on all seven qubits implements logical S for this basis:
    # simplex words have weight 0/4; the logical-one coset has weight 3/7.
    cflow = {control: dict.fromkeys(data, ca.inverse(ca.S))}

    # Keep the injection and both rounds in causal order. Reuse the sequential
    # round schedules so the seven-qubit magic register never causes a wide
    # simultaneous preparation of syndrome ancillas and teleportation targets.
    prepare = {node: time for node, time in before_schedule.prepare_time.items() if time is not None}
    measure = {node: time for node, time in before_schedule.measure_time.items() if time is not None}
    entangle = {edge: time for edge, time in before_schedule.entangle_time.items() if time is not None}
    injection_time = before_schedule.num_slices()
    prepare[probe] = injection_time
    measure[probe] = injection_time + 1
    measure.update(dict.fromkeys(magic_nodes, injection_time + 2))
    entangle.update({edge: injection_time for edge in graph.edges if probe in edge})
    offset = injection_time + 3
    prepare.update(
        {node_map[node]: offset + time for node, time in after_schedule.prepare_time.items() if time is not None}
    )
    measure.update(
        {node_map[node]: offset + time for node, time in after_schedule.measure_time.items() if time is not None}
    )
    entangle.update(
        {
            (node_map[a], node_map[b]): offset + time
            for (a, b), time in after_schedule.entangle_time.items()
            if time is not None
        }
    )
    control_time = offset + after_schedule.num_slices()
    measure[control] = control_time
    if readout_axis is not None:
        measure.update(dict.fromkeys(data, control_time + 1))
    scheduler = Scheduler(graph, xflow, zflow, cflow=cflow)
    scheduler.manual_schedule(prepare, measure, entangle)
    detectors = [{a, b} for a, b in zip(before_nodes, after_nodes, strict=True)]
    detector_tags = [f"repeat_{axis.name}_{row}" for axis in (Axis.X, Axis.Z) for row in range(3)]
    if readout_axis is not None:
        detectors.extend({data[qubit] for qubit in support} for support in STEANE_SUPPORTS)
        detector_tags.extend(f"data_{readout_axis.name}_{row}" for row in range(3))
    # All seven magic X records participate in the three X-stabilizer checks.
    detectors.extend({magic_nodes[qubit] for qubit in support} for support in STEANE_SUPPORTS)
    detector_tags.extend(f"magic_X_{row}" for row in range(3))
    feedforward = Feedforward(
        xflow,
        zflow,
        cflow=cflow,
        parity_check_group=detectors,
        parity_check_tags=detector_tags,
        logical_observables={0: {data[qubit] for qubit in LOGICAL_SUPPORT}} if readout_axis is not None else {},
    )
    pattern = qompile(graph, feedforward.backpropagate(graph), scheduler=scheduler)
    data_layers = tuple(tuple(before.data_nodes[q, layer] for q in range(7)) for layer in range(_INJECTION_LAYER + 1))
    data_layers += tuple(tuple(node_map[after.data_nodes[q, layer]] for q in range(7)) for layer in (3, 4))
    return SteanePattern(
        pattern, data, data_layers, before_nodes, after_nodes, magic_nodes, probe, control, readout_axis
    )


def run_steane_shot(
    steane: SteanePattern,
    logical_state: NDArray[np.complex128],
    *,
    seed: int = 0,
    fault: tuple[int, Axis] | None = None,
) -> SteaneShot:
    """Sample the readout pattern, optionally applying a post-syndrome fault.

    Returns
    -------
    SteaneShot
        The two injection bits, syndrome records, and readout parities.

    Raises
    ------
    ValueError
        If quantum output is requested or the fault qubit is outside the register.
    """
    if steane.readout_axis is None:
        msg = "Use PatternSimulator.simulate() for the quantum-output pattern."
        raise ValueError(msg)
    if fault is not None and not 0 <= fault[0] < len(steane.data_nodes):
        msg = "The Steane fault qubit must be between 0 and 6."
        raise ValueError(msg)
    simulator = PatternSimulator(deepcopy(steane.pattern), SimulatorBackend.StateVector, calc_prob=True)
    simulator.state = StateVector(steane_input(logical_state))
    rng = np.random.default_rng(seed)
    for command in steane.pattern:
        if fault is not None and isinstance(command, M) and command.node == steane.control_node:
            qubit, axis = fault
            operator = ca.to_matrix({Axis.X: ca.X, Axis.Y: ca.Y, Axis.Z: ca.Z}[axis])
            simulator.state.evolve(operator, simulator.node_indices.index(steane.data_nodes[qubit]))
        simulator.apply_cmd(command, rng=rng)
    records = simulator.results
    p = records[steane.probe_node]
    m = False
    for qubit in LOGICAL_SUPPORT:
        m ^= records[steane.magic_nodes[qubit]]
    before = tuple(records[node] for node in steane.before_nodes)
    after = tuple(records[node] for node in steane.after_nodes)
    frame = steane.pattern.frame
    detectors = tuple(bool(sum(records[node] for node in group) % 2) for group in frame.parity_check_group)
    observable = bool(sum(records[node] for node in frame.logical_observables[0]) % 2)
    return SteaneShot((p, m), records[steane.control_node], before, after, detectors, observable)


def validate_steane() -> dict[str, int]:
    """Compare quantum outputs with logical T and check all boundary Pauli faults.

    Returns
    -------
    dict[str, int]
        Numbers of quantum-state and single-Pauli-fault executions verified.

    Raises
    ------
    AssertionError
        If fidelity, correction control, or detector response is incorrect.
    """
    quantum = build_steane_pattern(None)
    logical = np.asarray([0.3 + 0.1j, 0.7 - 0.2j], dtype=np.complex128)
    logical /= np.linalg.norm(logical)
    zero, one = steane_basis()
    expected = logical[0] * zero + np.exp(1j * np.pi / 4) * logical[1] * one
    branches: set[tuple[bool, bool]] = set()
    for seed in range(16):
        simulator = PatternSimulator(deepcopy(quantum.pattern), SimulatorBackend.StateVector, calc_prob=True)
        simulator.state = StateVector(steane_input(logical))
        simulator.simulate(np.random.default_rng(seed))
        fidelity = abs(np.vdot(expected, np.asarray(simulator.state.state()).ravel())) ** 2
        p = simulator.results[quantum.probe_node]
        m = False
        for qubit in LOGICAL_SUPPORT:
            m ^= simulator.results[quantum.magic_nodes[qubit]]
        branches.add((p, m))
        if not np.isclose(fidelity, 1.0, rtol=0.0, atol=1e-12) or simulator.results[quantum.control_node] != p:
            msg = f"Incorrect logical T output at seed={seed}: fidelity={fidelity}, p={p}"
            raise AssertionError(msg)
        if any(simulator.results[node] for node in (*quantum.before_nodes, *quantum.after_nodes)):
            msg = f"Nonzero ideal Steane syndrome at seed={seed}"
            raise AssertionError(msg)
        for support in STEANE_SUPPORTS:
            if sum(simulator.results[quantum.magic_nodes[q]] for q in support) % 2:
                msg = f"Nonzero ideal magic X stabilizer at seed={seed}, support={support}"
                raise AssertionError(msg)
    if branches != set(itertools.product((False, True), repeat=2)):
        msg = f"Validation did not sample all four injection branches: {branches}"
        raise AssertionError(msg)
    fault_count = _validate_steane_faults(logical)
    return {"quantum_output": 16, "injection_branches": len(branches), "single_pauli_fault": fault_count}


def _validate_steane_faults(logical: NDArray[np.complex128]) -> int:
    """Check detector responses for both readouts and all boundary Pauli faults.

    Returns
    -------
    int
        The number of verified fault executions.

    Raises
    ------
    AssertionError
        If a detector or delayed control is incorrect, or a ZZ branch is missing.
    """
    fault_count = 0
    for readout in (Axis.X, Axis.Z):
        steane = build_steane_pattern(readout)
        # Both ZZ branches occur among these noiseless seeds. Faults occur after
        # the probe, so using the same seed preserves the sampled injection bit.
        controls: set[bool] = set()
        for seed in range(8):
            for qubit, axis in itertools.product(range(7), Axis):
                shot = run_steane_shot(steane, logical, seed=seed, fault=(qubit, axis))
                p = shot.injection[0]
                controls.add(p)
                effective = (Axis.Y if p else Axis.X) if readout == Axis.X else Axis.Z
                flip = axis != effective
                expected_detectors = (
                    (False,) * 6 + tuple(flip and qubit in support for support in STEANE_SUPPORTS) + (False,) * 3
                )
                if shot.detectors != expected_detectors or shot.correction_control != p:
                    msg = (
                        f"Incorrect Steane fault response: readout={readout}, seed={seed}, "
                        f"fault={(qubit, axis)}, {shot}"
                    )
                    raise AssertionError(msg)
                fault_count += 1
        if controls != {False, True}:
            msg = f"Fault validation did not sample both ZZ branches: {controls}"
            raise AssertionError(msg)
    return fault_count
