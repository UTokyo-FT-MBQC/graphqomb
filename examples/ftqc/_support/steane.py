"""Steane logical T gate teleportation with a supplied encoded magic state."""

from __future__ import annotations

import itertools
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import stim

from graphqomb import clifford_algebra as ca
from graphqomb.command import M
from graphqomb.common import Axis
from graphqomb.feedforward import Feedforward, dag_from_flow, inverse_dag_from_dag
from graphqomb.qompiler import qompile
from graphqomb.scheduler import ScheduleConfig, Scheduler, Strategy
from graphqomb.simulator import PatternSimulator, SimulatorBackend
from graphqomb.statevec import StateVector
from graphqomb.stim_glue.importer import stim_circuit_to_pattern

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from graphqomb.pattern import Pattern

STEANE_SUPPORTS = ((0, 1, 2, 3), (0, 1, 4, 5), (0, 2, 4, 6))
LOGICAL_SUPPORT = (4, 5, 6)
CODE_DISTANCE = 3
MEMORY_ROUNDS = CODE_DISTANCE
MAGIC_PATCH_OFFSET = (5.4, -1.6)
# Three quadrilateral plaquettes meet at q0; q4/q5/q6 form the lower boundary.
PATCH_VERTICES = ((0.0, 0.0), (-0.9, 0.52), (0.9, 0.52), (0.0, 2.08), (0.0, -1.04), (-1.8, -1.04), (1.8, -1.04))


@dataclass(frozen=True)
class SteaneSource:
    """Clifford Stim circuit and the records defining encoded teleportation."""

    circuit: stim.Circuit
    before_records: tuple[int, ...]
    magic_records: tuple[int, ...]
    after_records: tuple[tuple[int, ...], ...]
    consumed_records: tuple[int, ...]


@dataclass(frozen=True)
class SteanePattern:
    """Resource node identities for transversal-CNOT T teleportation."""

    pattern: Pattern
    source: SteaneSource
    data_nodes: tuple[int, ...]
    data_input_nodes: tuple[int, ...]
    magic_input_nodes: tuple[int, ...]
    before_nodes: tuple[int, ...]
    magic_before_nodes: tuple[int, ...]
    after_nodes: tuple[tuple[int, ...], ...]
    correction_nodes: tuple[int, ...]
    cnot_edges: tuple[tuple[int, int], ...]
    consumed_nodes: tuple[int, ...]
    control_node: int
    readout_axis: Axis | None


@dataclass(frozen=True)
class SteaneShot:
    """Teleportation control, compiled detectors, and logical bit."""

    teleportation_bit: bool
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
    """Construct encoded data and T-state input vectors directly.

    This supplies the initial state, without simulating a preparation circuit.

    Returns
    -------
    numpy.ndarray
        A normalized 14-qubit state in the pattern's input-qubit order.

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
    return np.asarray(np.kron(data, magic), dtype=np.complex128)


def _append_memory(circuit: stim.Circuit, offset: int, *, end_round: bool = True) -> tuple[int, ...]:
    """Append an ideal Steane memory round and return its six record indices.

    Returns
    -------
    tuple[int, ...]
        X-check records followed by Z-check records.
    """
    records = []
    for axis in (stim.target_x, stim.target_z):
        for support in STEANE_SUPPORTS:
            targets: list[stim.GateTarget] = []
            for qubit in support:
                if targets:
                    targets.append(stim.target_combiner())
                targets.append(axis(offset + qubit))
            records.append(circuit.num_measurements)
            circuit.append("MPP", targets)
    if end_round:
        circuit.append("TICK", [])
    return tuple(records)


def build_steane_circuit(readout_axis: Axis | None) -> SteaneSource:  # ruff:ignore[complex-structure, too-many-branches]
    """Describe the Clifford part of encoded T teleportation in Stim.

    Encoded inputs and the conditional S correction are supplied separately.
    MPP represents ideal syndrome extraction, not a physical FT extraction circuit.

    Returns
    -------
    SteaneSource
        Circuit and absolute measurement indices used for cflow and validation.

    Raises
    ------
    ValueError
        If the readout axis is unsupported.
    """
    if readout_axis not in {Axis.X, Axis.Z, None}:
        msg = "The Steane demo supports X, Z, or quantum output."
        raise ValueError(msg)
    circuit = stim.Circuit()
    for block in range(2):
        for qubit, (x, y) in enumerate(PATCH_VERTICES):
            coordinate = (x, y, 0.0) if block == 0 else (x + MAGIC_PATCH_OFFSET[0], y + MAGIC_PATCH_OFFSET[1], 0.0)
            circuit.append("QUBIT_COORDS", [7 * block + qubit], coordinate)
    before_records = _append_memory(circuit, 0, end_round=False)
    magic_records = _append_memory(circuit, 7)
    circuit.append("CX", [node for qubit in range(7) for node in (7 + qubit, qubit)])
    circuit.append("TICK", [])
    consumed_records = []
    for qubit in range(6):
        consumed_records.append(circuit.num_measurements)
        circuit.append("M", [qubit])
    circuit.append("TICK", [])
    # Relabel the last consumed Z result by the other logical-Z support bits.
    # This computes m with classical readout feedback, without a quantum relay.
    for qubit in LOGICAL_SUPPORT[:-1]:
        circuit.append("CX", [stim.target_rec(consumed_records[qubit] - circuit.num_measurements), 6])
    consumed_records.append(circuit.num_measurements)
    circuit.append("M", [6])
    circuit.append("TICK", [])
    # X^tensor7 is logical X. The Z factor of physical S-dagger is
    # included here; its S coset is attached as cflow after import.
    for qubit in range(7):
        circuit.append("CX", [stim.target_rec(consumed_records[6] - circuit.num_measurements), 7 + qubit])
        circuit.append("CZ", [stim.target_rec(consumed_records[6] - circuit.num_measurements), 7 + qubit])
    circuit.append("TICK", [])
    after_records = tuple(_append_memory(circuit, 7) for _ in range(MEMORY_ROUNDS))
    output_records: tuple[int, ...] = ()
    if readout_axis is not None:
        start = circuit.num_measurements
        circuit.append("MX" if readout_axis == Axis.X else "M", list(range(7, 14)))
        output_records = tuple(range(start, start + 7))

    def detector(records: set[int], tag: str) -> None:
        circuit.append(
            "DETECTOR", [stim.target_rec(index - circuit.num_measurements) for index in sorted(records)], tag=tag
        )

    for block_name, records in (("data", before_records), ("magic", magic_records)):
        for row, record in enumerate(records):
            detector(
                {record},
                f"initial_{block_name}_{'X' if row < len(STEANE_SUPPORTS) else 'Z'}_{row % len(STEANE_SUPPORTS)}",
            )
    # The correction preserves the codespace. Its first memory round checks
    # that space directly; subsequent rounds compare corresponding syndromes.
    # Across conditional S, an X-check transfer relation would depend on m
    # times the earlier Z syndrome and cannot be stored as an affine parity.
    for round_index, records in enumerate(after_records):
        for row, record in enumerate(records):
            group = {record}
            if round_index:
                group.add(after_records[round_index - 1][row])
            axis = "X" if row < len(STEANE_SUPPORTS) else "Z"
            detector(group, f"memory_{round_index + 1}_{axis}_{row % len(STEANE_SUPPORTS)}")
    if readout_axis is not None:
        for row, support in enumerate(STEANE_SUPPORTS):
            group = {output_records[q] for q in support}
            group.add(after_records[-1][row + (0 if readout_axis == Axis.X else 3)])
            detector(group, f"data_{readout_axis.name}_{row}")
        circuit.append(
            "OBSERVABLE_INCLUDE",
            [stim.target_rec(output_records[q] - circuit.num_measurements) for q in LOGICAL_SUPPORT],
            0,
        )
    # Z_data(out)=Z_data(in) Z_magic(in). Undo the pivot's parity relabelling.
    parity_inputs = {consumed_records[q] for q in LOGICAL_SUPPORT[:-1]}
    for row, support in enumerate(STEANE_SUPPORTS):
        group = {consumed_records[q] for q in support}
        if consumed_records[6] in group:
            group ^= parity_inputs
        group ^= {before_records[row + 3], magic_records[row + 3]}
        detector(group, f"consumed_Z_{row}")
    return SteaneSource(circuit, before_records, magic_records, after_records, tuple(consumed_records))


def build_steane_pattern(readout_axis: Axis | None) -> SteanePattern:  # ruff:ignore[too-many-locals, complex-structure]
    """Import the Stim circuit, attach conditional S, and compile Feedforward.

    Returns
    -------
    SteanePattern
        Transversal-CNOT teleportation with memory on both initial blocks.

    Raises
    ------
    ValueError
        If the correction and round-order constraints cannot be scheduled.
    """
    source = build_steane_circuit(readout_axis)
    imported = stim_circuit_to_pattern(source.circuit, coord_dims=3)
    frame = imported.pattern.frame
    graph = frame.graphstate
    outputs = {qubit: node for node, qubit in graph.output_node_indices.items()}
    data = tuple(outputs[imported.stim_to_final_qubit[7 + qubit]] for qubit in range(7))
    initial = {qubit: node for node, qubit in graph.input_node_indices.items()}
    data_inputs = tuple(initial[imported.stim_to_qubit[qubit]] for qubit in range(7))
    magic_inputs = tuple(initial[imported.stim_to_qubit[7 + qubit]] for qubit in range(7))
    consumed = tuple(imported.record_to_node[record] for record in source.consumed_records)
    after_nodes = tuple(
        tuple(imported.record_to_node[record] for record in records) for records in source.after_records
    )
    control = consumed[6]
    # The source Pauli feedback targets the seven wires immediately before
    # the post-correction memory. S times its Z factor gives physical S-dagger,
    # implementing logical S in this Steane basis; X precedes both factors.
    correction_nodes = tuple(sorted(frame.xflow[control]))
    zflow = {node: set(targets) for node, targets in frame.zflow.items()}
    # Earlier Pauli events must precede S. Commuting Z dependencies can order
    # them before the consumed Z readout without changing its reported bit.
    inverse = inverse_dag_from_dag(dag_from_flow(graph, frame.xflow, zflow))
    # First-round Z-check ancillas are future extraction events. Their Pauli
    # updates must follow S in the Clifford frame even though their graph
    # measurements precede the first data-wire measurements of that round.
    future_checks = {node for nodes in after_nodes for node in nodes}
    earlier: set[int] = set()
    pending = list(correction_nodes)
    while pending:
        for parent in inverse.get(pending.pop(), ()):
            if parent != control and parent not in future_checks and parent not in earlier:
                earlier.add(parent)
                pending.append(parent)
    for node in earlier:
        zflow.setdefault(node, set()).add(control)
    feedforward = Feedforward(
        frame.xflow,
        zflow,
        {control: dict.fromkeys(correction_nodes, ca.S)},
        parity_check_group=frame.parity_check_group,
        parity_check_tags=frame.parity_check_tags,
        logical_observables=frame.logical_observables,
    ).backpropagate(graph)
    scheduler = Scheduler(graph, feedforward.xflow, feedforward.zflow, cflow=feedforward.cflow)
    # Keep the requested round order even where offline Pauli compilation
    # would permit measurements to move across source-circuit TICKs.
    # cflow is applied to the output frame when the control is measured.
    # Its target nodes also serve as the first memory round's input nodes.
    stages = [(control,), *after_nodes]
    if readout_axis is not None:
        stages.append(data)
    for previous, following in itertools.pairwise(stages):
        for node in previous:
            scheduler.dag.setdefault(node, set()).update(following)
    if not scheduler.solve_schedule(ScheduleConfig(strategy=Strategy.MINIMIZE_SPACE)):
        msg = "Cannot schedule the conditional correction followed by Steane memory rounds."
        raise ValueError(msg)
    pattern = qompile(graph, feedforward, scheduler=scheduler)
    # Identify only the seven cross-block CZs produced by the CX transpilation.
    positions = {
        tuple(graph.coordinates[node][:2]): (block, qubit)
        for block, nodes in enumerate((data_inputs, magic_inputs))
        for qubit, node in enumerate(nodes)
    }
    cnot_edges = []
    for edge in graph.edges:
        pair = [positions.get(tuple(graph.coordinates[node][:2])) for node in edge]
        if pair[0] is not None and pair[1] is not None and pair[0][0] != pair[1][0] and pair[0][1] == pair[1][1]:
            cnot_edges.append(edge)
    return SteanePattern(
        pattern=pattern,
        source=source,
        data_nodes=data,
        data_input_nodes=data_inputs,
        magic_input_nodes=magic_inputs,
        before_nodes=tuple(imported.record_to_node[record] for record in source.before_records),
        magic_before_nodes=tuple(imported.record_to_node[record] for record in source.magic_records),
        after_nodes=after_nodes,
        correction_nodes=correction_nodes,
        cnot_edges=tuple(cnot_edges),
        consumed_nodes=consumed,
        control_node=control,
        readout_axis=readout_axis,
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
        The teleportation bit, compiled detector values, and logical readout.

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
        if fault is not None and isinstance(command, M) and command.node == steane.data_nodes[fault[0]]:
            qubit, axis = fault
            operator = ca.to_matrix({Axis.X: ca.X, Axis.Y: ca.Y, Axis.Z: ca.Z}[axis])
            simulator.state.evolve(operator, simulator.node_indices.index(steane.data_nodes[qubit]))
        simulator.apply_cmd(command, rng=rng)
    records = simulator.results
    frame = steane.pattern.frame
    detectors = tuple(bool(sum(records[node] for node in group) % 2) for group in frame.parity_check_group)
    observable = bool(sum(records[node] for node in frame.logical_observables[0]) % 2)
    return SteaneShot(records[steane.control_node], detectors, observable)
