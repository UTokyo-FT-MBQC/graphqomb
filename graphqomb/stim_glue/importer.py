"""Import supported Stim circuits into GraphQOMB patterns."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import combinations, pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import stim

from graphqomb.circuit import Circuit, CircuitScheduleStrategy, circuit2graph
from graphqomb.common import Axis, AxisMeasBasis, Sign
from graphqomb.gates import CZ, J
from graphqomb.graphstate import GraphState, compose, odd_neighbors
from graphqomb.qeccode import StabilizerGraphStateBuildResult, YFoliation, build_graph_state
from graphqomb.qompiler import qompile
from graphqomb.stim_glue._parse import (
    DIRECT_MEASUREMENT_AXES,
    MEASURE_RESET_AXES,
    PAIR_MEASUREMENT_AXES,
    RESET_AXES,
    PauliSupport,
    StimMppExtraction,
    collect_record_annotations,
    extract_qubit_coordinates,
    iter_instructions,
    mpp_targets_to_signed_products,
    pauli_products_commute,
    plain_qubit_target,
    record_targets_to_absolute_indices,
    stim_mpp_extraction_from_records,
)
from graphqomb.stim_glue.transpiler import STIM_GATE_J_ANGLES, UnsupportedInstructionError, transpile

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from collections.abc import Set as AbstractSet

    from graphqomb.pattern import Pattern
    from graphqomb.stim_glue._parse import RecordAnnotations


_PAULI_PRODUCT_MEASUREMENT_GATES = frozenset({"MPP", *PAIR_MEASUREMENT_AXES})
_FEEDBACK_AXES = {
    ("CX", 0): Axis.X,
    ("CY", 0): Axis.Y,
    ("CZ", 0): Axis.Z,
    ("CZ", 1): Axis.Z,
    ("XCZ", 1): Axis.X,
    ("YCZ", 1): Axis.Y,
}
_CONTROLLED_PAULI_GATES = frozenset(name for name, _position in _FEEDBACK_AXES)


@dataclass(frozen=True)
class StimImportResult:
    r"""Result of importing a supported Stim circuit.

    ``stim_to_qubit`` maps each Stim qubit id to the qubit index of its
    initial (input-side) wire. When a Stim qubit is used again after a direct
    single-qubit measurement or re-initialized by a mid-circuit reset, the
    importer issues a fresh internal qubit index for the new wire;
    ``qubit_to_stim`` maps every internal qubit index, including those
    continuation indices, back to its Stim qubit id. That mapping is
    many-to-one, so ``stim_to_final_qubit`` names the index of the last
    lifetime of each Stim qubit; the earlier indices of a reset qubit are its
    discarded history, whose wires may remain pattern outputs.

    A mid-circuit reset also issues a fresh *Stim* qubit id for the reset
    lifetime, and that id is what ``mpp_extractions`` reports in ``supports``,
    ``stim_to_column``, and ``column_to_stim``. ``wire_to_stim`` maps every
    such id back to the original circuit's Stim qubit id; it is the identity
    on qubits that are never reset mid-circuit.

    Attributes
    ----------
    pattern : Pattern
        Imported measurement pattern.
    stim_to_qubit : `dict`\[`int`, `int`\]
        Original Stim qubit id to the internal qubit index of its initial wire.
    qubit_to_stim : `dict`\[`int`, `int`\]
        Every internal qubit index to its original Stim qubit id.
    mpp_extractions : `tuple`\[`StimMppExtraction`, ...\]
        Stabilizer data of each imported MPP block, keyed by wire id.
    wire_to_stim : `dict`\[`int`, `int`\]
        Wire (post-split Stim) id to original Stim qubit id.
    stim_to_final_qubit : `dict`\[`int`, `int`\]
        Original Stim qubit id to the internal qubit index of its last
        lifetime.
    """

    pattern: Pattern
    stim_to_qubit: dict[int, int]
    qubit_to_stim: dict[int, int]
    mpp_extractions: tuple[StimMppExtraction, ...]
    wire_to_stim: dict[int, int]
    stim_to_final_qubit: dict[int, int]


@dataclass(frozen=True)
class _Fragment:
    graph: GraphState
    xflow: dict[int, set[int]]
    record_nodes: dict[int, int]
    feedback_targets: tuple[_FeedbackTarget, ...] = ()
    mpp_extractions: tuple[StimMppExtraction, ...] = ()
    zflow: dict[int, set[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class _ImportContext:
    stim_to_qubit: Mapping[int, int]
    coordinate_by_stim_id: Mapping[int, tuple[float, ...]]
    input_initialization_axes: Mapping[int, Axis]
    detector_record_indices: Sequence[frozenset[int]]
    detector_tags: Sequence[str]
    logical_observable_record_indices: Mapping[int, frozenset[int]]
    schedule_strategy: CircuitScheduleStrategy
    y_foliation: YFoliation
    split_wire_ids: frozenset[int]


@dataclass(frozen=True)
class _IdealizedCircuit:
    circuit: stim.Circuit
    zero_record_indices: frozenset[int]


@dataclass(frozen=True)
class _ResetLifetimeSplit:
    circuit: stim.Circuit
    split_to_original: dict[int, int]


@dataclass(frozen=True)
class _AnalyzedInstruction:
    instruction: stim.CircuitInstruction
    record_indices: tuple[int, ...]
    qubit_ids: frozenset[int]
    feedback_corrections: tuple[_FeedbackCorrection, ...]


@dataclass(frozen=True)
class _FeedbackCorrection:
    record_index: int
    stim_id: int
    axis: Axis


@dataclass(frozen=True)
class _FeedbackTarget:
    record_index: int
    node: int
    axis: Axis
    past_neighbors: frozenset[int] = frozenset()


@dataclass(frozen=True)
class _DirectMeasurement:
    stim_id: int
    record_index: int
    axis: Axis
    sign: Sign
    # True for measure-reset gates: the wire re-prepares the positive
    # eigenstate of `axis` unconditionally instead of keeping the outcome.
    resets: bool = False


@dataclass(frozen=True)
class _CircuitAnalysis:
    blocks: tuple[tuple[_AnalyzedInstruction, ...], ...]
    measurement_count: int
    input_initialization_axes: dict[int, Axis]
    direct_measurements: tuple[_DirectMeasurement, ...]


@dataclass(frozen=True)
class _FragmentBuildResult:
    fragments: list[_Fragment]
    stim_to_qubit: dict[int, int]
    qubit_to_stim: dict[int, int]


def stim_file_to_pattern(
    path: str | Path,
    *,
    coord_dims: int = 2,
    schedule_strategy: CircuitScheduleStrategy = CircuitScheduleStrategy.PARALLEL,
    y_foliation: YFoliation = YFoliation.TYPE_I,
) -> StimImportResult:
    """Import a supported Stim file into a GraphQOMB pattern.

    Returns
    -------
    `StimImportResult`
        Imported pattern, qubit mapping, and MPP extraction metadata.
    """
    return stim_text_to_pattern(
        Path(path).read_text(encoding="utf-8"),
        coord_dims=coord_dims,
        schedule_strategy=schedule_strategy,
        y_foliation=y_foliation,
    )


def stim_text_to_pattern(
    text: str,
    *,
    coord_dims: int = 2,
    schedule_strategy: CircuitScheduleStrategy = CircuitScheduleStrategy.PARALLEL,
    y_foliation: YFoliation = YFoliation.TYPE_I,
) -> StimImportResult:
    """Import supported Stim text into a GraphQOMB pattern.

    Returns
    -------
    `StimImportResult`
        Imported pattern, qubit mapping, and MPP extraction metadata.
    """
    return stim_circuit_to_pattern(
        stim.Circuit(text),
        coord_dims=coord_dims,
        schedule_strategy=schedule_strategy,
        y_foliation=y_foliation,
    )


def stim_circuit_to_pattern(  # ruff:ignore[too-many-locals]
    circuit: stim.Circuit,
    *,
    coord_dims: int = 2,
    schedule_strategy: CircuitScheduleStrategy = CircuitScheduleStrategy.PARALLEL,
    y_foliation: YFoliation = YFoliation.TYPE_I,
) -> StimImportResult:
    """Import a supported Stim circuit into a GraphQOMB pattern.

    The importer supports initial Pauli resets, Clifford unitary blocks, and
    Pauli measurement blocks. Stim ``R``, ``RX``, and ``RY`` instructions are
    imported as positive Z-, X-, and Y-eigenstate input initialization,
    respectively, when they occur before any other quantum operation on the
    target qubit. Adjacent Clifford gates are folded into those initial Pauli
    states when possible. Stim noise instructions and measurement-error
    probabilities are omitted because circuit-level noise is outside the
    GraphQOMB import model. Measurement-record-controlled Pauli gates are
    imported as X/Z correction-flow entries at their circuit positions; an X or
    Y feedback additionally contributes Z corrections on the neighbors its
    target becomes entangled with after the feedback position, so deferred
    frame-based application matches the circuit-position semantics.

    Unitary gates may share a TICK block with Pauli measurements. Single-qubit
    Cliffords ahead of a measurement are folded into the measurement basis
    when the measured qubit is never used again; any remaining mixed block is
    split into internal unitary, measurement, and feedback layers that
    preserve the original instruction order, each advancing the z coordinate
    by one layer.

    A Stim qubit may be used again after a direct single-qubit measurement.
    The importer issues a fresh internal qubit index for the post-measurement
    wire: a new node is placed at the qubit's XY coordinates on the next z
    layer, initialized in the positive eigenstate of the measurement axis, and
    the measurement outcome conditions the continuation through the pattern's
    correction flows. Measure-reset gates (``MR``, ``MRX``, ``MRY``) work the
    same way except that the reset re-prepares the positive eigenstate
    unconditionally, so the continuation carries no outcome correction. Reuse
    after an inverted plain measurement target (``M !q``) is not supported
    because negative-eigenstate initialization cannot be represented; an
    inverted measure-reset target may be reused since the reset discards the
    outcome. Reuse is decided on the normalized circuit: single-qubit
    Cliffords after a measurement that cancel to the identity during per-block
    optimization do not count as later use, so the measured wire ends and the
    pattern has one output fewer than the raw instruction stream suggests.

    A qubit may also be re-initialized with an explicit mid-circuit ``R``,
    ``RX``, or ``RY``. The importer splits each reset-started lifetime onto a
    fresh internal qubit index: the new wire starts at the qubit's XY
    coordinates on the next z layer in the positive eigenstate of the reset
    axis, with no outcome conditioning because the reset discards the previous
    state. A wire that was not consumed by a direct measurement before the
    reset remains a pattern output; leaving it coherent and untouched is
    channel-equivalent to Stim's trace-out reset on every recorded
    measurement, provided that discarded output is traced out rather than read.
    ``qubit_to_stim`` maps the fresh indices back to the original Stim qubit
    id, and ``stim_to_final_qubit`` selects the live one among them.

    ``DETECTOR`` instruction tags are preserved: each imported detector's tag
    is carried into ``pattern.pauli_frame.parity_check_tags`` (aligned with
    ``parity_check_group``) and re-emitted by
    `graphqomb.stim_glue.compiler.stim_compile`, so post-selection flag
    detectors (``DETECTOR[type=flag]``) survive the import/export round trip.

    Each reset lifetime also gets its own Stim qubit id internally, and the
    ``mpp_extractions`` metadata (``supports``, ``stim_to_column``, and
    ``column_to_stim``) reports the lifetime that an ``MPP`` product actually
    measured, so a qubit measured before and after a reset occupies a distinct
    column in each block. ``wire_to_stim`` maps those ids back to the original
    circuit ids.

    Returns
    -------
    `StimImportResult`
        Imported pattern, qubit mapping, and MPP extraction metadata.

    Raises
    ------
    ValueError
        If the circuit uses unsupported instructions or invalid coordinates.
    """
    if coord_dims not in {2, 3}:
        msg = "coord_dims must be 2 or 3."
        raise ValueError(msg)

    idealized = _idealize_circuit(circuit.flattened())
    reset_split = _split_reused_reset_wires(idealized.circuit)
    normalized_circuit, stim_ids = _normalize_import_circuit(reset_split.circuit)
    analysis = _CircuitAnalyzer().analyze(normalized_circuit)
    annotations = collect_record_annotations(normalized_circuit)
    if analysis.measurement_count == 0 and (annotations.detectors or annotations.logical_observables):
        msg = "DETECTOR and OBSERVABLE_INCLUDE require at least one imported measurement instruction."
        raise ValueError(msg)
    coordinate_by_stim_id = _coordinates_with_split_wires(
        idealized.circuit,
        reset_split.split_to_original,
        coord_dims=coord_dims,
    )
    _warn_on_partial_coordinate_coverage(stim_ids, coordinate_by_stim_id)
    stim_to_qubit = {stim_id: qubit for qubit, stim_id in enumerate(sorted(stim_ids))}
    context = _ImportContext(
        stim_to_qubit=stim_to_qubit,
        coordinate_by_stim_id=coordinate_by_stim_id,
        input_initialization_axes=analysis.input_initialization_axes,
        detector_record_indices=annotations.detectors,
        detector_tags=annotations.detector_tags,
        logical_observable_record_indices=annotations.logical_observables,
        schedule_strategy=schedule_strategy,
        y_foliation=y_foliation,
        split_wire_ids=frozenset(reset_split.split_to_original),
    )

    build = _fragments_from_blocks(analysis.blocks, context=context)
    fragment = _compose_fragments(build.fragments)
    fragment = _apply_single_measurements(
        fragment,
        analysis.direct_measurements,
        final_stim_to_qubit=build.stim_to_qubit,
    )
    parity_check_groups, parity_check_tags, logical_observables = _measurement_annotations(
        annotations,
        record_nodes=fragment.record_nodes,
        zero_record_indices=idealized.zero_record_indices,
    )
    pattern = qompile(
        fragment.graph,
        *_flows_with_feedback(fragment, zero_record_indices=idealized.zero_record_indices),
        parity_check_group=parity_check_groups,
        logical_observables=logical_observables,
        parity_check_tags=parity_check_tags,
    )
    return StimImportResult(
        pattern=pattern,
        stim_to_qubit={
            stim_id: qubit for stim_id, qubit in stim_to_qubit.items() if stim_id not in reset_split.split_to_original
        },
        qubit_to_stim={
            qubit: reset_split.split_to_original.get(stim_id, stim_id) for qubit, stim_id in build.qubit_to_stim.items()
        },
        mpp_extractions=fragment.mpp_extractions,
        wire_to_stim={wire_id: reset_split.split_to_original.get(wire_id, wire_id) for wire_id in sorted(stim_ids)},
        stim_to_final_qubit=_final_qubit_by_stim_id(build.stim_to_qubit, reset_split.split_to_original),
    )


def _final_qubit_by_stim_id(
    final_stim_to_qubit: Mapping[int, int],
    split_to_original: Mapping[int, int],
) -> dict[int, int]:
    r"""Return the internal qubit index of each Stim qubit's last lifetime.

    Returns
    -------
    `dict`\[`int`, `int`\]
        Original Stim qubit id to internal qubit index.

    Raises
    ------
    RuntimeError
        If a split wire id is not above its original Stim qubit id.
    """
    # `_split_reused_reset_wires` issues wire ids above every original id and
    # in circuit order, so the largest wire id of a Stim qubit is its last
    # reset lifetime; `final_stim_to_qubit` already resolves the
    # measurement-reuse continuations within one wire.
    if any(wire_id <= original_id for wire_id, original_id in split_to_original.items()):
        msg = "Split wire ids must be issued above their original Stim qubit ids; this is a bug."
        raise RuntimeError(msg)
    return {split_to_original.get(wire_id, wire_id): qubit for wire_id, qubit in sorted(final_stim_to_qubit.items())}


def _idealize_circuit(circuit: stim.Circuit) -> _IdealizedCircuit:
    result = stim.Circuit()
    zero_record_indices: set[int] = set()
    measurement_offset = 0

    for instruction in iter_instructions(circuit):
        gate_data = stim.gate_data(instruction.name)
        if instruction.name in DIRECT_MEASUREMENT_AXES:
            result.append(instruction.name, instruction.targets_copy())
        elif instruction.name in _PAULI_PRODUCT_MEASUREMENT_GATES:
            _append_ideal_pauli_measurements(result, instruction)
        elif instruction.name == "MPAD":
            targets = instruction.targets_copy()
            if any(int(target.value) != 0 for target in targets):
                msg = "MPAD 1 records are not supported because detector parity offsets are not represented."
                raise ValueError(msg)
            result.append("MPAD", targets)
            zero_record_indices.update(range(measurement_offset, measurement_offset + instruction.num_measurements))
        elif gate_data.is_noisy_gate and not gate_data.is_reset:
            if gate_data.produces_measurements:
                result.append("MPAD", [0] * instruction.num_measurements)
                zero_record_indices.update(range(measurement_offset, measurement_offset + instruction.num_measurements))
        else:
            result.append(instruction)

        measurement_offset += instruction.num_measurements

    return _IdealizedCircuit(result, frozenset(zero_record_indices))


def _tracked_qubits(instruction: stim.CircuitInstruction) -> set[int]:
    # Narrower than _parse.ANNOTATION_GATES on purpose: QUBIT_COORDS targets
    # must stay tracked so coordinate-only qubits become wires, while MPAD pad
    # bits are spelled like qubit targets but are not qubits.
    if instruction.name in {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "MPAD"}:
        return set()
    return {int(target.qubit_value) for target in instruction.targets_copy() if target.qubit_value is not None}


def _split_reused_reset_wires(circuit: stim.Circuit) -> _ResetLifetimeSplit:
    """Give every reset-started wire lifetime its own Stim qubit id.

    A ``R``/``RX``/``RY`` instruction targeting a qubit that already had a
    quantum operation starts a new wire: the reset and all later instructions
    on that qubit are remapped to a fresh Stim qubit id, so downstream import
    stages see initial resets only. Resets repeated before any quantum use
    keep their wire id, matching the leading-reset rule where the last reset
    determines the initialization axis.

    Returns
    -------
    `_ResetLifetimeSplit`
        Rewritten circuit and the mapping from fresh ids to original ids.
    """
    result = stim.Circuit()
    next_wire = circuit.num_qubits
    wire_of: dict[int, int] = {}
    used_wires: set[int] = set()
    split_to_original: dict[int, int] = {}

    for instruction in iter_instructions(circuit):
        name = instruction.name
        if name in {"QUBIT_COORDS", "MPAD"}:
            # Coordinates stay attached to original ids; MPAD pad bits are
            # spelled like qubit targets and must not be remapped.
            result.append(instruction)
            continue
        if name in RESET_AXES:
            wires: list[int] = []
            for target in instruction.targets_copy():
                stim_id = plain_qubit_target(target, name)
                wire = wire_of.get(stim_id, stim_id)
                if wire in used_wires:
                    wire = next_wire
                    next_wire += 1
                    split_to_original[wire] = stim_id
                    wire_of[stim_id] = wire
                wires.append(wire)
            result.append(name, wires, instruction.gate_args_copy(), tag=instruction.tag)
            continue
        result.append(
            name,
            [_remap_qubit_target(target, wire_of) for target in instruction.targets_copy()],
            instruction.gate_args_copy(),
            tag=instruction.tag,
        )
        if _is_quantum_operation(instruction):
            used_wires.update(wire_of.get(stim_id, stim_id) for stim_id in _tracked_qubits(instruction))

    return _ResetLifetimeSplit(result, split_to_original)


def _coordinates_with_split_wires(
    circuit: stim.Circuit,
    split_to_original: Mapping[int, int],
    *,
    coord_dims: int,
) -> dict[int, tuple[float, ...]]:
    r"""Extract qubit coordinates and share each original's XY with its split wires.

    Returns
    -------
    `dict`\[`int`, `tuple`\[`float`, ...\]\]
        Coordinates keyed by Stim qubit id, including split wire ids.
    """
    coordinates = extract_qubit_coordinates(circuit, coord_dims=coord_dims)
    for split_id, original_id in split_to_original.items():
        coord = coordinates.get(original_id)
        if coord is not None:
            coordinates[split_id] = coord
    return coordinates


def _warn_on_partial_coordinate_coverage(
    stim_ids: frozenset[int],
    coordinate_by_stim_id: Mapping[int, tuple[float, ...]],
) -> None:
    """Warn when only some imported wires carry ``QUBIT_COORDS``.

    Fully coordinated and fully uncoordinated circuits are both intentional
    inputs; a mixture usually means coordinate metadata was lost upstream, for
    example on reset lifetimes that were split onto fresh qubit ids by another
    tool, which the importer cannot map back to an original qubit.
    """
    uncoordinated = stim_ids - coordinate_by_stim_id.keys()
    if uncoordinated and uncoordinated != stim_ids:
        shown = sorted(uncoordinated)[:10]
        listing = ", ".join(str(wire) for wire in shown) + (", ..." if len(uncoordinated) > len(shown) else "")
        msg = (
            f"{len(uncoordinated)} of {len(stim_ids)} imported wires have no QUBIT_COORDS (wire ids {listing}); "
            "their pattern nodes get no coordinates. Reset lifetimes split onto fresh qubit ids outside the "
            "importer cannot inherit the original qubit's coordinates."
        )
        warnings.warn(msg, stacklevel=3)


def _remap_qubit_target(target: stim.GateTarget, wire_of: Mapping[int, int]) -> stim.GateTarget | int:
    """Map one Stim gate target onto its current wire id.

    Returns
    -------
    ``stim.GateTarget`` | `int`
        Remapped target, or the unchanged non-qubit target.
    """
    qubit_value = target.qubit_value
    if qubit_value is None:
        return target
    wire = wire_of.get(int(qubit_value), int(qubit_value))
    if target.pauli_type == "X":
        return stim.target_x(wire, invert=target.is_inverted_result_target)
    if target.pauli_type == "Y":
        return stim.target_y(wire, invert=target.is_inverted_result_target)
    if target.pauli_type == "Z":
        return stim.target_z(wire, invert=target.is_inverted_result_target)
    if target.is_inverted_result_target:
        return stim.target_inv(wire)
    return wire


@dataclass
class _TickBlock:
    """One TICK block accumulated during import normalization."""

    circuit: stim.Circuit = field(default_factory=stim.Circuit)
    last_index: int = -1
    has_unitary: bool = False
    has_single_measurement: bool = False
    has_measure_reset: bool = False
    has_mpp: bool = False
    has_feedback: bool = False
    measured_stim_ids: set[int] = field(default_factory=set)

    @property
    def has_measurement(self) -> bool:
        """Whether the block contains a single-qubit measurement or an MPP."""
        return self.has_single_measurement or self.has_mpp

    def add(self, instruction: stim.CircuitInstruction, index: int, instruction_qubits: set[int]) -> None:
        """Append one instruction and update the block's feature flags."""
        self.circuit.append(instruction)
        self.last_index = index
        self.has_unitary |= _is_unitary_instruction(instruction)
        self.has_measure_reset |= instruction.name in MEASURE_RESET_AXES
        self.has_mpp |= instruction.name == "MPP"
        self.has_feedback |= _is_feedback_instruction(instruction)
        if instruction.name in DIRECT_MEASUREMENT_AXES:
            self.has_single_measurement = True
            self.measured_stim_ids.update(instruction_qubits)


def _last_quantum_use_indices(instructions: Sequence[stim.CircuitInstruction]) -> dict[int, int]:
    r"""Return the last instruction index at which each Stim qubit is quantum-used.

    Returns
    -------
    `dict`\[`int`, `int`\]
        Last quantum-operation instruction index keyed by Stim qubit id.
    """
    last_quantum_use: dict[int, int] = {}
    for index, instruction in enumerate(instructions):
        if _is_quantum_operation(instruction):
            for stim_id in _tracked_qubits(instruction):
                last_quantum_use[stim_id] = index
    return last_quantum_use


def _normalize_import_circuit(circuit: stim.Circuit) -> tuple[stim.Circuit, frozenset[int]]:
    instructions = list(iter_instructions(circuit))
    last_quantum_use = _last_quantum_use_indices(instructions)
    result = stim.Circuit()
    block = _TickBlock()
    stim_ids: set[int] = set()
    block_number = 1

    def flush_block() -> None:
        nonlocal block
        parts = _normalize_block(block, last_quantum_use=last_quantum_use, block_number=block_number)
        for part_index, part in enumerate(parts):
            if part_index:
                result.append("TICK", [])
            result.append(part)
        block = _TickBlock()

    # `_split_reused_reset_wires` runs first, so every reset here is the
    # initial reset of its own wire id.
    for index, instruction in enumerate(instructions):
        instruction_qubits = _tracked_qubits(instruction)
        stim_ids.update(instruction_qubits)
        if instruction.name == "TICK":
            flush_block()
            result.append(instruction)
            block_number += 1
        else:
            block.add(instruction, index, instruction_qubits)

    flush_block()
    return result, frozenset(stim_ids)


def _normalize_block(
    block: _TickBlock,
    *,
    last_quantum_use: Mapping[int, int],
    block_number: int,
) -> list[stim.Circuit]:
    r"""Normalize one TICK block into unitary-only, measurement-only, and feedback-only parts.

    Returns
    -------
    `list`\[``stim.Circuit``\]
        Block parts to be emitted in order, separated by internal TICKs.
    """
    if not block.has_unitary and not block.has_feedback:
        if block.has_measurement or block.has_measure_reset:
            return _split_mixed_block(block.circuit)
        return [block.circuit]
    if not (block.has_measurement or block.has_measure_reset or block.has_feedback):
        return [_transpile_tick_block(block.circuit, block_number=block_number)]
    if block.has_mpp or block.has_measure_reset or block.has_feedback:
        # MPP, measure-reset, and record-controlled instructions cannot pass
        # through the J/CZ optimizer, so split first and transpile only the
        # purely unitary parts.
        return [
            _transpile_tick_block(part, block_number=block_number)
            if any(
                isinstance(instruction, stim.CircuitInstruction) and _is_unitary_instruction(instruction)
                for instruction in part
            )
            else part
            for part in _split_mixed_block(block.circuit)
        ]
    # A measurement on a qubit that is quantum-used again later must keep its
    # post-measurement state, so its basis is never folded away.
    preserved_qubits = {
        stim_id for stim_id in block.measured_stim_ids if last_quantum_use.get(stim_id, -1) > block.last_index
    }
    optimized = _transpile_tick_block(
        block.circuit,
        block_number=block_number,
        preserved_measurement_qubits=preserved_qubits,
    )
    return _split_mixed_block(optimized)


def _transpile_tick_block(
    block: stim.Circuit,
    *,
    block_number: int,
    preserved_measurement_qubits: AbstractSet[int] = frozenset(),
) -> stim.Circuit:
    try:
        return transpile(
            block,
            optimize=True,
            preserved_measurement_qubits=preserved_measurement_qubits,
        )
    except UnsupportedInstructionError as ex:
        msg = f"Stim unitary TICK block {block_number} failed to transpile: {ex}"
        raise UnsupportedInstructionError(msg) from ex


def _is_quantum_operation(instruction: stim.CircuitInstruction) -> bool:
    return (
        _is_unitary_instruction(instruction)
        or _is_feedback_instruction(instruction)
        or instruction.name in RESET_AXES
        or instruction.name in DIRECT_MEASUREMENT_AXES
        or instruction.name == "MPP"
    )


def _split_mixed_block(  # ruff:ignore[complex-structure, too-many-branches, too-many-locals]
    block: stim.Circuit,
) -> list[stim.Circuit]:
    r"""Split one TICK block into unitary-only, measurement-only, and feedback-only parts.

    Instructions are greedily placed into the earliest compatible part.
    Per-qubit instruction order is preserved: an instruction never lands in an
    earlier part than a previous touch of its qubit, and a single-qubit
    measurement is pushed strictly past a preceding single-qubit measurement
    of the same qubit. (A preceding MPP touch may share the measurement part;
    fragment construction still applies the direct measurement after the MPP.)
    Record-producing and record-consuming instructions (including
    record-controlled feedback) keep their relative order so ``rec[-k]``
    targets stay valid.

    Returns
    -------
    `list`\[``stim.Circuit``\]
        Block parts to be emitted in order, separated by internal TICKs.
    """
    part_kinds: list[str] = []
    part_instructions: list[list[stim.CircuitInstruction]] = []
    last_part_of_qubit: dict[int, int] = {}
    last_touch_is_single_measurement: dict[int, bool] = {}
    record_floor = 0

    for instruction in _atomized_block_instructions(block):
        name = instruction.name
        is_measurement = name == "MPP" or name in DIRECT_MEASUREMENT_AXES
        is_feedback = _is_feedback_instruction(instruction)
        is_quantum = _is_quantum_operation(instruction)
        if is_measurement:
            kind = "measurement"
            allowed_kinds: tuple[str, ...] = ("measurement",)
        elif is_feedback:
            kind = "feedback"
            allowed_kinds = ("feedback",)
        elif _is_unitary_instruction(instruction):
            kind = "unitary"
            allowed_kinds = ("unitary",)
        else:
            kind = "unitary"
            allowed_kinds = ("unitary", "measurement", "feedback")
        is_record_ordered = (
            instruction.num_measurements > 0
            or is_feedback
            or name in {"DETECTOR", "MPAD", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"}
        )

        lower = record_floor if is_record_ordered else 0
        if is_quantum:
            for qubit in _tracked_qubits(instruction):
                last_part = last_part_of_qubit.get(qubit)
                if last_part is None:
                    continue
                bump = 1 if is_measurement and last_touch_is_single_measurement[qubit] else 0
                lower = max(lower, last_part + bump)

        part_index = lower
        while part_index < len(part_kinds) and part_kinds[part_index] not in allowed_kinds:
            part_index += 1
        if part_index == len(part_kinds):
            part_kinds.append(kind)
            part_instructions.append([])
        part_instructions[part_index].append(instruction)

        if is_quantum:
            for qubit in _tracked_qubits(instruction):
                last_part_of_qubit[qubit] = part_index
                last_touch_is_single_measurement[qubit] = name in DIRECT_MEASUREMENT_AXES
        if is_record_ordered:
            record_floor = part_index

    parts: list[stim.Circuit] = []
    for instructions in part_instructions:
        part = stim.Circuit()
        for instruction in instructions:
            part.append(instruction)
        parts.append(part)
    return parts


def _defused_feedback_instructions(instruction: stim.CircuitInstruction) -> Iterator[stim.CircuitInstruction]:
    """Yield a controlled-Pauli instruction split per target group when it carries feedback.

    Stim fuses adjacent same-name instructions, so ``CZ rec[-1] 0`` next to a
    plain ``CZ 0 1`` arrives as one instruction mixing record-controlled and
    unitary target groups. Splitting per group lets the feedback groups take
    the feedback import path while the plain groups stay unitary gates.

    Yields
    ------
    ``stim.CircuitInstruction``
        The original instruction, or one instruction per target group.
    """
    if instruction.name in _CONTROLLED_PAULI_GATES and _is_feedback_instruction(instruction):
        groups = instruction.target_groups()
        if len(groups) > 1:
            gate_args = instruction.gate_args_copy()
            for group in groups:
                yield stim.CircuitInstruction(instruction.name, list(group), gate_args, tag=instruction.tag)
            return
    yield instruction


def _atomized_block_instructions(block: stim.Circuit) -> Iterator[stim.CircuitInstruction]:
    """Yield block instructions split into independently placeable units.

    Stim fuses adjacent same-name instructions, so a repeated measurement of
    one qubit can arrive as a single multi-target instruction (each target is
    one measurement record and must be placed independently), and a
    record-controlled ``CZ rec[-1] 0`` next to a plain ``CZ 0 1`` arrives as
    one instruction mixing feedback and unitary target groups. The split
    places each unit in its own part, and parts are emitted TICK-separated,
    so the units are not re-fused downstream.

    Yields
    ------
    ``stim.CircuitInstruction``
        Original instructions, with single-qubit measurement instructions
        emitted once per target and feedback-carrying controlled-Pauli
        instructions emitted once per target group.
    """
    for instruction in iter_instructions(block):
        if instruction.name in DIRECT_MEASUREMENT_AXES and instruction.num_measurements > 1:
            gate_args = instruction.gate_args_copy()
            for target in instruction.targets_copy():
                yield stim.CircuitInstruction(instruction.name, [target], gate_args, tag=instruction.tag)
        else:
            yield from _defused_feedback_instructions(instruction)


class _CircuitAnalyzer:
    """Mutable state for the single-pass Stim circuit analysis.

    ``DETECTOR`` and ``OBSERVABLE_INCLUDE`` annotations are resolved
    separately by `graphqomb.stim_glue._parse.collect_record_annotations`.
    """

    def __init__(self) -> None:
        self.blocks: list[tuple[_AnalyzedInstruction, ...]] = []
        self.current_block: list[_AnalyzedInstruction] = []
        self.measurement_count = 0
        self.input_initialization_axes: dict[int, Axis] = {}
        self.direct_measurements: list[_DirectMeasurement] = []
        self.block_has_unitary = False
        self.block_has_pauli_measurement = False

    def analyze(self, circuit: stim.Circuit) -> _CircuitAnalysis:
        for instruction in iter_instructions(circuit):
            self._process_instruction(instruction)
            self.measurement_count += instruction.num_measurements
        self._finish_block()
        return self._result()

    def _process_instruction(self, instruction: stim.CircuitInstruction) -> None:
        if instruction.name == "TICK":
            self._finish_block()
        # Not _parse.ANNOTATION_GATES: TICK is the block boundary handled
        # above, and MPAD still advances measurement_count in analyze() even
        # though it becomes no operation (idealization records its zero bits).
        elif instruction.name not in {"QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "MPAD"}:
            self._record_operation(instruction, _tracked_qubits(instruction))

    def _finish_block(self) -> None:
        self.blocks.append(tuple(self.current_block))
        self.current_block = []
        self.block_has_unitary = False
        self.block_has_pauli_measurement = False

    def _record_operation(self, instruction: stim.CircuitInstruction, instruction_qubits: set[int]) -> None:
        _validate_supported_instruction(instruction)
        record_indices = tuple(range(self.measurement_count, self.measurement_count + instruction.num_measurements))
        feedback_corrections = _feedback_corrections_from_instruction(
            instruction,
            measurement_count=self.measurement_count,
        )
        analyzed = _AnalyzedInstruction(
            instruction,
            record_indices,
            frozenset(instruction_qubits),
            feedback_corrections,
        )
        self.current_block.append(analyzed)

        is_unitary = _is_unitary_instruction(instruction)
        is_pauli_measurement = instruction.name == "MPP" or instruction.name in DIRECT_MEASUREMENT_AXES
        self._validate_block_separation(
            is_unitary=is_unitary,
            is_pauli_measurement=is_pauli_measurement,
        )
        reset_axis = RESET_AXES.get(instruction.name)
        if reset_axis is not None:
            self.input_initialization_axes.update(dict.fromkeys(instruction_qubits, reset_axis))

        if instruction.name in DIRECT_MEASUREMENT_AXES:
            self.direct_measurements.extend(_direct_measurements_from_instruction(analyzed))

    def _validate_block_separation(self, *, is_unitary: bool, is_pauli_measurement: bool) -> None:
        self.block_has_unitary |= is_unitary
        self.block_has_pauli_measurement |= is_pauli_measurement
        if self.block_has_unitary and self.block_has_pauli_measurement:
            msg = "Pauli measurement instructions must be separated from unitary gate instructions by TICK."
            raise ValueError(msg)

    def _result(self) -> _CircuitAnalysis:
        return _CircuitAnalysis(
            blocks=tuple(self.blocks),
            measurement_count=self.measurement_count,
            input_initialization_axes=self.input_initialization_axes,
            direct_measurements=tuple(self.direct_measurements),
        )


def _validate_supported_instruction(instruction: stim.CircuitInstruction) -> None:
    if (
        _is_unitary_instruction(instruction)
        or _is_feedback_instruction(instruction)
        or instruction.name in RESET_AXES
        or instruction.name in DIRECT_MEASUREMENT_AXES
        or instruction.name == "MPP"
    ):
        return
    msg = f"Unsupported Stim instruction(s): {instruction.name}."
    raise ValueError(msg)


def _is_unitary_instruction(instruction: stim.CircuitInstruction) -> bool:
    return bool(stim.gate_data(instruction.name).is_unitary and not _is_feedback_instruction(instruction))


def _is_feedback_instruction(instruction: stim.CircuitInstruction) -> bool:
    return bool(
        stim.gate_data(instruction.name).is_unitary
        and any(target.is_measurement_record_target for target in instruction.targets_copy())
    )


def _feedback_corrections_from_instruction(
    instruction: stim.CircuitInstruction,
    *,
    measurement_count: int,
) -> tuple[_FeedbackCorrection, ...]:
    if not _is_feedback_instruction(instruction):
        return ()

    corrections: list[_FeedbackCorrection] = []
    expected_group_size = 2
    for group in instruction.target_groups():
        record_positions = [index for index, target in enumerate(group) if target.is_measurement_record_target]
        if len(group) != expected_group_size or len(record_positions) != 1:
            msg = (
                f"{instruction.name} feedback groups must contain exactly one measurement record and one qubit target."
            )
            raise ValueError(msg)
        record_position = record_positions[0]
        axis = _FEEDBACK_AXES.get((instruction.name, record_position))
        if axis is None:
            msg = f"{instruction.name} does not support a measurement record target at position {record_position}."
            raise ValueError(msg)

        target_position = 1 - record_position
        stim_id = plain_qubit_target(group[target_position], instruction.name)
        record_indices = record_targets_to_absolute_indices(
            (group[record_position],),
            measurement_count=measurement_count,
            instruction_name=instruction.name,
        )
        corrections.append(_FeedbackCorrection(next(iter(record_indices)), stim_id, axis))
    return tuple(corrections)


def _direct_measurements_from_instruction(
    analyzed: _AnalyzedInstruction,
) -> list[_DirectMeasurement]:
    instruction = analyzed.instruction
    targets = instruction.targets_copy()
    if len(targets) != len(analyzed.record_indices):
        msg = f"{instruction.name} target count does not match its measurement-record count."
        raise ValueError(msg)

    measurements: list[_DirectMeasurement] = []
    seen_qubits: set[int] = set()
    axis = DIRECT_MEASUREMENT_AXES[instruction.name]
    resets = instruction.name in MEASURE_RESET_AXES
    for target, record_index in zip(targets, analyzed.record_indices, strict=True):
        stim_id = plain_qubit_target(target, instruction.name)
        if stim_id in seen_qubits:
            msg = f"{instruction.name} measures qubit {stim_id} more than once in one instruction."
            raise ValueError(msg)
        seen_qubits.add(stim_id)
        sign = Sign.MINUS if target.is_inverted_result_target else Sign.PLUS
        measurements.append(_DirectMeasurement(stim_id, record_index, axis, sign, resets=resets))
    return measurements


def _append_ideal_pauli_measurements(
    circuit: stim.Circuit,
    instruction: stim.CircuitInstruction,
) -> None:
    if instruction.name == "MPP":
        circuit.append("MPP", instruction.targets_copy())
        return

    targets = instruction.targets_copy()
    if any(target.is_inverted_result_target for target in targets):
        msg = f"Signed {instruction.name} products are not supported; inverted targets cannot be imported."
        raise ValueError(msg)

    axis = PAIR_MEASUREMENT_AXES[instruction.name]
    expected_group_size = 2

    target_factory = {Axis.X: stim.target_x, Axis.Y: stim.target_y, Axis.Z: stim.target_z}[axis]
    for group in instruction.target_groups():
        if len(group) != expected_group_size:
            msg = f"{instruction.name} target group must contain {expected_group_size} qubit(s)."
            raise ValueError(msg)
        product_targets: list[stim.GateTarget] = []
        for index, target in enumerate(group):
            if index:
                product_targets.append(stim.target_combiner())
            product_targets.append(
                target_factory(
                    plain_qubit_target(target, instruction.name),
                )
            )
        circuit.append("MPP", product_targets)


def _fragments_from_blocks(  # ruff:ignore[too-many-locals]
    blocks: Sequence[Sequence[_AnalyzedInstruction]],
    *,
    context: _ImportContext,
) -> _FragmentBuildResult:
    fragments = [_identity_fragment(context)]
    z_base = 0
    stim_to_qubit = dict(context.stim_to_qubit)
    qubit_to_stim = {qubit: stim_id for stim_id, qubit in stim_to_qubit.items()}
    live_stim_ids = set(stim_to_qubit) - context.split_wire_ids
    future_use = _stim_ids_used_after_block(blocks)

    for block_index, block in enumerate(blocks):
        new_wire_ids = sorted(
            {
                stim_id
                for analyzed in block
                if analyzed.instruction.name in RESET_AXES
                for stim_id in analyzed.qubit_ids
                if stim_id in context.split_wire_ids and stim_id not in live_stim_ids
            }
        )
        if new_wire_ids:
            z_base += 1
            fragments.append(_new_wire_fragment(new_wire_ids, z=z_base, stim_to_qubit=stim_to_qubit, context=context))
            live_stim_ids.update(new_wire_ids)

        direct_items = tuple(analyzed for analyzed in block if analyzed.instruction.name in DIRECT_MEASUREMENT_AXES)
        directly_measured_stim_ids = {stim_id for analyzed in direct_items for stim_id in analyzed.qubit_ids}
        reused_measurements = tuple(
            measurement
            for analyzed in direct_items
            for measurement in _direct_measurements_from_instruction(analyzed)
            if measurement.stim_id in future_use[block_index]
        )
        unitary_instructions = tuple(
            analyzed.instruction for analyzed in block if _is_unitary_instruction(analyzed.instruction)
        )
        feedback_corrections = tuple(correction for analyzed in block for correction in analyzed.feedback_corrections)
        if unitary_instructions:
            fragment, z_base = _unitary_fragment(
                unitary_instructions,
                live_stim_ids=live_stim_ids,
                z_base=z_base,
                stim_to_qubit=stim_to_qubit,
                qubit_to_stim=qubit_to_stim,
                context=context,
            )
            fragments.append(fragment)
        else:
            mpp_stim_ids = {
                stim_id for analyzed in block if analyzed.instruction.name == "MPP" for stim_id in analyzed.qubit_ids
            }
            mpp_items = tuple(analyzed for analyzed in block if analyzed.instruction.name == "MPP")
            if mpp_items:
                fragments.append(
                    _mpp_fragment(
                        mpp_items,
                        z_base=z_base,
                        io_stim_ids=(live_stim_ids - directly_measured_stim_ids) | mpp_stim_ids,
                        stim_to_qubit=stim_to_qubit,
                        context=context,
                    )
                )
                z_base += 2
                if reused_measurements:
                    z_base += 1
                    fragments.append(
                        _reuse_fragment(
                            reused_measurements,
                            z=z_base,
                            stim_to_qubit=stim_to_qubit,
                            qubit_to_stim=qubit_to_stim,
                            context=context,
                        )
                    )
            elif feedback_corrections:
                fragments.append(
                    _feedback_fragment(
                        feedback_corrections,
                        stim_to_qubit=stim_to_qubit,
                    )
                )
            elif directly_measured_stim_ids:
                continuing_stim_ids = live_stim_ids - directly_measured_stim_ids
                if continuing_stim_ids or reused_measurements:
                    z_base += 1
                    graph = GraphState()
                    _add_relocated_io_nodes(
                        graph,
                        continuing_stim_ids,
                        z=z_base,
                        stim_to_qubit=stim_to_qubit,
                        context=context,
                    )
                    fragments.append(
                        _reuse_fragment(
                            reused_measurements,
                            z=z_base,
                            graph=graph,
                            stim_to_qubit=stim_to_qubit,
                            qubit_to_stim=qubit_to_stim,
                            context=context,
                        )
                    )

        reused_stim_ids = {measurement.stim_id for measurement in reused_measurements}
        live_stim_ids.difference_update(directly_measured_stim_ids - reused_stim_ids)

    return _FragmentBuildResult(
        fragments=fragments,
        stim_to_qubit=stim_to_qubit,
        qubit_to_stim=qubit_to_stim,
    )


def _stim_ids_used_after_block(blocks: Sequence[Sequence[_AnalyzedInstruction]]) -> list[set[int]]:
    future: list[set[int]] = []
    seen: set[int] = set()
    for block in reversed(blocks):
        future.append(set(seen))
        for analyzed in block:
            seen.update(analyzed.qubit_ids)
    future.reverse()
    return future


def _reuse_fragment(  # ruff:ignore[too-many-arguments]
    reused_measurements: Sequence[_DirectMeasurement],
    *,
    z: float,
    stim_to_qubit: dict[int, int],
    qubit_to_stim: dict[int, int],
    context: _ImportContext,
    graph: GraphState | None = None,
) -> _Fragment:
    """Measure reused qubits in place and restart their wires on fresh qubit indices.

    Each reused measurement binds an internal measured node onto the qubit's
    current wire end and opens a continuation wire on a freshly issued qubit
    index. The continuation node reuses the qubit's XY coordinates at ``z``
    and is initialized in the positive eigenstate of the measurement axis.
    After a plain measurement the correction flows condition the continuation
    on the outcome; after a measure-reset gate the re-prepared eigenstate is
    outcome-independent, so no flow entry is added.

    Returns
    -------
    `_Fragment`
        Fragment holding the measured nodes, continuation wires, and flows.

    Raises
    ------
    ValueError
        If a reused plain measurement has an inverted target. (An inverted
        measure-reset target is fine: the reset discards the outcome.)
    """
    if graph is None:
        graph = GraphState()
    xflow: dict[int, set[int]] = {}
    zflow: dict[int, set[int]] = {}
    record_nodes: dict[int, int] = {}

    for measurement in reused_measurements:
        if measurement.sign is Sign.MINUS and not measurement.resets:
            msg = (
                f"Stim qubit {measurement.stim_id} is reused after an inverted single-qubit measurement; "
                "inverted measurement targets are only supported when they end the qubit's lifetime."
            )
            raise ValueError(msg)
        old_qubit = stim_to_qubit[measurement.stim_id]
        new_qubit = len(qubit_to_stim)
        measured_node = graph.add_node()
        graph.register_input(measured_node, old_qubit)
        graph.assign_meas_basis(measured_node, AxisMeasBasis(measurement.axis, measurement.sign))
        coord = context.coordinate_by_stim_id.get(measurement.stim_id)
        continuation_node = graph.add_node(coordinate=_coordinate_at_z(coord, z) if coord is not None else None)
        graph.register_input(continuation_node, new_qubit, init_axis=measurement.axis)
        graph.register_output(continuation_node, new_qubit)
        if measurement.resets:
            # A measure-reset gate re-prepares the positive eigenstate no
            # matter the outcome, so the continuation needs no correction.
            pass
        elif measurement.axis == Axis.Z:
            # The conditional X re-preparing |m> acts before later entangling
            # CZs, so it needs the derived odd-neighbor Z propagation that a
            # regular xflow entry receives.
            xflow[measured_node] = {continuation_node}
        else:
            # X and Y eigenstates are re-prepared by a conditional Z, which
            # commutes with later CZs and stays a bare zflow entry.
            zflow[measured_node] = {continuation_node}
        record_nodes[measurement.record_index] = measured_node
        stim_to_qubit[measurement.stim_id] = new_qubit
        qubit_to_stim[new_qubit] = measurement.stim_id

    return _Fragment(graph=graph, xflow=xflow, record_nodes=record_nodes, zflow=zflow)


def _feedback_fragment(
    corrections: Sequence[_FeedbackCorrection],
    *,
    stim_to_qubit: Mapping[int, int],
) -> _Fragment:
    graph = GraphState()
    node_by_stim_id: dict[int, int] = {}
    for correction in corrections:
        if correction.stim_id in node_by_stim_id:
            continue
        node = graph.add_node()
        qubit = stim_to_qubit[correction.stim_id]
        graph.register_input(node, qubit)
        graph.register_output(node, qubit)
        node_by_stim_id[correction.stim_id] = node

    return _Fragment(
        graph=graph,
        xflow={},
        record_nodes={},
        feedback_targets=tuple(
            _FeedbackTarget(
                record_index=correction.record_index,
                node=node_by_stim_id[correction.stim_id],
                axis=correction.axis,
            )
            for correction in corrections
        ),
    )


def _identity_fragment(context: _ImportContext) -> _Fragment:
    graph = GraphState()
    for stim_id, qubit_index in sorted(context.stim_to_qubit.items()):
        if stim_id in context.split_wire_ids:
            # Split wires start at their reset position, not at the circuit input.
            continue
        coord = context.coordinate_by_stim_id.get(stim_id)
        node = graph.add_node(coordinate=_coordinate_at_z(coord, 0) if coord is not None else None)
        graph.register_input(
            node,
            qubit_index,
            init_axis=context.input_initialization_axes.get(stim_id, Axis.X),
        )
        graph.register_output(node, qubit_index)
    return _Fragment(
        graph=graph,
        xflow={},
        record_nodes={},
    )


def _new_wire_fragment(
    stim_ids: Sequence[int],
    *,
    z: int,
    stim_to_qubit: Mapping[int, int],
    context: _ImportContext,
) -> _Fragment:
    """Open fresh wires for reset-started lifetimes at their reset position.

    Each wire reuses its original qubit's XY coordinates at ``z`` and is
    initialized in the positive eigenstate of its last leading reset axis.
    The reset discards the previous state, so the wire carries no outcome
    conditioning; the replaced wire simply ends (a measured wire ended at its
    measurement, an unmeasured wire remains a pattern output).

    Returns
    -------
    `_Fragment`
        Fragment holding the fresh input/output wires.
    """
    graph = GraphState()
    for stim_id in stim_ids:
        coord = context.coordinate_by_stim_id.get(stim_id)
        node = graph.add_node(coordinate=_coordinate_at_z(coord, z) if coord is not None else None)
        qubit = stim_to_qubit[stim_id]
        graph.register_input(node, qubit, init_axis=context.input_initialization_axes[stim_id])
        graph.register_output(node, qubit)
    return _Fragment(graph=graph, xflow={}, record_nodes={})


def _unitary_fragment(  # ruff:ignore[too-many-arguments]
    block: Sequence[stim.CircuitInstruction],
    *,
    live_stim_ids: set[int],
    z_base: int,
    stim_to_qubit: Mapping[int, int],
    qubit_to_stim: Mapping[int, int],
    context: _ImportContext,
) -> tuple[_Fragment, int]:
    ordered_stim_ids = sorted(live_stim_ids)
    stim_to_local = {stim_id: local_index for local_index, stim_id in enumerate(ordered_stim_ids)}
    local_to_global = {local_index: stim_to_qubit[stim_id] for stim_id, local_index in stim_to_local.items()}
    circuit = Circuit(len(ordered_stim_ids))
    for instruction in block:
        _append_unitary_instruction(circuit, instruction, stim_to_local)

    local_graph, local_xflow, _ = circuit2graph(circuit, schedule_strategy=context.schedule_strategy)
    graph, node_map = _copy_graph_with_qindices(local_graph, local_to_global)
    xflow = _remap_flow(local_xflow, node_map)
    z_span = _apply_unitary_coordinates(
        graph,
        xflow,
        z_base=z_base,
        qubit_to_stim=qubit_to_stim,
        context=context,
    )
    return (
        _Fragment(
            graph=graph,
            xflow=xflow,
            record_nodes={},
        ),
        z_base + z_span,
    )


def _apply_unitary_coordinates(
    graph: GraphState,
    xflow: Mapping[int, set[int]],
    *,
    z_base: int,
    qubit_to_stim: Mapping[int, int],
    context: _ImportContext,
) -> int:
    output_node_by_qubit = {qubit: node for node, qubit in graph.output_node_indices.items()}
    chains: dict[int, list[int]] = {}
    chain_nodes: set[int] = set()
    for input_node, qubit in graph.input_node_indices.items():
        output_node = output_node_by_qubit[qubit]
        chain = [input_node]
        visited = {input_node}
        while chain[-1] != output_node:
            targets = xflow.get(chain[-1])
            if targets is None or len(targets) != 1:
                msg = f"Transpiled unitary wire for qubit {qubit} is not a single X-flow chain."
                raise ValueError(msg)
            next_node = next(iter(targets))
            if next_node in visited:
                msg = f"Transpiled unitary wire for qubit {qubit} contains an X-flow cycle."
                raise ValueError(msg)
            chain.append(next_node)
            visited.add(next_node)
        chains[qubit] = chain
        chain_nodes.update(chain)

    if chain_nodes != graph.nodes:
        msg = "Transpiled unitary graph contains a node outside its data-wire X-flow chains."
        raise ValueError(msg)

    max_chain_depth = max((len(chain) - 1 for chain in chains.values()), default=0)
    z_span = max(1, max_chain_depth)
    for qubit, chain in chains.items():
        coord = context.coordinate_by_stim_id.get(qubit_to_stim[qubit])
        if coord is None:
            continue
        depth = len(chain) - 1
        if depth == 0:
            graph.set_coordinate(chain[0], _coordinate_at_z(coord, z_base + z_span))
            continue
        for layer, node in enumerate(chain):
            z = z_base + z_span * layer / depth
            graph.set_coordinate(node, _coordinate_at_z(coord, z))
    return z_span


def _copy_graph_with_qindices(
    graph: GraphState,
    local_to_global: Mapping[int, int],
) -> tuple[GraphState, dict[int, int]]:
    copied = GraphState()
    node_map = {node: copied.add_node() for node in sorted(graph.nodes)}
    for node1, node2 in graph.edges:
        copied.add_edge(node_map[node1], node_map[node2])
    for node, q_index in graph.input_node_indices.items():
        copied.register_input(node_map[node], local_to_global[q_index])
    for node, q_index in graph.output_node_indices.items():
        copied.register_output(node_map[node], local_to_global[q_index])
    for node, meas_basis in graph.meas_bases.items():
        copied.assign_meas_basis(node_map[node], meas_basis)
    for node, local_clifford in graph.local_cliffords.items():
        copied.apply_local_clifford(node_map[node], local_clifford)
    for node, coordinate in graph.coordinates.items():
        copied.set_coordinate(node_map[node], coordinate)
    return copied, node_map


def _append_unitary_instruction(
    circuit: Circuit,
    instruction: stim.CircuitInstruction,
    stim_to_qubit: Mapping[int, int],
) -> None:
    for group in instruction.target_groups():
        qubits = [plain_qubit_target(target, instruction.name) for target in group]
        mapped = [stim_to_qubit[qubit] for qubit in qubits]
        j_angle = STIM_GATE_J_ANGLES.get(instruction.name)
        if j_angle is not None:
            circuit.apply_macro_gate(J(mapped[0], j_angle))
        elif instruction.name == "CZ":
            circuit.apply_macro_gate(CZ((mapped[0], mapped[1])))
        else:
            msg = f"Stim parser emitted unsupported basis instruction: {instruction.name}."
            raise AssertionError(msg)


def _mpp_fragment(
    block: Sequence[_AnalyzedInstruction],
    *,
    z_base: int,
    io_stim_ids: set[int],
    stim_to_qubit: Mapping[int, int],
    context: _ImportContext,
) -> _Fragment:
    signed_products = tuple(
        product for analyzed in block for product in mpp_targets_to_signed_products(analyzed.instruction.targets_copy())
    )
    supports = tuple(support for support, _sign in signed_products)
    # `stim_mpp_extraction_from_records` keeps the product order, so product i
    # becomes stabilizer row i and therefore ancilla node `ancilla_nodes[i]`.
    negative_rows = frozenset(row for row, (_support, sign) in enumerate(signed_products) if sign is Sign.MINUS)
    _validate_commuting_mpp_supports(supports)
    record_indices = tuple(record_index for analyzed in block for record_index in analyzed.record_indices)
    extraction = stim_mpp_extraction_from_records(
        supports,
        record_indices,
        coordinate_by_stim_id=context.coordinate_by_stim_id,
        detector_record_indices=context.detector_record_indices,
        logical_observable_record_indices=context.logical_observable_record_indices,
        detector_tags=context.detector_tags,
    )
    fragment = _mpp_graph_fragment(
        extraction,
        record_indices=record_indices,
        negative_rows=negative_rows,
        z_base=z_base,
        io_stim_ids=io_stim_ids,
        stim_to_qubit=stim_to_qubit,
        context=context,
    )
    return _Fragment(
        graph=fragment.graph,
        xflow=fragment.xflow,
        record_nodes=fragment.record_nodes,
        mpp_extractions=(extraction,),
    )


def _validate_commuting_mpp_supports(supports: Sequence[PauliSupport]) -> None:
    for (left_index, left), (right_index, right) in combinations(enumerate(supports), 2):
        if not pauli_products_commute(left, right):
            msg = (
                f"MPP products within one TICK block must commute; products {left_index} and {right_index} anticommute."
            )
            raise ValueError(msg)


def _mpp_graph_fragment(  # ruff:ignore[too-many-arguments]
    extraction: StimMppExtraction,
    *,
    record_indices: Sequence[int],
    negative_rows: AbstractSet[int],
    z_base: int,
    io_stim_ids: set[int],
    stim_to_qubit: Mapping[int, int],
    context: _ImportContext,
) -> _Fragment:
    qubit_indices = {column: stim_to_qubit[stim_id] for column, stim_id in extraction.column_to_stim.items()}
    result = build_graph_state(
        extraction.code,
        z_base=z_base,
        y_foliation=context.y_foliation,
        data_as_io=True,
        qubit_indices=qubit_indices,
    )
    # A negative product sign flips only the measurement sign; the axis must stay
    # the one build_graph_state chose (Y for odd-Y-support rows under Type I).
    for row in sorted(negative_rows):
        ancilla_node = result.ancilla_nodes[row]
        result.graph.assign_meas_basis(ancilla_node, result.graph.meas_bases[ancilla_node].flip())
    active_stim_ids = set(extraction.stim_to_column)
    _add_relocated_io_nodes(
        result.graph,
        io_stim_ids - active_stim_ids,
        z=z_base + 2,
        stim_to_qubit=stim_to_qubit,
        context=context,
    )
    xflow = _mpp_flow(result)
    if len(record_indices) != len(result.ancilla_nodes):
        msg = "Imported MPP record count does not match the generated ancilla-node count."
        raise ValueError(msg)
    return _Fragment(
        graph=result.graph,
        xflow=xflow,
        record_nodes={record_indices[row]: node for row, node in result.ancilla_nodes.items()},
    )


def _add_relocated_io_nodes(
    graph: GraphState,
    stim_ids: set[int],
    *,
    z: int,
    stim_to_qubit: Mapping[int, int],
    context: _ImportContext,
) -> None:
    for stim_id in sorted(stim_ids):
        coord = context.coordinate_by_stim_id.get(stim_id)
        node = graph.add_node(coordinate=_coordinate_at_z(coord, z) if coord is not None else None)
        qubit = stim_to_qubit[stim_id]
        graph.register_input(node, qubit)
        graph.register_output(node, qubit)


def _coordinate_at_z(coord: tuple[float, ...], z: float) -> tuple[float, float, float]:
    return (float(coord[0]), float(coord[1]), float(z))


def _mpp_flow(
    result: StabilizerGraphStateBuildResult,
) -> dict[int, set[int]]:
    xflow: dict[int, set[int]] = {}
    for qubit in sorted({key[0] for key in result.data_nodes}):
        layer_nodes = [node for (data_qubit, _layer), node in sorted(result.data_nodes.items()) if data_qubit == qubit]
        for current_node, next_node in pairwise(layer_nodes):
            if current_node in result.graph.meas_bases:
                xflow[current_node] = {next_node}
    return xflow


def _compose_fragments(fragments: Sequence[_Fragment]) -> _Fragment:
    current = fragments[0]
    for fragment in fragments[1:]:
        graph, node_map1, node_map2 = compose(current.graph, fragment.graph)
        current = _Fragment(
            graph=graph,
            xflow=_merge_flows(_remap_flow(current.xflow, node_map1), _remap_flow(fragment.xflow, node_map2)),
            record_nodes=_remap_record_nodes(current.record_nodes, node_map1)
            | _remap_record_nodes(fragment.record_nodes, node_map2),
            feedback_targets=(
                *_remap_feedback_targets(current.feedback_targets, node_map1),
                *_capture_feedback_targets(fragment.feedback_targets, node_map2, graph),
            ),
            mpp_extractions=(*current.mpp_extractions, *fragment.mpp_extractions),
            zflow=_merge_flows(_remap_flow(current.zflow, node_map1), _remap_flow(fragment.zflow, node_map2)),
        )
    return current


def _merge_flows(flow1: Mapping[int, set[int]], flow2: Mapping[int, set[int]]) -> dict[int, set[int]]:
    r"""Combine two correction flows by XOR-ing target sets that share a source.

    Flow sources are measured nodes private to one fragment, so today the two
    maps never collide; XOR keeps a future collision Pauli-correct instead of
    silently dropping one side, as a plain dict union would.

    Returns
    -------
    `dict`\[`int`, `set`\[`int`\]\]
        Merged flow.
    """
    merged = {node: set(targets) for node, targets in flow1.items()}
    for node, targets in flow2.items():
        merged.setdefault(node, set()).symmetric_difference_update(targets)
    return merged


def _apply_single_measurements(
    fragment: _Fragment,
    direct_measurements: Sequence[_DirectMeasurement],
    *,
    final_stim_to_qubit: Mapping[int, int],
) -> _Fragment:
    output_node_by_qubit = {qubit: node for node, qubit in fragment.graph.output_node_indices.items()}
    record_nodes = dict(fragment.record_nodes)

    for measurement in direct_measurements:
        if measurement.record_index in record_nodes:
            # Reused measurements were bound to internal nodes during fragment construction.
            continue
        node = output_node_by_qubit[final_stim_to_qubit[measurement.stim_id]]
        fragment.graph.assign_meas_basis(node, AxisMeasBasis(measurement.axis, measurement.sign))
        record_nodes[measurement.record_index] = node

    return _Fragment(
        graph=fragment.graph,
        xflow=fragment.xflow,
        record_nodes=record_nodes,
        feedback_targets=fragment.feedback_targets,
        mpp_extractions=fragment.mpp_extractions,
        zflow=fragment.zflow,
    )


def _flows_with_feedback(
    fragment: _Fragment,
    *,
    zero_record_indices: frozenset[int],
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    xflow = {node: set(targets) for node, targets in fragment.xflow.items()}
    zflow = {node: odd_neighbors(targets, fragment.graph) for node, targets in fragment.xflow.items()}
    for node, targets in fragment.zflow.items():
        # Explicit entries carry the post-measurement continuation corrections.
        zflow.setdefault(node, set()).symmetric_difference_update(targets)

    for feedback in fragment.feedback_targets:
        source_nodes = _record_indices_to_nodes(
            frozenset({feedback.record_index}),
            fragment.record_nodes,
            zero_record_indices=zero_record_indices,
        )
        if not source_nodes:
            continue
        source = next(iter(source_nodes))
        if feedback.axis in {Axis.X, Axis.Y}:
            xflow.setdefault(source, set()).symmetric_difference_update({feedback.node})
            # Deferring the X to the target's measurement pushes it through the
            # CZ edges created after the feedback position, leaving Z there.
            deferred_z = fragment.graph.neighbors(feedback.node) - feedback.past_neighbors
            if deferred_z:
                zflow.setdefault(source, set()).symmetric_difference_update(deferred_z)
        if feedback.axis in {Axis.Z, Axis.Y}:
            zflow.setdefault(source, set()).symmetric_difference_update({feedback.node})
    return xflow, zflow


def _measurement_annotations(
    annotations: RecordAnnotations,
    *,
    record_nodes: Mapping[int, int],
    zero_record_indices: frozenset[int],
) -> tuple[list[set[int]], list[str], dict[int, set[int]]]:
    if not record_nodes and not zero_record_indices:
        return [], [], {}

    parity_check_groups = [
        _record_indices_to_nodes(record_indices, record_nodes, zero_record_indices=zero_record_indices)
        for record_indices in annotations.detectors
    ]
    logical_observables = {
        logical_idx: _record_indices_to_nodes(
            record_indices,
            record_nodes,
            zero_record_indices=zero_record_indices,
        )
        for logical_idx, record_indices in annotations.logical_observables.items()
    }
    return parity_check_groups, list(annotations.detector_tags), logical_observables


def _record_indices_to_nodes(
    record_indices: frozenset[int],
    record_nodes: Mapping[int, int],
    *,
    zero_record_indices: frozenset[int],
) -> set[int]:
    missing_records = sorted(
        record_index
        for record_index in record_indices
        if record_index not in record_nodes and record_index not in zero_record_indices
    )
    if missing_records:
        msg = f"Cannot map Stim measurement record(s) to imported Pauli-measurement nodes: {missing_records}."
        raise ValueError(msg)
    return {record_nodes[record_index] for record_index in record_indices if record_index in record_nodes}


def _remap_flow(flow: Mapping[int, set[int]], node_map: Mapping[int, int]) -> dict[int, set[int]]:
    return {node_map[node]: {node_map[target] for target in targets} for node, targets in flow.items()}


def _remap_record_nodes(record_nodes: Mapping[int, int], node_map: Mapping[int, int]) -> dict[int, int]:
    return {record_index: node_map[node] for record_index, node in record_nodes.items()}


def _remap_feedback_targets(
    feedback_targets: Sequence[_FeedbackTarget],
    node_map: Mapping[int, int],
) -> tuple[_FeedbackTarget, ...]:
    return tuple(
        _FeedbackTarget(
            record_index=feedback.record_index,
            node=node_map[feedback.node],
            axis=feedback.axis,
            past_neighbors=frozenset(node_map[node] for node in feedback.past_neighbors),
        )
        for feedback in feedback_targets
    )


def _capture_feedback_targets(
    feedback_targets: Sequence[_FeedbackTarget],
    node_map: Mapping[int, int],
    graph: GraphState,
) -> tuple[_FeedbackTarget, ...]:
    # A feedback fragment has no internal edges, so right after composition the
    # target's neighbors are exactly those entangled before the feedback position.
    return tuple(
        _FeedbackTarget(
            record_index=feedback.record_index,
            node=node_map[feedback.node],
            axis=feedback.axis,
            past_neighbors=frozenset(graph.neighbors(node_map[feedback.node])),
        )
        for feedback in feedback_targets
    )
