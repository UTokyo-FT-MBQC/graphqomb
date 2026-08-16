"""Rewrite Stim syndrome-extraction circuits into abstract MPP measurements.

This module recognizes the standard syndrome-extraction shape (reset ancillas,
apply Clifford unitaries, measure) and replaces each gate-level segment with
``MPP`` instructions that measure the inferred data Pauli products directly.

Correctness is established constructively instead of by after-the-fact
verification: measurement observables are pulled through the segment body with
a sparse F2 symplectic tableau (``U† P U``), fresh-ancilla stabilizers are
substituted only where their algebraic preconditions hold, and the residual
Clifford frame on surviving qubits is the symplectic restriction of the body
tableau, built only when that restriction exists. A segment whose
preconditions fail is emitted gate-level (``fallback="segment"``) or falls the
whole circuit back to its flattened source (``fallback="circuit"``).

This module provides:

- `rewrite_to_mpp`: Rewrite a noiseless syndrome-extraction circuit into MPP form.
- `MppRewriteResult`: Rewritten circuit with its per-measurement Pauli products.
- `CheckMapping`: Mapping from one measurement record to its Pauli product.
- `UnsupportedSyndromeCircuitError`: Error for unsupported syndrome circuits.
- `MppRewriteVerificationError`: Retained for API compatibility; no longer raised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import stim
from scipy.sparse import csr_array, lil_array

from graphqomb.common import Axis
from graphqomb.stim_glue._parse import (
    ANNOTATION_GATES,
    DIRECT_MEASUREMENT_AXES,
    MEASURE_RESET_AXES,
    PAIR_MEASUREMENT_AXES,
    RESET_AXES,
    RESET_GATES,
    SINGLE_MEASUREMENT_AXES,
    iter_instructions,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

_InstructionKind = Literal["annotation", "mpad", "reset", "measurement", "unitary"]
_FallbackMode = Literal["circuit", "segment"]

_PAIR_GROUP_SIZE = 2
# (x bit, z bit) of each Pauli axis in the [x | z] component convention.
_AXIS_BITS: dict[Axis, tuple[int, int]] = {
    Axis.X: (1, 0),
    Axis.Y: (1, 1),
    Axis.Z: (0, 1),
}
_SIGN_FROM_EXPONENT: tuple[complex, ...] = (1, 1j, -1, -1j)
# i**2 = -1: the phase of a deterministic-minus identity product.
_MINUS_PHASE = 2


class UnsupportedSyndromeCircuitError(ValueError):
    """Raised when a circuit is outside the supported syndrome-extraction form."""


class MppRewriteVerificationError(ValueError):
    """Retained for API compatibility; the rewrite is validated constructively and no longer raises this."""


@dataclass(frozen=True)
class CheckMapping:
    r"""Mapping from one measurement record to the Pauli product it measures.

    Attributes
    ----------
    measurement_index : `int`
        Global measurement-record index in the rewritten circuit.
    segment_index : `int`
        Index of the segment the measurement belongs to.
    product : ``stim.PauliString``
        Measured Pauli product in source-circuit qubit coordinates.
    source_qubit : `int` | `None`
        Measured qubit for single-qubit source measurements, `None` for
        source ``MPP``/``MXX``/``MYY``/``MZZ`` products.
    """

    measurement_index: int
    segment_index: int
    product: stim.PauliString
    source_qubit: int | None


@dataclass(frozen=True)
class MppRewriteResult:
    r"""Result of rewriting a syndrome-extraction circuit into MPP form.

    Attributes
    ----------
    circuit : ``stim.Circuit``
        Rewritten circuit with one measurement per source measurement.
    checks : `tuple`\[`CheckMapping`, ...\]
        One mapping per measurement record produced by a Pauli measurement.
        ``MPAD`` records are not listed.
    fallback_segments : `tuple`\[`int`, ...\]
        Indices of the segments emitted gate-level instead of in MPP form:
        every segment when the whole circuit took the gate-level fallback,
        only the individually failing segments under ``fallback="segment"``,
        and empty when every segment was rewritten.
    """

    circuit: stim.Circuit
    checks: tuple[CheckMapping, ...]
    fallback_segments: tuple[int, ...] = ()


@dataclass(frozen=True)
class _SourceObservable:
    observable: stim.PauliString
    source_qubit: int | None


@dataclass(frozen=True)
class _Pauli:
    """Pauli operator ``i**phase · X^x Z^z`` with components stored as ``[x | z]``."""

    bits: NDArray[np.bool_]
    phase: int


def _pauli_mul(left: _Pauli, right: _Pauli) -> _Pauli:
    """Multiply two Pauli operators in the ``i**phase · X^x Z^z`` convention.

    Returns
    -------
    `_Pauli`
        The product ``left * right``.
    """
    half = left.bits.shape[0] // 2
    cross = int(np.count_nonzero(left.bits[half:] & right.bits[:half])) & 1
    return _Pauli(bits=left.bits ^ right.bits, phase=(left.phase + right.phase + 2 * cross) % 4)


def _pauli_from_stim(pauli: stim.PauliString) -> _Pauli:
    """Convert a Stim Pauli string into component form.

    Returns
    -------
    `_Pauli`
        Component representation of the same operator.
    """
    xs, zs = pauli.to_numpy()
    y_count = int(np.count_nonzero(xs & zs))
    exponent = _SIGN_FROM_EXPONENT.index(pauli.sign)
    return _Pauli(bits=np.concatenate([xs, zs]), phase=(y_count + exponent) % 4)


def _pauli_to_stim(pauli: _Pauli) -> stim.PauliString:
    """Convert a component-form Pauli into a Stim Pauli string.

    Returns
    -------
    ``stim.PauliString``
        The same operator as a Stim Pauli string.
    """
    half = pauli.bits.shape[0] // 2
    xs = pauli.bits[:half]
    zs = pauli.bits[half:]
    result = stim.PauliString.from_numpy(xs=xs, zs=zs)
    y_count = int(np.count_nonzero(xs & zs))
    result.sign = _SIGN_FROM_EXPONENT[(pauli.phase - y_count) % 4]
    return result


@cache
def _gate_generator_images(name: str) -> tuple[tuple[tuple[int, ...], int], ...]:
    r"""Return the local generator images under conjugation by the inverse gate.

    For gate ``g`` this is ``g† P g`` for each local generator ``P`` in
    ``X_0..X_{k-1}, Z_0..Z_{k-1}`` order, so appending ``g`` to a circuit
    updates a backward map ``P -> U† P U`` by rewriting these generators. Each
    image is spelled as the ascending local generator indices whose ordered
    product it is, with its phase.

    Returns
    -------
    `tuple`\[`tuple`\[`tuple`\[`int`, ...\], `int`\], ...\]
        One ``(local generator indices, phase)`` pair per local generator.

    Raises
    ------
    UnsupportedSyndromeCircuitError
        If the gate has no tableau (it is not a unitary gate).
    """
    tableau = stim.gate_data(name).tableau
    if tableau is None:
        msg = f"{name} is not a unitary gate."
        raise UnsupportedSyndromeCircuitError(msg)
    inverse = tableau.inverse()
    num_qubits = len(inverse)
    outputs = [inverse.x_output(qubit) for qubit in range(num_qubits)]
    outputs += [inverse.z_output(qubit) for qubit in range(num_qubits)]
    images = []
    for output in outputs:
        pauli = _pauli_from_stim(output)
        images.append((tuple(int(index) for index in np.flatnonzero(pauli.bits)), pauli.phase))
    return tuple(images)


class _BodyTableau:
    """Backward conjugation map ``P -> U† P U`` of a Clifford body over F2.

    Row ``j`` holds the component support of the image of generator ``j``
    (``X_j`` for ``j < n``, ``Z_{j-n}`` otherwise) as an index set — the LIL
    layout — with phases tracked per generator modulo 4, because per-gate
    updates and pull-backs combine a handful of small supports at a time. The
    matrix-level work (the transposed column view behind the symplectic
    inverse) goes through scipy sparse, so one incremental structure serves
    product pull-back and residual-frame construction.
    """

    def __init__(self, num_qubits: int) -> None:
        self._num_qubits = num_qubits
        size = 2 * num_qubits
        self._supports: list[set[int]] = [{index} for index in range(size)]
        self._phases: list[int] = [0] * size
        self._gate_count = 0
        self._columns: csr_array[Any, tuple[int, int]] | None = None

    @property
    def gate_count(self) -> int:
        """Number of gates appended to the body."""
        return self._gate_count

    def _fold(self, indices: Sequence[int], phase: int) -> tuple[set[int], int]:
        r"""Multiply the stored generator images selected by ``indices`` in order.

        Returns
        -------
        `tuple`\[`set`\[`int`\], `int`\]
            Support and phase of ``i**phase · prod(A(gen_j))``.
        """
        half = self._num_qubits
        support: set[int] = set()
        for index in indices:
            row = self._supports[index]
            cross = sum(1 for component in row if component < half and component + half in support)
            phase = (phase + self._phases[index] + 2 * (cross & 1)) % 4
            support ^= row
        return support, phase

    def append(self, name: str, qubits: Sequence[int]) -> None:
        """Extend the body by one gate application."""
        images = _gate_generator_images(name)
        generator_indices = [*qubits, *(self._num_qubits + qubit for qubit in qubits)]
        folded = [
            self._fold([generator_indices[local] for local in image_support], image_phase)
            for image_support, image_phase in images
        ]
        for index, (support, phase) in zip(generator_indices, folded, strict=True):
            self._supports[index] = support
            self._phases[index] = phase
        self._gate_count += 1
        self._columns = None

    def pull(self, pauli: _Pauli) -> _Pauli:
        """Return ``U† P U`` for a Pauli measured after the body.

        Returns
        -------
        `_Pauli`
            The pulled-back operator.
        """
        support, phase = self._fold([int(index) for index in np.flatnonzero(pauli.bits)], pauli.phase)
        return _Pauli(bits=_dense_bits(support, 2 * self._num_qubits), phase=phase)

    def forward_generator(self, index: int) -> _Pauli:
        """Return ``U gen U†`` through the symplectic inverse of the stored map.

        Returns
        -------
        `_Pauli`
            Forward image of generator ``index``.
        """
        if self._columns is None:
            matrix = lil_array((2 * self._num_qubits, 2 * self._num_qubits), dtype=np.bool_)
            for row, support in enumerate(self._supports):
                columns = sorted(support)
                matrix.rows[row] = columns
                matrix.data[row] = [True] * len(columns)
            self._columns = csr_array(matrix.T)
        half = self._num_qubits
        partner = (index + half) % (2 * half)
        start, stop = self._columns.indptr[partner], self._columns.indptr[partner + 1]
        preimage = sorted((int(row) + half) % (2 * half) for row in self._columns.indices[start:stop])
        _, pulled_phase = self._fold(preimage, 0)
        return _Pauli(bits=_dense_bits(set(preimage), 2 * half), phase=(-pulled_phase) % 4)


def _dense_bits(support: set[int], size: int) -> NDArray[np.bool_]:
    r"""Materialize an index set as a dense boolean component vector.

    Returns
    -------
    ``NDArray``\[``np.bool_``\]
        The dense vector.
    """
    bits = np.zeros(size, dtype=np.bool_)
    bits[list(support)] = True
    return bits


@dataclass
class _SegmentBounds:
    """Segment-boundary state shared by segmentation and fallback numbering.

    A segment ends before a unitary that follows the segment's measurements,
    and before a measurement that follows a reset issued after those
    measurements (including the implicit reset of ``MR``).
    """

    seen_measurement: bool = False
    seen_late_reset: bool = False

    def starts_new_segment(self, instruction: stim.CircuitInstruction, kind: _InstructionKind) -> bool:
        """Return whether this instruction opens a new segment, and absorb it.

        Returns
        -------
        `bool`
            Whether the instruction belongs to a new segment.
        """
        boundary = (kind == "unitary" and self.seen_measurement) or (
            kind in {"measurement", "mpad"} and self.seen_late_reset
        )
        if boundary:
            self.seen_measurement = False
            self.seen_late_reset = False
        if kind in {"measurement", "mpad"}:
            self.seen_measurement = True
            if instruction.name in MEASURE_RESET_AXES:
                self.seen_late_reset = True
        elif kind == "reset" and self.seen_measurement:
            self.seen_late_reset = True
        return boundary


@dataclass
class _SourceSegment:
    """One reset/Clifford/measure segment of the flattened source circuit.

    Carries the classified instructions with per-position measured observables
    (``sources``), the measured-out qubits with their single-qubit measurement
    bases, reset/unreset bookkeeping for post-state tracking, and
    ``blocked_message`` when feedback or a reset-after-entangle rules the
    segment out regardless of its algebra.
    """

    index: int
    items: list[tuple[stim.CircuitInstruction, _InstructionKind]] = field(default_factory=list)
    sources: dict[int, tuple[_SourceObservable, ...]] = field(default_factory=dict)
    measured_bases: dict[int, set[Axis]] = field(default_factory=dict)
    unitary_qubits: set[int] = field(default_factory=set)
    reset_qubits: set[int] = field(default_factory=set)
    unreset_measured: set[int] = field(default_factory=set)
    seen_measurement: bool = False
    blocked_message: str | None = None


@dataclass(frozen=True)
class _Conflict:
    """A segment consumed a measurement post-state trashed by a rewritten segment."""

    producer: int
    message: str


def rewrite_to_mpp(circuit: stim.Circuit | str, *, fallback: _FallbackMode = "circuit") -> MppRewriteResult:
    """Rewrite a noiseless Stim syndrome-extraction circuit into MPP form.

    Each segment's measurement observables are pulled backwards through its
    Clifford body (``U† P U``) with a sparse F2 tableau. Stabilizers of
    freshly prepared, measured-out ancillas are substituted with +1 when every
    product's factor on that ancilla is the prepared basis or identity and the
    substituted products still commute pairwise per instruction. Clifford
    frames left on surviving qubits are appended as the symplectic restriction
    of the body tableau when that restriction exists; a component on a
    measured-out qubit must anticommute with every measurement basis of that
    qubit to be absorbed. Each source measurement maps to exactly one
    measurement in the rewritten circuit, so ``DETECTOR`` and
    ``OBSERVABLE_INCLUDE`` annotations are copied verbatim.

    ``REPEAT`` blocks are flattened before rewriting, which also bakes
    ``SHIFT_COORDS`` offsets into ``DETECTOR`` coordinate arguments.

    When a segment's preconditions fail, ``fallback="circuit"`` returns the
    flattened gate-level circuit unchanged, while ``fallback="segment"`` keeps
    only the failing segments gate-level, listed in
    `MppRewriteResult.fallback_segments`. In segment mode a segment that
    applies measurement-record-controlled feedback, or that consumes a
    measurement post-state an earlier rewritten segment trashed, also stays
    gate-level (the producing segment is forced gate-level as well, because
    only verbatim segments preserve their post-states exactly).

    Parameters
    ----------
    circuit : ``stim.Circuit`` | `str`
        Source circuit or Stim circuit text. Noise instructions and
        measurement noise arguments are rejected with
        `UnsupportedSyndromeCircuitError`; classical feedback is rejected
        under ``fallback="circuit"``.
    fallback : ``"circuit"`` | ``"segment"``, optional
        Whether a failing segment falls back the whole circuit (default) or
        only itself.

    Returns
    -------
    `MppRewriteResult`
        Rewritten circuit and per-measurement Pauli-product mappings.

    Raises
    ------
    ValueError
        If fallback is not ``"circuit"`` or ``"segment"``.
    """
    if fallback not in {"circuit", "segment"}:
        msg = f"fallback must be 'circuit' or 'segment', not {fallback!r}."
        raise ValueError(msg)
    source = circuit if isinstance(circuit, stim.Circuit) else stim.Circuit(circuit)
    flattened = source.flattened()
    segments = _split_segments(flattened)
    forced_verbatim: set[int] = set()
    while True:
        outcome = _rewrite_pass(
            segments,
            flattened.num_qubits,
            fallback=fallback,
            forced_verbatim=forced_verbatim,
        )
        if outcome is None:
            return _passthrough_result(flattened, segments)
        if isinstance(outcome, _Conflict):
            forced_verbatim.add(outcome.producer)
            continue
        return outcome


def _split_segments(circuit: stim.Circuit) -> list[_SourceSegment]:
    r"""Split a flattened circuit into segments and validate its instructions.

    Returns
    -------
    `list`\[`_SourceSegment`\]
        Segments in circuit order.
    """
    segments: list[_SourceSegment] = []
    bounds = _SegmentBounds()
    current = _SourceSegment(index=0)
    num_qubits = circuit.num_qubits
    for instruction in iter_instructions(circuit):
        kind = _instruction_kind(instruction)
        if bounds.starts_new_segment(instruction, kind) and current.items:
            segments.append(current)
            current = _SourceSegment(index=current.index + 1)
        _collect_item(current, instruction, kind, num_qubits)
    if current.items:
        segments.append(current)
    return segments


def _collect_item(
    segment: _SourceSegment,
    instruction: stim.CircuitInstruction,
    kind: _InstructionKind,
    num_qubits: int,
) -> None:
    """Append one classified instruction to a segment and update its metadata."""
    position = len(segment.items)
    segment.items.append((instruction, kind))
    if kind == "reset":
        for target in instruction.targets_copy():
            qubit = _plain_qubit(target, instruction.name)
            if not segment.seen_measurement and qubit in segment.unitary_qubits and segment.blocked_message is None:
                segment.blocked_message = (
                    f"Reset on qubit {qubit} after it was entangled in the same segment is not supported."
                )
            segment.reset_qubits.add(qubit)
            segment.unreset_measured.discard(qubit)
    elif kind == "unitary":
        _collect_unitary(segment, instruction)
    elif kind == "mpad":
        segment.seen_measurement = True
    elif kind == "measurement":
        _collect_measurement(segment, instruction, position, num_qubits)


def _collect_unitary(segment: _SourceSegment, instruction: stim.CircuitInstruction) -> None:
    """Record one unitary instruction's qubits and feedback usage.

    Raises
    ------
    UnsupportedSyndromeCircuitError
        If the unitary is controlled by a sweep bit.
    """
    for target in instruction.targets_copy():
        if target.is_sweep_bit_target:
            msg = f"Classical feedback is not supported: {instruction.name} with target {target!r}."
            raise UnsupportedSyndromeCircuitError(msg)
        if target.is_measurement_record_target:
            if segment.blocked_message is None:
                segment.blocked_message = (
                    f"Classical feedback is not supported: {instruction.name} with target {target!r}."
                )
        elif target.qubit_value is not None:
            segment.unitary_qubits.add(int(target.qubit_value))


def _collect_measurement(
    segment: _SourceSegment,
    instruction: stim.CircuitInstruction,
    position: int,
    num_qubits: int,
) -> None:
    """Record one measurement instruction's observables and measured-out qubits.

    Raises
    ------
    UnsupportedSyndromeCircuitError
        If the measurement carries noise arguments.
    """
    segment.seen_measurement = True
    if instruction.gate_args_copy():
        msg = f"Noisy measurement {instruction.name} with arguments is not supported."
        raise UnsupportedSyndromeCircuitError(msg)
    segment.sources[position] = tuple(_measurement_observables(instruction, num_qubits))
    if instruction.name in DIRECT_MEASUREMENT_AXES:
        axis = DIRECT_MEASUREMENT_AXES[instruction.name]
        for target in instruction.targets_copy():
            qubit = _plain_qubit(target, instruction.name)
            segment.measured_bases.setdefault(qubit, set()).add(axis)
            if instruction.name in SINGLE_MEASUREMENT_AXES:
                segment.unreset_measured.add(qubit)
            else:
                segment.reset_qubits.add(qubit)
                segment.unreset_measured.discard(qubit)


def _rewrite_pass(
    segments: Sequence[_SourceSegment],
    num_qubits: int,
    *,
    fallback: _FallbackMode,
    forced_verbatim: set[int],
) -> MppRewriteResult | _Conflict | None:
    r"""Rewrite every segment once with the given forced-verbatim set.

    Returns
    -------
    `MppRewriteResult` | `_Conflict` | `None`
        The finished rewrite, a post-state consumption conflict that forces a
        producing segment gate-level (segment mode), or `None` when the whole
        circuit must take the gate-level fallback (circuit mode).

    Raises
    ------
    UnsupportedSyndromeCircuitError
        In circuit mode, if a segment feeds back from a measurement record,
        resets an entangled qubit, or consumes a trashed post-state.
    """
    output = stim.Circuit()
    checks: list[CheckMapping] = []
    prepared: dict[int, Axis] = {}
    dirty: dict[int, int] = {}
    measurement_index = 0
    fallback_segments: list[int] = []
    for segment in segments:
        verbatim = segment.index in forced_verbatim or (fallback == "segment" and segment.blocked_message is not None)
        if not verbatim and segment.blocked_message is not None:
            raise UnsupportedSyndromeCircuitError(segment.blocked_message)
        analyzed: _SegmentRewrite | _Conflict | None = None
        if not verbatim:
            analyzed = _analyze_segment(
                segment,
                num_qubits,
                prepared=prepared,
                dirty=dirty,
                measurement_index=measurement_index,
                allow_negative_pads=fallback == "circuit",
            )
            if analyzed is None:
                if fallback == "circuit":
                    return None
                verbatim = True
        if verbatim:
            fallback_segments.append(segment.index)
            analyzed = _emit_verbatim(
                segment,
                prepared=prepared,
                dirty=dirty,
                measurement_index=measurement_index,
            )
        if isinstance(analyzed, _Conflict):
            if fallback == "circuit":
                raise UnsupportedSyndromeCircuitError(analyzed.message)
            return analyzed
        if analyzed is None:  # pragma: no cover - one branch above always assigns
            continue
        output += analyzed.circuit
        checks.extend(analyzed.checks)
        prepared = analyzed.exit_prepared
        dirty = analyzed.exit_dirty
        measurement_index = analyzed.exit_measurement_index
    return MppRewriteResult(circuit=output, checks=tuple(checks), fallback_segments=tuple(fallback_segments))


@dataclass(frozen=True)
class _SegmentRewrite:
    """One emitted segment with its exit bookkeeping."""

    circuit: stim.Circuit
    checks: tuple[CheckMapping, ...]
    exit_prepared: dict[int, Axis]
    exit_dirty: dict[int, int]
    exit_measurement_index: int


def _analyze_segment(  # ruff:ignore[too-many-arguments]
    segment: _SourceSegment,
    num_qubits: int,
    *,
    prepared: dict[int, Axis],
    dirty: dict[int, int],
    measurement_index: int,
    allow_negative_pads: bool,
) -> _SegmentRewrite | _Conflict | None:
    """Rewrite one segment into MPP form when its preconditions hold.

    Returns
    -------
    `_SegmentRewrite` | `_Conflict` | `None`
        The rewritten segment, a trashed-post-state consumption conflict, or
        `None` when the segment is not MPP-representable.
    """
    scan = _scan_segment(segment, num_qubits, prepared=prepared, dirty=dirty)
    if isinstance(scan, _Conflict):
        return scan
    products = _substituted_products(
        segment,
        scan.pulled,
        scan.prepared,
        num_qubits,
        allow_negative_pads=allow_negative_pads,
    )
    if products is None:
        return None
    frame = _residual_frame(scan.body, scan.body_touched, segment, num_qubits)
    if frame is None:
        return None

    circuit, checks, exit_measurement_index = _emit_segment(
        segment,
        products,
        frame,
        measurement_index=measurement_index,
        late_resets=scan.late_resets,
    )
    exit_prepared = {
        qubit: axis
        for qubit, axis in scan.prepared.items()
        if qubit not in scan.body_touched and qubit not in scan.measured_any
    }
    exit_prepared.update(scan.late_resets)
    exit_dirty = scan.dirty
    exit_dirty.update(dict.fromkeys(segment.unreset_measured, segment.index))
    return _SegmentRewrite(
        circuit=circuit,
        checks=checks,
        exit_prepared=exit_prepared,
        exit_dirty=exit_dirty,
        exit_measurement_index=exit_measurement_index,
    )


@dataclass
class _SegmentScan:
    """Algebraic state of one segment after streaming its instructions."""

    body: _BodyTableau
    prepared: dict[int, Axis]
    dirty: dict[int, int]
    late_resets: dict[int, Axis] = field(default_factory=dict)
    body_touched: set[int] = field(default_factory=set)
    measured_any: set[int] = field(default_factory=set)
    pulled: dict[int, list[_Pauli]] = field(default_factory=dict)


def _scan_segment(
    segment: _SourceSegment,
    num_qubits: int,
    *,
    prepared: dict[int, Axis],
    dirty: dict[int, int],
) -> _SegmentScan | _Conflict:
    """Stream one segment, building its body tableau and pulled products.

    Returns
    -------
    `_SegmentScan` | `_Conflict`
        The collected state, or a trashed-post-state consumption conflict.
    """
    scan = _SegmentScan(body=_BodyTableau(num_qubits), prepared=dict(prepared), dirty=dict(dirty))
    seen_measurement = False
    for position, (instruction, kind) in enumerate(segment.items):
        if kind == "reset":
            axis = RESET_AXES[instruction.name]
            for target in instruction.targets_copy():
                qubit = _plain_qubit(target, instruction.name)
                if seen_measurement:
                    scan.late_resets[qubit] = axis
                else:
                    scan.prepared[qubit] = axis
                scan.dirty.pop(qubit, None)
        elif kind == "unitary":
            conflict = _scan_unitary(scan, instruction)
            if conflict is not None:
                return conflict
        elif kind == "measurement":
            seen_measurement = True
            supports = {qubit for source in segment.sources[position] for qubit in source.observable.pauli_indices()}
            conflict = _consumed_dirty(supports, scan.dirty, instruction.name, remeasured=True)
            if conflict is not None:
                return conflict
            scan.measured_any |= supports
            scan.pulled[position] = [
                scan.body.pull(_pauli_from_stim(source.observable)) for source in segment.sources[position]
            ]
        elif kind == "mpad":
            seen_measurement = True
    return scan


def _scan_unitary(scan: _SegmentScan, instruction: stim.CircuitInstruction) -> _Conflict | None:
    """Fold one unitary instruction into the body tableau.

    Returns
    -------
    `_Conflict` | `None`
        A trashed-post-state consumption conflict, or `None`.
    """
    qubits = {int(target.qubit_value) for target in instruction.targets_copy() if target.qubit_value is not None}
    conflict = _consumed_dirty(qubits, scan.dirty, instruction.name, remeasured=False)
    if conflict is not None:
        return conflict
    for group in instruction.target_groups():
        scan.body.append(
            instruction.name,
            [_plain_qubit(target, instruction.name) for target in group],
        )
    scan.body_touched |= qubits
    return None


def _consumed_dirty(
    qubits: set[int], dirty: dict[int, int], instruction_name: str, *, remeasured: bool
) -> _Conflict | None:
    """Report a touch of a qubit whose post-state a rewritten segment trashed.

    Returns
    -------
    `_Conflict` | `None`
        The conflict, or `None` when no trashed qubit is touched.
    """
    consumed = sorted(qubits & dirty.keys())
    if not consumed:
        return None
    if remeasured:
        msg = (
            f"{instruction_name} measures qubit(s) {consumed} whose earlier measurement "
            "post-state was never reset; correlated re-measurement is not supported."
        )
    else:
        msg = (
            f"{instruction_name} acts on measured-but-not-reset qubit(s) {consumed}; "
            "reusing measurement post-states is not supported."
        )
    return _Conflict(producer=dirty[consumed[0]], message=msg)


def _substituted_products(
    segment: _SourceSegment,
    pulled: dict[int, list[_Pauli]],
    prepared: dict[int, Axis],
    num_qubits: int,
    *,
    allow_negative_pads: bool,
) -> dict[int, list[_Pauli]] | None:
    r"""Substitute fresh-ancilla stabilizers into the pulled products.

    Substitution on a prepared, measured-out qubit is applied only when every
    product's factor on that qubit is the prepared basis or identity, so the
    stabilizer still holds when each product is measured. The substituted
    products of one instruction must commute pairwise; otherwise the
    unsubstituted products are used, and if those also fail (anticommuting
    source products or a disallowed negative pad) the segment is not
    representable.

    Returns
    -------
    `dict`\[`int`, `list`\[`_Pauli`\]\] | `None`
        Products per measurement item position, or `None`.
    """
    substitutable = _substitutable_qubits(segment, pulled, prepared, num_qubits)
    for qubits in (substitutable, frozenset[int]()):
        candidate = {
            position: [_without_stabilizer_factors(product, qubits, num_qubits) for product in products]
            for position, products in pulled.items()
        }
        if _products_admissible(candidate, allow_negative_pads=allow_negative_pads):
            return candidate
        if not qubits:
            return None
    return None


def _substitutable_qubits(
    segment: _SourceSegment,
    pulled: dict[int, list[_Pauli]],
    prepared: dict[int, Axis],
    num_qubits: int,
) -> frozenset[int]:
    r"""Return the prepared, measured-out qubits whose stabilizer can be dropped.

    Returns
    -------
    `frozenset`\[`int`\]
        Qubits satisfying the substitution precondition.
    """
    candidates = set(prepared) & set(segment.measured_bases)
    for products in pulled.values():
        for product in products:
            for qubit in list(candidates):
                x_bit, z_bit = _AXIS_BITS[prepared[qubit]]
                factor = (
                    int(product.bits[qubit]),
                    int(product.bits[num_qubits + qubit]),
                )
                if factor not in {(0, 0), (x_bit, z_bit)}:
                    candidates.discard(qubit)
    return frozenset(candidates)


def _without_stabilizer_factors(product: _Pauli, qubits: frozenset[int], num_qubits: int) -> _Pauli:
    """Divide the prepared stabilizer factors on the given qubits out of a product.

    Returns
    -------
    `_Pauli`
        The reduced product.
    """
    bits = product.bits.copy()
    phase = product.phase
    for qubit in qubits:
        if bits[qubit] or bits[num_qubits + qubit]:
            if bits[qubit] and bits[num_qubits + qubit]:
                phase = (phase - 1) % 4
            bits[qubit] = False
            bits[num_qubits + qubit] = False
    return _Pauli(bits=bits, phase=phase)


def _products_admissible(products: dict[int, list[_Pauli]], *, allow_negative_pads: bool) -> bool:
    """Check pairwise commutation per instruction and the negative-pad rule.

    Returns
    -------
    `bool`
        Whether the products can be emitted.
    """
    for instruction_products in products.values():
        if len(instruction_products) > 1 and _anticommutation_matrix(instruction_products).any():
            return False
        for product in instruction_products:
            if not product.bits.any() and product.phase % 4 == _MINUS_PHASE and not allow_negative_pads:
                return False
    return True


def _anticommutation_matrix(paulis: Sequence[_Pauli]) -> NDArray[np.uint8]:
    r"""Return the pairwise anticommutation parities of a list of Paulis.

    Returns
    -------
    ``NDArray``\[``np.uint8``\]
        Symmetric 0/1 matrix with a zero diagonal.
    """
    half = paulis[0].bits.shape[0] // 2
    x_bits = np.array([pauli.bits[:half] for pauli in paulis], dtype=np.uint8)
    z_bits = np.array([pauli.bits[half:] for pauli in paulis], dtype=np.uint8)
    return (x_bits @ z_bits.T + z_bits @ x_bits.T) % 2


def _residual_frame(
    body: _BodyTableau,
    body_touched: set[int],
    segment: _SourceSegment,
    num_qubits: int,
) -> stim.Circuit | None:
    """Build the residual Clifford frame left on qubits that were not measured out.

    The frame is the body tableau restricted to the surviving qubits. A
    component on a measured-out qubit is absorbed by that qubit's measurement
    only when it anticommutes with every measurement basis of the qubit, and
    the restricted generators must satisfy the canonical symplectic relations;
    otherwise the segment is not MPP-representable.

    Returns
    -------
    ``stim.Circuit`` | `None`
        The residual frame circuit (possibly empty), or `None`.
    """
    if body.gate_count == 0:
        return stim.Circuit()
    survivors = sorted(body_touched - set(segment.measured_bases))
    if not survivors:
        return stim.Circuit()
    survivor_index = {qubit: local for local, qubit in enumerate(survivors)}
    xs: list[stim.PauliString] = []
    zs: list[stim.PauliString] = []
    for offset, collector in ((0, xs), (num_qubits, zs)):
        for qubit in survivors:
            forward = body.forward_generator(qubit + offset)
            restricted = _restrict_forward_image(forward, survivor_index, segment.measured_bases, num_qubits)
            if restricted is None:
                return None
            collector.append(restricted)
    if not _symplectic_generators(xs, zs):
        return None
    tableau = stim.Tableau.from_conjugated_generators(xs=xs, zs=zs)
    if tableau == stim.Tableau(len(survivors)):
        return stim.Circuit()
    return _remap_tableau_circuit(tableau, survivors)


def _restrict_forward_image(
    forward: _Pauli,
    survivor_index: dict[int, int],
    measured_bases: dict[int, set[Axis]],
    num_qubits: int,
) -> stim.PauliString | None:
    """Restrict a forward generator image to the surviving qubits.

    Returns
    -------
    ``stim.PauliString`` | `None`
        The restricted image, or `None` when a measured-out component fails to
        anticommute with a measurement basis of its qubit.
    """
    size = len(survivor_index)
    bits = np.zeros(2 * size, dtype=np.bool_)
    phase = forward.phase
    supports = np.flatnonzero(forward.bits[:num_qubits] | forward.bits[num_qubits:])
    for qubit_index in supports:
        qubit = int(qubit_index)
        x_bit = bool(forward.bits[qubit])
        z_bit = bool(forward.bits[num_qubits + qubit])
        local = survivor_index.get(qubit)
        if local is not None:
            bits[local] = x_bit
            bits[size + local] = z_bit
            continue
        axis = next(axis for axis, axis_bits in _AXIS_BITS.items() if axis_bits == (int(x_bit), int(z_bit)))
        if axis in measured_bases[qubit]:
            return None
        if x_bit and z_bit:
            phase = (phase - 1) % 4
    return _pauli_to_stim(_Pauli(bits=bits, phase=phase % 4))


def _symplectic_generators(xs: Sequence[stim.PauliString], zs: Sequence[stim.PauliString]) -> bool:
    """Check that restricted generator images satisfy the canonical symplectic relations.

    Returns
    -------
    `bool`
        Whether ``xs`` and ``zs`` define a valid Clifford tableau.
    """
    size = len(xs)
    generators = [_pauli_from_stim(pauli) for pauli in [*xs, *zs]]
    canonical = np.zeros((2 * size, 2 * size), dtype=np.uint8)
    for row in range(size):
        canonical[row, row + size] = 1
        canonical[row + size, row] = 1
    return np.array_equal(_anticommutation_matrix(generators), canonical)


def _emit_segment(
    segment: _SourceSegment,
    products: dict[int, list[_Pauli]],
    frame: stim.Circuit,
    *,
    measurement_index: int,
    late_resets: dict[int, Axis],
) -> tuple[stim.Circuit, tuple[CheckMapping, ...], int]:
    r"""Emit one rewritten segment.

    Returns
    -------
    `tuple`\[``stim.Circuit``, `tuple`\[`CheckMapping`, ...\], `int`\]
        The segment circuit, its check mappings, and the next measurement index.
    """
    output = stim.Circuit()
    checks: list[CheckMapping] = []
    seen_measurement = False
    for position, (instruction, kind) in enumerate(segment.items):
        if kind == "unitary":
            continue
        if kind in {"annotation", "reset"}:
            output.append(instruction)
            continue
        if kind == "mpad":
            output.append(instruction)
            measurement_index += instruction.num_measurements
            seen_measurement = True
            continue
        if seen_measurement:
            output.append("TICK", [])
        seen_measurement = True
        measurement_index = _emit_measurement_instruction(
            output,
            checks,
            instruction,
            sources=segment.sources[position],
            products=products[position],
            segment_index=segment.index,
            measurement_index=measurement_index,
            late_resets=late_resets,
        )
    output += frame
    return output, tuple(checks), measurement_index


def _emit_measurement_instruction(  # ruff:ignore[too-many-arguments]
    output: stim.Circuit,
    checks: list[CheckMapping],
    instruction: stim.CircuitInstruction,
    *,
    sources: tuple[_SourceObservable, ...],
    products: list[_Pauli],
    segment_index: int,
    measurement_index: int,
    late_resets: dict[int, Axis],
) -> int:
    """Emit one measurement instruction as its source form or inferred products.

    Returns
    -------
    `int`
        The next measurement-record index.
    """
    trivial = all(
        np.array_equal(product.bits, source_pauli.bits) and product.phase == source_pauli.phase
        for product, source_pauli in zip(
            products,
            (_pauli_from_stim(source.observable) for source in sources),
            strict=True,
        )
    )
    if trivial:
        output.append(instruction)
    else:
        _append_products(output, products, tag=instruction.tag)
    if instruction.name in MEASURE_RESET_AXES:
        axis = MEASURE_RESET_AXES[instruction.name]
        qubits = [_plain_qubit(target, instruction.name) for target in instruction.targets_copy()]
        if not trivial:
            output.append(RESET_GATES[axis], qubits, [], tag=instruction.tag)
        for qubit in qubits:
            late_resets[qubit] = axis
    for product, source in zip(products, sources, strict=True):
        checks.append(
            CheckMapping(
                measurement_index=measurement_index,
                segment_index=segment_index,
                product=_pauli_to_stim(product),
                source_qubit=source.source_qubit,
            )
        )
        measurement_index += 1
    return measurement_index


def _append_products(output: stim.Circuit, products: Sequence[_Pauli], *, tag: str) -> None:
    """Append inferred products, using MPAD for deterministic identities."""
    with_support = [product.bits.any() for product in products]
    if all(with_support):
        targets: list[stim.GateTarget] = []
        for product in products:
            # The stim stub mistypes the return as a single GateTarget; the
            # runtime returns a list of targets with combiners.
            targets.extend(
                cast(
                    "list[stim.GateTarget]",
                    stim.target_combined_paulis(_pauli_to_stim(product)),
                )
            )
        output.append("MPP", targets, [], tag=tag)
        return
    for product, has_support in zip(products, with_support, strict=True):
        if has_support:
            product_targets = cast(
                "list[stim.GateTarget]",
                stim.target_combined_paulis(_pauli_to_stim(product)),
            )
            output.append("MPP", product_targets, [], tag=tag)
        else:
            output.append("MPAD", [int(product.phase % 4 == _MINUS_PHASE)], [], tag=tag)


def _emit_verbatim(
    segment: _SourceSegment,
    *,
    prepared: dict[int, Axis],
    dirty: dict[int, int],
    measurement_index: int,
) -> _SegmentRewrite | _Conflict:
    """Emit one segment gate-level with exit-state bookkeeping only.

    Returns
    -------
    `_SegmentRewrite` | `_Conflict`
        The verbatim segment, or a trashed-post-state consumption conflict.
    """
    output = stim.Circuit()
    checks: list[CheckMapping] = []
    exit_prepared = dict(prepared)
    exit_dirty = dict(dirty)
    for position, (instruction, kind) in enumerate(segment.items):
        output.append(instruction)
        if kind == "reset":
            for target in instruction.targets_copy():
                qubit = _plain_qubit(target, instruction.name)
                exit_prepared[qubit] = RESET_AXES[instruction.name]
                exit_dirty.pop(qubit, None)
        elif kind == "unitary":
            qubits = {
                int(target.qubit_value) for target in instruction.targets_copy() if target.qubit_value is not None
            }
            conflict = _consumed_dirty(qubits, exit_dirty, instruction.name, remeasured=False)
            if conflict is not None:
                return conflict
            for qubit in qubits:
                exit_prepared.pop(qubit, None)
        elif kind == "measurement":
            outcome = _verbatim_measurement(
                checks,
                instruction,
                sources=segment.sources[position],
                segment_index=segment.index,
                measurement_index=measurement_index,
                prepared=exit_prepared,
                dirty=exit_dirty,
            )
            if isinstance(outcome, _Conflict):
                return outcome
            measurement_index = outcome
        elif kind == "mpad":
            measurement_index += instruction.num_measurements
    return _SegmentRewrite(
        circuit=output,
        checks=tuple(checks),
        exit_prepared=exit_prepared,
        exit_dirty=exit_dirty,
        exit_measurement_index=measurement_index,
    )


def _verbatim_measurement(  # ruff:ignore[too-many-arguments]
    checks: list[CheckMapping],
    instruction: stim.CircuitInstruction,
    *,
    sources: tuple[_SourceObservable, ...],
    segment_index: int,
    measurement_index: int,
    prepared: dict[int, Axis],
    dirty: dict[int, int],
) -> int | _Conflict:
    """Track one verbatim measurement's mappings and prepared-basis effects.

    Returns
    -------
    `int` | `_Conflict`
        The next measurement-record index, or a consumption conflict.
    """
    supports = {qubit for source in sources for qubit in source.observable.pauli_indices()}
    conflict = _consumed_dirty(supports, dirty, instruction.name, remeasured=True)
    if conflict is not None:
        return conflict
    for qubit in supports:
        prepared.pop(qubit, None)
    for source in sources:
        checks.append(
            CheckMapping(
                measurement_index=measurement_index,
                segment_index=segment_index,
                product=source.observable,
                source_qubit=source.source_qubit,
            )
        )
        measurement_index += 1
    if instruction.name in MEASURE_RESET_AXES:
        for target in instruction.targets_copy():
            prepared[_plain_qubit(target, instruction.name)] = MEASURE_RESET_AXES[instruction.name]
    return measurement_index


def _passthrough_result(circuit: stim.Circuit, segments: Sequence[_SourceSegment]) -> MppRewriteResult:
    """Return the flattened source circuit as a whole-circuit fallback result.

    Returns
    -------
    `MppRewriteResult`
        The unchanged circuit with pass-through check mappings.
    """
    checks: list[CheckMapping] = []
    measurement_index = 0
    for segment in segments:
        for position, (instruction, kind) in enumerate(segment.items):
            if kind == "mpad":
                measurement_index += instruction.num_measurements
            elif kind == "measurement":
                for source in segment.sources[position]:
                    checks.append(
                        CheckMapping(
                            measurement_index=measurement_index,
                            segment_index=segment.index,
                            product=source.observable,
                            source_qubit=source.source_qubit,
                        )
                    )
                    measurement_index += 1
    return MppRewriteResult(
        circuit=circuit,
        checks=tuple(checks),
        fallback_segments=tuple(segment.index for segment in segments),
    )


def _instruction_kind(instruction: stim.CircuitInstruction) -> _InstructionKind:
    """Classify one instruction for segmentation.

    Returns
    -------
    `_InstructionKind`
        The instruction class.

    Raises
    ------
    UnsupportedSyndromeCircuitError
        If the instruction is outside the supported basis.
    """
    name = instruction.name
    if name in ANNOTATION_GATES:
        return "annotation"
    if name == "MPAD":
        return "mpad"
    if name in RESET_AXES:
        return "reset"
    if name == "MPP" or name in DIRECT_MEASUREMENT_AXES or name in PAIR_MEASUREMENT_AXES:
        return "measurement"
    if stim.gate_data(name).is_unitary:
        return "unitary"
    msg = f"Unsupported instruction for MPP rewriting: {name}."
    raise UnsupportedSyndromeCircuitError(msg)


def _plain_qubit(target: stim.GateTarget, instruction_name: str) -> int:
    """Return the Stim qubit id a target acts on.

    Returns
    -------
    `int`
        Stim qubit id.

    Raises
    ------
    UnsupportedSyndromeCircuitError
        If the target does not act on a qubit.
    """
    qubit_value = target.qubit_value
    if qubit_value is None:
        msg = f"{instruction_name} contains unsupported target {target!r}; only qubit targets are supported."
        raise UnsupportedSyndromeCircuitError(msg)
    return int(qubit_value)


def _remap_tableau_circuit(tableau: stim.Tableau, qubits: Sequence[int]) -> stim.Circuit:
    """Synthesize a tableau and map its dense qubit indices to Stim qubit ids.

    Returns
    -------
    ``stim.Circuit``
        Synthesized Clifford circuit on qubits.

    Raises
    ------
    TypeError
        If Stim unexpectedly synthesizes a repeat block.
    """
    result = stim.Circuit()
    for instruction in tableau.to_circuit(method="elimination"):
        if isinstance(instruction, stim.CircuitRepeatBlock):  # pragma: no cover - synthesis emits no repeats
            msg = "Stim unexpectedly synthesized a REPEAT block for the residual Clifford frame."
            raise TypeError(msg)
        targets = [qubits[_plain_qubit(target, instruction.name)] for target in instruction.targets_copy()]
        result.append(instruction.name, targets, instruction.gate_args_copy(), tag=instruction.tag)
    return result


def _measurement_observables(instruction: stim.CircuitInstruction, num_qubits: int) -> list[_SourceObservable]:
    r"""Extract the measured observables of one measurement instruction.

    Returns
    -------
    `list`\[`_SourceObservable`\]
        One observable per measurement record, in record order.
    """
    name = instruction.name
    if name in DIRECT_MEASUREMENT_AXES:
        basis = DIRECT_MEASUREMENT_AXES[name]
        return [_single_qubit_observable(group, basis, name, num_qubits) for group in instruction.target_groups()]
    if name in PAIR_MEASUREMENT_AXES:
        basis = PAIR_MEASUREMENT_AXES[name]
        return [_pair_observable(group, basis, name, num_qubits) for group in instruction.target_groups()]
    return [_mpp_observable(group, num_qubits) for group in instruction.target_groups()]


def _single_qubit_observable(
    group: Sequence[stim.GateTarget], basis: Axis, name: str, num_qubits: int
) -> _SourceObservable:
    """Build the observable of one single-qubit measurement target.

    Returns
    -------
    `_SourceObservable`
        The measured observable.
    """
    (target,) = group
    qubit = _plain_qubit(target, name)
    observable = stim.PauliString(num_qubits)
    observable[qubit] = _pauli_code(basis)
    if target.is_inverted_result_target:
        observable.sign = -1
    return _SourceObservable(observable=observable, source_qubit=qubit)


def _pair_observable(group: Sequence[stim.GateTarget], basis: Axis, name: str, num_qubits: int) -> _SourceObservable:
    """Build the observable of one pair-measurement target group.

    Returns
    -------
    `_SourceObservable`
        The measured observable.

    Raises
    ------
    UnsupportedSyndromeCircuitError
        If the group is not a pair of distinct qubits.
    """
    if len(group) != _PAIR_GROUP_SIZE:
        msg = f"{name} expects qubit pairs."
        raise UnsupportedSyndromeCircuitError(msg)
    observable = stim.PauliString(num_qubits)
    sign = 1
    for target in group:
        qubit = _plain_qubit(target, name)
        if observable[qubit] != 0:
            msg = f"{name} pairs the same qubit {qubit} with itself."
            raise UnsupportedSyndromeCircuitError(msg)
        observable[qubit] = _pauli_code(basis)
        if target.is_inverted_result_target:
            sign = -sign
    observable.sign = sign
    return _SourceObservable(observable=observable, source_qubit=None)


def _mpp_observable(group: Sequence[stim.GateTarget], num_qubits: int) -> _SourceObservable:
    """Build the observable of one MPP product group.

    Returns
    -------
    `_SourceObservable`
        The measured observable.

    Raises
    ------
    UnsupportedSyndromeCircuitError
        If the product contains a non-Pauli target or is not Hermitian.
    """
    observable = stim.PauliString(num_qubits)
    for target in group:
        qubit = _plain_qubit(target, "MPP")
        pauli = target.pauli_type
        if pauli not in Axis.__members__:
            msg = f"MPP contains a non-Pauli target on qubit {qubit}."
            raise UnsupportedSyndromeCircuitError(msg)
        factor = stim.PauliString({qubit: _pauli_code(Axis[pauli])})
        if target.is_inverted_result_target:
            factor.sign = -1
        observable *= factor
    if observable.sign not in {1, -1}:
        msg = f"Non-Hermitian measurement product: {observable}."
        raise UnsupportedSyndromeCircuitError(msg)
    return _SourceObservable(observable=observable, source_qubit=None)


def _pauli_code(axis: Axis) -> int:
    """Return the Stim numeric Pauli code of an axis.

    Returns
    -------
    `int`
        1 for X, 2 for Y, 3 for Z.
    """
    return {Axis.X: 1, Axis.Y: 2, Axis.Z: 3}[axis]
