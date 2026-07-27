"""Rewrite Stim syndrome-extraction circuits into abstract MPP measurements.

This experimental module recognizes the standard syndrome-extraction shape
(reset ancillas, apply Clifford unitaries, measure ancillas) and replaces each
gate-level extraction block with ``MPP`` instructions that directly measure the
inferred data Pauli products. Circuit-level noise and classical feedback remain
unsupported.

This module provides:

- `rewrite_to_mpp`: Rewrite a noiseless syndrome-extraction circuit into MPP form.
- `MppRewriteResult`: Rewritten circuit with its per-measurement Pauli products.
- `CheckMapping`: Mapping from one measurement record to its Pauli product.
- `UnsupportedSyndromeCircuitError`: Error for unsupported syndrome circuits.
- `MppRewriteVerificationError`: Error for a failed segment verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Literal, cast

import stim

from graphqomb.stim_glue._parse import (
    MEASURE_RESET_AXES,
    PAIR_MEASUREMENT_AXES,
    RESET_AXES,
    RESET_GATES,
    SINGLE_MEASUREMENT_AXES,
    iter_instructions,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_ANNOTATION_GATES = frozenset({"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "TICK"})
# Pauli bases are spelled as Stim's "X"/"Y"/"Z" letters inside this module.
_RESET_BASES = {name: axis.name for name, axis in RESET_AXES.items()}
_SINGLE_MEASUREMENT_BASES = {name: axis.name for name, axis in SINGLE_MEASUREMENT_AXES.items()}
_MEASURE_RESET_GATES = {name: (axis.name, RESET_GATES[axis]) for name, axis in MEASURE_RESET_AXES.items()}
_PAIR_MEASUREMENT_BASES = {name: axis.name for name, axis in PAIR_MEASUREMENT_AXES.items()}
_RESET_GATES_BY_BASIS = {axis.name: gate for axis, gate in RESET_GATES.items()}
_PAULI_CODES = {"X": 1, "Y": 2, "Z": 3}
_PAIR_GROUP_SIZE = 2

_InstructionKind = Literal["annotation", "mpad", "reset", "measurement", "unitary"]


class UnsupportedSyndromeCircuitError(ValueError):
    """Raised when a circuit is outside the supported syndrome-extraction form."""


class MppRewriteVerificationError(ValueError):
    """Raised when a rewritten segment fails stabilizer-flow verification."""


class _ResidualFrameSynthesisError(RuntimeError):
    """Raised when the removed body cannot be represented by a residual Clifford frame."""


@dataclass(frozen=True)
class CheckMapping:
    """Sidecar mapping from one source measurement to its inferred Pauli product.

    Attributes
    ----------
    measurement_index : `int`
        Global measurement-record index, identical in source and rewritten circuits.
    segment_index : `int`
        Index of the extraction segment that produced this measurement.
    product : ``stim.PauliString``
        Inferred signed Pauli product measured at this record position.
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
    """

    circuit: stim.Circuit
    checks: tuple[CheckMapping, ...]


@dataclass(frozen=True)
class _SourceObservable:
    observable: stim.PauliString
    source_qubit: int | None


@dataclass
class _SegmentBounds:
    """Segment-boundary state shared by the rewriter and the fallback mapper.

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
            if instruction.name in _MEASURE_RESET_GATES:
                self.seen_late_reset = True
        elif kind == "reset" and self.seen_measurement:
            self.seen_late_reset = True
        return boundary


def rewrite_to_mpp(circuit: stim.Circuit | str) -> MppRewriteResult:
    """Rewrite a noiseless Stim syndrome-extraction circuit into MPP form.

    The inference conjugates each final measurement Pauli backwards through the
    segment's Clifford body (``U† P U`` via ``stim.PauliString.before``) and
    substitutes ``+1`` for stabilizers of freshly initialized, measured-out
    ancillas. A segment ends when a unitary follows its measurements, or when a
    measurement follows a reset issued after those measurements (including the
    implicit reset of ``MR``); trailing data measurements therefore start a
    fresh segment with an empty Clifford body and pass through verbatim.
    Clifford frames left on surviving qubits are retained after the inferred
    measurements. Each source measurement maps to exactly one measurement in
    the rewritten circuit, so ``DETECTOR`` and ``OBSERVABLE_INCLUDE``
    annotations are copied verbatim.

    ``REPEAT`` blocks are flattened before rewriting, which also bakes
    ``SHIFT_COORDS`` offsets into ``DETECTOR`` coordinate arguments.

    Every rewritten segment is cross-checked against its source by stabilizer-
    flow generators, with measured-out ancilla post-states reset in both copies
    before comparison. If a segment cannot be represented by MPP measurements
    plus a residual Clifford frame, or fails that check, the rewriter preserves
    the gate-level circuit and replaces reset-based qubit reuse with fresh Stim
    qubit ids. The check is not optional: it is what selects between the
    optimized rewrite, the unsubstituted retry, and the gate-level fallback.

    Parameters
    ----------
    circuit : ``stim.Circuit`` | `str`
        Source circuit or Stim circuit text. Noise instructions, measurement
        noise arguments, and classical feedback are rejected with
        `UnsupportedSyndromeCircuitError`.

    Returns
    -------
    `MppRewriteResult`
        Rewritten circuit and per-measurement Pauli-product mappings.
    """
    source = circuit if isinstance(circuit, stim.Circuit) else stim.Circuit(circuit)
    flattened = source.flattened()
    try:
        return _rewrite_flattened_to_mpp(flattened)
    except (MppRewriteVerificationError, _ResidualFrameSynthesisError):
        fallback = _split_reset_lifetimes(flattened)
        return MppRewriteResult(circuit=fallback, checks=_check_mappings(fallback))


def _rewrite_flattened_to_mpp(flattened: stim.Circuit) -> MppRewriteResult:
    """Rewrite a flattened circuit through the optimized MPP path.

    Returns
    -------
    `MppRewriteResult`
        Optimized rewrite result.

    Raises
    ------
    UnsupportedSyndromeCircuitError
        If the circuit is outside the supported syndrome-extraction form.
    RuntimeError
        If the rewrite changes the measurement count.
    """
    rewriter = _Rewriter(num_qubits=flattened.num_qubits)
    for instruction in flattened:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            msg = "REPEAT blocks must be flattened before rewriting."
            raise UnsupportedSyndromeCircuitError(msg)
        rewriter.process(instruction)
    result = rewriter.finish()
    if result.circuit.num_measurements != flattened.num_measurements:
        msg = "MPP rewrite changed the measurement count; this is a bug."
        raise RuntimeError(msg)
    return result


def _split_reset_lifetimes(circuit: stim.Circuit) -> stim.Circuit:
    """Replace reset-based qubit reuse with fresh Stim qubit ids.

    Returns
    -------
    ``stim.Circuit``
        Equivalent circuit where every reset after prior quantum use starts a
        fresh wire. Old post-measurement wires remain unused.
    """
    result = stim.Circuit()
    next_qubit = circuit.num_qubits
    current_qubit = {qubit: qubit for qubit in range(circuit.num_qubits)}
    used_qubits: set[int] = set()

    for instruction in iter_instructions(circuit):
        if instruction.name == "QUBIT_COORDS":
            coordinate_targets = [
                mapped
                for target in instruction.targets_copy()
                if (mapped := current_qubit[_plain_qubit(target, instruction.name)]) < circuit.num_qubits
            ]
            if coordinate_targets:
                result.append(instruction.name, coordinate_targets, instruction.gate_args_copy(), tag=instruction.tag)
            continue

        if instruction.name == "MPAD":
            # MPAD pad bits are spelled like qubit targets, so they must never
            # be routed through the lifetime map.
            result.append(instruction)
            continue

        targets = instruction.targets_copy()
        if instruction.name in _RESET_BASES:
            reset_targets: list[int] = []
            for target in targets:
                source_qubit = _plain_qubit(target, instruction.name)
                if source_qubit in used_qubits:
                    current_qubit[source_qubit] = next_qubit
                    next_qubit += 1
                used_qubits.add(source_qubit)
                reset_targets.append(current_qubit[source_qubit])
            result.append(instruction.name, reset_targets, instruction.gate_args_copy(), tag=instruction.tag)
            continue

        result.append(
            instruction.name,
            [_remap_gate_target(target, current_qubit) for target in targets],
            instruction.gate_args_copy(),
            tag=instruction.tag,
        )
        used_qubits.update(int(target.qubit_value) for target in targets if target.qubit_value is not None)
    return result


def _remap_gate_target(target: stim.GateTarget, current_qubit: dict[int, int]) -> stim.GateTarget | int:
    """Map one Stim gate target through the current qubit lifetime.

    Returns
    -------
    ``stim.GateTarget`` | `int`
        Remapped target, or the unchanged non-qubit target.
    """
    qubit_value = target.qubit_value
    if qubit_value is None:
        return target
    qubit = current_qubit[int(qubit_value)]
    if target.pauli_type == "X":
        return stim.target_x(qubit, invert=target.is_inverted_result_target)
    if target.pauli_type == "Y":
        return stim.target_y(qubit, invert=target.is_inverted_result_target)
    if target.pauli_type == "Z":
        return stim.target_z(qubit, invert=target.is_inverted_result_target)
    if target.is_inverted_result_target:
        return stim.target_inv(qubit)
    return qubit


def _check_mappings(circuit: stim.Circuit) -> tuple[CheckMapping, ...]:
    r"""Build pass-through measurement mappings for a gate-level fallback.

    Returns
    -------
    `tuple`\[`CheckMapping`, ...\]
        One mapping for each Pauli measurement record.
    """
    checks: list[CheckMapping] = []
    measurement_index = 0
    segment_index = 0
    bounds = _SegmentBounds()
    for instruction in iter_instructions(circuit):
        kind = _instruction_kind(instruction)
        if bounds.starts_new_segment(instruction, kind):
            segment_index += 1
        if kind == "mpad":
            measurement_index += instruction.num_measurements
        elif kind == "measurement":
            for source in _measurement_observables(instruction, circuit.num_qubits):
                checks.append(
                    CheckMapping(
                        measurement_index=measurement_index,
                        segment_index=segment_index,
                        product=source.observable,
                        source_qubit=source.source_qubit,
                    )
                )
                measurement_index += 1
    return tuple(checks)


def _instruction_kind(instruction: stim.CircuitInstruction) -> _InstructionKind:
    name = instruction.name
    if name in _ANNOTATION_GATES:
        return "annotation"
    if name == "MPAD":
        return "mpad"
    if name in _RESET_BASES:
        return "reset"
    if (
        name == "MPP"
        or name in _SINGLE_MEASUREMENT_BASES
        or name in _MEASURE_RESET_GATES
        or name in _PAIR_MEASUREMENT_BASES
    ):
        return "measurement"
    if stim.gate_data(name).is_unitary:
        return "unitary"
    msg = f"Unsupported instruction for MPP rewriting: {name}."
    raise UnsupportedSyndromeCircuitError(msg)


def _plain_qubit(target: stim.GateTarget, instruction_name: str) -> int:
    """Return the Stim qubit id a target acts on.

    Unlike ``_parse.plain_qubit_target``, Pauli-typed and inverted-result
    targets are accepted: the rewriter only needs the qubit id here, and it
    reports unsupported targets as `UnsupportedSyndromeCircuitError`.

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


def _check_no_feedback_targets(instruction: stim.CircuitInstruction) -> None:
    for target in instruction.targets_copy():
        if target.is_measurement_record_target or target.is_sweep_bit_target:
            msg = f"Classical feedback is not supported: {instruction.name} with target {target!r}."
            raise UnsupportedSyndromeCircuitError(msg)


class _Rewriter:
    """Streaming rewriter that buffers and rewrites one segment at a time."""

    def __init__(self, *, num_qubits: int) -> None:
        self._num_qubits = num_qubits
        self._output = stim.Circuit()
        self._checks: list[CheckMapping] = []
        self._measurement_index = 0
        self._segment_index = 0
        self._dirty: set[int] = set()
        self._entry_prepared: dict[int, str] = {}
        self._items: list[tuple[stim.CircuitInstruction, _InstructionKind]] = []
        self._bounds = _SegmentBounds()

    def process(self, instruction: stim.CircuitInstruction) -> None:
        """Buffer one instruction, flushing the current segment at a boundary.

        The boundary rule lives in `_SegmentBounds`, so the gate-level fallback
        numbers its segments the same way.
        """
        kind = _instruction_kind(instruction)
        if self._bounds.starts_new_segment(instruction, kind):
            self._flush_segment()
        self._items.append((instruction, kind))

    def finish(self) -> MppRewriteResult:
        """Flush the final segment and return the rewrite result.

        Returns
        -------
        `MppRewriteResult`
            Rewritten circuit and per-measurement Pauli-product mappings.
        """
        self._flush_segment()
        return MppRewriteResult(circuit=self._output, checks=tuple(self._checks))

    def _flush_segment(self) -> None:
        if not self._items:
            return
        try:
            segment = self._rewrite_segment(substitute_prepared=True)
        except (MppRewriteVerificationError, _ResidualFrameSynthesisError):
            segment = self._rewrite_segment(substitute_prepared=False)
        self._output += segment.output
        self._checks.extend(segment.checks)
        self._measurement_index = segment.measurement_index
        self._entry_prepared = segment.exit_prepared()
        self._dirty = segment.exit_dirty()
        self._segment_index += 1
        self._items = []

    def _rewrite_segment(self, *, substitute_prepared: bool) -> _Segment:
        """Rewrite and verify the buffered segment using one ancilla-substitution policy.

        Returns
        -------
        `_Segment`
            Completed segment rewrite.
        """
        segment = _Segment(
            num_qubits=self._num_qubits,
            segment_index=self._segment_index,
            entry_prepared=dict(self._entry_prepared),
            dirty=set(self._dirty),
            measurement_index=self._measurement_index,
            substitute_prepared=substitute_prepared,
        )
        segment.prescan(self._items)
        for instruction, kind in self._items:
            segment.process(instruction, kind)
        segment.append_residual_frame()
        segment.verify()
        return segment


class _Segment:
    """Rewrite state for one reset/Clifford/measure segment of the source circuit."""

    def __init__(  # ruff:ignore[too-many-arguments]
        self,
        *,
        num_qubits: int,
        segment_index: int,
        entry_prepared: dict[int, str],
        dirty: set[int],
        measurement_index: int,
        substitute_prepared: bool,
    ) -> None:
        self._num_qubits = num_qubits
        self._segment_index = segment_index
        self._prepared = entry_prepared
        self._dirty = dirty
        self._substitute_prepared = substitute_prepared
        self.measurement_index = measurement_index
        self.output = stim.Circuit()
        self.checks: list[CheckMapping] = []
        self._body = stim.Circuit()
        self._body_touched: set[int] = set()
        self._seen_measurement = False
        self._late_resets: dict[int, str] = {}
        self._measured_out: set[int] = set()
        self._measured_any: set[int] = set()
        self._measured_unreset: set[int] = set()
        # Source and rewritten copies of this segment as standalone channels.
        # Both residual-frame synthesis and `verify` compare their stabilizer
        # flows, so they are always mirrored while the segment is rewritten.
        self._orig_verify = stim.Circuit()
        self._conv_verify = stim.Circuit()
        for qubit, basis in sorted(entry_prepared.items()):
            for target_circuit in (self._orig_verify, self._conv_verify):
                target_circuit.append(_RESET_GATES_BY_BASIS[basis], [qubit])

    def process(self, instruction: stim.CircuitInstruction, kind: _InstructionKind) -> None:
        """Rewrite one buffered instruction of this segment."""
        if kind == "annotation":
            self.output.append(instruction)
        elif kind == "mpad":
            self._process_mpad(instruction)
        elif kind == "reset":
            self._process_reset(instruction)
        elif kind == "unitary":
            self._process_unitary(instruction)
        else:
            self._process_measurement(instruction)

    def prescan(self, items: Sequence[tuple[stim.CircuitInstruction, _InstructionKind]]) -> None:
        """Collect the segment-wide set of measured-out qubits before rewriting."""
        for instruction, kind in items:
            if kind != "measurement":
                continue
            name = instruction.name
            if name in _SINGLE_MEASUREMENT_BASES or name in _MEASURE_RESET_GATES:
                for target in instruction.targets_copy():
                    self._measured_out.add(_plain_qubit(target, name))

    def _append_verbatim(self, instruction: stim.CircuitInstruction) -> None:
        self.output.append(instruction)
        self._orig_verify.append(instruction)
        self._conv_verify.append(instruction)

    def _rewrite_targets(self) -> tuple[stim.Circuit, stim.Circuit]:
        r"""Return the circuits every rewritten instruction is appended to.

        Returns
        -------
        `tuple`\[``stim.Circuit``, ``stim.Circuit``\]
            The emitted output and its verification mirror.
        """
        return (self.output, self._conv_verify)

    def _process_mpad(self, instruction: stim.CircuitInstruction) -> None:
        self._append_verbatim(instruction)
        self.measurement_index += len(instruction.targets_copy())
        self._seen_measurement = True

    def _process_reset(self, instruction: stim.CircuitInstruction) -> None:
        basis = _RESET_BASES[instruction.name]
        for target in instruction.targets_copy():
            qubit = _plain_qubit(target, instruction.name)
            if self._seen_measurement:
                self._late_resets[qubit] = basis
            elif qubit in self._body_touched:
                msg = f"Reset on qubit {qubit} after it was entangled in the same segment is not supported."
                raise UnsupportedSyndromeCircuitError(msg)
            else:
                self._prepared[qubit] = basis
            self._dirty.discard(qubit)
            self._measured_unreset.discard(qubit)
        self._append_verbatim(instruction)

    def _process_unitary(self, instruction: stim.CircuitInstruction) -> None:
        _check_no_feedback_targets(instruction)
        touched = {int(target.qubit_value) for target in instruction.targets_copy() if target.qubit_value is not None}
        dirty_touched = sorted(touched & self._dirty)
        if dirty_touched:
            msg = (
                f"{instruction.name} acts on measured-but-not-reset qubit(s) {dirty_touched}; "
                "reusing measurement post-states is not supported."
            )
            raise UnsupportedSyndromeCircuitError(msg)
        self._body.append(instruction)
        self._orig_verify.append(instruction)
        self._body_touched |= touched

    def _process_measurement(self, instruction: stim.CircuitInstruction) -> None:
        if instruction.gate_args_copy():
            msg = f"Noisy measurement {instruction.name} with arguments is not supported."
            raise UnsupportedSyndromeCircuitError(msg)
        sources = _measurement_observables(instruction, self._num_qubits)
        self._validate_measured_qubits(instruction, sources)
        separate_from_previous = self._seen_measurement
        self._seen_measurement = True
        products = [self._infer_product(source) for source in sources]
        if any(not left.commutes(right) for left, right in combinations(products, 2)):
            msg = f"Segment {self._segment_index} ancilla substitution made commuting source measurements anticommute."
            raise _ResidualFrameSynthesisError(msg)
        for product, source in zip(products, sources, strict=True):
            self.checks.append(
                CheckMapping(
                    measurement_index=self.measurement_index,
                    segment_index=self._segment_index,
                    product=product,
                    source_qubit=source.source_qubit,
                )
            )
            self.measurement_index += 1
        if separate_from_previous:
            self.output.append("TICK", [])
            self._conv_verify.append("TICK", [])
        self._emit_measurement(instruction, sources, products)
        self._orig_verify.append(instruction)

    def _validate_measured_qubits(
        self, instruction: stim.CircuitInstruction, sources: Sequence[_SourceObservable]
    ) -> None:
        qubits = {qubit for source in sources for qubit in source.observable.pauli_indices()}
        dirty_measured = sorted(qubits & self._dirty)
        if dirty_measured:
            msg = (
                f"{instruction.name} measures qubit(s) {dirty_measured} whose earlier measurement "
                "post-state was never reset; correlated re-measurement is not supported."
            )
            raise UnsupportedSyndromeCircuitError(msg)
        self._measured_any |= qubits
        if instruction.name in _SINGLE_MEASUREMENT_BASES:
            self._measured_unreset |= qubits

    def _infer_product(self, source: _SourceObservable) -> stim.PauliString:
        pulled = source.observable.before(self._body) if len(self._body) else source.observable.copy()
        if not self._substitute_prepared:
            return pulled
        for qubit in pulled.pauli_indices():
            prepared_basis = self._prepared.get(qubit)
            if prepared_basis is None or qubit not in self._measured_out:
                continue
            if _PAULI_CODES[prepared_basis] == pulled[qubit]:
                pulled[qubit] = 0
        if pulled.sign not in {1, -1}:
            msg = f"Non-Hermitian inferred observable: {pulled}."
            raise UnsupportedSyndromeCircuitError(msg)
        return pulled

    def _emit_measurement(
        self,
        instruction: stim.CircuitInstruction,
        sources: Sequence[_SourceObservable],
        products: Sequence[stim.PauliString],
    ) -> None:
        trivial = all(product == source.observable for product, source in zip(products, sources, strict=True))
        if trivial:
            for target_circuit in self._rewrite_targets():
                target_circuit.append(instruction)
        else:
            self._emit_inferred_products(products, tag=instruction.tag)
        if instruction.name in _MEASURE_RESET_GATES:
            basis, reset_gate = _MEASURE_RESET_GATES[instruction.name]
            qubits = [_plain_qubit(target, instruction.name) for target in instruction.targets_copy()]
            if not trivial:
                for target_circuit in self._rewrite_targets():
                    target_circuit.append(reset_gate, qubits, [], tag=instruction.tag)
            for qubit in qubits:
                self._late_resets[qubit] = basis
                self._measured_unreset.discard(qubit)

    def _emit_inferred_products(self, products: Sequence[stim.PauliString], *, tag: str) -> None:
        """Append inferred products, using MPAD for deterministic identities."""
        product_has_support = [bool(product.pauli_indices()) for product in products]
        target_circuits = self._rewrite_targets()
        if all(product_has_support):
            targets: list[stim.GateTarget] = []
            for product in products:
                # The stim stub mistypes the return as a single GateTarget; the
                # runtime returns a list of targets with combiners.
                targets.extend(cast("list[stim.GateTarget]", stim.target_combined_paulis(product)))
            for target_circuit in target_circuits:
                target_circuit.append("MPP", targets, [], tag=tag)
            return

        for product, has_support in zip(products, product_has_support, strict=True):
            if has_support:
                product_targets = cast("list[stim.GateTarget]", stim.target_combined_paulis(product))
                for target_circuit in target_circuits:
                    target_circuit.append("MPP", product_targets, [], tag=tag)
            else:
                pad_bit = int(product.sign == -1)
                for target_circuit in target_circuits:
                    target_circuit.append("MPAD", [pad_bit], [], tag=tag)

    def append_residual_frame(self) -> None:
        """Preserve the Clifford frame left on qubits that were not measured out."""
        if len(self._body) == 0:
            return
        survivors = sorted(self._body_touched - self._measured_out)
        if not survivors:
            return

        try:
            residual_frame = self._residual_frame_from_body(survivors)
        except ValueError:
            residual_frame = self._residual_local_frame_from_flows(survivors)
        if len(residual_frame) == 0:
            return
        self.output += residual_frame
        self._conv_verify += residual_frame

    def _residual_frame_from_body(self, survivors: Sequence[int]) -> stim.Circuit:
        """Return the body tableau restricted to surviving qubits.

        Returns
        -------
        ``stim.Circuit``
            A synthesized residual Clifford circuit.
        """
        xs = [self._restricted_body_output(qubit, "X", survivors) for qubit in survivors]
        zs = [self._restricted_body_output(qubit, "Z", survivors) for qubit in survivors]
        tableau = stim.Tableau.from_conjugated_generators(xs=xs, zs=zs)
        if tableau == stim.Tableau(len(survivors)):
            return stim.Circuit()
        return _remap_tableau_circuit(tableau, survivors)

    def _residual_local_frame_from_flows(self, survivors: Sequence[int]) -> stim.Circuit:
        """Infer a tensor product of one-qubit residual Cliffords from channel flows.

        Returns
        -------
        ``stim.Circuit``
            A local Clifford circuit satisfying all matched flow generators.
        """
        original, converted = self._padded_channels()
        pairs = _matched_flow_output_pairs(original, converted, survivors, self._segment_index)
        axis_maps = _local_axis_maps(pairs, len(survivors), self._segment_index)
        local_tableaus = [_unsigned_local_tableau(axis_map, self._segment_index) for axis_map in axis_maps]
        equations = _local_sign_equations(pairs, local_tableaus, self._segment_index)
        return _materialize_local_frame(survivors, local_tableaus, _solve_binary_equations(equations))

    def _padded_channels(self) -> tuple[stim.Circuit, stim.Circuit]:
        r"""Return the source and rewritten channels with ancilla post-states reset.

        Resetting the measured-out ancillas in both copies models the
        optimized-rewrite semantics: the rewrite is only required to agree with
        its source on the qubits that survive the segment.

        Returns
        -------
        `tuple`\[``stim.Circuit``, ``stim.Circuit``\]
            Padded copies of the source and rewritten channels.
        """
        padding = sorted(self._measured_out)
        original = self._orig_verify.copy()
        converted = self._conv_verify.copy()
        if padding:
            original.append("R", padding)
            converted.append("R", padding)
        return original, converted

    def _restricted_body_output(
        self,
        qubit: int,
        basis: Literal["X", "Z"],
        survivors: Sequence[int],
    ) -> stim.PauliString:
        """Return one body-tableau generator with measured-out support removed.

        Returns
        -------
        ``stim.PauliString``
            Conjugated generator restricted to the surviving qubits.
        """
        source = stim.PauliString(self._num_qubits)
        source[qubit] = basis
        transformed = source.after(self._body)
        restricted = stim.PauliString(len(survivors))
        restricted.sign = transformed.sign
        for local_qubit, global_qubit in enumerate(survivors):
            restricted[local_qubit] = transformed[global_qubit]
        return restricted

    def verify(self) -> None:
        """Cross-check stabilizer-flow generators of the source and rewritten segment.

        Raises
        ------
        MppRewriteVerificationError
            If the rewritten segment is not flow-equivalent to its source.
        """
        if len(self._body) == 0 and self._orig_verify == self._conv_verify:
            return
        original, converted = self._padded_channels()
        original_flows = original.flow_generators()
        converted_flows = converted.flow_generators()
        # Stim currently returns a canonical sorted basis. Equality is a fast
        # path only; the per-flow fallback remains authoritative if the chosen
        # generator bases differ across equivalent circuits or Stim versions.
        if original_flows == converted_flows:
            return
        for flow in original_flows:
            if not converted.has_flow(flow):
                msg = _verification_message(self._segment_index, flow, "source flow missing from rewrite")
                raise MppRewriteVerificationError(msg)
        for flow in converted_flows:
            if not original.has_flow(flow):
                msg = _verification_message(self._segment_index, flow, "rewritten flow missing from source")
                raise MppRewriteVerificationError(msg)

    def exit_prepared(self) -> dict[int, str]:
        r"""Return the prepared-qubit map carried into the next segment.

        Returns
        -------
        `dict`\[`int`, `str`\]
            Prepared Pauli bases still valid at the next segment's start.
        """
        carried = {
            qubit: basis
            for qubit, basis in self._prepared.items()
            if qubit not in self._body_touched and qubit not in self._measured_any
        }
        carried.update(self._late_resets)
        return carried

    def exit_dirty(self) -> set[int]:
        r"""Return qubits whose post-measurement state diverges after this segment.

        Returns
        -------
        `set`\[`int`\]
            Qubits measured out without a subsequent reset.
        """
        return self._dirty | self._measured_unreset


def _remap_tableau_circuit(tableau: stim.Tableau, qubits: Sequence[int]) -> stim.Circuit:
    """Synthesize a tableau and map its dense qubit indices to Stim qubit ids.

    Returns
    -------
    ``stim.Circuit``
        Synthesized Clifford circuit on ``qubits``.

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


def _matched_flow_output_pairs(
    original: stim.Circuit,
    converted: stim.Circuit,
    qubits: Sequence[int],
    segment_index: int,
) -> list[tuple[stim.PauliString, stim.PauliString]]:
    r"""Match channel-flow outputs by their input Pauli and measurement parity.

    Returns
    -------
    `list`\[`tuple`\[``stim.PauliString``, ``stim.PauliString``\]\]
        Converted/original output-Pauli pairs restricted to ``qubits``.

    Raises
    ------
    _ResidualFrameSynthesisError
        If the two channels' canonical flow generators cannot be aligned.
    """
    original_by_key = {_flow_key(flow): flow for flow in original.flow_generators()}
    converted_by_key = {_flow_key(flow): flow for flow in converted.flow_generators()}
    if original_by_key.keys() != converted_by_key.keys():
        msg = f"Segment {segment_index} flow generators cannot be aligned to infer a residual frame."
        raise _ResidualFrameSynthesisError(msg)
    return [
        (
            _restrict_pauli(converted_by_key[key].output_copy(), qubits),
            _restrict_pauli(original_by_key[key].output_copy(), qubits),
        )
        for key in original_by_key
    ]


def _local_axis_maps(
    pairs: Sequence[tuple[stim.PauliString, stim.PauliString]],
    num_qubits: int,
    segment_index: int,
) -> list[dict[int, int]]:
    r"""Infer unsigned one-qubit Pauli-axis maps from matched flow outputs.

    Returns
    -------
    `list`\[`dict`\[`int`, `int`\]\]
        Source-to-target Pauli-axis maps for each dense qubit.

    Raises
    ------
    _ResidualFrameSynthesisError
        If a residual frame changes Pauli support or has inconsistent local
        axis constraints.
    """
    axis_maps: list[dict[int, int]] = [{} for _ in range(num_qubits)]
    for converted_pauli, original_pauli in pairs:
        if converted_pauli.pauli_indices() != original_pauli.pauli_indices():
            msg = f"Segment {segment_index} residual frame is not local on the surviving qubits."
            raise _ResidualFrameSynthesisError(msg)
        for qubit in converted_pauli.pauli_indices():
            source_axis = converted_pauli[qubit]
            target_axis = original_pauli[qubit]
            previous = axis_maps[qubit].setdefault(source_axis, target_axis)
            if previous != target_axis:
                msg = f"Segment {segment_index} has inconsistent local Clifford flow constraints."
                raise _ResidualFrameSynthesisError(msg)
    return axis_maps


def _local_sign_equations(
    pairs: Sequence[tuple[stim.PauliString, stim.PauliString]],
    tableaus: Sequence[stim.Tableau],
    segment_index: int,
) -> list[tuple[int, int]]:
    r"""Build Pauli-correction equations for local residual Clifford signs.

    Returns
    -------
    `list`\[`tuple`\[`int`, `int`\]\]
        Bit-packed GF(2) equations.

    Raises
    ------
    _ResidualFrameSynthesisError
        If the unsigned local tableaus do not satisfy the axis constraints.
    """
    equations: list[tuple[int, int]] = []
    for converted_pauli, original_pauli in pairs:
        mapped = _apply_local_tableaus(converted_pauli, tableaus)
        if mapped.pauli_indices() != original_pauli.pauli_indices() or any(
            mapped[qubit] != original_pauli[qubit] for qubit in mapped.pauli_indices()
        ):
            msg = f"Segment {segment_index} local Clifford axes do not satisfy its flow constraints."
            raise _ResidualFrameSynthesisError(msg)
        coefficients = 0
        for qubit in mapped.pauli_indices():
            axis = mapped[qubit]
            if axis in {2, 3}:
                coefficients ^= 1 << (2 * qubit)
            if axis in {1, 2}:
                coefficients ^= 1 << (2 * qubit + 1)
        equations.append((coefficients, int(mapped.sign != original_pauli.sign)))
    return equations


def _materialize_local_frame(
    qubits: Sequence[int],
    tableaus: Sequence[stim.Tableau],
    correction_bits: int,
) -> stim.Circuit:
    """Materialize local tableaus and their output-Pauli sign corrections.

    Returns
    -------
    ``stim.Circuit``
        Residual local Clifford circuit on the requested Stim qubits.
    """
    frame = stim.Circuit()
    for global_qubit, tableau in zip(qubits, tableaus, strict=True):
        if tableau != stim.Tableau(1):
            frame += _remap_tableau_circuit(tableau, [global_qubit])
    x_targets = [qubits[qubit] for qubit in range(len(qubits)) if correction_bits >> (2 * qubit) & 1]
    z_targets = [qubits[qubit] for qubit in range(len(qubits)) if correction_bits >> (2 * qubit + 1) & 1]
    if x_targets:
        frame.append("X", x_targets)
    if z_targets:
        frame.append("Z", z_targets)
    return frame


def _flow_key(flow: stim.Flow) -> tuple[str, tuple[int, ...]]:
    r"""Return the flow fields unchanged by an output Clifford frame.

    Returns
    -------
    `tuple`\[`str`, `tuple`\[`int`, ...\]\]
        Input-Pauli spelling and measurement-record parity.
    """
    return str(flow.input_copy()), tuple(flow.measurements_copy())


def _restrict_pauli(pauli: stim.PauliString, qubits: Sequence[int]) -> stim.PauliString:
    """Restrict a Pauli string to an ordered qubit subset.

    Returns
    -------
    ``stim.PauliString``
        Pauli string on dense indices corresponding to ``qubits``.
    """
    result = stim.PauliString(len(qubits))
    result.sign = pauli.sign
    for local_qubit, global_qubit in enumerate(qubits):
        # Stim trims trailing identity factors from flow-output Paulis.
        result[local_qubit] = pauli[global_qubit] if global_qubit < len(pauli) else 0
    return result


def _unsigned_local_tableau(axis_map: dict[int, int], segment_index: int) -> stim.Tableau:
    """Choose a one-qubit Clifford with the requested unsigned axis mapping.

    Returns
    -------
    ``stim.Tableau``
        A matching one-qubit Clifford tableau.

    Raises
    ------
    _ResidualFrameSynthesisError
        If no one-qubit Clifford satisfies the axis mapping.
    """
    for tableau in stim.Tableau.iter_all(1, unsigned=True):
        matches = True
        for source_axis, target_axis in axis_map.items():
            source = stim.PauliString(1)
            source[0] = source_axis
            if tableau(source)[0] != target_axis:
                matches = False
                break
        if matches:
            return tableau
    msg = f"Segment {segment_index} flow constraints do not define a one-qubit Clifford."
    raise _ResidualFrameSynthesisError(msg)


def _apply_local_tableaus(pauli: stim.PauliString, tableaus: Sequence[stim.Tableau]) -> stim.PauliString:
    """Conjugate a Pauli string by a tensor product of one-qubit tableaus.

    Returns
    -------
    ``stim.PauliString``
        Conjugated Pauli string.
    """
    result = stim.PauliString(len(pauli))
    result.sign = pauli.sign
    for qubit in pauli.pauli_indices():
        source = stim.PauliString(1)
        source[0] = pauli[qubit]
        transformed = tableaus[qubit](source)
        result[qubit] = transformed[0]
        result.sign *= transformed.sign
    return result


def _solve_binary_equations(equations: Sequence[tuple[int, int]]) -> int:
    """Solve bit-packed linear equations over GF(2), setting free variables to zero.

    Returns
    -------
    `int`
        Bit-packed solution.

    Raises
    ------
    _ResidualFrameSynthesisError
        If the equations are inconsistent.
    """
    pivots: dict[int, tuple[int, int]] = {}
    for initial_mask, initial_rhs in equations:
        mask = initial_mask
        rhs = initial_rhs
        while mask:
            pivot = (mask & -mask).bit_length() - 1
            existing = pivots.get(pivot)
            if existing is None:
                pivots[pivot] = (mask, rhs)
                break
            mask ^= existing[0]
            rhs ^= existing[1]
        if mask == 0 and rhs:
            msg = "Residual Clifford sign constraints are inconsistent."
            raise _ResidualFrameSynthesisError(msg)

    solution = 0
    for pivot in sorted(pivots, reverse=True):
        mask, rhs = pivots[pivot]
        if (mask & solution).bit_count() % 2 != rhs:
            solution |= 1 << pivot
    return solution


def _verification_message(segment_index: int, flow: stim.Flow, direction: str) -> str:
    return (
        f"Segment {segment_index} failed flow verification ({direction}): {flow}. "
        "The removed Clifford body likely acts on data qubits beyond measuring the inferred products."
    )


def _measurement_observables(instruction: stim.CircuitInstruction, num_qubits: int) -> list[_SourceObservable]:
    name = instruction.name
    if name in _SINGLE_MEASUREMENT_BASES or name in _MEASURE_RESET_GATES:
        basis = _SINGLE_MEASUREMENT_BASES.get(name) or _MEASURE_RESET_GATES[name][0]
        return [_single_qubit_observable(group, basis, name, num_qubits) for group in instruction.target_groups()]
    if name in _PAIR_MEASUREMENT_BASES:
        basis = _PAIR_MEASUREMENT_BASES[name]
        return [_pair_observable(group, basis, name, num_qubits) for group in instruction.target_groups()]
    return [_mpp_observable(group, num_qubits) for group in instruction.target_groups()]


def _single_qubit_observable(
    group: Sequence[stim.GateTarget], basis: str, name: str, num_qubits: int
) -> _SourceObservable:
    (target,) = group
    qubit = _plain_qubit(target, name)
    observable = stim.PauliString(num_qubits)
    observable[qubit] = basis
    if target.is_inverted_result_target:
        observable.sign = -1
    return _SourceObservable(observable=observable, source_qubit=qubit)


def _pair_observable(group: Sequence[stim.GateTarget], basis: str, name: str, num_qubits: int) -> _SourceObservable:
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
        observable[qubit] = basis
        if target.is_inverted_result_target:
            sign = -sign
    observable.sign = sign
    return _SourceObservable(observable=observable, source_qubit=None)


def _mpp_observable(group: Sequence[stim.GateTarget], num_qubits: int) -> _SourceObservable:
    observable = stim.PauliString(num_qubits)
    for target in group:
        qubit = _plain_qubit(target, "MPP")
        pauli = target.pauli_type
        if pauli not in _PAULI_CODES:
            msg = f"MPP contains a non-Pauli target on qubit {qubit}."
            raise UnsupportedSyndromeCircuitError(msg)
        factor = stim.PauliString({qubit: _PAULI_CODES[pauli]})
        if target.is_inverted_result_target:
            factor.sign = -1
        observable *= factor
    return _SourceObservable(observable=observable, source_qubit=None)
