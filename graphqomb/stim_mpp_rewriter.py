"""Rewrite Stim syndrome-extraction circuits into abstract MPP measurements.

This experimental module recognizes the standard syndrome-extraction shape
(reset ancillas, apply Clifford unitaries, measure ancillas) and replaces each
gate-level extraction block with ``MPP`` instructions that directly measure the
inferred data Pauli products. Each source measurement maps to exactly one
measurement in the rewritten circuit, so ``DETECTOR`` and
``OBSERVABLE_INCLUDE`` annotations are copied verbatim.

The inference conjugates each final measurement Pauli backwards through the
segment's Clifford body (``U† P U`` via :meth:`stim.PauliString.before`) and
substitutes ``+1`` for stabilizers of freshly initialized, measured-out
ancillas. A segment ends when a unitary follows its measurements, or when a
measurement follows a reset issued after those measurements (including the
implicit reset of ``MR``); trailing data measurements therefore start a fresh
segment with an empty Clifford body and pass through verbatim. Rewriting is
structural: circuit-level noise is rejected, and the post-measurement state of
measured-out ancillas is not preserved (they are left in their reset state
instead of the measurement outcome's eigenstate).

By default each rewritten segment is verified against its source segment by
cross-checking stabilizer-flow generators, with a reset appended to every
measured-out ancilla in both copies to discard the intentionally different
ancilla post-states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import stim

if TYPE_CHECKING:
    from collections.abc import Sequence

_ANNOTATION_GATES = frozenset({"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "TICK"})
_RESET_BASES = {"R": "Z", "RX": "X", "RY": "Y"}
_SINGLE_MEASUREMENT_BASES = {"M": "Z", "MX": "X", "MY": "Y"}
_MEASURE_RESET_GATES = {"MR": ("Z", "R"), "MRX": ("X", "RX"), "MRY": ("Y", "RY")}
_PAIR_MEASUREMENT_BASES = {"MXX": "X", "MYY": "Y", "MZZ": "Z"}
_PAULI_CODES = {"X": 1, "Y": 2, "Z": 3}
_PAIR_GROUP_SIZE = 2

_InstructionKind = Literal["annotation", "mpad", "reset", "measurement", "unitary"]


class UnsupportedSyndromeCircuitError(ValueError):
    """Raised when a circuit is outside the supported syndrome-extraction form."""


class MppRewriteVerificationError(ValueError):
    """Raised when a rewritten segment fails stabilizer-flow verification."""


@dataclass(frozen=True)
class CheckMapping:
    """Sidecar mapping from one source measurement to its inferred Pauli product.

    Attributes
    ----------
    measurement_index : int
        Global measurement-record index, identical in source and rewritten circuits.
    segment_index : int
        Index of the extraction segment that produced this measurement.
    product : stim.PauliString
        Inferred signed Pauli product measured at this record position.
    source_qubit : int | None
        Measured qubit for single-qubit source measurements, ``None`` for
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
    circuit : stim.Circuit
        Rewritten circuit with one measurement per source measurement.
    checks : `tuple`\\[`CheckMapping`, ...\\]
        One mapping per measurement record produced by a Pauli measurement.
        ``MPAD`` records are not listed.
    """

    circuit: stim.Circuit
    checks: tuple[CheckMapping, ...]


@dataclass(frozen=True)
class _SourceObservable:
    observable: stim.PauliString
    source_qubit: int | None


@dataclass(frozen=True)
class _RewriteConfig:
    num_qubits: int
    verify: bool


@dataclass
class _SegmentBuffer:
    items: list[tuple[stim.CircuitInstruction, _InstructionKind]] = field(default_factory=list)
    seen_measurement: bool = False
    seen_late_reset: bool = False


def rewrite_to_mpp(circuit: stim.Circuit | str, *, verify: bool = True) -> MppRewriteResult:
    """Rewrite a noiseless Stim syndrome-extraction circuit into MPP form.

    ``REPEAT`` blocks are flattened before rewriting, which also bakes
    ``SHIFT_COORDS`` offsets into ``DETECTOR`` coordinate arguments. Noise
    instructions, measurement noise arguments, and classical feedback are
    rejected with :class:`UnsupportedSyndromeCircuitError`.

    Parameters
    ----------
    circuit : stim.Circuit | str
        Source circuit or Stim circuit text.
    verify : bool
        Cross-check stabilizer-flow generators of every rewritten segment
        against its source segment. Verification models the documented
        semantics: measured-out ancilla post-states are reset in both copies
        before comparison.

    Returns
    -------
    MppRewriteResult
        Rewritten circuit and per-measurement Pauli-product mappings.

    Raises
    ------
    UnsupportedSyndromeCircuitError
        If the circuit is outside the supported syndrome-extraction form. When
        ``verify`` is enabled, :class:`MppRewriteVerificationError` is raised
        for segments that are not flow-equivalent to their source.
    RuntimeError
        If the rewrite changes the measurement count, which indicates a bug.
    """
    flattened = _coerce_circuit(circuit).flattened()
    rewriter = _Rewriter(num_qubits=flattened.num_qubits, verify=verify)
    for instruction in flattened:
        if isinstance(instruction, stim.CircuitRepeatBlock):  # pragma: no cover - removed by flattened()
            msg = "REPEAT blocks must be flattened before rewriting."
            raise UnsupportedSyndromeCircuitError(msg)
        rewriter.process(instruction)
    result = rewriter.finish()
    if result.circuit.num_measurements != flattened.num_measurements:  # pragma: no cover - internal invariant
        msg = "MPP rewrite changed the measurement count; this is a bug."
        raise RuntimeError(msg)
    return result


def _coerce_circuit(circuit: stim.Circuit | str) -> stim.Circuit:
    if isinstance(circuit, stim.Circuit):
        return circuit
    return stim.Circuit(circuit)


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
    """Streams flattened instructions, buffering and rewriting one segment at a time."""

    def __init__(self, *, num_qubits: int, verify: bool) -> None:
        self._config = _RewriteConfig(num_qubits=num_qubits, verify=verify)
        self._output = stim.Circuit()
        self._checks: list[CheckMapping] = []
        self._measurement_index = 0
        self._segment_index = 0
        self._dirty: set[int] = set()
        self._entry_prepared: dict[int, str] = {}
        self._buffer = _SegmentBuffer()

    def process(self, instruction: stim.CircuitInstruction) -> None:
        """Buffer one instruction, flushing the current segment at a boundary.

        A segment ends before a unitary that follows the segment's
        measurements, or before a measurement that follows a reset issued
        after those measurements (including the implicit reset of ``MR``).
        """
        kind = _instruction_kind(instruction)
        starts_new_segment = (kind == "unitary" and self._buffer.seen_measurement) or (
            kind in {"measurement", "mpad"} and self._buffer.seen_late_reset
        )
        if starts_new_segment:
            self._flush_segment()
        self._buffer.items.append((instruction, kind))
        if kind in {"measurement", "mpad"}:
            self._buffer.seen_measurement = True
            if instruction.name in _MEASURE_RESET_GATES:
                self._buffer.seen_late_reset = True
        elif kind == "reset" and self._buffer.seen_measurement:
            self._buffer.seen_late_reset = True

    def finish(self) -> MppRewriteResult:
        """Flush the final segment and return the rewrite result.

        Returns
        -------
        MppRewriteResult
            Rewritten circuit and per-measurement Pauli-product mappings.
        """
        self._flush_segment()
        return MppRewriteResult(circuit=self._output, checks=tuple(self._checks))

    def _flush_segment(self) -> None:
        if not self._buffer.items:
            return
        segment = _Segment(
            config=self._config,
            segment_index=self._segment_index,
            entry_prepared=dict(self._entry_prepared),
            dirty=set(self._dirty),
            measurement_index=self._measurement_index,
        )
        segment.prescan(self._buffer.items)
        for instruction, kind in self._buffer.items:
            segment.process(instruction, kind)
        self._output += segment.output
        self._checks.extend(segment.checks)
        self._measurement_index = segment.measurement_index
        if self._config.verify:
            segment.verify()
        self._entry_prepared = segment.exit_prepared()
        self._dirty = segment.exit_dirty()
        self._segment_index += 1
        self._buffer = _SegmentBuffer()


class _Segment:
    """Rewrites one reset/Clifford/measure segment of the source circuit."""

    def __init__(
        self,
        *,
        config: _RewriteConfig,
        segment_index: int,
        entry_prepared: dict[int, str],
        dirty: set[int],
        measurement_index: int,
    ) -> None:
        self._config = config
        self._segment_index = segment_index
        self._prepared = entry_prepared
        self._dirty = dirty
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
        self._orig_verify = stim.Circuit()
        self._conv_verify = stim.Circuit()
        if config.verify:
            for qubit, basis in sorted(entry_prepared.items()):
                for target_circuit in (self._orig_verify, self._conv_verify):
                    target_circuit.append(_reset_gate_for_basis(basis), [qubit])

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
        if self._config.verify:
            self._orig_verify.append(instruction)
            self._conv_verify.append(instruction)

    def _rewrite_targets(self) -> tuple[stim.Circuit, ...]:
        if self._config.verify:
            return (self.output, self._conv_verify)
        return (self.output,)

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
        if self._config.verify:
            self._orig_verify.append(instruction)
        self._body_touched |= touched

    def _process_measurement(self, instruction: stim.CircuitInstruction) -> None:
        if instruction.gate_args_copy():
            msg = f"Noisy measurement {instruction.name} with arguments is not supported."
            raise UnsupportedSyndromeCircuitError(msg)
        sources = _measurement_observables(instruction, self._config.num_qubits)
        self._validate_measured_qubits(instruction, sources)
        self._seen_measurement = True
        products = [self._infer_product(source) for source in sources]
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
        self._emit_measurement(instruction, sources, products)
        if self._config.verify:
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

    def verify(self) -> None:
        """Cross-check stabilizer-flow generators of the source and rewritten segment.

        Raises
        ------
        MppRewriteVerificationError
            If the rewritten segment is not flow-equivalent to its source.
        """
        if len(self._body) == 0 and self._orig_verify == self._conv_verify:
            return
        padding = sorted(self._measured_out)
        original = self._orig_verify.copy()
        converted = self._conv_verify.copy()
        if padding:
            original.append("R", padding)
            converted.append("R", padding)
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
        `dict`\\[`int`, `str`\\]
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
        `set`\\[`int`\\]
            Qubits measured out without a subsequent reset.
        """
        return self._dirty | self._measured_unreset


def _verification_message(segment_index: int, flow: stim.Flow, direction: str) -> str:
    return (
        f"Segment {segment_index} failed flow verification ({direction}): {flow}. "
        "The removed Clifford body likely acts on data qubits beyond measuring the inferred products."
    )


def _reset_gate_for_basis(basis: str) -> str:
    return {"Z": "R", "X": "RX", "Y": "RY"}[basis]


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
    if len(group) != _PAIR_GROUP_SIZE:  # pragma: no cover - stim enforces pairing
        msg = f"{name} expects qubit pairs."
        raise UnsupportedSyndromeCircuitError(msg)
    observable = stim.PauliString(num_qubits)
    sign = 1
    for target in group:
        qubit = _plain_qubit(target, name)
        if observable[qubit] != 0:  # pragma: no cover - stim rejects duplicate pair targets
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
