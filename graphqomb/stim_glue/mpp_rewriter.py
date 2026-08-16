"""Move Clifford bodies behind Pauli measurements and expose inferred MPPs.

For a Clifford body ``U`` and a Pauli measurement projector ``Pi_m(P)``,

``Pi_m(P) U = U Pi_m(U† P U)``.

The rewriter applies this identity directly. Clifford gates are accumulated
as a pending frame, every Pauli measurement is conjugated backwards through
that frame, and the unchanged frame is materialized at the next reset-like
barrier or at circuit exit. Correctness therefore does not depend on flow
verification, retries, or gate-level fallback.

When the pulled product contains the preparation Pauli of the reset qubit
that the source directly measures, that factor is a known ``+1`` stabilizer
and is removed. Standard reset/Clifford/measure check gadgets consequently
expose a data-only ``MPP`` while the exact Clifford body remains pending
behind it.

This module provides:

- `rewrite_to_mpp`: Rewrite a noiseless Clifford/Pauli-measurement circuit.
- `MppRewriteResult`: Rewritten circuit with its per-measurement products.
- `CheckMapping`: Mapping from one source record to its inferred product.
- `UnsupportedSyndromeCircuitError`: Error for unsupported noisy circuits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import stim

from graphqomb.stim_glue._parse import (
    ANNOTATION_GATES,
    MEASURE_RESET_AXES,
    PAIR_MEASUREMENT_AXES,
    RESET_AXES,
    RESET_GATES,
    SINGLE_MEASUREMENT_AXES,
    iter_instructions,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# Pauli bases are spelled as Stim's X/Y/Z letters inside this module.
_RESET_BASES = {name: axis.name for name, axis in RESET_AXES.items()}
_SINGLE_MEASUREMENT_BASES = {name: axis.name for name, axis in SINGLE_MEASUREMENT_AXES.items()}
_MEASURE_RESET_BASES = {name: axis.name for name, axis in MEASURE_RESET_AXES.items()}
_MEASURE_RESET_GATES = {name: RESET_GATES[axis] for name, axis in MEASURE_RESET_AXES.items()}
_MEASURE_RESET_MEASUREMENTS = {
    name: next(
        measurement for measurement, measurement_axis in SINGLE_MEASUREMENT_AXES.items() if measurement_axis == axis
    )
    for name, axis in MEASURE_RESET_AXES.items()
}
_PAIR_MEASUREMENT_BASES = {name: axis.name for name, axis in PAIR_MEASUREMENT_AXES.items()}
_PAULI_CODES = {"X": 1, "Y": 2, "Z": 3}
_PAIR_GROUP_SIZE = 2

_InstructionKind = Literal["annotation", "mpad", "reset", "measurement", "unitary"]
_FallbackMode = Literal["circuit", "segment"]


class UnsupportedSyndromeCircuitError(ValueError):
    """Raised when a circuit is outside the supported noiseless Clifford form."""


class MppRewriteVerificationError(ValueError):
    """Legacy compatibility error; the constructive rewriter never raises it."""


@dataclass(frozen=True)
class CheckMapping:
    """Sidecar mapping from one source measurement to its inferred product.

    Attributes
    ----------
    measurement_index : `int`
        Global measurement-record index, identical in both circuits.
    segment_index : `int`
        Source segment index under the historical boundary convention.
    product : ``stim.PauliString``
        Signed product emitted for this record.
    source_qubit : `int` | `None`
        Directly measured source qubit, or `None` for product measurements.
    """

    measurement_index: int
    segment_index: int
    product: stim.PauliString
    source_qubit: int | None


@dataclass(frozen=True)
class MppRewriteResult:
    r"""Result of moving Clifford gates behind Pauli measurements.

    ``fallback_segments`` is retained for API compatibility and is always
    empty because this pipeline has no optimized-versus-verbatim branches.

    Attributes
    ----------
    circuit : ``stim.Circuit``
        Exactly equivalent rewritten circuit.
    checks : `tuple`\[`CheckMapping`, ...\]
        One mapping per Pauli-measurement record; source ``MPAD`` records are
        not listed.
    fallback_segments : `tuple`\[`int`, ...\]
        Always empty.
    """

    circuit: stim.Circuit
    checks: tuple[CheckMapping, ...]
    fallback_segments: tuple[int, ...] = ()


@dataclass(frozen=True)
class _SourceObservable:
    observable: stim.PauliString
    source_qubit: int | None


@dataclass
class _SegmentBounds:
    """Historical segment numbering retained for `CheckMapping`."""

    seen_measurement: bool = False
    seen_late_reset: bool = False

    def starts_new_segment(self, instruction: stim.CircuitInstruction, kind: _InstructionKind) -> bool:
        """Return whether this source instruction opens a new segment.

        Returns
        -------
        `bool`
            Whether the instruction starts a new historical segment.
        """
        boundary = (kind == "unitary" and self.seen_measurement) or (
            kind in {"measurement", "mpad"} and self.seen_late_reset
        )
        if boundary:
            self.seen_measurement = False
            self.seen_late_reset = False
        if kind in {"measurement", "mpad"}:
            self.seen_measurement = True
            if instruction.name in _MEASURE_RESET_BASES:
                self.seen_late_reset = True
        elif kind == "reset" and self.seen_measurement:
            self.seen_late_reset = True
        return boundary


def rewrite_to_mpp(circuit: stim.Circuit | str, *, fallback: _FallbackMode = "circuit") -> MppRewriteResult:
    r"""Move Clifford gates behind measurements and expose pulled Pauli products.

    A pending Clifford circuit ``U`` is commuted through every Pauli
    measurement by replacing ``P`` with ``U† P U``. The unchanged ``U`` is
    then emitted at a deterministic barrier: a reset, measure-reset,
    measurement-record-controlled gate, or circuit exit. Measurement order,
    record indices, detector/observable annotations, and post-measurement
    states are preserved exactly.

    If a direct source measurement's pulled product contains the same Pauli
    on its source qubit as the most recent reset prepared, that factor is
    removed as a known ``+1`` stabilizer. A negative identity is deliberately
    left unsubstituted so the result remains a real signed measurement instead
    of the importer's unsupported ``MPAD 1``.

    ``REPEAT`` blocks are flattened before processing. Noise and noisy
    measurement arguments remain unsupported. The ``fallback`` argument is
    accepted for compatibility; both values select this same fallback-free
    pipeline.

    Parameters
    ----------
    circuit : ``stim.Circuit`` | `str`
        Source circuit or Stim text.
    fallback : ``"circuit"`` | ``"segment"``, optional
        Compatibility parameter; it does not change the rewrite.

    Returns
    -------
    `MppRewriteResult`
        Exactly equivalent rewritten circuit and measurement mappings.

    Raises
    ------
    ValueError
        If ``fallback`` is not a recognized compatibility value.
    RuntimeError
        If an internal bug changes the number of measurement records.
    """
    if fallback not in {"circuit", "segment"}:
        msg = f"fallback must be 'circuit' or 'segment', not {fallback!r}."
        raise ValueError(msg)
    source = circuit if isinstance(circuit, stim.Circuit) else stim.Circuit(circuit)
    flattened = source.flattened()
    rewriter = _PendingCliffordRewriter(flattened.num_qubits)
    for instruction in iter_instructions(flattened):
        rewriter.process(instruction)
    result = rewriter.finish()
    if result.circuit.num_measurements != flattened.num_measurements:
        msg = "MPP rewrite changed the measurement count; this is a bug."
        raise RuntimeError(msg)
    return result


class _PendingCliffordRewriter:
    """Streaming implementation of the exact measurement/Clifford identity."""

    def __init__(self, num_qubits: int) -> None:
        self._num_qubits = num_qubits
        self._output = stim.Circuit()
        self._pending = stim.Circuit()
        self._pending_touched: set[int] = set()
        self._prepared: dict[int, str] = {}
        self._checks: list[CheckMapping] = []
        self._measurement_index = 0
        self._segment_index = 0
        self._bounds = _SegmentBounds()

    def process(self, instruction: stim.CircuitInstruction) -> None:
        """Consume one flattened source instruction."""
        kind = _instruction_kind(instruction)
        if self._bounds.starts_new_segment(instruction, kind):
            self._segment_index += 1

        if kind == "annotation":
            self._output.append(instruction)
        elif kind == "mpad":
            self._output.append(instruction)
            self._measurement_index += instruction.num_measurements
        elif kind == "reset":
            self._process_reset(instruction)
        elif kind == "measurement":
            self._process_measurement(instruction)
        else:
            self._process_unitary(instruction)

    def finish(self) -> MppRewriteResult:
        """Materialize the final frame and return the rewrite result.

        Returns
        -------
        `MppRewriteResult`
            Completed exact rewrite.
        """
        self._flush_pending()
        return MppRewriteResult(circuit=self._output, checks=tuple(self._checks))

    def _process_reset(self, instruction: stim.CircuitInstruction) -> None:
        self._flush_pending()
        self._output.append(instruction)
        basis = _RESET_BASES[instruction.name]
        for target in instruction.targets_copy():
            self._prepared[_plain_qubit(target, instruction.name)] = basis

    def _process_unitary(self, instruction: stim.CircuitInstruction) -> None:
        targets = instruction.targets_copy()
        if any(target.is_sweep_bit_target for target in targets):
            target = next(target for target in targets if target.is_sweep_bit_target)
            msg = f"Classical feedback is not supported: {instruction.name} with target {target!r}."
            raise UnsupportedSyndromeCircuitError(msg)
        if any(target.is_measurement_record_target for target in targets):
            # Feedback is a deterministic barrier, not a fallback: the exact
            # pending frame is emitted before the source instruction.
            self._flush_pending()
            self._output.append(instruction)
            for qubit in _instruction_qubits(instruction):
                self._prepared.pop(qubit, None)
            return
        self._pending.append(instruction)
        self._pending_touched.update(_instruction_qubits(instruction))

    def _process_measurement(self, instruction: stim.CircuitInstruction) -> None:
        if instruction.gate_args_copy():
            msg = f"Noisy measurement {instruction.name} with arguments is not supported."
            raise UnsupportedSyndromeCircuitError(msg)
        sources = _measurement_observables(instruction, self._num_qubits)
        products: list[stim.PauliString] = []
        for source in sources:
            source_product = self._pull(source.observable)
            products.append(self._substitute_source_reset(source_product, source.source_qubit))
            # Targets within one measurement instruction still produce
            # records in sequence. An earlier anticommuting product can
            # destroy a reset stabilizer needed by a later target.
            self._invalidate_anticommuting_preparations((source_product,))
        self._append_checks(products, sources)

        is_measure_reset = instruction.name in _MEASURE_RESET_BASES
        trivial = all(product == source.observable for product, source in zip(products, sources, strict=True))
        if is_measure_reset and len(self._pending) == 0 and trivial:
            self._output.append(instruction)
            self._set_measure_reset_preparations(instruction)
            return

        if trivial:
            if is_measure_reset:
                self._append_measure_only(instruction)
            else:
                self._output.append(instruction)
        else:
            _append_products(self._output, products, tag=instruction.tag)

        if is_measure_reset:
            self._flush_pending()
            reset_gate = _MEASURE_RESET_GATES[instruction.name]
            qubits = [_plain_qubit(target, instruction.name) for target in instruction.targets_copy()]
            self._output.append(reset_gate, qubits, [], tag=instruction.tag)
            self._set_measure_reset_preparations(instruction)

    def _pull(self, observable: stim.PauliString) -> stim.PauliString:
        if len(self._pending) == 0:
            return observable.copy()
        return observable.before(self._pending)

    def _substitute_source_reset(
        self,
        product: stim.PauliString,
        source_qubit: int | None,
    ) -> stim.PauliString:
        """Remove the directly measured qubit's known reset stabilizer.

        Returns
        -------
        ``stim.PauliString``
            Product with a matching reset factor removed when safe.
        """
        if source_qubit is None:
            return product
        basis = self._prepared.get(source_qubit)
        if basis is None or product[source_qubit] != _PAULI_CODES[basis]:
            return product
        candidate = product.copy()
        candidate[source_qubit] = 0
        if not candidate.pauli_indices() and candidate.sign == -1:
            # Keep a real signed measurement instead of producing MPAD 1,
            # which Stim can represent but the GraphQOMB importer cannot.
            return product
        return candidate

    def _append_checks(
        self,
        products: Sequence[stim.PauliString],
        sources: Sequence[_SourceObservable],
    ) -> None:
        for product, source in zip(products, sources, strict=True):
            self._checks.append(
                CheckMapping(
                    measurement_index=self._measurement_index,
                    segment_index=self._segment_index,
                    product=product,
                    source_qubit=source.source_qubit,
                )
            )
            self._measurement_index += 1

    def _append_measure_only(self, instruction: stim.CircuitInstruction) -> None:
        measurement_gate = _MEASURE_RESET_MEASUREMENTS[instruction.name]
        self._output.append(measurement_gate, instruction.targets_copy(), [], tag=instruction.tag)

    def _set_measure_reset_preparations(self, instruction: stim.CircuitInstruction) -> None:
        basis = _MEASURE_RESET_BASES[instruction.name]
        for target in instruction.targets_copy():
            self._prepared[_plain_qubit(target, instruction.name)] = basis

    def _invalidate_anticommuting_preparations(self, products: Sequence[stim.PauliString]) -> None:
        invalidated = {
            qubit
            for qubit, basis in self._prepared.items()
            if any(product[qubit] not in {0, _PAULI_CODES[basis]} for product in products)
        }
        for qubit in invalidated:
            self._prepared.pop(qubit, None)

    def _flush_pending(self) -> None:
        if len(self._pending) == 0:
            return
        self._output += self._pending
        for qubit in self._pending_touched:
            self._prepared.pop(qubit, None)
        self._pending = stim.Circuit()
        self._pending_touched.clear()


def _append_products(output: stim.Circuit, products: Sequence[stim.PauliString], *, tag: str) -> None:
    """Append products in record order, using MPAD for positive identities."""
    with_support = [bool(product.pauli_indices()) for product in products]
    if all(with_support):
        targets: list[stim.GateTarget] = []
        for product in products:
            targets.extend(cast("list[stim.GateTarget]", stim.target_combined_paulis(product)))
        output.append("MPP", targets, [], tag=tag)
        return
    for product, has_support in zip(products, with_support, strict=True):
        if has_support:
            targets = cast("list[stim.GateTarget]", stim.target_combined_paulis(product))
            output.append("MPP", targets, [], tag=tag)
        else:
            output.append("MPAD", [int(product.sign == -1)], [], tag=tag)


def _instruction_kind(instruction: stim.CircuitInstruction) -> _InstructionKind:
    name = instruction.name
    if name in ANNOTATION_GATES:
        return "annotation"
    if name == "MPAD":
        return "mpad"
    if name in _RESET_BASES:
        return "reset"
    if (
        name == "MPP"
        or name in _SINGLE_MEASUREMENT_BASES
        or name in _MEASURE_RESET_BASES
        or name in _PAIR_MEASUREMENT_BASES
    ):
        return "measurement"
    if stim.gate_data(name).is_unitary:
        return "unitary"
    msg = f"Unsupported instruction for MPP rewriting: {name}."
    raise UnsupportedSyndromeCircuitError(msg)


def _instruction_qubits(instruction: stim.CircuitInstruction) -> set[int]:
    return {int(target.qubit_value) for target in instruction.targets_copy() if target.qubit_value is not None}


def _plain_qubit(target: stim.GateTarget, instruction_name: str) -> int:
    qubit_value = target.qubit_value
    if qubit_value is None:
        msg = f"{instruction_name} contains unsupported target {target!r}; only qubit targets are supported."
        raise UnsupportedSyndromeCircuitError(msg)
    return int(qubit_value)


def _measurement_observables(instruction: stim.CircuitInstruction, num_qubits: int) -> list[_SourceObservable]:
    name = instruction.name
    if name in _SINGLE_MEASUREMENT_BASES or name in _MEASURE_RESET_BASES:
        basis = _SINGLE_MEASUREMENT_BASES.get(name) or _MEASURE_RESET_BASES[name]
        return [_single_qubit_observable(group, basis, name, num_qubits) for group in instruction.target_groups()]
    if name in _PAIR_MEASUREMENT_BASES:
        basis = _PAIR_MEASUREMENT_BASES[name]
        return [_pair_observable(group, basis, name, num_qubits) for group in instruction.target_groups()]
    return [_mpp_observable(group, num_qubits) for group in instruction.target_groups()]


def _single_qubit_observable(
    group: Sequence[stim.GateTarget],
    basis: str,
    name: str,
    num_qubits: int,
) -> _SourceObservable:
    (target,) = group
    qubit = _plain_qubit(target, name)
    observable = stim.PauliString(num_qubits)
    observable[qubit] = basis
    if target.is_inverted_result_target:
        observable.sign = -1
    return _SourceObservable(observable=observable, source_qubit=qubit)


def _pair_observable(
    group: Sequence[stim.GateTarget],
    basis: str,
    name: str,
    num_qubits: int,
) -> _SourceObservable:
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
        factor = stim.PauliString(num_qubits)
        factor[qubit] = _PAULI_CODES[pauli]
        if target.is_inverted_result_target:
            factor.sign = -1
        observable *= factor
    if observable.sign not in {1, -1}:
        msg = f"Non-Hermitian measurement product: {observable}."
        raise UnsupportedSyndromeCircuitError(msg)
    return _SourceObservable(observable=observable, source_qubit=None)
