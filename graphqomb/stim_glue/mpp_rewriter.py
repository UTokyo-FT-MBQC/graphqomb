"""Normalize destructive readouts and replace local extraction circuits by MPPs.

Unrecognized intervals retain their gates and readouts for ordinary MBQC
lowering. No post-measurement state or pending-frame disposal check is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import stim

from graphqomb.stim_glue._gadget_contract import contract_extraction_gadgets
from graphqomb.stim_glue._measurement_lifetimes import normalize_measurement_lifetimes
from graphqomb.stim_glue._parse import (
    ANNOTATION_GATES,
    MEASURE_RESET_AXES,
    PAIR_MEASUREMENT_AXES,
    RESET_AXES,
    SINGLE_MEASUREMENT_AXES,
    iter_instructions,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_RESET_BASES = {name: axis.name for name, axis in RESET_AXES.items()}
_SINGLE_MEASUREMENT_BASES = {name: axis.name for name, axis in SINGLE_MEASUREMENT_AXES.items()}
_MEASURE_RESET_BASES = {name: axis.name for name, axis in MEASURE_RESET_AXES.items()}
_PAIR_MEASUREMENT_BASES = {name: axis.name for name, axis in PAIR_MEASUREMENT_AXES.items()}
_PAULI_CODES = {"X": 1, "Y": 2, "Z": 3}
_PAIR_GROUP_SIZE = 2
_InstructionKind = Literal["annotation", "mpad", "reset", "measurement", "unitary"]


class UnsupportedSyndromeCircuitError(ValueError):
    """Raised when a circuit is outside the supported noiseless Clifford form."""


@dataclass(frozen=True)
class CheckMapping:
    """A signed observable at its emitted circuit position and its source wire.

    measurement_index is the emitted record index, product is the emitted
    observable, and source_qubit is the original direct readout's Stim id
    (None for source product measurements). MPAD records have no check entry.
    """

    measurement_index: int
    product: stim.PauliString
    source_qubit: int | None


@dataclass(frozen=True)
class MppRewriteResult:
    r"""One destructive-measurement circuit and its record mappings.

    circuit preserves the record distribution of the input interpreted with
    destructive readouts and independent re-preparations. Terminal quantum
    outputs are outside this contract. checks describes emitted observables;
    foliation_circuit and foliation_checks are compatibility names for those
    same objects. foliation_record_to_source[j] identifies the original record
    of emitted column j. eliminated_qubits contains only original Stim ids
    completely absent after extraction, not ids with retained later lifetimes.
    """

    circuit: stim.Circuit
    checks: tuple[CheckMapping, ...]
    foliation_circuit: stim.Circuit
    eliminated_qubits: tuple[int, ...] = ()
    foliation_record_to_source: tuple[int, ...] = ()
    foliation_checks: tuple[CheckMapping, ...] = ()


@dataclass(frozen=True)
class _SourceObservable:
    observable: stim.PauliString
    source_qubit: int | None


def rewrite_to_mpp(circuit: stim.Circuit | str) -> MppRewriteResult:
    """Normalize photonic lifetimes and replace recognized extraction intervals.

    M/MX/MY consume their wire just like the measurement half of MR/MRX/MRY.
    Reuse prepares the positive eigenstate independently of the result, unless
    an explicit reset supplies another axis. Terminal readouts stop the wire.
    This intentionally differs from nondestructive Stim measurement semantics.

    Local controlled-Pauli extraction intervals become data-only MPPs. Other
    intervals keep their gates and readouts without duplicating an extraction
    body behind a pulled measurement. Mixed readouts can be reordered; all
    record references and the explicit record map follow that permutation.

    Returns
    -------
    MppRewriteResult
        Rewritten circuit and mappings under the destructive-readout contract.

    Raises
    ------
    UnsupportedSyndromeCircuitError
        If an instruction is noisy, unsupported, or has a sweep control.
    RuntimeError
        If an internal bug changes the number of measurement records.
    """
    source = circuit if isinstance(circuit, stim.Circuit) else stim.Circuit(circuit)
    source_qubits: dict[int, int | None] = {}
    record = 0
    for instruction in iter_instructions(source.flattened()):
        kind = _instruction_kind(instruction)
        if kind == "measurement":
            if instruction.gate_args_copy():
                msg = f"Noisy measurement {instruction.name} with arguments is not supported."
                raise UnsupportedSyndromeCircuitError(msg)
            for offset, observable in enumerate(_measurement_observables(instruction, source.num_qubits)):
                source_qubits[record + offset] = observable.source_qubit
        if any(t.is_sweep_bit_target for t in instruction.targets_copy()):
            msg = f"Classical feedback with sweep controls is not supported: {instruction.name}."
            raise UnsupportedSyndromeCircuitError(msg)
        record += instruction.num_measurements
    normalized = normalize_measurement_lifetimes(source)
    contracted = contract_extraction_gadgets(normalized)
    output = _separate_conflicting_mpp_products(contracted.circuit)
    checks: list[CheckMapping] = []
    record = 0
    for instruction in iter_instructions(output):
        if _instruction_kind(instruction) == "measurement":
            for offset, observable in enumerate(_measurement_observables(instruction, source.num_qubits)):
                index = record + offset
                checks.append(
                    CheckMapping(index, observable.observable, source_qubits[contracted.record_to_source[index]])
                )
        record += instruction.num_measurements
    if record != source.num_measurements:
        msg = "MPP rewrite changed the measurement count; this is a bug."
        raise RuntimeError(msg)
    mappings = tuple(checks)
    return MppRewriteResult(
        output, mappings, output, contracted.eliminated_qubits, contracted.record_to_source, mappings
    )


def _separate_conflicting_mpp_products(circuit: stim.Circuit) -> stim.Circuit:
    """Keep repeated or anticommuting Pauli products out of one Foliation layer.

    Products remain in record order. Distinct commuting products still share
    the source TICK interval; a repeated unsigned support or a product that
    anticommutes with one already in the layer starts the next internal layer.
    Pair measurements are converted to MPP before checking for conflicts.

    Returns
    -------
    ``stim.Circuit``
        Import-oriented circuit with conflicting MPP products separated by TICK.
    """
    result = stim.Circuit()
    supports_in_layer: list[tuple[tuple[int, str], ...]] = []
    for instruction in iter_instructions(circuit):
        if instruction.name == "TICK":
            result.append(instruction)
            supports_in_layer.clear()
        elif instruction.name == "MPP" or instruction.name in _PAIR_MEASUREMENT_BASES:
            for source_group in instruction.target_groups():
                if instruction.name in _PAIR_MEASUREMENT_BASES:
                    group = [
                        stim.target_pauli(
                            _plain_qubit(target, instruction.name),
                            _PAIR_MEASUREMENT_BASES[instruction.name],
                            invert=target.is_inverted_result_target,
                        )
                        for target in source_group
                    ]
                else:
                    group = source_group
                support = _mpp_group_support(group)
                if support in supports_in_layer or any(
                    _supports_anticommute(support, existing) for existing in supports_in_layer
                ):
                    result.append("TICK", [])
                    supports_in_layer.clear()
                result.append(
                    "MPP",
                    _combined_group_targets(group),
                    instruction.gate_args_copy(),
                    tag=instruction.tag,
                )
                supports_in_layer.append(support)
        else:
            result.append(instruction)
    return result


def _supports_anticommute(
    left: tuple[tuple[int, str], ...],
    right: tuple[tuple[int, str], ...],
) -> bool:
    """Return whether two unsigned Pauli supports anticommute.

    Returns
    -------
    `bool`
        Whether an odd number of factor pairs anticommute.
    """
    left_paulis = dict(left)
    differing = sum(1 for qubit, pauli in right if left_paulis.get(qubit, pauli) != pauli)
    return differing % 2 == 1


def _mpp_group_support(group: Sequence[stim.GateTarget]) -> tuple[tuple[int, str], ...]:
    """Return the unsigned, order-independent support of one MPP product.

    Returns
    -------
    `tuple`[`tuple`[`int`, `str`], ...]
        Canonically ordered qubit and Pauli pairs.

    Raises
    ------
    UnsupportedSyndromeCircuitError
        If the group contains a non-Pauli target.
    """
    support: list[tuple[int, str]] = []
    for target in group:
        pauli = target.pauli_type
        if pauli not in _PAULI_CODES:
            msg = f"MPP contains a non-Pauli target {target!r}."
            raise UnsupportedSyndromeCircuitError(msg)
        support.append((_plain_qubit(target, "MPP"), pauli))
    return tuple(sorted(support))


def _combined_group_targets(group: Sequence[stim.GateTarget]) -> list[stim.GateTarget]:
    """Restore Stim combiners between the factors of one MPP product.

    Returns
    -------
    `list`[`stim.GateTarget`]
        Product targets in Stim's combined-target representation.
    """
    targets: list[stim.GateTarget] = []
    for target in group:
        if targets:
            targets.append(stim.target_combiner())
        targets.append(target)
    return targets


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
