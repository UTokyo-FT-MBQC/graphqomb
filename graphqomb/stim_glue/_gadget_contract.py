"""Contract disposable extraction ancillas for the classical-output circuit.

This optimization is separate from the exact quantum-channel rewrite. A
controlled-Pauli identity replaces recognized gadgets with data-only MPPs
after discarding their measured ancillas. Those ancillas must be reset before their next use,
or have no further quantum use. Measurement permutations are explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import stim

from graphqomb.stim_glue._parse import ANNOTATION_GATES, iter_instructions

_RESET_BASES = {"R": "Z", "RX": "X", "RY": "Y"}
_MEASUREMENT_BASES = {"M": "Z", "MX": "X", "MY": "Y"}


@dataclass(frozen=True)
class GadgetContraction:
    """Optimized source and its emitted-record to original-record mapping."""

    circuit: stim.Circuit
    record_to_source: tuple[int, ...]
    discarded_qubits: frozenset[int]


def _qubits(instruction: stim.CircuitInstruction) -> set[int]:
    return {int(t.qubit_value) for t in instruction.targets_copy() if t.qubit_value is not None}


def _disposable_targets(instructions: list[stim.CircuitInstruction]) -> dict[int, set[int]]:
    next_use: dict[int, str] = {}
    disposable = {}
    for index in reversed(range(len(instructions))):
        instruction = instructions[index]
        if instruction.name in ANNOTATION_GATES or instruction.name == "MPAD":
            continue
        qubits = _qubits(instruction)
        if instruction.name in _MEASUREMENT_BASES:
            disposable[index] = {q for q in qubits if q not in next_use or next_use[q] in _RESET_BASES}
        next_use.update(dict.fromkeys(qubits, instruction.name))
    return disposable


def _append_products(circuit: stim.Circuit, products: list[stim.PauliString], tag: str) -> None:
    for product in products:
        circuit.append("MPP", stim.target_combined_paulis(product), tag=tag)


def _structural_products(  # ruff:ignore[too-many-branches, complex-structure, too-many-return-statements]
    pending: stim.Circuit,
    instruction: stim.CircuitInstruction,
    prepared: dict[int, str],
    disposable: set[int],
    num_qubits: int,
) -> tuple[list[int], list[stim.PauliString]]:
    """Apply the controlled-Pauli extraction identity without a flow oracle.

    Group gates by ancilla in measurement order. Swapping anticommuting
    controlled Paulis contributes CZ between their controls. The parity set
    tracks these algebraic terms; contraction applies only when they cancel.
    Within each ancilla, factors on a given data qubit must share an axis.
    Thus every grouped body is a Hermitian controlled Pauli, and X readout
    of its |+> control has Kraus operator (I + (-1)**m P) / 2.

    Returns
    -------
    tuple[list[int], list[stim.PauliString]]
        Selected measurement positions and their data products, or empty lists.
    """
    if not pending or instruction.name != "MX":
        return [], []
    targets = instruction.targets_copy()
    if len({t.value for t in targets}) != len(targets):
        return [], []
    rank = {t.value: p for p, t in enumerate(targets) if prepared.get(t.value) == "X" and t.value in disposable}
    # CX targets cannot be controls of the extraction gadgets recognized here.
    for gate in iter_instructions(pending):
        if gate.name not in {"CX", "CZ"}:
            return [], []
        if gate.name == "CX":
            for target in gate.targets_copy()[1::2]:
                rank.pop(target.value, None)
    supports: dict[int, dict[int, str]] = {}
    axes: dict[tuple[int, int], str] = {}
    preceding: dict[int, dict[int, str]] = {}
    phases: set[tuple[int, int]] = set()
    for gate in iter_instructions(pending):
        pairs = gate.targets_copy()
        for left, right in zip(pairs[::2], pairs[1::2], strict=True):
            a, q = left.value, right.value
            if not left.is_qubit_target or not right.is_qubit_target:
                return [], []
            if gate.name == "CZ" and a not in rank:
                a, q = q, a
            if a not in rank or q in rank:
                return [], []
            axis = "X" if gate.name == "CX" else "Z"
            if axes.get((a, q), axis) != axis:
                return [], []
            axes[a, q] = axis
            prior = preceding.setdefault(q, {})
            for b, previous_axis in prior.items():
                if rank[b] > rank[a] and previous_axis != axis:
                    pair = (a, b)
                    phases.symmetric_difference_update({pair})
            support = supports.setdefault(a, {})
            if q in support:
                del support[q]
                del prior[a]
            else:
                support[q] = axis
                prior[a] = axis
    if phases or not supports or any(not support for support in supports.values()):
        return [], []
    positions, products = [], []
    for position, target in enumerate(targets):
        if target.value not in supports:
            continue
        product = stim.PauliString(num_qubits)
        for q, axis in supports[target.value].items():
            product[q] = axis
        if target.is_inverted_result_target:
            product.sign = -1
        positions.append(position)
        products.append(product)
    return positions, products


def _update_preparations(prepared: dict[int, str], instruction: stim.CircuitInstruction, pending: stim.Circuit) -> None:
    if pending or instruction.num_measurements or stim.gate_data(instruction.name).is_unitary:
        # Feedback can change a prepared sign even with no pending body.
        prepared.clear()
    if instruction.name in _RESET_BASES:
        prepared.update(dict.fromkeys(_qubits(instruction), _RESET_BASES[instruction.name]))


def contract_disposable_gadgets(circuit: stim.Circuit) -> GadgetContraction:  # ruff:ignore[too-many-locals]
    """Contract recognized disposable gadgets, remapping records when readouts are mixed.

    Returns
    -------
    GadgetContraction
        Source circuit with recognized extraction bodies removed, plus the
        record permutation and original ancilla IDs eligible for cleanup.
    """
    instructions = list(iter_instructions(circuit.flattened()))
    disposable = _disposable_targets(instructions)
    num_qubits = circuit.num_qubits
    output, pending = stim.Circuit(), stim.Circuit()
    prepared: dict[int, str] = {}
    old_to_new: dict[int, int] = {}
    record_to_source: list[int] = []
    discarded: set[int] = set()
    old_count = 0
    for index, instruction in enumerate(instructions):
        name = instruction.name
        targets = instruction.targets_copy()
        # Includes DETECTOR, OBSERVABLE_INCLUDE, and measurement feedback.
        mapped_targets = [
            stim.target_rec(old_to_new[old_count + t.value] - len(record_to_source))
            if t.is_measurement_record_target
            else t
            for t in targets
        ]
        mapped = stim.CircuitInstruction(name, mapped_targets, instruction.gate_args_copy(), tag=instruction.tag)
        if name in ANNOTATION_GATES:
            output.append(mapped)
            continue
        if stim.gate_data(name).is_unitary and not any(t.is_measurement_record_target for t in targets):
            pending.append(instruction)
            continue
        positions: list[int] = []
        products: list[stim.PauliString] = []
        if name in _MEASUREMENT_BASES:
            positions, products = _structural_products(pending, instruction, prepared, disposable[index], num_qubits)
        if positions:
            chosen = set(positions)
            remaining = [p for p in range(len(targets)) if p not in chosen]
            _append_products(output, products, instruction.tag)
            if remaining:
                output.append("TICK", [])
                output.append(name, [targets[p] for p in remaining], tag=instruction.tag)
            discarded.update(targets[p].value for p in positions)
            order = positions + remaining
        else:
            output += pending
            output.append(mapped)
            order = list(range(instruction.num_measurements))
        _update_preparations(prepared, instruction, pending)
        pending = stim.Circuit()
        for offset in order:
            old_to_new[old_count + offset] = len(record_to_source)
            record_to_source.append(old_count + offset)
        old_count += instruction.num_measurements
    output += pending
    return GadgetContraction(output, tuple(record_to_source), frozenset(discarded))
