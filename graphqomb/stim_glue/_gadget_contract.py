"""Optional local MPP extraction after destructive measurement normalization.

Every direct readout already ends its wire. Recognizing a controlled-Pauli
extraction only replaces the implementation of that interval; unrecognized
intervals keep their gates and destructive readouts for ordinary MBQC lowering.
"""

from __future__ import annotations

from dataclasses import dataclass

import stim

from graphqomb.stim_glue._parse import ANNOTATION_GATES, RESET_AXES, SINGLE_MEASUREMENT_AXES, iter_instructions

_RESET_BASES = {name: axis.name for name, axis in RESET_AXES.items()}
_MEASUREMENT_BASES = {name: axis.name for name, axis in SINGLE_MEASUREMENT_AXES.items()}


@dataclass(frozen=True)
class GadgetContraction:
    """Rewritten circuit and emitted-record to original-record mapping."""

    circuit: stim.Circuit
    record_to_source: tuple[int, ...]
    eliminated_qubits: tuple[int, ...]


def _canonical_controlled_body(
    pending: stim.Circuit, basis: str, rank: dict[int, int]
) -> tuple[stim.Circuit, dict[int, int]] | None:
    """Convert Z-readout CNOT targets into X-readout controls of CZ.

    Remove data targets from the candidate control ranks. The returned body
    is used for algebra only; it is not emitted into the rewritten circuit.

    Returns
    -------
    tuple[stim.Circuit, dict[int, int]] | None
        Equivalent controlled-Pauli body and candidate control ranks, or None
        for another gate structure.
    """
    if basis == "Z":
        # H on a Z-readout ancilla turns data->ancilla CX into an
        # X-prepared/X-measured control of CZ. No gates are actually emitted.
        controls = {t.value for gate in iter_instructions(pending) for t in gate.targets_copy()[::2]}
        rank = {q: position for q, position in rank.items() if q not in controls}
        transformed = stim.Circuit()
        for gate in iter_instructions(pending):
            if gate.name != "CX":
                return None
            pairs = gate.targets_copy()
            for data, ancilla in zip(pairs[::2], pairs[1::2], strict=True):
                if ancilla.value not in rank or data.value in rank:
                    return None
                transformed.append("CZ", [ancilla, data])
        pending = transformed
    # CX targets cannot be controls of the extraction gadgets recognized here.
    for gate in iter_instructions(pending):
        if gate.name not in {"CX", "CZ"}:
            return None
        if gate.name == "CX":
            for target in gate.targets_copy()[1::2]:
                rank.pop(target.value, None)
    return pending, rank


def _structural_products(  # ruff:ignore[too-many-branches, complex-structure, too-many-return-statements, too-many-locals]
    pending: stim.Circuit,
    instruction: stim.CircuitInstruction,
    prepared: dict[int, str],
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
    if not pending or instruction.name not in {"MX", "M"}:
        return [], []
    targets = instruction.targets_copy()
    if len({t.value for t in targets}) != len(targets):
        return [], []
    basis = _MEASUREMENT_BASES[instruction.name]
    rank = {t.value: p for p, t in enumerate(targets) if prepared.get(t.value) == basis}
    canonical = _canonical_controlled_body(pending, basis, rank)
    if canonical is None:
        return [], []
    pending, rank = canonical
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


def _signed_product_targets(product: stim.PauliString) -> list[stim.GateTarget]:
    """Encode a product's overall sign on its first factor and insert combiners.

    Returns
    -------
    list[stim.GateTarget]
        Targets for a single MPP, independent of Stim's scalar return annotation
        on target_combined_paulis in the currently supported version.
    """
    targets: list[stim.GateTarget] = []
    for qubit in product.pauli_indices():
        inverted = not targets and product.sign == -1
        if targets:
            targets.append(stim.target_combiner())
        targets.append(stim.target_pauli(qubit, product[qubit], invert=inverted))
    return targets


def contract_extraction_gadgets(circuit: stim.Circuit) -> GadgetContraction:  # ruff:ignore[too-many-locals, complex-structure, too-many-branches, too-many-statements]
    """Replace controlled-Pauli extraction intervals with data-only MPPs.

    Preparation removal is tied to the particular lifetime that was replaced,
    never to every occurrence of a physical Stim id. An unrelated later
    lifetime or reset-only output retains its preparation.

    Returns
    -------
    GadgetContraction
        Rewritten circuit, record permutation, and fully eliminated source ids.
    """
    emitted: list[stim.CircuitInstruction] = []
    pending = stim.Circuit()
    buffered: list[stim.CircuitInstruction] = []
    prepared: dict[int, str] = {}
    preparation_sites: dict[int, list[int]] = {}
    removed: dict[int, set[int]] = {}
    old_to_new: dict[int, int] = {}
    record_to_source: list[int] = []
    old_count = 0
    contracted_qubits: set[int] = set()
    for instruction in iter_instructions(circuit):
        name = instruction.name
        targets = instruction.targets_copy()
        mapped = stim.CircuitInstruction(
            name,
            [
                stim.target_rec(old_to_new[old_count + t.value] - len(record_to_source))
                if t.is_measurement_record_target
                else t
                for t in targets
            ],
            instruction.gate_args_copy(),
            tag=instruction.tag,
        )
        if name in ANNOTATION_GATES:
            (buffered if pending else emitted).append(mapped)
            continue
        if stim.gate_data(name).is_unitary and not any(t.is_measurement_record_target for t in targets):
            pending.append(instruction)
            buffered.append(instruction)
            continue
        positions: list[int] = []
        products: list[stim.PauliString] = []
        if name in _MEASUREMENT_BASES:
            positions, products = _structural_products(pending, instruction, prepared, circuit.num_qubits)
        if positions:
            emitted.extend(op for op in buffered if op.name in ANNOTATION_GATES)
            chosen = set(positions)
            remaining = [p for p in range(len(targets)) if p not in chosen]
            for position, product in zip(positions, products, strict=True):
                emitted.append(stim.CircuitInstruction("MPP", _signed_product_targets(product), tag=instruction.tag))
                qubit = targets[position].value
                contracted_qubits.add(qubit)
                for site in preparation_sites[qubit]:
                    removed.setdefault(site, set()).add(qubit)
            if remaining:
                emitted.extend(
                    [
                        stim.CircuitInstruction("TICK", []),
                        stim.CircuitInstruction(name, [targets[p] for p in remaining], tag=instruction.tag),
                    ]
                )
            order = positions + remaining
        else:
            emitted.extend(buffered)
            emitted.append(mapped)
            order = list(range(instruction.num_measurements))
        if pending or instruction.num_measurements or stim.gate_data(name).is_unitary:
            # A measurement or a feedback boundary ends this extraction interval.
            touched = {
                int(t.qubit_value)
                for op in [*iter_instructions(pending), instruction]
                for t in op.targets_copy()
                if t.qubit_value is not None
            }
            for qubit in touched:
                prepared.pop(qubit, None)
                preparation_sites.pop(qubit, None)
        if name in _RESET_BASES:
            for target in targets:
                prepared[target.value] = _RESET_BASES[name]
                preparation_sites.setdefault(target.value, []).append(len(emitted) - 1)
        pending = stim.Circuit()
        buffered = []
        for offset in order:
            old_to_new[old_count + offset] = len(record_to_source)
            record_to_source.append(old_count + offset)
        old_count += instruction.num_measurements
    emitted.extend(buffered)
    output, eliminated = _remove_extracted_preparations(emitted, removed, contracted_qubits)
    return GadgetContraction(output, tuple(record_to_source), eliminated)


def _remove_extracted_preparations(
    emitted: list[stim.CircuitInstruction],
    removed: dict[int, set[int]],
    contracted_qubits: set[int],
) -> tuple[stim.Circuit, tuple[int, ...]]:
    """Remove preparation sites of replaced lifetimes, retaining round boundaries.

    Returns
    -------
    tuple[stim.Circuit, tuple[int, ...]]
        Cleaned circuit and the source ids absent from every remaining lifetime.
    """
    output = stim.Circuit()
    measurements_since_tick = False
    for index, instruction in enumerate(emitted):
        if index in removed:
            targets = [t for t in instruction.targets_copy() if t.value not in removed[index]]
            if targets:
                output.append(instruction.name, targets, instruction.gate_args_copy(), tag=instruction.tag)
            # Preserve the lifetime/round boundary even if its preparation vanished.
            if measurements_since_tick:
                output.append("TICK", [])
                measurements_since_tick = False
        else:
            output.append(instruction)
            if instruction.name == "TICK":
                measurements_since_tick = False
            else:
                measurements_since_tick |= instruction.num_measurements > 0
    retained = {
        int(t.qubit_value)
        for op in iter_instructions(output)
        if op.name not in ANNOTATION_GATES and op.name != "MPAD"
        for t in op.targets_copy()
        if t.qubit_value is not None
    }
    eliminated = contracted_qubits - retained
    cleaned = stim.Circuit()
    for instruction in iter_instructions(output):
        if instruction.name == "QUBIT_COORDS":
            targets = [t for t in instruction.targets_copy() if t.value not in eliminated]
            if targets:
                cleaned.append(instruction.name, targets, instruction.gate_args_copy(), tag=instruction.tag)
        else:
            cleaned.append(instruction)
    return cleaned, tuple(sorted(eliminated))
