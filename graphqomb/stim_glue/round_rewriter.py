"""Local extraction of closed Pauli probes, preserving successive instruments.

This pass never uses a known data state or a detector relation to delete a
measurement. Its default algebraic domain is a layer of independent
controlled-Pauli probes. Other blocks are copied, including their residual
quantum action.

With ``interacting_probes=True`` a window rejected by that algebra is analyzed
through its stabilizer flows instead. Probes may then interact with each
other, as in superdense extraction where two probes of a plaquette share a
Bell pair. Such a window is rewritten as data MPPs followed by Pauli gates
controlled by the new records, and only when the two circuits have identical
flow groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import stim

from graphqomb.stim_glue._parse import iter_instructions
from graphqomb.stim_glue._t_gates import T_ANGLES

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

_RESETS = {"R": "Z", "RX": "X", "RY": "Y"}
_MEASUREMENTS = {"M": "Z", "MX": "X", "MY": "Y"}
_METADATA = {"QUBIT_COORDS", "SHIFT_COORDS", "TICK", "DETECTOR", "OBSERVABLE_INCLUDE"}
_READOUT_NAMES = {"Z": "M", "X": "MX", "Y": "MY"}
_PAULI_X = 1
_PAULI_Y = 2
_PAULI_Z = 3


@dataclass(frozen=True)
class RoundRewriteResult:
    """Time-ordered circuit and source-to-output absolute record permutation.

    ``retained_reasons`` explains blocks with reset/readout candidates that
    did not fit the local algebra. Coordinates are never used as role labels.
    ``eliminated_probes`` counts lifetimes, not distinct physical qubit IDs.
    ``rewritten_rounds`` counts measurement-delimited windows, not inferred
    QEC rounds. Resets, TICKs, and record annotations do not split a window.
    """

    circuit: stim.Circuit
    record_map: tuple[int, ...]
    rewritten_rounds: int
    eliminated_probes: int
    retained_reasons: tuple[str, ...]


@cache
def _gate_tableau(name: str) -> stim.Tableau:
    return stim.Tableau.from_named_gate(name)


def _referenced_qubits(instruction: stim.CircuitInstruction) -> set[int]:
    """Return quantum and Pauli-observable targets, excluding coordinates and pad bits.

    Returns
    -------
    set[int]
        Physical qubit IDs referenced by the instruction.
    """
    if instruction.name in {"QUBIT_COORDS", "MPAD"}:
        return set()
    return {target.value for target in instruction.targets_copy() if target.qubit_value is not None}


def _remove_unused_probe_coordinates(circuit: stim.Circuit, eliminated: set[int]) -> stim.Circuit:
    """Remove coordinates only for eliminated IDs with no remaining reference.

    Returns
    -------
    stim.Circuit
        Circuit with unused eliminated-probe coordinate targets removed.
    """
    if not eliminated:
        return circuit
    unused = eliminated.copy()
    for inst in iter_instructions(circuit):
        unused.difference_update(_referenced_qubits(inst))
    if not unused:
        return circuit
    output = stim.Circuit()
    for inst in iter_instructions(circuit):
        if inst.name == "QUBIT_COORDS":
            targets = [target for target in inst.targets_copy() if target.value not in unused]
            if targets:
                output.append(inst.name, targets, inst.gate_args_copy(), tag=inst.tag)
        else:
            output.append(inst)
    return output


def rewrite_syndrome_rounds(  # ruff: ignore[complex-structure, too-many-branches, too-many-locals, too-many-statements]
    circuit: stim.Circuit,
    *,
    interacting_probes: bool = False,
) -> RoundRewriteResult:
    """Replace closed extraction layers with successive data MPPs.

    Each probe must have an explicit reset, a final single-qubit measurement,
    and no subsequent quantum use before another reset (or circuit end).
    Within a candidate layer, its interactions must share one control axis,
    its prepared/readout axes must match and be transverse to that control,
    and the collected data products must form a commuting Hermitian layer
    without a residual inter-probe phase. Data Clifford gates and preparation
    operations are retained. Failed factorizations leave the entire block
    unchanged. This default rule uses no runtime channel comparison or
    flow-generator check.

    The resulting instrument on all remaining wires and the complete record
    vector are preserved for arbitrary block input. Only closed probe outputs
    are discarded. This is a structural compilation policy, not inference of
    the intended logical subsystem. Existing MPPs are barriers, not optimized.
    Coordinates of eliminated probes are removed only if no operation or
    Pauli observable anywhere in the rewritten circuit still references them.
    T and T_DAG transport gates are barriers and are copied unchanged.

    Parameters
    ----------
    circuit : stim.Circuit
        Ideal Clifford circuit. Explicit noise must be removed by the caller.
    interacting_probes : bool, optional
        Also rewrite windows whose probes interact with each other or change
        their control axis, such as superdense extraction with a Bell pair per
        plaquette. A window rejected by the default rule is then analyzed
        together with the directly following single-qubit readouts. Each probe
        record must measure a Pauli product of the window input, and the
        remaining action must be the retained data operations followed by
        Pauli gates controlled by the probe records. The rewrite is emitted
        only if its stabilizer flows and those of the source window generate
        the same group, so it preserves the same instrument. Probes with a
        deterministic record are not extracted. By default False.

    Returns
    -------
    RoundRewriteResult
        Import ``result.circuit`` with ``stim_circuit_to_pattern``. Its record
        references already include the permutation. Pauli feedback is retained
        for lowering to the importer's xflow/zflow.

    Raises
    ------
    ValueError
        If noise is present.
    """
    instructions = [inst for inst in circuit.flattened() if isinstance(inst, stim.CircuitInstruction)]
    width = circuit.num_qubits
    # Whether each measured coordinate's next reference is a reset. Pauli
    # observables protect their qubits; coordinates and record bits do not.
    next_use: dict[int, str] = {}
    closed: dict[int, set[int]] = {}
    for index in range(len(instructions) - 1, -1, -1):
        inst = instructions[index]
        name = inst.name
        data = stim.gate_data(name)
        noiseless_measurements = {*_MEASUREMENTS, "MR", "MRX", "MRY", "MPP", "MXX", "MYY", "MZZ"}
        if data.is_noisy_gate and (name not in noiseless_measurements or any(inst.gate_args_copy())):
            msg = "rewrite_syndrome_rounds requires ideal input; remove noise explicitly."
            raise ValueError(msg)
        qubits = _referenced_qubits(inst)
        if name in _MEASUREMENTS:
            closed[index] = {q for q in qubits if q not in next_use or next_use[q] in _RESETS}
        for q in qubits:
            next_use[q] = name

    output = stim.Circuit()
    record_map: dict[int, int] = {}
    source_count = 0
    output_count = 0
    pending: list[tuple[stim.CircuitInstruction, int]] = []
    rounds = probes = 0
    eliminated: set[int] = set()
    reasons: list[str] = []

    def emit(inst: stim.CircuitInstruction, before: int, origins: list[int] | None = None) -> None:
        nonlocal output_count
        targets = [
            stim.target_rec(record_map[before + t.value] - output_count) if t.is_measurement_record_target else t
            for t in inst.targets_copy()
        ]
        count = inst.num_measurements
        output.append(inst.name, targets, inst.gate_args_copy(), tag=inst.tag)
        record_map.update(
            zip(
                range(before, before + count) if origins is None else origins,
                range(output_count, output_count + count),
                strict=True,
            )
        )
        output_count += count

    def emit_window(window: _InteractingWindow, readout_group: Sequence[stim.CircuitInstruction], before: int) -> int:
        tick = stim.CircuitInstruction("TICK")
        for old, old_before in pending[: window.prefix]:
            emit(old, old_before)
        emit(tick, before)
        for offset, product in zip(window.order, window.products, strict=False):
            emit(product, before, [before + offset])
        emit(tick, before)
        for retained in window.body:
            emit(retained, before)
        if window.finals:
            emit(tick, before)
        for offset, readout in zip(window.order[len(window.products) :], window.finals, strict=True):
            emit(readout, before, [before + offset])
        if window.feedback:
            emit(tick, before)
            # Lookbacks already count back from the last record of the window.
            for feedback in window.feedback:
                output.append(feedback)
            emit(tick, before)
        for inst in readout_group:
            if inst.name not in _MEASUREMENTS:
                emit(inst, before)
            before += inst.num_measurements
        return before

    index = 0
    while index < len(instructions):
        inst = instructions[index]
        count = inst.num_measurements
        if inst.name in _MEASUREMENTS:
            rewritten, removed, reason = _extract_layer(pending, inst, source_count, closed[index], width)
            if rewritten is None and interacting_probes:
                stop = _readout_group_end(instructions, index)
                group_closed = {q for i in range(index, stop) for q in closed.get(i, ())}
                window, window_reason = _extract_interacting_window(
                    [old for old, _ in pending], instructions[index:stop], group_closed
                )
                if window is not None:
                    source_count = emit_window(window, instructions[index:stop], source_count)
                    rounds += 1
                    probes += len(window.removed)
                    eliminated.update(window.removed)
                    pending.clear()
                    index = stop
                    continue
                reason = window_reason or reason
            if rewritten is None:
                for old, before in pending:
                    emit(old, before)
                emit(inst, source_count)
                if reason:
                    reasons.append(reason)
            else:
                for new, before, origins in rewritten:
                    emit(new, before, origins)
                rounds += 1
                probes += len(removed)
                eliminated.update(removed)
            pending.clear()
        elif count or inst.name in {"MR", "MRX", "MRY"} or inst.tag in T_ANGLES:
            for old, before in pending:
                emit(old, before)
            pending.clear()
            emit(inst, source_count)
        else:
            pending.append((inst, source_count))
        source_count += count
        index += 1
    for inst, before in pending:
        emit(inst, before)
    output = _remove_unused_probe_coordinates(output, eliminated)
    return RoundRewriteResult(output, tuple(record_map[i] for i in range(source_count)), rounds, probes, tuple(reasons))


def _refresh_support(
    support_index: dict[int, set[int]],
    product: stim.PauliString,
    probe: int,
    qubits: Iterable[int],
) -> None:
    """Keep ``support_index[q]`` equal to the probes whose data product is non-identity on ``q``.

    Only the qubits a gate or factor touched can change, so refreshing those keeps the index exact.
    """
    for qubit in qubits:
        probes = support_index.setdefault(qubit, set())
        if product[qubit]:
            probes.add(probe)
        else:
            probes.discard(probe)


def _extract_layer(  # ruff: ignore[complex-structure, too-many-branches, too-many-statements, too-many-locals, too-many-return-statements]
    pending: list[tuple[stim.CircuitInstruction, int]],
    readout: stim.CircuitInstruction,
    source_count: int,
    closed: set[int],
    width: int,
) -> tuple[list[tuple[stim.CircuitInstruction, int, list[int] | None]] | None, set[int], str]:
    resets = {t.value for inst, _ in pending if inst.name in _RESETS for t in inst.targets_copy()}
    measured = [t.value for t in readout.targets_copy()]
    candidates = resets & set(measured) & closed
    if not candidates:
        return None, set(), ""
    if len(set(measured)) != len(measured):
        return None, set(), "repeated readout target"
    # U = L_A (product_a controlled(P_a)) V_D. Reordering different
    # controls produces CZ phases on A; retain these symbolically until the
    # complete layer has been collected. No state of D enters this identity.
    local = {a: stim.Tableau(1) for a in candidates}
    prepared: dict[int, str] = {}
    pointer: dict[int, stim.PauliString] = {}
    products = {a: stim.PauliString(width) for a in candidates}
    # data qubit -> probes whose product currently acts on it; conjugation by a
    # data gate and factor multiplication only touch a few qubits, so every
    # product update below is proportional to the gate's locality, not to the
    # number of probes in the layer.
    support_index: dict[int, set[int]] = {}
    phases: set[tuple[int, int]] = set()
    result: list[tuple[stim.CircuitInstruction, int, list[int] | None]] = []
    touched: set[int] = set()
    for inst, before in pending:
        name = inst.name
        targets = inst.targets_copy()
        if name in _METADATA:
            if name == "OBSERVABLE_INCLUDE" and any(not t.is_measurement_record_target for t in targets):
                return None, set(), "Pauli-target observable boundary"
            result.append((inst, before, None))
            continue
        if any(not t.is_qubit_target for t in targets):
            return None, set(), "classical control or Pauli-product gate inside layer"
        if name in _RESETS:
            keep = []
            for target in targets:
                q = target.value
                if q in touched:
                    return None, set(), "reset after a quantum operation inside layer"
                if q in candidates:
                    if q in prepared:
                        return None, set(), "multiple probe resets inside layer"
                    prepared[q] = _RESETS[name]
                else:
                    keep.append(target)
            if keep:
                result.append((stim.CircuitInstruction(name, keep, tag=inst.tag), before, None))
            continue
        data = stim.gate_data(name)
        if not data.is_unitary or data.takes_pauli_targets:
            return None, set(), "operation outside controlled-Pauli layer"
        arity = 1 if data.is_single_qubit_gate else 2
        for start in range(0, len(targets), arity):
            group = targets[start : start + arity]
            qubits = [t.value for t in group]
            touched.update(qubits)
            anc = candidates.intersection(qubits)
            if not anc:
                affected: set[int] = set()
                for q in qubits:
                    affected |= support_index.get(q, set())
                for a in affected:
                    products[a] = products[a].after(_gate_tableau(name), targets=qubits)
                    _refresh_support(support_index, products[a], a, qubits)
                result.append((stim.CircuitInstruction(name, group, tag=inst.tag), before, None))
            elif len(anc) > 1:
                return None, set(), "interaction between probe candidates"
            else:
                a = next(iter(anc))
                if a not in prepared:
                    return None, set(), "probe used before reset"
                if arity == 1:
                    local[a] = local[a].then(_gate_tableau(name))
                    continue
                if name not in {"CX", "CY", "CZ"}:
                    return None, set(), "unsupported probe interaction"
                control_axis, data_axis = ("Z", name[-1]) if qubits[0] == a else (name[-1], "Z")
                pulled = local[a].inverse()(stim.PauliString(control_axis))
                if a in pointer and pointer[a] != pulled:
                    return None, set(), "probe control axis changes during extraction"
                pointer[a] = pulled
                if pulled.commutes(stim.PauliString(prepared[a])):
                    return None, set(), "probe preparation not transverse to control"
                d = qubits[1] if qubits[0] == a else qubits[0]
                factor = stim.PauliString(width)
                factor[d] = data_axis
                factor_d = factor[d]
                for b in support_index.get(d, ()):
                    if b < a and products[b][d] != factor_d:
                        phases.symmetric_difference_update({(b, a)})
                products[a] = factor * products[a]
                _refresh_support(support_index, products[a], a, (d,))
    if phases:
        return None, set(), "residual phase between probes"
    ordered = sorted(candidates)
    for a in ordered:
        product = products[a]
        if a not in pointer or product.weight == 0 or product.sign not in {1, -1}:
            return None, set(), "probe has trivial or non-Hermitian data product"
        # Disjoint products commute trivially; compare only against earlier probes sharing support.
        overlapping: set[int] = set()
        for q in product.pauli_indices():
            overlapping |= support_index.get(q, set())
        if any(b < a and not product.commutes(products[b]) for b in overlapping):
            return None, set(), "noncommuting final data products"
        pulled = local[a].inverse()(stim.PauliString(_MEASUREMENTS[readout.name]))
        positive = stim.PauliString(prepared[a])
        if pulled[0] != positive[0]:
            return None, set(), "residual probe phase requires a quantum correction"
        if pulled.sign == -1:
            products[a] *= -1
    # All probe projections precede data readout, even if Stim lists the data
    # first in a mixed readout instruction. Permute record references globally.
    result.append((stim.CircuitInstruction("TICK"), source_count, None))
    for index, target in enumerate(readout.targets_copy()):
        a = target.value
        if a not in candidates:
            continue
        p = products[a]
        negative = (p.sign == -1) ^ target.is_inverted_result_target
        mpp_targets: list[stim.GateTarget] = []
        for q in p.pauli_indices():
            if mpp_targets:
                mpp_targets.append(stim.target_combiner())
            mpp_targets.append(stim.target_pauli(q, "_XYZ"[p[q]], invert=negative and not mpp_targets))
        result.append(
            (stim.CircuitInstruction("MPP", mpp_targets, tag=readout.tag), source_count, [source_count + index])
        )
    result.append((stim.CircuitInstruction("TICK"), source_count, None))
    for index, target in enumerate(readout.targets_copy()):
        if target.value not in candidates:
            result.append(
                (stim.CircuitInstruction(readout.name, [target], tag=readout.tag), source_count, [source_count + index])
            )
    return result, candidates, ""


@dataclass(frozen=True)
class _InteractingWindow:
    """Rewrite of one window whose probes interact; see `_extract_interacting_window`.

    ``order`` lists the source-local record offsets in emitted order, first
    the extracted products and then the retained readouts. Feedback lookbacks
    count back from the last emitted record of the window.
    """

    prefix: int
    products: tuple[stim.CircuitInstruction, ...]
    body: tuple[stim.CircuitInstruction, ...]
    finals: tuple[stim.CircuitInstruction, ...]
    feedback: tuple[stim.CircuitInstruction, ...]
    order: tuple[int, ...]
    removed: frozenset[int]


class _Gf2Span:
    """Incremental span of bit vectors that remembers how each one was combined."""

    def __init__(self) -> None:
        self._basis: dict[int, tuple[int, int]] = {}
        self.dependent: list[int] = []

    def add(self, vector: int, combination: int) -> None:
        while vector:
            pivot = vector.bit_length() - 1
            if pivot not in self._basis:
                self._basis[pivot] = (vector, combination)
                return
            basis_vector, basis_combination = self._basis[pivot]
            vector ^= basis_vector
            combination ^= basis_combination
        self.dependent.append(combination)

    def solve(self, target: int) -> int | None:
        combination = 0
        while target:
            pivot = target.bit_length() - 1
            if pivot not in self._basis:
                return None
            basis_vector, basis_combination = self._basis[pivot]
            target ^= basis_vector
            combination ^= basis_combination
        return combination

    def reduce(self, vector: int) -> int:
        for pivot in sorted(self._basis, reverse=True):
            if vector >> pivot & 1:
                vector ^= self._basis[pivot][0]
        return vector


def _readout_group_end(instructions: Sequence[stim.CircuitInstruction], index: int) -> int:
    """Find the end of the single-qubit readouts starting at ``index``, allowing annotations between them.

    Returns
    -------
    int
        Index following the last readout instruction of the group.
    """
    stop = index + 1
    for position in range(index + 1, len(instructions)):
        name = instructions[position].name
        if name in _MEASUREMENTS:
            stop = position + 1
        elif name not in _METADATA:
            break
    return stop


def _pauli_bits(pauli: stim.PauliString) -> int:
    """Encode a Pauli string with bit ``2q`` for X and bit ``2q + 1`` for Z on qubit ``q``.

    Returns
    -------
    int
        Symplectic bit vector of the Pauli string, without its sign.
    """
    bits = 0
    for qubit in pauli.pauli_indices():
        kind = pauli[qubit]
        if kind in {_PAULI_X, _PAULI_Y}:
            bits |= 1 << (2 * qubit)
        if kind in {_PAULI_Y, _PAULI_Z}:
            bits |= 1 << (2 * qubit + 1)
    return bits


def _flow_is_negative(flow: stim.Flow) -> bool:
    return bool((flow.input_copy().sign * flow.output_copy().sign).real < 0)


def _combine_flows(flows: Sequence[stim.Flow], combination: int) -> stim.Flow | None:
    product: stim.Flow | None = None
    for index, flow in enumerate(flows):
        if combination >> index & 1:
            product = flow if product is None else product * flow
    return product


def _window_suffix_start(pending: Sequence[stim.CircuitInstruction]) -> int:
    """Find where the analyzable part of a window starts.

    A reset of a qubit already used in the window begins a new lifetime, and
    record-controlled or non-unitary operations cannot be moved. Everything
    before the last such position is copied unchanged.

    Returns
    -------
    int
        Number of leading window instructions to copy unchanged.
    """
    start = 0
    touched: set[int] = set()
    for position, inst in enumerate(pending):
        name = inst.name
        if name in _METADATA:
            continue
        qubits = _referenced_qubits(inst)
        targets = inst.targets_copy()
        data = stim.gate_data(name)
        if name in _RESETS:
            if touched & qubits:
                start = position
                touched = set()
        elif not data.is_unitary or data.takes_pauli_targets or any(not t.is_qubit_target for t in targets):
            start = position + 1
            touched = set()
            continue
        touched |= qubits
    return start


def _local_instruction(inst: stim.CircuitInstruction, local: Mapping[int, int]) -> stim.CircuitInstruction:
    targets = [
        stim.target_inv(local[t.value]) if t.is_inverted_result_target else stim.GateTarget(local[t.value])
        for t in inst.targets_copy()
    ]
    return stim.CircuitInstruction(inst.name, targets, inst.gate_args_copy())


def _without_probes(inst: stim.CircuitInstruction, probes: frozenset[int]) -> stim.CircuitInstruction | None:
    """Drop the probe resets and every gate application touching a probe.

    Returns
    -------
    stim.CircuitInstruction | None
        The remaining instruction, or None if nothing remains.
    """
    targets = inst.targets_copy()
    arity = 1 if inst.name in _RESETS or stim.gate_data(inst.name).is_single_qubit_gate else 2
    keep: list[stim.GateTarget] = []
    for start in range(0, len(targets), arity):
        group = targets[start : start + arity]
        if probes.isdisjoint(t.value for t in group):
            keep.extend(group)
    return stim.CircuitInstruction(inst.name, keep, inst.gate_args_copy(), tag=inst.tag) if keep else None


def _product_instruction(
    product: stim.PauliString, qubits: Sequence[int], *, negative: bool
) -> stim.CircuitInstruction:
    targets: list[stim.GateTarget] = []
    for local_qubit in product.pauli_indices():
        if targets:
            targets.append(stim.target_combiner())
        targets.append(
            stim.target_pauli(qubits[local_qubit], "_XYZ"[product[local_qubit]], invert=negative and not targets)
        )
    return stim.CircuitInstruction("MPP", targets)


def _extract_interacting_window(  # ruff: ignore[complex-structure, too-many-branches, too-many-statements, too-many-locals, too-many-return-statements]
    pending: Sequence[stim.CircuitInstruction],
    readout_group: Sequence[stim.CircuitInstruction],
    closed: set[int],
) -> tuple[_InteractingWindow | None, str]:
    r"""Rewrite a window with interacting probes as MPPs plus Pauli feedback.

    Let the window act on its non-probe wires as the instrument
    :math:`\{K_m\}`. The accepted form is
    :math:`K_m \propto B(m)\,V\,\prod_a (I + (-1)^{m_a} P_a)/2`, where the
    :math:`P_a` are commuting Pauli products of the window input, :math:`V` is
    the retained non-probe part of the window, and :math:`B(m)` is a Pauli
    operator linear in the probe records. The products are read from the
    stabilizer flows that map an input Pauli to a single probe record, and
    :math:`B` from the record terms by which the remaining flows differ from
    those of the measurements alone. The candidate is accepted only if its
    flow generators are flows of the source window and conversely, so both
    circuits generate the same flow group.

    Returns
    -------
    tuple[_InteractingWindow | None, str]
        The rewrite, or None with the reason for retaining the window.
    """
    start = _window_suffix_start(pending)
    suffix = pending[start:]
    readouts = [inst for inst in readout_group if inst.name in _MEASUREMENTS]
    for inst in (*suffix, *readout_group):
        if inst.name == "OBSERVABLE_INCLUDE" and any(not t.is_measurement_record_target for t in inst.targets_copy()):
            return None, "Pauli-target observable boundary"
    reset_qubits = {t.value for inst in suffix if inst.name in _RESETS for t in inst.targets_copy()}
    # (qubit, axis, inverted) per source record of the group.
    measured = [
        (t.value, _MEASUREMENTS[inst.name], t.is_inverted_result_target)
        for inst in readouts
        for t in inst.targets_copy()
    ]
    measured_qubits = [qubit for qubit, _, _ in measured]
    probes = frozenset(reset_qubits & set(measured_qubits) & closed)
    if not probes:
        return None, ""
    if len(set(measured_qubits)) != len(measured_qubits):
        return None, "repeated readout target"
    operations = [inst for inst in suffix if inst.name not in _METADATA]
    qubits = sorted({q for inst in (*operations, *readouts) for q in _referenced_qubits(inst)})
    local = {qubit: position for position, qubit in enumerate(qubits)}
    width = len(qubits)
    record_count = len(measured)
    probe_offsets = [offset for offset, (qubit, _, _) in enumerate(measured) if qubit in probes]

    # Both circuits end by resetting the probes, whose outputs are discarded.
    tail = stim.Circuit()
    tail.append("I", [width - 1])
    tail.append("R", sorted(local[probe] for probe in probes))
    source = stim.Circuit()
    for inst in (*operations, *readouts):
        source.append(_local_instruction(inst, local))
    source += tail
    source_flows = source.flow_generators()
    out_shift, record_shift = 2 * width, 4 * width

    def record_bits(flow: stim.Flow, record_of: Sequence[int] | None = None) -> int:
        bits = 0
        for record in flow.measurements_copy():
            offset = record % record_count
            bits ^= 1 << (offset if record_of is None else record_of[offset])
        return bits

    # Input products: flows from an input Pauli to exactly one probe record.
    by_output_and_record = _Gf2Span()
    for position, flow in enumerate(source_flows):
        key = _pauli_bits(flow.output_copy()) << out_shift | record_bits(flow) << record_shift
        by_output_and_record.add(key, 1 << position)
    products: list[stim.CircuitInstruction] = []
    for offset in probe_offsets:
        combination = by_output_and_record.solve(1 << (record_shift + offset))
        flow = None if combination is None else _combine_flows(source_flows, combination)
        if flow is None:
            return None, "probe record is not a Pauli product of the window input"
        product = flow.input_copy()
        if product.weight == 0 or product.sign not in {1, -1}:
            return None, "probe has trivial or non-Hermitian data product"
        if any(product[local[probe]] for probe in probes if local[probe] < len(product)):
            return None, "probe product acts on a probe"
        # The flow sign already includes an inverted readout target.
        products.append(_product_instruction(product, qubits, negative=_flow_is_negative(flow)))

    body: list[stim.CircuitInstruction] = []
    for inst in suffix:
        retained = inst if inst.name in _METADATA else _without_probes(inst, probes)
        if retained is not None:
            body.append(retained)
    finals = [
        stim.CircuitInstruction(_READOUT_NAMES[axis], [stim.target_inv(qubit) if inverted else stim.GateTarget(qubit)])
        for qubit, axis, inverted in measured
        if qubit not in probes
    ]
    order = (*probe_offsets, *(offset for offset, (qubit, _, _) in enumerate(measured) if qubit not in probes))
    emitted_of = {offset: emitted for emitted, offset in enumerate(order)}

    def candidate(feedback: Sequence[stim.CircuitInstruction]) -> stim.Circuit:
        result = stim.Circuit()
        for inst in (*products, *(inst for inst in body if inst.name not in _METADATA), *finals):
            targets = [
                stim.target_pauli(
                    local[t.value],
                    "_XYZ"[1 if t.is_x_target else 2 if t.is_y_target else 3],
                    invert=t.is_inverted_result_target,
                )
                if t.is_x_target or t.is_y_target or t.is_z_target
                else t
                if t.is_combiner
                else stim.target_inv(local[t.value])
                if t.is_inverted_result_target
                else stim.GateTarget(local[t.value])
                for t in inst.targets_copy()
            ]
            result.append(inst.name, targets, inst.gate_args_copy())
        for inst in feedback:
            targets = [
                t if t.is_measurement_record_target else stim.GateTarget(local[t.value]) for t in inst.targets_copy()
            ]
            result.append(inst.name, targets)
        return result + tail

    # Byproduct: record and sign terms by which the source flows differ from
    # those of the measurements alone, modulo the deterministic record
    # relations of the source window.
    by_input_and_output = _Gf2Span()
    for position, flow in enumerate(source_flows):
        key = _pauli_bits(flow.input_copy()) | _pauli_bits(flow.output_copy()) << out_shift
        by_input_and_output.add(key, 1 << position)
    relations = _Gf2Span()
    for combination in by_input_and_output.dependent:
        relation = _combine_flows(source_flows, combination)
        if relation is not None:
            relations.add(record_bits(relation) | _flow_is_negative(relation) << record_count, 0)
    system: dict[int, tuple[int, int]] = {}
    for flow in candidate(()).flow_generators():
        output_bits = _pauli_bits(flow.output_copy())
        combination = by_input_and_output.solve(_pauli_bits(flow.input_copy()) | output_bits << out_shift)
        if combination is None:
            return None, "window action is not Pauli products with Pauli feedback"
        matched = _combine_flows(source_flows, combination)
        difference = record_bits(flow, order) | _flow_is_negative(flow) << record_count
        if matched is not None:
            difference ^= record_bits(matched) | _flow_is_negative(matched) << record_count
        difference = relations.reduce(difference)
        # Swapping the X and Z bits turns the symplectic product into a dot product.
        row = (output_bits & int("01" * width, 2)) << 1 | (output_bits >> 1) & int("01" * width, 2)
        while row:
            pivot = row.bit_length() - 1
            if pivot not in system:
                system[pivot] = (row, difference)
                break
            row ^= system[pivot][0]
            difference ^= system[pivot][1]
        if not row and difference:
            return None, "window action is not Pauli products with Pauli feedback"
    for pivot in sorted(system):
        pivot_row, pivot_difference = system[pivot]
        for other in system:
            if other != pivot and system[other][0] >> pivot & 1:
                system[other] = (system[other][0] ^ pivot_row, system[other][1] ^ pivot_difference)
    byproducts = [0] * (record_count + 1)
    for pivot, (_, difference) in system.items():
        for record in range(record_count + 1):
            if difference >> record & 1:
                byproducts[record] |= 1 << pivot
    feedback: list[stim.CircuitInstruction] = []
    for offset, byproduct in enumerate(byproducts):
        if not byproduct:
            continue
        # The last entry is the record-independent part, left by inverted readouts.
        constant = offset == record_count
        if not constant and measured[offset][0] not in probes:
            return None, "Pauli byproduct controlled by a retained readout"
        for gates, shift in ((("X", "CX"), 0), (("Z", "CZ"), 1)):
            targets = [qubits[q] for q in range(width) if byproduct >> (2 * q + shift) & 1]
            if not probes.isdisjoint(targets):
                return None, "Pauli byproduct acts on a probe"
            if targets and constant:
                feedback.append(stim.CircuitInstruction(gates[0], targets))
            elif targets:
                control = stim.target_rec(emitted_of[offset] - record_count)
                feedback.append(
                    stim.CircuitInstruction(gates[1], [t for q in targets for t in (control, stim.GateTarget(q))])
                )

    rewritten = candidate(feedback)

    def relabeled(flows: Sequence[stim.Flow], record_of: Mapping[int, int] | Sequence[int]) -> list[stim.Flow]:
        return [
            stim.Flow(
                input=flow.input_copy(),
                output=flow.output_copy(),
                measurements=[record_of[record % record_count] for record in flow.measurements_copy()],
            )
            for flow in flows
        ]

    if not source.has_all_flows(relabeled(rewritten.flow_generators(), order)) or not rewritten.has_all_flows(
        relabeled(source_flows, emitted_of)
    ):
        return None, "flow comparison rejected the rewrite"
    return (
        _InteractingWindow(start, tuple(products), tuple(body), tuple(finals), tuple(feedback), order, probes),
        "",
    )
