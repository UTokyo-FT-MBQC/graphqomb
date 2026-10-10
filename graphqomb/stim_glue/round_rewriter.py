"""Local extraction of closed Pauli probes, preserving successive instruments.

This pass never uses a known data state or a detector relation to delete a
measurement. Its algebraic domain is a layer of independent controlled-Pauli
probes. Other blocks are copied, including their residual quantum action.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import stim

_RESETS = {"R": "Z", "RX": "X", "RY": "Y"}
_MEASUREMENTS = {"M": "Z", "MX": "X", "MY": "Y"}
_METADATA = {"QUBIT_COORDS", "SHIFT_COORDS", "TICK", "DETECTOR", "OBSERVABLE_INCLUDE"}


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


def rewrite_syndrome_rounds(  # ruff: ignore[complex-structure, too-many-branches, too-many-locals, too-many-statements]
    circuit: stim.Circuit,
) -> RoundRewriteResult:
    """Replace closed extraction layers with successive data MPPs.

    Each probe must have an explicit reset, a final single-qubit measurement,
    and no subsequent quantum use before another reset (or circuit end).
    Within a candidate layer, its interactions must share one control axis,
    its prepared/readout axes must match and be transverse to that control,
    and the collected data products must form a commuting Hermitian layer
    without a residual inter-probe phase. Data Clifford gates and preparation
    operations are retained. Failed factorizations leave the entire block
    unchanged. No runtime channel comparison or flow-generator check is used.

    The resulting instrument on all remaining wires and the complete record
    vector are preserved for arbitrary block input. Only closed probe outputs
    are discarded. This is a structural compilation policy, not inference of
    the intended logical subsystem. Existing MPPs are barriers, not optimized.

    Parameters
    ----------
    circuit : stim.Circuit
        Ideal Clifford circuit. Explicit noise must be removed by the caller.

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
    # Whether each measured coordinate's next quantum use is a reset. Metadata
    # (including Pauli observable annotations) does not constitute quantum use.
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
        if name in _METADATA and not (
            name == "OBSERVABLE_INCLUDE"
            and any(t.is_x_target or t.is_y_target or t.is_z_target for t in inst.targets_copy())
        ):
            continue
        qubits = [
            t.value for t in inst.targets_copy() if t.is_qubit_target or t.is_x_target or t.is_y_target or t.is_z_target
        ]
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

    for index, inst in enumerate(instructions):
        count = inst.num_measurements
        if inst.name in _MEASUREMENTS:
            rewritten, removed, reason = _extract_layer(pending, inst, source_count, closed[index], width)
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
                probes += removed
            pending.clear()
        elif count or inst.name in {"MR", "MRX", "MRY"}:
            for old, before in pending:
                emit(old, before)
            pending.clear()
            emit(inst, source_count)
        else:
            pending.append((inst, source_count))
        source_count += count
    for inst, before in pending:
        emit(inst, before)
    return RoundRewriteResult(output, tuple(record_map[i] for i in range(source_count)), rounds, probes, tuple(reasons))


def _extract_layer(  # ruff: ignore[complex-structure, too-many-branches, too-many-statements, too-many-locals, too-many-return-statements]
    pending: list[tuple[stim.CircuitInstruction, int]],
    readout: stim.CircuitInstruction,
    source_count: int,
    closed: set[int],
    width: int,
) -> tuple[list[tuple[stim.CircuitInstruction, int, list[int] | None]] | None, int, str]:
    resets = {t.value for inst, _ in pending if inst.name in _RESETS for t in inst.targets_copy()}
    measured = [t.value for t in readout.targets_copy()]
    candidates = resets & set(measured) & closed
    if not candidates:
        return None, 0, ""
    if len(set(measured)) != len(measured):
        return None, 0, "repeated readout target"
    # U = L_A (product_a controlled(P_a)) V_D. Reordering different
    # controls produces CZ phases on A; retain these symbolically until the
    # complete layer has been collected. No state of D enters this identity.
    local = {a: stim.Tableau(1) for a in candidates}
    prepared: dict[int, str] = {}
    pointer: dict[int, stim.PauliString] = {}
    products = {a: stim.PauliString(width) for a in candidates}
    phases: set[tuple[int, int]] = set()
    result: list[tuple[stim.CircuitInstruction, int, list[int] | None]] = []
    touched: set[int] = set()
    for inst, before in pending:
        name = inst.name
        targets = inst.targets_copy()
        if name in _METADATA:
            if name == "OBSERVABLE_INCLUDE" and any(not t.is_measurement_record_target for t in targets):
                return None, 0, "Pauli-target observable boundary"
            result.append((inst, before, None))
            continue
        if any(not t.is_qubit_target for t in targets):
            return None, 0, "classical control or Pauli-product gate inside layer"
        if name in _RESETS:
            keep = []
            for target in targets:
                q = target.value
                if q in touched:
                    return None, 0, "reset after a quantum operation inside layer"
                if q in candidates:
                    if q in prepared:
                        return None, 0, "multiple probe resets inside layer"
                    prepared[q] = _RESETS[name]
                else:
                    keep.append(target)
            if keep:
                result.append((stim.CircuitInstruction(name, keep, tag=inst.tag), before, None))
            continue
        data = stim.gate_data(name)
        if not data.is_unitary or data.takes_pauli_targets:
            return None, 0, "operation outside controlled-Pauli layer"
        arity = 1 if data.is_single_qubit_gate else 2
        for start in range(0, len(targets), arity):
            group = targets[start : start + arity]
            qubits = [t.value for t in group]
            touched.update(qubits)
            anc = candidates.intersection(qubits)
            if not anc:
                for a, p in products.items():
                    if any(p[q] for q in qubits):
                        products[a] = p.after(_gate_tableau(name), targets=qubits)
                result.append((stim.CircuitInstruction(name, group, tag=inst.tag), before, None))
            elif len(anc) > 1:
                return None, 0, "interaction between probe candidates"
            else:
                a = next(iter(anc))
                if a not in prepared:
                    return None, 0, "probe used before reset"
                if arity == 1:
                    local[a] = local[a].then(_gate_tableau(name))
                    continue
                if name not in {"CX", "CY", "CZ"}:
                    return None, 0, "unsupported probe interaction"
                control_axis, data_axis = ("Z", name[-1]) if qubits[0] == a else (name[-1], "Z")
                pulled = local[a].inverse()(stim.PauliString(control_axis))
                if a in pointer and pointer[a] != pulled:
                    return None, 0, "probe control axis changes during extraction"
                pointer[a] = pulled
                if pulled.commutes(stim.PauliString(prepared[a])):
                    return None, 0, "probe preparation not transverse to control"
                d = qubits[1] if qubits[0] == a else qubits[0]
                factor = stim.PauliString(width)
                factor[d] = data_axis
                for b, p in products.items():
                    if b < a and p[d] and p[d] != factor[d]:
                        phases.symmetric_difference_update({(b, a)})
                products[a] = factor * products[a]
    if phases:
        return None, 0, "residual phase between probes"
    ordered = sorted(candidates)
    for index, a in enumerate(ordered):
        if a not in pointer or products[a].weight == 0 or products[a].sign not in {1, -1}:
            return None, 0, "probe has trivial or non-Hermitian data product"
        if any(not products[a].commutes(products[b]) for b in ordered[:index]):
            return None, 0, "noncommuting final data products"
        pulled = local[a].inverse()(stim.PauliString(_MEASUREMENTS[readout.name]))
        positive = stim.PauliString(prepared[a])
        if pulled[0] != positive[0]:
            return None, 0, "residual probe phase requires a quantum correction"
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
    return result, len(candidates), ""
