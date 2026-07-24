"""Core Clifford transpilation and basis-gate optimization logic."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Literal

import stim

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

HS_STIM_GATE = "C_XNYZ"
HZ_STIM_GATE = "SQRT_Y"
HS_DAG_STIM_GATE = "C_XYZ"
# Maps each single-qubit basis gate, by its Stim spelling, to the angle of the
# one J(angle) = H Rz(angle) primitive implementing it. These are the four
# Clifford XY-plane measurements: X+ (H), Y+ (HS), X- (HZ), and Y- (HS_DAG).
STIM_GATE_J_ANGLES: dict[str, float] = {
    "H": 0.0,
    HS_STIM_GATE: math.pi / 2,
    HZ_STIM_GATE: math.pi,
    HS_DAG_STIM_GATE: -math.pi / 2,
}
_PauliAxis = Literal["X", "Y", "Z"]
_LOCAL_BASIS_GENERATORS = ("H", HS_STIM_GATE, HZ_STIM_GATE, HS_DAG_STIM_GATE)
_LOCAL_BASIS_GATES = frozenset(_LOCAL_BASIS_GENERATORS)
_BASIS_GATES = frozenset({"CZ", *_LOCAL_BASIS_GATES})
_PRESERVED_INSTRUCTIONS = frozenset(
    {
        "DETECTOR",
        "MPAD",
        "OBSERVABLE_INCLUDE",
        "QUBIT_COORDS",
        "SHIFT_COORDS",
        "TICK",
    }
)
_RESET_AXES: dict[str, _PauliAxis] = {"R": "Z", "RX": "X", "RY": "Y"}
_MEASUREMENT_AXES: dict[str, _PauliAxis] = {"M": "Z", "MX": "X", "MY": "Y"}
_RESETS = frozenset(_RESET_AXES)
_MEASUREMENTS = frozenset(_MEASUREMENT_AXES)
_BOUNDARY_OPERATIONS = _RESETS | _MEASUREMENTS
_OPTIMIZER_INSTRUCTIONS = _BASIS_GATES | _BOUNDARY_OPERATIONS
_PAULI_AXES: tuple[_PauliAxis, ...] = ("X", "Y", "Z")
_SINGLE_QUBIT_CLIFFORD_COUNT = 24
_SIGNED_PAULI_COUNT = 6


class UnsupportedInstructionError(ValueError):
    """Raised when an instruction cannot be represented by the supported basis."""


def transpile(
    circuit: stim.Circuit | str,
    *,
    optimize: bool = False,
    preserved_measurement_qubits: AbstractSet[int] = frozenset(),
) -> stim.Circuit:
    """Transpile a Stim Clifford circuit into the Clifford J/CZ gate basis.

    The single-qubit basis gates are the four Clifford ``J(angle)`` gates,
    i.e. the XY-plane Pauli measurements: ``H = J(0)`` (X+),
    ``HS = J(pi/2)`` (Y+, Stim's ``C_XNYZ``), ``HZ = J(pi)`` (X-, Stim's
    ``SQRT_Y``), and ``HS_DAG = J(-pi/2)`` (Y-, Stim's ``C_XYZ``).

    Parameters
    ----------
    circuit : ``stim.Circuit`` | `str`
        A Stim circuit or Stim circuit text. REPEAT blocks are handled
        recursively. TICK, QUBIT_COORDS, SHIFT_COORDS, DETECTOR,
        OBSERVABLE_INCLUDE, and MPAD instructions are preserved verbatim
        because they do not affect the quantum operation. Measurement-record
        targets are not interpreted by this standalone parser.
    optimize : `bool`, optional
        Remove redundant basis gates and simplify gates adjacent to
        R/RX/RY and M/MX/MY boundaries after transpilation.
    preserved_measurement_qubits : ``collections.abc.Set[int]``, optional
        Qubits whose post-measurement state must survive the optimized
        circuit. Measurements on these qubits are never folded into a
        rewritten Pauli basis, so single-qubit gates ahead of them stay
        explicit.

    Returns
    -------
    ``stim.Circuit``
        A new circuit containing only Clifford J gates, CZ, supported
        boundaries, and preserved instructions.
    """
    transpiled = _transpile_block(_coerce_circuit(circuit), context="circuit")
    if optimize:
        return optimize_j_cz(transpiled, preserved_measurement_qubits=preserved_measurement_qubits)
    return transpiled


def optimize_j_cz(
    circuit: stim.Circuit | str,
    *,
    preserved_measurement_qubits: AbstractSet[int] = frozenset(),
) -> stim.Circuit:
    """Remove redundant Clifford J and CZ gates.

    Every maximal run of single-qubit basis gates on one qubit is replaced by
    the shortest word over the four Clifford J gates (``H``, ``HS``, ``HZ``,
    ``HS_DAG``) with the same Clifford action; any single-qubit Clifford
    needs at most three J gates. CZ pairs cancel across commuting operations
    on other qubits. Single-qubit gates at R/RX/RY and M/MX/MY boundaries are
    folded into Pauli preparations or measurements when safe. TICK,
    coordinate annotations, DETECTOR, OBSERVABLE_INCLUDE, and MPAD are
    preserved verbatim and treated as optimization barriers. The repeated
    simplification passes are intended for TICK-bounded blocks; unusually
    large barrier-free inputs can take superlinear time.

    Parameters
    ----------
    circuit : ``stim.Circuit`` | `str`
        A circuit in the Clifford J/CZ basis.
    preserved_measurement_qubits : ``collections.abc.Set[int]``, optional
        Qubits whose post-measurement state must survive the optimized
        circuit. Measurements on these qubits are never folded into a
        rewritten Pauli basis.

    Returns
    -------
    ``stim.Circuit``
        The optimized circuit.
    """
    return _optimize_circuit(
        _coerce_circuit(circuit),
        context="circuit",
        terminal_measurements=True,
        preserved_measurement_qubits=preserved_measurement_qubits,
    )


def _coerce_circuit(circuit: stim.Circuit | str) -> stim.Circuit:
    if isinstance(circuit, str):
        return stim.Circuit(circuit)
    if not isinstance(circuit, stim.Circuit):
        msg = "circuit must be a stim.Circuit or Stim circuit text"
        raise TypeError(msg)
    return circuit


@dataclass(frozen=True)
class _AtomicGate:
    name: str
    targets: tuple[int, ...]
    tag: str
    gate_args: tuple[float, ...] = ()
    inverted: bool = False

    @property
    def qubits(self) -> frozenset[int]:
        return frozenset(self.targets)

    def append_to(self, circuit: stim.Circuit) -> None:
        targets: tuple[int | stim.GateTarget, ...]
        targets = (stim.target_inv(self.targets[0]),) if self.name in _MEASUREMENTS and self.inverted else self.targets
        circuit.append(
            self.name,
            targets,
            self.gate_args or None,
            tag=self.tag,
        )


def _optimize_circuit(
    circuit: stim.Circuit,
    *,
    context: str,
    terminal_measurements: bool,
    preserved_measurement_qubits: AbstractSet[int] = frozenset(),
) -> stim.Circuit:
    result = stim.Circuit()
    pending: list[_AtomicGate] = []

    def flush(*, allow_terminal_measurement_fold: bool) -> None:
        if not pending:
            return
        optimized = _simplify_boundaries(
            pending,
            allow_terminal_measurement_fold=allow_terminal_measurement_fold,
            preserved_measurement_qubits=preserved_measurement_qubits,
        )
        optimized = _cancel_redundant_gates(optimized)
        optimized = _simplify_boundaries(
            optimized,
            allow_terminal_measurement_fold=allow_terminal_measurement_fold,
            preserved_measurement_qubits=preserved_measurement_qubits,
        )
        for gate in optimized:
            gate.append_to(result)
        pending.clear()

    for instruction_index in range(len(circuit)):
        instruction = circuit[instruction_index]
        location = f"{context}, instruction {instruction_index}"
        if isinstance(instruction, stim.CircuitRepeatBlock):
            flush(allow_terminal_measurement_fold=False)
            body = _optimize_circuit(
                instruction.body_copy(),
                context=f"{location}, REPEAT body",
                terminal_measurements=False,
                preserved_measurement_qubits=preserved_measurement_qubits,
            )
            result.append(
                stim.CircuitRepeatBlock(
                    instruction.repeat_count,
                    body,
                    tag=instruction.tag,
                )
            )
            continue

        if instruction.name in _PRESERVED_INSTRUCTIONS:
            flush(allow_terminal_measurement_fold=False)
            result.append(instruction)
            continue
        if instruction.name not in _OPTIMIZER_INSTRUCTIONS:
            rendered = ", ".join(sorted(_OPTIMIZER_INSTRUCTIONS))
            msg = f"Instruction {instruction.name!r} at {location} is not in the Clifford J/CZ basis ({rendered})."
            raise UnsupportedInstructionError(msg)

        _validate_plain_qubit_targets(instruction, location=location)
        for group in instruction.target_groups():
            targets = tuple(_qubit_value(target) for target in group)
            if len(targets) != (2 if instruction.name == "CZ" else 1):
                msg = f"Instruction {instruction.name!r} at {location} has unsupported target grouping."
                raise UnsupportedInstructionError(msg)
            pending.append(
                _AtomicGate(
                    instruction.name,
                    targets,
                    instruction.tag,
                    tuple(instruction.gate_args_copy()),
                    instruction.name in _MEASUREMENTS and group[0].is_inverted_result_target,
                )
            )

    flush(allow_terminal_measurement_fold=terminal_measurements)
    return result


def _cancel_redundant_gates(gates: list[_AtomicGate]) -> list[_AtomicGate]:
    result = list(gates)
    while True:
        result = _normalize_single_qubit_runs(result)
        cancelled = _cancel_one_cz_pair(result)
        if cancelled is None:
            return result
        result = cancelled


def _single_qubit_runs(gates: list[_AtomicGate]) -> list[list[int]]:
    active_positions: dict[int, list[int]] = {}
    runs: list[list[int]] = []
    for index, gate in enumerate(gates):
        if gate.name in _LOCAL_BASIS_GATES:
            active_positions.setdefault(gate.targets[0], []).append(index)
            continue
        for qubit in gate.targets:
            positions = active_positions.pop(qubit, None)
            if positions is not None:
                runs.append(positions)
    runs.extend(active_positions.values())
    return runs


def _normalize_single_qubit_runs(gates: list[_AtomicGate]) -> list[_AtomicGate]:
    replacements: dict[int, tuple[_AtomicGate, ...]] = {}
    removed_positions: set[int] = set()
    for positions in _single_qubit_runs(gates):
        if len(positions) == 1:
            continue
        word = tuple(gates[index].name for index in positions)
        canonical = _single_qubit_normal_forms().unitary[_tableau_key(_single_qubit_tableau(word))]
        if canonical == word:
            continue
        head = gates[positions[0]]
        qubit = head.targets[0]
        replacement = tuple(_AtomicGate(name, (qubit,), head.tag) for name in canonical)
        replacements[positions[0]] = replacement
        removed_positions.update(positions)

    if not replacements:
        return gates
    result: list[_AtomicGate] = []
    for index, gate in enumerate(gates):
        if index in replacements:
            result.extend(replacements[index])
        if index not in removed_positions:
            result.append(gate)
    return result


def _cancel_one_cz_pair(gates: list[_AtomicGate]) -> list[_AtomicGate] | None:
    for start, gate in enumerate(gates):
        if gate.name != "CZ":
            continue
        for candidate_index in range(start + 1, len(gates)):
            candidate = gates[candidate_index]
            if candidate.name == "CZ" and candidate.qubits == gate.qubits:
                removed = {start, candidate_index}
                return [item for index, item in enumerate(gates) if index not in removed]
            if candidate.name != "CZ" and not gate.qubits.isdisjoint(candidate.qubits):
                break
    return None


def _simplify_boundaries(  # ruff:ignore[complex-structure, too-many-branches, too-many-statements]
    gates: list[_AtomicGate],
    *,
    allow_terminal_measurement_fold: bool,
    preserved_measurement_qubits: AbstractSet[int] = frozenset(),
) -> list[_AtomicGate]:
    result = list(gates)

    while True:
        changed = False

        # A reset discards prior single-qubit operations on the reset qubit.
        for reset_index, reset in enumerate(result):
            if reset.name not in _RESETS:
                continue
            positions = _local_word_before(result, reset_index, qubit=reset.targets[0])
            if positions:
                result = _replace_gate_positions(result, positions, ())
                changed = True
                break
        if changed:
            continue

        # Absorb a positive-axis state preparation into Stim's reset basis.
        for reset_index, reset in enumerate(result):
            if reset.name not in _RESETS:
                continue
            positions = _local_word_after(result, reset_index, qubit=reset.targets[0])
            if not positions:
                continue
            word = tuple(result[index].name for index in positions)
            state_key = _boundary_key(
                word,
                boundary="state",
                axis=_RESET_AXES[reset.name],
            )
            reset_name: str | None = {"+Z": "R", "+X": "RX", "+Y": "RY"}.get(state_key)
            if reset_name is None:
                continue
            replacement = _AtomicGate(
                reset_name,
                reset.targets,
                reset.tag,
                reset.gate_args,
            )
            result = _replace_boundary_and_word(
                result,
                boundary_index=reset_index,
                word_positions=positions,
                replacement=replacement,
            )
            changed = True
            break
        if changed:
            continue

        # Absorb a signed measurement basis into M/MX/MY and target inversion.
        for measurement_index, measurement in enumerate(result):
            if measurement.name not in _MEASUREMENTS:
                continue
            if not _measurement_state_is_discarded(
                result,
                measurement_index=measurement_index,
                allow_end=allow_terminal_measurement_fold,
                preserved_qubits=preserved_measurement_qubits,
            ):
                continue
            positions = _local_word_before(result, measurement_index, qubit=measurement.targets[0])
            if not positions:
                continue
            word = tuple(result[index].name for index in positions)
            measurement_key = _boundary_key(
                word,
                boundary="measurement",
                axis=_MEASUREMENT_AXES[measurement.name],
            )
            measurement_name: str = {
                "+Z": "M",
                "-Z": "M",
                "+X": "MX",
                "-X": "MX",
                "+Y": "MY",
                "-Y": "MY",
            }[measurement_key]
            replacement = _AtomicGate(
                measurement_name,
                measurement.targets,
                measurement.tag,
                measurement.gate_args,
                measurement.inverted ^ measurement_key.startswith("-"),
            )
            result = _replace_boundary_and_word(
                result,
                boundary_index=measurement_index,
                word_positions=positions,
                replacement=replacement,
            )
            changed = True
            break
        if changed:
            continue

        # After a reset, only the prepared stabilizer state matters.
        for reset_index, reset in enumerate(result):
            if reset.name not in _RESETS:
                continue
            positions = _local_word_after(result, reset_index, qubit=reset.targets[0])
            if positions:
                normalized = _normalize_boundary_positions(
                    result,
                    positions,
                    qubit=reset.targets[0],
                    boundary="state",
                    axis=_RESET_AXES[reset.name],
                )
                if normalized is not None:
                    result = normalized
                    changed = True
                    break
        if changed:
            continue

        # Before a Pauli measurement, only the signed observable matters.
        for measurement_index, measurement in enumerate(result):
            if measurement.name not in _MEASUREMENTS:
                continue
            positions = _local_word_before(result, measurement_index, qubit=measurement.targets[0])
            if positions:
                normalized = _normalize_boundary_positions(
                    result,
                    positions,
                    qubit=measurement.targets[0],
                    boundary="measurement",
                    axis=_MEASUREMENT_AXES[measurement.name],
                )
                if normalized is not None:
                    result = normalized
                    changed = True
                    break

        if not changed:
            return result


def _local_word_after(
    gates: list[_AtomicGate],
    boundary_index: int,
    *,
    qubit: int,
) -> list[int]:
    positions: list[int] = []
    for index in range(boundary_index + 1, len(gates)):
        candidate = gates[index]
        if qubit not in candidate.qubits:
            continue
        if candidate.name in _LOCAL_BASIS_GATES:
            positions.append(index)
            continue
        break
    return positions


def _local_word_before(
    gates: list[_AtomicGate],
    boundary_index: int,
    *,
    qubit: int,
) -> list[int]:
    positions: list[int] = []
    for index in range(boundary_index - 1, -1, -1):
        candidate = gates[index]
        if qubit not in candidate.qubits:
            continue
        if candidate.name in _LOCAL_BASIS_GATES:
            positions.append(index)
            continue
        break
    positions.reverse()
    return positions


def _normalize_boundary_positions(
    gates: list[_AtomicGate],
    positions: list[int],
    *,
    qubit: int,
    boundary: Literal["state", "measurement"],
    axis: _PauliAxis,
) -> list[_AtomicGate] | None:
    word = tuple(gates[index].name for index in positions)
    forms = _single_qubit_normal_forms()
    key = _boundary_key(word, boundary=boundary, axis=axis)
    replacement = (forms.state if boundary == "state" else forms.measurement)[axis, key]
    if replacement == word:
        return None
    tag = gates[positions[0]].tag
    replacement_gates = tuple(_AtomicGate(name, (qubit,), tag) for name in replacement)
    return _replace_gate_positions(gates, positions, replacement_gates)


def _replace_gate_positions(
    result: list[_AtomicGate],
    positions: list[int],
    replacement: tuple[_AtomicGate, ...],
) -> list[_AtomicGate]:
    position_set = set(positions)
    first = min(positions)
    output: list[_AtomicGate] = []
    for index, gate in enumerate(result):
        if index == first:
            output.extend(replacement)
        if index not in position_set:
            output.append(gate)
    return output


def _replace_boundary_and_word(
    gates: list[_AtomicGate],
    *,
    boundary_index: int,
    word_positions: list[int],
    replacement: _AtomicGate,
) -> list[_AtomicGate]:
    removed = set(word_positions)
    return [
        replacement if index == boundary_index else gate for index, gate in enumerate(gates) if index not in removed
    ]


def _measurement_state_is_discarded(
    gates: list[_AtomicGate],
    *,
    measurement_index: int,
    allow_end: bool,
    preserved_qubits: AbstractSet[int] = frozenset(),
) -> bool:
    qubit = gates[measurement_index].targets[0]
    if qubit in preserved_qubits:
        return False
    for candidate in gates[measurement_index + 1 :]:
        if qubit not in candidate.qubits:
            continue
        return candidate.name in _RESETS
    return allow_end


@dataclass(frozen=True)
class _NormalForms:
    unitary: dict[tuple[str, str], tuple[str, ...]]
    state: dict[tuple[str, str], tuple[str, ...]]
    measurement: dict[tuple[str, str], tuple[str, ...]]


@cache
def _single_qubit_normal_forms() -> _NormalForms:
    unitary_forms: dict[tuple[str, str], tuple[str, ...]] = {}
    state_forms: dict[tuple[str, str], tuple[str, ...]] = {}
    measurement_forms: dict[tuple[str, str], tuple[str, ...]] = {}
    queue: deque[tuple[str, ...]] = deque([()])

    # Breadth-first search over Clifford J words visits shorter words first,
    # so the first word reaching a tableau, prepared state, or measured
    # observable is a shortest representative.
    while queue:
        word = queue.popleft()
        tableau = _single_qubit_tableau(word)
        key = _tableau_key(tableau)
        if key in unitary_forms:
            continue
        unitary_forms[key] = word
        inverse = tableau.inverse()
        for axis in _PAULI_AXES:
            state_forms.setdefault((axis, str(_tableau_pauli_output(tableau, axis))), word)
            measurement_forms.setdefault((axis, str(_tableau_pauli_output(inverse, axis))), word)
        for generator in _LOCAL_BASIS_GENERATORS:
            queue.append((*word, generator))

    if (
        len(unitary_forms) != _SINGLE_QUBIT_CLIFFORD_COUNT
        or len(state_forms) != len(_PAULI_AXES) * _SIGNED_PAULI_COUNT
        or len(measurement_forms) != len(_PAULI_AXES) * _SIGNED_PAULI_COUNT
    ):
        msg = "Failed to enumerate single-qubit Clifford J normal forms"
        raise AssertionError(msg)
    return _NormalForms(unitary_forms, state_forms, measurement_forms)


def _tableau_key(tableau: stim.Tableau) -> tuple[str, str]:
    return str(tableau.x_output(0)), str(tableau.z_output(0))


def _boundary_key(
    word: tuple[str, ...],
    *,
    boundary: Literal["state", "measurement"],
    axis: _PauliAxis,
) -> str:
    tableau = _single_qubit_tableau(word)
    if boundary == "state":
        return str(_tableau_pauli_output(tableau, axis))
    return str(_tableau_pauli_output(tableau.inverse(), axis))


def _tableau_pauli_output(
    tableau: stim.Tableau,
    axis: _PauliAxis,
) -> stim.PauliString:
    return {
        "X": tableau.x_output(0),
        "Y": tableau.y_output(0),
        "Z": tableau.z_output(0),
    }[axis]


def _single_qubit_tableau(word: tuple[str, ...]) -> stim.Tableau:
    circuit = stim.Circuit()
    for name in word:
        circuit.append(name, [0])
    return stim.Tableau.from_circuit(circuit) if word else stim.Tableau(1)


def _transpile_block(circuit: stim.Circuit, *, context: str) -> stim.Circuit:
    result = stim.Circuit()

    for instruction_index in range(len(circuit)):
        instruction = circuit[instruction_index]
        location = f"{context}, instruction {instruction_index}"

        if isinstance(instruction, stim.CircuitRepeatBlock):
            body = _transpile_block(
                instruction.body_copy(),
                context=f"{location}, REPEAT body",
            )
            result.append(
                stim.CircuitRepeatBlock(
                    instruction.repeat_count,
                    body,
                    tag=instruction.tag,
                )
            )
            continue

        name = instruction.name
        if name in _PRESERVED_INSTRUCTIONS:
            result.append(instruction)
            continue
        if name in _BOUNDARY_OPERATIONS:
            _validate_plain_qubit_targets(instruction, location=location)
            result.append(instruction)
            continue

        try:
            gate_data = stim.gate_data(name)
        except IndexError as ex:
            msg = f"Unsupported instruction {name!r} at {location}."
            raise UnsupportedInstructionError(msg) from ex

        if not gate_data.is_unitary:
            rendered = ", ".join(sorted(_BASIS_GATES))
            msg = (
                f"Instruction {name!r} at {location} is not a unitary "
                f"Clifford gate and cannot be decomposed into the Clifford J/CZ basis ({rendered})."
            )
            raise UnsupportedInstructionError(msg)

        if name in _BASIS_GATES:
            _validate_plain_qubit_targets(instruction, location=location)
            result.append(instruction)
        elif name in {"SPP", "SPP_DAG"}:
            _append_spp_decomposition(
                result,
                instruction,
                location=location,
            )
        elif gate_data.is_single_qubit_gate or gate_data.is_two_qubit_gate:
            _append_fixed_gate_decomposition(
                result,
                instruction,
                gate_data=gate_data,
                location=location,
            )
        else:
            msg = f"Unitary instruction {name!r} at {location} has unsupported target semantics."
            raise UnsupportedInstructionError(msg)

    return result


def _validate_plain_qubit_targets(
    instruction: stim.CircuitInstruction,
    *,
    location: str,
) -> None:
    for target in instruction.targets_copy():
        if not target.is_qubit_target:
            msg = (
                f"Instruction {instruction.name!r} at {location} uses non-qubit target {target!s}; "
                "classical controls cannot be represented using the supported unitary gate basis."
            )
            raise UnsupportedInstructionError(msg)


def _qubit_value(target: stim.GateTarget) -> int:
    value = target.qubit_value
    if value is None:
        msg = f"Expected a qubit target, got {target!s}."
        raise AssertionError(msg)
    return int(value)


def _append_fixed_gate_decomposition(
    output: stim.Circuit,
    instruction: stim.CircuitInstruction,
    *,
    gate_data: stim.GateData,
    location: str,
) -> None:
    _validate_plain_qubit_targets(instruction, location=location)
    for group in instruction.target_groups():
        expected_size = 1 if gate_data.is_single_qubit_gate else 2
        if len(group) != expected_size:
            rendered_targets = " ".join(str(target) for target in group)
            msg = f"Instruction {instruction.name!r} at {location} uses unsupported targets {rendered_targets!r}."
            raise UnsupportedInstructionError(msg)

        qubits = tuple(_qubit_value(target) for target in group)
        _append_template(
            output,
            _fixed_gate_template(instruction.name),
            qubits=qubits,
            tag=instruction.tag,
        )


@cache
def _fixed_gate_template(name: str) -> stim.Circuit:
    tableau = stim.gate_data(name).tableau
    if tableau is None:
        msg = f"Stim did not provide tableau data for {name!r}"
        raise AssertionError(msg)
    return tableau.to_circuit(method="elimination")


def _append_spp_decomposition(
    output: stim.Circuit,
    instruction: stim.CircuitInstruction,
    *,
    location: str,
) -> None:
    for group in instruction.target_groups():
        if not group:
            msg = f"Instruction {instruction.name!r} at {location} has an empty Pauli product."
            raise UnsupportedInstructionError(msg)

        original_qubits = sorted({target.value for target in group})
        qubit_to_local = {qubit: index for index, qubit in enumerate(original_qubits)}
        local_targets: list[stim.GateTarget] = []

        for factor_index, target in enumerate(group):
            pauli = target.pauli_type
            if pauli not in {"X", "Y", "Z"}:
                msg = f"Instruction {instruction.name!r} at {location} has unsupported Pauli target {target!s}."
                raise UnsupportedInstructionError(msg)

            factory = {
                "X": stim.target_x,
                "Y": stim.target_y,
                "Z": stim.target_z,
            }[pauli]
            local_targets.append(
                factory(
                    qubit_to_local[target.value],
                    invert=target.is_inverted_result_target,
                )
            )
            if factor_index + 1 != len(group):
                local_targets.append(stim.target_combiner())

        local_circuit = stim.Circuit()
        local_circuit.append(instruction.name, local_targets)
        # Stim defers semantic validation of repeated-qubit SPP products (for
        # example, the anti-Hermitian X*Y) until tableau construction. Keep Stim
        # as the source of truth for ordered Pauli multiplication, then translate
        # its expected ValueError into this parser's location-aware error.
        try:
            template = stim.Tableau.from_circuit(local_circuit).to_circuit(method="elimination")
        except ValueError as ex:
            msg = f"Instruction {instruction.name!r} at {location} has an invalid Pauli product: {ex}"
            raise UnsupportedInstructionError(msg) from ex
        _append_template(
            output,
            template,
            qubits=tuple(original_qubits),
            tag=instruction.tag,
        )


def _append_template(
    output: stim.Circuit,
    template: stim.Circuit,
    *,
    qubits: tuple[int, ...],
    tag: str,
) -> None:
    for instruction_index in range(len(template)):
        instruction = template[instruction_index]
        if isinstance(instruction, stim.CircuitRepeatBlock):
            msg = "Stim unexpectedly synthesized a REPEAT block"
            raise TypeError(msg)

        targets = instruction.targets_copy()
        if instruction.name == "H":
            for target in targets:
                output.append(
                    "H",
                    [qubits[_qubit_value(target)]],
                    tag=tag,
                )
        elif instruction.name == "S":
            for target in targets:
                qubit = qubits[_qubit_value(target)]
                # Chronological HS (= H S in matrix order) followed by H
                # implements S, because H H S = S.
                output.append(HS_STIM_GATE, [qubit], tag=tag)
                output.append("H", [qubit], tag=tag)
        elif instruction.name == "CX":
            if len(targets) % 2:
                msg = "Stim synthesized CX with an odd target count"
                raise AssertionError(msg)
            for pair_start in range(0, len(targets), 2):
                control = qubits[_qubit_value(targets[pair_start])]
                target = qubits[_qubit_value(targets[pair_start + 1])]
                output.append("H", [target], tag=tag)
                output.append("CZ", [control, target], tag=tag)
                output.append("H", [target], tag=tag)
        else:
            msg = f"Stim's elimination synthesis unexpectedly emitted {instruction.name!r} instead of H, S, or CX"
            raise AssertionError(msg)
