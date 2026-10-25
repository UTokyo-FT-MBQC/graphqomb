r"""Pattern to Clifft compiler.

`Clifft <https://pypi.org/project/clifft/>`_ samples near-Clifford circuits
written in Stim syntax extended with non-Clifford gates such as ``T``. This
lets noisy patterns use non-stabilizer inputs, for example the magic state
:math:`|T\rangle = T|+\rangle`, and Clifford feedforward (cflow).

This module provides:

- `clifft_compile`: Function to compile a pattern into Clifft circuit text.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import typing_extensions

from graphqomb import clifford_algebra
from graphqomb.pauli_frame import CliffordFrame
from graphqomb.stim_glue.compiler import _StimCompiler

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from graphqomb.clifford_algebra import C1Element
    from graphqomb.noise_model import NoiseModel
    from graphqomb.pattern import Pattern

_SINGLE_QUBIT_GATES = frozenset(
    {
        "I",
        "X",
        "Y",
        "Z",
        "H",
        "H_XY",
        "H_YZ",
        "H_NXY",
        "H_NXZ",
        "H_NYZ",
        "S",
        "S_DAG",
        "SQRT_X",
        "SQRT_X_DAG",
        "SQRT_Y",
        "SQRT_Y_DAG",
        "C_XYZ",
        "C_ZYX",
        "C_NXYZ",
        "C_NZYX",
        "C_XNYZ",
        "C_XYNZ",
        "C_ZNYX",
        "C_ZYNX",
        "T",
        "T_DAG",
    }
)
_PARAMETRIZED_GATES = frozenset({"R_X", "R_Y", "R_Z", "U3"})
_GATE_PATTERN = re.compile(r"([A-Z][A-Z0-9_]*)(?:\(([^()]*)\))?")
_HEAD_PATTERN = re.compile(r"[A-Z][A-Z0-9_]*")
_QUBIT_TARGET_PATTERN = re.compile(r"!?[XYZ]?(\d+)")
_RECORD_INSTRUCTIONS = frozenset({"DETECTOR", "OBSERVABLE_INCLUDE", "REPEAT"})
# Stim gates for each transversal coset, in application order.
_COSET_GATES: dict[C1Element, tuple[str, ...]] = {
    clifford_algebra.IDENTITY: (),
    clifford_algebra.S: ("S",),
    clifford_algebra.H: ("H",),
    clifford_algebra.SH: ("H", "S"),
    clifford_algebra.HS: ("S", "H"),
    clifford_algebra.HSH: ("H", "S", "H"),
}


def _controlled_s(control: int, target: int) -> tuple[str, ...]:
    r"""Return controlled-S, whose phase exponent is a + t - (a XOR t) quarter turns.

    Returns
    -------
    `tuple`\[`str`, ...\]
        Clifft instructions in application order.
    """
    return (f"T {control} {target}", f"CX {control} {target}", f"T_DAG {target}", f"CX {control} {target}")


def _controlled_h(control: int, target: int) -> tuple[str, ...]:
    r"""Return controlled-H as V CZ V^dagger, with V = W T W^dagger.

    ``W = SQRT_X_DAG`` maps Z to Y, so V is a Y rotation by an eighth turn and
    V Z V^dagger equals H modulo phase.

    Returns
    -------
    `tuple`\[`str`, ...\]
        Clifft instructions in application order.
    """
    rotate = (f"SQRT_X {target}", f"T_DAG {target}", f"SQRT_X_DAG {target}")
    unrotate = (f"SQRT_X {target}", f"T {target}", f"SQRT_X_DAG {target}")
    return (*rotate, f"CZ {control} {target}", *unrotate)


def _controlled_clifford(element: C1Element, control: int, target: int) -> tuple[str, ...]:
    r"""Return Clifft instructions applying ``element`` to ``target`` when ``control`` is one.

    The control qubit must hold a computational basis state, so the
    control-dependent global phase of each gadget is unobservable.

    Parameters
    ----------
    element : `C1Element`
        The single-qubit Clifford to apply, modulo phase.
    control : `int`
        Control qubit index.
    target : `int`
        Target qubit index.

    Returns
    -------
    `tuple`\[`str`, ...\]
        Clifft instructions in application order.
    """
    coset, x_pauli, z_pauli = clifford_algebra.decompose(element)
    # element = coset * X^x * Z^z: Z acts first and the coset last.
    ops: list[str] = []
    if z_pauli:
        ops.append(f"CZ {control} {target}")
    if x_pauli:
        ops.append(f"CX {control} {target}")
    if coset == clifford_algebra.S:
        ops.extend(_controlled_s(control, target))
    elif coset == clifford_algebra.H:
        ops.extend(_controlled_h(control, target))
    elif coset == clifford_algebra.SH:
        ops.extend((*_controlled_h(control, target), *_controlled_s(control, target)))
    elif coset == clifford_algebra.HS:
        ops.extend((*_controlled_s(control, target), *_controlled_h(control, target)))
    elif coset == clifford_algebra.HSH:
        ops.extend((f"H {target}", *_controlled_s(control, target), f"H {target}"))
    return tuple(ops)


def _clifford(element: C1Element, target: int) -> tuple[str, ...]:
    r"""Return Stim instructions applying ``element`` to ``target`` modulo phase.

    Returns
    -------
    `tuple`\[`str`, ...\]
        Stim instructions in application order.
    """
    coset, x_pauli, z_pauli = clifford_algebra.decompose(element)
    # element = coset * X^x * Z^z: Z acts first and the coset last.
    gates = (*(("Z",) if z_pauli else ()), *(("X",) if x_pauli else ()), *_COSET_GATES[coset])
    return tuple(f"{gate} {target}" for gate in gates)


def _normalize_fixed_controls(pattern: Pattern, fixed_controls: Mapping[int, bool] | None) -> dict[int, bool]:
    r"""Validate the fixed values of Clifford feedforward sources.

    Returns
    -------
    `dict`\[`int`, `bool`\]
        The fixed value of each source.

    Raises
    ------
    ValueError
        If a key is not a source of Clifford feedforward.
    """
    sources = pattern.frame.cflow if isinstance(pattern.frame, CliffordFrame) else {}
    for node in fixed_controls or {}:
        if node not in sources:
            msg = f"Fixed control {node} is not a source of Clifford feedforward (cflow) in the pattern."
            raise ValueError(msg)
    return {node: bool(value) for node, value in (fixed_controls or {}).items()}


def _normalize_input_gates(
    pattern: Pattern, input_gates: Mapping[int, str | Sequence[str]] | None
) -> dict[int, tuple[str, ...]]:
    r"""Validate the gates applied after input resets.

    Returns
    -------
    `dict`\[`int`, `tuple`\[`str`, ...\]\]
        Gate instructions for each input node in application order.

    Raises
    ------
    ValueError
        If a key is not an input node or a gate is not a single-qubit unitary.
    """
    normalized: dict[int, tuple[str, ...]] = {}
    for node, gates in (input_gates or {}).items():
        if node not in pattern.input_node_indices:
            msg = f"Input gates target node {node}, which is not an input node of the pattern."
            raise ValueError(msg)
        sequence = (gates,) if isinstance(gates, str) else tuple(gates)
        for gate in sequence:
            match = _GATE_PATTERN.fullmatch(gate.strip()) if isinstance(gate, str) else None
            if match is None or not (
                (match[1] in _SINGLE_QUBIT_GATES and match[2] is None)
                or (match[1] in _PARAMETRIZED_GATES and match[2] is not None)
            ):
                msg = f"Unsupported input gate for node {node}: {gate!r}. Use a single-qubit unitary such as 'T'."
                raise ValueError(msg)
        normalized[node] = tuple(gate.strip() for gate in sequence)
    return normalized


def _input_circuit_qubits(input_circuit: str) -> set[int]:
    r"""Validate the input preparation circuit and collect its qubit targets.

    Returns
    -------
    `set`\[`int`\]
        Qubit indices targeted by the circuit.

    Raises
    ------
    ValueError
        If the circuit produces or references measurement records.
    """
    qubits: set[int] = set()
    for raw_line in input_circuit.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        head = _HEAD_PATTERN.match(line)
        name = head[0] if head else line
        if (
            name.startswith(("M", "HERALD"))
            or name in _RECORD_INSTRUCTIONS
            or "rec[" in line
            or any(brace in line for brace in "{}")
        ):
            msg = f"The input circuit must not produce or reference measurement records: {raw_line.strip()!r}."
            raise ValueError(msg)
        for token in line.split()[1:]:
            if target := _QUBIT_TARGET_PATTERN.fullmatch(token):
                qubits.add(int(target[1]))
    return qubits


class _ClifftCompiler(_StimCompiler):
    # Clifft rejects instruction tags outside LEVEL_TRANSITION.
    _emit_tags = False

    def __init__(  # ruff:ignore[too-many-arguments]
        self,
        pattern: Pattern,
        *,
        input_gates: dict[int, tuple[str, ...]],
        input_circuit: str,
        fixed_controls: dict[int, bool],
        emit_qubit_coords: bool,
        noise_models: Sequence[NoiseModel],
        tick_duration: float,
    ) -> None:
        super().__init__(
            pattern,
            backpropagate=False,
            emit_qubit_coords=emit_qubit_coords,
            noise_models=noise_models,
            tick_duration=tick_duration,
        )
        self._input_gates = input_gates
        self._input_circuit = input_circuit.strip()
        self._fixed_controls = fixed_controls
        qubits = set(pattern.frame.graphstate.nodes) | _input_circuit_qubits(input_circuit)
        self._ancilla = max(qubits, default=-1) + 1

    @typing_extensions.override
    def _emit_input_state(self, node: int) -> None:
        for gate in self._input_gates.get(node, ()):
            self._stim_io.write(f"{gate} {node}\n")

    @typing_extensions.override
    def _emit_input_nodes(self) -> None:
        super()._emit_input_nodes()
        if self._input_circuit:
            self._stim_io.write(f"{self._input_circuit}\n")

    @typing_extensions.override
    def _emit_pauli_corrections(self, node: int) -> None:
        """Execute retained Pauli and Clifford feedback in the frame's event order."""
        frame = self._pframe
        if not isinstance(frame, CliffordFrame):
            super()._emit_pauli_corrections(node)
            return
        sources = (
            frame.inv_xflow.get(node, set()) | frame.inv_zflow.get(node, set()) | frame.inv_cflow.get(node, {}).keys()
        )
        # The frame composes each source's events when its outcome is recorded.
        for source in sorted(sources, key=self._meas_order.__getitem__):
            record = f"rec[{self._meas_order[source] - self._rec_index}]"
            fixed = self._fixed_controls.get(source)
            for target, correction in frame.correction_events.get(source, ()):
                if target != node or fixed is False:
                    continue
                if fixed:
                    for op in _clifford(correction, node):
                        self._stim_io.write(f"{op}\n")
                elif correction == clifford_algebra.X:
                    self._stim_io.write(f"CX {record} {node}\n")
                elif correction == clifford_algebra.Z:
                    self._stim_io.write(f"CZ {record} {node}\n")
                else:
                    # Clifft feedback controls only CX/CZ: copy the record to a
                    # basis-state ancilla and apply the controlled-Clifford gadget.
                    self._stim_io.write(f"R {self._ancilla}\nCX {record} {self._ancilla}\n")
                    for op in _controlled_clifford(correction, self._ancilla, node):
                        self._stim_io.write(f"{op}\n")


def clifft_compile(  # ruff:ignore[too-many-arguments]
    pattern: Pattern,
    *,
    input_gates: Mapping[int, str | Sequence[str]] | None = None,
    input_circuit: str = "",
    fixed_controls: Mapping[int, bool] | None = None,
    emit_qubit_coords: bool = True,
    noise_models: Sequence[NoiseModel] | None = None,
    tick_duration: float = 1.0,
) -> str:
    r"""Compile a pattern to Clifft circuit text.

    The output is Stim syntax extended with Clifft's non-Clifford gates; pass
    it to ``clifft.compile``. Install the optional dependency with
    ``pip install graphqomb[clifft]`` (Clifft requires Python 3.12 or later).

    Parameters
    ----------
    pattern : `Pattern`
        The pattern to compile. Measurements must be Pauli measurements.
    input_gates : `collections.abc.Mapping`\[`int`, `str` | `collections.abc.Sequence`\[`str`\]\] | `None`, optional
        Single-qubit unitaries applied to input nodes right after their Pauli
        reset, in order. For example, ``{node: "T"}`` with the default ``X+``
        initialization prepares :math:`|T\rangle = T|+\rangle`. Fixed gates
        such as ``"T"``, ``"T_DAG"``, ``"H"`` or ``"S"`` and parametrized
        ``R_X``, ``R_Y``, ``R_Z`` or ``U3`` (``"R_Z(0.1)"``) are accepted.
        By default None.
    input_circuit : `str`, optional
        Clifft instructions inserted after all input preparations and before
        the first pattern command, for example an ideal encoder of logical
        input states. It may act on input nodes but must not produce or
        reference measurement records. By default empty.
    fixed_controls : `collections.abc.Mapping`\[`int`, `bool`\] | `None`, optional
        Fixed classical values of Clifford feedforward (cflow) sources. Every
        correction event of a listed source, Pauli or Clifford, is applied
        unconditionally when its value is True and omitted when False,
        instead of being controlled by the source's record. The source is
        still measured, and detectors and observables keep its record. See
        Notes for sampling with online decoding. By default None.
    emit_qubit_coords : `bool`, optional
        Whether to emit QUBIT_COORDS instructions for nodes with coordinates,
        by default True.
    noise_models : `collections.abc.Sequence`\[`NoiseModel`\] | `None`, optional
        Custom noise models, invoked for the same events as `stim_compile`.
        By default None.
    tick_duration : `float`, optional
        Duration associated with each TICK for idle noise, by default 1.0.

    Returns
    -------
    `str`
        The compiled Clifft circuit text.

    Notes
    -----
    Detector and observable supports are serialized as stored, as in
    ``stim_compile(pattern, backpropagate=False)``: call
    ``Feedforward.backpropagate`` before ``qompile`` to compile removable
    Pauli corrections into them. Every retained correction is executed right
    before the corrected node's measurement, or at the end for unmeasured
    outputs. Corrections onto one node follow the measurement order of their
    sources, and each source applies X, then Z, then its Clifford gate, as in
    `graphqomb.pauli_frame.CliffordFrame` and the pattern simulator.

    Clifft feedback controls only CX and CZ. A conditional Clifford therefore
    copies its control record to an ancilla qubit, numbered after all pattern
    and input-circuit qubits, and applies a controlled gadget built from
    Clifford gates and T: three T/T_DAG gates for controlled-S and two for
    controlled-H. Input gates, the input circuit, and correction gadgets are
    noiseless; noise models are not invoked for them.

    Fault-tolerant execution decides a conditional Clifford from a value
    decoded online from the records measured so far, not from the raw record
    of its source: a wrong Clifford cannot be repaired by offline Pauli
    decoding. To sample this, compile one circuit for each value ``c`` of the
    source with ``fixed_controls={source: c}``, sample the same number of
    shots from each, decode the value from the records preceding the
    correction, and keep the shots whose decoded value equals ``c``. The
    circuits agree before the correction, so a branch keeps the fraction of
    shots that the decoder sends into it, and the kept shots of all branches
    together follow the online-decoded execution. With fixed values, a
    pattern whose only non-Clifford parts are input gates or the input
    circuit compiles to Clifford instructions only.

    Examples
    --------
    Sample T gate teleportation with depolarizing noise:

    >>> # import clifft
    >>> # text = clifft_compile(pattern, input_gates={magic: "T"}, noise_models=[DepolarizingNoiseModel(1e-3)])
    >>> # result = clifft.sample(clifft.compile(text), shots=10_000)
    """
    compiler = _ClifftCompiler(
        pattern,
        input_gates=_normalize_input_gates(pattern, input_gates),
        input_circuit=input_circuit,
        fixed_controls=_normalize_fixed_controls(pattern, fixed_controls),
        emit_qubit_coords=emit_qubit_coords,
        noise_models=tuple(() if noise_models is None else noise_models),
        tick_duration=tick_duration,
    )
    return compiler.compile()
