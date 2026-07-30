"""Pattern text format (.ptn) module.

This module provides:

- `dump`: Write a pattern to a .ptn file or string.
- `load`: Read a pattern from a .ptn file or string.
- `dumps`: Serialize a pattern to a .ptn format string.
- `loads`: Deserialize a pattern from a .ptn format string.
"""

from __future__ import annotations

import math
import operator
import re
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import typing_extensions

from graphqomb._tag import escape_tag, unescape_tag
from graphqomb.command import TICK, Command, E, M, N
from graphqomb.common import (
    Axis,
    AxisMeasBasis,
    MeasBasis,
    Plane,
    PlannerMeasBasis,
    Sign,
    determine_pauli_axis,
    is_clifford_angle,
    is_close_angle,
)
from graphqomb.graphstate import BaseGraphState
from graphqomb.pattern import Pattern
from graphqomb.pauli_frame import PauliFrame

if TYPE_CHECKING:
    from collections.abc import Sequence

PTN_VERSION = 2
SUPPORTED_PTN_VERSIONS = frozenset({1, PTN_VERSION})

# A timeslice marker may skip ahead of the previous one, and every skipped slice becomes a TICK
# command.  The index is attacker-controlled in an untrusted .ptn file, so the expansion is bounded
# to keep memory use proportional to the input: a parse may materialize at most one TICK per input
# line, plus a fixed slack that covers hand-written files that legitimately start at, or jump to, a
# small non-zero timeslice.
_IMPLICIT_TICK_SLACK = 4096

# Angle formatting/parsing lookup tables
_ANGLE_TO_STR: dict[float, str] = {
    0.0: "0",
    math.pi: "pi",
    -math.pi: "-pi",
    math.pi / 2: "pi/2",
    -math.pi / 2: "-pi/2",
    math.pi / 4: "pi/4",
    -math.pi / 4: "-pi/4",
    3 * math.pi / 2: "3pi/2",
    3 * math.pi / 4: "3pi/4",
}

_STR_TO_ANGLE: dict[str, float] = {
    "0": 0.0,
    "pi": math.pi,
    "-pi": -math.pi,
    "pi/2": math.pi / 2,
    "-pi/2": -math.pi / 2,
    "pi/4": math.pi / 4,
    "-pi/4": -math.pi / 4,
    "3pi/2": 3 * math.pi / 2,
    "3pi/4": 3 * math.pi / 4,
}

_PI_PATTERN = re.compile(r"^(-?\d*)pi(?:/(\d+))?$")


def _format_angle(angle: float) -> str:
    r"""Format angle for output, using pi fractions where appropriate.

    Parameters
    ----------
    angle : `float`
        The angle in radians.

    Returns
    -------
    `str`
        Formatted angle string.
    """
    candidates = (
        ((ref_angle, label) for ref_angle, label in _ANGLE_TO_STR.items() if is_clifford_angle(ref_angle))
        if is_clifford_angle(angle)
        else _ANGLE_TO_STR.items()
    )
    ordered_candidates = sorted(candidates, key=lambda item: item[0] < 0 if angle >= 0 else item[0] >= 0)
    for ref_angle, label in ordered_candidates:
        if is_close_angle(angle, ref_angle):
            return label
    return f"{angle}"


def _parse_angle(s: str) -> float:
    r"""Parse angle string to float.

    Parameters
    ----------
    s : `str`
        Angle string (e.g., "0", "pi", "pi/2", "3pi/4", "1.5707963").

    Returns
    -------
    `float`
        The angle in radians.

    Raises
    ------
    ValueError
        If the angle is not a valid number or pi expression.
    """
    s = s.strip()
    if s in _STR_TO_ANGLE:
        return _STR_TO_ANGLE[s]

    pi_match = _PI_PATTERN.match(s)
    if pi_match:
        numerator = pi_match.group(1)
        denominator = pi_match.group(2)
        num = int(numerator) if numerator and numerator != "-" else (1 if numerator != "-" else -1)
        denom = int(denominator) if denominator else 1
        if denom == 0:
            msg = "Angle denominator cannot be zero"
            raise ValueError(msg)
        return num * math.pi / denom

    return float(s)


def _format_coord(coord: tuple[float, ...]) -> str:
    r"""Format coordinate tuple for output.

    Parameters
    ----------
    coord : `tuple`\[`float`, ...\]
        Coordinate tuple (2D or 3D).

    Returns
    -------
    `str`
        Space-separated coordinate string.
    """
    return " ".join(str(c) for c in coord)


def _parse_coord(parts: Sequence[str]) -> tuple[float, ...]:
    r"""Parse coordinate from string parts.

    Parameters
    ----------
    parts : `list`\[`str`\]
        List of coordinate value strings.

    Returns
    -------
    `tuple`\[`float`, ...\]
        Coordinate tuple.
    """
    return tuple(float(p) for p in parts)


# ============================================================
# Serialization (dumps/dump)
# ============================================================


def _write_header(out: StringIO, pattern: Pattern) -> None:
    """Write header section to output."""
    out.write(f"# GraphQOMB Pattern Format v{PTN_VERSION}\n")
    out.write("\n")
    out.write("#======== HEADER ========\n")
    out.write(f".version {PTN_VERSION}\n")

    if pattern.input_node_indices:
        input_parts = [
            f"{node}:{qidx}" for node, qidx in sorted(pattern.input_node_indices.items(), key=operator.itemgetter(1))
        ]
        out.write(f".input {' '.join(input_parts)}\n")
        input_basis_parts = [
            f"{node}:{axis.name}"
            for node, _qidx in sorted(pattern.input_node_indices.items(), key=operator.itemgetter(1))
            if (axis := pattern.input_initialization_axes.get(node, Axis.X)) is not Axis.X
        ]
        if input_basis_parts:
            out.write(f".input_basis {' '.join(input_basis_parts)}\n")

    if pattern.output_node_indices:
        output_parts = [
            f"{node}:{qidx}" for node, qidx in sorted(pattern.output_node_indices.items(), key=operator.itemgetter(1))
        ]
        out.write(f".output {' '.join(output_parts)}\n")

    out.writelines(
        f".coord {node} {_format_coord(coord)}\n" for node, coord in sorted(pattern.input_coordinates.items())
    )


def _write_command(out: StringIO, cmd: Command) -> None:
    """Write a single command to output."""
    if isinstance(cmd, N):
        if cmd.coordinate is not None:
            out.write(f"N {cmd.node} {_format_coord(cmd.coordinate)}\n")
        else:
            out.write(f"N {cmd.node}\n")
    elif isinstance(cmd, E):
        out.write(f"E {cmd.nodes[0]} {cmd.nodes[1]}\n")
    elif isinstance(cmd, M):
        _write_measurement(out, cmd)


def _is_positive_pauli_measurement(meas_basis: MeasBasis, pauli_axis: Axis) -> bool:
    """Return whether a Pauli measurement is on the positive eigenbasis.

    Returns
    -------
    bool
        True if the measurement basis is the positive Pauli eigenbasis.
    """
    angle = meas_basis.angle
    plane = meas_basis.plane
    if pauli_axis == Axis.X:
        positive_angle = math.pi / 2 if plane == Plane.XZ else 0.0
    elif pauli_axis == Axis.Y:
        positive_angle = math.pi / 2
    else:
        positive_angle = 0.0
    return is_close_angle(angle, positive_angle)


def _write_measurement(out: StringIO, cmd: M) -> None:
    """Write measurement command with appropriate format."""
    pauli_axis = determine_pauli_axis(cmd.meas_basis)
    if pauli_axis is not None:
        sign = "+" if _is_positive_pauli_measurement(cmd.meas_basis, pauli_axis) else "-"
        out.write(f"M {cmd.node} {pauli_axis.name} {sign}\n")
    else:
        plane_name = cmd.meas_basis.plane.name
        angle_str = _format_angle(cmd.meas_basis.angle)
        out.write(f"M {cmd.node} {plane_name} {angle_str}\n")


def _write_quantum_section(out: StringIO, pattern: Pattern) -> None:
    """Write quantum instructions section to output."""
    out.write("\n")
    out.write("#======== QUANTUM ========\n")

    timeslice = 0
    current_slice_commands: list[Command] = []

    def write_slice(slice_num: int, commands: list[Command]) -> None:
        out.write(f"[{slice_num}]\n")
        for cmd in commands:
            _write_command(out, cmd)

    for cmd in pattern.commands:
        if isinstance(cmd, TICK):
            write_slice(timeslice, current_slice_commands)
            current_slice_commands = []
            timeslice += 1
        else:
            current_slice_commands.append(cmd)

    if current_slice_commands or timeslice == 0 or (pattern.commands and isinstance(pattern.commands[-1], TICK)):
        write_slice(timeslice, current_slice_commands)


def _write_classical_section(out: StringIO, pauli_frame: PauliFrame) -> None:
    """Write classical frame section to output."""
    out.write("\n")
    out.write("#======== CLASSICAL ========\n")

    for source, targets in sorted(pauli_frame.xflow.items()):
        if targets:
            targets_str = " ".join(str(t) for t in sorted(targets))
            out.write(f".xflow {source} -> {targets_str}\n")

    for source, targets in sorted(pauli_frame.zflow.items()):
        if targets:
            targets_str = " ".join(str(t) for t in sorted(targets))
            out.write(f".zflow {source} -> {targets_str}\n")

    for group, tag in zip(pauli_frame.parity_check_group, pauli_frame.parity_check_tags, strict=True):
        if group:
            group_str = " ".join(str(n) for n in sorted(group))
            tag_suffix = f"[{escape_tag(tag)}]" if tag else ""
            out.write(f".detector{tag_suffix} {group_str}\n")

    for logical_idx, nodes in sorted(pauli_frame.logical_observables.items()):
        if nodes:
            nodes_str = " ".join(str(n) for n in sorted(nodes))
            out.write(f".observable {logical_idx} -> {nodes_str}\n")


def dumps(pattern: Pattern) -> str:
    """Serialize a pattern to a .ptn format string.

    Parameters
    ----------
    pattern : `Pattern`
        The pattern to serialize.

    Returns
    -------
    `str`
        The .ptn format string.
    """
    out = StringIO()
    _write_header(out, pattern)
    _write_quantum_section(out, pattern)
    _write_classical_section(out, pattern.pauli_frame)
    return out.getvalue()


def dump(pattern: Pattern, file: Path | str) -> None:
    """Write a pattern to a .ptn file.

    Parameters
    ----------
    pattern : `Pattern`
        The pattern to write.
    file : `pathlib.Path` | `str`
        The file path to write to.
    """
    path = Path(file)
    path.write_text(dumps(pattern), encoding="utf-8")


# ============================================================
# Deserialization (loads/load)
# ============================================================


def _parse_int(value: str, label: str) -> int:
    """Parse an integer field.

    Returns
    -------
    `int`
        Parsed integer.

    Raises
    ------
    ValueError
        If the field is not an integer.
    """
    try:
        return int(value)
    except ValueError as exc:
        msg = f"Invalid {label}: {value!r}"
        raise ValueError(msg) from exc


def _parse_node_qubit_pairs(parts: Sequence[str]) -> dict[int, int]:
    r"""Parse node:qubit pairs from string parts.

    Parameters
    ----------
    parts : `list`\[`str`\]
        List of "node:qubit" strings.

    Returns
    -------
    `dict`\[`int`, `int`\]
        Mapping from node to qubit index.

    Raises
    ------
    ValueError
        If any pair is malformed or duplicated.
    """
    result: dict[int, int] = {}
    for part in parts:
        pair = part.split(":")
        if len(pair) != 2:  # ruff:ignore[magic-value-comparison]
            msg = f"Invalid node:qubit pair: {part!r}"
            raise ValueError(msg)
        node_str, qidx_str = pair
        node = _parse_int(node_str, "node")
        qidx = _parse_int(qidx_str, "qubit index")
        if node in result:
            msg = f"Duplicate node mapping: {node}"
            raise ValueError(msg)
        if qidx in result.values():
            msg = f"Duplicate qubit index: {qidx}"
            raise ValueError(msg)
        result[node] = qidx
    return result


def _parse_node_axis_pairs(parts: Sequence[str]) -> dict[int, Axis]:
    r"""Parse node:axis pairs from string parts.

    Returns
    -------
    `dict`\[`int`, `Axis`\]
        Mapping from node to Pauli axis.

    Raises
    ------
    ValueError
        If any pair is malformed, duplicated, or uses an invalid axis.
    """
    result: dict[int, Axis] = {}
    for part in parts:
        pair = part.split(":")
        if len(pair) != 2:  # ruff:ignore[magic-value-comparison]
            msg = f"Invalid node:axis pair: {part!r}"
            raise ValueError(msg)
        node_str, axis_str = pair
        node = _parse_int(node_str, "node")
        if node in result:
            msg = f"Duplicate input basis node: {node}"
            raise ValueError(msg)
        try:
            result[node] = Axis[axis_str]
        except KeyError as exc:
            msg = f"Invalid input basis axis: {axis_str!r}"
            raise ValueError(msg) from exc
    return result


def _parse_node_set(parts: Sequence[str], label: str) -> set[int]:
    r"""Parse a non-empty set of node ids.

    Returns
    -------
    `set`\[`int`\]
        Parsed node ids.

    Raises
    ------
    ValueError
        If the node set is empty or contains invalid integers.
    """
    if not parts:
        msg = f"{label} requires at least one node"
        raise ValueError(msg)
    return {_parse_int(part, "node") for part in parts}


def _parse_arrow_mapping(line: str, label: str) -> tuple[int, set[int]]:
    r"""Parse a flow line (xflow or zflow).

    Parameters
    ----------
    line : `str`
        The flow line content after ".xflow" or ".zflow".

    Returns
    -------
    `tuple`\[`int`, `set`\[`int`\]\]
        Source node and set of target nodes.

    Raises
    ------
    ValueError
        If the mapping is malformed.
    """
    parts = line.split("->")
    if len(parts) != 2:  # ruff:ignore[magic-value-comparison]
        msg = f"{label} must contain exactly one '->'"
        raise ValueError(msg)
    source_part = parts[0].strip()
    target_parts = parts[1].strip().split()
    if not source_part:
        msg = f"{label} requires a source node"
        raise ValueError(msg)
    source = _parse_int(source_part, "source node")
    targets = _parse_node_set(target_parts, f"{label} targets")
    return source, targets


@dataclass(slots=True)
class _PatternData:
    """Container for parsed pattern data from .ptn format.

    Attributes
    ----------
    input_node_indices : `dict`[`int`, `int`]
        Mapping from node to qubit index for input nodes.
    output_node_indices : `dict`[`int`, `int`]
        Mapping from node to qubit index for output nodes.
    input_coordinates : `dict`[`int`, `tuple`[`float`, ...]]
        Coordinates for input nodes.
    input_initialization_axes : `dict`[`int`, `Axis`]
        Pauli initialization axes for input nodes.
    commands : `list`[`Command`]
        List of quantum commands.
    xflow : `dict`[`int`, `set`[`int`]]
        X correction flow mapping.
    zflow : `dict`[`int`, `set`[`int`]]
        Z correction flow mapping.
    parity_check_groups : `list`[`set`[`int`]]
        Parity check groups for error detection.
    parity_check_tags : `list`[`str`]
        Stim-style tag of each parity check group, aligned with
        ``parity_check_groups``; the empty string means untagged.
    """

    input_node_indices: dict[int, int] = field(default_factory=dict[int, int])
    output_node_indices: dict[int, int] = field(default_factory=dict[int, int])
    input_coordinates: dict[int, tuple[float, ...]] = field(default_factory=dict[int, tuple[float, ...]])
    input_initialization_axes: dict[int, Axis] = field(default_factory=dict[int, Axis])
    commands: list[Command] = field(default_factory=list[Command])
    xflow: dict[int, set[int]] = field(default_factory=dict[int, set[int]])
    zflow: dict[int, set[int]] = field(default_factory=dict[int, set[int]])
    parity_check_groups: list[set[int]] = field(default_factory=list[set[int]])
    parity_check_tags: list[str] = field(default_factory=list[str])
    logical_observables: dict[int, set[int]] = field(default_factory=dict[int, set[int]])


@dataclass(slots=True)
class _LoadedGraphState(BaseGraphState):
    """Read-only graph state reconstructed from a .ptn file."""

    _input_node_indices: dict[int, int]
    _output_node_indices: dict[int, int]
    _nodes: set[int]
    _edges: set[tuple[int, int]]
    _meas_bases: dict[int, MeasBasis]
    _coordinates: dict[int, tuple[float, ...]]
    _input_initialization_axes: dict[int, Axis]
    _neighbors: dict[int, set[int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._input_node_indices = dict(self._input_node_indices)
        self._output_node_indices = dict(self._output_node_indices)
        self._nodes = set(self._nodes)
        self._edges = {(node1, node2) if node1 < node2 else (node2, node1) for node1, node2 in self._edges}
        self._meas_bases = dict(self._meas_bases)
        self._coordinates = dict(self._coordinates)
        self._input_initialization_axes = dict(self._input_initialization_axes)
        self._neighbors: dict[int, set[int]] = {node: set() for node in self._nodes}
        for node1, node2 in self._edges:
            self._neighbors.setdefault(node1, set()).add(node2)
            self._neighbors.setdefault(node2, set()).add(node1)

    @property
    @typing_extensions.override
    def input_node_indices(self) -> dict[int, int]:
        return self._input_node_indices.copy()

    @property
    @typing_extensions.override
    def input_initialization_axes(self) -> dict[int, Axis]:
        return self._input_initialization_axes.copy()

    @property
    @typing_extensions.override
    def output_node_indices(self) -> dict[int, int]:
        return self._output_node_indices.copy()

    @property
    @typing_extensions.override
    def nodes(self) -> set[int]:
        return set(self._nodes)

    @property
    @typing_extensions.override
    def edges(self) -> set[tuple[int, int]]:
        return set(self._edges)

    @property
    @typing_extensions.override
    def meas_bases(self) -> MappingProxyType[int, MeasBasis]:
        return MappingProxyType(self._meas_bases)

    @property
    @typing_extensions.override
    def coordinates(self) -> dict[int, tuple[float, ...]]:
        return self._coordinates.copy()

    @typing_extensions.override
    def add_node(self, node: int | None = None, *, coordinate: tuple[float, ...] | None = None) -> int:
        msg = "Loaded .ptn graph states are read-only"
        raise NotImplementedError(msg)

    @typing_extensions.override
    def add_edge(self, node1: int, node2: int) -> None:
        msg = "Loaded .ptn graph states are read-only"
        raise NotImplementedError(msg)

    @typing_extensions.override
    def register_input(self, node: int, q_index: int, *, init_axis: Axis = Axis.X) -> None:
        msg = "Loaded .ptn graph states are read-only"
        raise NotImplementedError(msg)

    @typing_extensions.override
    def register_output(self, node: int, q_index: int) -> None:
        msg = "Loaded .ptn graph states are read-only"
        raise NotImplementedError(msg)

    @typing_extensions.override
    def assign_meas_basis(self, node: int, meas_basis: MeasBasis) -> None:
        msg = "Loaded .ptn graph states are read-only"
        raise NotImplementedError(msg)

    @typing_extensions.override
    def neighbors(self, node: int) -> set[int]:
        if node not in self._nodes:
            msg = f"Node does not exist node={node}"
            raise ValueError(msg)
        return self._neighbors.get(node, set()).copy()

    @typing_extensions.override
    def check_canonical_form(self) -> None:
        for node in self._nodes - self._output_node_indices.keys():
            if node not in self._meas_bases:
                msg = f"Measurement basis not set for node {node}"
                raise ValueError(msg)


def _command_nodes(cmd: Command) -> set[int]:
    r"""Return node ids referenced by a command.

    Returns
    -------
    `set`\[`int`\]
        Node ids referenced by the command.
    """
    if isinstance(cmd, (N, M)):
        return {cmd.node}
    if isinstance(cmd, E):
        return set(cmd.nodes)
    return set()


def _input_initialization_axes_from_data(data: _PatternData) -> dict[int, Axis]:
    r"""Validate and normalize parsed input initialization axes.

    Returns
    -------
    `dict`\[`int`, `Axis`\]
        Normalized input initialization axes.

    Raises
    ------
    ValueError
        If an input basis is specified for a non-input node.
    """
    non_input_basis_nodes = set(data.input_initialization_axes) - set(data.input_node_indices)
    if non_input_basis_nodes:
        msg = f"Input basis specified for non-input node(s): {sorted(non_input_basis_nodes)}"
        raise ValueError(msg)

    return {node: data.input_initialization_axes.get(node, Axis.X) for node in data.input_node_indices}


def _build_pattern(data: _PatternData) -> Pattern:
    """Build a Pattern from parsed .ptn data.

    Returns
    -------
    `Pattern`
        Reconstructed pattern.

    Raises
    ------
    ValueError
        If parsed commands contain invalid graph structure.
    """
    input_initialization_axes = _input_initialization_axes_from_data(data)

    nodes: set[int] = set(data.input_node_indices) | set(data.output_node_indices) | set(data.input_coordinates)
    edges: set[tuple[int, int]] = set()
    meas_bases: dict[int, MeasBasis] = {}
    coordinates = dict(data.input_coordinates)

    for cmd in data.commands:
        nodes.update(_command_nodes(cmd))
        if isinstance(cmd, E):
            node1, node2 = cmd.nodes
            edge = (node1, node2) if node1 < node2 else (node2, node1)
            if edge[0] == edge[1]:
                msg = f"Self-loop edge is not allowed: {cmd.nodes}"
                raise ValueError(msg)
            edges.add(edge)
        elif isinstance(cmd, M):
            meas_bases[cmd.node] = cmd.meas_basis
        elif isinstance(cmd, N) and cmd.coordinate is not None:
            coordinates[cmd.node] = cmd.coordinate

    for source, targets in data.xflow.items():
        nodes.add(source)
        nodes.update(targets)
    for source, targets in data.zflow.items():
        nodes.add(source)
        nodes.update(targets)
    for group in data.parity_check_groups:
        nodes.update(group)
    for nodes_in_observable in data.logical_observables.values():
        nodes.update(nodes_in_observable)

    graphstate = _LoadedGraphState(
        _input_node_indices=data.input_node_indices,
        _output_node_indices=data.output_node_indices,
        _nodes=nodes,
        _edges=edges,
        _meas_bases=meas_bases,
        _coordinates=coordinates,
        _input_initialization_axes=input_initialization_axes,
    )
    pauli_frame = PauliFrame(
        graphstate,
        data.xflow,
        data.zflow,
        parity_check_group=data.parity_check_groups,
        logical_observables=data.logical_observables,
        parity_check_tags=data.parity_check_tags,
    )
    return Pattern(
        input_node_indices=dict(data.input_node_indices),
        output_node_indices=dict(data.output_node_indices),
        commands=tuple(data.commands),
        pauli_frame=pauli_frame,
        input_coordinates=dict(data.input_coordinates),
        input_initialization_axes=input_initialization_axes,
    )


def _strip_comment(raw_line: str) -> str:
    r"""Strip a ``#`` comment, keeping ``#`` inside a .detector tag bracket.

    Detector tags use Stim's tag escape language, in which a literal ``]``
    is escaped as ``\\C``, so the first ``]`` always closes the tag.

    Returns
    -------
    `str`
        Line content without its trailing comment.
    """
    stripped = raw_line.lstrip()
    if stripped.startswith(".detector["):
        closing = stripped.find("]")
        if closing != -1:
            return stripped[: closing + 1] + stripped[closing + 1 :].split("#", 1)[0]
    return raw_line.split("#", 1)[0]


class _Parser:
    """Internal parser state for loads()."""

    def __init__(self) -> None:
        self.result = _PatternData()
        self.current_timeslice = -1
        self.version: int | None = None
        self.tick_budget = _IMPLICIT_TICK_SLACK

    def parse(self, s: str) -> Pattern:
        r"""Parse the input string and return Pattern.

        Parameters
        ----------
        s : `str`
            The .ptn format string.

        Returns
        -------
        `Pattern`
            Loaded measurement pattern.

        Raises
        ------
        ValueError
            If the format is invalid or unsupported version.
        """
        lines = s.splitlines()
        self.tick_budget = _IMPLICIT_TICK_SLACK + len(lines)
        for line_num, raw_line in enumerate(lines, 1):
            self._parse_line(line_num, raw_line)

        if self.version is None:
            msg = "Missing .version directive"
            raise ValueError(msg)
        if self.version == 1 and self.result.input_initialization_axes:
            msg = ".input_basis requires .ptn version 2 or later"
            raise ValueError(msg)
        if self.version == 1 and any(self.result.parity_check_tags):
            msg = ".detector tags require .ptn version 2 or later"
            raise ValueError(msg)

        return _build_pattern(self.result)

    def _parse_line(self, line_num: int, raw_line: str) -> None:
        """Parse a single line.

        Raises
        ------
        ValueError
            If the line is malformed.
        """
        line = _strip_comment(raw_line).strip()
        if not line:
            return

        try:
            self._parse_content_line(line)
        except ValueError as exc:
            msg = f"Line {line_num}: {exc}"
            raise ValueError(msg) from exc

    def _parse_content_line(self, line: str) -> None:
        if line.startswith("."):
            self._parse_directive(line)
        elif line.startswith("[") and line.endswith("]"):
            self._parse_timeslice(line)
        else:
            self._parse_command(line)

    def _parse_directive(self, line: str) -> None:
        """Parse a directive line (starts with '.').

        Raises
        ------
        ValueError
            If the directive is invalid.
        """
        if line.startswith(".detector"):
            # Handled before whitespace splitting because a detector tag
            # (".detector[type=flag] 1 2") may itself contain whitespace.
            self._handle_detector(line[len(".detector") :])
            return

        parts = line.split(maxsplit=1)
        directive = parts[0]
        content = parts[1] if len(parts) > 1 else ""

        if directive == ".version":
            self._handle_version(content)
        elif directive == ".input":
            self.result.input_node_indices = _parse_node_qubit_pairs(content.split())
        elif directive == ".input_basis":
            self.result.input_initialization_axes = _parse_node_axis_pairs(content.split())
        elif directive == ".output":
            self.result.output_node_indices = _parse_node_qubit_pairs(content.split())
        elif directive == ".coord":
            self._handle_coord(content)
        elif directive == ".xflow":
            source, targets = _parse_arrow_mapping(content, ".xflow")
            self.result.xflow[source] = targets
        elif directive == ".zflow":
            source, targets = _parse_arrow_mapping(content, ".zflow")
            self.result.zflow[source] = targets
        elif directive == ".observable":
            logical_idx, nodes = _parse_arrow_mapping(content, ".observable")
            self.result.logical_observables[logical_idx] = nodes
        else:
            msg = f"Unknown directive: {directive}"
            raise ValueError(msg)

    def _handle_detector(self, rest: str) -> None:
        """Handle a .detector directive with an optional bracketed tag.

        Raises
        ------
        ValueError
            If the directive or its tag is malformed.
        """
        tag = ""
        if rest.startswith("["):
            closing = rest.find("]")
            if closing == -1:
                msg = ".detector tag is missing its closing ']'"
                raise ValueError(msg)
            tag = unescape_tag(rest[1:closing])
            rest = rest[closing + 1 :]
            if rest and not rest[0].isspace():
                msg = ".detector tag must be followed by whitespace"
                raise ValueError(msg)
        elif rest and not rest[0].isspace():
            msg = f"Unknown directive: .detector{rest.split(maxsplit=1)[0]}"
            raise ValueError(msg)
        self.result.parity_check_groups.append(_parse_node_set(rest.split(), ".detector"))
        self.result.parity_check_tags.append(tag)

    def _handle_version(self, content: str) -> None:
        r"""Handle .version directive.

        Raises
        ------
        ValueError
            If the version is unsupported.
        """
        version = _parse_int(content, "version")
        if version not in SUPPORTED_PTN_VERSIONS:
            supported = ", ".join(str(supported_version) for supported_version in sorted(SUPPORTED_PTN_VERSIONS))
            msg = f"Unsupported .ptn version: {version} (supported: {supported})"
            raise ValueError(msg)
        self.version = version

    def _handle_coord(self, content: str) -> None:
        """Handle .coord directive.

        Raises
        ------
        ValueError
            If the coordinate directive is malformed.
        """
        coord_parts = content.split()
        if len(coord_parts) not in {3, 4}:
            msg = ".coord requires a node and 2D or 3D coordinates"
            raise ValueError(msg)
        node = _parse_int(coord_parts[0], "node")
        coord = _parse_coord(coord_parts[1:])
        self.result.input_coordinates[node] = coord

    def _parse_timeslice(self, line: str) -> None:
        """Parse timeslice marker [n].

        Raises
        ------
        ValueError
            If the timeslice marker is malformed, or if it skips so far ahead that the implied
            TICK commands would exceed the budget for this input.
        """
        slice_num = _parse_int(line[1:-1], "timeslice")
        if slice_num < 0:
            msg = "Timeslice must be non-negative"
            raise ValueError(msg)
        if slice_num < self.current_timeslice:
            msg = "Timeslices must be monotonically increasing"
            raise ValueError(msg)
        ticks_to_insert = slice_num if self.current_timeslice < 0 else slice_num - self.current_timeslice
        if ticks_to_insert > self.tick_budget:
            msg = (
                f"Timeslice {slice_num} would insert {ticks_to_insert} TICK commands, exceeding the "
                f"remaining budget of {self.tick_budget} for an input of this size"
            )
            raise ValueError(msg)
        self.tick_budget -= ticks_to_insert
        self.result.commands.extend(TICK() for _ in range(ticks_to_insert))
        self.current_timeslice = slice_num

    def _parse_command(self, line: str) -> None:
        r"""Parse a command line.

        Raises
        ------
        ValueError
            If the command type is unknown.
        """
        parts = line.split()
        cmd_type = parts[0]

        if cmd_type == "N":
            self._parse_n_command(parts)
        elif cmd_type == "E":
            self._parse_e_command(parts)
        elif cmd_type == "M":
            self._parse_m_command(parts)
        else:
            msg = f"Unknown command: {cmd_type}"
            raise ValueError(msg)

    def _parse_n_command(self, parts: Sequence[str]) -> None:
        """Parse N (node) command.

        Raises
        ------
        ValueError
            If the command is malformed.
        """
        if len(parts) not in {2, 4, 5}:
            msg = "N command requires a node and optional 2D or 3D coordinates"
            raise ValueError(msg)
        node = _parse_int(parts[1], "node")
        coord: tuple[float, ...] | None = _parse_coord(parts[2:]) if len(parts) > 2 else None  # ruff:ignore[magic-value-comparison]
        self.result.commands.append(N(node=node, coordinate=coord))

    def _parse_e_command(self, parts: Sequence[str]) -> None:
        """Parse E (entangle) command.

        Raises
        ------
        ValueError
            If the command is malformed.
        """
        if len(parts) != 3:  # ruff:ignore[magic-value-comparison]
            msg = "E command requires exactly two nodes"
            raise ValueError(msg)
        node1 = _parse_int(parts[1], "node")
        node2 = _parse_int(parts[2], "node")
        self.result.commands.append(E(nodes=(node1, node2)))

    def _parse_m_command(self, parts: Sequence[str]) -> None:
        """Parse M (measure) command.

        Raises
        ------
        ValueError
            If the command is malformed.
        """
        if len(parts) != 4:  # ruff:ignore[magic-value-comparison]
            msg = "M command requires a node, basis, and angle/sign"
            raise ValueError(msg)
        node = _parse_int(parts[1], "node")
        basis_spec = parts[2]
        meas_basis: MeasBasis

        if basis_spec in {"X", "Y", "Z"}:
            sign_str = parts[3]
            if sign_str not in {"+", "-"}:
                msg = f"Invalid Pauli measurement sign: {sign_str!r}"
                raise ValueError(msg)
            sign = Sign.PLUS if sign_str == "+" else Sign.MINUS
            axis = Axis[basis_spec]
            meas_basis = AxisMeasBasis(axis, sign)
        else:
            try:
                plane = Plane[basis_spec]
            except KeyError as exc:
                msg = f"Invalid measurement basis: {basis_spec!r}"
                raise ValueError(msg) from exc
            angle = _parse_angle(parts[3])
            meas_basis = PlannerMeasBasis(plane, angle)

        self.result.commands.append(M(node=node, meas_basis=meas_basis))


def loads(s: str) -> Pattern:
    """Deserialize a .ptn format string to a pattern.

    Parameters
    ----------
    s : `str`
        The .ptn format string.

    Returns
    -------
    `Pattern`
        The loaded pattern.

    See Also
    --------
    _Parser.parse : Internal parser that may raise ValueError for invalid input.
    """
    return _Parser().parse(s)


def load(file: Path | str) -> Pattern:
    """Read a pattern from a .ptn file.

    Parameters
    ----------
    file : `pathlib.Path` | `str`
        The file path to read from.

    Returns
    -------
    `Pattern`
        The loaded pattern.
        See `loads` for details.
    """
    path = Path(file)
    return loads(path.read_text(encoding="utf-8"))
