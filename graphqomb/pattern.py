"""Pattern module.

This module provides:

- `Pattern`: Pattern class
- `is_runnable`: Check if the pattern is runnable
- `print_pattern`: Print a pattern
"""

from __future__ import annotations

import dataclasses
import functools
import typing
from collections.abc import Sequence
from typing import TYPE_CHECKING

import typing_extensions

from graphqomb.command import TICK, Command, E, M, N
from graphqomb.common import Axis

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from graphqomb.pauli_frame import PauliFrame


@dataclasses.dataclass(frozen=True)
class Pattern(Sequence[Command]):
    r"""Pattern class.

    Attributes
    ----------
    input_node_indices : `dict`\[`int`, `int`\]
        The map of input nodes to their logical qubit indices
    output_node_indices : `dict`\[`int`, `int`\]
        The map of output nodes to their logical qubit indices
    commands : `tuple`\[`Command`, ...\]
        Commands of the pattern
    pauli_frame : `PauliFrame`
        Pauli frame of the pattern to track the Pauli state of each node
    input_coordinates : `dict`\[`int`, `tuple`\[`float`, ...\]\]
        Coordinates for input nodes (2D or 3D)
    input_initialization_axes : `dict`\[`int`, `Axis`\]
        Pauli initialization axes for input nodes. Missing inputs default to Axis.X.
    input_initialization_tags : `dict`\[`int`, `str`\]
        Stim-style instruction tags of input initializations. Missing inputs
        default to "" (untagged).
    """

    input_node_indices: dict[int, int]
    output_node_indices: dict[int, int]
    commands: tuple[Command, ...]
    pauli_frame: PauliFrame
    input_coordinates: dict[int, tuple[float, ...]] = dataclasses.field(default_factory=dict[int, tuple[float, ...]])
    input_initialization_axes: dict[int, Axis] = dataclasses.field(default_factory=dict[int, Axis])
    input_initialization_tags: dict[int, str] = dataclasses.field(default_factory=dict[int, str])

    @typing_extensions.override
    def __len__(self) -> int:
        return len(self.commands)

    @typing_extensions.override
    def __iter__(self) -> Iterator[Command]:
        return iter(self.commands)

    @typing.overload
    def __getitem__(self, index: int) -> Command: ...
    @typing.overload
    def __getitem__(self, index: slice) -> tuple[Command, ...]: ...
    @typing_extensions.override
    def __getitem__(self, index: int | slice) -> Command | tuple[Command, ...]:
        return self.commands[index]

    @functools.cached_property
    def max_space(self) -> int:
        """Maximum number of qubits prepared at any point in the pattern.

        Returns
        -------
        `int`
            Maximum number of qubits prepared at any point in the pattern
        """
        return max(self.space)

    @functools.cached_property
    def space(self) -> list[int]:
        r"""List of qubits prepared at each point in the pattern.

        Returns
        -------
        `list`\[`int`\]
            List of qubits prepared at each point in the pattern
        """
        nodes = len(self.input_node_indices)
        space_list = [nodes]
        for cmd in self.commands:
            if isinstance(cmd, N):
                nodes += 1
            elif isinstance(cmd, M):
                nodes -= 1
            elif isinstance(cmd, TICK):
                # TICK does not change the number of qubits
                space_list.append(nodes)
        return space_list

    @property
    def depth(self) -> int:
        """Depth of the pattern (number of TICK commands).

        Returns
        -------
        `int`
            Depth of the pattern
        """
        return sum(1 for cmd in self.commands if isinstance(cmd, TICK))

    @property
    def coordinates(self) -> dict[int, tuple[float, ...]]:
        r"""All node coordinates from N commands and input coordinates.

        Returns
        -------
        `dict`\[`int`, `tuple`\[`float`, ...\]\]
            mapping from node index to coordinate tuple (2D or 3D)
        """
        coords = dict(self.input_coordinates)
        for cmd in self.commands:
            if isinstance(cmd, N) and cmd.coordinate is not None:
                coords[cmd.node] = cmd.coordinate
        return coords

    @property
    def active_volume(self) -> int:
        """Calculate tha active volume, summation of space for each timeslice.

        Returns
        -------
        `int`
            Active volume of the pattern
        """
        return sum(self.space)

    @property
    def volume(self) -> int:
        """Calculate the volume, defined as max_space * depth.

        Returns
        -------
        `int`
            Volume of the pattern
        """
        return self.max_space * self.depth

    @property
    def idle_times(self) -> dict[int, int]:
        r"""Calculate the idle times for each qubit in the pattern.

        Returns
        -------
        `dict`\[`int`, `int`\]
            A dictionary mapping each qubit index to its idle time.
        """
        idle_times: dict[int, int] = {}
        prepared_time: dict[int, int] = dict.fromkeys(self.input_node_indices, 0)

        current_time = 0
        for cmd in self.commands:
            if isinstance(cmd, TICK):
                current_time += 1
            elif isinstance(cmd, N):
                prepared_time[cmd.node] = current_time
            elif isinstance(cmd, M):
                idle_times[cmd.node] = current_time - prepared_time[cmd.node]

        for output_node in self.output_node_indices:
            if output_node in prepared_time and output_node not in idle_times:
                idle_times[output_node] = current_time - prepared_time[output_node]

        return idle_times

    @property
    def throughput(self) -> float:
        """Calculate the number of measurements per TICK in the pattern.

        Returns
        -------
        `float`
            Number of measurements per TICK

        Raises
        ------
        ValueError
            If the pattern has zero depth (no TICK commands)
        """
        num_measurements = sum(1 for cmd in self.commands if isinstance(cmd, M))
        num_ticks = self.depth
        if num_ticks == 0:
            msg = "Cannot calculate throughput for a pattern with zero depth (no TICK commands)."
            raise ValueError(msg)
        return num_measurements / num_ticks


def is_runnable(pattern: Pattern) -> None:
    """Check if the pattern is runnable.

    Parameters
    ----------
    pattern : `Pattern`
        Pattern to check
    """
    _ensure_no_unmeasured_output_dependencies(pattern)
    _ensure_no_operations_on_measured_qubits(pattern)
    _ensure_no_unprepared_qubit_operations(pattern)
    _ensure_measurement_consistency(pattern)


def _ensure_no_unmeasured_output_dependencies(pattern: Pattern) -> None:
    """Ensure that no command depends on an output that has not yet been measured.

    Parameters
    ----------
    pattern : `Pattern`
        The sequence of commands to validate.

    Raises
    ------
    ValueError
        If any command depends on an output that has not yet been measured.
    """
    measured: set[int] = set()
    for cmd in pattern:
        if isinstance(cmd, M):
            measured.add(cmd.node)
            children_nodes = pattern.pauli_frame.parents(cmd.node)
            acausal_children = children_nodes - measured
            if acausal_children:
                msg = f"These nodes depend on a unmeasured output: {sorted(acausal_children)}"
                raise ValueError(msg)


def _ensure_no_operations_on_measured_qubits(pattern: Pattern) -> None:
    """Ensure that no command operates on a qubit that has already been measured.

    Parameters
    ----------
    pattern : `Pattern`
        The sequence of commands to validate.

    Raises
    ------
    ValueError
        If any command attempts to act on a qubit that has already been measured.
    TypeError
        If an unknown command type is encountered.
    """
    measured: set[int] = set()
    for cmd in pattern:
        if isinstance(cmd, M):
            if cmd.node in measured:
                msg = f"A measurement is repeated on the same qubit: {cmd}"
                raise ValueError(msg)
            measured.add(cmd.node)
        elif isinstance(cmd, E):
            if not set(cmd.nodes).isdisjoint(measured):
                msg = f"Entanglement operation targets a measured qubit: {cmd}"
                raise ValueError(msg)
        elif isinstance(cmd, N):
            if cmd.node in measured:
                msg = f"Operation on a measured qubit: {cmd}"
                raise ValueError(msg)
        elif isinstance(cmd, TICK):
            # TICK is a time separator and does not operate on qubits
            pass
        else:
            msg = f"Unknown command kind: {type(cmd)}"
            raise TypeError(msg)


def _ensure_no_unprepared_qubit_operations(pattern: Pattern) -> None:
    """Ensure that no command operates on a qubit that hasn't been prepared yet, except for input qubits.

    Parameters
    ----------
    pattern : `Pattern`
        The sequence of commands to validate.

    Raises
    ------
    ValueError
        If any command targets a qubit that hasn't been prepared.
    """
    prepared = set(pattern.input_node_indices)
    for cmd in pattern:
        if isinstance(cmd, N):
            prepared.add(cmd.node)
        elif isinstance(cmd, E):
            if cmd.nodes[0] not in prepared or cmd.nodes[1] not in prepared:
                msg = f"Entanglement operation targets a qubit that hasn't been prepared yet: {cmd}"
                raise ValueError(msg)
        elif isinstance(cmd, M) and cmd.node not in prepared:
            msg = f"Operation on a qubit that hasn't been prepared yet: {cmd}"
            raise ValueError(msg)
        elif isinstance(cmd, TICK):
            # TICK is a time separator and does not operate on qubits
            pass


def _ensure_measurement_consistency(pattern: Pattern) -> None:
    """Ensure that measurements are applied exactly to all non-output qubits.

    Parameters
    ----------
    pattern : `Pattern`
        The sequence of commands to validate.

    Raises
    ------
    ValueError
        If some non-output qubits are never measured.
    """
    output_nodes = set(pattern.output_node_indices)
    non_output_nodes = (
        {cmd.node for cmd in pattern if isinstance(cmd, N)} | set(pattern.input_node_indices)
    ) - output_nodes
    measured: set[int] = set()
    for cmd in pattern:
        if isinstance(cmd, M):
            measured.add(cmd.node)
    if not non_output_nodes.issubset(measured):
        missing = non_output_nodes - measured
        msg = f"Missing measurements on qubit(s): {sorted(missing)}"
        raise ValueError(msg)


def print_pattern(
    pattern: Pattern,
    *,
    file: typing.TextIO | None = None,
    lim: int = 40,
    cmd_filter: Callable[[Command], Command | None] | None = None,
) -> None:
    r"""Print a pattern.

    Parameters
    ----------
    pattern : `Pattern`
        Pattern to print
    file : `typing.TextIO`, optional
        File to print to, by default None (prints to stdout)
    lim : `int`, optional
        Maximum number of commands to print, by default 40
    cmd_filter : `typing.Callable`\[\[`Command`\], `Command` | `None`\] | `None`, optional
        Command filter, by default None
    """

    def identity_filter(cmd: Command) -> Command | None:
        return cmd if isinstance(cmd, (N, E, M, TICK)) else None

    if cmd_filter is None:
        cmd_filter = identity_filter

    nmax = min(lim, len(pattern))
    print_count = 0
    for i, cmd in enumerate(pattern):
        cmd_filtered = cmd_filter(cmd)
        if cmd_filtered is None:
            continue
        print(cmd_filtered, file=file)
        print_count += 1

        if print_count >= nmax:
            print(
                f"{len(pattern) - i - 1} more commands truncated. Change lim argument of print_pattern() to show more",
                file=file,
            )
            break
