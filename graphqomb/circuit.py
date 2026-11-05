"""Circuit classes for encoding quantum operations.

This module provides:

- `BaseCircuit`: An abstract base class for quantum circuits.
- `MBQCCircuit`: A circuit class composed solely of a unit gate set.
- `Circuit`: A class for circuits that include macro instructions.
- `CircuitScheduleStrategy`: Scheduling strategies for circuit conversion.
- `circuit2graph`: A function that converts a circuit to a graph state, gflow, and scheduler.
"""

from __future__ import annotations

import copy
import enum
import itertools
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

import typing_extensions

from graphqomb.common import Plane, PlannerMeasBasis
from graphqomb.gates import CZ, Gate, J, PhaseGadget, UnitGate
from graphqomb.graphstate import GraphState
from graphqomb.scheduler import Scheduler

if TYPE_CHECKING:
    from collections.abc import Sequence


class CircuitScheduleStrategy(Enum):
    """Enumeration for manual scheduling strategies derived from circuit structure."""

    PARALLEL = enum.auto()
    MINIMIZE_SPACE = enum.auto()


class _Circuit2GraphContext:
    """Internal helper for converting circuits with a given scheduling strategy."""

    graph: GraphState
    gflow: dict[int, set[int]]
    qindex2front_nodes: dict[int, int]
    qindex2timestep: dict[int, int]
    prepare_time: dict[int, int]
    measure_time: dict[int, int]
    minimize_qubits: bool
    current_time: int

    def __init__(self, graph: GraphState, strategy: CircuitScheduleStrategy) -> None:
        _validate_schedule_strategy(strategy)
        self.minimize_qubits = strategy == CircuitScheduleStrategy.MINIMIZE_SPACE

        self.graph = graph
        self.gflow = {}
        self.qindex2front_nodes = {}
        self.qindex2timestep = {}
        self.prepare_time = {}
        self.measure_time = {}
        self.current_time = 0

    def apply_instruction(self, instruction: UnitGate) -> None:
        """Apply a unit gate to the graph conversion context.

        Raises
        ------
        TypeError
            If the instruction type is not supported.
        """
        if isinstance(instruction, J):
            self._apply_j(instruction)
            return
        if isinstance(instruction, CZ):
            self._apply_cz(instruction)
            return
        if isinstance(instruction, PhaseGadget):
            self._apply_phase_gadget(instruction)
            return
        msg = f"Invalid instruction: {instruction}"
        raise TypeError(msg)

    def _apply_j(self, instruction: J) -> None:
        new_node = self.graph.add_node()
        self.graph.add_edge(self.qindex2front_nodes[instruction.qubit], new_node)
        self.graph.assign_meas_basis(
            self.qindex2front_nodes[instruction.qubit],
            PlannerMeasBasis(Plane.XY, -instruction.angle),
        )

        timestep = self.qindex2timestep[instruction.qubit]
        if self.minimize_qubits:
            timestep = max(self.current_time, timestep)
        self.prepare_time[new_node] = timestep
        self.measure_time[self.qindex2front_nodes[instruction.qubit]] = timestep + 1
        self.qindex2timestep[instruction.qubit] = timestep + 1
        if self.minimize_qubits:
            self.current_time = timestep + 1

        self.gflow[self.qindex2front_nodes[instruction.qubit]] = {new_node}
        self.qindex2front_nodes[instruction.qubit] = new_node

    def _apply_cz(self, instruction: CZ) -> None:
        node0 = self.qindex2front_nodes[instruction.qubits[0]]
        node1 = self.qindex2front_nodes[instruction.qubits[1]]
        # Repeated CZs on unchanged frontiers cancel because CZ**2 = I.
        if node1 in self.graph.neighbors(node0):
            self.graph.remove_edge(node0, node1)
        else:
            self.graph.add_edge(node0, node1)

        aligned_time = max(self.qindex2timestep[instruction.qubits[0]], self.qindex2timestep[instruction.qubits[1]])
        if self.minimize_qubits:
            aligned_time = max(self.current_time, aligned_time)
            self.current_time = aligned_time
        self.qindex2timestep[instruction.qubits[0]] = aligned_time
        self.qindex2timestep[instruction.qubits[1]] = aligned_time

    def _apply_phase_gadget(self, instruction: PhaseGadget) -> None:
        new_node = self.graph.add_node()
        self.graph.assign_meas_basis(new_node, PlannerMeasBasis(Plane.YZ, instruction.angle))
        for qubit in instruction.qubits:
            self.graph.add_edge(self.qindex2front_nodes[qubit], new_node)

        self.gflow[new_node] = {new_node}

        max_timestep = max(self.qindex2timestep[qubit] for qubit in instruction.qubits)
        if self.minimize_qubits:
            max_timestep = max(self.current_time, max_timestep)
            self.current_time = max_timestep + 1
        self.prepare_time[new_node] = max_timestep
        self.measure_time[new_node] = max_timestep + 1
        for qubit in instruction.qubits:
            self.qindex2timestep[qubit] = max_timestep + 1


class BaseCircuit(ABC):
    """
    Abstract base class for quantum circuits.

    This class defines the interface for quantum circuit objects.
    It enforces implementation of core methods that must be present
    in any subclass representing a specific type of quantum circuit.
    """

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """The number of qubits in the circuit.

        Returns
        -------
        `int`
            The number of qubits in the circuit
        """
        raise NotImplementedError

    @abstractmethod
    def instructions(self) -> list[Gate]:
        r"""Get the list of gate instructions in the circuit.

        Returns
        -------
        `list`\[`Gate`\]
            List of gate instructions in the circuit.
        """
        raise NotImplementedError

    @abstractmethod
    def unit_instructions(self) -> list[UnitGate]:
        r"""Get the list of unit gate instructions in the circuit.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gate instructions in the circuit.
        """
        raise NotImplementedError


class MBQCCircuit(BaseCircuit):
    """A circuit class composed solely of a unit gate set."""

    __num_qubits: int
    __gate_instructions: list[UnitGate]

    def __init__(self, num_qubits: int) -> None:
        self.__num_qubits = num_qubits
        self.__gate_instructions = []

    @property
    @typing_extensions.override
    def num_qubits(self) -> int:
        """The number of qubits in the circuit.

        Returns
        -------
        `int`
            The number of qubits in the circuit.
        """
        return self.__num_qubits

    @typing_extensions.override
    def instructions(self) -> list[Gate]:
        r"""Get the list of gate instructions in the circuit.

        Returns
        -------
        `list`\[`Gate`\]
            List of gate instructions in the circuit.
        """
        # For MBQCCircuit, Gate and UnitGate are the same
        return [copy.deepcopy(gate) for gate in self.__gate_instructions]

    @typing_extensions.override
    def unit_instructions(self) -> list[UnitGate]:
        r"""Get the list of unit gate instructions in the circuit.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gate instructions in the circuit.
        """
        return [copy.deepcopy(gate) for gate in self.__gate_instructions]

    def j(self, qubit: int, angle: float) -> None:
        """Add a J gate to the circuit.

        Parameters
        ----------
        qubit : `int`
            The qubit index.
        angle : `float`
            The angle of the J gate.
        """
        self.__gate_instructions.append(J(qubit=qubit, angle=angle))

    def cz(self, qubit1: int, qubit2: int) -> None:
        """Add a CZ gate to the circuit.

        Parameters
        ----------
        qubit1 : `int`
            The first qubit index.
        qubit2 : `int`
            The second qubit index.
        """
        self.__gate_instructions.append(CZ(qubits=(qubit1, qubit2)))

    def phase_gadget(self, qubits: Sequence[int], angle: float) -> None:
        r"""Add a phase gadget to the circuit.

        Parameters
        ----------
        qubits : `collections.abc.Sequence`\[`int`\]
            The qubit indices.
        angle : `float`
            The angle of the phase gadget
        """
        self.__gate_instructions.append(PhaseGadget(qubits=list(qubits), angle=angle))


class Circuit(BaseCircuit):
    """A class for circuits that include macro instructions."""

    __num_qubits: int
    __macro_gate_instructions: list[Gate]

    def __init__(self, num_qubits: int) -> None:
        self.__num_qubits = num_qubits
        self.__macro_gate_instructions = []

    @property
    @typing_extensions.override
    def num_qubits(self) -> int:
        """The number of qubits in the circuit.

        Returns
        -------
        `int`
            The number of qubits in the circuit.
        """
        return self.__num_qubits

    @typing_extensions.override
    def instructions(self) -> list[Gate]:
        r"""Get the list of gate instructions in the circuit.

        Returns
        -------
        `list`\[`Gate`\]
            List of gate instructions in the circuit.
        """
        return [copy.deepcopy(gate) for gate in self.__macro_gate_instructions]

    @typing_extensions.override
    def unit_instructions(self) -> list[UnitGate]:
        r"""Get the list of unit gate instructions in the circuit.

        Returns
        -------
        `list`\[`UnitGate`\]
            The list of unit gate instructions in the circuit.
        """
        return list(
            itertools.chain.from_iterable(macro_gate.unit_gates() for macro_gate in self.__macro_gate_instructions)
        )

    def apply_macro_gate(self, gate: Gate) -> None:
        """Apply a macro gate to the circuit.

        Parameters
        ----------
        gate : `Gate`
            The macro gate to apply.
        """
        self.__macro_gate_instructions.append(gate)


def circuit2graph(
    circuit: BaseCircuit,
    schedule_strategy: CircuitScheduleStrategy = CircuitScheduleStrategy.PARALLEL,
) -> tuple[GraphState, dict[int, set[int]], Scheduler]:
    r"""Convert a circuit to a graph state, gflow, and scheduler.

    Parameters
    ----------
    circuit : `BaseCircuit`
        The quantum circuit to convert.
    schedule_strategy : `CircuitScheduleStrategy`, optional
        Strategy for scheduling preparation and measurement times derived from the circuit,
        by default `CircuitScheduleStrategy.PARALLEL`.
        The strategies are:

        - `CircuitScheduleStrategy.PARALLEL`: schedule each qubit independently to reduce depth
        - `CircuitScheduleStrategy.MINIMIZE_SPACE`: serialize operations to reduce prepared qubits

    Returns
    -------
    `tuple`\[`GraphState`, `dict`\[`int`, `set`\[`int`\]\], `Scheduler`\]
        The graph state, gflow, and scheduler converted from the circuit.
        The scheduler is configured with automatic time scheduling derived from circuit structure.

    """
    context = _circuit2graph_context(circuit, schedule_strategy)
    scheduler = Scheduler(context.graph, context.gflow)
    scheduler.manual_schedule(context.prepare_time, context.measure_time)
    return context.graph, context.gflow, scheduler


def _validate_schedule_strategy(strategy: CircuitScheduleStrategy) -> None:
    """Validate the scheduling option retained by circuit and pattern entry points.

    Raises
    ------
    ValueError
        If the strategy is not a supported circuit scheduling strategy.
    """
    if strategy not in {CircuitScheduleStrategy.PARALLEL, CircuitScheduleStrategy.MINIMIZE_SPACE}:
        msg = f"Invalid schedule strategy: {strategy}"
        raise ValueError(msg)


def _circuit2graph_context(circuit: BaseCircuit, schedule_strategy: CircuitScheduleStrategy) -> _Circuit2GraphContext:
    """Build graph, flow, and circuit-derived timings without constructing a scheduler.

    Returns
    -------
    `_Circuit2GraphContext`
        Graph, flow, and circuit-derived timings.
    """
    graph = GraphState()
    context = _Circuit2GraphContext(graph, schedule_strategy)

    # input nodes
    for i in range(circuit.num_qubits):
        node = graph.add_node()
        graph.register_input(node, i)
        context.qindex2front_nodes[i] = node
        context.qindex2timestep[i] = 0

    for instruction in circuit.unit_instructions():
        context.apply_instruction(instruction)

    # output nodes
    for qindex, node in context.qindex2front_nodes.items():
        graph.register_output(node, qindex)

    return context
