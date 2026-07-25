"""Module for simulating circuits and Measurement Patterns.

This module provides:

- `SimulatorBackend` : Enum class for circuit simulator backends.
- `CircuitSimulator` : Class for simulating circuits.
- `PatternSimulator` : Class for simulating Measurement Patterns.
"""

from __future__ import annotations

import functools
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from graphqomb.command import TICK, E, M, N
from graphqomb.common import Axis, MeasBasis, Plane
from graphqomb.gates import MultiGate, SingleGate, TwoQubitGate
from graphqomb.pattern import is_runnable
from graphqomb.rng import ensure_rng
from graphqomb.statevec import StateVector

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from graphqomb.circuit import BaseCircuit
    from graphqomb.command import Command
    from graphqomb.gates import Gate
    from graphqomb.pattern import Pattern
    from graphqomb.simulator_backend import BaseFullStateSimulator

_X_MATRIX = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
_Z_MATRIX = np.asarray([[1, 0], [0, -1]], dtype=np.complex128)
_INPUT_STATE_VECTORS: dict[Axis, NDArray[np.complex128]] = {
    Axis.X: np.asarray([1.0, 1.0], dtype=np.complex128) / np.sqrt(2),
    Axis.Y: np.asarray([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2),
    Axis.Z: np.asarray([1.0, 0.0], dtype=np.complex128),
}


class SimulatorBackend(Enum):
    """Enum class for circuit simulator backend.

    Available backends are:
    - StateVector
    - DensityMatrix
    """

    StateVector = auto()
    DensityMatrix = auto()


class CircuitSimulator:
    r"""Class for simulating circuits.

    Attributes
    ----------
    state : `BaseFullStateSimulator`
        The quantum state of the simulator.
    gate_instructions : `list`\[`Gate`\]
        The list of gate instructions to be applied.
    """

    state: BaseFullStateSimulator
    gate_instructions: list[Gate]

    def __init__(self, mbqc_circuit: BaseCircuit, backend: SimulatorBackend) -> None:
        if backend == SimulatorBackend.StateVector:
            self.state = StateVector.from_num_qubits(mbqc_circuit.num_qubits)
        elif backend == SimulatorBackend.DensityMatrix:
            raise NotImplementedError
        else:
            msg = f"Invalid backend: {backend}"
            raise ValueError(msg)

        self.gate_instructions = mbqc_circuit.instructions()

    def apply_gate(self, gate: Gate) -> None:
        """Apply a gate to the circuit.

        Parameters
        ----------
        gate : `Gate`
            The gate to apply.

        Raises
        ------
        TypeError
            If the gate type is not supported.
        """
        operator = gate.matrix()

        # Get qubits that the gate acts on
        if isinstance(gate, SingleGate):
            # Single qubit gate
            qubits = [gate.qubit]
        elif isinstance(gate, (TwoQubitGate, MultiGate)):
            # Multi-qubit gate (both TwoQubitGate and MultiGate have qubits attribute)
            qubits = list(gate.qubits)
        else:
            msg = f"Cannot determine qubits for gate: {gate}"
            raise TypeError(msg)

        self.state.evolve(operator, qubits)

    def simulate(self) -> None:
        """Simulate the circuit."""
        for gate in self.gate_instructions:
            self.apply_gate(gate)


class PatternSimulator:
    r"""Class for simulating Measurement Patterns.

    Attributes
    ----------
    state : `BaseFullStateSimulator`
        The quantum state of the simulator.
    node_indices : `list`\[`int`\]
        The list of node indices in the pattern.
    results : `dict`\[`int`, `bool`\]
        The measurement results for each node.
    output_results : `dict`\[`int`, `bool`\]
        Measurement results for output nodes, keyed by logical output index.
    calc_prob : `bool`
        Whether to sample every measurement from its exact Born probability.
        If False, non-output measurements use the legacy 50/50 assumption.
    """

    state: BaseFullStateSimulator
    node_indices: list[int]
    results: dict[int, bool]
    output_results: dict[int, bool]
    calc_prob: bool
    __pattern: Pattern

    def __init__(
        self,
        pattern: Pattern,
        backend: SimulatorBackend,
        *,
        calc_prob: bool = True,
    ) -> None:
        self.node_indices = list(pattern.input_node_indices.keys())
        self.results = {}
        self.output_results = {}

        self.calc_prob = calc_prob
        self.__pattern = pattern

        # Pattern runnability check is done via is_runnable function
        is_runnable(self.__pattern)

        if backend == SimulatorBackend.StateVector:
            input_states = [
                _INPUT_STATE_VECTORS[self.__pattern.input_initialization_axes.get(node, Axis.X)]
                for node in self.node_indices
            ]
            self.state = StateVector.from_product_states(input_states)
        elif backend == SimulatorBackend.DensityMatrix:
            raise NotImplementedError
        else:
            msg = f"Invalid backend: {backend}"
            raise ValueError(msg)

    @functools.singledispatchmethod
    def apply_cmd(  # ruff:ignore[no-self-use]
        self, cmd: Command, *, rng: np.random.Generator
    ) -> None:
        """Apply a command to the state.

        Parameters
        ----------
        cmd : `Command`
            The command to apply.
        rng : `numpy.random.Generator`
            Random number generator to use.

        Raises
        ------
        TypeError
            If the command type is not supported by the simulator.
        """
        _ = rng
        msg = f"Unsupported command for pattern simulation: {type(cmd).__name__}"
        raise TypeError(msg)

    @apply_cmd.register
    def _(self, cmd: N, *, rng: np.random.Generator) -> None:  # ruff:ignore[unused-method-argument]
        self.state.add_node(1)
        self.node_indices.append(cmd.node)

    @apply_cmd.register
    def _(self, cmd: E, *, rng: np.random.Generator) -> None:  # ruff:ignore[unused-method-argument]
        node_id1 = self.node_indices.index(cmd.nodes[0])
        node_id2 = self.node_indices.index(cmd.nodes[1])
        self.state.entangle(node_id1, node_id2)

    def _updated_measurement_basis(self, cmd: M) -> MeasBasis:
        basis = cmd.meas_basis
        x_pauli = self.__pattern.pauli_frame.x_pauli[cmd.node]
        z_pauli = self.__pattern.pauli_frame.z_pauli[cmd.node]

        if cmd.meas_basis.plane == Plane.XY:
            if x_pauli:
                basis = basis.conjugate()
            if z_pauli:
                basis = basis.flip()
        elif cmd.meas_basis.plane == Plane.YZ:
            if x_pauli:
                basis = basis.flip()
            if z_pauli:
                basis = basis.conjugate()
        else:
            if x_pauli ^ z_pauli:
                basis = basis.conjugate()
            if x_pauli:
                basis = basis.flip()

        return basis

    def _apply_output_pauli_frame(self, node: int) -> None:
        node_id = self.node_indices.index(node)
        if self.__pattern.pauli_frame.x_pauli[node]:
            self.state.evolve(_X_MATRIX, node_id)
        if self.__pattern.pauli_frame.z_pauli[node]:
            self.state.evolve(_Z_MATRIX, node_id)

    @apply_cmd.register
    def _(self, cmd: M, *, rng: np.random.Generator) -> None:
        node_id = self.node_indices.index(cmd.node)
        meas_basis = self._updated_measurement_basis(cmd)
        if self.calc_prob or cmd.node in self.__pattern.output_node_indices:
            result = self.state.sample_measure(node_id, meas_basis, rng)
        else:
            result = rng.uniform() < 1 / 2
            self.state.measure(node_id, meas_basis, result)
        self.results[cmd.node] = result
        self.node_indices.remove(cmd.node)

        if cmd.node in self.__pattern.output_node_indices:
            qindex = self.__pattern.output_node_indices[cmd.node]
            self.output_results[qindex] = result

        # Measured outputs participate in feedforward like any other node.
        if result:
            self.__pattern.pauli_frame.meas_flip(cmd.node)

    @apply_cmd.register
    def _(self, cmd: TICK, *, rng: np.random.Generator) -> None:
        # TICK is a time separator that doesn't affect quantum state
        pass

    def simulate(self, rng: np.random.Generator | None = None) -> None:
        """
        Simulate the pattern.

        Parameters
        ----------
        rng : `numpy.random.Generator` | None, optional
            Random number generator to use for measurement outcomes.
            If None, a new generator will be created using the default random source. Default is None.

        """
        rng = ensure_rng(rng)
        for cmd in self.__pattern.commands:
            self.apply_cmd(cmd, rng=rng)

        # Only remaining output nodes still have state-vector axes; measured outputs
        # remain in output_node_indices but have been removed from node_indices.
        for node in self.node_indices:
            if node in self.__pattern.output_node_indices:
                self._apply_output_pauli_frame(node)

        # Measured outputs can leave sparse qindices among the remaining quantum
        # outputs. Reorder by each qindex's relative position, not by the qindex
        # value itself.
        output_qindices = [self.__pattern.output_node_indices[node] for node in self.node_indices]
        qindex_rank = {qindex: rank for rank, qindex in enumerate(sorted(output_qindices))}
        permutation = [qindex_rank[qindex] for qindex in output_qindices]

        self.state.reorder(permutation)
