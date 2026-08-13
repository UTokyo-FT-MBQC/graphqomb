"""State vector representation module.

This module provides:

- `StateVector`: State vector representation class.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import typing_extensions

from graphqomb.matrix import is_hermitian
from graphqomb.simulator_backend import BaseFullStateSimulator, QubitIndexManager

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from graphqomb.common import MeasBasis

CZ_TENSOR = np.asarray(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ]
)


class StateVector(BaseFullStateSimulator):
    r"""State vector representation."""

    __state: NDArray[np.complex128]
    __qindex_mng: QubitIndexManager

    def __init__(self, state: ArrayLike | None = None, *, copy: bool | None = None) -> None:
        if state is not None:
            state = np.asarray(state, dtype=np.complex128, copy=copy)
            size = state.size
            if size & (size - 1) or size == 0:
                msg = "State vector must have a size that is a power of 2."
                raise ValueError(msg)
            num_qubits = (size - 1).bit_length()
            self.__state = state.reshape((2,) * num_qubits)
        else:
            num_qubits = 0
            self.__state = np.zeros((2,) * 0, dtype=np.complex128)

        # Internal qubit ordering: maps external qubit index to internal index
        self.__qindex_mng = QubitIndexManager(num_qubits)

    def __array__(self, dtype: DTypeLike | None = None, copy: bool | None = None) -> NDArray[np.generic]:
        state = self.state()
        if dtype is not None:
            target_dtype = np.dtype(dtype)
            if target_dtype.kind in "bifu":
                if np.any(state.imag != 0):
                    msg = "Cannot convert a state vector with nonzero imaginary amplitudes to a real dtype."
                    raise ValueError(msg)
                return np.asarray(state.real, dtype=target_dtype, copy=copy)
        return np.asarray(state, dtype=dtype, copy=copy)

    @property
    @typing_extensions.override
    def num_qubits(self) -> int:
        """The number of qubits in the state vector.

        Returns
        -------
        `int`
            The number of qubits in the state vector.
        """
        return self.__state.ndim

    @typing_extensions.override
    def state(self) -> NDArray[np.complex128]:
        r"""Get the state vector in external qubit order.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            The state vector as a numpy array in external qubit order.
        """
        # If internal order matches external order, return directly
        if self.__qindex_mng.match(range(self.num_qubits)):
            return self.__state

        # Otherwise, reorder to external qubit order
        axes = self.__qindex_mng.inverse_permutation()
        return self.__state.transpose(axes)

    def copy(self) -> StateVector:
        """Create a copy of the state vector.

        Returns
        -------
        `StateVector`
            A new `StateVector` instance with the same state.
        """
        return StateVector(self.state(), copy=True)

    @staticmethod
    def from_num_qubits(num_qubits: int) -> StateVector:
        """Create a state vector in the plus state with given number of qubits.

        Parameters
        ----------
        num_qubits : `int`
            Number of qubits in the state vector.

        Returns
        -------
        `StateVector`
            The resulting state vector in the plus state.

        Raises
        ------
        ValueError
            If num_qubits is negative.
        """
        if num_qubits < 0:
            msg = "Number of qubits must be non-negative."
            raise ValueError(msg)
        return StateVector(np.full((2,) * num_qubits, 1 / math.sqrt(2**num_qubits), dtype=np.complex128))

    @staticmethod
    def from_product_states(states: Sequence[ArrayLike]) -> StateVector:
        r"""Create a state vector from one-qubit product states.

        Parameters
        ----------
        states : `collections.abc.Sequence`\[`numpy.typing.ArrayLike`\]
            One-qubit state vectors in external qubit order.

        Returns
        -------
        `StateVector`
            The resulting product state vector.

        Raises
        ------
        ValueError
            If any one-qubit state does not have exactly two amplitudes.
        """
        result = np.asarray(1.0, dtype=np.complex128)
        for state in states:
            one_qubit_state = np.asarray(state, dtype=np.complex128)
            if one_qubit_state.size != 2:  # ruff:ignore[magic-value-comparison]
                msg = "Each product-state factor must have exactly two amplitudes."
                raise ValueError(msg)
            result = np.asarray(np.kron(result, one_qubit_state.reshape(2)), dtype=np.complex128)
        return StateVector(result)

    @staticmethod
    def tensor_product(a: StateVector, b: StateVector) -> StateVector:
        """Tensor product with other state vector, self ⊗ other.

        Parameters
        ----------
        a : `StateVector`
            first state vector
        b : `StateVector`
            second state vector

        Returns
        -------
        `StateVector`
            The resulting state vector after tensor product.
        """
        return StateVector(np.kron(a.state(), b.state()))

    @typing_extensions.override
    def evolve(self, operator: NDArray[np.complex128], qubits: int | Sequence[int]) -> None:
        r"""Evolve the state by applying an operator to a subset of qubits.

        Parameters
        ----------
        operator : `numpy.typing.NDArray`\[`numpy.complex128`\]
            The operator to apply.
        qubits : `int` | `collections.abc.Sequence`\[`int`\]
            The qubits to apply the operator to.
        """
        # Convert external qubit indices to internal indices
        internal_qubits = self.__qindex_mng.external_to_internal(qubits)
        internal_qubits = (internal_qubits,) if isinstance(internal_qubits, int) else internal_qubits
        k = len(internal_qubits)

        rest = tuple(i for i in range(self.num_qubits) if i not in set(internal_qubits))
        perm = internal_qubits + rest
        inv_perm = np.argsort(perm)

        op_tensor = operator.reshape((2,) * (2 * k))

        contracted = np.tensordot(op_tensor, self.__state, axes=(range(k, 2 * k), internal_qubits))
        contracted = contracted.transpose(inv_perm)
        self.__state = np.asarray(contracted, dtype=np.complex128, copy=False)  # for type checker

    @typing_extensions.override
    def measure(self, qubit: int, meas_basis: MeasBasis, result: int) -> None:
        """Measure a qubit in a given measurement basis.

        Parameters
        ----------
        qubit : `int`
            The qubit to measure.
        meas_basis : `MeasBasis`
            The measurement basis to use.
        result : `int`
            The measurement result.
        """
        # Convert external qubit index to internal index
        internal_qubit = self.__qindex_mng.external_to_internal(qubit)

        meas_basis = meas_basis.flip() if result else meas_basis
        basis_vector = meas_basis.vector()
        new_state = np.tensordot(basis_vector.conjugate(), self.__state, axes=(0, internal_qubit))
        self.__state = np.asarray(new_state, dtype=np.complex128, copy=False)  # for type checker

        # Update qubit order: remove the measured qubit
        self.__qindex_mng.remove_qubit(internal_qubit)

        self.normalize()

    @typing_extensions.override
    def sample_measure(
        self,
        qubit: int,
        meas_basis: MeasBasis,
        rng: np.random.Generator,
    ) -> bool:
        """Sample and apply a measurement without repeating the sampled projection.

        Parameters
        ----------
        qubit : `int`
            The qubit to measure.
        meas_basis : `MeasBasis`
            The measurement basis to use.
        rng : `numpy.random.Generator`
            Random number generator used to sample the outcome.

        Returns
        -------
        `bool`
            The sampled measurement result.
        """
        internal_qubit = self.__qindex_mng.external_to_internal(qubit)
        norm_sq = float(np.real(np.vdot(self.__state, self.__state)))
        basis_vector = meas_basis.vector()
        projected = np.tensordot(basis_vector.conjugate(), self.__state, axes=(0, internal_qubit))
        projected_norm_sq = float(np.real(np.vdot(projected, projected)))
        prob_false = projected_norm_sq / norm_sq
        prob_false = min(1.0, max(0.0, prob_false))
        result = bool(rng.uniform() >= prob_false)

        if result:
            basis_vector = meas_basis.flip().vector()
            projected = np.tensordot(basis_vector.conjugate(), self.__state, axes=(0, internal_qubit))
            projected_norm_sq = float(np.real(np.vdot(projected, projected)))

        self.__state = np.asarray(projected / math.sqrt(projected_norm_sq), dtype=np.complex128)
        self.__qindex_mng.remove_qubit(internal_qubit)
        return result

    @typing_extensions.override
    def add_node(self, num_qubits: int) -> None:
        """Add plus state to the end of state vector.

        Parameters
        ----------
        num_qubits : `int`
            number of qubits to add
        """
        flat_state: NDArray[np.complex128] = self.__state.ravel()
        repeated_flat = np.repeat(flat_state, 1 << num_qubits) / math.sqrt(2**num_qubits)
        repeated_state = np.asarray(repeated_flat, dtype=np.complex128)
        self.__state = repeated_state.reshape((2,) * (self.num_qubits + num_qubits))
        # Append new qubits to the end of the qubit order
        self.__qindex_mng.add_qubits(num_qubits)

    @typing_extensions.override
    def entangle(self, qubit1: int, qubit2: int) -> None:
        r"""Entangle two qubits.

        Parameters
        ----------
        qubit1 : `int`
            first qubit index
        qubit2 : `int`
            second qubit index
        """
        self.evolve(CZ_TENSOR, (qubit1, qubit2))

    def normalize(self) -> None:
        """Normalize the state."""
        self.__state /= np.linalg.norm(self.__state)

    @typing_extensions.override
    def reorder(self, permutation: Sequence[int]) -> None:
        r"""Permute qubits.

        if permutation is [2, 0, 1], then
        # [q0, q1, q2] -> [q1, q2, q0]

        Parameters
        ----------
        permutation : `collections.abc.Sequence`\[`int`\]
            permutation list
        """
        # Update the internal qubit order only (no state reordering)
        self.__qindex_mng.reorder(permutation)

    @typing_extensions.override
    def norm(self) -> float:
        """Get norm of state vector.

        Returns
        -------
        `float`
            norm of state vector
        """
        return float(np.linalg.norm(self.__state))

    @typing_extensions.override
    def expectation(self, operator: NDArray[np.complex128], qubits: int | Sequence[int]) -> float:
        r"""Calculate expectation value of operator.

        Parameters
        ----------
        operator : `numpy.typing.NDArray`\[`numpy.complex128`\]
            Hermitian operator matrix
        qubits : `int` | `collections.abc.Sequence`\[`int`\]
            target qubits

        Returns
        -------
        `float`
            expectation value

        Raises
        ------
        ValueError
            if operator is not Hermitian
        """
        if not is_hermitian(operator):
            msg = "Operator must be Hermitian"
            raise ValueError(msg)

        # Convert external qubit indices to internal indices
        internal_qubits = self.__qindex_mng.external_to_internal(qubits)
        internal_qubits = (internal_qubits,) if isinstance(internal_qubits, int) else internal_qubits
        k = len(internal_qubits)

        rest = tuple(i for i in range(self.num_qubits) if i not in internal_qubits)
        perm = internal_qubits + rest

        state_perm = self.__state.transpose(perm)
        op_tensor = operator.reshape((2,) * (2 * k))

        # Apply operator: O|ψ⟩
        transformed_state = np.tensordot(op_tensor, state_perm, axes=(range(k, 2 * k), range(k)))

        # Calculate expectation value: ⟨ψ|O|ψ⟩
        norm_squared = np.real(np.vdot(state_perm, state_perm))
        expectation = np.real(np.vdot(state_perm, transformed_state))

        return float(expectation / norm_squared)
