"""Module for gates used in circuit representation.

This module provides:

- `Gate`: Abstract class for gates.
- `UnitGate`: Abstract class for unit gates.
- `J`: Class for the J gate.
- `CZ`: Class for the CZ gate.
- `PhaseGadget`: Class for the PhaseGadget gate.
- `Identity`: Class for the Identity gate.
- `X`: Class for the X gate.
- `Y`: Class for the Y gate.
- `Z`: Class for the Z gate.
- `H`: Class for the H gate.
- `S`: Class for the S gate.
- `T`: Class for the T gate.
- `Tdg`: Class for the Tdg gate.
- `Rx`: Class for the Rx gate.
- `Ry`: Class for the Ry gate.
- `Rz`: Class for the Rz gate.
- `U3`: Class for the U3 gate.
- `CNOT`: Class for the CNOT gate.
- `SWAP`: Class for the SWAP gate.
- `CRz`: Class for the CRz gate.
- `CRx`: Class for the CRx gate.
- `CU3`: Class for the CU3 gate.
- `Toffoli`: Class for the Toffoli gate.
- `CCZ`: Class for the CCZ gate.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import typing_extensions

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Gate(ABC):
    """Abstract class for gates."""

    @abstractmethod
    def unit_gates(self) -> list[UnitGate]:
        r"""Get the unit gates that make up the gate.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gates that make up the gate.
        """
        raise NotImplementedError

    @abstractmethod
    def matrix(self) -> NDArray[np.complex128]:
        r"""Get the matrix representation of the gate.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            Matrix representation of the gate.
        """
        raise NotImplementedError


class SingleGate(Gate):
    """Base class for single qubit macro gates."""

    qubit: int


class TwoQubitGate(Gate):
    """Base class for two qubit macro gates."""

    qubits: tuple[int, int]


class MultiGate(Gate):
    """Base class for multi qubit macro gates."""

    qubits: list[int]


@dataclass(frozen=True)
class J(SingleGate):
    r"""Class for the J gate.

    Attributes
    ----------
    qubit : `int`
        The qubit the gate acts on.
    angle : `float`
        The angle of the J gate.

        .. math::

            J = \frac{1}{\sqrt{2}}
            \begin{pmatrix}
            1 & e^{i\theta} \\
            1 & -e^{i\theta}
            \end{pmatrix}

    """

    qubit: int
    angle: float

    @typing_extensions.override
    def unit_gates(self) -> list[UnitGate]:
        r"""Get the unit gates that make up the gate.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gates that make up the gate.
        """
        return [self]

    @typing_extensions.override
    def matrix(self) -> NDArray[np.complex128]:
        r"""Get the matrix representation of the gate.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            Matrix representation of the gate.
        """
        array: NDArray[np.complex128] = np.asarray(
            [[1.0 + 0j, np.exp(1j * self.angle)], [1.0 + 0j, -np.exp(1j * self.angle)]],
            dtype=np.complex128,
        ) / np.sqrt(2)
        return array


@dataclass(frozen=True)
class CZ(TwoQubitGate):
    r"""Class for the CZ gate.

    Attributes
    ----------
    qubits : `tuple`\[`int`, `int`\]
        The qubits the gate acts on.

        .. math::

            CZ = \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 0 & 0 & -1
            \end{pmatrix}

    """

    qubits: tuple[int, int]

    @typing_extensions.override
    def unit_gates(self) -> list[UnitGate]:
        r"""Get the `unit_gates` that make up the gate.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gates that make up the gate.
        """
        return [self]

    @typing_extensions.override
    def matrix(
        self,
    ) -> NDArray[np.complex128]:
        r"""Get the `matrix` representation of the gate.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            Matrix representation of the gate.
        """
        return np.asarray(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
            dtype=np.complex128,
        )


@dataclass(frozen=True)
class PhaseGadget(MultiGate):
    r"""Class for the PhaseGadget gate.

    Attributes
    ----------
    qubits : `list`\[`int`\]
        The qubits the gate acts on.
    angle : `float`
        The angle of the PhaseGadget gate.

        .. math::

            PhaseGadget(\theta) = \exp\left(-i\frac{\theta}{2}\prod_{j}Z_j\right)

    """

    qubits: list[int]
    angle: float

    @typing_extensions.override
    def unit_gates(self) -> list[UnitGate]:
        r"""Get the `unit_gates` that make up the gate.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gates that make up the gate.
        """
        return [self]

    @typing_extensions.override
    def matrix(self) -> NDArray[np.complex128]:
        r"""Get the `matrix` representation of the gate.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            Matrix representation of the gate.
        """

        def count_ones_in_binary(array: NDArray[np.int64]) -> NDArray[np.int64]:
            def count_ones_single(x: np.int64) -> int:
                return int(x).bit_count()

            count_ones = np.vectorize(count_ones_single)
            return np.asarray(count_ones(array), dtype=np.int64)

        index_array: NDArray[np.int64] = np.arange(2 ** len(self.qubits), dtype=np.int64)
        z_sign = (-1) ** count_ones_in_binary(index_array)
        return np.diag(np.exp(-1j * self.angle / 2 * z_sign))


UnitGate: TypeAlias = J | CZ | PhaseGadget
"""Unit gate type"""


@dataclass(frozen=True)
class Identity(SingleGate):
    r"""Class for the Identity gate.

    Attributes
    ----------
    qubit : `int`
        The qubit the gate acts on.

        .. math::

            I = \begin{pmatrix}
            1 & 0 \\
            0 & 1
            \end{pmatrix}

    """

    qubit: int

    @typing_extensions.override
    def unit_gates(self) -> list[UnitGate]:
        r"""Get the `unit_gates` that make up the gate.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gates that make up the gate.
        """
        return [J(self.qubit, 0), J(self.qubit, 0)]

    @typing_extensions.override
    def matrix(self) -> NDArray[np.complex128]:
        r"""Get the `matrix` representation of the gate.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            Identity matrix.
        """
        return np.eye(2, dtype=np.complex128)


@dataclass(frozen=True)
class X(SingleGate):
    r"""Class for the X gate.

    Attributes
    ----------
    qubit : `int`
        The qubit the gate acts on.

        .. math::

            X = \begin{pmatrix}
            0 & 1 \\
            1 & 0
            \end{pmatrix}

    """

    qubit: int

    @typing_extensions.override
    def unit_gates(self) -> list[UnitGate]:
        r"""Get the `unit_gates` that make up the gate.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gates that make up the gate.
        """
        return [J(self.qubit, 0), J(self.qubit, math.pi)]

    @typing_extensions.override
    def matrix(self) -> NDArray[np.complex128]:
        r"""Get the `matrix` representation of the gate.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            X gate matrix.
        """
        return np.asarray([[0, 1], [1, 0]], dtype=np.complex128)


@dataclass(frozen=True)
class Y(SingleGate):
    r"""Class for the Y gate.

    Attributes
    ----------
    qubit : `int`
        The qubit the gate acts on.

        .. math::

            Y = \begin{pmatrix}
            0 & -i \\
            i & 0
            \end{pmatrix}

    """

    qubit: int

    @typing_extensions.override
    def unit_gates(self) -> list[UnitGate]:
        r"""Get the `unit_gates` that make up the gate.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gates that make up the gate.
        """
        return [
            J(self.qubit, math.pi / 2),
            J(self.qubit, math.pi),
            J(self.qubit, -math.pi / 2),
            J(self.qubit, 0),
        ]

    @typing_extensions.override
    def matrix(self) -> NDArray[np.complex128]:
        r"""Get the `matrix` representation of the gate.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            Y gate matrix.
        """
        return np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128)


@dataclass(frozen=True)
class Z(SingleGate):
    r"""Class for the Z gate.

    Attributes
    ----------
    qubit : `int`
        The qubit the gate acts on.

        .. math::

            Z = \begin{pmatrix}
            1 & 0 \\
            0 & -1
            \end{pmatrix}

    """

    qubit: int

    @typing_extensions.override
    def unit_gates(self) -> list[UnitGate]:
        r"""Get the `unit_gates` that make up the gate.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gates that make up the gate.
        """
        return [J(self.qubit, math.pi), J(self.qubit, 0)]

    @typing_extensions.override
    def matrix(self) -> NDArray[np.complex128]:
        r"""Get the `matrix` representation of the gate.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            Z gate matrix.
        """
        return np.asarray([[1, 0], [0, -1]], dtype=np.complex128)


@dataclass(frozen=True)
class H(SingleGate):
    r"""Class for the H gate.

    Attributes
    ----------
    qubit : `int`
        The qubit the gate acts on.

        .. math::

            H = \frac{1}{\sqrt{2}}\begin{pmatrix}
            1 & 1 \\
            1 & -1
            \end{pmatrix}

    """

    qubit: int

    @typing_extensions.override
    def unit_gates(self) -> list[UnitGate]:
        r"""Get the `unit_gates` that make up the gate.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gates that make up the gate.
        """
        return [J(self.qubit, 0)]

    @typing_extensions.override
    def matrix(self) -> NDArray[np.complex128]:
        r"""Get the `matrix` representation of the gate.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            H gate matrix.
        """
        array: NDArray[np.complex128] = (1 / np.sqrt(2)) * np.asarray([[1, 1], [1, -1]], dtype=np.complex128)
        return array


@dataclass(frozen=True)
class S(SingleGate):
    r"""Class for the S gate.

    Attributes
    ----------
    qubit : `int`
        The qubit the gate acts on.

        .. math::

            S = \begin{pmatrix}
            1 & 0 \\
            0 & i
            \end{pmatrix}

    """

    qubit: int

    @typing_extensions.override
    def unit_gates(self) -> list[UnitGate]:
        r"""Get the `unit_gates` that make up the gate.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gates that make up the gate.
        """
        return [J(self.qubit, math.pi / 2), J(self.qubit, 0)]

    @typing_extensions.override
    def matrix(self) -> NDArray[np.complex128]:
        r"""Get the `matrix` representation of the gate.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            S gate matrix.
        """
        return np.asarray([[1, 0], [0, 1j]], dtype=np.complex128)


@dataclass(frozen=True)
class T(SingleGate):
    r"""Class for the T gate.

    Attributes
    ----------
    qubit : `int`
        The qubit the gate acts on.

        .. math::

            T = \begin{pmatrix}
            1 & 0 \\
            0 & e^{i\pi/4}
            \end{pmatrix}

    """

    qubit: int

    @typing_extensions.override
    def unit_gates(self) -> list[UnitGate]:
        r"""Get the `unit_gates` that make up the gate.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gates that make up the gate.
        """
        return [J(self.qubit, math.pi / 4), J(self.qubit, 0)]

    @typing_extensions.override
    def matrix(self) -> NDArray[np.complex128]:
        r"""Get the `matrix` representation of the gate.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            T gate matrix.
        """
        return np.asarray([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=np.complex128)


@dataclass(frozen=True)
class Tdg(SingleGate):
    r"""Class for the Tdg gate.

    Attributes
    ----------
    qubit : `int`
        The qubit the gate acts on.

        .. math::

            T^\dagger = \begin{pmatrix}
            1 & 0 \\
            0 & e^{-i\pi/4}
            \end{pmatrix}

    """

    qubit: int

    @typing_extensions.override
    def unit_gates(self) -> list[UnitGate]:
        r"""Get the `unit_gates` that make up the gate.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gates that make up the gate.
        """
        return [J(self.qubit, -math.pi / 4), J(self.qubit, 0)]

    @typing_extensions.override
    def matrix(self) -> NDArray[np.complex128]:
        r"""Get the `matrix` representation of the gate.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            Tdg gate matrix.
        """
        return np.asarray([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=np.complex128)


@dataclass(frozen=True)
class Rx(SingleGate):
    r"""Class for the Rx gate.

    Attributes
    ----------
    qubit : `int`
        The qubit the gate acts on.
    angle : `float`
        The angle of the Rx gate.

        .. math::

            R_x(\theta) = \begin{pmatrix}
            \cos(\theta/2) & -i\sin(\theta/2) \\
            -i\sin(\theta/2) & \cos(\theta/2)
            \end{pmatrix}

    """

    qubit: int
    angle: float

    @typing_extensions.override
    def unit_gates(self) -> list[UnitGate]:
        r"""Get the `unit_gates` that make up the gate.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gates that make up the gate.
        """
        return [
            J(self.qubit, 0),
            J(self.qubit, self.angle),
        ]

    @typing_extensions.override
    def matrix(self) -> NDArray[np.complex128]:
        r"""Get the `matrix` representation of the gate.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            Rx gate matrix.
        """
        return np.asarray(
            [
                [np.cos(self.angle / 2), -1j * np.sin(self.angle / 2)],
                [-1j * np.sin(self.angle / 2), np.cos(self.angle / 2)],
            ],
            dtype=np.complex128,
        )


@dataclass(frozen=True)
class Ry(SingleGate):
    r"""Class for the Ry gate.

    Attributes
    ----------
    qubit : `int`
        The qubit the gate acts on.
    angle : `float`
        The angle of the Ry gate.

        .. math::

            R_y(\theta) = \begin{pmatrix}
            \cos(\theta/2) & -\sin(\theta/2) \\
            \sin(\theta/2) & \cos(\theta/2)
            \end{pmatrix}

    """

    qubit: int
    angle: float

    @typing_extensions.override
    def unit_gates(self) -> list[UnitGate]:
        r"""Get the `unit_gates` that make up the gate.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gates that make up the gate.
        """
        return [
            J(self.qubit, math.pi / 2),
            J(self.qubit, -self.angle),
            J(self.qubit, -math.pi / 2),
            J(self.qubit, 0),
        ]

    @typing_extensions.override
    def matrix(self) -> NDArray[np.complex128]:
        r"""Get the `matrix` representation of the gate.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            Ry gate matrix.
        """
        return np.asarray(
            [
                [np.cos(self.angle / 2), -np.sin(self.angle / 2)],
                [np.sin(self.angle / 2), np.cos(self.angle / 2)],
            ],
            dtype=np.complex128,
        )


@dataclass(frozen=True)
class Rz(SingleGate):
    r"""Class for the Rz gate.

    Attributes
    ----------
    qubit : `int`
        The qubit the gate acts on.
    angle : `float`
        The angle of the Rz gate.

        .. math::

            R_z(\theta) = \begin{pmatrix}
            e^{-i\theta/2} & 0 \\
            0 & e^{i\theta/2}
            \end{pmatrix}

    """

    qubit: int
    angle: float

    @typing_extensions.override
    def unit_gates(self) -> list[UnitGate]:
        r"""Get the `unit_gates` that make up the gate.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gates that make up the gate.
        """
        return [J(self.qubit, self.angle), J(self.qubit, 0)]

    @typing_extensions.override
    def matrix(self) -> NDArray[np.complex128]:
        r"""Get the `matrix` representation of the gate.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            Rz gate matrix.
        """
        return np.asarray(
            [[np.exp(-1j * self.angle / 2), 0], [0, np.exp(1j * self.angle / 2)]],
            dtype=np.complex128,
        )


@dataclass(frozen=True)
class U3(SingleGate):
    r"""Class for the U3 gate.

    Attributes
    ----------
    qubit : `int`
        The qubit the gate acts on.
    angle1 : `float`
        The first angle of the U3 gate.
    angle2 : `float`
        The second angle of the U3 gate.
    angle3 : `float`
        The third angle of the U3 gate.

        .. math::

            U3(\theta, \phi, \lambda) = \begin{pmatrix}
            \cos(\theta/2) & -e^{i\lambda}\sin(\theta/2) \\
            e^{i\phi}\sin(\theta/2) & e^{i(\phi+\lambda)}\cos(\theta/2)
            \end{pmatrix}

    """

    qubit: int
    angle1: float
    angle2: float
    angle3: float

    @typing_extensions.override
    def unit_gates(self) -> list[UnitGate]:
        r"""Get the `unit_gates` that make up the gate.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gates that make up the gate.
        """
        return [
            J(self.qubit, self.angle3 - math.pi / 2),
            J(self.qubit, self.angle1),
            J(self.qubit, self.angle2 + math.pi / 2),
            J(self.qubit, 0),
        ]

    @typing_extensions.override
    def matrix(self) -> NDArray[np.complex128]:
        r"""Get the `matrix` representation of the gate.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            U3 gate matrix.
        """
        return np.asarray(
            [
                [
                    np.cos(self.angle1 / 2),
                    -np.exp(1j * self.angle3) * np.sin(self.angle1 / 2),
                ],
                [
                    np.exp(1j * self.angle2) * np.sin(self.angle1 / 2),
                    np.exp(1j * (self.angle2 + self.angle3)) * np.cos(self.angle1 / 2),
                ],
            ],
            dtype=np.complex128,
        )


@dataclass(frozen=True)
class CNOT(TwoQubitGate):
    r"""Class for the CNOT gate.

    Attributes
    ----------
    qubits : `tuple`\[`int`, `int`\]
        The qubits the gate acts on [control target].

        .. math::

            CNOT = \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 1 \\
            0 & 0 & 1 & 0
            \end{pmatrix}

    """

    qubits: tuple[int, int]

    @typing_extensions.override
    def unit_gates(self) -> list[UnitGate]:
        r"""Get the `unit_gates` that make up the gate.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gates that make up the gate.
        """
        target = self.qubits[1]
        return [
            J(target, 0),
            CZ((self.qubits[0], self.qubits[1])),
            J(target, 0),
        ]

    @typing_extensions.override
    def matrix(self) -> NDArray[np.complex128]:
        r"""Get the `matrix` representation of the gate.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            Matrix representation of the gate.
        """
        return np.asarray(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=np.complex128,
        )


@dataclass(frozen=True)
class SWAP(TwoQubitGate):
    r"""Class for the SWAP gate.

    Attributes
    ----------
    qubits : `tuple`\[`int`, `int`\]
        The qubits the gate acts on [control target].

        .. math::

            SWAP = \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 1
            \end{pmatrix}

    """

    qubits: tuple[int, int]

    @typing_extensions.override
    def unit_gates(self) -> list[UnitGate]:
        r"""Get the `unit_gates` that make up the gate.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gates that make up the gate.
        """
        control, target = self.qubits
        macro_gates: list[Gate] = [
            CNOT(self.qubits),
            CNOT((target, control)),
            CNOT(self.qubits),
        ]
        unit_gates: list[UnitGate] = []
        for macro_gate in macro_gates:
            unit_gates.extend(macro_gate.unit_gates())
        return unit_gates

    @typing_extensions.override
    def matrix(self) -> NDArray[np.complex128]:
        r"""Get the `matrix` representation of the gate.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            Matrix representation of the gate.
        """
        return np.asarray(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.complex128,
        )


@dataclass(frozen=True)
class CRz(TwoQubitGate):
    r"""Class for the CRz gate.

    Attributes
    ----------
    qubits : `tuple`\[`int`, `int`\]
        The qubits the gate acts on [control target].
    angle : `float`
        The angle of the CRz gate.

        .. math::

            CR_z(\theta) = \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & e^{-i\theta/2} & 0 \\
            0 & 0 & 0 & e^{i\theta/2}
            \end{pmatrix}

    """

    qubits: tuple[int, int]
    angle: float

    @typing_extensions.override
    def unit_gates(self) -> list[UnitGate]:
        r"""Get the `unit_gates` that make up the gate.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gates that make up the gate.
        """
        target = self.qubits[1]
        macro_gates: list[Gate] = [
            Rz(target, self.angle / 2),
            CNOT(self.qubits),
            Rz(target, -self.angle / 2),
            CNOT(self.qubits),
        ]
        unit_gates: list[UnitGate] = []
        for macro_gate in macro_gates:
            unit_gates.extend(macro_gate.unit_gates())
        return unit_gates

    @typing_extensions.override
    def matrix(self) -> NDArray[np.complex128]:
        r"""Get the `matrix` representation of the gate.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            Matrix representation of the gate.
        """
        return np.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.exp(-1j * self.angle / 2), 0],
                [0, 0, 0, np.exp(1j * self.angle / 2)],
            ],
            dtype=np.complex128,
        )


@dataclass(frozen=True)
class CRx(TwoQubitGate):
    r"""Class for the CRx gate.

    Attributes
    ----------
    qubits : `tuple`\[`int`, `int`\]
        The qubits the gate acts on [control target].
    angle : `float`
        The angle of the CRx gate.

        .. math::

            CR_x(\theta) = \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & \cos(\theta/2) & -i\sin(\theta/2) \\
            0 & 0 & -i\sin(\theta/2) & \cos(\theta/2)
            \end{pmatrix}

    """

    qubits: tuple[int, int]
    angle: float

    @typing_extensions.override
    def unit_gates(self) -> list[UnitGate]:
        r"""Get the `unit_gates` that make up the gate.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gates that make up the gate.
        """
        target = self.qubits[1]
        macro_gates: list[Gate] = [
            H(target),
            CRz(self.qubits, self.angle),
            H(target),
        ]
        unit_gates: list[UnitGate] = []
        for macro_gate in macro_gates:
            unit_gates.extend(macro_gate.unit_gates())
        return unit_gates

    @typing_extensions.override
    def matrix(self) -> NDArray[np.complex128]:
        r"""Get the `matrix` representation of the gate.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            Matrix representation of the gate.
        """
        return np.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.cos(self.angle / 2), -1j * np.sin(self.angle / 2)],
                [0, 0, -1j * np.sin(self.angle / 2), np.cos(self.angle / 2)],
            ],
            dtype=np.complex128,
        )


@dataclass(frozen=True)
class CU3(TwoQubitGate):
    r"""Class for the CU3 gate.

    Attributes
    ----------
    qubits : `tuple`\[`int`, `int`\]
        The qubits the gate acts on.
    angle1 : `float`
        The first angle of the CU3 gate.
    angle2 : `float`
        The second angle of the CU3 gate.
    angle3 : `float`
        The third angle of the CU3 gate.

        .. math::

            CU3(\theta, \phi, \lambda) = \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & \cos(\theta/2) & -e^{i\lambda}\sin(\theta/2) \\
            0 & 0 & e^{i\phi}\sin(\theta/2) & e^{i(\phi+\lambda)}\cos(\theta/2)
            \end{pmatrix}

    """

    qubits: tuple[int, int]
    angle1: float
    angle2: float
    angle3: float

    @typing_extensions.override
    def unit_gates(self) -> list[UnitGate]:
        r"""Get the `unit_gates` that make up the gate.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gates that make up the gate.
        """
        control, target = self.qubits
        macro_gates: list[Gate] = [
            Rz(control, self.angle3 / 2 + self.angle2 / 2),
            Rz(target, self.angle3 / 2 - self.angle2 / 2),
            CNOT(self.qubits),
            U3(target, -self.angle1 / 2, 0, -(self.angle2 + self.angle3) / 2),
            CNOT(self.qubits),
            U3(target, self.angle1 / 2, self.angle2, 0),
        ]
        unit_gates: list[UnitGate] = []
        for macro_gate in macro_gates:
            unit_gates.extend(macro_gate.unit_gates())
        return unit_gates

    @typing_extensions.override
    def matrix(self) -> NDArray[np.complex128]:
        r"""Get the `matrix` representation of the gate.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            Matrix representation of the gate.
        """
        return np.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [
                    0,
                    0,
                    np.cos(self.angle1 / 2),
                    -np.exp(1j * self.angle3) * np.sin(self.angle1 / 2),
                ],
                [
                    0,
                    0,
                    np.exp(1j * self.angle2) * np.sin(self.angle1 / 2),
                    np.exp(1j * (self.angle2 + self.angle3)) * np.cos(self.angle1 / 2),
                ],
            ],
            dtype=np.complex128,
        )


@dataclass(frozen=True)
class Toffoli(MultiGate):
    r"""Class for the Toffoli gate.

    Attributes
    ----------
    qubits : `list`\[`int`\]
        The qubits the gate acts on [control1, control2, target].

        .. math::

            Toffoli = \begin{pmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
            \end{pmatrix}

    """

    qubits: list[int]

    @typing_extensions.override
    def unit_gates(self) -> list[UnitGate]:
        r"""Get the `unit_gates` that make up the gate.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gates that make up the gate.
        """
        control1, control2, target = self.qubits
        macro_gates: list[Gate] = [
            H(target),
            CNOT((control2, target)),
            Tdg(target),
            CNOT((control1, target)),
            T(target),
            CNOT((control2, target)),
            Tdg(target),
            CNOT((control1, target)),
            T(control2),
            T(target),
            H(target),
            CNOT((control1, control2)),
            T(control1),
            Tdg(control2),
            CNOT((control1, control2)),
        ]
        unit_gates: list[UnitGate] = []
        for macro_gate in macro_gates:
            unit_gates.extend(macro_gate.unit_gates())
        return unit_gates

    @typing_extensions.override
    def matrix(self) -> NDArray[np.complex128]:
        r"""Get the `matrix` representation of the gate.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            Matrix representation of the gate.
        """
        return np.asarray(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ],
            dtype=np.complex128,
        )


@dataclass(frozen=True)
class CCZ(MultiGate):
    r"""Class for the CCZ gate.

    Attributes
    ----------
    qubits : `list`\[`int`\]
        The qubits the gate acts on [control1, control2, target].

        .. math::

            CCZ = \begin{pmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & -1
            \end{pmatrix}

    """

    qubits: list[int]

    @typing_extensions.override
    def unit_gates(self) -> list[UnitGate]:
        r"""Get the `unit_gates` that make up the gate.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gates that make up the gate.
        """
        control1, control2, target = self.qubits
        macro_gates: list[Gate] = [
            H(target),
            Toffoli([control1, control2, target]),
            H(target),
        ]
        unit_gates: list[UnitGate] = []
        for macro_gate in macro_gates:
            unit_gates.extend(macro_gate.unit_gates())
        return unit_gates

    @typing_extensions.override
    def matrix(self) -> NDArray[np.complex128]:
        r"""Get the `matrix` representation of the gate.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            Matrix representation of the gate.
        """
        return np.asarray(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, -1],
            ],
            dtype=np.complex128,
        )
