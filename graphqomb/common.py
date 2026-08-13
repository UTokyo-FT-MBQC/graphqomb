"""Common classes and functions.

This module provides:

- `Plane`: Measurement planes for the MBQC.
- `Axis`: Measurement axis.
- `Sign`: Measurement sign.
- `Initialization`: Input-node initialization (Pauli axis and Stim-style tag).
- `MeasBasis`: Abstract class to represent a measurement basis.
- `PlannerMeasBasis`: Class to represent a planner measurement basis.
- `AxisMeasBasis`: Class to represent an axis measurement basis.
- `is_close_angle`: Check if an angle is close to a target angle.
- `is_clifford_angle`: Check if an angle is a Clifford angle.
- `determine_pauli_axis`: Function to determine Pauli axis for a measurement basis.
- `default_meas_basis`: Function to return the default measurement basis.
- `meas_basis`: Function to get the measurement basis vector.
"""

from __future__ import annotations

import abc
import cmath
import dataclasses
import enum
import math
from abc import ABC
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import typing_extensions

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Plane(Enum):
    """Measurement planes for MBQC.

    We distinguish the axial measurements from the planar measurements.
    """

    XY = enum.auto()
    YZ = enum.auto()
    XZ = enum.auto()


class Axis(Enum):
    """Measurement axis for Pauli measurement."""

    X = enum.auto()
    Y = enum.auto()
    Z = enum.auto()


class Sign(Enum):
    """Measurement sign for Pauli measurement."""

    PLUS = enum.auto()
    MINUS = enum.auto()


@dataclasses.dataclass(frozen=True)
class Initialization:
    """Input-node initialization: a positive Pauli eigenstate with an optional tag.

    Attributes
    ----------
    axis : `Axis`
        Pauli axis of the prepared positive eigenstate, by default Axis.X.
    tag : `str`
        Stim-style instruction tag of the initialization, by default ""
        (untagged).

    Raises
    ------
    TypeError
        If axis is not an `Axis` value or tag is not a `str`.
    """

    axis: Axis = Axis.X
    tag: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.axis, Axis):
            msg = "Initialization axis must be one of Axis.X, Axis.Y, Axis.Z"
            raise TypeError(msg)
        if not isinstance(self.tag, str):
            msg = "Initialization tag must be a str"
            raise TypeError(msg)


class MeasBasis(ABC):
    """Abstract class to represent a measurement basis."""

    @property
    @abc.abstractmethod
    def plane(self) -> Plane:
        """The measurement plane."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def angle(self) -> float:
        """The measurement angle."""
        raise NotImplementedError

    @abc.abstractmethod
    def flip(self) -> MeasBasis:
        """Flip the measurement basis."""
        raise NotImplementedError

    @abc.abstractmethod
    def conjugate(self) -> MeasBasis:
        """Return the angle-conjugated measurement basis."""
        raise NotImplementedError

    @abc.abstractmethod
    def vector(self) -> NDArray[np.complex128]:
        """Return the measurement basis vector."""
        raise NotImplementedError


class PlannerMeasBasis(MeasBasis):
    """Class to represent a planner measurement basis."""

    def __init__(self, plane: Plane, angle: float) -> None:
        self.__plane = plane
        self.__angle = angle

    @property
    @typing_extensions.override
    def plane(self) -> Plane:
        """The measurement plane.

        Returns
        -------
        `Plane`
            measurement plane
        """
        return self.__plane

    @property
    @typing_extensions.override
    def angle(self) -> float:
        """The measurement angle.

        Returns
        -------
        `float`
            measurement angle
        """
        return self.__angle

    @typing_extensions.override
    def flip(self) -> PlannerMeasBasis:
        """Flip the measurement basis.

        Returns
        -------
        `PlannerMeasBasis`
            flipped PlannerMeasBasis
        """
        return PlannerMeasBasis(self.plane, self.angle + math.pi)

    @typing_extensions.override
    def conjugate(self) -> PlannerMeasBasis:
        """Return the angle-conjugated PlannerMeasBasis object.

        This operation represents measurement-angle sign inversion. It is not
        defined as complex conjugation of the basis vector.

        Returns
        -------
        `PlannerMeasBasis`
            angle-conjugated PlannerMeasBasis

        Raises
        ------
        TypeError
            if the plane is not one of XY, YZ, XZ
        """
        if not isinstance(self.plane, Plane):
            msg = "The plane must be one of XY, YZ, XZ"
            raise TypeError(msg)
        return PlannerMeasBasis(self.plane, -self.angle)

    @typing_extensions.override
    def vector(self) -> NDArray[np.complex128]:
        r"""Return the measurement basis vector.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            measurement basis vector
        """
        return meas_basis(self.plane, self.angle)


class AxisMeasBasis(MeasBasis):
    """Class to represent an axis measurement basis.

    The aim is to pocess the accurate information of the axis measurement.

    Attributes
    ----------
    axis : `Axis`
        measurement axis
    sign : `Sign`
        measurement sign
    """

    def __init__(self, axis: Axis, sign: Sign) -> None:
        self.axis = axis
        self.sign = sign

    @property
    @typing_extensions.override
    def plane(self) -> Plane:
        """The measurement plane.

        Returns
        -------
        `Plane`
            measurement plane

        Raises
        ------
        TypeError
            if the axis is not one of X, Y, Z
        """
        if not isinstance(self.axis, Axis):
            msg = "The axis must be one of X, Y, Z"
            raise TypeError(msg)
        if self.axis == Axis.X:
            plane = Plane.XY
        elif self.axis == Axis.Y:
            plane = Plane.YZ
        elif self.axis == Axis.Z:
            plane = Plane.XZ
        else:
            typing_extensions.assert_never(self.axis)
        return plane

    @property
    @typing_extensions.override
    def angle(self) -> float:
        """The measurement angle.

        Returns
        -------
        `float`
            measurement angle

        Raises
        ------
        TypeError
            if the axis is not one of X, Y,
        """
        if not isinstance(self.axis, Axis):
            msg = "The axis must be one of X, Y, Z"
            raise TypeError(msg)
        if self.axis == Axis.Y:
            angle = math.pi / 2 if self.sign == Sign.PLUS else 3 * math.pi / 2
        else:
            angle = 0 if self.sign == Sign.PLUS else math.pi
        return angle

    @typing_extensions.override
    def flip(self) -> AxisMeasBasis:
        """Flip the measurement basis.

        Returns
        -------
        `AxisMeasBasis`
            flipped AxisMeasBasis
        """
        return AxisMeasBasis(self.axis, Sign.MINUS if self.sign == Sign.PLUS else Sign.PLUS)

    @typing_extensions.override
    def conjugate(self) -> AxisMeasBasis:
        """Return the conjugate of the AxisMeasBasis object.

        Returns
        -------
        `AxisMeasBasis`
            conjugate AxisMeasBasis
        """
        if self.axis == Axis.Y:
            return AxisMeasBasis(Axis.Y, Sign.MINUS if self.sign == Sign.PLUS else Sign.PLUS)
        return AxisMeasBasis(self.axis, self.sign)

    @typing_extensions.override
    def vector(self) -> NDArray[np.complex128]:
        r"""Return the measurement basis vector.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            measurement basis vector
        """
        return meas_basis(self.plane, self.angle)


def is_close_angle(angle: float, target: float, atol: float = 1e-9) -> bool:
    """Check if an angle is close to a target angle.

    Parameters
    ----------
    angle : `float`
        angle to check
    target : `float`
        target angle
    atol : `float`, optional
        absolute tolerance, by default 1e-9

    Returns
    -------
    `bool`
        `True` if the angle is close to the target angle
    """
    diff_angle = (angle - target) % (2 * math.pi)

    if diff_angle > math.pi:
        diff_angle = 2 * math.pi - diff_angle
    return bool(np.isclose(diff_angle, 0, atol=atol))


def is_clifford_angle(angle: float, atol: float = 1e-9) -> bool:
    """Check if an angle is a Clifford angle.

    Parameters
    ----------
    angle : `float`
        angle to check
    atol : `float`, optional
        absolute tolerance, by default 1e-9

    Returns
    -------
    `bool`
        `True` if the angle is a Clifford angle
    """
    angle_preprocessed = angle % (2 * math.pi)
    return any(
        is_close_angle(angle_preprocessed, target, atol=atol) for target in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)
    )


def determine_pauli_axis(meas_bases: MeasBasis) -> Axis | None:
    """Determine Pauli axis for a measurement basis if it's a Pauli measurement.

    Parameters
    ----------
    meas_bases : `MeasBasis`
        Measurement basis to check

    Returns
    -------
    `Axis` | None
        Pauli axis if this is a Pauli measurement, None otherwise
    """
    angle = meas_bases.angle

    if not is_clifford_angle(angle):
        return None

    if is_close_angle(angle, 0) or is_close_angle(angle, math.pi):
        # X measurement (XY plane, θ=0 or π) or Z measurement (XZ plane, θ=0 or π)
        if meas_bases.plane == Plane.XY:
            return Axis.X
        if meas_bases.plane in {Plane.XZ, Plane.YZ}:
            return Axis.Z
    elif is_close_angle(angle, math.pi / 2) or is_close_angle(angle, 3 * math.pi / 2):
        # Y measurement can occur in XY plane (θ=π/2 or 3π/2) or YZ plane (θ=π/2 or 3π/2)
        if meas_bases.plane in {Plane.XY, Plane.YZ}:
            return Axis.Y
        if meas_bases.plane == Plane.XZ:
            return Axis.X

    return None


def default_meas_basis() -> PlannerMeasBasis:
    """Return the default measurement basis.

    The default measurement basis is the XY plane with angle 0.

    Returns
    -------
    `PlannerMeasBasis`
        default measurement basis
    """
    return PlannerMeasBasis(Plane.XY, 0.0)


def meas_basis(plane: Plane, angle: float) -> NDArray[np.complex128]:
    r"""Return the measurement basis vector corresponding to the plane and angle.

    Parameters
    ----------
    plane : `Plane`
        measurement plane
    angle : `float`
        measurement angle

    Returns
    -------
    `numpy.typing.NDArray`\[`numpy.complex128`\]
        measurement basis vector

    Raises
    ------
    TypeError
        if the plane is not one of XY, YZ, XZ
    """
    if not isinstance(plane, Plane):
        msg = "The plane must be one of XY, YZ, XZ"
        raise TypeError(msg)
    if plane == Plane.XY:
        basis = np.asarray([1, cmath.exp(1j * angle)]) / math.sqrt(2)
    elif plane == Plane.YZ:
        basis = np.asarray([math.cos(angle / 2), 1j * math.sin(angle / 2)])
    elif plane == Plane.XZ:
        basis = np.asarray([math.cos(angle / 2), math.sin(angle / 2)])
    else:
        typing_extensions.assert_never(plane)
    return basis.astype(np.complex128)
