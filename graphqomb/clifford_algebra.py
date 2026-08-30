"""Exact algebra of the single-qubit Clifford group modulo global phase.

Each of the 24 elements is represented as a signed permutation of the Bloch
axes: the images of the X and Z axes under conjugation (the Y image follows
from right-handedness).  All group operations are exact; floating point only
appears in the matrix interop helpers.

This module provides:

- `C1Element`: A single-qubit Clifford (mod phase) as a signed axis permutation.
- `compose`: Compose two elements with the matrix-product convention.
- `inverse`: Invert an element.
- `decompose`: Split an element as ``coset * X^a * Z^b`` with the coset in `TRANSVERSAL`.
- `act_on_axis`: Image of a Pauli axis under conjugation.
- `act_on_plane_angle`: Action on measurement-basis labels ``(plane, angle)``.
- `is_z_axis_preserving`: Whether the element maps the Z axis to itself up to sign.
- `to_matrix`: A 2x2 unitary representative (arbitrary phase).
- `from_matrix`: Construct an element from a 2x2 unitary.
- `from_local_clifford`: Construct an element from a `graphqomb.euler.LocalClifford`.
- Constants: `IDENTITY`, `X`, `Y`, `Z`, `S`, `H`, `SH`, `HS`, `HSH`, `TRANSVERSAL`, `COSET_NAMES`.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import operator
from typing import TYPE_CHECKING

import numpy as np

from graphqomb.common import Axis, Plane

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from graphqomb.euler import LocalClifford

_AXES = (Axis.X, Axis.Y, Axis.Z)
_CYCLIC_PAIRS = frozenset({(Axis.X, Axis.Y), (Axis.Y, Axis.Z), (Axis.Z, Axis.X)})


def _third_axis(axis1: Axis, axis2: Axis) -> Axis:
    (axis,) = set(_AXES) - {axis1, axis2}
    return axis


def _cyclic_sign(axis1: Axis, axis2: Axis) -> int:
    return 1 if (axis1, axis2) in _CYCLIC_PAIRS else -1


@dataclasses.dataclass(frozen=True)
class C1Element:
    """Single-qubit Clifford modulo phase, as the signed images of the X and Z axes.

    ``c`` maps the X axis to ``x_sign * x_axis`` and the Z axis to
    ``z_sign * z_axis`` under conjugation; the Y image is derived.

    Attributes
    ----------
    x_axis : `Axis`
        Image axis of X under conjugation.
    x_sign : `int`
        Sign (+1 or -1) of the X image.
    z_axis : `Axis`
        Image axis of Z under conjugation.
    z_sign : `int`
        Sign (+1 or -1) of the Z image.
    """

    x_axis: Axis
    x_sign: int
    z_axis: Axis
    z_sign: int

    def __post_init__(self) -> None:
        if self.x_axis == self.z_axis:
            msg = "The images of the X and Z axes must be distinct."
            raise ValueError(msg)
        if self.x_sign not in {-1, 1} or self.z_sign not in {-1, 1}:
            msg = "Axis image signs must be +1 or -1."
            raise ValueError(msg)


def act_on_axis(c: C1Element, axis: Axis) -> tuple[Axis, int]:
    r"""Return the image of a Pauli axis under conjugation by ``c``.

    ``c P_axis c^\dagger = sign * P_new_axis``.

    Parameters
    ----------
    c : `C1Element`
        The Clifford element.
    axis : `Axis`
        The Pauli axis to conjugate.

    Returns
    -------
    `tuple`\[`Axis`, `int`\]
        Image axis and its sign.
    """
    if axis == Axis.X:
        return c.x_axis, c.x_sign
    if axis == Axis.Z:
        return c.z_axis, c.z_sign
    # Y = i X Z, so c Y c^t = x_sign * z_sign * (i P_x P_z) with
    # i P_a P_b = -cyclic_sign(a, b) P_third for orthogonal axes a, b.
    return _third_axis(c.x_axis, c.z_axis), -c.x_sign * c.z_sign * _cyclic_sign(c.x_axis, c.z_axis)


def compose(a: C1Element, b: C1Element) -> C1Element:
    """Compose two elements: apply ``b`` first, then ``a`` (matrix product ``a @ b``).

    Parameters
    ----------
    a : `C1Element`
        The element applied second (left matrix factor).
    b : `C1Element`
        The element applied first (right matrix factor).

    Returns
    -------
    `C1Element`
        The product ``a * b`` modulo phase.
    """
    x_axis1, x_sign1 = act_on_axis(b, Axis.X)
    x_axis2, x_sign2 = act_on_axis(a, x_axis1)
    z_axis1, z_sign1 = act_on_axis(b, Axis.Z)
    z_axis2, z_sign2 = act_on_axis(a, z_axis1)
    return C1Element(x_axis2, x_sign1 * x_sign2, z_axis2, z_sign1 * z_sign2)


def inverse(c: C1Element) -> C1Element:
    """Return the inverse element.

    Parameters
    ----------
    c : `C1Element`
        The element to invert.

    Returns
    -------
    `C1Element`
        The inverse of ``c`` modulo phase.
    """
    preimages: dict[Axis, tuple[Axis, int]] = {}
    for axis in _AXES:
        image_axis, sign = act_on_axis(c, axis)
        preimages[image_axis] = (axis, sign)
    x_axis, x_sign = preimages[Axis.X]
    z_axis, z_sign = preimages[Axis.Z]
    return C1Element(x_axis, x_sign, z_axis, z_sign)


IDENTITY = C1Element(Axis.X, 1, Axis.Z, 1)
X = C1Element(Axis.X, 1, Axis.Z, -1)
Y = C1Element(Axis.X, -1, Axis.Z, -1)
Z = C1Element(Axis.X, -1, Axis.Z, 1)
S = C1Element(Axis.Y, 1, Axis.Z, 1)
H = C1Element(Axis.Z, 1, Axis.X, 1)
SH = compose(S, H)
HS = compose(H, S)
HSH = compose(H, compose(S, H))

#: Transversal of the Pauli subgroup: C = coset * X^a * Z^b with coset drawn from here.
TRANSVERSAL = (IDENTITY, S, H, SH, HS, HSH)

#: Names of the transversal elements, used by the .ptn format.
COSET_NAMES: dict[C1Element, str] = {
    IDENTITY: "I",
    S: "S",
    H: "H",
    SH: "SH",
    HS: "HS",
    HSH: "HSH",
}

_COSET_BY_PERMUTATION: dict[tuple[Axis, Axis], C1Element] = {(d.x_axis, d.z_axis): d for d in TRANSVERSAL}


def decompose(c: C1Element) -> tuple[C1Element, bool, bool]:
    r"""Split ``c = coset * X^a * Z^b`` with the coset in `TRANSVERSAL`.

    Parameters
    ----------
    c : `C1Element`
        The element to decompose.

    Returns
    -------
    `tuple`\[`C1Element`, `bool`, `bool`\]
        The coset representative and the Pauli bits ``(a, b)``.
    """
    coset = _COSET_BY_PERMUTATION[c.x_axis, c.z_axis]
    residue = compose(inverse(coset), c)
    # The residue permutes no axes, so it is I, X, Y or Z: X flips the Z-image
    # sign and Z flips the X-image sign.
    return coset, residue.z_sign < 0, residue.x_sign < 0


# (cos-axis, sin-axis) of each measurement plane, matching `common.meas_basis`:
# the Bloch vector of (plane, theta) is cos(theta)*cos_axis + sin(theta)*sin_axis.
_PLANE_AXES: dict[Plane, tuple[Axis, Axis]] = {
    Plane.XY: (Axis.X, Axis.Y),
    Plane.YZ: (Axis.Z, Axis.Y),
    Plane.XZ: (Axis.Z, Axis.X),
}
_PLANE_BY_AXES: dict[frozenset[Axis], Plane] = {
    frozenset({Axis.X, Axis.Y}): Plane.XY,
    frozenset({Axis.Y, Axis.Z}): Plane.YZ,
    frozenset({Axis.X, Axis.Z}): Plane.XZ,
}


def act_on_plane_angle(c: C1Element, plane: Plane) -> tuple[Plane, int, int]:
    r"""Return how ``c`` transforms measurement-basis labels on ``plane``.

    The basis vector satisfies ``c |plane, theta> = |new_plane, eps*theta + k*pi/2>``
    up to phase, for the parameterizations of `graphqomb.common.meas_basis`.

    Parameters
    ----------
    c : `C1Element`
        The Clifford element applied to the basis vector.
    plane : `Plane`
        The nominal measurement plane.

    Returns
    -------
    `tuple`\[`Plane`, `int`, `int`\]
        ``(new_plane, eps, k)`` with ``eps`` in {-1, +1} and ``k`` in {0, 1, 2, 3}.
    """
    cos_axis, sin_axis = _PLANE_AXES[plane]
    cos_image, cos_sign = act_on_axis(c, cos_axis)
    sin_image, sin_sign = act_on_axis(c, sin_axis)
    new_plane = _PLANE_BY_AXES[frozenset({cos_image, sin_image})]
    new_cos_axis, _ = _PLANE_AXES[new_plane]
    if cos_image == new_cos_axis:
        quarter_turns = 0 if cos_sign > 0 else 2
        eps = cos_sign * sin_sign
    else:
        quarter_turns = 1 if cos_sign > 0 else 3
        eps = -cos_sign * sin_sign
    return new_plane, eps, quarter_turns


def is_z_axis_preserving(c: C1Element) -> bool:
    """Return whether ``c`` maps the Z axis to itself up to sign.

    True exactly for the 8 elements generated by S and the Paulis.

    Parameters
    ----------
    c : `C1Element`
        The element to check.

    Returns
    -------
    `bool`
        Whether the Z-axis image of ``c`` is +Z or -Z.
    """
    return c.z_axis == Axis.Z


_MATRIX_IDENTITY = np.eye(2, dtype=np.complex128)
_MATRIX_X = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
_MATRIX_Y = np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128)
_MATRIX_Z = np.asarray([[1, 0], [0, -1]], dtype=np.complex128)
_MATRIX_S = np.asarray([[1, 0], [0, 1j]], dtype=np.complex128)
_MATRIX_H = np.asarray([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
_COSET_MATRICES: dict[C1Element, NDArray[np.complex128]] = {
    IDENTITY: _MATRIX_IDENTITY,
    S: _MATRIX_S,
    H: _MATRIX_H,
    SH: _MATRIX_S @ _MATRIX_H,
    HS: _MATRIX_H @ _MATRIX_S,
    HSH: _MATRIX_H @ _MATRIX_S @ _MATRIX_H,
}
_AXIS_MATRICES: dict[Axis, NDArray[np.complex128]] = {Axis.X: _MATRIX_X, Axis.Y: _MATRIX_Y, Axis.Z: _MATRIX_Z}


def to_matrix(c: C1Element) -> NDArray[np.complex128]:
    r"""Return a 2x2 unitary representative of ``c`` (phase is arbitrary).

    Parameters
    ----------
    c : `C1Element`
        The element to represent.

    Returns
    -------
    `numpy.typing.NDArray`\[`numpy.complex128`\]
        A 2x2 unitary matrix.
    """
    coset, x_bit, z_bit = decompose(c)
    factors = [_COSET_MATRICES[coset]]
    if x_bit:
        factors.append(_MATRIX_X)
    if z_bit:
        factors.append(_MATRIX_Z)
    return np.asarray(functools.reduce(operator.matmul, factors), dtype=np.complex128)


def from_matrix(matrix: NDArray[np.complex128]) -> C1Element:
    r"""Construct an element from a 2x2 unitary (up to scale and phase).

    Parameters
    ----------
    matrix : `numpy.typing.NDArray`\[`numpy.complex128`\]
        A 2x2 matrix proportional to a single-qubit Clifford unitary.

    Returns
    -------
    `C1Element`
        The corresponding group element.

    Raises
    ------
    ValueError
        If the matrix is not proportional to a single-qubit Clifford unitary.
    """
    determinant = complex(np.linalg.det(matrix))
    if np.isclose(determinant, 0):
        msg = "The matrix is singular; expected a unitary matrix."
        raise ValueError(msg)
    unitary = matrix / math.sqrt(abs(determinant))

    images: dict[Axis, tuple[Axis, int]] = {}
    for axis in (Axis.X, Axis.Z):
        conjugated = unitary @ _AXIS_MATRICES[axis] @ unitary.conj().T
        for image_axis, image_matrix in _AXIS_MATRICES.items():
            if np.allclose(conjugated, image_matrix):
                images[axis] = (image_axis, 1)
                break
            if np.allclose(conjugated, -image_matrix):
                images[axis] = (image_axis, -1)
                break
        else:
            msg = "The matrix is not proportional to a single-qubit Clifford unitary."
            raise ValueError(msg)
    x_axis, x_sign = images[Axis.X]
    z_axis, z_sign = images[Axis.Z]
    return C1Element(x_axis, x_sign, z_axis, z_sign)


def from_local_clifford(lc: LocalClifford) -> C1Element:
    """Construct an element from a `graphqomb.euler.LocalClifford`.

    Parameters
    ----------
    lc : `LocalClifford`
        The local Clifford to convert.

    Returns
    -------
    `C1Element`
        The corresponding group element modulo phase.
    """
    return from_matrix(lc.matrix())
