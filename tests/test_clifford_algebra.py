"""Tests for the exact single-qubit Clifford algebra."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from graphqomb import clifford_algebra as ca
from graphqomb.common import Axis, Plane, PlannerMeasBasis, meas_basis
from graphqomb.euler import LocalClifford, update_lc_basis

_PAULI_MATRICES = {
    Axis.X: np.asarray([[0, 1], [1, 0]], dtype=np.complex128),
    Axis.Y: np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128),
    Axis.Z: np.asarray([[1, 0], [0, -1]], dtype=np.complex128),
}

_ANGLES = (0.0, 0.3, 1.1, -0.7, math.pi / 2, 2.0, math.pi)


def _all_elements() -> list[ca.C1Element]:
    return [
        ca.C1Element(x_axis, x_sign, z_axis, z_sign)
        for x_axis, z_axis in itertools.permutations((Axis.X, Axis.Y, Axis.Z), 2)
        for x_sign in (1, -1)
        for z_sign in (1, -1)
    ]


def _proportional(a: np.ndarray, b: np.ndarray) -> bool:
    """Return whether two 2x2 unitaries are equal up to global phase."""
    return bool(np.isclose(abs(np.trace(a.conj().T @ b)), 2.0))


def test_group_order() -> None:
    elements = _all_elements()
    assert len(elements) == 24
    assert len(set(elements)) == 24


def test_s_and_h_generate_the_group() -> None:
    generated = {ca.IDENTITY}
    frontier = [ca.IDENTITY]
    while frontier:
        current = frontier.pop()
        for generator in (ca.S, ca.H):
            product = ca.compose(generator, current)
            if product not in generated:
                generated.add(product)
                frontier.append(product)
    assert generated == set(_all_elements())


def test_closure() -> None:
    elements = set(_all_elements())
    for a, b in itertools.product(elements, repeat=2):
        assert ca.compose(a, b) in elements


def test_associativity() -> None:
    elements = _all_elements()
    for a, b, c in itertools.product(elements, repeat=3):
        assert ca.compose(a, ca.compose(b, c)) == ca.compose(ca.compose(a, b), c)


def test_inverse() -> None:
    for element in _all_elements():
        assert ca.compose(element, ca.inverse(element)) == ca.IDENTITY
        assert ca.compose(ca.inverse(element), element) == ca.IDENTITY


def test_pauli_constants_match_matrices() -> None:
    for element, matrix in (
        (ca.X, _PAULI_MATRICES[Axis.X]),
        (ca.Y, _PAULI_MATRICES[Axis.Y]),
        (ca.Z, _PAULI_MATRICES[Axis.Z]),
    ):
        assert _proportional(ca.to_matrix(element), matrix)


def test_decompose_recompose_identity() -> None:
    for element in _all_elements():
        coset, x_bit, z_bit = ca.decompose(element)
        assert coset in ca.TRANSVERSAL
        rebuilt = coset
        if x_bit:
            rebuilt = ca.compose(rebuilt, ca.X)
        if z_bit:
            rebuilt = ca.compose(rebuilt, ca.Z)
        assert rebuilt == element


def test_compose_is_matrix_product_up_to_phase() -> None:
    for a, b in itertools.product(_all_elements(), repeat=2):
        assert _proportional(ca.to_matrix(ca.compose(a, b)), ca.to_matrix(a) @ ca.to_matrix(b))


def test_act_on_axis_matches_matrix_conjugation() -> None:
    for element in _all_elements():
        matrix = ca.to_matrix(element)
        for axis in (Axis.X, Axis.Y, Axis.Z):
            image_axis, sign = ca.act_on_axis(element, axis)
            conjugated = matrix @ _PAULI_MATRICES[axis] @ matrix.conj().T
            np.testing.assert_allclose(conjugated, sign * _PAULI_MATRICES[image_axis], atol=1e-12)


def test_act_on_plane_angle_matches_matrix_action() -> None:
    for element in _all_elements():
        matrix = ca.to_matrix(element)
        for plane in Plane:
            new_plane, eps, quarter_turns = ca.act_on_plane_angle(element, plane)
            assert eps in {-1, 1}
            assert quarter_turns in {0, 1, 2, 3}
            for angle in _ANGLES:
                transformed = matrix @ meas_basis(plane, angle)
                expected = meas_basis(new_plane, eps * angle + quarter_turns * math.pi / 2)
                assert np.isclose(abs(np.vdot(expected, transformed)), 1.0)


@pytest.mark.parametrize(
    ("element", "plane", "expected"),
    [
        # Pauli rows must reproduce PatternSimulator._updated_measurement_basis exactly.
        (ca.X, Plane.XY, (Plane.XY, -1, 0)),
        (ca.Z, Plane.XY, (Plane.XY, 1, 2)),
        (ca.Y, Plane.XY, (Plane.XY, -1, 2)),
        (ca.X, Plane.YZ, (Plane.YZ, 1, 2)),
        (ca.Z, Plane.YZ, (Plane.YZ, -1, 0)),
        (ca.Y, Plane.YZ, (Plane.YZ, -1, 2)),
        (ca.X, Plane.XZ, (Plane.XZ, -1, 2)),
        (ca.Z, Plane.XZ, (Plane.XZ, -1, 0)),
        (ca.Y, Plane.XZ, (Plane.XZ, 1, 2)),
        # Convention anchors for the coset generators.
        (ca.S, Plane.XY, (Plane.XY, 1, 1)),
        (ca.S, Plane.YZ, (Plane.XZ, -1, 0)),
        (ca.S, Plane.XZ, (Plane.YZ, 1, 0)),
        (ca.H, Plane.XY, (Plane.YZ, -1, 0)),
        (ca.H, Plane.YZ, (Plane.XY, -1, 0)),
        (ca.H, Plane.XZ, (Plane.XZ, -1, 1)),
    ],
)
def test_act_on_plane_angle_table(element: ca.C1Element, plane: Plane, expected: tuple[Plane, int, int]) -> None:
    assert ca.act_on_plane_angle(element, plane) == expected


def test_is_z_axis_preserving_is_the_s_pauli_subgroup() -> None:
    preserving = {element for element in _all_elements() if ca.is_z_axis_preserving(element)}
    assert len(preserving) == 8
    s_pauli_subgroup = {
        ca.compose(s_power, pauli)
        for s_power in (ca.IDENTITY, ca.S, ca.Z, ca.compose(ca.S, ca.Z))
        for pauli in (ca.IDENTITY, ca.X)
    }
    assert preserving == s_pauli_subgroup
    assert not ca.is_z_axis_preserving(ca.H)
    assert not ca.is_z_axis_preserving(ca.HSH)


def test_from_matrix_roundtrip() -> None:
    for element in _all_elements():
        assert ca.from_matrix(ca.to_matrix(element)) == element
        # Global phase and scale must not matter.
        assert ca.from_matrix(1.7j * ca.to_matrix(element)) == element


def test_from_matrix_rejects_non_clifford() -> None:
    t_matrix = np.diag([1.0, np.exp(1j * math.pi / 4)]).astype(np.complex128)
    with pytest.raises(ValueError, match="not proportional to a single-qubit Clifford"):
        ca.from_matrix(t_matrix)
    with pytest.raises(ValueError, match="singular"):
        ca.from_matrix(np.zeros((2, 2), dtype=np.complex128))


def test_from_local_clifford() -> None:
    assert ca.from_local_clifford(LocalClifford(0, 0, math.pi / 2)) == ca.S
    assert ca.from_local_clifford(LocalClifford(math.pi / 2, math.pi / 2, math.pi / 2)) == ca.H
    assert ca.from_local_clifford(LocalClifford(0, math.pi, 0)) == ca.X
    assert ca.from_local_clifford(LocalClifford(0, 0, math.pi)) == ca.Z


def test_act_on_plane_angle_matches_update_lc_basis() -> None:
    lc_angles = (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)
    for alpha, beta, gamma in itertools.product(lc_angles, repeat=3):
        lc = LocalClifford(alpha, beta, gamma)
        element = ca.from_local_clifford(lc)
        for plane in Plane:
            for angle in (0.3, 1.1):
                reference = update_lc_basis(lc, PlannerMeasBasis(plane, angle))
                new_plane, eps, quarter_turns = ca.act_on_plane_angle(element, plane)
                assert new_plane == reference.plane
                derived = meas_basis(new_plane, eps * angle + quarter_turns * math.pi / 2)
                assert np.isclose(abs(np.vdot(derived, reference.vector())), 1.0)


def test_invalid_elements_rejected() -> None:
    with pytest.raises(ValueError, match="distinct"):
        ca.C1Element(Axis.X, 1, Axis.X, 1)
    with pytest.raises(ValueError, match="sign"):
        ca.C1Element(Axis.X, 2, Axis.Z, 1)
