from __future__ import annotations

import math

import pytest

from graphqomb.common import Axis, AxisMeasBasis, Plane, PlannerMeasBasis, Sign, is_clifford_angle, is_close_angle


def test_inverse_order_plane() -> None:
    with pytest.raises(AttributeError):
        _ = PlannerMeasBasis(Plane.YX, 0)  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    with pytest.raises(AttributeError):
        _ = PlannerMeasBasis(Plane.ZX, 0)  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    with pytest.raises(AttributeError):
        _ = PlannerMeasBasis(Plane.ZY, 0)  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]


def test_is_clifford_angle() -> None:
    assert is_clifford_angle(0)
    assert is_clifford_angle(math.pi / 2)
    assert is_clifford_angle(math.pi)
    assert is_clifford_angle(3 * math.pi / 2)
    assert not is_clifford_angle(math.pi / 3)
    assert not is_clifford_angle(math.pi / 4)
    assert not is_clifford_angle(math.pi / 6)


def test_is_close_angle() -> None:
    assert is_close_angle(0, 0)
    assert is_close_angle(math.pi / 2, math.pi / 2)
    assert is_close_angle(math.pi, math.pi)
    assert is_close_angle(3 * math.pi / 2, 3 * math.pi / 2)
    assert not is_close_angle(0, math.pi / 2)
    assert not is_close_angle(math.pi / 2, math.pi)
    assert not is_close_angle(math.pi, 3 * math.pi / 2)
    assert not is_close_angle(3 * math.pi / 2, 0)

    # add 2 * math.pi to the second angle
    assert is_close_angle(0, 2 * math.pi)
    assert is_close_angle(math.pi / 2, 2 * math.pi + math.pi / 2)
    assert is_close_angle(math.pi, 2 * math.pi + math.pi)
    assert is_close_angle(3 * math.pi / 2, 2 * math.pi + 3 * math.pi / 2)

    # minus 2 * math.pi to the second angle
    assert is_close_angle(0, -2 * math.pi)
    assert is_close_angle(math.pi / 2, -2 * math.pi + math.pi / 2)
    assert is_close_angle(math.pi, -2 * math.pi + math.pi)
    assert is_close_angle(3 * math.pi / 2, -2 * math.pi + 3 * math.pi / 2)

    # boundary cases
    assert is_close_angle(-1e-10, 1e-10)


@pytest.mark.parametrize("plane", list(Plane))
def test_planner_meas_basis_conjugate_negates_angle(plane: Plane) -> None:
    basis = PlannerMeasBasis(plane, 0.3)

    conjugated = basis.conjugate()

    assert conjugated.plane == plane
    assert is_close_angle(conjugated.angle, -0.3)


@pytest.mark.parametrize("axis", list(Axis))
@pytest.mark.parametrize("sign", list(Sign))
def test_axis_meas_basis_conjugate_negates_angle(axis: Axis, sign: Sign) -> None:
    basis = AxisMeasBasis(axis, sign)

    conjugated = basis.conjugate()

    expected_sign = sign
    if axis == Axis.Y:
        expected_sign = Sign.MINUS if sign == Sign.PLUS else Sign.PLUS

    assert conjugated.axis == axis
    assert conjugated.sign == expected_sign
    assert is_close_angle(conjugated.angle, -basis.angle)


def test_is_close_angle_rejects_non_finite_angles() -> None:
    assert not is_close_angle(math.nan, 0)
    assert not is_close_angle(math.inf, 0)
    assert not is_close_angle(0, -math.inf)


def test_is_close_angle_tolerance_boundary() -> None:
    assert is_close_angle(1e-10, 0)
    assert is_close_angle(2 * math.pi - 1e-10, 0)
    assert not is_close_angle(2e-9, 0, atol=1e-9)
    assert is_close_angle(2e-9, 0, atol=5e-9)
    assert isinstance(is_close_angle(0, 0), bool)
