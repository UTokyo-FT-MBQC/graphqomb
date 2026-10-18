"""Tests for QEC graph-state builders."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from scipy.sparse import csr_array

from graphqomb import qeccode
from graphqomb.common import Axis, AxisMeasBasis, Sign
from graphqomb.graphstate import GraphState
from graphqomb.qeccode import StabilizerCode, YFoliation, build_graph_state

if TYPE_CHECKING:
    from collections.abc import Sequence


def _matrix(data: list[list[int]]) -> csr_array[Any, tuple[int, int]]:
    return csr_array(data)


def _assert_axis_meas_basis(meas_basis: object, axis: Axis) -> None:
    assert isinstance(meas_basis, AxisMeasBasis)
    assert meas_basis.axis == axis
    assert meas_basis.sign == Sign.PLUS
    expected_angle = math.pi / 2 if axis == Axis.Y else 0.0
    assert math.isclose(meas_basis.angle, expected_angle)


def test_build_graph_state_connects_stabilizer_supports() -> None:
    matrix = _matrix(
        [
            [0, 0, 0, 1, 0, 2],  # Z support on q0 and q2.
            [0, 3, 0, 0, 0, 0],  # X support on q1.
            [1, 0, 0, 1, 0, 0],  # X and Z support on q0.
        ]
    )
    code = StabilizerCode(matrix)

    result = build_graph_state(code)
    graph = result.graph

    for qubit in range(3):
        assert graph.has_edge(result.data_nodes[qubit, 0], result.data_nodes[qubit, 1])

    ancilla0 = result.ancilla_nodes[0]
    assert graph.has_edge(ancilla0, result.data_nodes[0, 0])
    assert graph.has_edge(ancilla0, result.data_nodes[2, 0])

    ancilla1 = result.ancilla_nodes[1]
    assert graph.has_edge(ancilla1, result.data_nodes[1, 1])

    ancilla2 = result.ancilla_nodes[2]
    assert graph.has_edge(ancilla2, result.data_nodes[0, 0])
    assert graph.has_edge(ancilla2, result.data_nodes[0, 1])

    assert graph.number_of_nodes() == 9
    assert graph.number_of_edges() == 8


def test_build_graph_state_assigns_x_measurement_to_all_nodes() -> None:
    code = StabilizerCode(_matrix([[1, 0, 0, 1]]))

    result = build_graph_state(code)

    for node in result.graph.nodes:
        _assert_axis_meas_basis(result.graph.meas_bases[node], Axis.X)


@pytest.mark.parametrize(
    ("stabilizer_row", "expected_axis"),
    [
        ([1, 0, 0, 0, 0, 0], Axis.X),  # Zero Y supports.
        ([1, 0, 0, 1, 0, 0], Axis.Y),  # One Y support.
        ([1, 1, 0, 1, 1, 0], Axis.X),  # Two Y supports.
        ([1, 1, 1, 1, 1, 1], Axis.Y),  # Three Y supports.
    ],
)
def test_build_graph_state_type_i_selects_ancilla_basis_from_y_support_parity(
    stabilizer_row: list[int],
    expected_axis: Axis,
) -> None:
    code = StabilizerCode(_matrix([stabilizer_row]))

    result = build_graph_state(code, y_foliation=YFoliation.TYPE_I)

    _assert_axis_meas_basis(result.graph.meas_bases[result.ancilla_nodes[0]], expected_axis)


@pytest.mark.parametrize("y_foliation", [YFoliation.TYPE_I, YFoliation.TYPE_II])
@pytest.mark.parametrize(
    "stabilizer_matrix",
    [
        pytest.param(
            [
                [1, 0, 0, 1],  # X0 Z1.
                [0, 1, 1, 0],  # Z0 X1.
            ],
            id="xz-order",
        ),
        pytest.param(
            [
                [1, 0, 1, 1],  # Y0 Z1.
                [0, 1, 1, 1],  # Z0 Y1.
            ],
            id="yz-order",
        ),
    ],
)
def test_build_graph_state_connects_ancillas_for_odd_twisted_order_pairs(
    y_foliation: YFoliation,
    stabilizer_matrix: list[list[int]],
) -> None:
    result = build_graph_state(StabilizerCode(_matrix(stabilizer_matrix)), y_foliation=y_foliation)

    assert result.graph.has_edge(result.ancilla_nodes[0], result.ancilla_nodes[1])


@pytest.mark.parametrize("y_foliation", [YFoliation.TYPE_I, YFoliation.TYPE_II])
@pytest.mark.parametrize(
    "stabilizer_matrix",
    [
        pytest.param(
            [
                [0, 0, 1, 1],  # Z0 Z1.
                [1, 1, 0, 0],  # X0 X1: the same order occurs twice.
            ],
            id="same-order-twice",
        ),
        pytest.param(
            [
                [0, 0, 1, 1, 1, 1, 0, 0],  # Z0 Z1 X2 X3.
                [1, 1, 0, 0, 0, 0, 1, 1],  # X0 X1 Z2 Z3: two twists in each direction.
            ],
            id="even-twisted-pairs",
        ),
    ],
)
def test_build_graph_state_omits_ancilla_edge_without_odd_twisted_order_pairs(
    y_foliation: YFoliation,
    stabilizer_matrix: list[list[int]],
) -> None:
    result = build_graph_state(StabilizerCode(_matrix(stabilizer_matrix)), y_foliation=y_foliation)

    assert not result.graph.has_edge(result.ancilla_nodes[0], result.ancilla_nodes[1])


_YY_ODD = [
    [1, 1, 0, 1, 1, 0],  # Y0 Y1.
    [0, 1, 1, 0, 1, 1],  # Y1 Y2: one equal-Y overlap.
]
_YY_ODD_WITH_MIXED_PAIR = [
    [1, 0, 1, 1, 1, 0],  # Y0 Z1 X2.
    [1, 1, 0, 1, 0, 1],  # Y0 X1 Z2: equal-Y overlap plus one twist in each direction.
]
_YY_ODD_WITH_FORWARD_PAIRS = [
    [1, 0, 0, 1, 1, 1],  # Y0 Z1 Z2.
    [1, 1, 1, 1, 0, 0],  # Y0 X1 X2: equal-Y overlap plus two twists in one direction.
]
_YY_EVEN = [
    [1, 1, 1, 1, 1, 1, 1, 1],  # Y0 Y1 Y2 Y3.
    [0, 0, 1, 1, 0, 0, 1, 1],  # Y2 Y3: two equal-Y overlaps.
]


@pytest.mark.parametrize(
    ("stabilizer_matrix", "y_foliation", "expected_edge"),
    [
        pytest.param(_YY_ODD, YFoliation.TYPE_I, True, id="yy-odd-type-i"),
        pytest.param(_YY_ODD, YFoliation.TYPE_II, False, id="yy-odd-type-ii"),
        pytest.param(_YY_ODD_WITH_MIXED_PAIR, YFoliation.TYPE_I, False, id="yy-odd-mixed-pair-type-i"),
        pytest.param(_YY_ODD_WITH_MIXED_PAIR, YFoliation.TYPE_II, True, id="yy-odd-mixed-pair-type-ii"),
        pytest.param(_YY_ODD_WITH_FORWARD_PAIRS, YFoliation.TYPE_I, True, id="yy-odd-forward-pairs-type-i"),
        pytest.param(_YY_ODD_WITH_FORWARD_PAIRS, YFoliation.TYPE_II, False, id="yy-odd-forward-pairs-type-ii"),
        pytest.param(_YY_EVEN, YFoliation.TYPE_I, False, id="yy-even-type-i"),
        pytest.param(_YY_EVEN, YFoliation.TYPE_II, False, id="yy-even-type-ii"),
    ],
)
def test_build_graph_state_counts_equal_y_overlaps_in_twist_parity_for_type_i(
    stabilizer_matrix: list[list[int]],
    y_foliation: YFoliation,
    expected_edge: bool,
) -> None:
    result = build_graph_state(StabilizerCode(_matrix(stabilizer_matrix)), y_foliation=y_foliation)

    assert result.graph.has_edge(result.ancilla_nodes[0], result.ancilla_nodes[1]) == expected_edge


def test_build_graph_state_returns_index_to_node_maps() -> None:
    code = StabilizerCode(_matrix([[1, 0, 0, 1]]))

    result = build_graph_state(code)

    assert set(result.data_nodes) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert set(result.ancilla_nodes) == {0}
    assert set(result.data_nodes.values()).isdisjoint(result.ancilla_nodes.values())


def test_build_graph_state_registers_custom_data_io_indices() -> None:
    code = StabilizerCode(_matrix([[1, 0, 0, 1]]))

    result = build_graph_state(code, data_as_io=True, qubit_indices={0: 10, 1: 12})

    assert result.graph.input_node_indices == {result.data_nodes[0, 0]: 10, result.data_nodes[1, 0]: 12}
    assert result.graph.output_node_indices == {result.data_nodes[0, 2]: 10, result.data_nodes[1, 2]: 12}
    assert set(result.data_nodes) == {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}
    for qubit in range(2):
        _assert_axis_meas_basis(result.graph.meas_bases[result.data_nodes[qubit, 0]], Axis.X)
        _assert_axis_meas_basis(result.graph.meas_bases[result.data_nodes[qubit, 1]], Axis.X)
        assert result.data_nodes[qubit, 2] not in result.graph.meas_bases
        assert result.graph.has_edge(result.data_nodes[qubit, 0], result.data_nodes[qubit, 1])
        assert result.graph.has_edge(result.data_nodes[qubit, 1], result.data_nodes[qubit, 2])


@pytest.mark.parametrize(
    ("qubit_indices", "message"),
    [
        ({0: 10}, "map every stabilizer-code qubit"),
        ({0: 10, 1: 10}, "values must be unique"),
    ],
)
def test_build_graph_state_rejects_invalid_custom_data_io_indices(
    qubit_indices: dict[int, int],
    message: str,
) -> None:
    code = StabilizerCode(_matrix([[1, 0, 0, 1]]))

    with pytest.raises(ValueError, match=message):
        build_graph_state(code, data_as_io=True, qubit_indices=qubit_indices)


def test_build_graph_state_rejects_qubit_indices_without_data_io() -> None:
    code = StabilizerCode(_matrix([[1, 0, 0, 1]]))

    with pytest.raises(ValueError, match="data_as_io=True"):
        build_graph_state(code, qubit_indices={0: 10, 1: 12})


def test_build_graph_state_registers_type_ii_chain_endpoints_as_io() -> None:
    code = StabilizerCode(_matrix([[1, 1]]))

    result = build_graph_state(code, y_foliation=YFoliation.TYPE_II, data_as_io=True)

    assert result.graph.input_node_indices == {result.data_nodes[0, 0]: 0}
    assert result.graph.output_node_indices == {result.data_nodes[0, 3]: 0}
    assert set(result.data_nodes) == {(0, 0), (0, 1), (0, 2), (0, 3)}
    for layer in range(3):
        _assert_axis_meas_basis(result.graph.meas_bases[result.data_nodes[0, layer]], Axis.Y)
    assert result.data_nodes[0, 3] not in result.graph.meas_bases
    assert result.graph.has_edge(result.data_nodes[0, 2], result.data_nodes[0, 3])


def test_build_graph_state_type_ii_keeps_separate_output_for_non_y_support() -> None:
    code = StabilizerCode(_matrix([[1, 0]]))

    result = build_graph_state(code, y_foliation=YFoliation.TYPE_II, data_as_io=True)

    assert set(result.data_nodes) == {(0, 0), (0, 1), (0, 2)}
    assert result.graph.output_node_indices == {result.data_nodes[0, 2]: 0}
    _assert_axis_meas_basis(result.graph.meas_bases[result.data_nodes[0, 0]], Axis.X)
    _assert_axis_meas_basis(result.graph.meas_bases[result.data_nodes[0, 1]], Axis.X)
    assert result.data_nodes[0, 2] not in result.graph.meas_bases


def test_build_graph_state_type_ii_routes_y_support_through_three_node_y_chain() -> None:
    code = StabilizerCode(
        _matrix(
            [
                [0, 1],  # Z support.
                [1, 1],  # Y support.
                [1, 0],  # X support.
            ]
        )
    )

    result = build_graph_state(code, y_foliation=YFoliation.TYPE_II)
    graph = result.graph

    assert set(result.data_nodes) == {(0, 0), (0, 1), (0, 2)}
    assert graph.has_edge(result.data_nodes[0, 0], result.data_nodes[0, 1])
    assert graph.has_edge(result.data_nodes[0, 1], result.data_nodes[0, 2])
    assert graph.has_edge(result.ancilla_nodes[0], result.data_nodes[0, 0])
    assert graph.has_edge(result.ancilla_nodes[1], result.data_nodes[0, 1])
    assert graph.has_edge(result.ancilla_nodes[2], result.data_nodes[0, 2])
    assert not graph.has_edge(result.ancilla_nodes[1], result.data_nodes[0, 0])
    assert not graph.has_edge(result.ancilla_nodes[1], result.data_nodes[0, 2])

    for data_node in result.data_nodes.values():
        _assert_axis_meas_basis(graph.meas_bases[data_node], Axis.Y)
    for ancilla_node in result.ancilla_nodes.values():
        _assert_axis_meas_basis(graph.meas_bases[ancilla_node], Axis.X)

    assert graph.number_of_nodes() == 6
    assert graph.number_of_edges() == 5


def test_build_graph_state_type_ii_keeps_two_node_x_chain_without_y_support() -> None:
    code = StabilizerCode(
        _matrix(
            [
                [0, 1],  # Z support.
                [1, 0],  # X support.
            ]
        )
    )

    result = build_graph_state(code, y_foliation=YFoliation.TYPE_II)
    graph = result.graph

    assert set(result.data_nodes) == {(0, 0), (0, 1)}
    assert graph.has_edge(result.data_nodes[0, 0], result.data_nodes[0, 1])
    assert graph.has_edge(result.ancilla_nodes[0], result.data_nodes[0, 0])
    assert graph.has_edge(result.ancilla_nodes[1], result.data_nodes[0, 1])
    for data_node in result.data_nodes.values():
        _assert_axis_meas_basis(graph.meas_bases[data_node], Axis.X)


def test_build_graph_state_type_ii_moves_ancilla_off_overlapping_middle_node() -> None:
    code = StabilizerCode(
        _matrix([[1, 1]]),
        qubit_coords={0: (10.0, 20.0)},
    )

    result = build_graph_state(code, z_base=5, y_foliation=YFoliation.TYPE_II)
    coords = result.graph.coordinates

    assert set(result.data_nodes) == {(0, 5), (0, 6), (0, 7)}
    assert coords[result.data_nodes[0, 5]] == (10.0, 20.0, 5.0)
    assert coords[result.data_nodes[0, 6]] == (10.0, 20.0, 5.5)
    assert coords[result.data_nodes[0, 7]] == (10.0, 20.0, 6.0)
    assert coords[result.ancilla_nodes[0]] == (10.5, 20.0, 5.5)


def test_build_graph_state_type_i_moves_ancilla_off_same_xy_projection() -> None:
    code = StabilizerCode(
        _matrix([[1, 1]]),
        qubit_coords={0: (10.0, 20.0)},
    )

    result = build_graph_state(code, z_base=5, y_foliation=YFoliation.TYPE_I)
    coords = result.graph.coordinates

    assert coords[result.data_nodes[0, 5]] == (10.0, 20.0, 5.0)
    assert coords[result.data_nodes[0, 6]] == (10.0, 20.0, 6.0)
    assert coords[result.ancilla_nodes[0]] == (10.5, 20.0, 5.5)


def test_build_graph_state_moves_ancilla_off_data_qubit_at_support_centroid() -> None:
    code = StabilizerCode(
        _matrix([[1, 0, 1, 0, 0, 0]]),
        qubit_coords={
            0: (0.0, 0.0),
            1: (1.0, 0.0),
            2: (2.0, 0.0),
        },
    )

    result = build_graph_state(code)
    coords = result.graph.coordinates

    assert coords[result.data_nodes[1, 1]] == (1.0, 0.0, 1.0)
    assert coords[result.ancilla_nodes[0]] == (1.5, 0.0, 1.0)


def test_build_graph_state_scales_ancilla_clearance_with_coordinate_span() -> None:
    code = StabilizerCode(
        _matrix([[1, 0, 1, 0, 0, 0]]),
        qubit_coords={
            0: (0.0, 0.0),
            1: (1000.0, 0.0),
            2: (2000.0, 0.0),
        },
    )

    result = build_graph_state(code)

    assert result.graph.coordinates[result.ancilla_nodes[0]] == (1100.0, 0.0, 1.0)


def test_build_graph_state_ignores_nonfinite_coordinates_when_avoiding_overlap() -> None:
    code = StabilizerCode(
        _matrix([[1, 1, 0, 0, 0, 0, 0, 0]]),
        qubit_coords={
            0: (-1.0, 0.0),
            1: (1.0, 0.0),
            2: (0.0, 0.0),
            3: (float("nan"), 5.0),
        },
    )

    result = build_graph_state(code)

    assert result.graph.coordinates[result.ancilla_nodes[0]] == (0.5, 0.0, 1.0)


def test_build_graph_state_uses_next_escape_direction_when_first_is_occupied() -> None:
    code = StabilizerCode(
        _matrix([[1, 0, 0, 1, 0, 0, 0, 0]]),
        qubit_coords={
            0: (0.0, 0.0),
            1: (1.5, 0.0),
            2: (2.0, 0.0),
            3: (3.0, 0.0),
        },
    )

    result = build_graph_state(code)

    assert result.graph.coordinates[result.ancilla_nodes[0]] == (1.0, 0.0, 1.0)


def test_build_graph_state_expands_escape_radius_when_first_ring_is_occupied() -> None:
    escape_ring = [
        (0.5, 0.0),
        (-0.5, 0.0),
        (0.0, 0.5),
        (0.0, -0.5),
        (0.5, 0.5),
        (0.5, -0.5),
        (-0.5, 0.5),
        (-0.5, -0.5),
    ]
    code = StabilizerCode(
        _matrix([[1, 1, *([0] * 9), *([0] * 11)]]),
        qubit_coords={
            0: (-2.0, 0.0),
            1: (2.0, 0.0),
            2: (0.0, 0.0),
            **dict(enumerate(escape_ring, start=3)),
        },
    )

    result = build_graph_state(code)

    assert result.graph.coordinates[result.ancilla_nodes[0]] == (1.0, 0.0, 1.0)


@pytest.mark.parametrize("y_foliation", [YFoliation.TYPE_I, YFoliation.TYPE_II])
def test_build_graph_state_places_separate_io_output_at_end_of_unit(y_foliation: YFoliation) -> None:
    code = StabilizerCode(_matrix([[1, 1]]), qubit_coords={0: (10.0, 20.0)})

    result = build_graph_state(code, z_base=5, y_foliation=y_foliation, data_as_io=True)
    output_node = next(iter(result.graph.output_node_indices))

    assert result.graph.coordinates[output_node] == (10.0, 20.0, 7.0)
    assert output_node not in result.graph.meas_bases


def test_build_graph_state_lifts_coordinates_to_shifted_3d_layers() -> None:
    code = StabilizerCode(
        _matrix([[0, 1, 1, 0]]),
        qubit_coords={
            0: (10.0, 20.0),
            1: (30.0, 40.0, 999.0),
        },
    )

    result = build_graph_state(code, z_base=5)
    coords = result.graph.coordinates

    assert set(result.data_nodes) == {(0, 5), (0, 6), (1, 5), (1, 6)}
    assert coords[result.data_nodes[0, 5]] == (10.0, 20.0, 5.0)
    assert coords[result.data_nodes[0, 6]] == (10.0, 20.0, 6.0)
    assert coords[result.data_nodes[1, 5]] == (30.0, 40.0, 5.0)
    assert coords[result.data_nodes[1, 6]] == (30.0, 40.0, 6.0)
    assert coords[result.ancilla_nodes[0]] == (20.0, 30.0, 5.5)


def test_build_graph_state_explicit_ancilla_coordinate_overrides_average() -> None:
    code = StabilizerCode(
        _matrix([[0, 1, 1, 0]]),
        stabilizer_coords={0: (1.0, 2.0, 3.0)},
        qubit_coords={
            0: (10.0, 20.0),
            1: (30.0, 40.0),
        },
    )

    result = build_graph_state(code)

    assert result.graph.coordinates[result.ancilla_nodes[0]] == (1.0, 2.0, 3.0)


def test_build_graph_state_allows_missing_coordinates() -> None:
    code = StabilizerCode(_matrix([[1, 0, 0, 1]]))

    result = build_graph_state(code)

    assert result.graph.coordinates == {}


def test_stabilizer_code_preserves_higher_dimensional_coordinates() -> None:
    code = StabilizerCode(
        _matrix([[1, 0]]),
        stabilizer_coords={0: (1.0, 2.0, 3.0, 4.0)},
        qubit_coords={0: (5.0, 6.0, 7.0, 8.0)},
    )

    assert code.stabilizer_coord == {0: (1.0, 2.0, 3.0, 4.0)}
    assert code.qubit_coord == {0: (5.0, 6.0, 7.0, 8.0)}


def test_build_graph_state_rejects_non_integer_z_base() -> None:
    code = StabilizerCode(_matrix([[1, 0]]))

    with pytest.raises(TypeError, match="z_base must be an integer"):
        build_graph_state(code, z_base=0.5)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_build_graph_state_skips_distance_scan_when_centroid_is_free(monkeypatch: pytest.MonkeyPatch) -> None:
    """A free centroid is accepted from the occupancy index without a distance scan over all nodes."""
    code = StabilizerCode(
        _matrix([[1, 1, 0, 0]]),
        qubit_coords={0: (0.0, 0.0), 1: (1.0, 0.0)},
    )

    def fail(*_args: object) -> float:
        pytest.fail("free centroid must not trigger math.dist over occupied coordinates")

    monkeypatch.setattr(math, "dist", fail)
    result = build_graph_state(code)

    assert result.graph.coordinates[result.ancilla_nodes[0]] == (0.5, 0.0, 1.0)


def test_build_graph_state_detects_collision_within_tolerance_across_rounding_cells() -> None:
    """Points closer than the tolerance collide even when they round to different index cells."""
    centroid_x = 1.0 - 2e-10
    occupied_x = centroid_x + 8e-10
    code = StabilizerCode(
        _matrix([[1, 1, 0, 0, 0, 0]]),
        qubit_coords={
            0: (centroid_x - 1.0, 0.0),
            1: (centroid_x + 1.0, 0.0),
            2: (occupied_x, 0.0),
        },
    )

    result = build_graph_state(code)
    ancilla = result.graph.coordinates[result.ancilla_nodes[0]]

    assert not math.isclose(ancilla[0], centroid_x, abs_tol=1e-9)
    assert math.isclose(ancilla[0], centroid_x + 0.5)
    assert ancilla[1:] == (0.0, 1.0)


def test_build_graph_state_keeps_centroid_beyond_tolerance_in_same_rounding_cell() -> None:
    """A neighbor farther than the tolerance does not displace the ancilla."""
    code = StabilizerCode(
        _matrix([[1, 1, 0, 0, 0, 0]]),
        qubit_coords={
            0: (0.0, 0.0),
            1: (1.0, 0.0),
            2: (0.5 + 5e-7, 0.0),
        },
    )

    result = build_graph_state(code)

    assert result.graph.coordinates[result.ancilla_nodes[0]] == (0.5, 0.0, 1.0)


def test_build_graph_state_separates_ancillas_sharing_a_centroid() -> None:
    """Previously placed ancillas count as occupied for later stabilizers of the same block."""
    code = StabilizerCode(
        _matrix([[1, 1, 0, 0], [0, 0, 1, 1]]),
        qubit_coords={0: (0.0, 0.0), 1: (1.0, 0.0)},
    )

    result = build_graph_state(code, y_foliation=YFoliation.TYPE_I)
    coords = result.graph.coordinates
    ancilla_xy = [coords[result.ancilla_nodes[row]][:2] for row in range(2)]
    data_xy = {coords[node][:2] for node in result.data_nodes.values()}

    assert ancilla_xy[0] == (0.5, 0.0)
    assert ancilla_xy[1] != ancilla_xy[0]
    assert ancilla_xy[1] not in data_xy


def _reference_avoid_occupied_coordinate(
    occupied_coordinates: Sequence[tuple[float, ...]],
    coordinate: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Full-scan restatement of the escape search (the pre-index algorithm)."""
    tolerance = 1e-9
    occupied = [c for c in occupied_coordinates if math.isfinite(c[0]) and math.isfinite(c[1])]
    if not any(math.dist(coordinate[:2], o[:2]) <= tolerance for o in occupied):
        return coordinate
    xs = [o[0] for o in occupied]
    ys = [o[1] for o in occupied]
    span = max(max(xs) * 0.05 - min(xs) * 0.05, max(ys) * 0.05 - min(ys) * 0.05)
    clearance = max(0.5, span)
    directions = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))
    x, y, z = coordinate
    for radius in range(1, 3 * len(occupied) + 2):
        distance = radius * clearance
        for dx, dy in directions:
            candidate = (x + dx * distance, y + dy * distance, z)
            if all(math.dist(candidate[:2], o[:2]) >= clearance - tolerance for o in occupied):
                return candidate
    pytest.fail("reference search exhausted")


def _graph_with_coordinates(coordinates: Sequence[tuple[float, ...]]) -> GraphState:
    graph = GraphState()
    for coordinate in coordinates:
        graph.add_node(coordinate=coordinate)
    return graph


@pytest.mark.parametrize("seed", range(40))
def test_avoid_occupied_coordinate_matches_full_scan_reference(seed: int) -> None:
    """Indexed collision and clearance queries place ancillas exactly where the full scan did."""
    rng = np.random.default_rng(seed)
    count = int(rng.integers(1, 60))
    scale = float(rng.choice([1.0, 7.0, 400.0]))
    # Half-integer lattice with duplicates forces collisions and dense escape rings.
    occupied: list[tuple[float, ...]] = [
        (float(rng.integers(-6, 7)) * 0.5 * scale, float(rng.integers(-6, 7)) * 0.5 * scale, float(rng.integers(0, 3)))
        for _ in range(count)
    ]
    if rng.random() < 0.2:
        occupied.append((math.nan, 1.0, 0.0))
    queries = [occupied[int(rng.integers(0, len(occupied)))] for _ in range(4)]
    queries.extend([(0.25 * scale, 0.25 * scale, 1.0), (occupied[0][0] + 5e-10, occupied[0][1], 2.0)])

    graph = _graph_with_coordinates(occupied)
    occupancy = qeccode._CoordinateOccupancy.from_graph(graph)
    for query in queries:
        if not (math.isfinite(query[0]) and math.isfinite(query[1])):
            continue
        coordinate = (float(query[0]), float(query[1]), float(query[2]))
        expected = _reference_avoid_occupied_coordinate(occupied, coordinate)
        assert qeccode._avoid_occupied_coordinate(graph, coordinate, occupancy) == expected
        assert qeccode._avoid_occupied_coordinate(graph, coordinate) == expected


def test_avoid_occupied_coordinate_collision_work_is_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolving a collision touches only nearby coordinates, not every assigned coordinate."""
    far_away = [(1000.0 + 3.0 * i, 1000.0 + 3.0 * (i % 7), 0.0) for i in range(2000)]
    occupied = [*far_away, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    graph = _graph_with_coordinates(occupied)
    occupancy = qeccode._CoordinateOccupancy.from_graph(graph)
    expected = _reference_avoid_occupied_coordinate(occupied, (0.0, 0.0, 1.0))

    calls = 0
    real_dist = math.dist

    def counting(p: Sequence[float], q: Sequence[float]) -> float:
        nonlocal calls
        calls += 1
        return real_dist(p, q)

    monkeypatch.setattr(math, "dist", counting)
    placed = qeccode._avoid_occupied_coordinate(graph, (0.0, 0.0, 1.0), occupancy)

    assert placed == expected
    # The full-scan algorithm evaluated every coordinate once per collision check and per
    # escape candidate (about 10,000 distances here); local queries need a small fraction.
    assert calls < 1000


def test_coordinate_occupancy_handles_huge_magnitudes_exactly() -> None:
    """Coordinates beyond the indexable magnitude fall back to exact full scans."""
    occupied = [(1e12, 0.0, 0.0), (1e12 + 0.5, 0.0, 0.0), (3.0, 4.0, 0.0)]
    graph = _graph_with_coordinates(occupied)
    occupancy = qeccode._CoordinateOccupancy.from_graph(graph)

    for coordinate in [(1e12, 0.0, 1.0), (3.0, 4.0, 1.0), (5.0, 5.0, 1.0)]:
        expected = _reference_avoid_occupied_coordinate(occupied, coordinate)
        assert qeccode._avoid_occupied_coordinate(graph, coordinate, occupancy) == expected
