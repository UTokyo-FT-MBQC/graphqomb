"""Tests for building QEC codes from Stim MPP layers."""

from __future__ import annotations

import pytest

from graphqomb.qeccode import build_graph_state
from graphqomb.stim_glue._parse import stim_mpp_extraction_from_records
from graphqomb.stim_glue.mpp import stabilizer_code_from_stim_text


def test_stabilizer_code_from_stim_mpp_sets_x_z_and_y_support() -> None:
    extraction = stabilizer_code_from_stim_text(
        """
        QUBIT_COORDS(10, 20, 30) 0
        QUBIT_COORDS(11, 21, 31) 1
        QUBIT_COORDS(12, 22, 32) 2
        MPP X0*Y1*Z2
        """
    )

    matrix = (extraction.code.hx.toarray(), extraction.code.hz.toarray())

    assert extraction.stim_to_column == {0: 0, 1: 1, 2: 2}
    assert extraction.supports == (((0, "X"), (1, "Y"), (2, "Z")),)
    assert matrix[0].tolist() == [[True, True, False]]
    assert matrix[1].tolist() == [[False, True, True]]
    assert extraction.code.qubit_coord == {
        0: (10.0, 20.0),
        1: (11.0, 21.0),
        2: (12.0, 22.0),
    }


def test_stabilizer_code_from_stim_mpp_preserves_3d_coordinates_when_requested() -> None:
    extraction = stabilizer_code_from_stim_text(
        """
        QUBIT_COORDS(10, 20, 30) 0
        QUBIT_COORDS(11, 21, 31) 1
        MPP X0*Z1
        """,
        coord_dims=3,
    )

    assert extraction.code.qubit_coord == {
        0: (10.0, 20.0, 30.0),
        1: (11.0, 21.0, 31.0),
    }


def test_stabilizer_code_from_stim_mpp_preserves_higher_dimensional_coordinates() -> None:
    extraction = stabilizer_code_from_stim_text(
        """
        QUBIT_COORDS(10, 20) 0
        QUBIT_COORDS(30, 40) 0
        MPP X0
        """,
        coord_dims=4,
    )

    assert extraction.code.qubit_coord == {0: (10.0, 20.0, 30.0, 40.0)}


def test_stabilizer_code_from_stim_mpp_handles_sparse_stim_ids_and_multiple_products() -> None:
    extraction = stabilizer_code_from_stim_text(
        """
        QUBIT_COORDS(1, 2) 10
        QUBIT_COORDS(3, 4) 12
        QUBIT_COORDS(5, 6) 99
        MPP X10*Y12 Z99
        TICK
        MPP Z10*Z12
        """
    )

    hx = extraction.code.hx.toarray().tolist()
    hz = extraction.code.hz.toarray().tolist()

    assert extraction.stim_to_column == {10: 0, 12: 1, 99: 2}
    assert extraction.column_to_stim == {0: 10, 1: 12, 2: 99}
    assert extraction.supports == (((10, "X"), (12, "Y")), ((99, "Z"),))
    assert hx == [[True, True, False], [False, False, False]]
    assert hz == [[False, True, False], [False, False, True]]
    assert extraction.detector_rows == ()
    assert extraction.logical_observable_rows == {}


def test_stabilizer_code_from_stim_mpp_can_select_later_mpp_layer() -> None:
    extraction = stabilizer_code_from_stim_text(
        """
        QUBIT_COORDS(1, 2) 10
        QUBIT_COORDS(3, 4) 12
        MPP X10
        TICK
        MPP Z10*Z12
        """,
        mpp_layer=1,
    )

    assert extraction.stim_to_column == {10: 0, 12: 1}
    assert extraction.supports == (((10, "Z"), (12, "Z")),)
    assert extraction.code.hx.toarray().tolist() == [[False, False]]
    assert extraction.code.hz.toarray().tolist() == [[True, True]]


def test_stabilizer_code_from_stim_mpp_rejects_missing_layer() -> None:
    with pytest.raises(ValueError, match=r"has 1 MPP layer"):
        stabilizer_code_from_stim_text("MPP X0\n", mpp_layer=1)


def test_stabilizer_code_from_stim_mpp_rejects_signed_product() -> None:
    with pytest.raises(ValueError, match="Signed MPP products are not supported"):
        stabilizer_code_from_stim_text("MPP !X0*Y1\n")


def test_stabilizer_code_from_stim_mpp_builds_graph_state() -> None:
    extraction = stabilizer_code_from_stim_text(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        QUBIT_COORDS(2, 0) 2
        MPP X0*Y1*Z2
        MPP Z0*Z2
        """
    )

    result = build_graph_state(extraction.code)

    assert result.graph.number_of_nodes() == 8
    assert result.graph.number_of_edges() == 9
    assert result.graph.coordinates[result.data_nodes[0, 0]] == (0.0, 0.0, 0.0)
    assert result.graph.coordinates[result.data_nodes[0, 1]] == (0.0, 0.0, 1.0)


def test_stabilizer_code_from_stim_mpp_reads_detectors_and_observables() -> None:
    extraction = stabilizer_code_from_stim_text(
        """
        MPP X0*X1 Z2 X3
        DETECTOR rec[-1] rec[-3]
        OBSERVABLE_INCLUDE(5) rec[-2]
        """
    )

    result = build_graph_state(extraction.code)

    assert extraction.detector_rows == (frozenset({0, 2}),)
    assert extraction.logical_observable_rows == {5: frozenset({1})}
    assert extraction.detector_record_indices == (frozenset({0, 2}),)
    assert extraction.logical_observable_record_indices == {5: frozenset({1})}
    assert extraction.detector_groups(result.ancilla_nodes) == [
        {result.ancilla_nodes[0], result.ancilla_nodes[2]},
    ]
    assert extraction.logical_observables(result.ancilla_nodes) == {5: {result.ancilla_nodes[1]}}


def test_stabilizer_code_from_stim_mpp_accumulates_observable_rows_by_parity() -> None:
    extraction = stabilizer_code_from_stim_text(
        """
        MPP X0 X1
        OBSERVABLE_INCLUDE(2) rec[-1]
        OBSERVABLE_INCLUDE(2) rec[-2] rec[-1]
        """
    )

    assert extraction.logical_observable_rows == {2: frozenset({0})}
    assert extraction.logical_observable_record_indices == {2: frozenset({0})}


def test_stabilizer_code_from_stim_mpp_keeps_selected_rows_when_detector_refs_external_records() -> None:
    extraction = stabilizer_code_from_stim_text(
        """
        M 99
        MPP X0
        DETECTOR rec[-1] rec[-2]
        """
    )

    assert extraction.detector_rows == (frozenset({0}),)
    assert extraction.detector_record_indices == (frozenset({0, 1}),)


def test_stabilizer_code_from_stim_mpp_can_select_all_mpp_layers() -> None:
    extraction = stabilizer_code_from_stim_text(
        """
        MPP X0
        TICK
        MPP Z0
        DETECTOR rec[-1] rec[-2]
        """,
        mpp_layer=None,
    )

    assert extraction.supports == (((0, "X"),), ((0, "Z"),))
    assert extraction.detector_rows == (frozenset({0, 1}),)
    assert extraction.detector_record_indices == (frozenset({0, 1}),)


def test_stabilizer_code_from_stim_mpp_rejects_record_before_beginning_of_time() -> None:
    with pytest.raises(ValueError, match="before the beginning of time"):
        stabilizer_code_from_stim_text("DETECTOR rec[-1]\nMPP X0")


def test_stabilizer_code_from_stim_mpp_keeps_detector_tags_of_selected_detectors() -> None:
    extraction = stabilizer_code_from_stim_text(
        """
        M 99
        MPP X0 X1
        DETECTOR[type=flag] rec[-2]
        DETECTOR rec[-1]
        DETECTOR[dropped] rec[-3]
        """
    )

    # The last detector only references the non-MPP record, so it is not
    # selected and its tag is dropped with it.
    assert extraction.detector_rows == (frozenset({0}), frozenset({1}))
    assert extraction.detector_tags == ("type=flag", "")


def test_stim_mpp_extraction_rejects_misaligned_detector_tags() -> None:
    with pytest.raises(ValueError, match="Detector tag count"):
        stim_mpp_extraction_from_records(
            (((0, "X"),),),
            (0,),
            coordinate_by_stim_id={},
            detector_record_indices=(frozenset({0}),),
            logical_observable_record_indices={},
            detector_tags=("type=flag", "extra"),
        )


def test_stim_mpp_extraction_defaults_detector_tags_to_untagged() -> None:
    extraction = stim_mpp_extraction_from_records(
        (((0, "X"),),),
        (0,),
        coordinate_by_stim_id={},
        detector_record_indices=(frozenset({0}),),
        logical_observable_record_indices={},
    )

    assert extraction.detector_tags == ("",)
