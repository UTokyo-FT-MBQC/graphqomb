"""Tests for flag-detector post-selection helpers."""

from __future__ import annotations

import numpy as np
import pytest

from graphqomb.stim_glue import (
    CAT_FLAG_TAG,
    FLAG_DETECTOR_TAG,
    flag_detector_indices,
    flag_postselection_mask,
    stim_compile,
    stim_text_to_pattern,
)

stim = pytest.importorskip("stim")


def test_flag_detector_tag_value() -> None:
    assert FLAG_DETECTOR_TAG == "type=flag"


def test_flag_detector_indices_filters_by_tag() -> None:
    circuit = stim.Circuit(
        """
        RX 0 1 2
        MX 0 1 2
        DETECTOR[type=flag] rec[-3]
        DETECTOR rec[-2]
        DETECTOR[other] rec[-1]
        DETECTOR[type=flag] rec[-1]
        """
    )

    assert flag_detector_indices(circuit) == (0, 3)


def test_flag_detector_indices_counts_repeat_iterations() -> None:
    circuit = stim.Circuit(
        """
        RX 0
        MX 0
        DETECTOR rec[-1]
        REPEAT 3 {
            RX 0
            MX 0
            DETECTOR[type=flag] rec[-1]
        }
        """
    )

    assert flag_detector_indices(circuit) == (1, 2, 3)


def test_flag_postselection_mask_bit_packing() -> None:
    circuit = stim.Circuit()
    circuit.append("RX", [0])
    for index in range(10):
        circuit.append("MX", [0])
        tag = "type=flag" if index in {1, 9} else ""
        circuit.append(
            stim.CircuitInstruction("DETECTOR", [stim.target_rec(-1)], tag=tag),
        )

    mask = flag_postselection_mask(circuit)

    assert mask.dtype == np.uint8
    assert mask.tolist() == [2, 2]


def test_flag_postselection_mask_without_flags_is_zero() -> None:
    circuit = stim.Circuit("RX 0\nMX 0\nDETECTOR rec[-1]")

    assert flag_postselection_mask(circuit).tolist() == [0]


def test_flag_postselection_mask_on_compiled_pattern() -> None:
    result = stim_text_to_pattern(
        """
        RX 0 1 2
        TICK
        MPP X0*X1 X1*X2
        DETECTOR[type=flag] rec[-2]
        DETECTOR rec[-1]
        """
    )

    circuit = stim.Circuit(stim_compile(result.pattern))

    assert flag_detector_indices(circuit) == (0,)
    assert flag_postselection_mask(circuit).tolist() == [1]


def test_flag_helpers_select_the_requested_tag() -> None:
    circuit = stim.Circuit(
        """
        RX 0 1 2
        MX 0 1 2
        DETECTOR[type=flag] rec[-3]
        DETECTOR[other] rec[-2]
        DETECTOR rec[-1]
        DETECTOR[other] rec[-1]
        """
    )

    assert flag_detector_indices(circuit, "other") == (1, 3)
    assert flag_postselection_mask(circuit, "other").tolist() == [0b1010]
    assert flag_detector_indices(circuit, tag=FLAG_DETECTOR_TAG) == flag_detector_indices(circuit) == (0,)


def test_cat_flag_checks_are_selected_separately_from_source_flags() -> None:
    result = stim_text_to_pattern(
        """
        RX 0 1 2
        TICK
        MPP X0*X1*X2
        DETECTOR[type=flag] rec[-1]
        """,
        mpp_ancilla="cat",
    )

    circuit = stim.Circuit(stim_compile(result.pattern))

    assert CAT_FLAG_TAG != FLAG_DETECTOR_TAG
    assert flag_detector_indices(circuit) == (0,)
    assert flag_detector_indices(circuit, CAT_FLAG_TAG) == (1,)
    assert flag_postselection_mask(circuit, CAT_FLAG_TAG).tolist() == [2]
