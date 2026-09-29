"""Shared lifetime rules, including boundaries that carry no quantum use."""

from __future__ import annotations

import pytest
import stim

from graphqomb.stim_glue._measurement_lifetimes import normalize_measurement_lifetimes
from tests.stim_reference import assert_record_channel


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("MR[readout] !0", "M[readout] !0"),
        ("M[readout] 0\nH 0", "M[readout] 0\nR 0\nH 0"),
        ("MR[readout] 0\nH 0", "M[readout] 0\nR[readout] 0\nH 0"),
        ("MY 0\nRX[next] 0\nH 0", "MY 0\nRX[next] 0\nH 0"),
        ("MX !0 0", "MX !0\nRX 0\nMX 0"),
        ("M 0\nTICK\nH 1\nDETECTOR rec[-1]", "M 0\nTICK\nH 1\nDETECTOR rec[-1]"),
        ("M 0\nQUBIT_COORDS(1, 2) 0\nMPAD 0\nCX rec[-2] 1", "M 0\nQUBIT_COORDS(1, 2) 0\nMPAD 0\nCX rec[-2] 1"),
        ("MX 0\nCX rec[-1] 0", "MX 0\nRX 0\nCX rec[-1] 0"),
        ("MY 0\nMZZ 0 1", "MY 0\nRY 0\nMZZ 0 1"),
        ("MPP X0*X1\nH 0\nMXX 0 1", "MPP X0*X1\nH 0\nMXX 0 1"),
        ("REPEAT 2 {\n MX 0\n TICK\n}", "MX 0\nTICK\nRX 0\nMX 0\nTICK"),
    ],
)
def test_lifetime_normalization(text: str, expected: str) -> None:
    source = stim.Circuit(text)
    normalized = normalize_measurement_lifetimes(source)
    assert normalized == stim.Circuit(expected)
    assert normalize_measurement_lifetimes(normalized) == normalized
    assert_record_channel(source, normalized, tuple(range(source.num_measurements)))
