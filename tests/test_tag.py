"""Tests for the Stim-style tag escape helpers."""

from __future__ import annotations

import pytest

from graphqomb._tag import escape_tag, unescape_tag


@pytest.mark.parametrize(
    "tag",
    [
        "",
        "type=flag",
        "a]b",
        "a[b",
        "back\\slash",
        "new\nline",
        "cr\rreturn",
        "sp ace",
        "hash#x",
        "ünïcode",
        "\\C",
    ],
)
def test_escape_tag_roundtrip(tag: str) -> None:
    assert unescape_tag(escape_tag(tag)) == tag


def test_escape_tag_matches_stim_escape_language() -> None:
    assert escape_tag("]\\\r\n") == "\\C\\B\\r\\n"


def test_escape_tag_keeps_plain_characters() -> None:
    assert escape_tag("type=flag") == "type=flag"


def test_unescape_tag_rejects_unknown_escape() -> None:
    with pytest.raises(ValueError, match="Unrecognized tag escape sequence"):
        unescape_tag("a\\Qb")


def test_unescape_tag_rejects_trailing_backslash() -> None:
    with pytest.raises(ValueError, match="Unrecognized tag escape sequence"):
        unescape_tag("a\\")


def test_escaped_tag_parses_back_through_stim() -> None:
    stim = pytest.importorskip("stim")
    tag = "a]b\\c\nd\re [f#g h"
    circuit = stim.Circuit(f"M 0\nDETECTOR[{escape_tag(tag)}] rec[-1]")
    assert circuit[1].tag == tag
