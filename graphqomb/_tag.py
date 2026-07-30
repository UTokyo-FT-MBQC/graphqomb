r"""Stim-style instruction-tag text escaping.

Detector tags travel from imported Stim circuits through `PauliFrame` into
Stim text emitted by `graphqomb.stim_glue.compiler.stim_compile` and into the
``.ptn`` format. Both writers spell a tag inside square brackets, so the tag
text must never contain a raw closing bracket or line break. This module
implements Stim's tag escape language so the emitted text parses back to the
original tag: ``\\`` becomes ``\\B``, ``]`` becomes ``\\C``, a carriage return
becomes ``\\r``, and a line feed becomes ``\\n``. All other characters are
kept verbatim.

This module provides:

- `escape_tag`: Escape a tag for emission inside ``[...]``.
- `unescape_tag`: Invert `escape_tag`.
"""

from __future__ import annotations

_ESCAPES = {
    "\\": "\\B",
    "]": "\\C",
    "\r": "\\r",
    "\n": "\\n",
}
_UNESCAPES = {escaped[1]: raw for raw, escaped in _ESCAPES.items()}


def escape_tag(tag: str) -> str:
    """Escape a tag with Stim's tag escape sequences.

    Returns
    -------
    `str`
        Escaped tag text safe to emit between square brackets.
    """
    return "".join(_ESCAPES.get(char, char) for char in tag)


def unescape_tag(text: str) -> str:
    """Invert `escape_tag`.

    Returns
    -------
    `str`
        Original tag text.

    Raises
    ------
    ValueError
        If the text contains a backslash outside Stim's tag escape language,
        which Stim's own parser also rejects.
    """
    parts: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            escaped = text[index + 1] if index + 1 < len(text) else None
            if escaped is None or escaped not in _UNESCAPES:
                known = ", ".join(f"\\{key}" for key in _UNESCAPES)
                msg = f"Unrecognized tag escape sequence at index {index} in {text!r}; known sequences are {known}."
                raise ValueError(msg)
            parts.append(_UNESCAPES[escaped])
            index += 2
        else:
            parts.append(char)
            index += 1
    return "".join(parts)
