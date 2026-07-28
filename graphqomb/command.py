"""Command module for measurement pattern.

This module provides:

- `N`: Preparation command.
- `E`: Entanglement command.
- `M`: Measurement command.
- `TICK`: Time slice separator command.
- `Command`: Type alias of all commands.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import typing_extensions

if TYPE_CHECKING:
    from graphqomb.common import MeasBasis


@dataclasses.dataclass
class N:
    r"""Preparation command.

    Attributes
    ----------
    node : `int`
        The node index to be prepared.
    coordinate : `tuple`\[`float`, ...\] | `None`
        Optional coordinate for the node (2D or 3D).
    """

    node: int
    coordinate: tuple[float, ...] | None = None

    @typing_extensions.override
    def __str__(self) -> str:
        if self.coordinate is not None:
            return f"N: node={self.node}, coord={self.coordinate}"
        return f"N: node={self.node}"


@dataclasses.dataclass
class E:
    r"""Entanglement command.

    Attributes
    ----------
    nodes : `tuple`\[`int`, `int`\]
        The node indices to be entangled.
    """

    nodes: tuple[int, int]

    @typing_extensions.override
    def __str__(self) -> str:
        return f"E: nodes={self.nodes}"


@dataclasses.dataclass
class M:
    """Measurement command.

    Attributes
    ----------
    node : `int`
        The node index to be measured.
    meas_basis : MeasBasis
        The measurement basis.
    """

    node: int
    meas_basis: MeasBasis

    @typing_extensions.override
    def __str__(self) -> str:
        return f"M: node={self.node}, plane={self.meas_basis.plane}, angle={self.meas_basis.angle}"


@dataclasses.dataclass
class TICK:
    """Time slice separator command.

    Marks the boundary between time slices. Commands between two consecutive
    TICK commands can be executed in parallel within the same time slice.
    """

    @typing_extensions.override
    def __str__(self) -> str:
        return "TICK"


Command = N | E | M | TICK
