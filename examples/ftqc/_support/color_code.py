"""Triangular color-code memory experiments written with plaquette MPPs."""

from __future__ import annotations

from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import TYPE_CHECKING, Literal

import numpy as np
import stim

from graphqomb.feedforward import inverse_dag_from_dag
from graphqomb.noise_model import DepolarizingNoiseModel, MeasurementFlipNoiseModel
from graphqomb.stim_glue import stim_compile

if TYPE_CHECKING:
    from collections.abc import Sequence

    from graphqomb.pattern import Pattern
    from graphqomb.stim_glue import StimImportIR

_MIN_DISTANCE = 3
# Faces of the triangular lattice are the points of one of its three colors.
_FACE_COLOR = 1
_NEIGHBORS = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))


@dataclass(frozen=True)
class TriangularColorCode:
    """Self-dual CSS code: every plaquette is both an X check and a Z check."""

    distance: int
    coordinates: tuple[tuple[int, int], ...]
    plaquettes: tuple[tuple[int, ...], ...]
    logical: tuple[int, ...]


def triangular_color_code(distance: int) -> TriangularColorCode:
    r"""Build the triangular 6.6.6 color code with :math:`(3d^2 + 1)/4` qubits.

    Points :math:`(i, j)` with :math:`i, j \ge 0` and :math:`i + j \le 3(d-1)/2`
    form a triangular lattice colored by :math:`(i - j) \bmod 3`. One color
    holds the plaquettes and the other two the qubits, so a plaquette acts on
    its lattice neighbors: six in the bulk and four on a boundary. The logical
    operator is the boundary :math:`j = 0`.

    Returns
    -------
    `TriangularColorCode`
        Qubit coordinates, plaquette supports in cyclic order, and the logical support.

    Raises
    ------
    ValueError
        If the distance is even or smaller than 3.
    """
    if distance < _MIN_DISTANCE or distance % 2 == 0:
        msg = "The triangular color code needs an odd distance of at least 3."
        raise ValueError(msg)
    size = 3 * (distance - 1) // 2
    points = [(i, j) for i in range(size + 1) for j in range(size + 1 - i)]
    index = {point: qubit for qubit, point in enumerate(p for p in points if (p[0] - p[1]) % 3 != _FACE_COLOR)}
    plaquettes = tuple(
        tuple(index[i + di, j + dj] for di, dj in _NEIGHBORS if (i + di, j + dj) in index)
        for i, j in points
        if (i - j) % 3 == _FACE_COLOR
    )
    return TriangularColorCode(
        distance=distance,
        coordinates=tuple((2 * i + j, 2 * j) for i, j in index),
        plaquettes=plaquettes,
        logical=tuple(qubit for (_, j), qubit in index.items() if j == 0),
    )


def memory_circuit(code: TriangularColorCode, rounds: int, basis: Literal["X", "Z"] = "Z") -> stim.Circuit:
    """Write a noiseless memory experiment whose rounds are plaquette MPPs.

    Each round measures every X and Z plaquette in one ``MPP`` instruction.
    Detectors compare consecutive rounds, the first round and the final data
    readout check the plaquettes of the memory basis, and observable 0 is the
    logical operator of that basis.

    Returns
    -------
    `stim.Circuit`
        Circuit to import with ``mpp_ancilla="single"`` or ``"cat"``.

    Raises
    ------
    ValueError
        If rounds is not positive or basis is not X or Z.
    """
    if rounds < 1:
        msg = "A memory experiment needs at least one round."
        raise ValueError(msg)
    if basis not in {"X", "Z"}:
        msg = "The memory basis must be X or Z."
        raise ValueError(msg)
    num_qubits = len(code.coordinates)
    count = len(code.plaquettes)
    qubits = " ".join(map(str, range(num_qubits)))
    other = "X" if basis == "Z" else "Z"
    products = " ".join(
        "*".join(f"{pauli}{qubit}" for qubit in plaquette) for pauli in (other, basis) for plaquette in code.plaquettes
    )
    lines = [f"QUBIT_COORDS({x}, {y}) {qubit}" for qubit, (x, y) in enumerate(code.coordinates)]
    lines.append(f"R{'X' if basis == 'X' else ''} {qubits}")
    for round_index in range(rounds):
        lines.extend(("TICK", f"MPP {products}"))
        # This round's records: the other type first, then the memory type.
        for plaquette in range(count):
            memory = plaquette - count
            if round_index == 0:
                lines.append(f"DETECTOR rec[{memory}]")
            else:
                lines.extend(
                    (
                        f"DETECTOR rec[{memory - count}] rec[{memory - 3 * count}]",
                        f"DETECTOR rec[{memory}] rec[{memory - 2 * count}]",
                    )
                )
    lines.extend(("TICK", f"M{'X' if basis == 'X' else ''} {qubits}"))
    for plaquette, support in enumerate(code.plaquettes):
        data = " ".join(f"rec[{qubit - num_qubits}]" for qubit in support)
        lines.append(f"DETECTOR rec[{plaquette - count - num_qubits}] {data}")
    lines.append("OBSERVABLE_INCLUDE(0) " + " ".join(f"rec[{qubit - num_qubits}]" for qubit in code.logical))
    return stim.Circuit("\n".join(lines))


def random_cz_order(ir: StimImportIR, seed: int) -> list[tuple[int, int]]:
    r"""Shuffle the graph edges of an imported IR reproducibly.

    Returns
    -------
    `list`\[`tuple`\[`int`, `int`\]\]
        Every graph edge once, in an order fixed by seed.
    """
    edges = sorted(ir.graph.edges)
    return [edges[index] for index in np.random.default_rng(seed).permutation(len(edges))]


def pattern_with_cz_order(ir: StimImportIR, edge_order: Sequence[tuple[int, int]]) -> Pattern:
    """Lower an imported IR with one CZ per time slice, in the given order.

    Every node is prepared in slice 0 and measured after the last CZ in
    dependency order, so ``edge_order`` alone decides how a fault spreads
    through later CZ gates.

    Returns
    -------
    `Pattern`
        Pattern whose ``E`` commands follow ``edge_order``.

    Raises
    ------
    ValueError
        If edge_order does not list every graph edge exactly once.
    """
    edges = {frozenset(edge) for edge in ir.graph.edges}
    if len(edge_order) != len(edges) or {frozenset(edge) for edge in edge_order} != edges:
        msg = "edge_order must list every graph edge exactly once."
        raise ValueError(msg)
    scheduler = ir.make_scheduler()
    entangle_time = {edge: time for time, edge in enumerate(edge_order, start=1)}
    measure_order = TopologicalSorter(inverse_dag_from_dag(scheduler.dag)).static_order()
    measure_time = {
        node: time
        for time, node in enumerate(measure_order, start=len(entangle_time) + 1)
        if node in scheduler.measure_time
    }
    scheduler.manual_schedule(dict.fromkeys(scheduler.prepare_time, 0), measure_time, entangle_time)
    return ir.to_pattern(scheduler).pattern


def noisy_circuit(pattern: Pattern, p: float = 1e-3) -> stim.Circuit:
    """Export a pattern with depolarizing noise after each preparation and CZ, and measurement flips.

    Returns
    -------
    `stim.Circuit`
        Noisy circuit with the pattern's detectors and observables.
    """
    noise_models = [DepolarizingNoiseModel(p1=p, p2=p), MeasurementFlipNoiseModel(p=p)]
    return stim.Circuit(stim_compile(pattern, noise_models=noise_models))
