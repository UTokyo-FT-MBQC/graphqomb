"""Quantum Compiler(qompiler) module for Measurement-Based Quantum Computing (MBQC).

note: ``compile`` is used in Python built-in functions, so we use `qompile` instead.

This module provides:

- `qompile`: Compile graph state into a pattern with x/z correction flows.
"""

from __future__ import annotations

from graphlib import TopologicalSorter
from typing import TYPE_CHECKING

from graphqomb.command import TICK, Command, E, M, N
from graphqomb.feedforward import check_flow, dag_from_flow
from graphqomb.graphstate import odd_neighbors
from graphqomb.pattern import Pattern
from graphqomb.pauli_frame import PauliFrame
from graphqomb.scheduler import Scheduler

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from collections.abc import Set as AbstractSet

    from graphqomb.graphstate import BaseGraphState


def qompile(  # ruff:ignore[too-many-arguments]
    graph: BaseGraphState,
    xflow: Mapping[int, AbstractSet[int]],
    zflow: Mapping[int, AbstractSet[int]] | None = None,
    *,
    parity_check_group: Sequence[AbstractSet[int]] | None = None,
    logical_observables: Mapping[int, AbstractSet[int]] | None = None,
    scheduler: Scheduler | None = None,
) -> Pattern:
    r"""Compile graph state into pattern with x/z correction flows.

    Parameters
    ----------
    graph : `BaseGraphState`
        graph state
    xflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        x correction flow
    zflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`
        z correction flow
        if `None`, it is generated from xflow by odd neighbors
    parity_check_group : `collections.abc.Sequence`\[`collections.abc.Set`\[`int`\]\] | `None`
        parity check group for FTQC
    logical_observables : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`
        logical observables represented by logical index and seed nodes
    scheduler : `Scheduler` | `None`, optional
        scheduler to schedule the graph state preparation and measurements,
        if `None`, a `Scheduler` is constructed internally and solved with the
        default ``MINIMIZE_TIME`` strategy before compiling the pattern,
        otherwise the provided scheduler is validated before compiling the pattern,
        by default `None`

    Returns
    -------
    `Pattern`
        compiled pattern
    """
    graph.check_canonical_form()
    if zflow is None:
        zflow = {node: odd_neighbors(xflow[node], graph) for node in xflow}
    check_flow(graph, xflow, zflow)

    pauli_frame = PauliFrame(
        graph,
        xflow,
        zflow,
        parity_check_group=parity_check_group,
        logical_observables=logical_observables,
    )

    return _qompile(graph, pauli_frame, scheduler=scheduler)


def _qompile(
    graph: BaseGraphState,
    pauli_frame: PauliFrame,
    *,
    scheduler: Scheduler | None = None,
) -> Pattern:
    """Compile graph state into pattern with a given Pauli frame.

    note: This is an internal function of `qompile`.

    Parameters
    ----------
    graph : `BaseGraphState`
        graph state
    pauli_frame : `PauliFrame`
        Pauli frame to track the Pauli state of each node
    scheduler : `Scheduler` | `None`, optional
        scheduler to schedule the graph state preparation and measurements,
        if `None`, a `Scheduler` is constructed internally and solved with the
        default ``MINIMIZE_TIME`` strategy before compiling the pattern,
        otherwise the provided scheduler is validated before compiling the pattern,
        by default `None`

    Returns
    -------
    `Pattern`
        compiled pattern
    """
    meas_bases = graph.meas_bases
    graph_coords = graph.coordinates

    dag = dag_from_flow(graph, xflow=pauli_frame.xflow, zflow=pauli_frame.zflow)
    topo_order = list(TopologicalSorter(dag).static_order())
    topo_order.reverse()  # children first

    commands: list[Command] = []
    if scheduler is None:
        scheduler = Scheduler(graph, pauli_frame.xflow, pauli_frame.zflow)
        scheduler.solve_schedule()
    else:
        scheduler.validate_schedule()

    timeline = scheduler.timeline

    for time_idx in range(scheduler.num_slices()):
        prepare_nodes, entangle_edges, measure_nodes = timeline[time_idx]

        # Order within time slice: N -> E -> M
        # N commands include coordinates if available
        for node in prepare_nodes:
            coord = graph_coords.get(node)
            commands.append(N(node, coordinate=coord))
        for edge in entangle_edges:
            a, b = edge
            commands.append(E(nodes=(a, b)))
        commands.extend(M(node, meas_bases[node]) for node in measure_nodes)

        # Insert TICK between time slices
        commands.append(TICK())

    # Collect input node coordinates
    input_coords = {node: graph_coords[node] for node in graph.input_node_indices if node in graph_coords}

    return Pattern(
        input_node_indices=graph.input_node_indices,
        output_node_indices=graph.output_node_indices,
        commands=tuple(commands),
        pauli_frame=pauli_frame,
        input_coordinates=input_coords,
        input_initialization_axes=graph.input_initialization_axes,
    )
