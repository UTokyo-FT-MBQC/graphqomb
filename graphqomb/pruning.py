"""Pruning of MBQC compile inputs.

Measuring a graph-state node in the Z basis simply disentangles it, and a
Z-prepared input never interacts with its CZ edges, so both kinds of node can
be deleted from the resource graph before compilation. Dropping such a node
from the correction flows, parity checks, and logical observables at the same
time preserves the semantics of the pattern: in the pruned pattern the node is
never prepared or entangled, and every parity-check or observable product
loses exactly the Z factor that the deleted node supplied.

Deleting Z-basis nodes can disconnect the graph. A connected component that
contains neither an output node nor a logical observable seed cannot influence
the logical outcome, so it can be deleted as well. Such a component is not
entangled with the rest of the graph, so a deterministic parity check that
mixes its records with kept records factors into two independently
deterministic halves and stays deterministic after the deleted half is
dropped.

This module provides:

- `PruneResult`: Result of pruning nodes from a set of compile inputs.
- `prune_z_nodes`: Remove Z-prepared/Z-measured nodes from a graph state and
  the associated flows, parity checks, and logical observables.
- `prune_isolated_components`: Remove connected components that touch neither
  an output node nor a logical observable seed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from graphqomb.common import Axis, determine_pauli_axis
from graphqomb.graphstate import GraphState, odd_neighbors

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from collections.abc import Set as AbstractSet

    from graphqomb.graphstate import BaseGraphState


class PruneResult(NamedTuple):
    r"""Result of pruning nodes from a set of compile inputs.

    Node indices are preserved: every kept node keeps the index, qubit index,
    initialization, measurement basis, and coordinate it had in the source
    graph.

    Attributes
    ----------
    graph : `GraphState`
        Pruned copy of the source graph state.
    xflow : `dict`\[`int`, `set`\[`int`\]\]
        X correction flow without the pruned nodes.
    zflow : `dict`\[`int`, `set`\[`int`\]\]
        Z correction flow without the pruned nodes.
    parity_check_group : `list`\[`set`\[`int`\]\]
        Parity check groups without the pruned nodes. Groups whose every node
        was pruned are dropped entirely.
    parity_check_tags : `list`\[`str`\]
        Tags aligned with ``parity_check_group`` after empty groups are
        dropped.
    logical_observables : `dict`\[`int`, `set`\[`int`\]\]
        Logical observable seed nodes without the pruned nodes. Entries are
        kept even when their seed set becomes empty so that logical indices
        stay stable.
    removed_nodes : `frozenset`\[`int`\]
        Nodes removed from the source graph.
    """

    graph: GraphState
    xflow: dict[int, set[int]]
    zflow: dict[int, set[int]]
    parity_check_group: list[set[int]]
    parity_check_tags: list[str]
    logical_observables: dict[int, set[int]]
    removed_nodes: frozenset[int]


def prune_z_nodes(  # ruff:ignore[too-many-arguments]
    graph: BaseGraphState,
    xflow: Mapping[int, AbstractSet[int]],
    zflow: Mapping[int, AbstractSet[int]] | None = None,
    *,
    parity_check_group: Sequence[AbstractSet[int]] | None = None,
    parity_check_tags: Sequence[str] | None = None,
    logical_observables: Mapping[int, AbstractSet[int]] | None = None,
) -> PruneResult:
    r"""Remove Z-prepared/Z-measured nodes from a set of compile inputs.

    A node is pruned when it is not an output node and it is either an input
    node initialized in the positive or negative Z eigenstate or a node
    measured along the Pauli-Z axis. The pruned node disappears from the
    graph together with its edges, from the flows both as a corrector and as
    a correction target, and from every parity check group and logical
    observable seed set. Output nodes are always kept so that the pattern
    keeps its logical qubit interface.

    The pieces returned by this function can be passed directly to
    `graphqomb.qompiler.qompile` or to `prune_isolated_components`.

    Parameters
    ----------
    graph : `BaseGraphState`
        graph state
    xflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        x correction flow
    zflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`
        z correction flow
        if `None`, it is generated from xflow by odd neighbors of the source
        graph before pruning
    parity_check_group : `collections.abc.Sequence`\[`collections.abc.Set`\[`int`\]\] | `None`
        parity check group for FTQC, by default `None` (no parity checks)
    parity_check_tags : `collections.abc.Sequence`\[`str`\] | `None`
        Stim-style tag per parity check group, aligned with
        ``parity_check_group``. If `None`, every group is untagged.
    logical_observables : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`
        logical observables represented by logical index and seed nodes,
        by default `None` (no logical observables)

    Returns
    -------
    `PruneResult`
        Pruned graph, flows, parity checks, and logical observables.

    Raises
    ------
    ValueError
        If ``parity_check_tags`` is given but not aligned with
        ``parity_check_group``.
    """  # ruff:ignore[docstring-extraneous-exception]
    return _pruned_inputs(
        graph,
        _z_basis_nodes(graph),
        xflow,
        zflow,
        parity_check_group=parity_check_group,
        parity_check_tags=parity_check_tags,
        logical_observables=logical_observables,
    )


def prune_isolated_components(  # ruff:ignore[too-many-arguments]
    graph: BaseGraphState,
    xflow: Mapping[int, AbstractSet[int]],
    zflow: Mapping[int, AbstractSet[int]] | None = None,
    *,
    parity_check_group: Sequence[AbstractSet[int]] | None = None,
    parity_check_tags: Sequence[str] | None = None,
    logical_observables: Mapping[int, AbstractSet[int]] | None = None,
) -> PruneResult:
    r"""Remove connected components that touch neither an output node nor a logical observable seed.

    Every node of such a component is pruned from the graph, from the flows,
    and from every parity check group; parity checks contained entirely in a
    pruned component disappear, including ``type=flag`` checks, so pruning can
    change the acceptance rate of post-selected sampling (only by dropping
    flags whose firing cannot affect the logical outcome). Logical observable
    seed sets are unchanged by construction.

    Note that when the inputs declare no logical observables and the graph has
    no output nodes, every component is considered irrelevant and the whole
    graph is pruned. This function is meant for compile inputs whose semantics
    are carried by logical observables or output qubits, not for
    detector-only experiments.

    The pieces returned by this function can be passed directly to
    `graphqomb.qompiler.qompile`.

    Parameters
    ----------
    graph : `BaseGraphState`
        graph state
    xflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        x correction flow
    zflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`
        z correction flow
        if `None`, it is generated from xflow by odd neighbors of the source
        graph before pruning
    parity_check_group : `collections.abc.Sequence`\[`collections.abc.Set`\[`int`\]\] | `None`
        parity check group for FTQC, by default `None` (no parity checks)
    parity_check_tags : `collections.abc.Sequence`\[`str`\] | `None`
        Stim-style tag per parity check group, aligned with
        ``parity_check_group``. If `None`, every group is untagged.
    logical_observables : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`
        logical observables represented by logical index and seed nodes,
        by default `None` (no logical observables)

    Returns
    -------
    `PruneResult`
        Pruned graph, flows, parity checks, and logical observables.

    Raises
    ------
    ValueError
        If ``parity_check_tags`` is given but not aligned with
        ``parity_check_group``.
    """  # ruff:ignore[docstring-extraneous-exception]
    relevant_nodes = set(graph.output_node_indices)
    if logical_observables is not None:
        for seed_nodes in logical_observables.values():
            relevant_nodes.update(seed_nodes)

    removed_nodes: set[int] = set()
    for component in _connected_components(graph):
        if not (component & relevant_nodes):
            removed_nodes |= component

    return _pruned_inputs(
        graph,
        removed_nodes,
        xflow,
        zflow,
        parity_check_group=parity_check_group,
        parity_check_tags=parity_check_tags,
        logical_observables=logical_observables,
    )


def _pruned_inputs(  # ruff:ignore[too-many-arguments]
    graph: BaseGraphState,
    removed_nodes: AbstractSet[int],
    xflow: Mapping[int, AbstractSet[int]],
    zflow: Mapping[int, AbstractSet[int]] | None,
    *,
    parity_check_group: Sequence[AbstractSet[int]] | None,
    parity_check_tags: Sequence[str] | None,
    logical_observables: Mapping[int, AbstractSet[int]] | None,
) -> PruneResult:
    r"""Drop the given nodes from every compile input.

    Parameters
    ----------
    graph : `BaseGraphState`
        graph state
    removed_nodes : `collections.abc.Set`\[`int`\]
        nodes to drop
    xflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        x correction flow
    zflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`
        z correction flow, derived by odd neighbors of the source graph when
        `None`
    parity_check_group : `collections.abc.Sequence`\[`collections.abc.Set`\[`int`\]\] | `None`
        parity check group for FTQC
    parity_check_tags : `collections.abc.Sequence`\[`str`\] | `None`
        Stim-style tag per parity check group
    logical_observables : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`
        logical observables represented by logical index and seed nodes

    Returns
    -------
    `PruneResult`
        Pruned graph, flows, parity checks, and logical observables.

    Raises
    ------
    ValueError
        If ``parity_check_tags`` is given but not aligned with
        ``parity_check_group``.
    """
    if parity_check_group is None:
        parity_check_group = []
    if parity_check_tags is None:
        parity_check_tags = [""] * len(parity_check_group)
    elif len(parity_check_tags) != len(parity_check_group):
        msg = (
            f"parity_check_tags has {len(parity_check_tags)} tag(s) "
            f"for {len(parity_check_group)} parity check group(s)."
        )
        raise ValueError(msg)
    if logical_observables is None:
        logical_observables = {}
    if zflow is None:
        zflow = {node: odd_neighbors(xflow[node], graph) for node in xflow}

    removed_nodes = frozenset(removed_nodes)

    pruned_group: list[set[int]] = []
    pruned_tags: list[str] = []
    for check, tag in zip(parity_check_group, parity_check_tags, strict=True):
        pruned_check = set(check) - removed_nodes
        if pruned_check:
            pruned_group.append(pruned_check)
            pruned_tags.append(tag)
    pruned_observables = {
        logical_idx: set(seed_nodes) - removed_nodes for logical_idx, seed_nodes in logical_observables.items()
    }

    return PruneResult(
        graph=_copy_graph_without(graph, removed_nodes),
        xflow=_prune_flow(xflow, removed_nodes),
        zflow=_prune_flow(zflow, removed_nodes),
        parity_check_group=pruned_group,
        parity_check_tags=pruned_tags,
        logical_observables=pruned_observables,
        removed_nodes=removed_nodes,
    )


def _z_basis_nodes(graph: BaseGraphState) -> set[int]:
    r"""Collect the non-output nodes that are Z-prepared or Z-measured.

    Parameters
    ----------
    graph : `BaseGraphState`
        graph state

    Returns
    -------
    `set`\[`int`\]
        Nodes to prune.
    """
    input_initializations = graph.input_initializations
    meas_bases = graph.meas_bases
    removed: set[int] = set()
    for node in graph.nodes - graph.output_node_indices.keys():
        initialization = input_initializations.get(node)
        if initialization is not None and initialization.axis == Axis.Z:
            removed.add(node)
            continue
        meas_basis = meas_bases.get(node)
        if meas_basis is not None and determine_pauli_axis(meas_basis) == Axis.Z:
            removed.add(node)
    return removed


def _connected_components(graph: BaseGraphState) -> list[set[int]]:
    r"""Split the graph into connected components.

    Parameters
    ----------
    graph : `BaseGraphState`
        graph state

    Returns
    -------
    `list`\[`set`\[`int`\]\]
        Node sets of the connected components.
    """
    components: list[set[int]] = []
    unvisited = graph.nodes
    while unvisited:
        component = {unvisited.pop()}
        frontier = set(component)
        while frontier:
            frontier = {neighbor for node in frontier for neighbor in graph.neighbors(node) if neighbor in unvisited}
            unvisited -= frontier
            component |= frontier
        components.append(component)
    return components


def _copy_graph_without(graph: BaseGraphState, removed_nodes: AbstractSet[int]) -> GraphState:
    r"""Copy a graph state while skipping the given nodes, preserving node indices.

    Parameters
    ----------
    graph : `BaseGraphState`
        source graph state
    removed_nodes : `collections.abc.Set`\[`int`\]
        nodes to skip

    Returns
    -------
    `GraphState`
        Copied graph state without the given nodes.
    """
    coordinates = graph.coordinates
    pruned = GraphState()
    for node in graph.nodes - removed_nodes:
        pruned.add_node(node, coordinate=coordinates.get(node))
    for node1, node2 in graph.edges:
        if node1 not in removed_nodes and node2 not in removed_nodes:
            pruned.add_edge(node1, node2)
    for output_node, q_index in graph.output_node_indices.items():
        pruned.register_output(output_node, q_index)
    _copy_node_annotations(graph, pruned, removed_nodes)
    return pruned


def _copy_node_annotations(graph: BaseGraphState, pruned: GraphState, removed_nodes: AbstractSet[int]) -> None:
    r"""Copy input registrations, measurement bases, and local Cliffords of kept nodes.

    Parameters
    ----------
    graph : `BaseGraphState`
        source graph state
    pruned : `GraphState`
        destination graph state holding the kept nodes
    removed_nodes : `collections.abc.Set`\[`int`\]
        nodes skipped in the destination graph
    """
    input_initializations = graph.input_initializations
    for input_node, q_index in graph.input_node_indices.items():
        if input_node not in removed_nodes:
            pruned.register_input(input_node, q_index, init=input_initializations.get(input_node))
    for node, meas_basis in graph.meas_bases.items():
        if node not in removed_nodes:
            pruned.assign_meas_basis(node, meas_basis)
    if isinstance(graph, GraphState):
        for node, local_clifford in graph.local_cliffords.items():
            if node not in removed_nodes:
                pruned.apply_local_clifford(node, local_clifford)


def _prune_flow(flow: Mapping[int, AbstractSet[int]], removed_nodes: AbstractSet[int]) -> dict[int, set[int]]:
    r"""Drop pruned nodes from a correction flow.

    Pruned nodes are dropped both as correctors (their entry disappears) and
    as correction targets (they are discarded from every target set).

    Parameters
    ----------
    flow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        correction flow
    removed_nodes : `collections.abc.Set`\[`int`\]
        nodes to drop

    Returns
    -------
    `dict`\[`int`, `set`\[`int`\]\]
        Pruned correction flow.
    """
    return {node: set(targets) - removed_nodes for node, targets in flow.items() if node not in removed_nodes}
