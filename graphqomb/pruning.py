"""Prune graph-state compile inputs before passing them to `qompile`.

`prune_z_nodes` removes eligible Z-basis nodes, while
`prune_isolated_components` removes components unrelated to outputs or
logical observables. Both update all compile inputs and preserve node indices.

This module provides:

- `PruneResult`: Pruned compile inputs and the removed nodes.
- `prune_z_nodes`: Remove eligible Z-prepared and Z-measured nodes.
- `prune_isolated_components`: Remove components unrelated to outputs or logical observables.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

from graphqomb.common import Axis, determine_pauli_axis, is_close_angle
from graphqomb.graphstate import GraphState, odd_neighbors

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from collections.abc import Set as AbstractSet

    from graphqomb.common import MeasBasis
    from graphqomb.graphstate import BaseGraphState


class PruneResult(NamedTuple):
    """Pruned compile inputs and the nodes removed from the source graph.

    Empty parity checks are dropped with their tags. Logical-observable entries
    remain present when their seed sets become empty, preserving logical indices.
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
    prune_preparations: bool = True,
    prune_measurements: bool = True,
    prune_uncorrected_measurements: bool = False,
    protected_preparation_tags: AbstractSet[str] | None = None,
) -> PruneResult:
    r"""Remove eligible Z-prepared and Z-measured nodes from compile inputs.

    Output nodes are always kept. Preparation and measurement pruning can be
    disabled independently, and protected_preparation_tags protects matching
    Z-prepared inputs. A node both prepared and measured in Z is governed by
    the preparation controls.

    Parameters
    ----------
    graph : `BaseGraphState`
        Source graph state.
    xflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        X correction flow.
    zflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`
        Z correction flow, derived from xflow when omitted.
    parity_check_group : `collections.abc.Sequence`\[`collections.abc.Set`\[`int`\]\] | `None`
        Parity checks to update.
    parity_check_tags : `collections.abc.Sequence`\[`str`\] | `None`
        Tags aligned with parity_check_group.
    logical_observables : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`
        Logical-observable seed nodes to update.
    prune_preparations : `bool`
        Whether to remove Z-prepared inputs.
    prune_measurements : `bool`
        Whether to remove corrected Z measurements.
    prune_uncorrected_measurements : `bool`
        Whether to also remove Z measurements whose byproducts the flows do
        not correct, fixing one measurement branch. The inverted-record rule
        in the Notes still applies. Has no effect when prune_measurements is
        false.
    protected_preparation_tags : `collections.abc.Set`\[`str`\] | `None`
        Initialization tags that protect Z-prepared inputs.

    Returns
    -------
    `PruneResult`
        Pruned inputs suitable for `qompile` or further pruning.

    Raises
    ------
    ValueError
        If parity_check_tags is given but not aligned with
        parity_check_group.

    Notes
    -----
    Removing a node keeps the branch of its measurement that projects it onto
    the +Z eigenstate, fixing its record at 0 for a positive Z basis and at 1
    for an inverted one (angle pi). A record fixed at 1 fires the corrections
    the node sources, so a node measured in an inverted Z basis is never
    removed unless those corrections form a stabilizer of the initialized
    graph state, up to vacuous Z factors on Z-basis nodes.

    A Z-measured node is removed only when, additionally, each measurement
    branch leaves the state of the pruned circuit: its sourced corrections and
    the Z byproducts on its neighbors must combine into such stabilizers
    branch by branch; otherwise removal would select a measurement branch.
    That branch condition preserves the compiled Kraus map but is unnecessary
    when nothing downstream depends on the measurement outcome;
    prune_uncorrected_measurements then selects the byproduct-free branch
    anyway, still subject to the inverted-record rule above. Parity checks and
    logical observables keep their values (up to a deterministic flip when the
    record is the constant 1), while outputs carry the single kept branch
    instead of the mixture over branches.

    A Z-prepared input never entangles, so it is pruned along with its sourced
    corrections whenever the inverted-record rule allows, fixing the branch of
    the dropped record in the same way.
    """  # ruff:ignore[docstring-extraneous-exception]
    if zflow is None:
        zflow = {node: odd_neighbors(xflow[node], graph) for node in xflow}
    prunable = _prunable_z_nodes(
        graph,
        xflow,
        zflow,
        prune_preparations=prune_preparations,
        prune_measurements=prune_measurements,
        prune_uncorrected_measurements=prune_uncorrected_measurements,
        protected_preparation_tags=protected_preparation_tags,
    )
    return _pruned_inputs(
        graph,
        prunable,
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
    r"""Remove components unrelated to outputs or logical-observable seeds.

    Graph edges and correction-flow entries both connect nodes. With no output
    nodes or logical observables, every component is removed; this operation is
    therefore not intended for detector-only experiments.

    Parameters
    ----------
    graph : `BaseGraphState`
        Source graph state.
    xflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        X correction flow.
    zflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`
        Explicit Z correction flow, if any.
    parity_check_group : `collections.abc.Sequence`\[`collections.abc.Set`\[`int`\]\] | `None`
        Parity checks to update.
    parity_check_tags : `collections.abc.Sequence`\[`str`\] | `None`
        Tags aligned with parity_check_group.
    logical_observables : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`
        Logical-observable seeds that make components relevant.

    Returns
    -------
    `PruneResult`
        Pruned inputs suitable for `qompile`.

    Raises
    ------
    ValueError
        If parity_check_tags is given but not aligned with
        parity_check_group.
    """  # ruff:ignore[docstring-extraneous-exception]
    relevant_nodes = set(graph.output_node_indices)
    if logical_observables is not None:
        for seed_nodes in logical_observables.values():
            relevant_nodes.update(seed_nodes)

    # A zflow derived from odd neighbors of xflow targets cannot connect
    # anything beyond graph edges and xflow entries.
    flows = [xflow] if zflow is None else [xflow, zflow]

    removed_nodes = graph.nodes - _reachable_nodes(graph, flows, relevant_nodes)

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


def _prunable_z_nodes(  # ruff:ignore[too-many-arguments]
    graph: BaseGraphState,
    xflow: Mapping[int, AbstractSet[int]],
    zflow: Mapping[int, AbstractSet[int]],
    *,
    prune_preparations: bool,
    prune_measurements: bool,
    prune_uncorrected_measurements: bool,
    protected_preparation_tags: AbstractSet[str] | None,
) -> set[int]:
    input_initializations = graph.input_initializations
    z_basis_nodes = _z_basis_nodes(graph)
    z_vacuous = z_basis_nodes | {v for v, init in input_initializations.items() if init.axis == Axis.Z}
    prunable: set[int] = set()
    for node in z_basis_nodes:
        initialization = input_initializations.get(node)
        if initialization is not None and initialization.axis == Axis.Z:
            protected = protected_preparation_tags is not None and initialization.tag in protected_preparation_tags
            if (
                prune_preparations
                and not protected
                and _fixed_record_corrections_vacuous(node, graph, xflow, zflow, z_vacuous)
            ):
                prunable.add(node)
        elif prune_measurements and _z_measurement_removable(
            node, graph, xflow, zflow, z_vacuous, force=prune_uncorrected_measurements
        ):
            prunable.add(node)
    return prunable


def _z_measurement_removable(  # ruff:ignore[too-many-arguments]
    node: int,
    graph: BaseGraphState,
    xflow: Mapping[int, AbstractSet[int]],
    zflow: Mapping[int, AbstractSet[int]],
    z_vacuous: set[int],
    *,
    force: bool,
) -> bool:
    applied_x = set(xflow.get(node, ())) - {node}
    applied_z = set(zflow.get(node, ())) - {node}
    neighbors = set(graph.neighbors(node))
    allowed = z_vacuous | {node}
    if _inverted_z_record(graph.meas_bases.get(node)):
        # The kept branch has record 1 and fires the sourced corrections.
        if not _vacuous_stabilizer(graph, applied_x, applied_z, allowed):
            return False
        # The dropped branch leaves the byproduct Z on the neighbors.
        return force or _vacuous_stabilizer(graph, set(), neighbors, allowed)
    # The kept branch has record 0 and matches the pruned circuit exactly;
    # the dropped branch fires the corrections and the byproduct together.
    return force or _vacuous_stabilizer(graph, applied_x, applied_z ^ neighbors, allowed)


def _fixed_record_corrections_vacuous(
    node: int,
    graph: BaseGraphState,
    xflow: Mapping[int, AbstractSet[int]],
    zflow: Mapping[int, AbstractSet[int]],
    z_vacuous: set[int],
) -> bool:
    if not _inverted_z_record(graph.meas_bases.get(node)):
        return True
    applied_x = set(xflow.get(node, ())) - {node}
    applied_z = set(zflow.get(node, ())) - {node}
    return _vacuous_stabilizer(graph, applied_x, applied_z, z_vacuous | {node})


def _vacuous_stabilizer(
    graph: BaseGraphState,
    x_support: AbstractSet[int],
    z_support: AbstractSet[int],
    allowed_z: AbstractSet[int],
) -> bool:
    input_initializations = graph.input_initializations
    x_support_axes = {v: input_initializations[v].axis for v in x_support if v in input_initializations}
    if Axis.Z in x_support_axes.values():
        return False
    y_initialized = {v for v, axis in x_support_axes.items() if axis == Axis.Y}
    mismatch = set(z_support) ^ odd_neighbors(x_support, graph) ^ y_initialized
    return mismatch <= allowed_z


def _inverted_z_record(meas_basis: MeasBasis | None) -> bool:
    if meas_basis is None:
        return False
    return determine_pauli_axis(meas_basis) == Axis.Z and is_close_angle(meas_basis.angle, math.pi)


def _z_basis_nodes(graph: BaseGraphState) -> set[int]:
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


def _reachable_nodes(
    graph: BaseGraphState,
    flows: Sequence[Mapping[int, AbstractSet[int]]],
    seeds: AbstractSet[int],
) -> set[int]:
    adjacency: dict[int, set[int]] = {node: set(graph.neighbors(node)) for node in graph.nodes}
    for flow in flows:
        for source, targets in flow.items():
            for target in targets:
                adjacency[source].add(target)
                adjacency[target].add(source)
    reachable = set(seeds) & adjacency.keys()
    frontier = set(reachable)
    while frontier:
        frontier = {neighbor for node in frontier for neighbor in adjacency[node] if neighbor not in reachable}
        reachable |= frontier
    return reachable


def _copy_graph_without(graph: BaseGraphState, removed_nodes: AbstractSet[int]) -> GraphState:
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
    return {node: set(targets) - removed_nodes for node, targets in flow.items() if node not in removed_nodes}
