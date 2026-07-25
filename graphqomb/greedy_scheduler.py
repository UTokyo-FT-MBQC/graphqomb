"""Greedy heuristic scheduler for fast MBQC pattern scheduling.

This module provides fast greedy scheduling algorithms as an alternative to
CP-SAT based optimization. The greedy algorithms provide approximate solutions
with speedup compared to CP-SAT, making them suitable for large-scale
graphs or when optimality is not critical.

This module provides:

- `greedy_minimize_time`: Fast greedy scheduler optimizing for minimal execution time
- `greedy_minimize_space`: Fast greedy scheduler optimizing for minimal qubit usage
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from graphqomb.feedforward import TOPO_ORDER_CYCLE_ERROR_MSG, inverse_dag_from_dag, topo_order_from_inv_dag
from graphqomb.graphstate import unmeasured_output_nodes

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    from graphqomb.graphstate import BaseGraphState


def greedy_minimize_time(  # ruff:ignore[too-many-locals]
    graph: BaseGraphState,
    dag: Mapping[int, AbstractSet[int]],
    max_qubit_count: int | None = None,
) -> tuple[dict[int, int], dict[int, int]]:
    r"""Fast greedy scheduler optimizing for minimal execution time (makespan).

    This algorithm uses a single slice-by-slice strategy with slack-filling.
    If `max_qubit_count` is `None`, it is treated as no active-qubit limit.

    At each time slice, scheduling proceeds in two phases:

    1. Phase 1 (measurement phase): Measure every currently ready node.
       A node is ready when all DAG parents are already measured, all graph
       neighbors are prepared, and (for non-input nodes) the node itself is prepared.
    2. Phase 2 (preparation phase): Use remaining qubit capacity to prepare
       high-priority unprepared nodes that are likely to unblock future
       measurements (slack-filling).

    Parameters
    ----------
    graph : `BaseGraphState`
        The graph state to schedule
    dag : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        The directed acyclic graph representing measurement dependencies
    max_qubit_count : `int` | `None`, optional
        Maximum allowed number of active qubits. If None, no limit is enforced.

    Returns
    -------
    `tuple`\[`dict`\[`int`, `int`\], `dict`\[`int`, `int`\]\]
        A tuple of (prepare_time, measure_time) dictionaries

    Raises
    ------
    RuntimeError
        If the scheduling cannot proceed due to cyclic dependencies
        or if the max_qubit_count constraint is too tight to allow any progress.
    """
    unmeasured_outputs = unmeasured_output_nodes(graph)
    unmeasured = graph.nodes - unmeasured_outputs
    input_nodes = set(graph.input_node_indices.keys())

    inv_dag = inverse_dag_from_dag(dag, graph.nodes)

    # Cache neighbors to avoid repeated set constructions in tight loops
    neighbors_map = {node: graph.neighbors(node) for node in graph.nodes}

    # Single implementation for both bounded and unbounded capacity modes.
    prepare_time: dict[int, int] = {}
    measure_time: dict[int, int] = {}

    # Make mutable copies
    inv_dag_mut: dict[int, set[int]] = {node: set(parents) for node, parents in inv_dag.items()}
    unmeasured_mut: set[int] = set(unmeasured)

    prepared: set[int] = set(input_nodes)
    alive: set[int] = set(input_nodes)

    effective_max_qubit_count = max_qubit_count if max_qubit_count is not None else len(graph.nodes)

    if len(alive) > effective_max_qubit_count:
        msg = "Initial number of active qubits exceeds max_qubit_count."
        raise RuntimeError(msg)

    # Compute criticality for prioritizing preparations
    criticality = _compute_criticality(dag, unmeasured_outputs)

    current_time = 0

    while unmeasured_mut:
        ready_to_measure = _phase1_measure_ready_nodes(
            current_time,
            dag=dag,
            inv_dag=inv_dag_mut,
            neighbors_map=neighbors_map,
            input_nodes=input_nodes,
            prepared=prepared,
            alive=alive,
            unmeasured=unmeasured_mut,
            measure_time=measure_time,
        )
        prepared_in_phase2 = _phase2_prepare_nodes_with_slack(
            current_time,
            nodes=graph.nodes,
            max_qubit_count=effective_max_qubit_count,
            inv_dag=inv_dag_mut,
            neighbors_map=neighbors_map,
            prepared=prepared,
            alive=alive,
            unmeasured=unmeasured_mut,
            criticality=criticality,
            prepare_time=prepare_time,
        )

        # Check if we made progress
        if not ready_to_measure and not prepared_in_phase2 and unmeasured_mut:
            if current_time == 0 and _has_initial_input_input_entanglement_wait(
                unmeasured_mut,
                inv_dag_mut,
                neighbors_map,
                input_nodes,
                prepared,
            ):
                pass
            elif max_qubit_count is None:
                raise RuntimeError(TOPO_ORDER_CYCLE_ERROR_MSG)
            else:
                # No measurements and no room to prepare under qubit-capacity constraint.
                msg = (
                    "Cannot schedule more measurements without exceeding max qubit count. "
                    "Please increase max_qubit_count."
                )
                raise RuntimeError(msg)

        current_time += 1

        # Safety check for infinite loops
        if current_time > len(graph.nodes) * 2:
            msg = "Scheduling did not converge; possible cyclic dependency."
            raise RuntimeError(msg)

    _prepare_remaining_nodes_at_tail(
        current_time,
        nodes=graph.nodes,
        input_nodes=input_nodes,
        max_qubit_count=effective_max_qubit_count,
        prepared=prepared,
        alive=alive,
        prepare_time=prepare_time,
    )

    # Apply ALAP post-processing to minimize active volume
    prepare_time = alap_prepare_times(graph, prepare_time, measure_time)

    return prepare_time, measure_time


def _prepare_remaining_nodes_at_tail(  # ruff:ignore[too-many-arguments]
    current_time: int,
    *,
    nodes: AbstractSet[int],
    input_nodes: AbstractSet[int],
    max_qubit_count: int,
    prepared: set[int],
    alive: set[int],
    prepare_time: dict[int, int],
) -> None:
    # Any remaining nodes are outputs that were never needed for a measurement.
    # With no future measurements to free qubits, they must all fit in the final live set.
    remaining = nodes - input_nodes - prepared
    if not remaining:
        return

    if len(alive) + len(remaining) > max_qubit_count:
        msg = (
            "Cannot prepare remaining output nodes without exceeding max qubit count. Please increase max_qubit_count."
        )
        raise RuntimeError(msg)

    for node in sorted(remaining):
        prepare_time[node] = current_time
    prepared.update(remaining)
    alive.update(remaining)


def _phase1_measure_ready_nodes(  # ruff:ignore[too-many-arguments]
    current_time: int,
    *,
    dag: Mapping[int, AbstractSet[int]],
    inv_dag: dict[int, set[int]],
    neighbors_map: Mapping[int, AbstractSet[int]],
    input_nodes: AbstractSet[int],
    prepared: set[int],
    alive: set[int],
    unmeasured: set[int],
    measure_time: dict[int, int],
) -> set[int]:
    # Phase 1: measure all currently ready nodes.
    ready_to_measure: set[int] = set()
    for node in unmeasured:
        if inv_dag[node]:
            continue
        if not neighbors_map[node] <= prepared:
            continue
        if node not in input_nodes and node not in prepared:
            continue
        if current_time == 0 and _needs_initial_input_input_entanglement_wait(node, neighbors_map, input_nodes):
            continue
        ready_to_measure.add(node)

    for node in ready_to_measure:
        measure_time[node] = current_time
        unmeasured.remove(node)
        alive.discard(node)
        for child in dag.get(node, ()):
            inv_dag[child].discard(node)

    return ready_to_measure


def _needs_initial_input_input_entanglement_wait(
    node: int,
    neighbors_map: Mapping[int, AbstractSet[int]],
    input_nodes: AbstractSet[int],
) -> bool:
    return node in input_nodes and bool(neighbors_map[node] & input_nodes)


def _has_initial_input_input_entanglement_wait(
    unmeasured: AbstractSet[int],
    inv_dag: Mapping[int, AbstractSet[int]],
    neighbors_map: Mapping[int, AbstractSet[int]],
    input_nodes: AbstractSet[int],
    prepared: AbstractSet[int],
) -> bool:
    for node in unmeasured:
        if inv_dag[node]:
            continue
        if not neighbors_map[node] <= prepared:
            continue
        if _needs_initial_input_input_entanglement_wait(node, neighbors_map, input_nodes):
            return True
    return False


def _phase2_prepare_nodes_with_slack(  # ruff:ignore[too-many-arguments]
    current_time: int,
    *,
    nodes: AbstractSet[int],
    max_qubit_count: int,
    inv_dag: Mapping[int, AbstractSet[int]],
    neighbors_map: Mapping[int, AbstractSet[int]],
    prepared: set[int],
    alive: set[int],
    unmeasured: AbstractSet[int],
    criticality: Mapping[int, int],
    prepare_time: dict[int, int],
) -> bool:
    # Phase 2: prepare only nodes that can unblock the current DAG frontier.
    free_capacity = max_qubit_count - len(alive)
    if free_capacity <= 0:
        return False

    unprepared = nodes - prepared
    if not unprepared:
        return False

    prep_candidates = _get_prep_candidates_with_priority(
        inv_dag,
        neighbors_map,
        prepared,
        unmeasured,
        criticality,
    )

    prepared_in_phase2 = False
    for candidate, score in prep_candidates:
        if free_capacity <= 0 or score <= 0:
            break
        prepare_time[candidate] = current_time
        prepared.add(candidate)
        alive.add(candidate)
        prepared_in_phase2 = True
        free_capacity -= 1

    return prepared_in_phase2


def _compute_criticality(
    dag: Mapping[int, AbstractSet[int]],
    unmeasured_outputs: AbstractSet[int],
) -> dict[int, int]:
    # Compute criticality (remaining DAG depth) for each node.
    # Nodes with higher criticality should be prioritized for unblocking.
    criticality: dict[int, int] = {}

    # For criticality we need children first, so reverse dependency-first topo order.
    topo_order = topo_order_from_inv_dag(inverse_dag_from_dag(dag))
    topo_order.reverse()

    for node in topo_order:
        children_crits = [criticality.get(c, 0) for c in dag.get(node, ())]
        criticality[node] = 1 + max(children_crits, default=0)

    # Unmeasured output nodes have criticality 0 (they don't need to be measured)
    for node in unmeasured_outputs:
        criticality[node] = 0

    return criticality


def _get_prep_candidates_with_priority(
    inv_dag: Mapping[int, AbstractSet[int]],
    neighbors_map: Mapping[int, AbstractSet[int]],
    prepared: AbstractSet[int],
    unmeasured: AbstractSet[int],
    criticality: Mapping[int, int],
) -> list[tuple[int, float]]:
    # Get preparation candidates sorted by priority score.
    # Priority is based on how much preparing a node helps unblock measurements.
    # Find nodes that are DAG-ready but blocked by missing neighbors
    dag_ready_blocked: set[int] = set()
    missing_map: dict[int, set[int]] = {}

    for node in unmeasured:
        if inv_dag[node]:
            continue  # Not DAG-ready
        missing = set(neighbors_map[node]) - set(prepared)
        # Also check if the node itself needs preparation
        if node not in prepared:
            missing.add(node)
        if missing:
            dag_ready_blocked.add(node)
            missing_map[node] = missing

    if not dag_ready_blocked:
        return []

    useful_candidates = set().union(*missing_map.values())

    # Score each unprepared node
    scores: list[tuple[int, float]] = []
    for candidate in useful_candidates:
        score = 0.0
        for blocked_node in dag_ready_blocked:
            if candidate in missing_map[blocked_node]:
                crit = criticality.get(blocked_node, 1)
                score += crit / len(missing_map[blocked_node])

        scores.append((candidate, score))

    # Sort by score descending (higher score = higher priority).
    # Break ties by node id for deterministic schedules.
    scores.sort(key=lambda item: (-item[1], item[0]))

    return scores


def greedy_minimize_space(  # ruff:ignore[too-many-locals]
    graph: BaseGraphState,
    dag: Mapping[int, AbstractSet[int]],
) -> tuple[dict[int, int], dict[int, int]]:
    r"""Fast greedy scheduler optimizing for minimal qubit usage (space).

    This algorithm uses a greedy approach to minimize the number of active
    qubits at each time step:
    1. At each time step, select the next node to measure that minimizes the
       projected number of alive qubits after any required preparations.
    2. Prepare neighbors of the measured node just before measurement.

    Parameters
    ----------
    graph : `BaseGraphState`
        The graph state to schedule
    dag : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        The directed acyclic graph representing measurement dependencies

    Returns
    -------
    `tuple`\[`dict`\[`int`, `int`\], `dict`\[`int`, `int`\]\]
        A tuple of (prepare_time, measure_time) dictionaries

    Raises
    ------
    RuntimeError
        If no nodes can be measured at a given time step, indicating a possible
        cyclic dependency or incomplete preparation.
    """
    prepare_time: dict[int, int] = {}
    measure_time: dict[int, int] = {}

    unmeasured = graph.nodes - unmeasured_output_nodes(graph)

    inv_dag = inverse_dag_from_dag(dag, graph.nodes)
    topo_order = topo_order_from_inv_dag(inv_dag)  # from parents to children
    topo_rank = {node: i for i, node in enumerate(topo_order)}

    input_nodes = set(graph.input_node_indices.keys())
    prepared: set[int] = set(input_nodes)
    alive: set[int] = set(input_nodes)
    current_time = 0

    # Cache neighbors once as the graph is static during scheduling
    neighbors_map = {node: graph.neighbors(node) for node in graph.nodes}

    measure_candidates: set[int] = {node for node in unmeasured if not inv_dag[node]}

    while unmeasured:
        if not measure_candidates:
            raise RuntimeError(TOPO_ORDER_CYCLE_ERROR_MSG)

        # calculate costs and pick the best node to measure
        default_rank = len(topo_rank)
        candidates = iter(measure_candidates)
        best_node = next(candidates)
        best_cost = _calc_activate_cost(best_node, neighbors_map, prepared, alive, input_nodes)
        best_rank = topo_rank.get(best_node, default_rank)
        for node in candidates:
            cost = _calc_activate_cost(node, neighbors_map, prepared, alive, input_nodes)
            rank = topo_rank.get(node, default_rank)
            if cost < best_cost or (cost == best_cost and rank < best_rank):
                best_cost = cost
                best_rank = rank
                best_node = node

        # Prepare neighbors and the node itself (if non-input) at current_time
        new_neighbors = neighbors_map[best_node] - prepared
        needs_self_prep = best_node not in input_nodes and best_node not in prepared
        to_prepare = new_neighbors | ({best_node} if needs_self_prep else set())
        needs_prep = bool(to_prepare)
        if needs_prep:
            for node_to_prep in to_prepare:
                prepare_time[node_to_prep] = current_time
            prepared.update(to_prepare)
            alive.update(to_prepare)

        # Measure at current_time if no prep or initial input-input entanglement wait is needed.
        needs_initial_entangle_wait = current_time == 0 and _needs_initial_input_input_entanglement_wait(
            best_node, neighbors_map, input_nodes
        )
        meas_time = current_time + 1 if needs_prep or needs_initial_entangle_wait else current_time
        measure_time[best_node] = meas_time
        unmeasured.remove(best_node)
        alive.remove(best_node)

        measure_candidates.remove(best_node)

        # Remove measured node from dependencies of all its children in the DAG
        for child in dag.get(best_node, ()):
            inv_dag[child].remove(best_node)
            if not inv_dag[child] and child in unmeasured:
                measure_candidates.add(child)

        current_time = meas_time + 1

    # Any nodes left here were never needed to unlock a measurement.
    remaining = graph.nodes - input_nodes - prepared
    for node in sorted(remaining):
        prepare_time[node] = current_time

    return prepare_time, measure_time


def _calc_activate_cost(
    node: int,
    neighbors_map: Mapping[int, AbstractSet[int]],
    prepared: AbstractSet[int],
    alive: AbstractSet[int],
    input_nodes: AbstractSet[int],
) -> int:
    r"""Calculate the projected number of alive qubits if measuring this node next.

    If neighbors or the node itself must be prepared, they become alive at the
    current time slice while the node itself remains alive until the next slice.
    If no preparation is needed, the node is measured in the current slice and removed.

    Parameters
    ----------
    node : `int`
        The node to evaluate.
    neighbors_map : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        Cached neighbor sets for graph nodes.
    prepared : `collections.abc.Set`\[`int`\]
        The set of currently prepared nodes.
    alive : `collections.abc.Set`\[`int`\]
        The set of currently active (prepared but not yet measured) nodes.
    input_nodes : `collections.abc.Set`\[`int`\]
        The set of input nodes (already prepared at the start).

    Returns
    -------
    `int`
        The activation cost for the node.
    """
    new_neighbors = neighbors_map[node] - prepared
    needs_self_prep = node not in input_nodes and node not in prepared
    num_to_prepare = len(new_neighbors) + (1 if needs_self_prep else 0)
    if num_to_prepare > 0:
        return len(alive) + num_to_prepare
    # No preparation needed -> node is measured in the current slice, so alive decreases by 1.
    return max(len(alive) - 1, 0)


def _delay_prepare_times(
    prepare_time: Mapping[int, int],
    deadline: Mapping[int, int],
) -> dict[int, int]:
    delayed_prepare_time: dict[int, int] = {}
    for node, original_time in prepare_time.items():
        latest_time = deadline.get(node)
        if latest_time is None:
            delayed_prepare_time[node] = original_time
            continue
        if latest_time < original_time:
            msg = f"ALAP would move node {node} earlier than its existing preparation time."
            raise RuntimeError(msg)
        delayed_prepare_time[node] = latest_time
    return delayed_prepare_time


def alap_prepare_times(
    graph: BaseGraphState,
    prepare_time: Mapping[int, int],
    measure_time: Mapping[int, int],
) -> dict[int, int]:
    r"""Recompute preparation times using ALAP (As Late As Possible) strategy.

    Given fixed measurement times and an existing valid preparation schedule,
    this delays each prepared node as late as possible while respecting the
    constraint that all neighbors must be prepared before a node is measured.

    This post-processing reduces active volume (sum of qubit lifetimes) without
    changing the measurement schedule or depth.

    Parameters
    ----------
    graph : `BaseGraphState`
        The graph state
    prepare_time : `collections.abc.Mapping`\[`int`, `int`\]
        Existing preparation times for non-input nodes
    measure_time : `collections.abc.Mapping`\[`int`, `int`\]
        Fixed measurement times for non-output nodes

    Returns
    -------
    `dict`\[`int`, `int`\]
        ALAP preparation times for non-input nodes
    """
    input_nodes = set(graph.input_node_indices.keys())

    # deadline[v] = latest time v can be prepared
    deadline: dict[int, int] = {}

    # Each measured node and its neighbors must be prepared before that measurement.
    for node, meas_t in measure_time.items():
        for dependency in itertools.chain((node,), graph.neighbors(node)):
            if dependency in input_nodes:
                continue  # Input nodes don't need prep
            current_deadline = deadline.get(dependency)
            if current_deadline is None or (meas_t - 1) < current_deadline:
                deadline[dependency] = meas_t - 1

    return _delay_prepare_times(prepare_time, deadline)
