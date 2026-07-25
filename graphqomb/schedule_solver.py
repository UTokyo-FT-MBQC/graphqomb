"""Scheduling solver for optimizing MBQC pattern execution.

This module provides:

- `Strategy`: Enumeration of scheduling optimization strategies
- `ScheduleConfig`: Configuration for scheduling parameters and constraints
- `solve_schedule`: Solve scheduling optimization using constraint programming
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from ortools.sat.python import cp_model

from graphqomb.graphstate import unmeasured_output_nodes

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    from graphqomb.graphstate import BaseGraphState


class Strategy(Enum):
    """Enumeration for scheduling strategies."""

    MINIMIZE_SPACE = enum.auto()
    MINIMIZE_TIME = enum.auto()


@dataclass
class ScheduleConfig:
    """Configuration for scheduling strategy, constraints, and parameters."""

    strategy: Strategy
    max_qubit_count: int | None = None
    max_time: int | None = None
    use_greedy: bool = False


@dataclass
class _ModelContext:
    """Internal context for model and graph data."""

    model: cp_model.CpModel
    graph: BaseGraphState


def _add_constraints(
    model: cp_model.CpModel,
    graph: BaseGraphState,
    dag: Mapping[int, AbstractSet[int]],
    node2prep: Mapping[int, cp_model.IntVar],
    node2meas: Mapping[int, cp_model.IntVar],
) -> None:
    """Add constraints to the scheduling model."""
    # Measurement order constraints
    for node, children in dag.items():
        for child in children:
            if node in node2meas and child in node2meas:
                model.add(node2meas[node] < node2meas[child])

    # A non-input, measured node must be prepared before it is measured.
    for node in graph.nodes:
        if node in node2prep and node in node2meas:
            model.add(node2prep[node] < node2meas[node])

    # Edge constraints
    for node in node2meas:
        for neighbor in graph.neighbors(node):
            if neighbor in graph.input_node_indices:
                if node in graph.input_node_indices:
                    model.add(node2meas[node] > 0)
                continue
            model.add(node2prep[neighbor] < node2meas[node])


def _set_objective(
    ctx: _ModelContext,
    node2prep: Mapping[int, cp_model.IntVar],
    node2meas: Mapping[int, cp_model.IntVar],
    config: ScheduleConfig,
    max_time: int,
) -> None:
    """Set the objective function for the scheduling model.

    Raises
    ------
    ValueError
        If the scheduling strategy is unknown.
    """
    if config.strategy == Strategy.MINIMIZE_SPACE:
        _set_minimize_space_objective(ctx, node2prep, node2meas, max_time)
    elif config.strategy == Strategy.MINIMIZE_TIME:
        _set_minimize_time_objective(ctx, node2prep, node2meas, max_time, config.max_qubit_count)
    else:
        msg = f"Unknown scheduling strategy: {config.strategy}"
        raise ValueError(msg)


def _compute_alive_nodes_at_time(
    ctx: _ModelContext,
    node2prep: Mapping[int, cp_model.IntVar],
    node2meas: Mapping[int, cp_model.IntVar],
    t: int,
) -> list[cp_model.IntVar]:
    """Compute the list of alive nodes at time t.

    Returns
    -------
    list[cp_model.IntVar]
        Boolean variables indicating whether each node is alive at time t.
    """
    alive_at_t: list[cp_model.IntVar] = []
    for node in ctx.graph.nodes:
        a_pre = ctx.model.new_bool_var(f"alive_pre_{node}_{t}")
        if node in ctx.graph.input_node_indices:
            ctx.model.add(a_pre == 1)
        else:
            p = node2prep[node]
            ctx.model.add(p <= t).only_enforce_if(a_pre)
            ctx.model.add(p > t).only_enforce_if(a_pre.negated())

        measured_before_t = ctx.model.new_bool_var(f"measured_before_{node}_{t}")
        if node not in node2meas:
            ctx.model.add(measured_before_t == 0)
        else:
            q = node2meas[node]
            ctx.model.add(q < t).only_enforce_if(measured_before_t)
            ctx.model.add(q >= t).only_enforce_if(measured_before_t.negated())

        alive = ctx.model.new_bool_var(f"alive_{node}_{t}")
        ctx.model.add_implication(alive, a_pre)
        ctx.model.add_implication(alive, measured_before_t.negated())
        ctx.model.add(a_pre - measured_before_t <= alive)
        alive_at_t.append(alive)

    return alive_at_t


def _set_minimize_space_objective(
    ctx: _ModelContext,
    node2prep: Mapping[int, cp_model.IntVar],
    node2meas: Mapping[int, cp_model.IntVar],
    max_time: int,
) -> None:
    """Set objective to minimize the maximum number of qubits used at any time."""
    max_space = ctx.model.new_int_var(0, len(ctx.graph.nodes), "max_space")
    for t in range(max_time + 1):
        alive_at_t = _compute_alive_nodes_at_time(ctx, node2prep, node2meas, t)
        ctx.model.add(max_space >= sum(alive_at_t))
    ctx.model.minimize(max_space)


def _set_minimize_time_objective(
    ctx: _ModelContext,
    node2prep: Mapping[int, cp_model.IntVar],
    node2meas: Mapping[int, cp_model.IntVar],
    max_time: int,
    max_qubit_count: int | None = None,
) -> None:
    """Set objective to minimize the total execution time."""
    # Add space constraint if max_qubit_count is specified
    if max_qubit_count is not None:
        for t in range(max_time + 1):
            alive_at_t = _compute_alive_nodes_at_time(ctx, node2prep, node2meas, t)
            ctx.model.add(sum(alive_at_t) <= max_qubit_count)

    # Time objective: minimize makespan. Pure input-output graphs have no
    # measurement/preparation variables, but can still have edge constraints.
    meas_vars = list(node2meas.values()) or list(node2prep.values())
    if not meas_vars:
        ctx.model.minimize(0)
        return
    makespan = ctx.model.new_int_var(0, max_time, "makespan")
    ctx.model.add_max_equality(makespan, meas_vars)
    ctx.model.minimize(makespan)


def solve_schedule(
    graph: BaseGraphState,
    dag: Mapping[int, AbstractSet[int]],
    config: ScheduleConfig,
    timeout: int = 60,
) -> tuple[dict[int, int], dict[int, int]] | None:
    r"""Solve the scheduling problem for the given graph.

    Parameters
    ----------
    graph : `BaseGraphState`
        The graph state to optimize.
    dag : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        The directed acyclic graph representing dependencies.
    config : `ScheduleConfig`
        The scheduling configuration including strategy and constraints.
    timeout : `int`, optional
        Maximum solve time in seconds, by default 60

    Returns
    -------
    `tuple`\[`dict`\[`int`, `int`\], `dict`\[`int`, `int`\]\] | `None`
        A tuple of (prepare_time, measure_time) dictionaries if solved,
        None if no solution found.
    """
    # Construct model
    model = cp_model.CpModel()

    # Determine max_time from config or calculate default
    max_time = config.max_time if config.max_time is not None else 2 * len(graph.nodes)

    # Create variables
    unmeasured_outputs = unmeasured_output_nodes(graph)
    node2prep: dict[int, cp_model.IntVar] = {}
    node2meas: dict[int, cp_model.IntVar] = {}
    for node in graph.nodes:
        if node not in graph.input_node_indices:
            node2prep[node] = model.new_int_var(0, max_time, f"prep_{node}")
        if node not in unmeasured_outputs:
            node2meas[node] = model.new_int_var(0, max_time, f"meas_{node}")

    # Add constraints
    _add_constraints(model, graph, dag, node2prep, node2meas)

    # Set objective
    ctx = _ModelContext(model, graph)
    _set_objective(ctx, node2prep, node2meas, config, max_time)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    status = solver.Solve(model)

    if status in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        prepare_time: dict[int, int] = {node: int(solver.Value(var)) for node, var in node2prep.items()}
        measure_time: dict[int, int] = {node: int(solver.Value(var)) for node, var in node2meas.items()}
        return prepare_time, measure_time

    return None
