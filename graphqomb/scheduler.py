"""Graph scheduler for measurement and preparation timing in MBQC patterns.

This module provides:

- `compress_schedule`: Compress preparation and measurement times by removing gaps.
- `Scheduler`: Schedule graph node preparation and measurement operations
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, NamedTuple

from graphqomb.feedforward import dag_from_flow
from graphqomb.graphstate import unmeasured_output_nodes
from graphqomb.greedy_scheduler import greedy_minimize_space, greedy_minimize_time
from graphqomb.schedule_solver import ScheduleConfig, Strategy, solve_schedule

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    from graphqomb.graphstate import BaseGraphState


class ScheduleTimings(NamedTuple):
    """Scheduling timings for preparation, entanglement, and measurement."""

    prepare_time: dict[int, int | None]  #: Mapping from node indices to their preparation time.
    measure_time: dict[int, int | None]  #: Mapping from node indices to their measurement time.
    entangle_time: dict[tuple[int, int], int | None]  #: Mapping from edges to their entanglement time.


class TimeSlice(NamedTuple):
    """Operations for a single time slice in the schedule."""

    prepare_nodes: set[int]  #: Set of node indices to prepare in this time slice.
    entangle_edges: set[tuple[int, int]]  #: Set of edges to entangle in this time slice.
    measure_nodes: set[int]  #: Set of node indices to measure in this time slice.


def compress_schedule(  # ruff:ignore[complex-structure, too-many-branches]
    prepare_time: Mapping[int, int | None],
    measure_time: Mapping[int, int | None],
    entangle_time: Mapping[tuple[int, int], int | None] | None = None,
) -> ScheduleTimings:
    r"""Compress a schedule by removing gaps in time indices.

    This function shifts all time indices forward to remove unused time slots,
    reducing the total number of slices without changing the relative ordering.

    Parameters
    ----------
    prepare_time : `collections.abc.Mapping`\[`int`, `int` | `None`\]
        A mapping from node indices to their preparation time.
    measure_time : `collections.abc.Mapping`\[`int`, `int` | `None`\]
        A mapping from node indices to their measurement time.
    entangle_time : `collections.abc.Mapping`\[`tuple`\[`int`, `int`\], `int` | `None`\] | `None`, optional
        A mapping from edges (as tuples) to their entanglement time.

    Returns
    -------
    ScheduleTimings
        A NamedTuple containing compressed timing information:

        - prepare_time: `dict`\[`int`, `int` | `None`\]
        - measure_time: `dict`\[`int`, `int` | `None`\]
        - entangle_time: `dict`\[`tuple`\[`int`, `int`\], `int` | `None`\]
    """
    # Collect all used time indices
    all_times: set[int] = set()

    for time in prepare_time.values():
        if time is not None:
            all_times.add(time)

    for time in measure_time.values():
        if time is not None:
            all_times.add(time)

    if entangle_time is not None:
        for time in entangle_time.values():
            if time is not None:
                all_times.add(time)

    if not all_times:
        compressed_entangle_time: dict[tuple[int, int], int | None] = (
            dict(entangle_time) if entangle_time is not None else {}
        )
        return ScheduleTimings(dict(prepare_time), dict(measure_time), compressed_entangle_time)

    # Create mapping from old time to new compressed time
    sorted_times = sorted(all_times)
    time_mapping = {old_time: new_time for new_time, old_time in enumerate(sorted_times)}

    # Apply compression to preparation times
    compressed_prepare_time: dict[int, int | None] = {}
    for node, old_time in prepare_time.items():
        if old_time is not None:
            compressed_prepare_time[node] = time_mapping[old_time]
        else:
            compressed_prepare_time[node] = None

    # Apply compression to measurement times
    compressed_measure_time: dict[int, int | None] = {}
    for node, old_time in measure_time.items():
        if old_time is not None:
            compressed_measure_time[node] = time_mapping[old_time]
        else:
            compressed_measure_time[node] = None

    # Apply compression to entanglement times
    compressed_entangle_time = {}
    if entangle_time is not None:
        for edge, old_time in entangle_time.items():
            if old_time is not None:
                compressed_entangle_time[edge] = time_mapping[old_time]
            else:
                compressed_entangle_time[edge] = None

    return ScheduleTimings(compressed_prepare_time, compressed_measure_time, compressed_entangle_time)


class Scheduler:
    r"""Schedule graph preparation and measurements.

    Attributes
    ----------
    graph : `BaseGraphState`
        The graph state to be scheduled.
    dag : `dict`\[`int`, `set`\[`int`\]\]
        The directed acyclic graph representing dependencies.
    prepare_time : `dict`\[`int`, `int` | `None`\]
        A mapping from node indices to their preparation time.
    measure_time : `dict`\[`int`, `int` | `None`\]
        A mapping from node indices to their measurement time.
    entangle_time : `dict`\[`tuple`\[`int`, `int`\], `int` | `None`\]
        A mapping from edge (as tuple of two node indices) to their entanglement time.
    """

    graph: BaseGraphState
    dag: dict[int, set[int]]
    prepare_time: dict[int, int | None]
    measure_time: dict[int, int | None]
    entangle_time: dict[tuple[int, int], int | None]

    def __init__(
        self,
        graph: BaseGraphState,
        xflow: Mapping[int, AbstractSet[int]],
        zflow: Mapping[int, AbstractSet[int]] | None = None,
    ) -> None:
        self.graph = graph
        self.dag = dag_from_flow(graph, xflow, zflow)
        self.prepare_time = dict.fromkeys(graph.nodes - graph.input_node_indices.keys())
        self.measure_time = dict.fromkeys(graph.nodes - unmeasured_output_nodes(graph))
        # Initialize entangle_time for all edges
        self.entangle_time = dict.fromkeys(graph.edges)

    def num_slices(self) -> int:
        r"""Return the number of slices in the schedule.

        Returns
        -------
        `int`
            The number of slices, which is the maximum time across all nodes and edges plus one.
        """
        return (
            max(
                max((t for t in self.prepare_time.values() if t is not None), default=0),
                max((t for t in self.measure_time.values() if t is not None), default=0),
                max((t for t in self.entangle_time.values() if t is not None), default=0),
            )
            + 1
        )

    @property
    def timeline(self) -> list[TimeSlice]:
        r"""Per-slice operations for preparation, entanglement, and measurement.

        Returns
        -------
        `list`\[`TimeSlice`\]
            Each element is a `TimeSlice` containing three sets for each time slice:

            - prepare_nodes: Nodes to prepare
            - entangle_edges: Edges to entangle
            - measure_nodes: Nodes to measure
        """
        prep_time: defaultdict[int, set[int]] = defaultdict(set)
        for node, time in self.prepare_time.items():
            if time is not None:
                prep_time[time].add(node)

        ent_time: defaultdict[int, set[tuple[int, int]]] = defaultdict(set)
        for edge, time in self.entangle_time.items():
            if time is not None:
                ent_time[time].add(edge)

        meas_time: defaultdict[int, set[int]] = defaultdict(set)
        for node, time in self.measure_time.items():
            if time is not None:
                meas_time[time].add(node)

        return [TimeSlice(prep_time[time], ent_time[time], meas_time[time]) for time in range(self.num_slices())]

    def manual_schedule(
        self,
        prepare_time: Mapping[int, int | None],
        measure_time: Mapping[int, int | None],
        entangle_time: Mapping[tuple[int, int], int | None] | None = None,
    ) -> None:
        r"""Set the schedule manually.

        Parameters
        ----------
        prepare_time : `collections.abc.Mapping`\[`int`, `int` | `None`\]
            A mapping from node indices to their preparation time.
        measure_time : `collections.abc.Mapping`\[`int`, `int` | `None`\]
            A mapping from node indices to their measurement time.
        entangle_time : `collections.abc.Mapping`\[`tuple`\[`int`, `int`\], `int` | `None`\] | `None`, optional
            A mapping from edges (as tuples) to their entanglement time.
            If None, unscheduled entanglement times are auto-scheduled based on preparation times.

        Notes
        -----
        After setting preparation and measurement times, any unscheduled entanglement times
        (with `None` value) are automatically scheduled using `auto_schedule_entanglement()`.

        The graph is treated as undirected. For convenience, `entangle_time` accepts edges
        in either order: both ``(u, v)`` and ``(v, u)`` are recognized. If both keys are
        provided, the canonical order (as returned by :attr:`BaseGraphState.edges`)
        takes precedence, even when the value is ``None``.
        """
        self.prepare_time = {
            node: prepare_time.get(node, None) for node in self.graph.nodes - self.graph.input_node_indices.keys()
        }
        self.measure_time = {
            node: measure_time.get(node, None) for node in self.graph.nodes - unmeasured_output_nodes(self.graph)
        }
        if entangle_time is not None:
            resolved_entangle_time: dict[tuple[int, int], int | None] = {}
            for edge in self.entangle_time:
                if edge in entangle_time:
                    resolved_entangle_time[edge] = entangle_time[edge]
                else:
                    u, v = edge
                    resolved_entangle_time[edge] = entangle_time.get((v, u), None)
            self.entangle_time = resolved_entangle_time

        # Auto-schedule unscheduled entanglement times
        if any(time is None for time in self.entangle_time.values()):
            self.auto_schedule_entanglement()

    def _validate_node_sets(self) -> None:
        """Validate that node sets are correctly configured.

        Raises
        ------
        ValueError
            If input/output nodes are incorrectly included in prepare/measure times,
            or if node sets do not match expected sets.
        """
        input_nodes = self.graph.input_node_indices.keys()
        unmeasured_outputs = unmeasured_output_nodes(self.graph)
        nodes = self.graph.nodes

        # Input nodes should not be in prepare_time
        invalid_prep = input_nodes & self.prepare_time.keys()
        if invalid_prep:
            msg = f"Input nodes {sorted(invalid_prep)} should not be in prepare_time"
            raise ValueError(msg)

        # Outputs without a measurement basis should not be in measure_time
        invalid_meas = unmeasured_outputs & self.measure_time.keys()
        if invalid_meas:
            msg = f"Unmeasured output nodes {sorted(invalid_meas)} should not be in measure_time"
            raise ValueError(msg)

        # Check expected node sets
        expected_prep_nodes = nodes - input_nodes
        expected_meas_nodes = nodes - unmeasured_outputs

        if self.prepare_time.keys() != expected_prep_nodes:
            missing = expected_prep_nodes - self.prepare_time.keys()
            unexpected = self.prepare_time.keys() - expected_prep_nodes
            msg_parts: list[str] = []
            if missing:
                msg_parts.append(f"missing nodes {sorted(missing)}")
            if unexpected:
                msg_parts.append(f"unexpected nodes {sorted(unexpected)}")
            msg = f"prepare_time has incorrect node set: {', '.join(msg_parts)}"
            raise ValueError(msg)

        if self.measure_time.keys() != expected_meas_nodes:
            missing = expected_meas_nodes - self.measure_time.keys()
            unexpected = self.measure_time.keys() - expected_meas_nodes
            msg_parts = []
            if missing:
                msg_parts.append(f"missing nodes {sorted(missing)}")
            if unexpected:
                msg_parts.append(f"unexpected nodes {sorted(unexpected)}")
            msg = f"measure_time has incorrect node set: {', '.join(msg_parts)}"
            raise ValueError(msg)

    def _validate_all_nodes_scheduled(self) -> None:
        """Validate that all required nodes are scheduled.

        Raises
        ------
        ValueError
            If any node in prepare_time or measure_time has None as its time value.
        """
        # All nodes in prepare_time must have non-None values
        unscheduled_prep = [node for node, time in self.prepare_time.items() if time is None]
        if unscheduled_prep:
            msg = f"Nodes {sorted(unscheduled_prep)} have no preparation time scheduled (time is None)"
            raise ValueError(msg)

        # All nodes in measure_time must have non-None values
        unscheduled_meas = [node for node, time in self.measure_time.items() if time is None]
        if unscheduled_meas:
            msg = f"Nodes {sorted(unscheduled_meas)} have no measurement time scheduled (time is None)"
            raise ValueError(msg)

    def _validate_executable_times_are_nonnegative(self) -> None:
        """Validate that all executable schedule times are non-negative.

        Raises
        ------
        ValueError
            If any executable preparation, measurement, or entanglement time is negative.
        """
        for schedule_name, schedule in (
            ("preparation", self.prepare_time),
            ("measurement", self.measure_time),
            ("entanglement", self.entangle_time),
        ):
            negative_times = {item: time for item, time in schedule.items() if time is not None and time < 0}
            if negative_times:
                msg = f"{schedule_name.capitalize()} schedule contains negative executable times: {negative_times}"
                raise ValueError(msg)

    def _validate_dag_constraints(self) -> None:
        """Validate that measurement order respects DAG dependencies.

        Raises
        ------
        ValueError
            If measurement times violate DAG ordering constraints
            (a node must be measured before all its successors in the DAG).
        """
        for u, successors in self.dag.items():
            u_time = self.measure_time.get(u)
            if u_time is None:
                continue
            for v in successors:
                v_time = self.measure_time.get(v)
                if v_time is not None and u_time >= v_time:
                    msg = (
                        f"DAG violation: node {u} (measure_time={u_time}) "
                        f"must be measured before node {v} (measure_time={v_time})"
                    )
                    raise ValueError(msg)

    def auto_schedule_entanglement(self) -> None:
        r"""Automatically schedule entanglement operations based on preparation times.

        Each edge is scheduled at the time when both of its endpoints are prepared.
        For edges involving input nodes, they are scheduled when the non-input node is prepared.
        Input nodes are considered to be prepared at time -1 (before the first time slice).

        Note
        ----
        Only schedules entanglement for edges with `None` time. Preserves manually set times.
        Validation of measurement causality is performed by `validate_schedule()`.
        """
        for edge in self.graph.edges:
            node1, node2 = edge

            # Get preparation times (input nodes are considered prepared at time -1)
            time1 = self.prepare_time.get(node1)
            if time1 is None and node1 in self.graph.input_node_indices:
                time1 = -1

            time2 = self.prepare_time.get(node2)
            if time2 is None and node2 in self.graph.input_node_indices:
                time2 = -1

            # Edge can be created when both nodes are prepared
            # Only schedule if not already scheduled (preserve manual settings)
            if time1 is not None and time2 is not None and self.entangle_time[edge] is None:
                # Keep executable slices non-negative even when both endpoints are inputs.
                self.entangle_time[edge] = max(time1, time2, 0)

    def _validate_entangle_time_constraints(self) -> None:
        """Validate that entanglement times respect preparation and measurement constraints.

        Checks that:
        - Entanglement happens AFTER both nodes are prepared
        - Entanglement happens BEFORE either node is measured

        Raises
        ------
        ValueError
            If entanglement times violate preparation or measurement causality constraints.
        """
        for edge, ent_time in self.entangle_time.items():
            if ent_time is None:
                # Entanglement not scheduled yet is okay (might be auto-scheduled later)
                continue

            node1, node2 = edge

            # Get preparation times (input nodes are considered prepared at time -1)
            time1 = self.prepare_time.get(node1)
            if time1 is None and node1 in self.graph.input_node_indices:
                time1 = -1

            time2 = self.prepare_time.get(node2)
            if time2 is None and node2 in self.graph.input_node_indices:
                time2 = -1

            # Both nodes must be prepared before or at entanglement time
            if time1 is None or time2 is None:
                # Cannot validate if preparation times are not set
                msg = f"Edge {edge} entanglement validation failed: preparation times not set"
                raise ValueError(msg)

            if ent_time < time1:
                msg = f"Edge {edge} entanglement at time {ent_time} is before node {node1} preparation at time {time1}"
                raise ValueError(msg)

            if ent_time < time2:
                msg = f"Edge {edge} entanglement at time {ent_time} is before node {node2} preparation at time {time2}"
                raise ValueError(msg)

            # Entanglement must happen BEFORE measurement of either node
            # Get measurement times (output nodes are not measured)
            meas_time1 = self.measure_time.get(node1)
            meas_time2 = self.measure_time.get(node2)

            # If node is measured, entanglement must be strictly before measurement
            if meas_time1 is not None and ent_time >= meas_time1:
                msg = (
                    f"Edge {edge} entanglement at time {ent_time} "
                    f"is not before node {node1} measurement at time {meas_time1}"
                )
                raise ValueError(msg)

            if meas_time2 is not None and ent_time >= meas_time2:
                msg = (
                    f"Edge {edge} entanglement at time {ent_time} "
                    f"is not before node {node2} measurement at time {meas_time2}"
                )
                raise ValueError(msg)

    def _validate_time_ordering(self) -> None:
        """Validate ordering within same time slice.

        Raises
        ------
        ValueError
            If any node is both prepared and measured at the same time.
        """
        # Group nodes by time
        time_to_prep_nodes: defaultdict[int, set[int]] = defaultdict(set)
        time_to_meas_nodes: defaultdict[int, set[int]] = defaultdict(set)

        for node, time in self.prepare_time.items():
            if time is not None:
                time_to_prep_nodes[time].add(node)

        for node, time in self.measure_time.items():
            if time is not None:
                time_to_meas_nodes[time].add(node)

        # Check that no node is both prepared and measured at the same time
        all_times = time_to_prep_nodes.keys() | time_to_meas_nodes.keys()
        for time in all_times:
            prep_nodes = time_to_prep_nodes[time]
            meas_nodes = time_to_meas_nodes[time]
            conflicting_nodes = prep_nodes & meas_nodes
            if conflicting_nodes:
                msg = f"Nodes {sorted(conflicting_nodes)} cannot be both prepared and measured at time {time}"
                raise ValueError(msg)

    def validate_schedule(self) -> None:
        r"""Validate that the schedule is consistent with the graph state and DAG.

        Checks:
        - Input nodes are not prepared (assumed to be prepared before time 0)
        - Outputs without a measurement basis are not measured
        - All non-input nodes have a preparation time
        - All nodes with a measurement basis have a measurement time
        - Measurement order respects DAG dependencies
        - Within same time slice, measurements happen before preparations
        - Entanglement times respect causality constraints (if entanglement is scheduled):

          - Entanglement happens AFTER both nodes are prepared
          - Entanglement happens BEFORE either node is measured
        """
        self._validate_node_sets()
        self._validate_all_nodes_scheduled()
        self._validate_executable_times_are_nonnegative()
        self._validate_dag_constraints()
        self._validate_time_ordering()

        # Validate entanglement times only if at least one edge has a scheduled time
        if any(time is not None for time in self.entangle_time.values()):
            self._validate_entangle_time_constraints()

    def solve_schedule(
        self,
        config: ScheduleConfig | None = None,
        timeout: int = 60,
    ) -> bool:
        r"""Compute the schedule using constraint programming or greedy heuristics.

        Parameters
        ----------
        config : `ScheduleConfig` | `None`, optional
            The scheduling configuration. If None, defaults to MINIMIZE_TIME strategy.
        timeout : `int`, optional
            Maximum solve time in seconds for CP-SAT solver, by default 60.
            Ignored when use_greedy=True.

        Returns
        -------
        `bool`
            True if a solution was found and applied, False otherwise.

        Note
        ----
        After solving, any unscheduled entanglement times (with `None` value) are
        automatically scheduled using `auto_schedule_entanglement()`.
        """
        if config is None:
            config = ScheduleConfig(Strategy.MINIMIZE_TIME)

        result: tuple[dict[int, int], dict[int, int]] | None
        if config.use_greedy:
            # Use fast greedy heuristics
            if config.strategy == Strategy.MINIMIZE_TIME:
                result = greedy_minimize_time(self.graph, self.dag, max_qubit_count=config.max_qubit_count)
            else:  # Strategy.MINIMIZE_SPACE
                result = greedy_minimize_space(self.graph, self.dag)
        else:
            # Use CP-SAT solver for optimal solution
            result = solve_schedule(self.graph, self.dag, config, timeout)

        if result is None:
            return False

        prepare_time, measure_time = result
        prep_time = {
            node: prepare_time.get(node, None) for node in self.graph.nodes - self.graph.input_node_indices.keys()
        }
        meas_time = {
            node: measure_time.get(node, None) for node in self.graph.nodes - unmeasured_output_nodes(self.graph)
        }

        self.prepare_time = prep_time
        self.measure_time = meas_time

        # Auto-schedule unscheduled entanglement times before compression so
        # slice-0 entanglement remains part of the executable time set.
        if any(time is None for time in self.entangle_time.values()):
            self.auto_schedule_entanglement()

        # Compress the schedule to minimize time indices
        timings = compress_schedule(self.prepare_time, self.measure_time, self.entangle_time)
        self.prepare_time = timings.prepare_time
        self.measure_time = timings.measure_time
        self.entangle_time = timings.entangle_time

        return True
