"""Feedforward correction functions.

This module provides:

- `dag_from_flow`: Construct a directed acyclic graph (DAG) from a flowlike object.
- `inverse_dag_from_dag`: Construct an inverse DAG (node -> dependencies).
- `topo_order_from_inv_dag`: Construct a topological order from an inverse DAG.
- `check_dag`: Check if a directed acyclic graph (DAG) does not contain a cycle.
- `check_flow`: Check if the flowlike object is causal with respect to the graph state.
- `signal_shifting`: Convert the correction maps into more parallel-friendly forms using signal shifting.
- `propagate_correction_map`: Propagate the correction map through a measurement at the target node.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from graphlib import CycleError, TopologicalSorter
from typing import Any, TypeGuard

import typing_extensions

from graphqomb.common import Axis, Plane, determine_pauli_axis
from graphqomb.graphstate import BaseGraphState, odd_neighbors, unmeasured_output_nodes

TOPO_ORDER_CYCLE_ERROR_MSG = "No nodes can be measured; possible cyclic dependency or incomplete preparation."


def _is_flow(flowlike: Mapping[int, Any]) -> TypeGuard[Mapping[int, int]]:
    r"""Check if the flowlike object is a flow.

    Parameters
    ----------
    flowlike : `collections.abc.Mapping`\[`int`, `typing.Any`\]
        A flowlike object to check

    Returns
    -------
    `bool`
        True if the flowlike object is a flow, False otherwise
    """
    return all(isinstance(v, int) for v in flowlike.values())


def _is_gflow(flowlike: Mapping[int, Any]) -> TypeGuard[Mapping[int, AbstractSet[int]]]:
    r"""Check if the flowlike object is a GFlow.

    Parameters
    ----------
    flowlike : `collections.abc.Mapping`\[`int`, `typing.Any`\]
        A flowlike object to check

    Returns
    -------
    `bool`
        True if the flowlike object is a GFlow, False otherwise
    """
    return all(isinstance(v, AbstractSet) for v in flowlike.values())


def dag_from_flow(
    graph: BaseGraphState,
    xflow: Mapping[int, int] | Mapping[int, AbstractSet[int]],
    zflow: Mapping[int, int] | Mapping[int, AbstractSet[int]] | None = None,
) -> dict[int, set[int]]:
    r"""Construct a directed acyclic graph (DAG) from a flowlike object.

    Parameters
    ----------
    graph : `BaseGraphState`
        The graph state
    xflow : `collections.abc.Mapping`\[`int`, `int`\] | `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        The X correction flow (flow and gflow are included)
    zflow : `collections.abc.Mapping`\[`int`, `int`\] | `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`
        The Z correction flow. If `None`, it is generated from xflow by odd neighbors.

    Returns
    -------
    `dict`\[`int`, `set`\[`int`\]\]
        The directed acyclic graph

    Raises
    ------
    TypeError
        If the flowlike object is not a Flow or GFlow
    """  # ruff:ignore[line-too-long]
    dag: dict[int, set[int]] = {}
    unmeasured_outputs = unmeasured_output_nodes(graph)
    measured_nodes = graph.nodes - unmeasured_outputs
    if _is_flow(xflow):
        xflow = {node: {xflow[node]} for node in xflow}
    elif _is_gflow(xflow):
        xflow = {node: set(xflow[node]) for node in xflow}
    else:
        msg = "Invalid flowlike object"
        raise TypeError(msg)

    if zflow is None:
        zflow = {node: odd_neighbors(xflow[node], graph) for node in xflow}
    elif _is_flow(zflow):
        zflow = {node: {zflow[node]} for node in zflow}
    elif _is_gflow(zflow):
        zflow = {node: set(zflow[node]) for node in zflow}
    else:
        msg = "Invalid zflow object"
        raise TypeError(msg)
    for node in measured_nodes:
        target_nodes = (xflow.get(node, set()) | zflow.get(node, set())) - {node}  # remove self-loops
        dag[node] = target_nodes
    for output in unmeasured_outputs:
        dag[output] = set()

    return dag


def check_dag(dag: Mapping[int, Iterable[int]]) -> None:
    r"""Check if a directed acyclic graph (DAG) does not contain a cycle.

    Parameters
    ----------
    dag : `collections.abc.Mapping`\[`int`, `collections.abc.Iterable`\[`int`\]\]
        directed acyclic graph

    Raises
    ------
    ValueError
        If the graph contains a cycle
    """
    inv_dag = inverse_dag_from_dag(dag)
    try:
        tuple(TopologicalSorter(inv_dag).static_order())
    except CycleError as exc:
        cycle = " -> ".join(map(str, exc.args[1]))
        msg = f"Cycle detected in the graph: {cycle}"
        raise ValueError(msg) from exc


def inverse_dag_from_dag(
    dag: Mapping[int, Iterable[int]],
    all_nodes: Iterable[int] | None = None,
) -> dict[int, set[int]]:
    r"""Build inverse DAG (node -> dependencies) from parent->children DAG.

    Parameters
    ----------
    dag : `collections.abc.Mapping`\[`int`, `collections.abc.Iterable`\[`int`\]\]
        DAG represented as parent node -> children.
    all_nodes : `collections.abc.Iterable`\[`int`\] | `None`, optional
        Optional full node set to include isolated nodes.

    Returns
    -------
    `dict`\[`int`, `set`\[`int`\]\]
        Inverse DAG represented as node -> dependencies.
    """
    nodes = set(all_nodes) if all_nodes is not None else set(dag)
    for children in dag.values():
        nodes.update(children)

    inv_dag: dict[int, set[int]] = {node: set() for node in nodes}
    for parent, children in dag.items():
        for child in children:
            inv_dag[child].add(parent)

    return inv_dag


def topo_order_from_inv_dag(inv_dag: Mapping[int, Iterable[int]]) -> list[int]:
    r"""Build topological order from an inverse DAG (node -> dependencies).

    Parameters
    ----------
    inv_dag : `collections.abc.Mapping`\[`int`, `collections.abc.Iterable`\[`int`\]\]
        Inverse DAG where each node maps to the nodes it depends on.

    Returns
    -------
    `list`\[`int`\]
        Topological order from dependencies to dependents.

    Raises
    ------
    RuntimeError
        If topological ordering is not possible due to a cycle.
    """
    try:
        return list(TopologicalSorter(inv_dag).static_order())
    except CycleError as exc:
        raise RuntimeError(TOPO_ORDER_CYCLE_ERROR_MSG) from exc


def check_flow(
    graph: BaseGraphState,
    xflow: Mapping[int, int] | Mapping[int, AbstractSet[int]],
    zflow: Mapping[int, int] | Mapping[int, AbstractSet[int]] | None = None,
) -> None:
    r"""Check if the flowlike object is causal with respect to the graph state.

    Parameters
    ----------
    graph : `BaseGraphState`
        The graph state
    xflow : `collections.abc.Mapping`\[`int`, `int`\] | `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        The  X correction flow (flow and gflow are included)
    zflow : `collections.abc.Mapping`\[`int`, `int`\] | `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`
        The  Z correction flow. If `None`, it is generated from xflow by odd neighbors.
    """  # ruff:ignore[line-too-long]
    dag = dag_from_flow(graph, xflow, zflow)
    check_dag(dag)


def signal_shifting(
    graph: BaseGraphState, xflow: Mapping[int, AbstractSet[int]], zflow: Mapping[int, AbstractSet[int]] | None = None
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    r"""Convert the correction maps into more parallel-friendly forms using signal shifting.

    Parameters
    ----------
    graph : `BaseGraphState`
        Underlying graph state.
    xflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        Correction map for X.
    zflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`
        Correction map for Z. If `None`, it is generated from xflow by odd neighbors.

    Returns
    -------
    `tuple`\[`dict`\[`int`, `set`\[`int`\]\], `dict`\[`int`, `set`\[`int`\]\]\]
        Updated correction maps for X and Z after signal shifting.
    """
    if zflow is None:
        zflow = {node: odd_neighbors(xflow[node], graph) - {node} for node in xflow}

    dag = dag_from_flow(graph, xflow, zflow)
    topo_order = list(TopologicalSorter(dag).static_order())
    topo_order.reverse()  # from parents to children

    for output in graph.output_node_indices:
        topo_order.remove(output)

    new_xflow = {k: set(vs) for k, vs in xflow.items()}
    new_zflow = {k: set(vs) for k, vs in zflow.items()}

    for target_node in topo_order:
        new_xflow, new_zflow = propagate_correction_map(target_node, graph, new_xflow, new_zflow)

    return new_xflow, new_zflow


def propagate_correction_map(  # ruff:ignore[complex-structure, too-many-branches]
    target_node: int,
    graph: BaseGraphState,
    xflow: Mapping[int, AbstractSet[int]],
    zflow: Mapping[int, AbstractSet[int]] | None = None,
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    r"""Propagate the correction map through a measurement at the target node.

    Parameters
    ----------
    target_node : `int`
        Node at which the measurement is performed.
    graph : `BaseGraphState`
        Underlying graph state.
    xflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        Correction map for X.
    zflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`
        Correction map for Z. If `None`, it is generated from xflow by odd neighbors.

    Returns
    -------
    `tuple`\[`dict`\[`int`, `set`\[`int`\]\], `dict`\[`int`, `set`\[`int`\]\]\]
        Updated correction maps for X and Z after measurement at the target node.

    Raises
    ------
    ValueError
        If the target node is an output node.
    ValueError
        If the measurement plane is unsupported.


    Notes
    -----
    This function converts the correction maps into more parallel-friendly forms.
    It is equivalent to the signal shifting technique in the measurement calculus.
    """
    if target_node in graph.output_node_indices:
        msg = "Cannot propagate flow for output nodes."
        raise ValueError(msg)

    if zflow is None:
        zflow = {node: odd_neighbors(xflow[node], graph) - {node} for node in xflow}

    inv_xflow: dict[int, set[int]] = {}
    inv_zflow: dict[int, set[int]] = {}
    for k, vs in xflow.items():
        for v in vs:
            inv_xflow.setdefault(v, set()).add(k)
    for k, vs in zflow.items():
        for v in vs:
            inv_zflow.setdefault(v, set()).add(k)

    new_xflow = {k: set(vs) for k, vs in xflow.items()}
    new_zflow = {k: set(vs) for k, vs in zflow.items()}

    meas_basis = graph.meas_bases[target_node]

    if meas_basis.plane == Plane.XY:
        target_parents = inv_zflow.get(target_node, set())
        for parent in target_parents:
            new_zflow[parent] -= {target_node}
    elif meas_basis.plane == Plane.YZ:
        target_parents = inv_xflow.get(target_node, set())
        for parent in target_parents:
            new_xflow[parent] -= {target_node}
    elif meas_basis.plane == Plane.XZ:
        target_parents = inv_xflow.get(target_node, set()) & inv_zflow.get(target_node, set())
        for parent in target_parents:
            new_xflow[parent] -= {target_node}
            new_zflow[parent] -= {target_node}
    else:
        typing_extensions.assert_never(meas_basis.plane)
        msg = f"Unsupported measurement plane: {meas_basis.plane}"
        raise ValueError(msg)

    for child_x in xflow.get(target_node, set()):
        for parent in target_parents:
            new_xflow[parent] ^= {child_x}
    for child_z in zflow.get(target_node, set()):
        for parent in target_parents:
            new_zflow[parent] ^= {child_z}

    return new_xflow, new_zflow


def pauli_simplification(  # ruff:ignore[complex-structure, too-many-branches]
    graph: BaseGraphState,
    xflow: Mapping[int, AbstractSet[int]],
    zflow: Mapping[int, AbstractSet[int]] | None = None,
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    r"""Simplify the correction maps by removing redundant Pauli corrections.

    Parameters
    ----------
    graph : `BaseGraphState`
        Underlying graph state.
    xflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        Correction map for X.
    zflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`
        Correction map for Z. If `None`, it is generated from xflow by odd neighbors.

    Returns
    -------
    `tuple`\[`dict`\[`int`, `set`\[`int`\]\], `dict`\[`int`, `set`\[`int`\]\]\]
        Updated correction maps for X and Z after simplification.
    """
    if zflow is None:
        zflow = {node: odd_neighbors(xflow[node], graph) - {node} for node in xflow}

    new_xflow = {k: set(vs) for k, vs in xflow.items()}
    new_zflow = {k: set(vs) for k, vs in zflow.items()}

    inv_xflow: dict[int, set[int]] = {}
    inv_zflow: dict[int, set[int]] = {}
    for k, vs in xflow.items():
        for v in vs:
            inv_xflow.setdefault(v, set()).add(k)
    for k, vs in zflow.items():
        for v in vs:
            inv_zflow.setdefault(v, set()).add(k)

    for node in graph.nodes - graph.output_node_indices.keys():
        meas_basis = graph.meas_bases.get(node)
        if meas_basis is None:
            continue
        meas_axis = determine_pauli_axis(meas_basis)
        if meas_axis is None:
            continue

        if meas_axis == Axis.X:
            for parent in inv_xflow.get(node, set()):
                new_xflow[parent] -= {node}
        elif meas_axis == Axis.Z:
            for parent in inv_zflow.get(node, set()):
                new_zflow[parent] -= {node}
        elif meas_axis == Axis.Y:
            for parent in inv_xflow.get(node, set()) & inv_zflow.get(node, set()):
                new_xflow[parent] -= {node}
                new_zflow[parent] -= {node}

    return new_xflow, new_zflow
