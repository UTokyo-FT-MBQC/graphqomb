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
from typing import TYPE_CHECKING, Any, TypeGuard

import typing_extensions

from graphqomb import clifford_algebra
from graphqomb.common import Axis, Plane, determine_pauli_axis
from graphqomb.graphstate import BaseGraphState, odd_neighbors, unmeasured_output_nodes

if TYPE_CHECKING:
    from collections.abc import Collection

    from graphqomb.clifford_algebra import C1Element

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
    cflow: Mapping[int, Mapping[int, C1Element]] | None = None,
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
    cflow : `collections.abc.Mapping`\[`int`, `collections.abc.Mapping`\[`int`, `C1Element`\]\] | `None`
        The Clifford correction flow. Its targets join the DAG edges.

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
    if cflow is None:
        cflow = {}
    # Validate sources before unmeasured outputs lose their outgoing DAG edges.
    # Empty entries are harmless; self-targets remain valid on measured nodes.
    for flow in (xflow, zflow, cflow):
        _check_flow_sources(measured_nodes, flow)
    for node in measured_nodes:
        # remove self-loops
        target_nodes = (xflow.get(node, set()) | zflow.get(node, set()) | cflow.get(node, {}).keys()) - {node}
        dag[node] = target_nodes
    for output in unmeasured_outputs:
        dag[output] = set()

    return dag


def _check_flow_sources(measured_nodes: AbstractSet[int], flow: Mapping[int, Collection[int]]) -> None:
    """Check that each nonempty correction map has a measured source.

    Raises
    ------
    ValueError
        If a correction source has no measurement outcome.
    """
    for source, targets in flow.items():
        if targets and source not in measured_nodes:
            msg = f"Flow source {source} is not measured; unmeasured outputs cannot control corrections."
            raise ValueError(msg)


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
    cflow: Mapping[int, Mapping[int, C1Element]] | None = None,
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
    cflow : `collections.abc.Mapping`\[`int`, `collections.abc.Mapping`\[`int`, `C1Element`\]\] | `None`
        The Clifford correction flow.

    Notes
    -----
    Nonempty flows require measured sources; the combined x/z/Clifford
    dependency graph must be acyclic. Self-targets are allowed and excluded
    from dependencies and runtime corrections.
    """  # ruff:ignore[line-too-long]
    dag = dag_from_flow(graph, xflow, zflow, cflow)
    check_dag(dag)


def _reject_cflow(cflow: Mapping[int, Mapping[int, C1Element]] | None, operation: str) -> None:
    """Raise for a nonempty Clifford correction flow.

    Raises
    ------
    NotImplementedError
        If any cflow correction is present.
    """
    if cflow and any(targets for targets in cflow.values()):
        msg = f"{operation} over Clifford feedforward (cflow) is not supported yet."
        raise NotImplementedError(msg)


def _normalized_correction_dag(
    graph: BaseGraphState,
    xflow: Mapping[int, AbstractSet[int]],
    zflow: Mapping[int, AbstractSet[int]],
    cflow: Mapping[int, Mapping[int, C1Element]],
) -> dict[int, set[int]]:
    r"""Use the same-source Pauli cancellation convention of CliffordFrame.

    Returns
    -------
    `dict`\[`int`, `set`\[`int`\]\]
        Dependency DAG after normalization, without changing the supplied maps.
    """
    normalized_x = {source: set(targets) for source, targets in xflow.items()}
    normalized_z = {source: set(targets) for source, targets in zflow.items()}
    normalized_c: dict[int, dict[int, C1Element]] = {}
    for source, targets in cflow.items():
        for target, gate in targets.items():
            coset, x_bit, z_bit = clifford_algebra.decompose(gate)
            if x_bit:
                normalized_x.setdefault(source, set()).symmetric_difference_update({target})
            if z_bit:
                normalized_z.setdefault(source, set()).symmetric_difference_update({target})
            if coset != clifford_algebra.IDENTITY:
                normalized_c.setdefault(source, {})[target] = coset
    return dag_from_flow(graph, normalized_x, normalized_z, normalized_c)


def _signal_shifting_context(
    graph: BaseGraphState,
    xflow: Mapping[int, AbstractSet[int]],
    zflow: Mapping[int, AbstractSet[int]],
    cflow: Mapping[int, Mapping[int, C1Element]] | None,
) -> tuple[dict[int, set[int]], set[int]]:
    r"""Build the execution DAG and the conservative Clifford boundary.

    A shift advances the target's outgoing Pauli events to its parents. Freezing
    every cflow target and its ancestors prevents those events from crossing a
    Clifford correction, even when the shifted node has no incident cflow edge.
    Pauli-valued cflow entries are also protected because this pass returns only
    xflow/zflow and cannot rewrite the supplied cflow's control records.

    Use normalized corrections for the execution DAG, as CliffordFrame does.
    For protection, restore the supplied cflow edges: shifting just the X/Z
    half of a cancelled correction would break its cancellation with cflow.

    Returns
    -------
    `tuple`\[`dict`\[`int`, `set`\[`int`\]\], `set`\[`int`\]\]
        Normalized dependency DAG and the protected backward closure.
    """
    if not cflow:
        return dag_from_flow(graph, xflow, zflow), set()
    dag = _normalized_correction_dag(graph, xflow, zflow, cflow)

    protected = {
        target
        for source, targets in cflow.items()
        for target, gate in targets.items()
        if source != target and gate != clifford_algebra.IDENTITY
    }
    if not protected:
        return dag, protected
    protection_dag = {source: set(targets) for source, targets in dag.items()}
    for source, targets in cflow.items():
        protection_dag.setdefault(source, set()).update(
            target for target, gate in targets.items() if source != target and gate != clifford_algebra.IDENTITY
        )
    inverse_dag = inverse_dag_from_dag(protection_dag)
    pending = list(protected)
    while pending:
        for parent in inverse_dag.get(pending.pop(), set()):
            if parent not in protected:
                protected.add(parent)
                pending.append(parent)
    return dag, protected


def signal_shifting(
    graph: BaseGraphState,
    xflow: Mapping[int, AbstractSet[int]],
    zflow: Mapping[int, AbstractSet[int]] | None = None,
    cflow: Mapping[int, Mapping[int, C1Element]] | None = None,
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
    cflow : `collections.abc.Mapping`\[`int`, `collections.abc.Mapping`\[`int`, `C1Element`\]\] | `None`
        Clifford correction flow, preserved unchanged. Its effective targets
        and their causal past are excluded from signal shifting.

    Returns
    -------
    `tuple`\[`dict`\[`int`, `set`\[`int`\]\], `dict`\[`int`, `set`\[`int`\]\]\]
        Updated correction maps for X and Z after signal shifting. Pass the
        original cflow alongside these maps when compiling the result.

    Notes
    -----
    Only Pauli regions outside the backward dependency closure of cflow targets
    are rewritten. This keeps Clifford control records and noncommuting event
    order unchanged; it does not distribute Clifford gates over XOR controls.
    Self-target and identity cflow entries do not create a boundary. Pauli gates
    supplied through cflow do create one; use xflow/zflow to make them eligible
    for shifting. Measurement records in rewritten regions are relabelled, as
    in ordinary signal shifting, so this pass does not update parity seeds.
    """
    if zflow is None:
        zflow = {node: odd_neighbors(xflow[node], graph) - {node} for node in xflow}

    dag, protected = _signal_shifting_context(graph, xflow, zflow, cflow)
    topo_order = list(TopologicalSorter(dag).static_order())
    topo_order.reverse()  # from parents to children

    for output in graph.output_node_indices:
        topo_order.remove(output)

    new_xflow = {k: set(vs) for k, vs in xflow.items()}
    new_zflow = {k: set(vs) for k, vs in zflow.items()}

    for target_node in topo_order:
        if target_node in protected:
            continue
        # The original protected set is closed under predecessors. A shift
        # outside it cannot introduce a correction into it, so compute it once.
        new_xflow, new_zflow = propagate_correction_map(target_node, graph, new_xflow, new_zflow)

    return new_xflow, new_zflow


def propagate_correction_map(  # ruff:ignore[complex-structure, too-many-branches]
    target_node: int,
    graph: BaseGraphState,
    xflow: Mapping[int, AbstractSet[int]],
    zflow: Mapping[int, AbstractSet[int]] | None = None,
    cflow: Mapping[int, Mapping[int, C1Element]] | None = None,
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
    cflow : `collections.abc.Mapping`\[`int`, `collections.abc.Mapping`\[`int`, `C1Element`\]\] | `None`
        Clifford correction flow, preserved unchanged. If the target is in
        the causal past of an effective cflow target (or is one), return copies
        of the unmodified X and Z maps.

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
    With cflow, the same conservative boundary as `signal_shifting` preserves
    control records and correction order. Pass cflow on every individual call;
    it must also accompany the returned maps when compiling the result.
    """
    if target_node in graph.output_node_indices:
        msg = "Cannot propagate flow for output nodes."
        raise ValueError(msg)

    if zflow is None:
        zflow = {node: odd_neighbors(xflow[node], graph) - {node} for node in xflow}

    new_xflow = {k: set(vs) for k, vs in xflow.items()}
    new_zflow = {k: set(vs) for k, vs in zflow.items()}
    if cflow:
        dag, protected = _signal_shifting_context(graph, xflow, zflow, cflow)
        check_dag(dag)
        if target_node in protected:
            return new_xflow, new_zflow

    inv_xflow: dict[int, set[int]] = {}
    inv_zflow: dict[int, set[int]] = {}
    for k, vs in xflow.items():
        for v in vs:
            inv_xflow.setdefault(v, set()).add(k)
    for k, vs in zflow.items():
        for v in vs:
            inv_zflow.setdefault(v, set()).add(k)

    meas_basis = graph.meas_bases[target_node]

    if meas_basis.plane == Plane.XY:
        target_parents = inv_zflow.get(target_node, set()) - {target_node}
        for parent in target_parents:
            new_zflow[parent] -= {target_node}
    elif meas_basis.plane == Plane.YZ:
        target_parents = inv_xflow.get(target_node, set()) - {target_node}
        for parent in target_parents:
            new_xflow[parent] -= {target_node}
    elif meas_basis.plane == Plane.XZ:
        target_parents = (inv_xflow.get(target_node, set()) & inv_zflow.get(target_node, set())) - {target_node}
        for parent in target_parents:
            new_xflow[parent] -= {target_node}
            new_zflow[parent] -= {target_node}
    else:
        typing_extensions.assert_never(meas_basis.plane)
        msg = f"Unsupported measurement plane: {meas_basis.plane}"
        raise ValueError(msg)

    for child_x in xflow.get(target_node, set()) - {target_node}:
        for parent in target_parents:
            new_xflow.setdefault(parent, set()).symmetric_difference_update({child_x})
    for child_z in zflow.get(target_node, set()) - {target_node}:
        for parent in target_parents:
            new_zflow.setdefault(parent, set()).symmetric_difference_update({child_z})

    return new_xflow, new_zflow


def pauli_simplification(  # ruff:ignore[complex-structure, too-many-branches]
    graph: BaseGraphState,
    xflow: Mapping[int, AbstractSet[int]],
    zflow: Mapping[int, AbstractSet[int]] | None = None,
    cflow: Mapping[int, Mapping[int, C1Element]] | None = None,
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
    cflow : `collections.abc.Mapping`\[`int`, `collections.abc.Mapping`\[`int`, `C1Element`\]\] | `None`
        Clifford correction flow. Must be empty; simplifying it is unsupported.

    Returns
    -------
    `tuple`\[`dict`\[`int`, `set`\[`int`\]\], `dict`\[`int`, `set`\[`int`\]\]\]
        Updated correction maps for X and Z after simplification.
    """
    _reject_cflow(cflow, "Pauli simplification")
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
