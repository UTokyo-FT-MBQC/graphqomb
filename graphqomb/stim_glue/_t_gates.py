"""Internal T-gate lowering shared by the near-Clifford importer."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import stim

from graphqomb import clifford_algebra as ca
from graphqomb.common import Axis, AxisMeasBasis, Plane, Sign, determine_pauli_axis, is_close_angle

if TYPE_CHECKING:
    from graphqomb.clifford_algebra import C1Element
    from graphqomb.graphstate import GraphState

# Private, tagged identities are a transport IR, never Clifford stand-ins for T.
# They are optimizer barriers and become J(theta), J(0) before graph conversion.
T_ANGLES = {"graphqomb.internal.T": math.pi / 4, "graphqomb.internal.T_DAG": -math.pi / 4}


def parse_t_circuit(text: str) -> stim.Circuit:
    """Parse the Stim+T subset using optimizer-protected private transport tags.

    Returns
    -------
    stim.Circuit
        Circuit whose tagged identities must be lowered before execution.

    Raises
    ------
    ValueError
        If the input uses reserved transport tags.
    """
    if any(tag in text for tag in T_ANGLES):
        msg = "Input text must not contain graphqomb.internal.T transport tags."
        raise ValueError(msg)
    encoded = re.sub(
        r"(?m)^([ \t]*)(T_DAG|T)(?=\s|$)",
        lambda match: f"{match[1]}I[graphqomb.internal.{match[2]}]",
        text,
    )
    return stim.Circuit(encoded)


def inject_t_measurements(
    graph: GraphState, xflow: dict[int, set[int]], zflow: dict[int, set[int]]
) -> tuple[dict[int, str], dict[int, dict[int, C1Element]]]:
    """Replace each T-basis measurement with a ZZ-probe injection gadget.

    Mutate the graph and correction maps before scheduling. The additional
    Z on a probe propagates every incoming X correction through its new CZ.

    Returns
    -------
    tuple
        Scheduled magic-preparation gates and conditional S/S-dagger corrections.

    Raises
    ------
    ValueError
        If a non-Pauli measurement is not an XY measurement at +/-pi/4.
    """
    input_gates: dict[int, str] = {}
    cflow: dict[int, dict[int, C1Element]] = {}
    for node, basis in tuple(graph.meas_bases.items()):
        if determine_pauli_axis(basis) is not None:
            continue
        if basis.plane is not Plane.XY or not is_close_angle(abs(basis.angle), math.pi / 4):
            msg = f"T injection requires an XY measurement at +/-pi/4, got {basis}."
            raise ValueError(msg)
        # M(-theta) implements J(theta). Inject T^(sign theta) before MX.
        positive = basis.angle < 0
        coord = graph.coordinates.get(node)
        magic = graph.add_node(coordinate=coord)
        probe = graph.add_node(coordinate=coord)
        graph.add_edge(probe, node)
        graph.add_edge(probe, magic)
        for measured in (node, magic, probe):
            graph.assign_meas_basis(measured, AxisMeasBasis(Axis.X, Sign.PLUS))
        input_gates[magic] = "T" if positive else "T_DAG"
        cflow[probe] = {node: ca.S if positive else ca.inverse(ca.S)}
        zflow[magic] = {node}
        for source, targets in xflow.items():
            if node in targets:
                zflow.setdefault(source, set()).symmetric_difference_update({probe})
    return input_gates, cflow
