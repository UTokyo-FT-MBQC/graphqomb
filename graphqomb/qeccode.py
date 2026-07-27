"""QEC Code object."""

from __future__ import annotations

import math
from collections import defaultdict
from enum import Enum, auto
from itertools import combinations
from typing import TYPE_CHECKING, Any, NamedTuple

from scipy.sparse import csr_array

from graphqomb.common import Axis, AxisMeasBasis, Sign
from graphqomb.graphstate import GraphState

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


_TYPE_II_CHAIN_LENGTH = 3
_MIN_ANCILLA_COLLISION_CLEARANCE = 0.5
_ANCILLA_COLLISION_CLEARANCE_RATIO = 0.05
_COORDINATE_TOLERANCE = 1e-9
_ANCILLA_ESCAPE_DIRECTIONS = (
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (1.0, 1.0),
    (1.0, -1.0),
    (-1.0, 1.0),
    (-1.0, -1.0),
)
Coordinate = tuple[float, ...]


class YFoliation(Enum):
    """Y-foliation graph-state builder variant."""

    TYPE_I = auto()
    TYPE_II = auto()


class StabilizerCode:
    """A stabilizer code."""

    def __init__(
        self,
        stabilizer_matrix: csr_array[Any, tuple[int, int]],
        *,
        stabilizer_coords: Mapping[int, Coordinate] | None = None,
        qubit_coords: Mapping[int, Coordinate] | None = None,
    ) -> None:
        self.hx = csr_array(stabilizer_matrix[:, : stabilizer_matrix.shape[1] // 2])
        self.hz = csr_array(stabilizer_matrix[:, stabilizer_matrix.shape[1] // 2 :])

        self.stabilizer_coord = stabilizer_coords
        self.qubit_coord = qubit_coords

    @property
    def num_stabilizers(self) -> int:
        """Number of stabilizers."""
        return int(self.hx.shape[0])

    @property
    def num_qubits(self) -> int:
        """Number of qubits."""
        return int(self.hx.shape[1])


class StabilizerGraphStateBuildResult(NamedTuple):
    """Result of building a graph state from a stabilizer code.

    ``graph`` is the constructed graph state with data and ancilla nodes,
    edges, coordinates, and measurement bases. ``data_nodes`` maps each
    ``(physical_qubit, data_layer)`` pair to its graph node id.
    ``ancilla_nodes`` maps each stabilizer row index to its ancilla graph node
    id.
    """

    graph: GraphState
    data_nodes: dict[tuple[int, int], int]
    ancilla_nodes: dict[int, int]


class _DataLayerPlan(NamedTuple):
    """Data-node layer layout for graph-state construction."""

    data_layers: dict[int, tuple[int, ...]]
    measurement_layers: dict[int, tuple[int, ...]]
    coordinate_z_by_layer: dict[tuple[int, int], float]
    meas_basis_by_qubit: dict[int, AxisMeasBasis]
    y_foliation: YFoliation
    data_as_io: bool
    qubit_indices: Mapping[int, int] | None


class _StabilizerSupport(NamedTuple):
    """Sparse support sets for one stabilizer row."""

    hx: set[int]
    hz: set[int]


def build_graph_state(
    code: StabilizerCode,
    z_base: int = 0,
    *,
    y_foliation: YFoliation = YFoliation.TYPE_I,
    data_as_io: bool = False,
    qubit_indices: Mapping[int, int] | None = None,
) -> StabilizerGraphStateBuildResult:
    r"""Build a graph-state unit from a stabilizer code.

    Parameters
    ----------
    code : `StabilizerCode`
        Stabilizer code to convert. The X support is connected to the upper
        data layer and the Z support is connected to the lower data layer.
    z_base : `int`, optional
        Lower stabilizer-measurement data-layer index, by default 0. Type I
        uses two measurement layers; Type II uses three for Y support. When
        ``data_as_io`` is enabled, a separate output layer is appended.
    y_foliation : `YFoliation`, optional
        Foliation variant. Type I measures each stabilizer ancilla in X when
        its row has an even number of Y supports, including zero, and in Y
        when the number is odd. Type II uses a three-node Y-measured data chain
        only for qubits that have an Hx=Hz=1 support in at least one stabilizer
        row. Both variants add a CZ between stabilizer ancillas when an odd
        number of shared-data-qubit pairs have opposite local interaction
        order under Z-before-Y-before-X ordering.
    data_as_io : `bool`, optional
        Whether to register the first stabilizer-measurement data nodes as
        inputs and append separate unmeasured output nodes, by default False.
    qubit_indices : `collections.abc.Mapping`\[`int`, `int`\] | `None`, optional
        Mapping from stabilizer-code qubit columns to graph qindices when
        ``data_as_io`` is enabled. If omitted, code qubit columns are used.

    Returns
    -------
    `StabilizerGraphStateBuildResult`
        Graph state and maps from stabilizer/data indices to graph nodes.

    Raises
    ------
    TypeError
        If z_base is not an integer.
    ValueError
        If ``qubit_indices`` is invalid for the requested data I/O layout.
    """
    if not isinstance(z_base, int):
        msg = "z_base must be an integer."
        raise TypeError(msg)
    if qubit_indices is not None:
        if not data_as_io:
            msg = "qubit_indices can only be used when data_as_io=True."
            raise ValueError(msg)
        expected_qubits = set(range(code.num_qubits))
        provided_qubits = set(qubit_indices)
        if provided_qubits != expected_qubits:
            missing = sorted(expected_qubits - provided_qubits)
            unexpected = sorted(provided_qubits - expected_qubits)
            msg = (
                "qubit_indices must map every stabilizer-code qubit exactly once; "
                f"missing={missing}, unexpected={unexpected}."
            )
            raise ValueError(msg)
        qindices = list(qubit_indices.values())
        if len(qindices) != len(set(qindices)):
            msg = "qubit_indices values must be unique."
            raise ValueError(msg)

    graph = GraphState()
    x_meas_basis = AxisMeasBasis(Axis.X, Sign.PLUS)

    data_layer_plan = _data_layer_plan(
        code,
        z_base=z_base,
        y_foliation=y_foliation,
        data_as_io=data_as_io,
        qubit_indices=qubit_indices,
    )
    data_nodes = _add_layered_data_nodes(graph, code, data_layer_plan)
    ancilla_nodes = _add_ancilla_nodes(graph, code, data_nodes, data_layer_plan, x_meas_basis)

    return StabilizerGraphStateBuildResult(graph, data_nodes, ancilla_nodes)


def _data_layer_plan(
    code: StabilizerCode,
    *,
    z_base: int,
    y_foliation: YFoliation,
    data_as_io: bool,
    qubit_indices: Mapping[int, int] | None,
) -> _DataLayerPlan:
    data_layers: dict[int, tuple[int, ...]] = {}
    measurement_layers: dict[int, tuple[int, ...]] = {}
    coordinate_z_by_layer: dict[tuple[int, int], float] = {}
    meas_basis_by_qubit: dict[int, AxisMeasBasis] = {}
    x_meas_basis = AxisMeasBasis(Axis.X, Sign.PLUS)
    y_meas_basis = AxisMeasBasis(Axis.Y, Sign.PLUS)
    y_chain_qubits: set[int] = _qubits_with_y_support(code) if y_foliation is YFoliation.TYPE_II else set()

    for qubit in range(code.num_qubits):
        measured_layers: tuple[int, ...]
        measured_coordinate_zs: tuple[float, ...]
        if qubit in y_chain_qubits:
            measured_layers = (z_base, z_base + 1, z_base + 2)
            measured_coordinate_zs = (
                float(z_base),
                float(z_base) + 0.5,
                float(z_base + 1),
            )
            meas_basis = y_meas_basis
        else:
            measured_layers = (z_base, z_base + 1)
            measured_coordinate_zs = (float(z_base), float(z_base + 1))
            meas_basis = x_meas_basis

        if data_as_io:
            layers = (*measured_layers, measured_layers[-1] + 1)
            coordinate_zs = (*measured_coordinate_zs, float(z_base + 2))
        else:
            layers = measured_layers
            coordinate_zs = measured_coordinate_zs

        data_layers[qubit] = layers
        measurement_layers[qubit] = measured_layers
        meas_basis_by_qubit[qubit] = meas_basis
        for layer, coordinate_z in zip(layers, coordinate_zs, strict=True):
            coordinate_z_by_layer[qubit, layer] = coordinate_z

    return _DataLayerPlan(
        data_layers=data_layers,
        measurement_layers=measurement_layers,
        coordinate_z_by_layer=coordinate_z_by_layer,
        meas_basis_by_qubit=meas_basis_by_qubit,
        y_foliation=y_foliation,
        data_as_io=data_as_io,
        qubit_indices=qubit_indices,
    )


def _add_layered_data_nodes(
    graph: GraphState,
    code: StabilizerCode,
    data_layer_plan: _DataLayerPlan,
) -> dict[tuple[int, int], int]:
    """Add layered data nodes.

    Returns
    -------
    `dict`[`tuple`[`int`, `int`], `int`]
        Mapping from physical qubit and z layer to graph node.
    """
    data_nodes: dict[tuple[int, int], int] = {}
    for qubit in range(code.num_qubits):
        previous_node: int | None = None
        layers = data_layer_plan.data_layers[qubit]
        measured_layers = set(data_layer_plan.measurement_layers[qubit])
        for layer in layers:
            node = graph.add_node(
                coordinate=_data_coordinate(code, qubit, data_layer_plan.coordinate_z_by_layer[qubit, layer])
            )
            if previous_node is not None:
                graph.add_edge(previous_node, node)
            if layer in measured_layers:
                graph.assign_meas_basis(node, data_layer_plan.meas_basis_by_qubit[qubit])
            data_nodes[qubit, layer] = node
            previous_node = node

        if data_layer_plan.data_as_io:
            q_index = data_layer_plan.qubit_indices[qubit] if data_layer_plan.qubit_indices is not None else qubit
            graph.register_input(data_nodes[qubit, data_layer_plan.measurement_layers[qubit][0]], q_index)
            graph.register_output(data_nodes[qubit, layers[-1]], q_index)

    return data_nodes


def _add_ancilla_nodes(
    graph: GraphState,
    code: StabilizerCode,
    data_nodes: dict[tuple[int, int], int],
    data_layer_plan: _DataLayerPlan,
    meas_basis: AxisMeasBasis,
) -> dict[int, int]:
    """Add ancilla nodes and stabilizer-support edges.

    Returns
    -------
    `dict`[`int`, `int`]
        Mapping from stabilizer row index to graph node.
    """
    ancilla_nodes: dict[int, int] = {}
    supports = _stabilizer_supports(code)
    y_meas_basis = AxisMeasBasis(Axis.Y, Sign.PLUS)
    for stabilizer, support in enumerate(supports):
        explicit_ancilla_coord = _explicit_ancilla_coordinate(code, stabilizer)
        ancilla_node = graph.add_node(coordinate=explicit_ancilla_coord)
        has_odd_y_support = len(support.hx & support.hz) % 2 == 1
        ancilla_meas_basis = (
            y_meas_basis if data_layer_plan.y_foliation is YFoliation.TYPE_I and has_odd_y_support else meas_basis
        )
        graph.assign_meas_basis(ancilla_node, ancilla_meas_basis)
        ancilla_nodes[stabilizer] = ancilla_node

        connected_data_nodes = _connect_stabilizer_support(
            graph,
            ancilla_node=ancilla_node,
            support=support,
            data_nodes=data_nodes,
            data_layer_plan=data_layer_plan,
        )

        if explicit_ancilla_coord is None:
            inferred_coord = _average_node_coordinates(graph, connected_data_nodes)
            if inferred_coord is not None:
                graph.set_coordinate(ancilla_node, _avoid_occupied_coordinate(graph, inferred_coord))

    for left_stabilizer, right_stabilizer in _twisted_stabilizer_pairs(supports):
        graph.add_edge(ancilla_nodes[left_stabilizer], ancilla_nodes[right_stabilizer])

    return ancilla_nodes


def _twisted_stabilizer_pairs(supports: Sequence[_StabilizerSupport]) -> list[tuple[int, int]]:
    """Return stabilizer pairs whose ancillas require a CZ edge.

    Local interactions on each shared data qubit are ordered Z, Y, then X.
    For a stabilizer pair, let ``forward`` and ``reverse`` be the numbers of
    shared qubits with the two possible strict orders. The number of twisted
    qubit pairs is ``forward * reverse``, so only the parity of each direction
    is required. Equal-Pauli overlaps have no strict order and are ignored.

    Candidate stabilizer pairs are generated from per-qubit incidence lists,
    avoiding a scan over all stabilizer pairs when supports are sparse.

    Returns
    -------
    `list`[`tuple`[`int`, `int`]]
        Sorted stabilizer-row pairs requiring an ancilla CZ edge.
    """
    incidences_by_qubit: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for stabilizer, support in enumerate(supports):
        qubits_by_order = (
            support.hz - support.hx,
            support.hx & support.hz,
            support.hx - support.hz,
        )
        for order, qubits in enumerate(qubits_by_order):
            for qubit in qubits:
                incidences_by_qubit[qubit].append((stabilizer, order))

    # Incidence lists are in ascending stabilizer order, so combinations
    # always yield stabilizer_a < stabilizer_b.
    parities: dict[tuple[int, int], list[bool]] = defaultdict(lambda: [False, False])
    for incidences in incidences_by_qubit.values():
        for (stabilizer_a, order_a), (stabilizer_b, order_b) in combinations(incidences, 2):
            if order_a == order_b:
                continue
            direction = 0 if order_a < order_b else 1
            parities[stabilizer_a, stabilizer_b][direction] ^= True

    return sorted(pair for pair, (forward_odd, reverse_odd) in parities.items() if forward_odd and reverse_odd)


def _connect_stabilizer_support(
    graph: GraphState,
    *,
    ancilla_node: int,
    support: _StabilizerSupport,
    data_nodes: dict[tuple[int, int], int],
    data_layer_plan: _DataLayerPlan,
) -> list[int]:
    connected_data_nodes: list[int] = []
    for qubit in sorted(support.hx | support.hz):
        layers = data_layer_plan.measurement_layers[qubit]
        if data_layer_plan.y_foliation is YFoliation.TYPE_II and len(layers) == _TYPE_II_CHAIN_LENGTH:
            layer = _type_ii_support_layer(layers, has_x=qubit in support.hx, has_z=qubit in support.hz)
            data_node = data_nodes[qubit, layer]
            graph.add_edge(ancilla_node, data_node)
            connected_data_nodes.append(data_node)
            continue

        if qubit in support.hz:
            data_node = data_nodes[qubit, layers[0]]
            graph.add_edge(ancilla_node, data_node)
            connected_data_nodes.append(data_node)
        if qubit in support.hx:
            data_node = data_nodes[qubit, layers[-1]]
            graph.add_edge(ancilla_node, data_node)
            connected_data_nodes.append(data_node)

    return connected_data_nodes


def _type_ii_support_layer(layers: tuple[int, ...], *, has_x: bool, has_z: bool) -> int:
    if has_z and not has_x:
        return layers[0]
    if has_z and has_x:
        return layers[1]
    return layers[2]


def _stabilizer_supports(code: StabilizerCode) -> list[_StabilizerSupport]:
    """Return sparse X/Z support sets for every stabilizer row.

    Returns
    -------
    `list`[`_StabilizerSupport`]
        Support sets indexed by stabilizer row.
    """
    hx = code.hx.copy()
    hz = code.hz.copy()
    hx.eliminate_zeros()
    hz.eliminate_zeros()
    return [
        _StabilizerSupport(
            hx=set(_row_support(hx, stabilizer)),
            hz=set(_row_support(hz, stabilizer)),
        )
        for stabilizer in range(code.num_stabilizers)
    ]


def _qubits_with_y_support(code: StabilizerCode) -> set[int]:
    y_qubits: set[int] = set()
    for support in _stabilizer_supports(code):
        y_qubits.update(support.hx & support.hz)
    return y_qubits


def _data_coordinate(code: StabilizerCode, qubit: int, z: float) -> tuple[float, float, float] | None:
    """Return the 3D coordinate of a layered data node.

    Returns
    -------
    `tuple`[`float`, `float`, `float`] | `None`
        Lifted 3D coordinate, or None when the qubit has no coordinate.
    """
    if code.qubit_coord is None or qubit not in code.qubit_coord:
        return None
    coord = code.qubit_coord[qubit]
    return (float(coord[0]), float(coord[1]), float(z))


def _explicit_ancilla_coordinate(code: StabilizerCode, stabilizer: int) -> tuple[float, float, float] | None:
    """Return an explicitly supplied ancilla coordinate, if present.

    Returns
    -------
    `tuple`[`float`, `float`, `float`] | `None`
        Explicit 3D coordinate, or None when no coordinate is supplied.
    """
    if code.stabilizer_coord is None or stabilizer not in code.stabilizer_coord:
        return None
    coord = code.stabilizer_coord[stabilizer]
    return (float(coord[0]), float(coord[1]), float(coord[2]))


def _row_support(matrix: csr_array, row: int) -> list[int]:
    """Return nonzero column indices in a CSR sparse row.

    Returns
    -------
    `list`[`int`]
        Nonzero column indices for the row.
    """
    start = int(matrix.indptr[row])
    end = int(matrix.indptr[row + 1])
    return [int(col) for col in matrix.indices[start:end]]


def _average_node_coordinates(graph: GraphState, nodes: list[int]) -> tuple[float, float, float] | None:
    """Return the componentwise average of node coordinates when all are available.

    Returns
    -------
    `tuple`[`float`, `float`, `float`] | `None`
        Average coordinate, or None when no average can be inferred.
    """
    if not nodes:
        return None
    coordinates = graph.coordinates
    if any(node not in coordinates for node in nodes):
        return None

    return (
        sum(coordinates[node][0] for node in nodes) / len(nodes),
        sum(coordinates[node][1] for node in nodes) / len(nodes),
        sum(coordinates[node][2] for node in nodes) / len(nodes),
    )


def _avoid_occupied_coordinate(
    graph: GraphState,
    coordinate: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Move an inferred ancilla coordinate aside when its centroid is occupied.

    The temporal coordinate is preserved. Candidate positions expand
    deterministically in the data plane until one has enough clearance from
    every node whose coordinate has already been assigned.

    Returns
    -------
    `tuple`[`float`, `float`, `float`]
        The original coordinate when unoccupied, otherwise a nearby free one.

    Raises
    ------
    ValueError
        If no finite candidate can be found within the bounded search.
    """
    if not all(math.isfinite(value) for value in coordinate[:2]):
        return coordinate

    occupied_coordinates = tuple(
        occupied for occupied in graph.coordinates.values() if all(math.isfinite(value) for value in occupied[:2])
    )
    if not any(math.dist(coordinate[:2], occupied[:2]) <= _COORDINATE_TOLERANCE for occupied in occupied_coordinates):
        return coordinate

    x_coordinates = tuple(occupied[0] for occupied in occupied_coordinates)
    y_coordinates = tuple(occupied[1] for occupied in occupied_coordinates)
    scaled_coordinate_span = max(
        max(x_coordinates) * _ANCILLA_COLLISION_CLEARANCE_RATIO
        - min(x_coordinates) * _ANCILLA_COLLISION_CLEARANCE_RATIO,
        max(y_coordinates) * _ANCILLA_COLLISION_CLEARANCE_RATIO
        - min(y_coordinates) * _ANCILLA_COLLISION_CLEARANCE_RATIO,
    )
    clearance = max(_MIN_ANCILLA_COLLISION_CLEARANCE, scaled_coordinate_span)

    x, y, z = coordinate
    max_radius = 3 * len(occupied_coordinates) + 1
    for radius in range(1, max_radius + 1):
        distance = radius * clearance
        for dx, dy in _ANCILLA_ESCAPE_DIRECTIONS:
            candidate = (x + dx * distance, y + dy * distance, z)
            if all(math.isfinite(value) for value in candidate[:2]) and all(
                math.dist(candidate[:2], occupied[:2]) >= clearance - _COORDINATE_TOLERANCE
                for occupied in occupied_coordinates
            ):
                return candidate

    msg = "could not find a finite coordinate for the inferred ancilla"
    raise ValueError(msg)
