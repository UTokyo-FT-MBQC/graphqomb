"""PyZX integration utilities.

This module provides:

- `PyZXDiagram`: Protocol describing the PyZX graph interface used for import.
- `VertexData`: Collected PyZX vertex metadata used during import.
- `EdgeData`: Collected PyZX edge metadata used during import.
- `from_pyzx`: Convert a graph-like PyZX diagram into a `GraphState`.
"""
# ignore D102: Undocumented public method especially for Protocols, which are primarily for internal use
# and may not be directly instantiated by users.
# ruff:file-ignore[undocumented-public-method]

from __future__ import annotations

import dataclasses
import importlib
import math
from typing import TYPE_CHECKING, Protocol, SupportsFloat, TypeAlias, runtime_checkable

from graphqomb.common import Plane, PlannerMeasBasis
from graphqomb.graphstate import GraphState

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, MutableMapping
    from collections.abc import Set as AbstractSet

FloatInt: TypeAlias = float | int
FractionLike: TypeAlias = SupportsFloat
VertexType: TypeAlias = int
EdgeType: TypeAlias = int


class _PyZXVertexTypeNamespace(Protocol):
    """Static view of the PyZX vertex-type enum namespace."""

    BOUNDARY: int
    Z: int
    X: int


class _PyZXEdgeTypeNamespace(Protocol):
    """Static view of the PyZX edge-type enum namespace."""

    HADAMARD: int
    SIMPLE: int


class PyZXDiagram(Protocol):
    """Protocol covering the PyZX graph surface used by this module."""

    def copy(self) -> PyZXDiagram: ...

    def edges(self) -> Iterable[object]: ...

    def edge_st(self, edge: object) -> tuple[int, int]: ...

    def edge_type(self, edge: object) -> EdgeType: ...

    def inputs(self) -> tuple[int, ...]: ...

    def is_ground(self, vertex: int) -> bool: ...

    def neighbors(self, vertex: int) -> Iterable[int]: ...

    def outputs(self) -> tuple[int, ...]: ...

    def phase(self, vertex: int) -> FractionLike: ...

    def qubit(self, vertex: int) -> FloatInt: ...

    def remove_vertex(self, vertex: int) -> None: ...

    def row(self, vertex: int) -> FloatInt: ...

    def set_phase(self, vertex: int, phase: FractionLike) -> None: ...

    def set_type(self, vertex: int, vertex_type: VertexType) -> None: ...

    def type(self, vertex: int) -> VertexType: ...

    def vertex_degree(self, vertex: int) -> int: ...

    def vertices(self) -> Iterable[int]: ...


@runtime_checkable
class PyZXModule(Protocol):
    """Protocol for the runtime `pyzx` module attributes used here."""

    EdgeType: _PyZXEdgeTypeNamespace
    VertexType: _PyZXVertexTypeNamespace

    def is_graph_like(self, diagram: PyZXDiagram, *, strict: bool = ...) -> bool: ...


_PYZX_INSTALL_HINT = "PyZX support requires the optional dependency `graphqomb[pyzx]`."


def _require_pyzx() -> PyZXModule:
    """Import PyZX on demand for optional integration paths.

    Returns
    -------
    `PyZXModule`
        Imported `pyzx` module.

    Raises
    ------
    ModuleNotFoundError
        If the optional `pyzx` dependency is not installed.
    TypeError
        If the imported `pyzx` module does not provide the required API.
    """
    try:
        zx = importlib.import_module("pyzx")
    except ModuleNotFoundError as exc:
        msg = f"{_PYZX_INSTALL_HINT} Install it with `pip install graphqomb[pyzx]`."
        raise ModuleNotFoundError(msg) from exc

    if not isinstance(zx, PyZXModule):
        msg = "Imported `pyzx` module does not provide the API required by graphqomb."
        raise TypeError(msg)

    return zx


@dataclasses.dataclass(frozen=True, slots=True)
class VertexData:
    """Collected PyZX vertex metadata used during import.

    Attributes
    ----------
    vertex_id : `int`
        Original PyZX vertex id.
    vertex_type : ``VertexType``
        PyZX vertex type.
    phase : ``FractionLike``
        PyZX vertex phase in multiples of pi.
    qubit : ``FloatInt``
        PyZX qubit coordinate.
    row : ``FloatInt``
        PyZX row coordinate.
    is_ground : `bool`
        Whether the vertex is marked as ground in PyZX.
    """

    vertex_id: int
    vertex_type: VertexType
    phase: FractionLike
    qubit: FloatInt
    row: FloatInt
    is_ground: bool


@dataclasses.dataclass(frozen=True, slots=True)
class EdgeData:
    """Collected PyZX edge metadata used during import.

    Attributes
    ----------
    source : `int`
        Smaller endpoint id of the undirected edge.
    target : `int`
        Larger endpoint id of the undirected edge.
    edge_type : ``EdgeType``
        PyZX edge type.
    """

    source: int
    target: int
    edge_type: EdgeType


@dataclasses.dataclass(frozen=True, slots=True)
class _PyZXImportData:
    """Collected import data used to initialize a `GraphState`."""

    nodes: dict[int, VertexData]
    edges: dict[tuple[int, int], EdgeData]
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    meas_bases: dict[int, PlannerMeasBasis]
    coordinates: dict[int, tuple[float, float]]


def from_pyzx(diagram: PyZXDiagram, *, recognize_pg: bool = False) -> GraphState:
    r"""Convert a graph-like PyZX diagram into a graph state.

    Parameters
    ----------
    diagram : `PyZXDiagram`
        Input PyZX diagram in graph-like form.
    recognize_pg : `bool`, optional
        Whether to recognize supported lone-Z phase gadgets and import their
        neighbors as YZ-plane measurements.

    Returns
    -------
    `GraphState`
        Imported graph state.

    Raises
    ------
    ValueError
        If the input diagram is not in strict graph-like form or contains
        ground vertices.
    """
    pyzx = _require_pyzx()

    # Check whether the diagram is in graph-like form.
    if not pyzx.is_graph_like(diagram, strict=True):
        msg = "The input diagram is not in graph-like form. Please apply the graph-like transformation first."
        raise ValueError(msg)

    import_data = _collect_import_data(diagram, recognize_pg=recognize_pg)

    graph, _ = GraphState.from_graph(
        nodes=import_data.nodes,
        edges=import_data.edges,
        inputs=import_data.inputs,
        outputs=import_data.outputs,
        meas_bases=import_data.meas_bases,
        coordinates=import_data.coordinates,
    )

    return graph


def _collect_import_data(diagram: PyZXDiagram, *, recognize_pg: bool) -> _PyZXImportData:
    """Collect the import-time graph data derived from a PyZX diagram.

    Returns
    -------
    `_PyZXImportData`
        Topology, boundary registration, measurement bases, and coordinates
        used to initialize the imported `GraphState`.
    """
    node_map = _collect_node_map(diagram)
    _validate_no_ground_vertices(node_map)
    edge_map = _collect_edge_map(diagram)

    rewritten_inputs = _rewrite_input_boundary_maps(diagram, node_map, edge_map)
    rewritten_outputs = _rewrite_output_boundary_maps(diagram, node_map, edge_map)
    output_nodes = set(rewritten_outputs)

    pg_meas_bases: dict[int, PlannerMeasBasis] = {}
    if recognize_pg:
        pg_meas_bases = _collect_phase_gadget_meas_bases(diagram, node_map, edge_map)

    meas_bases = dict(pg_meas_bases)
    meas_bases.update(
        _build_meas_basis_map(
            node_map,
            output_nodes=output_nodes,
            excluded_nodes=set(pg_meas_bases),
        )
    )

    return _PyZXImportData(
        nodes=node_map,
        edges=edge_map,
        inputs=rewritten_inputs,
        outputs=rewritten_outputs,
        meas_bases=meas_bases,
        coordinates=_build_coordinate_map(node_map),
    )


def _validate_no_ground_vertices(node_map: Mapping[int, VertexData]) -> None:
    r"""Reject PyZX diagrams containing ground vertices.

    Parameters
    ----------
    node_map : `collections.abc.Mapping`\[`int`, `VertexData`\]
        Imported PyZX vertex metadata keyed by vertex id.

    Raises
    ------
    ValueError
        If the diagram contains ground vertices.
    """
    grounded_vertices = sorted(vertex_id for vertex_id, vertex_data in node_map.items() if vertex_data.is_ground)
    if grounded_vertices:
        msg = f"PyZX diagrams containing ground vertices are not supported for GraphState import: {grounded_vertices}"
        raise ValueError(msg)


def _build_meas_basis_map(
    node_map: Mapping[int, VertexData],
    *,
    output_nodes: AbstractSet[int],
    excluded_nodes: AbstractSet[int] | None = None,
) -> dict[int, PlannerMeasBasis]:
    r"""Build GraphState measurement bases from PyZX vertex metadata.

    Parameters
    ----------
    node_map : `collections.abc.Mapping`\[`int`, `VertexData`\]
        Imported PyZX vertex metadata keyed by vertex id.
    output_nodes : `collections.abc.Set`\[`int`\]
        Node ids that should be treated as outputs and skipped.
    excluded_nodes : `collections.abc.Set`\[`int`\] | `None`, optional
        Non-output nodes to exclude from default measurement-basis collection.

    Returns
    -------
    `dict`\[`int`, `PlannerMeasBasis`\]
        Measurement-basis assignments for imported nodes.

    Raises
    ------
    ValueError
        If an imported vertex type cannot be represented as a measurement basis.
    """
    pyzx = _require_pyzx()
    meas_bases: dict[int, PlannerMeasBasis] = {}
    skipped_nodes: set[int] = set() if excluded_nodes is None else set(excluded_nodes)

    for vertex_id, vertex_data in node_map.items():
        if (
            vertex_id in output_nodes
            or vertex_id in skipped_nodes
            or vertex_data.vertex_type == pyzx.VertexType.BOUNDARY
        ):
            continue

        if vertex_data.vertex_type == pyzx.VertexType.Z:
            plane = Plane.XY
        elif vertex_data.vertex_type == pyzx.VertexType.X:
            plane = Plane.YZ
        else:
            msg = f"Unsupported PyZX vertex type for GraphState import: {vertex_data.vertex_type}"
            raise ValueError(msg)

        meas_bases[vertex_id] = PlannerMeasBasis(plane, _phase_to_angle(vertex_data.phase))

    return meas_bases


def _build_coordinate_map(node_map: Mapping[int, VertexData]) -> dict[int, tuple[float, float]]:
    r"""Build 2D coordinates from PyZX row and qubit placement.

    Parameters
    ----------
    node_map : `collections.abc.Mapping`\[`int`, `VertexData`\]
        Imported PyZX vertex metadata keyed by vertex id.

    Returns
    -------
    `dict`\[`int`, `tuple`\[`float`, `float`\]\]
        Coordinate map keyed by PyZX vertex id.
    """
    return {
        vertex_id: (float(vertex_data.row), float(vertex_data.qubit)) for vertex_id, vertex_data in node_map.items()
    }


def _phase_to_angle(phase: FractionLike) -> float:
    """Convert a PyZX phase expressed in multiples of pi into radians.

    Parameters
    ----------
    phase : ``FractionLike``
        PyZX phase value expressed in multiples of pi.

    Returns
    -------
    `float`
        Phase angle in radians.

    Raises
    ------
    TypeError
        If the phase is symbolic and cannot be converted to a float.
    """
    try:
        return float(phase) * math.pi
    except TypeError as exc:
        msg = f"Unsupported symbolic PyZX phase for GraphState import: {phase!r}"
        raise TypeError(msg) from exc


def _collect_node_map(
    diagram: PyZXDiagram,
) -> dict[int, VertexData]:
    r"""Collect vertex metadata from a PyZX diagram.

    Parameters
    ----------
    diagram : `PyZXDiagram`
        Input PyZX diagram.

    Returns
    -------
    `dict`\[`int`, `VertexData`\]
        Vertex metadata keyed by PyZX vertex id.
    """
    node_map: dict[int, VertexData] = {}

    for vertex_id in diagram.vertices():
        node_map[vertex_id] = VertexData(
            vertex_id=vertex_id,
            vertex_type=diagram.type(vertex_id),
            phase=diagram.phase(vertex_id),
            qubit=diagram.qubit(vertex_id),
            row=diagram.row(vertex_id),
            is_ground=diagram.is_ground(vertex_id),
        )

    return node_map


def _collect_edge_map(
    diagram: PyZXDiagram,
) -> dict[tuple[int, int], EdgeData]:
    r"""Collect edge metadata from a PyZX diagram.

    Parameters
    ----------
    diagram : `PyZXDiagram`
        Input PyZX diagram.

    Returns
    -------
    `dict`\[`tuple`\[`int`, `int`\], `EdgeData`\]
        Edge metadata keyed by canonical undirected endpoint pairs.

    Raises
    ------
    ValueError
        If the diagram contains parallel edges.
    """
    edge_map: dict[tuple[int, int], EdgeData] = {}

    for edge in diagram.edges():
        source, target = diagram.edge_st(edge)

        edge_key = _edge_key(source, target)
        if edge_key in edge_map:
            msg = f"Parallel edges are not supported for PyZX import: {edge_key}"
            raise ValueError(msg)

        edge_map[edge_key] = EdgeData(
            source=edge_key[0],
            target=edge_key[1],
            edge_type=diagram.edge_type(edge),
        )

    return edge_map


def _edge_key(source: int, target: int) -> tuple[int, int]:
    r"""Return a canonical key for an undirected edge.

    Parameters
    ----------
    source : `int`
        One endpoint of the edge.
    target : `int`
        The other endpoint of the edge.

    Returns
    -------
    `tuple`\[`int`, `int`\]
        Edge endpoints ordered increasingly.
    """
    return (source, target) if source <= target else (target, source)


def _rewrite_input_boundary_maps(
    diagram: PyZXDiagram,
    node_map: MutableMapping[int, VertexData],
    edge_map: MutableMapping[tuple[int, int], EdgeData],
) -> tuple[int, ...]:
    r"""Rewrite input boundaries into GraphState-compatible node and edge maps.

    Parameters
    ----------
    diagram : `PyZXDiagram`
        Input PyZX diagram.
    node_map : `collections.abc.MutableMapping`\[`int`, `VertexData`\]
        Mutable imported vertex metadata keyed by vertex id.
    edge_map : `collections.abc.MutableMapping`\[`tuple`\[`int`, `int`\], `EdgeData`\]
        Mutable imported edge metadata keyed by canonical endpoint pairs.

    Returns
    -------
    `tuple`\[`int`, ...\]
        Imported input nodes in logical-qubit order.

    Raises
    ------
    ValueError
        If a boundary shape is unsupported or inconsistent with graph-like form.
    """
    pyzx = _require_pyzx()
    rewritten_inputs: list[int] = []
    input_vertices = diagram.inputs()

    for input_vertex in input_vertices:
        if input_vertex not in node_map:
            msg = f"Missing input vertex in collected node map: {input_vertex}"
            raise ValueError(msg)

        vertex_data = node_map[input_vertex]
        if vertex_data.vertex_type != pyzx.VertexType.BOUNDARY:
            msg = f"Input vertex must be a boundary vertex: {input_vertex}"
            raise ValueError(msg)

        neighbors = list(diagram.neighbors(input_vertex))
        if len(neighbors) != 1:
            msg = f"Input boundary must have exactly one neighbor: {input_vertex}"
            raise ValueError(msg)

        neighbor = neighbors[0]
        edge_key = _edge_key(input_vertex, neighbor)
        edge_data = edge_map.get(edge_key)
        if edge_data is None:
            msg = f"Missing incident edge for input boundary: {input_vertex}"
            raise ValueError(msg)

        if edge_data.edge_type == pyzx.EdgeType.HADAMARD:
            del node_map[input_vertex]
            del edge_map[edge_key]
            rewritten_inputs.append(neighbor)
            continue

        if edge_data.edge_type == pyzx.EdgeType.SIMPLE:
            node_map[input_vertex] = dataclasses.replace(
                vertex_data,
                vertex_type=pyzx.VertexType.Z,
                phase=0,
                is_ground=False,
            )
            edge_map[edge_key] = dataclasses.replace(edge_data, edge_type=pyzx.EdgeType.HADAMARD)
            rewritten_inputs.append(input_vertex)
            continue

        msg = f"Unsupported edge type for input boundary: {edge_data.edge_type}"
        raise ValueError(msg)

    return tuple(rewritten_inputs)


def _rewrite_output_boundary_maps(
    diagram: PyZXDiagram,
    node_map: MutableMapping[int, VertexData],
    edge_map: MutableMapping[tuple[int, int], EdgeData],
) -> tuple[int, ...]:
    r"""Rewrite output boundaries into GraphState-compatible node and edge maps.

    Parameters
    ----------
    diagram : `PyZXDiagram`
        Input PyZX diagram.
    node_map : `collections.abc.MutableMapping`\[`int`, `VertexData`\]
        Mutable imported vertex metadata keyed by vertex id.
    edge_map : `collections.abc.MutableMapping`\[`tuple`\[`int`, `int`\], `EdgeData`\]
        Mutable imported edge metadata keyed by canonical endpoint pairs.

    Returns
    -------
    `tuple`\[`int`, ...\]
        Imported output nodes in logical-qubit order.

    Raises
    ------
    ValueError
        If a boundary shape is unsupported or inconsistent with graph-like form.
    """
    pyzx = _require_pyzx()
    rewritten_outputs: list[int] = []
    output_vertices = diagram.outputs()
    next_vertex_id = max(node_map, default=-1) + 1

    for output_vertex in output_vertices:
        if output_vertex not in node_map:
            msg = f"Missing output vertex in collected node map: {output_vertex}"
            raise ValueError(msg)

        vertex_data = node_map[output_vertex]
        if vertex_data.vertex_type != pyzx.VertexType.BOUNDARY:
            msg = f"Output vertex must be a boundary vertex: {output_vertex}"
            raise ValueError(msg)

        neighbors = list(diagram.neighbors(output_vertex))
        if len(neighbors) != 1:
            msg = f"Output boundary must have exactly one neighbor: {output_vertex}"
            raise ValueError(msg)

        neighbor = neighbors[0]
        edge_key = _edge_key(output_vertex, neighbor)
        edge_data = edge_map.get(edge_key)
        if edge_data is None:
            msg = f"Missing incident edge for output boundary: {output_vertex}"
            raise ValueError(msg)

        if edge_data.edge_type == pyzx.EdgeType.HADAMARD:
            rewritten_outputs.append(output_vertex)
            continue

        if edge_data.edge_type == pyzx.EdgeType.SIMPLE:
            node_map[output_vertex] = dataclasses.replace(
                vertex_data,
                vertex_type=pyzx.VertexType.Z,
                phase=0,
                is_ground=False,
            )
            edge_map[edge_key] = dataclasses.replace(edge_data, edge_type=pyzx.EdgeType.HADAMARD)

            new_output_vertex = next_vertex_id
            next_vertex_id += 1
            node_map[new_output_vertex] = VertexData(
                vertex_id=new_output_vertex,
                vertex_type=pyzx.VertexType.BOUNDARY,
                phase=0,
                qubit=vertex_data.qubit,
                row=vertex_data.row + 1,
                is_ground=False,
            )
            new_output_edge_key = _edge_key(output_vertex, new_output_vertex)
            edge_map[new_output_edge_key] = EdgeData(
                source=new_output_edge_key[0],
                target=new_output_edge_key[1],
                edge_type=pyzx.EdgeType.HADAMARD,
            )
            rewritten_outputs.append(new_output_vertex)

            continue

        msg = f"Unsupported edge type for output boundary: {edge_data.edge_type}"
        raise ValueError(msg)

    return tuple(rewritten_outputs)


def _collect_phase_gadget_meas_bases(
    diagram: PyZXDiagram,
    node_map: MutableMapping[int, VertexData],
    edge_map: MutableMapping[tuple[int, int], EdgeData],
) -> dict[int, PlannerMeasBasis]:
    r"""Rewrite supported lone-`Z` phase gadgets for GraphState import.

    Supported patterns are phaseful degree-1 `Z` spiders connected by a
    Hadamard edge to a phase-free `Z` spider. The lone spider is removed from
    the imported node and edge maps, and its neighbor is imported as a
    `YZ`-plane measurement with the lone spider's phase.

    Parameters
    ----------
    diagram : `PyZXDiagram`
        Input PyZX diagram.
    node_map : `collections.abc.MutableMapping`\[`int`, `VertexData`\]
        Mutable imported vertex metadata keyed by vertex id.
    edge_map : `collections.abc.MutableMapping`\[`tuple`\[`int`, `int`\], `EdgeData`\]
        Mutable imported edge metadata keyed by canonical endpoint pairs.

    Returns
    -------
    `dict`\[`int`, `PlannerMeasBasis`\]
        Measurement-basis overrides for recognized phase-gadget neighbors.
    """
    candidates: list[tuple[int, int]] = []
    neighbor_counts: dict[int, int] = {}

    for vertex_id, vertex_data in node_map.items():
        neighbor = _phase_gadget_neighbor(diagram, node_map, edge_map, vertex_id, vertex_data)
        if neighbor is None:
            continue

        candidates.append((vertex_id, neighbor))
        neighbor_counts[neighbor] = neighbor_counts.get(neighbor, 0) + 1

    meas_basis_overrides: dict[int, PlannerMeasBasis] = {}
    for lone_z_spider, phase_gadget_neighbor in candidates:
        if neighbor_counts[phase_gadget_neighbor] != 1:
            continue

        lone_vertex_data = node_map.get(lone_z_spider)
        if lone_vertex_data is None:
            continue

        edge_key = _edge_key(lone_z_spider, phase_gadget_neighbor)
        if edge_key not in edge_map:
            continue

        meas_basis_overrides[phase_gadget_neighbor] = PlannerMeasBasis(
            Plane.YZ,
            _phase_to_angle(lone_vertex_data.phase),
        )
        del node_map[lone_z_spider]
        del edge_map[edge_key]

    return meas_basis_overrides


def _phase_gadget_neighbor(
    diagram: PyZXDiagram,
    node_map: Mapping[int, VertexData],
    edge_map: Mapping[tuple[int, int], EdgeData],
    vertex_id: int,
    vertex_data: VertexData,
) -> int | None:
    """Return the supported phase-gadget neighbor for a lone `Z` spider.

    Returns
    -------
    `int` | `None`
        Neighbor vertex id when the lone spider matches a supported
        phase-gadget pattern, otherwise `None`.
    """
    pyzx = _require_pyzx()
    if vertex_data.vertex_type != pyzx.VertexType.Z or vertex_data.phase == 0:
        return None
    if diagram.vertex_degree(vertex_id) != 1:
        return None

    neighbor = next(iter(diagram.neighbors(vertex_id)))
    neighbor_data = node_map.get(neighbor)
    if neighbor_data is None:
        return None
    if neighbor_data.vertex_type != pyzx.VertexType.Z or neighbor_data.phase != 0:
        return None

    edge_data = edge_map.get(_edge_key(vertex_id, neighbor))
    if edge_data is None or edge_data.edge_type != pyzx.EdgeType.HADAMARD:
        return None

    return neighbor
