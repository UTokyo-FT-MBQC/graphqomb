"""Visualization tool.

This module provides:

- `visualize` : Visualize the GraphState.
"""

from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING, NamedTuple

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import patches
from matplotlib.lines import Line2D

from graphqomb.common import Axis, Plane, determine_pauli_axis

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    from matplotlib.axes import Axes

    from graphqomb.graphstate import BaseGraphState

# Minimum number of coordinate dimensions required for 2D visualization
_MIN_2D_COORDS = 2

if sys.version_info >= (3, 11):
    from enum import StrEnum

    class _ColorMap(StrEnum):
        pass

else:
    from enum import Enum

    class _ColorMap(str, Enum):
        pass


class ColorMap(_ColorMap):
    """Color map for the nodes."""

    XY = "#2ECC71"  # Emerald green
    YZ = "#E74C3C"  # Vibrant red
    XZ = "#3498DB"  # Bright blue
    OUTPUT = "#95A5A6"  # Cool grey


class FigureSetup(NamedTuple):
    """Parameters for setting up the figure."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    padding: float
    fig_width: float
    fig_height: float


def visualize(  # ruff:ignore[too-many-arguments]
    graph: BaseGraphState,
    *,
    ax: Axes | None = None,
    show_node_labels: bool = True,
    node_size: float = 300,
    show_legend: bool = True,
    use_graph_coordinates: bool = True,
) -> Axes:
    r"""Visualize the GraphState.

    Parameters
    ----------
    graph : `BaseGraphState`
        GraphState to visualize.
    ax : `matplotlib.axes.Axes` | `None`, optional
        Matplotlib Axes to draw on, by default None
    show_node_labels : `bool`, optional
        Whether to show node index labels, by default True
    node_size : `float`, optional
        Size of nodes (scatter size), by default 300
    show_legend : `bool`, optional
        Whether to show color legend, by default True
    use_graph_coordinates : `bool`, optional
        Whether to use coordinates stored in the graph. If True and the graph
        has coordinates, those coordinates are used (projected to 2D for 3D
        coordinates). Nodes without coordinates will use auto-calculated
        positions. By default True.

    Returns
    -------
    `matplotlib.axes.Axes`
        The Axes object containing the visualization

    Notes
    -----
    Currently only 2D visualization is supported. For 3D coordinates, only the
    x and y components are used; the z component is ignored. 3D visualization
    support is planned for a future release.
    """
    node_pos = _determine_node_positions(graph, use_graph_coordinates)

    node_colors = _determine_node_colors(graph)

    # Setup figure with proper aspect ratio
    figure_setup = _setup_figure(node_pos)

    if ax is None:
        # Create figure with proper aspect ratio using plt
        _, ax = plt.subplots(figsize=(figure_setup.fig_width, figure_setup.fig_height))

    # Always set equal aspect ratio to ensure circles appear circular
    ax.set_aspect("equal")

    # Set plot limits before drawing nodes so coordinate transformation works correctly
    if node_pos:
        ax.set_xlim(figure_setup.x_min - figure_setup.padding, figure_setup.x_max + figure_setup.padding)
        ax.set_ylim(figure_setup.y_min - figure_setup.padding, figure_setup.y_max + figure_setup.padding)

    # Remove tick marks and labels for cleaner appearance
    ax.set_xticks([])
    ax.set_yticks([])

    # All nodes use the same base size for consistency

    # Draw nodes with special handling for Pauli measurements
    pauli_nodes = _find_pauli_nodes(graph)

    for node in graph.nodes:
        if node in pauli_nodes:
            # Calculate accurate patch radius for this specific position
            x, y = node_pos[node]
            patch_radius = _scatter_size_to_patch_radius(ax, x, y, node_size)
            _draw_pauli_node(ax, node_pos[node], pauli_nodes[node], patch_radius)
        else:
            # Ensure all nodes have a color, fallback to default if missing
            node_color = node_colors.get(node, ColorMap.OUTPUT)  # Default to output color
            ax.scatter(*node_pos[node], color=node_color, s=node_size, zorder=2)

    for edge in graph.edges:
        x0, y0 = node_pos[edge[0]]
        x1, y1 = node_pos[edge[1]]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color="black",
            zorder=1,
        )

    # Draw node labels if requested
    if show_node_labels:
        pauli_nodes = _find_pauli_nodes(graph)

        # Draw labels manually for better center alignment
        for node in graph.nodes:
            x, y = node_pos[node]

            # All nodes now have the same size, so use same font size calculation
            font_size = _calculate_font_size(node_size)

            ax.text(
                x,
                y,
                str(node),
                fontsize=font_size,
                ha="center",  # horizontal alignment: center
                va="center",  # vertical alignment: center
                fontweight="bold",
                color="black",
                zorder=4,  # Above all node patches
            )

    # Add color legend if requested
    if show_legend:
        _add_legend(ax, graph)
    return ax


def _setup_figure(node_pos: Mapping[int, tuple[float, float]]) -> FigureSetup:
    r"""Calculate figure dimensions and plot limits based on node positions.

    Parameters
    ----------
    node_pos : `collections.abc.Mapping`\[`int`, `tuple`\[`float`, `float`\]\]
        Dictionary mapping node indices to (x, y) positions

    Returns
    -------
    `FigureSetup`
        NamedTuple containing
        x_min, x_max, y_min, y_max, padding, fig_width, fig_height values
    """
    if node_pos:
        x_coords = [pos[0] for pos in node_pos.values()]
        y_coords = [pos[1] for pos in node_pos.values()]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Add padding around the graph
        padding = 0.5
        x_range = max(x_max - x_min, 0.5) + 2 * padding  # Minimum range to avoid too narrow plots
        y_range = max(y_max - y_min, 0.5) + 2 * padding

        # Calculate figure size to maintain equal aspect ratio
        # This ensures circles appear as circles, not ellipses
        base_size = 8.0
        # Use the same dimension for both to maintain 1:1 aspect ratio visually
        max_range = max(x_range, y_range)
        fig_width = base_size * (x_range / max_range)
        fig_height = base_size * (y_range / max_range)

        # Ensure minimum figure size for readability
        min_size = 4.0
        if fig_width < min_size or fig_height < min_size:
            scale = min_size / min(fig_width, fig_height)
            fig_width *= scale
            fig_height *= scale
    else:
        # Default size if no nodes
        fig_width = fig_height = 8.0
        x_min = x_max = y_min = y_max = 0
        padding = 0.5

    return FigureSetup(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        padding=padding,
        fig_width=fig_width,
        fig_height=fig_height,
    )


def _determine_node_positions(
    graph: BaseGraphState,
    use_graph_coordinates: bool,
) -> dict[int, tuple[float, float]]:
    r"""Get node positions, using graph coordinates if available and requested.

    Parameters
    ----------
    graph : `BaseGraphState`
        GraphState to visualize.
    use_graph_coordinates : `bool`
        Whether to use coordinates stored in the graph.

    Returns
    -------
    `dict`\[`int`, `tuple`\[`float`, `float`\]\]
        Mapping of node indices to their (x, y) positions.
    """
    if use_graph_coordinates and graph.coordinates:
        # Use graph coordinates (project 3D to 2D by using x, y only)
        node_pos: dict[int, tuple[float, float]] = {}
        for node, coord in graph.coordinates.items():
            # Take first two coordinates for 2D projection (1D uses y=0.0)
            node_pos[node] = (coord[0], coord[1] if len(coord) >= _MIN_2D_COORDS else 0.0)

        # For nodes without coordinates, calculate positions
        missing_nodes = graph.nodes - node_pos.keys()
        if missing_nodes:
            # Calculate auto positions for all nodes
            auto_pos = _calc_node_positions(graph)
            # Add only the missing nodes' positions
            for node in missing_nodes:
                node_pos[node] = auto_pos[node]

        return node_pos

    # Fall back to auto-calculated positions
    return _calc_node_positions(graph)


def _calc_node_positions(graph: BaseGraphState) -> dict[int, tuple[float, float]]:
    r"""Calculate node positions for visualization with input/output nodes arranged vertically.

    Parameters
    ----------
    graph : `BaseGraphState`
        GraphState to visualize.

    Returns
    -------
    `dict`\[`int`, `tuple`\[`float`, `float`\]\]
        Mapping of node indices to their (x, y) positions.
    """
    internal_nodes = graph.nodes - graph.input_node_indices.keys() - graph.output_node_indices.keys()

    pos: dict[int, tuple[float, float]] = {}

    # Arrange input nodes vertically on the left
    for node in graph.input_node_indices:
        pos[node] = (0.0, -graph.input_node_indices[node])

    # Arrange output nodes vertically on the right
    max_x = 2.0
    for node in graph.output_node_indices:
        pos[node] = (max_x, -graph.output_node_indices[node])

    # For internal nodes, use networkx layout to minimize crossings
    if internal_nodes:
        # Create subgraph of internal nodes and their connections
        internal_edges = [edge for edge in graph.edges if edge[0] in internal_nodes and edge[1] in internal_nodes]

        if internal_edges:
            # Use spring layout for internal nodes
            nx_graph: nx.Graph[int] = nx.Graph()
            nx_graph.add_nodes_from(internal_nodes)
            nx_graph.add_edges_from(internal_edges)
            internal_pos_raw = nx.spring_layout(nx_graph, k=1, iterations=50)
            internal_pos: dict[int, tuple[float, float]] = {
                node: (float(coord[0]), float(coord[1])) for node, coord in internal_pos_raw.items()
            }

            # Scale and position internal nodes in the middle
            for node, (x, y) in internal_pos.items():
                pos[node] = (1.0 + x * 0.8, y * 2.0)  # Center between input and output
        else:
            # If no internal edges, arrange internal nodes in a column
            for i, node in enumerate(sorted(internal_nodes)):
                pos[node] = (1.0, -i)

    return pos


def _determine_node_colors(graph: BaseGraphState) -> dict[int, ColorMap]:
    node_colors: dict[int, ColorMap] = {}
    pauli_nodes = _find_pauli_nodes(graph)

    # Set colors for all nodes with measurement bases
    for node, meas_bases in graph.meas_bases.items():
        # Skip Pauli measurements as they will be handled separately
        if node in pauli_nodes:
            continue

        if meas_bases.plane == Plane.XY:
            node_colors[node] = ColorMap.XY
        elif meas_bases.plane == Plane.YZ:
            node_colors[node] = ColorMap.YZ
        elif meas_bases.plane == Plane.XZ:
            node_colors[node] = ColorMap.XZ

    # Set colors for output nodes (may override measurement colors)
    for output_node in graph.output_node_indices:
        node_colors[output_node] = ColorMap.OUTPUT

    return node_colors


def _find_pauli_nodes(graph: BaseGraphState) -> dict[int, Axis]:
    r"""Identify nodes with Pauli measurements (Clifford angles).

    Returns
    -------
    `dict`\[`int`, `Axis`\]
        Dictionary mapping node indices to Pauli axis
    """
    pauli_nodes: dict[int, Axis] = {}

    for node, meas_bases in graph.meas_bases.items():
        pauli_axis = determine_pauli_axis(meas_bases)
        if pauli_axis:
            pauli_nodes[node] = pauli_axis

    return pauli_nodes


def _scatter_size_to_patch_radius(ax: Axes, x: float, y: float, scatter_size: float) -> float:
    """Convert scatter size to patch radius for equal display area.

    This function converts matplotlib scatter size (points²) to the equivalent
    radius in data coordinates for patches, ensuring patches have the same
    display area as scatter points with equal aspect ratio.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Matplotlib axes object
    x : `float`
        X position of the node in data coordinates
    y : `float`
        Y position of the node in data coordinates
    scatter_size : `float`
        Scatter plot size parameter (area in points²)

    Returns
    -------
    `float`
        Equivalent radius in data coordinates for patches
    """
    # Convert scatter size (points²) to radius in points
    radius_pt = math.sqrt(scatter_size / math.pi)

    # Convert points to pixels
    dpi = ax.figure.dpi if ax.figure is not None else 100.0  # Default DPI if figure is None
    radius_px = radius_pt * dpi / 72.0

    # Get transformation from data to display coordinates
    trans = ax.transData
    inv = trans.inverted()

    # Find display coordinates of the node position
    x_disp, y_disp = trans.transform((x, y))

    # Calculate data coordinate offsets in both X and Y directions
    x_offset_data = inv.transform((x_disp + radius_px, y_disp))[0] - x
    y_offset_data = inv.transform((x_disp, y_disp + radius_px))[1] - y

    # For equal aspect ratio, we want equal display area
    # Area = π * rx * ry where rx and ry are the semi-axes in data coordinates
    # For a circle with equal display area: π * r² = π * rx * ry
    # So r = sqrt(rx * ry) to maintain equal display area
    return math.sqrt(abs(x_offset_data) * abs(y_offset_data))


def _calculate_font_size(node_size: float) -> int:
    """Calculate appropriate font size based on node size that fits within the node.

    Parameters
    ----------
    node_size : `float`
        Node size parameter (scatter size in points^2)

    Returns
    -------
    `int`
        Font size for node labels that fit within the node
    """
    # Calculate the diameter of the node in points
    # scatter size is area in points^2, so diameter = 2 * sqrt(area / π)
    node_diameter_points = 2 * math.sqrt(node_size / math.pi)

    # Font size should be roughly 60% of the node diameter to fit comfortably
    # Empirically determined factor for good readability within circular nodes
    font_size = node_diameter_points * 0.4

    # Clamp to reasonable range (minimum for readability, maximum to avoid overflow)
    return max(6, min(16, int(font_size)))


def _draw_pauli_node(ax: Axes, pos: tuple[float, float], pauli_axis: Axis, node_radius: float) -> None:
    r"""Draw a Pauli measurement node with hatch patterns.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Matplotlib axes object
    pos : `tuple`\[`float`, `float`\]
        Node position (x, y)
    pauli_axis : `Axis`
        Pauli axis
    node_radius : `float`
        Radius for the node patches
    """
    x, y = pos

    # Use unified design for all Pauli measurements
    # Base color depends on the measurement plane, stripe color is contrasting
    if pauli_axis == Axis.X:
        # X measurement: XY plane
        face_color = ColorMap.XY
        edge_color = ColorMap.XZ  # Contrasting color
    elif pauli_axis == Axis.Y:
        # Y measurement: YZ plane
        face_color = ColorMap.YZ
        edge_color = ColorMap.XY  # Contrasting color
    elif pauli_axis == Axis.Z:
        # Z measurement: XZ plane
        face_color = ColorMap.XZ
        edge_color = ColorMap.YZ  # Contrasting color
    else:
        # Fallback to solid color
        circle = patches.Circle((x, y), node_radius, facecolor="black", edgecolor="none", linewidth=0, zorder=2)
        ax.add_patch(circle)
        return

    # Unified hatch pattern for all Pauli measurements
    hatch_pattern = "////////"  # Diagonal stripes for all Pauli nodes

    # Create circle patch with hatch pattern - same size as regular scatter nodes
    circle = patches.Circle(
        (x, y),
        node_radius,
        facecolor=face_color,
        edgecolor=edge_color,
        linewidth=0,  # No boundary, only hatch pattern
        hatch=hatch_pattern,
        zorder=2,
    )
    ax.add_patch(circle)


def _add_legend(ax: Axes, graph: BaseGraphState) -> None:
    """Add color legend to the plot.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Matplotlib axes object to add legend to
    graph : `BaseGraphState`
        GraphState to analyze for legend items
    """
    planes_present, pauli_measurements = _analyze_graph_measurements(graph)
    legend_elements = _create_legend_elements(graph, planes_present, pauli_measurements)

    # Add legend to the plot if there are elements to show
    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)


def _analyze_graph_measurements(graph: BaseGraphState) -> tuple[set[Plane], set[Axis]]:
    r"""Analyze graph measurements to determine legend content.

    Parameters
    ----------
    graph : `BaseGraphState`
        GraphState to analyze

    Returns
    -------
    `tuple`\[`set`\[`Plane`\], `set`\[`Axis`\]\]
        Tuple of (planes_present, pauli_measurements)
    """
    planes_present: set[Plane] = set()
    pauli_measurements: set[Axis] = set()

    for meas_bases in graph.meas_bases.values():
        planes_present.add(meas_bases.plane)

        # Check for Pauli measurements using the shared helper function
        pauli_axis = determine_pauli_axis(meas_bases)
        if pauli_axis is not None:
            pauli_measurements.add(pauli_axis)

    return planes_present, pauli_measurements


def _create_legend_elements(
    graph: BaseGraphState, planes_present: AbstractSet[Plane], pauli_measurements: AbstractSet[Axis]
) -> list[Line2D | patches.Circle]:
    r"""Create legend elements for the plot.

    Parameters
    ----------
    graph : `BaseGraphState`
        GraphState object
    planes_present : `collections.abc.Set`\[`Plane`\]
        Set of measurement planes present in graph
    pauli_measurements : `collections.abc.Set`\[`Axis`\]
        Set of Pauli measurement axes present in graph

    Returns
    -------
    `list`\[`matplotlib.lines.Line2D` | `matplotlib.patches.Circle`\]
        List of matplotlib legend elements (Line2D and Circle patches)
    """
    legend_elements: list[Line2D | patches.Circle] = []

    # Add legend entries for measurement planes
    if Plane.XY in planes_present:
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=ColorMap.XY, markersize=8, label="XY measurement")
        )

    if Plane.YZ in planes_present:
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=ColorMap.YZ, markersize=8, label="YZ measurement")
        )

    if Plane.XZ in planes_present:
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=ColorMap.XZ, markersize=8, label="XZ measurement")
        )

    # Add legend entry for output nodes if present
    if graph.output_node_indices:
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=ColorMap.OUTPUT, markersize=8, label="Output node")
        )

    # Add legend entries for Pauli measurements if present
    pauli_entries: list[patches.Circle] = []
    for pauli_axis in sorted(pauli_measurements, key=lambda x: x.name):
        # Create hatch pattern legend entry using Circle patch with same pattern as nodes
        if pauli_axis == Axis.X:
            face_color = ColorMap.XY
            edge_color = ColorMap.XZ
        elif pauli_axis == Axis.Y:
            face_color = ColorMap.YZ
            edge_color = ColorMap.XY
        elif pauli_axis == Axis.Z:
            face_color = ColorMap.XZ
            edge_color = ColorMap.YZ
        else:
            continue

        hatch_pattern = "////////"  # Dense stripes for 50/50 coverage

        # Create a circle patch for the legend with same pattern as actual nodes
        circle_patch = patches.Circle(
            (0, 0),
            0.15,  # Small radius for legend
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=0,  # No boundary, only hatch pattern
            hatch=hatch_pattern,
            label=f"Pauli {pauli_axis.name}",
        )
        pauli_entries.append(circle_patch)

    legend_elements.extend(pauli_entries)

    return legend_elements
