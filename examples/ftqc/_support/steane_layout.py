"""XY Steane geometry and a resource graph stacked along the time Z axis."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axis3d import Axis as Axis3D

from .steane import CODE_DISTANCE, LOGICAL_SUPPORT, MEMORY_ROUNDS, PATCH_VERTICES, STEANE_SUPPORTS

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    from .steane import SteanePattern


def plot_steane_injection(steane: SteanePattern) -> Figure:
    """Draw logical gate teleportation from a supplied encoded T state.

    Returns
    -------
    matplotlib.figure.Figure
        Spatial and temporal coordinates with no temporal displacement in XY.
    """
    fig = plt.figure(figsize=(16, 13))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 4], width_ratios=[3, 1])
    logical_ax = fig.add_subplot(grid[0, :])
    resource_ax = fig.add_subplot(grid[1, 0], projection="3d")
    references = grid[1, 1].subgridspec(2, 1, height_ratios=[1, 2], hspace=0.2)
    coupling_ax = fig.add_subplot(references[0])
    patch_ax = fig.add_subplot(references[1])
    _draw_logical_protocol(logical_ax)
    _draw_resource(resource_ax, steane)
    _draw_cnot_xy(coupling_ax, steane)
    _draw_xy_reference(patch_ax)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.10, top=0.96, hspace=0.10, wspace=0.04)
    return fig


def _draw_logical_protocol(ax: Axes) -> None:
    """Show magic-controlled transversal CX and transfer to the magic block."""
    ax.plot([0.0, 4.7], [1.0, 1.0], color="black")
    ax.plot([0.0, 10.7], [0.0, 0.0], color="black")
    ax.text(-0.15, 1.0, r"Data $|\psi_L\rangle$", ha="right", va="center")
    ax.text(-0.15, 0.0, r"Input $|T_L\rangle$", ha="right", va="center")
    for y in (0.0, 1.0):
        ax.text(1.0, y, "Memory", ha="center", va="center", bbox={"facecolor": "#EAF2F8", "edgecolor": "#34495E"})
    ax.plot([3.0, 3.0], [0.0, 1.0], color="#D55E00", linewidth=1.4)
    ax.plot(3.0, 0.0, "o", color="#D55E00", markersize=6)
    ax.plot(3.0, 1.0, "o", markerfacecolor="white", markeredgecolor="#D55E00", markersize=14)
    ax.plot([2.85, 3.15], [1.0, 1.0], color="#D55E00")
    ax.plot([3.0, 3.0], [0.85, 1.15], color="#D55E00")
    ax.text(3.0, -0.4, r"$\mathrm{CX}^{\otimes 7}$", ha="center", color="#D55E00")
    ax.text(4.7, 1.0, r"$M_{Z_L}$", ha="center", va="center", bbox={"facecolor": "white", "edgecolor": "black"})
    ax.text(4.7, 1.35, "Consumed data", ha="center", fontsize=9)
    ax.text(6.1, 0.0, r"$S_L^m X_L^m$", ha="center", va="center", bbox={"facecolor": "white", "edgecolor": "#8E44AD"})
    ax.text(
        8.2,
        0.0,
        f"Memory\n$r=d={CODE_DISTANCE}$ rounds",
        ha="center",
        va="center",
        bbox={"facecolor": "#EAF2F8", "edgecolor": "#34495E"},
    )
    ax.text(10.7, 0.0, r"$M_{X_L/Z_L}$", ha="center", va="center", bbox={"facecolor": "white", "edgecolor": "black"})
    ax.text(9.0, -0.6, r"Transferred output: $T_L|\psi_L\rangle$", ha="center")
    ax.annotate(
        "m",
        xy=(6.1, 0.2),
        xytext=(5.1, 1.0),
        color="#8E44AD",
        arrowprops={"arrowstyle": "->", "linestyle": "--", "color": "#8E44AD", "connectionstyle": "arc3,rad=-0.2"},
    )
    ax.set_xlim(-1.8, 12.1)
    ax.set_ylim(-1.0, 1.9)
    ax.set_title("(a) Logical T gate teleportation with a supplied encoded T state", loc="left")
    ax.axis("off")


def _draw_xyz_patch(ax: Axes3D, nodes: tuple[int, ...], coordinates: Mapping[int, tuple[float, float, float]]) -> None:
    """Place the three stabilizer faces in one constant-Z code plane."""
    for support, color in zip(STEANE_SUPPORTS, ("#F7C6C7", "#C7E9C0", "#C6DBEF"), strict=True):
        vertices = [coordinates[nodes[q]] for q in support]
        center_x = sum(x for x, _, _ in vertices) / len(vertices)
        center_y = sum(y for _, y, _ in vertices) / len(vertices)
        vertices.sort(key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x))
        ax.add_collection3d(
            Poly3DCollection([vertices], facecolor=color, edgecolor="#AAB7B8", alpha=0.20, linewidth=0.5)
        )


def _draw_syndrome_ancillas(
    ax: Axes3D, steane: SteanePattern, coordinates: Mapping[int, tuple[float, float, float]]
) -> None:
    """Label the data input, magic input, and surviving-block memory checks."""
    stages = [("D0", steane.before_nodes), ("M0", steane.magic_before_nodes)]
    stages.extend((f"M{index}", nodes) for index, nodes in enumerate(steane.after_nodes, 1))
    for stage, nodes in stages:
        for row, node in enumerate(nodes):
            x, y, z = coordinates[node]
            # Matplotlib accepts float Z coordinates; its untyped zs=0 default is inferred as int.
            ax.scatter(
                x,
                y,
                z,  # pyright: ignore[reportArgumentType]
                marker="s",
                s=80,
                facecolor="#34495E",
                depthshade=False,
            )
            if row % len(STEANE_SUPPORTS) == 0:
                label = f"{stage}:{'X' if row < len(STEANE_SUPPORTS) else 'Z'}1"
                ax.text(x - 1.0, y - 0.25, z + 0.10, label, fontsize=8)


def _draw_resource(ax: Axes3D, steane: SteanePattern) -> None:
    """Draw the imported graph, seven pairwise CX couplings, and classical controls."""
    graph = steane.pattern.frame.graphstate
    coordinates = {node: (coord[0], coord[1], coord[2]) for node, coord in graph.coordinates.items()}
    cnot_edges = {frozenset(edge) for edge in steane.cnot_edges}
    for edge in graph.edges:
        coupling = frozenset(edge) in cnot_edges
        color = "#D55E00" if coupling else "#66777D"
        ax.plot(
            *zip(*(coordinates[node] for node in edge), strict=True),
            color=color,
            linewidth=1.8 if coupling else 0.8,
            alpha=1.0 if coupling else 0.65,
        )
    for nodes in (steane.data_input_nodes, steane.magic_input_nodes, steane.data_nodes):
        _draw_xyz_patch(ax, nodes, coordinates)
    for x, y, z in coordinates.values():
        ax.scatter(x, y, z, s=60, facecolor="white", edgecolor="black", depthshade=False)  # pyright: ignore[reportArgumentType]
    _draw_syndrome_ancillas(ax, steane, coordinates)
    control = coordinates[steane.control_node]
    for qubit in LOGICAL_SUPPORT[:-1]:
        origin = coordinates[steane.consumed_nodes[qubit]]
        ax.quiver(
            *origin,
            *(b - a for a, b in zip(origin, control, strict=True)),
            color="#0072B2",
            linestyle="-.",
            linewidth=1.0,
            arrow_length_ratio=0.08,
        )
    for target in steane.correction_nodes:
        destination = coordinates[target]
        ax.quiver(
            *control,
            *(b - a for a, b in zip(control, destination, strict=True)),
            color="#8E44AD",
            linestyle="--",
            linewidth=0.9,
            arrow_length_ratio=0.08,
        )
    ax.text(control[0] - 0.3, control[1] + 0.5, control[2] + 0.5, "m: logical Z parity", fontsize=9, color="#0072B2")
    xs, ys, zs = zip(*coordinates.values(), strict=True)
    ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
    ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)
    ax.set_zlim(
        min(point[2] for point in coordinates.values()) - 0.3, max(point[2] for point in coordinates.values()) + 0.8
    )
    ax.set_xlabel("X: spatial position", labelpad=10)
    ax.set_ylabel("Y: spatial position", labelpad=10)
    ax.set_zlabel("Z: resource layer", labelpad=10)
    ax.set_box_aspect((max(xs) - min(xs) + 2, max(ys) - min(ys) + 2, max(zs) - min(zs) + 2))
    ax.view_init(elev=24, azim=-65)
    ax.set_proj_type("ortho")
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        if isinstance(axis, Axis3D):
            axis.pane.fill = False
    ax.set_title("(b) Compact MBQC resource; translated Steane patches", loc="left", pad=5)
    ax.legend(
        handles=[
            Line2D([], [], color="#66777D", label="CZ / foliation"),
            Line2D([], [], color="#D55E00", label="Seven pairwise CX couplings (CZ edges)"),
            Line2D([], [], color="#0072B2", linestyle="-.", label="Classical XOR into Z readout"),
            Line2D([], [], color="#8E44AD", linestyle="--", label="Conditional logical S (with X correction)"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=8,
    )


def _draw_cnot_xy(ax: Axes, steane: SteanePattern) -> None:
    """Show all seven couplings without occlusion from the 3D projection."""
    coordinates = steane.pattern.frame.graphstate.coordinates
    for edge in steane.cnot_edges:
        points = [coordinates[node][:2] for node in edge]
        ax.plot(*zip(*points, strict=True), color="#D55E00", linewidth=1.2, zorder=1)
    for name, nodes in (("Data", steane.data_input_nodes), ("Magic / output", steane.magic_input_nodes)):
        for qubit, node in enumerate(nodes):
            x, y = coordinates[node][:2]
            ax.scatter(x, y, s=130, facecolor="white", edgecolor="black", zorder=2)
            ax.text(x, y, str(qubit), ha="center", va="center", fontsize=8, zorder=3)
        center_x = sum(coordinates[node][0] for node in nodes) / len(nodes)
        top = max(coordinates[node][1] for node in nodes)
        ax.text(center_x, top + 0.5, name, ha="center", fontsize=9)
    ax.margins(0.12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(c) Seven transversal pairs (XY)", loc="left", fontsize=10)


def _draw_xy_reference(ax: Axes) -> None:
    """Show physical qubit labels and stabilizer faces in the spatial plane."""
    for row, (support, color) in enumerate(zip(STEANE_SUPPORTS, ("#F7C6C7", "#C7E9C0", "#C6DBEF"), strict=True), 1):
        vertices = [PATCH_VERTICES[q] for q in support]
        center_x = sum(x for x, _ in vertices) / len(vertices)
        center_y = sum(y for _, y in vertices) / len(vertices)
        vertices.sort(key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x))
        ax.add_patch(Polygon(vertices, facecolor=color, edgecolor="#AAB7B8", linewidth=0.8))
        ax.text(center_x, center_y, f"X{row} / Z{row}", ha="center", va="center", fontsize=9)
    for qubit, (x, y) in enumerate(PATCH_VERTICES):
        ax.scatter(x, y, s=280, facecolor="white", edgecolor="black", zorder=3)
        ax.text(x, y, f"q{qubit}", ha="center", va="center", fontsize=9, zorder=4)
    ax.plot([-1.8, 1.8], [-1.15, -1.15], color="#009E73", linewidth=2)
    ax.text(
        0.0, -1.55, r"$X_L=X_4X_5X_6$" + "\n" + r"$Z_L=Z_4Z_5Z_6$", ha="center", va="top", fontsize=10, color="#009E73"
    )
    ax.text(
        0.0,
        -2.5,
        "Each colored face defines\none X and one Z stabilizer.\nFaces are not CZ gates.",
        ha="center",
        va="top",
        fontsize=9,
    )
    ax.text(
        0.0,
        -3.9,
        "D0/M0: input memory rounds.\n"
        f"M1-M{MEMORY_ROUNDS}: memory after S.\n"
        "Data Z readout transfers the state.\n"
        "m controls logical X and S.",
        ha="center",
        va="top",
        fontsize=9,
        color="#0072B2",
    )
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-5.5, 2.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(d) Steane stabilizer supports", loc="left", fontsize=10)
