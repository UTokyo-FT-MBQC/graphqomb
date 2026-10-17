"""XY Steane geometry and a resource graph stacked along the time Z axis."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axis3d import Axis as Axis3D

from .steane import LOGICAL_SUPPORT, PATCH_VERTICES, STEANE_SUPPORTS

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    from .steane import SteanePattern


def plot_steane_injection(steane: SteanePattern) -> Figure:
    """Show the logical protocol, a Z-stacked resource, and its XY code patch.

    Returns
    -------
    matplotlib.figure.Figure
        Spatial and temporal coordinates with no temporal displacement in XY.
    """
    fig = plt.figure(figsize=(16, 12))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 4], width_ratios=[3, 1])
    logical_ax = fig.add_subplot(grid[0, :])
    resource_ax = fig.add_subplot(grid[1, 0], projection="3d")
    patch_ax = fig.add_subplot(grid[1, 1])
    _draw_logical_protocol(logical_ax)
    _draw_resource(resource_ax, steane)
    _draw_xy_reference(patch_ax)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.10, top=0.96, hspace=0.10, wspace=0.04)
    return fig


def _draw_logical_protocol(ax: Axes) -> None:
    """Draw the operation on the two encoded blocks and its classical controls."""
    ax.plot([0.0, 10.0], [1.0, 1.0], color="black")
    ax.plot([0.0, 4.5], [0.0, 0.0], color="black")
    ax.text(-0.15, 1.0, r"Data $|\psi_L\rangle$", ha="right", va="center")
    ax.text(-0.15, 0.0, r"Magic $|T_L\rangle$", ha="right", va="center")
    for x_position, label in ((1.0, "Syndrome\nround 1"), (6.0, "Syndrome\nround 2")):
        ax.text(x_position, 1.0, label, ha="center", va="center", bbox={"facecolor": "#EAF2F8", "edgecolor": "#34495E"})
    ax.add_patch(Rectangle((2.55, -0.2), 0.9, 1.4, facecolor="white", edgecolor="#D55E00", zorder=3))
    ax.text(3.0, 0.5, r"$M_{Z_L Z_L}$", ha="center", va="center", zorder=4)
    ax.text(3.0, -0.3, "ZZ bit p", ha="center", va="top")
    ax.text(4.5, 0.0, r"$M_{X_L}$", ha="center", va="center", bbox={"facecolor": "white", "edgecolor": "black"})
    ax.text(4.5, -0.3, "Magic parity m", ha="center", va="top")
    ax.text(8.5, 1.0, r"$S_L^p Z_L^m$", ha="center", va="center", bbox={"facecolor": "white", "edgecolor": "#8E44AD"})
    ax.text(10.0, 1.0, r"$M_{X_L}$", ha="center", va="center", bbox={"facecolor": "white", "edgecolor": "black"})
    ax.text(10.6, 1.0, "Logical X readout", ha="left", va="center")
    ax.text(9.15, 1.45, r"$T_L|\psi_L\rangle$", ha="center")
    ax.annotate(
        "",
        xy=(8.5, 1.2),
        xytext=(3.0, 1.2),
        arrowprops={"arrowstyle": "->", "linestyle": "--", "color": "#8E44AD", "connectionstyle": "arc3,rad=-0.2"},
    )
    ax.annotate(
        "",
        xy=(8.35, 0.8),
        xytext=(4.7, 0.1),
        arrowprops={"arrowstyle": "->", "linestyle": "--", "color": "#009E73"},
    )
    ax.set_xlim(-1.8, 12.0)
    ax.set_ylim(-0.8, 2.4)
    ax.set_title("(a) Logical T injection with an encoded data block and an encoded magic block", loc="left")
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
    """Label the X/Z check ancillas in both syndrome rounds."""
    for nodes in (steane.before_nodes, steane.after_nodes):
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
            label = f"{'X' if row < len(STEANE_SUPPORTS) else 'Z'}{row % len(STEANE_SUPPORTS) + 1} ({node})"
            ax.text(x + 0.15, y, z + 0.10, label, fontsize=7)


def _draw_resource(ax: Axes3D, steane: SteanePattern) -> None:
    """Draw all quantum nodes and edges in their saved XYZ coordinates."""
    graph = steane.pattern.frame.graphstate
    coordinates = {node: (coord[0], coord[1], coord[2]) for node, coord in graph.coordinates.items()}
    data_nodes = {node for layer in steane.data_layers for node in layer}
    for source, target in graph.edges:
        color = "#D55E00" if steane.probe_node in {source, target} else "#66777D"
        ax.plot(*zip(coordinates[source], coordinates[target], strict=True), color=color, linewidth=1.0, alpha=0.85)
    for nodes in (*steane.data_layers, steane.magic_nodes):
        _draw_xyz_patch(ax, nodes, coordinates)
    for node in data_nodes | set(steane.magic_nodes):
        x, y, z = coordinates[node]
        ax.scatter(
            x,
            y,
            z,  # pyright: ignore[reportArgumentType]  # Matplotlib infers zs=0 as int.
            s=115,
            facecolor="white",
            edgecolor="black",
            depthshade=False,
        )
        ax.text(x, y, z + 0.06, str(node), ha="center", va="bottom", fontsize=7)
    _draw_syndrome_ancillas(ax, steane, coordinates)
    for node, label, color in (
        (steane.probe_node, "ZZ probe: p", "#D55E00"),
        (steane.control_node, "35: stored p\nimplementation relay", "#0072B2"),
    ):
        x, y, z = coordinates[node]
        ax.scatter(
            x,
            y,
            z,  # pyright: ignore[reportArgumentType]  # Matplotlib infers zs=0 as int.
            s=150,
            facecolor="white",
            edgecolor=color,
            depthshade=False,
        )
        ax.text(x + 0.25, y, z, label, color=color, fontsize=9)
    probe = coordinates[steane.probe_node]
    control = coordinates[steane.control_node]
    ax.quiver(
        *probe,
        *(b - a for a, b in zip(probe, control, strict=True)),
        color="#0072B2",
        linestyle="-.",
        linewidth=1.2,
        arrow_length_ratio=0.08,
    )
    for target in steane.data_nodes:
        destination = coordinates[target]
        ax.quiver(
            *control,
            *(b - a for a, b in zip(control, destination, strict=True)),
            color="#8E44AD",
            linestyle="--",
            linewidth=0.9,
            arrow_length_ratio=0.08,
        )
    # m is the XOR of the three logical-X boundary records, not a new quantum node.
    magic_boundary = [coordinates[steane.magic_nodes[q]] for q in LOGICAL_SUPPORT]
    parity_origin = tuple(sum(point[axis] for point in magic_boundary) / len(magic_boundary) for axis in range(3))
    for qubit in LOGICAL_SUPPORT:
        destination = coordinates[steane.data_nodes[qubit]]
        ax.quiver(
            *parity_origin,
            *(b - a for a, b in zip(parity_origin, destination, strict=True)),
            color="#009E73",
            linestyle="--",
            linewidth=1.0,
            arrow_length_ratio=0.06,
        )
    ax.text(
        -2.0,
        -4.5,
        coordinates[steane.magic_nodes[0]][2],
        r"Encoded $|T_L\rangle$" + "\nAll 7 measured in X",
        ha="right",
        fontsize=9,
    )
    for z, label in ((0.0, "Input"), (1.0, "Round 1"), (2.0, "Injection"), (3.5, "Round 2"), (5.0, "Readout")):
        ax.text(2.0, 2.3, z + 0.12, f"{label}: Z={z:g}", fontsize=9, ha="left")
    ax.set_xlim(-3.0, 6.3)
    ax.set_ylim(-7.0, 2.8)
    ax.set_zlim(-0.15, 5.7)
    ax.set_xlabel("X: spatial position", labelpad=10)
    ax.set_ylabel("Y: spatial position", labelpad=10)
    ax.set_zlabel("Z: time / foliation layer", labelpad=10)
    ax.set_box_aspect((7, 8, 11))
    ax.view_init(elev=22, azim=-60)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        if isinstance(axis, Axis3D):
            axis.pane.fill = False
    ax.set_title("(b) XY code patches stacked in Z", loc="left", pad=5)
    ax.legend(
        handles=[
            Line2D([], [], color="#66777D", label="CZ"),
            Line2D([], [], color="#D55E00", label="CZ for logical ZZ"),
            Line2D([], [], color="#0072B2", linestyle="-.", label="Copy p to relay 35"),
            Line2D([], [], color="#8E44AD", linestyle="--", label="Conditional S"),
            Line2D([], [], color="#009E73", linestyle="--", label="Logical Z from m"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=3,
        fontsize=8,
    )


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
        "Node 35 stores the measured p bit\n"
        "until the Pauli events are finished.\n"
        "It is an IR ordering workaround,\n"
        "not part of logical T injection.",
        ha="center",
        va="top",
        fontsize=9,
        color="#0072B2",
    )
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-5.5, 2.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(c) XY Steane patch", loc="left")
