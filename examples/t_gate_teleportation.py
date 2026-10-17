r"""T Gate Injection via ZZ Measurement
=======================================

The data qubit remains the output, while a separate magic-state input
:math:`|T\rangle = T|+\rangle` is consumed. A :math:`|+\rangle` probe couples
to both inputs by CZ gates; measuring the probe in X realizes

.. math::

    P_p = \frac{I + (-1)^p Z_{\mathrm{data}} Z_{\mathrm{magic}}}{2}.

The magic qubit is then measured in X with result :math:`(-1)^m`. The data
state is :math:`Z^m T^{(-1)^p}|\psi\rangle`, up to global phase. Applying
:math:`S^p Z^m` yields :math:`T|\psi\rangle` for all four measurement branches.

Signal shifting preserves the correction maps of this minimal injection
gadget: both measurement records feed the data output, a Clifford target.
"""

# %%
import math
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from graphqomb import clifford_algebra
from graphqomb.common import Axis, AxisMeasBasis, Sign
from graphqomb.feedforward import signal_shifting
from graphqomb.graphstate import GraphState
from graphqomb.pattern import print_pattern
from graphqomb.qompiler import qompile
from graphqomb.scheduler import Scheduler
from graphqomb.simulator import PatternSimulator, SimulatorBackend
from graphqomb.statevec import StateVector
from graphqomb.visualizer import visualize

# %%
# Build separate data and magic inputs, joined by a ZZ-measurement probe.
graph = GraphState()
data_node = graph.add_node(coordinate=(0.0, 1.0))
magic_node = graph.add_node(coordinate=(0.0, -1.0))
probe_node = graph.add_node(coordinate=(1.6, 0.0))
graph.register_input(data_node, 0)
graph.register_input(magic_node, 1)
graph.register_output(data_node, 0)
for node in (data_node, magic_node):
    graph.add_edge(probe_node, node)
for node in (probe_node, magic_node):
    graph.assign_meas_basis(node, AxisMeasBasis(Axis.X, Sign.PLUS))

# %%
# Correct the data with S for odd ZZ parity, and Z for the magic X result.
xflow: dict[int, set[int]] = {}
zflow = {magic_node: {data_node}}
cflow = {probe_node: {data_node: clifford_algebra.S}}
shifted_xflow, shifted_zflow = signal_shifting(graph, xflow, zflow, cflow=cflow)
print("Shifted X flow:", shifted_xflow)
print("Shifted Z flow:", shifted_zflow)
print("Clifford flow:", cflow)

# %%
# Visualize the two-qubit protocol and its three-node GraphQOMB realization.
fig, (protocol_ax, graph_ax) = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [1.5, 1]})
protocol_ax.plot([0.0, 5.2], [1.0, 1.0], color="black")
protocol_ax.plot([0.0, 2.8], [0.0, 0.0], color="black")
protocol_ax.text(-0.1, 1.0, r"Data $|\psi\rangle$", ha="right", va="center")
protocol_ax.text(-0.1, 0.0, r"Magic $|T\rangle$", ha="right", va="center")
protocol_ax.text(5.3, 1.0, r"$T|\psi\rangle$", ha="left", va="center")
protocol_ax.add_patch(Rectangle((1.1, -0.15), 0.9, 1.3, facecolor="white", edgecolor="black", zorder=3))
protocol_ax.text(1.55, 0.5, r"$M_{ZZ}$", ha="center", va="center", zorder=4)
protocol_ax.text(1.55, -0.25, r"$(-1)^p$", ha="center", va="top")
protocol_ax.text(2.8, 0.0, r"$M_X$", ha="center", va="center", bbox={"facecolor": "white", "edgecolor": "black"})
protocol_ax.text(2.8, -0.25, r"$(-1)^m$", ha="center", va="top")
protocol_ax.text(4.0, 1.0, r"$S^p Z^m$", ha="center", va="center", bbox={"facecolor": "white", "edgecolor": "#8E44AD"})
protocol_ax.annotate(
    "",
    xy=(4.0, 1.18),
    xytext=(1.55, 1.18),
    arrowprops={"arrowstyle": "->", "linestyle": "--", "color": "#8E44AD", "connectionstyle": "arc3,rad=-0.3"},
)
protocol_ax.annotate(
    "",
    xy=(3.85, 0.85),
    xytext=(2.95, 0.15),
    arrowprops={"arrowstyle": "->", "linestyle": "--", "color": "#8E44AD"},
)
protocol_ax.set_xlim(-1.5, 6.1)
protocol_ax.set_ylim(-0.65, 1.9)
protocol_ax.set_title("(a) ZZ-based T injection")
protocol_ax.axis("off")

# Solid graph edges represent CZ gates. The probe's X result is the ZZ parity.
visualize(graph, ax=graph_ax, show_node_labels=True, show_legend=False, node_size=600)
graph_ax.set_xlim(-0.85, 3.25)
graph_ax.set_ylim(-1.8, 1.8)
graph_ax.text(0.0, 1.25, "Data input / output", ha="center", va="bottom")
graph_ax.text(0.0, -1.25, r"Magic input $|T\rangle$" + "\nX measurement: m", ha="center", va="top")
graph_ax.text(1.85, 0.0, r"Probe $|+\rangle$" + "\nX measurement: p", ha="left", va="center")
graph_ax.set_title("(b) GraphQOMB resource graph")
graph_ax.axis("off")
fig.tight_layout()
plt.show()

# %%
# Measure the ZZ probe first, then consume the magic input by X measurement.
scheduler = Scheduler(graph, shifted_xflow, shifted_zflow, cflow=cflow)
scheduler.manual_schedule(prepare_time={probe_node: 0}, measure_time={probe_node: 1, magic_node: 2})
pattern = qompile(graph, shifted_xflow, shifted_zflow, cflow=cflow, scheduler=scheduler)
print_pattern(pattern)

# %%
# Prepare the input state and the magic state.
# Supply an arbitrary input and the magic state directly to the statevector
# backend: input initialization metadata currently describes only Pauli states.
psi = np.asarray([0.3 + 0.1j, 0.7 - 0.2j], dtype=np.complex128)
psi /= np.linalg.norm(psi)
t_gate = np.diag([1.0, np.exp(1j * math.pi / 4)])
magic_state = t_gate @ (np.ones(2, dtype=np.complex128) / math.sqrt(2))

# %%
# Simulate one measurement branch.
# Simulation updates the correction frame; copy it so this cell can be rerun.
simulator = PatternSimulator(deepcopy(pattern), SimulatorBackend.StateVector, calc_prob=True)
simulator.state = StateVector.from_product_states([psi, magic_state])
simulator.simulate(np.random.default_rng(7))

# %%
# Compare the corrected output with the expected T-gate result.
output = np.asarray(simulator.state.state()).ravel()
print("ZZ parity bit p:", int(simulator.results[probe_node]))
print("Magic X bit m:", int(simulator.results[magic_node]))
print("T output fidelity:", abs(np.vdot(t_gate @ psi, output)) ** 2)
