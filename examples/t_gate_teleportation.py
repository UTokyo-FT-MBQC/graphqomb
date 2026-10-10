"""
T gate teleportation with Clifford feedforward
============================================================

A five-node chain implements T using one magic-state input and a conditional
S correction. The first and last teleportations supply the Hadamards around
the three-node H T H injection gadget. Every measurement branch is accepted.

Signal shifting retains the Clifford target and its causal past, while moving
the final Pauli-only measurement's outcome flip into the output frame.
"""

import math

import numpy as np

from graphqomb import clifford_algebra
from graphqomb.common import Axis, AxisMeasBasis, Sign
from graphqomb.feedforward import signal_shifting
from graphqomb.graphstate import GraphState
from graphqomb.qompiler import qompile
from graphqomb.simulator import PatternSimulator, SimulatorBackend
from graphqomb.statevec import StateVector

graph = GraphState()
for _ in range(5):
    graph.add_node()
graph.register_input(0, 0)
graph.register_input(2, 1)
graph.register_output(4, 0)
for node in range(4):
    graph.add_edge(node, node + 1)
    graph.assign_meas_basis(node, AxisMeasBasis(Axis.X, Sign.PLUS))

xflow = {node: {node + 1} for node in range(4)}
zflow = {0: {2}, 1: {3}, 2: {4}}
cflow = {1: {2: clifford_algebra.S}}
shifted_xflow, shifted_zflow = signal_shifting(graph, xflow, zflow, cflow=cflow)
pattern = qompile(graph, shifted_xflow, shifted_zflow, cflow=cflow)

# Supply an arbitrary input and the magic state directly to the statevector
# backend: input initialization metadata currently describes only Pauli states.
psi = np.asarray([0.3 + 0.1j, 0.7 - 0.2j], dtype=np.complex128)
psi /= np.linalg.norm(psi)
t_gate = np.diag([1.0, np.exp(1j * math.pi / 4)])
magic_state = t_gate @ (np.ones(2, dtype=np.complex128) / math.sqrt(2))
simulator = PatternSimulator(pattern, SimulatorBackend.StateVector, calc_prob=True)
simulator.state = StateVector.from_product_states([psi, magic_state])
simulator.simulate(np.random.default_rng(7))

output = np.asarray(simulator.state.state()).ravel()
print("Measurement results:", simulator.results)
print("T output fidelity:", abs(np.vdot(t_gate @ psi, output)) ** 2)
print("Shifted X flow:", shifted_xflow)
print("Shifted Z flow:", shifted_zflow)
