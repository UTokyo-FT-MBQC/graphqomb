Fault-tolerant quantum computing
================================

This example draws and simulates logical T gate teleportation using a supplied
encoded magic state. T-state preparation and distillation circuits are outside
the simulation. Reference checks cover quantum output, detectors, and logical
readout.

``steane_t_injection.py`` applies a logical T gate to a ``[[7,1,3]]`` Steane
block using an encoded magic state, seven transversal CNOTs, destructive
data Z readout, and conditional logical X/S correction. Both input blocks
have an initial memory round. After correction, the surviving block has
``r = d = 3`` memory rounds before logical measurement.
The Clifford operations are written in Stim and imported before adding cflow.

The example explains the correction equations, detector rules, exports, and
validation scope. Its gallery-style cells export ``.ptn`` patterns to the
gitignored ``output/`` directory beside it.
