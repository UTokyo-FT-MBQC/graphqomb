Fault-tolerant quantum computing
================================

The Steane examples draw and simulate logical T gate teleportation using a
supplied encoded magic state. T-state preparation and distillation circuits are outside
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

``steane_t_injection_noise.py`` samples the same teleportation with
measurement noise in Clifft (``graphqomb[clifft]``, Python 3.12 or later). The
conditional S is decided by a teleportation bit decoded online from the
destructive readout, sampled by fixing each branch with ``fixed_controls``
and keeping the shots whose decoded bit selects it. Kept shots are decoded
with a lookup decoder of each branch's detector error model; the logical
error rate scales as ``p^2``.

``cultivation_import.py`` imports the distance-3 T-gate cultivation circuit
distributed with Clifft, downloaded from a pinned commit, using both T
lowerings of ``clifft_text_to_pattern``. Clifft sampling after MBQC
compilation is compared with sampling of the source circuit.
