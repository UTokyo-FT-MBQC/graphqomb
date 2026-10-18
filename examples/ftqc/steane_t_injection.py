r"""Logical T Injection in the Steane Code
======================================

Run the example from the repository root:

.. code-block:: sh

    uv run --extra stim python -m examples.ftqc.steane_t_injection

Encoded data and an encoded magic state occupy two ``[[7,1,3]]`` Steane blocks.
Both input blocks have a parallel initial memory round. Seven transversal
CNOTs use the magic block as control and the data block as target. Destructive
data Z readout transfers the logical state to the magic block. Its conditional
logical X/S correction is followed by ``r = d = 3`` memory rounds and then
logical measurement. X/Z readout and unmeasured quantum-output variants are
provided.

Circuit and corrections
-----------------------

``build_steane_circuit(...)`` writes the Clifford operations, ideal MPP memory
rounds, Pauli feedback, and annotations as a ``stim.Circuit``.
``build_steane_pattern(...)`` imports it with ``stim_circuit_to_pattern``, adds
conditional S as ``cflow``, transforms ``Feedforward``, and lowers it with
``qompile``. It constructs no graph nodes or edges directly.
``StimImportResult.record_to_node`` identifies the graph measurement for each
absolute source record.

Each block has X and Z stabilizers on ``{0,1,2,3}``, ``{0,1,4,5}``, and
``{0,2,4,6}``. Logical X/Z use ``{4,5,6}``. Logical zero is the equal
superposition of the eight simplex words; logical one is its complement coset.
With ``|T_L> = (|0_L> + exp(i*pi/4)|1_L>)/sqrt(2)``, the transfer is:

.. code-block:: text

    CX(magic -> data), followed by Z readout of data:
        surviving magic block = T_L X_L^m |psi_L>
        m = XOR of data Z outcomes on {4,5,6}

    Correction, applied X first and S second:
        S_L^m X_L^m T_L X_L^m |psi_L> = T_L |psi_L>

The equality is up to global phase. The output is on the former magic block;
there is no shared ZZ probe. For transversal logical teleportation with the
Steane code, see `Ryan-Anderson et al. <https://arxiv.org/abs/2404.16728>`_.
Physical ``(S^dagger)^tensor7`` implements logical S in this codeword convention.
The Stim source supplies transversal conditional X and the conditional Z
factor of physical S-dagger. Import is followed by the S coset as cflow at
those same seven correction nodes, before memory starts. The logical X
correction uses its transversal weight-seven representative.

Stim cannot express record-controlled S. The final consumed Z result is
classically relabelled by XORing the other two logical-Z support records,
using record-controlled X before its Z readout. That reported bit is ``m``,
which controls cflow without an extra quantum relay. All seven data Z results
are obtained before correction. The control result updates the Clifford frame
before the first post-correction syndrome measurement. The correction targets
are also that round's data-input nodes; no identity teleportation layers are
inserted. Its Z-check ancilla updates follow S in the frame's event order.
Commuting Z dependencies order earlier Pauli events before S without changing
the Z result. Consumed-block check supports undo the classical relabelling.
The schedule orders the control result, all three memory rounds, and final
readout, retaining that order after backpropagation.

Memory rounds and detectors
---------------------------

There are six memory checks on each input block, then three six-check rounds
on the corrected surviving block. The first post-correction round checks its
codespace directly. The next two compare corresponding syndrome results
between rounds. Final physical readout parities are compared with the last
round's syndrome on the measured axis.

Detectors include twelve initialization checks, six first-round checks,
twelve differences between rounds, three consumed-block Z checks, and,
for measured outputs, three final readout checks. X/Z variants therefore
have 36 detectors and one observable; quantum output has 33 detectors and
no classical observable.

The consumed-block checks use ``Z_data(out) = Z_magic(in) XOR Z_data(in)``.
An X-syndrome relation across conditional S would contain the product of
the logical control bit and the earlier Z syndrome. Such a relation cannot
be represented by a fixed affine detector support, so no detector joins the
pre-correction X syndrome directly to the first post-correction round.

Backpropagation runs before lowering and preserves Clifford targets and
their dependency ancestors. Parities are evaluated from stored pattern
supports, without calling legacy frame-side compilation. Correction nodes
and their ancestors retain their feedforward, while the Clifford memory
tail and logical readout compile into detector and observable supports.

Exports and figure: steane-t-injection
--------------------------------------

The gitignored ``output/`` directory receives X, Z, and quantum ``.ptn`` files,
their ``.stim`` Clifford source circuits, and PNG/PDF/SVG figures. Source Stim
circuits omit the encoded states and the conditional S coset. PTN patterns
preserve the complete correction maps; encoded inputs are supplied by
``steane_input()`` in Python because PTN initialization metadata describes only
Pauli eigenstates. Stim's default product inputs do not reproduce this experiment.
All artifacts are saved before the figure window opens, so an open window
does not leave older PTN exports in place.

The resource has 128 nodes, 14 encoded input qubits, and at most 15 live qubits
under the space-minimizing schedule. The input blocks have distinct XY
patches with the same orientation, related by the translation ``(5.4, -1.6)``;
the output occupies the magic patch. Corresponding CNOT segments are parallel
and lie on distinct XY lines. The importer supplies consecutive resource Z
layers; terminal readouts and classical feedback do not add idle layers.
These coordinates describe the resource geometry; execution timings are
stored separately in the pattern schedule. Solid edges are CZ; the seven
orange cross-block edges come from the transversal CNOTs. Dashed arrows show
classical parity aggregation and conditional S. Colored faces define code supports.

Panel (a) shows the protocol, panel (b) the imported graph, panel (c) the
seven couplings in XY without 3D occlusion, and panel (d) the physical
supports. ``_support/steane_layout.py`` generates the figure with
DejaVu Sans and editable-text SVG output. Patch geometry follows
`Postler et al., Fig. 1(c) <https://doi.org/10.1103/PRXQuantum.5.030326>`_, with
labels matched to this example's stabilizers.

Validation and scope
--------------------

.. code-block:: sh

    uv run --extra stim pytest tests/test_ftqc_steane_t_injection.py

Tests independently check all sixteen allowed encoded Z-readout words and
their ``T_L X_L^m`` branches. Statevector checks cover both logical branches,
positive/negative logical X/Z readout, the correction/round/readout order,
PTN round trips, parallel input memories, translated non-overlapping couplings,
all post-memory single-qubit Pauli faults, and a fault on every magic input
before its memory round. An incoming X error checks that S rotates its syndrome
and output state in the odd branch before post-correction memory starts.
All fifteen two-qubit Pauli faults at each of the seven CNOT entanglers are
checked for detection or an unchanged corrected output in two sampled runs.
The example itself demonstrates output fidelity and sampled logical readout;
the regression suite owns the comprehensive branch and fault checks.

Memory MPPs are ideal syndrome measurements. The imported graph does not
supply verified ancilla preparation, flag extraction, decoding, or recovery.
The example implements transversal inter-block coupling and syndrome checks;
it does not establish circuit-level fault tolerance for the complete
injection or resource preparation.
"""

# %%
import sys
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from graphqomb.common import Axis
from graphqomb.pattern import print_pattern
from graphqomb.ptn_format import dump
from graphqomb.simulator import PatternSimulator, SimulatorBackend
from graphqomb.statevec import StateVector

if TYPE_CHECKING or __package__:
    from ._support.steane import (
        build_steane_pattern,
        run_steane_shot,
        steane_basis,
        steane_input,
    )
    from ._support.steane_layout import plot_steane_injection
else:
    from _support.steane import (  # ruff:ignore[import-private-name]
        build_steane_pattern,
        run_steane_shot,
        steane_basis,
        steane_input,
    )
    from _support.steane_layout import plot_steane_injection  # ruff:ignore[import-private-name]

# %%
# Build X/Z readout patterns and an unmeasured-output variant for fidelity checks.
steane_x = build_steane_pattern(Axis.X)
steane_z = build_steane_pattern(Axis.Z)
steane_quantum = build_steane_pattern(None)
pattern = steane_x.pattern
graph = pattern.frame.graphstate
print("Resource nodes:", len(graph.nodes))
print("Pattern depth:", pattern.depth)
print("Maximum space:", pattern.max_space)
print("Transversal CNOT couplings:", steane_x.cnot_edges)
print("Logical Z correction record:", steane_x.control_node)
print_pattern(pattern)

# %%
# Each data layer and the magic input are triangular seven-qubit code patches.
fig = plot_steane_injection(steane_x)

# %%
# Export pristine patterns and the logical/resource figure.
# Encoded inputs are supplied below; .ptn initialization metadata is Pauli-only.
script_path = Path(globals().get("__file__", sys.argv[0])).resolve()
output_dir = script_path.parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / "steane_t_injection.png", dpi=150)
with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):
    fig.savefig(output_dir / "steane_t_injection.pdf")
    fig.savefig(output_dir / "steane_t_injection.svg")
for name, steane in (
    ("steane_t_injection_x", steane_x),
    ("steane_t_injection_z", steane_z),
    ("steane_t_injection_quantum", steane_quantum),
):
    output_path = output_dir / f"{name}.ptn"
    dump(steane.pattern, output_path)
    output_path.with_suffix(".stim").write_text(
        "# Clifford skeleton: encoded inputs and the S coset are supplied by the Python example.\n"
        + str(steane.source.circuit)
        + "\n"
    )
    print("Saved pattern:", output_path)

plt.show()

# %%
# Inspect the supports compiled before lowering, retaining conditional S.
# Parity compilation does not certify encoded-input stabilizer determinism.
for readout, steane in (("X", steane_x), ("Z", steane_z)):
    frame = steane.pattern.frame
    print(f"\nLogical {readout} readout: compiled feedforward")
    for tag, support in zip(frame.parity_check_tags, frame.parity_check_group, strict=True):
        print(f"  {tag}: support={sorted(support)}")
    print("  Logical observables:", {index: sorted(group) for index, group in frame.logical_observables.items()})
    print("  Retained Clifford controls:", sorted(frame.cflow))

# %%
# Prepare arbitrary encoded data and the encoded magic state.
logical_state = np.asarray([0.3 + 0.1j, 0.7 - 0.2j], dtype=np.complex128)
logical_state /= np.linalg.norm(logical_state)
zero_l, one_l = steane_basis()
expected = logical_state[0] * zero_l + np.exp(1j * np.pi / 4) * logical_state[1] * one_l

# %%
# Compare the corrected quantum output with the expected logical T state.
# Simulation mutates the frame; keep the exported pattern reusable.
simulator = PatternSimulator(deepcopy(steane_quantum.pattern), SimulatorBackend.StateVector, calc_prob=True)
simulator.state = StateVector(steane_input(logical_state))
simulator.simulate(np.random.default_rng(7))
output = np.asarray(simulator.state.state()).ravel()
print("Logical T output fidelity:", abs(np.vdot(expected, output)) ** 2)
print("Teleportation bit m:", int(simulator.results[steane_quantum.control_node]))

# %%
# Applying T to the chosen input gives a deterministic positive logical X result.
logical_x_input = np.asarray([1.0, np.exp(-1j * np.pi / 4)], dtype=np.complex128) / np.sqrt(2)
for seed in range(4):
    shot = run_steane_shot(steane_x, logical_x_input, seed=seed)
    print(f"seed={seed}, teleportation bit m={shot.teleportation_bit}")
    print("  Compiled detector parities:", tuple(int(bit) for bit in shot.detectors))
    print("  Compiled logical X parity:", int(shot.observable))

# %%
