r"""Logical T Injection in the Steane Code
==========================================

Two :math:`[[7,1,3]]` blocks supply the data and an encoded magic state
:math:`|T_L\rangle`. A probe measures :math:`Z_L Z_L` with bit :math:`p`,
and transversal X measurements consume the magic block with logical bit
:math:`m`. The data branch is :math:`Z_L^m T_L^{(-1)^p}|\psi_L\rangle`.

The correction :math:`S_L^p Z_L^m` completes T injection. With the codeword
convention used here, physical :math:`(S^\dagger)^{\otimes7}` implements
:math:`S_L`. Syndrome rounds before and after the injection supply repeated
detectors; the final X/Z readout supplies three additional checks and one
logical observable. All seven magic X records contribute to three magic-block
stabilizer checks. The exported patterns retain the Clifford feedforward.

Feedforward backpropagation runs before lowering and retains Clifford
boundaries and their control dependencies. Readout checks evaluate the
compiled parity supports against independent logical and fault predictions.
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
        validate_steane,
    )
    from ._support.steane_layout import plot_steane_injection
else:
    from _support.steane import (  # ruff:ignore[import-private-name]
        build_steane_pattern,
        run_steane_shot,
        steane_basis,
        steane_input,
        validate_steane,
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
print("Logical ZZ probe:", steane_x.probe_node)
print("Delayed S correction control:", steane_x.control_node)
print_pattern(pattern)

# %%
# Each data layer and the magic input are triangular seven-qubit code patches.
fig = plot_steane_injection(steane_x)
plt.show()

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
    print("Saved pattern:", output_path)

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
# Prepare arbitrary encoded data, the encoded magic state, and the correction control.
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
print("ZZ parity bit:", int(simulator.results[steane_quantum.probe_node]))
print("Delayed correction bit:", int(simulator.results[steane_quantum.control_node]))

# %%
# Applying T to the chosen input gives a deterministic positive logical X result.
logical_x_input = np.asarray([1.0, np.exp(-1j * np.pi / 4)], dtype=np.complex128) / np.sqrt(2)
for seed in range(4):
    shot = run_steane_shot(steane_x, logical_x_input, seed=seed)
    print(f"seed={seed}, injection bits (p,m)={shot.injection}")
    print("  Compiled repeated/data-readout/magic parities:", tuple(int(bit) for bit in shot.detectors))
    print("  Compiled logical X parity:", int(shot.observable))

# %%
# Sample all four injection branches and verify every post-syndrome Pauli fault.
counts = validate_steane()
print("Validated quantum outputs:", counts["quantum_output"])
print("Sampled injection branches:", counts["injection_branches"])
print("Validated single-Pauli-fault shots:", counts["single_pauli_fault"])
print("Compiled parity and independent logical/fault checks passed.")
