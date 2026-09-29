# 15-to-1 Clifford proxy regression input

`15to1_first_mixed_readout.stim` is the prefix through the first mixed data/syndrome
readout and its following annotations from the existing k=1, d=3 input
`FTQC-compiler-survey/stim2gqomb/lattice_surgery/msd15to1/msd15to1_proxy_k1_noiseless.stim`.
The original coordinates and qubit IDs are preserved. Partial OBSERVABLE_INCLUDE
terms are omitted: their complete parities require the full factory circuit. The first reordered record
is 255 (zero based); this prefix contains 464 measurements
and 330 detectors.

Full input SHA-256: `74262d7e87e9440cb4254af30c412ab79ab753aa808bfefccb0b8c9d4fa11463`.
Full input counts: 332 qubits, 8,361 measurements, 6,714 detectors, 5 observables.
The full input is a Clifford proxy, not T-state injection. Maximum graph degree
four is a regression condition for this 15-to-1 construction only, not a
requirement on logical Y or arbitrary Clifford circuits.

## Full-input validation

The full k=1 input was compiled in the separate development worktree with
`coord_dims=2`, `merge_safe_ticks=False`, and the default greedy scheduler.
Existing survey outputs were not overwritten.

| Check | Result |
| --- | --- |
| Original/emitted measurement records | 8,361 / 8,361 |
| Record permutation and original source qubits | All 8,361 matched |
| Detector references and tags | All 6,714 matched after inverse record mapping |
| Observable references | All 5 matched after inverse record mapping |
| Pattern graph | 23,502 nodes, 38,154 edges |
| Maximum degree, including temporal edges | 4 |
| Detector determinism | 6,714 / 6,714 |
| Rewritten noiseless samples | 8 shots, seed 0, no detector or observable flips |

The local suite passed 2,041 tests, including the independent eager-reset
record-channel comparisons and postselected state-vector comparisons. Ruff
(check and format), mypy, pyright, ty, and a warning-as-error Sphinx build
passed. No degree bound is imposed on logical Y or general Clifford examples.

Reproduce the full compile from the GraphQOMB worktree in this workspace:

```sh
uv run --no-sync python ../FTQC-compiler-survey/stim2gqomb/lattice_surgery/msd15to1/compile_graphqomb.py \
  --input ../FTQC-compiler-survey/stim2gqomb/lattice_surgery/msd15to1/msd15to1_proxy_k1_noiseless.stim \
  --output /tmp/photonic-15to1/msd15to1_proxy_k1.ptn
```

The external input and compile script are not needed for the checked-in prefix
test. The script's older option description still names disposable gadgets;
the installed GraphQOMB worktree selects the new implementation. Its JSON
records the input hash, source hashes, environment, options, and output counts.

Validation environment: Python 3.13.2, graphqomb 0.5.3, stim 1.16.0, numpy 2.4.6, networkx 3.6.1, ortools 9.15.6755.
