# Clifford+T cultivation import comparison

This experiment compares two MBQC lowerings of the actual T-gate distance-3
cultivation circuit. It uses Clifft for direct execution and for execution
after GraphQOMB compilation. The recorded run was performed on 2026-10-27.

## Input and reproduction

The input is Clifft's
[`circuit_d3_t_gate_p0.001.stim`](https://github.com/unitaryfoundation/clifft/blob/847757e2bc8c4f7439b9e92011550fd61719eb85/docs/guide/circuits/circuit_d3_t_gate_p0.001.stim),
from snapshot `847757e2bc8c4f7439b9e92011550fd61719eb85`. Its header attributes
the circuit to SOFT's released magic-state-cultivation circuits, Apache-2.0.
The file contains 15 physical qubits, 20 detectors, one observable, and 29
individual T/T_DAG applications, including the final diagnostic readout.
The input is not vendored into this repository.

SHA-256:
`094ae095e249c90ac7b0cf82ef187afdf634a0ef8a82331a81b4b9100a902798`.

Run from the GraphQOMB repository with its Clifft-enabled environment:

```bash
curl -L https://raw.githubusercontent.com/unitaryfoundation/clifft/847757e2bc8c4f7439b9e92011550fd61719eb85/docs/guide/circuits/circuit_d3_t_gate_p0.001.stim -o /tmp/cultivation_d3.stim
.venv/bin/python benchmarks/cultivation_import.py /tmp/cultivation_d3.stim \
  --shots 16384 --seed 17 --p 0.001 \
  --output benchmarks/results/cultivation_d3.json
```

The JSON records dependency versions, input hash, implementation-file hashes,
seed, noise parameters, resource metrics, raw counts, and timing. The run used
Python 3.13.2, GraphQOMB 0.6.0 with the accompanying changes, Stim 1.16.0, and
Clifft 0.12.0. The implementation was based on GraphQOMB commit
`b110e442ebda377f2b7afb6c07a96407329a43fb`; the JSON hashes identify the local
implementation rather than treating that base commit as the modified version.

## The two lowerings

`clifft_text_to_pattern(text, t_mode="measurement")` lowers T into
`J(pi/4), J(0)`, where `J(theta) = H Rz(theta)`. The first teleportation
measurement has angle `-pi/4`. Its observable `(X-Y)/sqrt(2)` is a Hermitian
Clifford, but is not a Pauli observable. T_DAG uses the opposite angle.
`clifft_compile` executes the stored Pauli/Clifford corrections before
applying the inverse basis rotation and measuring X. This preserves angle
adaptation rather than discarding the preceding Pauli frame.

`t_mode="injection"` replaces each such measurement with a Pauli X readout
preceded by a ZZ-probe injection gadget. A separate `T|+>` resource is
consumed; for ZZ parity `p` and magic-X outcome `m`, the data state before
correction is `Z^m T^((-1)^p)|psi>`, up to global phase. The correction
`S^p Z^m` yields `T|psi>`. T_DAG uses `T_DAG|+>` and `S_DAG^p`.
Incoming X corrections also add Z on the new probe when propagated through
its CZ edge.

Magic resources are scheduled internal preparations, not initial inputs.
Pass the returned `preparation_gates` to `clifft_compile`. The scheduler
orders each ZZ probe before the corresponding magic-X readout. Without
this order, all 29 resources can be consumed at the start, creating a
simultaneous active width of 29; that sampling trial was interrupted.
The implemented ordering reduces the compiled peak active width to 10.

## Results

The default greedy schedule, without `merge_safe_ticks`, gives:

| Quantity | Measurement lowering | Injection lowering |
| --- | ---: | ---: |
| Non-Pauli measurements | 29 | 0 |
| Scheduled magic preparations | 0 | 29 |
| Maximum live qubits | 35 | 36 |
| Pattern depth | 46 | 54 |
| Active volume | 1301 | 1515 |
| Clifft peak active width | 5 | 10 |
| Noiseless accepted shots | 16384/16384 | 16384/16384 |
| Noiseless observable flips | 0 | 0 |

Direct noiseless execution also accepts all 16384 shots and has zero
observable flips; its peak active width is 4. All runs use seed 17 and
disable syndrome normalization, so reference sampling cannot conceal a
nonzero detector or observable offset.

The script also inserts a deterministic `X 3` immediately after the first
T_DAG, with all source noise removed. All three implementations reject
those shots. Detector and observable marginals agree within the recorded
sampling tolerance of 0.0332: the maximum discrepancy is 0.00312 for
measurement lowering and 0.0107 for injection lowering. These checks
establish sampled agreement for this input and fault, not full channel
equivalence or an exhaustive fault analysis.

With `DepolarizingNoiseModel(p1=0.001, p2=0.001)` on MBQC preparations and
CZ edges:

| Trial | Accepted | Observable flips among accepted |
| --- | ---: | ---: |
| Original source noise | 11365/16384 | 0 |
| Measurement MBQC noise | 10357/16384 | 34 |
| Injection MBQC noise | 9643/16384 | 31 |

All-detector postselection is used. These are raw observable-parity counts
without decoding. A zero count does not establish zero logical error.

The three noisy rows use different error locations and are not equivalent
noise models. Import removes source noise; MBQC noise is then applied to
the complete compiled pattern, including diagnostic-readout gadgets.
The model includes no measurement-flip or idle noise. Basis-rotation gates
and conditional-Clifford synthesis gadgets are ideal. Magic preparation
rotations are ideal followed by the configured preparation noise. These
runs demonstrate executable noisy paths, not a reproduction of the
published cultivation infidelity or an overhead-optimized implementation.

## Verification

The new unit tests check T/T_DAG signs and cancellation in all sampled
injection branches, analytic T-basis readout probabilities, and joint
source-record distributions through entanglement, record feedback,
REPEAT, and resets. They also check the injection preparation order,
preparation-noise placement, and rejection of non-Pauli export by Stim.
The related Stim, Clifft, and Steane injection suite passes 743 tests;
Ruff and targeted Mypy checks pass.
