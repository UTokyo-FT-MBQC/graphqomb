# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **MPP rewriter validated constructively**: `rewrite_to_mpp` now pulls measurement products and residual Clifford frames out of an in-house sparse F2 symplectic tableau of each segment body, applying ancilla substitution and the residual frame only when their algebraic preconditions hold. This replaces the mirrored-circuit `flow_generators` verification and the flow-matching local-frame search, and takes Stim off the analysis path (surface-code memory d=9 r=9: 0.26 s → 0.18 s). `MppRewriteVerificationError` is removed; residual frames that were only representable through the flow-matching search now fall back gate-level.

### Added

- **In-Place Graph Composition**: `graphstate.compose_into(graph1, graph2)` composes `graph2` into `graph1` by mutation with the same connection rule and validation as `compose`, keeping `graph1` node indices stable, plus `GraphState.unregister_output()` to drop an output registration. The Stim importer's fragment fold now uses it, replacing the per-step full-graph copy (quadratic in total) with a linear fold: on the 15-to-1 lattice-surgery proxy the compose stage drops from 5.0 s to 0.4 s (k=1, 23k nodes) and 54 s to 3.3 s (k=2, 101k nodes) — end-to-end import from 175 s to 82 s at k=2. The composed graph is identical up to node relabeling.

## [0.5.3] - 2026-08-14

### Added

- **Compile-input pruning**: The new `graphqomb.pruning` module offers `prune_z_nodes`, which removes Z-prepared inputs and byproduct-corrected Z-measured nodes from a graph state and drops them from the correction flows, parity check groups, and logical observables in one pass (a Z-measured node whose byproduct is not corrected by the flows is kept), and `prune_isolated_components`, which deletes components touching neither an output node nor a logical observable seed (e.g. fragments left behind by Z pruning), where graph edges and correction-flow entries both count as connections. Preparation and measurement pruning toggle independently via `prune_preparations`/`prune_measurements`, `prune_uncorrected_measurements` also removes Z-measured nodes whose byproducts the flows do not correct (fixing the branch without a Z byproduct on their neighbors, for simulations whose results do not depend on those outcomes), and `protected_preparation_tags` protects Z-prepared inputs by their initialization tag. Pruning is sign-aware: a node measured in an inverted Z basis (angle π) fixes record 1 in the kept branch, which fires the corrections the node sources, so such a node — including a Z-prepared input — is removed only when those corrections are vacuous. Both return a `PruneResult` that preserves node indices and feeds directly into `qompile`.

- **Initialization Tags**: Stim reset instruction tags (`R[tag]`, `RX[tag]`, `RY[tag]`) survive the import/export round trip, mirroring detector tags. The new `Initialization` dataclass (axis + tag) is carried from `register_input(init=...)` through `Pattern.input_initializations`, and `stim_compile` re-emits each input's tag on its reset instruction. The `.ptn` format serializes tags as `.input_tag[tag] node` lines under new format version 4.

- **Stim Import Scheduler Passthrough**: `stim_circuit_to_pattern()`, `stim_text_to_pattern()`, and `stim_file_to_pattern()` accept `schedule_config` and `schedule_timeout`. When a `ScheduleConfig` is given, the importer builds a `Scheduler` from the imported graph and flows, solves it with that configuration (e.g. `ScheduleConfig(Strategy.MINIMIZE_TIME, use_greedy=False)` for the CP-SAT solver), and compiles the pattern with the solved schedule; a `ValueError` is raised when no schedule is found. When omitted, `qompile` keeps scheduling with its default greedy strategy. This restores external control over the emitted command order — e.g. entangling-order studies that reorder `E` commands within contiguous runs can request the CP-SAT schedule instead of pinning an old library version.

### Changed (Breaking)

- **Input initialization API**: `register_input(init_axis=...)` is now `register_input(init=Initialization(axis, tag))`, and the `input_initialization_axes` properties on graph states and `Pattern` (and the matching `from_graph` argument) are replaced by `input_initializations` holding `Initialization` values.
- **Scheduler subpackage**: `graphqomb.scheduler`, `graphqomb.greedy_scheduler`, and `graphqomb.schedule_solver` moved to `graphqomb.scheduler.core`, `graphqomb.scheduler.greedy`, and `graphqomb.scheduler.solver`. The `graphqomb.scheduler` package re-exports the public API, so import everything scheduling-related from `graphqomb.scheduler`.
- **Greedy scheduling by default**: `ScheduleConfig.use_greedy` now defaults to `True`. Pass `use_greedy=False` for the optimal CP-SAT solver.

## [0.5.2] - 2026-08-04

### Added

- **MPP Rewriter Segment Fallback**: `rewrite_to_mpp(..., fallback="segment")` keeps unsupported segments gate-level while rewriting the rest to `MPP`; the default remains `"circuit"`. Measurement post-state reuse, record feedback, and dependent producer segments fall back together, while noise remains rejected. `MppRewriteResult.fallback_segments` reports retained segments, and deterministic-minus feedback records stay signed measurements instead of unsupported `MPAD 1`. On Gidney's logical-Y circuits, node count falls from 9951 directly imported to 6237 at d=9, r=9 (8217 with safe TICK merging alone).
- **Stim Import Safe TICK Merging**: `stim_circuit_to_pattern()`, `stim_text_to_pattern()`, and `stim_file_to_pattern()` accept `merge_safe_ticks` (default off). It removes semantically redundant TICK barriers so Clifford gates can cancel or fold into resets and measurements without reordering instructions or measurement records. TICKs between Pauli-product measurement blocks remain to preserve `StimMppExtraction` grouping and commutation checks. Gidney logical-Y imports shrink by 17-19% (d=9, r=3: 7167 to 5913 nodes).

### Fixed

- **Detector Determinism with Z Measurements and Z Initialization**: `PauliFrame.detector_determinism()` credited every Z-measured node with the single-qubit Z stabilizer that only Z-initialized nodes possess (false positives), while Z support landing on Z-initialized neighbors was compared exactly instead of being absorbed by that stabilizer (false negatives). Verdicts now agree with Stim's determinism analysis of the compiled circuit.

## [0.5.1] - 2026-08-03

### Added

- **Detector Tags and Post-Selection Flags**: Stim `DETECTOR` instruction tags now survive the import/export round trip, so flag detectors marked `DETECTOR[type=flag]` can be post-selected with sinter on the compiled circuit. `qompile` accepts `parity_check_tags` (aligned with `parity_check_group`) and stores them on the new `PauliFrame.parity_check_tags`; the Stim importer fills them from the source circuit's `DETECTOR` tags and `stim_compile` re-emits each group's tag on its `DETECTOR` instruction, escaped with Stim's tag escape language. `StimMppExtraction` gains a matching `detector_tags` field aligned with `detector_rows`. The new `graphqomb.stim_glue.postselect` module exposes `FLAG_DETECTOR_TAG`, `flag_detector_indices()`, and `flag_postselection_mask()`, the latter returning a bit-packed mask suitable for `sinter.Task(postselection_mask=...)`. The `.ptn` format serializes tagged groups as `.detector[tag] ...` lines and introduces format version 3 for them: a file is written with the smallest version whose grammar it uses, so tag-free files keep the version 2 header and stay readable by external version 2 parsers, while tagged files announce `.version 3` and older parsers reject them with a clear version error.
- **ty Type Checking**: Added [ty](https://docs.astral.sh/ty/) as a third type checker alongside mypy and pyright, configured under `[tool.ty]` with every disabled-by-default rule promoted to `error`, and wired into the `Type Checking` workflow.
- **Explicit Override Markers**: Every method overriding a base-class method now carries `@typing_extensions.override`.
- **Partial Coordinate Coverage Warning**: `stim_circuit_to_pattern()` emits a `UserWarning` when only some imported wires carry `QUBIT_COORDS`, since a mixture usually means coordinate metadata was lost upstream.

### Changed

- **MPP Rewriter Fallback Returns the Source Unsplit**: The gate-level fallback of `rewrite_to_mpp` returns the flattened source circuit unchanged instead of pre-splitting reset lifetimes onto fresh qubit ids, which dropped their `QUBIT_COORDS`; the importer handles reset reuse natively and inherits coordinates. `CheckMapping` now always reports original-circuit qubit ids.

### Security

- **Bounded `.ptn` Timeslice Expansion**: `graphqomb.ptn_format.loads()` materialized one `TICK` command for every timeslice a `[n]` marker skipped, so a handful of bytes of untrusted input (for example `[100000000000]`) could exhaust memory before the parse finished. The number of `TICK` commands a parse may materialize from timeslice markers is now bounded by the input size (one per line, plus a fixed slack of 4096), and an input that exceeds it raises `ValueError` instead of allocating. Files written by `dumps()` never skip timeslices and are unaffected.
- **Pillow 12.3.0 in the Lock File**: Bumped the locked `pillow` (a transitive dependency of `matplotlib`) from 12.2.0 to 12.3.0, clearing the 12.2.0 image-decoding advisories reported against the lock file. No direct dependency constraints changed.
- **Release Workflow Shell Hardening**: The `Publish to PyPI` workflow interpolated the pushed tag name straight into a `run:` script and closed its `$GITHUB_OUTPUT` heredoc with a fixed `EOF` delimiter. The tag now reaches the script through the environment, and the delimiter is randomly generated, so neither a crafted tag name nor a changelog line can inject shell commands or forge step outputs in the job that holds the trusted-publishing token.

### Fixed

- **Signed MPP Rows Keep Their Measurement Axis**: A negative `MPP` product sign now flips the sign of the ancilla's assigned measurement basis instead of forcing a negative X measurement, which clobbered the Y basis of Type I odd-Y-support rows and made detectors non-deterministic (for example `MPP !Y0`).
- **Type I Twist Edges for Shared Y-Y Support**: Equal-Y overlaps between stabilizers now count toward both twist direction parities, fixing missing or spurious ancilla CZ edges that compiled to non-deterministic detectors (for example two rounds of `MPP Y0*Y1 Y1*Y2`). Type II foliation is unaffected.
- **Observable Index Parsing on Python 3.10 and 3.11**: `observable_index` used `int.is_integer()`, which requires Python 3.12, so an integer `OBSERVABLE_INCLUDE` argument raised `AttributeError`.

## [0.5.0] - 2026-07-27

### Changed (Breaking)

- **Stim modules consolidated under `graphqomb.stim_glue`**: All Stim interoperability now lives in one subpackage, and the `graphqomb.qec` package is dissolved. `graphqomb.stim_compiler` moved to `graphqomb.stim_glue.compiler`, `graphqomb.stim_importer` to `graphqomb.stim_glue.importer`, `graphqomb.stim_parser` to `graphqomb.stim_glue.transpiler`, `graphqomb.stim_mpp_rewriter` to `graphqomb.stim_glue.mpp_rewriter`, and `graphqomb.qec.stim_mpp` to `graphqomb.stim_glue.mpp`; the shared parsing helpers formerly in `graphqomb.qec._stim` are now `graphqomb.stim_glue._parse`. The Stim-independent stabilizer-code builder `graphqomb.qec.qeccode` moved to the top-level `graphqomb.qeccode` module. `graphqomb.stim_glue` re-exports the public API (`stim_compile`, `stim_circuit_to_pattern`, `transpile`, `rewrite_to_mpp`, `stabilizer_code_from_stim_text`, and friends), so `from graphqomb.stim_glue import stim_compile` is the recommended import; importing `graphqomb.stim_glue` requires the optional `stim` dependency.

### Added

- **Stim Mid-Circuit Reset Import**: `R`, `RX`, and `RY` on a previously used qubit are now imported instead of rejected. Import-time preprocessing splits each reset-started lifetime onto its own Stim qubit id, so the importer starts a fresh wire on a new internal qubit index at the qubit's XY coordinates on the next z layer, initialized in the positive eigenstate of the reset axis with no outcome conditioning; mixed TICK blocks, MPP blocks, detectors, observables, and classical feedback work unchanged. Repeated resets before the next quantum use keep one wire, with the last reset selecting the axis. A wire that was not consumed by a direct measurement before the reset remains a pattern output, which is channel-equivalent to Stim's trace-out reset on every recorded measurement as long as that output is traced out rather than read. `StimImportResult.qubit_to_stim` maps the fresh indices back to the original Stim qubit id, and the new `stim_to_final_qubit` names the index of each Stim qubit's last lifetime so callers can tell the live wire from its discarded history. Because each reset lifetime also gets its own Stim qubit id, the `StimMppExtraction` metadata of an MPP block (`supports`, `stim_to_column`, `column_to_stim`) reports the lifetime that each product measured; the new `StimImportResult.wire_to_stim` maps those ids back to the original circuit ids and is the identity for qubits that are never reset mid-circuit.
- **Experimental Stim MPP Rewriter**: Added `graphqomb.stim_glue.mpp_rewriter.rewrite_to_mpp` to rewrite noiseless gate-level syndrome-extraction circuits into abstract `MPP` measurements by conjugating each ancilla measurement Pauli backwards through its segment's Clifford body. Every source measurement keeps its record position, so `DETECTOR` and `OBSERVABLE_INCLUDE` annotations are copied verbatim, and a `CheckMapping` sidecar links each record to its inferred Pauli product. Rewritten segments are always cross-checked against their source by stabilizer-flow generators; a segment that cannot be represented or verified falls back to the gate-level circuit with a fresh Stim qubit id per reset lifetime. Because that cross-check is what selects between the optimized rewrite, the unsubstituted retry, and the fallback, it cannot be disabled. Noise, classical feedback, and reuse of unreset measurement post-states raise `UnsupportedSyndromeCircuitError`.
- **Stim Classical Feedback Import**: Import measurement-record-controlled `CX`, `CY`, `CZ`, `XCZ`, and `YCZ` targets into the pattern's X/Z correction flows, preserving instruction position and applying the feedback only after graph-derived odd-neighbor Z-flow construction. Because frame corrections are applied at the target's measurement, an X or Y feedback entry also XORs Z corrections onto the neighbors entangled with the target after the feedback position; neighbors entangled before the feedback receive no compensation, and Z feedback needs none.
- **Stim Import Mixed TICK Blocks**: `stim_circuit_to_pattern()` now accepts unitary gates and Pauli measurements inside the same TICK block. Single-qubit Cliffords ahead of a measurement fold into the measurement basis when the measured qubit's lifetime ends there; any remaining mixed block (including MPP, CZ, and record-controlled feedback combinations) is split into internal unitary, measurement, and feedback layers that preserve per-qubit instruction order and measurement-record order, each advancing the temporal Z coordinate by one layer.
- **Stim Import Qubit Reuse After Measurement**: A Stim qubit may now be used again after a direct single-qubit measurement (including repeated measurement in one TICK block, such as `M 0 0`). The importer issues a fresh internal qubit index for the post-measurement wire, places its continuation node at the qubit's XY coordinates on the next Z layer, initializes it in the positive eigenstate of the measurement axis, and conditions it on the measurement outcome through the X/Z correction flows; the measured node becomes an internal measured node so its outcome propagates through the Pauli frame. `StimImportResult.qubit_to_stim` maps every internal qubit index, including continuation indices, back to its Stim qubit id. Reuse after an inverted measurement target (for example `M !0`) is rejected because negative-eigenstate initialization cannot be represented.
- **Stim Parser Preserved Measurement Qubits**: `transpile()` and `optimize_j_cz()` accept `preserved_measurement_qubits`, a set of qubits whose post-measurement state must survive optimization; measurements on these qubits are never folded into a rewritten Pauli basis.
- **Stim Import Fused Feedback De-fusing**: A controlled-Pauli instruction that stim fused across record-controlled and plain target groups (for example `CZ rec[-1] 0` adjacent to `CZ 0 1`) is split per target group, so the feedback groups import as correction-flow entries while the plain groups remain unitary gates.
- **Stim Measure-Reset Import**: `MR`, `MRX`, and `MRY` (and the `MRZ` alias) are imported as a direct measurement whose record participates in detectors, observables, and feedback like `M`/`MX`/`MY`. When the qubit is used again, the continuation wire starts on a fresh internal qubit index initialized in the positive eigenstate of the reset axis with no outcome conditioning, because the reset re-prepares that state unconditionally. Inverted measure-reset targets (for example `MR !0`) may be reused since the reset discards the outcome.

### Changed

- **Signed Stim MPP Import**: `stim_circuit_to_pattern()` now preserves inverted-result signs on `MPP` products by assigning the corresponding negative X measurement to the generated MPP ancilla. The unsigned `StabilizerCode` extraction helpers continue to reject signed products because stabilizer signs are not retained there.
- **First-Class Measured Outputs**: Output nodes with an assigned measurement basis are now scheduled and executed like ordinary measurements instead of being appended after the timeline. `dag_from_flow` keeps correction-flow dependencies whose source is a measured output, the `Scheduler`, CP-SAT solver, and both greedy schedulers include measured outputs in `measure_time`, `qompile` emits their `M` commands inside the scheduled slices, and `PatternSimulator` applies their feedforward corrections via `meas_flip`. This makes classical feedback sourced from direct Stim measurements causal and correctly simulated. Manual schedules must now provide measurement times for measured outputs. Added `unmeasured_output_nodes()` to `graphqomb.graphstate`, and `compose()` now rejects connecting through a measured output of the first graph because the wire is consumed by the readout.

## [0.4.2] - 2026-07-23

### Added

- **Detector Determinism**: Added `PauliFrame.detector_stabilizers()` and `PauliFrame.detector_determinism()` to construct closure-expanded detector stabilizers and report whether they exactly match the unsigned product of assigned Pauli measurement axes on the detector support. The construction respects Pauli input initialization, replaces Z-measured graph stabilizers with single-qubit Z, removes their incident graph edges, and omits unmeasured outputs from comparison.
- **Pauli Frame Logical Observables**: Added `PauliFrame.logical_observable_groups()` to return all indexed logical observables after dependent-chain expansion, matching the all-groups behavior of `detector_groups()`; Stim export now uses the new aggregate API.

### Fixed

- **QEC Graph-State Coordinates**: Move automatically positioned MPP ancillas aside when their support centroid's XY projection is already occupied by another node, using clearance scaled to the graph's coordinate span while preserving the inferred temporal coordinate and explicit ancilla coordinates.

## [0.4.1] - 2026-07-20

### Added

- **Stim Clifford Parser**: Added `graphqomb.stim_parser` for normalizing Stim Clifford circuits to the Clifford `J`/`CZ` basis: `H = J(0)`, `HS = J(pi/2)` (Stim `C_XNYZ`), `HZ = J(pi)` (Stim `SQRT_Y`), and `HS_DAG = J(-pi/2)` (Stim `C_XYZ`), corresponding to X+/Y+/X-/Y- measurements. The parser covers fixed one- and two-qubit Clifford gates, `SPP`/`SPP_DAG`, nested repeats, `R`/`RX`/`RY` and `M`/`MX`/`MY` boundaries, preserved record/annotation instructions, and detailed rejection of unsupported instructions. Optimization canonicalizes every single-qubit gate run to its shortest word over the four `J` gates (at most three per single-qubit Clifford), cancels redundant `CZ` pairs, and folds gates at those Pauli reset and measurement boundaries.

### Changed

- **Stim Circuit Import**: Normalize reset/unitary TICK blocks through the new Stim parser before circuit analysis, folding leading Clifford gates into Pauli input initialization, expanding accepted Clifford gates, and removing identity blocks such as repeated CZ pairs without advancing their temporal coordinate.

## [0.4.0] - 2026-07-19

### Added

- **QEC Stim MPP Import**: Added utilities for building `StabilizerCode` inputs from unsigned Stim `MPP` layers, including sparse Stim qubit id mapping, coordinate import, multi-layer selection, detector/logical-observable import, and the optional `graphqomb[stim]` extra. Signed products using inverted Pauli targets are rejected because stabilizer signs are not retained.
- **Stim Circuit Import**: Added `stim_file_to_pattern()`, `stim_text_to_pattern()`, and `stim_circuit_to_pattern()` for converting supported Stim circuits into GraphQOMB patterns. The importer handles initial Pauli resets (`R`/`RZ`, `RX`, and `RY`), Clifford unitary blocks, and `TICK`-separated Pauli measurements (`M`/`MZ`, `MX`, `MY`, `MXX`, `MYY`, `MZZ`, and `MPP`), assigns single-qubit measurement bases directly to their graph nodes, terminates each qubit lifetime at its direct measurement while allowing disjoint qubits to continue, validates that same-block MPP products commute, supports Type I and Type II Y foliation, keeps MPP ancilla nodes out of the X-correction flow, derives the complete Z-correction flow from the composed data flow without Pauli simplification, places gate and MPP blocks on explicit temporal Z layers while preserving the first two Stim coordinate components, aligns parallel gate outputs across different transpiled depths, relocates idle I/O nodes without adding graph nodes or edges, and resolves detector/logical-observable records once across the full flattened circuit. Circuit-level noise and measurement-error probabilities are intentionally omitted because GraphQOMB uses an MBQC-specific noise model; mid-circuit reset, measurement-reset, and feedback instructions remain unsupported.
- **Input Initialization Bases**: Added per-input positive Pauli eigenstate initialization via `GraphState.register_input(..., init_axis=...)`, with `X+`, `Y+`, and `Z+` propagated through `qompile()`, `Pattern`, `PatternSimulator`, and Stim export.

### Changed

- **Development Tooling**: Use uv as the default dependency manager for local development, CI, documentation builds, and publishing workflows.
- **Graph State API**: Replaced legacy `physical_*` graph methods/properties with standard graph-style `nodes`, `edges`, `add_node()`, `add_edge()`, `remove_node()`, `remove_edge()`, and count/query helpers.
- **Pattern Simulator**: Materialize pending output Pauli-frame corrections when returning output statevectors or explicit output measurement results.
- **Pattern Simulator**: Sample all measurements from exact Born probabilities by default, with `calc_prob=False` retaining the legacy 50/50 assumption for non-output measurements.
- **PTN Format**: Bumped exported `.ptn` files to format version 2 to record non-default input initialization bases with `.input_basis`; version 1 files remain readable and default inputs to `X+`.
- **Stim Compiler**: Input reset instructions now follow the stored initialization basis (`RX`, `RY`, or `R`) while non-input preparations remain `RX`.
- **Stim Coordinate Import**: Reject `QUBIT_COORDS` whose first two components collide across qubits, because graph nodes are placed by the XY projection and colliding projections produce coincident nodes.

### Fixed

- **Pattern Simulator**: Raise `TypeError` for unsupported commands instead of recursively redispatching them.
- **QEC Graph-State Builder**: Add the required ancilla CZ edge when shared data qubits contain an odd number of oppositely ordered stabilizer-interaction pairs, using the same `Z`-before-`Y`-before-`X` rule for Type I and Type II foliation.
- **Stim Circuit Import**: Preserve the existing data-lane endpoint and its coordinate when importing a terminal single-qubit measurement.
- **Stim Circuit Import Coordinates**: Generate consistent 3D spacetime coordinates across gate, idle, and MPP fragments instead of mixing raw 2D gate coordinates with MPP Z layers.
- **Stim Compiler**: Preserve minus-signed axis measurement bases using inverted Stim measurement targets.
- **State Vector Array Conversion**: Convert to a real NumPy dtype without warnings when every amplitude is real-valued, and reject the conversion when it would discard nonzero imaginary amplitudes.

### Removed

- **Pattern Commands**: Removed explicit `graphqomb.command.X` and `graphqomb.command.Z` correction commands. Output corrections are now represented by `PauliFrame` only, and `.ptn` files containing `X`/`Z` command lines are rejected.

## [0.3.1] - 2026-05-17

### Added

- **PTN Format**: Human-readable text format (`.ptn`) for pattern serialization
  - `ptn_format.dumps()` / `ptn_format.dump()`: Serialize patterns to text
  - `ptn_format.loads()` / `ptn_format.load()`: Deserialize patterns from text
  - Format separates quantum instructions and classical feedforward processing
  - Timeslice markers `[n]` indicate parallel execution groups
  - Pauli measurements use compact notation (`X +`, `Y -`, `Z +`)
  - Non-Pauli measurements use plane+angle format (`XY pi/4`)
  - Support for node coordinates, logical observables, and inline comments
- **Non-Unitary Parity Projection Example**: Added `examples/nonunitary_parity_projection.py` demonstrating measurement-induced entanglement via a 3-node star graph parity projector

### Fixed

- **Qompiler**: `qompile()` now validates a provided scheduler before pattern generation, so invalid manual schedules fail early with `ValueError`.

## [0.3.0] - 2026-04-08

### Added

- **Noise Model Module**: Added `graphqomb.noise_model` for event-driven noise injection during Stim compilation
  - Added `NoiseModel` hooks for `on_prepare`, `on_entangle`, `on_measure`, and `on_idle`
  - Added frozen, validated `NoiseOp` dataclasses: `PauliChannel1`, `PauliChannel2`, `HeraldedPauliChannel1`, `HeraldedErase`, `RawStimOp`, `MeasurementFlip`
  - Added event dataclasses `PrepareEvent`, `EntangleEvent`, `MeasureEvent`, `IdleEvent`, plus `NodeInfo` and `Coordinate`
  - Added `NoisePlacement`, `noise_op_to_stim()`, `depolarize1_probs()`, and `depolarize2_probs()`

- **Built-in Noise Models**: Ready-to-use noise model implementations
  - Added `DepolarizingNoiseModel` for single and two-qubit depolarizing noise
  - Added `MeasurementFlipNoiseModel` for measurement bit-flip errors using Stim's built-in `MX(p)` syntax

- **Stim Compiler Noise Integration**: Added noise-model-driven Stim compilation
  - Support for multiple noise models via `Sequence[NoiseModel]`
  - Added `tick_duration` parameter for idle noise calculations
  - Automatic measurement record tracking for heralded noise operations when emitting detectors and observables

- **Greedy Scheduler**: Fast greedy scheduling algorithms as an alternative to CP-SAT optimization
  - Added `greedy_minimize_time()` for minimal execution time scheduling with ALAP preparation optimization
  - Added `greedy_minimize_space()` for minimal qubit usage scheduling

- **Schedule Solver**: Added constraint that every non-input, non-output node must be prepared strictly before it is measured (`node2prep[node] < node2meas[node]`)

- **Circuit Conversion**: Added circuit-derived pre-scheduling support in `circuit2graph()`.
  - Added `CircuitScheduleStrategy` with `PARALLEL` and `MINIMIZE_SPACE`.
  - Added `schedule_strategy` argument to `circuit2graph()`.
  - `circuit2graph()` now returns `(graph, gflow, scheduler)` and pre-populates `Scheduler` via manual scheduling.
- **PyZX Integration**: Added optional `graphqomb.zx_util` utilities for importing strict graph-like PyZX diagrams into `GraphState`.
  - Added `from_pyzx()` to convert PyZX diagrams into a `GraphState`.
  - Added boundary rewriting and metadata import helpers to preserve graph structure, measurement bases, and coordinates during conversion.
  - Added optional phase-gadget recognition for supported lone-`Z` gadget patterns via `recognize_pg=True`, importing the adjacent node as a `YZ`-plane measurement.

- **Documentation**: Added comprehensive Sphinx documentation for the noise model module

### Changed

- **Stim Compiler API**: `stim_compile()` now has signature `stim_compile(pattern, *, emit_qubit_coords=True, noise_models=None, tick_duration=1.0)`
- **Stim Compiler**: Refactored internal structure to support event-driven noise model integration
- **Measurement Flip Semantics**: `MeasurementFlipNoiseModel` and custom `MeasurementFlip` ops now compile to Stim's native `MX(p)` / `MY(p)` / `MZ(p)` instructions instead of emitting separate Pauli error instructions
- **Noise Extension API**: `NoiseOp` values are now represented as plain frozen dataclasses collected under the `NoiseOp` union, improving type safety for custom `NoiseModel` implementations
- **Noise Validation**: Centralized noise parameter validation in `noise_model`
  - `NoiseOp` dataclasses now validate and normalize their inputs at construction time
  - `DepolarizingNoiseModel` and `MeasurementFlipNoiseModel` now reject invalid probabilities when instantiated
  - `MeasurementFlip` is now enforced as a measurement-only noise operation during Stim compilation
- **Graph State**: Made `meas_bases` read-only by returning `MappingProxyType` to avoid external mutation.
- **Graph State**: Added caching for `physical_nodes` snapshots and proper cache invalidation on node add/remove.
- **Docs/Examples**: Updated circuit conversion usage in README and `examples/pattern_from_circuit.py` for the new `circuit2graph()` return signature.
- **Packaging/Docs**: Added the optional `graphqomb[pyzx]` extra, documented PyZX installation in the README, and published Sphinx API reference pages for `graphqomb.zx_util`.
- **CI**: Split PyZX-marked tests into a dedicated GitHub Actions job and installed the optional dependency in coverage runs.

### Fixed

- **Stim Compiler**: Detector and observable record indices now stay aligned when noise models emit heralded instructions that add measurement records
- **Pattern Simulator**: Fixed adaptive measurement conjugation so non-Pauli measurements apply the missing angle sign flip from the Pauli frame during simulation ([#139](https://github.com/UTokyo-FT-MBQC/graphqomb/issues/139))
- **Feedforward**: Fixed operator precedence bug in `dag_from_flow` where self-loops were only removed from `zflow` but not from `xflow`. The expression `xflow | zflow - {node}` was evaluated as `xflow | (zflow - {node})` due to `-` binding tighter than `|`. Corrected to `(xflow | zflow) - {node}`.

### Tests

- **Noise Model / Stim Compiler**: Added comprehensive tests for `graphqomb.noise_model` and noise-aware `stim_compile()`, including heralded record tracking, `MeasurementFlip` validation, and removed legacy kwargs
- **Greedy Scheduler**: Added tests for greedy scheduling algorithms
- **Schedule Solver**: Added integration test verifying that CP-SAT MINIMIZE_SPACE strategy enforces node preparation before measurement
- **Circuit Conversion**: Expanded scheduling tests in `tests/test_circuit.py`, including scheduler return contract, J/CZ/phase-gadget timing behavior, schedule validation, and `MINIMIZE_SPACE` behavior.
- **Integration**: Added circuit-level integration tests for `signal_shifting()` and `pauli_simplification()` with circuit-vs-pattern statevector equivalence checks.
- **Pattern Simulator / Measurement Bases**: Added regression tests for measurement-basis conjugation semantics and adaptive simulation of non-Pauli measurements affected by the Pauli frame.
- **Stim Compiler / Pauli Frame**: Updated tests to explicitly pass parity-check groups where logical-observable and cache initialization paths are exercised.
- **PyZX Integration**: Added unit tests for vertex/edge collection, boundary rewrites, lone-spider phase-gadget recognition, and end-to-end `from_pyzx()` conversion behavior.

### Removed

- **Stim Compiler Legacy Noise Args**: Removed `p_depol_after_clifford` and `p_before_meas_flip` from `stim_compile()`
  - Use `noise_models=[DepolarizingNoiseModel(...), MeasurementFlipNoiseModel(...)]` instead

## [0.2.1] - 2026-01-16

### Added

- **Type Hints**: Added `py.typed` marker for PEP 561 compliance, enabling type checkers (mypy, pyright) to recognize the package as typed when installed from PyPI.

### Changed

- **Python Support**: Dropped Python 3.9 support, added Python 3.14 support. Now requires Python >=3.10, <3.15.

## [0.2.0] - 2025-12-26

### Added

- **TICK Command**: Time slice boundary marker for temporal scheduling in MBQC patterns
  - Added TICK command type to mark boundaries between time slices
  - Integrated TICK command handling in PatternSimulator
  - Integrated TICK command processing in Stim compiler

- **Edge Scheduler**: Automatic entanglement operation scheduling based on node preparation times ([#99](https://github.com/UTokyo-FT-MBQC/graphqomb/issues/99))
  - Added `entangle_time` attribute to Scheduler for tracking entanglement operation timing
  - Added `auto_schedule_entanglement()` method to automatically schedule CZ gates when both nodes are prepared
  - Extended the `timeline` property to include entanglement operations
  - Added entanglement time validation in schedule validation
  - Added `compress_schedule()` function to support entanglement time compression

- **Pattern**: Added the `depth` attribute into `Pattern`, which represents the depth of parallel execution.

- **Pattern**: Added pattern resource/throughput metrics (`active_volume`, `volume`, `idle_times`, `throughput`).

- **Scheduler Integration**: Enhanced qompile() to support temporal scheduling with TICK commands
  - Added `scheduler` parameter to qompile() for custom scheduling
  - Automatically inserts TICK commands between time slices

- **Examples**: Added entanglement_scheduling_demo.py demonstrating edge scheduler features

- **Feedforward Optimization**: Added a `signal_shifting` method as a feedforward optimization.
  - This optimization is equivalent to the operation of the same name in the measurement calculus, and makes the measurement pattern as parallel as possible.
  - The optimization is now self-contained within the feedforward module.

- **Feedforward Optimization**: Added `pauli_simplification()` to remove redundant Pauli corrections in correction maps when measuring in Pauli bases.

### Changed

- **Pattern**: Updated command sequence generation to support TICK commands
- **Command**: Extended Command type alias to include TICK
- The default strategy of `Scheduler.solve_schedule` is now `MINIMIZE_TIME` instead of `MINIMIZE_SPACE` for the compilation performance.

### Fixed

- **Scheduler**: Accept `entangle_time` edges in either order in `Scheduler.manual_schedule()`.

### Tests

- **Stim Compiler**: Add coverage that manual `entangle_time` determines CZ time slices in both Pattern and Stim output.

## [0.1.2] - 2025-10-31

### Added

- **Graph State**: Bulk initialization methods for GraphState ([#120](https://github.com/UTokyo-FT-MBQC/graphqomb/issues/120))
  - Added `from_graph()` class method for direct graph-based initialization
  - Added `from_base_graph_state()` class method for initialization from base GraphState objects
  - Improved initialization flexibility for diverse use cases

### Performance

- **Pauli Frame**: Optimized `_collect_dependent_chain` method with memoization and caching
  - Added Pauli axis cache to avoid redundant basis computations
  - Implemented chain memoization cache to prevent recalculating dependent chains
  - Optimized set operations for better performance in large graph states

### Tests

- **TICK Command**: Added comprehensive test suite for TICK command functionality
  - Added `test_simulator_with_tick_commands()` for TICK command handling in PatternSimulator
  - Added `test_stim_compile_with_tick_commands()` for TICK command compilation to Stim format
  - Extended scheduler integration tests with comprehensive edge scheduling validation

- **Pauli Frame**: Added comprehensive test suite for PauliFrame module
  - Added tests for basic methods (x_flip, z_flip, meas_flip, children, parents)
  - Added tests for Pauli axis cache initialization and chain cache memoization
  - Added tests for dependent chain collection across X, Y, Z measurement axes
  - Added tests for detector groups and logical observables
  - Improved test coverage from 77.78% to 97% for pauli_frame.py
- **Graph State**: Added comprehensive test suite for bulk initialization methods
  - Added tests for `from_graph()` initialization
  - Added tests for `from_base_graph_state()` initialization
  - Added tests for graph consistency and state equivalence

## [0.1.1] - 2025-10-23

### Added

- **Stim Compiler**: Pattern to Stim circuit compiler with detector and observable support for fault-tolerant quantum computing ([#67](https://github.com/UTokyo-FT-MBQC/graphqomb/issues/67))
  - Compile MBQC patterns into Stim format for error correction analysis
  - Support for detectors, observables, and error models
  - Configurable depolarization noise after Clifford gates and measurements

### Changed

- **Pauli Frame**: Extended with detector and syndrome analysis capabilities
  - Added `detector_groups` for detector grouping
  - Added `syndrome_parity_group` for syndrome extraction
  - Added parity check grouping for X and Z corrections

### Fixed

- Fixed inverse flow construction to avoid self-loops
- Fixed type hints in `graphstate.compose` for better type safety

## [0.1.0] - 2025-10-22

### Added

- **Core Infrastructure**: Initial repository setup with project structure, build system, and CI/CD workflows ([#12](https://github.com/UTokyo-FT-MBQC/graphqomb/pull/12))
- **Mathematical Foundations**: Euler angle computations for quantum operations ([#24](https://github.com/UTokyo-FT-MBQC/graphqomb/pull/24))
- **Graph State**: Graph state representation and manipulation ([#34](https://github.com/UTokyo-FT-MBQC/graphqomb/pull/34))
- **Pattern Module**: MBQC pattern data structures and operations ([#47](https://github.com/UTokyo-FT-MBQC/graphqomb/pull/47))
- **Feedforward System**: Feedforward strategy design and implementation for adaptive measurements ([#40](https://github.com/UTokyo-FT-MBQC/graphqomb/pull/40))
- **Command Module**: Measurement command definitions ([#43](https://github.com/UTokyo-FT-MBQC/graphqomb/pull/43))
- **Qompiler**: Pattern compiler with Pauli frame implementation ([#55](https://github.com/UTokyo-FT-MBQC/graphqomb/pull/55))
- **Scheduler**: Prepare time and measurement time scheduling for efficient execution ([#74](https://github.com/UTokyo-FT-MBQC/graphqomb/pull/74))
- **Circuit Framework**: Quantum gate definitions and circuit representation ([#73](https://github.com/UTokyo-FT-MBQC/graphqomb/pull/73))
- **Simulation Backend**: Statevector simulator backend for quantum state evolution ([#62](https://github.com/UTokyo-FT-MBQC/graphqomb/pull/62))
- **Pattern and Circuit Simulation**: Complete simulation support for both patterns and circuits ([#78](https://github.com/UTokyo-FT-MBQC/graphqomb/pull/78))
- **Visualization**: Basic visualizer for graph states and patterns ([#83](https://github.com/UTokyo-FT-MBQC/graphqomb/pull/83))
- **Documentation**: Comprehensive documentation on Read the Docs (https://graphqomb.readthedocs.io/)
