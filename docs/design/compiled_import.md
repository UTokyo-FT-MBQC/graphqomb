# Design proposal: importing compiled feedforward, detectors, and logical observables

Status: draft for discussion
Target branch: `claude/graphqomb-feedforward-interface-3u88co`

## 1. Motivation

GraphQOMB's pipeline today is compile-centric: the supported entry points are a
circuit (`circuit2graph` → `qompile`), a full Stim circuit
(`stim_glue.importer`), or a fully scheduled `.ptn` pattern. However, several
workflows start from *already compiled* artifacts rather than a circuit:

- an external FT compiler (e.g. `ls-pattern-compile`) that emits a resource
  graph, feedforward maps, detectors, and logical observables directly;
- hand-constructed or generated lattice layouts whose detectors/observables
  come from code structure, not from a circuit;
- re-analysis of previously exported data (Stim `DETECTOR`/`OBSERVABLE_INCLUDE`
  lines, DEM-level tooling) against a known graph and feedforward.

For these, the user wants GraphQOMB's *analysis* machinery — resource
estimation (`Pattern.max_space`, `depth`, `volume`, `active_volume`,
`idle_times`, `throughput`), determinism checking
(`PauliFrame.detector_determinism`, `detector_stabilizers`), Stim export,
simulation, and visualization — without owning a circuit.

This document proposes an import interface that takes compiled objects
(feedforward maps, detectors, logical observables, plus the resource graph)
and converts them into the useful in-memory objects: `GraphState`,
`PauliFrame`, `Scheduler`, and `Pattern`.

## 2. Current state

The relevant machinery already exists but is only reachable through
compile-from-circuit paths:

| Capability | Where it lives | How it is reached today |
| --- | --- | --- |
| Lowering graph + flows (+ detector seeds, observables) to a pattern | `qompiler.qompile` | public, but detector/observable input only in *seed* form |
| Detector/observable expansion, stabilizers, determinism | `pauli_frame.PauliFrame` | constructed inside `qompile`; public class but no convenience entry |
| Resource metrics | `pattern.Pattern` properties | requires full lowering (a schedule) |
| Graph construction from external ids | `GraphState.from_graph` | public; returns `(graph, node_map)` |
| Scheduled-pattern serialization | `ptn_format` | requires a full command stream |
| Stim circuit import | `stim_glue.importer` | requires a complete Stim circuit |

Two facts shape the design:

1. **`PauliFrame` is already the canonical container** for compiled classical
   structure: graph + `xflow`/`zflow` + `parity_check_group` (+ tags) +
   `logical_observables`. Determinism checking needs *only* a `PauliFrame` —
   no schedule, no commands.
2. **Detector/observable groups have two representations**, and only one
   direction of conversion exists (see §3).

## 3. Two group representations: corrected vs. raw

A detector (or logical observable) is a parity of measurement outcomes, but
"outcome" is ambiguous in MBQC because Pauli corrections are tracked virtually
in the frame rather than applied physically:

- **Corrected-outcome form** (today's "seed" form): the parity is over
  *frame-corrected* outcomes. This is what `qompile` /
  `PauliFrame.parity_check_group` accept, what `.ptn` `.detector` lines store,
  and what circuit-level tools naturally produce (a circuit either applies its
  corrections physically or has none).
- **Raw-outcome form** (today's "expanded" form): the parity is over
  *physical, uncorrected* outcomes. This is what
  `PauliFrame.detector_groups()` computes and what `stim_compile` emits as
  `DETECTOR` targets. MBQC-native tooling and DEM-level analysis of emitted
  Stim circuits naturally live here.

The two are related by the outcome dependency chains
(`PauliFrame._collect_dependent_chain`): with `c(n)` the chain of node `n`,

```
raw(S) = XOR_{n in S} c(n)          # implemented today (seed -> expanded)
```

`c(n) = {n} XOR c(p1) XOR c(p2) ...` where the parents `p_i` are the
axis-appropriate inverse-flow sets, all measured strictly before `n` (causality
is guaranteed by `check_flow`). Over GF(2) this map is linear and
unitriangular with respect to any topological order of the correction DAG, so
it is **invertible**, and the inverse is a cheap back-substitution:

```
corrected_from_raw(E):
    R = set(E); S = {}
    for n in reverse_topological_order(dag_from_flow(graph, xflow, zflow)):
        if n in R:
            S.add(n)
            R ^= c(n)          # removes n; only toggles strictly earlier nodes
    if R: raise ValueError     # group touches unmeasured/unknown nodes
    return S
```

Cost is `O(sum |c(n)|)` with the same memoized chains PauliFrame already
computes. This inverse is the one genuinely missing algorithm; everything else
in this proposal is packaging.

An import interface must let the caller state which form their data is in and
canonicalize at ingestion. Canonical internal form stays **corrected** (what
`PauliFrame` stores), so all existing consumers (`stim_compile`, `.ptn` dump,
determinism) are unchanged.

## 4. Proposed API

### 4.1 Flow algebra additions — `graphqomb.feedforward`

Pure flow/outcome algebra belongs next to `signal_shifting` and
`pauli_simplification`:

```python
def outcome_chains(
    graph: BaseGraphState,
    xflow: Mapping[int, AbstractSet[int]],
    zflow: Mapping[int, AbstractSet[int]] | None = None,
    nodes: Iterable[int] | None = None,
) -> dict[int, frozenset[int]]:
    """Memoized dependent chains c(n) for Pauli-measured nodes."""

def raw_groups_from_corrected(
    graph, xflow, zflow, groups: Sequence[AbstractSet[int]],
) -> list[set[int]]:
    """Corrected (seed) parities -> raw-outcome parities. Same math as
    PauliFrame.detector_groups(), exposed as a standalone function."""

def corrected_groups_from_raw(
    graph, xflow, zflow, groups: Sequence[AbstractSet[int]],
) -> list[set[int]]:
    """Raw-outcome parities -> corrected (seed) parities (back-substitution
    of section 3). Raises ValueError if a group is not solvable (nodes
    without a Pauli measurement basis, unknown nodes)."""
```

`PauliFrame._collect_dependent_chain` is refactored to delegate to
`outcome_chains` (behavior-preserving; the memo cache moves with it).

### 4.2 New module — `graphqomb.compiled`

A thin adapter layer; `qompile` remains the only lowering path.

```python
class GroupForm(Enum):
    CORRECTED = auto()   # qompile-native seed form
    RAW = auto()         # stim DETECTOR-native form

@dataclass(frozen=True)
class Detector:
    nodes: frozenset[int]
    tag: str = ""        # Stim-style tag, e.g. "type=flag"
```

`Detector` also fixes an existing papercut: `qompile` currently takes parallel
sequences `parity_check_group` / `parity_check_tags` that are easy to
misalign.

**Analysis tier (no schedule needed):**

```python
def pauli_frame_from_flows(
    graph: BaseGraphState,
    xflow: Mapping[int, AbstractSet[int]],
    zflow: Mapping[int, AbstractSet[int]] | None = None,
    *,
    detectors: Sequence[Detector] = (),
    observables: Mapping[int, AbstractSet[int]] | None = None,
    group_form: GroupForm = GroupForm.CORRECTED,
) -> PauliFrame:
    """Validate (check_flow, canonical form, group membership), convert RAW
    groups to corrected form, and build the PauliFrame directly."""
```

This is the cheapest entry for the determinism use case: build the frame,
call `detector_determinism()` — no scheduling, no commands.

**Lowering tier:**

```python
def pattern_from_flows(
    graph, xflow, zflow=None, *,
    detectors=(), observables=None, group_form=GroupForm.CORRECTED,
    scheduler: Scheduler | None = None,
) -> Pattern:
    """pauli_frame_from_flows + qompile lowering. `scheduler` passthrough
    keeps imported/manual schedules (Scheduler.manual_schedule) usable."""

def reannotate(
    pattern: Pattern, *,
    detectors: Sequence[Detector] | None = None,
    observables: Mapping[int, AbstractSet[int]] | None = None,
    group_form: GroupForm = GroupForm.CORRECTED,
) -> Pattern:
    """Attach or replace detectors/observables on an existing compiled
    pattern (same graph, flows, and commands; new PauliFrame). Supports the
    'detectors arrive from a separate QEC tool after compilation' workflow."""
```

**Bundle tier (external node ids):**

External compilers use their own node identifiers; today remapping through
`GraphState.from_graph`'s `node_map` is manual and error-prone across flows,
detectors, and observables. A single bundle handles it once:

```python
@dataclass(frozen=True)
class CompiledLayout(Generic[NodeT]):
    nodes: Sequence[NodeT]
    edges: Sequence[tuple[NodeT, NodeT]]
    inputs: Sequence[NodeT] = ()
    outputs: Sequence[NodeT] = ()
    meas_bases: Mapping[NodeT, MeasBasis] = ...
    input_initialization_axes: Mapping[NodeT, Axis] = ...
    coordinates: Mapping[NodeT, tuple[float, ...]] = ...
    xflow: Mapping[NodeT, AbstractSet[NodeT]] = ...
    zflow: Mapping[NodeT, AbstractSet[NodeT]] | None = None
    detectors: Sequence[tuple[AbstractSet[NodeT], str]] = ()
    observables: Mapping[int, AbstractSet[NodeT]] = ...
    group_form: GroupForm = GroupForm.CORRECTED

@dataclass(frozen=True)
class CompiledImportResult:
    graph: GraphState
    node_map: dict[Any, int]          # external id -> internal node index
    pauli_frame: PauliFrame
    pattern: Pattern | None           # None when lower=False

def import_compiled(
    layout: CompiledLayout[NodeT], *,
    scheduler: Scheduler | None = None,
    lower: bool = True,
) -> CompiledImportResult
```

`import_compiled` builds the graph via `GraphState.from_graph`, remaps every
mapping through `node_map`, then calls `pauli_frame_from_flows` and (when
`lower=True`) `qompile`. `lower=False` serves determinism-only analysis.
Mirrors the existing `StimImportResult` precedent.

**Reporting conveniences:**

```python
@dataclass(frozen=True)
class DetectorVerdict:
    tag: str
    corrected_nodes: frozenset[int]
    raw_nodes: frozenset[int]
    stabilizer: Mapping[int, Axis]
    deterministic: bool

def determinism_report(frame: PauliFrame) -> list[DetectorVerdict]

class ResourceSummary(NamedTuple):
    max_space: int
    depth: int
    volume: int
    active_volume: int
    num_measurements: int
    throughput: float

def resource_summary(pattern: Pattern) -> ResourceSummary
```

These are thin views over existing `PauliFrame`/`Pattern` members, packaged
for the import-then-analyze workflow the interface exists for.

### 4.3 Validation rules

`pauli_frame_from_flows` (and therefore everything above it) validates:

1. `graph.check_canonical_form()` and `check_flow(graph, xflow, zflow)`
   (causality / DAG acyclicity) — same gates as `qompile`.
2. Every detector/observable node exists in the graph.
3. Every detector/observable node carries a **Pauli** measurement basis
   (`determine_pauli_axis` succeeds), or is an unmeasured output only where
   `detector_determinism` tolerates it. RAW-form input strictly requires
   Pauli bases along all involved chains (the inverse conversion needs them).
4. `Detector.tag` uses the Stim tag escape language (reuse `_tag.escape_tag`).

Errors should name the offending detector index/tag and node ids — imported
data is external and the error message is the debugging interface.

### 4.4 Worked examples

External FT compiler output (external ids, raw-form detectors):

```python
layout = CompiledLayout(
    nodes=nodes, edges=edges, outputs=outs,
    meas_bases=bases, xflow=xflow,
    detectors=[(d.nodes, d.tag) for d in external_detectors],
    observables={0: obs_nodes},
    group_form=GroupForm.RAW,
)
result = import_compiled(layout, lower=False)
report = determinism_report(result.pauli_frame)     # determinism check

result = import_compiled(layout)                    # default schedule
summary = resource_summary(result.pattern)          # resource estimation
stim_text = stim_compile(result.pattern)            # downstream export
```

Re-attaching detectors computed after compilation:

```python
pattern = qompile(graph, xflow)
pattern = reannotate(pattern, detectors=[Detector(frozenset(g)) for g in groups],
                     group_form=GroupForm.RAW)
```

### 4.5 Serialization (phase 2)

`.ptn` stays the *scheduled pattern* format (commands are its core). The
compiled bundle is a different pipeline stage — unscheduled graph + classical
structure — and deserves a separate, trivially-emittable JSON format:

```json
{
  "format": "graphqomb-compiled", "version": 1,
  "nodes": [0, 1, 2], "edges": [[0, 1], [1, 2]],
  "inputs": [0], "outputs": [2],
  "meas_bases": {"0": {"plane": "XY", "angle": "pi/2"}, "1": {"axis": "X"}},
  "input_axes": {"0": "X"},
  "coordinates": {"0": [0.0, 0.0]},
  "xflow": {"0": [1], "1": [2]},
  "zflow": null,
  "detectors": [{"nodes": [0, 1], "tag": "type=flag"}],
  "observables": {"0": [1, 2]},
  "group_form": "raw",
  "schedule": null
}
```

`load_layout`/`dump_layout` map this to/from `CompiledLayout`. The optional
`schedule` section (prepare/measure/entangle times) feeds
`Scheduler.manual_schedule`. Angle strings reuse the `.ptn` angle grammar.

## 5. Implementation plan

1. **Flow algebra** (`feedforward.py`): `outcome_chains`,
   `raw_groups_from_corrected`, `corrected_groups_from_raw`;
   `PauliFrame` delegates. Tests: inverse round-trip property on
   `random_objects` flows; equivalence of `raw_groups_from_corrected` with
   `PauliFrame.detector_groups()`.
2. **Analysis tier** (`compiled.py`): `GroupForm`, `Detector`,
   `pauli_frame_from_flows`, `determinism_report`. Tests: determinism on
   `qeccode.build_graph_state` codes reached without `qompile`; RAW-form
   ingestion round-trips against `stim_compile` output parsed back
   (`DETECTOR` targets → raw groups → corrected seeds == original seeds).
3. **Lowering + bundle tier**: `pattern_from_flows`, `reannotate`,
   `CompiledLayout`, `import_compiled`, `resource_summary`. Tests: external-id
   remapping, scheduler passthrough, `.ptn`/Stim round trips.
4. **Serialization** (phase 2): JSON layout format + docs page
   (`docs/source/compiled.rst`), example in the gallery.

Each phase is independently shippable; phase 1 + 2 already cover the stated
use cases (determinism checking, then resource estimation via phase 3's thin
wrappers or plain `qompile`).

## 6. Open questions

1. **Naming.** Module `graphqomb.compiled` vs `graphqomb.imports`;
   `GroupForm.CORRECTED/RAW` vs `SEED/EXPANDED`; `CompiledLayout` vs
   `CompiledArtifact`.
2. **Should `qompile` itself accept `Sequence[Detector]`** (deprecating the
   parallel `parity_check_group`/`parity_check_tags` lists), making
   `Detector` the single vocabulary type across `qompile`, `compiled`, and
   `stim_glue`?
3. **Where does the inverse conversion live** — module functions in
   `feedforward` (proposed) or methods on `PauliFrame`?
4. **Serialization ownership**: should the JSON layout format live here or be
   co-designed with `graphqomb-studio` / `ls-pattern-compile` so producers and
   consumers share a schema from day one?
5. **Non-Pauli graphs**: detectors require Pauli bases, but observables on
   non-Pauli graphs could still be stored (they only expand at Stim export).
   Reject at import, or defer the check as today?
