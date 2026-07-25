Architecture Overview
=====================

GraphQOMB is built around a simple idea: keep the major MBQC concerns separate until lowering time.

Instead of mixing graph structure, classical dependency logic, and execution order into one mutable object, GraphQOMB treats them as distinct compiler interfaces. This matches how MBQC workflows are often optimized in practice: graph rewrites, feedforward simplification, and scheduling are related, but they are not the same problem.

Three compiler interfaces
-------------------------

The main GraphQOMB workflow uses three explicit inputs:

- **Labelled graph state**
  The resource graph together with measurement bases and input/output registration.
- **Feedforward maps**
  `xflow` and optional `zflow` record how measurement outcomes propagate as classical corrections.
- **Scheduler**
  A schedule that assigns preparation, entanglement, and measurement work to executable slices.

The entry point is :func:`graphqomb.qompiler.qompile`, which lowers those inputs to a :class:`graphqomb.pattern.Pattern`.

Lowering output
---------------

The lowered pattern combines:

- a scheduled command stream,
- a :class:`graphqomb.pauli_frame.PauliFrame` used for dependency tracking,
- derived metrics such as depth, space usage, and active volume.

Most scheduled work is serialized as prepare, entangle, and measure commands separated by ``TICK`` slice boundaries.
Pauli corrections are retained in the :class:`graphqomb.pauli_frame.PauliFrame` rather than emitted as ``X``/``Z`` commands, so the executable command stream is limited to ``N``, ``E``, ``M``, and ``TICK``.
Output nodes with an assigned measurement basis are projective readouts of the output register: they are scheduled and measured inside the timeline like any other node, and their results participate in feedforward.
Outputs without a basis remain quantum, and the pattern simulator materializes their pending frame corrections when returning an output statevector.

Feedforward and causality
-------------------------

GraphQOMB assumes that feedforward dependencies are causally executable.

- `xflow` is always required.
- If `zflow` is omitted, :func:`graphqomb.qompiler.qompile` derives it from odd neighborhoods.
- The lowering path validates the dependency structure by building a dependency DAG and rejecting cyclic feedforward.

This makes the library suitable for static, branch-free MBQC patterns. Deterministic semantics still depend on the feedforward structure you provide: standard flow, gflow, or related stabilizer-derived constructions remain the usual way to guarantee deterministic execution.

Scheduling and `TICK`
---------------------

The scheduler is an explicit object, not a hidden post-processing step.

This matters because schedule choices change the executable properties of the pattern:

- depth,
- maximum live-qubit count,
- active volume,
- entanglement timing seen by downstream tooling.

When a scheduler is present, each time slice becomes a ``TICK``-delimited block in the lowered pattern. This lets you inspect, simulate, and export the same executable schedule that was optimized upstream.

Workflow boundaries
-------------------

GraphQOMB is most useful in the middle of a broader MBQC toolchain:

- **Upstream**
  Circuit transpilation, graph construction, and graph rewrites can happen before lowering. Tools such as PyZX fit naturally here.
- **Core**
  GraphQOMB handles feedforward validation, schedule construction, pattern lowering, simulation, and pattern-level metrics.
- **Downstream**
  Compatible patterns can be exported with :func:`graphqomb.stim_compiler.stim_compile` for Stim-oriented workflows.

This separation is the main reason the package is documented as a compiler framework rather than a thin wrapper around another MBQC package.
