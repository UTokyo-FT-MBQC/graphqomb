Feedforward
===========

:mod:`graphqomb.feedforward` module
++++++++++++++++++++++++++++++++++++

.. automodule:: graphqomb.feedforward

Parity compilation before lowering
----------------------------------

``Feedforward`` owns correction maps together with detector supports,
logical-observable supports, and detector tags. A support always refers to
the measurement records produced while its maps are in effect. Backpropagation
is an explicit transformation of this IR, before scheduling and lowering::

    from graphqomb.feedforward import Feedforward
    from graphqomb.qompiler import qompile

    feedforward = Feedforward(
        xflow, zflow, cflow,
        parity_check_group=detector_seeds,
        logical_observables=observable_seeds,
        parity_check_tags=detector_tags,
    )
    compiled = feedforward.backpropagate(graph)
    pattern = qompile(graph, compiled)

``backpropagate`` normalizes Pauli factors of cflow, then protects every
nontrivial Clifford target and all its ancestors in the combined correction
DAG. Quantum outputs and non-Pauli measurements, with their ancestors, are
protected too. Incoming corrections on protected nodes remain executable.
This preserves the original Clifford control bits and noncommuting event order.

Outside that region, incoming Pauli corrections are removed. Their outcome
relabelings are compiled into each detector and observable by XOR substitution:
X measurements inherit Z controls, Z measurements inherit X controls, and Y
measurements inherit the symmetric difference of both. Substitution stops at
protected records. Retained corrections are not absorbed into the parities a
second time. Clifford feedback remains in the returned IR, rather than being
expressed as a static XOR over nominal-basis records.

The input object is unchanged. Reapplying backpropagation is idempotent.
Individual records can change convention; detector/observable values,
retained adaptive behavior, and corrected quantum outputs are preserved.
This transformation does not certify detector determinism, including for
encoded inputs or branch-dependent Clifford measurements.

``qompile(graph, compiled)`` copies the residual maps and supports into the
runtime frame. It does not apply another parity transformation. The existing
map-based qompile arguments remain available. ``pattern.frame.feedforward``
returns an independent static IR snapshot for transformation and recompilation.
The stored ``frame.parity_check_group`` and ``frame.logical_observables``
are the supports over that pattern's reported records. The frame's
``detector_groups`` method provides nominal-basis supports for determinism
diagnostics; it is not a compilation step and rejects Clifford-influenced
raw-record expansion. The frame's ``logical_observable_groups`` and
``logical_observables_group`` methods also remain available for legacy
nominal-basis compilation. Frame-side compilation and the legacy Stim export
path are scheduled for deprecation in v0.8.0; no warning is emitted yet.
New workflows transform observables together with correction maps using
``Feedforward.backpropagate``.

Parity expansion uses one iterative affine-substitution implementation for
compilation and diagnostics, without recursive closures or persistent frame
caches.

``stim_compile(pattern, backpropagate=False)`` serializes stored supports
directly and executes any retained Pauli feedforward as record-controlled
corrections at measurement/output boundaries, without further backpropagation.
For compatibility, ``stim_compile(pattern)`` defaults to ``backpropagate=True``:
it expands supports through the frame's legacy methods and emits nominal
measurements without executing retained correction maps. In particular, the
legacy path leaves quantum-output corrections implicit. Both paths reject
nontrivial cflow.

.. autoclass:: graphqomb.feedforward.Feedforward
   :members:
   :member-order: bysource

Signal shifting with Clifford feedforward
-----------------------------------------

``signal_shifting`` and ``propagate_correction_map`` accept ``cflow`` while
preserving its control records and correcting gates. They return the usual
pair of X/Z maps; pass the original ``cflow`` when compiling those maps::

    shifted_xflow, shifted_zflow = signal_shifting(graph, xflow, zflow, cflow=cflow)
    pattern = qompile(graph, shifted_xflow, shifted_zflow, cflow=cflow)

The optimization uses a static boundary. In the combined X/Z/Clifford
dependency DAG, let ``B`` contain every effective ``cflow`` target and all
of its ancestors. Signal shifting applies the ordinary measurement-plane
rules only outside ``B``. An individual propagation call at a node in ``B``
returns copies of the unchanged X/Z maps. Identity and self-target cflow
entries have no runtime action and create no boundary. Pauli-valued cflow
entries are protected too, since their controls cannot be rewritten by a
pass returning only X/Z maps; use ``xflow``/``zflow`` for optimizable Pauli
corrections.

This boundary preserves both the Clifford control bits and the ordering of
noncommuting corrections. Protecting only the endpoints of cflow edges would
not suffice: shifting an earlier measurement can advance its outgoing Pauli
correction past a Clifford on another target. The backward closure prevents
this because no node outside ``B`` sends corrections into ``B``. It also
preserves all dependency paths ordering corrections on Clifford targets.
The closure is computed once before the whole signal-shifting pass, with no
branch enumeration or runtime checks. Pauli regions after the boundary and
independent regions remain eligible for optimization.

Execution order is computed after the same Pauli-factor normalization used
by the Clifford frame, so identity entries and cancelled Pauli corrections
do not create spurious cycles. The protection graph additionally retains
the original effective cflow edges, even when they cancel an X/Z correction;
rewriting only the X/Z half would break that cancellation.

This is a conservative optimization boundary, not an operation that resets
the frame or deletes pending corrections. The simulator still executes the
retained Clifford feedforward. Fully shifting through these boundaries would
require ordered Clifford events with parity-valued controls: in general
``S ** (a XOR b)`` cannot be replaced by ``S ** a`` followed by ``S ** b``.

As with Pauli-only signal shifting, records in rewritten regions are
relabelled. These functions do not transform detector or logical-observable
seeds. Use the joint IR transformation above to compile parity supports while
retaining Clifford boundaries. ``pauli_simplification`` still requires an empty
cflow. Do not run a map-only record-relabeling pass after parity compilation
without updating the associated supports.

The :doc:`gallery/t_gate_teleportation` example implements T injection by
measuring ZZ between separate data and magic inputs, then measuring the magic
input in X. A third node, prepared in ``|+>`` and CZ-coupled to both inputs,
implements the ZZ measurement through its own X measurement. The data remains
the output and receives conditional S and Z corrections. Signal shifting
leaves this minimal gadget unchanged because both records are in the causal
past of the data output, a Clifford target. The example supplies the magic
state through the statevector simulator's input state; graph input
initialization metadata remains restricted to Pauli eigenstates.

Functions
---------

.. autofunction:: graphqomb.feedforward.dag_from_flow

.. autofunction:: graphqomb.feedforward.inverse_dag_from_dag

.. autofunction:: graphqomb.feedforward.topo_order_from_inv_dag

.. autofunction:: graphqomb.feedforward.check_dag

.. autofunction:: graphqomb.feedforward.check_flow

.. autofunction:: graphqomb.feedforward.signal_shifting

.. autofunction:: graphqomb.feedforward.propagate_correction_map
