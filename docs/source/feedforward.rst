Feedforward
===========

:mod:`graphqomb.feedforward` module
++++++++++++++++++++++++++++++++++++

.. automodule:: graphqomb.feedforward

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
seeds. Clifford-dependent parity backpropagation remains unsupported, and
``pauli_simplification`` still requires an empty cflow.

The :doc:`gallery/t_gate_teleportation` example compiles a complete magic-state
T injection, applies this pass, and simulates the result with the statevector
backend. It supplies the magic state through the simulator's input state;
graph input initialization metadata remains restricted to Pauli eigenstates.

Functions
---------

.. autofunction:: graphqomb.feedforward.dag_from_flow

.. autofunction:: graphqomb.feedforward.inverse_dag_from_dag

.. autofunction:: graphqomb.feedforward.topo_order_from_inv_dag

.. autofunction:: graphqomb.feedforward.check_dag

.. autofunction:: graphqomb.feedforward.check_flow

.. autofunction:: graphqomb.feedforward.signal_shifting

.. autofunction:: graphqomb.feedforward.propagate_correction_map
