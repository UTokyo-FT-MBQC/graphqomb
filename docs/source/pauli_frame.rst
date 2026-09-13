Clifford Frame
==============

:mod:`graphqomb.pauli_frame` module
++++++++++++++++++++++++++++++++++++

.. automodule:: graphqomb.pauli_frame

Residual convention and correction order
----------------------------------------

The frame records the residual :math:`F`, with
:math:`|\psi_{\rm actual}\rangle=F|\psi_{\rm corrected}\rangle`.
The gates supplied in ``cflow`` are **correcting gates**. If the reference
applies them in order :math:`C_1,\ldots,C_r`, then

.. math::

   K=C_r\cdots C_1,\qquad F=K^\dagger=C_1^\dagger\cdots C_r^\dagger.

Each measurement event updates :math:`F\leftarrow F C^\dagger`.
The executed observable is :math:`F A F^\dagger=K^\dagger A K`, and
output correction applies :math:`F^\dagger=K`.
For example, ``cflow={u: {v: S}}`` means apply correcting gate :math:`S`
on outcome one: the residual is :math:`S^\dagger`, and nominal X readout
becomes :math:`-Y`. It does not become :math:`Y`.

``correction_events`` records ordered tuples for each source, using the same
representation for commuting and noncommuting events, including Pauli events.
Within a source, application order is X, Z, then the normalized coset.
A map value represents one composite gate for each source/target pair;
normalization combines only Pauli factors with that same control and target.
Sources execute in causal measurement order. Noncommuting corrections to a
common target require comparable sources in the dependency DAG. Commuting
sources can run in either order without changing the product modulo phase.

Self-targets are allowed in all three flow maps, but excluded from dependency
edges and runtime correcting events. Sources with nonempty normalized maps
must be measured; unmeasured outputs cannot provide control bits. This is
checked by ``check_flow`` when constructing a frame.
The .ptn representation stores the normalized maps from which this event order
is reconstructed; it stores program corrections, not the live residual state.

Scope
-----

``qompile(..., cflow=...)`` accepts general single-qubit Clifford corrections
already placed at measurement/output boundaries. This is an execution
interface, not a general conditional-gate transpiler or a determinism test.
The current circuit transpilers do not lower arbitrary conditional Cliffords.
The planned new circuit-side support is limited to diagonal :math:`S`,
:math:`S^\dagger`, and :math:`Z`: they commute with CZ, and deferral must stop
at the next non-diagonal operation or frontier change. General conditional
gate decomposition and entangling feedback are outside that extension.

General determinism criteria for branch-dependent measurement labels are
future work. No Lambda-flow theorem or finder is implemented or claimed.
Causality and the existing Pauli detector checks keep their separate meanings.

Clifford Frame Class
--------------------

.. autoclass:: graphqomb.pauli_frame.CliffordFrame
    :members:
    :member-order: bysource

.. data:: graphqomb.pauli_frame.PauliFrame

    Backwards-compatible alias of :class:`graphqomb.pauli_frame.CliffordFrame`.
