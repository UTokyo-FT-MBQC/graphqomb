Stim MPP rewriter (experimental)
================================

Install the optional Stim integration before importing this module:

.. code-block:: console

   uv add "graphqomb[stim]"

The rewriter exposes the Pauli products measured by Clifford syndrome
extraction without deleting or approximating the extraction circuit. For a
Clifford body ``U`` and Pauli measurement ``P``, it uses the exact instrument
identity

.. math::

   \Pi_m(P) U = U \Pi_m(U^\dagger P U).

Clifford instructions accumulate in a pending frame. At a source Pauli
measurement, the rewriter emits the pulled product ``U† P U`` and leaves the
unchanged frame behind that measurement. A reset, measure-reset,
measurement-record-controlled gate, or circuit exit materializes the pending
frame. This is one deterministic path: there is no verification retry or
gate-level fallback, and measurement post-states are preserved exactly.

Reset stabilizer substitution
-----------------------------

When a source direct measurement's pulled product contains the same Pauli on
that source qubit as its most recent reset preparation, that factor has known
eigenvalue ``+1`` and is removed. Only the directly measured source qubit is
eligible; reset data-qubit factors remain in the product. Standard check
gadgets therefore expose a data-only ``MPP``:

.. code-block:: text

   R 4                       R 4
   CX 0 4 1 4       ->       MPP Z0*Z1
   M 4                       CX 0 4 1 4

The trailing Clifford body is intentional. Applying it after the pulled
measurement makes the transformed circuit exactly equivalent to the source,
including the measurement record and post-measurement quantum state. A factor
that does not match the reset basis is retained. For example,
``RX 2; CZ 0 2; S 2; MX 2`` becomes
``RX 2; MPP !Z0*Y2; CZ 0 2; S 2``.

Factors are considered in measurement-record order. An earlier product that
anticommutes with a stored reset stabilizer invalidates it before later
products are simplified. A negative identity is kept as a real signed
measurement rather than emitted as ``MPAD 1``, which the GraphQOMB importer
does not support.

Barriers and annotations
------------------------

``MR``, ``MRX``, and ``MRY`` are split only when needed: the pulled
measurement is emitted, the exact pending frame follows, and then the reset is
applied. Classical record feedback is copied verbatim after materializing the
frame. Sweep-bit controls, circuit-level noise, and noisy measurement
arguments are rejected. ``DETECTOR``, ``OBSERVABLE_INCLUDE``, tags, and qubit
coordinates are copied while retaining their measurement-record indices.
``REPEAT`` blocks are flattened first.

Commuting pulled products emitted inside one source ``TICK`` interval are
collected by the importer into one MPP graph fragment, even when tags keep them
as separate Stim instructions. A source ``TICK`` is a hard boundary: products
on opposite sides are built as separate graph layers and are never coalesced.

The historical ``fallback`` argument and ``fallback_segments`` result field
remain for API compatibility. ``fallback="circuit"`` and
``fallback="segment"`` select the same exact pipeline, and
``fallback_segments`` is always empty.

.. code-block:: python

   from graphqomb.stim_glue.mpp_rewriter import rewrite_to_mpp

   result = rewrite_to_mpp(
       """
       R 4
       CX 0 4 1 4
       M 4
       """
   )
   assert str(result.checks[0].product) == "+ZZ___"

API reference
-------------

.. automodule:: graphqomb.stim_glue.mpp_rewriter
   :members:
   :show-inheritance:
