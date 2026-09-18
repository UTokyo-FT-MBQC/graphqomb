Stim MPP rewriter
=================

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
unchanged frame behind that measurement. A reset, measurement-record-controlled
gate, or circuit exit materializes the pending frame. At a measure-reset whose
source-ancilla reset factor was removed, the rewriter compares the local
reset/body/measurement channel with the reduced MPP/reset channel. Equal
canonical Stim flows certify that the circuit-foliation MPP ancilla has replaced the
source extraction ancilla, so the now-redundant pending body is discarded.
Otherwise the exact body is materialized unchanged. There is no gate-level
fallback, and measurement post-states are preserved exactly.

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

For a measure-reset check gadget, the reset provides a stronger local
boundary. If ``reset + body + measure-reset`` and ``reset + reduced MPP +
reset`` have identical canonical Stim flows, the body is discarded: the
circuit-foliation graph constructed from the MPP already supplies the check ancilla's
initialization, interaction, and measurement. ``result.circuit`` retains any
independent reset-only source outputs so it stays exactly equivalent as a
standalone Stim channel. ``result.foliation_circuit`` additionally removes
those idle source ancillas for import, and ``result.eliminated_qubits`` reports
their Stim ids. This certificate is exactly as sound as Stim's flow analysis,
so the Stim extra requires ``stim>=1.16``.

Factors are considered in measurement-record order. An earlier product that
anticommutes with a stored reset stabilizer invalidates it before later
products are simplified. Substitution is skipped if it would remove the last
Pauli factor, for either sign. A known noiseless result still has a physical
readout: ``R 0; M 0`` remains a reset and measurement, not ``MPAD 0``.
In particular, terminal data measurements survive circuit foliation. Source
``MPAD`` records are copied; explicitly supplied identity products can still
be represented as padding. This does not add support for noisy input circuits.

Removal order and cost
----------------------

The flow check runs at an eligible measure-reset boundary, before any qubit
is removed. It compares two circuits containing the currently tracked
preparations, pending Clifford body, and local readout/reset. On equality,
the entire pending body (including its CNOTs) is discarded immediately.
At circuit exit, the rewriter scans the resulting circuit and removes only
certified ancillas that now occur in resets or coordinates alone. Ancillas
with remaining quantum uses are retained. Plain measurement alone does not
permit discarding the pending body.

Each attempted certificate with a nonempty pending body calls
``flow_generators()`` twice. This is stabilizer algebra, without enumerating
measurement outcomes. The cost depends on circuit width, pending-body size,
and the number of eligible reset boundaries. Currently the comparison includes
all tracked preparations and uses the original qubit ids; it does not compact
to just one ancilla's neighbors or cache repeated certificates. Large or sparse
qubit ids can therefore increase cost. ``REPEAT`` is flattened, so each round
is processed separately. The final idle-qubit removal is a circuit scan and
does not call ``flow_generators()`` again.

Barriers and annotations
------------------------

``MR``, ``MRX``, and ``MRY`` are split only when needed: the pulled
measurement is emitted, an exactly contractible source-ancilla extraction body
is discarded (otherwise it follows unchanged), and then the reset is applied.
Classical record feedback is copied verbatim after materializing the frame.
Sweep-bit controls, circuit-level noise, and noisy measurement
arguments are rejected. ``DETECTOR``, ``OBSERVABLE_INCLUDE``, tags, and qubit
coordinates are copied while retaining their measurement-record indices.
``REPEAT`` blocks are flattened first.

Commuting pulled products emitted inside one source ``TICK`` interval are
collected by the importer into one MPP graph fragment, even when tags keep them
as separate Stim instructions. A source ``TICK`` is a hard boundary: products
on opposite sides are built as separate graph layers and are never coalesced.
Removing a contracted measure-reset ancilla also preserves its round boundary:
if there is no source ``TICK`` before the next MPP, ``foliation_circuit`` inserts
one. More generally, a repeated identical Pauli support, or a product that
anticommutes with one already in the layer, starts a new internal layer even
within one source interval. Distinct commuting supports continue to share a
layer, so repeated identical checks are never fused together and every
imported MPP block commutes internally.
Pair measurements (``MXX``, ``MYY``, and ``MZZ``) are normalized to ``MPP``
in ``foliation_circuit`` and follow the same layer-separation rules, including
when mixed with explicit MPP products.

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

   # Use this circuit for the StabilizerCode circuit-foliation importer path.
   import_circuit = result.foliation_circuit

API reference
-------------

.. automodule:: graphqomb.stim_glue.mpp_rewriter
   :members:
   :show-inheritance:
