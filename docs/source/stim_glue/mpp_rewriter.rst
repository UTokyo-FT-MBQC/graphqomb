Stim MPP rewriter
=================

Install the optional Stim integration before importing this module:

.. code-block:: console

   uv add "graphqomb[stim]"

The rewriter exposes the Pauli products measured by Clifford syndrome
extraction. ``result.circuit`` preserves the complete measurement and quantum
channel; ``result.foliation_circuit`` preserves the measurement-record
distribution up to the explicit ``foliation_record_to_source`` permutation,
contracts structurally recognized disposable check gadgets, and omits the final pending
Clifford for import into MBQC. Detector and observable meanings are preserved.
For a Clifford body ``U`` and Pauli measurement ``P``, it uses the exact
instrument identity

.. math::

   \Pi_m(P) U = U \Pi_m(U^\dagger P U).

Clifford instructions accumulate in a pending frame. At a source Pauli
measurement, the rewriter emits the pulled product ``U† P U`` and leaves the
unchanged frame behind that measurement. A reset, measurement-record-controlled
gate, or circuit exit materializes the pending frame in ``result.circuit``.
At a measure-reset whose
source-ancilla reset factor was removed, the rewriter compares the local
reset/body/measurement channel with the reduced MPP/reset channel. Equal
canonical Stim flows certify that the circuit-foliation MPP ancilla has replaced the
source extraction ancilla, so the now-redundant pending body is discarded.
Otherwise the exact body is materialized unchanged. There is no gate-level
fallback, and measurement post-states are preserved exactly in ``result.circuit``.

Terminal Clifford removal for circuit foliation
-----------------------------------------------

``result.foliation_circuit`` is intended for pipelines that consume measurement
records and discard terminal quantum states. It omits the pending Clifford at
circuit exit, after all measurements have been pulled. For each complete record
``m``, the unnormalized conditional state ``rho_m`` satisfies

.. math::

   p(m) = \operatorname{Tr}(U\rho_m U^\dagger)
        = \operatorname{Tr}(\rho_m).

Thus the joint record distribution, including detector and observable parities,
is unchanged. No flow analysis or equivalence check is performed for this
omission, even for a multi-qubit Clifford. For example, ``H 0; M 0`` becomes
``MPP X0; H 0`` in ``result.circuit`` and ``MPP X0`` in
``result.foliation_circuit``. This also omits a final pending body when some
qubits are unmeasured, or the circuit contains no measurements: quantum outputs
are outside the foliation result's contract. Use ``result.circuit`` when those
outputs are needed.

In the exact quantum-channel result, reset and feedback boundaries still
materialize the pending frame unless measure-reset contraction succeeds. A reset discards
only its target subsystem. It does not erase a preceding Clifford's action on
other qubits: ``X 0; CX 0 1; R 0; M 1`` still measures ``1`` on qubit 1.
Resetting every qubit on which the pending body acts would allow that body to
be omitted without a flow check, but this optimization is not applied here.

Disposable check gadgets for circuit foliation
----------------------------------------------

The foliation path also recognizes a plain syndrome measurement followed by a
separate reset, or by no further quantum use of that source ancilla. This
additional pass uses a constructive controlled-Pauli rule, with no
``flow_generators`` or ``has_flow`` calls. It recognizes X-prepared,
X-measured controls of CX/CZ gates, with no independent data Clifford.

To group the controlled factors in measurement order, exchanging two
anticommuting factors with distinct controls introduces CZ between the controls.
The pass tracks these phase terms modulo two and requires them to cancel.
Factors sharing a control and a data qubit must have the same axis, so each
grouped body is a Hermitian controlled Pauli. For each resulting product P,
the readout Kraus operator is exactly

.. math::

   \langle m_X|C(P)|+\rangle = (I + (-1)^m P)/2.

Thus the grouped bodies implement the emitted sequential Pauli measurements
for arbitrary data inputs, including inputs entangled with other systems.
This is a sufficient structural rule, not a general equivalence decision
procedure. Other shapes retain their original body. Feedback and intervening
uses of a measured ancilla prevent contraction. The exact ``result.circuit``
is unchanged by this pass.

This avoids importing both the original syndrome-extraction circuit and its
replacement MPP gadget. For surface-code extraction this duplication could
produce resource-state vertices with degree greater than four.

When one direct-measurement instruction interleaves syndrome ancillas and data
readouts, recognized syndrome MPPs are emitted first, followed by a ``TICK`` and
the data readouts. The original measurements on distinct source qubits commute;
their record order changes, so all detector, observable, and feedback references
are remapped. ``result.foliation_record_to_source[j]`` gives the original record
index of foliation record ``j`` (including padding records).
``result.foliation_checks`` describes the products in this new order, while
``result.checks`` still describes the exact result in original source order.
Callers comparing raw sample columns must apply this permutation. Callers
using detector or observable annotations need no additional remapping.

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

In ``result.circuit``, the trailing Clifford body is intentional. Applying it
after the pulled measurement makes the transformed circuit exactly equivalent to the source,
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
those idle source ancillas and omits the final pending Clifford for import.
``result.eliminated_qubits`` reports the removed reset-only ancillas' Stim ids.
This certificate is exactly as sound as Stim's flow analysis,
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
At circuit exit, ``foliation_circuit`` is built before the final pending body
is appended to ``result.circuit``. The rewriter removes certified ancillas that
now occur in resets or coordinates alone from the foliation result. Ancillas
with remaining quantum uses are retained. Plain measurement alone does not
permit discarding the pending body when preserving quantum outputs; the
foliation result discards that final body because it preserves only records.

Each attempted certificate with a nonempty pending body calls
``flow_generators()`` twice. This is stabilizer algebra, without enumerating
measurement outcomes. The cost depends on circuit width, pending-body size,
and the number of eligible reset boundaries. Currently the comparison includes
all tracked preparations and uses the original qubit ids; it does not compact
to just one ancilla's neighbors or cache repeated certificates. Large or sparse
qubit ids can therefore increase cost. ``REPEAT`` is flattened, so each round
is processed separately. The final idle-qubit removal is a circuit scan and
does not call ``flow_generators()`` again.

The new disposable-gadget prepass adds no stabilizer-flow verification. Let N
be the flattened instruction/target count (including record references), G the
number of controlled factors, A the number of emitted check products, Q the
largest qubit id plus one, and D the largest number of distinct candidate
controls touching a data qubit in one buffered body. Its scans and phase
updates cost O(N + G D), apart from sorting/serialization. Dense Stim Pauli
products add O(A Q) bits of initialization/scanning (packed internally), so
this implementation is not strictly linear in N when width grows. For bounded
surface-code incidence D is constant; arbitrary dense incidence can make the
phase bookkeeping quadratic. The record permutation and flattened source
require O(N) storage. The public API also builds the exact result, and, when
contraction succeeds, performs a second exact rewrite of the contracted source;
that transformation cost is additional, not correctness verification.
The older measure-reset flow certificates described above remain separate
and were not introduced by the disposable-gadget fix.

Barriers and annotations
------------------------

``MR``, ``MRX``, and ``MRY`` are split only when needed: the pulled
measurement is emitted, an exactly contractible source-ancilla extraction body
is discarded (otherwise it follows unchanged), and then the reset is applied.
In the exact result, classical record feedback is copied after materializing the frame.
Sweep-bit controls, circuit-level noise, and noisy measurement
arguments are rejected. ``DETECTOR``, ``OBSERVABLE_INCLUDE``, tags, and qubit
coordinates are retained. The foliation path remaps record references when
structural mixed-readout contraction permutes measurements.
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
