Stim circuit import
===================

Install the optional Stim integration before importing this module:

.. code-block:: console

   uv add "graphqomb[stim]"

The circuit importer converts supported Stim circuits into GraphQOMB
measurement patterns. It accepts initial Pauli resets, Clifford unitary blocks,
and Pauli measurement blocks separated by ``TICK``. The supported Pauli
measurement instructions are ``M``/``MZ``, ``MX``, ``MY``, ``MXX``, ``MYY``,
``MZZ``, and ``MPP``.

``StimImportResult.record_to_node`` maps absolute source measurement-record
indices to graph measurement nodes, enabling external conditional-Clifford
feedforward to use source record identities after import. Records removed as
deterministic zero padding have no graph node and are omitted from the mapping.

Import before scheduling
------------------------

:func:`~graphqomb.stim_glue.importer.stim_circuit_to_ir`,
:func:`~graphqomb.stim_glue.importer.stim_text_to_ir`, and
:func:`~graphqomb.stim_glue.importer.stim_file_to_ir` return a frozen
:class:`~graphqomb.stim_glue.importer.StimImportIR` before constructing or
solving any scheduler. It contains the resource graph, a
:class:`~graphqomb.feedforward.Feedforward` with corrections and parity
annotations, and the same source mappings as ``StimImportResult``.

Use ``ir.make_scheduler()`` to construct an unsolved scheduler. Inspect
``ir.graph.coordinates`` and ``ir.record_to_node`` when choosing operation
times, then call ``manual_schedule`` or ``solve_schedule``. Lower with
``ir.to_pattern(scheduler)`` to obtain a ``StimImportResult``. Lowering delegates
schedule validation to ``qompile`` and checks that the scheduler uses the IR's graph.
The existing ``*_to_pattern`` entry points perform these steps using the
requested configuration or the default greedy ``MINIMIZE_TIME`` strategy.

.. code-block:: python

   from dataclasses import replace
   from graphlib import TopologicalSorter

   from graphqomb.feedforward import inverse_dag_from_dag
   from graphqomb.stim_glue import stim_text_to_ir

   ir = stim_text_to_ir("RX 0\nTICK\nMPP X0\nTICK\nMX 0")
   ir = replace(ir, feedforward=ir.feedforward.backpropagate(ir.graph))
   scheduler = ir.make_scheduler()
   # Prepare at 0, entangle serially, then follow measurement dependencies.
   edge_times = {edge: t for t, edge in enumerate(ir.graph.edges, start=1)}
   order = TopologicalSorter(inverse_dag_from_dag(scheduler.dag)).static_order()
   scheduler.manual_schedule(
       dict.fromkeys(scheduler.prepare_time, 0),
       {
           node: t
           for t, node in enumerate(order, start=len(edge_times) + 1)
           if node in scheduler.measure_time
       },
       edge_times,
   )
   result = ir.to_pattern(scheduler)

Use ``dataclasses.replace(ir, feedforward=...)`` to apply transformed feedforward
before scheduling and lowering; both methods use the resulting IR's feedforward.
Export with ``stim_compile(result.pattern, backpropagate=False)`` to serialize
those supports.

Graph construction
------------------

Every unitary block is normalized by :mod:`graphqomb.stim_glue.transpiler` to the four
Clifford ``J(angle)`` gates (``H = J(0)``, ``HS = J(pi/2)``, ``HZ = J(pi)``,
``HS_DAG = J(-pi/2)``, corresponding to X+/Y-/X-/Y+ measurements) plus ``CZ``
before graph conversion. This accepts all fixed one- and two-qubit Clifford
gates exposed by Stim as well as ``SPP`` and ``SPP_DAG``. Each basis gate
becomes one GraphQOMB ``J`` primitive.

Here, **circuit foliation** means constructing an MBQC pattern from the
operations of a Clifford circuit, including gates, resets, measurements, and
feedback. We distinguish this circuit construction from code foliation for
memory operations such as CSS-code memories. ``YFoliation`` selects the graph
construction for products containing Y factors.

For closed syndrome-extraction windows, first apply :doc:`round_rewriter`
and pass its ``result.circuit`` to this importer. Record references already
include any readout permutation; Pauli feedback follows the ordinary
``xflow``/``zflow`` import path.

Initial reset instructions
--------------------------

Without an explicit reset, an input uses the ``Z+`` state (``|0>``), matching
Stim's default initialization. Both ``M 0`` and ``R 0; M 0`` therefore return
zero noiselessly. To prepare ``|+>``, specify ``RX`` before the qubit's first
quantum use. This default applies to Stim circuit import; GraphQOMB's general
input initialization default remains ``X+``.

Leading reset instructions determine the positive Pauli eigenstate used for an
input: ``R``/``RZ`` initializes ``Z+``, ``RX`` initializes ``X+``, and ``RY``
initializes ``Y+``. A reset is initial when its target qubit has not previously
participated in a unitary or measurement operation. Operations on other qubits
do not prevent a later initial reset. Repeated leading resets are accepted, and
the last reset on each qubit determines its initialization state.

Leading resets are normalized together with the Clifford gates in their
``TICK`` block. When the resulting state is a positive Pauli eigenstate, the
gates are absorbed into ``R``, ``RX``, or ``RY`` before graph construction.
For example, ``R`` followed by ``H`` becomes ``RX`` and does not add a gate
node or shift that input to an earlier ``z`` layer.

Stim canonicalizes the Z-axis aliases internally, so the importer accepts both
``M`` and ``MZ`` for Z measurement and both ``R`` and ``RZ`` for Z reset. Stim
export uses ``MZ`` for Z measurement and ``R`` for Z reset.

A mid-circuit reset starts a new wire on a fresh internal qubit index: the
reset and every later operation on that Stim qubit continue on the new wire,
which is initialized in the positive eigenstate of the reset axis at the
qubit's XY coordinates on the next ``z`` layer. The reset discards the
previous state, so the new wire carries no outcome conditioning; when several
resets act on the qubit before its next quantum use, the last reset determines
the axis. A replaced wire that was not consumed by a direct measurement
remains a pattern output, which is channel-equivalent to the trace-out reset
on every recorded measurement. Combined measurement-reset instructions
(``MR``/``MRZ``, ``MRX``, and ``MRY``) are imported as a direct measurement
whose continuation re-prepares the measured axis eigenstate unconditionally.

``StimImportResult.qubit_to_stim`` maps every fresh index back to the original
Stim qubit id. Because that mapping is many-to-one for a reset qubit,
``stim_to_final_qubit`` gives the index of its last lifetime; the earlier
indices are discarded history. Equivalence to the source circuit is equivalence
of the recorded measurements, not of the pattern's output channel: a replaced
live wire is a purification of Stim's trace-out reset, so a caller that reads
every pattern output must trace out the outputs that are not named by
``stim_to_final_qubit``.

Internally each reset lifetime also receives its own Stim qubit id, and the
``StimMppExtraction`` metadata of an MPP block (``supports``,
``stim_to_column``, and ``column_to_stim``) reports the lifetime that each
product measured. A qubit measured before and after a reset therefore occupies
a different column in each block, and the same physical qubit may carry
different ids across blocks. ``StimImportResult.wire_to_stim`` maps every such
id back to the original circuit id; it is the identity for qubits that are
never reset mid-circuit.

Within one optimization block, single-qubit Cliffords preceding a terminal
measurement are absorbed into its signed Pauli basis when its post-state is
not reused. For example, ``H 0; M 0`` becomes ``MX 0``, and
``S 0; MX 0`` becomes ``MY !0``. These readouts need no extra gate node.
``TICK`` barriers and measurements whose qubits are reused limit this folding;
the latter must preserve the post-measurement state.

Single-qubit measurements assign an ``AxisMeasBasis`` directly to the measured
data-lane endpoint without replacing that node or its coordinate. They do not
create an ``MPP`` extraction or an ancillary parity measurement node. Inverted
single-qubit measurement targets select the minus sign of that node's basis. A
direct single-qubit measurement ends that wire: a later quantum operation on
the same qubit continues on a fresh internal qubit index initialized in the
positive eigenstate of the measurement axis and conditioned on the outcome
through the correction flows. Reuse after an inverted plain target (``M !q``)
is rejected because negative-eigenstate initialization cannot be represented,
while an inverted measure-reset target may be reused because the reset
discards the outcome.

The first two components of ``QUBIT_COORDS`` are used as the fixed spatial
``(x, y)`` position of each data lane. The importer supplies the temporal ``z``
component. Wires split from a mid-circuit reset inherit the original qubit's
``(x, y)``. Fully coordinated and fully uncoordinated circuits both import
silently; when only some wires carry ``QUBIT_COORDS`` the importer emits a
`UserWarning`, because a mixture usually means coordinate metadata was lost
upstream — for example reset lifetimes split onto fresh qubit ids by another
tool, which cannot be mapped back to an original qubit. Every unitary ``TICK``
block is transpiled before placement. Its
input layer starts at the preceding block's output ``z``, and all of its output
nodes share the maximum transpiled depth of the block. A shorter data-wire
chain is spread across that same interval. A live qubit with no operation in
the block remains a single input/output node and is relocated directly to the
common output layer; this adds no graph node or edge and does not change the
circuit semantics.

Terminal single-qubit readouts and classical Pauli feedback introduce no new
graph layers on continuing wires. Only measurements followed by quantum reuse
need a continuation node on a new layer. Resource Z coordinates describe graph
placement; execution times are stored independently in the pattern schedule.

Two-qubit measurements are parity measurements and are lowered to equivalent
unsigned ``MPP`` products. Inverted targets in ``MXX``, ``MYY``, and ``MZZ``
are rejected by the importer and retained unchanged by ``rewrite_syndrome_rounds``.
Express these measurements as signed ``MPP`` products before import; for example,
replace ``MXX !0 1`` with ``MPP !X0*X1``. Signed ``MPP`` is supported: its
product sign changes the ancilla measurement basis. The Type-I Y construction also contributes its
own phase correction. Extraction ``supports`` and check matrices describe
unsigned support; the pattern retains the measurement sign.

Classical Pauli feedback
------------------------

Stim measurement-record controls are imported as explicit Pauli-frame
corrections. ``CX rec[-k] q``/``CNOT rec[-k] q`` adds ``q`` to the
controlling measurement node's X-correction flow, ``CZ rec[-k] q`` adds it to
the Z-correction flow, and ``CY rec[-k] q`` adds it to both. The equivalent
reverse-target spellings ``XCZ q rec[-k]`` and ``YCZ q rec[-k]`` are also
supported, as is the symmetric ``CZ q rec[-k]`` form. Batched target pairs in
one Stim instruction are supported, including instructions mixing quantum and
feedback pairs. The importer splits those pairs in source order; this also
handles Stim's automatic fusion of adjacent instructions.

A feedback instruction is a graph-fragment boundary, so its correction targets
the data-lane node at that exact circuit position. Repeating the same
record-controlled Pauli correction cancels by parity. A feedback record must
refer to an earlier imported measurement; records idealized to zero produce no
correction.

Frame corrections are applied at their target's measurement, after every graph
edge on the target exists. An X or Y feedback therefore also XORs Z
corrections onto the neighbors the target becomes entangled with after the
feedback position, which is what pushing the deferred X through those later CZ
edges leaves behind. Neighbors entangled before the feedback position receive
no compensation, and Z feedback commutes with CZ so it needs none.

A feedback record may come from an ``MPP`` ancilla or from a direct
single-qubit measurement. Direct measurements are imported as measured output
nodes, which are scheduled like ordinary measurements, so the controlling
readout is always measured before any node that depends on its correction.

All ``MPP`` instructions within each feedback-free portion of one ``TICK``
block are represented by one combined extraction and are validated to commute.
Anticommuting products in the same portion are rejected. Within a combined
block, local stabilizer interactions are ordered ``Z -> Y -> X`` on each
shared data qubit. If an odd number of shared-data-qubit pairs reverse the
order of the same two stabilizers, the graph-state builder adds the required CZ
edge between their ancillas. This rule is applied automatically for both Type
I and Type II circuit foliation.

Only data-wire nodes contribute to the graph-generated X-correction flow; MPP
ancilla nodes do not produce those corrections. After composing all graph
fragments, the importer first derives the Z-correction flow from the odd
neighborhood of this graph-generated X flow. It then XORs Stim's explicit
classical X/Y/Z feedback into the two correction maps, including the
deferred-X compensation described above; a feedback entry never adopts the
full odd-neighbor rule, only the neighbors entangled after its position.
Both completed maps are passed directly to ``qompile()`` without Pauli
simplification or an importer-specific fallback. Non-commuting measurements
must be separated by ``TICK`` in the source circuit. Each unit has a distinct
unmeasured output layer, which is composed with the next unitary or measurement
fragment by qubit index. Pass
``y_foliation=YFoliation.TYPE_II`` to any of the three import entry points to use
the three-layer Y-measurement construction; Type I is the default.

An MPP block starts at the preceding gate or MPP output layer and ends two
``z`` units later. Live lanes not used by that MPP block are relocated to the
same output layer without adding nodes. Consequently, a composed output and
the next active fragment input have the same ``z`` coordinate, and imported
patterns do not mix 2D spatial coordinates with 3D spacetime coordinates.

The flattened ideal circuit is normalized before measurement-record analysis.
The original qubit inventory is retained even when a unitary block optimizes
to the identity. Records from single-qubit measurements, pair measurements,
``MPP``, and ideal-zero ``MPAD`` replacements share one absolute index space.
``DETECTOR`` and ``OBSERVABLE_INCLUDE`` targets are resolved against that
whole-circuit index space, and each ``StimMppExtraction`` retains the
corresponding absolute record sets even when an annotation also references
records outside that MPP unit.

Cat-state ancillas
------------------

By default an ``MPP`` product is measured through one ancilla node joined to
every data leg of the product. An X fault on that node between two of its CZ
gates acts as Z on each data wire coupled later, so one fault can reach
several data wires.

Pass ``mpp_ancilla="cat"`` to any import entry point, including the
Clifford+T ones, to measure each product with at least three data legs
through a ring. Leg :math:`d_i` is joined to its own cat node :math:`a_i`.
Connector :math:`c_i` joins :math:`a_i` to :math:`a_{i+1}`, and the last
connector closes the ring. Legs are graph edges, so a Y factor has two legs
under Type I foliation. A product with fewer legs keeps its single ancilla.
The legs follow the order of the product's targets, so the cat nodes of
neighboring targets are neighbors on the ring.

An instruction tag selects the ancilla for the products of one instruction
and overrides ``mpp_ancilla``:

.. code-block:: text

   MPP[ancilla=cat] X0*X1*X2*X3*X4*X5
   MPP[ancilla=single] Z0*Z1*Z2*Z3
   MPP X0*X1*X2*X3

The first instruction uses a ring, the second a single ancilla, and the third
follows ``mpp_ancilla``. Any other tag is ignored. A product is measured
through a full ring or through one ancilla; a ring with fewer cat nodes than
legs is not supported.

The original ancilla becomes the last cat node. Its measurement basis, its
edges to other ancillas, and its entry in ``record_to_node`` are unchanged,
and its corrected record is still the outcome of the product. The added cat
nodes and the connectors are measured in X.

The parity of the connector outcomes is fixed. The importer appends it to the
parity checks with the tag
:data:`~graphqomb.stim_glue.importer.CAT_FLAG_TAG`, after the source
detectors, which keep their indices. One fault on the ring then flips this
flag check or reaches at most one data wire, in any CZ order. Select the flag
checks of a compiled circuit by their tag to decode or post-select them
separately from the checks of the code:

.. code-block:: python

   import stim

   from graphqomb.stim_glue import (
       CAT_FLAG_TAG,
       flag_postselection_mask,
       stim_compile,
       stim_text_to_pattern,
   )

   result = stim_text_to_pattern("MPP Z0*Z1*Z2*Z3", mpp_ancilla="cat")
   circuit = stim.Circuit(stim_compile(result.pattern))
   mask = flag_postselection_mask(circuit, CAT_FLAG_TAG)

A flag check detects one fault per ring; two faults on the same ring can
cancel in it. The fault distance of a compiled circuit also depends on the
order of its CZ gates, which the schedule decides.
:doc:`/gallery/ftqc/color_code_cat_ancilla` computes the fault distance of a
color-code memory experiment for both ancilla kinds and several CZ orders.

Leg order
---------

A fault spreads only through the CZ gates applied after it, so the order in
which an ancilla node is entangled with its legs decides which data wires one
fault reaches. The importer takes this order from the source circuit: the
legs of a product follow the order of its targets. ``ir.mpp_legs`` maps the
measurement record of each product to its legs, as ``(ancilla node, data
node)`` edges in that order. A Y factor under Type I foliation contributes
its Z leg and then its X leg.

The default schedule entangles an edge as soon as both of its nodes are
prepared, and the edges of one time slice are in no particular order.
:meth:`~graphqomb.stim_glue.importer.StimImportIR.sequence_mpp_legs` rewrites
a schedule through ``Scheduler.manual_schedule`` so that the legs sharing an
ancilla node are entangled one per time slice, in target order:

.. code-block:: python

   from graphqomb.stim_glue import stim_text_to_ir

   ir = stim_text_to_ir("MPP Z0*Z1*Z2*Z3")
   scheduler = ir.make_scheduler()
   scheduler.solve_schedule()
   ir.sequence_mpp_legs(scheduler)
   result = ir.to_pattern(scheduler)

The prepare and measure order of the schedule is kept. A time slice is split
where an ancilla node has several legs in it, and a data node is prepared
earlier where a later target's data node would otherwise be measured before
it exists.

With a single ancilla on four targets, a fault on the ancilla between its
second and third CZ gate then leaves an error on the data wires of the last
two targets. Order the targets so that this pair does not shorten a logical
operator of the code. To schedule the legs by hand instead, pass entanglement
times for the edges of ``ir.mpp_legs`` to ``Scheduler.manual_schedule``.

Legs on different ancilla nodes are not ordered against each other, and
neither are the edges inside a cat ring. A cat node has one leg, so cat
ancillas add no time slices.

The ``*_to_pattern`` entry points keep the default schedule unless
``sequence_mpp_legs=True`` is passed, which applies the same rewrite to the
solved schedule:

.. code-block:: python

   result = stim_text_to_pattern("MPP Z0*Z1*Z2*Z3", sequence_mpp_legs=True)

``result.mpp_legs`` and ``ir.mpp_legs`` hold the same legs.

Noise policy
------------

Circuit-level Stim noise is intentionally not imported. GraphQOMB applies
noise to the compiled MBQC pattern through its own noise-model API, where
preparation, entanglement, measurement, and idle events differ from the source
circuit operations.

Pure noise instructions are omitted during import. Error probabilities attached
to Pauli measurements are also omitted while retaining the ideal measurement.
Heralded noise records are retained as ideal zero-valued record positions so
that later ``DETECTOR`` and ``OBSERVABLE_INCLUDE`` references remain aligned.

.. code-block:: python

   from graphqomb.qeccode import YFoliation
   from graphqomb.stim_glue.importer import stim_text_to_pattern

   result = stim_text_to_pattern(
       """
       RY 0
       R 1 2
       MX 0
       MYY 1 2
       DETECTOR rec[-2] rec[-1]
       """,
       y_foliation=YFoliation.TYPE_II,
   )
   pattern = result.pattern

API reference
-------------

.. automodule:: graphqomb.stim_glue.importer
   :members:
   :show-inheritance:
