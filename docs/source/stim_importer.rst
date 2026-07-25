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

Every unitary block is normalized by :mod:`graphqomb.stim_parser` to the four
Clifford ``J(angle)`` gates (``H = J(0)``, ``HS = J(pi/2)``, ``HZ = J(pi)``,
``HS_DAG = J(-pi/2)``, corresponding to X+/Y+/X-/Y- measurements) plus ``CZ``
before graph conversion. This accepts all fixed one- and two-qubit Clifford
gates exposed by Stim as well as ``SPP`` and ``SPP_DAG``. Each basis gate
becomes one GraphQOMB ``J`` primitive.

Initial reset instructions
--------------------------

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

Mid-circuit resets are rejected because GraphQOMB patterns do not currently
represent multiple lifetimes for one logical qubit. Combined measurement-reset
instructions (``MR``/``MRZ``, ``MRX``, and ``MRY``) remain unsupported.

Single-qubit measurements assign an ``AxisMeasBasis`` directly to the measured
data-lane endpoint without replacing that node or its coordinate. They do not
create an ``MPP`` extraction or an ancillary parity measurement node. Inverted
single-qubit measurement targets select the minus sign of that node's basis. A
direct single-qubit measurement terminates that qubit's lifetime: a later
quantum operation on the same qubit is rejected, while operations on other
qubits may continue. A measured qubit cannot begin a new lifetime later in the
circuit.

The first two components of ``QUBIT_COORDS`` are used as the fixed spatial
``(x, y)`` position of each data lane. The importer supplies the temporal ``z``
component. Every unitary ``TICK`` block is transpiled before placement. Its
input layer starts at the preceding block's output ``z``, and all of its output
nodes share the maximum transpiled depth of the block. A shorter data-wire
chain is spread across that same interval. A live qubit with no operation in
the block remains a single input/output node and is relocated directly to the
common output layer; this adds no graph node or edge and does not change the
circuit semantics.

Two-qubit measurements are parity measurements and are lowered to equivalent
unsigned ``MPP`` products. Inverted targets in ``MXX``, ``MYY``, ``MZZ``, and
``MPP`` are rejected because GraphQOMB does not currently retain the
corresponding parity offset.

Classical Pauli feedback
------------------------

Stim measurement-record controls are imported as explicit Pauli-frame
corrections. ``CX rec[-k] q``/``CNOT rec[-k] q`` adds ``q`` to the
controlling measurement node's X-correction flow, ``CZ rec[-k] q`` adds it to
the Z-correction flow, and ``CY rec[-k] q`` adds it to both. The equivalent
reverse-target spellings ``XCZ q rec[-k]`` and ``YCZ q rec[-k]`` are also
supported, as is the symmetric ``CZ q rec[-k]`` form. Batched target pairs in
one Stim instruction are supported when every pair is a feedback pair. Mixing
quantum and feedback pairs in one instruction remains unsupported; use
separate instructions.

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
I and Type II foliation.

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

   from graphqomb.qec.qeccode import YFoliation
   from graphqomb.stim_importer import stim_text_to_pattern

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

.. automodule:: graphqomb.stim_importer
   :members:
   :show-inheritance:
