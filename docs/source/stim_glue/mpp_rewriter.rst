Stim MPP rewriter
=================

Install the optional Stim integration before importing this module:

.. code-block:: console

   uv add "graphqomb[stim]"

``rewrite_to_mpp`` normalizes photonic measurement lifetimes and replaces
recognized syndrome-extraction intervals with data-only Pauli-product
measurements. Other intervals keep their gates and readouts for the ordinary
circuit importer. A measured wire ends in either case; ending it never depends
on recognizing a gadget or comparing stabilizer flows.

Measurement contract
--------------------

Single-qubit ``M/MX/MY`` and ``MR/MRX/MRY`` consume their target. Reuse starts
a new wire in the positive eigenstate of the measured axis, independently of
the outcome. An explicit following reset supplies its own axis. Terminal
readouts create no preparation. Inverted targets change the recorded bit only;
repeated targets are processed sequentially. Classical record references and
coordinate annotations do not restart a measured wire. Explicit feedback acts
on its target's current lifetime.

This intentionally changes the interpretation of plain Stim measurements:
``X 0; M 0 0`` records ``1, 0`` under the new contract, rather than Stim's
``1, 1``. There is no legacy post-state-preservation mode. ``MPP`` and pair
measurements retain their parity-measurement semantics and do not consume
individual data wires. The importer uses the same lifetime normalization.

The result preserves the joint measurement-record distribution of the input
under this contract, up to ``foliation_record_to_source``. It does not promise
equivalence of terminal quantum outputs. Source noise, noisy measurement
arguments, and sweep-bit controls remain unsupported. ``REPEAT`` is flattened.

Local extraction optimization
-----------------------------

An X-prepared, X-measured control interacting through CX/CZ implements a
controlled Pauli product. For a Hermitian product P, its readout has Kraus
operator

.. math::

   \langle m_X|C(P)|+\rangle = (I + (-1)^m P)/2.

The rewriter replaces that extraction interval by the corresponding MPP and
removes the preparation belonging to that lifetime. Z-prepared, Z-measured
CNOT targets implement the equivalent Z-product extraction. Signed readouts
become signed products. For example:

.. code-block:: text

   R 4                      MPP Z0*Z1
   CX 0 4 1 4      ->
   M 4

For multiple controls, grouping controlled factors into measurement order can
introduce CZ phases between controls when factors anticommute. The optimizer
tracks their parity and applies the replacement only when they cancel. Factors
on the same control/data pair must share an axis. Independent data Cliffords,
other local basis wrappers, Y-readout gadgets, or non-cancelling phases keep
their original gate implementation. These are limits of the optional local
optimization, not restrictions on wire termination or circuit import.

For example, ``RX 2; CZ 2 0; S 2; MX 2`` keeps its gates and its direct
readout. There is no pulled MPP followed by a duplicate extraction body, and
no runtime ``flow_generators`` or ``has_flow`` certificate. The tests compare
record channels against an independent eager measure/reset reference instead.

Preparations are tracked by their instruction site and lifetime. Removing one
extraction's preparation cannot remove another use of the same Stim ID. A
later reset-only output also remains. ``eliminated_qubits`` contains an ID
only when no quantum instruction on any of its lifetimes remains; its
``QUBIT_COORDS`` annotation is then removed as well. Known deterministic direct
readouts remain physical measurements rather than constant ``MPAD`` records.

Records and graph layers
------------------------

In a mixed readout, extracted syndrome products precede the remaining direct
data readouts. The original measurements on different source qubits commute.
The rewriter records the permutation and updates every detector, observable,
and feedback reference. ``MPAD`` records occupy their original source-record
positions in this map even though they have no ``CheckMapping`` entry.

``result.foliation_record_to_source[j]`` is the original record index for
emitted record j. Each ``CheckMapping`` contains the emitted record index,
the signed observable at its emitted circuit position, and the original
direct-readout qubit ID (or ``None`` for a source product measurement).

Source ``TICK`` boundaries remain. Removing a preparation preserves a round
boundary after earlier measurements. Repeated unsigned supports and
anticommuting products start distinct MPP layers; distinct commuting products
can share a layer. Pair measurements are normalized to MPPs before this check.

Maximum graph degree four is a regression condition for the repository's
15-to-1 Clifford-proxy fixture only. It is not a requirement on logical Y,
surface-code generators in general, or arbitrary Clifford circuits. The
fixture includes a mixed data/syndrome readout; full-factory validation also
checks detector and observable correspondence.

Result and migration
--------------------

``result.circuit`` is the single rewritten circuit. ``result.foliation_circuit``
is a compatibility name referring to that same object. Similarly, ``checks``
and ``foliation_checks`` refer to the same tuple in emitted record order.
Neither name selects a different quantum-output contract. Callers comparing
sample columns must apply ``foliation_record_to_source``; consumers of the
rewritten detector, observable, and feedback annotations already have updated
references.

.. code-block:: python

   from graphqomb.stim_glue import rewrite_to_mpp, stim_circuit_to_pattern

   result = rewrite_to_mpp("R 4\nCX 0 4 1 4\nMR 4\nCX 0 4 1 4\nM 4")
   imported = stim_circuit_to_pattern(result.circuit)
   assert result.circuit is result.foliation_circuit
   assert result.eliminated_qubits == (4,)

API reference
-------------

.. automodule:: graphqomb.stim_glue.mpp_rewriter
   :members:
   :undoc-members:
   :show-inheritance:
