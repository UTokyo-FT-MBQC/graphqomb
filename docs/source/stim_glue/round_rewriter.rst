Round-preserving syndrome extraction
====================================

Use this pass when the resource graph must retain successive data chains and
measurement layers. It accepts a Stim circuit alone. It does not need a code
tableau, logical boundary, coordinate convention, or fixed data-qubit set.

.. code-block:: python

    import stim
    from graphqomb.stim_glue import rewrite_syndrome_rounds, stim_circuit_to_pattern

    source = stim.Circuit.from_file("experiment.stim")
    result = rewrite_syndrome_rounds(source.without_noise())
    imported = stim_circuit_to_pattern(result.circuit)

Noise removal is an explicit caller decision. This pass preserves the ideal
instrument, not a circuit-level noise model or fault-distance guarantee.

Contract and boundary policy
----------------------------

Windows are defined by instruction boundaries after expanding ``REPEAT``.
``M``, ``MX``, and ``MY`` end a window and trigger an extraction attempt.
Other record-producing instructions, including ``MR`` variants, ``MPP``, pair
measurements, and ``MPAD``, flush the accumulated window unchanged. Resets,
``TICK``, and annotations do not themselves split a window; instructions left
at circuit end are copied unchanged. A multi-target measurement instruction
is one boundary. If a QEC round reads X and Z checks in separate instructions,
those instructions end separate windows, which can limit extraction coverage.
``rewritten_rounds`` counts rewritten windows, not inferred QEC rounds.

An extraction window ends at a single-qubit measurement instruction. Candidate
probe lifetimes have an explicit reset inside that window, are measured at its
end, and have no further quantum use before another reset or circuit end.
All other coordinates are treated as arbitrary quantum inputs to the window.
The policy permits discarding the candidate probes' terminal states; it does
not infer a user's intended logical subsystem from detector annotations.

An accepted rewrite preserves the outcome-conditioned quantum map on every
remaining coordinate, together with every measurement record. In particular:

* Data preparation and readout remain actual operations.
* No known initial data stabilizer, detector parity, or previous-round outcome
  is used to delete a later check measurement.
* The same Pauli check in two rounds remains two MPP measurements.
* Native MPP instructions are retained, including their existing layer boundaries.
* ``M`` remains nondemolition. There is no ``M`` to ``MR`` replacement.
* Record-controlled Pauli gates are retained, with record references remapped.
  The ordinary graph importer lowers them into ``xflow``/``zflow``.

An arbitrary logical-channel compiler still requires logical boundary metadata.
This restricted pass instead exposes a structural, local discard policy. A
quantum output on an eliminated probe is explicitly outside its contract.

Constructive deletion rule
--------------------------

The unitary body is factored algebraically as

.. math::

    U = L_A\,\Phi_A\,\left(\prod_a C_a(P_a)\right)V_D.

``V_D`` contains the retained data Clifford gates. ``L_A`` contains local probe
Cliffords. Each probe's interactions must share a single control axis in its
preparation coordinates. Moving data gates to ``V_D`` conjugates the accumulated
``P_a``. Exchanging controlled Paulis on different probes contributes a ``CZ``
to ``Phi_A`` when their data factors anticommute. Multiplying Pauli strings
retains their full complex phase, including minus signs.

The deletion rule applies when ``Phi_A`` cancels, each nontrivial ``P_a`` is
Hermitian, the final products commute, and each pulled-back readout matches
the prepared probe axis up to sign and is transverse to its control axis.
For each outcome its data Kraus operator is then the signed Pauli projector,
times the retained ``V_D``. The probe can be replaced by that MPP without any
channel-equivalence search. Probes are measured before any mixed data readout;
``record_map`` records the necessary permutation of source records. All
``DETECTOR``, ``OBSERVABLE_INCLUDE``, and feedback references are remapped.

The implementation does not call ``flow_generators`` or compare a candidate
channel against the source at runtime. Independent tests compare dense branch
Choi operators, including arbitrary data inputs, interleaved anticommuting
couplings, signed measurements, and residual quantum action.

Scope and retained blocks
-------------------------

Currently, probe interactions use ``CX``, ``CY``, or ``CZ`` with compatible
one-qubit Clifford frames. Probe-to-probe interactions, changing control axes,
uncancelled probe phases, trivial or non-Hermitian data products, and a
readout requiring a residual quantum correction retain the original block.
Classical control inside a candidate window also retains that block. This
does not prevent the ordinary importer from handling the retained feedback.
Clifford feedforward is outside the supported scope.

For example, ``RX 1; CZ 1 0; S 1; MX 1`` is retained: its random outcome also
leaves ``S_DAG`` and a conditional ``Z`` on data. It cannot be replaced by just
a random bit. Special logical-Y transition windows may similarly stay in
gate form while ordinary extraction windows become MPPs.

This pass changes the implementation of measurements, not necessarily the
total number of graph vertices. The graph backend's MPP foliation may require
more vertices than its direct Clifford-gate construction. Compare complete
graph construction, not just the number of removed probe lifetimes.

The implementation flattens repeats, uses packed Stim Pauli strings per probe,
and checks pairwise commutation within each window. Its storage includes the
flattened input and output; it is not a bounded-memory streaming interface.

Migration from the previous rewriter
-------------------------------------

The old ``rewrite_to_mpp`` API and its flow-generator-based contraction engine
have been removed. Use ``rewrite_syndrome_rounds(source).circuit`` as the
importer input. There is no ``foliation_circuit`` or global classical normal
form: the supported local rewrites retain the remaining quantum instrument,
and unsupported windows remain in gate form. If an application needs the
terminal states of candidate probes, import its original Stim circuit directly.

API
---

.. autofunction:: graphqomb.stim_glue.rewrite_syndrome_rounds

.. autoclass:: graphqomb.stim_glue.RoundRewriteResult
   :members:
