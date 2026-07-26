Stim MPP rewriter (experimental)
================================

Install the optional Stim integration before importing this module:

.. code-block:: console

   uv add "graphqomb[stim]"

The rewriter recognizes the standard syndrome-extraction shape — reset
ancillas, apply Clifford unitaries, measure ancillas — and replaces each
gate-level extraction block with ``MPP`` instructions measuring the inferred
data Pauli products. The inference conjugates each measurement Pauli backwards
through the block's Clifford body (:meth:`stim.PauliString.before`) and
substitutes ``+1`` for stabilizers of freshly initialized, measured-out
ancillas.

Each source measurement maps to exactly one measurement record in the
rewritten circuit, so ``DETECTOR`` and ``OBSERVABLE_INCLUDE`` annotations are
copied verbatim and keep their relative record indices. ``MR`` splits into an
``MPP`` product followed by a reset. ``REPEAT`` blocks are flattened first,
which bakes ``SHIFT_COORDS`` offsets into detector coordinates.

Existing ``MPP`` products pass through verbatim, including valid repeated
factors such as ``X0*X1*X0``. Their sidecar mapping contains the reduced Pauli
product. When initialized-ancilla substitution reduces an inferred product to
``+I`` or ``-I``, the rewrite emits ``MPAD 0`` or ``MPAD 1`` respectively so
the deterministic measurement record is preserved.

.. code-block:: python

   from graphqomb.stim_mpp_rewriter import rewrite_to_mpp

   result = rewrite_to_mpp(
       """
       R 4
       H 4
       CX 4 0 4 1 4 2 4 3
       H 4
       M 4
       """
   )
   assert str(result.circuit).strip() == "R 4\nMPP X0*X1*X2*X3"

The rewrite is structural: circuit-level noise instructions are rejected
instead of being folded into ``MPP`` arguments, classical feedback is not
supported, and measured-out ancillas are left in their reset state rather
than the measurement outcome's eigenstate. The original gate schedule is
discarded, so hook-fault analysis must rely on the returned
:class:`graphqomb.stim_mpp_rewriter.CheckMapping` sidecar and the source
circuit.

By default every rewritten segment is verified against its source segment by
cross-checking stabilizer-flow generators (with resets appended to measured-out
ancillas in both copies), so residual data-qubit unitaries or unsupported
flag-style structures raise
:class:`graphqomb.stim_mpp_rewriter.MppRewriteVerificationError` instead of
silently changing the circuit's semantics. Pass ``verify=False`` to skip this
check.

API reference
-------------

.. automodule:: graphqomb.stim_mpp_rewriter
   :members:
   :show-inheritance:
