Stim MPP rewriter (experimental)
================================

Install the optional Stim integration before importing this module:

.. code-block:: console

   uv add "graphqomb[stim]"

The rewriter recognizes the standard syndrome-extraction shape — reset
ancillas, apply Clifford unitaries, measure ancillas — and replaces each
gate-level extraction block with ``MPP`` instructions measuring the inferred
data Pauli products. The inference conjugates each measurement Pauli backwards
through the block's Clifford body (``stim.PauliString.before``) and
substitutes ``+1`` for stabilizers of freshly initialized, measured-out
ancillas.

Each source measurement maps to exactly one measurement record in the
rewritten circuit, so ``DETECTOR`` and ``OBSERVABLE_INCLUDE`` annotations are
copied verbatim and keep their relative record indices. ``MR`` splits into an
``MPP`` product followed by a reset. ``REPEAT`` blocks are flattened first,
which bakes ``SHIFT_COORDS`` offsets into detector coordinates.

A segment ends when a unitary follows its measurements, or when a measurement
follows a reset issued after those measurements (including the implicit reset
of ``MR``). Trailing data measurements — such as the final transversal
readout of ``stim.Circuit.generated`` memory circuits — therefore start a
fresh segment with an empty Clifford body and pass through verbatim instead
of dragging reset-ancilla Pauli factors into their products.

Existing ``MPP`` products pass through verbatim, including valid repeated
factors such as ``X0*X1*X0``. Their sidecar mapping contains the reduced Pauli
product. When initialized-ancilla substitution reduces an inferred product to
``+I`` or ``-I``, the rewrite emits ``MPAD 0`` or ``MPAD 1`` respectively so
the deterministic measurement record is preserved. Inverted source targets
such as ``M !4`` produce signed ``MPP`` products.

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
supported, and optimized rewrites leave measured-out ancillas in their reset
state rather than the measurement outcome's eigenstate. The optimized path
preserves any residual Clifford frame on surviving qubits. It discards the
original gate schedule, so hook-fault analysis must rely on the returned
:class:`graphqomb.stim_mpp_rewriter.CheckMapping` sidecar and the source
circuit.

Every rewritten segment is verified against its source segment by
cross-checking stabilizer-flow generators (with resets appended to measured-out
ancillas in both copies). If initialized-ancilla substitution cannot preserve
the segment, the rewriter retries with the exact pulled-back products. If the
result still cannot be represented and verified as ``MPP`` measurements plus a
residual Clifford frame, it returns the original gate-level circuit with each
reset lifetime assigned a fresh Stim qubit id. This fallback preserves gate
order and measurement-record annotations while avoiding importer ambiguity
from reusing the same Stim id after reset.

The cross-check is not an optional assertion: it is the decision procedure that
picks between the optimized rewrite, the unsubstituted retry, and the
gate-level fallback, so it always runs.

API reference
-------------

.. automodule:: graphqomb.stim_mpp_rewriter
   :members:
   :show-inheritance:
