Stim Clifford parser
====================

Install the optional Stim integration before importing this module:

.. code-block:: console

   uv add "graphqomb[stim]"

The parser normalizes Stim Clifford circuits to GraphQOMB's unitary basis:
the four Clifford ``J(angle)`` gates plus ``CZ``. Each single-qubit basis
gate is one :class:`graphqomb.gates.J` primitive, i.e. one XY-plane Pauli
measurement in MBQC (gate names use algebraic matrix order):

.. list-table::
   :header-rows: 1

   * - Basis gate
     - Definition
     - Measurement
     - Stim name
   * - ``H``
     - ``J(0)``
     - X+
     - ``H``
   * - ``HS``
     - ``H S = J(pi/2)``
     - Y+
     - ``C_XNYZ``
   * - ``HZ``
     - ``H Z = J(pi)``
     - X-
     - ``SQRT_Y``
   * - ``HS_DAG``
     - ``H S† = J(-pi/2)``
     - Y-
     - ``C_XYZ``

.. code-block:: python

   from graphqomb.stim_glue.transpiler import transpile

   normalized = transpile("S_DAG 0\nCX 0 1", optimize=True)

The parser supports all fixed one- and two-qubit Clifford gates exposed by
Stim, arbitrary-length ``SPP`` and ``SPP_DAG`` rotations, nested ``REPEAT``
blocks, Pauli reset and measurement boundaries, and coordinate/TICK
annotations. ``DETECTOR``, ``OBSERVABLE_INCLUDE``, and ``MPAD`` are preserved
verbatim as optimization barriers; the standalone parser does not interpret
their measurement-record targets. Noise, non-Clifford instructions,
measurement-reset instructions, MPP measurements, and classically controlled
targets are rejected.

Hermitian ``SPP`` products may contain the same qubit more than once. Their
ordered Pauli factors and target inversions retain Stim's product semantics.

With ``optimize=True``, every maximal single-qubit gate run is replaced by
the shortest equivalent word over the four J gates, so any single-qubit
Clifford costs at most three J primitives (three graph nodes in the compiled
pattern). Optimization repeatedly scans a barrier-free block, so very large
circuits without ``TICK`` or annotation barriers can take superlinear time.
The importer normalizes each unitary ``TICK`` block independently, so
TICK-partitioned inputs bound this cost.

The Stim importer resolves the preserved detector and observable records,
handles MPP operations, and applies GraphQOMB's circuit-level noise policy.

API reference
-------------

.. automodule:: graphqomb.stim_glue.transpiler
   :members:
   :show-inheritance:
