Pattern Text Format
===================

GraphQOMB reads format versions 1 through 5 and writes the smallest version
required by a pattern:

* Version 2 is the base format and adds per-input Pauli initialization.
* Version 3 adds tagged detectors: ``.detector[tag] 0 1``.
* Version 4 adds input initialization tags: ``.input_tag[tag] 0``.
* Version 5 adds conditional Clifford corrections: ``.cflow 0 -> 1:S``.

Patterns without tags or Clifford corrections continue to use version 2.
Older readers must support the version declared by the file before loading it.

Input initialization
--------------------

Version 2 adds optional per-input positive Pauli eigenstate initialization:

.. code-block:: text

   .version 2
   .input 0:0 1:1 2:2
   .input_basis 1:Y 2:Z

Each ``.input_basis`` entry has the form ``node:X``, ``node:Y``, or ``node:Z``
and must reference a node declared by ``.input``. ``X`` initialization is the
default and is omitted when serializing, so the example initializes nodes 0,
1, and 2 as ``X+``, ``Y+``, and ``Z+`` respectively.

Version 1 files remain readable. Inputs in a version 1 file, or in a version 2
file without a corresponding ``.input_basis`` entry, are initialized as
``X+``. The ``.input_basis`` directive is rejected in version 1 files.

Clifford feedforward
--------------------

Version 5 stores the normalized Clifford cosets alongside ``.xflow`` and
``.zflow``. A ``.cflow`` entry maps a measured source node to target/coset pairs:

.. code-block:: text

   .cflow 0 -> 2:S 3:H

The accepted coset names are ``S``, ``H``, ``SH``, ``HS``, and ``HSH``.
Pauli factors are stored in the X/Z maps. These directives describe correcting
gates controlled by source outcome one, rather than the live residual frame.
Within each source, correction order is X, Z, then the coset; source events
follow causal measurement order. See :doc:`pauli_frame` for the residual
convention and restrictions on noncommuting corrections.

Loading reconstructs a ``CliffordFrame`` when nontrivial cosets are present,
and a ``PauliFrame`` otherwise. The runtime X/Z bits and coset accumulators are
initialized anew; serialization preserves the program, not a simulation state.

:mod:`graphqomb.ptn_format` module
++++++++++++++++++++++++++++++++++

.. automodule:: graphqomb.ptn_format
   :members:
   :member-order: bysource
