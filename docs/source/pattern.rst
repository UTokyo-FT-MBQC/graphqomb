Pattern
=======

:mod:`graphqomb.pattern` module
++++++++++++++++++++++++++++++++

.. automodule:: graphqomb.pattern

Pattern Classes
---------------

The correction frame is available as ``pattern.frame``. It is a
:class:`graphqomb.pauli_frame.PauliFrame`, optionally extended to a
:class:`graphqomb.pauli_frame.CliffordFrame` for Clifford feedforward.

Construct a pattern with ``Pattern(..., frame=frame)``. The fourth positional
argument is also the frame; the following positional arguments remain input
coordinates and input initializations. Omitting the frame or supplying multiple
frame arguments raises ``TypeError``.

The stored dataclass field is ``frame``. Use
``dataclasses.replace(pattern, frame=frame)`` to replace it. The pattern remains
a frozen dataclass.

For compatibility, ``Pattern(..., pauli_frame=frame)`` and
``Pattern(..., clifford_frame=frame)`` are accepted without deprecation warnings.
Both ``pattern.pauli_frame`` and ``pattern.clifford_frame`` are read-only
aliases of ``pattern.frame`` and return the complete frame, including for
Clifford patterns. These aliases are not dataclass fields; use ``frame`` for
``dataclasses.replace`` and dataclass field inspection.

:func:`graphqomb.qompiler.qompile` and ``.ptn`` loading select
:class:`graphqomb.pauli_frame.PauliFrame` for Pauli-only normalized corrections
and :class:`graphqomb.pauli_frame.CliffordFrame` for nontrivial Clifford cosets.

.. autoclass:: graphqomb.pattern.Pattern
    :members:
    :member-order: bysource

Helper Functions
----------------

.. autofunction:: graphqomb.pattern.is_runnable

.. autofunction:: graphqomb.pattern.print_pattern
