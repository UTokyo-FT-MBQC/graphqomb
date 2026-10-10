Pattern
=======

:mod:`graphqomb.pattern` module
++++++++++++++++++++++++++++++++

.. automodule:: graphqomb.pattern

Pattern Classes
---------------

The correction frame is available as ``pattern.clifford_frame`` or
``pattern.pauli_frame``. Both names refer to the same instance, including when
it is a :class:`graphqomb.pauli_frame.CliffordFrame`.

Construction accepts ``Pattern(..., clifford_frame=frame)`` or the
backwards-compatible ``Pattern(..., pauli_frame=frame)`` without deprecation
warnings. The fourth positional argument is also the frame; the following
positional arguments remain input coordinates and input initializations.
Omitting the frame or supplying both frame keywords raises ``TypeError``.

The stored dataclass field is ``clifford_frame``; ``pauli_frame`` is a read-only
property. Use ``dataclasses.replace(pattern, clifford_frame=frame)`` to replace
the frame. The pattern remains a frozen dataclass.

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
