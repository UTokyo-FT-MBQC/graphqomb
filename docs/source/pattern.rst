Pattern
=======

:mod:`graphqomb.pattern` module
++++++++++++++++++++++++++++++++

.. automodule:: graphqomb.pattern

Pattern Classes
---------------

The correction frame is available as ``pattern.clifford_frame`` and passed to
the constructor as ``Pattern(..., clifford_frame=frame)``. Both replace the
former ``pauli_frame`` name. The frame tracks Pauli and Clifford corrections.

.. autoclass:: graphqomb.pattern.Pattern
    :members:
    :member-order: bysource

Helper Functions
----------------

.. autofunction:: graphqomb.pattern.is_runnable

.. autofunction:: graphqomb.pattern.print_pattern
