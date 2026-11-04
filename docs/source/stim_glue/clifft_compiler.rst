Clifft Compiler
===============

Install the optional Clifft integration before using this module (Clifft
requires Python 3.12 or later):

.. code-block:: console

   uv add "graphqomb[clifft]"

:func:`graphqomb.stim_glue.clifft_compiler.clifft_compile` emits Stim syntax
extended with Clifft's non-Clifford gates. Unlike
:func:`graphqomb.stim_glue.compiler.stim_compile`, it accepts patterns with
Clifford feedforward (cflow) and non-stabilizer inputs such as
:math:`|T\rangle`. Pass the returned text to ``clifft.compile`` and sample it
with ``clifft.sample``.

Clifft does not accept instruction tags on detectors or resets, so they are
omitted. Detector ``i`` corresponds to
``pattern.frame.parity_check_tags[i]``; for example, build a Clifft
post-selection mask with
``[int(tag == FLAG_DETECTOR_TAG) for tag in pattern.frame.parity_check_tags]``.

Online decoding of Clifford feedforward
---------------------------------------

In fault-tolerant execution, a conditional Clifford is decided by a value
decoded online from the records measured before it, not by the raw record of
its cflow source: a wrong Clifford cannot be repaired by offline Pauli
decoding. Clifft cannot call a decoder during sampling, so sample each branch
separately:

1. For each value ``c`` of the source, compile with
   ``fixed_controls={source: c}``. Every correction event of the source is
   then applied unconditionally (``c=True``) or omitted (``c=False``), while
   the source is still measured and its record stays in the detectors.
2. Sample the same number of shots from each branch circuit.
3. Decode the control value from the records preceding the correction and
   keep the shots whose decoded value equals ``c``.

The branch circuits agree before the correction, so each branch keeps the
fraction of shots that the decoder sends into it, and the kept shots of all
branches together follow the online-decoded execution. This is exact
sampling, not physical post-selection; with ``k`` fixed sources it needs
``2**k`` circuits and keeps about ``2**-k`` of the shots. Decode the kept
shots of each branch with a detector error model of that branch, for example
from the Clifford circuit obtained by replacing non-stabilizer inputs with
stabilizer inputs that make the observable deterministic.

.. automodule:: graphqomb.stim_glue.clifft_compiler
   :members:
   :show-inheritance:
