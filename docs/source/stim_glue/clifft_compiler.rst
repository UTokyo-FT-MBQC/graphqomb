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

Clifford+T import
-----------------

The :mod:`graphqomb.stim_glue.clifft_importer` frontend accepts the existing
Stim-import subset extended by ``T`` and ``T_DAG``. It offers two lowerings:

* ``t_mode="measurement"`` uses XY-plane measurements at
  :math:`\pm\pi/4`, whose observables :math:`(X\pm Y)/\sqrt{2}` are Hermitian
  Cliffords. These are non-Pauli measurements. At export, ``clifft_compile``
  rotates by the negative measurement angle using T/T_DAG and measures X.
* ``t_mode="injection"`` uses Pauli measurements, scheduled magic resources,
  ZZ probes, and conditional S/S-dagger corrections. Preparation gates are
  returned separately and must be passed to ``clifft_compile``.

.. code-block:: python

   import clifft
   from graphqomb.stim_glue import clifft_file_to_pattern, clifft_compile

   imported = clifft_file_to_pattern("cultivation.stim", t_mode="injection")
   text = clifft_compile(
       imported.pattern, preparation_gates=imported.preparation_gates
   )
   sample = clifft.sample(clifft.compile(text), shots=4096, seed=17)

Write each T instruction on its own line, including inside REPEAT blocks.
Other Clifft extensions, such as arbitrary rotations, are not imported.
Source noise is removed by import. MBQC noise must be supplied separately
through ``noise_models``. T-basis readout rotations and controlled-Clifford
correction gadgets are ideal. Scheduled magic preparation gates run after
the reset and before any preparation noise. The injection scheduler orders
each ZZ probe before its magic X readout to avoid consuming all resources
at circuit start; this ordering does not change the ideal instrument.
For a T-basis measurement, ``MeasureEvent.axis`` denotes the physical X
readout. BEFORE measurement noise acts before the ideal basis rotation;
measurement-flip noise acts on the resulting X record.

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

.. automodule:: graphqomb.stim_glue.clifft_importer
   :members:
   :show-inheritance:
