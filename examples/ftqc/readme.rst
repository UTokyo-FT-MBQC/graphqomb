Fault-tolerant quantum computing
================================

This example provides an encoded T-injection pattern and reference checks
for its quantum output, detectors, and logical readout.

``steane_t_injection.py`` applies a logical T gate to a ``[[7,1,3]]`` Steane
block using an encoded magic state, logical ZZ measurement, syndrome rounds,
and conditional transversal S correction.

The script uses gallery-style cells and exports its ``.ptn`` patterns to
the gitignored ``output/`` directory beside it.
See ``README.md`` in this directory for the equations and validation scope.
