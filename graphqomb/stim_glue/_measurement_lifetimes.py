"""Normalize destructive single-qubit measurements and independent preparations."""

from __future__ import annotations

import stim

from graphqomb.stim_glue._parse import (
    ANNOTATION_GATES,
    DIRECT_MEASUREMENT_AXES,
    MEASURE_RESET_AXES,
    RESET_AXES,
    RESET_GATES,
    SINGLE_MEASUREMENT_AXES,
    iter_instructions,
)

_MEASUREMENT_GATES = {axis: name for name, axis in SINGLE_MEASUREMENT_AXES.items()}


def normalize_measurement_lifetimes(circuit: stim.Circuit) -> stim.Circuit:
    """End each measured wire and prepare its next lifetime only when used.

    Plain and measure-reset readouts both consume their target. A later
    quantum operation starts in the positive eigenstate of the measured axis,
    independently of the result. An explicit reset supplies its own axis.
    Preparations are delayed until that next use; terminal readouts therefore
    never create an unused continuation. Record order and readout tags survive.

    Returns
    -------
    stim.Circuit
        Flattened circuit with only plain readouts and explicit preparations.
        Applying this normalization again leaves the circuit unchanged.
    """
    result = stim.Circuit()
    pending: dict[int, tuple[str, str]] = {}
    for instruction in iter_instructions(circuit.flattened()):
        name = instruction.name
        if name in ANNOTATION_GATES or name == "MPAD":
            result.append(instruction)
            continue
        targets = instruction.targets_copy()
        if name in RESET_AXES:
            for target in targets:
                pending.pop(target.value, None)
            result.append(instruction)
            continue
        # Handle readouts sequentially, including repeated targets. Stim
        # coalesces adjacent distinct targets back into one readout instruction.
        groups = [[target] for target in targets] if name in DIRECT_MEASUREMENT_AXES else [targets]
        for group in groups:
            for target in group:
                qubit = target.qubit_value
                if qubit is not None and int(qubit) in pending:
                    reset, tag = pending.pop(int(qubit))
                    result.append(reset, [int(qubit)], tag=tag)
            if name in DIRECT_MEASUREMENT_AXES:
                axis = DIRECT_MEASUREMENT_AXES[name]
                result.append(_MEASUREMENT_GATES[axis], group, instruction.gate_args_copy(), tag=instruction.tag)
                pending[group[0].value] = (RESET_GATES[axis], instruction.tag if name in MEASURE_RESET_AXES else "")
            else:
                result.append(name, group, instruction.gate_args_copy(), tag=instruction.tag)
    return result
