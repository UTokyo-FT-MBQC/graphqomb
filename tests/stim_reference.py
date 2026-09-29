"""Independent reference semantics for photonic Stim import tests."""

from __future__ import annotations

import stim


def photonic_reference(source: stim.Circuit) -> stim.Circuit:
    """Expand every direct readout to an eager measure/reset pair.

    Unlike production's lazy preparation and wire splitting, the reference
    resets immediately, even at terminal readouts. Terminal states are traced
    out in record-channel comparisons. Later explicit resets override the
    eager reset, so the observable behavior is the same.

    Returns
    -------
    stim.Circuit
        Ordinary Stim circuit implementing destructive-readout semantics.
    """
    measurements = {
        "M": ("M", "R"),
        "MX": ("MX", "RX"),
        "MY": ("MY", "RY"),
        "MR": ("M", "R"),
        "MRX": ("MX", "RX"),
        "MRY": ("MY", "RY"),
    }
    result = stim.Circuit()
    for instruction in source.flattened():
        if isinstance(instruction, stim.CircuitInstruction) and instruction.name in measurements:
            readout, reset = measurements[instruction.name]
            for target in instruction.targets_copy():
                result.append(readout, [target], instruction.gate_args_copy(), tag=instruction.tag)
                result.append(reset, [target.value])
        else:
            result.append(instruction)
    return result


def assert_record_channel(source: stim.Circuit, rewritten: stim.Circuit, permutation: tuple[int, ...]) -> None:
    """Compare complete signed record channels for arbitrary input states."""
    left, right = photonic_reference(source), rewritten.copy()
    width = max(source.num_qubits, rewritten.num_qubits)
    left.append("R", range(width))
    right.append("R", range(width))
    assert left.num_measurements == right.num_measurements
    assert sorted(permutation) == list(range(source.num_measurements))
    inverse = {old: new for new, old in enumerate(permutation)}
    for a, b, mapping in ((left, right, inverse), (right, left, dict(enumerate(permutation)))):
        for flow in a.flow_generators():
            output = flow.output_copy()
            # Stim 1.16's canonical generators may omit MPAD 1's sign.
            if not a.has_flow(flow):
                output = -output
                assert a.has_flow(
                    stim.Flow(input=flow.input_copy(), output=output, measurements=flow.measurements_copy())
                )
            mapped = stim.Flow(
                input=flow.input_copy(), output=output, measurements=[mapping[m] for m in flow.measurements_copy()]
            )
            assert b.has_flow(mapped), (source, rewritten, flow, mapped)
