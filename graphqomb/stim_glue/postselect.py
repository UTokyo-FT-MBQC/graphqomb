"""Post-selection helpers for flag detectors.

A detector tagged ``type=flag`` (``DETECTOR[type=flag] ...``) is a flag
detector: a sample in which it fires is meant to be rejected. GraphQOMB
carries the tag through import (`graphqomb.stim_glue.importer`), the compiled
pattern (`graphqomb.pauli_frame.PauliFrame.parity_check_tags`), and Stim
export (`graphqomb.stim_glue.compiler.stim_compile`), so the helpers here work
on the original circuit and on the compiled circuit alike.

This module provides:

- `FLAG_DETECTOR_TAG`: The Stim instruction tag marking a flag detector.
- `flag_detector_indices`: Indices of flag detectors in a Stim circuit.
- `flag_postselection_mask`: Bit-packed sinter post-selection mask.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt
    import stim

FLAG_DETECTOR_TAG = "type=flag"
"""Stim instruction tag marking a post-selection flag detector."""


def flag_detector_indices(circuit: stim.Circuit) -> tuple[int, ...]:
    r"""Return the detector indices tagged ``type=flag`` in a Stim circuit.

    Repeat blocks are flattened first, so an instruction inside a
    ``REPEAT`` block contributes one detector index per iteration.

    Returns
    -------
    `tuple`\[`int`, ...\]
        Ascending indices of flag detectors, counting every ``DETECTOR``
        instruction in circuit order.
    """
    indices: list[int] = []
    detector_index = 0
    for instruction in circuit.flattened():
        if instruction.name != "DETECTOR":
            continue
        if instruction.tag == FLAG_DETECTOR_TAG:
            indices.append(detector_index)
        detector_index += 1
    return tuple(indices)


def flag_postselection_mask(circuit: stim.Circuit) -> npt.NDArray[np.uint8]:
    r"""Return a bit-packed post-selection mask over the flag detectors of a circuit.

    Detector k is marked when bit ``mask[k // 8] & (1 << (k % 8))`` is
    set, matching the postselection_mask argument of ``sinter.Task``:
    samples in which a marked detector fires are discarded.

    Returns
    -------
    `numpy.typing.NDArray`\[`numpy.uint8`\]
        Mask of ``ceil(num_detectors / 8)`` bytes with the bits of
        ``type=flag`` detectors set.

    Examples
    --------
    >>> import stim
    >>> from graphqomb.stim_glue import flag_postselection_mask
    >>> circuit = stim.Circuit('''
    ...     RX 0 1
    ...     MX 0 1
    ...     DETECTOR rec[-2]
    ...     DETECTOR[type=flag] rec[-1]
    ... ''')
    >>> flag_postselection_mask(circuit)
    array([2], dtype=uint8)
    """
    mask = np.zeros(math.ceil(circuit.num_detectors / 8), dtype=np.uint8)
    for index in flag_detector_indices(circuit):
        mask[index // 8] |= 1 << (index % 8)
    return mask
