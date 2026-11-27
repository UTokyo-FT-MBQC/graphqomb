"""Fault distance of a noisy Stim circuit, from Stim alone."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import stim

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

    Level = tuple[NDArray[np.uint64], NDArray[np.uint64], NDArray[np.intp]]

# Sets of at most two mechanisms are enumerated, which decides weights up to four.
_MAX_EXHAUSTIVE_WEIGHT = 4
_MAX_OBSERVABLES = 64
_HASH_SEED = 2603


def drop_detectors(circuit: stim.Circuit, tag: str) -> stim.Circuit:
    """Remove the detectors carrying an instruction tag, as if they were not read.

    Returns
    -------
    `stim.Circuit`
        Flattened circuit without the tagged detectors.
    """
    kept = stim.Circuit()
    for instruction in circuit.flattened():
        if instruction.name != "DETECTOR" or instruction.tag != tag:
            kept.append(instruction)
    return kept


def fault_distance_bounds(circuit: stim.Circuit, *, exhaustive_weight: int = _MAX_EXHAUSTIVE_WEIGHT) -> tuple[int, int]:
    r"""Bound the fewest error mechanisms that flip an observable without a detection event.

    Error mechanisms are those of the circuit's detector error model, without
    decomposition into graphlike parts. Every set of at most exhaustive_weight
    mechanisms is examined by matching the detection events of two smaller
    sets, so a logical error of that weight is found with certainty. When none
    exists, the lower bound is ``exhaustive_weight + 1`` and the upper bound
    comes from ``stim.Circuit.search_for_undetectable_logical_errors``, which
    returns an actual logical error but not necessarily a smallest one.

    Returns
    -------
    `tuple`\[`int`, `int`\]
        Lower and upper bound; they agree when the distance is determined.

    Raises
    ------
    ValueError
        If exhaustive_weight is outside 0 to 4, or the circuit has more than 64 observables.
    """
    if not 0 <= exhaustive_weight <= _MAX_EXHAUSTIVE_WEIGHT:
        msg = f"exhaustive_weight must be between 0 and {_MAX_EXHAUSTIVE_WEIGHT}."
        raise ValueError(msg)
    if circuit.num_observables > _MAX_OBSERVABLES:
        msg = f"At most {_MAX_OBSERVABLES} observables are supported."
        raise ValueError(msg)
    mechanisms = error_mechanisms(circuit)
    levels = _levels(mechanisms, circuit.num_detectors)
    for weight in range(1, exhaustive_weight + 1):
        # Pairs are needed from weight three on; a lighter logical error returns before they are built.
        while len(levels) <= (weight + 1) // 2:
            levels.append(_pairs(levels[1]))
        if _has_logical_error(mechanisms, levels[(weight + 1) // 2], levels[weight // 2]):
            return weight, weight
    found = circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=6,
        dont_explore_edges_with_degree_above=6,
        dont_explore_edges_increasing_symptom_degree=True,
    )
    return exhaustive_weight + 1, len(found)


def error_mechanisms(circuit: stim.Circuit) -> list[tuple[frozenset[int], int]]:
    r"""List the distinct error mechanisms as detector sets and observable bit masks.

    Returns
    -------
    `list`\[`tuple`\[`frozenset`\[`int`\], `int`\]\]
        Mechanisms that flip at least one detector or observable.
    """
    distinct: set[tuple[frozenset[int], int]] = set()
    for instruction in circuit.detector_error_model().flattened():
        if not isinstance(instruction, stim.DemInstruction) or instruction.type != "error":
            continue
        detectors: set[int] = set()
        observables = 0
        for target in instruction.targets_copy():
            # A repeated target cancels, as in a decomposed error.
            if isinstance(target, stim.DemTarget) and target.is_relative_detector_id():
                detectors ^= {target.val}
            elif isinstance(target, stim.DemTarget) and target.is_logical_observable_id():
                observables ^= 1 << target.val
        if detectors or observables:
            distinct.add((frozenset(detectors), observables))
    return sorted(distinct, key=lambda mechanism: (sorted(mechanism[0]), mechanism[1]))


def _levels(mechanisms: list[tuple[frozenset[int], int]], num_detectors: int) -> list[Level]:
    r"""Hash the detection events of the empty set and of every single mechanism.

    Returns
    -------
    `list`\[`Level`\]
        Per set size, the hashed detection events, the observable masks, and the members of each set.
    """
    keys = np.random.default_rng(_HASH_SEED).integers(1, 2**63, size=max(num_detectors, 1), dtype=np.uint64)
    events = np.zeros(len(mechanisms), dtype=np.uint64)
    for index, (detectors, _) in enumerate(mechanisms):
        events[index] = np.bitwise_xor.reduce(keys[sorted(detectors)], initial=np.uint64(0))
    observables = np.array([mask for _, mask in mechanisms], dtype=np.uint64)
    return [
        (np.zeros(1, dtype=np.uint64), np.zeros(1, dtype=np.uint64), np.empty((1, 0), dtype=np.intp)),
        (events, observables, np.arange(len(mechanisms), dtype=np.intp)[:, np.newaxis]),
    ]


def _pairs(singles: Level) -> Level:
    """Combine every two distinct mechanisms.

    Returns
    -------
    `Level`
        Hashed detection events, observable masks, and members of each pair.
    """
    events, observables, _ = singles
    first, second = np.triu_indices(events.size, k=1)
    return events[first] ^ events[second], observables[first] ^ observables[second], np.stack([first, second], axis=1)


def _has_logical_error(mechanisms: list[tuple[frozenset[int], int]], left: Level, right: Level) -> bool:
    """Find a left set and a right set with equal detection events and different observable flips.

    Equal detection events always hash equally, so no logical error is missed.
    A hash match is confirmed on the detector sets before it is accepted.

    Returns
    -------
    `bool`
        Whether the union of some left set and some right set is an undetectable logical error.
    """
    events, observables, members = left
    # Sort the right sets by detection events, then by observable mask.
    order = np.lexsort((right[1], right[0]))
    right_events, right_observables = right[0][order], right[1][order]
    start = np.searchsorted(right_events, events, side="left")
    stop = np.searchsorted(right_events, events, side="right")
    matched = np.flatnonzero(start < stop)
    # A group holds a different mask exactly if its first or its last mask differs.
    differs = (right_observables[start[matched]] != observables[matched]) | (
        right_observables[stop[matched] - 1] != observables[matched]
    )
    for candidate in matched[differs]:
        for position in range(int(start[candidate]), int(stop[candidate])):
            if right_observables[position] != observables[candidate] and _cancels(
                mechanisms, (*members[candidate], *right[2][order[position]])
            ):
                return True
    return False


def _cancels(mechanisms: list[tuple[frozenset[int], int]], members: Iterable[np.intp]) -> bool:
    """Check on the detector sets that the members leave no detection event.

    Returns
    -------
    `bool`
        Whether every detector is flipped an even number of times.
    """
    remaining: set[int] = set()
    for member in members:
        remaining ^= mechanisms[int(member)][0]
    return not remaining
