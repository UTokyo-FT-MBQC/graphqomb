"""Noisy Clifft sampling of Steane T teleportation with an online-decoded S control."""

from __future__ import annotations

import importlib
import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import stim

from graphqomb.command import M
from graphqomb.stim_glue.clifft_compiler import clifft_compile

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from graphqomb.noise_model import NoiseModel

    from .steane import SteanePattern


@dataclass(frozen=True)
class LogicalErrorEstimate:
    """Decoded logical failures among the shots following the executed protocol."""

    failures: int
    shots: int

    @property
    def rate(self) -> float:
        """Logical error rate."""
        return self.failures / self.shots

    @property
    def standard_error(self) -> float:
        """Binomial standard error of the rate."""
        return float(np.sqrt(self.rate * (1 - self.rate) / self.shots))


def _encode(nodes: Sequence[int]) -> list[str]:
    """Return the encoder after qubit 1 holds the logical state.

    Pivots 3, 5, 6 add the simplex rows to |0>; the logical-one coset
    contains {1, 2, 4}, disjoint from the pivots.

    Returns
    -------
    list[str]
        Noiseless Clifft instructions.
    """
    lines = [f"CX {nodes[1]} {nodes[2]} {nodes[1]} {nodes[4]}", f"H {nodes[3]} {nodes[5]} {nodes[6]}"]
    for pivot, others in ((3, (0, 1, 2)), (5, (0, 1, 4)), (6, (0, 2, 4))):
        lines.append("CX " + " ".join(f"{nodes[pivot]} {nodes[other]}" for other in others))
    return lines


def steane_encoder(nodes: Sequence[int], state_gates: Sequence[str]) -> str:
    """Encode the state that ``state_gates`` prepare from |0> on qubit 1.

    Returns
    -------
    str
        Noiseless Clifft instructions, starting with a reset of the block.
    """
    lines = [f"R {' '.join(map(str, nodes))}", *(f"{gate} {nodes[1]}" for gate in state_gates)]
    return "\n".join([*lines, *_encode(nodes)])


def t_input_circuit(steane: SteanePattern) -> str:
    """Encode data T^dagger|+> and the magic state T|+>; T teleportation gives |+>.

    Returns
    -------
    str
        Noiseless Clifft instructions; the logical X readout is deterministically 0.
    """
    return "\n".join(
        (
            steane_encoder(steane.data_input_nodes, ("H", "T_DAG")),
            steane_encoder(steane.magic_input_nodes, ("H", "T")),
        )
    )


def branch_input_circuit(steane: SteanePattern, value: bool) -> str:
    """Encode a Clifford input whose teleportation bit is ``value`` and X readout is fixed.

    Branch 0 uses (|00> + |11>)/sqrt(2) on (data, magic); the readout leaves
    magic in |+>. Branch 1 uses (|10> + i|01>)/sqrt(2); X then S map the
    remaining |+i> to |+>.

    Returns
    -------
    str
        Noiseless Stim instructions.
    """
    data, magic = steane.data_input_nodes, steane.magic_input_nodes
    lines = [f"R {' '.join(map(str, (*data, *magic)))}", f"H {magic[1]}", f"CX {magic[1]} {data[1]}"]
    if value:
        lines += [f"X {data[1]}", f"S {magic[1]}"]
    return "\n".join([*lines, *_encode(data), *_encode(magic)])


def branch_circuit(steane: SteanePattern, value: bool, noise_models: Sequence[NoiseModel]) -> stim.Circuit:
    """Build the Clifford circuit of one fixed branch, for its detector error model.

    Detector and observable flips of a Pauli fault depend only on the Clifford
    operations after it, so the error model also describes the T-input branch.

    Returns
    -------
    stim.Circuit
        The branch circuit with deterministic detectors and observable.
    """
    text = clifft_compile(
        steane.pattern,
        input_circuit=branch_input_circuit(steane, value),
        fixed_controls={steane.control_node: value},
        noise_models=noise_models,
    )
    return stim.Circuit(text)


def _measurement_columns(steane: SteanePattern) -> dict[int, int]:
    return {command.node: index for index, command in enumerate(c for c in steane.pattern if isinstance(c, M))}


def decode_teleportation_bit(steane: SteanePattern, measurements: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Decode the teleportation bit by a Hamming lookup on the destructive readout syndrome.

    Each consumed-Z detector without its earlier Z-check records is the
    syndrome bit of the readout word alone. The Z-check columns of q4, q5,
    q6 are (0,1,1), (0,1,0), (0,0,1), so a single flip lies on the logical
    support exactly when s1 = 0 and the syndrome is nonzero.

    Returns
    -------
    numpy.ndarray
        The decoded bit of each shot, using only records preceding the correction.
    """
    frame = steane.pattern.frame
    columns = _measurement_columns(steane)
    checks = set(steane.before_nodes) | set(steane.magic_before_nodes)
    syndrome = [
        np.bitwise_xor.reduce(measurements[:, [columns[node] for node in sorted(set(group) - checks)]], axis=1)
        for group, tag in zip(frame.parity_check_group, frame.parity_check_tags, strict=True)
        if tag.startswith("consumed_Z")
    ]
    flip = ~syndrome[0] & (syndrome[1] | syndrome[2])
    return np.asarray(measurements[:, columns[steane.control_node]] ^ flip, dtype=np.bool_)


def _syndrome_keys(detectors: NDArray[np.bool_]) -> list[int]:
    packed = np.packbits(detectors, axis=1, bitorder="little")
    return [int.from_bytes(row.tobytes(), "little") for row in packed]


@dataclass(frozen=True)
class ErrorMechanism:
    """One independent error of a detector error model."""

    probability: float
    detectors: frozenset[int]
    flips_observable: bool


def error_mechanisms(dem: stim.DetectorErrorModel) -> list[ErrorMechanism]:
    """List the errors of a single-observable detector error model.

    Returns
    -------
    list[ErrorMechanism]
        The errors with their detectors and observable flip.
    """
    mechanisms: list[ErrorMechanism] = []
    for instruction in dem.flattened():
        if not isinstance(instruction, stim.DemInstruction) or instruction.type != "error":
            continue
        detectors: set[int] = set()
        flip = False
        for target in instruction.targets_copy():
            if isinstance(target, stim.DemTarget) and target.is_relative_detector_id():
                detectors ^= {target.val}
            elif isinstance(target, stim.DemTarget) and target.is_logical_observable_id():
                flip = not flip
        mechanisms.append(ErrorMechanism(instruction.args_copy()[0], frozenset(detectors), flip))
    return mechanisms


@dataclass(frozen=True)
class LookupDecoder:
    """Most likely observable flip for each syndrome of at most two error mechanisms."""

    table: dict[int, bool]

    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel) -> LookupDecoder:
        """Tabulate the empty set, every mechanism, and every pair of mechanisms.

        Returns
        -------
        LookupDecoder
            The decoder; unlisted syndromes predict no flip.
        """
        mechanisms = [
            (mechanism.probability, sum(1 << index for index in mechanism.detectors), mechanism.flips_observable)
            for mechanism in error_mechanisms(dem)
        ]
        weights: defaultdict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
        weights[0][0] = 1.0
        for probability, syndrome, flip in mechanisms:
            weights[syndrome][flip] += probability
        for (p1, s1, f1), (p2, s2, f2) in itertools.combinations(mechanisms, 2):
            weights[s1 ^ s2][f1 ^ f2] += p1 * p2
        return cls({syndrome: weight[1] > weight[0] for syndrome, weight in weights.items()})

    def predict(self, detectors: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Predict the observable flip of each shot.

        Returns
        -------
        numpy.ndarray
            The predicted flip of each shot.
        """
        return np.asarray([self.table.get(key, False) for key in _syndrome_keys(detectors)], dtype=np.bool_)


def sample_logical_error(
    steane: SteanePattern,
    noise_models: Sequence[NoiseModel],
    shots: int,
    *,
    seed: int = 0,
    decoders: Mapping[bool, LookupDecoder] | None = None,
) -> LogicalErrorEstimate:
    """Sample decoded logical X failures of T teleportation with the S control decoded online.

    Each fixed-branch circuit samples ``shots`` shots and keeps those whose
    decoded teleportation bit selects that branch; each kept shot is decoded
    with the lookup decoder of its branch.

    Parameters
    ----------
    steane : SteanePattern
        The X-readout pattern.
    noise_models : collections.abc.Sequence[NoiseModel]
        Noise models of the sampled circuits.
    shots : int
        Shots sampled from each branch circuit.
    seed : int, optional
        Seed of the first branch; the second uses ``seed + 1``. By default 0.
    decoders : collections.abc.Mapping[bool, LookupDecoder] | None, optional
        Decoder of each branch. By default built from the detector error
        model of each branch under ``noise_models``.

    Returns
    -------
    LogicalErrorEstimate
        Failures among the kept shots.
    """
    clifft: Any = importlib.import_module("clifft")
    if decoders is None:
        decoders = {
            value: LookupDecoder.from_dem(branch_circuit(steane, value, noise_models).detector_error_model())
            for value in (False, True)
        }
    failures = kept = 0
    for value in (False, True):
        text = clifft_compile(
            steane.pattern,
            input_circuit=t_input_circuit(steane),
            fixed_controls={steane.control_node: value},
            noise_models=noise_models,
        )
        result = clifft.sample(clifft.compile(text), shots=shots, seed=seed + value)
        keep = decode_teleportation_bit(steane, np.asarray(result.measurements, dtype=np.bool_)) == value
        detectors = np.asarray(result.detectors, dtype=np.bool_)[keep]
        observables = np.asarray(result.observables, dtype=np.bool_)[keep, 0]
        failures += int(np.count_nonzero(decoders[value].predict(detectors) ^ observables))
        kept += int(np.count_nonzero(keep))
    return LogicalErrorEstimate(failures, kept)
