"""Checks for noisy Steane T teleportation with an online-decoded S control."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import typing_extensions

from examples.ftqc._support.steane import build_steane_pattern
from examples.ftqc._support.steane_noise import (
    LookupDecoder,
    branch_circuit,
    decode_teleportation_bit,
    error_mechanisms,
    sample_logical_error,
    t_input_circuit,
)
from graphqomb.command import M
from graphqomb.common import Axis
from graphqomb.noise_model import MeasureEvent, MeasurementFlip, MeasurementFlipNoiseModel, NoiseModel
from graphqomb.stim_glue.clifft_compiler import clifft_compile

if TYPE_CHECKING:
    from collections.abc import Sequence

    from examples.ftqc._support.steane import SteanePattern
    from graphqomb.noise_model import NoiseOp


@pytest.fixture(scope="module")
def steane() -> SteanePattern:
    return build_steane_pattern(Axis.X)


class _FlipRecord(NoiseModel):
    """Flip one measurement record, a single classical readout fault."""

    def __init__(self, node: int, p: float) -> None:
        self._node = node
        self._p = p

    @typing_extensions.override
    def on_measure(self, event: MeasureEvent) -> Sequence[NoiseOp]:
        return (MeasurementFlip(self._p, target=self._node),) if event.node.id == self._node else ()


@pytest.mark.parametrize("value", [False, True])
def test_branch_input_fixes_the_teleportation_bit_and_readout(steane: SteanePattern, value: bool) -> None:
    circuit = branch_circuit(steane, value, [])
    measurements = circuit.compile_sampler(seed=1).sample(50)
    detectors, observables = circuit.compile_detector_sampler(seed=1).sample(50, separate_observables=True)
    control = [command.node for command in steane.pattern if isinstance(command, M)].index(steane.control_node)
    assert np.all(measurements[:, control] == value)
    assert not detectors.any()
    assert not observables.any()


@pytest.mark.parametrize("value", [False, True])
def test_lookup_decoder_corrects_every_single_measurement_flip(steane: SteanePattern, value: bool) -> None:
    dem = branch_circuit(steane, value, [MeasurementFlipNoiseModel(1e-3)]).detector_error_model()
    decoder = LookupDecoder.from_dem(dem)
    mechanisms = error_mechanisms(dem)
    assert mechanisms
    for mechanism in mechanisms:
        detectors = np.zeros((1, dem.num_detectors), dtype=np.bool_)
        detectors[0, sorted(mechanism.detectors)] = True
        assert decoder.predict(detectors)[0] == mechanism.flips_observable


@pytest.mark.parametrize("qubit", range(7))
def test_online_decoded_control_tolerates_a_readout_fault(steane: SteanePattern, qubit: int) -> None:
    clifft = pytest.importorskip("clifft")
    noise = [_FlipRecord(steane.consumed_nodes[qubit], 0.5)]
    kept = 0
    for value in (False, True):
        text = clifft_compile(
            steane.pattern,
            input_circuit=t_input_circuit(steane),
            noise_models=noise,
            fixed_controls={steane.control_node: value},
        )
        result = clifft.sample(clifft.compile(text), shots=400, seed=7)
        keep = decode_teleportation_bit(steane, np.asarray(result.measurements, dtype=bool)) == value
        assert np.asarray(result.detectors, dtype=bool)[keep].any()
        assert not np.asarray(result.observables, dtype=bool)[keep, 0].any()
        kept += int(keep.sum())
    assert 300 < kept < 500


def test_every_single_measurement_flip_is_corrected(steane: SteanePattern) -> None:
    pytest.importorskip("clifft")
    noise = [MeasurementFlipNoiseModel(1e-3)]
    decoders = {
        value: LookupDecoder.from_dem(branch_circuit(steane, value, noise).detector_error_model())
        for value in (False, True)
    }
    for command in steane.pattern:
        if not isinstance(command, M):
            continue
        estimate = sample_logical_error(steane, [_FlipRecord(command.node, 1.0)], 50, seed=9, decoders=decoders)
        assert estimate.shots > 0
        assert estimate.failures == 0, f"a flip of node {command.node} causes a logical failure"
