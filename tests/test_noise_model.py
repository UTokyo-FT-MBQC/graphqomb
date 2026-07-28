"""Tests for noise_model module."""

from __future__ import annotations

import math

import pytest
import typing_extensions

from graphqomb.common import Axis
from graphqomb.noise_model import (
    PAULI_CHANNEL_2_ORDER,
    Coordinate,
    DepolarizingNoiseModel,
    EntangleEvent,
    HeraldedErase,
    HeraldedPauliChannel1,
    IdleEvent,
    MeasureEvent,
    MeasurementFlip,
    MeasurementFlipNoiseModel,
    NodeInfo,
    NoiseModel,
    NoisePlacement,
    PauliChannel1,
    PauliChannel2,
    PrepareEvent,
    RawStimOp,
    default_noise_placement,
    depolarize1_probs,
    depolarize2_probs,
    noise_op_to_stim,
)

# ---- Coordinate Tests ----


class TestCoordinate:
    """Tests for Coordinate dataclass."""

    def test_xy_with_2d(self) -> None:
        """Test xy property with 2D coordinates."""
        coord = Coordinate((1.0, 2.0))
        assert coord.xy == (1.0, 2.0)

    def test_xy_with_3d(self) -> None:
        """Test xy property with 3D coordinates."""
        coord = Coordinate((1.0, 2.0, 3.0))
        assert coord.xy == (1.0, 2.0)

    def test_xy_with_1d(self) -> None:
        """Test xy property with 1D coordinates returns None."""
        coord = Coordinate((1.0,))
        assert coord.xy is None

    def test_xyz_with_3d(self) -> None:
        """Test xyz property with 3D coordinates."""
        coord = Coordinate((1.0, 2.0, 3.0))
        assert coord.xyz == (1.0, 2.0, 3.0)

    def test_xyz_with_2d(self) -> None:
        """Test xyz property with 2D coordinates returns None."""
        coord = Coordinate((1.0, 2.0))
        assert coord.xyz is None

    def test_xyz_with_4d(self) -> None:
        """Test xyz property with 4D coordinates."""
        coord = Coordinate((1.0, 2.0, 3.0, 4.0))
        assert coord.xyz == (1.0, 2.0, 3.0)


# ---- NodeInfo Tests ----


class TestNodeInfo:
    """Tests for NodeInfo dataclass."""

    def test_with_coordinate(self) -> None:
        """Test NodeInfo with coordinate."""
        coord = Coordinate((1.0, 2.0))
        info = NodeInfo(id=5, coord=coord)
        assert info.id == 5
        assert info.coord is not None
        assert info.coord.xy == (1.0, 2.0)

    def test_without_coordinate(self) -> None:
        """Test NodeInfo without coordinate."""
        info = NodeInfo(id=3, coord=None)
        assert info.id == 3
        assert info.coord is None


# ---- Event Tests ----


class TestPrepareEvent:
    """Tests for PrepareEvent dataclass."""

    def test_basic(self) -> None:
        """Test basic PrepareEvent creation."""
        node = NodeInfo(id=0, coord=None)
        event = PrepareEvent(time=0, node=node, is_input=True)
        assert event.time == 0
        assert event.node.id == 0
        assert event.is_input is True


class TestEntangleEvent:
    """Tests for EntangleEvent dataclass."""

    def test_basic(self) -> None:
        """Test basic EntangleEvent creation."""
        node0 = NodeInfo(id=0, coord=None)
        node1 = NodeInfo(id=1, coord=None)
        event = EntangleEvent(time=1, node0=node0, node1=node1, edge=(0, 1))
        assert event.time == 1
        assert event.node0.id == 0
        assert event.node1.id == 1
        assert event.edge == (0, 1)


class TestMeasureEvent:
    """Tests for MeasureEvent dataclass."""

    def test_basic(self) -> None:
        """Test basic MeasureEvent creation."""
        node = NodeInfo(id=2, coord=None)
        event = MeasureEvent(time=2, node=node, axis=Axis.X)
        assert event.time == 2
        assert event.node.id == 2
        assert event.axis == Axis.X


class TestIdleEvent:
    """Tests for IdleEvent dataclass."""

    def test_basic(self) -> None:
        """Test basic IdleEvent creation."""
        nodes = [NodeInfo(id=i, coord=None) for i in range(3)]
        event = IdleEvent(time=1, nodes=nodes, duration=1.0)
        assert event.time == 1
        assert len(event.nodes) == 3
        assert math.isclose(event.duration, 1.0)


# ---- NoiseOp Tests ----


class TestPauliChannel1:
    """Tests for PauliChannel1 noise operation."""

    def test_basic(self) -> None:
        """Test basic PauliChannel1 creation and conversion."""
        op = PauliChannel1(px=0.01, py=0.02, pz=0.03, targets=[0, 1])
        text, delta = noise_op_to_stim(op)
        assert text == "PAULI_CHANNEL_1(0.01,0.02,0.03) 0 1"
        assert delta == 0

    def test_single_target(self) -> None:
        """Test PauliChannel1 with single target."""
        op = PauliChannel1(px=0.1, py=0.0, pz=0.0, targets=[5])
        text, delta = noise_op_to_stim(op)
        assert text == "PAULI_CHANNEL_1(0.1,0.0,0.0) 5"
        assert delta == 0

    def test_empty_targets(self) -> None:
        """Test PauliChannel1 with empty targets returns empty string."""
        op = PauliChannel1(px=0.01, py=0.01, pz=0.01, targets=[])
        text, delta = noise_op_to_stim(op)
        assert not text
        assert delta == 0

    def test_placement_before(self) -> None:
        """Test PauliChannel1 with BEFORE placement."""
        op = PauliChannel1(px=0.01, py=0.0, pz=0.0, targets=[0], placement=NoisePlacement.BEFORE)
        assert op.placement == NoisePlacement.BEFORE

    def test_targets_converted_to_tuple(self) -> None:
        """Test that targets list is converted to tuple."""
        op = PauliChannel1(px=0.01, py=0.0, pz=0.0, targets=[0, 1, 2])
        assert isinstance(op.targets, tuple)
        assert op.targets == (0, 1, 2)

    def test_negative_probability_raises(self) -> None:
        """Test PauliChannel1 with negative probability raises ValueError."""
        with pytest.raises(ValueError, match=r"PauliChannel1\.px must be within \[0, 1\]"):
            PauliChannel1(px=-0.01, py=0.0, pz=0.0, targets=[0])

    def test_probability_sum_greater_than_one_raises(self) -> None:
        """Test PauliChannel1 with probabilities summing to >1 raises ValueError."""
        with pytest.raises(ValueError, match=r"PauliChannel1 probabilities must sum to <= 1"):
            PauliChannel1(px=0.5, py=0.4, pz=0.2, targets=[0])


class TestPauliChannel2:
    """Tests for PauliChannel2 noise operation."""

    def test_with_mapping(self) -> None:
        """Test PauliChannel2 with mapping probabilities."""
        op = PauliChannel2.from_mapping(probabilities={"ZZ": 0.01, "XX": 0.005}, targets=[(0, 1)])
        text, delta = noise_op_to_stim(op)
        assert "PAULI_CHANNEL_2" in text
        assert "0 1" in text
        assert delta == 0

    def test_with_sequence(self) -> None:
        """Test PauliChannel2 with sequence probabilities."""
        probs = [0.0] * 15
        probs[14] = 0.01  # ZZ
        op = PauliChannel2.from_sequence(probabilities=probs, targets=[(2, 3)])
        text, delta = noise_op_to_stim(op)
        assert "PAULI_CHANNEL_2" in text
        assert "2 3" in text
        assert delta == 0

    def test_multiple_pairs(self) -> None:
        """Test PauliChannel2 with multiple target pairs."""
        op = PauliChannel2.from_mapping(probabilities={"ZZ": 0.01}, targets=[(0, 1), (2, 3)])
        text, delta = noise_op_to_stim(op)
        assert "0 1 2 3" in text
        assert delta == 0

    def test_empty_targets(self) -> None:
        """Test PauliChannel2 with empty targets returns empty string."""
        op = PauliChannel2.from_mapping(probabilities={"ZZ": 0.01}, targets=[])
        text, delta = noise_op_to_stim(op)
        assert not text
        assert delta == 0

    def test_unknown_key_raises(self) -> None:
        """Test PauliChannel2 with unknown key raises ValueError."""
        with pytest.raises(ValueError, match="Unknown PAULI_CHANNEL_2 keys"):
            PauliChannel2.from_mapping(probabilities={"ZZZ": 0.01}, targets=[(0, 1)])

    def test_wrong_sequence_length_raises(self) -> None:
        """Test PauliChannel2 with wrong sequence length raises ValueError."""
        with pytest.raises(ValueError, match="PAULI_CHANNEL_2 expects 15 probabilities"):
            PauliChannel2.from_sequence(probabilities=[0.01] * 10, targets=[(0, 1)])

    def test_internal_representation_is_normalized(self) -> None:
        """Test that PauliChannel2 stores canonical tuple values."""
        op = PauliChannel2.from_mapping(probabilities={"ZZ": 0.01}, targets=[(0, 1), (2, 3)])
        assert isinstance(op.probabilities, tuple)
        assert isinstance(op.targets, tuple)
        assert all(isinstance(pair, tuple) for pair in op.targets)

    def test_pauli_channel_2_order_has_15_elements(self) -> None:
        """Test that PAULI_CHANNEL_2_ORDER has exactly 15 elements."""
        assert len(PAULI_CHANNEL_2_ORDER) == 15

    def test_probability_out_of_range_raises(self) -> None:
        """Test PauliChannel2 with out-of-range probability raises ValueError."""
        with pytest.raises(ValueError, match=r"PauliChannel2\.probabilities\[ZZ\] must be within \[0, 1\]"):
            PauliChannel2.from_mapping(probabilities={"ZZ": 1.1}, targets=[(0, 1)])

    def test_probability_sum_greater_than_one_raises(self) -> None:
        """Test PauliChannel2 with probabilities summing to >1 raises ValueError."""
        with pytest.raises(ValueError, match=r"PauliChannel2 probabilities must sum to <= 1"):
            PauliChannel2.from_mapping(probabilities={"ZZ": 0.8, "XX": 0.3}, targets=[(0, 1)])


class TestDepolarize1Probs:
    """Tests for depolarize1_probs utility function."""

    def test_returns_3_elements(self) -> None:
        """Test that depolarize1_probs returns px, py, pz."""
        probs = depolarize1_probs(0.03)
        assert len(probs) == 3
        assert set(probs.keys()) == {"px", "py", "pz"}

    def test_each_probability_is_p_over_3(self) -> None:
        """Test that each probability is p/3."""
        p = 0.03
        probs = depolarize1_probs(p)
        expected = p / 3
        for key, prob in probs.items():
            assert prob == expected, f"Expected {expected} for {key}, got {prob}"

    def test_can_be_used_with_pauli_channel_1(self) -> None:
        """Test that depolarize1_probs works with PauliChannel1."""
        probs = depolarize1_probs(0.03)
        op = PauliChannel1(px=probs["px"], py=probs["py"], pz=probs["pz"], targets=[0])
        text, delta = noise_op_to_stim(op)
        assert "PAULI_CHANNEL_1" in text
        assert delta == 0

    def test_invalid_probability_raises(self) -> None:
        """Test that depolarize1_probs validates probability range."""
        with pytest.raises(ValueError, match=r"depolarize1_probs\.p must be within \[0, 1\]"):
            depolarize1_probs(-0.1)


class TestDepolarize2Probs:
    """Tests for depolarize2_probs utility function."""

    def test_returns_15_elements(self) -> None:
        """Test that depolarize2_probs returns 15 Pauli pairs."""
        probs = depolarize2_probs(0.15)
        assert len(probs) == 15

    def test_each_probability_is_p_over_15(self) -> None:
        """Test that each probability is p/15."""
        p = 0.15
        probs = depolarize2_probs(p)
        expected = p / 15
        for pauli, prob in probs.items():
            assert prob == expected, f"Expected {expected} for {pauli}, got {prob}"

    def test_contains_all_pauli_pairs(self) -> None:
        """Test that all 15 Pauli pairs are present."""
        probs = depolarize2_probs(0.1)
        for pauli in PAULI_CHANNEL_2_ORDER:
            assert pauli in probs

    def test_can_be_used_with_pauli_channel_2(self) -> None:
        """Test that depolarize2_probs works with PauliChannel2."""
        probs = depolarize2_probs(0.15)
        op = PauliChannel2.from_mapping(probabilities=probs, targets=[(0, 1)])
        text, delta = noise_op_to_stim(op)
        assert "PAULI_CHANNEL_2" in text
        assert delta == 0

    def test_invalid_probability_raises(self) -> None:
        """Test that depolarize2_probs validates probability range."""
        with pytest.raises(ValueError, match=r"depolarize2_probs\.p must be within \[0, 1\]"):
            depolarize2_probs(1.2)


class TestHeraldedPauliChannel1:
    """Tests for HeraldedPauliChannel1 noise operation."""

    def test_basic(self) -> None:
        """Test basic HeraldedPauliChannel1 creation and conversion."""
        op = HeraldedPauliChannel1(pi=0.0, px=0.01, py=0.0, pz=0.0, targets=[5])
        text, delta = noise_op_to_stim(op)
        assert text == "HERALDED_PAULI_CHANNEL_1(0.0,0.01,0.0,0.0) 5"
        assert delta == 1

    def test_multiple_targets(self) -> None:
        """Test HeraldedPauliChannel1 with multiple targets."""
        op = HeraldedPauliChannel1(pi=0.1, px=0.0, py=0.0, pz=0.0, targets=[0, 1, 2])
        text, delta = noise_op_to_stim(op)
        assert text == "HERALDED_PAULI_CHANNEL_1(0.1,0.0,0.0,0.0) 0 1 2"
        assert delta == 3

    def test_empty_targets(self) -> None:
        """Test HeraldedPauliChannel1 with empty targets returns empty string."""
        op = HeraldedPauliChannel1(pi=0.0, px=0.01, py=0.0, pz=0.0, targets=[])
        text, delta = noise_op_to_stim(op)
        assert not text
        assert delta == 0

    def test_targets_converted_to_tuple(self) -> None:
        """Test that targets list is converted to tuple."""
        op = HeraldedPauliChannel1(pi=0.0, px=0.01, py=0.0, pz=0.0, targets=[0, 1])
        assert isinstance(op.targets, tuple)

    def test_probability_sum_greater_than_one_raises(self) -> None:
        """Test HeraldedPauliChannel1 with probabilities summing to >1 raises ValueError."""
        with pytest.raises(ValueError, match=r"HeraldedPauliChannel1 probabilities must sum to <= 1"):
            HeraldedPauliChannel1(pi=0.4, px=0.4, py=0.3, pz=0.0, targets=[0])


class TestHeraldedErase:
    """Tests for HeraldedErase noise operation."""

    def test_basic(self) -> None:
        """Test basic HeraldedErase creation and conversion."""
        op = HeraldedErase(p=0.05, targets=[0])
        text, delta = noise_op_to_stim(op)
        assert text == "HERALDED_ERASE(0.05) 0"
        assert delta == 1

    def test_multiple_targets(self) -> None:
        """Test HeraldedErase with multiple targets."""
        op = HeraldedErase(p=0.1, targets=[0, 1, 2])
        text, delta = noise_op_to_stim(op)
        assert text == "HERALDED_ERASE(0.1) 0 1 2"
        assert delta == 3

    def test_empty_targets(self) -> None:
        """Test HeraldedErase with empty targets returns empty string."""
        op = HeraldedErase(p=0.05, targets=[])
        text, delta = noise_op_to_stim(op)
        assert not text
        assert delta == 0

    def test_targets_converted_to_tuple(self) -> None:
        """Test that targets list is converted to tuple."""
        op = HeraldedErase(p=0.05, targets=[0, 1, 2])
        assert isinstance(op.targets, tuple)

    def test_probability_out_of_range_raises(self) -> None:
        """Test HeraldedErase with out-of-range probability raises ValueError."""
        with pytest.raises(ValueError, match=r"HeraldedErase\.p must be within \[0, 1\]"):
            HeraldedErase(p=1.1, targets=[0])


class TestRawStimOp:
    """Tests for RawStimOp noise operation."""

    def test_basic(self) -> None:
        """Test basic RawStimOp creation and conversion."""
        op = RawStimOp(text="X_ERROR(0.001) 0 1 2")
        text, delta = noise_op_to_stim(op)
        assert text == "X_ERROR(0.001) 0 1 2"
        assert delta == 0

    def test_with_record_delta(self) -> None:
        """Test RawStimOp with custom record delta."""
        op = RawStimOp(text="MR 5", record_delta=1)
        text, delta = noise_op_to_stim(op)
        assert text == "MR 5"
        assert delta == 1

    def test_empty_text(self) -> None:
        """Test RawStimOp with empty text."""
        op = RawStimOp(text="")
        text, delta = noise_op_to_stim(op)
        assert not text
        assert delta == 0

    def test_placement_before(self) -> None:
        """Test RawStimOp with BEFORE placement."""
        op = RawStimOp(text="Z_ERROR(0.01) 0", placement=NoisePlacement.BEFORE)
        assert op.placement == NoisePlacement.BEFORE

    def test_multiline_text_raises(self) -> None:
        """Test RawStimOp rejects multiline text."""
        with pytest.raises(ValueError, match="single Stim instruction line"):
            RawStimOp(text="M 0\nM 1", record_delta=0)

    def test_negative_record_delta_raises(self) -> None:
        """Test RawStimOp rejects negative record_delta."""
        with pytest.raises(ValueError, match="must be non-negative"):
            RawStimOp(text="X_ERROR(0.01) 0", record_delta=-1)

    def test_record_delta_mismatch_raises(self) -> None:
        """Test RawStimOp validates record_delta consistency for inferable instructions."""
        with pytest.raises(ValueError, match="record_delta mismatch"):
            RawStimOp(text="MR 5", record_delta=0)


# ---- NoiseModel Tests ----


class TestNoiseModel:
    """Tests for NoiseModel base class."""

    def test_default_on_prepare_returns_empty(self) -> None:
        """Test that default on_prepare returns empty list."""
        model = NoiseModel()
        node = NodeInfo(id=0, coord=None)
        event = PrepareEvent(time=0, node=node, is_input=False)
        result = list(model.on_prepare(event))
        assert result == []

    def test_default_on_entangle_returns_empty(self) -> None:
        """Test that default on_entangle returns empty list."""
        model = NoiseModel()
        node0 = NodeInfo(id=0, coord=None)
        node1 = NodeInfo(id=1, coord=None)
        event = EntangleEvent(time=0, node0=node0, node1=node1, edge=(0, 1))
        result = list(model.on_entangle(event))
        assert result == []

    def test_default_on_measure_returns_empty(self) -> None:
        """Test that default on_measure returns empty list."""
        model = NoiseModel()
        node = NodeInfo(id=0, coord=None)
        event = MeasureEvent(time=0, node=node, axis=Axis.X)
        result = list(model.on_measure(event))
        assert result == []

    def test_default_on_idle_returns_empty(self) -> None:
        """Test that default on_idle returns empty list."""
        model = NoiseModel()
        nodes = [NodeInfo(id=i, coord=None) for i in range(2)]
        event = IdleEvent(time=0, nodes=nodes, duration=1.0)
        result = list(model.on_idle(event))
        assert result == []

    def test_default_noise_placement_for_measure_is_before(self) -> None:
        """Test that default_noise_placement returns BEFORE for MeasureEvent."""
        node = NodeInfo(id=0, coord=None)
        event = MeasureEvent(time=0, node=node, axis=Axis.X)
        assert default_noise_placement(event) == NoisePlacement.BEFORE

    def test_default_noise_placement_for_prepare_is_after(self) -> None:
        """Test that default_noise_placement returns AFTER for PrepareEvent."""
        node = NodeInfo(id=0, coord=None)
        event = PrepareEvent(time=0, node=node, is_input=False)
        assert default_noise_placement(event) == NoisePlacement.AFTER

    def test_default_noise_placement_for_entangle_is_after(self) -> None:
        """Test that default_noise_placement returns AFTER for EntangleEvent."""
        node0 = NodeInfo(id=0, coord=None)
        node1 = NodeInfo(id=1, coord=None)
        event = EntangleEvent(time=0, node0=node0, node1=node1, edge=(0, 1))
        assert default_noise_placement(event) == NoisePlacement.AFTER

    def test_default_noise_placement_for_idle_is_after(self) -> None:
        """Test that default_noise_placement returns AFTER for IdleEvent."""
        nodes = [NodeInfo(id=i, coord=None) for i in range(2)]
        event = IdleEvent(time=0, nodes=nodes, duration=1.0)
        assert default_noise_placement(event) == NoisePlacement.AFTER


class TestNoisePlacementAuto:
    """Tests for AUTO placement behavior."""

    def test_auto_is_default_for_pauli_channel_1(self) -> None:
        """Test that AUTO is the default placement for PauliChannel1."""
        op = PauliChannel1(px=0.01, py=0.0, pz=0.0, targets=[0])
        assert op.placement == NoisePlacement.AUTO

    def test_auto_is_default_for_pauli_channel_2(self) -> None:
        """Test that AUTO is the default placement for PauliChannel2."""
        op = PauliChannel2.from_mapping(probabilities={"ZZ": 0.01}, targets=[(0, 1)])
        assert op.placement == NoisePlacement.AUTO

    def test_auto_is_default_for_heralded_pauli_channel_1(self) -> None:
        """Test that AUTO is the default placement for HeraldedPauliChannel1."""
        op = HeraldedPauliChannel1(pi=0.0, px=0.01, py=0.0, pz=0.0, targets=[0])
        assert op.placement == NoisePlacement.AUTO

    def test_auto_is_default_for_heralded_erase(self) -> None:
        """Test that AUTO is the default placement for HeraldedErase."""
        op = HeraldedErase(p=0.01, targets=[0])
        assert op.placement == NoisePlacement.AUTO

    def test_auto_is_default_for_raw_stim_op(self) -> None:
        """Test that AUTO is the default placement for RawStimOp."""
        op = RawStimOp(text="X_ERROR(0.01) 0")
        assert op.placement == NoisePlacement.AUTO


class _CustomNoiseModel(NoiseModel):
    """Test noise model that adds noise on all events."""

    def __init__(self, p: float) -> None:
        self.p = p

    @typing_extensions.override
    def on_prepare(self, event: PrepareEvent) -> list[PauliChannel1]:
        return [PauliChannel1(px=self.p, py=0.0, pz=0.0, targets=[event.node.id])]

    @typing_extensions.override
    def on_entangle(self, event: EntangleEvent) -> list[PauliChannel2]:
        return [PauliChannel2.from_mapping(probabilities={"ZZ": self.p}, targets=[(event.node0.id, event.node1.id)])]

    @typing_extensions.override
    def on_measure(self, event: MeasureEvent) -> list[HeraldedPauliChannel1]:
        return [HeraldedPauliChannel1(pi=0.0, px=self.p, py=0.0, pz=0.0, targets=[event.node.id])]

    @typing_extensions.override
    def on_idle(self, event: IdleEvent) -> list[PauliChannel1]:
        p = self.p * event.duration
        targets = [n.id for n in event.nodes]
        return [PauliChannel1(px=p, py=p, pz=p, targets=targets)]


class TestCustomNoiseModel:
    """Tests for custom NoiseModel subclass."""

    def test_on_prepare(self) -> None:
        """Test custom on_prepare implementation."""
        model = _CustomNoiseModel(p=0.01)
        node = NodeInfo(id=5, coord=None)
        event = PrepareEvent(time=0, node=node, is_input=False)
        ops = list(model.on_prepare(event))
        assert len(ops) == 1
        assert isinstance(ops[0], PauliChannel1)
        assert math.isclose(ops[0].px, 0.01)
        assert 5 in ops[0].targets

    def test_on_entangle(self) -> None:
        """Test custom on_entangle implementation."""
        model = _CustomNoiseModel(p=0.02)
        node0 = NodeInfo(id=0, coord=None)
        node1 = NodeInfo(id=1, coord=None)
        event = EntangleEvent(time=1, node0=node0, node1=node1, edge=(0, 1))
        ops = list(model.on_entangle(event))
        assert len(ops) == 1
        assert isinstance(ops[0], PauliChannel2)

    def test_on_measure(self) -> None:
        """Test custom on_measure implementation."""
        model = _CustomNoiseModel(p=0.03)
        node = NodeInfo(id=2, coord=None)
        event = MeasureEvent(time=2, node=node, axis=Axis.Z)
        ops = list(model.on_measure(event))
        assert len(ops) == 1
        assert isinstance(ops[0], HeraldedPauliChannel1)

    def test_on_idle(self) -> None:
        """Test custom on_idle implementation."""
        model = _CustomNoiseModel(p=0.001)
        nodes = [NodeInfo(id=i, coord=None) for i in range(3)]
        event = IdleEvent(time=1, nodes=nodes, duration=2.0)
        ops = list(model.on_idle(event))
        assert len(ops) == 1
        assert isinstance(ops[0], PauliChannel1)
        assert math.isclose(ops[0].px, 0.002)  # p * duration


# ---- MeasurementFlip Tests ----


class TestMeasurementFlip:
    """Tests for MeasurementFlip noise operation."""

    def test_basic(self) -> None:
        """Test basic MeasurementFlip creation."""
        op = MeasurementFlip(p=0.01, target=5)
        assert math.isclose(op.p, 0.01)
        assert op.target == 5
        assert op.placement == NoisePlacement.AUTO

    def test_to_stim_returns_empty(self) -> None:
        """Test that noise_op_to_stim returns empty string for MeasurementFlip.

        MeasurementFlip is handled specially by modifying the measurement
        instruction itself, so it should not emit a separate instruction.
        """
        op = MeasurementFlip(p=0.01, target=0)
        text, delta = noise_op_to_stim(op)
        assert not text
        assert delta == 0

    def test_invalid_probability_raises(self) -> None:
        """Test MeasurementFlip validates probability range."""
        with pytest.raises(ValueError, match=r"MeasurementFlip\.p must be within \[0, 1\]"):
            MeasurementFlip(p=-0.1, target=0)


# ---- DepolarizingNoiseModel Tests ----


class TestDepolarizingNoiseModel:
    """Tests for DepolarizingNoiseModel built-in noise model."""

    def test_on_prepare_emits_depolarize1(self) -> None:
        """Test that on_prepare returns DEPOLARIZE1 instruction."""
        model = DepolarizingNoiseModel(p1=0.01)
        node = NodeInfo(id=5, coord=None)
        event = PrepareEvent(time=0, node=node, is_input=False)
        ops = list(model.on_prepare(event))
        assert len(ops) == 1
        text, _ = noise_op_to_stim(ops[0])
        assert text == "DEPOLARIZE1(0.01) 5"

    def test_on_entangle_emits_depolarize2(self) -> None:
        """Test that on_entangle returns DEPOLARIZE2 instruction."""
        model = DepolarizingNoiseModel(p1=0.01)
        node0 = NodeInfo(id=0, coord=None)
        node1 = NodeInfo(id=1, coord=None)
        event = EntangleEvent(time=1, node0=node0, node1=node1, edge=(0, 1))
        ops = list(model.on_entangle(event))
        assert len(ops) == 1
        text, _ = noise_op_to_stim(ops[0])
        assert text == "DEPOLARIZE2(0.01) 0 1"

    def test_p2_defaults_to_p1(self) -> None:
        """Test that p2 defaults to p1 when not specified."""
        model = DepolarizingNoiseModel(p1=0.02)
        node0 = NodeInfo(id=2, coord=None)
        node1 = NodeInfo(id=3, coord=None)
        event = EntangleEvent(time=1, node0=node0, node1=node1, edge=(2, 3))
        ops = list(model.on_entangle(event))
        text, _ = noise_op_to_stim(ops[0])
        assert "DEPOLARIZE2(0.02)" in text

    def test_different_p1_and_p2(self) -> None:
        """Test DepolarizingNoiseModel with different p1 and p2."""
        model = DepolarizingNoiseModel(p1=0.001, p2=0.01)
        # Check prepare uses p1
        node = NodeInfo(id=0, coord=None)
        prepare_event = PrepareEvent(time=0, node=node, is_input=False)
        ops = list(model.on_prepare(prepare_event))
        text, _ = noise_op_to_stim(ops[0])
        assert "DEPOLARIZE1(0.001)" in text

        # Check entangle uses p2
        node0 = NodeInfo(id=0, coord=None)
        node1 = NodeInfo(id=1, coord=None)
        entangle_event = EntangleEvent(time=1, node0=node0, node1=node1, edge=(0, 1))
        ops = list(model.on_entangle(entangle_event))
        text, _ = noise_op_to_stim(ops[0])
        assert "DEPOLARIZE2(0.01)" in text

    def test_zero_probability_returns_empty(self) -> None:
        """Test that zero probability returns empty sequence."""
        model = DepolarizingNoiseModel(p1=0.0)
        node = NodeInfo(id=0, coord=None)
        event = PrepareEvent(time=0, node=node, is_input=False)
        ops = list(model.on_prepare(event))
        assert len(ops) == 0

    def test_invalid_p1_raises(self) -> None:
        """Test that invalid p1 is rejected at construction time."""
        with pytest.raises(ValueError, match=r"DepolarizingNoiseModel\.p1 must be within \[0, 1\]"):
            DepolarizingNoiseModel(p1=1.1)

    def test_invalid_p2_raises(self) -> None:
        """Test that invalid p2 is rejected at construction time."""
        with pytest.raises(ValueError, match=r"DepolarizingNoiseModel\.p2 must be within \[0, 1\]"):
            DepolarizingNoiseModel(p1=0.1, p2=-0.1)

    def test_nan_probability_raises(self) -> None:
        """Test that NaN probabilities are rejected at construction time."""
        with pytest.raises(ValueError, match=r"DepolarizingNoiseModel\.p1 must be within \[0, 1\]"):
            DepolarizingNoiseModel(p1=float("nan"))


# ---- MeasurementFlipNoiseModel Tests ----


class TestMeasurementFlipNoiseModel:
    """Tests for MeasurementFlipNoiseModel built-in noise model."""

    def test_on_measure_returns_measurement_flip(self) -> None:
        """Test that on_measure returns MeasurementFlip operation."""
        model = MeasurementFlipNoiseModel(p=0.01)
        node = NodeInfo(id=5, coord=None)
        event = MeasureEvent(time=0, node=node, axis=Axis.X)
        ops = list(model.on_measure(event))
        assert len(ops) == 1
        assert isinstance(ops[0], MeasurementFlip)
        assert math.isclose(ops[0].p, 0.01)
        assert ops[0].target == 5

    def test_zero_probability_returns_empty(self) -> None:
        """Test that zero probability returns empty sequence."""
        model = MeasurementFlipNoiseModel(p=0.0)
        node = NodeInfo(id=0, coord=None)
        event = MeasureEvent(time=0, node=node, axis=Axis.Z)
        ops = list(model.on_measure(event))
        assert len(ops) == 0

    def test_different_axes(self) -> None:
        """Test MeasurementFlipNoiseModel works with all measurement axes."""
        model = MeasurementFlipNoiseModel(p=0.005)
        for axis in [Axis.X, Axis.Y, Axis.Z]:
            node = NodeInfo(id=0, coord=None)
            event = MeasureEvent(time=0, node=node, axis=axis)
            ops = list(model.on_measure(event))
            assert len(ops) == 1
            assert isinstance(ops[0], MeasurementFlip)
            assert math.isclose(ops[0].p, 0.005)

    def test_invalid_probability_raises(self) -> None:
        """Test that invalid p is rejected at construction time."""
        with pytest.raises(ValueError, match=r"MeasurementFlipNoiseModel\.p must be within \[0, 1\]"):
            MeasurementFlipNoiseModel(p=1.1)

    def test_nan_probability_raises(self) -> None:
        """Test that NaN p is rejected at construction time."""
        with pytest.raises(ValueError, match=r"MeasurementFlipNoiseModel\.p must be within \[0, 1\]"):
            MeasurementFlipNoiseModel(p=float("nan"))
