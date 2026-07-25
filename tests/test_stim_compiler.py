"""Tests for stim_compiler module."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from graphqomb.command import TICK, E, M, N
from graphqomb.common import Axis, AxisMeasBasis, Plane, PlannerMeasBasis, Sign
from graphqomb.graphstate import GraphState
from graphqomb.noise_model import (
    DepolarizingNoiseModel,
    EntangleEvent,
    HeraldedPauliChannel1,
    IdleEvent,
    MeasureEvent,
    MeasurementFlip,
    MeasurementFlipNoiseModel,
    NoiseModel,
    PrepareEvent,
)
from graphqomb.qompiler import qompile
from graphqomb.schedule_solver import ScheduleConfig, Strategy
from graphqomb.scheduler import Scheduler
from graphqomb.stim_compiler import stim_compile

if TYPE_CHECKING:
    from graphqomb.pattern import Pattern


def create_simple_pattern_x_measurement() -> tuple[Pattern, int, int]:
    """Create a simple pattern with X measurement for testing.

    Returns
    -------
    tuple[Pattern, int, int]
        Pattern and expected node for X measurement
    """
    graph = GraphState()
    in_node = graph.add_node()
    meas_node = graph.add_node()
    out_node = graph.add_node()

    q_idx = 0
    graph.register_input(in_node, q_idx)
    graph.register_output(out_node, q_idx)

    graph.add_edge(in_node, meas_node)
    graph.add_edge(meas_node, out_node)

    # X measurement: XY plane with angle 0
    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(meas_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(out_node, PlannerMeasBasis(Plane.XY, 0.0))

    xflow = {in_node: {meas_node}, meas_node: {out_node}}
    pattern = qompile(graph, xflow)

    return pattern, meas_node, in_node


def create_simple_pattern_y_measurement() -> tuple[Pattern, int, int]:
    """Create a simple pattern with Y measurement for testing.

    Returns
    -------
    tuple[Pattern, int, int]
        Pattern and expected node for Y measurement
    """
    graph = GraphState()
    in_node = graph.add_node()
    meas_node = graph.add_node()
    out_node = graph.add_node()

    q_idx = 0
    graph.register_input(in_node, q_idx)
    graph.register_output(out_node, q_idx)

    graph.add_edge(in_node, meas_node)
    graph.add_edge(meas_node, out_node)

    # Y measurement: XY plane with angle π/2
    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, math.pi / 2))
    graph.assign_meas_basis(meas_node, PlannerMeasBasis(Plane.XY, math.pi / 2))
    graph.assign_meas_basis(out_node, PlannerMeasBasis(Plane.XY, math.pi / 2))

    xflow = {in_node: {meas_node}, meas_node: {out_node}}
    pattern = qompile(graph, xflow)

    return pattern, meas_node, in_node


def create_simple_pattern_z_measurement() -> tuple[Pattern, int, int]:
    """Create a simple pattern with Z measurement for testing.

    Returns
    -------
    tuple[Pattern, int, int]
        Pattern and expected node for Z measurement
    """
    graph = GraphState()
    in_node = graph.add_node()
    meas_node = graph.add_node()
    out_node = graph.add_node()

    q_idx = 0
    graph.register_input(in_node, q_idx)
    graph.register_output(out_node, q_idx)

    graph.add_edge(in_node, meas_node)
    graph.add_edge(meas_node, out_node)

    # Z measurement: XZ plane with angle 0
    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XZ, 0.0))
    graph.assign_meas_basis(meas_node, PlannerMeasBasis(Plane.XZ, 0.0))
    graph.assign_meas_basis(out_node, PlannerMeasBasis(Plane.XZ, 0.0))

    xflow = {in_node: {meas_node}, meas_node: {out_node}}
    pattern = qompile(graph, xflow)

    return pattern, meas_node, in_node


def test_stim_compile_basic_pattern() -> None:
    """Test basic pattern compilation to stim format."""
    pattern, _, _ = create_simple_pattern_x_measurement()

    stim_str = stim_compile(pattern)

    # Check basic structure
    assert "RX" in stim_str
    assert "CZ" in stim_str
    assert "MX" in stim_str
    assert stim_str.count("\n") > 0


@pytest.mark.parametrize(
    ("init_axis", "expected_reset"),
    [
        (Axis.X, "RX"),
        (Axis.Y, "RY"),
        (Axis.Z, "R"),
    ],
)
def test_stim_compile_uses_input_initialization_axis(init_axis: Axis, expected_reset: str) -> None:
    """Input initialization axes choose the corresponding Stim reset instruction."""
    graph = GraphState()
    in_node = graph.add_node()
    out_node = graph.add_node()

    graph.register_input(in_node, 0, init_axis=init_axis)
    graph.register_output(out_node, 0)
    graph.add_edge(in_node, out_node)
    graph.assign_meas_basis(in_node, AxisMeasBasis(Axis.X, Sign.PLUS))

    pattern = qompile(graph, {in_node: {out_node}})
    stim_lines = stim_compile(pattern).splitlines()

    assert f"{expected_reset} {in_node}" in stim_lines


def test_stim_compile_keeps_non_input_preparations_in_x_basis() -> None:
    """Only input reset instructions use the input initialization axis."""
    graph = GraphState()
    in_node = graph.add_node()
    mid_node = graph.add_node()
    out_node = graph.add_node()

    graph.register_input(in_node, 0, init_axis=Axis.Z)
    graph.register_output(out_node, 0)
    graph.add_edge(in_node, mid_node)
    graph.add_edge(mid_node, out_node)
    graph.assign_meas_basis(in_node, AxisMeasBasis(Axis.X, Sign.PLUS))
    graph.assign_meas_basis(mid_node, AxisMeasBasis(Axis.X, Sign.PLUS))

    pattern = qompile(graph, {in_node: {mid_node}, mid_node: {out_node}})
    stim_lines = stim_compile(pattern).splitlines()

    assert f"R {in_node}" in stim_lines
    assert f"RX {mid_node}" in stim_lines


def test_stim_compile_x_measurement() -> None:
    """Test X measurement compilation."""
    pattern, meas_node, in_node = create_simple_pattern_x_measurement()

    stim_str = stim_compile(pattern)

    # X measurement should generate MX command
    assert "MX" in stim_str
    assert f"MX {meas_node}" in stim_str or f"MX {in_node}" in stim_str


def test_stim_compile_y_measurement() -> None:
    """Test Y measurement compilation."""
    pattern, meas_node, in_node = create_simple_pattern_y_measurement()

    stim_str = stim_compile(pattern)

    # Y measurement should generate MY command
    assert "MY" in stim_str
    assert f"MY {meas_node}" in stim_str or f"MY {in_node}" in stim_str


def test_stim_compile_z_measurement() -> None:
    """Test Z measurement compilation."""
    pattern, meas_node, in_node = create_simple_pattern_z_measurement()

    stim_str = stim_compile(pattern)

    # Z measurement should generate MZ command
    assert "MZ" in stim_str
    assert f"MZ {meas_node}" in stim_str or f"MZ {in_node}" in stim_str


def test_stim_compile_with_depolarization() -> None:
    """Test that depolarization error is correctly inserted using DepolarizingNoiseModel."""
    pattern, _, _ = create_simple_pattern_x_measurement()

    stim_str = stim_compile(pattern, noise_models=[DepolarizingNoiseModel(p1=0.01)])

    # Check DEPOLARIZE instructions are present
    assert "DEPOLARIZE1(0.01)" in stim_str
    assert "DEPOLARIZE2(0.01)" in stim_str


def test_stim_compile_with_measurement_errors_x() -> None:
    """Test that X measurement errors are correctly inserted using MeasurementFlipNoiseModel."""
    pattern, _, _ = create_simple_pattern_x_measurement()

    stim_str = stim_compile(pattern, noise_models=[MeasurementFlipNoiseModel(p=0.01)])

    # For X measurement, error probability is attached to MX instruction
    assert "MX(0.01)" in stim_str


def test_stim_compile_with_measurement_errors_y() -> None:
    """Test that Y measurement errors are correctly inserted using MeasurementFlipNoiseModel."""
    pattern, _, _ = create_simple_pattern_y_measurement()

    stim_str = stim_compile(pattern, noise_models=[MeasurementFlipNoiseModel(p=0.01)])

    # For Y measurement, error probability is attached to MY instruction
    assert "MY(0.01)" in stim_str


def test_stim_compile_with_measurement_errors_z() -> None:
    """Test that Z measurement errors are correctly inserted using MeasurementFlipNoiseModel."""
    pattern, _, _ = create_simple_pattern_z_measurement()

    stim_str = stim_compile(pattern, noise_models=[MeasurementFlipNoiseModel(p=0.01)])

    # For Z measurement, error probability is attached to MZ instruction
    assert "MZ(0.01)" in stim_str


def test_stim_compile_combines_measurement_flip_probabilities() -> None:
    """Multiple MeasurementFlip models should combine as independent events."""
    pattern, _, _ = create_simple_pattern_x_measurement()

    stim_str = stim_compile(
        pattern,
        noise_models=[
            MeasurementFlipNoiseModel(p=0.1),
            MeasurementFlipNoiseModel(p=0.2),
        ],
    )

    expected = (1 - 0.1) * 0.2 + 0.1 * (1 - 0.2)
    mx_lines = [line for line in stim_str.splitlines() if line.startswith("MX(")]
    assert mx_lines
    for line in mx_lines:
        prob = float(line.split("(", 1)[1].split(")", 1)[0])
        assert math.isclose(prob, expected)


def test_stim_compile_removed_legacy_noise_parameters() -> None:
    """Removed legacy noise parameters should no longer be accepted."""
    pattern, _, _ = create_simple_pattern_x_measurement()

    with pytest.raises(TypeError, match="unexpected keyword argument 'p_before_meas_flip'"):
        stim_compile(pattern, p_before_meas_flip=0.01)  # type: ignore[call-arg]


def test_stim_compile_with_detectors() -> None:
    """Test DETECTOR generation with parity check groups."""
    graph = GraphState()
    in_node = graph.add_node()
    meas_node = graph.add_node()
    out_node = graph.add_node()

    q_idx = 0
    graph.register_input(in_node, q_idx)
    graph.register_output(out_node, q_idx)

    graph.add_edge(in_node, meas_node)
    graph.add_edge(meas_node, out_node)

    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(meas_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(out_node, PlannerMeasBasis(Plane.XY, 0.0))

    xflow = {in_node: {meas_node}, meas_node: {out_node}}
    # Add parity check groups
    parity_check_group = [{in_node}]
    pattern = qompile(graph, xflow, parity_check_group=parity_check_group)

    stim_str = stim_compile(pattern)

    # Check DETECTOR instruction is present
    assert "DETECTOR" in stim_str
    # DETECTOR may be empty if the dependent chain resolves to empty set
    # This is valid behavior for certain graph configurations


class _HeraldedNoise(NoiseModel):
    """Test noise model that adds heralded Pauli channel on measurements."""

    def on_measure(self, event: MeasureEvent) -> list[HeraldedPauliChannel1]:
        return [HeraldedPauliChannel1(0.0, 0.0, 0.0, 0.1, targets=[event.node.id])]


class _MismatchedMeasurementFlipNoise(NoiseModel):
    """Test noise model with intentionally mismatched MeasurementFlip target."""

    def on_measure(self, event: MeasureEvent) -> list[MeasurementFlip]:
        return [MeasurementFlip(p=0.1, target=event.node.id + 999)]


class _PrepareMeasurementFlipNoise(NoiseModel):
    """Test noise model with invalid MeasurementFlip on prepare."""

    def on_prepare(self, event: PrepareEvent) -> list[MeasurementFlip]:
        return [MeasurementFlip(p=0.1, target=event.node.id)]


class _EntangleMeasurementFlipNoise(NoiseModel):
    """Test noise model with invalid MeasurementFlip on entangle."""

    def on_entangle(self, event: EntangleEvent) -> list[MeasurementFlip]:
        return [MeasurementFlip(p=0.1, target=event.node0.id)]


class _IdleMeasurementFlipNoise(NoiseModel):
    """Test noise model with invalid MeasurementFlip on idle."""

    def on_idle(self, event: IdleEvent) -> list[MeasurementFlip]:
        return [MeasurementFlip(p=0.1, target=event.nodes[0].id)]


def _parse_stim_measurements(stim_str: str) -> tuple[dict[int, int], int]:
    """Parse stim string to extract measurement order and total record count."""
    rec_index = 0
    actual_meas_order: dict[int, int] = {}
    for raw_line in stim_str.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        opcode = stripped.split()[0].split("(", 1)[0]
        if opcode == "HERALDED_PAULI_CHANNEL_1":
            targets = stripped.split(")", 1)[1].strip().split()
            rec_index += len(targets)
        elif opcode in {"MX", "MY", "MZ"}:
            node = int(stripped.split()[1])
            actual_meas_order[node] = rec_index
            rec_index += 1
    return actual_meas_order, rec_index


def _normalize_detector(line: str) -> str:
    """Normalize detector line by sorting targets."""
    parts = line.strip().split()
    if len(parts) <= 1:
        return "DETECTOR"
    targets = sorted(parts[1:])
    return f"DETECTOR {' '.join(targets)}"


def test_stim_compile_with_heralded_noise_updates_detectors() -> None:
    """Heralded noise should shift rec indices used by detectors."""
    graph = GraphState()
    in_node = graph.add_node()
    meas_node = graph.add_node()
    out_node = graph.add_node()

    q_idx = 0
    graph.register_input(in_node, q_idx)
    graph.register_output(out_node, q_idx)

    graph.add_edge(in_node, meas_node)
    graph.add_edge(meas_node, out_node)

    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(meas_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(out_node, PlannerMeasBasis(Plane.XY, 0.0))

    xflow = {in_node: {meas_node}, meas_node: {out_node}}
    parity_check_group = [{in_node}]
    pattern = qompile(graph, xflow, parity_check_group=parity_check_group)

    stim_str = stim_compile(pattern, noise_models=[_HeraldedNoise()])

    actual_meas_order, total_measurements = _parse_stim_measurements(stim_str)

    check_groups = pattern.pauli_frame.detector_groups()
    expected_detectors = {
        _normalize_detector(
            f"DETECTOR {' '.join(f'rec[{actual_meas_order[check] - total_measurements}]' for check in checks)}"
        )
        for checks in check_groups
    }
    actual_detectors = {_normalize_detector(line) for line in stim_str.splitlines() if line.startswith("DETECTOR")}
    assert expected_detectors == actual_detectors


def test_stim_compile_rejects_mismatched_measurement_flip_target() -> None:
    """MeasurementFlip target must match the current measurement node."""
    pattern, _, _ = create_simple_pattern_x_measurement()

    with pytest.raises(ValueError, match="MeasurementFlip target mismatch"):
        stim_compile(pattern, noise_models=[_MismatchedMeasurementFlipNoise()])


@pytest.mark.parametrize(
    "noise_model",
    [
        _PrepareMeasurementFlipNoise(),
        _EntangleMeasurementFlipNoise(),
        _IdleMeasurementFlipNoise(),
    ],
)
def test_stim_compile_rejects_measurement_flip_outside_measurement(noise_model: NoiseModel) -> None:
    """MeasurementFlip can only be emitted during measurement events."""
    pattern, _, _ = create_simple_pattern_x_measurement()

    with pytest.raises(TypeError, match=r"MeasurementFlip can only be returned from NoiseModel\.on_measure"):
        stim_compile(pattern, noise_models=[noise_model])


def test_stim_compile_with_logical_observables() -> None:
    """Issue #167: logical observables should compile without parity_check_group."""
    graph = GraphState()
    in_node = graph.add_node()
    meas_node = graph.add_node()
    out_node = graph.add_node()

    q_idx = 0
    graph.register_input(in_node, q_idx)
    graph.register_output(out_node, q_idx)

    graph.add_edge(in_node, meas_node)
    graph.add_edge(meas_node, out_node)

    # X measurement: XY plane with angle 0
    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(meas_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(out_node, PlannerMeasBasis(Plane.XY, 0.0))

    xflow = {in_node: {meas_node}, meas_node: {out_node}}
    pattern = qompile(graph, xflow, logical_observables={0: {meas_node}})

    stim_str = stim_compile(pattern)

    # Check OBSERVABLE_INCLUDE instruction is present
    assert "OBSERVABLE_INCLUDE(0)" in stim_str
    # OBSERVABLE_INCLUDE may be empty if the dependent chain resolves to empty set
    # This is valid behavior for certain graph configurations


def test_stim_compile_uses_logical_observables_from_qompile() -> None:
    """Stored logical observables should be emitted when stim_compile omits them."""
    graph = GraphState()
    in_node = graph.add_node()
    meas_node = graph.add_node()
    out_node = graph.add_node()

    q_idx = 0
    graph.register_input(in_node, q_idx)
    graph.register_output(out_node, q_idx)

    graph.add_edge(in_node, meas_node)
    graph.add_edge(meas_node, out_node)

    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(meas_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(out_node, PlannerMeasBasis(Plane.XY, 0.0))

    xflow = {in_node: {meas_node}, meas_node: {out_node}}
    pattern = qompile(graph, xflow, logical_observables={0: {meas_node}})

    assert pattern.pauli_frame.logical_observables == {0: {meas_node}}
    assert "OBSERVABLE_INCLUDE(0)" in stim_compile(pattern)


def test_stim_compile_unsupported_basis() -> None:
    """Test that unsupported measurement basis raises ValueError."""
    graph = GraphState()
    in_node = graph.add_node()
    meas_node = graph.add_node()
    out_node = graph.add_node()

    q_idx = 0
    graph.register_input(in_node, q_idx)
    graph.register_output(out_node, q_idx)

    graph.add_edge(in_node, meas_node)
    graph.add_edge(meas_node, out_node)

    # Non-Pauli measurement: XY plane with arbitrary angle
    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.1))
    graph.assign_meas_basis(meas_node, PlannerMeasBasis(Plane.XY, 0.1))
    graph.assign_meas_basis(out_node, PlannerMeasBasis(Plane.XY, 0.1))

    xflow = {in_node: {meas_node}, meas_node: {out_node}}
    pattern = qompile(graph, xflow)

    # Should raise ValueError for unsupported measurement basis
    with pytest.raises(ValueError, match="Unsupported measurement basis"):
        stim_compile(pattern)


def test_stim_compile_unmeasured_output_has_no_correction_commands() -> None:
    """Unmeasured outputs should compile without terminal correction commands."""
    graph = GraphState()
    in_node = graph.add_node()
    out_node = graph.add_node()

    q_idx = 0
    graph.register_input(in_node, q_idx)
    graph.register_output(out_node, q_idx)

    graph.add_edge(in_node, out_node)
    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.0))

    xflow = {in_node: {out_node}}
    pattern = qompile(graph, xflow)

    assert all(isinstance(cmd, (N, E, M, TICK)) for cmd in pattern.commands)
    assert all(not (isinstance(cmd, M) and cmd.node == out_node) for cmd in pattern.commands)

    stim_str = stim_compile(pattern)

    assert isinstance(stim_str, str)
    assert stim_str


def test_stim_compile_empty_pattern() -> None:
    """Test compilation of minimal pattern."""
    graph = GraphState()
    in_node = graph.add_node()
    out_node = graph.add_node()

    q_idx = 0
    graph.register_input(in_node, q_idx)
    graph.register_output(out_node, q_idx)

    graph.add_edge(in_node, out_node)
    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(out_node, PlannerMeasBasis(Plane.XY, 0.0))

    xflow = {in_node: {out_node}}
    pattern = qompile(graph, xflow)

    stim_str = stim_compile(pattern)

    # Should compile without errors
    assert isinstance(stim_str, str)
    assert len(stim_str) > 0


def test_stim_compile_axis_meas_basis() -> None:
    """Test compilation with AxisMeasBasis."""
    graph = GraphState()
    in_node = graph.add_node()
    meas_node = graph.add_node()
    out_node = graph.add_node()

    q_idx = 0
    graph.register_input(in_node, q_idx)
    graph.register_output(out_node, q_idx)

    graph.add_edge(in_node, meas_node)
    graph.add_edge(meas_node, out_node)

    # Use AxisMeasBasis instead of PlannerMeasBasis
    graph.assign_meas_basis(in_node, AxisMeasBasis(Axis.X, Sign.PLUS))
    graph.assign_meas_basis(meas_node, AxisMeasBasis(Axis.Y, Sign.PLUS))
    graph.assign_meas_basis(out_node, AxisMeasBasis(Axis.X, Sign.PLUS))

    xflow = {in_node: {meas_node}, meas_node: {out_node}}
    pattern = qompile(graph, xflow)

    stim_str = stim_compile(pattern)

    # Should compile with both MX and MY
    assert "MX" in stim_str
    assert "MY" in stim_str


def test_stim_compile_with_tick_commands() -> None:
    """Test that TICK commands are properly compiled to Stim format."""
    # Create a simple graph and compile with TICK commands
    graph = GraphState()
    node0 = graph.add_node()
    node1 = graph.add_node()
    node2 = graph.add_node()
    graph.add_edge(node0, node1)
    graph.add_edge(node1, node2)
    qindex = 0
    graph.register_input(node0, qindex)
    graph.register_output(node2, qindex)

    graph.assign_meas_basis(node0, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(node1, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(node2, PlannerMeasBasis(Plane.XY, 0.0))

    flow = {node0: {node1}, node1: {node2}}
    scheduler = Scheduler(graph, flow)
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME)
    scheduler.solve_schedule(config)

    # Compile with scheduler-driven TICK commands (entanglement auto-scheduled by solve_schedule)
    pattern = qompile(graph, flow, scheduler=scheduler)

    # Verify TICK commands are present in pattern
    tick_count = sum(1 for cmd in pattern if isinstance(cmd, TICK))
    assert tick_count > 0, "Pattern should contain TICK commands"
    assert tick_count == scheduler.num_slices(), "Each time slice should yield one TICK command"

    # Compile to Stim format
    stim_str = stim_compile(pattern)

    # Verify TICK instructions are present in Stim output
    assert "TICK" in stim_str, "Stim output should contain TICK instructions"

    # Count TICK instructions in output
    stim_tick_count = stim_str.count("TICK")
    assert stim_tick_count == tick_count, "Number of TICK instructions should match pattern"


def _entanglement_slices_from_pattern(pattern: Pattern) -> dict[tuple[int, int], int]:
    r"""Extract entanglement time slices from a pattern.

    Parameters
    ----------
    pattern : `Pattern`
        The pattern to inspect.

    Returns
    -------
    `dict`\[`tuple`\[`int`, `int`\], `int`\]
        A mapping from entanglement edge ``(u, v)`` to the time slice index, represented as the
        number of preceding `TICK` commands.
    """
    ticks = 0
    entangle_slice: dict[tuple[int, int], int] = {}
    for cmd in pattern:
        if isinstance(cmd, TICK):
            ticks += 1
        elif isinstance(cmd, E):
            entangle_slice[cmd.nodes] = ticks
    return entangle_slice


def _cz_slices_from_stim(stim_str: str) -> dict[tuple[int, int], int]:
    r"""Extract CZ time slices from a stim circuit string.

    Parameters
    ----------
    stim_str : `str`
        The stim circuit string to inspect.

    Returns
    -------
    `dict`\[`tuple`\[`int`, `int`\], `int`\]
        A mapping from CZ targets ``(q1, q2)`` to the time slice index, represented as the number
        of preceding ``TICK`` instructions.
    """
    ticks = 0
    cz_slice: dict[tuple[int, int], int] = {}
    for line in stim_str.splitlines():
        if line == "TICK":
            ticks += 1
        elif line.startswith("CZ "):
            _, q1, q2 = line.split()
            cz_slice[int(q1), int(q2)] = ticks
    return cz_slice


def test_stim_compile_respects_manual_entangle_time() -> None:
    """Manual entanglement times should determine the CZ slice in both Pattern and Stim output."""
    graph = GraphState()
    in_node = graph.add_node()
    mid_node = graph.add_node()
    out_node = graph.add_node()

    graph.register_input(in_node, 0)
    graph.register_output(out_node, 0)

    graph.add_edge(in_node, mid_node)
    graph.add_edge(mid_node, out_node)

    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(mid_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(out_node, PlannerMeasBasis(Plane.XY, 0.0))

    scheduler = Scheduler(graph, {in_node: {mid_node}, mid_node: {out_node}})

    scheduler.manual_schedule(
        prepare_time={mid_node: 0, out_node: 0},
        measure_time={in_node: 3, mid_node: 4, out_node: 5},
        # Intentionally provide edges in reversed order to ensure they are still accepted.
        entangle_time={
            (mid_node, in_node): 2,
            (out_node, mid_node): 1,
        },
    )
    scheduler.validate_schedule()

    pattern = qompile(graph, {in_node: {mid_node}, mid_node: {out_node}}, scheduler=scheduler)
    entangle_slice = _entanglement_slices_from_pattern(pattern)

    assert entangle_slice[in_node, mid_node] == 2
    assert entangle_slice[mid_node, out_node] == 1

    cz_slice = _cz_slices_from_stim(stim_compile(pattern))

    assert cz_slice[in_node, mid_node] == 2
    assert cz_slice[mid_node, out_node] == 1


# ---- Coordinate Tests ----


def test_stim_compile_with_coordinates() -> None:
    """Test that QUBIT_COORDS instructions are emitted for nodes with coordinates."""
    graph = GraphState()
    in_node = graph.add_node(coordinate=(0.0, 0.0))
    mid_node = graph.add_node(coordinate=(1.0, 0.0))
    out_node = graph.add_node(coordinate=(2.0, 0.0))

    graph.register_input(in_node, 0)
    graph.register_output(out_node, 0)

    graph.add_edge(in_node, mid_node)
    graph.add_edge(mid_node, out_node)

    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(mid_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(out_node, PlannerMeasBasis(Plane.XY, 0.0))

    pattern = qompile(graph, {in_node: {mid_node}, mid_node: {out_node}})
    stim_str = stim_compile(pattern)

    # Check QUBIT_COORDS instructions are present
    assert f"QUBIT_COORDS(0.0, 0.0) {in_node}" in stim_str
    assert f"QUBIT_COORDS(1.0, 0.0) {mid_node}" in stim_str
    assert f"QUBIT_COORDS(2.0, 0.0) {out_node}" in stim_str


def test_stim_compile_with_3d_coordinates() -> None:
    """Test that 3D coordinates are correctly emitted."""
    graph = GraphState()
    in_node = graph.add_node(coordinate=(0.0, 0.0, 0.0))
    out_node = graph.add_node(coordinate=(1.0, 1.0, 1.0))

    graph.register_input(in_node, 0)
    graph.register_output(out_node, 0)

    graph.add_edge(in_node, out_node)
    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(out_node, PlannerMeasBasis(Plane.XY, 0.0))

    pattern = qompile(graph, {in_node: {out_node}})
    stim_str = stim_compile(pattern)

    assert f"QUBIT_COORDS(0.0, 0.0, 0.0) {in_node}" in stim_str
    assert f"QUBIT_COORDS(1.0, 1.0, 1.0) {out_node}" in stim_str


def test_stim_compile_without_coordinates() -> None:
    """Test that no QUBIT_COORDS are emitted when emit_qubit_coords is False."""
    graph = GraphState()
    in_node = graph.add_node(coordinate=(0.0, 0.0))
    out_node = graph.add_node(coordinate=(1.0, 0.0))

    graph.register_input(in_node, 0)
    graph.register_output(out_node, 0)

    graph.add_edge(in_node, out_node)
    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(out_node, PlannerMeasBasis(Plane.XY, 0.0))

    pattern = qompile(graph, {in_node: {out_node}})
    stim_str = stim_compile(pattern, emit_qubit_coords=False)

    assert "QUBIT_COORDS" not in stim_str


def test_pattern_coordinates_property() -> None:
    """Test that Pattern.coordinates aggregates coordinates from N commands and input nodes."""
    graph = GraphState()
    in_node = graph.add_node(coordinate=(0.0, 0.0))
    mid_node = graph.add_node(coordinate=(1.0, 0.0))
    out_node = graph.add_node(coordinate=(2.0, 0.0))

    graph.register_input(in_node, 0)
    graph.register_output(out_node, 0)

    graph.add_edge(in_node, mid_node)
    graph.add_edge(mid_node, out_node)

    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(mid_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(out_node, PlannerMeasBasis(Plane.XY, 0.0))

    pattern = qompile(graph, {in_node: {mid_node}, mid_node: {out_node}})

    # Check pattern coordinates
    assert pattern.coordinates[in_node] == (0.0, 0.0)
    assert pattern.coordinates[mid_node] == (1.0, 0.0)
    assert pattern.coordinates[out_node] == (2.0, 0.0)
