"""Tests for ptn_format module."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pytest

from graphqomb import clifford_algebra
from graphqomb.command import TICK, E, M, N
from graphqomb.common import Axis, AxisMeasBasis, Initialization, Plane, PlannerMeasBasis, Sign, determine_pauli_axis
from graphqomb.graphstate import GraphState
from graphqomb.pattern import Pattern
from graphqomb.pauli_frame import PauliFrame
from graphqomb.ptn_format import (
    dump,
    dumps,
    load,
    loads,
)
from graphqomb.qompiler import qompile
from graphqomb.stim_glue.compiler import stim_compile

if TYPE_CHECKING:
    from pathlib import Path

    from graphqomb.command import Command


def create_simple_pattern() -> Pattern:
    """Create a simple pattern for testing.

    Returns
    -------
    Pattern
        A compiled MBQC pattern for testing.
    """
    graph = GraphState()
    in_node = graph.add_node(coordinate=(0.0, 0.0))
    mid_node = graph.add_node(coordinate=(1.0, 0.0))
    out_node = graph.add_node(coordinate=(2.0, 0.0))

    graph.register_input(in_node, 0)
    graph.register_output(out_node, 0)

    graph.add_edge(in_node, mid_node)
    graph.add_edge(mid_node, out_node)

    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(mid_node, PlannerMeasBasis(Plane.XY, math.pi / 2))

    xflow = {in_node: {mid_node}, mid_node: {out_node}}
    return qompile(graph, xflow)


def create_measured_output_pattern_with_observable() -> Pattern:
    """Create a stim-compatible pattern with a logical observable."""
    graph = GraphState()
    in_node = graph.add_node(coordinate=(10.0, 0.0))
    out_node = graph.add_node(coordinate=(20.0, 0.0))

    graph.register_input(in_node, 0)
    graph.register_output(out_node, 0)
    graph.add_edge(in_node, out_node)
    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(out_node, PlannerMeasBasis(Plane.XY, 0.0))

    return qompile(graph, {in_node: {out_node}}, logical_observables={0: {in_node}})


def create_measured_output_pattern_with_detector() -> Pattern:
    """Create a stim-compatible pattern with a detector."""
    graph = GraphState()
    in_node = graph.add_node(coordinate=(30.0, 0.0))
    out_node = graph.add_node(coordinate=(40.0, 0.0))

    graph.register_input(in_node, 0)
    graph.register_output(out_node, 0)
    graph.add_edge(in_node, out_node)
    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(out_node, PlannerMeasBasis(Plane.XY, 0.0))

    return qompile(graph, {in_node: {out_node}}, parity_check_group=[{in_node}])


def command_signature(cmd: Command) -> tuple[Any, ...]:
    """Return a behavior-level signature for a pattern command."""
    if isinstance(cmd, N):
        return ("N", cmd.node, cmd.coordinate)
    if isinstance(cmd, E):
        return ("E", cmd.nodes)
    if isinstance(cmd, M):
        pauli_axis = determine_pauli_axis(cmd.meas_basis)
        if pauli_axis is not None:
            return ("M", cmd.node, pauli_axis, cmd.meas_basis.angle)
        return ("M", cmd.node, cmd.meas_basis.plane, cmd.meas_basis.angle)
    if isinstance(cmd, TICK):
        return ("TICK",)
    return ("UNKNOWN", type(cmd).__name__)


def assert_pattern_equivalent(actual: Pattern, expected: Pattern) -> None:
    """Assert that serialized pattern content survived a roundtrip."""
    assert actual.input_node_indices == expected.input_node_indices
    assert actual.output_node_indices == expected.output_node_indices
    assert actual.input_coordinates == expected.input_coordinates
    assert actual.input_initializations == expected.input_initializations
    assert actual.pauli_frame.xflow == expected.pauli_frame.xflow
    assert actual.pauli_frame.zflow == expected.pauli_frame.zflow
    assert actual.pauli_frame.cflow == expected.pauli_frame.cflow
    assert actual.pauli_frame.parity_check_group == expected.pauli_frame.parity_check_group
    assert actual.pauli_frame.parity_check_tags == expected.pauli_frame.parity_check_tags
    assert actual.pauli_frame.logical_observables == expected.pauli_frame.logical_observables
    assert [command_signature(cmd) for cmd in actual.commands] == [command_signature(cmd) for cmd in expected.commands]


def test_dumps_basic() -> None:
    """Test basic pattern serialization."""
    pattern = create_simple_pattern()
    ptn_str = dumps(pattern)

    assert ".version 2" in ptn_str
    assert ".input" in ptn_str
    assert ".output" in ptn_str
    assert "#======== QUANTUM ========" in ptn_str
    assert "#======== CLASSICAL ========" in ptn_str


def test_dumps_omits_default_input_basis() -> None:
    """Default X-basis input initialization is implicit in .ptn output."""
    pattern = create_simple_pattern()

    assert ".input_basis" not in dumps(pattern)


def test_ptn_roundtrip_preserves_input_initialization_axes() -> None:
    """Non-default input initialization axes survive .ptn roundtrip."""
    graph = GraphState()
    in0 = graph.add_node()
    in1 = graph.add_node()
    out0 = graph.add_node()
    out1 = graph.add_node()

    graph.register_input(in0, 0, init=Initialization(axis=Axis.Y))
    graph.register_input(in1, 1, init=Initialization(axis=Axis.Z))
    graph.register_output(out0, 0)
    graph.register_output(out1, 1)
    graph.add_edge(in0, out0)
    graph.add_edge(in1, out1)
    graph.assign_meas_basis(in0, AxisMeasBasis(Axis.X, Sign.PLUS))
    graph.assign_meas_basis(in1, AxisMeasBasis(Axis.X, Sign.PLUS))

    pattern = qompile(graph, {in0: {out0}, in1: {out1}})
    ptn_str = dumps(pattern)
    loaded = loads(ptn_str)

    assert ".input_basis" in ptn_str
    assert loaded.input_initializations == {in0: Initialization(axis=Axis.Y), in1: Initialization(axis=Axis.Z)}


def test_ptn_loads_legacy_input_without_input_basis_as_x() -> None:
    """Existing .ptn files without .input_basis default inputs to X initialization."""
    ptn_str = """# GraphQOMB Pattern Format v1
.version 1
.input 0:0
.output 1:0

[0]
E 0 1
M 0 X +

.xflow 0 -> 1
"""

    loaded = loads(ptn_str)

    assert loaded.input_initializations == {0: Initialization()}


def test_ptn_load_rejects_input_basis_for_non_input_node() -> None:
    """Input basis directives can only reference input nodes."""
    ptn_str = """# GraphQOMB Pattern Format v2
.version 2
.input 0:0
.input_basis 1:Z
.output 1:0

[0]
E 0 1
M 0 X +

.xflow 0 -> 1
"""

    with pytest.raises(ValueError, match="Input basis specified for non-input node"):
        loads(ptn_str)


def create_tagged_input_pattern(tag: str) -> Pattern:
    """Create a compiled pattern whose single input carries an initialization tag.

    Returns
    -------
    Pattern
        Compiled pattern with a tagged input initialization.
    """
    graph = GraphState()
    in_node = graph.add_node()
    out_node = graph.add_node()
    graph.register_input(in_node, 0, init=Initialization(tag=tag))
    graph.register_output(out_node, 0)
    graph.add_edge(in_node, out_node)
    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.0))

    return qompile(graph, {in_node: {out_node}})


def test_dumps_omits_untagged_input_tag_and_keeps_base_version() -> None:
    """Untagged input initializations stay off the file and off version 4."""
    ptn_str = dumps(create_simple_pattern())

    assert ".input_tag" not in ptn_str
    assert ".version 2" in ptn_str


def test_dumps_writes_input_tag_with_version_4() -> None:
    """A tagged input initialization uses the version 4 grammar."""
    ptn_str = dumps(create_tagged_input_pattern("init_data"))

    assert ".version 4" in ptn_str
    assert ".input_tag[init_data] 0" in ptn_str


@pytest.mark.parametrize("tag", ["init_data", "a]b", "back\\slash", "new\nline", "sp ace#hash"])
def test_ptn_roundtrip_preserves_input_initialization_tags(tag: str) -> None:
    """Input initialization tags survive a .ptn roundtrip, including escapes."""
    pattern = create_tagged_input_pattern(tag)
    ptn_str = dumps(pattern)
    loaded = loads(ptn_str)

    assert loaded.input_initializations == pattern.input_initializations
    assert dumps(loaded) == ptn_str


def test_ptn_load_rejects_input_tag_for_non_input_node() -> None:
    """Input tag directives can only reference input nodes."""
    ptn_str = """# GraphQOMB Pattern Format v4
.version 4
.input 0:0
.input_tag[init] 1
.output 1:0

[0]
E 0 1
M 0 X +

.xflow 0 -> 1
"""

    with pytest.raises(ValueError, match="Input tag specified for non-input node"):
        loads(ptn_str)


def test_ptn_load_rejects_input_tag_in_older_version() -> None:
    """Version 3 .ptn files cannot use the version 4 input-tag directive."""
    ptn_str = """# GraphQOMB Pattern Format v3
.version 3
.input 0:0
.input_tag[init] 0
.output 1:0

[0]
E 0 1
M 0 X +

.xflow 0 -> 1
"""

    with pytest.raises(ValueError, match=r"\.input_tag requires \.ptn version 4"):
        loads(ptn_str)


def test_ptn_load_rejects_input_tag_without_bracket() -> None:
    """The .input_tag directive requires a bracketed tag."""
    ptn_str = """.version 4
.input 0:0
.input_tag 0
.output 1:0

[0]
E 0 1
M 0 X +
"""

    with pytest.raises(ValueError, match=r"\.input_tag requires a bracketed tag"):
        loads(ptn_str)


def test_ptn_load_rejects_unclosed_input_tag() -> None:
    """The .input_tag bracket must be closed."""
    with pytest.raises(ValueError, match=r"\.input_tag tag is missing its closing"):
        loads(".version 4\n.input 0:0\n.input_tag[init 0\n")


def test_ptn_load_rejects_duplicate_input_tag() -> None:
    """A node may carry at most one input tag."""
    ptn_str = """.version 4
.input 0:0
.input_tag[first] 0
.input_tag[second] 0
"""

    with pytest.raises(ValueError, match=r"\.input_tag specified more than once"):
        loads(ptn_str)


def test_ptn_load_keeps_hash_inside_input_tag() -> None:
    """A ``#`` inside the tag bracket is tag text, not a comment."""
    ptn_str = """.version 4
.input 0:0
.input_tag[tag # not comment] 0
.output 1:0

[0]
E 0 1
M 0 X +

.xflow 0 -> 1
"""

    loaded = loads(ptn_str)

    assert loaded.input_initializations == {0: Initialization(tag="tag # not comment")}


def test_ptn_load_rejects_input_basis_in_legacy_version() -> None:
    """Version 1 .ptn files cannot use the version 2 input-basis directive."""
    ptn_str = """# GraphQOMB Pattern Format v1
.version 1
.input 0:0
.input_basis 0:Z
.output 1:0

[0]
E 0 1
M 0 X +

.xflow 0 -> 1
"""

    with pytest.raises(ValueError, match=r"\.input_basis requires \.ptn version 2"):
        loads(ptn_str)


def test_dumps_contains_commands() -> None:
    """Test that dumps includes all command types."""
    pattern = create_simple_pattern()
    ptn_str = dumps(pattern)

    # Check for command types
    assert "N " in ptn_str  # Node creation
    assert "E " in ptn_str  # Entanglement
    assert "M " in ptn_str  # Measurement


def test_dumps_coordinates() -> None:
    """Test that coordinates are correctly serialized."""
    pattern = create_simple_pattern()
    ptn_str = dumps(pattern)

    assert ".coord 0 0.0 0.0" in ptn_str


def test_dumps_pauli_measurements() -> None:
    """Test that Pauli measurements are correctly formatted with +/- signs."""
    pattern = create_simple_pattern()
    ptn_str = dumps(pattern)

    # X measurement (XY plane, angle 0) should be formatted as "X +"
    assert "M 0 X +" in ptn_str
    # Y measurement (XY plane, angle pi/2) should be formatted as "Y +"
    assert "M 1 Y +" in ptn_str


def test_dumps_formats_near_known_angle() -> None:
    """Angles near known constants should serialize canonically."""
    graph = GraphState()
    node = graph.add_node()
    pattern = Pattern(
        input_node_indices={},
        output_node_indices={},
        commands=(M(node, PlannerMeasBasis(Plane.XY, math.pi / 4 + 1e-10)),),
        pauli_frame=PauliFrame(graph, xflow={}, zflow={}),
    )

    assert f"M {node} XY pi/4" in dumps(pattern)


def test_dumps_preserves_xz_plane_x_pauli_sign() -> None:
    """Plane.XZ X measurements should serialize with the correct Pauli sign."""
    graph = GraphState()
    plus_node = graph.add_node()
    minus_node = graph.add_node()
    pattern = Pattern(
        input_node_indices={},
        output_node_indices={},
        commands=(
            M(plus_node, PlannerMeasBasis(Plane.XZ, math.pi / 2)),
            M(minus_node, PlannerMeasBasis(Plane.XZ, 3 * math.pi / 2)),
        ),
        pauli_frame=PauliFrame(graph, xflow={}, zflow={}),
    )

    ptn_str = dumps(pattern)
    result = loads(ptn_str)

    assert f"M {plus_node} X +" in ptn_str
    assert f"M {minus_node} X -" in ptn_str
    measurements = {cmd.node: cmd for cmd in result.commands if isinstance(cmd, M)}
    plus_basis = measurements[plus_node].meas_basis
    minus_basis = measurements[minus_node].meas_basis
    assert isinstance(plus_basis, AxisMeasBasis)
    assert plus_basis.axis == Axis.X
    assert plus_basis.sign == Sign.PLUS
    assert isinstance(minus_basis, AxisMeasBasis)
    assert minus_basis.axis == Axis.X
    assert minus_basis.sign == Sign.MINUS


def test_dumps_preserves_consecutive_trailing_ticks() -> None:
    """Empty final timeslices should preserve consecutive trailing TICK commands."""
    graph = GraphState()
    node = graph.add_node()
    pattern = Pattern(
        input_node_indices={},
        output_node_indices={},
        commands=(N(node), TICK(), TICK()),
        pauli_frame=PauliFrame(graph, xflow={}, zflow={}),
    )

    ptn_str = dumps(pattern)
    result = loads(ptn_str)

    assert f"[0]\nN {node}\n[1]\n[2]\n" in ptn_str
    assert [command_signature(cmd) for cmd in result.commands] == [
        ("N", node, None),
        ("TICK",),
        ("TICK",),
    ]


def test_loads_basic() -> None:
    """Test basic pattern deserialization."""
    ptn_str = """# Test pattern
.version 1
.input 0:0
.output 2:0
.coord 0 0.0 0.0

#======== QUANTUM ========
[0]
N 1
E 0 1
M 0 XY 0

#======== CLASSICAL ========
.xflow 0 -> 1
"""
    result = loads(ptn_str)

    assert result.input_node_indices == {0: 0}
    assert result.output_node_indices == {2: 0}
    assert result.input_coordinates == {0: (0.0, 0.0)}


def test_loads_commands() -> None:
    """Test that commands are correctly parsed."""
    ptn_str = """
.version 1
.input 0:0
.output 2:0

[0]
N 1
N 3 1.0 2.0
E 0 1
M 0 XY 0
M 1 XY pi/2
"""
    result = loads(ptn_str)

    commands = result.commands
    # Check command types
    assert any(isinstance(c, N) and c.node == 1 for c in commands)
    assert any(isinstance(c, N) and c.node == 3 and c.coordinate == (1.0, 2.0) for c in commands)
    assert any(isinstance(c, E) and c.nodes == (0, 1) for c in commands)
    assert any(isinstance(c, M) and c.node == 0 for c in commands)


def test_loads_timeslices() -> None:
    """Test that timeslice markers generate TICK commands."""
    ptn_str = """
.version 1
.input 0:0
.output 1:0

[0]
E 0 1
[1]
M 0 XY 0
[2]
M 1 XY 0
"""
    result = loads(ptn_str)

    # Count TICK commands
    tick_count = sum(1 for c in result.commands if isinstance(c, TICK))
    assert tick_count == 2


def test_loads_initial_empty_timeslices() -> None:
    """Test that an initial non-zero timeslice marker creates the matching TICKs."""
    ptn_str = """
.version 1

[2]
M 0 XY 0
"""
    result = loads(ptn_str)

    assert [command_signature(cmd) for cmd in result.commands] == [
        ("TICK",),
        ("TICK",),
        ("M", 0, Axis.X, 0),
    ]


def test_loads_rejects_timeslice_expansion_bomb() -> None:
    """A tiny input must not be able to allocate an unbounded number of TICK commands."""
    with pytest.raises(ValueError, match="exceeding the remaining budget"):
        loads(".version 1\n[100000000000]\nM 0 XY 0\n")


def test_loads_rejects_cumulative_timeslice_expansion() -> None:
    """Repeated moderate skips must not add up to an unbounded expansion either."""
    ptn_str = ".version 1\n" + "".join(f"[{i * 1000}]\n" for i in range(1, 20))
    with pytest.raises(ValueError, match="exceeding the remaining budget"):
        loads(ptn_str)


def test_loads_allows_empty_timeslices_within_budget() -> None:
    """Skipping ahead by a modest number of empty timeslices still parses."""
    result = loads(".version 1\n[4000]\nM 0 XY 0\n")

    assert sum(1 for cmd in result.commands if isinstance(cmd, TICK)) == 4000


def test_loads_timeslice_budget_scales_with_input_length() -> None:
    """A long input may legitimately carry more TICK commands than the fixed slack."""
    ptn_str = "\n".join([".version 1", *(f"[{i}]" for i in range(6000))]) + "\n"
    result = loads(ptn_str)

    assert sum(1 for cmd in result.commands if isinstance(cmd, TICK)) == 5999


def test_loads_angle_parsing() -> None:
    """Test various angle format parsing."""
    ptn_str = """
.version 1
.input 0:0
.output 5:0

[0]
M 0 XY 0
M 1 XY pi
M 2 XY pi/2
M 3 XY pi/4
M 4 XY 3pi/4
"""
    result = loads(ptn_str)

    measurements = [c for c in result.commands if isinstance(c, M)]
    angles = {m.node: m.meas_basis.angle for m in measurements}

    assert math.isclose(angles[0], 0.0)
    assert math.isclose(angles[1], math.pi)
    assert math.isclose(angles[2], math.pi / 2)
    assert math.isclose(angles[3], math.pi / 4)
    assert math.isclose(angles[4], 3 * math.pi / 4)


def test_loads_pauli_measurements() -> None:
    """Test parsing of Pauli measurement format (X/Y/Z +/-)."""
    ptn_str = """
.version 1
.input 0:0
.output 6:0

[0]
M 0 X +
M 1 X -
M 2 Y +
M 3 Y -
M 4 Z +
M 5 Z -
"""
    result = loads(ptn_str)

    measurements = [c for c in result.commands if isinstance(c, M)]
    assert len(measurements) == 6

    # Check that Pauli measurements are parsed correctly
    m0 = next(m for m in measurements if m.node == 0)
    assert math.isclose(m0.meas_basis.angle, 0.0)  # X +

    m1 = next(m for m in measurements if m.node == 1)
    assert math.isclose(m1.meas_basis.angle, math.pi)  # X -

    m2 = next(m for m in measurements if m.node == 2)
    assert math.isclose(m2.meas_basis.angle, math.pi / 2)  # Y +

    m3 = next(m for m in measurements if m.node == 3)
    assert math.isclose(m3.meas_basis.angle, 3 * math.pi / 2)  # Y -

    m4 = next(m for m in measurements if m.node == 4)
    assert math.isclose(m4.meas_basis.angle, 0.0)  # Z +

    m5 = next(m for m in measurements if m.node == 5)
    assert math.isclose(m5.meas_basis.angle, math.pi)  # Z -


def test_loads_flow_parsing() -> None:
    """Test xflow and zflow parsing."""
    ptn_str = """
.version 1
.input 0:0
.output 2:0

[0]
N 1
E 0 1
M 0 XY 0

.xflow 0 -> 1 2
.zflow 0 -> 3 4
"""
    result = loads(ptn_str)

    assert result.pauli_frame.xflow == {0: {1, 2}}
    assert result.pauli_frame.zflow == {0: {3, 4}}


def test_loads_detector_parsing() -> None:
    """Test detector (parity check group) parsing."""
    ptn_str = """
.version 1
.input 0:0
.output 2:0

[0]
M 0 XY 0

.detector 0 1 2
.detector 3 4
"""
    result = loads(ptn_str)

    assert len(result.pauli_frame.parity_check_group) == 2
    assert result.pauli_frame.parity_check_group[0] == {0, 1, 2}
    assert result.pauli_frame.parity_check_group[1] == {3, 4}


def test_loads_observable_parsing() -> None:
    """Test logical observable parsing."""
    ptn_str = """
.version 1
.input 0:0
.output 1:0

[0]
M 0 X +
M 1 X +

.observable 0 -> 0 1
"""
    result = loads(ptn_str)

    assert result.pauli_frame.logical_observables == {0: {0, 1}}


def test_loads_missing_version() -> None:
    """Test that missing version raises ValueError."""
    ptn_str = """
.input 0:0
.output 1:0
[0]
M 0 XY 0
"""
    with pytest.raises(ValueError, match=r"Missing \.version directive"):
        loads(ptn_str)


def test_loads_unsupported_version() -> None:
    """Test that unsupported version raises ValueError."""
    ptn_str = """
.version 99
.input 0:0
.output 1:0
"""
    with pytest.raises(ValueError, match=r"Unsupported \.ptn version"):
        loads(ptn_str)


def test_loads_unknown_command() -> None:
    """Test that unknown command raises ValueError."""
    ptn_str = """
.version 1
.input 0:0
.output 1:0

[0]
UNKNOWN 0
"""
    with pytest.raises(ValueError, match="Unknown command"):
        loads(ptn_str)


def test_roundtrip() -> None:
    """Test that dumps followed by loads preserves data."""
    pattern = create_simple_pattern()
    ptn_str = dumps(pattern)
    result = loads(ptn_str)

    assert_pattern_equivalent(result, pattern)


def test_dump_and_load_file(tmp_path: Path) -> None:
    """Test file I/O operations."""
    pattern = create_simple_pattern()
    filepath = tmp_path / "test.ptn"

    dump(pattern, filepath)

    assert filepath.exists()

    result = load(filepath)

    assert_pattern_equivalent(result, pattern)


def test_multiple_input_output_qubits() -> None:
    """Test pattern with multiple input/output qubits."""
    graph = GraphState()
    in0 = graph.add_node()
    in1 = graph.add_node()
    out0 = graph.add_node()
    out1 = graph.add_node()

    graph.register_input(in0, 0)
    graph.register_input(in1, 1)
    graph.register_output(out0, 0)
    graph.register_output(out1, 1)

    graph.add_edge(in0, out0)
    graph.add_edge(in1, out1)

    graph.assign_meas_basis(in0, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(in1, PlannerMeasBasis(Plane.XY, 0.0))

    xflow = {in0: {out0}, in1: {out1}}
    pattern = qompile(graph, xflow)

    ptn_str = dumps(pattern)
    result = loads(ptn_str)

    assert len(result.input_node_indices) == 2
    assert len(result.output_node_indices) == 2


def test_3d_coordinates() -> None:
    """Test 3D coordinate serialization and parsing."""
    ptn_str = """
.version 1
.input 0:0
.output 1:0
.coord 0 1.0 2.0 3.0

[0]
N 1 4.0 5.0 6.0
M 0 XY 0
"""
    result = loads(ptn_str)

    assert result.input_coordinates[0] == (1.0, 2.0, 3.0)

    # Check N command coordinate
    n_cmd = next(c for c in result.commands if isinstance(c, N))
    assert n_cmd.coordinate == (4.0, 5.0, 6.0)


def test_different_measurement_planes() -> None:
    """Test all measurement planes are correctly handled."""
    ptn_str = """
.version 1
.input 0:0
.output 3:0

[0]
M 0 XY pi/4
M 1 XZ pi/4
M 2 YZ pi/4
"""
    result = loads(ptn_str)

    measurements = [c for c in result.commands if isinstance(c, M)]
    planes = {m.node: m.meas_basis.plane for m in measurements}

    assert planes[0] == Plane.XY
    assert planes[1] == Plane.XZ
    assert planes[2] == Plane.YZ


def test_empty_flow() -> None:
    """Test pattern with empty flow mappings."""
    ptn_str = """
.version 1
.input 0:0
.output 1:0

[0]
M 0 XY 0

#======== CLASSICAL ========
"""
    result = loads(ptn_str)

    assert result.pauli_frame.xflow == {}
    assert result.pauli_frame.zflow == {}


def test_comments_ignored() -> None:
    """Test that comments are properly ignored."""
    ptn_str = """
# This is a comment
.version 1
# Another comment
.input 0:0  # inline comment should be parsed as part of content
.output 1:0

# Comment in quantum section
[0]
M 0 XY 0
"""
    result = loads(ptn_str)
    assert result.input_node_indices == {0: 0}


def test_roundtrip_preserves_logical_observables_for_stim() -> None:
    """Logical observables should survive .ptn serialization."""
    pattern = create_measured_output_pattern_with_observable()
    ptn_str = dumps(pattern)

    assert ".observable 0 -> 0" in ptn_str

    result = loads(ptn_str)

    assert_pattern_equivalent(result, pattern)
    assert stim_compile(result) == stim_compile(pattern)


def test_roundtrip_preserves_detectors_for_stim() -> None:
    """Detectors should survive .ptn serialization."""
    pattern = create_measured_output_pattern_with_detector()
    ptn_str = dumps(pattern)

    assert ".detector 0" in ptn_str

    result = loads(ptn_str)

    assert_pattern_equivalent(result, pattern)
    assert stim_compile(result) == stim_compile(pattern)


def test_loads_preserves_non_contiguous_node_ids() -> None:
    """Loading should preserve node ids instead of remapping them."""
    ptn_str = """
.version 1
.input 10:0
.output 30:0
.coord 10 1.0 2.0

[0]
N 20 3.0 4.0
E 10 20
E 20 30
[1]
M 10 X +
[2]
M 20 Y -

.xflow 10 -> 20
.zflow 20 -> 30
"""
    result = loads(ptn_str)

    assert result.input_node_indices == {10: 0}
    assert result.output_node_indices == {30: 0}
    assert result.pauli_frame.graphstate.nodes == {10, 20, 30}
    assert result.pauli_frame.graphstate.edges == {(10, 20), (20, 30)}
    assert result.pauli_frame.graphstate.has_node(20)
    assert result.pauli_frame.graphstate.has_edge(30, 20)
    assert result.pauli_frame.graphstate.number_of_nodes() == 3
    assert result.pauli_frame.graphstate.number_of_edges() == 2
    with pytest.raises(NotImplementedError, match="read-only"):
        result.pauli_frame.graphstate.add_node()
    with pytest.raises(NotImplementedError, match="read-only"):
        result.pauli_frame.graphstate.add_edge(10, 30)
    assert any(isinstance(cmd, N) and cmd.node == 20 for cmd in result.commands)


@pytest.mark.parametrize(
    ("ptn_str", "message"),
    [
        (".version 1\n.foo whatever\n", "Unknown directive"),
        (".version 1\n[0]\nM 0 X bad\n", "Invalid Pauli measurement sign"),
        (".version 1\n[0]\nM 0 X + junk\n", "M command requires"),
        (".version 1\n.xflow 0 1\n", "must contain exactly one"),
        (".version 1\n[-1]\n", "Timeslice must be non-negative"),
        (".version 1\n[1]\n[0]\n", "monotonically increasing"),
        (".version 1\n[0]\nM 0 XY pi/0\n", "denominator"),
        (".version 1\n.detector\n", "requires at least one node"),
        (".version 1\n[0]\nX 0\n", "Unknown command: X"),
        (".version 1\n[0]\nZ 0\n", "Unknown command: Z"),
    ],
)
def test_loads_rejects_malformed_input(ptn_str: str, message: str) -> None:
    """Malformed .ptn input should fail instead of being guessed."""
    with pytest.raises(ValueError, match=message):
        loads(ptn_str)


def create_measured_output_pattern_with_tagged_detector(tag: str) -> Pattern:
    """Create a stim-compatible pattern whose detector carries a tag."""
    graph = GraphState()
    in_node = graph.add_node(coordinate=(50.0, 0.0))
    out_node = graph.add_node(coordinate=(60.0, 0.0))

    graph.register_input(in_node, 0)
    graph.register_output(out_node, 0)
    graph.add_edge(in_node, out_node)
    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(out_node, PlannerMeasBasis(Plane.XY, 0.0))

    return qompile(graph, {in_node: {out_node}}, parity_check_group=[{in_node}], parity_check_tags=[tag])


def test_dumps_writes_detector_tag() -> None:
    pattern = create_measured_output_pattern_with_tagged_detector("type=flag")
    ptn_str = dumps(pattern)

    assert any(line.startswith(".detector[type=flag] ") for line in ptn_str.splitlines())
    assert ".version 3" in ptn_str


def test_dumps_keeps_version_2_without_detector_tags() -> None:
    """Tag-free files keep the version 2 header for external version 2 parsers."""
    pattern = create_measured_output_pattern_with_detector()
    ptn_str = dumps(pattern)

    assert ".version 2" in ptn_str
    assert ".version 3" not in ptn_str


def test_dumps_keeps_version_2_for_tag_on_empty_group() -> None:
    """A tag on an empty (unwritten) group does not force the version 3 header."""
    pattern = create_simple_pattern()
    pattern.pauli_frame.parity_check_group.append(set())
    pattern.pauli_frame.parity_check_tags.append("type=flag")

    ptn_str = dumps(pattern)

    assert ".version 2" in ptn_str


@pytest.mark.parametrize("tag", ["type=flag", "a]b", "back\\slash", "sp ace#hash"])
def test_ptn_roundtrip_preserves_detector_tags(tag: str) -> None:
    pattern = create_measured_output_pattern_with_tagged_detector(tag)
    loaded = loads(dumps(pattern))

    assert_pattern_equivalent(loaded, pattern)
    assert loaded.pauli_frame.parity_check_tags == [tag]


def test_loads_detector_tag_parsing() -> None:
    ptn_str = """
.version 3
.input 0:0
.output 1:0
[0]
N 0
N 1
E 0 1
M 0 XY 0
.detector[type=flag] 0 # trailing comment
.detector[sp ace#hash] 0
.detector[a\\Cb\\Bc] 0
.detector 0
"""
    pattern = loads(ptn_str)

    assert pattern.pauli_frame.parity_check_tags == ["type=flag", "sp ace#hash", "a]b\\c", ""]
    assert pattern.pauli_frame.parity_check_group == [{0}, {0}, {0}, {0}]


def test_loads_rejects_unclosed_detector_tag() -> None:
    ptn_str = """
.version 2
.input 0:0
.output 0:0
.detector[type=flag 0
"""
    with pytest.raises(ValueError, match=r"missing its closing"):
        loads(ptn_str)


def test_loads_rejects_glued_detector_tag() -> None:
    ptn_str = """
.version 2
.input 0:0
.output 0:0
.detector[type=flag]0
"""
    with pytest.raises(ValueError, match=r"must be followed by whitespace"):
        loads(ptn_str)


def test_loads_rejects_unknown_detector_like_directive() -> None:
    ptn_str = """
.version 2
.input 0:0
.output 0:0
.detectorx 0
"""
    with pytest.raises(ValueError, match=r"Unknown directive: \.detectorx"):
        loads(ptn_str)


@pytest.mark.parametrize("version", [1, 2])
def test_loads_rejects_detector_tag_in_older_version(version: int) -> None:
    ptn_str = f"""
.version {version}
.input 0:0
.output 1:0
[0]
N 0
N 1
E 0 1
M 0 XY 0
.detector[type=flag] 0
"""
    with pytest.raises(ValueError, match=r"\.detector tags require \.ptn version 3"):
        loads(ptn_str)


@pytest.mark.parametrize("version", [1, 2])
def test_loads_rejects_empty_detector_tag_in_older_version(version: int) -> None:
    ptn_str = f"""
.version {version}
.input 0:0
.output 1:0
[0]
N 0
N 1
E 0 1
M 0 XY 0
.detector[] 0
"""
    with pytest.raises(ValueError, match=r"\.detector tags require \.ptn version 3"):
        loads(ptn_str)


def test_loads_accepts_empty_detector_tag_in_version_3() -> None:
    ptn_str = """
.version 3
.input 0:0
.output 1:0
[0]
N 0
N 1
E 0 1
M 0 XY 0
.detector[] 0
"""
    pattern = loads(ptn_str)
    assert list(pattern.pauli_frame.parity_check_tags) == [""]


def _pattern_with_cflow() -> Pattern:
    graph = GraphState()
    n0 = graph.add_node()
    n1 = graph.add_node()
    n2 = graph.add_node()
    graph.register_input(n0, 0)
    graph.register_output(n2, 0)
    graph.add_edge(n0, n1)
    graph.add_edge(n1, n2)
    graph.assign_meas_basis(n0, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(n1, PlannerMeasBasis(Plane.XY, 0.0))
    return qompile(graph, xflow={n0: {n1}, n1: {n2}}, cflow={n0: {n1: clifford_algebra.S, n2: clifford_algebra.HSH}})


def test_cflow_roundtrip() -> None:
    pattern = _pattern_with_cflow()
    ptn_str = dumps(pattern)
    assert ".version 5" in ptn_str
    assert ".cflow 0 -> 1:S 2:HSH" in ptn_str

    result = loads(ptn_str)
    assert_pattern_equivalent(result, pattern)
    assert result.pauli_frame.cflow == pattern.pauli_frame.cflow


def test_cflow_requires_version_5() -> None:
    ptn_str = """
.version 4
.input 0:0
.output 1:0
[0]
N 1
E 0 1
M 0 XY 0
.cflow 0 -> 1:S
"""
    with pytest.raises(ValueError, match=r"\.cflow requires \.ptn version 5"):
        loads(ptn_str)


def test_cflow_rejects_unknown_coset_name() -> None:
    ptn_str = """
.version 5
.input 0:0
.output 1:0
[0]
N 1
E 0 1
M 0 XY 0
.cflow 0 -> 1:T
"""
    with pytest.raises(ValueError, match=r"Invalid \.cflow coset name"):
        loads(ptn_str)


def test_cflow_rejects_malformed_pair() -> None:
    ptn_str = """
.version 5
.input 0:0
.output 1:0
[0]
N 1
E 0 1
M 0 XY 0
.cflow 0 -> 1
"""
    with pytest.raises(ValueError, match="Invalid target:coset pair"):
        loads(ptn_str)
