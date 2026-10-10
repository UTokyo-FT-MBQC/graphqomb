"""Pauli-specific execution and backwards-compatible Pattern APIs."""

from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest
import stim

from graphqomb import clifford_algebra as ca
from graphqomb.command import TICK
from graphqomb.common import Axis, AxisMeasBasis, Initialization, Plane, PlannerMeasBasis, Sign
from graphqomb.graphstate import GraphState
from graphqomb.pattern import Pattern
from graphqomb.pauli_frame import CliffordFrame, PauliFrame, make_frame
from graphqomb.ptn_format import dumps, loads
from graphqomb.qompiler import qompile
from graphqomb.simulator import PatternSimulator, SimulatorBackend
from graphqomb.stim_glue.compiler import stim_compile
from graphqomb.stim_glue.importer import stim_circuit_to_pattern

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def graph() -> GraphState:
    graph = GraphState()
    for _ in range(3):
        graph.add_node()
    graph.register_input(0, 0)
    graph.register_output(2, 0)
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    for node in (0, 1):
        graph.assign_meas_basis(node, AxisMeasBasis(Axis.X, Sign.PLUS))
    return graph


@pytest.mark.parametrize("frame_type", [PauliFrame, CliffordFrame])
def test_pattern_frame_names_share_instance(graph: GraphState, frame_type: type[PauliFrame]) -> None:
    frame = frame_type(graph, {}, {})
    coordinates: dict[int, tuple[float, ...]] = {0: (1.0, 2.0)}
    initializations = {0: Initialization(Axis.Y)}
    positional = Pattern({0: 0}, {2: 0}, (), frame, coordinates, initializations)
    legacy = Pattern({0: 0}, {2: 0}, (), pauli_frame=frame)
    current = Pattern({0: 0}, {2: 0}, (), frame=frame)
    clifford_alias = Pattern({0: 0}, {2: 0}, (), clifford_frame=frame)

    for pattern in (positional, legacy, current, clifford_alias):
        assert pattern.frame is pattern.pauli_frame is pattern.clifford_frame is frame
    assert positional.input_coordinates is coordinates
    assert positional.input_initializations is initializations
    assert legacy == current == clifford_alias
    assert legacy.input_coordinates is not current.input_coordinates
    assert legacy.input_initializations is not current.input_initializations


def test_legacy_attribute_preserves_clifford_corrections(graph: GraphState) -> None:
    frame = CliffordFrame(graph, {}, {}, cflow={0: {2: ca.S}})
    pattern = Pattern({0: 0}, {2: 0}, (), pauli_frame=frame)
    pattern.pauli_frame.meas_flip(0)

    assert pattern.frame is frame
    assert frame.coset[2] == ca.S
    assert frame.z_pauli[2]


@pytest.mark.parametrize("name", ["pauli_frame", "clifford_frame"])
def test_legacy_frame_keywords_warn_at_call_site(graph: GraphState, name: str) -> None:
    frame = PauliFrame(graph, {}, {})
    with pytest.warns(DeprecationWarning, match=rf"Pattern\({name}=.*v0\.8\.0.*use frame=") as recorded:
        pattern = (
            Pattern({}, {}, (), pauli_frame=frame)
            if name == "pauli_frame"
            else Pattern({}, {}, (), clifford_frame=frame)
        )

    assert pattern.frame is frame
    assert len(recorded) == 1
    assert recorded[0].filename == __file__


def test_canonical_frame_construction_and_replace_do_not_warn(graph: GraphState) -> None:
    frame = PauliFrame(graph, {}, {})
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        positional = Pattern({}, {}, (), frame)
        keyword = Pattern({}, {}, (), frame=frame)
        replaced = dataclasses.replace(keyword, commands=(TICK(),))

    assert positional.frame is keyword.frame is replaced.frame is frame


def test_pattern_requires_frame() -> None:
    with pytest.raises(TypeError, match="correction frame is required"):
        Pattern({}, {}, ())


@pytest.mark.parametrize(
    "names",
    [
        ("frame", "pauli_frame"),
        ("frame", "clifford_frame"),
        ("pauli_frame", "clifford_frame"),
        ("frame", "pauli_frame", "clifford_frame"),
    ],
)
def test_pattern_rejects_conflicting_frame_names(
    graph: GraphState,
    names: tuple[str, ...],
) -> None:
    frame = PauliFrame(graph, {}, {})
    with pytest.raises(TypeError, match="Specify exactly one"):
        Pattern(
            {},
            {},
            (),
            frame=frame if "frame" in names else None,
            pauli_frame=frame if "pauli_frame" in names else None,
            clifford_frame=frame if "clifford_frame" in names else None,
        )


@pytest.mark.parametrize("attribute", ["commands", "frame", "clifford_frame", "pauli_frame"])
def test_pattern_remains_frozen(graph: GraphState, attribute: str) -> None:
    pattern = Pattern({}, {}, (), pauli_frame=PauliFrame(graph, {}, {}))
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(pattern, attribute, ())


def test_dataclass_replace_preserves_frame_alias(graph: GraphState) -> None:
    frame = PauliFrame(graph, {}, {})
    pattern = Pattern({}, {}, (), pauli_frame=frame)
    updated = dataclasses.replace(pattern, commands=(TICK(),))
    replacement = CliffordFrame(graph, {}, {})
    reframed = dataclasses.replace(updated, frame=replacement)

    assert pattern.commands == ()
    assert updated.commands == (TICK(),)
    assert updated.pauli_frame is frame
    assert reframed.frame is reframed.pauli_frame is reframed.clifford_frame is replacement
    assert [field.name for field in dataclasses.fields(pattern)] == [
        "input_node_indices",
        "output_node_indices",
        "commands",
        "frame",
        "input_coordinates",
        "input_initializations",
    ]


@pytest.mark.parametrize("name", ["pauli_frame", "clifford_frame"])
def test_dataclass_replace_legacy_keyword_explains_migration(graph: GraphState, name: str) -> None:
    frame = PauliFrame(graph, {}, {})
    pattern = Pattern({}, {}, (), frame=frame)
    with pytest.raises(TypeError, match=r"For dataclasses\.replace\(\), use frame="):
        (
            dataclasses.replace(pattern, pauli_frame=frame)  # type: ignore[call-arg]
            if name == "pauli_frame"
            else dataclasses.replace(pattern, clifford_frame=frame)  # type: ignore[call-arg]
        )


@pytest.mark.parametrize("coset", ca.TRANSVERSAL)
@pytest.mark.parametrize("pauli", [ca.IDENTITY, ca.X, ca.Y, ca.Z])
def test_make_frame_selects_and_normalizes_all_cliffords(
    graph: GraphState, coset: ca.C1Element, pauli: ca.C1Element
) -> None:
    xflow = {0: {1}, 1: {2}}
    zflow = {0: {2}}
    gate = ca.compose(coset, pauli)
    cflow = {0: {1: gate}}
    frame = make_frame(graph, xflow, zflow, cflow=cflow)
    explicit = CliffordFrame(graph, xflow, zflow, cflow=cflow)

    assert type(frame) is (PauliFrame if coset == ca.IDENTITY else CliffordFrame)
    assert frame.xflow == explicit.xflow
    assert frame.zflow == explicit.zflow
    assert frame.cflow == explicit.cflow
    frame.meas_flip(0)
    explicit.meas_flip(0)
    assert frame.x_pauli == explicit.x_pauli
    assert frame.z_pauli == explicit.z_pauli
    if isinstance(frame, CliffordFrame):
        assert frame.coset == explicit.coset
        assert frame.correction_events == explicit.correction_events
    assert xflow == {0: {1}, 1: {2}}
    assert zflow == {0: {2}}
    assert cflow == {0: {1: gate}}


@pytest.mark.parametrize("cflow", [None, {}, {0: {}}, {0: {1: ca.IDENTITY}}])
def test_make_frame_empty_corrections_keep_metadata(
    graph: GraphState, cflow: dict[int, dict[int, ca.C1Element]] | None
) -> None:
    frame = make_frame(graph, {0: {1}, 1: {2}}, {}, [{0}], {1: {1}}, parity_check_tags=["type=flag"], cflow=cflow)
    assert type(frame) is PauliFrame
    assert frame.cflow == {}
    assert frame.parity_check_group == [{0}]
    assert frame.logical_observables == {1: {1}}
    assert frame.parity_check_tags == ["type=flag"]


def test_pauli_frame_cflow_is_read_only_and_not_stored(graph: GraphState) -> None:
    frame = PauliFrame(graph, {}, {})
    empty = frame.cflow
    empty[0] = {1: ca.S}
    assert frame.cflow == {}
    assert {"cflow", "_cflow", "inv_cflow", "coset", "correction_events"}.isdisjoint(vars(frame))
    with pytest.raises(AttributeError):
        setattr(frame, "cflow", {})  # ruff:ignore[set-attr-with-constant]


def test_stim_import_and_ptn_reload_preserve_empty_cflow_access() -> None:
    pattern = stim_circuit_to_pattern(stim.Circuit("RX 0\nM 0")).pattern
    restored = loads(dumps(pattern))
    for item in (pattern, restored):
        assert type(item.frame) is PauliFrame
        assert len(item.clifford_frame.cflow) == 0
        assert not item.clifford_frame.cflow


@pytest.mark.parametrize("cflow", [None, {}, {0: {}}, {0: {2: ca.IDENTITY}}])
def test_no_clifford_corrections_choose_lightweight_frame(
    graph: GraphState,
    cflow: dict[int, dict[int, ca.C1Element]] | None,
) -> None:
    pattern = qompile(graph, {0: {1}, 1: {2}}, cflow=cflow)
    assert type(pattern.frame) is PauliFrame
    assert pattern.clifford_frame.cflow == {}
    assert {"cflow", "_cflow", "inv_cflow", "coset", "correction_events"}.isdisjoint(vars(pattern.frame))
    restored = loads(dumps(pattern))
    assert type(restored.pauli_frame) is PauliFrame
    assert ".version 2" in dumps(pattern)


@pytest.mark.parametrize("gate", [ca.IDENTITY, ca.X, ca.Y, ca.Z])
def test_pauli_cflow_normalizes_without_mutating_inputs(graph: GraphState, gate: ca.C1Element) -> None:
    xflow = {0: {1}, 1: {2}}
    zflow = {0: {2}}
    cflow = {0: {1: gate}}
    pattern = qompile(graph, xflow, zflow, cflow)
    frame = pattern.pauli_frame

    assert type(frame) is PauliFrame
    assert frame.xflow == {0: set() if gate in {ca.X, ca.Y} else {1}, 1: {2}}
    assert frame.zflow == {0: {1, 2} if gate in {ca.Y, ca.Z} else {2}}
    assert xflow == {0: {1}, 1: {2}}
    assert zflow == {0: {2}}
    assert cflow == {0: {1: gate}}
    frame.meas_flip(0)
    assert frame.x_pauli == {0: False, 1: gate not in {ca.X, ca.Y}, 2: False}
    assert frame.z_pauli == {0: False, 1: gate in {ca.Y, ca.Z}, 2: True}


def test_nontrivial_coset_selects_clifford_frame(graph: GraphState) -> None:
    cflow = {0: {1: ca.compose(ca.S, ca.X)}}
    pattern = qompile(graph, {0: {1}, 1: {2}}, {}, cflow)
    frame = pattern.frame
    assert isinstance(frame, CliffordFrame)
    assert frame.cflow == {0: {1: ca.S}}
    assert frame.xflow == {0: set(), 1: {2}}
    assert cflow == {0: {1: ca.compose(ca.S, ca.X)}}
    restored = loads(dumps(pattern))
    assert isinstance(restored.frame, CliffordFrame)
    assert restored.frame.cflow == frame.cflow
    assert restored.frame.correction_events == frame.correction_events


def test_pauli_and_clifford_frames_agree_on_pauli_operations(graph: GraphState) -> None:
    frames = [
        frame_type(graph, {0: {0, 1}, 1: {2}}, {0: {0, 2}}, [{1}], {0: {1}}, parity_check_tags=["type=flag"])
        for frame_type in (PauliFrame, CliffordFrame)
    ]
    for frame in frames:
        frame.meas_flip(0)
        assert frame.x_pauli == {0: False, 1: True, 2: False}
        assert frame.z_pauli == {0: False, 1: False, 2: True}
        assert frame.children(0) == {1, 2}
        assert frame.parents(1) == {0}
        assert frame.detector_groups() == [{1}]
        assert frame.logical_observable_groups() == {0: {1}}
    pauli, clifford = frames
    assert pauli.detector_stabilizers() == clifford.detector_stabilizers()
    assert pauli.detector_determinism() == clifford.detector_determinism()
    pattern = qompile(
        graph,
        pauli.xflow,
        pauli.zflow,
        parity_check_group=[{1}],
        logical_observables={0: {1}},
        parity_check_tags=["type=flag"],
    )
    other = dataclasses.replace(pattern, frame=clifford)
    assert stim_compile(pattern) == stim_compile(other)
    assert dumps(pattern) == dumps(other)
    assert type(loads(dumps(other)).pauli_frame) is PauliFrame


@pytest.mark.parametrize("plane", list(Plane))
@pytest.mark.parametrize("seed", [2, 7])
def test_pauli_and_clifford_simulation_agree(
    graph: GraphState,
    plane: Plane,
    seed: int,
) -> None:
    for node in (0, 1):
        graph.assign_meas_basis(node, PlannerMeasBasis(plane, 0.31))
    pattern = qompile(graph, {0: {1}, 1: {2}}, {0: {2}})
    other = dataclasses.replace(pattern, frame=CliffordFrame(graph, {0: {1}, 1: {2}}, {0: {2}}))
    simulators = [PatternSimulator(item, SimulatorBackend.StateVector) for item in (pattern, other)]
    for simulator in simulators:
        simulator.simulate(np.random.default_rng(seed))
    assert simulators[0].results == simulators[1].results
    np.testing.assert_allclose(simulators[0].state.state(), simulators[1].state.state(), atol=1e-12)
    assert pattern.pauli_frame.x_pauli == other.frame.x_pauli
    assert pattern.pauli_frame.z_pauli == other.frame.z_pauli


@pytest.mark.parametrize("frame_type", [PauliFrame, CliffordFrame])
def test_direct_frame_construction_checks_invalid_sources(graph: GraphState, frame_type: type[PauliFrame]) -> None:
    with pytest.raises(ValueError, match="Flow source 2 is not measured"):
        frame_type(graph, {2: {1}}, {})
    with pytest.raises(ValueError, match="Cycle detected"):
        frame_type(graph, {0: {1}, 1: {0}}, {})


@pytest.mark.parametrize("constructor", [CliffordFrame, make_frame])
def test_clifford_construction_checks_combined_flow(graph: GraphState, constructor: Callable[..., PauliFrame]) -> None:
    with pytest.raises(ValueError, match="Flow source 2 is not measured"):
        constructor(graph, {}, {}, cflow={2: {1: ca.S}})
    with pytest.raises(ValueError, match="Cycle detected"):
        constructor(graph, {0: {1}}, {}, cflow={1: {0: ca.S}})
    with pytest.raises(ValueError, match="do not commute"):
        constructor(graph, {0: {2}}, {}, cflow={1: {2: ca.S}})


def test_clifford_dependent_chain_checks_indirect_influence(graph: GraphState) -> None:
    graph.assign_meas_basis(2, AxisMeasBasis(Axis.X, Sign.PLUS))
    frame = CliffordFrame(graph, {}, {1: {2}}, parity_check_group=[{2}], cflow={0: {1: ca.S}})
    with pytest.raises(NotImplementedError, match="Node 1 is subject to Clifford feedforward"):
        frame.detector_groups()


@pytest.mark.parametrize("frame_type", [PauliFrame, CliffordFrame])
def test_density_matrix_backend_remains_unsupported(graph: GraphState, frame_type: type[PauliFrame]) -> None:
    pattern = qompile(graph, {0: {1}, 1: {2}})
    pattern = dataclasses.replace(pattern, frame=frame_type(graph, {0: {1}, 1: {2}}, {0: {2}}))
    with pytest.raises(NotImplementedError):
        PatternSimulator(pattern, SimulatorBackend.DensityMatrix)
