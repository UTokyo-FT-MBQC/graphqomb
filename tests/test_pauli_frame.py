"""Tests for pauli_frame module."""

from __future__ import annotations

import itertools
import math

import pytest

from graphqomb.common import Axis, AxisMeasBasis, Plane, PlannerMeasBasis, Sign
from graphqomb.graphstate import GraphState
from graphqomb.pauli_frame import PauliFrame


@pytest.fixture
def simple_graph_with_flows() -> tuple[GraphState, dict[int, set[int]], dict[int, set[int]]]:
    """Create a simple graph with X and Z flows for testing.

    Returns
    -------
    tuple[GraphState, dict[int, set[int]], dict[int, set[int]]]
        GraphState, xflow, and zflow
    """
    graph = GraphState()
    n0 = graph.add_node()
    n1 = graph.add_node()
    n2 = graph.add_node()

    q_idx = 0
    graph.register_input(n0, q_idx)
    graph.register_output(n2, q_idx)

    graph.add_edge(n0, n1)
    graph.add_edge(n1, n2)

    # Z measurement on n0, X measurement on n1
    graph.assign_meas_basis(n0, PlannerMeasBasis(Plane.XZ, 0.0))
    graph.assign_meas_basis(n1, PlannerMeasBasis(Plane.XY, 0.0))

    xflow = {n0: {n1}, n1: {n2}}
    zflow = {n0: {n0}}

    return graph, xflow, zflow


@pytest.fixture
def simple_pauli_frame(
    simple_graph_with_flows: tuple[GraphState, dict[int, set[int]], dict[int, set[int]]],
) -> PauliFrame:
    """Create a simple PauliFrame for testing.

    Parameters
    ----------
    simple_graph_with_flows : tuple[GraphState, dict[int, set[int]], dict[int, set[int]]]
        Graph, xflow, and zflow from fixture

    Returns
    -------
    PauliFrame
        A simple PauliFrame instance
    """
    graph, xflow, zflow = simple_graph_with_flows
    # Provide parity_check_group to enable _pauli_axis_cache initialization
    parity_check_group = [set(graph.nodes)]
    return PauliFrame(graph, xflow, zflow, parity_check_group)


@pytest.fixture
def simple_nodes(simple_graph_with_flows: tuple[GraphState, dict[int, set[int]], dict[int, set[int]]]) -> list[int]:
    """Get list of nodes from simple graph.

    Parameters
    ----------
    simple_graph_with_flows : tuple[GraphState, dict[int, set[int]], dict[int, set[int]]]
        Graph, xflow, and zflow from fixture

    Returns
    -------
    list[int]
        List of node IDs
    """
    graph, _, _ = simple_graph_with_flows
    return list(graph.nodes)


@pytest.fixture
def x_axis_pauli_frame() -> PauliFrame:
    """Create a PauliFrame with X axis measurements.

    Returns
    -------
    PauliFrame
        PauliFrame with X axis measurements
    """
    graph = GraphState()
    n0 = graph.add_node()
    n1 = graph.add_node()
    n2 = graph.add_node()

    graph.register_input(n0, 0)
    graph.register_output(n2, 0)

    graph.add_edge(n0, n1)
    graph.add_edge(n1, n2)

    # X measurement (XY plane, angle 0)
    graph.assign_meas_basis(n0, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(n1, PlannerMeasBasis(Plane.XY, 0.0))

    xflow = {n0: {n1}, n1: {n2}}
    zflow: dict[int, set[int]] = {}

    # Provide parity_check_group to enable _pauli_axis_cache initialization
    parity_check_group = [set(graph.nodes)]
    return PauliFrame(graph, xflow, zflow, parity_check_group)


@pytest.fixture
def y_axis_pauli_frame() -> PauliFrame:
    """Create a PauliFrame with Y axis measurements.

    Returns
    -------
    PauliFrame
        PauliFrame with Y axis measurements
    """
    graph = GraphState()
    n0 = graph.add_node()
    n1 = graph.add_node()
    n2 = graph.add_node()

    graph.register_input(n0, 0)
    graph.register_output(n2, 0)

    graph.add_edge(n0, n1)
    graph.add_edge(n1, n2)

    # Y measurement (XY plane, angle pi/2)
    graph.assign_meas_basis(n0, PlannerMeasBasis(Plane.XY, math.pi / 2))
    graph.assign_meas_basis(n1, PlannerMeasBasis(Plane.XY, math.pi / 2))

    xflow = {n0: {n1}, n1: {n2}}
    zflow = {n0: {n0}}

    # Provide parity_check_group to enable _pauli_axis_cache initialization
    parity_check_group = [set(graph.nodes)]
    return PauliFrame(graph, xflow, zflow, parity_check_group)


@pytest.fixture
def z_axis_pauli_frame() -> PauliFrame:
    """Create a PauliFrame with Z axis measurements.

    Returns
    -------
    PauliFrame
        PauliFrame with Z axis measurements
    """
    graph = GraphState()
    n0 = graph.add_node()
    n1 = graph.add_node()
    n2 = graph.add_node()

    graph.register_input(n0, 0)
    graph.register_output(n2, 0)

    graph.add_edge(n0, n1)
    graph.add_edge(n1, n2)

    # Z measurement (XZ plane, angle 0)
    graph.assign_meas_basis(n0, PlannerMeasBasis(Plane.XZ, 0.0))
    graph.assign_meas_basis(n1, PlannerMeasBasis(Plane.XZ, 0.0))

    xflow = {n0: {n1}, n1: {n2}}
    zflow: dict[int, set[int]] = {}

    # Provide parity_check_group to enable _pauli_axis_cache initialization
    parity_check_group = [set(graph.nodes)]
    return PauliFrame(graph, xflow, zflow, parity_check_group)


def test_x_flip(simple_pauli_frame: PauliFrame, simple_nodes: list[int]) -> None:
    """Test X Pauli flip operation."""
    pframe = simple_pauli_frame
    node = simple_nodes[0]

    # Initially False
    assert pframe.x_pauli[node] is False

    # Flip once
    pframe.x_flip(node)
    assert pframe.x_pauli[node] is True

    # Flip again
    pframe.x_flip(node)
    assert pframe.x_pauli[node] is False


def test_z_flip(simple_pauli_frame: PauliFrame, simple_nodes: list[int]) -> None:
    """Test Z Pauli flip operation."""
    pframe = simple_pauli_frame
    node = simple_nodes[0]

    # Initially False
    assert pframe.z_pauli[node] is False

    # Flip once
    pframe.z_flip(node)
    assert pframe.z_pauli[node] is True

    # Flip again
    pframe.z_flip(node)
    assert pframe.z_pauli[node] is False


def test_meas_flip(simple_pauli_frame: PauliFrame, simple_nodes: list[int]) -> None:
    """Test measurement flip operation updates Pauli frame correctly."""
    pframe = simple_pauli_frame
    n0, n1, n2 = simple_nodes[0], simple_nodes[1], simple_nodes[2]

    # Initially all False
    assert pframe.x_pauli[n1] is False
    assert pframe.z_pauli[n0] is False

    # Flip n0: should affect xflow[n0] = {n1} and zflow[n0] = {n0}
    pframe.meas_flip(n0)
    assert pframe.x_pauli[n1] is True  # n1 is in xflow[n0]
    assert pframe.z_pauli[n0] is True  # n0 is in zflow[n0]

    # Flip n1: should affect xflow[n1] = {n2}
    pframe.meas_flip(n1)
    assert pframe.x_pauli[n2] is True  # n2 is in xflow[n1]


def test_children(simple_pauli_frame: PauliFrame, simple_nodes: list[int]) -> None:
    """Test getting children of a node in the Pauli frame."""
    pframe = simple_pauli_frame
    n0, n1, n2 = simple_nodes[0], simple_nodes[1], simple_nodes[2]

    # n0 has children from xflow and zflow (excluding itself)
    children_n0 = pframe.children(n0)
    assert n1 in children_n0  # from xflow[n0]
    assert n0 not in children_n0  # self is excluded

    # n1 has child n2 from xflow
    children_n1 = pframe.children(n1)
    assert n2 in children_n1


def test_parents(simple_pauli_frame: PauliFrame, simple_nodes: list[int]) -> None:
    """Test getting parents of a node in the Pauli frame."""
    pframe = simple_pauli_frame
    n0, n1, n2 = simple_nodes[0], simple_nodes[1], simple_nodes[2]

    # n1 has parent n0 from inv_xflow
    parents_n1 = pframe.parents(n1)
    assert n0 in parents_n1
    assert n1 not in parents_n1

    # n2 has parent n1 from inv_xflow
    parents_n2 = pframe.parents(n2)
    assert n1 in parents_n2


def test_pauli_axis_cache_initialization(simple_pauli_frame: PauliFrame, simple_nodes: list[int]) -> None:
    """Test that Pauli axis cache is correctly initialized."""
    pframe = simple_pauli_frame
    n0, n1 = simple_nodes[0], simple_nodes[1]

    # n0 has Z measurement (XZ plane, angle 0)
    assert n0 in pframe._pauli_axis_cache
    assert pframe._pauli_axis_cache[n0] == Axis.Z

    # n1 has X measurement (XY plane, angle 0)
    assert n1 in pframe._pauli_axis_cache
    assert pframe._pauli_axis_cache[n1] == Axis.X


def test_pauli_axis_cache_not_initialized_without_observables(
    simple_graph_with_flows: tuple[GraphState, dict[int, set[int]], dict[int, set[int]]],
) -> None:
    """Cache should stay empty when neither parity checks nor logical observables are provided."""
    graph, xflow, zflow = simple_graph_with_flows
    pframe = PauliFrame(graph, xflow, zflow)

    assert pframe._pauli_axis_cache == {}


def test_chain_cache_memoization(simple_pauli_frame: PauliFrame, simple_nodes: list[int]) -> None:
    """Test that chain cache memoization works correctly."""
    pframe = simple_pauli_frame
    n1 = simple_nodes[1]

    # First call should compute and cache
    chain1 = pframe._collect_dependent_chain(n1)
    assert n1 in pframe._chain_cache

    # Second call should return cached result
    chain2 = pframe._collect_dependent_chain(n1)
    assert chain1 == chain2

    # Verify cache hit by checking the cached value
    cached_value = pframe._chain_cache[n1]
    assert set(cached_value) == chain1


def test_collect_dependent_chain_x_axis(x_axis_pauli_frame: PauliFrame) -> None:
    """Test dependent chain collection for X axis measurement."""
    pframe = x_axis_pauli_frame
    # Node 1 is the second node in the graph
    nodes = list(pframe.graphstate.nodes)
    n1 = nodes[1]

    # For X axis, parents come from inv_zflow
    chain = pframe._collect_dependent_chain(n1)
    assert isinstance(chain, set)
    assert n1 in chain


def test_collect_dependent_chain_y_axis(y_axis_pauli_frame: PauliFrame) -> None:
    """Test dependent chain collection for Y axis measurement."""
    pframe = y_axis_pauli_frame
    # Node 1 is the second node in the graph
    nodes = list(pframe.graphstate.nodes)
    n1 = nodes[1]

    # For Y axis, parents come from symmetric difference of inv_xflow and inv_zflow
    chain = pframe._collect_dependent_chain(n1)
    assert isinstance(chain, set)
    assert n1 in chain


def test_collect_dependent_chain_z_axis(z_axis_pauli_frame: PauliFrame) -> None:
    """Test dependent chain collection for Z axis measurement."""
    pframe = z_axis_pauli_frame
    # Node 1 is the second node in the graph
    nodes = list(pframe.graphstate.nodes)
    n1 = nodes[1]

    # For Z axis, parents come from inv_xflow
    chain = pframe._collect_dependent_chain(n1)
    assert isinstance(chain, set)
    assert n1 in chain


def test_detector_groups() -> None:
    """Test detector groups generation with parity check groups."""
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
    graph.assign_meas_basis(n2, PlannerMeasBasis(Plane.XY, 0.0))

    xflow = {n0: {n1}, n1: {n2}}
    zflow: dict[int, set[int]] = {n0: {n0, n2}}
    parity_check_group = [{n1}, {n1, n2}]

    pframe = PauliFrame(graph, xflow, zflow, parity_check_group)

    # Get detector groups
    groups = pframe.detector_groups()
    assert isinstance(groups, list)
    assert len(groups) == 2
    for group in groups:
        assert isinstance(group, set)

    assert groups[0] == {n1}
    assert groups[1] == {n0, n1, n2}  # n0 is included via dependent chain


def test_detector_stabilizer_and_determinism() -> None:
    """A detector stabilizer is the product of graph-state stabilizers."""
    graph = GraphState()
    center = graph.add_node()
    unmeasured_output = graph.add_node()

    graph.register_output(unmeasured_output, 0)
    graph.add_edge(center, unmeasured_output)
    graph.assign_meas_basis(center, AxisMeasBasis(Axis.X, Sign.PLUS))

    pframe = PauliFrame(graph, xflow={}, zflow={}, parity_check_group=[{center}])

    assert pframe.detector_stabilizers() == [
        {
            center: Axis.X,
            unmeasured_output: Axis.Z,
        }
    ]
    assert pframe.detector_determinism() == [True]

    # Once the output is measured, its basis participates in the comparison.
    graph.assign_meas_basis(unmeasured_output, AxisMeasBasis(Axis.X, Sign.PLUS))
    assert pframe.detector_determinism() == [False]


def test_detector_determinism_requires_exact_support() -> None:
    """A matching measurement outside the detector group cannot extend its product."""
    graph = GraphState()
    detector_node = graph.add_node()
    outside_node = graph.add_node()
    graph.add_edge(detector_node, outside_node)
    graph.assign_meas_basis(detector_node, AxisMeasBasis(Axis.X, Sign.PLUS))
    graph.assign_meas_basis(outside_node, AxisMeasBasis(Axis.X, Sign.PLUS))

    pframe = PauliFrame(graph, xflow={}, zflow={}, parity_check_group=[{detector_node}])

    assert pframe.detector_stabilizers() == [{detector_node: Axis.X, outside_node: Axis.Z}]
    assert pframe.detector_determinism() == [False]


def test_z_measurement_supported_by_neighbor_stabilizer() -> None:
    """A Z-measured node draws its Z support from the graph stabilizers of its neighbors."""
    graph = GraphState()
    x_node = graph.add_node()
    z_node = graph.add_node()
    graph.add_edge(x_node, z_node)
    graph.assign_meas_basis(x_node, AxisMeasBasis(Axis.X, Sign.PLUS))
    graph.assign_meas_basis(z_node, AxisMeasBasis(Axis.Z, Sign.MINUS))

    pframe = PauliFrame(graph, xflow={}, zflow={}, parity_check_group=[{x_node, z_node}])

    assert pframe.detector_stabilizers() == [{x_node: Axis.X, z_node: Axis.Z}]
    assert pframe.detector_determinism() == [True]


def test_z_measurement_alone_on_plus_state_is_not_deterministic() -> None:
    """Z on a plus-state node has no single-qubit Z stabilizer, so it stays random."""
    graph = GraphState()
    x_node = graph.add_node()
    z_node = graph.add_node()
    graph.add_edge(x_node, z_node)
    graph.assign_meas_basis(x_node, AxisMeasBasis(Axis.X, Sign.PLUS))
    graph.assign_meas_basis(z_node, AxisMeasBasis(Axis.Z, Sign.PLUS))

    pframe = PauliFrame(graph, xflow={}, zflow={}, parity_check_group=[{z_node}])

    assert pframe.detector_stabilizers() == [{}]
    assert pframe.detector_determinism() == [False]


def test_z_measured_node_z_parity_must_not_cancel_out() -> None:
    """Neighbor Z supports canceling on a Z-measured node leave its Z uncertified."""
    graph = GraphState()
    left = graph.add_node()
    center = graph.add_node()
    right = graph.add_node()
    graph.add_edge(left, center)
    graph.add_edge(center, right)
    graph.assign_meas_basis(left, AxisMeasBasis(Axis.X, Sign.PLUS))
    graph.assign_meas_basis(center, AxisMeasBasis(Axis.Z, Sign.PLUS))
    graph.assign_meas_basis(right, AxisMeasBasis(Axis.X, Sign.PLUS))

    pframe = PauliFrame(
        graph,
        xflow={},
        zflow={},
        parity_check_group=[{left, center, right}, {left, center}],
    )

    # X(left) X(right) cancels the Z supports on the center, leaving its Z
    # measurement unmatched, while X(left) Z(center) is the graph stabilizer
    # of the left node.
    assert pframe.detector_determinism() == [False, True]


def test_detector_stabilizer_multiplication() -> None:
    """Overlapping X and Z support multiplies to Y up to phase."""
    graph = GraphState()
    n0 = graph.add_node()
    n1 = graph.add_node()
    graph.add_edge(n0, n1)
    graph.assign_meas_basis(n0, AxisMeasBasis(Axis.Y, Sign.PLUS))
    graph.assign_meas_basis(n1, AxisMeasBasis(Axis.Y, Sign.MINUS))

    pframe = PauliFrame(graph, xflow={}, zflow={}, parity_check_group=[{n0, n1}])

    assert pframe.detector_stabilizers() == [{n0: Axis.Y, n1: Axis.Y}]
    assert pframe.detector_determinism() == [True]

    graph.assign_meas_basis(n1, AxisMeasBasis(Axis.X, Sign.PLUS))
    assert pframe.detector_determinism() == [False]


@pytest.mark.parametrize(
    ("init_axis", "expected_stabilizer", "expected_determinism"),
    [
        # Measuring X or Y on a node entangled with a neighbor is random: the
        # entangling edge keeps Z support on the Z-measured |+> neighbor.
        (Axis.X, {0: Axis.X, 1: Axis.Z}, False),
        (Axis.Y, {0: Axis.Y, 1: Axis.Z}, False),
        # A Z-initialized input keeps its single-qubit Z stabilizer.
        (Axis.Z, {0: Axis.Z}, True),
    ],
)
def test_input_detector_stabilizer(
    init_axis: Axis,
    expected_stabilizer: dict[int, Axis],
    expected_determinism: bool,
) -> None:
    """Input detector stabilizers respect their preparation axes."""
    graph = GraphState()
    input_node = graph.add_node()
    neighbor = graph.add_node()
    graph.register_input(input_node, 0, init_axis=init_axis)
    graph.add_edge(input_node, neighbor)
    graph.assign_meas_basis(input_node, AxisMeasBasis(init_axis, Sign.PLUS))
    graph.assign_meas_basis(neighbor, AxisMeasBasis(Axis.Z, Sign.PLUS))

    pframe = PauliFrame(graph, xflow={}, zflow={}, parity_check_group=[{input_node}])

    assert (input_node, neighbor) == (0, 1)
    assert pframe.detector_stabilizers() == [expected_stabilizer]
    assert pframe.detector_determinism() == [expected_determinism]


def test_z_initialization_prevents_incident_support_cancellation() -> None:
    """Removing Z-initialized nodes from neighbor sets prevents support cancellation."""
    graph = GraphState()
    z_input = graph.add_node()
    neighbor = graph.add_node()
    graph.register_input(z_input, 0, init_axis=Axis.Z)
    graph.add_edge(z_input, neighbor)
    graph.assign_meas_basis(z_input, AxisMeasBasis(Axis.Z, Sign.PLUS))
    graph.assign_meas_basis(neighbor, AxisMeasBasis(Axis.X, Sign.PLUS))

    pframe = PauliFrame(graph, xflow={}, zflow={}, parity_check_group=[{z_input, neighbor}])

    assert pframe.detector_stabilizers() == [{z_input: Axis.Z, neighbor: Axis.X}]
    assert pframe.detector_determinism() == [True]


def test_input_initialization_can_make_detector_non_deterministic() -> None:
    """A detector is non-deterministic when its measurement disagrees with input preparation."""
    graph = GraphState()
    z_input = graph.add_node()
    graph.register_input(z_input, 0, init_axis=Axis.Z)
    graph.assign_meas_basis(z_input, AxisMeasBasis(Axis.X, Sign.PLUS))

    pframe = PauliFrame(graph, xflow={}, zflow={}, parity_check_group=[{z_input}])

    assert pframe.detector_stabilizers() == [{z_input: Axis.Z}]
    assert pframe.detector_determinism() == [False]


def test_z_initialized_neighbor_leaves_measurement_deterministic() -> None:
    """A CZ edge to a Z-initialized node does not entangle, so X on the neighbor stays certain."""
    graph = GraphState()
    z_input = graph.add_node()
    neighbor = graph.add_node()
    graph.register_input(z_input, 0, init_axis=Axis.Z)
    graph.add_edge(z_input, neighbor)
    graph.assign_meas_basis(z_input, AxisMeasBasis(Axis.X, Sign.PLUS))
    graph.assign_meas_basis(neighbor, AxisMeasBasis(Axis.X, Sign.PLUS))

    pframe = PauliFrame(graph, xflow={}, zflow={}, parity_check_group=[{neighbor}])

    assert pframe.detector_stabilizers() == [{neighbor: Axis.X}]
    assert pframe.detector_determinism() == [True]


def _assert_determinism_matches_stim(
    edges: tuple[tuple[int, int], ...],
    init_axes: tuple[Axis, ...],
    meas_axes: tuple[Axis, ...],
    groups: list[set[int]],
) -> None:
    """Compare `detector_determinism` with the stim expectation value oracle."""
    stim = pytest.importorskip("stim")
    reset_instr = {Axis.X: "RX", Axis.Y: "RY", Axis.Z: "R"}

    graph = GraphState()
    nodes = [graph.add_node() for _ in init_axes]
    for qubit, init_axis in enumerate(init_axes):
        if init_axis is not Axis.X:
            graph.register_input(nodes[qubit], qubit, init_axis=init_axis)
    for node0, node1 in edges:
        graph.add_edge(nodes[node0], nodes[node1])
    for qubit, meas_axis in enumerate(meas_axes):
        graph.assign_meas_basis(nodes[qubit], AxisMeasBasis(meas_axis, Sign.PLUS))

    pframe = PauliFrame(
        graph,
        xflow={},
        zflow={},
        parity_check_group=[{nodes[qubit] for qubit in group} for group in groups],
    )

    resets = "\n".join(f"{reset_instr[init_axis]} {qubit}" for qubit, init_axis in enumerate(init_axes))
    entanglements = "\n".join(f"CZ {node0} {node1}" for node0, node1 in edges)
    simulator = stim.TableauSimulator()
    simulator.do(stim.Circuit(f"{resets}\n{entanglements}"))

    for group, claimed in zip(groups, pframe.detector_determinism(), strict=True):
        pauli = ["_"] * len(init_axes)
        for qubit in group:
            pauli[qubit] = meas_axes[qubit].name
        expectation = simulator.peek_observable_expectation(stim.PauliString("".join(pauli)))
        assert claimed == (expectation != 0), (
            f"init={init_axes} meas={meas_axes} group={sorted(group)}: "
            f"checker={claimed}, stim expectation={expectation}"
        )


@pytest.mark.parametrize("edges", [((0, 1),), ((0, 1), (1, 2))])
def test_detector_determinism_matches_stim(edges: tuple[tuple[int, int], ...]) -> None:
    """Sweep every init/measurement-axis combination and compare with stim.

    The stim compiler prepares each node along its initialization axis and
    applies CZ for every edge, so a detector is deterministic exactly when the
    product of the measured Pauli axes on its group has a nonzero expectation
    value on that resource state.
    """
    axes = [Axis.X, Axis.Y, Axis.Z]
    num_nodes = max(max(edge) for edge in edges) + 1
    groups = [
        set(combo) for size in range(1, num_nodes + 1) for combo in itertools.combinations(range(num_nodes), size)
    ]

    for init_axes in itertools.product(axes, repeat=num_nodes):
        for meas_axes in itertools.product(axes, repeat=num_nodes):
            _assert_determinism_matches_stim(edges, init_axes, meas_axes, groups)


def test_logical_observables_group() -> None:
    """Test logical observables group generation."""
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
    graph.assign_meas_basis(n2, PlannerMeasBasis(Plane.XY, 0.0))

    xflow = {n0: {n1}, n1: {n2}}
    zflow: dict[int, set[int]] = {n0: {n0, n2}}

    # Provide parity_check_group to enable _pauli_axis_cache initialization
    parity_check_group = [set(graph.nodes)]
    pframe = PauliFrame(graph, xflow, zflow, parity_check_group)

    # Get logical observables group
    target_nodes = [n2]
    group = pframe.logical_observables_group(target_nodes)
    assert isinstance(group, set)
    assert group == {n0, n2}  # n0 is included via dependent chain


def test_logical_observable_groups() -> None:
    """Test closure expansion for all indexed logical observables."""
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
    graph.assign_meas_basis(n2, PlannerMeasBasis(Plane.XY, 0.0))

    xflow = {n0: {n1}, n1: {n2}}
    zflow: dict[int, set[int]] = {n0: {n0, n2}}
    logical_observables = {3: {n1}, 7: {n2}}

    pframe = PauliFrame(graph, xflow, zflow, logical_observables=logical_observables)

    assert pframe.logical_observable_groups() == {
        3: {n1},
        7: {n0, n2},
    }


def test_collect_dependent_chain_cache_hit() -> None:
    """Test that cache is correctly used when same node is queried multiple times."""
    graph = GraphState()
    n0 = graph.add_node()
    n1 = graph.add_node()
    n2 = graph.add_node()
    n3 = graph.add_node()

    graph.register_input(n0, 0)
    graph.register_output(n3, 0)

    graph.add_edge(n0, n1)
    graph.add_edge(n1, n2)
    graph.add_edge(n2, n3)

    # Mix of X and Z measurements
    graph.assign_meas_basis(n0, PlannerMeasBasis(Plane.XZ, 0.0))  # Z
    graph.assign_meas_basis(n1, PlannerMeasBasis(Plane.XY, 0.0))  # X
    graph.assign_meas_basis(n2, PlannerMeasBasis(Plane.XZ, 0.0))  # Z

    xflow = {n0: {n1}, n1: {n2}, n2: {n3}}
    zflow = {n0: {n0}}

    # Provide parity_check_group to enable _pauli_axis_cache initialization
    parity_check_group = [set(graph.nodes)]
    pframe = PauliFrame(graph, xflow, zflow, parity_check_group)

    # First call to n2
    chain1 = pframe._collect_dependent_chain(n2)
    assert n2 in pframe._chain_cache

    # Clear internal cache to test that memoization returns correct result
    cached_result = pframe._chain_cache[n2]

    # Second call should use cache
    chain2 = pframe._collect_dependent_chain(n2)

    # Results should be identical
    assert chain1 == chain2
    assert set(cached_result) == chain1


def test_collect_dependent_chain_diamond_cancellation() -> None:
    """Test that nodes reached via multiple paths are correctly XOR'd.

    Diamond graph structure (5 nodes with n4 as output):
        n0 → n1, n0 → n2, n1 → n3, n2 → n3, n3 → n4

    When collecting dependent chain for n3:
    - chain(n0) = {n0}
    - chain(n1) = {n1} ^ chain(n0) = {n0, n1}
    - chain(n2) = {n2} ^ chain(n0) = {n0, n2}
    - chain(n3) = {n3} ^ chain(n1) ^ chain(n2) = {n3} ^ {n0, n1} ^ {n0, n2} = {n1, n2, n3}

    Node n0 should be canceled out because it's reached via two paths.
    """
    graph = GraphState()
    n0 = graph.add_node()
    n1 = graph.add_node()
    n2 = graph.add_node()
    n3 = graph.add_node()
    n4 = graph.add_node()

    graph.register_input(n0, 0)
    graph.register_output(n4, 0)

    # Diamond edges + edge to output
    graph.add_edge(n0, n1)
    graph.add_edge(n0, n2)
    graph.add_edge(n1, n3)
    graph.add_edge(n2, n3)
    graph.add_edge(n3, n4)

    # All Z measurements (XZ plane, angle 0) so parents come from inv_xflow
    graph.assign_meas_basis(n0, PlannerMeasBasis(Plane.XZ, 0.0))
    graph.assign_meas_basis(n1, PlannerMeasBasis(Plane.XZ, 0.0))
    graph.assign_meas_basis(n2, PlannerMeasBasis(Plane.XZ, 0.0))
    graph.assign_meas_basis(n3, PlannerMeasBasis(Plane.XZ, 0.0))

    # xflow: n0 → {n1, n2}, n1 → {n3}, n2 → {n3}, n3 → {n4}
    xflow = {n0: {n1, n2}, n1: {n3}, n2: {n3}, n3: {n4}}
    zflow: dict[int, set[int]] = {}

    # Provide parity_check_group to enable _pauli_axis_cache initialization
    parity_check_group = [set(graph.nodes)]
    pframe = PauliFrame(graph, xflow, zflow, parity_check_group)

    # Verify the chain for n3
    chain_n3 = pframe._collect_dependent_chain(n3)

    # n0 should be canceled out (reached via n1 and n2)
    assert n0 not in chain_n3, f"n0 should be canceled out but chain is {chain_n3}"
    assert chain_n3 == {n1, n2, n3}, f"Expected {{n1, n2, n3}} but got {chain_n3}"

    # Also verify intermediate chains
    chain_n1 = pframe._collect_dependent_chain(n1)
    assert chain_n1 == {n0, n1}, f"Expected {{n0, n1}} but got {chain_n1}"

    chain_n2 = pframe._collect_dependent_chain(n2)
    assert chain_n2 == {n0, n2}, f"Expected {{n0, n2}} but got {chain_n2}"


def _two_group_frame_args() -> tuple[GraphState, dict[int, set[int]], dict[int, set[int]], list[set[int]]]:
    """Build a small graph with two parity check groups.

    Returns
    -------
    tuple[GraphState, dict[int, set[int]], dict[int, set[int]], list[set[int]]]
        GraphState, xflow, zflow, and parity check groups.
    """
    graph = GraphState()
    n0 = graph.add_node()
    n1 = graph.add_node()
    graph.register_input(n0, 0)
    graph.register_output(n1, 0)
    graph.add_edge(n0, n1)
    graph.assign_meas_basis(n0, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(n1, PlannerMeasBasis(Plane.XY, 0.0))
    return graph, {n0: {n1}}, {}, [{n0}, {n1}]


def test_parity_check_tags_default_to_untagged() -> None:
    graph, xflow, zflow, groups = _two_group_frame_args()
    pframe = PauliFrame(graph, xflow, zflow, groups)

    assert pframe.parity_check_tags == ["", ""]


def test_parity_check_tags_are_stored_in_group_order() -> None:
    graph, xflow, zflow, groups = _two_group_frame_args()
    pframe = PauliFrame(graph, xflow, zflow, groups, parity_check_tags=["type=flag", ""])

    assert pframe.parity_check_tags == ["type=flag", ""]


def test_parity_check_tags_length_mismatch_raises() -> None:
    graph, xflow, zflow, groups = _two_group_frame_args()
    with pytest.raises(ValueError, match="parity_check_tags has 1 tag"):
        PauliFrame(graph, xflow, zflow, groups, parity_check_tags=["type=flag"])


def test_parity_check_tags_without_groups_default_empty() -> None:
    graph, xflow, zflow, _groups = _two_group_frame_args()
    pframe = PauliFrame(graph, xflow, zflow)

    assert pframe.parity_check_tags == []
