"""Behavior and serialization checks for the Steane logical T example."""

from __future__ import annotations

import itertools
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import stim

from examples.ftqc._support.steane import (
    CODE_DISTANCE,
    MAGIC_PATCH_OFFSET,
    MEMORY_ROUNDS,
    STEANE_SUPPORTS,
    build_steane_circuit,
    build_steane_pattern,
    run_steane_shot,
    steane_basis,
    steane_input,
)
from graphqomb import clifford_algebra as ca
from graphqomb.command import E, M
from graphqomb.common import Axis
from graphqomb.ptn_format import dump, dumps, load
from graphqomb.simulator import PatternSimulator, SimulatorBackend
from graphqomb.statevec import StateVector

if TYPE_CHECKING:
    from pathlib import Path

_X = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
_Z = np.diag([1, -1]).astype(np.complex128)
_S_DAGGER = np.diag([1, -1j]).astype(np.complex128)


def test_encoded_basis_and_transversal_s_convention() -> None:
    zero, one = steane_basis()
    np.testing.assert_allclose(np.vdot(zero, one), 0.0, atol=1e-12)
    for state in (zero, one):
        np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=1e-12)
        for support, operator in itertools.product(STEANE_SUPPORTS, (_X, _Z)):
            vector = StateVector(state)
            for qubit in support:
                vector.evolve(operator, qubit)
            np.testing.assert_allclose(vector.state().ravel(), state, atol=1e-12)
    for state, phase in ((zero, 1.0), (one, 1j)):
        vector = StateVector(state)
        for qubit in range(7):
            vector.evolve(_S_DAGGER, qubit)
        np.testing.assert_allclose(vector.state().ravel(), phase * state, atol=1e-12)


def test_stim_source_has_memory_on_both_blocks_and_seven_disjoint_cnot_pairs() -> None:
    source = build_steane_circuit(None)
    circuit = source.circuit
    ops = []
    for index in range(len(circuit)):
        op = circuit[index]
        assert isinstance(op, stim.CircuitInstruction)
        ops.append(op)
    first_cx = next(index for index, op in enumerate(ops) if op.name == "CX")
    memory_ops = [op for op in ops[:first_cx] if op.name == "MPP"]
    assert sum(op.num_measurements for op in memory_ops) == 12
    assert source.magic_records == tuple(range(6, 12))
    cx = ops[first_cx]
    targets = [target.value for target in cx.targets_copy()]
    assert list(zip(targets[::2], targets[1::2], strict=True)) == [(7 + q, q) for q in range(7)]
    assert source.consumed_records == tuple(range(12, 19))
    assert source.after_records == tuple(tuple(range(19 + 6 * r, 25 + 6 * r)) for r in range(3))
    post_memory_ops = [op for op in ops[first_cx + 1 :] if op.name == "MPP"]
    assert sum(op.num_measurements for op in post_memory_ops) == 18
    assert all(op.name != "H" for op in ops)
    steane = build_steane_pattern(None)
    assert len(steane.pattern.input_node_indices) == 14
    assert len(steane.cnot_edges) == 7
    assert len({node for edge in steane.cnot_edges for node in edge}) == 14
    positions = steane.pattern.frame.graphstate.coordinates
    assert {positions[node][:2] for node in steane.data_nodes} == {
        positions[node][:2] for node in steane.magic_input_nodes
    }


def test_encoded_transversal_cnot_teleports_all_sixteen_allowed_z_words() -> None:
    """Independently verify all readout words before MBQC lowering."""
    logical = np.asarray([0.3 + 0.1j, 0.7 - 0.2j], dtype=np.complex128)
    logical /= np.linalg.norm(logical)
    zero, one = steane_basis()
    initial = steane_input(logical)
    indices = np.arange(initial.size)
    permuted = indices.copy()
    for qubit in range(7):
        permuted ^= ((indices >> (6 - qubit)) & 1) << (13 - qubit)
    after = np.empty_like(initial)
    after[permuted] = initial
    branches = 0
    for word, output in enumerate(after.reshape(128, 128)):
        probability = float(np.linalg.norm(output) ** 2)
        if probability < 1e-14:
            continue
        bits = [(word >> (6 - qubit)) & 1 for qubit in range(7)]
        assert all(sum(bits[q] for q in support) % 2 == 0 for support in STEANE_SUPPORTS)
        m = bool(sum(bits[q] for q in (4, 5, 6)) % 2)
        a, b = logical[::-1] if m else logical
        expected = a * zero + np.exp(1j * np.pi / 4) * b * one
        assert abs(np.vdot(expected, output / np.sqrt(probability))) ** 2 == pytest.approx(1.0, abs=1e-12)
        assert probability == pytest.approx(1 / 16, abs=1e-12)
        branches += 1
    assert branches == 16


def test_quantum_output_is_logical_t_in_all_sampled_injection_branches() -> None:
    steane = build_steane_pattern(None)
    assert steane.pattern.frame.logical_observables == {}
    logical = np.asarray([0.3 + 0.1j, 0.7 - 0.2j], dtype=np.complex128)
    logical /= np.linalg.norm(logical)
    zero, one = steane_basis()
    expected = logical[0] * zero + np.exp(1j * np.pi / 4) * logical[1] * one
    branches = set()
    for seed in range(16):
        simulator = PatternSimulator(deepcopy(steane.pattern), SimulatorBackend.StateVector, calc_prob=True)
        simulator.state = StateVector(steane_input(logical))
        simulator.simulate(np.random.default_rng(seed))
        output = np.asarray(simulator.state.state()).ravel()
        assert output.shape == (128,)
        assert abs(np.vdot(expected, output)) ** 2 == pytest.approx(1.0, abs=1e-12)
        records = simulator.results
        branches.add(records[steane.control_node])
        assert not any(sum(records[node] for node in group) % 2 for group in steane.pattern.frame.parity_check_group)
    assert branches == {False, True}
    assert steane.pattern.max_space <= 15


@pytest.mark.parametrize("readout", [Axis.X, Axis.Z])
@pytest.mark.parametrize("expected_bit", [False, True])
def test_logical_readout_preserves_both_eigenvalue_signs(readout: Axis, expected_bit: bool) -> None:
    steane = build_steane_pattern(readout)
    if readout == Axis.X:
        # Independent single-logical-qubit oracle: T turns these into |+> / |->.
        logical = np.asarray([1.0, (-1) ** expected_bit * np.exp(-1j * np.pi / 4)]) / np.sqrt(2)
    else:
        logical = np.asarray([0.0, 1.0] if expected_bit else [1.0, 0.0], dtype=np.complex128)
    for seed in range(8):
        shot = run_steane_shot(steane, logical, seed=seed)
        assert shot.observable == expected_bit
        assert not any(shot.detectors)


@pytest.mark.parametrize("readout", [Axis.X, Axis.Z])
def test_every_boundary_fault_has_the_expected_syndrome_column_and_logical_flip(readout: Axis) -> None:
    steane = build_steane_pattern(readout)
    # Both inputs yield deterministic zero logical readout after the T gate.
    logical = (
        np.asarray([1.0, np.exp(-1j * np.pi / 4)]) / np.sqrt(2)
        if readout == Axis.X
        else np.asarray([1.0, 0.0], dtype=np.complex128)
    )
    controls = set()
    for seed in range(4):
        for qubit, fault in itertools.product(range(7), Axis):
            shot = run_steane_shot(steane, logical, seed=seed, fault=(qubit, fault))
            m = shot.teleportation_bit
            controls.add(m)
            # S has already acted before the three memory rounds.
            flip = fault != readout
            column = (qubit in {0, 1, 2, 3}, qubit in {0, 1, 4, 5}, qubit in {0, 2, 4, 6})
            expected = tuple(
                flip and tag.startswith("data_") and column[int(tag.rsplit("_", 1)[1])]
                for tag in steane.pattern.frame.parity_check_tags
            )
            assert shot.detectors == expected
            assert shot.observable == (flip and qubit in {4, 5, 6})
    assert controls == {False, True}


@pytest.mark.parametrize("readout", [Axis.X, Axis.Z, None])
def test_ptn_roundtrip_preserves_encoded_injection_and_readout(tmp_path: Path, readout: Axis | None) -> None:
    steane = build_steane_pattern(readout)
    path = tmp_path / "steane.ptn"
    dump(steane.pattern, path)
    restored = load(path)
    assert dumps(restored) == dumps(steane.pattern)
    logical = np.asarray([1.0, 0.0], dtype=np.complex128)
    simulator = PatternSimulator(restored, SimulatorBackend.StateVector, calc_prob=True)
    simulator.state = StateVector(steane_input(logical))
    simulator.simulate(np.random.default_rng(0))
    assert simulator.results[steane.control_node] in {False, True}
    if readout is None:
        zero, _ = steane_basis()
        assert abs(np.vdot(zero, simulator.state.state().ravel())) ** 2 == pytest.approx(1.0, abs=1e-12)
    else:
        shot = run_steane_shot(replace(steane, pattern=load(path)), logical, seed=0)
        assert not any(shot.detectors)


def test_steane_rejects_unsupported_readout_and_invalid_logical_input() -> None:
    with pytest.raises(ValueError, match="X, Z, or quantum"):
        build_steane_pattern(Axis.Y)
    with pytest.raises(ValueError, match="normalized two-component"):
        steane_input(np.zeros(2, dtype=np.complex128))


def test_export_stacks_time_in_z_and_preserves_the_xy_code_patch(tmp_path: Path) -> None:
    """Data copies share XY positions; the serialized Z coordinates order the logical stages."""
    steane = build_steane_pattern(Axis.X)
    path = tmp_path / "steane_xyz.ptn"
    dump(steane.pattern, path)
    restored_frame = load(path).frame
    coordinates = restored_frame.graphstate.coordinates
    assert restored_frame.parity_check_tags == steane.pattern.frame.parity_check_tags
    assert restored_frame.parity_check_group == steane.pattern.frame.parity_check_group
    assert restored_frame.logical_observables == steane.pattern.frame.logical_observables
    assert all(len(coord) == 3 for coord in coordinates.values())
    for qubit in range(7):
        assert coordinates[steane.consumed_nodes[qubit]][:2] == coordinates[steane.data_input_nodes[qubit]][:2]
        assert coordinates[steane.data_nodes[qubit]][:2] == coordinates[steane.magic_input_nodes[qubit]][:2]
        assert coordinates[steane.data_nodes[qubit]][2] > coordinates[steane.magic_input_nodes[qubit]][2]


def test_input_memories_are_parallel_and_cnot_couplings_do_not_overlap() -> None:
    steane = build_steane_pattern(Axis.X)
    coordinates = steane.pattern.frame.graphstate.coordinates
    assert {coordinates[node][2] for node in steane.before_nodes} == {
        coordinates[node][2] for node in steane.magic_before_nodes
    }
    data_xy = np.array([coordinates[node][:2] for node in steane.data_input_nodes])
    magic_xy = np.array([coordinates[node][:2] for node in steane.magic_input_nodes])
    np.testing.assert_allclose(magic_xy - data_xy, np.tile(MAGIC_PATCH_OFFSET, (7, 1)), atol=1e-12)
    # All seven coupling segments are parallel and lie on distinct XY lines.
    dx, dy = MAGIC_PATCH_OFFSET
    for a, b in itertools.combinations(data_xy, 2):
        separation = b - a
        assert abs(dx * separation[1] - dy * separation[0]) > 1e-12
    layers = sorted({coordinate[2] for coordinate in coordinates.values()})
    assert layers == list(range(len(layers)))


@pytest.mark.parametrize("axis", [Axis.X, Axis.Z])
def test_cflow_precedes_three_memory_rounds_and_logical_readout(axis: Axis) -> None:
    """The compiled schedule retains S, then d complete rounds, then readout."""
    steane = build_steane_pattern(axis)
    pattern = steane.pattern
    frame = pattern.frame
    assert pattern.max_space <= 15
    assert MEMORY_ROUNDS == CODE_DISTANCE == 3
    assert len(steane.after_nodes) == 3
    assert all(len(nodes) == 6 for nodes in steane.after_nodes)
    assert len(pattern.frame.parity_check_group) == 36
    # S-dagger is stored as the S coset plus a conditional Z factor.
    assert frame.cflow == {steane.control_node: dict.fromkeys(steane.correction_nodes, ca.S)}
    assert len(steane.correction_nodes) == 7
    assert set(steane.correction_nodes).isdisjoint(steane.data_nodes)
    assert set(steane.correction_nodes) <= frame.zflow[steane.control_node]
    measurement_order = {command.node: index for index, command in enumerate(pattern) if isinstance(command, M)}
    assert measurement_order[steane.control_node] < min(measurement_order[node] for node in steane.correction_nodes)
    # S updates the output frame when the control result arrives. The first
    # round's input-wire measurements share its targets, with no identity relay.
    stages = [(steane.control_node,), *steane.after_nodes, steane.data_nodes]
    for previous, following in itertools.pairwise(stages):
        assert max(measurement_order[node] for node in previous) < min(measurement_order[node] for node in following)
    groups = [
        group
        for tag, group in zip(frame.parity_check_tags, frame.parity_check_group, strict=True)
        if tag.startswith("data_")
    ]
    assert len(groups) == 3
    assert all(group - set(steane.data_nodes) for group in groups)
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    simulator.state = StateVector(steane_input(np.asarray([1.0, 0.0], dtype=np.complex128)))
    simulator.simulate(np.random.default_rng(0))
    assert all(not sum(simulator.results[node] for node in group) % 2 for group in frame.parity_check_group)


@pytest.mark.parametrize("axis", [Axis.X, Axis.Z])
def test_logical_observable_backpropagates_through_memory_to_cflow(axis: Axis) -> None:
    """The observable compiles the Clifford tail and retains the early S boundary."""
    steane = build_steane_pattern(axis)
    pattern = steane.pattern
    support = pattern.frame.logical_observables[0]
    assert support & (set(steane.correction_nodes) | {steane.control_node})
    assert support != {steane.data_nodes[q] for q in (4, 5, 6)}
    logical = (
        np.asarray([1.0, np.exp(-1j * np.pi / 4)]) / np.sqrt(2)
        if axis == Axis.X
        else np.asarray([1.0, 0.0], dtype=np.complex128)
    )
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    simulator.state = StateVector(steane_input(logical))
    simulator.simulate(np.random.default_rng(1))
    assert not sum(simulator.results[node] for node in pattern.frame.logical_observables[0]) % 2


def test_shot_evaluates_the_patterns_stored_parity_supports() -> None:
    steane = build_steane_pattern(Axis.Z)
    frame = steane.pattern.frame
    frame.parity_check_group = [{steane.control_node}]
    frame.parity_check_tags = [""]
    frame.logical_observables = {0: {steane.control_node}}
    results = set()
    for seed in range(4):
        shot = run_steane_shot(steane, np.asarray([1.0, 0.0], dtype=np.complex128), seed=seed)
        assert shot.detectors == (shot.teleportation_bit,)
        assert shot.observable == shot.teleportation_bit
        results.add(shot.observable)
    assert results == {False, True}


@pytest.mark.parametrize("qubit", range(7))
def test_magic_memory_detects_a_fault_on_every_magic_input(qubit: int) -> None:
    """Every magic input passes through a memory round before the transversal CNOT."""
    steane = build_steane_pattern(None)
    logical = np.asarray([1.0, 0.0], dtype=np.complex128)
    simulator = PatternSimulator(deepcopy(steane.pattern), SimulatorBackend.StateVector, calc_prob=True)
    simulator.state = StateVector(steane_input(logical))
    simulator.state.evolve(_Z, simulator.node_indices.index(steane.magic_input_nodes[qubit]))
    rng = np.random.default_rng(0)
    simulator.simulate(rng)
    frame = steane.pattern.frame
    syndrome = tuple(
        bool(sum(simulator.results[node] for node in support) % 2)
        for tag, support in zip(frame.parity_check_tags, frame.parity_check_group, strict=True)
        if tag.startswith("initial_magic_X_")
    )
    # Independent columns of the specified three-check parity matrix.
    columns = (
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, True, True),
        (False, True, False),
        (False, False, True),
    )
    assert syndrome == columns[qubit]
    assert any(syndrome)


@pytest.mark.parametrize("pair", range(7))
def test_every_two_qubit_pauli_fault_at_a_cnot_coupling_is_flagged_or_harmless(pair: int) -> None:
    """Test the 15 local faults on each pairwise entangler in both sampled runs."""
    steane = build_steane_pattern(None)
    edge = steane.cnot_edges[pair]
    logical = np.asarray([0.3 + 0.1j, 0.7 - 0.2j], dtype=np.complex128)
    logical /= np.linalg.norm(logical)
    zero, one = steane_basis()
    expected = logical[0] * zero + np.exp(1j * np.pi / 4) * logical[1] * one
    for seed in range(2):
        for faults in itertools.product((ca.IDENTITY, ca.X, ca.Y, ca.Z), repeat=2):
            if faults == (ca.IDENTITY, ca.IDENTITY):
                continue
            pattern = deepcopy(steane.pattern)
            simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
            simulator.state = StateVector(steane_input(logical))
            rng = np.random.default_rng(seed)
            fault_sites = 0
            for command in pattern:
                simulator.apply_cmd(command, rng=rng)
                if isinstance(command, E) and set(command.nodes) == set(edge):
                    fault_sites += 1
                    for node, fault in zip(edge, faults, strict=True):
                        simulator.state.evolve(ca.to_matrix(fault), simulator.node_indices.index(node))
            assert fault_sites == 1, edge
            # apply_cmd permits faults at gate boundaries; finalize the remaining outputs.
            for node in simulator.node_indices:
                simulator._apply_output_frame(node)
            simulator.state.reorder([steane.data_nodes.index(node) for node in simulator.node_indices])
            flagged = any(
                sum(simulator.results[node] for node in group) % 2 for group in pattern.frame.parity_check_group
            )
            fidelity = abs(np.vdot(expected, simulator.state.state().ravel())) ** 2
            assert flagged or fidelity == pytest.approx(1.0, abs=1e-12), (pair, seed, faults, fidelity)


def test_post_correction_memory_sees_s_rotated_error_in_both_branches() -> None:
    """An incoming X0 error becomes Y0 for odd m before memory starts."""
    steane = build_steane_pattern(None)
    logical = np.asarray([0.3 + 0.1j, 0.7 - 0.2j], dtype=np.complex128)
    logical /= np.linalg.norm(logical)
    zero, one = steane_basis()
    ideal = logical[0] * zero + np.exp(1j * np.pi / 4) * logical[1] * one
    y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    branches = set()
    for seed in range(8):
        simulator = PatternSimulator(deepcopy(steane.pattern), SimulatorBackend.StateVector)
        simulator.state = StateVector(steane_input(logical))
        simulator.state.evolve(_X, simulator.node_indices.index(steane.magic_input_nodes[0]))
        simulator.simulate(np.random.default_rng(seed))
        m = simulator.results[steane.control_node]
        branches.add(m)
        frame = steane.pattern.frame
        checks = {
            tag: bool(sum(simulator.results[node] for node in group) % 2)
            for tag, group in zip(frame.parity_check_tags, frame.parity_check_group, strict=True)
        }
        # q0 belongs to every Steane support: Y flips X and Z checks; X only Z.
        assert tuple(checks[f"memory_1_X_{row}"] for row in range(3)) == (m, m, m)
        assert tuple(checks[f"memory_1_Z_{row}"] for row in range(3)) == (True, True, True)
        assert not any(value for tag, value in checks.items() if tag.startswith(("memory_2_", "memory_3_")))
        expected = StateVector(ideal)
        expected.evolve(y if m else _X, 0)
        assert abs(np.vdot(expected.state().ravel(), simulator.state.state().ravel())) ** 2 == pytest.approx(
            1.0, abs=1e-12
        )
    assert branches == {False, True}
