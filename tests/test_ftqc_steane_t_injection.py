"""Behavior and serialization checks for the Steane logical T example."""

from __future__ import annotations

import itertools
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from examples.ftqc._support.steane import (
    STEANE_SUPPORTS,
    build_steane_pattern,
    run_steane_shot,
    steane_basis,
    steane_input,
)
from graphqomb import clifford_algebra as ca
from graphqomb.command import M
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


def test_quantum_output_is_logical_t_in_all_sampled_injection_branches() -> None:
    steane = build_steane_pattern(None)
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
        magic_bit = records[steane.magic_nodes[4]] ^ records[steane.magic_nodes[5]] ^ records[steane.magic_nodes[6]]
        branches.add((records[steane.probe_node], magic_bit))
        assert records[steane.control_node] == records[steane.probe_node]
        assert not any(records[node] for node in (*steane.before_nodes, *steane.after_nodes))
    assert branches == set(itertools.product((False, True), repeat=2))
    assert steane.pattern.max_space <= 16


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
        assert not any(shot.before)
        assert not any(shot.after)


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
            p = shot.injection[0]
            controls.add(p)
            # A physical S correction rotates X to Y only in the odd ZZ branch.
            effective_axis = (Axis.Y if p else Axis.X) if readout == Axis.X else Axis.Z
            flip = fault != effective_axis
            column = (qubit in {0, 1, 2, 3}, qubit in {0, 1, 4, 5}, qubit in {0, 2, 4, 6})
            assert shot.detectors == (False,) * 6 + tuple(flip and entry for entry in column) + (False,) * 3
            assert shot.observable == (flip and qubit in {4, 5, 6})
            assert shot.correction_control == p
            assert not any(shot.before)
            assert not any(shot.after)
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
    assert simulator.results[steane.control_node] == simulator.results[steane.probe_node]
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
        points = [coordinates[layer[qubit]] for layer in steane.data_layers]
        assert all(point[:2] == points[0][:2] for point in points)
        assert [point[2] for point in points] == [0.0, 1.0, 2.0, 3.5, 5.0]
    assert [coordinates[node][2] for node in steane.magic_nodes] == pytest.approx([2.5] * 7)
    assert coordinates[steane.probe_node][2] < coordinates[steane.control_node][2]
    assert coordinates[steane.control_node][2] < min(coordinates[node][2] for node in steane.data_nodes)


@pytest.mark.parametrize("axis", [Axis.X, Axis.Z])
def test_final_readout_detectors_stop_at_cflow(axis: Axis) -> None:
    """Final-data parities remain over adapted boundary records, with S retained."""
    steane = build_steane_pattern(axis)
    pattern = steane.pattern
    frame = pattern.frame
    assert pattern.max_space <= 16
    assert len(pattern.frame.parity_check_group) == 12
    # S-dagger is stored as the S coset plus a conditional Z factor.
    assert frame.cflow == {steane.control_node: dict.fromkeys(steane.data_nodes, ca.S)}
    assert set(steane.data_nodes) <= frame.zflow[steane.control_node]
    assert frame.parity_check_group[6:9] == [{steane.data_nodes[q] for q in support} for support in STEANE_SUPPORTS]
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    simulator.state = StateVector(steane_input(np.asarray([1.0, 0.0], dtype=np.complex128)))
    simulator.simulate(np.random.default_rng(0))
    assert all(not sum(simulator.results[node] for node in group) % 2 for group in frame.parity_check_group)


@pytest.mark.parametrize("axis", [Axis.X, Axis.Z])
def test_logical_observable_stops_at_cflow(axis: Axis) -> None:
    """Logical parity uses boundary records instead of crossing a Clifford correction."""
    steane = build_steane_pattern(axis)
    pattern = steane.pattern
    assert pattern.frame.logical_observables == {0: {steane.data_nodes[q] for q in (4, 5, 6)}}
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
        assert shot.detectors == (shot.correction_control,)
        assert shot.observable == shot.correction_control
        results.add(shot.observable)
    assert results == {False, True}


def test_quantum_output_backpropagation_has_no_classical_observable_and_preserves_t() -> None:
    steane = build_steane_pattern(None)
    assert steane.pattern.frame.logical_observables == {}
    assert steane.pattern.max_space <= 16
    logical = np.asarray([0.3 + 0.1j, 0.7 - 0.2j])
    logical /= np.linalg.norm(logical)
    simulator = PatternSimulator(steane.pattern, SimulatorBackend.StateVector)
    simulator.state = StateVector(steane_input(logical))
    simulator.simulate(np.random.default_rng(0))
    zero, one = steane_basis()
    expected = logical[0] * zero + np.exp(1j * np.pi / 4) * logical[1] * one
    assert abs(np.vdot(expected, simulator.state.state().ravel())) ** 2 == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("qubit", range(7))
def test_every_magic_qubit_contributes_to_the_x_stabilizer_checks(qubit: int) -> None:
    """A Z fault on any magic input is detected, including the four outside logical-Z support."""
    steane = build_steane_pattern(Axis.X)
    logical = np.asarray([1.0, 0.0], dtype=np.complex128)
    simulator = PatternSimulator(deepcopy(steane.pattern), SimulatorBackend.StateVector, calc_prob=True)
    simulator.state = StateVector(steane_input(logical))
    rng = np.random.default_rng(0)
    for command in steane.pattern:
        if isinstance(command, M) and command.node == steane.magic_nodes[qubit]:
            simulator.state.evolve(_Z, simulator.node_indices.index(command.node))
        simulator.apply_cmd(command, rng=rng)
    magic_check_groups = steane.pattern.frame.parity_check_group[-3:]
    syndrome = tuple(bool(sum(simulator.results[node] for node in group) % 2) for group in magic_check_groups)
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
