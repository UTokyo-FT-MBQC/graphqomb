"""Checks for the color-code memory demo with cat-state ancillas."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from operator import itemgetter
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
import stim

from examples.ftqc._support.color_code import (
    memory_circuit,
    noisy_circuit,
    pattern_with_cz_order,
    random_cz_order,
    triangular_color_code,
)
from examples.ftqc._support.fault_distance import drop_detectors, error_mechanisms, fault_distance_bounds
from graphqomb.stim_glue import CAT_FLAG_TAG, flag_detector_indices, stim_circuit_to_ir, stim_circuit_to_pattern

if TYPE_CHECKING:
    from graphqomb.stim_glue import StimImportIR


@pytest.mark.parametrize("distance", [3, 5, 7])
def test_triangular_color_code_is_a_self_dual_css_code(distance: int) -> None:
    code = triangular_color_code(distance)
    num_qubits = (3 * distance**2 + 1) // 4

    assert len(code.coordinates) == num_qubits
    assert len(set(code.coordinates)) == num_qubits
    assert len(code.plaquettes) == (num_qubits - 1) // 2
    assert set(Counter(len(plaquette) for plaquette in code.plaquettes)) <= {4, 6}
    assert {qubit for plaquette in code.plaquettes for qubit in plaquette} == set(range(num_qubits))
    # X and Z plaquettes share supports, so every pair must overlap evenly.
    assert all(len(set(left) & set(right)) % 2 == 0 for left, right in combinations(code.plaquettes, 2))
    assert len(code.logical) == distance
    assert all(len(set(code.logical) & set(plaquette)) % 2 == 0 for plaquette in code.plaquettes)


@pytest.mark.parametrize("distance", [2, 4, 1])
def test_triangular_color_code_rejects_unsupported_distances(distance: int) -> None:
    with pytest.raises(ValueError, match="odd distance of at least 3"):
        triangular_color_code(distance)


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("distance", [3, 5])
def test_memory_circuit_has_code_distance_under_data_noise(distance: int, basis: Literal["X", "Z"]) -> None:
    code = triangular_color_code(distance)
    rounds = 2
    circuit = memory_circuit(code, rounds, basis)
    count = len(code.plaquettes)

    assert circuit.num_qubits == len(code.coordinates)
    assert circuit.num_measurements == 2 * count * rounds + len(code.coordinates)
    # Round 1 checks one type, later rounds both, and the readout the memory type.
    assert circuit.num_detectors == 2 * count * rounds
    assert circuit.num_observables == 1
    assert circuit.detector_error_model().num_errors == 0
    assert not np.asarray(circuit.compile_detector_sampler(seed=5).sample(32, append_observables=True)).any()

    # A data error of the flipping type before the final readout: the distance is the code distance.
    flip = "X_ERROR" if basis == "Z" else "Z_ERROR"
    noisy = stim.Circuit()
    for instruction in circuit:
        if instruction.name in {"M", "MX"}:
            noisy.append(flip, range(len(code.coordinates)), 1e-3)
        noisy.append(instruction)
    assert fault_distance_bounds(noisy) == (distance, distance)


def test_memory_circuit_rejects_invalid_arguments() -> None:
    code = triangular_color_code(3)
    with pytest.raises(ValueError, match="at least one round"):
        memory_circuit(code, 0)
    with pytest.raises(ValueError, match="basis must be"):
        memory_circuit(code, 2, "Y")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("task", "distance", "expected"),
    [
        ("repetition_code:memory", 3, 3),
        ("repetition_code:memory", 5, 5),
        ("surface_code:rotated_memory_z", 3, 3),
        ("surface_code:unrotated_memory_x", 3, 3),
    ],
)
def test_fault_distance_bounds_match_known_circuit_distances(task: str, distance: int, expected: int) -> None:
    circuit = stim.Circuit.generated(
        task,
        distance=distance,
        rounds=2,
        after_clifford_depolarization=1e-3,
        before_measure_flip_probability=1e-3,
        after_reset_flip_probability=1e-3,
    )

    assert fault_distance_bounds(circuit) == (expected, expected)


def test_fault_distance_bounds_report_an_uncertified_gap() -> None:
    circuit = stim.Circuit.generated(
        "repetition_code:memory", distance=5, rounds=2, before_round_data_depolarization=1e-3
    )

    # Without the exhaustive pass only Stim's search bounds the distance from above.
    assert fault_distance_bounds(circuit, exhaustive_weight=0) == (1, 5)
    assert fault_distance_bounds(circuit, exhaustive_weight=2) == (3, 5)
    with pytest.raises(ValueError, match="exhaustive_weight"):
        fault_distance_bounds(circuit, exhaustive_weight=5)


def test_fault_distance_bounds_find_a_single_undetected_fault() -> None:
    circuit = stim.Circuit("R 0\nX_ERROR(0.1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]")

    assert fault_distance_bounds(circuit) == (1, 1)


def test_drop_detectors_removes_only_the_tagged_ones() -> None:
    circuit = stim.Circuit(
        """
        R 0 1
        M 0 1
        DETECTOR[a] rec[-1]
        DETECTOR rec[-2]
        DETECTOR[a] rec[-2] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )

    dropped = drop_detectors(circuit, "a")

    assert dropped.num_detectors == 1
    assert dropped.num_observables == 1
    assert str(dropped) == "R 0 1\nM 0 1\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]"


def _cz_orders(ir: StimImportIR, seeds: int) -> list[list[tuple[int, int]]]:
    return [sorted(ir.graph.edges), *(random_cz_order(ir, seed) for seed in range(seeds))]


@pytest.mark.parametrize("weight", [4, 6])
def test_one_fault_on_a_cat_ring_reaches_one_data_wire_in_every_cz_order(weight: int) -> None:
    qubits = " ".join(map(str, range(weight)))
    product = "*".join(f"Z{qubit}" for qubit in range(weight))
    # Detector k compares the X readouts of qubits k and k + 1, so it locates Z errors up to the product.
    detectors = "\n".join(f"DETECTOR rec[{k - weight}] rec[{k + 1 - weight}]" for k in range(weight - 1))
    source = stim.Circuit(f"RX {qubits}\nTICK\nMPP {product}\nTICK\nMX {qubits}\n{detectors}")
    one_wire = {frozenset(), *(frozenset({k - 1, k}) & frozenset(range(weight - 1)) for k in range(weight))}

    def unflagged_signatures(kind: Literal["single", "cat"], order_seed: int | None) -> set[frozenset[int]]:
        ir = stim_circuit_to_ir(source, mpp_ancilla=kind)
        order = sorted(ir.graph.edges) if order_seed is None else random_cz_order(ir, order_seed)
        circuit = noisy_circuit(pattern_with_cz_order(ir, order))
        flags = set(flag_detector_indices(circuit, CAT_FLAG_TAG))
        return {fired for fired, _ in error_mechanisms(circuit) if not fired & flags}

    for order_seed in (None, *range(8)):
        assert unflagged_signatures("cat", order_seed) <= one_wire
        assert not unflagged_signatures("single", order_seed) <= one_wire


def test_cat_ancillas_keep_the_distance_three_in_every_cz_order() -> None:
    source = memory_circuit(triangular_color_code(3), 2)
    single = stim_circuit_to_ir(source)
    cat = stim_circuit_to_ir(source, mpp_ancilla="cat")

    for order in _cz_orders(single, seeds=3):
        assert fault_distance_bounds(noisy_circuit(pattern_with_cz_order(single, order))) == (2, 2)
    for order in _cz_orders(cat, seeds=3):
        circuit = noisy_circuit(pattern_with_cz_order(cat, order))
        assert fault_distance_bounds(circuit) == (3, 3)
        # Without the flag checks one fault on a cat ring is a two-qubit data error again.
        assert fault_distance_bounds(drop_detectors(circuit, CAT_FLAG_TAG)) == (2, 2)


def test_cat_ancillas_keep_the_distance_five() -> None:
    source = memory_circuit(triangular_color_code(5), 2)
    single = stim_circuit_to_ir(source)
    cat = stim_circuit_to_ir(source, mpp_ancilla="cat")

    assert fault_distance_bounds(noisy_circuit(pattern_with_cz_order(single, sorted(single.graph.edges)))) == (3, 3)
    assert fault_distance_bounds(noisy_circuit(pattern_with_cz_order(cat, sorted(cat.graph.edges)))) == (5, 5)


def test_default_schedule_keeps_the_distance_with_cat_ancillas() -> None:
    source = memory_circuit(triangular_color_code(3), 3, "X")

    single = noisy_circuit(stim_circuit_to_pattern(source).pattern)
    cat = noisy_circuit(stim_circuit_to_pattern(source, mpp_ancilla="cat").pattern)

    assert fault_distance_bounds(single) == (2, 2)
    assert fault_distance_bounds(cat) == (3, 3)


def _sequenced(source: stim.Circuit, kind: Literal["single", "cat"] = "single") -> stim.Circuit:
    """Compile with the legs of each ancilla node entangled in the order of the MPP targets."""
    ir = stim_circuit_to_ir(source, mpp_ancilla=kind)
    scheduler = ir.make_scheduler()
    assert scheduler.solve_schedule()
    ir.sequence_mpp_legs(scheduler)
    return noisy_circuit(ir.to_pattern(scheduler).pattern)


def test_target_order_schedule_keeps_the_distance_with_cat_ancillas() -> None:
    source = memory_circuit(triangular_color_code(3), 2)

    assert fault_distance_bounds(_sequenced(source)) == (2, 2)
    assert fault_distance_bounds(_sequenced(source, "cat")) == (3, 3)


def _rotated_surface_code_memory(distance: int, rounds: int, *, hooks_cross_logicals: bool) -> stim.Circuit:
    """Write a Z-basis memory experiment of the rotated surface code with plaquette MPPs.

    The logical Z is a row of qubits and the logical X is a column. With
    hooks_cross_logicals the last two targets of an X plaquette share a row
    and those of a Z plaquette share a column, so the two-qubit error of a
    single ancilla crosses the logical operator of its type. Otherwise the
    pairs lie along the logical operators.
    """
    index = {(x, y): x + distance * y for y in range(distance) for x in range(distance)}
    checks: dict[str, list[list[tuple[int, int]]]] = {"X": [], "Z": []}
    for j in range(-1, distance):
        for i in range(-1, distance):
            cells = [cell for cell in ((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)) if cell in index]
            kind = "X" if (i + j) % 2 == 0 else "Z"
            # Weight-two checks are X pairs on the top and bottom rows and Z pairs on the side columns.
            if len(cells) == 4 or (len(cells) == 2 and (cells[0][1] == cells[1][1]) == (kind == "X")):
                by_row = (kind == "X") == hooks_cross_logicals
                checks[kind].append(sorted(cells, key=itemgetter(1, 0) if by_row else itemgetter(0, 1)))
    num_qubits, num_x, count = distance**2, len(checks["X"]), len(checks["X"]) + len(checks["Z"])
    assert count == num_qubits - 1
    qubits = " ".join(map(str, range(num_qubits)))
    products = " ".join("*".join(f"{kind}{index[cell]}" for cell in cells) for kind in "XZ" for cells in checks[kind])
    lines = [f"R {qubits}"]
    for round_index in range(rounds):
        lines.extend(("TICK", f"MPP {products}"))
        for check in range(0 if round_index else num_x, count):
            previous = f" rec[{check - 2 * count}]" if round_index else ""
            lines.append(f"DETECTOR rec[{check - count}]{previous}")
    lines.extend(("TICK", f"M {qubits}"))
    for check, cells in enumerate(checks["Z"], start=num_x):
        data = " ".join(f"rec[{index[cell] - num_qubits}]" for cell in cells)
        lines.append(f"DETECTOR rec[{check - count - num_qubits}] {data}")
    lines.append("OBSERVABLE_INCLUDE(0) " + " ".join(f"rec[{index[x, 0] - num_qubits}]" for x in range(distance)))
    return stim.Circuit("\n".join(lines))


@pytest.mark.parametrize("distance", [3, 5])
def test_single_ancillas_keep_the_surface_code_distance_when_the_target_order_fits(distance: int) -> None:
    fitting = _rotated_surface_code_memory(distance, 2, hooks_cross_logicals=True)
    unfitting = _rotated_surface_code_memory(distance, 2, hooks_cross_logicals=False)
    assert stim.Circuit(str(fitting)).detector_error_model().num_errors == 0

    assert fault_distance_bounds(_sequenced(fitting)) == (distance, distance)
    assert fault_distance_bounds(_sequenced(unfitting)) == ((distance + 1) // 2, (distance + 1) // 2)


def test_pattern_with_cz_order_applies_one_cz_per_slice_in_order() -> None:
    ir = stim_circuit_to_ir(stim.Circuit("RX 0 1 2\nTICK\nMPP Z0*Z1*Z2\nTICK\nMX 0 1 2"), mpp_ancilla="cat")
    order = sorted(ir.graph.edges, reverse=True)

    circuit = stim.Circuit(str(noisy_circuit(pattern_with_cz_order(ir, order), p=0.0)))

    applied = [
        tuple(sorted(target.value for target in inst.targets_copy()))
        for inst in circuit
        if isinstance(inst, stim.CircuitInstruction) and inst.name == "CZ"
    ]
    assert applied == [tuple(sorted(edge)) for edge in order]
    assert circuit.detector_error_model().num_errors == 0


def test_random_cz_order_is_a_reproducible_permutation() -> None:
    ir = stim_circuit_to_ir(stim.Circuit("RX 0 1 2\nTICK\nMPP Z0*Z1*Z2\nTICK\nMX 0 1 2"), mpp_ancilla="cat")

    assert sorted(random_cz_order(ir, 0)) == sorted(ir.graph.edges)
    assert random_cz_order(ir, 0) == random_cz_order(ir, 0)
    assert random_cz_order(ir, 0) != random_cz_order(ir, 1)


def test_pattern_with_cz_order_rejects_an_incomplete_order() -> None:
    ir = stim_circuit_to_ir(stim.Circuit("RX 0 1\nTICK\nMPP Z0*Z1\nTICK\nMX 0 1"))

    with pytest.raises(ValueError, match="every graph edge exactly once"):
        pattern_with_cz_order(ir, sorted(ir.graph.edges)[1:])
    with pytest.raises(ValueError, match="every graph edge exactly once"):
        pattern_with_cz_order(ir, [*sorted(ir.graph.edges), next(iter(ir.graph.edges))])
