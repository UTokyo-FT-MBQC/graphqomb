"""Shared internal helpers for parsing Stim QEC data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.sparse import csr_array, lil_array

from graphqomb.common import Axis, Sign
from graphqomb.qec.qeccode import Coordinate, StabilizerCode

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import stim


PauliSupport = tuple[tuple[int, str], ...]

# Stim canonicalizes Z-axis aliases when parsing circuits: RZ becomes R,
# MZ becomes M, and MRZ becomes MR before instructions reach these tables.
RESET_AXES: dict[str, Axis] = {"R": Axis.Z, "RX": Axis.X, "RY": Axis.Y}
SINGLE_MEASUREMENT_AXES: dict[str, Axis] = {"M": Axis.Z, "MX": Axis.X, "MY": Axis.Y}
# Measure-reset gates measure and re-prepare along the same axis.
MEASURE_RESET_AXES: dict[str, Axis] = {"MR": Axis.Z, "MRX": Axis.X, "MRY": Axis.Y}
DIRECT_MEASUREMENT_AXES: dict[str, Axis] = {**SINGLE_MEASUREMENT_AXES, **MEASURE_RESET_AXES}
PAIR_MEASUREMENT_AXES: dict[str, Axis] = {"MXX": Axis.X, "MYY": Axis.Y, "MZZ": Axis.Z}
RESET_GATES: dict[Axis, str] = {axis: name for name, axis in RESET_AXES.items()}


@dataclass(frozen=True)
class StimMppExtraction:
    r"""Stabilizer-code data extracted from Stim MPP products.

    Attributes
    ----------
    code : StabilizerCode
        Dense-column stabilizer code using the ``[Hx | Hz]`` convention.
    stim_to_column : `dict`\[`int`, `int`\]
        Mapping from original Stim qubit ids to dense matrix columns.
    column_to_stim : `dict`\[`int`, `int`\]
        Inverse dense-column mapping.
    supports : `tuple`\[``PauliSupport``, ...\]
        Original Stim Pauli supports, one support per stabilizer row.
    detector_rows : `tuple`\[`frozenset`\[`int`\], ...\]
        Detector groups as selected-MPP stabilizer row indices.
    logical_observable_rows : `dict`\[`int`, `frozenset`\[`int`\]\]
        Logical observables as selected-MPP stabilizer row indices.
    detector_record_indices : `tuple`\[`frozenset`\[`int`\], ...\]
        Absolute Stim measurement-record indices for selected detectors.
    logical_observable_record_indices : `dict`\[`int`, `frozenset`\[`int`\]\]
        Absolute Stim record indices for selected logical observables.
    """

    code: StabilizerCode
    stim_to_column: dict[int, int]
    column_to_stim: dict[int, int]
    supports: tuple[PauliSupport, ...]
    detector_rows: tuple[frozenset[int], ...]
    logical_observable_rows: dict[int, frozenset[int]]
    detector_record_indices: tuple[frozenset[int], ...] = ()
    logical_observable_record_indices: dict[int, frozenset[int]] = field(default_factory=dict)

    def detector_groups(self, ancilla_nodes: Mapping[int, int]) -> list[set[int]]:
        r"""Return detector groups mapped to graph node ids for ``qompile``.

        Returns
        -------
        `list`\[`set`\[`int`\]\]
            Detector groups suitable for ``qompile``.
        """
        return [_map_rows_to_nodes(rows, ancilla_nodes, "detector") for rows in self.detector_rows]

    def logical_observables(self, ancilla_nodes: Mapping[int, int]) -> dict[int, set[int]]:
        r"""Return logical observables mapped to graph node ids for ``qompile``.

        Returns
        -------
        `dict`\[`int`, `set`\[`int`\]\]
            Logical-observable node groups keyed by Stim observable index.
        """
        return {
            logical_idx: _map_rows_to_nodes(rows, ancilla_nodes, f"logical observable {logical_idx}")
            for logical_idx, rows in self.logical_observable_rows.items()
        }


def stim_mpp_extraction_from_records(
    supports: Sequence[PauliSupport],
    record_indices: Sequence[int],
    *,
    coordinate_by_stim_id: Mapping[int, Coordinate],
    detector_record_indices: Sequence[frozenset[int]],
    logical_observable_record_indices: Mapping[int, frozenset[int]],
) -> StimMppExtraction:
    """Build an MPP extraction from globally indexed measurement records.

    Returns
    -------
    StimMppExtraction
        Extracted stabilizer data and record metadata.

    Raises
    ------
    ValueError
        If the support and record counts differ.
    """
    if len(supports) != len(record_indices):
        msg = "MPP support count does not match its measurement-record count."
        raise ValueError(msg)

    record_to_row = {record_index: row for row, record_index in enumerate(record_indices)}
    selected_detector_rows: list[frozenset[int]] = []
    selected_detector_records: list[frozenset[int]] = []
    for records in detector_record_indices:
        rows = frozenset(record_to_row[record] for record in records if record in record_to_row)
        if rows:
            selected_detector_rows.append(rows)
            selected_detector_records.append(records)

    selected_logical_rows: dict[int, frozenset[int]] = {}
    selected_logical_records: dict[int, frozenset[int]] = {}
    for logical_idx, records in logical_observable_record_indices.items():
        rows = frozenset(record_to_row[record] for record in records if record in record_to_row)
        if rows:
            selected_logical_rows[logical_idx] = rows
            selected_logical_records[logical_idx] = records

    matrix, stim_to_column, column_to_stim, qubit_coords = _build_stabilizer_data(
        supports,
        coordinate_by_stim_id,
    )

    return StimMppExtraction(
        code=StabilizerCode(matrix, qubit_coords=qubit_coords),
        stim_to_column=stim_to_column,
        column_to_stim=column_to_stim,
        supports=tuple(supports),
        detector_rows=tuple(selected_detector_rows),
        logical_observable_rows=selected_logical_rows,
        detector_record_indices=tuple(selected_detector_records),
        logical_observable_record_indices=selected_logical_records,
    )


def extract_qubit_coordinates(
    circuit: stim.Circuit,
    *,
    coord_dims: int,
) -> dict[int, Coordinate]:
    """Return final Stim qubit coordinates projected to ``coord_dims``.

    Returns
    -------
    dict[int, Coordinate]
        Final coordinates keyed by Stim qubit id.

    Raises
    ------
    ValueError
        If a coordinate has fewer dimensions than requested, or if two qubits
        share the same XY projection. Graph nodes are placed by the first two
        coordinate components, so XY collisions produce coincident nodes.
    """
    coordinates: dict[int, Coordinate] = {}
    stim_ids_by_xy: dict[Coordinate, list[int]] = {}
    for stim_id, values in circuit.get_final_qubit_coordinates().items():
        if len(values) < coord_dims:
            msg = (
                f"QUBIT_COORDS for qubit {stim_id} has {len(values)} coordinate(s), "
                f"fewer than requested coord_dims={coord_dims}."
            )
            raise ValueError(msg)
        coordinate = tuple(float(value) for value in values[:coord_dims])
        coordinates[int(stim_id)] = coordinate
        stim_ids_by_xy.setdefault(coordinate[:2], []).append(int(stim_id))
    duplicates = {xy: stim_ids for xy, stim_ids in stim_ids_by_xy.items() if len(stim_ids) > 1}
    if duplicates:
        described = "; ".join(f"qubits {sorted(stim_ids)} share {xy}" for xy, stim_ids in sorted(duplicates.items()))
        msg = f"QUBIT_COORDS must have distinct XY projections: {described}."
        raise ValueError(msg)
    return coordinates


def record_targets_to_absolute_indices(
    targets: Sequence[stim.GateTarget],
    *,
    measurement_count: int,
    instruction_name: str,
) -> frozenset[int]:
    """Resolve relative Stim record targets to absolute parity indices.

    Returns
    -------
    frozenset[int]
        Absolute record indices after parity cancellation.

    Raises
    ------
    ValueError
        If a target is invalid or references a record before time began.
    """
    record_indices: set[int] = set()
    for target in targets:
        if not target.is_measurement_record_target:
            msg = f"{instruction_name} contains unsupported target {target!r}; only rec targets are supported."
            raise ValueError(msg)
        record_index = measurement_count + int(target.value)
        if not 0 <= record_index < measurement_count:
            msg = f"{instruction_name} refers to measurement record {record_index} before the beginning of time."
            raise ValueError(msg)
        if record_index in record_indices:
            record_indices.remove(record_index)
        else:
            record_indices.add(record_index)
    return frozenset(record_indices)


def observable_index(instruction: stim.CircuitInstruction) -> int:
    """Return the logical-observable index from a Stim annotation.

    Returns
    -------
    int
        Logical-observable index.

    Raises
    ------
    ValueError
        If the annotation does not have one integer argument.
    """
    args = instruction.gate_args_copy()
    if len(args) != 1 or not args[0].is_integer():
        msg = "OBSERVABLE_INCLUDE must have one integer observable index."
        raise ValueError(msg)
    return int(args[0])


def mpp_targets_to_products(targets: Sequence[stim.GateTarget]) -> list[PauliSupport]:
    """Parse Stim MPP targets into unsigned Pauli products.

    Returns
    -------
    list[PauliSupport]
        Parsed Pauli products in target order.

    Raises
    ------
    ValueError
        If the target sequence is signed or malformed.
    """
    signed_products = mpp_targets_to_signed_products(targets)
    if any(sign is Sign.MINUS for _support, sign in signed_products):
        msg = "Signed MPP products are not supported; inverted Pauli targets cannot be imported."
        raise ValueError(msg)
    return [support for support, _sign in signed_products]


def mpp_targets_to_signed_products(
    targets: Sequence[stim.GateTarget],
) -> list[tuple[PauliSupport, Sign]]:
    """Parse Stim MPP targets into Pauli supports and measurement signs.

    Returns
    -------
    list[tuple[PauliSupport, Sign]]
        Parsed products and their result signs in target order.

    Raises
    ------
    ValueError
        If the target sequence is malformed.
    """
    products: list[tuple[PauliSupport, Sign]] = []
    current: list[tuple[int, str]] = []
    seen_in_current: set[int] = set()
    expect_pauli = True
    negative = False

    for target in targets:
        if target.is_combiner:
            if expect_pauli:
                msg = "Invalid MPP target list: unexpected combiner."
                raise ValueError(msg)
            expect_pauli = True
            continue

        pauli = _target_pauli(target)
        if current and not expect_pauli:
            products.append((tuple(current), Sign.MINUS if negative else Sign.PLUS))
            current = []
            seen_in_current = set()
            negative = False

        qid = int(target.value)
        if qid in seen_in_current:
            msg = f"Invalid MPP product: qubit {qid} appears more than once."
            raise ValueError(msg)
        current.append((qid, pauli))
        seen_in_current.add(qid)
        negative ^= target.is_inverted_result_target
        expect_pauli = False

    if expect_pauli:
        msg = "Invalid MPP target list: trailing combiner or empty product."
        raise ValueError(msg)
    products.append((tuple(current), Sign.MINUS if negative else Sign.PLUS))
    return products


def pauli_products_commute(left: PauliSupport, right: PauliSupport) -> bool:
    """Return whether two unsigned Pauli products commute.

    Returns
    -------
    bool
        Whether the two products commute.
    """
    right_by_qubit = dict(right)
    anticommuting_overlaps = sum(qubit in right_by_qubit and pauli != right_by_qubit[qubit] for qubit, pauli in left)
    return anticommuting_overlaps % 2 == 0


def plain_qubit_target(target: stim.GateTarget, instruction_name: str) -> int:
    """Return a plain Stim qubit target.

    Rejects Pauli-typed and inverted-result targets, which the importer cannot
    represent. ``stim_mpp_rewriter._plain_qubit`` deliberately accepts those
    because it only needs the qubit id an instruction acts on.

    Returns
    -------
    int
        Stim qubit id.

    Raises
    ------
    ValueError
        If the target is not a plain qubit target.
    """
    qubit_value = target.qubit_value
    if qubit_value is None or not target.is_qubit_target:
        msg = f"{instruction_name} contains unsupported target {target!r}; only plain qubit targets are supported."
        raise ValueError(msg)
    return int(qubit_value)


def _build_stabilizer_data(
    supports: Sequence[PauliSupport],
    coordinate_by_stim_id: Mapping[int, Coordinate],
) -> tuple[csr_array[Any, tuple[int, int]], dict[int, int], dict[int, int], dict[int, Coordinate]]:
    stim_ids = sorted({qid for support in supports for qid, _pauli in support})
    stim_to_column = {qid: column for column, qid in enumerate(stim_ids)}
    column_to_stim = {column: qid for qid, column in stim_to_column.items()}

    num_qubits = len(stim_ids)
    matrix = lil_array((len(supports), 2 * num_qubits), dtype=np.bool_)
    for row, support in enumerate(supports):
        for stim_id, pauli in support:
            column = stim_to_column[stim_id]
            if pauli in {"X", "Y"}:
                matrix[row, column] = True
            if pauli in {"Z", "Y"}:
                matrix[row, num_qubits + column] = True

    qubit_coords = {stim_to_column[qid]: coord for qid, coord in coordinate_by_stim_id.items() if qid in stim_to_column}
    stabilizer_matrix = csr_array(matrix, shape=(len(supports), 2 * num_qubits))
    return stabilizer_matrix, stim_to_column, column_to_stim, qubit_coords


def _target_pauli(target: stim.GateTarget) -> str:
    if target.is_x_target:
        return "X"
    if target.is_y_target:
        return "Y"
    if target.is_z_target:
        return "Z"
    msg = f"Unsupported MPP target: {target!r}."
    raise ValueError(msg)


def _map_rows_to_nodes(rows: frozenset[int], ancilla_nodes: Mapping[int, int], label: str) -> set[int]:
    missing_rows = sorted(row for row in rows if row not in ancilla_nodes)
    if missing_rows:
        msg = f"Cannot map {label}; ancilla node map is missing stabilizer row(s): {missing_rows}."
        raise ValueError(msg)
    return {ancilla_nodes[row] for row in rows}
