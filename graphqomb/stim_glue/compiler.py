"""Pattern to stim compiler.

This module provides:

- `stim_compile`: Function to compile a pattern into stim format.
"""

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING

from graphqomb._tag import escape_tag
from graphqomb.command import TICK, E, M, N
from graphqomb.common import Axis, AxisMeasBasis, MeasBasis, Sign, determine_pauli_axis
from graphqomb.noise_model import (
    Coordinate,
    EntangleEvent,
    IdleEvent,
    MeasureEvent,
    MeasurementFlip,
    NodeInfo,
    NoiseModel,
    NoiseOp,
    NoisePlacement,
    PrepareEvent,
    default_noise_placement,
    noise_op_to_stim,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from graphqomb.pattern import Pattern


class _StimCompiler:
    def __init__(
        self,
        pattern: Pattern,
        *,
        emit_qubit_coords: bool,
        noise_models: Sequence[NoiseModel],
        tick_duration: float,
    ) -> None:
        self._pattern = pattern
        self._pframe = pattern.pauli_frame
        self._coord_lookup = pattern.coordinates
        self._emit_qubit_coords = emit_qubit_coords
        self._noise_models = noise_models
        self._tick_duration = tick_duration
        self._stim_io = StringIO()
        self._meas_order: dict[int, int] = {}
        self._rec_index = 0
        self._alive_nodes: set[int] = set(pattern.input_node_indices)
        self._touched_nodes: set[int] = set()
        self._tick = 0

    def compile(self) -> str:
        self._emit_input_nodes()
        self._process_commands()
        total = self._rec_index
        self._emit_detectors(total)
        if self._pframe.logical_observables:
            self._emit_observables(total)
        return self._stim_io.getvalue().strip()

    def _emit_detectors(self, total_measurements: int) -> None:
        for checks, tag in zip(self._pframe.detector_groups(), self._pframe.parity_check_tags, strict=True):
            targets = [f"rec[{self._meas_order[c] - total_measurements}]" for c in checks]
            tag_suffix = f"[{escape_tag(tag)}]" if tag else ""
            self._stim_io.write(f"DETECTOR{tag_suffix} {' '.join(targets)}\n")

    def _emit_observables(self, total_measurements: int) -> None:
        for log_idx, group in sorted(self._pframe.logical_observable_groups().items()):
            targets = [f"rec[{self._meas_order[n] - total_measurements}]" for n in group]
            self._stim_io.write(f"OBSERVABLE_INCLUDE({log_idx}) {' '.join(targets)}\n")

    def _emit_input_nodes(self) -> None:
        coordinates = self._pattern.input_coordinates if self._emit_qubit_coords else None
        for node in self._pattern.input_node_indices:
            coord = coordinates.get(node) if coordinates else None
            axis = self._pattern.input_initialization_axes.get(node, Axis.X)
            self._process_prepare(node, coord, is_input=True, init_axis=axis)

    def _process_commands(self) -> None:
        for cmd in self._pattern:
            if isinstance(cmd, N):
                self._process_prepare(cmd.node, cmd.coordinate, is_input=False)
            elif isinstance(cmd, E):
                self._handle_entangle(cmd.nodes)
            elif isinstance(cmd, M):
                self._handle_measure(cmd.node, cmd.meas_basis)
            elif isinstance(cmd, TICK):
                self._handle_tick()
            else:
                msg = f"Unsupported command for stim compilation: {type(cmd).__name__}"
                raise TypeError(msg)

    def _process_prepare(
        self,
        node: int,
        coordinate: tuple[float, ...] | None,
        *,
        is_input: bool,
        init_axis: Axis = Axis.X,
    ) -> None:
        event = PrepareEvent(time=self._tick, node=self._node_info(node), is_input=is_input)
        ops = self._validate_ops_for_event(event, self._collect_noise_ops_from_models(lambda m: m.on_prepare(event)))
        default_placement = default_noise_placement(event)
        self._rec_index += self._emit_noise_ops(ops, NoisePlacement.BEFORE, default_placement)

        coord = coordinate if self._emit_qubit_coords else None
        if coord is not None:
            self._stim_io.write(f"QUBIT_COORDS({', '.join(str(c) for c in coord)}) {node}\n")
        reset_instr = {Axis.X: "RX", Axis.Y: "RY", Axis.Z: "R"}[init_axis]
        self._stim_io.write(f"{reset_instr} {node}\n")

        self._rec_index += self._emit_noise_ops(ops, NoisePlacement.AFTER, default_placement)
        self._alive_nodes.add(node)
        self._touched_nodes.add(node)

    def _handle_entangle(self, nodes: tuple[int, int]) -> None:
        n0, n1 = nodes
        edge: tuple[int, int] = (n0, n1) if n0 < n1 else (n1, n0)
        event = EntangleEvent(time=self._tick, node0=self._node_info(n0), node1=self._node_info(n1), edge=edge)
        ops = self._validate_ops_for_event(event, self._collect_noise_ops_from_models(lambda m: m.on_entangle(event)))
        default_placement = default_noise_placement(event)
        self._rec_index += self._emit_noise_ops(ops, NoisePlacement.BEFORE, default_placement)

        self._stim_io.write(f"CZ {n0} {n1}\n")
        self._touched_nodes.update(nodes)
        self._rec_index += self._emit_noise_ops(ops, NoisePlacement.AFTER, default_placement)

    def _handle_measure(self, node: int, meas_basis: MeasBasis) -> None:
        axis = determine_pauli_axis(meas_basis)
        if axis is None:
            msg = f"Unsupported measurement basis: {meas_basis.plane, meas_basis.angle}"
            raise ValueError(msg)
        event = MeasureEvent(time=self._tick, node=self._node_info(node), axis=axis)
        ops = self._validate_ops_for_event(event, self._collect_noise_ops_from_models(lambda m: m.on_measure(event)))

        # `_validate_ops_for_event` already checked that each MeasurementFlip
        # targets this node; combine their flip probabilities by XOR.
        other_ops = [op for op in ops if not isinstance(op, MeasurementFlip)]
        meas_flip_p = 0.0
        for op in ops:
            if isinstance(op, MeasurementFlip):
                meas_flip_p = meas_flip_p * (1 - op.p) + (1 - meas_flip_p) * op.p

        default_placement = default_noise_placement(event)
        self._rec_index += self._emit_noise_ops(other_ops, NoisePlacement.BEFORE, default_placement)

        # Emit measurement with optional flip probability
        meas_instr = {Axis.X: "MX", Axis.Y: "MY", Axis.Z: "MZ"}[axis]
        is_inverted = isinstance(meas_basis, AxisMeasBasis) and meas_basis.sign is Sign.MINUS
        meas_target = f"!{node}" if is_inverted else str(node)
        if meas_flip_p > 0.0:
            self._stim_io.write(f"{meas_instr}({meas_flip_p}) {meas_target}\n")
        else:
            self._stim_io.write(f"{meas_instr} {meas_target}\n")

        self._meas_order[node] = self._rec_index
        self._rec_index += 1
        self._alive_nodes.discard(node)
        self._touched_nodes.add(node)
        self._rec_index += self._emit_noise_ops(other_ops, NoisePlacement.AFTER, default_placement)

    def _handle_tick(self) -> None:
        idle_nodes = sorted(self._alive_nodes - self._touched_nodes)
        if idle_nodes and self._noise_models:
            event = IdleEvent(
                time=self._tick,
                nodes=tuple(self._node_info(node) for node in idle_nodes),
                duration=self._tick_duration,
            )
            ops = self._validate_ops_for_event(event, self._collect_noise_ops_from_models(lambda m: m.on_idle(event)))
            default_placement = default_noise_placement(event)
        else:
            ops = ()
            default_placement = NoisePlacement.AFTER
        self._rec_index += self._emit_noise_ops(ops, NoisePlacement.BEFORE, default_placement)
        self._stim_io.write("TICK\n")
        self._rec_index += self._emit_noise_ops(ops, NoisePlacement.AFTER, default_placement)
        self._touched_nodes.clear()
        self._tick += 1

    def _node_info(self, node: int) -> NodeInfo:
        coord_raw = self._coord_lookup.get(node)
        coord = Coordinate(tuple(coord_raw)) if coord_raw is not None else None
        return NodeInfo(id=node, coord=coord)

    def _collect_noise_ops_from_models(self, get_ops: Callable[[NoiseModel], Iterable[NoiseOp]]) -> tuple[NoiseOp, ...]:
        ops: list[NoiseOp] = []
        for model in self._noise_models:
            ops.extend(get_ops(model))
        return tuple(ops)

    @staticmethod
    def _validate_ops_for_event(
        event: PrepareEvent | EntangleEvent | MeasureEvent | IdleEvent,
        ops: tuple[NoiseOp, ...],
    ) -> tuple[NoiseOp, ...]:
        for op in ops:
            if not isinstance(op, MeasurementFlip):
                continue
            if not isinstance(event, MeasureEvent):
                msg = (
                    "MeasurementFlip can only be returned from NoiseModel.on_measure(). "
                    f"Received it for {type(event).__name__}."
                )
                raise TypeError(msg)
            if op.target != event.node.id:
                msg = (
                    f"MeasurementFlip target mismatch: measurement on node {event.node.id}, "
                    f"but flip targets node {op.target}"
                )
                raise ValueError(msg)
        return ops

    def _emit_noise_ops(
        self, ops: Iterable[NoiseOp], placement: NoisePlacement, default_placement: NoisePlacement
    ) -> int:
        record_delta = 0
        for op in ops:
            op_placement = op.placement
            if op_placement is NoisePlacement.AUTO:
                op_placement = default_placement
            if op_placement is not placement:
                continue
            text, delta = noise_op_to_stim(op)
            if text:
                self._stim_io.write(f"{text}\n")
            record_delta += delta
        return record_delta


def stim_compile(
    pattern: Pattern,
    *,
    emit_qubit_coords: bool = True,
    noise_models: Sequence[NoiseModel] | None = None,
    tick_duration: float = 1.0,
) -> str:
    r"""Compile a pattern to stim format.

    Parameters
    ----------
    pattern : `Pattern`
        The pattern to compile.
    emit_qubit_coords : `bool`, optional
        Whether to emit QUBIT_COORDS instructions for nodes with coordinates,
        by default True.
    noise_models : `collections.abc.Sequence`\[`NoiseModel`\] | `None`, optional
        Custom noise models for injecting Stim noise instructions, by default None.
        Use `DepolarizingNoiseModel` for gate noise and `MeasurementFlipNoiseModel`
        for measurement errors.
    tick_duration : `float`, optional
        Duration associated with each TICK for idle noise, by default 1.0.

    Returns
    -------
    `str`
        The compiled stim string.

    Notes
    -----
    Stim only supports Clifford gates, therefore this compiler only supports
    Pauli measurements (X, Y, Z basis) which correspond to Clifford operations.
    Non-Pauli measurements will raise a ValueError.

    Each parity check group compiles to a ``DETECTOR`` instruction, and the
    group's tag from `graphqomb.pauli_frame.PauliFrame.parity_check_tags` is
    emitted as a Stim instruction tag (for example ``DETECTOR[type=flag]``).
    Use `graphqomb.stim_glue.postselect.flag_postselection_mask` on the
    compiled circuit to post-select flag detectors with sinter.

    Examples
    --------
    Basic compilation without noise:

    >>> # stim_str = stim_compile(pattern)

    With depolarizing and measurement flip noise:

    >>> from graphqomb.noise_model import DepolarizingNoiseModel, MeasurementFlipNoiseModel
    >>> # stim_str = stim_compile(
    >>> #     pattern,
    >>> #     noise_models=[
    >>> #         DepolarizingNoiseModel(p1=0.001, p2=0.01),
    >>> #         MeasurementFlipNoiseModel(p=0.001)
    >>> #     ]
    >>> # )
    """
    compiler = _StimCompiler(
        pattern,
        emit_qubit_coords=emit_qubit_coords,
        noise_models=tuple(() if noise_models is None else noise_models),
        tick_duration=tick_duration,
    )
    return compiler.compile()
