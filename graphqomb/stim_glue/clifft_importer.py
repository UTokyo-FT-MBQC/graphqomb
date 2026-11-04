"""Import Clifford+T circuit text into unscheduled IRs or measurement patterns.

This module provides:

- `clifft_text_to_ir`, `clifft_file_to_ir`: Import before scheduling and lowering.
- `clifft_text_to_pattern`, `clifft_file_to_pattern`: Import and lower with a solved schedule.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from graphqomb.circuit import CircuitScheduleStrategy
from graphqomb.qeccode import YFoliation
from graphqomb.stim_glue._t_gates import parse_t_circuit
from graphqomb.stim_glue.importer import StimImportIR, StimImportResult, _import_circuit_ir, _schedule_and_lower

if TYPE_CHECKING:
    from graphqomb.scheduler import ScheduleConfig


def clifft_text_to_ir(  # ruff:ignore[too-many-arguments]
    text: str,
    *,
    t_mode: Literal["measurement", "injection"] = "measurement",
    coord_dims: int = 2,
    schedule_strategy: CircuitScheduleStrategy = CircuitScheduleStrategy.PARALLEL,
    y_foliation: YFoliation = YFoliation.TYPE_I,
    merge_safe_ticks: bool = False,
) -> StimImportIR:
    r"""Import Stim text extended with T and T_DAG without scheduling or lowering.

    Write each T instruction on its own line, including inside REPEAT blocks.
    Source noise is removed. The other options are those of
    `stim_circuit_to_ir`.

    Parameters
    ----------
    text : `str`
        The supported Stim subset extended with T and T_DAG.
    t_mode : {"measurement", "injection"}, optional
        ``"measurement"`` lowers T to an XY measurement at -pi/4.
        ``"injection"`` consumes a scheduled T|+> resource through a ZZ probe
        and conditional S, so all measurements are Pauli. T_DAG uses the
        opposite signs. By default ``"measurement"``.

    Returns
    -------
    `StimImportIR`
        Imported graph, feedforward, source mappings, and injection measurement
        dependencies. Use ``ir.make_scheduler()`` to retain probe-before-magic
        order, then ``ir.to_pattern(scheduler)`` to lower. Pass the returned
        ``preparation_gates`` to `clifft_compile`.

    Raises
    ------
    ValueError
        If t_mode is unknown, or the text uses reserved transport tags or
        unsupported syntax.
    """
    if t_mode not in {"measurement", "injection"}:
        msg = f"Unknown T lowering mode: {t_mode!r}."
        raise ValueError(msg)
    # Stim handles the grammar, target validation, repeats and rec offsets.
    # Tagged I instructions only transport T to the graph-building boundary;
    # they are explicitly protected from the Clifford optimizer there.
    return _import_circuit_ir(
        parse_t_circuit(text),
        t_mode=t_mode,
        coord_dims=coord_dims,
        schedule_strategy=schedule_strategy,
        y_foliation=y_foliation,
        merge_safe_ticks=merge_safe_ticks,
    )


def clifft_file_to_ir(  # ruff:ignore[too-many-arguments]
    path: str | Path,
    *,
    t_mode: Literal["measurement", "injection"] = "measurement",
    coord_dims: int = 2,
    schedule_strategy: CircuitScheduleStrategy = CircuitScheduleStrategy.PARALLEL,
    y_foliation: YFoliation = YFoliation.TYPE_I,
    merge_safe_ticks: bool = False,
) -> StimImportIR:
    """Import a Stim+T file without scheduling; see `clifft_text_to_ir`.

    Returns
    -------
    `StimImportIR`
        Resource graph, feedforward, and source mappings before scheduling.
    """
    return clifft_text_to_ir(
        Path(path).read_text(encoding="utf-8"),
        t_mode=t_mode,
        coord_dims=coord_dims,
        schedule_strategy=schedule_strategy,
        y_foliation=y_foliation,
        merge_safe_ticks=merge_safe_ticks,
    )


def clifft_text_to_pattern(  # ruff:ignore[too-many-arguments]
    text: str,
    *,
    t_mode: Literal["measurement", "injection"] = "measurement",
    coord_dims: int = 2,
    schedule_strategy: CircuitScheduleStrategy = CircuitScheduleStrategy.PARALLEL,
    y_foliation: YFoliation = YFoliation.TYPE_I,
    merge_safe_ticks: bool = False,
    schedule_config: ScheduleConfig | None = None,
    schedule_timeout: int = 60,
) -> StimImportResult:
    """Import Stim text extended with T and T_DAG into a GraphQOMB pattern.

    Import options are those of `clifft_text_to_ir`. schedule_config and
    schedule_timeout select the pattern-level scheduler as in
    `stim_circuit_to_pattern`.

    Returns
    -------
    `StimImportResult`
        Imported pattern and source mappings. Pass its ``preparation_gates``
        to `clifft_compile`.
    """
    return _schedule_and_lower(
        clifft_text_to_ir(
            text,
            t_mode=t_mode,
            coord_dims=coord_dims,
            schedule_strategy=schedule_strategy,
            y_foliation=y_foliation,
            merge_safe_ticks=merge_safe_ticks,
        ),
        schedule_config=schedule_config,
        schedule_timeout=schedule_timeout,
    )


def clifft_file_to_pattern(  # ruff:ignore[too-many-arguments]
    path: str | Path,
    *,
    t_mode: Literal["measurement", "injection"] = "measurement",
    coord_dims: int = 2,
    schedule_strategy: CircuitScheduleStrategy = CircuitScheduleStrategy.PARALLEL,
    y_foliation: YFoliation = YFoliation.TYPE_I,
    merge_safe_ticks: bool = False,
    schedule_config: ScheduleConfig | None = None,
    schedule_timeout: int = 60,
) -> StimImportResult:
    """Import a Stim file extended with T and T_DAG; see `clifft_text_to_pattern`.

    Returns
    -------
    `StimImportResult`
        Imported pattern and record mapping.
    """
    return clifft_text_to_pattern(
        Path(path).read_text(encoding="utf-8"),
        t_mode=t_mode,
        coord_dims=coord_dims,
        schedule_strategy=schedule_strategy,
        y_foliation=y_foliation,
        merge_safe_ticks=merge_safe_ticks,
        schedule_config=schedule_config,
        schedule_timeout=schedule_timeout,
    )
