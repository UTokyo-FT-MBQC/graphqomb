"""Import Clifford+T circuit text using measurement or injection lowering."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from graphqomb.circuit import CircuitScheduleStrategy
from graphqomb.qeccode import YFoliation
from graphqomb.stim_glue._t_gates import parse_t_circuit
from graphqomb.stim_glue.importer import StimImportResult, _import_circuit

if TYPE_CHECKING:
    from graphqomb.scheduler import ScheduleConfig


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
    r"""Import Stim-format Clifford+T text into an MBQC pattern.

    Parameters
    ----------
    text : str
        The existing Stim-import subset, extended by T and T_DAG. Write each
        T instruction on its own line, including inside REPEAT blocks.
        Source noise is removed, just as for the Stim importer.
    t_mode : {"measurement", "injection"}
        ``measurement`` represents each T as J(pi/4), J(0), using an
        XY measurement at -pi/4. ``injection`` replaces that measurement
        with a ZZ-probe gadget, a scheduled magic resource, and conditional S correction;
        T_DAG uses the opposite signs. All measurements are then Pauli.
    coord_dims, schedule_strategy, y_foliation, merge_safe_ticks, schedule_config, schedule_timeout
        See `stim_circuit_to_pattern`.

    Returns
    -------
    StimImportResult
        Pattern and original measurement-record mapping. Pass its
        ``preparation_gates`` to ``clifft_compile`` to prepare injection resources.
        These resources and their synthesis rotations are ideal; use explicit
        noise models when evaluating noisy MBQC execution.

    Raises
    ------
    ValueError
        For an unknown mode, private transport tags, or unsupported syntax.
    """
    if t_mode not in {"measurement", "injection"}:
        msg = f"Unknown T lowering mode: {t_mode!r}."
        raise ValueError(msg)
    # Stim handles the grammar, target validation, repeats and rec offsets.
    # Tagged I instructions only transport T to the graph-building boundary;
    # they are explicitly protected from the Clifford optimizer there.
    return _import_circuit(
        parse_t_circuit(text),
        t_mode=t_mode,
        coord_dims=coord_dims,
        schedule_strategy=schedule_strategy,
        y_foliation=y_foliation,
        merge_safe_ticks=merge_safe_ticks,
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
    """Import a Clifford+T file; see `clifft_text_to_pattern` for options.

    Returns
    -------
    StimImportResult
        Pattern, record mapping and any required scheduled magic preparations.
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
