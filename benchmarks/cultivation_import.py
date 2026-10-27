"""Compare direct and MBQC execution of a supplied Clifford+T cultivation file."""

# ruff: file-ignore[implicit-namespace-package]

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import clifft
import numpy as np

from graphqomb.command import M
from graphqomb.common import determine_pauli_axis
from graphqomb.noise_model import DepolarizingNoiseModel
from graphqomb.stim_glue import clifft_compile, clifft_text_to_pattern
from graphqomb.stim_glue._t_gates import T_ANGLES, parse_t_circuit  # ruff:ignore[import-private-name]


@dataclass
class Trial:
    """Sampling counters and compilation cost for one execution condition."""

    name: str
    shots: int
    seed: int
    peak_active_width: int
    detectors: int
    observables: int
    accepted: int
    accepted_observable_flips: int
    detector_ones: list[int]
    observable_ones: list[int]
    compile_seconds: float
    sample_seconds: float
    max_space: int | None = None
    depth: int | None = None
    active_volume: int | None = None
    non_pauli_measurements: int | None = None
    magic_preparations: int | None = None
    import_seconds: float | None = None


def sample_trial(text: str, name: str, shots: int, seed: int) -> Trial:
    """Sample all records, accepting a shot only if all detectors are zero.

    Returns
    -------
    Trial
        Raw observable-parity flips; no decoder or confidence estimate is implied.
    """
    start = time.perf_counter()
    program = clifft.compile(text, normalize_syndromes=False)
    compiled = time.perf_counter()
    sample = clifft.sample(program, shots=shots, seed=seed)
    sampled = time.perf_counter()
    detectors = np.asarray(sample.detectors, dtype=bool)
    observables = np.asarray(sample.observables, dtype=bool)
    keep = ~detectors.any(axis=1)
    return Trial(
        name=name,
        shots=shots,
        seed=seed,
        peak_active_width=program.peak_active_width,
        detectors=detectors.shape[1],
        observables=observables.shape[1],
        accepted=int(keep.sum()),
        accepted_observable_flips=int(observables[keep].any(axis=1).sum()),
        detector_ones=detectors.sum(axis=0).tolist(),
        observable_ones=observables.sum(axis=0).tolist(),
        compile_seconds=compiled - start,
        sample_seconds=sampled - compiled,
    )


def main() -> None:  # ruff:ignore[too-many-locals]
    """Record source provenance, software versions, and both lowering trials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("circuit", type=Path)
    parser.add_argument("--shots", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--p", type=float, default=0.001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.circuit.read_text(encoding="utf-8")
    # Let Stim remove source noise without interpreting the tagged T transport
    # operations; restore T before direct Clifft execution. This benchmark's
    # input must use ordinary Pauli noise, without heralded measurement records.
    ideal_source = str(parse_t_circuit(source).without_noise())
    for tag in T_ANGLES:
        ideal_source = ideal_source.replace(f"I[{tag}]", tag.rsplit(".", 1)[1])
    fault_lines = ideal_source.splitlines()
    for index, line in enumerate(fault_lines):
        if line.split()[0] in {"T", "T_DAG"}:
            fault_instruction = f"X {line.split()[1]}"
            fault_lines.insert(index + 1, fault_instruction)
            break
    else:
        parser.error("The comparison requires at least one T or T_DAG instruction.")
    fault_source = "\n".join(fault_lines)
    trials = [
        sample_trial(ideal_source, "source_ideal", args.shots, args.seed),
        sample_trial(source, "source_original_noise", args.shots, args.seed),
        sample_trial(fault_source, "source_x_fault", args.shots, args.seed),
    ]
    for mode in ("measurement", "injection"):
        start = time.perf_counter()
        imported = clifft_text_to_pattern(source, t_mode=mode)
        import_seconds = time.perf_counter() - start
        pattern = imported.pattern
        for noisy in (False, True):
            text = clifft_compile(
                pattern,
                preparation_gates=imported.preparation_gates,
                noise_models=[DepolarizingNoiseModel(args.p)] if noisy else [],
            )
            trial = sample_trial(text, f"{mode}_{'mbqc_noise' if noisy else 'ideal'}", args.shots, args.seed)
            trial.max_space = pattern.max_space
            trial.depth = pattern.depth
            trial.active_volume = pattern.active_volume
            trial.non_pauli_measurements = sum(
                isinstance(command, M) and determine_pauli_axis(command.meas_basis) is None for command in pattern
            )
            trial.magic_preparations = len(imported.preparation_gates)
            trial.import_seconds = import_seconds
            trials.append(trial)
        fault_import = clifft_text_to_pattern(fault_source, t_mode=mode)
        fault_text = clifft_compile(fault_import.pattern, preparation_gates=fault_import.preparation_gates)
        trials.append(sample_trial(fault_text, f"{mode}_x_fault", args.shots, args.seed))
    root = Path(__file__).resolve().parents[1]
    source_files = [Path(__file__), *sorted((root / "graphqomb" / "stim_glue").glob("*.py"))]
    by_name = {trial.name: trial for trial in trials}
    tolerance = max(0.02, 6 / math.sqrt(2 * args.shots))
    checks = {}
    for mode in ("measurement", "injection"):
        for condition in ("ideal", "x_fault"):
            reference = by_name[f"source_{condition}"]
            actual = by_name[f"{mode}_{condition}"]
            reference_counts = np.asarray(reference.detector_ones + reference.observable_ones)
            actual_counts = np.asarray(actual.detector_ones + actual.observable_ones)
            discrepancy = float(np.max(np.abs(reference_counts - actual_counts)) / args.shots)
            checks[f"{mode}_{condition}"] = {
                "max_marginal_discrepancy": discrepancy,
                "tolerance": tolerance,
                "passed": discrepancy <= tolerance,
            }
    report = {
        "source_name": args.circuit.name,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "environment": {name: version(name) for name in ("graphqomb", "stim", "clifft", "numpy", "ortools")},
        "python": platform.python_version(),
        "implementation_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files
        },
        "mbqc_noise": {"model": "DepolarizingNoiseModel", "p1": args.p, "p2": args.p},
        "single_fault": f"{fault_instruction} immediately after the first T/T_DAG, with source noise removed.",
        "acceptance": "All detectors zero; raw observable parity, without decoding or syndrome normalization.",
        "limitations": "Source and MBQC noise act at different locations; these noisy rates are not equivalent models.",
        "equivalence_checks": checks,
        "trials": [asdict(trial) for trial in trials],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["trials"], indent=2))  # ruff:ignore[print]


if __name__ == "__main__":
    main()
