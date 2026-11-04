r"""Importing a T-Gate Cultivation Circuit
=======================================

Run the example from the repository root (Clifft requires Python 3.12 or later):

.. code-block:: sh

    uv run --extra clifft python -m examples.ftqc.cultivation_import

This example imports the distance-3 T-gate cultivation circuit distributed
with Clifft (originally from SOFT, Apache-2.0) and compares Clifft sampling of
the source circuit with sampling after MBQC compilation. The circuit is
downloaded from a pinned commit and checked against its SHA-256 hash.

``clifft_text_to_pattern`` offers two T lowerings. ``measurement`` keeps
:math:`\pm\pi/4` XY-plane measurements, and ``injection`` consumes scheduled
:math:`|T\rangle` resources with only Pauli measurements. Pass the returned
``preparation_gates`` to ``clifft_compile``.

Both lowerings are checked against the source without noise and with a single
X fault after the first T gate. Noisy rows are raw observable flips among
shots with all detectors zero. Source and MBQC noise act at different
locations, so these rows are not equivalent noise models.
"""

# %%
import hashlib
import sys
import urllib.request
from pathlib import Path

import clifft
import numpy as np
from numpy.typing import NDArray

from graphqomb.command import M
from graphqomb.common import determine_pauli_axis
from graphqomb.noise_model import DepolarizingNoiseModel
from graphqomb.stim_glue import clifft_compile, clifft_text_to_pattern

URL = (
    "https://raw.githubusercontent.com/unitaryfoundation/clifft/"
    "847757e2bc8c4f7439b9e92011550fd61719eb85/docs/guide/circuits/circuit_d3_t_gate_p0.001.stim"
)
SHA256 = "094ae095e249c90ac7b0cf82ef187afdf634a0ef8a82331a81b4b9100a902798"
SHOTS = 16384
SEED = 17

# %%
# Download the circuit once into the gitignored ``output/`` directory.
script_path = Path(globals().get("__file__", sys.argv[0])).resolve()
path = script_path.parent / "output" / "cultivation_d3.stim"
if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=30) as response:
        path.write_bytes(response.read())
source = path.read_text(encoding="utf-8")
if hashlib.sha256(source.encode()).hexdigest() != SHA256:
    msg = f"{path} does not match the pinned cultivation circuit."
    raise RuntimeError(msg)

# %%
# Insert one X fault after the first T gate.
lines = source.splitlines()
first_t = next(index for index, line in enumerate(lines) if line.split()[:1] in (["T"], ["T_DAG"]))
lines.insert(first_t + 1, f"X {lines[first_t].split()[1]}")
fault = "\n".join(lines)


def sample(text: str, *, noiseless: bool = False) -> tuple[NDArray[np.float64], int, int]:
    r"""Sample detector and observable marginals and postselected observable flips.

    Returns
    -------
    `tuple`\[`numpy.ndarray`, `int`, `int`\]
        Marginal frequencies, accepted shots, and observable flips among them.
    """
    passes = clifft.default_hir_pass_manager()
    if noiseless:
        passes.add(clifft.RemoveNoisePass())
    program = clifft.compile(text, normalize_syndromes=False, hir_passes=passes)
    result = clifft.sample(program, shots=SHOTS, seed=SEED)
    detectors = np.asarray(result.detectors, dtype=bool)
    observables = np.asarray(result.observables, dtype=bool)
    keep = ~detectors.any(axis=1)
    marginals = np.concatenate([detectors.mean(axis=0), observables.mean(axis=0)])
    return marginals, int(keep.sum()), int(observables[keep].any(axis=1).sum())


# The importer removes source noise, so compare with the noiseless source.
references = {"ideal": sample(source, noiseless=True), "fault": sample(fault, noiseless=True)}
_marginals, accepted, flips = sample(source)
print(f"Source with its own noise: accepted={accepted}/{SHOTS}, observable flips={flips}")

# %%
# Import with each lowering and compare it with the source.
for mode in ("measurement", "injection"):
    imported = clifft_text_to_pattern(source, t_mode=mode)
    pattern = imported.pattern
    non_pauli = sum(isinstance(c, M) and determine_pauli_axis(c.meas_basis) is None for c in pattern)
    print(
        f"{mode}: non-Pauli measurements={non_pauli}, magic preparations={len(imported.preparation_gates)}, "
        f"max space={pattern.max_space}, depth={pattern.depth}, active volume={pattern.active_volume}"
    )
    for condition, text in (("ideal", source), ("fault", fault)):
        lowered = clifft_text_to_pattern(text, t_mode=mode)
        lowered_text = clifft_compile(lowered.pattern, preparation_gates=lowered.preparation_gates)
        marginals, accepted, _flips = sample(lowered_text)
        discrepancy = np.max(np.abs(marginals - references[condition][0]))
        print(f"  {condition}: accepted={accepted}/{SHOTS}, max marginal discrepancy={discrepancy:.4f}")
    noise_models = [DepolarizingNoiseModel(1e-3)]
    noisy = clifft_compile(pattern, preparation_gates=imported.preparation_gates, noise_models=noise_models)
    _marginals, accepted, flips = sample(noisy)
    print(f"  MBQC noise p=1e-3: accepted={accepted}/{SHOTS}, observable flips={flips}")

# %%
