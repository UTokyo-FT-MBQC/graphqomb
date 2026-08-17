"""Stim interoperability for GraphQOMB.

Importing this package requires the optional ``stim`` dependency
(``pip install graphqomb[stim]``).

This package provides:

- `stim_compile`: Function to compile a pattern into Stim circuit text.
- `stim_circuit_to_pattern`: Function to import a supported Stim circuit into a pattern.
- `stim_text_to_pattern`: Function to import supported Stim text into a pattern.
- `stim_file_to_pattern`: Function to import a supported Stim file into a pattern.
- `StimImportResult`: Result of importing a supported Stim circuit.
- `transpile`: Function to transpile a Stim Clifford circuit into the Clifford J/CZ basis.
- `optimize_j_cz`: Function to remove redundant Clifford J and CZ gates.
- `STIM_GATE_J_ANGLES`: Mapping from single-qubit basis gates to J angles.
- `UnsupportedInstructionError`: Error for instructions outside the supported basis.
- `rewrite_to_mpp`: Function to rewrite a syndrome-extraction circuit into MPP form.
- `MppRewriteResult`: Rewritten circuit with its per-measurement Pauli products.
- `CheckMapping`: Mapping from one measurement record to its Pauli product.
- `UnsupportedSyndromeCircuitError`: Error for unsupported syndrome circuits.
- `stabilizer_code_from_stim_text`: Function to build a stabilizer code from Stim MPP text.
- `stabilizer_code_from_stim_file`: Function to build a stabilizer code from a Stim file.
- `StimMppExtraction`: Stabilizer-code data extracted from Stim MPP products.
- `FLAG_DETECTOR_TAG`: The Stim instruction tag marking a post-selection flag detector.
- `flag_detector_indices`: Function to list flag-detector indices in a Stim circuit.
- `flag_postselection_mask`: Function to build a sinter post-selection mask from flag detectors.
"""

from graphqomb.stim_glue._parse import StimMppExtraction
from graphqomb.stim_glue.compiler import stim_compile
from graphqomb.stim_glue.importer import (
    StimImportResult,
    stim_circuit_to_pattern,
    stim_file_to_pattern,
    stim_text_to_pattern,
)
from graphqomb.stim_glue.mpp import stabilizer_code_from_stim_file, stabilizer_code_from_stim_text
from graphqomb.stim_glue.mpp_rewriter import (
    CheckMapping,
    MppRewriteResult,
    UnsupportedSyndromeCircuitError,
    rewrite_to_mpp,
)
from graphqomb.stim_glue.postselect import FLAG_DETECTOR_TAG, flag_detector_indices, flag_postselection_mask
from graphqomb.stim_glue.transpiler import (
    STIM_GATE_J_ANGLES,
    UnsupportedInstructionError,
    optimize_j_cz,
    transpile,
)

__all__ = [
    "FLAG_DETECTOR_TAG",
    "STIM_GATE_J_ANGLES",
    "CheckMapping",
    "MppRewriteResult",
    "StimImportResult",
    "StimMppExtraction",
    "UnsupportedInstructionError",
    "UnsupportedSyndromeCircuitError",
    "flag_detector_indices",
    "flag_postselection_mask",
    "optimize_j_cz",
    "rewrite_to_mpp",
    "stabilizer_code_from_stim_file",
    "stabilizer_code_from_stim_text",
    "stim_circuit_to_pattern",
    "stim_compile",
    "stim_file_to_pattern",
    "stim_text_to_pattern",
    "transpile",
]
