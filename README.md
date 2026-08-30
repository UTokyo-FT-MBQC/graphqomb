# GraphQOMB

![License](https://img.shields.io/github/license/UTokyo-FT-MBQC/graphqomb)
[![PyPI version](https://badge.fury.io/py/graphqomb.svg)](https://pypi.org/project/graphqomb/)
[![Python Versions](https://img.shields.io/pypi/pyversions/graphqomb.svg)](https://pypi.org/project/graphqomb/)
[![Documentation Status](https://readthedocs.org/projects/graphqomb/badge/?version=latest)](https://graphqomb.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/UTokyo-FT-MBQC/graphqomb/branch/main/graph/badge.svg)](https://codecov.io/gh/UTokyo-FT-MBQC/graphqomb)
[![pytest](https://github.com/UTokyo-FT-MBQC/graphqomb/actions/workflows/pytest.yml/badge.svg)](https://github.com/UTokyo-FT-MBQC/graphqomb/actions/workflows/pytest.yml)
[![typecheck](https://github.com/UTokyo-FT-MBQC/graphqomb/actions/workflows/typecheck.yml/badge.svg)](https://github.com/UTokyo-FT-MBQC/graphqomb/actions/workflows/typecheck.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**GraphQOMB** (Qompiler for Measurement-Based Quantum Computing, pronounced _graphcomb_) is a compiler framework for measurement-based quantum computation (MBQC). It keeps the resource-state structure, classical feedforward, and execution schedule as separate first-class objects, then lowers them to an executable measurement pattern with a Pauli frame.

This design makes GraphQOMB useful both as an executable MBQC compiler and as a foundation for fault-tolerant workflows. The same core pipeline can be used to build patterns, study schedule-dependent resource tradeoffs, simulate them with statevector or density-matrix backends, and export compatible patterns to Stim-oriented downstream tooling.

## Core Workflow

GraphQOMB is organized around three explicit compiler interfaces:

- **Labelled graph state**: the resource state, measurement bases, and I/O registration.
- **Feedforward maps**: explicit `xflow` and optional `zflow`/`cflow` describing classical dependencies.
- **Scheduler**: preparation, entanglement, and measurement order for executable slices.

These are lowered with `qompile(...)` into a `Pattern` carrying:

- a command stream for scheduled MBQC execution,
- a `PauliFrame` for classical dependency tracking,
- metrics such as `max_space`, `depth`, and `active_volume`.

## Features

- **Explicit IR boundaries**: work directly with graph-state, feedforward, and schedule objects instead of mixing them into a single representation.
- **Pattern lowering**: compile MBQC IRs into executable patterns with `TICK`-delimited slices and Pauli-frame tracking.
- **Schedule analysis**: compare depth-oriented and space-oriented schedules and inspect resulting resource metrics.
- **Simulation**: run circuit or pattern simulations with statevector and density-matrix backends.
- **Stim export**: compile compatible patterns to Stim text for downstream FT-oriented analysis.
- **Toolchain interoperability**: use GraphQOMB downstream of circuit transpilation and graph-rewrite tooling such as PyZX and flow-finding utilities.

## Installation

### From PyPI

```bash
uv add graphqomb
```

Install optional PyZX integration:

```bash
uv add "graphqomb[pyzx]"
```

Install optional Stim integration:

```bash
uv add "graphqomb[stim]"
```

### From Source

```bash
git clone https://github.com/UTokyo-FT-MBQC/graphqomb.git
cd graphqomb
uv sync
```

Install development dependencies:

```bash
uv sync --extra dev
```

Install documentation dependencies:

```bash
uv sync --extra doc
```

## Quick Start

The quickest way to see the compiler pipeline is to start from an MBQC-native circuit, derive the graph/feedforward/schedule objects, and lower them into a pattern:

```python
import numpy as np

from graphqomb.circuit import MBQCCircuit, circuit2graph
from graphqomb.qompiler import qompile
from graphqomb.simulator import PatternSimulator, SimulatorBackend

circuit = MBQCCircuit(3)
circuit.j(0, 0.5 * np.pi)
circuit.cz(0, 1)
circuit.cz(0, 2)
circuit.j(1, 0.75 * np.pi)
circuit.j(2, 0.25 * np.pi)

graphstate, xflow, scheduler = circuit2graph(circuit)
pattern = qompile(graphstate, xflow, scheduler=scheduler)

print("pattern depth:", pattern.depth)
print("pattern max space:", pattern.max_space)

simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
simulator.simulate()
print(simulator.state.state())
```

If you already have a graph-state design and explicit feedforward maps, you can skip `circuit2graph(...)` and call `qompile(...)` directly.

## Documentation

- **Getting started**: https://graphqomb.readthedocs.io/en/latest/getting_started.html
- **Architecture overview**: https://graphqomb.readthedocs.io/en/latest/architecture.html
- **Example gallery**: https://graphqomb.readthedocs.io/en/latest/gallery/index.html
- **API reference**: https://graphqomb.readthedocs.io/en/latest/references.html
- **QEC graph-state builder reference**: https://graphqomb.readthedocs.io/en/latest/qeccode.html
- **Stim MPP import reference**: https://graphqomb.readthedocs.io/en/latest/stim_glue/mpp.html
- **Stim MPP rewriter reference (experimental)**: https://graphqomb.readthedocs.io/en/latest/stim_glue/mpp_rewriter.html
- **Stim Clifford transpiler reference**: https://graphqomb.readthedocs.io/en/latest/stim_glue/transpiler.html
- **Stim circuit import reference**: https://graphqomb.readthedocs.io/en/latest/stim_glue/importer.html
- **Stim compiler reference**: https://graphqomb.readthedocs.io/en/latest/stim_glue/compiler.html

## Current Scope

GraphQOMB currently targets static, branch-free MBQC workflows. It is designed around causal feedforward dependencies and explicit scheduling, which makes it a good fit for pattern generation, simulation, and offline analysis of executable or fault-tolerant MBQC pipelines.

Patterns support classically-controlled single-qubit Clifford feedforward: an optional `cflow` map assigns each measured node a conditional Clifford correction on its targets, tracked at runtime by the pattern's Clifford frame (`CliffordFrame`, of which `PauliFrame` is now an alias). Stim export and detector/observable certification remain Pauli-frame-only for now.

## Development

### Running Tests

```bash
uv run pytest
uv run pytest tests/test_specific.py
```

### Code Quality

```bash
uv run ruff check
uv run ruff format
uv run mypy
uv run pyright
```

### Building Documentation

```bash
cd docs
uv run sphinx-build -W source build
```

## Contributing

Contributions are welcome. Please open an issue or pull request with:

1. A clear description of the change.
2. Tests for behavioral changes when applicable.
3. Documentation updates for user-facing features.

## Related Projects

- [graphix](https://github.com/TeamGraphix/graphix): MBQC software stack with a different abstraction strategy.
- [PyZX](https://github.com/Quantomatic/pyzx): ZX-calculus tooling that can be used upstream of GraphQOMB.
- [swiflow](https://github.com/TeamGraphix/swiflow): Flow-finding utilities for MBQC dependency structures.
- [Stim](https://github.com/quantumlib/Stim): Fast stabilizer-circuit simulator targeted by the Stim export path.

Our ongoing projects

- [ls-pattern-compile](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile): Lattice-surgery compiler for MBQC backends.
- [graphqomb-studio](https://github.com/UTokyo-FT-MBQC/graphqomb-studio): 2D&3D GUI editor and visualizer for `GraphQOMB` package.

## License

[MIT License](LICENSE)

## Citation

If you use GraphQOMB in your research, please cite:

```bibtex
@software{graphqomb,
  title = {GraphQOMB: A Modular Graph State Qompiler for Measurement-Based Quantum Computation},
  author = {Masato Fukushima, Sora Shiratani, Yuki Watanabe, and Daichi Sasaki},
  year = {2025},
  url = {https://github.com/UTokyo-FT-MBQC/graphqomb}
}
```

## Acknowledgements

We acknowledge the [NICT Quantum Camp](https://nqc.nict.go.jp/) for supporting our development.

Special thanks to Fixstars Amplify:

<p><a href="https://amplify.fixstars.com/en/">
<img src="https://github.com/TeamGraphix/graphix/raw/master/docs/imgs/fam_logo.png" alt="amplify" width="200"/>
</a></p>
