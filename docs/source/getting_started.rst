Getting Started
===============

This tutorial walks through GraphQOMB's main compiler path:

1. describe a computation with a circuit,
2. derive compiler IR objects,
3. lower them to an executable pattern,
4. inspect schedule-dependent metrics,
5. simulate the result.

If you prefer a runnable script first, start with :doc:`gallery/pattern_from_circuit`.

Compiler pipeline
-----------------

GraphQOMB separates the main MBQC compilation steps into explicit objects:

- a labelled graph state,
- feedforward maps (`xflow` and optional `zflow`),
- a scheduler describing executable time slices.

These are lowered with :func:`graphqomb.qompiler.qompile` into a :class:`graphqomb.pattern.Pattern`.

Build an MBQC-native circuit
----------------------------

Use :class:`graphqomb.circuit.MBQCCircuit` when you want an MBQC-native circuit description:

.. code-block:: python

   import numpy as np

   from graphqomb.circuit import MBQCCircuit

   circuit = MBQCCircuit(3)
   circuit.j(0, 0.5 * np.pi)
   circuit.cz(0, 1)
   circuit.cz(0, 2)
   circuit.j(1, 0.75 * np.pi)
   circuit.j(2, 0.25 * np.pi)

For circuit descriptions that use macro gates such as `H`, `CNOT`, or `Rz`, see :doc:`gallery/circuit_simulation_example`.

Derive compiler IRs
-------------------

Use :func:`graphqomb.circuit.circuit2graph` to derive the graph-state IR, feedforward map, and a schedule seed from the circuit:

.. code-block:: python

   from graphqomb.circuit import circuit2graph

   graphstate, xflow, scheduler = circuit2graph(circuit)

   print("graph nodes:", len(graphstate.nodes))
   print("graph edges:", len(graphstate.edges))
   print("feedforward entries:", len(xflow))
   print("scheduled slices:", scheduler.num_slices())

At this stage:

- ``graphstate`` carries the resource graph, measurement bases, and I/O registration,
- ``xflow`` stores explicit classical dependencies,
- ``scheduler`` stores prepare/entangle/measure slice timing.

Lower to an executable pattern
------------------------------

Lower these IR objects with :func:`graphqomb.qompiler.qompile`:

.. code-block:: python

   from graphqomb.qompiler import qompile

   pattern = qompile(graphstate, xflow, scheduler=scheduler)

   print("pattern depth:", pattern.depth)
   print("pattern max space:", pattern.max_space)
   print("pattern active volume:", pattern.active_volume)

Passing ``scheduler`` is optional. If you omit it, :func:`graphqomb.qompiler.qompile`
constructs a :class:`graphqomb.scheduler.Scheduler` internally, solves it with the
default ``MINIMIZE_TIME`` strategy, and still emits `TICK`-delimited time slices.

By default, :func:`graphqomb.qompiler.qompile` derives ``zflow`` from odd neighborhoods when you do not pass it explicitly. This is convenient for standard deterministic workflows, while still allowing explicit ``zflow`` control when you need a custom feedforward design.

The resulting :class:`graphqomb.pattern.Pattern` contains:

- scheduled commands,
- a :class:`graphqomb.pauli_frame.PauliFrame`,
- resource metrics such as `depth`, `max_space`, and `active_volume`.

Inspect the result
------------------

Patterns can be inspected directly:

.. code-block:: python

   from graphqomb.pattern import print_pattern

   print_pattern(pattern, lim=20)
   print("space profile:", pattern.space)

The emitted command stream is sliced with `TICK` markers according to the active
schedule. This happens both when you provide ``scheduler`` explicitly and when
:func:`graphqomb.qompiler.qompile` creates one internally with its default
strategy. Those slice boundaries are the executable schedule seen by pattern
simulators and downstream exporters.

Simulate the pattern
--------------------

Use :class:`graphqomb.simulator.PatternSimulator` for direct execution:

.. code-block:: python

   from graphqomb.simulator import PatternSimulator, SimulatorBackend

   simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
   simulator.simulate()
   state = simulator.state.state()

For deeper validation against the source circuit, see :doc:`gallery/pattern_from_circuit`, which compares circuit and pattern simulation results.

Where to go next
----------------

- :doc:`architecture` explains the three-IR compiler model and the lowering semantics.
- :doc:`gallery/scheduler_pattern_demo` compares space- and time-oriented scheduling strategies.
- :doc:`gallery/entanglement_scheduling_demo` focuses on explicit entanglement timing and `TICK` slices.
- :doc:`stim_glue/compiler` documents the Stim export path for compatible patterns.
