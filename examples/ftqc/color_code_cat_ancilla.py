r"""Color-Code Memory with Cat-State Ancillas
=========================================

Run the example from the repository root:

.. code-block:: sh

    uv run --extra stim python -m examples.ftqc.color_code_cat_ancilla

The Stim importer measures an ``MPP`` product through one ancilla node joined
to every data wire of the product. An X fault on that node between two of its
CZ gates acts as Z on every data wire coupled later. Such a hook error turns
one fault into an error on up to half of a plaquette, and the fault distance
of the compiled circuit falls below the code distance.

``mpp_ancilla="cat"`` measures each product with at least three legs through
a cat state, with one ancilla node per leg. The tag ``MPP[ancilla=cat]``
selects the same for the products of one instruction. This example writes a memory
experiment of the triangular color code with plaquette ``MPP`` products,
compiles it with both ancilla kinds, and computes the fault distance of the
exported noisy Stim circuits for distances 3 and 5.

Cat ring
--------

For a product with legs :math:`d_1, \dots, d_w` the importer builds a ring of
cat nodes :math:`a_i` and connectors :math:`c_i`, all measured in X. For
:math:`w = 4`:

.. code-block:: text

    d1        d2        d3        d4
    |         |         |         |
    a1 - c1 - a2 - c2 - a3 - c3 - a4
    |                             |
    +------------- c4 ------------+

Measuring the connectors leaves the cat nodes in a cat state. The parity of
the cat outcomes is the outcome of the product. The parity of the connector
outcomes is fixed, and the importer adds it as a parity check tagged
``CAT_FLAG_TAG``. One fault on the ring flips this flag or reaches at most
one data wire, in any CZ order. The legs follow the order of the ``MPP``
targets, so the cat nodes of neighboring targets are neighbors on the ring.

CZ order
--------

A fault spreads only through the CZ gates applied after it, so a fault
distance belongs to a CZ order and not to the graph alone. The ``E`` commands
of a pattern fix that order. The example compiles each experiment in four
orders:

- ``scheduler``: the default schedule, which leaves the edges of one time
  slice in no particular order.
- ``targets``: the default schedule with ``sequence_mpp_legs=True``, which
  entangles the legs of each ancilla node one per time slice in the order of
  the ``MPP`` targets. The plaquettes list their qubits in cyclic order.
- ``sorted`` and ``random``: every node prepared first, then one CZ per time
  slice in sorted or shuffled edge order.

The single-ancilla values below belong to the orders tried here. Gate
schedules chosen per plaquette color avoid the harmful hook errors in the
bulk of the code, though not at its boundary; see
`Kishony and Fowler <https://arxiv.org/abs/2603.28852>`_. This example does
not tune the order.

Fault distance
--------------

The fault distance is the smallest number of error mechanisms of the detector
error model that flip the logical observable without a detection event. Flag
checks count as detectors. ``fault_distance_bounds`` examines every set of at
most four mechanisms, so it finds a distance up to four with certainty and
otherwise proves that the distance is at least five. Stim's
``search_for_undetectable_logical_errors`` then gives an upper bound. A
MaxSAT solver applied to ``stim.Circuit.shortest_error_sat_problem`` is an
independent check.

The noise is depolarization after each preparation and CZ, and a flip of each
measurement. Idle noise is not included.

Result
------

With single ancillas the fault distance is 2 and 3. With cat ancillas it is
3 and 5, the code distance, in every CZ order tried. Without the flag checks
the cat circuits are back at 2 and 3.

A flag check detects one fault per ring. Two faults on the same ring can
cancel in its flag and leave an error on several legs of that plaquette. At
distance 5 the search finds no logical error of four mechanisms, so one flag
check per ring is enough for this code. The example does not cover larger
distances or a decoder that uses the flags.
"""

# %%
from typing import TYPE_CHECKING, Literal

from graphqomb.stim_glue import CAT_FLAG_TAG, flag_detector_indices, stim_circuit_to_ir, stim_circuit_to_pattern

if TYPE_CHECKING or __package__:
    from ._support.color_code import (
        memory_circuit,
        noisy_circuit,
        pattern_with_cz_order,
        random_cz_order,
        triangular_color_code,
    )
    from ._support.fault_distance import drop_detectors, fault_distance_bounds
else:
    from _support.color_code import (  # ruff:ignore[import-private-name]
        memory_circuit,
        noisy_circuit,
        pattern_with_cz_order,
        random_cz_order,
        triangular_color_code,
    )
    from _support.fault_distance import drop_detectors, fault_distance_bounds  # ruff:ignore[import-private-name]

if TYPE_CHECKING:
    import stim

DISTANCES = (3, 5)
ROUNDS = 3
RANDOM_ORDERS = 1

# %%
# Every plaquette of the triangular color code is an X check and a Z check.
codes = {distance: triangular_color_code(distance) for distance in DISTANCES}
for distance, code in codes.items():
    weights = [len(plaquette) for plaquette in code.plaquettes]
    print(
        f"d={distance}: {len(code.coordinates)} qubits, {weights.count(4)} plaquettes of weight 4, "
        f"{weights.count(6)} of weight 6, logical weight {len(code.logical)}"
    )

# %%
# The experiment is written once. The cat import adds a ring and one flag check per product.
sources = {distance: memory_circuit(code, ROUNDS) for distance, code in codes.items()}
print(str(sources[3]).split("TICK")[1].strip().splitlines()[0])
for distance, source in sources.items():
    for kind in ("single", "cat"):
        ir = stim_circuit_to_ir(source, mpp_ancilla=kind)
        tags = ir.feedforward.parity_check_tags
        print(
            f"d={distance} {kind:6s}: {len(ir.graph.nodes):4d} nodes, {len(ir.graph.edges):4d} CZ gates, "
            f"{len(tags) - tags.count(CAT_FLAG_TAG)} code checks, {tags.count(CAT_FLAG_TAG)} flag checks"
        )


# %%
# Compile one experiment in several CZ orders.
def noisy_circuits(source: "stim.Circuit", kind: Literal["single", "cat"]) -> dict[str, "stim.Circuit"]:
    in_target_order = stim_circuit_to_pattern(source, mpp_ancilla=kind, sequence_mpp_legs=True)
    circuits = {
        "scheduler": noisy_circuit(stim_circuit_to_pattern(source, mpp_ancilla=kind).pattern),
        "targets": noisy_circuit(in_target_order.pattern),
    }
    ir = stim_circuit_to_ir(source, mpp_ancilla=kind)
    orders = {"sorted": sorted(ir.graph.edges)}
    orders.update({f"random {seed}": random_cz_order(ir, seed) for seed in range(RANDOM_ORDERS)})
    for name, order in orders.items():
        circuits[name] = noisy_circuit(pattern_with_cz_order(ir, order))
    return circuits


def describe(bounds: tuple[int, int]) -> str:
    lower, upper = bounds
    return str(lower) if lower == upper else f"{lower}..{upper}"


# %%
# Fault distance per ancilla kind and CZ order. The flags of a compiled circuit are selected by their tag.
for distance, source in sources.items():
    single = noisy_circuits(source, "single")
    cat = noisy_circuits(source, "cat")
    flags = flag_detector_indices(cat["sorted"], CAT_FLAG_TAG)
    print(f"d={distance}: {len(flags)} of {cat['sorted'].num_detectors} detectors of the cat circuit are flag checks")
    rows = {
        "single": single,
        "cat, flags ignored": {name: drop_detectors(circuit, CAT_FLAG_TAG) for name, circuit in cat.items()},
        "cat": cat,
    }
    print(f"  {'ancilla':20s}" + "".join(f"{name:>11s}" for name in cat))
    for label, circuits in rows.items():
        values = "".join(f"{describe(fault_distance_bounds(circuit)):>11s}" for circuit in circuits.values())
        print(f"  {label:20s}{values}")

# %%
