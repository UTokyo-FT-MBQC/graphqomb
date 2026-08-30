"""Clifford frame for Measurement-based Quantum Computing.

This module provides:

- `CliffordFrame`: A class to track the correction frame of a quantum computation.
- `PauliFrame`: Backwards-compatible alias of `CliffordFrame`.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import TYPE_CHECKING

from graphqomb import clifford_algebra
from graphqomb.common import Axis, determine_pauli_axis

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence
    from collections.abc import Set as AbstractSet

    from graphqomb.clifford_algebra import C1Element
    from graphqomb.graphstate import BaseGraphState


class CliffordFrame:
    r"""Clifford frame tracker.

    Each node's runtime frame has the normal form ``D * X^a * Z^b`` with the
    coset ``D`` drawn from `graphqomb.clifford_algebra.TRANSVERSAL`.  The
    supplied ``cflow`` corrections are normalized at construction: the Pauli
    part is folded into ``xflow``/``zflow`` and only nontrivial cosets are
    stored.

    Attributes
    ----------
    graphstate : `BaseGraphState`
        Set of nodes in the resource graph
    xflow : `dict`\[`int`, `set`\[`int`\]
        X correction flow for each measurement flip
    zflow : `dict`\[`int`, `set`\[`int`\]
        Z correction flow for each  measurement flip
    cflow : `dict`\[`int`, `dict`\[`int`, `C1Element`\]\]
        Clifford correction flow (normalized, nontrivial cosets only)
    x_pauli : `dict`\[`int`, `bool`\]
        Current X Pauli state for each node
    z_pauli : `dict`\[`int`, `bool`\]
        Current Z Pauli state for each node
    coset : `dict`\[`int`, `C1Element`\]
        Current coset accumulator for each node (identity by default)
    parity_check_group : `list`\[`set`\[`int`\]\]
        Parity check group for FTQC
    parity_check_tags : `list`\[`str`\]
        Stim-style tag of each parity check group, aligned with
        `parity_check_group`. The empty string means untagged. The tag
        ``type=flag`` marks a flag detector whose samples are meant to be
        post-selected (rejected when the detector fires).
    inv_xflow : `dict`\[`int`, `set`\[`int`\]\]
        Inverse X correction flow for each measurement flip
    inv_zflow : `dict`\[`int`, `set`\[`int`\]\]
        Inverse Z correction flow for each measurement flip
    inv_cflow : `dict`\[`int`, `dict`\[`int`, `C1Element`\]\]
        Inverse Clifford correction flow (target -> source -> coset)
    """

    graphstate: BaseGraphState
    xflow: dict[int, set[int]]
    zflow: dict[int, set[int]]
    cflow: dict[int, dict[int, C1Element]]
    x_pauli: dict[int, bool]
    z_pauli: dict[int, bool]
    coset: dict[int, C1Element]
    parity_check_group: list[set[int]]
    parity_check_tags: list[str]
    logical_observables: dict[int, set[int]]
    inv_xflow: dict[int, set[int]]
    inv_zflow: dict[int, set[int]]
    inv_cflow: dict[int, dict[int, C1Element]]
    _pauli_axis_cache: dict[int, Axis | None]
    _chain_cache: dict[int, frozenset[int]]

    def __init__(  # ruff:ignore[too-many-arguments]
        self,
        graphstate: BaseGraphState,
        xflow: Mapping[int, AbstractSet[int]],
        zflow: Mapping[int, AbstractSet[int]],
        parity_check_group: Sequence[AbstractSet[int]] | None = None,
        logical_observables: Mapping[int, AbstractSet[int]] | None = None,
        *,
        parity_check_tags: Sequence[str] | None = None,
        cflow: Mapping[int, Mapping[int, C1Element]] | None = None,
    ) -> None:
        if parity_check_group is None:
            parity_check_group = []
        if logical_observables is None:
            logical_observables = {}
        self.graphstate = graphstate
        self.xflow = {node: set(targets) for node, targets in xflow.items()}
        self.zflow = {node: set(targets) for node, targets in zflow.items()}
        self._normalize_cflow(cflow)
        self.x_pauli = dict.fromkeys(graphstate.nodes, False)
        self.z_pauli = dict.fromkeys(graphstate.nodes, False)
        self.coset = dict.fromkeys(graphstate.nodes, clifford_algebra.IDENTITY)
        self.parity_check_group = [set(item) for item in parity_check_group]
        if parity_check_tags is None:
            parity_check_tags = [""] * len(self.parity_check_group)
        elif len(parity_check_tags) != len(self.parity_check_group):
            msg = (
                f"parity_check_tags has {len(parity_check_tags)} tag(s) "
                f"for {len(self.parity_check_group)} parity check group(s)."
            )
            raise ValueError(msg)
        self.parity_check_tags = list(parity_check_tags)
        self.logical_observables = {
            logical_idx: set(seed_nodes) for logical_idx, seed_nodes in logical_observables.items()
        }

        self._build_inverse_flows()
        self._check_schedule_independence()

        # Pre-compute Pauli axes for performance optimization.
        # NOTE: if non-Pauli measurements are involved, the stim_compile func will error out earlier
        self._pauli_axis_cache = (
            {node: determine_pauli_axis(meas_basis) for node, meas_basis in graphstate.meas_bases.items()}
            if parity_check_group or logical_observables
            else {}
        )
        # Cache for memoization of dependent chains
        self._chain_cache = {}

    def _normalize_cflow(self, cflow: Mapping[int, Mapping[int, C1Element]] | None) -> None:
        """Fold the Pauli part of each cflow correction into xflow/zflow, keeping nontrivial cosets."""
        self.cflow = {}
        if not cflow:
            return
        for source, targets in cflow.items():
            for target, element in targets.items():
                coset, x_bit, z_bit = clifford_algebra.decompose(element)
                if x_bit:
                    self.xflow.setdefault(source, set()).symmetric_difference_update({target})
                if z_bit:
                    self.zflow.setdefault(source, set()).symmetric_difference_update({target})
                if coset != clifford_algebra.IDENTITY:
                    self.cflow.setdefault(source, {})[target] = coset

    def _build_inverse_flows(self) -> None:
        """Build the inverse correction flows, dropping self-corrections."""
        self.inv_xflow = defaultdict(set)
        self.inv_zflow = defaultdict(set)
        for node, targets in self.xflow.items():
            for target in targets:
                self.inv_xflow[target].add(node)
            self.inv_xflow[node] -= {node}
        for node, targets in self.zflow.items():
            for target in targets:
                self.inv_zflow[target].add(node)
            self.inv_zflow[node] -= {node}
        self.inv_cflow = {}
        for node, coset_targets in self.cflow.items():
            for target, coset in coset_targets.items():
                if target != node:
                    self.inv_cflow.setdefault(target, {})[node] = coset

    def _correction_element(self, source: int, target: int) -> C1Element:
        """Return the total correction ``D * X^x * Z^z`` sent from source to target.

        Returns
        -------
        `C1Element`
            The total correction element modulo phase.
        """
        element = self.cflow.get(source, {}).get(target, clifford_algebra.IDENTITY)
        if target in self.xflow.get(source, set()):
            element = clifford_algebra.compose(element, clifford_algebra.X)
        if target in self.zflow.get(source, set()):
            element = clifford_algebra.compose(element, clifford_algebra.Z)
        return element

    def _descendants(self, node: int, cache: dict[int, frozenset[int]]) -> frozenset[int]:
        r"""Return all nodes reachable from ``node`` through the correction flows.

        Returns
        -------
        `frozenset`\[`int`\]
            The reachable nodes, excluding ``node`` itself unless it lies on a cycle.
        """
        if node in cache:
            return cache[node]
        seen: set[int] = set()
        stack = [node]
        while stack:
            for child in self.children(stack.pop()):
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        result = frozenset(seen)
        cache[node] = result
        return result

    def _check_schedule_independence(self) -> None:
        """Reject non-commuting corrections from sources not ordered by the dependency DAG.

        Raises
        ------
        ValueError
            If a target receives corrections that do not commute mod phase
            from two sources that are incomparable in the dependency DAG, so
            the accumulated frame would depend on the measurement schedule.
        """
        if not self.cflow:
            return
        descendants_cache: dict[int, frozenset[int]] = {}
        for target, coset_sources in self.inv_cflow.items():
            sources = set(coset_sources) | self.inv_xflow.get(target, set()) | self.inv_zflow.get(target, set())
            corrections = {source: self._correction_element(source, target) for source in sources}
            for source1, source2 in itertools.combinations(sorted(sources), 2):
                if source1 not in coset_sources and source2 not in coset_sources:
                    continue  # Pauli-Pauli pairs always commute mod phase
                element1, element2 = corrections[source1], corrections[source2]
                if clifford_algebra.compose(element1, element2) == clifford_algebra.compose(element2, element1):
                    continue
                if source2 in self._descendants(source1, descendants_cache) or source1 in self._descendants(
                    source2, descendants_cache
                ):
                    continue
                msg = (
                    f"Corrections onto node {target} from sources {source1} and {source2} do not commute "
                    "(mod phase) and the sources are not ordered by the dependency DAG, so the accumulated "
                    "frame would depend on the measurement schedule. Order the sources or use commuting "
                    "corrections."
                )
                raise ValueError(msg)

    def x_flip(self, node: int) -> None:
        """Flip the X Pauli mask for the given node.

        Parameters
        ----------
        node : `int`
            The node to flip.
        """
        self.x_pauli[node] = not self.x_pauli[node]

    def z_flip(self, node: int) -> None:
        """Flip the Z Pauli mask for the given node.

        Parameters
        ----------
        node : `int`
            The node to flip.
        """
        self.z_pauli[node] = not self.z_pauli[node]

    def meas_flip(self, node: int) -> None:
        """Update the frame for a measurement flip based on the given correction flows.

        Each frame stays in the normal form ``D * X^a * Z^b``: Pauli corrections
        are conjugated through the target's coset before toggling the bits, and
        coset corrections are left-multiplied and re-normalized.

        Parameters
        ----------
        node : `int`
            The node to flip.
        """
        if not self.cflow:
            for target in self.xflow.get(node, set()):
                self.x_pauli[target] = not self.x_pauli[target]
            for target in self.zflow.get(node, set()):
                self.z_pauli[target] = not self.z_pauli[target]
            return
        for target in self.xflow.get(node, set()):
            self._absorb_pauli(target, Axis.X)
        for target in self.zflow.get(node, set()):
            self._absorb_pauli(target, Axis.Z)
        for target, coset in self.cflow.get(node, {}).items():
            new_coset, x_bit, z_bit = clifford_algebra.decompose(clifford_algebra.compose(coset, self.coset[target]))
            self.coset[target] = new_coset
            if x_bit:
                self.x_pauli[target] = not self.x_pauli[target]
            if z_bit:
                self.z_pauli[target] = not self.z_pauli[target]

    def _absorb_pauli(self, target: int, axis: Axis) -> None:
        """Absorb a Pauli correction into the target frame, conjugating it through the coset."""
        coset = self.coset[target]
        if coset != clifford_algebra.IDENTITY:
            axis, _sign = clifford_algebra.act_on_axis(clifford_algebra.inverse(coset), axis)
        if axis is not Axis.Z:
            self.x_pauli[target] = not self.x_pauli[target]
        if axis is not Axis.X:
            self.z_pauli[target] = not self.z_pauli[target]

    def children(self, node: int) -> set[int]:
        r"""Get the children of a node in the correction frame.

        Parameters
        ----------
        node : `int`
            The node to get children for.

        Returns
        -------
        `set`\[`int`\]
            The set of child nodes.
        """
        return (self.xflow.get(node, set()) | self.zflow.get(node, set()) | self.cflow.get(node, {}).keys()) - {node}

    def parents(self, node: int) -> set[int]:
        r"""Get the parents of a node in the correction frame.

        Parameters
        ----------
        node : `int`
            The node to get parents for.

        Returns
        -------
        `set`\[`int`\]
            The set of parent nodes.
        """
        return self.inv_xflow.get(node, set()) | self.inv_zflow.get(node, set()) | self.inv_cflow.get(node, {}).keys()

    def detector_groups(self) -> list[set[int]]:
        r"""Get the parity check groups.

        Returns
        -------
        `list`\[`set`\[`int`\]\]
            The parity check groups.
        """
        groups: list[set[int]] = []

        for syndrome_group in self.parity_check_group:
            mbqc_group: set[int] = set()
            for node in syndrome_group:
                mbqc_group ^= self._collect_dependent_chain(node)
            groups.append(mbqc_group)

        return groups

    def detector_stabilizers(self) -> list[dict[int, Axis]]:
        r"""Get the graph-state stabilizer associated with each detector.

        The detector groups are expanded through their dependent chains before
        constructing the stabilizers.  Every node that is not Z-measured
        contributes the resource-state stabilizer obtained from its
        initialization axis (X on itself and Z on each neighbor for the
        default \|+> preparation).  A Z-measured node contributes the
        single-qubit Z stabilizer of its own preparation when it is
        Z-initialized, and nothing otherwise: its Z support has to be supplied
        by the graph stabilizers of its neighbors.  Z-initialized nodes are
        excluded from all neighbor sets because their single-qubit Z
        stabilizers make any Z support on them freely adjustable.

        Global phases are discarded when multiplying the Pauli operators.

        Returns
        -------
        `list`\[`dict`\[`int`, `Axis`\]\]
            Detector stabilizers, represented by their non-identity Pauli axes.
        """
        z_measurements = self._z_measurements()
        return [self._detector_stabilizer(group, z_measurements=z_measurements) for group in self.detector_groups()]

    def detector_determinism(self) -> list[bool]:
        r"""Determine whether each detector has deterministic measurement parity.

        A detector is deterministic when its stabilizer is exactly equal to the
        product of the Pauli measurement axes on its detector group.  Measurement
        signs and output nodes without an assigned measurement basis are omitted
        from the comparison.

        Returns
        -------
        `list`\[`bool`\]
            Determinism flags in the same order as `detector_groups`.
        """
        groups = self.detector_groups()
        z_measurements = self._z_measurements()
        stabilizers = [self._detector_stabilizer(group, z_measurements=z_measurements) for group in groups]
        unmeasured_outputs = self.graphstate.output_node_indices.keys() - self.graphstate.meas_bases.keys()
        results: list[bool] = []

        for group, stabilizer in zip(groups, stabilizers, strict=True):
            compared_stabilizer = {node: axis for node, axis in stabilizer.items() if node not in unmeasured_outputs}
            measurement_product = self._detector_measurement_product(group)
            results.append(measurement_product is not None and compared_stabilizer == measurement_product)

        return results

    def _detector_measurement_product(self, detector_group: AbstractSet[int]) -> dict[int, Axis] | None:
        r"""Construct the unsigned Pauli measurement product on a detector group.

        Parameters
        ----------
        detector_group : `collections.abc.Set`\[`int`\]
            Closure-expanded detector group.

        Returns
        -------
        `dict`\[`int`, `Axis`\] | `None`
            Pauli measurement axes, or None if a compared node does not have a
            Pauli measurement basis.
        """
        measurement_product: dict[int, Axis] = {}
        output_nodes = self.graphstate.output_node_indices
        meas_bases = self.graphstate.meas_bases

        for node in detector_group:
            meas_basis = meas_bases.get(node)
            if meas_basis is None and node in output_nodes:
                continue
            if meas_basis is None or (axis := determine_pauli_axis(meas_basis)) is None:
                return None
            measurement_product[node] = axis

        return measurement_product

    def _z_measurements(self) -> set[int]:
        r"""Return all nodes assigned a Z measurement.

        Returns
        -------
        `set`\[`int`\]
            Nodes with a Z measurement basis.
        """
        return {
            node
            for node, meas_basis in self.graphstate.meas_bases.items()
            if determine_pauli_axis(meas_basis) is Axis.Z
        }

    def _detector_stabilizer(
        self,
        detector_group: AbstractSet[int],
        *,
        z_measurements: AbstractSet[int],
    ) -> dict[int, Axis]:
        r"""Construct the product stabilizer for an expanded detector group.

        Parameters
        ----------
        detector_group : `collections.abc.Set`\[`int`\]
            Closure-expanded detector group.
        z_measurements : `collections.abc.Set`\[`int`\]
            Nodes with a Z measurement basis.

        Returns
        -------
        `dict`\[`int`, `Axis`\]
            Non-identity Pauli axes in the detector stabilizer.
        """
        x_support: set[int] = set()
        z_support: set[int] = set()
        input_axes = {node: init.axis for node, init in self.graphstate.input_initializations.items()}
        z_initialized = {node for node, axis in input_axes.items() if axis is Axis.Z}

        for node in detector_group:
            axis = input_axes.get(node, Axis.X)

            if node in z_measurements:
                # Only a Z-initialized node owns a single-qubit Z stabilizer that
                # certifies its Z measurement.  Any other Z-measured node must have
                # its Z support supplied by the graph stabilizers of its neighbors.
                if axis is Axis.Z:
                    z_support.symmetric_difference_update({node})
                continue

            if axis in {Axis.X, Axis.Y}:
                x_support.symmetric_difference_update({node})
                z_support.symmetric_difference_update(self.graphstate.neighbors(node) - z_initialized)
            if axis in {Axis.Y, Axis.Z}:
                z_support.symmetric_difference_update({node})

        stabilizer: dict[int, Axis] = {}
        for node in x_support | z_support:
            has_x = node in x_support
            has_z = node in z_support
            stabilizer[node] = Axis.Y if has_x and has_z else Axis.X if has_x else Axis.Z
        return stabilizer

    def logical_observable_groups(self) -> dict[int, set[int]]:
        r"""Get all logical observable groups after dependent-chain expansion.

        Returns
        -------
        `dict`\[`int`, `set`\[`int`\]\]
            The expanded logical observable groups keyed by logical index.
        """
        return {
            logical_idx: self.logical_observables_group(target_nodes)
            for logical_idx, target_nodes in self.logical_observables.items()
        }

    def logical_observables_group(self, target_nodes: Collection[int]) -> set[int]:
        r"""Get the logical observables group for the given target nodes.

        Parameters
        ----------
        target_nodes : `collections.abc.Collection`\[`int`\]
            The target nodes to get the logical observables group for.

        Returns
        -------
        `set`\[`int`\]
            The logical observables group for the given target nodes.
        """
        group: set[int] = set()
        for node in target_nodes:
            group ^= self._collect_dependent_chain(node=node)

        return group

    def _collect_dependent_chain(self, node: int) -> set[int]:
        r"""Generalized dependent-chain collector that respects measurement planes.

        Uses recursive memoization to correctly XOR nodes reached via multiple paths.

        Parameters
        ----------
        node : `int`
            The starting node.

        Returns
        -------
        `set`\[`int`\]
            The set of dependent nodes in the chain.

        Raises
        ------
        ValueError
            If an unexpected output basis or measurement plane is encountered.
        NotImplementedError
            If the node is subject to Clifford feedforward.
        """
        if self.cflow and (
            self.inv_cflow.get(node) or self.coset.get(node, clifford_algebra.IDENTITY) != clifford_algebra.IDENTITY
        ):
            msg = (
                f"Node {node} is subject to Clifford feedforward (cflow); symbolic detector and observable "
                "certification over a Clifford frame is future work."
            )
            raise NotImplementedError(msg)
        # Check memoization cache
        if node in self._chain_cache:
            return set(self._chain_cache[node])

        chain: set[int] = {node}

        # Use pre-computed Pauli axis from cache
        axis = self._pauli_axis_cache[node]

        # NOTE: might have to support plane instead of axis
        if axis == Axis.X:
            parents = self.inv_zflow[node]
        elif axis == Axis.Y:
            parents = self.inv_xflow[node].symmetric_difference(self.inv_zflow[node])
        elif axis == Axis.Z:
            parents = self.inv_xflow[node]
        else:
            msg = f"Unexpected measurement axis: {axis}"
            raise ValueError(msg)

        # Recursively collect and XOR parent chains
        for parent in parents:
            parent_chain = self._collect_dependent_chain(parent)
            chain ^= parent_chain

        # Store result in cache for future calls
        self._chain_cache[node] = frozenset(chain)

        return chain


#: Backwards-compatible alias of `CliffordFrame`.
PauliFrame = CliffordFrame
