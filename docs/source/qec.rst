QEC graph-state builder
=======================

``graphqomb.qeccode`` provides the stabilizer-code representation and the
Type I and Type II graph-state builders used by the QEC helper workflow.

Coordinate tuples are retained without restricting their dimensionality. The
graph-state builder uses the first two qubit-coordinate components for the data
plane and the first three explicitly supplied stabilizer-coordinate components
for ancilla placement. An ancilla without an explicit coordinate is placed at
the centroid of its connected data nodes. If that centroid's XY projection is
already occupied, the ancilla is moved in the data plane to a nearby candidate
position with clearance from existing nodes. The clearance scales with the XY
coordinate span so that it remains visible for large coordinate ranges; the
temporal coordinate is left unchanged.

With ``data_as_io=True``, the stabilizer-measurement unit has a separate,
unmeasured output layer. Type I therefore has two measured data layers followed
by an output layer. Type II uses three Y-measured layers for qubits with Y
support, followed by an output layer; qubits without Y support retain the Type I
two-measurement-layer layout. Ancilla support edges only touch measurement
layers, and the output of one composed unit becomes the input of the next.

Both foliation variants use the local stabilizer-interaction order
``Z -> Y -> X`` on each shared data qubit. For a pair of stabilizers, the
builder adds a CZ edge between their ancillas when an odd number of shared
data-qubit pairs reverse that order: one qubit applies stabilizer ``a`` before
``b`` while the other applies ``b`` before ``a``. Equal-Pauli overlaps have no
strict order and do not contribute. The builder tracks only the parity of the
two directions and considers only stabilizer pairs that share a data qubit, so
sparse codes do not require an all-pairs stabilizer scan.

.. automodule:: graphqomb.qeccode
   :members:
   :show-inheritance:
