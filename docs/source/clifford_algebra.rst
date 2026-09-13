Clifford Algebra
================

:mod:`graphqomb.clifford_algebra` module
+++++++++++++++++++++++++++++++++++++++++

.. automodule:: graphqomb.clifford_algebra

Why the quotient is S3
----------------------

The subscript 3 counts the three objects being permuted: the unsigned X, Y,
and Z axes. The symmetric group on these axes has 3! = 6 elements.

The single-qubit Clifford group modulo global phase has 24 elements.
Conjugation permutes the unsigned axes X, Y, Z. Ignoring signs, S exchanges
X and Y, and H exchanges X and Z; these generate all six permutations.
The kernel consists of I, X, Y, Z, whose conjugations only change signs.
Thus :math:`\mathrm{Cl}_1/\mathcal P_1\cong S_3` by the first isomorphism theorem.
This is a quotient by the normal Pauli subgroup, not deletion of its elements.

Each coset has four elements. ``TRANSVERSAL`` chooses six representatives,
so :math:`D X^a Z^b` identifies all 24 Cliffords. The representatives are not
a subgroup (:math:`S^2=Z`). In particular :math:`S^\dagger=SZ` has the same
unsigned permutation as S but a different Pauli part. Frame updates and
measurement outcomes require the full signed element, not only its coset.

Clifford Element Class
----------------------

.. autoclass:: graphqomb.clifford_algebra.C1Element
    :members:
    :member-order: bysource

Functions
---------

.. autofunction:: graphqomb.clifford_algebra.compose

.. autofunction:: graphqomb.clifford_algebra.inverse

.. autofunction:: graphqomb.clifford_algebra.decompose

.. autofunction:: graphqomb.clifford_algebra.act_on_axis

.. autofunction:: graphqomb.clifford_algebra.act_on_plane_angle

.. autofunction:: graphqomb.clifford_algebra.is_z_axis_preserving

.. autofunction:: graphqomb.clifford_algebra.to_matrix

.. autofunction:: graphqomb.clifford_algebra.from_matrix

.. autofunction:: graphqomb.clifford_algebra.from_local_clifford
