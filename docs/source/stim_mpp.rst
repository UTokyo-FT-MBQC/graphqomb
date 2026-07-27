Stim MPP import
===============

Install the optional Stim integration before importing this module:

.. code-block:: console

   uv add "graphqomb[stim]"

The importer converts unsigned Stim ``MPP`` products into a
:class:`graphqomb.qeccode.StabilizerCode`. It does not retain stabilizer
signs. Consequently, signed products written with an inverted Pauli target,
such as ``MPP !X0*Z1``, raise :class:`ValueError` instead of silently losing
the sign.

The result also provides dense-column mappings for sparse Stim qubit ids and
can map selected ``DETECTOR`` and ``OBSERVABLE_INCLUDE`` records to graph-state
ancilla nodes.

Set ``coord_dims`` to the number of leading ``QUBIT_COORDS`` components to
retain. The importer does not restrict this value to two or three dimensions;
each referenced qubit's final accumulated coordinate must contain at least the
requested number of components. Repeated declarations are combined according
to Stim's coordinate semantics.

.. code-block:: python

   from graphqomb.qeccode import build_graph_state
   from graphqomb.stim_glue.mpp import stabilizer_code_from_stim_text

   extraction = stabilizer_code_from_stim_text("MPP X0*Z2")
   graph_result = build_graph_state(extraction.code)
   detector_groups = extraction.detector_groups(graph_result.ancilla_nodes)

API reference
-------------

.. autoclass:: graphqomb.stim_glue.mpp.StimMppExtraction
   :members:
   :show-inheritance:

.. automodule:: graphqomb.stim_glue.mpp
   :members:
   :show-inheritance:
