Slit filling and density
========================

SilicaMS can filter a larger guest reservoir into the accessible region of a
generated slit and estimate the resulting density.

.. code-block:: bash

   silicams-fill-slit --help
   silicams-slit-density --help

The equivalent Python entry points are ``silicams.fill_slit`` and
``silicams.estimate_guest_density``. Both return structured dataclass reports.
The filling implementation preserves complete guest residues, applies
all-atom clash and aromatic-ring crossing checks, and can restrict guests to
the detected slit interval.

Density estimates retain the raw probe-radius and repeated-sampling values.
Any downstream plotting workflow must export companion plot data as CSV.
