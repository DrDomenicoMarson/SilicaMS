Slit filling and density
========================

SilicaMS can filter a larger guest reservoir into the mean-plane interval of a
generated slit and estimate the resulting density.

.. code-block:: bash

   silicams-fill-slit --help
   silicams-slit-density --help

For example, reuse geometry written by the construction workflow explicitly:

.. code-block:: bash

   silicams-fill-slit \
       --guest guest.gro \
       --slit functionalized.gro \
       --slit-geometry functionalized.yml \
       --output filled.gro

   silicams-slit-density \
       --input filled.gro \
       --slit-geometry filled.yml

The equivalent Python entry points are ``silicams.fill_slit`` and
``silicams.estimate_guest_density``. Both return structured dataclass reports.
The filling implementation preserves complete guest residues, applies
all-atom clash and aromatic-ring crossing checks, and can restrict guests to
the slit interval. Geometry may be supplied as a ``PeriodicSlitGeometry``
object through Python or as an explicit schema-v1 YAML path through either
interface. When neither is supplied, SilicaMS infers two mean planes from
hydroxylated surface Si atoms. It never discovers a neighboring YAML file
automatically. Filling writes the geometry after output-axis permutation to
``<output_stem>.yml``.

The signed ``surface_plane_padding_nm`` contracts the mean-plane interval when
positive and expands it when negative. The resulting width must stay positive
and no larger than the normal box length. Density sampling is uniform inside
this padded geometric interval. Framework van der Waals radii enlarged by the
probe radius determine the **probe-free fraction and volume**. These values do
not test connectivity or reachability and therefore are not experimental or
connectivity-based accessible pore volumes.

Density estimates retain the raw probe-radius, seed, fraction, volume, and
density values. Any downstream plotting workflow must export companion plot
data as CSV.

Python example
--------------

.. code-block:: python

   from pathlib import Path

   import silicams as sms

   report = sms.estimate_guest_density(
       sms.SlitDensityConfig(
           input_path=Path("filled.gro"),
           slit_geometry_path=Path("filled.yml"),
           surface_plane_padding_nm=0.05,
           random_seed=17,
       )
   )
   print(report.density_estimate.geometric_slit_volume_nm3)
   print(report.density_estimate.probe_estimates[0].probe_free_volume_mean_nm3)
