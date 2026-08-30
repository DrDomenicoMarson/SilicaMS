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
The filling implementation preserves complete guest residues and supports
mixed reservoirs. It applies center cropping, the optional surface-plane
filter, the all-atom clash check, and every applicable aromatic-ring crossing
check to all residue types. ``target_resname`` selects the species used for
density and target-specific summary fields; ``GuestResidueFilterSummary``
records filtering outcomes for every residue name. Monatomic and non-aromatic
residues skip only inapplicable bond or reverse-ring checks. Geometry may be
supplied as a ``PeriodicSlitGeometry``
object through Python or as an explicit schema-v1 YAML path through either
interface. When neither is supplied, SilicaMS infers two mean planes from
hydroxylated surface Si atoms. It never discovers a neighboring YAML file
automatically. Filling writes the geometry after output-axis permutation to
``<output_stem>.yml``.

``SlitFillConfig.general_cutoff_nm`` is a continuous, user-selectable
all-atom cutoff. Its ``0.10 nm`` default is deliberately permissive so dense
reservoirs can provide useful starting configurations. ``0.15`` and ``0.20
nm`` are useful progressively stricter values; they remove more guest
molecules. The historical TEPS workflow uses an especially permissive ``0.04
nm`` cutoff. These are construction heuristics rather than force-field-valid
contact distances, and the aromatic-ring crossing checks remain independent
of the selected cutoff. Inspect and minimize every filled system before
equilibration.

The public configuration dataclasses reject NaN and infinite scientific
parameters. Counts, repetitions, indices, and seeds require genuine integer
values; boolean and integer-valued float inputs are rejected.

The merged GRO, output geometry YAML, and human-readable log are staged and
promoted together. If a handled late failure occurs, existing outputs remain
unchanged and a new partial set is not exposed.

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
