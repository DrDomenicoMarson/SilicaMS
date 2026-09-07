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
       --slit-geometry filled.yml \
       --framework-resname OM \
       --framework-resname SI \
       --framework-resname SL \
       --framework-resname SLG \
       --framework-resname TPS \
       --framework-resname TPSG

The equivalent Python entry points are ``silicams.fill_slit`` and
``silicams.estimate_guest_density``. Both return structured dataclass reports.
Density's implementation and types are in ``silicams.slit_density``; filling
remains in ``silicams.slit_fill``. Package-root imports remain unchanged.
After updating an existing installation, rerun the installation command,
including for editable installs, to refresh the ``silicams-slit-density``
console entry point.

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
``<output_stem>.yml`` together with the source-derived framework/mobile
residue names and counts. This provenance is descriptive; standalone analysis
does not silently use it as a selector. Filling rejects a residue name present
in both source files because that name would become ambiguous after merging.

Standalone analysis classifies every residue name explicitly.
``framework_resnames`` identifies the slit atoms used for probe exclusion;
``mobile_resnames`` identifies non-target mobile species; and
``target_resname`` is automatically mobile and selects only the density
numerator. The default framework names are ``OM``, ``SI``, ``SL``, and
``SLG``. User-defined graft names and every co-guest must be added explicitly.
On the CLI, repeat ``--framework-resname`` or ``--mobile-resname`` as needed;
supplying any framework options replaces the default set. Overlap,
unclassified residue names, and an empty resolved framework are errors.

The output slit normal is on ``z``. Coordinates, optional input velocities,
box lengths, and geometry use the same axis permutation. Centering and
whole-residue wrapping affect positions only; input arrays are not modified.
Velocity columns are omitted when neither input supplies velocities. If only
one input supplies them, the other input's output velocities are zero-filled.

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
this padded geometric interval. A target molecule belongs to the density
numerator when its mass-weighted center, calculated after periodic
whole-residue reconstruction, lies inside the same interval. Full-box target
mass is used only for box-average density. Geometric and probe-free slit
densities use the interval-assigned mass, and reports include full-box, inside,
and outside molecule counts. Filling's optional all-atoms-inside plane filter
remains a stricter construction rule; its density report still uses the shared
center-of-mass analysis rule on the finalized coordinates.

Framework van der Waals radii enlarged by the probe radius determine the
**probe-free fraction and volume**. Other mobile mixture components do not
exclude this framework-accessible volume, so target species analyzed with the
same probe radius share one denominator. These values do not test connectivity
or reachability and therefore are not experimental or connectivity-based
accessible pore volumes.

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
           framework_resnames=("OM", "SI", "SL", "SLG", "TPS", "TPSG"),
           mobile_resnames=("SOL",),
           surface_plane_padding_nm=0.05,
           random_seed=17,
       )
   )
   print(report.density_estimate.geometric_slit_volume_nm3)
   print(report.density_estimate.probe_estimates[0].probe_free_volume_mean_nm3)
