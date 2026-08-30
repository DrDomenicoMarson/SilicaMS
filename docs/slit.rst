Silica slit preparation
=======================

Bare slits
----------

``ExperimentalSiliconStateTarget`` stores ``Q2/Q3/Q4/T2/T3`` fractions over
all silicon atoms. It also requires ``surface_silicon_fraction``: the
physically justified fraction of the experimental silicon population
represented by the modeled surface. SilicaMS does not infer this value from
the generated wall geometry. Bare slits require zero ``T2`` and ``T3``
fractions.

.. code-block:: python

   import silicams as sms

   config = sms.AmorphousSlitConfig(
       name="bare_silica_slit",
       slit_width_nm=7.0,
       repeat_y=2,
       surface_target=sms.ExperimentalSiliconStateTarget(
           q2_fraction=0.0170,
           q3_fraction=0.1675,
           surface_silicon_fraction=0.60,
       ),
   )
   result = sms.write_bare_amorphous_slit("output/bare", config)

The high-level function returns a ``SlitPreparationResult`` containing the
finalized ``SilicaSlit``, composition report, resolved silica topology, and
bare charge diagnostics. ``AmorphousSlitBuilder.prepare()`` exposes the same
preparation phase without writing files.

``result.report.slit_geometry`` is a ``PeriodicSlitGeometry`` fitted from the
two periodic mean surface-Si planes. It distinguishes the projected area of
one interface from the total projected area of both interfaces and records the
mean-plane volume and normal RMS roughness separately. High-level writers
store the unit-explicit schema-v1 mapping in ``<system_name>.yml`` for explicit
reuse by filling and density analysis.

Functionalized slits
--------------------

.. code-block:: python

   import silicams as sms
   from silicams.generic import tms

   config = sms.FunctionalizedAmorphousSlitConfig(
       slit_config=sms.AmorphousSlitConfig(
           name="functionalized_silica_slit",
           repeat_y=1,
           surface_target=sms.ExperimentalSiliconStateTarget(
               q2_fraction=0.0133,
               q3_fraction=0.1735,
               t2_fraction=0.0195,
               t3_fraction=0.0367,
               surface_silicon_fraction=0.571,
           ),
       ),
       ligand=sms.SilaneAttachmentConfig(
           molecule=tms(),
           mount=0,
           axis=(0, 1),
       ),
   )
   result = sms.write_functionalized_amorphous_slit(
       "output/functionalized",
       config,
   )

Without ``SilaneTopologyConfig``, the workflow writes coordinates and reports
but no functionalized ITP/TOP pair. Full topology export requires a
self-contained flat ligand ITP. Silica junction terms are configured through
``AmorphousSlitConfig.silica_topology``; use
``sms.default_silica_topology()`` to obtain an editable copy.

Contact settings and minimization
---------------------------------

The construction defaults deliberately favor obtaining a usable starting
configuration over enforcing a force-field contact criterion. For grafting,
``FunctionalizedSlitStericConfig(clearance_scale=0.60)`` is the permissive
default. ``0.75`` and ``0.85`` are progressively stricter continuous values
that remained practical in tests on the bundled TEPS series; dense exact
targets can become very slow or unrealizable near ``0.90`` and above. The
chosen settings are included in the preparation report.

These values are system-specific heuristics, not physical presets. Inspect
the coordinates and perform a gentle unconstrained energy minimization before
introducing constraints and proceeding to equilibration.

Writers and object output
-------------------------

High-level write functions finalize once and pass one immutable snapshot to
the structure and topology writers. Object serialization is opt-in through
``write_object_files=True`` and supports SilicaMS objects only.

GRO, PDB, and mmCIF connectivity validation defaults to ``strict``. Invalid
assembled chemistry prevents output unless ``warn`` or ``off`` is selected
explicitly. High-level structure, metadata, report, and optional topology
files are staged and promoted together, preserving the previous complete set
when a handled late export failure occurs.
