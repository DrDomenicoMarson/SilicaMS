Silica slit preparation
=======================

Bare slits
----------

``ExperimentalSiliconStateTarget`` stores ``Q2/Q3/Q4/T2/T3`` fractions over
all silicon atoms. Bare slits require zero ``T2`` and ``T3`` fractions.

.. code-block:: python

   import silicams as sms

   config = sms.AmorphousSlitConfig(
       name="bare_silica_slit",
       slit_width_nm=7.0,
       repeat_y=2,
       surface_target=sms.ExperimentalSiliconStateTarget(
           q2_fraction=66 / 40000,
           q3_fraction=650 / 40000,
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
               q2_fraction=65 / 957,
               q3_fraction=651 / 957,
               q4_fraction=239 / 957,
               t2_fraction=1 / 957,
               t3_fraction=1 / 957,
               alpha_override=1.0,
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

Writers and object output
-------------------------

High-level write functions finalize once and pass one immutable snapshot to
the structure and topology writers. Object serialization is opt-in through
``write_object_files=True`` and supports SilicaMS objects only.
