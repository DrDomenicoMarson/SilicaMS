SilicaMS
========

SilicaMS builds periodic amorphous and functionalized silica structures for
molecular simulation. Its supported high-level construction is currently a
silica slit. Retained lower-level geometry primitives are extension points,
not claims of support for other pore geometries.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   slit
   filling
   molecule
   api
   provenance

Quick start
-----------

.. code-block:: python

   import silicams as sms

   config = sms.AmorphousSlitConfig(name="bare_silica_slit")
   result = sms.write_bare_amorphous_slit("output/bare", config)
   print(result.report.final_surface)

SilicaMS object files use the ``silicams`` Python namespace. Objects serialized
by PoreMS are intentionally unsupported.
