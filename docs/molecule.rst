Molecules and reusable primitives
=================================

``Molecule`` remains part of the focused package root because ligands and
standalone structure export use it directly:

.. code-block:: python

   import silicams as sms

   molecule = sms.Molecule("example", "EXM")
   molecule.add("C", [0.0, 0.0, 0.0], name="C1")
   molecule.add("O", [0.14, 0.0, 0.0], name="O1")

Other construction primitives require explicit imports:

.. code-block:: python

   from silicams.matrix import Matrix
   from silicams.pattern import AlphaCristobalit
   from silicams.shape import Cylinder

Their availability supports future geometry-specific silica builders. It does
not make cylindrical, spherical, or capsule pore construction a supported
SilicaMS workflow today.
