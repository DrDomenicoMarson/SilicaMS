Origin, attribution, and citation
=================================

Project lineage
---------------

SilicaMS is derived from the GPL-3.0-licensed
`PoreMS project <https://github.com/PoreMS/PoreMS>`_. The complete inherited
Git history is retained, while the active package has been substantially
refactored and is independently maintained.

For the inherited pore-construction framework, cite:

  Kraus et al., *PoreMS: a software tool for generating silica pore models
  with user-defined surface functionalisation and pore dimensions*, Molecular
  Simulation 47 (2021), 306–316,
  doi:`10.1080/08927022.2020.1871478 <https://doi.org/10.1080/08927022.2020.1871478>`_.

Bundled amorphous template
--------------------------

The packaged ``silicams/templates/amorph.gro`` file contains 20,000 Si and
40,000 O atoms in a 9.605 nm cubic periodic box. Its nominal density is about
2.252 g cm :sup:`-3`. It was inherited from
`PoreMS commit 8d19b0
<https://github.com/PoreMS/PoreMS/commit/8d19b0393dd98482d86c60876dac5e53dfecb2e6>`_,
which introduced the amorphous builder later released in
`PoreMS 0.2.4 <https://pypi.org/project/porems/0.2.4/>`_. Except for the
descriptive first line, the SilicaMS and original PoreMS GRO files are
byte-identical. The SHA-256 checksum of lines 2 onward is
``d2df315fb075d7a094c0fd84994535fe620b1888f4b0128c416f7f79dab2fd46``.

The original PoreMS implementation attributes this 9.605 nm block to:

  Vink and Barkema, *Large well-relaxed models of vitreous silica,
  coordination numbers, and entropy*, Physical Review B 67, 245201 (2003),
  doi:`10.1103/PhysRevB.67.245201 <https://doi.org/10.1103/PhysRevB.67.245201>`_.

The closest description of the specific artifact is Chapter 4 of
`Vink's 2002 dissertation
<https://dspace.library.uu.nl/handle/1874/680>`_. It reports a 60,000-atom
vitreous-silica network at 2.25 g cm :sup:`-3`, generated from a 20,000-atom
amorphous-silicon continuous random network by oxygen insertion, WWW-style
bond transpositions with the Tu-Tersoff potential, and a final BKS quench with
volume optimization. The atom count, composition, box, and density agree with
the bundled template.

The journal article reports a related 60,000-atom model held at
2.20 g cm :sup:`-3`. It supports the scientific generation and validation
method, but an independently hosted original coordinate file or checksum has
not been located. Exact identity with the Vink artifact therefore remains a
strong, repository-documented attribution rather than an independent
byte-level verification.

The historical PoreMS phrase "amorphous beta-cristobalite block" is not used
as a scientific description here. Vink describes an amorphous-silicon
continuous-random-network backbone, not a beta-cristobalite starting crystal.
The template is consequently described as vitreous or amorphous silica.

PoreMS reconstructed connectivity using a 0.140--0.180 nm Si-O interval and
then removed the exceptional pair ``(57790, 2524)``. SilicaMS retains those
defaults. Their use is a PoreMS/SilicaMS processing choice; the GRO coordinate
file contains no bond list, and the reason for the exceptional split was not
documented upstream.

Citation scope
--------------

The PoreMS paper describes version 0.2.0, in which only beta-cristobalite was
implemented. The amorphous builder was added in PoreMS 0.2.4 after the paper.
The PoreMS paper should therefore be cited for the construction framework and
the Vink-Barkema work for the amorphous bulk model and its scientific basis.

The paper DOI and historical PoreMS Zenodo records do not identify SilicaMS
releases. See ``NOTICE.md`` and ``CITATION.cff`` in the repository for the
complete provenance statement.
