# SilicaMS provenance notice

SilicaMS is a modified work derived from
[PoreMS](https://github.com/PoreMS/PoreMS), which was originally developed by
Hamzeh Kraus and the PoreMS contributors and distributed under the GNU General
Public License, version 3.

The complete inherited Git history is retained to preserve authorship and
change provenance. Historical contributors visible in that history include
Hamzeh Kraus, Marc Högler, Domenico Marson, and other contributors recorded
under their GitHub or Git identities.

During 2026, Domenico Marson substantially modified the software to focus it
on periodic amorphous and functionalized silica slits. Major changes include a
dedicated silica-slit domain model, geometry-specific construction, separated
structure and topology writers, explicit surface-composition and charge
diagnostics, slit filling and density workflows, and the breaking migration
from the `porems` namespace to `silicams`.

SilicaMS is independently maintained and is not an official continuation
endorsed by the original PoreMS maintainers.

The original PoreMS construction framework is described in:

> Kraus et al., “PoreMS: a software tool for generating silica pore models
> with user-defined surface functionalisation and pore dimensions,”
> *Molecular Simulation* 47 (2021), 306–316.
> <https://doi.org/10.1080/08927022.2020.1871478>

That paper documents PoreMS 0.2.0, where only beta-cristobalite was
implemented. The bundled amorphous template was introduced later in
[PoreMS commit `8d19b0`](https://github.com/PoreMS/PoreMS/commit/8d19b0393dd98482d86c60876dac5e53dfecb2e6)
and released with [PoreMS 0.2.4](https://pypi.org/project/porems/0.2.4/). The
original source attributes its 60,000-atom, 9.605 nm cubic vitreous-silica
block to:

> R. L. C. Vink and G. T. Barkema, “Large well-relaxed models of vitreous
> silica, coordination numbers, and entropy,” *Physical Review B* 67, 245201
> (2003). <https://doi.org/10.1103/PhysRevB.67.245201>

The dimensions and nominal density of approximately 2.252 g cm^-3 closely
match the 60,000-atom, 2.25 g cm^-3 model described in Chapter 4 of:

> R. L. C. Vink, *Computer Simulations of Amorphous Semiconductors*, Utrecht
> University (2002). <https://dspace.library.uu.nl/handle/1874/680>

Except for its descriptive first line, the current SilicaMS GRO file is
byte-identical to the original PoreMS artifact. Lines 2 onward have SHA-256
checksum
`d2df315fb075d7a094c0fd84994535fe620b1888f4b0128c416f7f79dab2fd46`.
No independently hosted copy or checksum of the original Vink coordinates has
been located, so the coordinate identity is repository-attributed rather than
independently verified. See `methods.md` for the complete scientific and
processing provenance.

The paper DOI and historical PoreMS Zenodo records refer to the original
project and must not be represented as SilicaMS release identifiers.

SilicaMS remains licensed as a whole under GPL-3.0. The complete license text
is provided in `LICENSE`.
