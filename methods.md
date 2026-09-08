# SilicaMS construction methods

## Domain boundary

SilicaMS 0.5 supports periodic amorphous silica slits. The public
`AmorphousSlitBuilder` constructs an attach-ready `SilicaSlit`; the mutable
silica block, connectivity matrix, binding-site cache, and bridge insertion
details remain private to the silica chemistry engine.

Reusable molecular, connectivity, pattern, shape, and geometry components are
retained for future geometry-specific builders. Their presence does not imply
that non-slit pore construction is supported.

## Slit construction

The builder loads the packaged amorphous silica template, replicates it along
the requested periodic direction, reconstructs Si–O connectivity within the
configured distance range, applies template-specific bond splits, and removes
the central cuboid defining the slit. It then removes invalid or orphaned
surface atoms, identifies exposed silicon sites, assigns flat-wall normals,
and validates the remaining scaffold.

### Bundled amorphous template

The packaged `silicams/templates/amorph.gro` file contains 60,000 atoms:
20,000 Si and 40,000 O atoms in a cubic 9.605 nm periodic box. Using atomic
masses of 28.0855 u for Si and 15.999 u for O, this corresponds to a nominal
bulk density of approximately 2.252 g cm^-3. The complete current file has
SHA-256 checksum
`f276b0a82a4e76f26d5a7657034195e97ac0563788fd90e39b4a64590ae02fe6`.

The coordinate payload was introduced by
[PoreMS commit `8d19b0393dd98482d86c60876dac5e53dfecb2e6`](https://github.com/PoreMS/PoreMS/commit/8d19b0393dd98482d86c60876dac5e53dfecb2e6)
and released with [PoreMS 0.2.4](https://pypi.org/project/porems/0.2.4/) on
21 December 2021. The inherited PoreMS source attributes the block to Vink
and Barkema, *Large well-relaxed models of vitreous silica, coordination
numbers, and entropy*, Physical Review B 67, 245201 (2003),
<https://doi.org/10.1103/PhysRevB.67.245201>. Apart from the descriptive first
line of the GRO file, the current SilicaMS template is byte-identical to that
PoreMS artifact. Lines 2 onward have SHA-256 checksum
`d2df315fb075d7a094c0fd84994535fe620b1888f4b0128c416f7f79dab2fd46`
in both versions.

The dimensions and density strongly associate the artifact with the
60,000-atom vitreous-silica model described in Chapter 4 of R. L. C. Vink,
*Computer Simulations of Amorphous Semiconductors*, Utrecht University (2002),
<https://dspace.library.uu.nl/handle/1874/680>. That model was constructed from
a periodic 20,000-atom amorphous-silicon continuous random network by placing
oxygen atoms on the Si-Si bonds, scaling toward the experimental density,
performing approximately one million attempted WWW-style bond transpositions
with the Tu-Tersoff potential, and finally quenching with the BKS potential
including volume optimization. The dissertation reports a final density of
2.25 g cm^-3, mean O-Si-O and Si-O-Si angles of 109.4 and 151.2 degrees, and
fully coordinated Si and O populations under a 1.80 Angstrom Si-O cutoff.

The later journal article reports a related 60,000-atom network constrained to
2.20 g cm^-3 without volume optimization during its final BKS quench. It is an
appropriate source for the generation method and its validation against
neutron-scattering data, but it does not by itself establish byte-level
identity with this 9.605 nm coordinate file. No independently hosted copy or
checksum of Vink's original coordinates has been located. The identification
of the bundled artifact therefore rests on the explicit PoreMS attribution
together with its matching atom count, composition, dimensions, and density.

The original PoreMS docstring called the structure an "amorphous
beta-cristobalite block." That label should not be interpreted as its
generation history: Vink's dissertation describes an amorphous-silicon
continuous-random-network backbone rather than a beta-cristobalite starting
crystal. SilicaMS consequently refers to it as vitreous or amorphous silica.

The GRO file stores coordinates and box dimensions, but not connectivity.
PoreMS reconstructed Si-O bonds in the range 0.140-0.180 nm and then explicitly
removed the pair `(57790, 2524)`. SilicaMS retains that range and pair as the
default `amorph_bond_range_nm` and `template_split_pairs` processing choices.
The reason for the exceptional split was not documented upstream, so it must
not be represented as part of the published Vink coordinate model.

The requested `Q2/Q3/Q4/T2/T3` fractions are interpreted over all active
silicon atoms in the corresponding experimental population. The user must
provide the physically justified fraction of those silicon atoms represented
by the modeled surface population (`surface_silicon_fraction`, often denoted
`alpha`). The builder
uses that mapping to derive the surface-only target. It never estimates alpha
from the fraction of surface atoms in the generated wall because template
replication and wall thickness are model-construction choices rather than
experimental population measurements. The physical surface fraction must be
at least the total experimental non-`Q4` fraction. Exact integer compositions
are preferred; optional tolerance fallback selects the nearest realizable
composition within the configured per-state fraction tolerance.

Siloxane bridge candidates are chosen from eligible surface-silicon pairs.
Candidate-pair distances, bridge vectors, adjacency, and steric clearances use
one orthorhombic minimum-image convention, including pairs that cross a box
face. Candidate oxygen placements are generated perpendicular to the pair
axis and wrapped through the periodic cell. A local steric prescreen orders the
candidates, after which a full-scaffold clearance check accepts the first
candidate whose local and global clearances are no more than ``1e-12`` nm below
zero. This tolerance absorbs binary64 roundoff at exact contact. Default
ordering is deterministic. Setting `random_seed` randomizes chemically
equivalent bridge and attachment choices reproducibly.

The physical slit geometry is fitted from the normal coordinates of the
initial exposed surface-silicon sites. Sites are divided into two periodic face
clusters, and each surface plane is the periodic mean of its cluster. The
pooled normal RMS displacement is stored as a roughness diagnostic rather than
being folded into the width. Geometric slit volume is the mean-plane
separation multiplied by the projected periodic area of one face. Reported
total surface area is twice that face area; it is not the full area of the
orthorhombic box.

## Functionalization

Functionalized targets use explicit `T2` and `T3` fractions. Ligands are
attached through configured mount and axis atoms, using the slit surface normal
and optional rotation about the molecular axis. Each batch records requested,
successful, and rejected site identifiers. The final report stores the
realized silicon-state composition, surface-edit diagnostics, attachment
records, contact settings, and stage timings.

The graft-placement screen uses sums of covalent radii multiplied by the
continuous `clearance_scale` parameter. Its default value of `0.60` is a
deliberately permissive construction heuristic chosen to make exact dense
targets practical. Increasing the scale rejects more crowded placements but
can make an exact target slow or unrealizable. It should not be interpreted as
a force-field nonbonded criterion, and every generated functionalized slit
requires staged energy minimization before equilibration.

Coordinate-only functionalized output does not require a ligand topology. A
self-contained functionalized GROMACS topology additionally requires an
explicit flat ligand ITP and, for geminal sites, explicit generated cross
terms. Silica scaffold and graft-junction parameters are taken from the
resolved `SilicaTopologyModel`.

Each flat ligand ITP must define exactly one `[ moleculetype ]`. Atom names and
molecule-local indices are independently unique, and every `[ bonds ]`,
`[ pairs ]`, `[ angles ]`, and `[ dihedrals ]` atom reference is validated
against the declared `[ atoms ]` indices before assembly. Repeated interaction
sections remain supported within the single molecule definition, while a
second molecule definition is rejected rather than merged implicitly.

## Export and validation

Finalization is idempotent. One immutable export snapshot defines atom and
residue ordering, molecule membership, box dimensions, source identifiers,
assembled bonds and angles, graft junctions, and residue counts. Structure and
topology writers consume the same snapshot so numbering and connectivity stay
consistent across formats.

Bare exports must satisfy the silica charge and coordination identities.
Functionalized topology exports additionally validate expected fragment
charges and final neutrality. Connectivity validation supports `off`, `warn`,
and `strict` modes; `strict` is the default for GRO, PDB, and mmCIF output.
High-level slit output sets and GROMACS ITP/TOP pairs are rendered to staging
paths and promoted only after every requested file succeeds. Existing files
are restored after a handled promotion failure. This is process-level
transaction safety, not a claim of portable multi-file power-loss atomicity.

## Slit filling and density

The filling workflow selects complete guest residues from a larger reservoir.
For every residue type in a mixed reservoir, it rejects general all-atom
clashes, checks every inferred guest bond against slit aromatic rings, checks
slit bonds against every guest ring containing the configured six aromatic
atoms, and optionally requires every guest atom to lie inside a signed-padded
mean-plane slit interval. Monatomic and non-aromatic residues skip only the
bond or ring direction that is not physically applicable. `target_resname`
selects the species used for density and target-specific metrics, while the
report retains per-residue-name filtering outcomes. Positive padding contracts
both faces; negative padding expands the interval, but the validated padded
width must remain positive and cannot exceed the periodic box length along the
slit normal.

The merged GRO, geometry YAML, and report log are staged and promoted as one
exception-safe output set. Filling also writes source-derived component
provenance: residue names from the slit input are recorded as framework and
residue names from the guest reservoir as mobile. This metadata is descriptive
and is not an implicit standalone-analysis selector. A residue name present in
both inputs is rejected because it would become ambiguous after merging.

Output axes are permuted to place the slit normal on z. The same permutation
is applied to slit and guest coordinates, their optional Cartesian velocities
(nm/ps), box lengths, and geometry metadata. Center-crop translations and
whole-residue periodic image shifts affect positions only, not velocities.
Input arrays remain unchanged. If either input contains velocities, all output
atoms receive velocity columns, with missing input velocities filled by zeros;
if neither input contains velocities, those columns are omitted.

The general all-atom cutoff is also a construction heuristic. Its `0.10 nm`
default is deliberately permissive; larger user-selected values remove more
guests before minimization, while the aromatic-ring crossing checks operate
independently. This geometric prefilter does not replace force-field energy
minimization or establish that the retained configuration is equilibrated.

Geometry is resolved from an explicit `PeriodicSlitGeometry`, an explicitly
named schema-v1 YAML file, or hydroxylated-surface-Si inference. No neighboring
metadata file is discovered automatically. The explicit geometry route is
therefore required when chemical functionalization removes the hydroxylated
surface signature needed for inference.

For each configured probe radius, density analysis samples points uniformly
inside the padded geometric slit interval, not throughout the full simulation
box. Standalone analysis assigns exact residue names to `framework_resnames`
or `mobile_resnames`; `target_resname` is always mobile and selects only the
density species. The default framework selector contains the native silica
residues `OM`, `SI`, `SL`, and `SLG`. User-named surface functional groups and
all non-target guests require explicit classification. Selectors must be
disjoint and exhaustive for the residue names present, so ambiguous inputs
fail instead of being assigned by complement.

The resolved standalone density-report path is validated against both the
merged GRO input and any explicit geometry input before either file is parsed.
Canonical path aliases, symbolic links, and existing hard links are treated as
collisions. Report text is first written to a same-directory staging file and
then atomically promoted; a handled staging or promotion failure preserves any
previous report and never replaces either scientific input.

A point is probe-free when it lies outside every explicitly selected framework
van der Waals radius enlarged by the probe radius under orthorhombic periodic
boundary conditions. Mobile species, including non-target mixture components,
do not exclude framework-accessible volume. Consequently all target species in
one mixture can use the same framework denominator. The probe-free fraction is
normalized by the padded geometric slit volume.

The target population is defined independently of framework membership. Each
target residue is reconstructed around its first atom with orthorhombic
minimum-image displacements, and its mass-weighted center determines whether
the whole molecule belongs to the signed-padded mean-plane interval. Full-box,
inside-interval, and outside-interval counts are reported. Full-box target mass
divided by full-box volume gives the box-average density. Interval-assigned
target mass divided by padded geometric volume gives the geometric slit
density, and the same interval mass divided by probe-free volume gives the
probe-free slit density. Thus changing padding updates both the population and
the corresponding control volume. Filling retains its stricter optional rule
that every atom must lie inside the interval as a construction filter, but its
density calculation applies the same center-of-mass population definition as
standalone analysis to the finalized output coordinates.

Repeated seeded estimates retain all seed-level values. This local geometric
exclusion does not test whether free regions are connected to one another or
reachable from a reservoir, so it must not be interpreted as a
connectivity-based or experimental accessible pore volume.

## Lineage

The slit-construction workflow descends from PoreMS, while the bundled bulk
amorphous coordinates are attributed through PoreMS to the Vink-Barkema
vitreous-silica work described above. The PoreMS paper documents version 0.2.0,
which implemented beta-cristobalite only; the amorphous builder was added later
in PoreMS 0.2.4. See `NOTICE.md` and `CITATION.cff` for citation details.
