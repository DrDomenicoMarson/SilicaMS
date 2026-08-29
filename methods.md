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
axis, wrapped through the periodic cell, and rejected when local steric
clearance is negative. Default ordering is deterministic. Setting
`random_seed` randomizes chemically equivalent bridge and attachment choices
reproducibly.

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

## Export and validation

Finalization is idempotent. One immutable export snapshot defines atom and
residue ordering, molecule membership, box dimensions, source identifiers,
assembled bonds and angles, graft junctions, and residue counts. Structure and
topology writers consume the same snapshot so numbering and connectivity stay
consistent across formats.

Bare exports must satisfy the silica charge and coordination identities.
Functionalized topology exports additionally validate expected fragment
charges and final neutrality. Connectivity validation supports `off`, `warn`,
and `strict` modes.

## Slit filling and density

The filling workflow selects complete guest residues from a larger reservoir,
rejects general all-atom clashes and aromatic-ring crossings, and optionally
requires every guest atom to lie inside a signed-padded mean-plane slit
interval. Positive padding contracts both faces; negative padding expands the
interval, but the validated padded width must remain positive and cannot
exceed the periodic box length along the slit normal.

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
box. A point is probe-free when it lies outside every framework van der Waals
radius enlarged by that probe radius under orthorhombic periodic boundary
conditions. The probe-free fraction is normalized by the padded geometric
slit volume, and the guest mass divided by the corresponding probe-free volume
gives the reported probe-free density. Repeated seeded estimates retain all
seed-level values. This local geometric exclusion does not test whether free
regions are connected to one another or reachable from a reservoir, so it
must not be interpreted as a connectivity-based or experimental accessible
pore volume.

## Lineage

The amorphous silica construction methodology descends from PoreMS. See
`NOTICE.md` and `CITATION.cff` for upstream attribution and citation details.
