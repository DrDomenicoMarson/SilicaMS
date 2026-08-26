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
silicon atoms. The builder derives the corresponding surface-only target using
the automatically measured surface-to-total silicon fraction (`alpha`) unless
`alpha_override` is supplied. Exact integer compositions are preferred;
optional tolerance fallback selects the nearest realizable composition within
the configured per-state fraction tolerance.

Siloxane bridge candidates are chosen from eligible surface-silicon pairs.
Candidate oxygen placements are generated perpendicular to the pair axis,
wrapped through the periodic cell, and rejected when local steric clearance is
negative. Default ordering is deterministic. Setting `random_seed` randomizes
chemically equivalent bridge and attachment choices reproducibly.

## Functionalization

Functionalized targets use explicit `T2` and `T3` fractions. Ligands are
attached through configured mount and axis atoms, using the slit surface normal
and optional rotation about the molecular axis. Each batch records requested,
successful, and rejected site identifiers. The final report stores the
realized silicon-state composition, surface-edit diagnostics, attachment
records, and stage timings.

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
restricts guest atoms to the detected accessible slit interval. Density
analysis estimates accessible volume for configured probe radii through
repeated seeded sampling and records all reusable numerical results in its
structured report.

## Lineage

The amorphous silica construction methodology descends from PoreMS. See
`NOTICE.md` and `CITATION.cff` for upstream attribution and citation details.
