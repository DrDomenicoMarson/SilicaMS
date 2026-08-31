# SilicaMS

[![Tests](https://github.com/DrDomenicoMarson/SilicaMS/actions/workflows/tests.yml/badge.svg)](https://github.com/DrDomenicoMarson/SilicaMS/actions/workflows/tests.yml)
[![Documentation](https://github.com/DrDomenicoMarson/SilicaMS/actions/workflows/docs.yml/badge.svg)](https://github.com/DrDomenicoMarson/SilicaMS/actions/workflows/docs.yml)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

SilicaMS builds periodic amorphous silica structures for molecular simulation.
Its supported high-level workflow prepares bare or surface-functionalized
silica slits, validates their connectivity and charge contracts, and writes
coordinate, topology, and structured report files.

The package currently supports slit construction only. Molecule, connectivity,
pattern, shape, and geometry primitives remain available through explicit
submodules so future silica geometries can use the same domain and writer
boundaries without restoring the former general-purpose constructors.

Documentation is published at
[drdomenicomarson.github.io/SilicaMS](https://drdomenicomarson.github.io/SilicaMS/).

## Installation

SilicaMS 0.5.0 requires Python 3.14 or newer. Until the first public package
release, install it from a clone:

```bash
git clone https://github.com/DrDomenicoMarson/SilicaMS.git
cd SilicaMS
/absolute/path/to/python3 -m pip install -e .
```

For a reproducible server deployment, install a specific Git commit:

```bash
/absolute/path/to/python3 -m pip install \
    "SilicaMS @ git+https://github.com/DrDomenicoMarson/SilicaMS.git@COMMIT_SHA"
```

Install the development dependencies with:

```bash
/absolute/path/to/python3 -m pip install -e '.[test,docs,dev]'
```

## Bare silica slit

```python
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
print(result.report.final_surface)
print(result.bare_charge_diagnostics.is_neutral)
```

`ExperimentalSiliconStateTarget` always describes fractions over all silicon
atoms. Omitted `q4_fraction` values are derived from the remaining
`Q2/Q3/T2/T3` fractions. The required `surface_silicon_fraction` is the
physically justified fraction of all sample silicon represented by the
modeled surface population (the quantity often called alpha). SilicaMS does
not estimate this mapping from the simulated wall geometry: wall thickness
and template replication are modeling choices, not measurements of the
experimental sample. Builds are deterministic unless `random_seed` is set.

## Functionalized silica slit

```python
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
print(result.report.final_surface)
```

Coordinate-only functionalized output is supported without a ligand ITP.
Pass `SilaneTopologyConfig` with a self-contained flat ligand ITP when a full
functionalized GROMACS ITP/TOP pair is required. Junction terms are controlled
through `AmorphousSlitConfig.silica_topology` and
`sms.default_silica_topology()`.

### Construction contact policy

The contact defaults are deliberately permissive construction heuristics.
They are intended to make exact slit and loading targets practical and to
produce starting coordinates for staged molecular-mechanics minimization;
they do not assert a force-field-valid local minimum.

For ligand grafting, `FunctionalizedSlitStericConfig.clearance_scale=0.60`
scales sums of covalent radii. Continuous values remain user-selectable. In
the bundled TEPS series, `0.75` and `0.85` are useful progressively stricter
values; values of `0.90` or above can make dense exact targets very slow or
impossible to realize. These observations are system-specific, not universal
physical thresholds. The selected settings are stored in the preparation
report.

For filling, `SlitFillConfig.general_cutoff_nm=0.10` is likewise permissive.
Values of `0.15` and `0.20` nm are useful stricter starting points, with the
expected tradeoff of removing more guest molecules. The historical TEPS
workflow uses `0.04` nm and is especially permissive. Ring-crossing checks
remain active independently of this cutoff. Inspect the generated structure
and use an unconstrained gentle minimization before constrained relaxation
and equilibration.

## Slit filling and density

The packaged command-line interfaces are:

```bash
silicams-fill-slit --help
silicams-slit-density --help
```

The equivalent Python APIs are `sms.fill_slit(...)` and
`sms.estimate_guest_density(...)`. Both workflows use
`sms.PeriodicSlitGeometry`, which records an orthorhombic box, the slit-normal
axis, two fitted mean surface planes, projected interfacial area, geometric
slit volume, surface support count, and normal RMS roughness. Construction
writes this unit-explicit schema-v1 geometry to `<system_name>.yml`; filling
writes the output-frame geometry to `<output_stem>.yml`.

Filling puts the slit normal on the output z axis, applying the same axis
permutation to coordinates, any input velocities, box lengths, and geometry.
Centering and whole-residue wrapping change positions only. Velocity columns
are omitted when neither input contains velocities; when only one does, the
other input's velocities are written as zeros.

The standalone density workflow, its configuration, and its report types live
in `silicams.slit_density`; filling lives in `silicams.slit_fill`. The package-root
APIs shown above are unchanged. After updating an existing installation, rerun
your installation command (including for editable installs) to refresh the
`silicams-slit-density` console entry point. Its name and options are unchanged.

Filling supports mixed guest reservoirs. Cropping, surface-plane filtering,
the general all-atom cutoff, and applicable ring-crossing checks are applied
to every residue type. `target_resname` selects only the species used for
density calculations and the target-specific summary fields; the structured
report also includes per-residue-name filtering counts for the complete
mixture. Monatomic and non-aromatic residues skip only the bond or ring check
that is physically inapplicable.

Numeric configuration values are validated before construction: NaN and
infinite values are rejected, and counts, indices, repetitions, and seeds must
be actual integers rather than integer-valued floats or booleans.

GRO, PDB, and mmCIF writers validate connectivity in `strict` mode by default.
Invalid assembled chemistry therefore prevents output unless the caller
explicitly selects `warn` or `off`. Core multi-file slit and filling exports
are staged and promoted as a set; a handled late failure preserves the prior
complete output set rather than leaving partial new files.

Pass geometry explicitly with `slit_geometry=` in Python or
`--slit-geometry PATH` on the command line. If neither an object nor a path is
provided, geometry is inferred from hydroxylated surface Si atoms. Neighboring
YAML files are never selected automatically, so fully functionalized systems
whose surface Si atoms can no longer be inferred should use the geometry file
written by construction or filling.

Density Monte Carlo points are sampled uniformly only inside the signed-padded
mean-plane interval. Framework van der Waals radii plus each requested probe
radius then define the reported **probe-free volume**. This is a geometric
exclusion estimate; it does not establish solvent connectivity, reachability,
or experimentally accessible pore volume. Reports retain every seed-level
fraction, volume, and density. Plotting workflows must also export their plot
data as CSV.

## Output and extension APIs

The high-level write functions finalize one `SilicaSlit`, create one immutable
snapshot, and share it between:

- `StructureWriter` for GRO, PDB/CONECT, mmCIF bonds, XYZ, LAMMPS, validation,
  and structural object output.
- `GromacsTopologyWriter` for charge diagnostics and slit ITP/TOP output.
- `AntechamberWriter` for standalone molecule helper inputs.

Object output is opt-in. SilicaMS object files contain the `silicams` module
namespace; object files serialized by PoreMS are intentionally unsupported.

Reusable lower-level components are imported explicitly, for example:

```python
from silicams.molecule import Molecule
from silicams.pattern import AlphaCristobalit
from silicams.shape import Cylinder
```

These primitives do not imply that cylindrical pores are currently supported.
A future pore geometry should be introduced through a dedicated builder using
the existing `SilicaSlit`-style domain and shared writer snapshot.

## Development and verification

Use the project environment directly:

```bash
/Users/dm/miniforge3/envs/md/bin/python3 -m pytest \
    -n auto --dist loadgroup --cov=silicams --cov-report=term-missing
/Users/dm/miniforge3/envs/md/bin/python3 -m sphinx -W -b html docs docs/_build/html
/Users/dm/miniforge3/envs/md/bin/python3 -m pip wheel . \
    --no-deps --no-build-isolation --wheel-dir /tmp/silicams-wheel
```

The scope-aware parallel distribution keeps each expensive slit fixture in a
single worker while running independent large-bare, small-bare, and
functionalized workflows concurrently. Use plain ``python -m pytest`` when a
serial run is preferable for debugging.

The `dev` extra installs the declared Setuptools build backend so the explicit
no-build-isolation wheel check runs against the prepared project environment.

The `examples/` directory contains minimal bare, functionalized, and filling
workflows. The larger `user_examples/TEPS_example/` directory preserves the
complete historical TEPS study inputs and generated comparison campaigns from
`PoreMS/PoreMS@038b034238d0609e0e0f660d5866089e5d3701c8`; its construction and
filling scripts have been migrated to the current SilicaMS interfaces.

## Origin and attribution

SilicaMS began as a modified fork of
[PoreMS](https://github.com/PoreMS/PoreMS), originally developed by Hamzeh
Kraus and the PoreMS contributors. This repository preserves the complete
upstream commit history but contains substantial changes focused on periodic
amorphous and functionalized silica slits. SilicaMS is independently
maintained by Domenico Marson and is not an official continuation endorsed by
the original PoreMS maintainers.

The inherited construction methods should continue to acknowledge:

> Kraus et al., *PoreMS: a software tool for generating silica pore models
> with user-defined surface functionalisation and pore dimensions*, Molecular
> Simulation 47 (2021), 306–316.
> [doi:10.1080/08927022.2020.1871478](https://doi.org/10.1080/08927022.2020.1871478)

The bundled 60,000-atom amorphous template was added after that paper, in
[PoreMS 0.2.4](https://pypi.org/project/porems/0.2.4/). The original PoreMS
source attributes its 9.605 nm cubic block to
Vink and Barkema's vitreous-silica work. Its 20,000 Si and 40,000 O atoms give
a nominal density of approximately 2.252 g cm^-3, closely matching the 2.25
g cm^-3 model described in
[Vink's 2002 dissertation](https://dspace.library.uu.nl/handle/1874/680).
The related peer-reviewed method and validation are reported in
[Vink and Barkema, Physical Review B 67, 245201
(2003)](https://doi.org/10.1103/PhysRevB.67.245201). The repository records
this as a strong upstream attribution rather than independent coordinate-level
verification because no separately hosted original coordinate checksum has
been located.

That DOI and the historical PoreMS Zenodo records identify the original work,
not a SilicaMS release. See [NOTICE.md](NOTICE.md) and
[CITATION.cff](CITATION.cff) for citation metadata and
[methods.md](methods.md#bundled-amorphous-template) for the complete template
provenance, fingerprints, generation history, and limitations.

## License

SilicaMS is distributed under the GNU General Public License, version 3. See
[LICENSE](LICENSE).
