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
        q2_fraction=66 / 40000,
        q3_fraction=650 / 40000,
    ),
)

result = sms.write_bare_amorphous_slit("output/bare", config)
print(result.report.final_surface)
print(result.bare_charge_diagnostics.is_neutral)
```

`ExperimentalSiliconStateTarget` always describes fractions over all silicon
atoms. Omitted `q4_fraction` values are derived from the remaining
`Q2/Q3/T2/T3` fractions. Builds are deterministic unless `random_seed` is set.

## Functionalized silica slit

```python
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
print(result.report.final_surface)
```

Coordinate-only functionalized output is supported without a ligand ITP.
Pass `SilaneTopologyConfig` with a self-contained flat ligand ITP when a full
functionalized GROMACS ITP/TOP pair is required. Junction terms are controlled
through `AmorphousSlitConfig.silica_topology` and
`sms.default_silica_topology()`.

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
workflows.

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

That DOI and the historical PoreMS Zenodo records identify the original work,
not a SilicaMS release. See [NOTICE.md](NOTICE.md) and
[CITATION.cff](CITATION.cff) for provenance and citation metadata.

## License

SilicaMS is distributed under the GNU General Public License, version 3. See
[LICENSE](LICENSE).
