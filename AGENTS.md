# AGENTS.md

## Project scope

- SilicaMS is a living personal project for constructing periodic amorphous and
  functionalized silica slits for molecular simulation.
- The supported scientific and public behavior is documented in `README.md`
  and `methods.md`. Treat those descriptions as contracts unless the user
  explicitly requests a change.
- Preserve the repository's PoreMS provenance and attribution in `NOTICE.md`,
  `CITATION.cff`, and the README.

## Working agreement

- Inspect the relevant implementation, tests, documentation, and current Git
  diff before proposing or making changes. Preserve unrelated user changes.
- For an audit, assessment, review, or diagnosis, report evidence-backed
  findings and a scoped plan before editing files.
- For an explicit fix, change, refactor, or implementation request, make the
  change and verify it without asking for routine confirmations.
- Ask before making a change that alters the scientific interpretation of an
  existing result or documented model. Ordinary bug fixes that restore the
  current contract do not require an additional approval.
- Do not preserve legacy APIs, CLIs, behavior, or tests for compatibility.
  Remove or update superseded interfaces instead of adding aliases,
  deprecation layers, or compatibility shims.
- Tests must describe the behavior after the change. Update or remove tests
  that exist only to enforce superseded behavior.

## Python environments

- On Domenico's Mac, use the interpreter directly at
  `/Users/dm/miniforge3/envs/md/bin/python3` for Python commands.
- On that Mac, do not use `conda run` or `mamba run`; Codex cannot reliably use
  those commands because of the local permission setup.
- The `conda run`/`mamba run` restriction is local to Domenico's Mac. On
  GitHub-hosted agents and other computers, use the environment tooling
  appropriate to that system and a prepared Python 3.14-or-newer environment.
- Development dependencies may be installed into the selected environment as
  needed. Do not modify project dependency declarations merely to repair a
  transient local environment.

## Code and API design

- Prefer dataclasses to unstructured dictionaries for domain data with a
  stable schema.
- Do not introduce a configuration dataclass when a small, clear set of
  function arguments is sufficient.
- Add proper docstrings to new functions and classes. When modifying a
  function or class, update its docstring at the same time. Document every
  argument and, where applicable, return values, raised exceptions, units, and
  scientific meaning.
- Keep public interfaces typed and preserve the package's `py.typed` contract.
- Favor clear domain boundaries and explicit data flow over implicit mutable
  state. Refactors that improve logic, usability, or maintainability are
  welcome when they preserve the requested scientific behavior.

## Scientific and output contracts

- Preserve the documented meaning of `Q2/Q3/Q4/T2/T3` fractions, including
  their normalization over all active silicon atoms.
- Preserve deterministic construction by default and reproducible randomness
  when `random_seed` is supplied.
- Keep coordinate and topology writers consistent by deriving them from the
  same finalized export snapshot. Do not allow atom numbering, connectivity,
  residue counts, box dimensions, or charge accounting to diverge between
  formats.
- Maintain the documented coordination, connectivity, steric, fragment-charge,
  and final-neutrality validations. Add focused regression tests whenever one
  of these contracts is fixed or extended.
- Hardcoded aromatic-ring atom names are an accepted project assumption. All
  aromatic ring atoms are named `CA1`, `CA2`, `CA3`, `CA4`, `CA5`, and `CA6`.
- Every plot must have a companion CSV containing the reusable numerical data.
  Include clear column names and units where applicable.

## Documentation

- Update `README.md` whenever installation, CLI/API usage, documented behavior,
  or available functionality changes.
- Update the relevant files under `docs/` for public API or user-workflow
  changes.
- Update `methods.md` when construction, validation, topology, filling, or
  density methodology changes.
- Keep docstrings, README examples, Sphinx documentation, CLI help, and the
  implementation mutually consistent.

## Dependencies

- Prefer current supported dependency versions and an explicit minimum version
  over weakening a better implementation to accommodate an obsolete release.
- Record required minimum versions in `pyproject.toml` and verify them in the
  prepared development environment and CI.
- Do not add fallbacks for obsolete dependencies or bump unrelated dependency
  minimums without evidence. Document user-visible installation changes.

## Verification

Use the selected interpreter directly in place of `<python>` below.

- During development, run the narrowest relevant tests first.
- Before handing off a code change, run the complete critical lint and test
  suite:

  ```bash
  <python> -m ruff check .
  <python> -m pytest --cov=silicams --cov-report=term-missing
  ```

- For public API, documentation, or user-workflow changes, also build the
  documentation with warnings treated as errors:

  ```bash
  <python> -m sphinx -W -b html docs docs/_build/html
  ```

- For packaging, build-system, or dependency changes, also build a wheel using
  the prepared environment:

  ```bash
  <python> -m pip wheel . --no-deps --no-build-isolation \
      --wheel-dir /tmp/silicams-wheel
  ```

- If a required check cannot run, report the exact command, failure, and the
  remaining uncertainty. Do not describe unrun checks as passing.
