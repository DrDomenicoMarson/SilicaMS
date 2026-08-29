# TEPS example

This directory was recovered in full from
`PoreMS/PoreMS@038b034238d0609e0e0f660d5866089e5d3701c8`. It contains the TEPS
and thymol inputs, the original generated comparison campaigns, and the
historical minimization files. The generated files are retained as reference
artifacts and were not regenerated during the migration to SilicaMS.

The two `_0_create_slit_alpha_*.py` scripts use the current `silicams` package.
Both preserve their historical, explicitly supplied physical
`surface_silicon_fraction` values; the `alpha_new` and `alpha_old` variants
therefore remain scientifically distinct inputs rather than compatibility
alternatives. SilicaMS does not replace these values with a wall-geometry
estimate.

Run the current-alpha construction series from the repository root with:

```bash
/Users/dm/miniforge3/envs/md/bin/python3 \
  user_examples/TEPS_example/_0_create_slit_alpha_new.py
```

An optional `--seed-base INTEGER` produces reproducible alternative surface
arrangements. The base seed is offset by the system's position in the series.

After constructing the five systems, fill them with the bundled thymol box:

```bash
user_examples/TEPS_example/_1_fill_slit.sh
```

The fill script expects the installed `silicams-fill-slit` command. Set
`SILICAMS_FILL_COMMAND` to an alternative executable path when needed.

The construction scripts currently use the deliberately permissive graft
default, `FunctionalizedSlitStericConfig(clearance_scale=0.60)`. Pass an
explicit `steric_settings` object to `FunctionalizedAmorphousSlitConfig` to
try a stricter continuous value such as `0.75` or `0.85`. The historical fill
script explicitly retains its especially permissive `--general-cutoff 0.04`;
`0.10`, `0.15`, and `0.20 nm` are progressively stricter alternatives. A
stricter graft choice may prevent an exact target from being realized; a
stricter fill cutoff retains fewer thymol molecules. Either choice should be
recorded and validated per system.

## Verification boundary

A fresh `msn_9_1` construction and full topology export were exercised with
SilicaMS 0.5, and the filling command completed with the bundled thymol box.
GROMACS 2026.3 accepted the generated coordinates and topology with `grompp`.
The archived `min.mdp` is not yet a warning-clean modern minimization workflow:
it omits `periodic-molecules = yes`, which is required for this
single-molecule infinite silica network. Adding that setting removes the
periodic-shift and domain-decomposition failures, but the permissively built
starting structure still produces transient LINCS warnings under the archived
constrained minimization protocol. In the representative smoke test, an
initial unconstrained steepest-descent stage using a smaller `0.002 nm` maximum
step converged without warnings to `Fmax < 1000 kJ mol-1 nm-1`; the constrained
stage then completed without LINCS warnings. This supports staged relaxation
of the permissive construction, but the protocol should be validated across
the complete TEPS series before it replaces the archived input.
