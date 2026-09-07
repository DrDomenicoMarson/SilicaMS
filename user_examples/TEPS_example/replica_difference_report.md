# TEPS campaign realization and composition differences

## Scope

This report compares the five slit compositions stored in each of:

- `deterministic_alphaOLD`;
- `seed33_alphaNEW`;
- `seed11023_alphaNEW`.

The comparison uses the realized `Q2/Q3/Q4/T2/T3` counts in the archived
preparation reports and the periodic surface geometry of the exported slits.
It addresses how these campaigns should be combined in subsequent analysis;
it does not assess equilibration or production-trajectory convergence.

## Main conclusion

The three directories describe the same qualitative progression from bare
silica to high TEPS coverage. The `alphaOLD` structures are not in a different
surface-morphology or coverage regime. Their TEPS nearest-neighbor spacings,
T2/T3 balance, topology, and monotonic coverage ordering are comparable to the
two `alphaNEW` realizations.

They are nevertheless **not three fixed-composition replicas**. A smaller
physical surface-silicon fraction, alpha, maps the same experimental
all-silicon fractions onto larger fractions of the 957 modeled surface sites.
Consequently, `alphaOLD` contains systematically more grafted TEPS and more
silanol groups at the same nominal `9_1`, `8_2`, `7_3`, or `6_4` label. Even
the nominally bare `0_0` surface has a different silanol density.

The appropriate interpretation is therefore:

- `seed33_alphaNEW` and `seed11023_alphaNEW` are two spatial realizations at
  identical surface composition;
- `deterministic_alphaOLD` is a distinct spatial realization **and** an
  off-grid composition/sensitivity series;
- all three can test the robustness of a qualitative coverage trend, but they
  must not be averaged as three equivalent replicas at each nominal label.

This distinction is consistent with the example [README](README.md), which
treats the old and new alpha values as scientifically distinct explicit
inputs, and with the current physical-alpha definition in
[`methods.md`](../../methods.md).

## Realized surface composition

For analysis, use the two physical interfaces of the periodic slit:

$$
A_{\mathrm{periodic}}
 = 2(9.605\ \mathrm{nm})(9.605\ \mathrm{nm})
 = 184.51205\ \mathrm{nm^2}.
$$

Define the number of TEPS groups and the number of remaining surface hydroxyl
groups as

$$
N_{\mathrm{TEPS}} = N_{T2} + N_{T3},
$$

$$
N_{\mathrm{OH}} = 2N_{Q2} + N_{Q3} + N_{T2}.
$$

The latter expression counts two hydroxyls on Q2, one on Q3, one on T2, and
none on Q4 or T3. The corresponding two-face densities are

$$
\Gamma_{\mathrm{TEPS}} = N_{\mathrm{TEPS}}/A_{\mathrm{periodic}},
\qquad
\Gamma_{\mathrm{OH}} = N_{\mathrm{OH}}/A_{\mathrm{periodic}}.
$$

The two `alphaNEW` directories have identical counts, so they share one row in
the following table.

| Nominal system | Alpha set | TEPS count | TEPS / nm² | OH count | OH / nm² | T2 / (T2 + T3) |
|---|---:|---:|---:|---:|---:|---:|
| `0_0` | old | 0 | 0.000 | 370 | 2.005 | n/a |
| `0_0` | new | 0 | 0.000 | 322 | 1.745 | n/a |
| `9_1` | old | 106 | 0.574 | 412 | 2.233 | 0.349 |
| `9_1` | new | 94 | 0.509 | 368 | 1.994 | 0.351 |
| `8_2` | old | 254 | 1.377 | 470 | 2.547 | 0.382 |
| `8_2` | new | 225 | 1.219 | 419 | 2.271 | 0.382 |
| `7_3` | old | 392 | 2.125 | 468 | 2.536 | 0.352 |
| `7_3` | new | 358 | 1.940 | 426 | 2.309 | 0.352 |
| `6_4` | old | 614 | 3.328 | 474 | 2.569 | 0.248 |
| `6_4` | new | 586 | 3.176 | 452 | 2.450 | 0.247 |

Relative to `alphaNEW`, the `alphaOLD` TEPS density is higher by approximately
12.8%, 12.9%, 9.5%, and 4.8% for `9_1`, `8_2`, `7_3`, and `6_4`, respectively.
Its OH density is higher by approximately 14.9%, 12.0%, 12.2%, 9.9%, and 4.9%
for `0_0`, `9_1`, `8_2`, `7_3`, and `6_4`.

The almost identical T2 fraction at each nominal functionalized composition
is important: alpha mainly moves each system along the TEPS/OH-density trend;
it does not substantially change the relative T2/T3 chemistry.

## Spatial realization comparison

The median lateral nearest-neighbor distance between TEPS silicon atoms on
the same interface is also similar across campaigns:

| Nominal system | `alphaOLD` / nm | `seed33_alphaNEW` / nm | `seed11023_alphaNEW` / nm |
|---|---:|---:|---:|
| `9_1` | 0.636 | 0.598 | 0.637 |
| `8_2` | 0.489 | 0.485 | 0.478 |
| `7_3` | 0.437 | 0.425 | 0.428 |
| `6_4` | 0.311 | 0.315 | 0.315 |

The face-to-face allocation varies between realizations, especially at low
coverage where a difference of a few grafts is proportionally large. This is
ordinary finite-system realization variability rather than evidence for a
different coverage regime. It should still be considered when interpreting a
property that is sensitive to face asymmetry or local clustering.

## Recommended normalization and analysis strategy

### 1. Correct the geometric normalization

Use `184.51205 nm²` for quantities normalized over both periodic interfaces,
or `92.256025 nm²` when reporting a quantity for one interface. Do not use the
archived `shape_00.surface` values of roughly `226-228 nm²`; those values
include lateral cuboid faces that do not exist as silica-fluid interfaces
under periodic boundary conditions.

If a historical observable was calculated as

$$
Y_{\mathrm{legacy}} = X/A_{\mathrm{legacy}},
$$

correct it without rerunning the trajectory as

$$
Y_{\mathrm{periodic}}
 = Y_{\mathrm{legacy}}
   A_{\mathrm{legacy}}/184.51205.
$$

The required multiplier is system-dependent but lies between approximately
1.227 and 1.235 for these files. This correction applies only to genuinely
surface-extensive quantities. It must not be applied to diffusion
coefficients, relaxation times, densities, or other already intensive
observables.

### 2. Replace nominal coverage by realized descriptors

For every trajectory or reported observable, retain at least these analysis
columns:

- `campaign`;
- `nominal_system`;
- `alpha`;
- `teps_count` and `teps_density_nm2`;
- `oh_count` and `oh_density_nm2`;
- `t2_fraction_of_teps`;
- `seed` or realization identifier.

Use `teps_density_nm2` as the primary horizontal coordinate for a TEPS
coverage trend. Include `oh_density_nm2` as a second explanatory variable,
colour scale, or companion panel for observables controlled by hydrogen
bonding. The two densities are correlated in this small design, so a
multi-parameter fit should not be presented as independently identifying a
TEPS effect and an OH effect without additional systems that vary them
separately.

Alpha itself is an input used to construct these realized counts. It is not a
factor by which a simulated observable should subsequently be multiplied or
divided.

### 3. Treat replica uncertainty and composition uncertainty separately

At a fixed `alphaNEW` composition, use `seed33` and `seed11023` to estimate
spatial-realization variability. With only two realizations, report the two
values or their range alongside the mean rather than implying a well-resolved
population standard deviation.

Plot the `alphaOLD` value at its actual TEPS and OH densities as a separate
sensitivity point. Do not include it in an unqualified three-replica mean at
the nominal label. If a common-grid comparison is needed, interpolate a
smooth trend to a stated TEPS density and retain the interpolation uncertainty;
do not rescale the observable by the old/new TEPS-count ratio unless the
observable is known a priori to be strictly extensive and linear in graft
count.

### 4. Interpret agreement and disagreement

A consistent monotonic direction across the three campaigns is evidence that
the qualitative coverage trend is robust to both the alpha mapping and the
surface realization. The expected `alphaOLD` result for a smooth response is
usually a displacement along the same response curve because its realized
coverage is modestly higher.

An offset is especially plausible for hydrogen-bond-sensitive observables,
because the OH density differs as well as the TEPS density. A reversal of a
strong trend should not be attributed to alpha alone without checking
equilibration, face asymmetry, local clustering, guest loading, and statistical
uncertainty in the trajectories.

## Suggested terminology

For future text and plots, describe the datasets as:

- **two `alphaNEW` spatial realizations at fixed composition**; and
- **one `alphaOLD` composition-sensitivity realization**.

Collectively they are three realizations of the same broad TEPS-coverage
series, but only the first two are replicas in the strict fixed-input sense.
