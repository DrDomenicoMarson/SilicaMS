"""Guest density analysis and probe-free slit-volume estimates.

The numerical helpers are shared with filling; this module owns the standalone
density workflow, its configuration and reports, and its command-line adapter.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import secrets
from typing import Sequence
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

import silicams.database as db
from .database import _infer_element_from_atom_name
from ._gro_io import _GroSystem, _build_residue_spans, _load_gro_system
from ._output_transaction import staged_output_paths
from ._slit_geometry_io import _resolve_slit_geometry
from ._slit_report import _format_geometry_block, _format_value_lines
from ._validation import finite_real, integral
from .slit_geometry import (
    PeriodicSlitGeometry,
    minimum_image_displacements,
    wrap_positions,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

DEFAULT_DENSITY_PROBE_RADII_NM = (0.00, 0.14, 0.20)
DEFAULT_FRAMEWORK_RESNAMES = ("OM", "SI", "SL", "SLG")
GRAMS_PER_DA = 1.66053906660e-24
CUBIC_CENTIMETERS_PER_NM3 = 1.0e-21

__all__ = [
    "DEFAULT_DENSITY_PROBE_RADII_NM",
    "DEFAULT_FRAMEWORK_RESNAMES",
    "DensityProbeEstimate",
    "TargetPopulationSummary",
    "DensityEstimate",
    "SlitDensityConfig",
    "SlitDensityReport",
    "estimate_guest_density",
]


@dataclass(frozen=True)
class DensityProbeEstimate:
    """Probe-free slit-volume density summary for one probe radius.

    Parameters
    ----------
    probe_radius_nm : float
        Probe radius used for the probe-free volume estimate.
    seed_values : tuple[int, ...]
        Actual random seeds used for repeated estimates.
    probe_free_fractions : tuple[float, ...]
        Probe-free fractions of the padded geometric slit volume.
    probe_free_volumes_nm3 : tuple[float, ...]
        Probe-free slit volumes measured for each repeat.
    probe_free_densities_g_cm3 : tuple[float, ...]
        Interval-assigned target mass divided by each probe-free slit volume.
    probe_free_fraction_mean : float
        Mean probe-free fraction across repeats.
    probe_free_fraction_std : float
        Standard deviation of the probe-free fraction across repeats.
    probe_free_volume_mean_nm3 : float
        Mean probe-free volume across repeats.
    probe_free_volume_std_nm3 : float
        Standard deviation of the probe-free volume across repeats.
    probe_free_density_mean_g_cm3 : float
        Mean probe-free density across repeats.
    probe_free_density_std_g_cm3 : float
        Standard deviation of the probe-free density across repeats.
    """

    probe_radius_nm: float
    seed_values: tuple[int, ...]
    probe_free_fractions: tuple[float, ...]
    probe_free_volumes_nm3: tuple[float, ...]
    probe_free_densities_g_cm3: tuple[float, ...]
    probe_free_fraction_mean: float
    probe_free_fraction_std: float
    probe_free_volume_mean_nm3: float
    probe_free_volume_std_nm3: float
    probe_free_density_mean_g_cm3: float
    probe_free_density_std_g_cm3: float


@dataclass(frozen=True)
class TargetPopulationSummary:
    """Target-residue population assigned to the density control volumes.

    Parameters
    ----------
    total_molecule_count : int
        Number of selected target residues in the full periodic box.
    total_atom_count : int
        Number of atoms belonging to those full-box target residues.
    interval_molecule_count : int
        Number of selected target residues whose periodically reconstructed
        center of mass lies inside the signed-padded mean-plane interval.
    interval_atom_count : int
        Number of atoms belonging to the interval-assigned target residues.

    Notes
    -----
    Each molecule is assigned as a whole through its center of mass. Atom
    counts therefore describe the assigned molecular populations rather than
    the number of individual atoms geometrically inside the interval.
    """

    total_molecule_count: int
    total_atom_count: int
    interval_molecule_count: int
    interval_atom_count: int

    @property
    def outside_interval_molecule_count(self) -> int:
        """Return the number of full-box target residues outside the interval."""

        return self.total_molecule_count - self.interval_molecule_count

    @property
    def outside_interval_atom_count(self) -> int:
        """Return atoms belonging to target residues assigned outside the interval."""

        return self.total_atom_count - self.interval_atom_count


@dataclass(frozen=True)
class DensityEstimate:
    """Density metrics derived for full-box and slit-interval target populations.

    Parameters
    ----------
    guest_molecule_mass_da : float
        Mass of one target guest molecule in daltons.
    target_population : TargetPopulationSummary
        Full-box and padded-interval target population counts.
    total_guest_mass_da : float
        Full-box selected target mass in daltons.
    interval_guest_mass_da : float
        Target mass assigned to the padded slit interval in daltons.
    box_volume_nm3 : float
        Full periodic box volume in cubic nanometers.
    box_average_density_g_cm3 : float
        Target density obtained by dividing the full-box target mass by the
        full periodic box volume.
    geometric_slit_volume_nm3 : float
        Geometric mean-plane slit volume after signed surface-plane padding.
    geometric_slit_density_g_cm3 : float
        Interval target mass divided by the padded geometric slit volume.
    surface_plane_padding_nm : float
        Signed surface-plane padding used for geometric and probe-free volumes.
    sample_count_per_seed : int
        Monte Carlo sample count used for each repeated estimate.
    seed_count : int
        Number of repeated estimates used for each probe radius.
    probe_estimates : tuple[DensityProbeEstimate, ...]
        Probe-dependent probe-free slit-volume density summaries.
    """

    guest_molecule_mass_da: float
    target_population: TargetPopulationSummary
    total_guest_mass_da: float
    interval_guest_mass_da: float
    box_volume_nm3: float
    box_average_density_g_cm3: float
    geometric_slit_volume_nm3: float
    geometric_slit_density_g_cm3: float
    surface_plane_padding_nm: float
    sample_count_per_seed: int
    seed_count: int
    probe_estimates: tuple[DensityProbeEstimate, ...]


@dataclass(frozen=True)
class SlitDensityConfig:
    """Configuration for merged-slit guest-density analysis.

    Parameters
    ----------
    input_path : Path, optional
        GRO file containing the already merged slit-plus-guest structure.
    log_path : Path or None, optional
        Human-readable density report path. When omitted, the report is written
        next to ``input_path`` with the suffix ``_density.log``. The resolved
        report path must not alias the GRO or explicit geometry input.
    slit_geometry : PeriodicSlitGeometry or None, optional
        Explicit slit geometry. Supply this or ``slit_geometry_path``, but not
        both. When neither is supplied, geometry is inferred from
        hydroxylated surface silicon atoms in the framework.
    slit_geometry_path : Path or None, optional
        Explicit schema-v1 slit geometry YAML path. Neighboring YAML files are
        never discovered automatically.
    target_resname : str, optional
        Residue name used to identify guest molecules.
    framework_resnames : tuple[str, ...], optional
        Exact residue names classified as the slit framework. The default
        contains the silica residue names emitted by SilicaMS. Supplying this
        field replaces that default; functional groups must be listed
        explicitly.
    mobile_resnames : tuple[str, ...], optional
        Exact non-target residue names classified as mobile components. The
        target residue is always classified as mobile automatically.
    density_probe_radii_nm : tuple[float, ...], optional
        Probe radii used for probe-free slit-volume density estimates.
    density_sample_count : int, optional
        Monte Carlo sample count used for each density repeat.
    density_seed_count : int, optional
        Number of repeated density estimates per probe radius.
    surface_plane_padding_nm : float, optional
        Signed padding applied to both mean surface planes before sampling.
    random_seed : int or None, optional
        Optional seed used to generate deterministic Monte Carlo seeds. When
        omitted, entropy-backed random seeds are used.
    """

    input_path: Path = Path("merged_guest_slit_ring_check.gro")
    log_path: Path | None = None
    slit_geometry: PeriodicSlitGeometry | None = None
    slit_geometry_path: Path | None = None
    target_resname: str = "THY"
    framework_resnames: tuple[str, ...] = DEFAULT_FRAMEWORK_RESNAMES
    mobile_resnames: tuple[str, ...] = ()
    density_probe_radii_nm: tuple[float, ...] = DEFAULT_DENSITY_PROBE_RADII_NM
    density_sample_count: int = 200000
    density_seed_count: int = 5
    surface_plane_padding_nm: float = 0.0
    random_seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration values that do not require file inspection."""

        if not self.density_probe_radii_nm:
            raise ValueError("At least one density probe radius must be provided.")
        framework_resnames = _normalize_resname_selectors(
            "framework_resnames", self.framework_resnames
        )
        mobile_resnames = _normalize_resname_selectors(
            "mobile_resnames", self.mobile_resnames
        )
        if self.slit_geometry is not None and self.slit_geometry_path is not None:
            raise ValueError(
                "Supply either slit_geometry or slit_geometry_path, not both."
            )
        surface_plane_padding_nm = finite_real(
            "surface_plane_padding_nm",
            self.surface_plane_padding_nm,
        )
        probe_radii = tuple(
            finite_real("density probe radius", radius)
            for radius in self.density_probe_radii_nm
        )
        if any(radius < 0.0 for radius in probe_radii):
            raise ValueError("All density probe radii must be non-negative.")
        density_sample_count = integral(
            "density_sample_count", self.density_sample_count
        )
        density_seed_count = integral("density_seed_count", self.density_seed_count)
        if density_sample_count <= 0:
            raise ValueError("The density sample count must be strictly positive.")
        if density_seed_count <= 0:
            raise ValueError("The density seed count must be strictly positive.")
        random_seed = None
        if self.random_seed is not None:
            random_seed = integral("random_seed", self.random_seed)
            if random_seed < 0:
                raise ValueError("The random seed must be non-negative.")
        object.__setattr__(self, "surface_plane_padding_nm", surface_plane_padding_nm)
        object.__setattr__(self, "framework_resnames", framework_resnames)
        object.__setattr__(self, "mobile_resnames", mobile_resnames)
        object.__setattr__(self, "density_probe_radii_nm", probe_radii)
        object.__setattr__(self, "density_sample_count", density_sample_count)
        object.__setattr__(self, "density_seed_count", density_seed_count)
        object.__setattr__(self, "random_seed", random_seed)


@dataclass(frozen=True)
class SlitDensityReport:
    """Summary of density analysis for one merged slit system.

    Parameters
    ----------
    framework_atom_count : int
        Number of explicitly selected atoms treated as the slit framework.
    framework_residue_count : int
        Number of explicitly selected residues treated as the slit framework.
    framework_resnames : tuple[str, ...]
        Framework residue names present in the analyzed structure.
    mobile_resnames : tuple[str, ...]
        Mobile residue names present in the analyzed structure, including the
        target residue.
    slit_geometry : PeriodicSlitGeometry
        Slit geometry used to define the sampled mean-plane interval.
    density_estimate : DensityEstimate
        Density metrics derived for the target guest population.
    """

    framework_atom_count: int
    framework_residue_count: int
    framework_resnames: tuple[str, ...]
    mobile_resnames: tuple[str, ...]
    slit_geometry: PeriodicSlitGeometry
    density_estimate: DensityEstimate


def _normalize_resname_selectors(
    field_name: str, residue_names: Sequence[str]
) -> tuple[str, ...]:
    """Validate and normalize one exact residue-name selector sequence.

    Parameters
    ----------
    field_name : str
        Configuration field name used in validation messages.
    residue_names : sequence[str]
        Exact GRO residue names to normalize.

    Returns
    -------
    tuple[str, ...]
        Residue names in caller order with native string values.

    Raises
    ------
    TypeError
        Raised when a selector is not a string.
    ValueError
        Raised when a selector is empty, contains surrounding whitespace, or
        is repeated.
    """

    if isinstance(residue_names, str):
        raise TypeError(f"{field_name} must be a sequence of residue names.")

    normalized: list[str] = []
    seen: set[str] = set()
    for residue_name in residue_names:
        if not isinstance(residue_name, str):
            raise TypeError(f"Every {field_name} selector must be a string.")
        if not residue_name or residue_name != residue_name.strip():
            raise ValueError(
                f"Every {field_name} selector must be a non-empty residue name "
                "without surrounding whitespace."
            )
        if residue_name in seen:
            raise ValueError(
                f"Residue name {residue_name!r} is repeated in {field_name}."
            )
        seen.add(residue_name)
        normalized.append(residue_name)
    return tuple(normalized)


def _residue_counts_by_name(system: _GroSystem) -> dict[str, int]:
    """Count contiguous residues by exact residue name.

    Parameters
    ----------
    system : _GroSystem
        Loaded GRO records with contiguous residue spans.

    Returns
    -------
    dict[str, int]
        Residue counts keyed by exact residue name.
    """

    counts: dict[str, int] = {}
    for residue_span in system.residue_spans:
        counts[residue_span.residue_name] = counts.get(residue_span.residue_name, 0) + 1
    return counts


def _resolve_component_resnames(
    merged_system: _GroSystem,
    target_resname: str,
    framework_resnames: tuple[str, ...],
    mobile_resnames: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve an exhaustive, disjoint component classification.

    Parameters
    ----------
    merged_system : _GroSystem
        Merged framework-plus-mobile structure.
    target_resname : str
        Density target, which is always classified as mobile.
    framework_resnames : tuple[str, ...]
        Exact residue names selected as framework.
    mobile_resnames : tuple[str, ...]
        Exact non-target residue names selected as mobile.

    Returns
    -------
    tuple[tuple[str, ...], tuple[str, ...]]
        Sorted framework and mobile residue names that are present in the
        merged structure.

    Raises
    ------
    ValueError
        Raised when selectors overlap, a residue remains unclassified, or no
        framework residue is present.
    """

    framework = set(framework_resnames)
    mobile = set(mobile_resnames)
    mobile.add(target_resname)
    overlap = sorted(framework & mobile)
    if overlap:
        raise ValueError(
            "Residue names cannot be classified as both framework and mobile: "
            + ", ".join(overlap)
            + "."
        )

    residue_counts = _residue_counts_by_name(merged_system)
    present = set(residue_counts)
    unclassified = sorted(present - framework - mobile)
    if unclassified:
        details = ", ".join(
            f"{residue_name} ({residue_counts[residue_name]} residues)"
            for residue_name in unclassified
        )
        raise ValueError(
            f"Unclassified residue names: {details}. Classify every component "
            "with framework_resnames or mobile_resnames; target_resname is "
            "automatically mobile."
        )

    present_framework = tuple(sorted(present & framework))
    if not present_framework:
        raise ValueError(
            "The input GRO file does not contain any residues selected by "
            "framework_resnames, so no framework is available for probe-free "
            "volume estimation."
        )
    return present_framework, tuple(sorted(present & mobile))


def _paths_refer_to_same_file(first_path: Path, second_path: Path) -> bool:
    """Return whether two paths name the same prospective or existing file.

    Parameters
    ----------
    first_path : Path
        First path to compare.
    second_path : Path
        Second path to compare.

    Returns
    -------
    bool
        ``True`` for canonical path aliases, symlinks, or existing hard links.

    Notes
    -----
    Non-strict resolution detects lexical aliases even when the report does
    not exist yet. ``Path.samefile`` additionally detects existing hard links.
    """

    first_resolved = first_path.expanduser().resolve(strict=False)
    second_resolved = second_path.expanduser().resolve(strict=False)
    if first_resolved == second_resolved:
        return True
    try:
        return first_path.samefile(second_path)
    except OSError:
        return False


def _resolve_density_config(config: SlitDensityConfig) -> SlitDensityConfig:
    """Resolve the report path and reject aliases with density input files.

    Parameters
    ----------
    config : SlitDensityConfig
        User-provided density configuration.

    Returns
    -------
    SlitDensityConfig
        Configuration with ``log_path`` populated.

    Raises
    ------
    ValueError
        Raised when the report path aliases the merged GRO input or explicit
        slit-geometry input, including through a symlink or hard link.
    """

    resolved = (
        config
        if config.log_path is not None
        else replace(
            config,
            log_path=config.input_path.with_name(
                f"{config.input_path.stem}_density.log"
            ),
        )
    )
    assert resolved.log_path is not None
    protected_inputs = [("merged GRO input", resolved.input_path)]
    if resolved.slit_geometry_path is not None:
        protected_inputs.append(("slit-geometry input", resolved.slit_geometry_path))
    for input_label, input_path in protected_inputs:
        if _paths_refer_to_same_file(resolved.log_path, input_path):
            raise ValueError(
                f"Density report path {resolved.log_path} must differ from the "
                f"{input_label} path {input_path}."
            )
    return resolved


def _validate_density_config(
    config: SlitDensityConfig, merged_system: _GroSystem
) -> None:
    """Validate density settings against the loaded merged system.

    Parameters
    ----------
    config : SlitDensityConfig
        Density-analysis configuration.
    merged_system : _GroSystem
        Loaded merged slit-plus-guest system.

    Raises
    ------
    ValueError
        Raised when the target residue is missing. Component-selection
        validation is performed separately after this check.
    """

    if config.target_resname not in merged_system.residue_names:
        available_residues = sorted(set(merged_system.residue_names))
        raise ValueError(
            f"Residue name {config.target_resname!r} was not found in {config.input_path}. "
            f"Available residue names: {', '.join(available_residues)}"
        )


def _compute_target_residue_mass_da(
    guest_system: _GroSystem,
    target_resname: str,
) -> float:
    """Compute the representative mass from the first matching target residue.

    Parameters
    ----------
    guest_system : _GroSystem
        Source atom names and residue spans in file order. Target residues
        are assumed to share the first matching residue's molecular mass.
    target_resname : str
        Exact residue name identifying the guest species.

    Returns
    -------
    float
        Sum of the first target residue's element masses in daltons.

    Raises
    ------
    ValueError
        Raised when no target residue exists or an atom name cannot be mapped
        to a supported element and mass.
    """

    for residue_span in guest_system.residue_spans:
        if residue_span.residue_name != target_resname:
            continue

        return float(
            sum(
                db.get_mass(
                    _infer_element_from_atom_name(guest_system.atom_names[atom_index])
                )
                for atom_index in range(residue_span.start, residue_span.stop)
            )
        )

    raise ValueError(
        f"No residue named {target_resname!r} was found for mass estimation."
    )


def _select_target_population(
    guest_system: _GroSystem,
    guest_coordinates: FloatArray,
    slit_geometry: PeriodicSlitGeometry,
    surface_plane_padding_nm: float,
    target_resname: str,
    selected_residue_mask: BoolArray | None = None,
) -> TargetPopulationSummary:
    """Assign selected target residues to the padded slit by center of mass.

    Parameters
    ----------
    guest_system : _GroSystem
        Source atom names and contiguous residue spans.
    guest_coordinates : ndarray
        Guest coordinates in nanometers, shape ``(N, 3)``, expressed in the
        same frame as ``slit_geometry``.
    slit_geometry : PeriodicSlitGeometry
        Periodic mean-plane interval and orthorhombic box.
    surface_plane_padding_nm : float
        Signed padding applied to both mean planes in nanometers.
    target_resname : str
        Exact residue name identifying the target species.
    selected_residue_mask : ndarray or None, optional
        Boolean mask over residue spans. ``None`` selects every residue; filling
        supplies its final retained-residue mask so full-box counts describe
        only molecules written to the output.

    Returns
    -------
    TargetPopulationSummary
        Full-box and interval-assigned molecule and atom counts.

    Raises
    ------
    ValueError
        Raised when coordinate or mask shapes are inconsistent, coordinates
        are non-finite, or a target atom name has no supported elemental mass.

    Notes
    -----
    Each residue is reconstructed around its first atom with orthorhombic
    minimum-image displacements before its mass-weighted center is calculated.
    The center is then classified by the same periodic signed-padded interval
    used for geometric and probe-free volume sampling.
    """

    coordinates = np.asarray(guest_coordinates, dtype=np.float64)
    if coordinates.shape != (guest_system.atom_count, 3):
        raise ValueError("Guest coordinates must have shape (N, 3).")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Guest coordinates must be finite.")

    residue_count = len(guest_system.residue_spans)
    if selected_residue_mask is None:
        selected = np.ones(residue_count, dtype=bool)
    else:
        selected = np.asarray(selected_residue_mask)
        if selected.shape != (residue_count,) or selected.dtype.kind != "b":
            raise ValueError("Selected residue mask must be boolean with shape (R,).")

    total_molecule_count = 0
    total_atom_count = 0
    interval_molecule_count = 0
    interval_atom_count = 0
    box_lengths = np.asarray(slit_geometry.box_lengths_nm, dtype=np.float64)

    for residue_index, residue_span in enumerate(guest_system.residue_spans):
        if not selected[residue_index] or residue_span.residue_name != target_resname:
            continue

        atom_count = residue_span.stop - residue_span.start
        total_molecule_count += 1
        total_atom_count += atom_count
        residue_coordinates = coordinates[residue_span.start : residue_span.stop]
        atom_masses_da = np.array(
            [
                db.get_mass(
                    _infer_element_from_atom_name(guest_system.atom_names[index])
                )
                for index in range(residue_span.start, residue_span.stop)
            ],
            dtype=np.float64,
        )
        anchor_coordinate = residue_coordinates[0]
        whole_residue_coordinates = anchor_coordinate + minimum_image_displacements(
            anchor_coordinate,
            residue_coordinates,
            box_lengths,
        )
        center_of_mass_nm = np.average(
            whole_residue_coordinates,
            axis=0,
            weights=atom_masses_da,
        )
        if bool(
            slit_geometry.contains_positions(
                center_of_mass_nm,
                surface_plane_padding_nm,
            )
        ):
            interval_molecule_count += 1
            interval_atom_count += atom_count

    return TargetPopulationSummary(
        total_molecule_count=total_molecule_count,
        total_atom_count=total_atom_count,
        interval_molecule_count=interval_molecule_count,
        interval_atom_count=interval_atom_count,
    )


def _compute_repeated_std(values: tuple[float, ...]) -> float:
    """Compute the sample standard deviation across repeated estimates.

    Parameters
    ----------
    values : tuple[float, ...]
        Repeated scalar estimates in a common unit.

    Returns
    -------
    float
        Standard deviation with ``ddof=1`` in the input unit. Fewer than two
        values return zero, including a single non-finite value. Otherwise,
        any non-finite input produces positive infinity.
    """

    if len(values) < 2:
        return 0.0

    array = np.array(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        return float("inf")
    return float(np.std(array, ddof=1))


def _seed_values(seed_count: int, random_seed: int | None) -> tuple[int, ...]:
    """Return the repeated Monte Carlo seeds for one probe estimate.

    Parameters
    ----------
    seed_count : int
        Number of seeds required; validated by the workflow configuration.
    random_seed : int or None
        Optional seed for NumPy's default generator. ``None`` requests
        independent 63-bit draws from the system entropy source.

    Returns
    -------
    tuple[int, ...]
        Seed values used for repeated estimates, in draw order, in
        ``[0, 2**63)``.
    """

    if random_seed is None:
        return tuple(int(secrets.randbits(63)) for _ in range(seed_count))

    rng = np.random.default_rng(random_seed)
    return tuple(
        int(value) for value in rng.integers(0, 2**63, size=seed_count, dtype=np.int64)
    )


def _estimate_probe_free_volume_nm3(
    framework_system: _GroSystem,
    framework_coordinates: FloatArray,
    slit_geometry: PeriodicSlitGeometry,
    surface_plane_padding_nm: float,
    probe_radius_nm: float,
    sample_count: int,
    random_seed: int,
) -> tuple[float, float]:
    """Estimate probe-free volume inside the padded mean-plane interval.

    Parameters
    ----------
    framework_system : _GroSystem
        Framework atoms and atom names used to assign exclusion radii.
    framework_coordinates : ndarray
        Framework coordinates in nanometers, shape ``(N, 3)``. They are
        wrapped for the periodic search without changing the input array.
    slit_geometry : PeriodicSlitGeometry
        Periodic mean-plane interval and orthorhombic box.
    surface_plane_padding_nm : float
        Signed padding applied to both mean planes in nanometers.
    probe_radius_nm : float
        Radius added to every framework van der Waals radius in nanometers.
    sample_count : int
        Number of uniformly sampled points.
    random_seed : int
        Random seed for this Monte Carlo repeat.

    Returns
    -------
    tuple[float, float]
        Probe-free volume in cubic nanometers and the probe-free fraction of
        the padded geometric slit volume.

    Notes
    -----
    Sampling is confined to the signed-padded mean-plane interval and uses
    batches of 10,000 points. A point at or inside any framework atom's van
    der Waals radius plus the probe radius is excluded under minimum-image
    distances. This measures local geometric free space, not connectivity or
    reservoir accessibility.

    Raises
    ------
    ValueError
        Raised for unsupported framework atom names/radii or invalid geometry
        and padding. Workflow configurations validate sampling parameters.
    """

    box_lengths = np.asarray(slit_geometry.box_lengths_nm, dtype=np.float64)
    framework_wrapped = wrap_positions(framework_coordinates, box_lengths)
    exclusion_radii_nm = np.array(
        [
            db.get_vdw_radius(_infer_element_from_atom_name(atom_name))
            + probe_radius_nm
            for atom_name in framework_system.atom_names
        ],
        dtype=np.float64,
    )
    maximum_exclusion_radius_nm = float(np.max(exclusion_radii_nm))
    exclusion_radii_squared_nm2 = exclusion_radii_nm * exclusion_radii_nm
    framework_tree = cKDTree(framework_wrapped, boxsize=box_lengths)
    random_number_generator = np.random.default_rng(random_seed)

    probe_free_point_count = 0
    batch_size = 10000
    geometric_slit_volume_nm3 = slit_geometry.padded_geometric_volume_nm3(
        surface_plane_padding_nm
    )

    for batch_start in range(0, sample_count, batch_size):
        current_batch_size = min(batch_size, sample_count - batch_start)
        sample_points = slit_geometry.sample_uniform_positions(
            current_batch_size,
            random_number_generator,
            surface_plane_padding_nm,
        )
        neighbor_lists = framework_tree.query_ball_point(
            sample_points,
            r=maximum_exclusion_radius_nm,
            workers=-1,
        )
        neighbor_counts = np.fromiter(
            (len(neighbor_indices) for neighbor_indices in neighbor_lists),
            dtype=np.int32,
            count=current_batch_size,
        )
        total_neighbor_count = int(np.sum(neighbor_counts))
        if total_neighbor_count == 0:
            probe_free_point_count += current_batch_size
            continue

        sample_indices = np.repeat(
            np.arange(current_batch_size, dtype=np.int32), neighbor_counts
        )
        atom_indices = np.concatenate(
            [
                np.asarray(neighbor_indices, dtype=np.int32)
                for neighbor_indices in neighbor_lists
                if neighbor_indices
            ]
        )
        delta_vectors = minimum_image_displacements(
            sample_points[sample_indices],
            framework_wrapped[atom_indices],
            box_lengths,
        )
        squared_distances_nm2 = np.einsum("ij,ij->i", delta_vectors, delta_vectors)
        excluded_pairs = (
            squared_distances_nm2 <= exclusion_radii_squared_nm2[atom_indices]
        )
        excluded_points = np.zeros(current_batch_size, dtype=bool)
        if np.any(excluded_pairs):
            excluded_points[np.unique(sample_indices[excluded_pairs])] = True
        probe_free_point_count += int(np.count_nonzero(~excluded_points))

    probe_free_fraction = probe_free_point_count / sample_count
    probe_free_volume_nm3 = probe_free_fraction * geometric_slit_volume_nm3
    return probe_free_volume_nm3, probe_free_fraction


def _compute_density_estimate(
    guest_system: _GroSystem,
    framework_system: _GroSystem,
    framework_coordinates: FloatArray,
    slit_geometry: PeriodicSlitGeometry,
    surface_plane_padding_nm: float,
    target_resname: str,
    target_population: TargetPopulationSummary,
    probe_radii_nm: tuple[float, ...],
    sample_count: int,
    seed_count: int,
    random_seed: int | None,
) -> DensityEstimate:
    """Compute box-average, geometric-slit, and probe-free density estimates.

    Parameters
    ----------
    guest_system : _GroSystem
        System that defines the target guest molecular mass.
    framework_system : _GroSystem
        Framework system used for probe exclusion.
    framework_coordinates : ndarray
        Framework coordinates in nanometers, shape ``(N, 3)``, in the same
        reference frame as ``slit_geometry``.
    slit_geometry : PeriodicSlitGeometry
        Periodic mean-plane slit geometry.
    surface_plane_padding_nm : float
        Signed padding applied to both mean surface planes in nanometers.
    target_resname : str
        Residue name used to identify one guest molecule.
    target_population : TargetPopulationSummary
        Full-box and padded-interval target population counts.
    probe_radii_nm : tuple[float, ...]
        Probe radii in nanometers.
    sample_count : int
        Monte Carlo sample count per repeat.
    seed_count : int
        Number of repeats per probe radius.
    random_seed : int or None
        Optional deterministic seed source.

    Returns
    -------
    DensityEstimate
        Box-average, geometric-slit, and probe-free slit-volume density metrics.

    Warns
    -----
    UserWarning
        For each repeat with non-positive probe-free volume; that repeat's
        density is positive infinity, preserving the recorded volume/fraction.

    Notes
    -----
    Probe radii are processed in caller order. A deterministic seed source is
    incremented once per probe radius, and all resulting repeat seeds are
    recorded. Repeated densities are averaged individually, not recomputed
    from the mean volume. Neither source arrays nor the geometry are mutated.
    """

    guest_molecule_mass_da = _compute_target_residue_mass_da(
        guest_system, target_resname
    )
    total_guest_mass_da = (
        float(target_population.total_molecule_count) * guest_molecule_mass_da
    )
    interval_guest_mass_da = (
        float(target_population.interval_molecule_count) * guest_molecule_mass_da
    )
    box_volume_nm3 = float(np.prod(slit_geometry.box_lengths_nm))
    geometric_slit_volume_nm3 = slit_geometry.padded_geometric_volume_nm3(
        surface_plane_padding_nm
    )
    box_average_density_g_cm3 = (
        total_guest_mass_da
        * GRAMS_PER_DA
        / (box_volume_nm3 * CUBIC_CENTIMETERS_PER_NM3)
    )
    geometric_slit_density_g_cm3 = (
        interval_guest_mass_da
        * GRAMS_PER_DA
        / (geometric_slit_volume_nm3 * CUBIC_CENTIMETERS_PER_NM3)
    )

    probe_estimates: list[DensityProbeEstimate] = []
    deterministic_seed = random_seed
    for probe_radius_nm in probe_radii_nm:
        seed_values = _seed_values(seed_count, deterministic_seed)
        if deterministic_seed is not None:
            deterministic_seed += 1

        probe_free_volumes_nm3: list[float] = []
        probe_free_fractions: list[float] = []
        probe_free_densities_g_cm3: list[float] = []

        for seed_value in seed_values:
            probe_free_volume_nm3, probe_free_fraction = (
                _estimate_probe_free_volume_nm3(
                    framework_system=framework_system,
                    framework_coordinates=framework_coordinates,
                    slit_geometry=slit_geometry,
                    surface_plane_padding_nm=surface_plane_padding_nm,
                    probe_radius_nm=probe_radius_nm,
                    sample_count=sample_count,
                    random_seed=seed_value,
                )
            )
            probe_free_volumes_nm3.append(probe_free_volume_nm3)
            probe_free_fractions.append(probe_free_fraction)

            if probe_free_volume_nm3 <= 0.0:
                warnings.warn(
                    (
                        "Probe-free slit volume is non-positive for density probe radius "
                        f"{probe_radius_nm:.3f} nm and seed {seed_value}; reporting infinite "
                        "probe-free density for this repeat."
                    ),
                    stacklevel=2,
                )
                probe_free_densities_g_cm3.append(float("inf"))
            else:
                probe_free_densities_g_cm3.append(
                    interval_guest_mass_da
                    * GRAMS_PER_DA
                    / (probe_free_volume_nm3 * CUBIC_CENTIMETERS_PER_NM3)
                )

        probe_free_volume_values = tuple(probe_free_volumes_nm3)
        probe_free_fraction_values = tuple(probe_free_fractions)
        probe_free_density_values = tuple(probe_free_densities_g_cm3)

        volume_mean_nm3 = float(
            np.mean(np.array(probe_free_volume_values, dtype=np.float64))
        )
        fraction_mean = float(
            np.mean(np.array(probe_free_fraction_values, dtype=np.float64))
        )
        if all(np.isfinite(value) for value in probe_free_density_values):
            density_mean_g_cm3 = float(
                np.mean(np.array(probe_free_density_values, dtype=np.float64))
            )
        else:
            density_mean_g_cm3 = float("inf")

        probe_estimates.append(
            DensityProbeEstimate(
                probe_radius_nm=probe_radius_nm,
                seed_values=seed_values,
                probe_free_fractions=probe_free_fraction_values,
                probe_free_volumes_nm3=probe_free_volume_values,
                probe_free_densities_g_cm3=probe_free_density_values,
                probe_free_fraction_mean=fraction_mean,
                probe_free_fraction_std=_compute_repeated_std(
                    probe_free_fraction_values
                ),
                probe_free_volume_mean_nm3=volume_mean_nm3,
                probe_free_volume_std_nm3=_compute_repeated_std(
                    probe_free_volume_values
                ),
                probe_free_density_mean_g_cm3=density_mean_g_cm3,
                probe_free_density_std_g_cm3=_compute_repeated_std(
                    probe_free_density_values
                ),
            )
        )

    return DensityEstimate(
        guest_molecule_mass_da=guest_molecule_mass_da,
        target_population=target_population,
        total_guest_mass_da=total_guest_mass_da,
        interval_guest_mass_da=interval_guest_mass_da,
        box_volume_nm3=box_volume_nm3,
        box_average_density_g_cm3=box_average_density_g_cm3,
        geometric_slit_volume_nm3=geometric_slit_volume_nm3,
        geometric_slit_density_g_cm3=geometric_slit_density_g_cm3,
        surface_plane_padding_nm=surface_plane_padding_nm,
        sample_count_per_seed=sample_count,
        seed_count=seed_count,
        probe_estimates=tuple(probe_estimates),
    )


def _build_framework_system(
    merged_system: _GroSystem, framework_resnames: tuple[str, ...]
) -> _GroSystem:
    """Extract explicitly selected framework arrays from a merged slit system.

    Parameters
    ----------
    merged_system : _GroSystem
        Merged atom records in file order, including optional velocities.
    framework_resnames : tuple[str, ...]
        Exact residue names to include in the framework.

    Returns
    -------
    _GroSystem
        Selected framework atoms in their original order, with copied
        coordinates, identifiers, box lengths, optional velocities, and
        rebuilt contiguous residue indexing. Original atom/residue identifiers
        are retained.
    """

    selected_resnames = set(framework_resnames)
    framework_atom_mask = np.fromiter(
        (
            residue_name in selected_resnames
            for residue_name in merged_system.residue_names
        ),
        dtype=bool,
        count=merged_system.atom_count,
    )
    framework_residue_ids = merged_system.residue_ids[framework_atom_mask].copy()
    framework_residue_names = [
        residue_name
        for atom_index, residue_name in enumerate(merged_system.residue_names)
        if framework_atom_mask[atom_index]
    ]
    framework_atom_names = [
        atom_name
        for atom_index, atom_name in enumerate(merged_system.atom_names)
        if framework_atom_mask[atom_index]
    ]
    framework_atom_ids = merged_system.atom_ids[framework_atom_mask].copy()
    framework_coordinates = merged_system.coordinates[framework_atom_mask].copy()
    framework_velocities = None
    if merged_system.velocities is not None:
        framework_velocities = merged_system.velocities[framework_atom_mask].copy()

    framework_residue_spans, framework_atom_to_residue_index = _build_residue_spans(
        residue_ids=framework_residue_ids,
        residue_names=framework_residue_names,
    )
    return _GroSystem(
        title=f"{merged_system.title} [framework only]",
        residue_ids=framework_residue_ids,
        residue_names=framework_residue_names,
        atom_names=framework_atom_names,
        atom_ids=framework_atom_ids,
        coordinates=framework_coordinates,
        velocities=framework_velocities,
        box_lengths=merged_system.box_lengths.copy(),
        residue_spans=tuple(framework_residue_spans),
        atom_to_residue_index=framework_atom_to_residue_index,
    )


def _format_probe_block(probe_estimate: DensityProbeEstimate) -> str:
    """Format one human-readable probe-density block.

    Parameters
    ----------
    probe_estimate : DensityProbeEstimate
        Probe-density summary to render.

    Returns
    -------
    str
        Formatted probe block.
    """

    density_values = " ".join(
        "inf" if not np.isfinite(value) else f"{value:.6f}"
        for value in probe_estimate.probe_free_densities_g_cm3
    )
    density_mean = (
        "inf"
        if not np.isfinite(probe_estimate.probe_free_density_mean_g_cm3)
        else f"{probe_estimate.probe_free_density_mean_g_cm3:.6f}"
    )
    density_std = (
        "inf"
        if not np.isfinite(probe_estimate.probe_free_density_std_g_cm3)
        else f"{probe_estimate.probe_free_density_std_g_cm3:.6f}"
    )
    rows = [
        ("Probe radius", f"{probe_estimate.probe_radius_nm:.3f} nm"),
        (
            "Seed values",
            " ".join(str(seed_value) for seed_value in probe_estimate.seed_values),
        ),
        (
            "Probe-free fractions",
            " ".join(f"{value:.6f}" for value in probe_estimate.probe_free_fractions),
        ),
        (
            "Probe-free volumes",
            " ".join(f"{value:.6f}" for value in probe_estimate.probe_free_volumes_nm3)
            + " nm^3",
        ),
        ("Probe-free densities", density_values + " g/cm^3"),
        ("Mean fraction", f"{probe_estimate.probe_free_fraction_mean:.6f}"),
        ("Std fraction", f"{probe_estimate.probe_free_fraction_std:.6f}"),
        ("Mean volume", f"{probe_estimate.probe_free_volume_mean_nm3:.6f} nm^3"),
        ("Std volume", f"{probe_estimate.probe_free_volume_std_nm3:.6f} nm^3"),
        ("Mean density", density_mean + " g/cm^3"),
        ("Std density", density_std + " g/cm^3"),
    ]
    return _format_value_lines(f"Probe {probe_estimate.probe_radius_nm:.2f} nm", rows)


def _build_density_report_text(
    config: SlitDensityConfig, report: SlitDensityReport
) -> str:
    """Render the standalone density report without writing files.

    Parameters
    ----------
    config : SlitDensityConfig
        Resolved input/output paths, target name, and analysis settings.
    report : SlitDensityReport
        Completed counts, geometry, and repeated density estimates.

    Returns
    -------
    str
        Newline-terminated report retaining probe details before the final
        density summary, with explicit units and recorded repeat seeds.
    """

    inputs = _format_value_lines(
        "Inputs",
        [
            ("Merged input", str(config.input_path)),
            ("Report file", str(config.log_path)),
            ("Target residue", config.target_resname),
        ],
    )
    counts = _format_value_lines(
        "Target population",
        [
            ("Membership rule", "periodic center of mass inside padded interval"),
            (
                "Molecules in full box",
                str(report.density_estimate.target_population.total_molecule_count),
            ),
            (
                "Atoms in full-box molecules",
                str(report.density_estimate.target_population.total_atom_count),
            ),
            (
                "Molecules in padded interval",
                str(report.density_estimate.target_population.interval_molecule_count),
            ),
            (
                "Atoms in interval molecules",
                str(report.density_estimate.target_population.interval_atom_count),
            ),
            (
                "Molecules outside padded interval",
                str(
                    report.density_estimate.target_population.outside_interval_molecule_count
                ),
            ),
        ],
    )
    framework = _format_value_lines(
        "Framework",
        [
            ("Framework residue names", ", ".join(report.framework_resnames)),
            ("Mobile residue names", ", ".join(report.mobile_resnames)),
            ("Framework atoms", str(report.framework_atom_count)),
            ("Framework residues", str(report.framework_residue_count)),
        ],
    )
    geometry = _format_geometry_block(
        "Slit geometry",
        report.slit_geometry,
        config.surface_plane_padding_nm,
    )
    density_summary = _format_value_lines(
        "Density summary",
        [
            (
                "Guest molecule mass",
                f"{report.density_estimate.guest_molecule_mass_da:.5f} Da",
            ),
            (
                "Full-box target mass",
                f"{report.density_estimate.total_guest_mass_da:.5f} Da",
            ),
            (
                "Padded-interval target mass",
                f"{report.density_estimate.interval_guest_mass_da:.5f} Da",
            ),
            ("Box volume", f"{report.density_estimate.box_volume_nm3:.5f} nm^3"),
            (
                "Padded geometric slit volume",
                f"{report.density_estimate.geometric_slit_volume_nm3:.5f} nm^3",
            ),
            (
                "Box-average density",
                f"{report.density_estimate.box_average_density_g_cm3:.5f} g/cm^3",
            ),
            (
                "Geometric slit density",
                f"{report.density_estimate.geometric_slit_density_g_cm3:.5f} g/cm^3",
            ),
            (
                "Samples per seed",
                str(report.density_estimate.sample_count_per_seed),
            ),
            ("Seed count", str(report.density_estimate.seed_count)),
        ],
    )
    probe_details = "Probe details\n-------------\n" + "\n\n".join(
        _format_probe_block(probe_estimate)
        for probe_estimate in report.density_estimate.probe_estimates
    )

    sections = [
        "Slit Density Report\n===================",
        inputs,
        counts,
        framework,
        geometry,
        probe_details,
        density_summary,
    ]
    return "\n\n".join(sections) + "\n"


def estimate_guest_density(config: SlitDensityConfig) -> SlitDensityReport:
    """Estimate guest density inside an already merged slit system.

    Parameters
    ----------
    config : SlitDensityConfig
        Density-analysis settings and output paths.

    Returns
    -------
    SlitDensityReport
        Structured density-analysis report.

    Notes
    -----
    Framework and mobile components are selected by exact residue name. The
    target is always mobile, selector overlap is rejected, and every residue
    present in the input must be classified. Geometry is supplied explicitly,
    loaded from the explicitly named YAML file, or inferred from hydroxylated
    surface silicon; neighboring metadata is never discovered automatically.
    The text report is written to the resolved log path.

    Raises
    ------
    ValueError
        Raised for invalid input structures or geometry, overlapping or
        incomplete component selectors, a missing target population or
        framework, or unsupported atom names.
    OSError
        Raised when an input cannot be read or the report cannot be written.
    """

    config = _resolve_density_config(config)
    merged_system = _load_gro_system(config.input_path)
    _validate_density_config(config, merged_system)
    framework_resnames, mobile_resnames = _resolve_component_resnames(
        merged_system=merged_system,
        target_resname=config.target_resname,
        framework_resnames=config.framework_resnames,
        mobile_resnames=config.mobile_resnames,
    )

    framework_system = _build_framework_system(
        merged_system=merged_system,
        framework_resnames=framework_resnames,
    )
    # Density estimation wraps framework coordinates before the periodic
    # neighbor search, so nominal out-of-box coordinates are harmless here and
    # the fill-time warning only adds noise for already merged systems.
    slit_geometry = _resolve_slit_geometry(
        slit_system=framework_system,
        slit_coordinates=framework_system.coordinates,
        explicit_geometry=config.slit_geometry,
        geometry_path=config.slit_geometry_path,
        padding_nm=config.surface_plane_padding_nm,
    )
    target_population = _select_target_population(
        guest_system=merged_system,
        guest_coordinates=merged_system.coordinates,
        slit_geometry=slit_geometry,
        surface_plane_padding_nm=config.surface_plane_padding_nm,
        target_resname=config.target_resname,
    )
    density_estimate = _compute_density_estimate(
        guest_system=merged_system,
        framework_system=framework_system,
        framework_coordinates=framework_system.coordinates,
        slit_geometry=slit_geometry,
        surface_plane_padding_nm=config.surface_plane_padding_nm,
        target_resname=config.target_resname,
        target_population=target_population,
        probe_radii_nm=config.density_probe_radii_nm,
        sample_count=config.density_sample_count,
        seed_count=config.density_seed_count,
        random_seed=config.random_seed,
    )
    report = SlitDensityReport(
        framework_atom_count=framework_system.atom_count,
        framework_residue_count=len(framework_system.residue_spans),
        framework_resnames=framework_resnames,
        mobile_resnames=mobile_resnames,
        slit_geometry=slit_geometry,
        density_estimate=density_estimate,
    )
    report_text = _build_density_report_text(config, report)
    assert config.log_path is not None
    with staged_output_paths((config.log_path,)) as staging_paths:
        staging_paths[config.log_path].write_text(report_text, encoding="utf-8")
    return report


def _build_density_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for merged-slit density analysis.

    Returns
    -------
    argparse.ArgumentParser
        Parser with the density command's component selectors, numerical
        options, paths, and defaults.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Estimate target guest density inside an already merged slit structure "
            "by assigning target molecular centers to the mean-plane interval "
            "and computing its framework-excluded probe-free volume."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SlitDensityConfig.input_path,
        help="Merged slit-plus-guest GRO file. Default: %(default)s",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help=(
            "Optional report path. If omitted, the command writes "
            "<input_stem>_density.log next to the input GRO. The report must "
            "not alias the GRO or explicit geometry input."
        ),
    )
    parser.add_argument(
        "--slit-geometry",
        type=Path,
        default=None,
        help=(
            "Explicit schema-v1 slit geometry YAML file. When omitted, geometry "
            "is inferred from hydroxylated surface Si atoms; neighboring files "
            "are not discovered automatically."
        ),
    )
    parser.add_argument(
        "--target-resname",
        default=SlitDensityConfig.target_resname,
        help="Residue name used to identify guest molecules. Default: %(default)s",
    )
    parser.add_argument(
        "--framework-resname",
        action="append",
        help=(
            "Exact residue name classified as framework. Repeat for multiple "
            "names. Supplying any values replaces the default framework set "
            f"{', '.join(DEFAULT_FRAMEWORK_RESNAMES)}."
        ),
    )
    parser.add_argument(
        "--mobile-resname",
        action="append",
        help=(
            "Exact non-target residue name classified as mobile. Repeat for "
            "multiple names. The target residue is automatically mobile."
        ),
    )
    parser.add_argument(
        "--density-probe-radius",
        type=float,
        action="append",
        help=(
            "Probe radius in nm used to estimate probe-free slit volume. "
            "Repeat this option to request multiple probe radii. The default "
            "set is 0.00, 0.14, and 0.20 nm."
        ),
    )
    parser.add_argument(
        "--density-samples",
        type=int,
        default=SlitDensityConfig.density_sample_count,
        help=(
            "Number of Monte Carlo sample points used for each density repeat. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--density-seed-count",
        type=int,
        default=SlitDensityConfig.density_seed_count,
        help=(
            "Number of independent Monte Carlo repeats used for each probe "
            "radius. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--surface-plane-padding",
        type=float,
        default=SlitDensityConfig.surface_plane_padding_nm,
        help=(
            "Signed padding in nm applied on each side of the mean-plane slit "
            "interval before sampling. Positive values shrink the interval; "
            "negative values expand it. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional seed used to generate deterministic Monte Carlo seeds.",
    )
    return parser


def estimate_guest_density_main(argv: Sequence[str] | None = None) -> SlitDensityReport:
    """Parse CLI arguments and run merged-slit density analysis.

    Parameters
    ----------
    argv : sequence[str] or None, optional
        Optional argument vector. When omitted, arguments are read from the
        process command line.

    Returns
    -------
    SlitDensityReport
        Structured density-analysis report.

    Raises
    ------
    SystemExit
        Raised by the parser for help requests or invalid CLI arguments.
    ValueError
        Raised when configuration or density-analysis validation fails.
    OSError
        Raised when an input cannot be read or the report cannot be written.
    """

    parser = _build_density_argument_parser()
    args = parser.parse_args(argv)
    probe_radii = tuple(
        float(radius)
        for radius in (
            args.density_probe_radius
            if args.density_probe_radius is not None
            else DEFAULT_DENSITY_PROBE_RADII_NM
        )
    )
    config = SlitDensityConfig(
        input_path=args.input,
        log_path=args.log,
        slit_geometry_path=args.slit_geometry,
        target_resname=args.target_resname,
        framework_resnames=(
            tuple(args.framework_resname)
            if args.framework_resname is not None
            else DEFAULT_FRAMEWORK_RESNAMES
        ),
        mobile_resnames=tuple(args.mobile_resname or ()),
        density_probe_radii_nm=probe_radii,
        density_sample_count=args.density_samples,
        density_seed_count=args.density_seed_count,
        surface_plane_padding_nm=args.surface_plane_padding,
        random_seed=args.random_seed,
    )
    return estimate_guest_density(config)


def _estimate_guest_density_console_main() -> int:
    """Run the slit-density console entry point and return a process exit code.

    Returns
    -------
    int
        Successful process exit status.
    """

    estimate_guest_density_main()
    return 0
