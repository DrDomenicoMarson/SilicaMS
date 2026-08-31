################################################################################
# Slit Filling Helpers                                                         #
#                                                                              #
"""Guest filling and density analysis helpers for amorphous silica slits."""
################################################################################


from __future__ import annotations

import argparse
import secrets
import warnings

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from numpy.typing import NDArray
from scipy.spatial import cKDTree

import silicams.database as db
from silicams._gro_io import (
    BOX_LINE_TOLERANCE_NM,
    _GroSystem,
    _build_residue_spans,
    _load_gro_system,
    _write_merged_gro,
)
from silicams._output_transaction import staged_output_paths
from silicams._validation import boolean, finite_real, integral
from silicams.slit_geometry import (
    AXIS_NAMES,
    PeriodicSlitGeometry,
    minimum_image_displacements,
    wrap_positions,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]
BoolArray = NDArray[np.bool_]

DEFAULT_DENSITY_PROBE_RADII_NM = (0.00, 0.14, 0.20)
SLIT_COORDINATE_TOLERANCE_NM = 1.0e-6
GRAMS_PER_DA = 1.66053906660e-24
CUBIC_CENTIMETERS_PER_NM3 = 1.0e-21

__all__ = [
    "DEFAULT_DENSITY_PROBE_RADII_NM",
    "SlitFillConfig",
    "DensityProbeEstimate",
    "DensityEstimate",
    "GuestResidueFilterSummary",
    "SlitFillReport",
    "SlitDensityConfig",
    "SlitDensityReport",
    "fill_slit",
    "estimate_guest_density",
]


@dataclass(frozen=True)
class SlitFillConfig:
    """Configuration for the slit guest-filling workflow.

    Parameters
    ----------
    guest_path : Path, optional
        GRO file containing the larger guest reservoir box.
    slit_path : Path, optional
        GRO file containing the grafted silica slit.
    output_path : Path, optional
        GRO path written after filtering and merging the guest molecules into
        the slit cell.
    log_path : Path or None, optional
        Human-readable report path. When omitted, the report is written next to
        ``output_path`` with the suffix ``.log``.
    slit_geometry : PeriodicSlitGeometry or None, optional
        Explicit slit geometry. Supply this or ``slit_geometry_path``, but not
        both. When neither is supplied, the geometry is inferred from
        hydroxylated surface silicon atoms in ``slit_path``.
    slit_geometry_path : Path or None, optional
        Schema-v1 slit geometry YAML file. The file is read only when supplied
        explicitly; neighboring files are never discovered automatically.
    target_resname : str, optional
        Residue name used for density calculations and target-specific report
        fields. All cropped residue types are screened by the physical filling
        filters.
    general_cutoff_nm : float, optional
        Lower all-atom clash cutoff in nanometers. The ``0.10 nm`` default is
        deliberately permissive and intended to produce a starting
        configuration for staged energy minimization. ``0.15`` and ``0.20
        nm`` are useful progressively stricter values, but no value is a
        universal force-field contact criterion.
    ring_atom_prefix : str, optional
        Prefix used to identify the six aromatic ring atoms, for example
        ``"CA"`` for atom names such as ``CA1`` and ``CA6``.
    ring_plane_tolerance_nm : float, optional
        Maximum bond-to-plane distance that still counts as a ring crossing.
    ring_polygon_padding_nm : float, optional
        Additional in-plane padding applied around the aromatic ring polygon.
    include_hydrogen_bonds_in_ring_check : bool, optional
        When ``True``, bonds to hydrogen atoms also participate in the explicit
        ring-crossing checks.
    use_surface_plane_filter : bool, optional
        When ``True``, remove any cropped residue with atoms outside the detected
        slit interval.
    surface_plane_padding_nm : float, optional
        Signed padding applied on both sides of the detected slit interval.
        Positive values shrink the mean-plane interval and negative values
        expand it.
    density_probe_radii_nm : tuple[float, ...], optional
        Probe radii used for probe-free slit-volume density estimates.
    density_sample_count : int, optional
        Monte Carlo sample count used for each density repeat.
    density_seed_count : int, optional
        Number of repeated density estimates per probe radius.
    wrap_output : bool, optional
        When ``True``, wrap kept guest residues back into the final output box.
    random_seed : int or None, optional
        Optional seed used to generate deterministic Monte Carlo seeds. When
        omitted, entropy-backed random seeds are used.
    """

    guest_path: Path = Path("confout.gro")
    slit_path: Path = Path("msn_9_1.gro")
    output_path: Path = Path("merged_guest_slit_ring_check.gro")
    log_path: Path | None = None
    slit_geometry: PeriodicSlitGeometry | None = None
    slit_geometry_path: Path | None = None
    target_resname: str = "THY"
    general_cutoff_nm: float = 0.1
    ring_atom_prefix: str = "CA"
    ring_plane_tolerance_nm: float = 0.04
    ring_polygon_padding_nm: float = 0.02
    include_hydrogen_bonds_in_ring_check: bool = True
    use_surface_plane_filter: bool = True
    surface_plane_padding_nm: float = 0.0
    density_probe_radii_nm: tuple[float, ...] = DEFAULT_DENSITY_PROBE_RADII_NM
    density_sample_count: int = 200000
    density_seed_count: int = 5
    wrap_output: bool = True
    random_seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration values that do not require file inspection."""

        general_cutoff_nm = finite_real("general_cutoff_nm", self.general_cutoff_nm)
        ring_plane_tolerance_nm = finite_real(
            "ring_plane_tolerance_nm",
            self.ring_plane_tolerance_nm,
        )
        ring_polygon_padding_nm = finite_real(
            "ring_polygon_padding_nm",
            self.ring_polygon_padding_nm,
        )
        surface_plane_padding_nm = finite_real(
            "surface_plane_padding_nm",
            self.surface_plane_padding_nm,
        )
        if general_cutoff_nm <= 0.0:
            raise ValueError(
                "The general clash cutoff must be finite and strictly positive."
            )
        if ring_plane_tolerance_nm < 0.0:
            raise ValueError("The ring-plane tolerance must be non-negative.")
        if ring_polygon_padding_nm < 0.0:
            raise ValueError("The ring polygon padding must be non-negative.")
        if self.slit_geometry is not None and self.slit_geometry_path is not None:
            raise ValueError(
                "Supply either slit_geometry or slit_geometry_path, not both."
            )
        if not self.density_probe_radii_nm:
            raise ValueError("At least one density probe radius must be provided.")
        probe_radii = tuple(
            finite_real("density probe radius", radius)
            for radius in self.density_probe_radii_nm
        )
        if any(radius < 0.0 for radius in probe_radii):
            raise ValueError("All density probe radii must be non-negative.")
        density_sample_count = integral("density_sample_count", self.density_sample_count)
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
        boolean("include_hydrogen_bonds_in_ring_check", self.include_hydrogen_bonds_in_ring_check)
        boolean("use_surface_plane_filter", self.use_surface_plane_filter)
        boolean("wrap_output", self.wrap_output)
        object.__setattr__(self, "general_cutoff_nm", general_cutoff_nm)
        object.__setattr__(self, "ring_plane_tolerance_nm", ring_plane_tolerance_nm)
        object.__setattr__(self, "ring_polygon_padding_nm", ring_polygon_padding_nm)
        object.__setattr__(self, "surface_plane_padding_nm", surface_plane_padding_nm)
        object.__setattr__(self, "density_probe_radii_nm", probe_radii)
        object.__setattr__(self, "density_sample_count", density_sample_count)
        object.__setattr__(self, "density_seed_count", density_seed_count)
        object.__setattr__(self, "random_seed", random_seed)


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
        Densities derived from the probe-free slit volumes.
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
class DensityEstimate:
    """Density metrics derived for the retained guest population.

    Parameters
    ----------
    guest_molecule_mass_da : float
        Mass of one target guest molecule in daltons.
    total_guest_mass_da : float
        Total retained guest mass in daltons.
    box_volume_nm3 : float
        Full periodic box volume in cubic nanometers.
    box_average_density_g_cm3 : float
        Guest density obtained by dividing the retained guest mass by the full
        periodic box volume.
    geometric_slit_volume_nm3 : float
        Geometric mean-plane slit volume after signed surface-plane padding.
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
    total_guest_mass_da: float
    box_volume_nm3: float
    box_average_density_g_cm3: float
    geometric_slit_volume_nm3: float
    surface_plane_padding_nm: float
    sample_count_per_seed: int
    seed_count: int
    probe_estimates: tuple[DensityProbeEstimate, ...]


@dataclass(frozen=True)
class GuestResidueFilterSummary:
    """Residue-level filling outcomes for one guest residue name.

    Parameters
    ----------
    residue_name : str
        Guest residue name summarized by this record.
    initial_residues : int
        Number of matching residues in the input reservoir.
    cropped_residues : int
        Number retained by center cropping.
    removed_outside_crop_residues : int
        Number excluded by center cropping.
    surface_plane_filtered_residues : int
        Number remaining after the optional slit-plane filter.
    removed_by_surface_plane_residues : int
        Number removed by the slit-plane filter.
    removed_by_general_cutoff_residues : int
        Number intersecting the general all-atom cutoff.
    removed_by_forward_ring_residues : int
        Number whose bonds cross a slit aromatic ring.
    removed_by_reverse_ring_residues : int
        Number whose aromatic ring is crossed by a slit bond.
    removed_by_any_ring_residues : int
        Number removed by either ring-crossing direction.
    removed_by_any_clash_residues : int
        Number removed by the general or ring filters.
    removed_total_residues : int
        Total removed by cropping, plane filtering, or clash filtering.
    remaining_residues : int
        Number written to the merged output.
    """

    residue_name: str
    initial_residues: int
    cropped_residues: int
    removed_outside_crop_residues: int
    surface_plane_filtered_residues: int
    removed_by_surface_plane_residues: int
    removed_by_general_cutoff_residues: int
    removed_by_forward_ring_residues: int
    removed_by_reverse_ring_residues: int
    removed_by_any_ring_residues: int
    removed_by_any_clash_residues: int
    removed_total_residues: int
    remaining_residues: int


@dataclass(frozen=True)
class SlitFillReport:
    """Summary of one slit guest-filling workflow.

    Parameters
    ----------
    residue_filter_summaries : tuple[GuestResidueFilterSummary, ...]
        Per-residue-name filtering outcomes for every guest type present in the
        input reservoir.
    initial_guest_molecules : int
        Number of target guest residues found in the input guest box.
    cropped_guest_molecules : int
        Number of target guest residues retained after centered box cropping.
    removed_outside_crop_guest_molecules : int
        Number of target guest residues removed during centered box cropping.
    input_slit_geometry : PeriodicSlitGeometry
        Slit geometry in the input slit coordinate axes.
    output_slit_geometry : PeriodicSlitGeometry
        The same slit geometry after the output-axis permutation.
    surface_plane_filtered_guest_molecules : int
        Number of target residues still selected after the optional surface
        plane filter.
    removed_by_surface_plane_guest_molecules : int
        Number of target residues removed by the surface-plane filter.
    removed_by_general_cutoff_guest_molecules : int
        Number of target residues removed by the lower all-atom cutoff.
    removed_by_forward_ring_guest_molecules : int
        Number of target residues removed because a guest bond crosses a slit
        aromatic ring.
    removed_by_reverse_ring_guest_molecules : int
        Number of target residues removed because a slit bond crosses a guest
        aromatic ring.
    removed_by_any_ring_guest_molecules : int
        Number of target residues removed by either ring-crossing rule.
    removed_by_general_only_guest_molecules : int
        Number of target residues removed only by the lower all-atom cutoff.
    removed_by_forward_ring_only_guest_molecules : int
        Number of target residues removed only by the forward ring-crossing
        rule.
    removed_by_reverse_ring_only_guest_molecules : int
        Number of target residues removed only by the reverse ring-crossing
        rule.
    removed_by_general_and_forward_ring_only_guest_molecules : int
        Number of target residues removed by the general cutoff and the forward
        ring rule, but not by the reverse ring rule.
    removed_by_general_and_reverse_ring_only_guest_molecules : int
        Number of target residues removed by the general cutoff and the reverse
        ring rule, but not by the forward ring rule.
    removed_by_forward_and_reverse_ring_only_guest_molecules : int
        Number of target residues removed by both ring rules, but not by the
        general cutoff.
    removed_by_general_and_forward_and_reverse_ring_guest_molecules : int
        Number of target residues removed by all three clash rules.
    removed_by_clash_guest_molecules : int
        Number of target residues removed by any clash rule after crop and
        surface-plane selection.
    removed_guest_molecules : int
        Total number of removed target residues.
    remaining_guest_molecules : int
        Number of retained target residues written to the merged output.
    slit_aromatic_ring_count : int
        Number of aromatic slit rings used in the forward ring-crossing check.
    cropped_guest_ring_count : int
        Number of aromatic rings built across all selected guest residues.
    guest_bond_template_count : int
        Number of distinct guest residue bond templates used.
    guest_bond_count_checked : int
        Number of guest bond segments checked against slit aromatic rings.
    slit_bond_template_count : int
        Number of unique slit residue bond templates used in the reverse
        ring-crossing check.
    slit_bond_count_checked : int
        Number of slit bond segments checked against guest aromatic rings.
    density_estimate : DensityEstimate
        Density metrics derived for the retained guest population.
    slit_atom_count : int
        Number of atoms copied from the slit structure.
    final_atom_count : int
        Number of atoms written to the merged GRO file.
    final_residue_count : int
        Number of residues written to the merged GRO file.
    output_axis_permutation : tuple[int, int, int]
        Axis permutation used to place the detected slit normal on ``z`` in
        the output file.
    crop_window_start_nm : ndarray
        Lower corner of the centered crop window in the original guest box.
    output_box_nm : ndarray
        Orthorhombic box lengths written to the merged output.
    """

    residue_filter_summaries: tuple[GuestResidueFilterSummary, ...]
    initial_guest_molecules: int
    cropped_guest_molecules: int
    removed_outside_crop_guest_molecules: int
    input_slit_geometry: PeriodicSlitGeometry
    output_slit_geometry: PeriodicSlitGeometry
    surface_plane_filtered_guest_molecules: int
    removed_by_surface_plane_guest_molecules: int
    removed_by_general_cutoff_guest_molecules: int
    removed_by_forward_ring_guest_molecules: int
    removed_by_reverse_ring_guest_molecules: int
    removed_by_any_ring_guest_molecules: int
    removed_by_general_only_guest_molecules: int
    removed_by_forward_ring_only_guest_molecules: int
    removed_by_reverse_ring_only_guest_molecules: int
    removed_by_general_and_forward_ring_only_guest_molecules: int
    removed_by_general_and_reverse_ring_only_guest_molecules: int
    removed_by_forward_and_reverse_ring_only_guest_molecules: int
    removed_by_general_and_forward_and_reverse_ring_guest_molecules: int
    removed_by_clash_guest_molecules: int
    removed_guest_molecules: int
    remaining_guest_molecules: int
    slit_aromatic_ring_count: int
    cropped_guest_ring_count: int
    guest_bond_template_count: int
    guest_bond_count_checked: int
    slit_bond_template_count: int
    slit_bond_count_checked: int
    density_estimate: DensityEstimate
    slit_atom_count: int
    final_atom_count: int
    final_residue_count: int
    output_axis_permutation: tuple[int, int, int]
    crop_window_start_nm: FloatArray
    output_box_nm: FloatArray


@dataclass(frozen=True)
class SlitDensityConfig:
    """Configuration for merged-slit guest-density analysis.

    Parameters
    ----------
    input_path : Path, optional
        GRO file containing the already merged slit-plus-guest structure.
    log_path : Path or None, optional
        Human-readable density report path. When omitted, the report is written
        next to ``input_path`` with the suffix ``_density.log``.
    slit_geometry : PeriodicSlitGeometry or None, optional
        Explicit slit geometry. Supply this or ``slit_geometry_path``, but not
        both. When neither is supplied, geometry is inferred from
        hydroxylated surface silicon atoms in the framework.
    slit_geometry_path : Path or None, optional
        Explicit schema-v1 slit geometry YAML path. Neighboring YAML files are
        never discovered automatically.
    target_resname : str, optional
        Residue name used to identify guest molecules.
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
    density_probe_radii_nm: tuple[float, ...] = DEFAULT_DENSITY_PROBE_RADII_NM
    density_sample_count: int = 200000
    density_seed_count: int = 5
    surface_plane_padding_nm: float = 0.0
    random_seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration values that do not require file inspection."""

        if not self.density_probe_radii_nm:
            raise ValueError("At least one density probe radius must be provided.")
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
        density_sample_count = integral("density_sample_count", self.density_sample_count)
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
        object.__setattr__(self, "density_probe_radii_nm", probe_radii)
        object.__setattr__(self, "density_sample_count", density_sample_count)
        object.__setattr__(self, "density_seed_count", density_seed_count)
        object.__setattr__(self, "random_seed", random_seed)


@dataclass(frozen=True)
class SlitDensityReport:
    """Summary of density analysis for one merged slit system.

    Parameters
    ----------
    guest_molecule_count : int
        Number of target guest residues found in the merged structure.
    guest_atom_count : int
        Number of target guest atoms found in the merged structure.
    framework_atom_count : int
        Number of non-target atoms treated as the slit framework.
    framework_residue_count : int
        Number of non-target residues treated as the slit framework.
    slit_geometry : PeriodicSlitGeometry
        Slit geometry used to define the sampled mean-plane interval.
    density_estimate : DensityEstimate
        Density metrics derived for the target guest population.
    """

    guest_molecule_count: int
    guest_atom_count: int
    framework_atom_count: int
    framework_residue_count: int
    slit_geometry: PeriodicSlitGeometry
    density_estimate: DensityEstimate


@dataclass(frozen=True)
class _SurfacePlaneSelection:
    """Result of filtering guest residues against the detected slit interval."""

    selected_residue_mask: BoolArray
    removed_residue_mask: BoolArray
    slit_geometry: PeriodicSlitGeometry


@dataclass(frozen=True)
class _BondDefinition:
    """One guessed covalent bond within one residue template."""

    start_atom_index: int
    stop_atom_index: int
    start_atom_name: str
    stop_atom_name: str


@dataclass(frozen=True)
class _ResidueBondTemplate:
    """Guessed bond template for one residue topology."""

    residue_name: str
    atom_names: tuple[str, ...]
    bond_definitions: tuple[_BondDefinition, ...]


@dataclass(frozen=True)
class _RingTemplate:
    """Local atom-index template for one aromatic ring."""

    residue_name: str
    atom_prefix: str
    local_atom_indices: tuple[int, ...]
    ring_atom_names: tuple[str, ...]


@dataclass(frozen=True)
class _RingGeometry:
    """Geometric model of one aromatic ring."""

    residue_index: int
    residue_name: str
    center: FloatArray
    wrapped_center: FloatArray
    normal: FloatArray
    basis_u: FloatArray
    basis_v: FloatArray
    polygon_2d: FloatArray
    max_radius_nm: float


@dataclass(frozen=True)
class _BondSegmentGeometry:
    """Geometric representation of one residue bond segment."""

    residue_index: int
    residue_name: str
    start_atom_name: str
    stop_atom_name: str
    start_point: FloatArray
    stop_point: FloatArray
    midpoint: FloatArray
    wrapped_midpoint: FloatArray
    half_length_nm: float


@dataclass(frozen=True)
class _ClashSelection:
    """Residue-level clash results for the general and ring filters."""

    removed_residue_mask: BoolArray
    removed_by_general_mask: BoolArray
    removed_by_forward_ring_mask: BoolArray
    removed_by_reverse_ring_mask: BoolArray
    removed_by_any_ring_mask: BoolArray


@dataclass(frozen=True)
class _RingCheckCache:
    """Cached geometry data reused for reporting after clash detection."""

    slit_ring_geometries: tuple[_RingGeometry, ...]
    guest_bond_templates: tuple[_ResidueBondTemplate, ...]
    guest_bond_geometries: tuple[_BondSegmentGeometry, ...]
    guest_ring_geometries: tuple[_RingGeometry, ...]
    slit_bond_templates: tuple[_ResidueBondTemplate, ...]
    slit_bond_geometries: tuple[_BondSegmentGeometry, ...]


def _resolve_fill_config(config: SlitFillConfig) -> SlitFillConfig:
    """Return a fill configuration with a resolved log-file path.

    Parameters
    ----------
    config : SlitFillConfig
        User-provided slit-fill configuration.

    Returns
    -------
    SlitFillConfig
        Configuration with ``log_path`` populated.

    Raises
    ------
    ValueError
        Raised when the GRO, geometry YAML, and log destinations are not
        distinct.
    """

    resolved = (
        config
        if config.log_path is not None
        else replace(config, log_path=config.output_path.with_suffix(".log"))
    )
    output_paths = (
        resolved.output_path,
        resolved.output_path.with_suffix(".yml"),
        resolved.log_path,
    )
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("Fill GRO, geometry YAML, and log paths must be distinct.")
    return resolved


def _resolve_density_config(config: SlitDensityConfig) -> SlitDensityConfig:
    """Return a density configuration with a resolved log-file path.

    Parameters
    ----------
    config : SlitDensityConfig
        User-provided density configuration.

    Returns
    -------
    SlitDensityConfig
        Configuration with ``log_path`` populated.
    """

    if config.log_path is not None:
        return config
    return replace(
        config,
        log_path=config.input_path.with_name(f"{config.input_path.stem}_density.log"),
    )


def _validate_slit_coordinates(
    slit_system: _GroSystem,
    tolerance_nm: float = SLIT_COORDINATE_TOLERANCE_NM,
) -> None:
    """Warn when slit coordinates lie noticeably outside their stated box.

    Parameters
    ----------
    slit_system : _GroSystem
        Loaded slit system.
    tolerance_nm : float, optional
        Allowed coordinate tolerance in nanometers.
    """

    below_box = (-tolerance_nm) - slit_system.coordinates
    above_box = slit_system.coordinates - (slit_system.box_lengths + tolerance_nm)
    out_of_range_mask = np.any((below_box > 0.0) | (above_box > 0.0), axis=1)
    if not np.any(out_of_range_mask):
        return

    worst_below_nm = float(np.max(np.maximum(below_box, 0.0)))
    worst_above_nm = float(np.max(np.maximum(above_box, 0.0)))
    worst_offset_nm = max(worst_below_nm, worst_above_nm)
    warnings.warn(
        (
            f"{np.count_nonzero(out_of_range_mask)} slit atoms fall outside the nominal slit "
            f"box range [-{tolerance_nm:.1e}, box + {tolerance_nm:.1e}]; worst offset = "
            f"{worst_offset_nm:.6f} nm."
        ),
        stacklevel=2,
    )


def _validate_fill_config(
    config: SlitFillConfig,
    guest_system: _GroSystem,
    slit_system: _GroSystem,
) -> None:
    """Validate fill settings against the loaded systems.

    Parameters
    ----------
    config : SlitFillConfig
        Slit-fill configuration.
    guest_system : _GroSystem
        Loaded guest reservoir system.
    slit_system : _GroSystem
        Loaded slit system.

    Raises
    ------
    ValueError
        Raised when the target residue is missing or when the slit box cannot
        be cropped from the guest box.
    """

    if config.target_resname not in guest_system.residue_names:
        available_residues = sorted(set(guest_system.residue_names))
        raise ValueError(
            f"Residue name {config.target_resname!r} was not found in {config.guest_path}. "
            f"Available residue names: {', '.join(available_residues)}"
        )

    if np.any(slit_system.box_lengths > guest_system.box_lengths):
        raise ValueError(
            "The slit box is larger than the guest box in at least one dimension, "
            "so the guest reservoir cannot be center-cropped into the slit cell."
        )


def _validate_density_config(config: SlitDensityConfig, merged_system: _GroSystem) -> None:
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
        Raised when the target residue is missing or when the input file does
        not contain any framework atoms.
    """

    if config.target_resname not in merged_system.residue_names:
        available_residues = sorted(set(merged_system.residue_names))
        raise ValueError(
            f"Residue name {config.target_resname!r} was not found in {config.input_path}. "
            f"Available residue names: {', '.join(available_residues)}"
        )
    if all(residue_name == config.target_resname for residue_name in merged_system.residue_names):
        raise ValueError(
            "The input GRO file does not contain any non-target atoms, so no slit "
            "framework is available for probe-free volume estimation."
        )


def _wrap_residues(
    system: _GroSystem,
    coordinates: FloatArray,
    keep_atom_mask: BoolArray,
    box_lengths: FloatArray,
) -> FloatArray:
    """Wrap kept residues into the final box by whole-residue image shifts.

    Parameters
    ----------
    system : _GroSystem
        Source system that provides residue spans.
    coordinates : ndarray
        Coordinates to wrap.
    keep_atom_mask : ndarray
        Boolean atom mask marking atoms kept in the final output.
    box_lengths : ndarray
        Orthorhombic box lengths in nanometers.

    Returns
    -------
    ndarray
        Wrapped coordinates with each kept residue shifted by a whole-box image
        so that its center of geometry lies in the output box.
    """

    wrapped_coordinates = coordinates.copy()
    for residue_span in system.residue_spans:
        if not np.all(keep_atom_mask[residue_span.start:residue_span.stop]):
            continue

        residue_coordinates = wrapped_coordinates[residue_span.start:residue_span.stop]
        center_of_geometry = np.mean(residue_coordinates, axis=0)
        image_shift = box_lengths * np.floor(center_of_geometry / box_lengths)
        wrapped_coordinates[residue_span.start:residue_span.stop] = residue_coordinates - image_shift

    return wrapped_coordinates


def _infer_element_from_atom_name(atom_name: str) -> str:
    """Infer a chemical element symbol from one GRO atom name.

    Parameters
    ----------
    atom_name : str
        Atom name read from the GRO file.

    Returns
    -------
    str
        Normalized chemical element symbol.

    Raises
    ------
    ValueError
        Raised when the atom name cannot be mapped to a supported element.
    """

    try:
        return db.get_pdb_element(atom_name)
    except ValueError as error:
        raise ValueError(
            f"Unsupported element inferred from atom name {atom_name!r}."
        ) from error


def _find_hydroxylated_surface_silicon_indices(slit_system: _GroSystem) -> IntArray:
    """Return slit Si atoms that belong to hydroxylated surface residues.

    Parameters
    ----------
    slit_system : _GroSystem
        Loaded slit system.

    Returns
    -------
    ndarray
        Integer atom indices of hydroxylated surface silicon atoms.

    Raises
    ------
    ValueError
        Raised when no hydroxylated surface silicon atoms can be identified.
    """

    silicon_indices: list[int] = []
    for residue_span in slit_system.residue_spans:
        residue_elements = tuple(
            _infer_element_from_atom_name(atom_name)
            for atom_name in slit_system.atom_names[residue_span.start:residue_span.stop]
        )
        has_oxygen = any(element == "O" for element in residue_elements)
        has_hydrogen = any(element == "H" for element in residue_elements)
        if not has_oxygen or not has_hydrogen:
            continue

        for local_atom_index, element in enumerate(residue_elements):
            if element == "Si":
                silicon_indices.append(residue_span.start + local_atom_index)

    if not silicon_indices:
        raise ValueError(
            "Could not identify any hydroxylated surface Si atoms in the slit "
            "structure, so the slit interval cannot be inferred."
        )

    return np.array(silicon_indices, dtype=np.int32)


def _infer_slit_geometry(
    slit_system: _GroSystem,
    slit_coordinates: FloatArray,
    box_lengths: FloatArray,
) -> PeriodicSlitGeometry:
    """Infer periodic mean slit planes and the lower-occupancy framework arc.

    Parameters
    ----------
    slit_system : _GroSystem
        Loaded slit system.
    slit_coordinates : ndarray
        Slit coordinates in the final slit reference frame.
    box_lengths : ndarray
        Orthorhombic box lengths in nanometers.
    Returns
    -------
    PeriodicSlitGeometry
        Inferred orthorhombic slit geometry.

    Notes
    -----
    Two periodic planes define complementary arcs. After fitting the planes,
    this function chooses the arc containing fewer framework atom centers. A
    tie retains the largest surface-position gap selected by the fitter.

    Raises
    ------
    ValueError
        Raised when too few surface Si atoms are available or their positions
        cannot be separated into two faces.
    """

    surface_silicon_indices = _find_hydroxylated_surface_silicon_indices(slit_system)
    if surface_silicon_indices.size < 2:
        raise ValueError(
            "At least two hydroxylated surface Si atoms are required to infer "
            "the slit planes."
        )

    candidate_geometry = PeriodicSlitGeometry.from_surface_positions(
        box_lengths_nm=box_lengths,
        surface_positions_nm=slit_coordinates[surface_silicon_indices],
    )
    complementary_geometry = PeriodicSlitGeometry(
        box_lengths_nm=candidate_geometry.box_lengths_nm,
        normal_axis_index=candidate_geometry.normal_axis_index,
        lower_plane_nm=candidate_geometry.upper_plane_nm,
        upper_plane_nm=candidate_geometry.lower_plane_nm,
        surface_support_count=candidate_geometry.surface_support_count,
        normal_roughness_rms_nm=candidate_geometry.normal_roughness_rms_nm,
    )
    candidate_framework_count = int(
        np.count_nonzero(candidate_geometry.contains_positions(slit_coordinates))
    )
    complementary_framework_count = int(
        np.count_nonzero(complementary_geometry.contains_positions(slit_coordinates))
    )
    if complementary_framework_count < candidate_framework_count:
        return complementary_geometry
    return candidate_geometry


def _load_slit_geometry(path: Path) -> PeriodicSlitGeometry:
    """Load one schema-v1 periodic slit geometry YAML file.

    Parameters
    ----------
    path : Path
        Explicit YAML file path.

    Returns
    -------
    PeriodicSlitGeometry
        Parsed and validated slit geometry.

    Raises
    ------
    ValueError
        Raised when the YAML root, schema version, or geometry mapping is
        invalid.
    """

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    if payload.get("schema_version") != 1:
        raise ValueError(f"{path} must declare schema_version: 1.")
    geometry_payload = payload.get("slit_geometry")
    if not isinstance(geometry_payload, dict):
        raise ValueError(f"{path} must contain a slit_geometry mapping.")
    try:
        return PeriodicSlitGeometry.from_dict(geometry_payload)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid slit geometry in {path}: {error}") from error


def _validate_geometry_box(
    slit_geometry: PeriodicSlitGeometry,
    box_lengths: FloatArray,
) -> None:
    """Validate that geometry and GRO orthorhombic box lengths agree.

    Parameters
    ----------
    slit_geometry : PeriodicSlitGeometry
        Geometry supplied for the GRO structure.
    box_lengths : ndarray
        GRO box lengths in nanometers.

    Raises
    ------
    ValueError
        Raised when the box lengths differ beyond the GRO precision tolerance.
    """

    if not np.allclose(
        np.asarray(slit_geometry.box_lengths_nm),
        box_lengths,
        rtol=0.0,
        atol=BOX_LINE_TOLERANCE_NM,
    ):
        raise ValueError(
            "The slit geometry box lengths do not match the GRO box lengths: "
            f"geometry={slit_geometry.box_lengths_nm}, GRO={tuple(box_lengths)}."
        )


def _resolve_slit_geometry(
    slit_system: _GroSystem,
    slit_coordinates: FloatArray,
    explicit_geometry: PeriodicSlitGeometry | None,
    geometry_path: Path | None,
    padding_nm: float,
) -> PeriodicSlitGeometry:
    """Resolve explicit, file-backed, or inferred periodic slit geometry.

    Parameters
    ----------
    slit_system : _GroSystem
        Framework system used when inference is required.
    slit_coordinates : ndarray
        Framework coordinates in the same frame as ``slit_system.box_lengths``.
    explicit_geometry : PeriodicSlitGeometry or None
        Geometry supplied directly through the Python API.
    geometry_path : Path or None
        Explicit schema-v1 geometry YAML file.
    padding_nm : float
        Signed plane padding to validate against the resolved geometry.

    Returns
    -------
    PeriodicSlitGeometry
        Resolved and validated geometry.
    """

    if explicit_geometry is not None:
        slit_geometry = explicit_geometry
    elif geometry_path is not None:
        slit_geometry = _load_slit_geometry(geometry_path)
    else:
        slit_geometry = _infer_slit_geometry(
            slit_system=slit_system,
            slit_coordinates=slit_coordinates,
            box_lengths=slit_system.box_lengths,
        )
    _validate_geometry_box(slit_geometry, slit_system.box_lengths)
    slit_geometry.padded_width_nm(padding_nm)
    return slit_geometry


def _write_slit_geometry_metadata(
    path: Path,
    slit_geometry: PeriodicSlitGeometry,
    padding_nm: float,
) -> None:
    """Write schema-v1 output geometry and analysis metadata.

    Parameters
    ----------
    path : Path
        YAML output path.
    slit_geometry : PeriodicSlitGeometry
        Geometry expressed in the output coordinate frame.
    padding_nm : float
        Signed plane padding used by filtering and density analysis.
    """

    payload = {
        "schema_version": 1,
        "slit_geometry": slit_geometry.to_dict(),
        "slit_analysis": {
            "surface_plane_padding_nm": padding_nm,
            "padded_width_nm": slit_geometry.padded_width_nm(padding_nm),
            "padded_geometric_volume_nm3": slit_geometry.padded_geometric_volume_nm3(
                padding_nm
            ),
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _build_output_axis_permutation(normal_axis_index: int) -> tuple[int, int, int]:
    """Build the axis permutation that moves the slit normal onto ``z``."""

    if normal_axis_index == 2:
        return (0, 1, 2)
    if normal_axis_index == 1:
        return (0, 2, 1)
    if normal_axis_index == 0:
        return (2, 1, 0)
    raise ValueError(f"Unsupported axis index {normal_axis_index}.")


def _permute_vector_axes(
    vectors: FloatArray,
    axis_permutation: tuple[int, int, int],
) -> FloatArray:
    """Return Cartesian vectors expressed in the permuted output frame.

    Parameters
    ----------
    vectors : ndarray
        Positions in nanometers or velocities in nm/ps, shape ``(N, 3)``.
    axis_permutation : tuple[int, int, int]
        Input-axis indices in output-axis order.

    Returns
    -------
    ndarray
        Independent array with permuted components and unchanged units. No
        translation, wrapping, or mutation of the input is performed.
    """

    return vectors[:, axis_permutation].copy()


def _permute_box_axes(
    box_lengths: FloatArray,
    axis_permutation: tuple[int, int, int],
) -> FloatArray:
    """Permute orthorhombic box lengths consistently with rotated coordinates."""

    return box_lengths[np.array(axis_permutation, dtype=np.int32)].copy()


def _apply_surface_plane_filter(
    guest_system: _GroSystem,
    translated_guest_coordinates: FloatArray,
    selected_residue_mask: BoolArray,
    slit_geometry: PeriodicSlitGeometry,
    padding_nm: float,
) -> _SurfacePlaneSelection:
    """Remove selected residues that fall outside the detected slit planes.

    Parameters
    ----------
    guest_system : _GroSystem
        Loaded guest system.
    translated_guest_coordinates : ndarray
        Guest coordinates already translated into the slit reference frame.
    selected_residue_mask : ndarray
        Residue mask after center-cropping.
    slit_geometry : PeriodicSlitGeometry
        Periodic mean-plane slit geometry.
    padding_nm : float
        Signed padding applied to both mean surface planes.

    Returns
    -------
    _SurfacePlaneSelection
        Updated residue selection and the surface-plane removal mask.
    """

    filtered_residue_mask = selected_residue_mask.copy()
    removed_residue_mask = np.zeros(len(guest_system.residue_spans), dtype=bool)

    for residue_index, residue_span in enumerate(guest_system.residue_spans):
        if not selected_residue_mask[residue_index]:
            continue

        residue_coordinates = translated_guest_coordinates[
            residue_span.start:residue_span.stop
        ]
        if np.all(slit_geometry.contains_positions(residue_coordinates, padding_nm)):
            continue

        filtered_residue_mask[residue_index] = False
        removed_residue_mask[residue_index] = True

    return _SurfacePlaneSelection(
        selected_residue_mask=filtered_residue_mask,
        removed_residue_mask=removed_residue_mask,
        slit_geometry=slit_geometry,
    )


def _unwrap_residue_coordinates(
    residue_coordinates: FloatArray,
    box_lengths: FloatArray,
) -> FloatArray:
    """Unwrap one residue relative to its first atom under orthorhombic PBC."""

    reference = residue_coordinates[0]
    return reference + minimum_image_displacements(
        reference,
        residue_coordinates,
        box_lengths,
    )


def _guess_bond_definitions(
    atom_names: tuple[str, ...],
    residue_coordinates: FloatArray,
    include_hydrogen_bonds: bool,
) -> tuple[_BondDefinition, ...]:
    """Guess covalent bonds for one residue from atom names and coordinates."""

    bond_definitions: list[_BondDefinition] = []
    element_symbols = [_infer_element_from_atom_name(atom_name) for atom_name in atom_names]
    atom_count = len(atom_names)
    scale_factor = 1.25

    for start_atom_index in range(atom_count):
        for stop_atom_index in range(start_atom_index + 1, atom_count):
            start_element = element_symbols[start_atom_index]
            stop_element = element_symbols[stop_atom_index]
            if not include_hydrogen_bonds and (start_element == "H" or stop_element == "H"):
                continue

            distance_nm = float(
                np.linalg.norm(
                    residue_coordinates[stop_atom_index] - residue_coordinates[start_atom_index]
                )
            )
            cutoff_nm = scale_factor * (
                db.get_covalent_radius(start_element) + db.get_covalent_radius(stop_element)
            )
            if 0.05 < distance_nm <= cutoff_nm:
                bond_definitions.append(
                    _BondDefinition(
                        start_atom_index=start_atom_index,
                        stop_atom_index=stop_atom_index,
                        start_atom_name=atom_names[start_atom_index],
                        stop_atom_name=atom_names[stop_atom_index],
                    )
                )

    return tuple(bond_definitions)


def _build_ring_template_from_atom_names(
    residue_name: str,
    atom_names: tuple[str, ...],
    ring_atom_prefix: str,
) -> _RingTemplate | None:
    """Build an aromatic-ring template from one residue atom-name list."""

    local_atom_indices = tuple(
        atom_index
        for atom_index, atom_name in enumerate(atom_names)
        if atom_name.startswith(ring_atom_prefix)
    )
    if len(local_atom_indices) != 6:
        return None

    return _RingTemplate(
        residue_name=residue_name,
        atom_prefix=ring_atom_prefix,
        local_atom_indices=local_atom_indices,
        ring_atom_names=tuple(atom_names[atom_index] for atom_index in local_atom_indices),
    )


def _compute_target_residue_mass_da(
    guest_system: _GroSystem,
    target_resname: str,
) -> float:
    """Compute the mass of one target residue from its atom names."""

    for residue_span in guest_system.residue_spans:
        if residue_span.residue_name != target_resname:
            continue

        return float(
            sum(
                db.get_mass(_infer_element_from_atom_name(guest_system.atom_names[atom_index]))
                for atom_index in range(residue_span.start, residue_span.stop)
            )
        )

    raise ValueError(f"No residue named {target_resname!r} was found for mass estimation.")


def _compute_repeated_std(values: tuple[float, ...]) -> float:
    """Compute the standard deviation across repeated estimates."""

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
        Number of seeds required.
    random_seed : int or None
        Optional deterministic seed source.

    Returns
    -------
    tuple[int, ...]
        Seed values used for repeated estimates.
    """

    if random_seed is None:
        return tuple(int(secrets.randbits(63)) for _ in range(seed_count))

    rng = np.random.default_rng(random_seed)
    return tuple(int(value) for value in rng.integers(0, 2**63, size=seed_count, dtype=np.int64))


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
        Framework coordinates in nanometers.
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
    """

    box_lengths = np.asarray(slit_geometry.box_lengths_nm, dtype=np.float64)
    framework_wrapped = wrap_positions(framework_coordinates, box_lengths)
    exclusion_radii_nm = np.array(
        [
            db.get_vdw_radius(_infer_element_from_atom_name(atom_name)) + probe_radius_nm
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

        sample_indices = np.repeat(np.arange(current_batch_size, dtype=np.int32), neighbor_counts)
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
        excluded_pairs = squared_distances_nm2 <= exclusion_radii_squared_nm2[atom_indices]
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
    remaining_guest_molecules: int,
    probe_radii_nm: tuple[float, ...],
    sample_count: int,
    seed_count: int,
    random_seed: int | None,
) -> DensityEstimate:
    """Compute box-average and probe-free slit guest-density estimates.

    Parameters
    ----------
    guest_system : _GroSystem
        System that defines the target guest molecular mass.
    framework_system : _GroSystem
        Framework system used for probe exclusion.
    framework_coordinates : ndarray
        Framework coordinates in nanometers.
    slit_geometry : PeriodicSlitGeometry
        Periodic mean-plane slit geometry.
    surface_plane_padding_nm : float
        Signed padding applied to both mean surface planes in nanometers.
    target_resname : str
        Residue name used to identify one guest molecule.
    remaining_guest_molecules : int
        Number of guest molecules included in the mass.
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
        Box-average and probe-free slit-volume density metrics.
    """

    guest_molecule_mass_da = _compute_target_residue_mass_da(guest_system, target_resname)
    total_guest_mass_da = float(remaining_guest_molecules) * guest_molecule_mass_da
    box_volume_nm3 = float(np.prod(slit_geometry.box_lengths_nm))
    geometric_slit_volume_nm3 = slit_geometry.padded_geometric_volume_nm3(
        surface_plane_padding_nm
    )
    box_average_density_g_cm3 = (
        total_guest_mass_da * GRAMS_PER_DA / (box_volume_nm3 * CUBIC_CENTIMETERS_PER_NM3)
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
            probe_free_volume_nm3, probe_free_fraction = _estimate_probe_free_volume_nm3(
                framework_system=framework_system,
                framework_coordinates=framework_coordinates,
                slit_geometry=slit_geometry,
                surface_plane_padding_nm=surface_plane_padding_nm,
                probe_radius_nm=probe_radius_nm,
                sample_count=sample_count,
                random_seed=seed_value,
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
                    total_guest_mass_da
                    * GRAMS_PER_DA
                    / (probe_free_volume_nm3 * CUBIC_CENTIMETERS_PER_NM3)
                )

        probe_free_volume_values = tuple(probe_free_volumes_nm3)
        probe_free_fraction_values = tuple(probe_free_fractions)
        probe_free_density_values = tuple(probe_free_densities_g_cm3)

        volume_mean_nm3 = float(np.mean(np.array(probe_free_volume_values, dtype=np.float64)))
        fraction_mean = float(np.mean(np.array(probe_free_fraction_values, dtype=np.float64)))
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
                probe_free_fraction_std=_compute_repeated_std(probe_free_fraction_values),
                probe_free_volume_mean_nm3=volume_mean_nm3,
                probe_free_volume_std_nm3=_compute_repeated_std(probe_free_volume_values),
                probe_free_density_mean_g_cm3=density_mean_g_cm3,
                probe_free_density_std_g_cm3=_compute_repeated_std(probe_free_density_values),
            )
        )

    return DensityEstimate(
        guest_molecule_mass_da=guest_molecule_mass_da,
        total_guest_mass_da=total_guest_mass_da,
        box_volume_nm3=box_volume_nm3,
        box_average_density_g_cm3=box_average_density_g_cm3,
        geometric_slit_volume_nm3=geometric_slit_volume_nm3,
        surface_plane_padding_nm=surface_plane_padding_nm,
        sample_count_per_seed=sample_count,
        seed_count=seed_count,
        probe_estimates=tuple(probe_estimates),
    )


def _build_ring_geometry(
    residue_index: int,
    residue_name: str,
    residue_coordinates: FloatArray,
    ring_template: _RingTemplate,
    final_box_lengths: FloatArray,
) -> _RingGeometry:
    """Build one aromatic-ring geometry from residue coordinates and a template."""

    ring_coordinates = _unwrap_residue_coordinates(
        residue_coordinates[np.array(ring_template.local_atom_indices, dtype=np.int32)],
        final_box_lengths,
    )
    ring_center = np.mean(ring_coordinates, axis=0)
    wrapped_center = wrap_positions(ring_center[np.newaxis, :], final_box_lengths)[0]
    image_shift = ring_center - wrapped_center
    ring_coordinates = ring_coordinates - image_shift
    ring_center = ring_center - image_shift

    centered_coordinates = ring_coordinates - ring_center
    _, _, right_singular_vectors = np.linalg.svd(centered_coordinates, full_matrices=False)
    basis_u = right_singular_vectors[0] / np.linalg.norm(right_singular_vectors[0])
    basis_v = right_singular_vectors[1] / np.linalg.norm(right_singular_vectors[1])
    normal = right_singular_vectors[2] / np.linalg.norm(right_singular_vectors[2])

    ring_coordinates_2d = np.column_stack((centered_coordinates @ basis_u, centered_coordinates @ basis_v))
    angles = np.arctan2(ring_coordinates_2d[:, 1], ring_coordinates_2d[:, 0])
    order = np.argsort(angles)
    polygon_2d = ring_coordinates_2d[order]
    max_radius_nm = float(np.max(np.linalg.norm(polygon_2d, axis=1)))

    return _RingGeometry(
        residue_index=residue_index,
        residue_name=residue_name,
        center=ring_center,
        wrapped_center=wrapped_center,
        normal=normal,
        basis_u=basis_u,
        basis_v=basis_v,
        polygon_2d=polygon_2d,
        max_radius_nm=max_radius_nm,
    )


def _build_slit_ring_geometries(
    slit_system: _GroSystem,
    slit_coordinates: FloatArray,
    final_box_lengths: FloatArray,
    ring_atom_prefix: str,
) -> tuple[_RingGeometry, ...]:
    """Build aromatic-ring geometries for slit residues that contain one ring."""

    ring_geometries: list[_RingGeometry] = []
    for residue_index, residue_span in enumerate(slit_system.residue_spans):
        atom_names = tuple(slit_system.atom_names[residue_span.start:residue_span.stop])
        ring_template = _build_ring_template_from_atom_names(
            residue_name=residue_span.residue_name,
            atom_names=atom_names,
            ring_atom_prefix=ring_atom_prefix,
        )
        if ring_template is None:
            continue

        residue_coordinates = slit_coordinates[residue_span.start:residue_span.stop]
        ring_geometries.append(
            _build_ring_geometry(
                residue_index=residue_index,
                residue_name=residue_span.residue_name,
                residue_coordinates=residue_coordinates,
                ring_template=ring_template,
                final_box_lengths=final_box_lengths,
            )
        )

    return tuple(ring_geometries)


def _build_guest_filter_geometries(
    guest_system: _GroSystem,
    translated_guest_coordinates: FloatArray,
    selected_residue_mask: BoolArray,
    final_box_lengths: FloatArray,
    ring_atom_prefix: str,
    include_hydrogen_bonds: bool,
) -> tuple[
    tuple[_ResidueBondTemplate, ...],
    tuple[_BondSegmentGeometry, ...],
    tuple[_RingGeometry, ...],
]:
    """Build cached bond and optional ring geometry for all selected guests.

    Parameters
    ----------
    guest_system : _GroSystem
        Loaded guest reservoir.
    translated_guest_coordinates : ndarray
        Guest coordinates translated into the slit reference frame.
    selected_residue_mask : ndarray
        Residues retained by cropping and optional plane filtering.
    final_box_lengths : ndarray
        Periodic slit box lengths in nanometers.
    ring_atom_prefix : str
        Prefix identifying the accepted six aromatic ring atoms.
    include_hydrogen_bonds : bool
        Whether guessed bonds to hydrogen participate in forward ring checks.

    Returns
    -------
    bond_templates : tuple[_ResidueBondTemplate, ...]
        Unique bond templates keyed by residue name and atom-name signature.
    bond_geometries : tuple[_BondSegmentGeometry, ...]
        Actual bond segments for all selected multi-atom residues.
    ring_geometries : tuple[_RingGeometry, ...]
        Aromatic ring geometries for every selected ring-bearing residue.

    Raises
    ------
    ValueError
        Raised when a selected multi-atom residue has no safely inferred
        covalent bonds. A residue whose only bonds are explicitly excluded
        hydrogen bonds is accepted with no forward-ring segments.
    """

    template_cache: dict[tuple[str, tuple[str, ...]], _ResidueBondTemplate] = {}
    bond_geometries: list[_BondSegmentGeometry] = []
    ring_geometries: list[_RingGeometry] = []
    for residue_index, residue_span in enumerate(guest_system.residue_spans):
        if not selected_residue_mask[residue_index]:
            continue

        residue_coordinates = translated_guest_coordinates[residue_span.start:residue_span.stop]
        atom_names = tuple(guest_system.atom_names[residue_span.start:residue_span.stop])
        template_key = (residue_span.residue_name, atom_names)
        bond_template = template_cache.get(template_key)
        if bond_template is None:
            bond_definitions = _guess_bond_definitions(
                atom_names=atom_names,
                residue_coordinates=residue_coordinates,
                include_hydrogen_bonds=include_hydrogen_bonds,
            )
            if len(atom_names) > 1 and not bond_definitions:
                all_bond_definitions = _guess_bond_definitions(
                    atom_names=atom_names,
                    residue_coordinates=residue_coordinates,
                    include_hydrogen_bonds=True,
                )
                if not all_bond_definitions:
                    raise ValueError(
                        "No covalent bonds were guessed for selected multi-atom "
                        f"guest residue {residue_span.residue_name!r} with atom "
                        f"signature {atom_names!r}."
                    )
            bond_template = _ResidueBondTemplate(
                residue_name=residue_span.residue_name,
                atom_names=atom_names,
                bond_definitions=bond_definitions,
            )
            template_cache[template_key] = bond_template

        for bond_definition in bond_template.bond_definitions:
            start_point = residue_coordinates[bond_definition.start_atom_index]
            stop_point = residue_coordinates[bond_definition.stop_atom_index]
            midpoint = 0.5 * (start_point + stop_point)
            bond_geometries.append(
                _BondSegmentGeometry(
                    residue_index=residue_index,
                    residue_name=residue_span.residue_name,
                    start_atom_name=bond_definition.start_atom_name,
                    stop_atom_name=bond_definition.stop_atom_name,
                    start_point=start_point,
                    stop_point=stop_point,
                    midpoint=midpoint,
                    wrapped_midpoint=wrap_positions(
                        midpoint[np.newaxis, :],
                        final_box_lengths,
                    )[0],
                    half_length_nm=0.5 * float(np.linalg.norm(stop_point - start_point)),
                )
            )

        ring_template = _build_ring_template_from_atom_names(
            residue_name=residue_span.residue_name,
            atom_names=atom_names,
            ring_atom_prefix=ring_atom_prefix,
        )
        if ring_template is not None:
            ring_geometries.append(
                _build_ring_geometry(
                    residue_index=residue_index,
                    residue_name=residue_span.residue_name,
                    residue_coordinates=residue_coordinates,
                    ring_template=ring_template,
                    final_box_lengths=final_box_lengths,
                )
            )

    return (
        tuple(template_cache.values()),
        tuple(bond_geometries),
        tuple(ring_geometries),
    )


def _build_slit_bond_geometries(
    slit_system: _GroSystem,
    slit_coordinates: FloatArray,
    final_box_lengths: FloatArray,
) -> tuple[tuple[_ResidueBondTemplate, ...], tuple[_BondSegmentGeometry, ...]]:
    """Build cached slit bond templates and bond-segment geometries."""

    template_cache: dict[tuple[str, tuple[str, ...]], _ResidueBondTemplate] = {}
    bond_geometries: list[_BondSegmentGeometry] = []

    for residue_index, residue_span in enumerate(slit_system.residue_spans):
        atom_names = tuple(slit_system.atom_names[residue_span.start:residue_span.stop])
        template_key = (residue_span.residue_name, atom_names)
        residue_coordinates = _unwrap_residue_coordinates(
            slit_coordinates[residue_span.start:residue_span.stop],
            final_box_lengths,
        )

        residue_bond_template = template_cache.get(template_key)
        if residue_bond_template is None:
            residue_bond_template = _ResidueBondTemplate(
                residue_name=residue_span.residue_name,
                atom_names=atom_names,
                bond_definitions=_guess_bond_definitions(
                    atom_names=atom_names,
                    residue_coordinates=residue_coordinates,
                    include_hydrogen_bonds=True,
                ),
            )
            template_cache[template_key] = residue_bond_template

        for bond_definition in residue_bond_template.bond_definitions:
            start_point = residue_coordinates[bond_definition.start_atom_index]
            stop_point = residue_coordinates[bond_definition.stop_atom_index]
            midpoint = 0.5 * (start_point + stop_point)
            wrapped_midpoint = wrap_positions(midpoint[np.newaxis, :], final_box_lengths)[0]
            bond_geometries.append(
                _BondSegmentGeometry(
                    residue_index=residue_index,
                    residue_name=residue_span.residue_name,
                    start_atom_name=bond_definition.start_atom_name,
                    stop_atom_name=bond_definition.stop_atom_name,
                    start_point=start_point,
                    stop_point=stop_point,
                    midpoint=midpoint,
                    wrapped_midpoint=wrapped_midpoint,
                    half_length_nm=0.5 * float(np.linalg.norm(stop_point - start_point)),
                )
            )

    return tuple(template_cache.values()), tuple(bond_geometries)


def _point_inside_polygon_with_padding(
    point_2d: FloatArray,
    polygon_2d: FloatArray,
    padding_nm: float,
) -> bool:
    """Check whether a 2D point lies inside or very near a polygon."""

    inside = False
    point_x = float(point_2d[0])
    point_y = float(point_2d[1])
    vertex_count = polygon_2d.shape[0]

    for vertex_index in range(vertex_count):
        next_index = (vertex_index + 1) % vertex_count
        x1, y1 = polygon_2d[vertex_index]
        x2, y2 = polygon_2d[next_index]
        intersects = ((y1 > point_y) != (y2 > point_y)) and (
            point_x < (x2 - x1) * (point_y - y1) / (y2 - y1 + 1.0e-12) + x1
        )
        if intersects:
            inside = not inside

    if inside or padding_nm <= 0.0:
        return inside

    minimum_distance_nm = np.inf
    for vertex_index in range(vertex_count):
        next_index = (vertex_index + 1) % vertex_count
        segment_start = polygon_2d[vertex_index]
        segment_stop = polygon_2d[next_index]
        segment = segment_stop - segment_start
        denominator = float(np.dot(segment, segment))
        if denominator <= 1.0e-12:
            projection = segment_start
        else:
            fraction = float(
                np.clip(
                    np.dot(point_2d - segment_start, segment) / denominator,
                    0.0,
                    1.0,
                )
            )
            projection = segment_start + fraction * segment
        minimum_distance_nm = min(minimum_distance_nm, float(np.linalg.norm(point_2d - projection)))

    return minimum_distance_nm <= padding_nm


def _shift_residue_near_reference(
    residue_coordinates: FloatArray,
    reference_point: FloatArray,
    box_lengths: FloatArray,
) -> FloatArray:
    """Shift one residue by a whole-box image so it sits near a reference point."""

    residue_center = np.mean(residue_coordinates, axis=0)
    image_shift = box_lengths * np.round((residue_center - reference_point) / box_lengths)
    return residue_coordinates - image_shift


def _shift_bond_near_reference(
    start_point: FloatArray,
    stop_point: FloatArray,
    reference_point: FloatArray,
    box_lengths: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Shift one bond by a whole-box image so it sits near a reference point."""

    midpoint = 0.5 * (start_point + stop_point)
    image_shift = box_lengths * np.round((midpoint - reference_point) / box_lengths)
    return start_point - image_shift, stop_point - image_shift


def _bond_crosses_ring(
    bond_start: FloatArray,
    bond_stop: FloatArray,
    ring_geometry: _RingGeometry,
    plane_tolerance_nm: float,
    polygon_padding_nm: float,
) -> bool:
    """Check whether a bond passes through an aromatic-ring polygon."""

    bond_direction = bond_stop - bond_start
    denominator = float(np.dot(ring_geometry.normal, bond_direction))
    if abs(denominator) > 1.0e-12:
        fraction = float(
            np.clip(
                np.dot(ring_geometry.normal, ring_geometry.center - bond_start) / denominator,
                0.0,
                1.0,
            )
        )
    else:
        start_distance = abs(float(np.dot(bond_start - ring_geometry.center, ring_geometry.normal)))
        stop_distance = abs(float(np.dot(bond_stop - ring_geometry.center, ring_geometry.normal)))
        fraction = 0.0 if start_distance <= stop_distance else 1.0

    closest_point = bond_start + fraction * bond_direction
    plane_distance_nm = abs(float(np.dot(closest_point - ring_geometry.center, ring_geometry.normal)))
    if plane_distance_nm > plane_tolerance_nm:
        return False

    projected_point = closest_point - ring_geometry.center
    projected_point_2d = np.array(
        [
            np.dot(projected_point, ring_geometry.basis_u),
            np.dot(projected_point, ring_geometry.basis_v),
        ],
        dtype=np.float64,
    )
    if np.linalg.norm(projected_point_2d) > (ring_geometry.max_radius_nm + polygon_padding_nm):
        return False

    return _point_inside_polygon_with_padding(
        point_2d=projected_point_2d,
        polygon_2d=ring_geometry.polygon_2d,
        padding_nm=polygon_padding_nm,
    )


def _center_crop_guest_residues(
    guest_system: _GroSystem,
    final_box_lengths: FloatArray,
) -> tuple[FloatArray, BoolArray, FloatArray]:
    """Center-crop guest residues from the larger reservoir box to the slit cell."""

    crop_window_start = 0.5 * (guest_system.box_lengths - final_box_lengths)
    crop_window_stop = crop_window_start + final_box_lengths
    translated_coordinates = np.zeros_like(guest_system.coordinates)
    selected_residues = np.zeros(len(guest_system.residue_spans), dtype=bool)
    tolerance = 1.0e-6

    for residue_index, residue_span in enumerate(guest_system.residue_spans):
        residue_coordinates = _unwrap_residue_coordinates(
            guest_system.coordinates[residue_span.start:residue_span.stop],
            guest_system.box_lengths,
        )
        is_inside_crop = bool(
            np.all(residue_coordinates >= (crop_window_start - tolerance))
            and np.all(residue_coordinates < (crop_window_stop + tolerance))
        )
        if is_inside_crop:
            translated_coordinates[residue_span.start:residue_span.stop] = residue_coordinates - crop_window_start
            selected_residues[residue_index] = True

    return translated_coordinates, selected_residues, crop_window_start


def _identify_clashing_guest_residues(
    guest_system: _GroSystem,
    slit_system: _GroSystem,
    translated_guest_coordinates: FloatArray,
    selected_residue_mask: BoolArray,
    slit_coordinates: FloatArray,
    final_box_lengths: FloatArray,
    general_cutoff_nm: float,
    ring_atom_prefix: str,
    ring_plane_tolerance_nm: float,
    ring_polygon_padding_nm: float,
    include_hydrogen_bonds_in_ring_check: bool,
) -> tuple[_ClashSelection, _RingCheckCache]:
    """Mark every selected guest residue that clashes with the slit.

    Parameters
    ----------
    guest_system : _GroSystem
        Loaded guest reservoir.
    slit_system : _GroSystem
        Loaded slit structure.
    translated_guest_coordinates : ndarray
        Guest coordinates translated into the slit reference frame.
    selected_residue_mask : ndarray
        Residues retained by cropping and optional plane filtering.
    slit_coordinates : ndarray
        Slit coordinates in the output-cell reference frame.
    final_box_lengths : ndarray
        Orthorhombic periodic box lengths in nanometers.
    general_cutoff_nm : float
        General all-atom guest-to-slit clash cutoff in nanometers.
    ring_atom_prefix : str
        Atom-name prefix identifying six-atom aromatic rings.
    ring_plane_tolerance_nm : float
        Maximum bond-to-ring-plane distance for a crossing.
    ring_polygon_padding_nm : float
        Additional in-plane ring-polygon padding.
    include_hydrogen_bonds_in_ring_check : bool
        Whether guessed bonds to hydrogen participate in forward ring checks.

    Returns
    -------
    clash_selection : _ClashSelection
        Per-residue masks for every clash mechanism.
    ring_check_cache : _RingCheckCache
        Cached templates and geometries used for reporting.
    """

    residue_count = len(guest_system.residue_spans)
    removed_by_general = np.zeros(residue_count, dtype=bool)
    removed_by_forward_ring = np.zeros(residue_count, dtype=bool)
    removed_by_reverse_ring = np.zeros(residue_count, dtype=bool)

    slit_wrapped = wrap_positions(slit_coordinates, final_box_lengths)
    slit_tree = cKDTree(slit_wrapped, boxsize=final_box_lengths)

    candidate_atoms = np.zeros(guest_system.atom_count, dtype=bool)
    for residue_index, residue_span in enumerate(guest_system.residue_spans):
        if selected_residue_mask[residue_index]:
            candidate_atoms[residue_span.start:residue_span.stop] = True

    if np.any(candidate_atoms):
        candidate_atom_indices = np.flatnonzero(candidate_atoms)
        wrapped_candidate_coordinates = wrap_positions(
            translated_guest_coordinates[candidate_atom_indices],
            final_box_lengths,
        )
        nearest_distances, _ = slit_tree.query(wrapped_candidate_coordinates, k=1, workers=-1)
        clashing_candidate_atoms = np.zeros(guest_system.atom_count, dtype=bool)
        clashing_candidate_atoms[candidate_atom_indices] = nearest_distances <= general_cutoff_nm
        if np.any(clashing_candidate_atoms):
            residue_indices = guest_system.atom_to_residue_index[clashing_candidate_atoms]
            removed_by_general[np.unique(residue_indices)] = True

    guest_bond_templates, guest_bond_geometries, guest_ring_geometries = (
        _build_guest_filter_geometries(
            guest_system=guest_system,
            translated_guest_coordinates=translated_guest_coordinates,
            selected_residue_mask=selected_residue_mask,
            final_box_lengths=final_box_lengths,
            ring_atom_prefix=ring_atom_prefix,
            include_hydrogen_bonds=include_hydrogen_bonds_in_ring_check,
        )
    )
    slit_ring_geometries = _build_slit_ring_geometries(
        slit_system=slit_system,
        slit_coordinates=slit_coordinates,
        final_box_lengths=final_box_lengths,
        ring_atom_prefix=ring_atom_prefix,
    )
    slit_bond_templates, slit_bond_geometries = _build_slit_bond_geometries(
        slit_system=slit_system,
        slit_coordinates=slit_coordinates,
        final_box_lengths=final_box_lengths,
    )
    ring_check_cache = _RingCheckCache(
        slit_ring_geometries=slit_ring_geometries,
        guest_bond_templates=guest_bond_templates,
        guest_bond_geometries=guest_bond_geometries,
        guest_ring_geometries=guest_ring_geometries,
        slit_bond_templates=slit_bond_templates,
        slit_bond_geometries=slit_bond_geometries,
    )

    if slit_ring_geometries and guest_bond_geometries:
        slit_ring_center_tree = cKDTree(
            np.array([ring_geometry.wrapped_center for ring_geometry in slit_ring_geometries]),
            boxsize=final_box_lengths,
        )
        maximum_slit_ring_radius_nm = max(
            ring_geometry.max_radius_nm for ring_geometry in slit_ring_geometries
        )
        for bond_geometry in guest_bond_geometries:
            candidate_ring_indices = slit_ring_center_tree.query_ball_point(
                bond_geometry.wrapped_midpoint,
                r=(
                    bond_geometry.half_length_nm
                    + maximum_slit_ring_radius_nm
                    + ring_plane_tolerance_nm
                    + ring_polygon_padding_nm
                ),
            )
            for ring_index in candidate_ring_indices:
                ring_geometry = slit_ring_geometries[ring_index]
                shifted_start_point, shifted_stop_point = _shift_bond_near_reference(
                    start_point=bond_geometry.start_point,
                    stop_point=bond_geometry.stop_point,
                    reference_point=ring_geometry.center,
                    box_lengths=final_box_lengths,
                )
                if _bond_crosses_ring(
                    bond_start=shifted_start_point,
                    bond_stop=shifted_stop_point,
                    ring_geometry=ring_geometry,
                    plane_tolerance_nm=ring_plane_tolerance_nm,
                    polygon_padding_nm=ring_polygon_padding_nm,
                ):
                    removed_by_forward_ring[bond_geometry.residue_index] = True
                    break

    if guest_ring_geometries and slit_bond_geometries:
        slit_bond_midpoint_tree = cKDTree(
            np.array([bond_geometry.wrapped_midpoint for bond_geometry in slit_bond_geometries]),
            boxsize=final_box_lengths,
        )
        maximum_slit_bond_half_length_nm = max(
            bond_geometry.half_length_nm for bond_geometry in slit_bond_geometries
        )

        for guest_ring_geometry in guest_ring_geometries:
            candidate_bond_indices = slit_bond_midpoint_tree.query_ball_point(
                guest_ring_geometry.wrapped_center,
                r=(
                    guest_ring_geometry.max_radius_nm
                    + maximum_slit_bond_half_length_nm
                    + ring_plane_tolerance_nm
                    + ring_polygon_padding_nm
                ),
            )
            for bond_index in candidate_bond_indices:
                bond_geometry = slit_bond_geometries[bond_index]
                shifted_start_point, shifted_stop_point = _shift_bond_near_reference(
                    start_point=bond_geometry.start_point,
                    stop_point=bond_geometry.stop_point,
                    reference_point=guest_ring_geometry.center,
                    box_lengths=final_box_lengths,
                )
                if _bond_crosses_ring(
                    bond_start=shifted_start_point,
                    bond_stop=shifted_stop_point,
                    ring_geometry=guest_ring_geometry,
                    plane_tolerance_nm=ring_plane_tolerance_nm,
                    polygon_padding_nm=ring_polygon_padding_nm,
                ):
                    removed_by_reverse_ring[guest_ring_geometry.residue_index] = True
                    break

    removed_by_any_ring = removed_by_forward_ring | removed_by_reverse_ring
    clash_selection = _ClashSelection(
        removed_residue_mask=removed_by_general | removed_by_any_ring,
        removed_by_general_mask=removed_by_general,
        removed_by_forward_ring_mask=removed_by_forward_ring,
        removed_by_reverse_ring_mask=removed_by_reverse_ring,
        removed_by_any_ring_mask=removed_by_any_ring,
    )
    return clash_selection, ring_check_cache


def _build_kept_guest_atom_mask(
    guest_system: _GroSystem,
    selected_residue_mask: BoolArray,
    removed_residue_mask: BoolArray,
) -> BoolArray:
    """Build an atom mask for residues that survive the full filtering flow."""

    keep_mask = np.zeros(guest_system.atom_count, dtype=bool)
    for residue_index, residue_span in enumerate(guest_system.residue_spans):
        if selected_residue_mask[residue_index] and not removed_residue_mask[residue_index]:
            keep_mask[residue_span.start:residue_span.stop] = True
    return keep_mask


def _build_guest_filter_summaries(
    guest_system: _GroSystem,
    cropped_residue_mask: BoolArray,
    selected_residue_mask: BoolArray,
    surface_plane_removed_mask: BoolArray,
    clash_selection: _ClashSelection,
) -> tuple[GuestResidueFilterSummary, ...]:
    """Summarize filtering outcomes for every guest residue name.

    Parameters
    ----------
    guest_system : _GroSystem
        Loaded guest reservoir.
    cropped_residue_mask : ndarray
        Residues retained by center cropping.
    selected_residue_mask : ndarray
        Residues remaining after optional plane filtering.
    surface_plane_removed_mask : ndarray
        Residues removed by the plane filter.
    clash_selection : _ClashSelection
        General and ring clash masks.

    Returns
    -------
    summaries : tuple[GuestResidueFilterSummary, ...]
        Deterministic summaries ordered by residue name.
    """

    residue_names = np.array(
        [residue_span.residue_name for residue_span in guest_system.residue_spans],
        dtype=object,
    )
    summaries = []
    for residue_name in sorted(set(residue_names)):
        name_mask = residue_names == residue_name
        initial = int(np.count_nonzero(name_mask))
        cropped = int(np.count_nonzero(cropped_residue_mask & name_mask))
        surface_filtered = int(np.count_nonzero(selected_residue_mask & name_mask))
        removed_by_surface = int(
            np.count_nonzero(surface_plane_removed_mask & name_mask)
        )
        removed_by_general = int(
            np.count_nonzero(clash_selection.removed_by_general_mask & name_mask)
        )
        removed_by_forward = int(
            np.count_nonzero(clash_selection.removed_by_forward_ring_mask & name_mask)
        )
        removed_by_reverse = int(
            np.count_nonzero(clash_selection.removed_by_reverse_ring_mask & name_mask)
        )
        removed_by_ring = int(
            np.count_nonzero(clash_selection.removed_by_any_ring_mask & name_mask)
        )
        removed_by_clash = int(
            np.count_nonzero(clash_selection.removed_residue_mask & name_mask)
        )
        removed_outside_crop = initial - cropped
        removed_total = removed_outside_crop + removed_by_surface + removed_by_clash
        summaries.append(
            GuestResidueFilterSummary(
                residue_name=str(residue_name),
                initial_residues=initial,
                cropped_residues=cropped,
                removed_outside_crop_residues=removed_outside_crop,
                surface_plane_filtered_residues=surface_filtered,
                removed_by_surface_plane_residues=removed_by_surface,
                removed_by_general_cutoff_residues=removed_by_general,
                removed_by_forward_ring_residues=removed_by_forward,
                removed_by_reverse_ring_residues=removed_by_reverse,
                removed_by_any_ring_residues=removed_by_ring,
                removed_by_any_clash_residues=removed_by_clash,
                removed_total_residues=removed_total,
                remaining_residues=initial - removed_total,
            )
        )
    return tuple(summaries)


def _build_framework_system(merged_system: _GroSystem, target_resname: str) -> _GroSystem:
    """Extract the non-target framework from a merged slit-plus-guest system."""

    framework_atom_mask = np.array(
        [residue_name != target_resname for residue_name in merged_system.residue_names],
        dtype=bool,
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


def _count_target_molecules(merged_system: _GroSystem, target_resname: str) -> tuple[int, int]:
    """Count target residues and target atoms in one merged system."""

    target_residue_count = 0
    target_atom_count = 0
    for residue_span in merged_system.residue_spans:
        if residue_span.residue_name != target_resname:
            continue
        target_residue_count += 1
        target_atom_count += residue_span.stop - residue_span.start
    return target_residue_count, target_atom_count


def _format_value_lines(title: str, rows: Sequence[tuple[str, str]]) -> str:
    """Format one aligned report section from label/value rows.

    Parameters
    ----------
    title : str
        Section heading.
    rows : sequence[tuple[str, str]]
        Label/value pairs rendered in the section body.

    Returns
    -------
    str
        Formatted text section.
    """

    width = max((len(label) for label, _ in rows), default=0)
    body = "\n".join(f"  {label:<{width}} : {value}" for label, value in rows)
    return f"{title}\n{'-' * len(title)}\n{body}" if body else f"{title}\n{'-' * len(title)}"


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
        ("Seed values", " ".join(str(seed_value) for seed_value in probe_estimate.seed_values)),
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


def _format_geometry_block(
    title: str,
    slit_geometry: PeriodicSlitGeometry,
    padding_nm: float,
) -> str:
    """Format one periodic slit geometry report section.

    Parameters
    ----------
    title : str
        Section heading.
    slit_geometry : PeriodicSlitGeometry
        Geometry to report.
    padding_nm : float
        Signed plane padding used by the workflow.

    Returns
    -------
    str
        Human-readable geometry section.
    """

    roughness = (
        "not available"
        if slit_geometry.normal_roughness_rms_nm is None
        else f"{slit_geometry.normal_roughness_rms_nm:.5f} nm"
    )
    support_count = (
        "not available"
        if slit_geometry.surface_support_count is None
        else str(slit_geometry.surface_support_count)
    )
    return _format_value_lines(
        title,
        [
            ("Normal axis", slit_geometry.normal_axis_name),
            ("Axis index", str(slit_geometry.normal_axis_index)),
            ("Lower mean plane", f"{slit_geometry.lower_plane_nm:.5f} nm"),
            ("Upper mean plane", f"{slit_geometry.upper_plane_nm:.5f} nm"),
            ("Interval wraps", str(slit_geometry.interval_wraps)),
            ("Mean-plane separation", f"{slit_geometry.plane_separation_nm:.5f} nm"),
            ("Signed padding", f"{padding_nm:.5f} nm"),
            ("Padded width", f"{slit_geometry.padded_width_nm(padding_nm):.5f} nm"),
            (
                "Projected area per face",
                f"{slit_geometry.projected_area_per_face_nm2:.5f} nm^2",
            ),
            (
                "Total projected surface area",
                f"{slit_geometry.total_projected_surface_area_nm2:.5f} nm^2",
            ),
            (
                "Geometric slit volume",
                f"{slit_geometry.geometric_slit_volume_nm3:.5f} nm^3",
            ),
            (
                "Padded geometric volume",
                f"{slit_geometry.padded_geometric_volume_nm3(padding_nm):.5f} nm^3",
            ),
            ("Surface support positions", support_count),
            ("Normal RMS roughness", roughness),
        ],
    )


def _build_fill_report_text(config: SlitFillConfig, report: SlitFillReport) -> str:
    """Build the human-readable text report for the slit-fill workflow."""

    inputs = _format_value_lines(
        "Inputs",
        [
            ("Guest reservoir", str(config.guest_path)),
            ("Slit structure", str(config.slit_path)),
            ("Merged output", str(config.output_path)),
            ("Report file", str(config.log_path)),
        ],
    )
    selection = _format_value_lines(
        "Selection",
        [
            ("Density/report residue", config.target_resname),
            ("General cutoff", f"{config.general_cutoff_nm:.3f} nm"),
            ("Surface-plane filter", str(config.use_surface_plane_filter)),
            ("Surface-plane padding", f"{config.surface_plane_padding_nm:.3f} nm"),
            ("Initial guest molecules", str(report.initial_guest_molecules)),
            ("Cropped guest molecules", str(report.cropped_guest_molecules)),
            ("Removed outside crop", str(report.removed_outside_crop_guest_molecules)),
            (
                "After surface-plane filter",
                str(report.surface_plane_filtered_guest_molecules),
            ),
            (
                "Removed by surface-plane filter",
                str(report.removed_by_surface_plane_guest_molecules),
            ),
        ],
    )
    input_geometry = _format_geometry_block(
        "Input slit geometry",
        report.input_slit_geometry,
        config.surface_plane_padding_nm,
    )
    output_geometry = _format_geometry_block(
        "Output slit geometry",
        report.output_slit_geometry,
        config.surface_plane_padding_nm,
    )
    clash_filters = _format_value_lines(
        "Clash filters",
        [
            (
                "Removed by general cutoff",
                str(report.removed_by_general_cutoff_guest_molecules),
            ),
            ("Removed by any clash", str(report.removed_by_clash_guest_molecules)),
            ("Removed total", str(report.removed_guest_molecules)),
            ("Remaining guest molecules", str(report.remaining_guest_molecules)),
            ("General only", str(report.removed_by_general_only_guest_molecules)),
            (
                "General + forward ring only",
                str(report.removed_by_general_and_forward_ring_only_guest_molecules),
            ),
            (
                "General + reverse ring only",
                str(report.removed_by_general_and_reverse_ring_only_guest_molecules),
            ),
            (
                "General + forward + reverse",
                str(report.removed_by_general_and_forward_and_reverse_ring_guest_molecules),
            ),
        ],
    )
    ring_checks = _format_value_lines(
        "Ring checks",
        [
            ("Ring atom prefix", config.ring_atom_prefix),
            ("Ring plane tolerance", f"{config.ring_plane_tolerance_nm:.3f} nm"),
            ("Ring polygon padding", f"{config.ring_polygon_padding_nm:.3f} nm"),
            (
                "Include hydrogen bonds",
                str(config.include_hydrogen_bonds_in_ring_check),
            ),
            ("Slit aromatic rings", str(report.slit_aromatic_ring_count)),
            ("Guest aromatic rings", str(report.cropped_guest_ring_count)),
            ("Guest bond templates", str(report.guest_bond_template_count)),
            ("Guest bonds checked", str(report.guest_bond_count_checked)),
            ("Slit bond templates", str(report.slit_bond_template_count)),
            ("Slit bonds checked", str(report.slit_bond_count_checked)),
            ("Removed by forward ring", str(report.removed_by_forward_ring_guest_molecules)),
            ("Removed by reverse ring", str(report.removed_by_reverse_ring_guest_molecules)),
            ("Removed by any ring", str(report.removed_by_any_ring_guest_molecules)),
            (
                "Forward ring only",
                str(report.removed_by_forward_ring_only_guest_molecules),
            ),
            (
                "Reverse ring only",
                str(report.removed_by_reverse_ring_only_guest_molecules),
            ),
            (
                "Forward + reverse ring only",
                str(report.removed_by_forward_and_reverse_ring_only_guest_molecules),
            ),
        ],
    )
    density = _format_value_lines(
        "Density",
        [
            (
                "Guest molecule mass",
                f"{report.density_estimate.guest_molecule_mass_da:.5f} Da",
            ),
            ("Total guest mass", f"{report.density_estimate.total_guest_mass_da:.5f} Da"),
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
                "Samples per seed",
                str(report.density_estimate.sample_count_per_seed),
            ),
            ("Seed count", str(report.density_estimate.seed_count)),
        ],
    )
    probe_details = "\n\n".join(
        _format_probe_block(probe_estimate)
        for probe_estimate in report.density_estimate.probe_estimates
    )
    residue_filter_details = "\n\n".join(
        _format_value_lines(
            f"Residue filtering: {summary.residue_name}",
            [
                ("Initial", str(summary.initial_residues)),
                ("Cropped", str(summary.cropped_residues)),
                ("Removed outside crop", str(summary.removed_outside_crop_residues)),
                ("After surface-plane filter", str(summary.surface_plane_filtered_residues)),
                ("Removed by surface plane", str(summary.removed_by_surface_plane_residues)),
                ("Removed by general cutoff", str(summary.removed_by_general_cutoff_residues)),
                ("Removed by forward ring", str(summary.removed_by_forward_ring_residues)),
                ("Removed by reverse ring", str(summary.removed_by_reverse_ring_residues)),
                ("Removed by any clash", str(summary.removed_by_any_clash_residues)),
                ("Removed total", str(summary.removed_total_residues)),
                ("Remaining", str(summary.remaining_residues)),
            ],
        )
        for summary in report.residue_filter_summaries
    )
    output = _format_value_lines(
        "Output",
        [
            ("Slit atoms", str(report.slit_atom_count)),
            ("Final atom count", str(report.final_atom_count)),
            ("Final residue count", str(report.final_residue_count)),
            (
                "Axis order",
                " ".join(AXIS_NAMES[axis_index] for axis_index in report.output_axis_permutation),
            ),
            (
                "Crop window start",
                " ".join(f"{value:.5f}" for value in report.crop_window_start_nm) + " nm",
            ),
            (
                "Output box",
                " ".join(f"{value:.5f}" for value in report.output_box_nm) + " nm",
            ),
            ("Wrapped into box", str(config.wrap_output)),
        ],
    )

    sections = [
        "Slit Fill Report\n================",
        inputs,
        selection,
        input_geometry,
        output_geometry,
        clash_filters,
        ring_checks,
        residue_filter_details,
        density,
        "Probe details\n-------------\n" + probe_details,
        output,
    ]
    return "\n\n".join(sections) + "\n"


def _build_density_report_text(config: SlitDensityConfig, report: SlitDensityReport) -> str:
    """Build the human-readable text report for density analysis."""

    inputs = _format_value_lines(
        "Inputs",
        [
            ("Merged input", str(config.input_path)),
            ("Report file", str(config.log_path)),
            ("Target residue", config.target_resname),
        ],
    )
    counts = _format_value_lines(
        "Counts",
        [
            ("Guest molecules", str(report.guest_molecule_count)),
            ("Guest atoms", str(report.guest_atom_count)),
        ],
    )
    framework = _format_value_lines(
        "Framework",
        [
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
            ("Total guest mass", f"{report.density_estimate.total_guest_mass_da:.5f} Da"),
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


def fill_slit(config: SlitFillConfig) -> SlitFillReport:
    """Run the full slit guest-filling workflow.

    Parameters
    ----------
    config : SlitFillConfig
        Slit-fill settings and output paths.

    Returns
    -------
    SlitFillReport
        Structured report for the completed workflow.

    Notes
    -----
    Every cropped residue type is physically screened. ``target_resname``
    selects only the density population and target-specific report fields. The
    GRO, geometry YAML, and log are promoted together after all three succeed.
    Positions, optional velocities, box lengths, and geometry are expressed in
    the same output frame with the slit normal on ``z``. Translations and
    whole-residue wrapping affect positions only. If either input contains
    velocities, missing velocities in the other input are written as zeros;
    otherwise velocity columns are omitted. Input arrays are not modified.
    """

    config = _resolve_fill_config(config)
    guest_system = _load_gro_system(config.guest_path)
    slit_system = _load_gro_system(config.slit_path)
    _validate_slit_coordinates(slit_system)
    _validate_fill_config(config, guest_system, slit_system)

    final_box_lengths = slit_system.box_lengths.copy()
    centered_slit_coordinates = slit_system.coordinates.copy()
    translated_guest_coordinates, cropped_residue_mask, crop_window_start = _center_crop_guest_residues(
        guest_system=guest_system,
        final_box_lengths=final_box_lengths,
    )

    input_slit_geometry = _resolve_slit_geometry(
        slit_system=slit_system,
        slit_coordinates=centered_slit_coordinates,
        explicit_geometry=config.slit_geometry,
        geometry_path=config.slit_geometry_path,
        padding_nm=config.surface_plane_padding_nm,
    )
    surface_plane_removed_mask = np.zeros(len(guest_system.residue_spans), dtype=bool)
    selected_residue_mask = cropped_residue_mask
    if config.use_surface_plane_filter:
        surface_plane_selection = _apply_surface_plane_filter(
            guest_system=guest_system,
            translated_guest_coordinates=translated_guest_coordinates,
            selected_residue_mask=cropped_residue_mask,
            slit_geometry=input_slit_geometry,
            padding_nm=config.surface_plane_padding_nm,
        )
        selected_residue_mask = surface_plane_selection.selected_residue_mask
        surface_plane_removed_mask = surface_plane_selection.removed_residue_mask

    clash_selection, ring_check_cache = _identify_clashing_guest_residues(
        guest_system=guest_system,
        slit_system=slit_system,
        translated_guest_coordinates=translated_guest_coordinates,
        selected_residue_mask=selected_residue_mask,
        slit_coordinates=centered_slit_coordinates,
        final_box_lengths=final_box_lengths,
        general_cutoff_nm=config.general_cutoff_nm,
        ring_atom_prefix=config.ring_atom_prefix,
        ring_plane_tolerance_nm=config.ring_plane_tolerance_nm,
        ring_polygon_padding_nm=config.ring_polygon_padding_nm,
        include_hydrogen_bonds_in_ring_check=config.include_hydrogen_bonds_in_ring_check,
    )
    kept_guest_mask = _build_kept_guest_atom_mask(
        guest_system=guest_system,
        selected_residue_mask=selected_residue_mask,
        removed_residue_mask=clash_selection.removed_residue_mask,
    )
    residue_filter_summaries = _build_guest_filter_summaries(
        guest_system=guest_system,
        cropped_residue_mask=cropped_residue_mask,
        selected_residue_mask=selected_residue_mask,
        surface_plane_removed_mask=surface_plane_removed_mask,
        clash_selection=clash_selection,
    )

    output_axis_permutation = _build_output_axis_permutation(
        input_slit_geometry.normal_axis_index
    )
    output_slit_geometry = input_slit_geometry.permute_axes(output_axis_permutation)
    output_box_lengths = _permute_box_axes(
        box_lengths=final_box_lengths,
        axis_permutation=output_axis_permutation,
    )
    output_slit_coordinates = _permute_vector_axes(
        vectors=centered_slit_coordinates,
        axis_permutation=output_axis_permutation,
    )
    output_guest_coordinates = _permute_vector_axes(
        vectors=translated_guest_coordinates,
        axis_permutation=output_axis_permutation,
    )
    output_slit_velocities = (
        None
        if slit_system.velocities is None
        else _permute_vector_axes(slit_system.velocities, output_axis_permutation)
    )
    output_guest_velocities = (
        None
        if guest_system.velocities is None
        else _permute_vector_axes(guest_system.velocities, output_axis_permutation)
    )
    if config.wrap_output:
        output_guest_coordinates = _wrap_residues(
            system=guest_system,
            coordinates=output_guest_coordinates,
            keep_atom_mask=kept_guest_mask,
            box_lengths=output_box_lengths,
        )

    final_atom_count = slit_system.atom_count + int(np.count_nonzero(kept_guest_mask))
    final_residue_count = len(slit_system.residue_spans) + int(
        np.count_nonzero(
            selected_residue_mask & ~clash_selection.removed_residue_mask
        )
    )

    target_residue_mask = np.array(
        [residue_span.residue_name == config.target_resname for residue_span in guest_system.residue_spans],
        dtype=bool,
    )
    general_mask = clash_selection.removed_by_general_mask & target_residue_mask
    forward_mask = clash_selection.removed_by_forward_ring_mask & target_residue_mask
    reverse_mask = clash_selection.removed_by_reverse_ring_mask & target_residue_mask
    any_ring_mask = clash_selection.removed_by_any_ring_mask & target_residue_mask
    removed_mask = clash_selection.removed_residue_mask & target_residue_mask
    cropped_mask = cropped_residue_mask & target_residue_mask
    surface_plane_mask = surface_plane_removed_mask & target_residue_mask
    surface_plane_filtered_mask = selected_residue_mask & target_residue_mask

    initial_guest_molecules = int(np.count_nonzero(target_residue_mask))
    cropped_guest_molecules = int(np.count_nonzero(cropped_mask))
    removed_outside_crop_guest_molecules = initial_guest_molecules - cropped_guest_molecules
    surface_plane_filtered_guest_molecules = int(np.count_nonzero(surface_plane_filtered_mask))
    removed_by_surface_plane_guest_molecules = int(np.count_nonzero(surface_plane_mask))
    removed_by_general_cutoff_guest_molecules = int(np.count_nonzero(general_mask))
    removed_by_forward_ring_guest_molecules = int(np.count_nonzero(forward_mask))
    removed_by_reverse_ring_guest_molecules = int(np.count_nonzero(reverse_mask))
    removed_by_any_ring_guest_molecules = int(np.count_nonzero(any_ring_mask))

    removed_by_general_only_guest_molecules = int(np.count_nonzero(general_mask & ~forward_mask & ~reverse_mask))
    removed_by_forward_ring_only_guest_molecules = int(np.count_nonzero(~general_mask & forward_mask & ~reverse_mask))
    removed_by_reverse_ring_only_guest_molecules = int(np.count_nonzero(~general_mask & ~forward_mask & reverse_mask))
    removed_by_general_and_forward_ring_only_guest_molecules = int(
        np.count_nonzero(general_mask & forward_mask & ~reverse_mask)
    )
    removed_by_general_and_reverse_ring_only_guest_molecules = int(
        np.count_nonzero(general_mask & ~forward_mask & reverse_mask)
    )
    removed_by_forward_and_reverse_ring_only_guest_molecules = int(
        np.count_nonzero(~general_mask & forward_mask & reverse_mask)
    )
    removed_by_general_and_forward_and_reverse_ring_guest_molecules = int(
        np.count_nonzero(general_mask & forward_mask & reverse_mask)
    )

    removed_by_clash_guest_molecules = int(np.count_nonzero(removed_mask))
    removed_guest_molecules = (
        removed_outside_crop_guest_molecules
        + removed_by_surface_plane_guest_molecules
        + removed_by_clash_guest_molecules
    )
    remaining_guest_molecules = initial_guest_molecules - removed_guest_molecules

    density_estimate = _compute_density_estimate(
        guest_system=guest_system,
        framework_system=slit_system,
        framework_coordinates=centered_slit_coordinates,
        slit_geometry=input_slit_geometry,
        surface_plane_padding_nm=config.surface_plane_padding_nm,
        target_resname=config.target_resname,
        remaining_guest_molecules=remaining_guest_molecules,
        probe_radii_nm=config.density_probe_radii_nm,
        sample_count=config.density_sample_count,
        seed_count=config.density_seed_count,
        random_seed=config.random_seed,
    )

    report = SlitFillReport(
        residue_filter_summaries=residue_filter_summaries,
        initial_guest_molecules=initial_guest_molecules,
        cropped_guest_molecules=cropped_guest_molecules,
        removed_outside_crop_guest_molecules=removed_outside_crop_guest_molecules,
        input_slit_geometry=input_slit_geometry,
        output_slit_geometry=output_slit_geometry,
        surface_plane_filtered_guest_molecules=surface_plane_filtered_guest_molecules,
        removed_by_surface_plane_guest_molecules=removed_by_surface_plane_guest_molecules,
        removed_by_general_cutoff_guest_molecules=removed_by_general_cutoff_guest_molecules,
        removed_by_forward_ring_guest_molecules=removed_by_forward_ring_guest_molecules,
        removed_by_reverse_ring_guest_molecules=removed_by_reverse_ring_guest_molecules,
        removed_by_any_ring_guest_molecules=removed_by_any_ring_guest_molecules,
        removed_by_general_only_guest_molecules=removed_by_general_only_guest_molecules,
        removed_by_forward_ring_only_guest_molecules=removed_by_forward_ring_only_guest_molecules,
        removed_by_reverse_ring_only_guest_molecules=removed_by_reverse_ring_only_guest_molecules,
        removed_by_general_and_forward_ring_only_guest_molecules=removed_by_general_and_forward_ring_only_guest_molecules,
        removed_by_general_and_reverse_ring_only_guest_molecules=removed_by_general_and_reverse_ring_only_guest_molecules,
        removed_by_forward_and_reverse_ring_only_guest_molecules=removed_by_forward_and_reverse_ring_only_guest_molecules,
        removed_by_general_and_forward_and_reverse_ring_guest_molecules=removed_by_general_and_forward_and_reverse_ring_guest_molecules,
        removed_by_clash_guest_molecules=removed_by_clash_guest_molecules,
        removed_guest_molecules=removed_guest_molecules,
        remaining_guest_molecules=remaining_guest_molecules,
        slit_aromatic_ring_count=len(ring_check_cache.slit_ring_geometries),
        cropped_guest_ring_count=len(ring_check_cache.guest_ring_geometries),
        guest_bond_template_count=len(ring_check_cache.guest_bond_templates),
        guest_bond_count_checked=len(ring_check_cache.guest_bond_geometries),
        slit_bond_template_count=len(ring_check_cache.slit_bond_templates),
        slit_bond_count_checked=len(ring_check_cache.slit_bond_geometries),
        density_estimate=density_estimate,
        slit_atom_count=slit_system.atom_count,
        final_atom_count=final_atom_count,
        final_residue_count=final_residue_count,
        output_axis_permutation=output_axis_permutation,
        crop_window_start_nm=crop_window_start,
        output_box_nm=output_box_lengths,
    )

    report_text = _build_fill_report_text(config, report)
    assert config.log_path is not None
    metadata_path = config.output_path.with_suffix(".yml")
    final_paths = (config.output_path, metadata_path, config.log_path)
    with staged_output_paths(final_paths) as staging_paths:
        written_atom_count, written_residue_count = _write_merged_gro(
            output_path=staging_paths[config.output_path],
            slit_system=slit_system,
            slit_coordinates=output_slit_coordinates,
            slit_velocities=output_slit_velocities,
            guest_system=guest_system,
            guest_coordinates=output_guest_coordinates,
            guest_velocities=output_guest_velocities,
            kept_guest_mask=kept_guest_mask,
            final_box_lengths=output_box_lengths,
        )
        if (written_atom_count, written_residue_count) != (
            final_atom_count,
            final_residue_count,
        ):
            raise RuntimeError("Staged GRO counts diverged from the fill report.")
        _write_slit_geometry_metadata(
            staging_paths[metadata_path],
            output_slit_geometry,
            config.surface_plane_padding_nm,
        )
        staging_paths[config.log_path].write_text(report_text, encoding="utf-8")
    return report


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
    """

    config = _resolve_density_config(config)
    merged_system = _load_gro_system(config.input_path)
    _validate_density_config(config, merged_system)

    framework_system = _build_framework_system(
        merged_system=merged_system,
        target_resname=config.target_resname,
    )
    # Density estimation wraps framework coordinates before the periodic
    # neighbor search, so nominal out-of-box coordinates are harmless here and
    # the fill-time warning only adds noise for already merged systems.
    guest_molecule_count, guest_atom_count = _count_target_molecules(
        merged_system=merged_system,
        target_resname=config.target_resname,
    )
    slit_geometry = _resolve_slit_geometry(
        slit_system=framework_system,
        slit_coordinates=framework_system.coordinates,
        explicit_geometry=config.slit_geometry,
        geometry_path=config.slit_geometry_path,
        padding_nm=config.surface_plane_padding_nm,
    )
    density_estimate = _compute_density_estimate(
        guest_system=merged_system,
        framework_system=framework_system,
        framework_coordinates=framework_system.coordinates,
        slit_geometry=slit_geometry,
        surface_plane_padding_nm=config.surface_plane_padding_nm,
        target_resname=config.target_resname,
        remaining_guest_molecules=guest_molecule_count,
        probe_radii_nm=config.density_probe_radii_nm,
        sample_count=config.density_sample_count,
        seed_count=config.density_seed_count,
        random_seed=config.random_seed,
    )
    report = SlitDensityReport(
        guest_molecule_count=guest_molecule_count,
        guest_atom_count=guest_atom_count,
        framework_atom_count=framework_system.atom_count,
        framework_residue_count=len(framework_system.residue_spans),
        slit_geometry=slit_geometry,
        density_estimate=density_estimate,
    )
    report_text = _build_density_report_text(config, report)
    assert config.log_path is not None
    config.log_path.write_text(report_text, encoding="utf-8")
    return report


def _build_fill_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the slit-fill command."""

    parser = argparse.ArgumentParser(
        description=(
            "Center-crop a larger guest box to the slit cell, remove clashing "
            "guest molecules outside the detected slit planes or within a small "
            "all-atom cutoff, then reject guest residues involved in symmetric "
            "aromatic-ring crossings."
        )
    )
    parser.add_argument(
        "--guest",
        type=Path,
        default=SlitFillConfig.guest_path,
        help="GRO file containing the guest reservoir box. Default: %(default)s",
    )
    parser.add_argument(
        "--slit",
        type=Path,
        default=SlitFillConfig.slit_path,
        help="GRO file containing the grafted silica slit. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SlitFillConfig.output_path,
        help="Path to the merged GRO file. Default: %(default)s",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional report path. Defaults to <output_stem>.log next to the output GRO.",
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
        default=SlitFillConfig.target_resname,
        help=(
            "Residue name used for density and target-specific reporting; all "
            "cropped residue types are physically filtered. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--general-cutoff",
        type=float,
        default=SlitFillConfig.general_cutoff_nm,
        help=(
            "All-atom clash cutoff in nm. Guest residues with any atom closer "
            "than this distance to any slit atom are removed. The 0.10 nm "
            "default is deliberately permissive; 0.15 and 0.20 nm are useful "
            "progressively stricter starting points. Ring-crossing checks are "
            "applied separately. Every output still requires minimization."
        ),
    )
    parser.add_argument(
        "--ring-atom-prefix",
        default=SlitFillConfig.ring_atom_prefix,
        help="Atom-name prefix used to identify aromatic-ring atoms. Default: %(default)s",
    )
    parser.add_argument(
        "--ring-plane-tolerance",
        type=float,
        default=SlitFillConfig.ring_plane_tolerance_nm,
        help=(
            "Maximum distance in nm between a bond and a ring plane for the "
            "contact to count as crossing-like. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--ring-polygon-padding",
        type=float,
        default=SlitFillConfig.ring_polygon_padding_nm,
        help=(
            "Extra in-plane padding in nm added around the aromatic polygon "
            "during the ring-crossing tests. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--exclude-hydrogen-bonds-from-ring-check",
        action="store_true",
        help="Skip O-H and C-H bonds in the explicit ring-crossing tests.",
    )
    parser.add_argument(
        "--disable-surface-plane-filter",
        action="store_true",
        help=(
            "Skip the automatic slit-plane filter that removes guest residues "
            "outside the hydroxylated surface Si planes."
        ),
    )
    parser.add_argument(
        "--surface-plane-padding",
        type=float,
        default=SlitFillConfig.surface_plane_padding_nm,
        help=(
            "Signed padding in nm applied on each side of the detected slit "
            "interval before guest selection. Positive values shrink the "
            "allowed region; negative values expand it into the matrix. "
            "Default: %(default)s"
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
        default=SlitFillConfig.density_sample_count,
        help=(
            "Number of Monte Carlo sample points used for each density repeat. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--density-seed-count",
        type=int,
        default=SlitFillConfig.density_seed_count,
        help=(
            "Number of independent Monte Carlo repeats used for each probe "
            "radius. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional seed used to generate deterministic Monte Carlo seeds.",
    )
    parser.add_argument(
        "--no-wrap-output",
        action="store_true",
        help="Write coordinates exactly as merged instead of wrapping kept guest residues into the final box.",
    )
    return parser


def _build_density_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for merged-slit density analysis."""

    parser = argparse.ArgumentParser(
        description=(
            "Estimate target guest density inside an already merged slit structure "
            "by counting target molecules and computing framework-excluded, "
            "probe-free volume inside the mean-plane slit interval."
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
            "<input_stem>_density.log next to the input GRO."
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


def fill_slit_main(argv: Sequence[str] | None = None) -> SlitFillReport:
    """Parse CLI arguments and run the slit-fill workflow.

    Parameters
    ----------
    argv : sequence[str] or None, optional
        Optional argument vector. When omitted, arguments are read from the
        process command line.

    Returns
    -------
    SlitFillReport
        Structured slit-fill report.
    """

    parser = _build_fill_argument_parser()
    args = parser.parse_args(argv)
    probe_radii = tuple(
        float(radius)
        for radius in (
            args.density_probe_radius
            if args.density_probe_radius is not None
            else DEFAULT_DENSITY_PROBE_RADII_NM
        )
    )
    config = SlitFillConfig(
        guest_path=args.guest,
        slit_path=args.slit,
        output_path=args.output,
        log_path=args.log,
        slit_geometry_path=args.slit_geometry,
        target_resname=args.target_resname,
        general_cutoff_nm=args.general_cutoff,
        ring_atom_prefix=args.ring_atom_prefix,
        ring_plane_tolerance_nm=args.ring_plane_tolerance,
        ring_polygon_padding_nm=args.ring_polygon_padding,
        include_hydrogen_bonds_in_ring_check=not args.exclude_hydrogen_bonds_from_ring_check,
        use_surface_plane_filter=not args.disable_surface_plane_filter,
        surface_plane_padding_nm=args.surface_plane_padding,
        density_probe_radii_nm=probe_radii,
        density_sample_count=args.density_samples,
        density_seed_count=args.density_seed_count,
        wrap_output=not args.no_wrap_output,
        random_seed=args.random_seed,
    )
    return fill_slit(config)


def _fill_slit_console_main() -> int:
    """Run the slit-fill console entry point and return a process exit code.

    Returns
    -------
    int
        Successful process exit status.
    """

    fill_slit_main()
    return 0


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
