################################################################################
# Slit Filling Helpers                                                         #
#                                                                              #
"""Guest filling and clash filtering for amorphous silica slits."""
################################################################################

from __future__ import annotations

import argparse
import warnings

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from numpy.typing import NDArray
from silicams._gro_io import (
    _GroSystem,
    _load_gro_system,
    _write_merged_gro,
)
from silicams._output_transaction import staged_output_paths
from silicams._slit_geometry_io import (
    _resolve_slit_geometry,
    _write_slit_geometry_metadata,
)
from silicams._slit_guest_filter import (
    _ClashSelection,
    _apply_surface_plane_filter,
    _build_kept_guest_atom_mask,
    _center_crop_guest_residues,
    _identify_clashing_guest_residues,
)
from silicams._slit_report import _format_geometry_block, _format_value_lines
from silicams._validation import boolean, finite_real, integral
from silicams.slit_density import (
    DEFAULT_DENSITY_PROBE_RADII_NM,
    DensityEstimate,
    _compute_density_estimate,
    _format_probe_block,
    _select_target_population,
)
from silicams.slit_geometry import (
    AXIS_NAMES,
    PeriodicSlitGeometry,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]
BoolArray = NDArray[np.bool_]

SLIT_COORDINATE_TOLERANCE_NM = 1.0e-6

__all__ = [
    "SlitFillConfig",
    "GuestResidueFilterSummary",
    "SlitFillReport",
    "fill_slit",
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
        boolean(
            "include_hydrogen_bonds_in_ring_check",
            self.include_hydrogen_bonds_in_ring_check,
        )
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
        Full-box and padded-interval density metrics for retained target
        residues in the finalized output coordinates.
    framework_resnames : tuple[str, ...]
        Residue names originating in the slit input and therefore classified
        as framework for density analysis.
    mobile_resnames : tuple[str, ...]
        Residue names originating in the guest input and therefore classified
        as mobile for density analysis.
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
    framework_resnames: tuple[str, ...]
    mobile_resnames: tuple[str, ...]
    slit_atom_count: int
    final_atom_count: int
    final_residue_count: int
    output_axis_permutation: tuple[int, int, int]
    crop_window_start_nm: FloatArray
    output_box_nm: FloatArray


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
        Raised when the target residue is missing, a residue name occurs in
        both source systems, or the slit box cannot be cropped from the guest
        box.
    """

    if config.target_resname not in guest_system.residue_names:
        available_residues = sorted(set(guest_system.residue_names))
        raise ValueError(
            f"Residue name {config.target_resname!r} was not found in {config.guest_path}. "
            f"Available residue names: {', '.join(available_residues)}"
        )

    slit_resnames = set(slit_system.residue_names)
    guest_resnames = set(guest_system.residue_names)
    overlapping_resnames = sorted(slit_resnames & guest_resnames)
    if overlapping_resnames:
        raise ValueError(
            "Residue names occur in both the slit and guest inputs and would be "
            "ambiguous after merging: "
            + ", ".join(overlapping_resnames)
            + ". Rename the residues so framework and mobile selectors remain "
            "disjoint."
        )

    if np.any(slit_system.box_lengths > guest_system.box_lengths):
        raise ValueError(
            "The slit box is larger than the guest box in at least one dimension, "
            "so the guest reservoir cannot be center-cropped into the slit cell."
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
        if not np.all(keep_atom_mask[residue_span.start : residue_span.stop]):
            continue

        residue_coordinates = wrapped_coordinates[
            residue_span.start : residue_span.stop
        ]
        center_of_geometry = np.mean(residue_coordinates, axis=0)
        image_shift = box_lengths * np.floor(center_of_geometry / box_lengths)
        wrapped_coordinates[residue_span.start : residue_span.stop] = (
            residue_coordinates - image_shift
        )

    return wrapped_coordinates


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


def _build_fill_report_text(config: SlitFillConfig, report: SlitFillReport) -> str:
    """Build the human-readable text report for the slit-fill workflow.

    Parameters
    ----------
    config : SlitFillConfig
        Resolved fill inputs, outputs, and scientific settings.
    report : SlitFillReport
        Completed filtering, component, geometry, and density results.

    Returns
    -------
    str
        Newline-terminated report with explicit component roles and units.
    """

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
            ("Framework residue names", ", ".join(report.framework_resnames)),
            ("Mobile residue names", ", ".join(report.mobile_resnames)),
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
                str(
                    report.removed_by_general_and_forward_and_reverse_ring_guest_molecules
                ),
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
            (
                "Removed by forward ring",
                str(report.removed_by_forward_ring_guest_molecules),
            ),
            (
                "Removed by reverse ring",
                str(report.removed_by_reverse_ring_guest_molecules),
            ),
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
            ("Membership rule", "periodic center of mass inside padded interval"),
            (
                "Target molecules in full box",
                str(report.density_estimate.target_population.total_molecule_count),
            ),
            (
                "Target molecules in padded interval",
                str(report.density_estimate.target_population.interval_molecule_count),
            ),
            (
                "Target molecules outside padded interval",
                str(
                    report.density_estimate.target_population.outside_interval_molecule_count
                ),
            ),
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
                (
                    "After surface-plane filter",
                    str(summary.surface_plane_filtered_residues),
                ),
                (
                    "Removed by surface plane",
                    str(summary.removed_by_surface_plane_residues),
                ),
                (
                    "Removed by general cutoff",
                    str(summary.removed_by_general_cutoff_residues),
                ),
                (
                    "Removed by forward ring",
                    str(summary.removed_by_forward_ring_residues),
                ),
                (
                    "Removed by reverse ring",
                    str(summary.removed_by_reverse_ring_residues),
                ),
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
                " ".join(
                    AXIS_NAMES[axis_index]
                    for axis_index in report.output_axis_permutation
                ),
            ),
            (
                "Crop window start",
                " ".join(f"{value:.5f}" for value in report.crop_window_start_nm)
                + " nm",
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
    slit and guest input files define disjoint framework/mobile components;
    overlapping residue names are rejected and the source-derived roles are
    written to the geometry YAML. The GRO, geometry YAML, and log are promoted
    together after all three succeed.
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
    framework_resnames = tuple(sorted(set(slit_system.residue_names)))
    mobile_resnames = tuple(sorted(set(guest_system.residue_names)))

    final_box_lengths = slit_system.box_lengths.copy()
    centered_slit_coordinates = slit_system.coordinates.copy()
    translated_guest_coordinates, cropped_residue_mask, crop_window_start = (
        _center_crop_guest_residues(
            guest_system=guest_system,
            final_box_lengths=final_box_lengths,
        )
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
        np.count_nonzero(selected_residue_mask & ~clash_selection.removed_residue_mask)
    )

    target_residue_mask = np.array(
        [
            residue_span.residue_name == config.target_resname
            for residue_span in guest_system.residue_spans
        ],
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
    removed_outside_crop_guest_molecules = (
        initial_guest_molecules - cropped_guest_molecules
    )
    surface_plane_filtered_guest_molecules = int(
        np.count_nonzero(surface_plane_filtered_mask)
    )
    removed_by_surface_plane_guest_molecules = int(np.count_nonzero(surface_plane_mask))
    removed_by_general_cutoff_guest_molecules = int(np.count_nonzero(general_mask))
    removed_by_forward_ring_guest_molecules = int(np.count_nonzero(forward_mask))
    removed_by_reverse_ring_guest_molecules = int(np.count_nonzero(reverse_mask))
    removed_by_any_ring_guest_molecules = int(np.count_nonzero(any_ring_mask))

    removed_by_general_only_guest_molecules = int(
        np.count_nonzero(general_mask & ~forward_mask & ~reverse_mask)
    )
    removed_by_forward_ring_only_guest_molecules = int(
        np.count_nonzero(~general_mask & forward_mask & ~reverse_mask)
    )
    removed_by_reverse_ring_only_guest_molecules = int(
        np.count_nonzero(~general_mask & ~forward_mask & reverse_mask)
    )
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

    final_selected_residue_mask = (
        selected_residue_mask & ~clash_selection.removed_residue_mask
    )
    target_population = _select_target_population(
        guest_system=guest_system,
        guest_coordinates=output_guest_coordinates,
        slit_geometry=output_slit_geometry,
        surface_plane_padding_nm=config.surface_plane_padding_nm,
        target_resname=config.target_resname,
        selected_residue_mask=final_selected_residue_mask,
    )
    if target_population.total_molecule_count != remaining_guest_molecules:
        raise RuntimeError(
            "Final target population diverged from the filling filter counts."
        )

    density_estimate = _compute_density_estimate(
        guest_system=guest_system,
        framework_system=slit_system,
        framework_coordinates=output_slit_coordinates,
        slit_geometry=output_slit_geometry,
        surface_plane_padding_nm=config.surface_plane_padding_nm,
        target_resname=config.target_resname,
        target_population=target_population,
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
        framework_resnames=framework_resnames,
        mobile_resnames=mobile_resnames,
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
            framework_resnames=framework_resnames,
            mobile_resnames=mobile_resnames,
            target_resname=config.target_resname,
            framework_atom_count=slit_system.atom_count,
            framework_residue_count=len(slit_system.residue_spans),
        )
        staging_paths[config.log_path].write_text(report_text, encoding="utf-8")
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
