################################################################################
# Slit Preparation Helpers                                                     #
#                                                                              #
"""High-level helpers for preparing amorphous silica slit surfaces."""
################################################################################

from __future__ import annotations

import copy
import json
import os
import sys

from dataclasses import asdict, dataclass, field, replace
from time import perf_counter

import numpy as np
import yaml
from tqdm.auto import tqdm as _tqdm_auto

from . import utils
from ._output_transaction import staged_output_directory
from ._silica_placement import (
    _BridgeStericCache,
    _best_bridge_position,
    _bridge_candidate_positions as _bridge_candidate_positions_from_arrays,
    _bridge_pair_frame,
)
from ._validation import boolean, finite_range, finite_real, integral
from silicams.dice import Dice
from silicams.matrix import Matrix
from silicams.molecule import Molecule
from silicams.slit_geometry import (
    PeriodicSlitGeometry,
    pairwise_minimum_image_distances,
)
from silicams.slit_targets import (
    ExperimentalSiliconStateTarget,
    SiliconStateComposition,
    SiliconStateFractions,
    _nearest_integer_composition,
    _prepared_target_from_final,
    _prepared_target_is_compatible,
    _surface_target_candidates,
    _surface_target_from_experimental,
)
from silicams.topology import (
    BareSilicaChargeDiagnostics,
    FunctionalizedSlitChargeDiagnostics,
    GromacsAngleParameters,
    GromacsBondParameters,
    SilicaTopologyModel,
    default_silica_topology,
)
from silicams.slit_system import SilicaSlit, SurfacePreparationDiagnostics
from silicams.writers import GromacsTopologyWriter, StructureWriter


_BRIDGE_OFFSET_NM = 0.09
_BRIDGE_STERIC_DISTANCE_CUTOFF_NM = 0.30
_BRIDGE_STERIC_GRAPH_DEPTH = 6
_BRIDGE_MIN_CLEARANCE_BY_TYPE_NM = {
    "O": 0.18,
    "Si": 0.22,
    "H": 0.12,
}
_BRIDGE_CANDIDATE_ROTATIONS_DEG = (0, 45, -45, 90, -90, 135, -135, 180)


@dataclass(frozen=True)
class FunctionalizedSlitProgressConfig:
    """Progress-bar settings for functionalized slit preparation.

    Parameters
    ----------
    enabled : bool or None, optional
        Progress-display mode. ``None`` enables auto mode, which shows
        progress in interactive terminals and notebooks while staying quiet in
        non-interactive or pytest-driven contexts.
    leave : bool, optional
        When ``True``, keep completed progress bars visible instead of clearing
        them after completion.
    """

    enabled: bool | None = None
    leave: bool = False

    def __post_init__(self):
        """Validate the progress-display switches.

        Raises
        ------
        TypeError
            Raised when ``enabled`` is neither a boolean nor ``None``, or when
            ``leave`` is not a boolean.
        """

        if self.enabled is not None:
            boolean("enabled", self.enabled)
        boolean("leave", self.leave)


@dataclass
class _NullProgressBar:
    """No-op progress bar used when live progress is disabled."""

    total: int | None = None
    desc: str = ""
    unit: str = "it"
    leave: bool = False
    n: int = 0

    def update(self, value=1):
        """Advance the internal counter without rendering output."""
        self.n += value

    def set_description_str(self, desc, refresh=True):
        """Store the last description without rendering output."""
        del refresh
        self.desc = desc

    def close(self):
        """Close the no-op progress bar."""
        return None


@dataclass
class _FunctionalizedProgressTracker:
    """Outer progress tracker for the functionalized slit workflow."""

    total_stages: int
    progress_config: FunctionalizedSlitProgressConfig
    stage_bar: object = field(init=False)

    def __post_init__(self):
        """Create the outer workflow progress bar."""
        self.stage_bar = _create_progress_bar(
            total=self.total_stages,
            desc="Functionalized slit",
            progress_config=self.progress_config,
            unit="stage",
        )

    def set_stage(self, desc):
        """Update the current outer-stage description."""
        self.stage_bar.set_description_str(desc)

    def update_stage(self, value=1):
        """Advance the outer-stage bar."""
        self.stage_bar.update(value)

    def site_bar(self, desc, total):
        """Create an inner per-site progress bar."""
        return _create_progress_bar(
            total=total,
            desc=desc,
            progress_config=self.progress_config,
            unit="site",
        )

    def close(self):
        """Close the outer workflow bar."""
        self.stage_bar.close()


@dataclass(frozen=True)
class _AttachmentPhaseProgressContext:
    """Progress metadata for one batched slit-attachment phase.

    Parameters
    ----------
    phase_name : str
        Human-readable attachment phase label such as ``"T2 attachment"``.
    requested_count : int
        Number of attachment slots requested for the current candidate surface.
    candidate_index : int or None, optional
        One-based candidate-surface attempt index shown in progress text.
    total_candidates : int or None, optional
        Total number of candidate surfaces considered in the current workflow.
    """

    phase_name: str
    requested_count: int
    candidate_index: int | None = None
    total_candidates: int | None = None


@dataclass(frozen=True)
class _SlitSiteArrayCache:
    """Array-backed site geometry cache used for slit adjacency searches.

    Parameters
    ----------
    site_ids : tuple[int, ...]
        Surface silicon identifiers in the cached array order.
    positions : np.ndarray
        Cartesian silicon positions with shape ``(n, 3)``.
    site_index : dict[int, int]
        Mapping from silicon identifiers to row indices in ``positions``.
    direct_connection_mask : np.ndarray
        Boolean ``(n, n)`` mask marking silicon pairs that already share a
        bonded oxygen atom.
    """

    site_ids: tuple[int, ...]
    positions: np.ndarray
    site_index: dict[int, int]
    direct_connection_mask: np.ndarray


def _is_interactive_progress_environment():
    """Return whether live progress bars should auto-enable."""
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False

    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None and shell.__class__.__name__ in {
            "TerminalInteractiveShell",
            "ZMQInteractiveShell",
        }:
            return True
    except Exception:
        pass

    return any(
        getattr(stream, "isatty", lambda: False)()
        for stream in (sys.stderr, sys.stdout)
    )


def _progress_is_enabled(progress_config):
    """Return whether a progress bar should render live output.

    Parameters
    ----------
    progress_config : FunctionalizedSlitProgressConfig
        Functionalized slit progress settings.

    Returns
    -------
    enabled : bool
        True when live progress should be displayed.
    """
    if progress_config.enabled is not None:
        return progress_config.enabled
    return _is_interactive_progress_environment()


def _create_progress_bar(total, desc, progress_config, unit="it"):
    """Create one built-in progress bar or a no-op fallback.

    Parameters
    ----------
    total : int or None
        Total number of expected updates.
    desc : str
        Human-readable progress description.
    progress_config : FunctionalizedSlitProgressConfig
        Functionalized slit progress settings.
    unit : str, optional
        Unit label shown by the progress bar backend.

    Returns
    -------
    bar : object
        Live ``tqdm`` progress bar when enabled, otherwise a no-op fallback
        with the same ``update`` / ``set_description_str`` / ``close``
        interface.
    """
    if not _progress_is_enabled(progress_config):
        return _NullProgressBar(total=total, desc=desc, unit=unit, leave=progress_config.leave)

    return _tqdm_auto(
        total=total,
        desc=desc,
        leave=progress_config.leave,
        unit=unit,
        dynamic_ncols=True,
    )


def _candidate_attempt_suffix(candidate_index, total_candidates):
    """Return a compact candidate-attempt suffix for progress text.

    Parameters
    ----------
    candidate_index : int or None
        One-based candidate-surface attempt index.
    total_candidates : int or None
        Total number of candidate surfaces considered.

    Returns
    -------
    suffix : str
        Empty string when no candidate context is available, otherwise a
        bracketed ``"[candidate i/n]"`` suffix.
    """
    if candidate_index is None or total_candidates is None or total_candidates <= 1:
        return ""

    return f" [candidate {candidate_index}/{total_candidates}]"


def _set_candidate_stage(progress_tracker, stage_name, candidate_index=None, total_candidates=None):
    """Update the outer stage label, optionally including candidate context.

    Parameters
    ----------
    progress_tracker : _FunctionalizedProgressTracker or None
        Outer workflow progress tracker.
    stage_name : str
        Human-readable workflow stage label.
    candidate_index : int or None, optional
        One-based candidate-surface attempt index shown in progress text.
    total_candidates : int or None, optional
        Total number of candidate surfaces considered.

    Returns
    -------
    None
        The tracker is updated in-place when it exists.
    """
    if progress_tracker is None:
        return

    progress_tracker.set_stage(stage_name)
    suffix = _candidate_attempt_suffix(candidate_index, total_candidates)
    if suffix:
        progress_tracker.set_stage(f"{stage_name}{suffix}")


def _attachment_progress_description(progress_context, attached_count):
    """Return the progress-bar description for one attachment batch.

    Parameters
    ----------
    progress_context : _AttachmentPhaseProgressContext
        Metadata describing the current batched attachment phase.
    attached_count : int
        Number of successfully attached molecules so far.

    Returns
    -------
    desc : str
        Human-readable description summarizing attached count and, when
        available, candidate-attempt context.
    """
    return (
        f"{progress_context.phase_name} "
        f"{attached_count}/{progress_context.requested_count} attached"
        f"{_candidate_attempt_suffix(progress_context.candidate_index, progress_context.total_candidates)}"
    )


@dataclass(frozen=True)
class AmorphousSlitConfig:
    """Configuration for a periodic bare amorphous silica slit.

    Parameters
    ----------
    surface_target : ExperimentalSiliconStateTarget
        Experimental silicon-state ratios over all Si atoms together with the
        mandatory physical surface-silicon fraction used to derive the modeled
        surface target.
    name : str, optional
        Base name used for stored slit files.
    slit_width_nm : float, optional
        Width of the slit in nanometers.
    repeat_y : int, optional
        Number of amorphous template copies stacked along the slit-normal
        direction.
    temperature_k : float, optional
        Target simulation temperature in Kelvin.
    amorph_bond_range_nm : tuple, optional
        Accepted ``Si-O`` bond-length range for the amorphous template.
    siloxane_distance_range_nm : tuple, optional
        Accepted ``Si-Si`` distance range for custom siloxane formation.
    surface_fraction_tolerance : float, optional
        Allowed absolute fraction deviation per silicon state when the exact
        integer target cannot be realized on the current slit.
    random_seed : int or None, optional
        Optional seed used to randomize chemically equivalent surface-editing
        and grafting choices. When omitted, deterministic candidate ordering
        is used. Supplying the same seed reproduces the same slit variant,
        while different seeds can produce different arrangements with the same
        requested silicon-state composition.
    template_split_pairs : tuple, optional
        Template-specific bond pairs that must be disconnected after
        reconstructing the amorphous connectivity matrix.
    silica_topology : SilicaTopologyModel or None, optional
        Optional editable silica force-field model used by slit full-topology
        export. When omitted, the package defaults are used. Use
        :func:`silicams.topology.default_silica_topology` to inspect the active
        defaults and create one editable copy for local overrides.
    """

    surface_target: ExperimentalSiliconStateTarget
    name: str = "bare_amorphous_silica_slit"
    slit_width_nm: float = 7.0
    repeat_y: int = 2
    temperature_k: float = 300.0
    amorph_bond_range_nm: tuple[float, float] = (0.160 - 0.02, 0.160 + 0.02)
    siloxane_distance_range_nm: tuple[float, float] = (0.40, 0.65)
    surface_fraction_tolerance: float = 0.005
    random_seed: int | None = None
    template_split_pairs: tuple[tuple[int, int], ...] = ((57790, 2524),)
    silica_topology: SilicaTopologyModel | None = None

    def __post_init__(self):
        """Validate the slit configuration.

        Raises
        ------
        ValueError
            Raised when a finite scientific value, range, count domain, seed,
            or template split pair is invalid.
        TypeError
            Raised when ``surface_target`` or any integer, range, or split-pair
            field uses an unsupported type.
        """
        if not isinstance(self.surface_target, ExperimentalSiliconStateTarget):
            raise TypeError(
                "surface_target must be an ExperimentalSiliconStateTarget instance."
            )
        slit_width_nm = finite_real("slit_width_nm", self.slit_width_nm)
        repeat_y = integral("repeat_y", self.repeat_y)
        temperature_k = finite_real("temperature_k", self.temperature_k)
        amorph_bond_range_nm = finite_range(
            "amorph_bond_range_nm",
            self.amorph_bond_range_nm,
            positive=True,
        )
        siloxane_distance_range_nm = finite_range(
            "siloxane_distance_range_nm",
            self.siloxane_distance_range_nm,
            positive=True,
        )
        surface_fraction_tolerance = finite_real(
            "surface_fraction_tolerance",
            self.surface_fraction_tolerance,
        )
        if slit_width_nm <= 0:
            raise ValueError("The slit width must be positive.")
        if repeat_y < 1:
            raise ValueError("The amorphous slit requires at least one y-repeat.")
        if temperature_k <= 0:
            raise ValueError("The temperature must be positive.")
        if surface_fraction_tolerance < 0:
            raise ValueError("The surface fraction tolerance must be non-negative.")
        random_seed = None
        if self.random_seed is not None:
            random_seed = integral("random_seed", self.random_seed)
            if random_seed < 0:
                raise ValueError("The random seed must be non-negative.")
        if not isinstance(self.template_split_pairs, tuple):
            raise TypeError("template_split_pairs must be a tuple.")
        normalized_split_pairs = []
        for pair in self.template_split_pairs:
            if not isinstance(pair, tuple):
                raise TypeError("Each template split pair must be a tuple.")
            if len(pair) != 2:
                raise ValueError("Each template split pair must contain two atom indices.")
            atom_a = integral("template split atom index", pair[0])
            atom_b = integral("template split atom index", pair[1])
            if atom_a < 0 or atom_b < 0:
                raise ValueError("Template split atom indices must be non-negative.")
            if atom_a == atom_b:
                raise ValueError("Template split atom indices must be distinct.")
            normalized_split_pairs.append((atom_a, atom_b))

        object.__setattr__(self, "slit_width_nm", slit_width_nm)
        object.__setattr__(self, "repeat_y", repeat_y)
        object.__setattr__(self, "temperature_k", temperature_k)
        object.__setattr__(self, "amorph_bond_range_nm", amorph_bond_range_nm)
        object.__setattr__(
            self,
            "siloxane_distance_range_nm",
            siloxane_distance_range_nm,
        )
        object.__setattr__(
            self,
            "surface_fraction_tolerance",
            surface_fraction_tolerance,
        )
        object.__setattr__(self, "random_seed", random_seed)
        object.__setattr__(self, "template_split_pairs", tuple(normalized_split_pairs))


@dataclass(frozen=True)
class SlitPreparationReport:
    """Summary of a prepared or functionalized amorphous slit.

    Parameters
    ----------
    name : str
        Slit-system name.
    temperature_k : float
        Target simulation temperature in Kelvin.
    requested_slit_width_nm : float
        User-requested slit width in nanometers. The fitted physical separation
        is stored on ``slit_geometry``.
    slit_geometry : PeriodicSlitGeometry
        Fitted periodic mean-plane geometry used by downstream workflows.
    site_ex : int
        Number of exterior surface sites.
    siloxane_bridges : int
        Number of siloxane bridges introduced during surface editing.
    siloxane_distance_range_nm : tuple[float, float]
        Accepted ``Si-Si`` distance range used during custom surface editing.
    surface_fraction_tolerance : float
        Allowed absolute fraction deviation per silicon state for fallback
        target selection.
    random_seed : int or None
        Seed used to randomize surface-editing and grafting choices, or
        ``None`` when the deterministic ordering was used.
    used_surface_tolerance : bool
        Whether the selected target was accepted through the tolerance fallback
        rather than matched exactly.
    experimental_target : ExperimentalSiliconStateTarget
        Experimental all-silicon target supplied by the caller.
    derived_surface_target : SiliconStateFractions
        Surface-only fractions derived from the experimental target and
        its required physical surface-silicon fraction.
    initial_surface : SiliconStateComposition
        Surface composition before custom condensation.
    target_surface : SiliconStateComposition
        Selected integer final surface target.
    prepared_surface : SiliconStateComposition
        Surface composition after Q-state preparation and before optional
        grafting.
    final_surface : SiliconStateComposition
        Final post-grafting surface composition. For bare slits this is
        identical to ``prepared_surface``.
    preparation_diagnostics : SurfacePreparationDiagnostics
        Surface-cleanup and bridge-insertion diagnostics collected across
        preparation, Q-state editing, and optional grafting.
    functionalization_steric_settings : FunctionalizedSlitStericConfig or None, optional
        Permissive graft-placement contact settings used for a functionalized
        build. Bare-slit reports store ``None``.
    timing_summary : SlitTimingSummary, optional
        Lightweight wall-clock timing summary for the major slit-preparation
        stages. Bare-slit builds leave all values at zero unless later export
        stages populate them.
    """

    name: str
    temperature_k: float
    requested_slit_width_nm: float
    slit_geometry: PeriodicSlitGeometry
    site_ex: int
    siloxane_bridges: int
    siloxane_distance_range_nm: tuple[float, float]
    surface_fraction_tolerance: float
    random_seed: int | None
    used_surface_tolerance: bool
    experimental_target: ExperimentalSiliconStateTarget
    derived_surface_target: SiliconStateFractions
    initial_surface: SiliconStateComposition
    target_surface: SiliconStateComposition
    prepared_surface: SiliconStateComposition
    final_surface: SiliconStateComposition
    preparation_diagnostics: SurfacePreparationDiagnostics
    functionalization_steric_settings: "FunctionalizedSlitStericConfig | None" = None
    timing_summary: "SlitTimingSummary" = field(default_factory=lambda: SlitTimingSummary())


@dataclass
class SlitPreparationResult:
    """Prepared bare slit system together with its report.

    Parameters
    ----------
    system : SilicaSlit
        Bare slit system associated with the preparation report.
    report : SlitPreparationReport
        Summary of the generated slit geometry and surface composition.
    silica_topology : SilicaTopologyModel
        Resolved silica topology model actually associated with this slit
        result.
    bare_charge_diagnostics : BareSilicaChargeDiagnostics or None, optional
        Bare-slit charge-neutrality diagnostics computed for finalized bare
        exports. Prepared but not yet finalized results leave this as
        ``None``. :func:`write_bare_amorphous_slit` populates this field on
        the returned result.
    """

    system: SilicaSlit
    report: SlitPreparationReport
    silica_topology: SilicaTopologyModel
    bare_charge_diagnostics: BareSilicaChargeDiagnostics | None = None


@dataclass(frozen=True)
class SilaneTopologyConfig:
    """Flat ligand-topology input used by the slit full-topology exporter.

    Parameters
    ----------
    itp_path : str
        Path to one self-contained flat GROMACS ``.itp`` file describing the
        base post-condensation ``T3`` silane fragment, including the
        replacement surface silicon atom.
    moleculetype_name : str, optional
        Optional explicit molecule-type name expected inside ``itp_path``.
        When omitted, the parser accepts the file's own name.
    geminal_cross_terms : SilaneGeminalCrossTerms or None, optional
        Optional explicit bonded terms used only when the exporter internally
        augments the base ``T3`` fragment into a geminal ``T2`` site by
        adding one silica ``OH`` group. This same helper also carries the
        optional retained-scaffold ``O-Si(mount)-first_ligand_atom`` angle
        needed by ligands whose first atom bound to the mount silicon is not
        oxygen. When the requested target includes geminal ``T2`` sites,
        these terms are required. When no geminal sites are exported and the
        first ligand atom is oxygen, this can remain ``None``.
    Notes
    -----
    The supplied flat ITP must be self-contained, must use atom names that
    exactly match the configured :class:`SilaneAttachmentConfig.molecule`, and
    must already satisfy the total-charge target derived from the active
    silica model.
    """

    itp_path: str
    moleculetype_name: str = ""
    geminal_cross_terms: "SilaneGeminalCrossTerms | None" = None

    def __post_init__(self):
        """Validate the silane topology configuration payload.

        Raises
        ------
        TypeError
            Raised when ``itp_path`` is not a string or when nested topology
            helper payloads use the wrong type. This catches accidental
            trailing commas that turn ``geminal_cross_terms`` into a one-item
            tuple instead of a :class:`SilaneGeminalCrossTerms` instance.
        """
        if not isinstance(self.itp_path, str):
            raise TypeError("itp_path must be a string path.")
        if not isinstance(self.moleculetype_name, str):
            raise TypeError("moleculetype_name must be a string.")
        if self.geminal_cross_terms is not None and not isinstance(
            self.geminal_cross_terms,
            SilaneGeminalCrossTerms,
        ):
            raise TypeError(
                "geminal_cross_terms must be a SilaneGeminalCrossTerms "
                f"instance or None, not {type(self.geminal_cross_terms).__name__}."
            )
@dataclass(frozen=True)
class GeminalMountDihedralSpec:
    """One optional generated geminal dihedral around the mount silicon.

    Parameters
    ----------
    fourth_atom_name : str
        Ligand-side atom name used as the fourth atom in the generated
        ``O(geminal)-Si(mount)-first_ligand_atom-X`` dihedral.
    function : int
        GROMACS dihedral function type.
    parameters : tuple[str, ...], optional
        Optional raw parameter tokens written after ``function``.
    """

    fourth_atom_name: str
    function: int
    parameters: tuple[str, ...] = ()

    def __post_init__(self):
        """Validate one generated geminal dihedral specification.

        Raises
        ------
        TypeError
            Raised when names, function identifiers, or parameter tokens use
            unsupported types.
        ValueError
            Raised when the atom name is empty or the function is not positive.
        """

        if not isinstance(self.fourth_atom_name, str):
            raise TypeError("fourth_atom_name must be a string.")
        if not self.fourth_atom_name:
            raise ValueError("fourth_atom_name must not be empty.")
        function = integral("function", self.function)
        if function <= 0:
            raise ValueError("function must be strictly positive.")
        if not isinstance(self.parameters, tuple) or not all(
            isinstance(parameter, str) for parameter in self.parameters
        ):
            raise TypeError("parameters must be a tuple of strings.")
        object.__setattr__(self, "function", function)


@dataclass(frozen=True)
class SilaneGeminalCrossTerms:
    """Explicit bonded cross terms around the ligand mount silicon.

    Parameters
    ----------
    first_ligand_atom_name : str
        Atom name of the first ligand-side atom bonded to the mount silicon in
        the base ``T3`` fragment topology.
    geminal_oxygen_mount_ligand_angle : GromacsAngleParameters
        Angle parameters used for the generated
        ``O(geminal)-Si(mount)-first_ligand_atom`` term.
    scaffold_oxygen_mount_ligand_angle : GromacsAngleParameters or None, optional
        Optional angle parameters used for retained scaffold
        ``O(scaffold)-Si(mount)-first_ligand_atom`` terms. This is required
        when the first ligand atom is not oxygen because the generic silica
        ``O-Si-O`` defaults no longer apply.
    geminal_dihedrals : tuple[GeminalMountDihedralSpec, ...], optional
        Optional explicit generated dihedrals of the form
        ``O(geminal)-Si(mount)-first_ligand_atom-X``.
    """

    first_ligand_atom_name: str
    geminal_oxygen_mount_ligand_angle: GromacsAngleParameters
    scaffold_oxygen_mount_ligand_angle: GromacsAngleParameters | None = None
    geminal_dihedrals: tuple[GeminalMountDihedralSpec, ...] = ()

    def __post_init__(self):
        """Validate the geminal cross-term payload.

        Raises
        ------
        TypeError
            Raised when the angle or any optional generated dihedral uses the
            wrong helper type.
        """
        if not isinstance(self.first_ligand_atom_name, str):
            raise TypeError("first_ligand_atom_name must be a string.")
        if (
            self.scaffold_oxygen_mount_ligand_angle is not None
            and not isinstance(
                self.scaffold_oxygen_mount_ligand_angle,
                GromacsAngleParameters,
            )
        ):
            raise TypeError(
                "scaffold_oxygen_mount_ligand_angle must be a "
                "GromacsAngleParameters instance or None."
            )
        if not isinstance(
            self.geminal_oxygen_mount_ligand_angle,
            GromacsAngleParameters,
        ):
            raise TypeError(
                "geminal_oxygen_mount_ligand_angle must be a "
                "GromacsAngleParameters instance."
            )
        if not isinstance(self.geminal_dihedrals, tuple):
            raise TypeError("geminal_dihedrals must be a tuple.")
        for dihedral in self.geminal_dihedrals:
            if not isinstance(dihedral, GeminalMountDihedralSpec):
                raise TypeError(
                    "geminal_dihedrals must contain only "
                    "GeminalMountDihedralSpec instances."
                )


@dataclass(frozen=True)
class SilaneAttachmentConfig:
    """Attachment settings for one silane family.

    Parameters
    ----------
    molecule : Molecule
        Base post-condensation ligand fragment used for both single and
        geminal attachment. When ``topology`` is supplied, the molecule's atom
        names must match the atom names used in that flat ITP bundle.
    mount : int
        Atom id placed onto the selected silicon surface site.
    axis : tuple[int, int]
        Two atom ids defining the molecular attachment axis.
    rotate_about_axis : bool, optional
        True to scan several rotations around ``axis`` and keep the least
        crowded pose before giving up on one site.
    rotate_step_deg : float, optional
        Angular step in degrees used when ``rotate_about_axis`` is enabled.
    topology : SilaneTopologyConfig or None, optional
        Optional flat ligand-topology input used to assemble a full
        self-contained GROMACS slit topology during export. When omitted,
        functionalized slit preparation and coordinate export still work, but
        no functionalized slit ``.top`` / ``.itp`` pair is written.
    """

    molecule: Molecule
    mount: int
    axis: tuple[int, int]
    rotate_about_axis: bool = True
    rotate_step_deg: float = 10.0
    topology: SilaneTopologyConfig | None = None

    def __post_init__(self):
        """Validate the ligand attachment geometry and scan settings.

        Raises
        ------
        TypeError
            Raised when the molecule, atom indices, switches, or nested
            topology configuration use unsupported types.
        ValueError
            Raised when an atom index is out of range, the axis is degenerate,
            or the rotation step is not finite and strictly positive.
        """

        if not isinstance(self.molecule, Molecule):
            raise TypeError("molecule must be a Molecule instance.")
        mount = integral("mount", self.mount)
        if not isinstance(self.axis, tuple):
            raise TypeError("axis must be a tuple.")
        if len(self.axis) != 2:
            raise ValueError("axis must contain exactly two atom indices.")
        axis = (integral("axis atom index", self.axis[0]), integral("axis atom index", self.axis[1]))
        atom_count = self.molecule.get_num()
        if mount < 0 or mount >= atom_count or any(index < 0 or index >= atom_count for index in axis):
            raise ValueError("Attachment atom indices must refer to atoms in molecule.")
        if axis[0] == axis[1]:
            raise ValueError("Attachment axis atom indices must be distinct.")
        boolean("rotate_about_axis", self.rotate_about_axis)
        rotate_step_deg = finite_real("rotate_step_deg", self.rotate_step_deg)
        if rotate_step_deg <= 0.0:
            raise ValueError("rotate_step_deg must be strictly positive.")
        if self.topology is not None and not isinstance(self.topology, SilaneTopologyConfig):
            raise TypeError("topology must be a SilaneTopologyConfig instance or None.")
        object.__setattr__(self, "mount", mount)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "rotate_step_deg", rotate_step_deg)


@dataclass(frozen=True)
class FunctionalizedSlitStericConfig:
    """Slit-only steric acceptance settings for exact silane placement.

    Parameters
    ----------
    enabled : bool, optional
        True to reject sterically crowded slit-grafting poses during the exact
        ``T2/T3`` attachment workflow.
    clearance_scale : float, optional
        Multiplicative factor applied to the sum of covalent radii when
        estimating the minimum allowed atom-pair separation for the slit-only
        steric screen. The default ``0.60`` is deliberately permissive and is
        intended to produce a starting structure for staged energy
        minimization. Larger values reject more crowded poses but can make an
        exact surface target slow or impossible to realize. TEPS testing found
        ``0.75`` and ``0.85`` useful progressively stricter alternatives.
    """

    enabled: bool = True
    clearance_scale: float = 0.60

    def __post_init__(self):
        """Validate the slit steric settings.

        Raises
        ------
        ValueError
            Raised when the steric clearance scale is not finite and strictly
            positive.
        """
        boolean("enabled", self.enabled)
        clearance_scale = finite_real("clearance_scale", self.clearance_scale)
        if clearance_scale <= 0:
            raise ValueError(
                "The slit steric clearance scale must be finite and greater "
                "than zero."
            )
        object.__setattr__(self, "clearance_scale", clearance_scale)


@dataclass(frozen=True)
class SlitTimingSummary:
    """Lightweight wall-clock timings for major slit-preparation stages.

    Parameters
    ----------
    base_slit_build_s : float, optional
        Seconds spent building and preparing the unfunctionalized slit.
    q_state_preparation_s : float, optional
        Seconds spent editing the slit surface to the requested prepared
        ``Q``-state composition before any optional grafting.
    t2_attachment_s : float, optional
        Seconds spent attaching the requested ``T2`` population.
    t3_attachment_s : float, optional
        Seconds spent attaching the requested ``T3`` population.
    finalize_s : float, optional
        Seconds spent in :meth:`silicams.slit_system.SilicaSlit.finalize`.
    export_s : float, optional
        Seconds spent writing the requested coordinate/topology export files.
    """

    base_slit_build_s: float = 0.0
    q_state_preparation_s: float = 0.0
    t2_attachment_s: float = 0.0
    t3_attachment_s: float = 0.0
    finalize_s: float = 0.0
    export_s: float = 0.0


@dataclass(frozen=True)
class FunctionalizedAmorphousSlitConfig:
    """Configuration for an exactly targeted functionalized amorphous slit.

    Parameters
    ----------
    slit_config : AmorphousSlitConfig
        Base slit configuration, including the unified experimental target.
    ligand : SilaneAttachmentConfig
        Silane attachment definition used to realize ``T2`` and ``T3``.
        Coordinate-only exports require only the molecular fragment, while
        full-topology exports additionally require
        :class:`SilaneTopologyConfig`.
    steric_settings : FunctionalizedSlitStericConfig, optional
        Slit-only steric acceptance settings forwarded to the exact
        deterministic ``T2/T3`` attachment workflow.
    progress_settings : FunctionalizedSlitProgressConfig, optional
        Built-in progress-bar settings for the exact functionalized slit
        workflow.
    """

    slit_config: AmorphousSlitConfig
    ligand: SilaneAttachmentConfig
    steric_settings: FunctionalizedSlitStericConfig = field(
        default_factory=FunctionalizedSlitStericConfig
    )
    progress_settings: FunctionalizedSlitProgressConfig = field(
        default_factory=FunctionalizedSlitProgressConfig
    )

    def __post_init__(self):
        """Validate the functionalized slit configuration payload.

        Raises
        ------
        TypeError
            Raised when any nested configuration object has the wrong type.
            This commonly catches accidental trailing commas that turn the
            ``ligand`` payload into a one-element tuple instead of a single
            :class:`SilaneAttachmentConfig`.
        """
        if not isinstance(self.slit_config, AmorphousSlitConfig):
            raise TypeError(
                "slit_config must be an AmorphousSlitConfig instance."
            )
        if not isinstance(self.ligand, SilaneAttachmentConfig):
            raise TypeError(
                "ligand must be a SilaneAttachmentConfig instance, not "
                f"{type(self.ligand).__name__}."
            )
        if not isinstance(self.steric_settings, FunctionalizedSlitStericConfig):
            raise TypeError(
                "steric_settings must be a FunctionalizedSlitStericConfig "
                "instance."
            )
        if not isinstance(self.progress_settings, FunctionalizedSlitProgressConfig):
            raise TypeError(
                "progress_settings must be a FunctionalizedSlitProgressConfig "
                "instance."
            )


@dataclass
class FunctionalizedSlitResult:
    """Prepared functionalized slit system together with its report.

    Parameters
    ----------
    system : SilicaSlit
        Functionalized slit system associated with the preparation report.
    report : SlitPreparationReport
        Summary of the generated slit geometry and surface composition.
    silica_topology : SilicaTopologyModel
        Resolved silica topology model actually associated with this slit
        result.
    charge_diagnostics : FunctionalizedSlitChargeDiagnostics or None, optional
        Functionalized full-topology charge diagnostics when an explicit
        silane topology bundle was used during export. Preparation-only
        results and coordinate-only exports leave this as ``None``.
        :func:`write_functionalized_amorphous_slit` populates this field only
        when a functionalized full-topology export was requested and
        completed.
    """

    system: SilicaSlit
    report: SlitPreparationReport
    silica_topology: SilicaTopologyModel
    charge_diagnostics: FunctionalizedSlitChargeDiagnostics | None = None


@dataclass
class _SurfaceTargetAttempt:
    """Successful realization of one final slit-surface composition.

    Parameters
    ----------
    system : SilicaSlit
        Edited slit system that satisfies the selected final target.
    target_surface : SiliconStateComposition
        Selected integer final target.
    prepared_surface : SiliconStateComposition
        Intermediate prepared bare surface before optional grafting.
    final_surface : SiliconStateComposition
        Final post-grafting surface composition.
    siloxane_bridges : int
        Number of siloxane bridges introduced during Q-state preparation.
    used_surface_tolerance : bool
        Whether the selected target came from the tolerance fallback rather
        than the exact integer target.
    timing_summary : SlitTimingSummary
        Wall-clock timings collected while realizing the selected target.
    """

    system: SilicaSlit
    target_surface: SiliconStateComposition
    prepared_surface: SiliconStateComposition
    final_surface: SiliconStateComposition
    siloxane_bridges: int
    used_surface_tolerance: bool
    timing_summary: SlitTimingSummary = field(default_factory=SlitTimingSummary)


@dataclass
class _BaseSlitBuild:
    """Base slit system prepared before target realization.

    Parameters
    ----------
    system : SilicaSlit
        Prepared slit system before custom siloxane formation.
    total_surface_si : int
        Number of tracked surface silicon sites.
    initial_surface : SiliconStateComposition
        Initial surface composition before custom siloxane formation.
    """

    system: SilicaSlit
    total_surface_si: int
    initial_surface: SiliconStateComposition


def _amorphous_template_path():
    """Return the built-in amorphous silica template path.

    Returns
    -------
    template_path : str
        Absolute path to the amorphous silica template.
    """
    return os.path.join(os.path.dirname(__file__), "templates", "amorph.gro")


def resolve_silane_topology_config(ligand_config):
    """Return the explicit full-topology input for one silane attachment.

    Parameters
    ----------
    ligand_config : SilaneAttachmentConfig or None
        Functionalized slit ligand configuration.

    Returns
    -------
    topology_config : SilaneTopologyConfig or None
        Explicit user-supplied topology config when available. When omitted,
        functionalized coordinate export can still proceed, but no
        self-contained functionalized slit ``.top`` / ``.itp`` pair is
        written. The returned object is used for ligand-bundle parsing only;
        the flat ITP is interpreted as one base ``T3`` fragment.
    """
    if ligand_config is None:
        return None

    return ligand_config.topology


def resolve_silica_topology(config):
    """Return the resolved silica topology model for one slit workflow.

    Parameters
    ----------
    config : AmorphousSlitConfig
        Base slit configuration carrying the optional explicit silica model.
    Returns
    -------
    silica_topology : SilicaTopologyModel
        Deep-copied explicit model when supplied, otherwise a fresh package
        default.
    """
    if config.silica_topology is not None:
        return copy.deepcopy(config.silica_topology)

    return default_silica_topology()


def _replicate_along_y(base, repeat_y):
    """Replicate the amorphous template along the ``y`` axis.

    Parameters
    ----------
    base : Molecule
        Base amorphous silica template.
    repeat_y : int
        Number of copies along ``y``.

    Returns
    -------
    replicated : Molecule
        Replicated amorphous structure.
    """
    box = base.get_box()
    copies = []
    for copy_id in range(repeat_y):
        block = copy.deepcopy(base)
        block.translate([0, box[1] * copy_id, 0])
        copies.append(block)

    replicated = Molecule(inp=copies)
    replicated.set_box([box[0], box[1] * repeat_y, box[2]])
    replicated.set_name("replicated_amorphous_silica")
    replicated.set_short("AMO")

    return replicated


def _duplicate_template_splits(matrix, atoms_per_copy, repeat_y, split_pairs):
    """Apply template-specific bond removals to each replicated copy.

    Parameters
    ----------
    matrix : Matrix
        Connectivity matrix for the replicated amorphous structure.
    atoms_per_copy : int
        Number of atoms in one amorphous template copy.
    repeat_y : int
        Number of copies stacked along ``y``.
    split_pairs : tuple[tuple[int, int], ...]
        Pairwise atom indices that must be disconnected in each copy.
    """
    for copy_id in range(repeat_y):
        offset = copy_id * atoms_per_copy
        for atom_a, atom_b in split_pairs:
            matrix.split(atom_a + offset, atom_b + offset)


def _attached_state_counts(system, ligand):
    """Count attached ``T2`` and ``T3`` sites for one silane family.

    Parameters
    ----------
    system : SilicaSlit
        Current slit system.
    ligand : SilaneAttachmentConfig or None
        Silane family tracked in the current build. ``None`` means no grafted
        silicon states are present.

    Returns
    -------
    counts : tuple[int, int]
        Attached ``T2`` and ``T3`` counts in that order.
    """
    if ligand is None:
        return (0, 0)

    base_short = ligand.molecule.get_short()
    return system.attached_state_counts(base_short)


def _interior_attached_molecule_counts(system):
    """Return non-silanol interior molecule counts.

    Parameters
    ----------
    system : SilicaSlit
        Current slit system.

    Returns
    -------
    counts : dict[str, int]
        Attached interior molecule counts keyed by residue short name.
    """
    return system.attached_molecule_counts()


def _surface_composition(total_surface_si, sites, t2_sites=0, t3_sites=0):
    """Summarize the current five-state surface composition.

    Parameters
    ----------
    total_surface_si : int
        Total number of tracked surface silicon atoms.
    sites : dict[int, BindingSite]
        Current slit binding sites keyed by silicon identifier.
    t2_sites : int, optional
        Number of attached ``T2`` states already realized on the slit.
    t3_sites : int, optional
        Number of attached ``T3`` states already realized on the slit.

    Returns
    -------
    composition : SiliconStateComposition
        Five-state surface composition.
    """
    raw_q2_sites = sum(
        1 for site in sites.values() if site.site_type == "in" and site.is_geminal
    )
    raw_q3_sites = sum(
        1 for site in sites.values() if site.site_type == "in" and site.oxygen_count == 1
    )
    q2_sites = raw_q2_sites - t2_sites
    q3_sites = raw_q3_sites - t3_sites
    q4_sites = total_surface_si - q2_sites - q3_sites - t2_sites - t3_sites

    return SiliconStateComposition(
        total_surface_si=total_surface_si,
        q2_sites=q2_sites,
        q3_sites=q3_sites,
        q4_sites=q4_sites,
        t2_sites=t2_sites,
        t3_sites=t3_sites,
    )


def _realization_rng(random_seed):
    """Return a pseudorandom generator for one slit realization.

    Parameters
    ----------
    random_seed : int or None
        Optional seed from :class:`AmorphousSlitConfig`.

    Returns
    -------
    rng : numpy.random.Generator or None
        Seeded generator when ``random_seed`` is supplied, otherwise ``None``
        to keep deterministic candidate ordering.
    """
    return None if random_seed is None else np.random.default_rng(random_seed)


def _ordered_candidates(candidates, rng):
    """Return candidates in deterministic or seeded-random order.

    Parameters
    ----------
    candidates : iterable
        Candidate values to order.
    rng : numpy.random.Generator or None
        Optional generator used to shuffle the candidate order.

    Returns
    -------
    ordered : list
        Candidate values in their original order when ``rng`` is ``None``, or
        in a seeded pseudorandom order otherwise.
    """
    ordered = list(candidates)
    if rng is None or len(ordered) < 2:
        return ordered

    return [ordered[int(index)] for index in rng.permutation(len(ordered))]


def _are_sites_directly_connected(matrix, site_a, site_b):
    """Check whether two surface silicon atoms already share a bridge oxygen.

    Parameters
    ----------
    matrix : Matrix
        Current connectivity matrix.
    site_a : int
        First silicon site identifier.
    site_b : int
        Second silicon site identifier.

    Returns
    -------
    is_connected : bool
        True if the sites already share a bonded oxygen atom.
    """
    atoms_a = matrix.get_matrix()[site_a]["atoms"]
    atoms_b = matrix.get_matrix()[site_b]["atoms"]
    return any(atom_o in atoms_b for atom_o in atoms_a)


def _build_slit_site_array_cache(kit, site_ids):
    """Build array-backed site geometry data for slit adjacency searches.

    Parameters
    ----------
    kit : SilicaSlit
        Slit system under preparation.
    site_ids : list[int]
        Surface silicon identifiers that should populate the cache.

    Returns
    -------
    cache : _SlitSiteArrayCache
        Array-backed site geometry and direct-connection metadata.
    """
    site_ids = tuple(site_ids)
    if not site_ids:
        return _SlitSiteArrayCache(
            site_ids=(),
            positions=np.empty((0, 3), dtype=float),
            site_index={},
            direct_connection_mask=np.empty((0, 0), dtype=bool),
        )

    site_index = {site_id: idx for idx, site_id in enumerate(site_ids)}
    positions = kit.atom_positions(site_ids)
    direct_connection_mask = np.zeros((len(site_ids), len(site_ids)), dtype=bool)

    oxygen_owners = {}
    for site_id in site_ids:
        row_index = site_index[site_id]
        for oxygen_id in kit.atom_neighbors(site_id):
            oxygen_owners.setdefault(oxygen_id, []).append(row_index)

    for owners in oxygen_owners.values():
        if len(owners) < 2:
            continue
        owner_indices = np.asarray(owners, dtype=int)
        direct_connection_mask[np.ix_(owner_indices, owner_indices)] = True

    np.fill_diagonal(direct_connection_mask, False)
    return _SlitSiteArrayCache(
        site_ids=site_ids,
        positions=positions,
        site_index=site_index,
        direct_connection_mask=direct_connection_mask,
    )


def _build_slit_site_adjacency(kit, site_ids, distance_range):
    """Build a static neighbor graph for potential siloxane formation.

    Parameters
    ----------
    kit : SilicaSlit
        Slit system under preparation.
    site_ids : list[int]
        Surface silicon identifiers.
    distance_range : tuple[float, float]
        Accepted ``Si-Si`` distance range for siloxane formation.

    Returns
    -------
    adjacency : dict[int, list[tuple[int, float]]]
        Mapping of surface silicon identifiers to sorted neighbor lists.
    """
    adjacency = {site: [] for site in site_ids}
    cache = _build_slit_site_array_cache(kit, site_ids)
    if not cache.site_ids:
        return adjacency

    distances = pairwise_minimum_image_distances(cache.positions, kit.box_nm)
    pair_mask = (
        (distances >= distance_range[0])
        & (distances <= distance_range[1])
        & (~cache.direct_connection_mask)
    )
    pair_mask &= np.triu(np.ones(distances.shape, dtype=bool), k=1)

    row_indices, col_indices = np.where(pair_mask)
    for row_index, col_index in zip(row_indices.tolist(), col_indices.tolist()):
        site_a = cache.site_ids[row_index]
        site_b = cache.site_ids[col_index]
        distance = float(distances[row_index, col_index])
        adjacency[site_a].append((site_b, distance))
        adjacency[site_b].append((site_a, distance))

    for site in adjacency:
        adjacency[site].sort(key=lambda item: (item[1], item[0]))

    return adjacency


def _find_pair(sites, adjacency, first_count, second_count):
    """Find the next eligible siloxane pair for the requested site types.

    Parameters
    ----------
    sites : dict[int, BindingSite]
        Current binding sites keyed by silicon identifier.
    adjacency : dict[int, list[tuple[int, float]]]
        Precomputed slit neighbor graph.
    first_count : int
        Required number of free oxygen atoms on the first site.
    second_count : int
        Required number of free oxygen atoms on the second site.

    Returns
    -------
    pair : tuple[int, int] or None
        Pair of silicon site identifiers, or ``None`` if no eligible pair was
        found.
    """
    for site_a in sorted(sites):
        if sites[site_a].site_type != "in" or sites[site_a].oxygen_count != first_count:
            continue

        for site_b, _distance in adjacency.get(site_a, []):
            if site_b not in sites:
                continue
            if sites[site_b].site_type != "in" or sites[site_b].oxygen_count != second_count:
                continue
            if first_count == second_count and site_b < site_a:
                continue

            return (site_a, site_b)

    return None


def _find_placeable_pair(kit, sites, adjacency, first_count, second_count, rng=None):
    """Find the next eligible siloxane pair with a valid bridge placement.

    Parameters
    ----------
    kit : SilicaSlit
        Slit system under preparation.
    sites : dict[int, BindingSite]
        Current binding sites keyed by silicon identifier.
    adjacency : dict[int, list[tuple[int, float]]]
        Precomputed slit neighbor graph.
    first_count : int
        Required number of free oxygen atoms on the first site.
    second_count : int
        Required number of free oxygen atoms on the second site.
    rng : numpy.random.Generator or None, optional
        Optional generator used to randomize the chemically equivalent site and
        neighbor traversal order.

    Returns
    -------
    result : tuple[tuple[int, int], list[float]] or tuple[None, None]
        Pair of silicon identifiers and the selected bridge position, or
        ``(None, None)`` when no currently placeable pair exists.
    """
    for site_a in _ordered_candidates(sorted(sites), rng):
        if sites[site_a].site_type != "in" or sites[site_a].oxygen_count != first_count:
            continue

        for site_b, _distance in _ordered_candidates(adjacency.get(site_a, []), rng):
            if site_b not in sites:
                continue
            if sites[site_b].site_type != "in" or sites[site_b].oxygen_count != second_count:
                continue
            if first_count == second_count and site_b < site_a:
                continue

            pair = (site_a, site_b)
            bridge_position = _siloxane_bridge_position(kit, pair)
            if bridge_position is not None:
                return pair, bridge_position

    return None, None


def _bridge_candidate_positions(kit, pair):
    """Collect live geometry and return bridge candidates for one silicon pair.

    Parameters
    ----------
    kit : SilicaSlit
        Slit system under preparation.
    pair : tuple[int, int]
        Pair of silicon site identifiers.

    Returns
    -------
    positions : list[list[float]]
        Box-wrapped candidate positions for the bridging oxygen.

    Notes
    -----
    This adapter owns live-system access. Array-only frame construction,
    direction selection, rotations, and wrapping belong to the placement
    module.
    """
    box = kit.box_nm
    pos_a = kit.atom_position(pair[0])
    pos_b = kit.atom_position(pair[1])
    center_pos, axis_unit = _bridge_pair_frame(pos_a, pos_b, box)
    surface_normals = (
        kit.site_normal(pair[0], center_pos),
        kit.site_normal(pair[1], center_pos),
    )
    return _bridge_candidate_positions_from_arrays(
        center_pos,
        axis_unit,
        surface_normals,
        box,
        _BRIDGE_OFFSET_NM,
        _BRIDGE_CANDIDATE_ROTATIONS_DEG,
    )


def _min_clearance_by_atom_ids(system, atom_ids):
    """Return per-atom steric cutoff distances for the selected atoms.

    Parameters
    ----------
    system : SilicaSlit
        Active slit system.
    atom_ids : np.ndarray
        Atom identifiers whose steric cutoff distances should be collected.

    Returns
    -------
    min_distances : np.ndarray
        Per-atom steric cutoffs in nanometers.
    """
    return np.asarray(
        [
            _BRIDGE_MIN_CLEARANCE_BY_TYPE_NM.get(system.atom_type(atom_id), 0.18)
            for atom_id in atom_ids.tolist()
        ],
        dtype=float,
    )


def _build_bridge_steric_cache(kit, pair):
    """Build array-backed steric data for one silicon-pair bridge search.

    Parameters
    ----------
    kit : SilicaSlit
        Slit system under preparation.
    pair : tuple[int, int]
        Pair of silicon site identifiers.

    Returns
    -------
    cache : _BridgeStericCache
        Cached local and global steric-search arrays for ``pair``.
    """
    box = np.asarray(kit.box_nm, dtype=float)
    sites = kit.binding_sites
    consumed_oxygen_ids = {
        sites[pair[0]].oxygen_ids[0],
        sites[pair[1]].oxygen_ids[0],
    }
    excluded_ids = {
        pair[0],
        pair[1],
        *consumed_oxygen_ids,
    }

    frontier = list(pair)
    local_ids = set(pair)
    for _depth in range(_BRIDGE_STERIC_GRAPH_DEPTH):
        next_frontier = []
        for atom_id in frontier:
            for neighbor_id in kit.atom_neighbors(atom_id):
                if neighbor_id not in local_ids:
                    local_ids.add(neighbor_id)
                    next_frontier.append(neighbor_id)
        if not next_frontier:
            break
        frontier = next_frontier

    global_ids = np.asarray(
        [atom_id for atom_id in kit.active_atom_ids() if atom_id not in excluded_ids],
        dtype=int,
    )
    local_ids = np.asarray(
        [atom_id for atom_id in local_ids if atom_id not in excluded_ids],
        dtype=int,
    )

    global_positions = (
        kit.atom_positions(global_ids)
        if global_ids.size
        else np.empty((0, 3), dtype=float)
    )
    local_positions = (
        kit.atom_positions(local_ids)
        if local_ids.size
        else np.empty((0, 3), dtype=float)
    )

    return _BridgeStericCache(
        box=box,
        local_positions=local_positions,
        local_min_distances=_min_clearance_by_atom_ids(kit, local_ids),
        global_positions=global_positions,
        global_min_distances=_min_clearance_by_atom_ids(kit, global_ids),
    )


def _siloxane_bridge_position(kit, pair):
    """Return the least crowded bridge-oxygen position for one silicon pair.

    Parameters
    ----------
    kit : SilicaSlit
        Slit system under preparation.
    pair : tuple[int, int]
        Pair of silicon site identifiers.

    Returns
    -------
    position : list[float] or None
        Bridging oxygen position, or ``None`` when no sterically acceptable
        candidate was found for the pair.
    """
    steric_cache = _build_bridge_steric_cache(kit, pair)
    candidate_positions = _bridge_candidate_positions(kit, pair)
    return _best_bridge_position(
        candidate_positions,
        steric_cache,
        _BRIDGE_STERIC_DISTANCE_CUTOFF_NM,
    )


def _bridge_pair(kit, pair, bridge_position=None):
    """Create one siloxane bridge between two specific surface silicon sites.

    Parameters
    ----------
    kit : SilicaSlit
        Slit system under preparation.
    pair : tuple[int, int]
        Pair of silicon site identifiers.
    bridge_position : list[float] or None, optional
        Optional preselected bridge-oxygen position. When omitted, the helper
        searches for the least crowded valid bridge position automatically.

    Returns
    -------
    bridge_count : int
        Number of siloxane bridges created, always one on success.
    """
    sites = kit.binding_sites
    if pair[0] not in sites or pair[1] not in sites:
        raise ValueError("Cannot bridge a silicon pair that is no longer present in the site dictionary.")

    bridge_position = _siloxane_bridge_position(kit, pair) if bridge_position is None else bridge_position
    if bridge_position is None:
        raise ValueError("Cannot bridge a silicon pair without a sterically acceptable bridge-oxygen position.")
    kit.insert_siloxane_bridge(pair, bridge_position)
    return 1


def _consume_pair(adjacency, pair):
    """Remove a bridged silicon pair from the static slit adjacency graph.

    Parameters
    ----------
    adjacency : dict[int, list[tuple[int, float]]]
        Precomputed slit neighbor graph.
    pair : tuple[int, int]
        Silicon identifiers that have already been bridged once.
    """
    site_a, site_b = pair
    adjacency[site_a] = [item for item in adjacency.get(site_a, []) if item[0] != site_b]
    adjacency[site_b] = [item for item in adjacency.get(site_b, []) if item[0] != site_a]


def _refresh_single_slit_tracking(kit, total_surface_si, composition):
    """Refresh slit-only site tracking after Q-state editing or attachment.

    Parameters
    ----------
    kit : SilicaSlit
        Slit system under preparation.
    total_surface_si : int
        Total number of tracked surface silicon atoms.
    composition : SiliconStateComposition
        Current five-state slit surface composition.
    """
    del total_surface_si, composition
    kit.refresh_site_tracking()


def _enforce_surface_target(kit, total_surface_si, target_surface, distance_range, rng=None):
    """Condense the slit surface until the prepared ``Q`` counts are met.

    Parameters
    ----------
    kit : SilicaSlit
        Slit system under preparation.
    total_surface_si : int
        Total number of tracked surface silicon atoms.
    target_surface : SiliconStateComposition
        Bare pre-grafting target surface composition. ``T2/T3`` must be zero.
    distance_range : tuple[float, float]
        Accepted ``Si-Si`` distance range for siloxane formation.
    rng : numpy.random.Generator or None, optional
        Optional generator used to randomize the selection of eligible
        siloxane bridge pairs.

    Returns
    -------
    bridge_count : int
        Number of siloxane bridges introduced.
    """
    adjacency = _build_slit_site_adjacency(kit, kit.interior_site_ids, distance_range)
    bridge_count = 0
    sites = kit.binding_sites
    current_surface = _surface_composition(total_surface_si, sites)

    while current_surface.q3_sites < target_surface.q3_sites:
        pair, bridge_position = _find_placeable_pair(kit, sites, adjacency, 2, 2, rng=rng)
        if pair is None:
            raise ValueError("No remaining Q2/Q2 siloxane pair is available to increase the Q3 population.")

        bridge_count += _bridge_pair(kit, pair, bridge_position=bridge_position)
        _consume_pair(adjacency, pair)
        sites = kit.binding_sites
        current_surface = _surface_composition(total_surface_si, sites)

    if current_surface.q2_sites < target_surface.q2_sites:
        raise ValueError("The slit surface cannot increase Q3 to the requested value without undershooting the requested Q2 count.")

    while current_surface.q2_sites > target_surface.q2_sites:
        q2_delta = current_surface.q2_sites - target_surface.q2_sites
        pair, bridge_position = _find_placeable_pair(kit, sites, adjacency, 2, 1, rng=rng)

        if pair is None:
            if q2_delta < 2:
                raise ValueError("The slit surface cannot reach the requested Q2 count with the available siloxane pairs.")
            pair, bridge_position = _find_placeable_pair(kit, sites, adjacency, 2, 2, rng=rng)
            if pair is None:
                raise ValueError("No remaining Q2/Q2 siloxane pair is available to reduce the Q2 population.")

        bridge_count += _bridge_pair(kit, pair, bridge_position=bridge_position)
        _consume_pair(adjacency, pair)
        sites = kit.binding_sites
        current_surface = _surface_composition(total_surface_si, sites)

    while current_surface.q3_sites > target_surface.q3_sites:
        if (current_surface.q3_sites - target_surface.q3_sites) < 2:
            raise ValueError("The requested Q3 count is incompatible with the siloxane editing parity constraints.")

        pair, bridge_position = _find_placeable_pair(kit, sites, adjacency, 1, 1, rng=rng)
        if pair is None:
            raise ValueError("No remaining Q3/Q3 siloxane pair is available to reach the requested Q3 count.")

        bridge_count += _bridge_pair(kit, pair, bridge_position=bridge_position)
        _consume_pair(adjacency, pair)
        sites = kit.binding_sites
        current_surface = _surface_composition(total_surface_si, sites)

    if current_surface.q2_sites != target_surface.q2_sites or current_surface.q3_sites != target_surface.q3_sites:
        raise ValueError("The slit surface could not be edited to the requested prepared Q-state composition.")

    _refresh_single_slit_tracking(kit, total_surface_si, current_surface)

    return bridge_count


def _attach_to_specific_sites(
    kit,
    ligand,
    candidate_site_ids,
    requested_count,
    allow_geminal,
    steric_settings=None,
    progress_bar=None,
    progress_context=None,
):
    """Attach one silane family to a deterministic batch of candidate sites.

    Parameters
    ----------
    kit : SilicaSlit
        Slit system under preparation.
    ligand : SilaneAttachmentConfig
        Silane attachment settings.
    candidate_site_ids : list[int]
        Ordered site ids considered for batched attachment.
    requested_count : int
        Number of attachments requested from ``candidate_site_ids``.
    allow_geminal : bool
        Forwarded geminal-allowance flag for the internal attachment helper.
    steric_settings : FunctionalizedSlitStericConfig or None, optional
        Slit-only steric acceptance settings. When ``None``, the default
        relaxed slit steric settings are used.
    progress_bar : object or None, optional
        Progress-bar-like object updated once per requested site.
    progress_context : _AttachmentPhaseProgressContext or None, optional
        Optional metadata used to format live progress text for the current
        attachment batch.

    Returns
    -------
    mols : list
        Molecules accepted by :meth:`silicams.SilicaSlit.attach_ligands`.
    """
    if requested_count <= 0 or not candidate_site_ids:
        if progress_bar is not None:
            progress_bar.close()
        return []

    steric_settings = (
        FunctionalizedSlitStericConfig()
        if steric_settings is None
        else steric_settings
    )
    progress_bar = (
        _NullProgressBar(total=requested_count)
        if progress_bar is None
        else progress_bar
    )
    progress_context = (
        _AttachmentPhaseProgressContext(
            phase_name="Attachment",
            requested_count=requested_count,
        )
        if progress_context is None
        else progress_context
    )

    try:
        progress_bar.set_description_str(
            _attachment_progress_description(progress_context, attached_count=0)
        )

        def _progress_callback(requested_count, attached_count, success):
            """Update the attachment progress bar after one requested slot.

            Parameters
            ----------
            requested_count : int
                One-based number of attempted attachment slots so far.
            attached_count : int
                Number of successfully attached molecules so far.
            success : bool
                Whether the latest requested slot produced one attachment.
            """
            del success
            progress_bar.update(1)
            progress_bar.set_description_str(
                _attachment_progress_description(
                    progress_context,
                    attached_count=attached_count,
                )
            )

        attachment_result = kit.attach_ligands(
            molecule=ligand.molecule,
            mount=ligand.mount,
            axis=ligand.axis,
            site_ids=candidate_site_ids,
            requested_count=requested_count,
            allow_geminal=allow_geminal,
            rotate_about_axis=ligand.rotate_about_axis,
            rotate_step_deg=ligand.rotate_step_deg,
            check_sterics=steric_settings.enabled,
            steric_clearance_scale=steric_settings.clearance_scale,
            progress_callback=_progress_callback,
        )
    finally:
        progress_bar.close()

    return list(attachment_result.molecules)


def _available_site_ids(system, oxygen_count):
    """Return sorted available interior site ids for one oxygen-count class.

    Parameters
    ----------
    system : SilicaSlit
        Current slit system.
    oxygen_count : int
        Required number of oxygen handles on the surface site.

    Returns
    -------
    site_ids : list[int]
        Sorted interior site ids matching the requested oxygen count and still
        available for attachment.
    """
    return list(system.available_site_ids(oxygen_count=oxygen_count))


def _realize_surface_target(
    base_system,
    total_surface_si,
    initial_surface,
    target,
    exact_target,
    tolerance,
    distance_range,
    ligand=None,
    steric_settings=None,
    progress_tracker=None,
    rng=None,
):
    """Select and realize a compatible final slit-surface composition.

    Parameters
    ----------
    base_system : SilicaSlit
        Prepared slit system before custom siloxane formation.
    total_surface_si : int
        Total number of tracked surface silicon atoms.
    initial_surface : SiliconStateComposition
        Initial surface composition before custom condensation.
    target : SiliconStateFractions
        Requested surface-only five-state fractions.
    exact_target : SiliconStateComposition
        Preferred exact integer target derived from the requested fractions.
    tolerance : float
        Allowed absolute fraction deviation per silicon state for fallback
        target selection.
    distance_range : tuple[float, float]
        Accepted ``Si-Si`` distance range for siloxane formation.
    ligand : SilaneAttachmentConfig or None, optional
        Optional silane attachment definition used to realize ``T2`` and
        ``T3``.
    steric_settings : FunctionalizedSlitStericConfig or None, optional
        Slit-only steric acceptance settings used for the deterministic
        functionalized attachment path.
    progress_tracker : _FunctionalizedProgressTracker or None, optional
        Optional outer workflow progress tracker used to update live stage
        descriptions and inner attachment bars.
    rng : numpy.random.Generator or None, optional
        Optional generator used to randomize chemically equivalent siloxane
        pair and graft-site choices.

    Returns
    -------
    attempt : _SurfaceTargetAttempt
        Successfully realized slit surface and selected target metadata.

    Raises
    ------
    ValueError
        Raised when no exact or tolerance-compatible target can be realized on
        the current slit.
    """
    candidate_specs = [(exact_target, False)]
    candidate_specs.extend(
        (candidate.composition, True)
        for candidate in _surface_target_candidates(
            total_surface_si,
            target,
            exact_target,
            tolerance,
        )
    )

    total_candidates = len(candidate_specs)

    for candidate_index, (candidate_surface, used_tolerance) in enumerate(candidate_specs, start=1):
        prepared_target = _prepared_target_from_final(candidate_surface)
        if not _prepared_target_is_compatible(initial_surface, prepared_target):
            continue

        trial_system = base_system.clone()
        _set_candidate_stage(
            progress_tracker,
            "Q-state preparation",
            candidate_index=candidate_index,
            total_candidates=total_candidates,
        )
        q_state_start = perf_counter()
        try:
            bridge_count = _enforce_surface_target(
                trial_system,
                total_surface_si,
                prepared_target,
                distance_range,
                rng=rng,
            )
        except ValueError:
            continue
        q_state_preparation_s = perf_counter() - q_state_start

        prepared_surface = _surface_composition(
            total_surface_si,
            trial_system.binding_sites,
        )
        t2_attachment_s = 0.0
        t3_attachment_s = 0.0

        if ligand is not None:
            geminal_sites = _ordered_candidates(
                _available_site_ids(trial_system, oxygen_count=2),
                rng,
            )
            if len(geminal_sites) < candidate_surface.t2_sites:
                continue
            t2_bar = None
            _set_candidate_stage(
                progress_tracker,
                "T2 attachment",
                candidate_index=candidate_index,
                total_candidates=total_candidates,
            )
            t2_context = _AttachmentPhaseProgressContext(
                phase_name="T2 attachment",
                requested_count=candidate_surface.t2_sites,
                candidate_index=candidate_index,
                total_candidates=total_candidates,
            )
            if progress_tracker is not None and candidate_surface.t2_sites:
                t2_bar = progress_tracker.site_bar(
                    _attachment_progress_description(t2_context, attached_count=0),
                    candidate_surface.t2_sites,
                )
            t2_start = perf_counter()
            attached_t2_molecules = _attach_to_specific_sites(
                trial_system,
                ligand,
                geminal_sites,
                candidate_surface.t2_sites,
                allow_geminal=True,
                steric_settings=steric_settings,
                progress_bar=t2_bar,
                progress_context=t2_context,
            )
            t2_attachment_s = perf_counter() - t2_start
            if len(attached_t2_molecules) < candidate_surface.t2_sites:
                continue

            single_sites = _ordered_candidates(
                _available_site_ids(trial_system, oxygen_count=1),
                rng,
            )
            if len(single_sites) < candidate_surface.t3_sites:
                continue
            t3_bar = None
            _set_candidate_stage(
                progress_tracker,
                "T3 attachment",
                candidate_index=candidate_index,
                total_candidates=total_candidates,
            )
            t3_context = _AttachmentPhaseProgressContext(
                phase_name="T3 attachment",
                requested_count=candidate_surface.t3_sites,
                candidate_index=candidate_index,
                total_candidates=total_candidates,
            )
            if progress_tracker is not None and candidate_surface.t3_sites:
                t3_bar = progress_tracker.site_bar(
                    _attachment_progress_description(t3_context, attached_count=0),
                    candidate_surface.t3_sites,
                )
            t3_start = perf_counter()
            attached_t3_molecules = _attach_to_specific_sites(
                trial_system,
                ligand,
                single_sites,
                candidate_surface.t3_sites,
                allow_geminal=False,
                steric_settings=steric_settings,
                progress_bar=t3_bar,
                progress_context=t3_context,
            )
            t3_attachment_s = perf_counter() - t3_start
            if len(attached_t3_molecules) < candidate_surface.t3_sites:
                continue

        attached_t2, attached_t3 = _attached_state_counts(trial_system, ligand)
        final_surface = _surface_composition(
            total_surface_si,
            trial_system.binding_sites,
            t2_sites=attached_t2,
            t3_sites=attached_t3,
        )
        if final_surface != candidate_surface:
            continue

        _refresh_single_slit_tracking(trial_system, total_surface_si, final_surface)
        return _SurfaceTargetAttempt(
            system=trial_system,
            target_surface=candidate_surface,
            prepared_surface=prepared_surface,
            final_surface=final_surface,
            siloxane_bridges=bridge_count,
            used_surface_tolerance=used_tolerance,
            timing_summary=SlitTimingSummary(
                q_state_preparation_s=q_state_preparation_s,
                t2_attachment_s=t2_attachment_s,
                t3_attachment_s=t3_attachment_s,
            ),
        )

    raise ValueError(
        "The slit surface could not be edited to the requested silicon-state composition within the allowed tolerance."
    )


def _build_base_slit_system(config):
    """Build the base amorphous slit before target realization.

    Parameters
    ----------
    config : AmorphousSlitConfig
        Slit preparation configuration.

    Returns
    -------
    build : _BaseSlitBuild
        Base slit system together with the initial surface metadata.

    Raises
    ------
    ValueError
        Raised when the generated slit unexpectedly contains exterior sites.
    """
    base = Molecule(inp=_amorphous_template_path())
    replicated = _replicate_along_y(base, config.repeat_y)

    dice = Dice(replicated, 0.4, True)
    matrix = Matrix(
        dice.find(None, ["Si", "O"], list(config.amorph_bond_range_nm))
    )
    _duplicate_template_splits(
        matrix,
        base.get_num(),
        config.repeat_y,
        config.template_split_pairs,
    )
    system = SilicaSlit._from_block(
        replicated,
        matrix,
        slit_width_nm=config.slit_width_nm,
        name=config.name,
    )

    total_surface_si = len(system.interior_site_ids)
    initial_surface = _surface_composition(total_surface_si, system.binding_sites)
    _refresh_single_slit_tracking(system, total_surface_si, initial_surface)

    return _BaseSlitBuild(
        system=system,
        total_surface_si=total_surface_si,
        initial_surface=initial_surface,
    )


def _build_report(
    config,
    derived_surface_target,
    target_attempt,
    initial_surface,
    steric_settings=None,
    timing_summary=None,
):
    """Create a slit preparation report for a bare or functionalized build.

    Parameters
    ----------
    config : AmorphousSlitConfig
        Base slit configuration.
    derived_surface_target : SiliconStateFractions
        Surface-only fractions derived from the experimental target.
    target_attempt : _SurfaceTargetAttempt
        Successful target realization payload.
    initial_surface : SiliconStateComposition
        Surface composition before custom condensation.
    steric_settings : FunctionalizedSlitStericConfig or None, optional
        Permissive contact settings used during ligand placement. Bare-slit
        builds leave this as ``None``.
    timing_summary : SlitTimingSummary or None, optional
        Timing summary to store in the report. When omitted, the timings
        collected inside ``target_attempt`` are used.

    Returns
    -------
    report : SlitPreparationReport
        Report summarizing the slit build.
    """
    system = target_attempt.system
    diagnostics = system.preparation_diagnostics
    timing_summary = (
        target_attempt.timing_summary
        if timing_summary is None
        else timing_summary
    )

    return SlitPreparationReport(
        name=config.name,
        temperature_k=config.temperature_k,
        requested_slit_width_nm=config.slit_width_nm,
        slit_geometry=system.geometry,
        site_ex=0,
        siloxane_bridges=target_attempt.siloxane_bridges,
        siloxane_distance_range_nm=tuple(config.siloxane_distance_range_nm),
        surface_fraction_tolerance=config.surface_fraction_tolerance,
        random_seed=config.random_seed,
        used_surface_tolerance=target_attempt.used_surface_tolerance,
        experimental_target=config.surface_target,
        derived_surface_target=derived_surface_target,
        initial_surface=initial_surface,
        target_surface=target_attempt.target_surface,
        prepared_surface=target_attempt.prepared_surface,
        final_surface=target_attempt.final_surface,
        preparation_diagnostics=diagnostics,
        functionalization_steric_settings=steric_settings,
        timing_summary=timing_summary,
    )


@dataclass(frozen=True)
class AmorphousSlitBuilder:
    """Build bare or functionalized periodic amorphous silica slits.

    Parameters
    ----------
    config : AmorphousSlitConfig
        Base slit geometry, surface target, and silica-topology settings.
    """

    config: AmorphousSlitConfig

    def prepare(self):
        """Prepare a bare attach-ready slit.

        Returns
        -------
        result : SlitPreparationResult
            Prepared bare slit, report, and resolved silica topology.
        """
        return _prepare_bare_amorphous_slit_surface(self.config)

    def prepare_functionalized(
        self,
        ligand,
        steric_settings=None,
        progress_settings=None,
    ):
        """Prepare an exactly targeted functionalized slit.

        Parameters
        ----------
        ligand : SilaneAttachmentConfig
            Ligand geometry and optional topology bundle.
        steric_settings : FunctionalizedSlitStericConfig or None, optional
            Slit attachment steric settings. Defaults to the standard values.
        progress_settings : FunctionalizedSlitProgressConfig or None, optional
            Progress-display settings. Defaults to automatic display mode.

        Returns
        -------
        result : FunctionalizedSlitResult
            Prepared functionalized slit and associated report.
        """
        config = FunctionalizedAmorphousSlitConfig(
            slit_config=self.config,
            ligand=ligand,
            steric_settings=(
                FunctionalizedSlitStericConfig()
                if steric_settings is None
                else steric_settings
            ),
            progress_settings=(
                FunctionalizedSlitProgressConfig()
                if progress_settings is None
                else progress_settings
            ),
        )
        progress_tracker = _FunctionalizedProgressTracker(
            total_stages=4,
            progress_config=config.progress_settings,
        )
        try:
            return _prepare_functionalized_amorphous_slit_surface(
                config,
                progress_tracker=progress_tracker,
            )
        finally:
            progress_tracker.close()


def prepare_amorphous_slit_surface(config):
    """Prepare a bare amorphous slit through :class:`AmorphousSlitBuilder`.

    Parameters
    ----------
    config : AmorphousSlitConfig
        Bare slit configuration, including the required physical
        surface-silicon fraction.

    Returns
    -------
    result : SlitPreparationResult
        Prepared attach-ready bare slit and report.
    """
    return AmorphousSlitBuilder(config).prepare()


def _prepare_bare_amorphous_slit_surface(config):
    """Prepare a bare slit from a physically mapped experimental target.

    Parameters
    ----------
    config : AmorphousSlitConfig
        Bare slit preparation configuration.

    Returns
    -------
    result : SlitPreparationResult
        Attach-ready bare slit system, its preparation report, and the
        resolved silica topology model that would be used for full-slab slit
        topology export. Bare charge-neutrality diagnostics are only populated
        after finalized export.

    Raises
    ------
    ValueError
        Raised when the provided target contains non-zero ``T2/T3`` fractions
        or when the slit cannot realize the requested bare surface.

    Examples
    --------
    >>> import silicams as sms
    >>> config = sms.AmorphousSlitConfig(
    ...     name="bare_slit",
    ...     slit_width_nm=7.0,
    ...     repeat_y=2,
    ...     surface_target=sms.ExperimentalSiliconStateTarget(
    ...         q2_fraction=0.0170,
    ...         q3_fraction=0.1675,
    ...         surface_silicon_fraction=0.60,
    ...     ),
    ... )
    >>> result = sms.prepare_amorphous_slit_surface(config)
    >>> _ = result.report.final_surface
    >>> _ = result.silica_topology.to_yaml()
    """
    silica_topology = resolve_silica_topology(config)
    if config.surface_target.t2_fraction or config.surface_target.t3_fraction:
        raise ValueError("Bare slit preparation requires t2_fraction == 0 and t3_fraction == 0.")

    build = _build_base_slit_system(config)
    derived_surface_target = _surface_target_from_experimental(config.surface_target)
    exact_target = _nearest_integer_composition(build.total_surface_si, derived_surface_target)
    rng = _realization_rng(config.random_seed)
    target_attempt = _realize_surface_target(
        build.system,
        build.total_surface_si,
        build.initial_surface,
        derived_surface_target,
        exact_target,
        config.surface_fraction_tolerance,
        tuple(config.siloxane_distance_range_nm),
        ligand=None,
        rng=rng,
    )
    report = _build_report(
        config,
        derived_surface_target,
        target_attempt,
        build.initial_surface,
        timing_summary=SlitTimingSummary(),
    )
    return SlitPreparationResult(
        system=target_attempt.system,
        report=report,
        silica_topology=silica_topology,
    )


def prepare_functionalized_amorphous_slit_surface(config):
    """Prepare an exactly targeted functionalized amorphous slit surface.

    Parameters
    ----------
    config : FunctionalizedAmorphousSlitConfig
        Functionalized slit configuration.

    Returns
    -------
    result : FunctionalizedSlitResult
        Attach-ready functionalized slit system, its preparation report, and
        the resolved silica topology model that would be used for full-slab
        slit topology export.

    Examples
    --------
    >>> import silicams as sms
    >>> from silicams.generic import tms
    >>> config = sms.FunctionalizedAmorphousSlitConfig(
    ...     slit_config=sms.AmorphousSlitConfig(
    ...         name="functionalized_slit",
    ...         slit_width_nm=7.0,
    ...         repeat_y=1,
    ...         surface_target=sms.ExperimentalSiliconStateTarget(
    ...             q2_fraction=63 / 20000,
    ...             q3_fraction=648 / 20000,
    ...             surface_silicon_fraction=1.0,
    ...             t2_fraction=3 / 20000,
    ...             t3_fraction=4 / 20000,
    ...         ),
    ...     ),
    ...     ligand=sms.SilaneAttachmentConfig(
    ...         molecule=tms(),
    ...         mount=0,
    ...         axis=(0, 1),
    ...     ),
    ... )
    >>> result = sms.prepare_functionalized_amorphous_slit_surface(config)
    >>> _ = result.report.final_surface
    """
    return AmorphousSlitBuilder(config.slit_config).prepare_functionalized(
        ligand=config.ligand,
        steric_settings=config.steric_settings,
        progress_settings=config.progress_settings,
    )


def _prepare_functionalized_amorphous_slit_surface(config, progress_tracker):
    """Prepare a functionalized slit using an optional shared progress tracker.

    Parameters
    ----------
    config : FunctionalizedAmorphousSlitConfig
        Functionalized slit configuration.
    progress_tracker : _FunctionalizedProgressTracker
        Shared outer workflow progress tracker.

    Returns
    -------
    result : FunctionalizedSlitResult
        Attach-ready functionalized slit system, its preparation report, and
        the resolved silica topology model used for slit topology export.
    """
    slit_config = config.slit_config
    silica_topology = resolve_silica_topology(slit_config)
    progress_tracker.set_stage("Base slit build")
    build_start = perf_counter()
    build = _build_base_slit_system(slit_config)
    base_slit_build_s = perf_counter() - build_start
    progress_tracker.update_stage(1)
    derived_surface_target = _surface_target_from_experimental(
        slit_config.surface_target
    )
    exact_target = _nearest_integer_composition(build.total_surface_si, derived_surface_target)
    rng = _realization_rng(slit_config.random_seed)
    target_attempt = _realize_surface_target(
        build.system,
        build.total_surface_si,
        build.initial_surface,
        derived_surface_target,
        exact_target,
        slit_config.surface_fraction_tolerance,
        tuple(slit_config.siloxane_distance_range_nm),
        ligand=config.ligand,
        steric_settings=config.steric_settings,
        progress_tracker=progress_tracker,
        rng=rng,
    )
    progress_tracker.update_stage(3)
    timing_summary = replace(
        target_attempt.timing_summary,
        base_slit_build_s=base_slit_build_s,
    )
    report = _build_report(
        slit_config,
        derived_surface_target,
        target_attempt,
        build.initial_surface,
        steric_settings=config.steric_settings,
        timing_summary=timing_summary,
    )
    return FunctionalizedSlitResult(
        system=target_attempt.system,
        report=report,
        silica_topology=silica_topology,
    )


def _write_slit_structure_outputs(
    system,
    output_dir,
    write_object_files,
    write_pdb,
    write_pdb_conect,
    write_cif,
    write_cif_bonds,
    validate_connectivity,
):
    """Write shared finalized coordinate, metadata, and object outputs.

    Parameters
    ----------
    system : SilicaSlit
        Finalized slit domain model.
    output_dir : str or os.PathLike
        Output directory.
    write_object_files : bool
        Whether to serialize the shared snapshot and full slit state.
    write_pdb : bool
        Whether to write PDB coordinates.
    write_pdb_conect : bool
        Whether PDB output should contain ``CONECT`` records.
    write_cif : bool
        Whether to write mmCIF coordinates.
    write_cif_bonds : bool
        Whether mmCIF output should contain ``_struct_conn`` rows.
    validate_connectivity : str
        Connectivity-validation mode for coordinate writers.

    Returns
    -------
    snapshot : StructureSnapshot
        Immutable snapshot shared with the topology writer.
    """
    utils.mkdirp(output_dir)
    snapshot = system.export_snapshot()
    writer = StructureWriter(snapshot, output_dir)
    writer.write_gro(
        use_atom_names=True,
        validate_connectivity=validate_connectivity,
    )
    if write_pdb:
        writer.write_pdb(
            use_atom_names=True,
            write_conect=write_pdb_conect,
            validate_connectivity=validate_connectivity,
        )
    if write_cif:
        writer.write_cif(
            use_atom_names=True,
            write_bonds=write_cif_bonds,
            validate_connectivity=validate_connectivity,
        )
    if write_object_files:
        writer.write_object()
        utils.save(
            system,
            os.path.join(output_dir, f"{system.name}_system.obj"),
        )

    metadata_path = os.path.join(output_dir, f"{system.name}.yml")
    with open(metadata_path, "w", encoding="utf-8") as file_out:
        yaml.safe_dump(system.metadata(), file_out, sort_keys=False)
    return snapshot


def write_bare_amorphous_slit(
    output_dir,
    config,
    write_object_files=False,
    write_pdb=False,
    write_pdb_conect=True,
    write_cif=False,
    write_cif_bonds=True,
    validate_connectivity="strict",
):
    """Prepare, finalize, and store a bare amorphous silica slit.

    Parameters
    ----------
    output_dir : str
        Output directory for the generated slit files and JSON report.
    config : AmorphousSlitConfig
        Bare slit preparation configuration, including the required physical
        surface-silicon fraction.
    write_object_files : bool, optional
        When ``True``, also serialize the finalized structural snapshot and
        full :class:`silicams.slit_system.SilicaSlit` state as ``.obj`` files.
        The default is ``False`` so object exports remain an explicit opt-in.
    write_pdb : bool, optional
        When ``True``, also write a PDB structure file for inspection.
    write_pdb_conect : bool, optional
        When ``True`` (the default), emit inspection-oriented ``CONECT``
        records in the written PDB file whenever PDB output is requested.
    write_cif : bool, optional
        When ``True``, also write an mmCIF structure file for inspection.
    write_cif_bonds : bool, optional
        When ``True`` (the default), emit an inspection-oriented
        ``_struct_conn`` loop in the written mmCIF file whenever mmCIF output
        is requested.
    validate_connectivity : str, optional
        Connectivity validation mode forwarded to structure writers.
        Supported values are ``"off"``, ``"warn"``, and ``"strict"``. The
        default is ``"strict"``.

    Returns
    -------
    result : SlitPreparationResult
        Finalized bare slit system and its preparation report. The export
        writes a self-contained full-slab ``.itp`` / ``.top`` pair and
        exposes the resolved silica topology model plus finalized bare-slit
        charge-neutrality diagnostics on ``result``.

    Examples
    --------
    >>> import silicams as sms
    >>> result = sms.write_bare_amorphous_slit(
    ...     "output/bare_amorphous_slit",
    ...     sms.AmorphousSlitConfig(
    ...         surface_target=sms.ExperimentalSiliconStateTarget(
    ...             q2_fraction=0.0170,
    ...             q3_fraction=0.1675,
    ...             surface_silicon_fraction=0.60,
    ...         ),
    ...     ),
    ... )
    >>> _ = result.bare_charge_diagnostics.is_neutral
    """
    result = prepare_amorphous_slit_surface(config=config)
    return _write_prepared_bare_result(
        result=result,
        output_dir=output_dir,
        write_object_files=write_object_files,
        write_pdb=write_pdb,
        write_pdb_conect=write_pdb_conect,
        write_cif=write_cif,
        write_cif_bonds=write_cif_bonds,
        validate_connectivity=validate_connectivity,
    )


def _write_prepared_bare_result(
    result,
    output_dir,
    write_object_files=False,
    write_pdb=False,
    write_pdb_conect=True,
    write_cif=False,
    write_cif_bonds=True,
    validate_connectivity="strict",
):
    """Finalize and export an already prepared bare slit result.

    This internal boundary lets tests and higher-level workflows separate the
    expensive scientific preparation stage from deterministic serialization.
    The supplied result is finalized in place and returned. All requested files
    are staged and promoted together after serialization succeeds.

    Parameters
    ----------
    result : SlitPreparationResult
        Prepared, unfinalized bare-slit result to export.
    output_dir : str or os.PathLike
        Directory receiving structure, topology, metadata, and report files.
    write_object_files : bool, optional
        Whether to serialize the structural snapshot and full slit state.
    write_pdb : bool, optional
        Whether to write PDB coordinates.
    write_pdb_conect : bool, optional
        Whether PDB output includes ``CONECT`` records.
    write_cif : bool, optional
        Whether to write mmCIF coordinates.
    write_cif_bonds : bool, optional
        Whether mmCIF output includes ``_struct_conn`` rows.
    validate_connectivity : str, optional
        Connectivity-validation mode forwarded to coordinate writers. The
        default is ``"strict"``.

    Returns
    -------
    SlitPreparationResult
        The finalized result with bare-slit charge diagnostics populated.
    """

    result.system.finalize()
    with staged_output_directory(output_dir) as staged_output_dir:
        snapshot = _write_slit_structure_outputs(
            result.system,
            staged_output_dir,
            write_object_files,
            write_pdb,
            write_pdb_conect,
            write_cif,
            write_cif_bonds,
            validate_connectivity,
        )
        topology_writer = GromacsTopologyWriter(snapshot, staged_output_dir)
        result.bare_charge_diagnostics = topology_writer.bare_charge_diagnostics(
            silica_topology=result.silica_topology,
        )
        topology_writer.write_full_slit(
            silica_topology=result.silica_topology,
        )

        report_path = staged_output_dir / f"{result.report.name}_report.json"
        with open(report_path, "w") as file_out:
            json.dump(asdict(result.report), file_out, indent=2)

    return result


def write_functionalized_amorphous_slit(
    output_dir,
    config,
    write_object_files=False,
    write_pdb=False,
    write_pdb_conect=True,
    write_cif=False,
    write_cif_bonds=True,
    validate_connectivity="strict",
):
    """Prepare, finalize, and store a functionalized amorphous silica slit.

    Parameters
    ----------
    output_dir : str
        Output directory for the generated slit files and JSON report.
    config : FunctionalizedAmorphousSlitConfig
        Functionalized slit preparation configuration.
    write_object_files : bool, optional
        When ``True``, also serialize the finalized structural snapshot and
        full :class:`silicams.slit_system.SilicaSlit` state as ``.obj`` files.
        The default is ``False`` so object exports remain an explicit opt-in.
    write_pdb : bool, optional
        When ``True``, also write a PDB structure file for inspection.
    write_pdb_conect : bool, optional
        When ``True`` (the default), emit inspection-oriented ``CONECT``
        records in the written PDB file whenever PDB output is requested.
    write_cif : bool, optional
        When ``True``, also write an mmCIF structure file for inspection.
    write_cif_bonds : bool, optional
        When ``True`` (the default), emit an inspection-oriented
        ``_struct_conn`` loop in the written mmCIF file whenever mmCIF output
        is requested.
    validate_connectivity : str, optional
        Connectivity validation mode forwarded to structure writers.
        Supported values are ``"off"``, ``"warn"``, and ``"strict"``. The
        default is ``"strict"``.

    Returns
    -------
    result : FunctionalizedSlitResult
        Finalized functionalized slit system and its preparation report.
        When explicit flat ligand topology input is available, the export
        writes a self-contained full-slab ``.itp`` / ``.top`` pair and
        exposes the resolved silica topology model plus functionalized charge
        diagnostics on ``result``. When no explicit
        :class:`SilaneTopologyConfig` is supplied, the writer still stores the
        finalized coordinates and reports but skips functionalized topology
        files.

    Notes
    -----
    Functionalized full-topology export interprets the supplied flat ITP as
    one self-contained base ``T3`` silane fragment. The atom names in that
    bundle must match ``config.ligand.molecule``, the bundle charge must
    satisfy the active silica-model target, and geminal ``T2`` targets also
    require explicit :class:`SilaneGeminalCrossTerms`.

    Examples
    --------
    Coordinate-only export without a flat ligand topology bundle:

    >>> import silicams as sms
    >>> from silicams.generic import tms
    >>> config = sms.FunctionalizedAmorphousSlitConfig(
    ...     slit_config=sms.AmorphousSlitConfig(
    ...         surface_target=sms.ExperimentalSiliconStateTarget(
    ...             q2_fraction=63 / 20000,
    ...             q3_fraction=648 / 20000,
    ...             surface_silicon_fraction=1.0,
    ...             t2_fraction=3 / 20000,
    ...             t3_fraction=4 / 20000,
    ...         ),
    ...     ),
    ...     ligand=sms.SilaneAttachmentConfig(
    ...         molecule=tms(),
    ...         mount=0,
    ...         axis=(0, 1),
    ...     ),
    ... )
    >>> result = sms.write_functionalized_amorphous_slit(
    ...     "output/functionalized_coordinates",
    ...     config,
    ... )
    >>> _ = (result.charge_diagnostics is None)

    Full-topology export with an explicit base ``T3`` ITP and geminal cross
    terms:

    >>> import silicams as sms
    >>> from silicams.generic import tms
    >>> topology = sms.SilaneTopologyConfig(
    ...     itp_path="path/to/tms_base_t3.itp",
    ...     moleculetype_name="TMS",
    ...     geminal_cross_terms=sms.SilaneGeminalCrossTerms(
    ...         first_ligand_atom_name="O1",
    ...         geminal_oxygen_mount_ligand_angle=sms.GromacsAngleParameters.harmonic(
    ...             angle_deg=105.56,
    ...             force_constant=384.223760,
    ...         ),
    ...         geminal_dihedrals=(
    ...             sms.GeminalMountDihedralSpec(
    ...                 fourth_atom_name="Si2",
    ...                 function=1,
    ...                 parameters=("0.00000", "1.60387", "3"),
    ...             ),
    ...         ),
    ...     ),
    ... )
    >>> config = sms.FunctionalizedAmorphousSlitConfig(
    ...     slit_config=sms.AmorphousSlitConfig(
    ...         surface_target=sms.ExperimentalSiliconStateTarget(
    ...             q2_fraction=63 / 20000,
    ...             q3_fraction=648 / 20000,
    ...             surface_silicon_fraction=1.0,
    ...             t2_fraction=3 / 20000,
    ...             t3_fraction=4 / 20000,
    ...         ),
    ...     ),
    ...     ligand=sms.SilaneAttachmentConfig(
    ...         molecule=tms(),
    ...         mount=0,
    ...         axis=(0, 1),
    ...         topology=topology,
    ...     ),
    ... )
    >>> result = sms.write_functionalized_amorphous_slit(
    ...     "output/functionalized_full_topology",
    ...     config,
    ... )
    >>> _ = result.charge_diagnostics.is_valid
    """
    progress_tracker = _FunctionalizedProgressTracker(
        total_stages=6,
        progress_config=config.progress_settings,
    )
    try:
        result = _prepare_functionalized_amorphous_slit_surface(
            config,
            progress_tracker=progress_tracker,
        )
        return _write_prepared_functionalized_result(
            result=result,
            output_dir=output_dir,
            config=config,
            progress_tracker=progress_tracker,
            write_object_files=write_object_files,
            write_pdb=write_pdb,
            write_pdb_conect=write_pdb_conect,
            write_cif=write_cif,
            write_cif_bonds=write_cif_bonds,
            validate_connectivity=validate_connectivity,
        )
    finally:
        progress_tracker.close()


def _write_prepared_functionalized_result(
    result,
    output_dir,
    config,
    progress_tracker,
    write_object_files=False,
    write_pdb=False,
    write_pdb_conect=True,
    write_cif=False,
    write_cif_bonds=True,
    validate_connectivity="strict",
):
    """Finalize and export an already prepared functionalized slit result.

    This internal boundary separates expensive surface realization from
    deterministic finalization and serialization. The supplied result is
    finalized in place and returned. The caller owns the progress tracker's
    lifecycle. All requested files are staged and promoted together after
    serialization succeeds.

    Parameters
    ----------
    result : FunctionalizedSlitResult
        Prepared, unfinalized functionalized-slit result to export.
    output_dir : str or os.PathLike
        Directory receiving structure, topology, metadata, and report files.
    config : FunctionalizedAmorphousSlitConfig
        Ligand topology and output settings corresponding to ``result``.
    progress_tracker : _FunctionalizedProgressTracker
        Active workflow tracker receiving finalization and export stages.
    write_object_files : bool, optional
        Whether to serialize the structural snapshot and full slit state.
    write_pdb : bool, optional
        Whether to write PDB coordinates.
    write_pdb_conect : bool, optional
        Whether PDB output includes ``CONECT`` records.
    write_cif : bool, optional
        Whether to write mmCIF coordinates.
    write_cif_bonds : bool, optional
        Whether mmCIF output includes ``_struct_conn`` rows.
    validate_connectivity : str, optional
        Connectivity-validation mode forwarded to coordinate writers. The
        default is ``"strict"``.

    Returns
    -------
    FunctionalizedSlitResult
        Finalized result with timing and charge diagnostics populated.
    """

    topology_config = resolve_silane_topology_config(config.ligand)

    progress_tracker.set_stage("Finalize")
    finalize_start = perf_counter()
    result.system.finalize()
    finalize_s = perf_counter() - finalize_start
    progress_tracker.update_stage(1)

    progress_tracker.set_stage("Finalize/export")
    with staged_output_directory(output_dir) as staged_output_dir:
        export_start = perf_counter()
        snapshot = _write_slit_structure_outputs(
            result.system,
            staged_output_dir,
            write_object_files,
            write_pdb,
            write_pdb_conect,
            write_cif,
            write_cif_bonds,
            validate_connectivity,
        )
        if topology_config is not None:
            result.charge_diagnostics = GromacsTopologyWriter(
                snapshot,
                staged_output_dir,
            ).write_full_slit(
                base_ligand_short=config.ligand.molecule.get_short(),
                silane_topology_config=topology_config,
                silica_topology=result.silica_topology,
            )
        export_s = perf_counter() - export_start
        progress_tracker.update_stage(1)
        result.report = replace(
            result.report,
            timing_summary=replace(
                result.report.timing_summary,
                finalize_s=finalize_s,
                export_s=export_s,
            ),
        )

        report_path = staged_output_dir / f"{result.report.name}_report.json"
        with open(report_path, "w") as file_out:
            json.dump(asdict(result.report), file_out, indent=2)

    return result
