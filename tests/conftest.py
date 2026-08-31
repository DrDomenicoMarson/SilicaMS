from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field, replace
import math
from pathlib import Path
import os
import shutil

import pytest
import numpy as np


def _configure_test_environment() -> None:
    """Configure writable cache directories for test-time imports.

    Pytest imports test modules before fixtures run, so Matplotlib/XDG-related
    environment variables must be set during ``conftest`` import.
    """
    env_root = Path("/tmp/silicams_pytest_env")
    xdg_cache = env_root / "xdg_cache"
    mpl_config = env_root / "mpl_config"
    xdg_cache.mkdir(parents=True, exist_ok=True)
    mpl_config.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("MPLBACKEND", "Agg")


_configure_test_environment()


@dataclass(frozen=True)
class ModuleWorkspace:
    """Temporary working tree for one pytest module."""

    root: Path
    data_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class DiceExecutionContext:
    """Shared data for the dice execution tests."""

    dice: object
    search_args: tuple[object, list[str], list[float]]
    expected: list[list[object]]
    repo_root: str


@dataclass(frozen=True)
class BareSlitContext:
    """Shared prepared/stored bare slit artifacts for slit tests."""

    output_dir: Path
    surface_target: object
    config: object
    prepared_result: object
    prepared_report: object
    stored_result: object
    stored_report: object


@dataclass(frozen=True)
class FunctionalizedAttachmentCall:
    """Observed ligand-attachment call made during slit preparation.

    Parameters
    ----------
    requested_count : int or None
        Number of attachments requested in the call.
    site_count : int
        Number of candidate site identifiers supplied.
    allow_geminal : bool or None
        Whether the call accepted geminal attachment sites.
    has_progress_callback : bool
        Whether the call supplied a live progress callback.
    steric_clearance_scale : float or None
        Steric-clearance scaling factor forwarded to the attachment engine.
    """

    requested_count: int | None
    site_count: int
    allow_geminal: bool | None
    has_progress_callback: bool
    steric_clearance_scale: float | None


@dataclass
class ProgressBarTrace:
    """In-memory progress bar used to verify workflow instrumentation.

    Parameters
    ----------
    total : int
        Expected number of updates.
    desc : str
        Initial displayed description.
    unit : str
        Unit label for the progress counter.
    leave : bool
        Whether a rendered progress bar would remain after completion.
    n : int, optional
        Recorded current count.
    closed : bool, optional
        Whether the bar has been closed.
    descriptions : list[str], optional
        Ordered history of displayed descriptions.
    """

    total: int
    desc: str
    unit: str
    leave: bool
    n: int = 0
    closed: bool = False
    descriptions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize the description history."""

        self.descriptions.append(self.desc)

    def update(self, value: int = 1) -> None:
        """Advance the recorded count.

        Parameters
        ----------
        value : int, optional
            Increment added to the current count.
        """

        self.n += value

    def set_description_str(self, desc: str, refresh: bool = True) -> None:
        """Record a new displayed description.

        Parameters
        ----------
        desc : str
            New progress description.
        refresh : bool, optional
            Accepted for compatibility with ``tqdm`` and otherwise ignored.
        """

        del refresh
        self.desc = desc
        self.descriptions.append(desc)

    def close(self) -> None:
        """Record closure of the progress bar."""

        self.closed = True


@dataclass(frozen=True)
class FunctionalizedSlitContext:
    """Shared exact-target functionalized slit used by behavior tests.

    Parameters
    ----------
    config : FunctionalizedAmorphousSlitConfig
        Configuration used for the shared scientific preparation.
    prepared_result : FunctionalizedSlitResult
        Unfinalized result retaining attachment capability.
    finalized_result : FunctionalizedSlitResult
        Mutation-blocked result used for deterministic read-only exports.
    attachment_calls : tuple[FunctionalizedAttachmentCall, ...], optional
        Real attachment calls observed while preparing the result.
    progress_bars : tuple[ProgressBarTrace, ...], optional
        Real progress instrumentation observed during preparation.
    """

    config: object
    prepared_result: object
    finalized_result: object
    attachment_calls: tuple[FunctionalizedAttachmentCall, ...] = ()
    progress_bars: tuple[ProgressBarTrace, ...] = ()

    def clone_result(self) -> object:
        """Return an independent result suitable for finalization or export.

        Returns
        -------
        result : FunctionalizedSlitResult
            Deeply independent mutable system and topology with the immutable
            preparation report reused.
        """

        return replace(
            self.prepared_result,
            system=self.prepared_result.system.clone(),
            silica_topology=copy.deepcopy(self.prepared_result.silica_topology),
            charge_diagnostics=None,
        )

    def copy_finalized_result(self) -> object:
        """Return independent metadata around the shared immutable system.

        Returns
        -------
        result : FunctionalizedSlitResult
            Result that reuses the finalized, mutation-blocked system while
            owning independent topology, report, and diagnostics fields.
        """

        return replace(
            self.finalized_result,
            silica_topology=copy.deepcopy(self.finalized_result.silica_topology),
            charge_diagnostics=None,
        )


@dataclass(frozen=True)
class SmallBareSlitContext:
    """Shared repeat-one bare slit used by size-independent behavior tests.

    Parameters
    ----------
    config : AmorphousSlitConfig
        Configuration used for the shared scientific preparation.
    prepared_result : SlitPreparationResult
        Unfinalized result retaining attachment capability.
    finalized_result : SlitPreparationResult
        Mutation-blocked result used for deterministic read-only exports.
    """

    config: object
    prepared_result: object
    finalized_result: object

    def clone_result(self) -> object:
        """Return an independent prepared bare-slit result.

        Returns
        -------
        result : SlitPreparationResult
            Result with an independent mutable system and topology.
        """

        return replace(
            self.prepared_result,
            system=self.prepared_result.system.clone(),
            silica_topology=copy.deepcopy(self.prepared_result.silica_topology),
            bare_charge_diagnostics=None,
        )


def _gro_atom_line(
    residue_id: int,
    residue_name: str,
    atom_name: str,
    atom_id: int,
    x: float,
    y: float,
    z: float,
) -> str:
    """Return one independently formatted minimal GRO atom line.

    Parameters
    ----------
    residue_id : int
        Residue identifier, wrapped at five digits.
    residue_name : str
        Residue name, truncated to five characters.
    atom_name : str
        Atom name, truncated to five characters.
    atom_id : int
        Atom identifier, wrapped at five digits.
    x, y, z : float
        Cartesian position components in nanometers.

    Returns
    -------
    str
        Newline-terminated atom record with three-decimal coordinate precision.
    """

    return (
        f"{residue_id % 100000:5d}"
        f"{residue_name[:5]:<5}"
        f"{atom_name[:5]:>5}"
        f"{atom_id % 100000:5d}"
        f"{x:8.3f}"
        f"{y:8.3f}"
        f"{z:8.3f}\n"
    )


def _write_gro(
    path: Path,
    atoms: list[tuple[int, str, str, int, float, float, float]],
    box_values: tuple[float, ...],
    title: str = "test",
    velocities: np.ndarray | None = None,
) -> None:
    """Write an independently formatted GRO fixture.

    Parameters
    ----------
    path : Path
        Fixture destination.
    atoms : list[tuple]
        Residue id/name, atom name/id, and three coordinates in nanometers.
    box_values : tuple[float, ...]
        GRO box components in nanometers.
    title : str, optional
        First line of the fixture.
    velocities : ndarray or None, optional
        Per-atom Cartesian velocities, shape ``(N, 3)``, in nm/ps. Omit the
        velocity columns when absent.
    """

    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{title}\n")
        handle.write(f"{len(atoms)}\n")
        for atom_index, atom in enumerate(atoms):
            line = _gro_atom_line(*atom)
            if velocities is not None:
                line = line.rstrip("\n") + "".join(
                    f"{component:8.4f}" for component in velocities[atom_index]
                ) + "\n"
            handle.write(line)
        if len(box_values) == 3:
            handle.write(
                f"{box_values[0]:10.5f}{box_values[1]:10.5f}{box_values[2]:10.5f}\n"
            )
        else:
            handle.write(" ".join(f"{value:10.5f}" for value in box_values) + "\n")


def _surface_residue(
    residue_id: int,
    atom_id_start: int,
    x: float,
    y: float,
    z: float,
    residue_name: str = "SUR",
) -> list[tuple[int, str, str, int, float, float, float]]:
    """Return one hydroxylated residue used for slit-plane inference.

    Parameters
    ----------
    residue_id : int
        Residue identifier shared by the Si, O, and H records.
    atom_id_start : int
        First of three sequential atom identifiers.
    x, y, z : float
        Silicon position in nanometers; O and H are offset along x.
    residue_name : str, optional
        Residue name for all three records.

    Returns
    -------
    list[tuple]
        Atom records accepted by the shared GRO fixture writer.
    """

    return [
        (residue_id, residue_name, "SI1", atom_id_start, x, y, z),
        (residue_id, residue_name, "O1", atom_id_start + 1, x + 0.05, y, z),
        (residue_id, residue_name, "H1", atom_id_start + 2, x + 0.08, y, z),
    ]


def _ring_residue(
    residue_id: int,
    residue_name: str,
    atom_id_start: int,
    center: tuple[float, float, float],
    radius: float = 0.14,
) -> list[tuple[int, str, str, int, float, float, float]]:
    """Return one six-atom aromatic ring residue in the xy plane.

    Parameters
    ----------
    residue_id : int
        Residue identifier shared by all ring atoms.
    residue_name : str
        Residue name shared by all ring atoms.
    atom_id_start : int
        First of six sequential atom identifiers.
    center : tuple[float, float, float]
        Ring center in nanometers.
    radius : float, optional
        Ring radius in nanometers.

    Returns
    -------
    list[tuple]
        Atom records named CA1 through CA6 for the shared GRO fixture writer.
    """

    atoms = []
    center_x, center_y, center_z = center
    for index, angle_deg in enumerate(range(0, 360, 60), start=1):
        angle_rad = math.radians(angle_deg)
        atoms.append(
            (
                residue_id,
                residue_name,
                f"CA{index}",
                atom_id_start + index - 1,
                center_x + radius * math.cos(angle_rad),
                center_y + radius * math.sin(angle_rad),
                center_z,
            )
        )
    return atoms


def _bond_residue(
    residue_id: int,
    residue_name: str,
    atom_id_start: int,
    start: tuple[float, float, float],
    stop: tuple[float, float, float],
) -> list[tuple[int, str, str, int, float, float, float]]:
    """Return one simple two-carbon bond residue.

    Parameters
    ----------
    residue_id : int
        Residue identifier shared by both atoms.
    residue_name : str
        Residue name shared by both atoms.
    atom_id_start : int
        First of two sequential atom identifiers.
    start, stop : tuple[float, float, float]
        Carbon positions defining the segment endpoints in nanometers.

    Returns
    -------
    list[tuple]
        Two atom records accepted by the shared GRO fixture writer.
    """

    return [
        (residue_id, residue_name, "C1", atom_id_start, *start),
        (residue_id, residue_name, "C2", atom_id_start + 1, *stop),
    ]


def _write_basic_slit(path: Path, include_ring: bool = False, include_crossing_bond: bool = False) -> None:
    """Write a 2 nm cubic slit fixture with two hydroxylated surface residues.

    Parameters
    ----------
    path : Path
        GRO fixture destination.
    include_ring : bool, optional
        Add an aromatic ring centered at (1, 1, 1) nm.
    include_crossing_bond : bool, optional
        Add a two-carbon segment passing through the same center along z.
    """

    atoms: list[tuple[int, str, str, int, float, float, float]] = []
    atoms.extend(_surface_residue(1, 1, 0.2, 0.5, 0.5))
    atoms.extend(_surface_residue(2, 4, 1.8, 1.5, 1.5))
    next_atom_id = 7
    if include_ring:
        atoms.extend(_ring_residue(3, "SLR", next_atom_id, (1.0, 1.0, 1.0), radius=0.12))
        next_atom_id += 6
    if include_crossing_bond:
        atoms.extend(
            _bond_residue(
                4,
                "BND",
                next_atom_id,
                (1.0, 1.0, 0.92),
                (1.0, 1.0, 1.08),
            )
        )
    _write_gro(path, atoms, (2.0, 2.0, 2.0), title="slit")


def _write_guest_box(
    path: Path,
    include_inside_ring: bool = True,
    include_outside_ring: bool = True,
    include_crossing_bond: bool = False,
    ring_atom_count: int = 6,
    residue_name: str = "THY",
) -> None:
    """Write a 3 nm cubic guest reservoir for centered cropping tests.

    Parameters
    ----------
    path : Path
        GRO fixture destination.
    include_inside_ring : bool, optional
        Include a target residue centered at (1.5, 1.5, 1.5) nm.
    include_outside_ring : bool, optional
        Include a target residue centered outside the centered 2 nm crop.
    include_crossing_bond : bool, optional
        Append a two-carbon segment to the inside residue.
    ring_atom_count : int, optional
        Number of inside-ring atoms retained, up to six.
    residue_name : str, optional
        Residue name used for guest records.
    """

    atoms: list[tuple[int, str, str, int, float, float, float]] = []
    atom_id = 1
    if include_inside_ring:
        ring_atoms = _ring_residue(1, residue_name, atom_id, (1.5, 1.5, 1.5))
        if ring_atom_count < 6:
            ring_atoms = ring_atoms[:ring_atom_count]
        atoms.extend(ring_atoms)
        atom_id += len(ring_atoms)
        if include_crossing_bond:
            atoms.extend(
                _bond_residue(
                    1,
                    residue_name,
                    atom_id,
                    (1.0, 1.0, 0.92),
                    (1.0, 1.0, 1.08),
                )
            )
            atom_id += 2
    if include_outside_ring:
        atoms.extend(_ring_residue(2, residue_name, atom_id, (0.2, 1.5, 1.5)))
    _write_gro(path, atoms, (3.0, 3.0, 3.0), title="guest")


@pytest.fixture
def write_gro() -> Callable[..., None]:
    """Return the shared, independently formatted GRO fixture writer."""

    return _write_gro


@pytest.fixture
def surface_residue() -> Callable[..., list[tuple]]:
    """Return the shared hydroxylated surface-residue builder."""

    return _surface_residue


@pytest.fixture
def ring_residue() -> Callable[..., list[tuple]]:
    """Return the shared ring residue fixture builder."""

    return _ring_residue


@pytest.fixture
def bond_residue() -> Callable[..., list[tuple]]:
    """Return the shared bond residue fixture builder."""

    return _bond_residue


@pytest.fixture
def write_basic_slit() -> Callable[..., None]:
    """Return the shared write basic slit fixture builder."""

    return _write_basic_slit


@pytest.fixture
def write_guest_box() -> Callable[..., None]:
    """Return the shared write guest box fixture builder."""

    return _write_guest_box


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root directory."""

    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def tests_dir() -> Path:
    """Return the tests directory."""

    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def data_dir(tests_dir: Path) -> Path:
    """Return the shared test-data directory."""

    return tests_dir / "data"


@pytest.fixture(scope="module")
def module_workspace(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
    data_dir: Path,
) -> ModuleWorkspace:
    """Create one isolated module workspace with copied test data."""

    workspace_root = tmp_path_factory.mktemp(request.module.__name__.split(".")[-1])
    workspace_data = workspace_root / "data"
    shutil.copytree(data_dir, workspace_data)
    workspace_output = workspace_root / "output"
    workspace_output.mkdir()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(workspace_root)
    try:
        yield ModuleWorkspace(
            root=workspace_root,
            data_dir=workspace_data,
            output_dir=workspace_output,
        )
    finally:
        monkeypatch.undo()


@pytest.fixture(scope="session")
def dice_execution_context(repo_root: Path) -> DiceExecutionContext:
    """Build the shared dice-search context used by dice execution tests."""

    from silicams.dice import Dice
    from silicams.pattern import BetaCristobalit

    block = BetaCristobalit().generate([2, 2, 2], "z")
    dice = Dice(block, 0.2, True)
    search_args = (None, ["Si", "O"], [0.155 - 1e-2, 0.155 + 1e-2])
    expected = sorted(
        [[entry[0], sorted(entry[1])] for entry in dice.find(*search_args)],
        key=lambda entry: entry[0],
    )

    return DiceExecutionContext(
        dice=dice,
        search_args=search_args,
        expected=expected,
        repo_root=str(repo_root),
    )


@pytest.fixture(scope="session")
def bare_slit_context(tmp_path_factory: pytest.TempPathFactory) -> BareSlitContext:
    """Prepare one shared bare slit and export an independent clone."""

    import silicams as sms
    import silicams.slit as slit_mod

    output_dir = tmp_path_factory.mktemp("bare_amorphous_slit_preparation")
    surface_target = sms.ExperimentalSiliconStateTarget(
        q2_fraction=0.069,
        q3_fraction=0.681,
        surface_silicon_fraction=1.0,
    )
    config = sms.AmorphousSlitConfig(
        name="test_bare_amorphous_slit",
        surface_target=surface_target,
    )
    prepared_result = sms.prepare_amorphous_slit_surface(config=config)
    stored_result = replace(
        prepared_result,
        system=prepared_result.system.clone(),
        silica_topology=copy.deepcopy(prepared_result.silica_topology),
        bare_charge_diagnostics=None,
    )
    stored_result = slit_mod._write_prepared_bare_result(
        result=stored_result,
        output_dir=str(output_dir),
    )

    return BareSlitContext(
        output_dir=output_dir,
        surface_target=surface_target,
        config=config,
        prepared_result=prepared_result,
        prepared_report=prepared_result.report,
        stored_result=stored_result,
        stored_report=stored_result.report,
    )


@pytest.fixture(scope="session")
def functionalized_slit_context() -> FunctionalizedSlitContext:
    """Prepare one exact-target TMS slit for independent behavior checks.

    Returns
    -------
    context : FunctionalizedSlitContext
        Reusable prepared/finalized results plus real workflow traces.
    """

    import silicams as sms
    import silicams.slit as slit_mod
    from silicams import generic

    target = sms.ExperimentalSiliconStateTarget(
        q2_fraction=63 / 957,
        q3_fraction=648 / 957,
        q4_fraction=239 / 957,
        t2_fraction=3 / 957,
        t3_fraction=4 / 957,
        surface_silicon_fraction=1.0,
    )
    config = sms.FunctionalizedAmorphousSlitConfig(
        slit_config=sms.AmorphousSlitConfig(
            name="functionalized_exact_slit",
            repeat_y=1,
            surface_target=target,
        ),
        ligand=sms.SilaneAttachmentConfig(
            molecule=generic.tms(),
            mount=0,
            axis=(0, 1),
            rotate_about_axis=False,
        ),
        steric_settings=sms.FunctionalizedSlitStericConfig(clearance_scale=0.55),
        progress_settings=sms.FunctionalizedSlitProgressConfig(enabled=True),
    )
    attachment_calls = []
    progress_bars = []
    original_attach = sms.SilicaSlit.attach_ligands

    def recording_attach(system, *args, **kwargs):
        """Record attachment arguments while running the real implementation.

        Parameters
        ----------
        system : SilicaSlit
            Slit receiving the attachment request.
        *args : object
            Positional arguments forwarded to ``attach_ligands``.
        **kwargs : object
            Keyword arguments forwarded to ``attach_ligands``.

        Returns
        -------
        result : SlitAttachmentResult
            Result returned by the real attachment implementation.
        """

        attachment_calls.append(
            FunctionalizedAttachmentCall(
                requested_count=kwargs.get("requested_count"),
                site_count=len(kwargs["site_ids"]),
                allow_geminal=kwargs.get("allow_geminal"),
                has_progress_callback=kwargs.get("progress_callback") is not None,
                steric_clearance_scale=kwargs.get("steric_clearance_scale"),
            )
        )
        return original_attach(system, *args, **kwargs)

    def create_progress_bar(total, desc, progress_config, unit="it"):
        """Create and retain one in-memory progress bar.

        Parameters
        ----------
        total : int
            Expected number of updates.
        desc : str
            Initial progress description.
        progress_config : FunctionalizedSlitProgressConfig
            Workflow progress configuration, accepted for interface parity.
        unit : str, optional
            Unit label for the progress counter.

        Returns
        -------
        bar : ProgressBarTrace
            Newly retained in-memory progress bar.
        """

        bar = ProgressBarTrace(
            total=total,
            desc=desc,
            unit=unit,
            leave=progress_config.leave,
        )
        progress_bars.append(bar)
        return bar

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(sms.SilicaSlit, "attach_ligands", recording_attach)
    monkeypatch.setattr(slit_mod, "_create_progress_bar", create_progress_bar)
    try:
        prepared_result = sms.prepare_functionalized_amorphous_slit_surface(config)
    finally:
        monkeypatch.undo()
    finalized_result = replace(
        prepared_result,
        system=prepared_result.system.clone(),
        silica_topology=copy.deepcopy(prepared_result.silica_topology),
        charge_diagnostics=None,
    )
    finalized_result.system.finalize()
    return FunctionalizedSlitContext(
        config=config,
        prepared_result=prepared_result,
        finalized_result=finalized_result,
        attachment_calls=tuple(attachment_calls),
        progress_bars=tuple(progress_bars),
    )


@pytest.fixture(scope="session")
def small_bare_slit_context() -> SmallBareSlitContext:
    """Prepare one repeat-one bare slit for reusable behavior checks.

    Returns
    -------
    context : SmallBareSlitContext
        Reusable attachable and finalized bare-slit results.
    """

    import silicams as sms

    config = sms.AmorphousSlitConfig(
        name="thin_bare_amorphous_slit",
        repeat_y=1,
        surface_target=sms.ExperimentalSiliconStateTarget(
            q2_fraction=0.069,
            q3_fraction=0.681,
            surface_silicon_fraction=1.0,
        ),
    )
    prepared_result = sms.prepare_amorphous_slit_surface(config)
    finalized_result = replace(
        prepared_result,
        system=prepared_result.system.clone(),
        silica_topology=copy.deepcopy(prepared_result.silica_topology),
        bare_charge_diagnostics=None,
    )
    finalized_result.system.finalize()
    return SmallBareSlitContext(
        config=config,
        prepared_result=prepared_result,
        finalized_result=finalized_result,
    )


@pytest.fixture(scope="session")
def teps_slit_context(repo_root: Path) -> FunctionalizedSlitContext:
    """Prepare one exact-target TEPS slit for timing and format checks.

    Parameters
    ----------
    repo_root : Path
        Repository root containing the explicit-bond TEPS fixture.

    Returns
    -------
    context : FunctionalizedSlitContext
        Reusable prepared and finalized TEPS slit results.
    """

    import silicams as sms

    target = sms.ExperimentalSiliconStateTarget(
        q2_fraction=65 / 957,
        q3_fraction=651 / 957,
        q4_fraction=239 / 957,
        t2_fraction=1 / 957,
        t3_fraction=1 / 957,
        surface_silicon_fraction=1.0,
    )
    config = sms.FunctionalizedAmorphousSlitConfig(
        slit_config=sms.AmorphousSlitConfig(
            name="functionalized_teps_smoke",
            repeat_y=1,
            surface_target=target,
        ),
        ligand=sms.SilaneAttachmentConfig(
            molecule=sms.Molecule(
                "TEPS",
                "TEPS",
                str(repo_root / "tests" / "data" / "TEPS.pdb"),
            ),
            mount=0,
            axis=(0, 1),
            rotate_about_axis=False,
        ),
        steric_settings=sms.FunctionalizedSlitStericConfig(clearance_scale=0.60),
        progress_settings=sms.FunctionalizedSlitProgressConfig(enabled=False),
    )
    prepared_result = sms.prepare_functionalized_amorphous_slit_surface(config)
    finalized_result = replace(
        prepared_result,
        system=prepared_result.system.clone(),
        silica_topology=copy.deepcopy(prepared_result.silica_topology),
        charge_diagnostics=None,
    )
    finalized_result.system.finalize()
    return FunctionalizedSlitContext(
        config=config,
        prepared_result=prepared_result,
        finalized_result=finalized_result,
    )
