from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
import os
import shutil

import pytest


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
