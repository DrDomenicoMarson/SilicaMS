from __future__ import annotations

from dataclasses import dataclass
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
    """Build the shared bare amorphous slit context used by slit tests."""

    import silicams as sms

    output_dir = tmp_path_factory.mktemp("bare_amorphous_slit_preparation")
    surface_target = sms.ExperimentalSiliconStateTarget(
        q2_fraction=0.069,
        q3_fraction=0.681,
        alpha_override=1.0,
    )
    config = sms.AmorphousSlitConfig(
        name="test_bare_amorphous_slit",
        surface_target=surface_target,
    )
    prepared_result = sms.prepare_amorphous_slit_surface(config=config)
    stored_result = sms.write_bare_amorphous_slit(
        str(output_dir),
        config=config,
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
