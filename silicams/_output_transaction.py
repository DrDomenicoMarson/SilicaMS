"""Exception-safe staging and promotion for related output files."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import Iterator, Sequence


def _remove_if_present(path: Path) -> None:
    """Remove one regular staging file when it exists.

    Parameters
    ----------
    path : Path
        File to remove.
    """

    try:
        path.unlink()
    except FileNotFoundError:
        pass


def commit_staged_files(staged_to_final: Sequence[tuple[Path, Path]]) -> None:
    """Promote a complete file set, restoring prior outputs on failure.

    Parameters
    ----------
    staged_to_final : sequence[tuple[Path, Path]]
        Pairs of fully written staging paths and their final destinations.

    Raises
    ------
    FileNotFoundError
        Raised when any staged output is missing.
    ValueError
        Raised when final destinations are duplicated.
    OSError
        Raised when backup, promotion, or rollback fails.

    Notes
    -----
    Each replacement is atomic because staging and final paths share a parent
    directory. The set-level rollback protects against handled process errors;
    portable power-loss atomicity across multiple files is not claimed.
    """

    pairs = tuple((Path(staged), Path(final)) for staged, final in staged_to_final)
    final_paths = tuple(final for _, final in pairs)
    if len(final_paths) != len(set(final_paths)):
        raise ValueError("Transactional output destinations must be unique.")
    missing = [staged for staged, _ in pairs if not staged.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing staged output files: {missing!r}")

    backups: dict[Path, Path] = {}
    promoted: list[Path] = []
    try:
        for _, final in pairs:
            if not final.exists():
                continue
            descriptor, backup_name = tempfile.mkstemp(
                prefix=f".{final.name}.silicams-backup-",
                dir=final.parent,
            )
            os.close(descriptor)
            backup = Path(backup_name)
            backup.unlink()
            os.replace(final, backup)
            backups[final] = backup

        for staged, final in pairs:
            os.replace(staged, final)
            promoted.append(final)
    except BaseException as promotion_error:
        rollback_errors = []
        for final in reversed(promoted):
            try:
                _remove_if_present(final)
            except OSError as error:
                rollback_errors.append(error)
        for final, backup in reversed(tuple(backups.items())):
            try:
                if backup.exists():
                    os.replace(backup, final)
            except OSError as error:
                rollback_errors.append(error)
        if rollback_errors:
            raise OSError(
                "Output promotion failed and prior outputs could not be fully restored."
            ) from promotion_error
        raise
    else:
        for backup in backups.values():
            _remove_if_present(backup)


@contextmanager
def staged_output_paths(final_paths: Sequence[Path]) -> Iterator[dict[Path, Path]]:
    """Yield same-directory staging paths and commit them on successful exit.

    Parameters
    ----------
    final_paths : sequence[Path]
        Final output paths that form one logical output set.

    Yields
    ------
    staging_paths : dict[Path, Path]
        Mapping from each normalized final path to its writable staging path.
    """

    normalized_finals = tuple(Path(path) for path in final_paths)
    if len(normalized_finals) != len(set(normalized_finals)):
        raise ValueError("Transactional output destinations must be unique.")

    staging_paths: dict[Path, Path] = {}
    created_directories: list[Path] = []
    try:
        for final in normalized_finals:
            parent = final.parent
            if not parent.exists():
                parent.mkdir(parents=True)
                created_directories.append(parent)
            descriptor, staging_name = tempfile.mkstemp(
                prefix=f".{final.name}.silicams-stage-",
                dir=parent,
            )
            os.close(descriptor)
            staging = Path(staging_name)
            staging.unlink()
            staging_paths[final] = staging
        yield staging_paths
        commit_staged_files(
            tuple((staging_paths[final], final) for final in normalized_finals)
        )
    finally:
        for staging in staging_paths.values():
            _remove_if_present(staging)
        for directory in reversed(created_directories):
            try:
                directory.rmdir()
            except OSError:
                pass


@contextmanager
def staged_output_directory(final_directory: Path) -> Iterator[Path]:
    """Stage a flat output directory and promote all generated files together.

    Parameters
    ----------
    final_directory : Path
        Directory receiving the completed output set.

    Yields
    ------
    staging_directory : Path
        Temporary sibling directory into which the complete set must be written.
    """

    final_directory = Path(final_directory)
    parent = final_directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{final_directory.name}.silicams-stage-",
        dir=parent,
    ) as staging_name:
        staging_directory = Path(staging_name)
        yield staging_directory
        staged_files = tuple(
            sorted(
                (path for path in staging_directory.iterdir() if path.is_file()),
                key=lambda path: path.name,
            )
        )
        final_directory_existed = final_directory.exists()
        final_directory.mkdir(parents=True, exist_ok=True)
        try:
            commit_staged_files(
                tuple(
                    (staged, final_directory / staged.name)
                    for staged in staged_files
                )
            )
        except BaseException:
            if not final_directory_existed:
                try:
                    final_directory.rmdir()
                except OSError:
                    pass
            raise
