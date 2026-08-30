"""Regression tests for exception-safe output promotion."""

from pathlib import Path

import pytest

import silicams._output_transaction as transaction_mod


def test_staged_output_paths_replace_complete_set_and_preserve_unrelated(
    tmp_path: Path,
) -> None:
    """Promote all staged files while leaving unrelated files untouched."""

    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    unrelated = tmp_path / "unrelated.txt"
    first.write_text("old first", encoding="utf-8")
    unrelated.write_text("keep me", encoding="utf-8")

    with transaction_mod.staged_output_paths((first, second)) as staged:
        staged[first].write_text("new first", encoding="utf-8")
        staged[second].write_text("new second", encoding="utf-8")

    assert first.read_text(encoding="utf-8") == "new first"
    assert second.read_text(encoding="utf-8") == "new second"
    assert unrelated.read_text(encoding="utf-8") == "keep me"


def test_staged_output_paths_restore_prior_set_when_promotion_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore every prior file if a later promotion raises."""

    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("old first", encoding="utf-8")
    second.write_text("old second", encoding="utf-8")
    real_replace = transaction_mod.os.replace
    call_count = 0

    def fail_on_second_promotion(source, destination):
        """Raise once after the first staged file has been promoted."""

        nonlocal call_count
        call_count += 1
        if call_count == 4:
            raise OSError("injected promotion failure")
        return real_replace(source, destination)

    monkeypatch.setattr(transaction_mod.os, "replace", fail_on_second_promotion)

    with pytest.raises(OSError, match="injected promotion failure"):
        with transaction_mod.staged_output_paths((first, second)) as staged:
            staged[first].write_text("new first", encoding="utf-8")
            staged[second].write_text("new second", encoding="utf-8")

    assert first.read_text(encoding="utf-8") == "old first"
    assert second.read_text(encoding="utf-8") == "old second"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["first.txt", "second.txt"]


def test_staged_output_paths_leave_existing_files_when_stage_is_incomplete(
    tmp_path: Path,
) -> None:
    """Reject an incomplete staged set before touching prior outputs."""

    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("old first", encoding="utf-8")
    second.write_text("old second", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Missing staged output"):
        with transaction_mod.staged_output_paths((first, second)) as staged:
            staged[first].write_text("new first", encoding="utf-8")

    assert first.read_text(encoding="utf-8") == "old first"
    assert second.read_text(encoding="utf-8") == "old second"
