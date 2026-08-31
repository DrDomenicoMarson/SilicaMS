"""Focused contracts for private GRO parsing and serialization."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from io import StringIO
import inspect
from pathlib import Path

import numpy as np
import pytest

import silicams as sms
import silicams._gro_io as gro_io


@pytest.fixture
def gro_system() -> gro_io._GroSystem:
    """Return five atoms in three independently indexed residue spans."""

    return gro_io._GroSystem(
        title="source system",
        residue_ids=np.array([99999, 99999, 0, 99999, 99999], dtype=np.int32),
        residue_names=["LIG", "LIG", "ION", "LIG", "LIG"],
        atom_names=["C1", "H1", "NA", "C2", "O2"],
        atom_ids=np.array([99999, 0, 1, 2, 3], dtype=np.int32),
        coordinates=np.arange(1.0, 16.0).reshape(5, 3) / 10.0,
        velocities=np.arange(1.0, 16.0).reshape(5, 3) / 100.0,
        box_lengths=np.array([3.0, 4.0, 5.0]),
        residue_spans=(
            gro_io._ResidueSpan(99999, "LIG", 0, 2),
            gro_io._ResidueSpan(0, "ION", 2, 3),
            gro_io._ResidueSpan(99999, "LIG", 3, 5),
        ),
        atom_to_residue_index=np.array([0, 0, 1, 2, 2], dtype=np.int32),
    )


@pytest.mark.parametrize("velocity_records", ((), (0, 1, 2), (1,)))
def test_load_gro_preserves_fields_and_optional_velocities(
    tmp_path: Path,
    velocity_records: tuple[int, ...],
) -> None:
    """Read literal records without relying on the production formatter."""

    lines = [
        "99999LIG     C199999   0.125  -0.250   3.500",
        "99999LIG     H1    0   0.250  -0.125   3.625",
        "    0ION     NA    1   0.500   0.600   0.700",
    ]
    for index in velocity_records:
        lines[index] += "  1.2500 -2.5000  3.7500"
    path = tmp_path / "literal.gro"
    path.write_text("literal title  \n3\n" + "\n".join(lines) + "\n2.0 3.0 4.0\n")

    system = gro_io._load_gro_system(path)

    assert system.title == "literal title  "
    assert system.atom_count == 3
    assert type(system.atom_count) is int
    assert system.residue_names == ["LIG", "LIG", "ION"]
    assert system.atom_names == ["C1", "H1", "NA"]
    np.testing.assert_array_equal(system.residue_ids, [99999, 99999, 0])
    np.testing.assert_array_equal(system.atom_ids, [99999, 0, 1])
    np.testing.assert_array_equal(
        system.coordinates,
        [
            [0.125, -0.250, 3.500],
            [0.250, -0.125, 3.625],
            [0.500, 0.600, 0.700],
        ],
    )
    np.testing.assert_array_equal(system.box_lengths, [2.0, 3.0, 4.0])
    assert system.residue_spans == (
        gro_io._ResidueSpan(99999, "LIG", 0, 2),
        gro_io._ResidueSpan(0, "ION", 2, 3),
    )
    np.testing.assert_array_equal(system.atom_to_residue_index, [0, 0, 1])
    for array in (system.residue_ids, system.atom_ids, system.atom_to_residue_index):
        assert array.dtype == np.int32
    for array in (system.coordinates, system.box_lengths):
        assert array.dtype == np.float64
    if velocity_records:
        expected = np.zeros((3, 3))
        expected[list(velocity_records)] = [1.25, -2.5, 3.75]
        np.testing.assert_array_equal(system.velocities, expected)
        assert system.velocities.dtype == np.float64
    else:
        assert system.velocities is None


def test_load_empty_gro_preserves_array_shapes(tmp_path: Path) -> None:
    """An empty atom section still produces correctly shaped typed arrays."""

    path = tmp_path / "empty.gro"
    path.write_text("empty\n0\n2 3 4\n")
    system = gro_io._load_gro_system(path)

    assert system.atom_count == 0
    assert system.coordinates.shape == (0, 3)
    assert system.coordinates.dtype == np.float64
    assert system.residue_ids.shape == system.atom_ids.shape == (0,)
    assert system.atom_to_residue_index.shape == (0,)
    assert system.atom_to_residue_index.dtype == np.int32
    assert system.residue_names == system.atom_names == []
    assert system.residue_spans == ()
    assert system.velocities is None


@pytest.mark.parametrize("component", range(6))
@pytest.mark.parametrize(
    "offset,accepted",
    (
        (1.0e-6, True),
        (-1.0e-6, True),
        (np.nextafter(1.0e-6, np.inf), False),
        (np.nextafter(-1.0e-6, -np.inf), False),
    ),
)
def test_nine_value_box_tolerance_is_inclusive_on_every_off_diagonal(
    tmp_path: Path,
    component: int,
    offset: float,
    accepted: bool,
) -> None:
    """Keep the exact existing tolerance, including the first excluded float."""

    box = [2.0, 3.0, 4.0] + [0.0] * 6
    box[3 + component] = float(offset)
    path = tmp_path / "box.gro"
    path.write_text("box\n0\n" + " ".join(map(str, box)) + "\n")
    assert gro_io.BOX_LINE_TOLERANCE_NM == 1.0e-6
    if accepted:
        np.testing.assert_array_equal(
            gro_io._load_gro_system(path).box_lengths, box[:3]
        )
    else:
        with pytest.raises(ValueError) as caught:
            gro_io._load_gro_system(path)
        assert str(caught.value) == (
            f"{path} uses a non-orthorhombic 9-value GRO box with non-negligible "
            f"off-diagonal terms: {box[3:]}"
        )


@pytest.mark.parametrize("component_count", (0, 2, 4, 8, 10))
def test_unsupported_box_lengths_keep_the_existing_error(
    tmp_path: Path,
    component_count: int,
) -> None:
    """Reject unsupported box-line lengths without changing diagnostics."""

    path = tmp_path / "unsupported.gro"
    path.write_text("box\n0\n" + " ".join(["1"] * component_count) + "\n")
    with pytest.raises(ValueError) as caught:
        gro_io._load_gro_system(path)
    assert str(caught.value) == (
        f"{path} uses a non-orthorhombic box with {component_count} values; "
        "this workflow currently supports only orthorhombic GRO boxes."
    )


def test_gro_reading_propagates_file_and_numeric_errors(tmp_path: Path) -> None:
    """Retain native read and numeric-conversion failures."""

    path = tmp_path / "missing.gro"
    with pytest.raises(FileNotFoundError):
        gro_io._load_gro_system(path)
    path.write_text("bad count\nnot-an-integer\n2 3 4\n")
    with pytest.raises(ValueError, match="invalid literal for int"):
        gro_io._load_gro_system(path)


@pytest.mark.parametrize("empty", (False, True))
def test_residue_spans_follow_contiguous_id_and_name_pairs(empty: bool) -> None:
    """Repeated ids and changed names retain their distinct file-order spans."""

    ids = np.array(
        [] if empty else [99999, 99999, 0, 0, 99999, 99999, 99999], dtype=np.int32
    )
    names = [] if empty else ["LIG", "LIG", "ION", "ION", "LIG", "ALT", "ALT"]
    spans, atom_to_residue = gro_io._build_residue_spans(ids, names)

    assert spans == (
        []
        if empty
        else [
            gro_io._ResidueSpan(99999, "LIG", 0, 2),
            gro_io._ResidueSpan(0, "ION", 2, 4),
            gro_io._ResidueSpan(99999, "LIG", 4, 5),
            gro_io._ResidueSpan(99999, "ALT", 5, 7),
        ]
    )
    np.testing.assert_array_equal(
        atom_to_residue, [] if empty else [0, 0, 1, 1, 2, 3, 3]
    )
    assert atom_to_residue.dtype == np.int32


def test_gro_dataclasses_keep_frozen_fields(gro_system: gro_io._GroSystem) -> None:
    """Keep the existing frozen records and scalar atom-count property."""

    assert gro_system.atom_count == 5
    assert hash(gro_system.residue_spans[0]) == hash(
        gro_io._ResidueSpan(99999, "LIG", 0, 2)
    )
    with pytest.raises(FrozenInstanceError):
        gro_system.title = "reassigned"
    with pytest.raises(FrozenInstanceError):
        gro_system.residue_spans[0].start = 10


@pytest.mark.parametrize("with_velocity", (False, True))
def test_atom_format_precision_truncation_and_rollover(with_velocity: bool) -> None:
    """Compare the complete fixed-width line to independently written text."""

    velocity = np.array([0.123456, -0.234567, 0.345678]) if with_velocity else None
    line = gro_io._format_gro_atom_line(
        100001,
        "LONGRES",
        "ATOMLONG",
        200002,
        np.array([1.23456, -2.34567, 3.45678]),
        velocity,
    )
    expected = "    1LONGRATOML    2   1.235  -2.346   3.457"
    if with_velocity:
        expected += "  0.1235 -0.2346  0.3457"
    assert line == expected + "\n"


@pytest.mark.parametrize(
    "selection,retained_spans",
    (
        (None, (0, 1, 2)),
        ((True, True, False, True, True), (0, 2)),
        ((True, False, True, False, False), (1,)),
        ((False, False, False, False, False), ()),
    ),
)
@pytest.mark.parametrize(
    "write_velocities,supplied_velocities",
    (
        (False, True),
        (True, False),
        (True, True),
    ),
)
def test_atom_writing_keeps_whole_residues_and_explicit_output_vectors(
    gro_system: gro_io._GroSystem,
    selection: tuple[bool, ...] | None,
    retained_spans: tuple[int, ...],
    write_velocities: bool,
    supplied_velocities: bool,
) -> None:
    """Serialize supplied vectors, not source vectors, without splitting residues."""

    coordinates = gro_system.coordinates + 1.0
    velocities = gro_system.velocities - 2.0 if supplied_velocities else None
    mask = None if selection is None else np.array(selection, dtype=bool)
    arrays = (coordinates, velocities, gro_system.coordinates, gro_system.velocities)
    originals = [(array, array.copy()) for array in arrays if array is not None]
    for array, _ in originals:
        array.setflags(write=False)
    handle = StringIO()
    next_residue, next_atom = gro_io._write_system_atoms(
        handle,
        gro_system,
        coordinates,
        velocities,
        mask,
        write_velocities,
        99999,
        99998,
    )

    expected_records = [
        (99999 + residue_offset, atom_index)
        for residue_offset, span_index in enumerate(retained_spans)
        for atom_index in range(
            gro_system.residue_spans[span_index].start,
            gro_system.residue_spans[span_index].stop,
        )
    ]
    assert (next_residue, next_atom) == (
        99999 + len(retained_spans),
        99998 + len(expected_records),
    )
    lines = handle.getvalue().splitlines()
    assert len(lines) == len(expected_records)
    for atom_offset, (line, (residue_id, atom_index)) in enumerate(
        zip(lines, expected_records, strict=True)
    ):
        assert int(line[:5]) == residue_id % 100000
        assert line[5:10] == gro_system.residue_names[atom_index].ljust(5)
        assert line[10:15] == gro_system.atom_names[atom_index].rjust(5)
        assert int(line[15:20]) == (99998 + atom_offset) % 100000
        np.testing.assert_allclose(
            [float(line[start : start + 8]) for start in (20, 28, 36)],
            coordinates[atom_index],
            rtol=0,
            atol=1e-12,
        )
        if write_velocities:
            np.testing.assert_allclose(
                [float(line[start : start + 8]) for start in (44, 52, 60)],
                velocities[atom_index] if supplied_velocities else [0.0, 0.0, 0.0],
                rtol=0,
                atol=1e-12,
            )
        assert len(line) == (68 if write_velocities else 44)
    for array, original in originals:
        np.testing.assert_array_equal(array, original)


@pytest.mark.parametrize(
    "slit_has_velocities,guest_has_velocities",
    (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ),
)
@pytest.mark.parametrize("keep_guests", (False, True))
def test_merged_gro_counts_order_and_round_trip_precision(
    tmp_path: Path,
    gro_system: gro_io._GroSystem,
    slit_has_velocities: bool,
    guest_has_velocities: bool,
    keep_guests: bool,
) -> None:
    """Merge explicit vectors with the established formatting and velocity policy."""

    output_path = tmp_path / "merged.gro"
    slit_coordinates = gro_system.coordinates + 0.123456
    guest_coordinates = gro_system.coordinates - 0.234567
    slit_velocities = gro_system.velocities + 0.123456 if slit_has_velocities else None
    guest_velocities = (
        gro_system.velocities - 0.234567 if guest_has_velocities else None
    )
    mask = np.array([False, False, keep_guests, keep_guests, keep_guests])
    counts = gro_io._write_merged_gro(
        output_path,
        gro_system,
        slit_coordinates,
        slit_velocities,
        gro_system,
        guest_coordinates,
        guest_velocities,
        mask,
        np.array([8.123456, 9.234567, 10.345678]),
    )

    expected_atom_count, expected_residue_count = (8, 5) if keep_guests else (5, 3)
    assert counts == (expected_atom_count, expected_residue_count)
    lines = output_path.read_text().splitlines()
    assert lines[:2] == [
        "Merged slit + ring-check filtered guest",
        str(expected_atom_count),
    ]
    assert len(lines[2:-1]) == expected_atom_count
    assert lines[-1] == "   8.12346   9.23457  10.34568"
    parsed = gro_io._load_gro_system(output_path)
    assert len(parsed.residue_spans) == expected_residue_count
    assert parsed.atom_names == gro_system.atom_names + (
        gro_system.atom_names[2:] if keep_guests else []
    )
    np.testing.assert_array_equal(
        parsed.atom_ids, np.arange(1, expected_atom_count + 1)
    )
    np.testing.assert_array_equal(
        parsed.residue_ids, [1, 1, 2, 3, 3] + ([4, 5, 5] if keep_guests else [])
    )
    np.testing.assert_allclose(
        parsed.coordinates,
        np.vstack((slit_coordinates, guest_coordinates[mask])),
        rtol=0,
        atol=0.0005,
    )
    np.testing.assert_array_equal(parsed.box_lengths, [8.12346, 9.23457, 10.34568])
    if slit_has_velocities or guest_has_velocities:
        expected_velocities = np.vstack(
            (
                slit_velocities if slit_has_velocities else np.zeros((5, 3)),
                (guest_velocities if guest_has_velocities else np.zeros((5, 3)))[mask],
            )
        )
        np.testing.assert_allclose(
            parsed.velocities, expected_velocities, rtol=0, atol=0.00005
        )
    else:
        assert parsed.velocities is None


def test_merged_writer_propagates_open_errors(
    tmp_path: Path,
    gro_system: gro_io._GroSystem,
) -> None:
    """Destination creation and transactional cleanup remain caller concerns."""

    with pytest.raises(FileNotFoundError):
        gro_io._write_merged_gro(
            tmp_path / "missing" / "merged.gro",
            gro_system,
            gro_system.coordinates,
            None,
            gro_system,
            gro_system.coordinates,
            None,
            np.ones(5, dtype=bool),
            gro_system.box_lengths,
        )


def test_gro_module_stays_private_and_has_no_workflow_dependencies() -> None:
    """Keep GRO I/O below the scientific workflow and outside the public API."""

    assert gro_io.__all__ == []
    for name in ("_GroSystem", "_ResidueSpan", "_load_gro_system", "_write_merged_gro"):
        assert not hasattr(sms, name)
    tree = ast.parse(inspect.getsource(gro_io))
    imports = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imports <= {"__future__", "dataclasses", "pathlib", "typing", "numpy"}
