"""Direct contracts for construction steric batches and local periodic indexing."""

from dataclasses import FrozenInstanceError, fields
from itertools import product

import numpy as np
import pytest

import silicams as sms
import silicams._silica_sterics as sterics
from silicams._silica_sterics import (
    _MoleculeStericBatch,
    _StericAtomBatch,
    _StericGrid,
)


def _assert_batch(batch, positions, radii, atom_ids):
    """Check aligned batch values, shapes, and native dtypes.

    Parameters
    ----------
    batch : _StericAtomBatch
        Actual atom batch to inspect.
    positions : array-like
        Expected Cartesian coordinates in nanometers, reshapeable to ``(n, 3)``.
    radii : array-like
        Expected covalent radii in nanometers with shape ``(n,)``.
    atom_ids : array-like
        Expected scaffold identifiers with shape ``(n,)``.
    """
    np.testing.assert_array_equal(batch.positions, np.asarray(positions).reshape(-1, 3))
    np.testing.assert_array_equal(batch.radii, radii)
    np.testing.assert_array_equal(batch.block_atom_ids, atom_ids)
    assert batch.positions.dtype == np.dtype(float)
    assert batch.radii.dtype == np.dtype(float)
    assert batch.block_atom_ids.dtype == np.dtype(int)


def test_steric_types_remain_private():
    """Keep the extracted implementation out of the package's supported API."""
    assert sterics.__all__ == []
    for name in ("_StericAtomBatch", "_MoleculeStericBatch", "_StericGrid"):
        assert not hasattr(sms, name)


def test_empty_batches_have_independent_arrays_and_native_dtypes():
    """Allocate independent empty coordinate, radius, and identifier arrays."""
    first = _StericAtomBatch.empty()
    second = _StericAtomBatch()
    _assert_batch(first, [], [], [])
    _assert_batch(second, [], [], [])
    for name in ("positions", "radii", "block_atom_ids"):
        assert getattr(first, name) is not getattr(second, name)
    first.append([[1, 2, 3]], [0.1], [7])
    _assert_batch(second, [], [], [])


def test_direct_batch_construction_preserves_array_aliases_and_dtypes():
    """Do not add constructor copying, normalization, or frozen fields."""
    positions = np.array([[1, 2, 3]], dtype=np.float32)
    radii = np.array([0.125], dtype=np.float32)
    atom_ids = np.array([7], dtype=np.int16)
    batch = _StericAtomBatch(positions, radii, atom_ids)
    assert batch.positions is positions
    assert batch.radii is radii
    assert batch.block_atom_ids is atom_ids
    positions[0, 0] = 9
    assert batch.positions[0, 0] == 9
    replacement = np.array([8], dtype=np.int16)
    batch.block_atom_ids = replacement
    assert batch.block_atom_ids is replacement
    assert [field.name for field in fields(batch)] == [
        "positions", "radii", "block_atom_ids",
    ]


@pytest.mark.parametrize("batches", ([], [None], [None, _StericAtomBatch(), None]))
def test_concatenate_without_atoms_returns_native_empty_batch(batches):
    """Ignore absent/empty entries, including an entirely empty input."""
    _assert_batch(_StericAtomBatch.concatenate(batches), [], [], [])


def test_concatenate_preserves_order_dtypes_and_copies_even_one_batch():
    """Concatenation retains input row order without forcing native dtypes."""
    first = _StericAtomBatch(
        np.array([[3, 2, 1], [6, 5, 4]], dtype=np.float32),
        np.array([0.125, 0.25], dtype=np.float32),
        np.array([9, 2], dtype=np.int16),
    )
    second = _StericAtomBatch(
        np.array([[9, 8, 7]], dtype=np.float32),
        np.array([0.5], dtype=np.float32),
        np.array([-1], dtype=np.int16),
    )
    # Emptiness is determined by radii; skipped entries are not shape-validated.
    skipped = _StericAtomBatch(np.ones((1, 3)), np.empty(0), np.array([99]))
    merged = _StericAtomBatch.concatenate(iter([None, first, skipped, second]))
    single = _StericAtomBatch.concatenate([first])
    np.testing.assert_array_equal(merged.positions, [[3, 2, 1], [6, 5, 4], [9, 8, 7]])
    np.testing.assert_array_equal(merged.radii, [0.125, 0.25, 0.5])
    np.testing.assert_array_equal(merged.block_atom_ids, [9, 2, -1])
    assert merged.positions.dtype == np.float32
    assert merged.radii.dtype == np.float32
    assert merged.block_atom_ids.dtype == np.int16
    for name in ("positions", "radii", "block_atom_ids"):
        assert not np.shares_memory(getattr(first, name), getattr(single, name))
        assert not np.shares_memory(getattr(first, name), getattr(merged, name))
        assert not np.shares_memory(getattr(second, name), getattr(merged, name))


@pytest.mark.parametrize("existing", (False, True))
def test_append_converts_copies_and_preserves_row_order(existing):
    """Copy supplied rows for both initial allocation and subsequent append."""
    batch = _StericAtomBatch.empty()
    if existing:
        batch.append([6, 5, 4], 0.25, 2)
    positions = np.array([3, 2, 1, 9, 8, 7], dtype=np.int16)
    radii = np.array([[0.125, 0.5]], dtype=np.float32)
    atom_ids = np.array([[9, -1]], dtype=np.int16)
    assert batch.append(positions, radii, atom_ids) is None
    positions[:] = 0
    radii[:] = 0
    atom_ids[:] = 0
    _assert_batch(
        batch,
        ([[6, 5, 4]] if existing else []) + [[3, 2, 1], [9, 8, 7]],
        ([0.25] if existing else []) + [0.125, 0.5],
        ([2] if existing else []) + [9, -1],
    )


@pytest.mark.parametrize("existing", (False, True))
def test_empty_append_and_empty_removal_leave_arrays_untouched(existing):
    """No-op requests preserve the exact array objects, not only values."""
    batch = _StericAtomBatch.empty()
    if existing:
        batch.append([1, 2, 3], [0.125], [7])
    arrays = (batch.positions, batch.radii, batch.block_atom_ids)
    assert batch.append([], [], []) is None
    assert batch.remove_block_atoms([]) is None
    if not existing:
        assert batch.remove_block_atoms([7]) is None
    assert batch.positions is arrays[0]
    assert batch.radii is arrays[1]
    assert batch.block_atom_ids is arrays[2]


def test_batch_removal_filters_all_matching_rows_without_reordering():
    """Remove repeated identifiers while retaining unrelated and sentinel rows."""
    batch = _StericAtomBatch.empty()
    batch.append(np.arange(15).reshape(5, 3), [0.125] * 5, [7, -1, 2, 7, 9])
    assert batch.remove_block_atoms(np.array([[7, 99]], dtype=np.int16)) is None
    _assert_batch(batch, [[3, 4, 5], [6, 7, 8], [12, 13, 14]], [0.125] * 3, [-1, 2, 9])
    arrays = (batch.positions, batch.radii, batch.block_atom_ids)
    batch.remove_block_atoms([99])
    for old, new in zip(arrays, (batch.positions, batch.radii, batch.block_atom_ids)):
        np.testing.assert_array_equal(old, new)
        assert not np.shares_memory(old, new)
    batch.remove_block_atoms([-1, 2, 9])
    _assert_batch(batch, [], [], [])


def test_invalid_batch_inputs_keep_numpy_failure_behavior():
    """Preserve NumPy shape/conversion failures without adding validation."""
    batch = _StericAtomBatch.empty()
    with pytest.raises(ValueError):
        batch.append([1, 2], [0.1], [1])
    with pytest.raises(ValueError):
        batch.remove_block_atoms(["not an integer"])
    with pytest.raises(ValueError):
        _StericAtomBatch.concatenate([
            _StericAtomBatch(np.zeros((1, 2)), np.ones(1), np.ones(1, dtype=int)),
            _StericAtomBatch(np.zeros((1, 3)), np.ones(1), np.ones(1, dtype=int)),
        ])
    _assert_batch(batch, [], [], [])


def test_molecule_batch_is_frozen_but_retains_mutable_input_arrays():
    """Freeze field assignment, not the arrays or their constructor aliases."""
    atom_ids = np.array([4], dtype=np.int16)
    positions = np.array([[1, 2, 3]], dtype=np.float32)
    radii = np.array([0.125], dtype=np.float32)
    batch = _MoleculeStericBatch(atom_ids, positions, radii)
    assert batch.atom_ids is atom_ids
    assert batch.positions is positions
    assert batch.radii is radii
    assert [field.name for field in fields(batch)] == ["atom_ids", "positions", "radii"]
    with pytest.raises(FrozenInstanceError):
        batch.positions = np.zeros((1, 3))
    batch.positions[0, 0] = 9
    assert positions[0, 0] == 9
    atom_ids[0] = 5
    assert batch.atom_ids[0] == 5


def test_grid_defaults_derived_dimensions_and_mapping_ownership():
    """Keep derived dimensions outside the dataclass constructor schema."""
    grid = _StericGrid((1.125, 0.875, 0.125))
    other = _StericGrid((1.125, 0.875, 0.125))
    assert grid.cell_size_nm == sterics._STERIC_GRID_CELL_SIZE_NM == 0.25
    assert grid.dims == (4, 3, 1)
    assert grid.cells == grid.block_cells == {}
    assert grid.cells is not other.cells
    assert grid.block_cells is not other.block_cells
    assert [field.name for field in fields(grid)] == [
        "box", "cell_size_nm", "cells", "block_cells",
    ]
    assert _StericGrid((1.125, 0.875, 0.125), cell_size_nm=0.5).dims == (2, 1, 1)
    cells = {(0, 0, 0): _StericAtomBatch.empty()}
    block_cells = {}
    supplied = _StericGrid((1, 1, 1), cells=cells, block_cells=block_cells)
    assert supplied.cells is cells
    assert supplied.block_cells is block_cells
    supplied.add_block_atom(7, [0.1, 0.1, 0.1], 0.125)
    assert block_cells == {7: (0, 0, 0)}
    _assert_batch(cells[(0, 0, 0)], [[0.1, 0.1, 0.1]], [0.125], [7])


@pytest.mark.parametrize(
    ("box", "position", "expected"),
    (
        ((1, 1.25, 1.5), (0, 0, 0), (0, 0, 0)),
        ((1, 1.25, 1.5), (0.25, 0.5, 0.75), (1, 2, 3)),
        ((1, 1.25, 1.5), (1, 1.25, 1.5), (0, 0, 0)),
        ((1, 1.25, 1.5), (-0.125, 1.375, 3.125), (3, 0, 0)),
        ((1, 1.25, 1.5), (-2.125, -2.625, -3.125), (3, 4, 5)),
        ((1.125, 0.875, 0.625), (1.0625, 0.8125, 0.5625), (3, 2, 1)),
        ((0.125, 0.5, 0.75), (0.375, -0.125, 1.375), (0, 1, 2)),
    ),
)
def test_grid_cell_assignment_wraps_each_box_axis(box, position, expected):
    """Use explicit cell expectations at periodic seams and remainder cells."""
    assert _StericGrid(box)._cell_key(position) == expected


def test_grid_cell_boundary_uses_existing_half_open_partition():
    """Keep exact cell boundaries and adjacent representable values unchanged."""
    grid = _StericGrid((1, 1, 1))
    lower = np.nextafter(0.25, 0)
    upper = np.nextafter(0.25, 1)
    assert grid._cell_key((lower, 0.25, upper)) == (0, 1, 1)
    assert grid._cell_key((np.nextafter(1.0, 0), 1.0, np.nextafter(1.0, 2))) == (3, 0, 0)


def test_nonpositive_box_components_keep_existing_unwrapped_behavior():
    """Document the existing fallback without introducing new box validation."""
    grid = _StericGrid((0, -1, 1))
    assert grid.dims == (1, 1, 4)
    assert grid._wrap_component(-0.375, 0) == -0.375
    assert grid._wrap_component(2.125, 1) == 2.125
    assert grid._wrap_component(2.125, 2) == 0.125
    with pytest.raises(ZeroDivisionError):
        _StericGrid((1, 1, 1), cell_size_nm=0)


def test_grid_addition_stores_unwrapped_copies_and_tracks_only_scaffold_ids():
    """Wrap cell keys, not stored positions, and reserve -1 for attached rows."""
    grid = _StericGrid((1, 1, 1))
    positions = np.array([[-0.125, 0.125, 1.125], [0.875, 0.125, 0.125]])
    radii = np.array([0.125, 0.25])
    ids = np.array([7, 2])
    assert grid.add_block_atoms(ids, positions, radii) is None
    assert grid.add_attached_atoms(positions[:1], radii[:1]) is None
    positions[:] = 0
    radii[:] = 0
    ids[:] = 0
    assert grid.block_cells == {7: (3, 0, 0), 2: (3, 0, 0)}
    _assert_batch(
        grid.cells[(3, 0, 0)],
        [[-0.125, 0.125, 1.125], [0.875, 0.125, 0.125], [-0.125, 0.125, 1.125]],
        [0.125, 0.25, 0.125], [7, 2, -1],
    )
    grid.remove_block_atoms([-1, 7, 2, 99])
    assert grid.block_cells == {}
    _assert_batch(grid.cells[(3, 0, 0)], [[-0.125, 0.125, 1.125]], [0.125], [-1])


def test_grid_removal_prunes_cells_and_supports_successive_attachment_updates():
    """Model scaffold removal followed by inserting each successful attachment."""
    grid = _StericGrid((1, 1, 1))
    grid.add_block_atoms(
        [7, 2, 9],
        [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [0.6, 0.1, 0.1]],
        [0.125] * 3,
    )
    assert grid.remove_block_atoms([7, 7, 99]) is None
    assert grid.block_cells == {2: (0, 0, 0), 9: (2, 0, 0)}
    _assert_batch(grid.cells[(0, 0, 0)], [[0.2, 0.1, 0.1]], [0.125], [2])
    grid.add_attached_atoms([[0.15, 0.1, 0.1]], [0.25])
    grid.remove_block_atoms([2, 9])
    assert set(grid.cells) == {(0, 0, 0)}
    assert grid.block_cells == {}
    _assert_batch(grid.cells[(0, 0, 0)], [[0.15, 0.1, 0.1]], [0.25], [-1])
    grid.add_block_atom(7, [0.6, 0.1, 0.1], 0.125)
    assert grid.block_cells == {7: (2, 0, 0)}
    grid.remove_block_atoms([7])
    grid.add_attached_atoms([[0.65, 0.1, 0.1]], [0.5])
    assert grid.block_cells == {}
    _assert_batch(grid.cells[(2, 0, 0)], [[0.65, 0.1, 0.1]], [0.5], [-1])


def test_grid_empty_operations_and_missing_cell_removal():
    """Ignore empty inputs and tolerate a stale mapping to a missing cell."""
    grid = _StericGrid((1, 1, 1))
    grid.add_block_atoms([], [], [])
    grid.add_attached_atoms([], [])
    grid.remove_block_atoms([])
    grid.block_cells[7] = (0, 0, 0)
    grid.remove_block_atoms([7, 99])
    assert grid.cells == grid.block_cells == {}
    _assert_batch(grid.neighbor_batch([0, 0, 0]), [], [], [])


def test_grid_batch_addition_retains_existing_shortest_input_iteration():
    """Do not introduce length validation while extracting the existing loops."""
    grid = _StericGrid((1, 1, 1))
    grid.add_block_atoms([7, 8], [[0.1, 0.1, 0.1]], [0.125, 0.25])
    grid.add_attached_atoms([[0.2, 0.1, 0.1], [0.3, 0.1, 0.1]], [0.5])
    assert grid.block_cells == {7: (0, 0, 0)}
    _assert_batch(grid.cells[(0, 0, 0)], [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]], [0.125, 0.5], [7, -1])


@pytest.mark.parametrize(
    ("box", "center", "axis_order"),
    (
        ((1.125, 1.375, 1.625), (0, 0, 0), ((3, 0, 1), (4, 0, 1), (5, 0, 1))),
        ((1.125, 1.375, 1.625), (2, 3, 4), ((1, 2, 3), (2, 3, 4), (3, 4, 5))),
        ((0.125, 0.5, 1.0), (0, 0, 0), ((0,), (1, 0), (3, 0, 1))),
        ((0.5, 0.5, 0.5), (1, 1, 1), ((0, 1), (0, 1), (0, 1))),
        ((0.125, 0.125, 0.125), (0, 0, 0), ((0,), (0,), (0,))),
    ),
)
def test_neighbor_membership_order_and_uniqueness_against_independent_cells(
    box, center, axis_order,
):
    """Check explicit periodic cell neighborhoods, not a global distance query."""
    grid = _StericGrid(box)
    expected_rows = {}
    all_keys = list(product(*(range(dim) for dim in grid.dims)))
    # Insert cells in reverse order: lookup order must not follow dict insertion.
    for index, key in reversed(list(enumerate(all_keys))):
        position = np.array(key) * 0.25 + 0.0625
        ids = [2 * index + 2, 2 * index + 1]
        grid.add_block_atoms(ids, [position, position], [0.125, 0.25])
        expected_rows[key] = (position, ids)
    query = np.array(center) * 0.25 + 0.0625
    result = grid.neighbor_batch(query)
    # Per-axis cell orders are specified above independently of grid traversal.
    expected_keys = list(product(*axis_order))
    expected_ids = [atom_id for key in expected_keys for atom_id in expected_rows[key][1]]
    expected_positions = [expected_rows[key][0] for key in expected_keys for _ in range(2)]
    _assert_batch(result, expected_positions, [0.125, 0.25] * len(expected_keys), expected_ids)
    assert len(set(result.block_atom_ids)) == len(result.block_atom_ids)
    # Independently select cells by circular index distance to the query cell.
    member_keys = {
        key for key in all_keys
        if all(min(abs(k - c), size - abs(k - c)) <= 1
               for k, c, size in zip(key, center, grid.dims))
    }
    assert set(expected_keys) == member_keys
    shifted = grid.neighbor_batch(query + np.asarray(box) * [2, -1, 3])
    np.testing.assert_array_equal(shifted.block_atom_ids, result.block_atom_ids)
    for cell in grid.cells.values():
        assert not np.shares_memory(result.positions, cell.positions)
        assert not np.shares_memory(result.radii, cell.radii)
        assert not np.shares_memory(result.block_atom_ids, cell.block_atom_ids)


def test_neighbor_query_skips_empty_and_distant_cells_without_radius_filtering():
    """Include all rows in local cells but never expand the fixed neighborhood."""
    grid = _StericGrid((1, 1, 1))
    grid.cells[(0, 0, 0)] = _StericAtomBatch.empty()
    grid.add_block_atom(7, [0.875, 0.875, 0.875], 0.0)
    grid.add_block_atom(9, [0.625, 0.625, 0.625], 100.0)
    _assert_batch(grid.neighbor_batch([0, 0, 0]), [[0.875, 0.875, 0.875]], [0.0], [7])
