"""Direct numerical contracts for construction clearance and pose selection."""

from dataclasses import FrozenInstanceError
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

import silicams as sms
import silicams.generic as generic
import silicams._silica_placement as placement
from silicams._numba_kernels import minimum_clearance_against_batch
from silicams._silica_placement import (
    _BridgeStericCache,
    _best_bridge_position,
    _best_pose_positions,
    _bridge_base_direction,
    _bridge_candidate_positions,
    _bridge_clearance_from_arrays,
    _bridge_pair_frame,
    _clearance_is_acceptable,
    _filtered_steric_batch_arrays,
    _positions_clearance,
    _rotate_positions_around_axis,
    _rotation_angles,
    _rotation_matrix,
)
from silicams._silica_engine import _SilicaChemistryEngine
from silicams._silica_sterics import (
    _MoleculeStericBatch,
    _StericAtomBatch,
    _StericGrid,
)


def _brute_force_clearance(
    positions,
    radii,
    reference_positions,
    reference_radii,
    box,
    scale,
):
    """Return an independent scalar periodic clearance reference.

    Parameters
    ----------
    positions, reference_positions : array-like
        Candidate and reference coordinates in nanometers with shapes ``(n, 3)``
        and ``(m, 3)``.
    radii, reference_radii : array-like
        Candidate and reference covalent radii in nanometers.
    box : array-like
        Orthorhombic box lengths in nanometers. Nonpositive axes are not wrapped.
    scale : float
        Multiplier applied to summed radii.

    Returns
    -------
    clearance : float
        Smallest pair distance minus the corresponding scaled summed radii, or
        ``inf`` when either collection is empty.
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    radii = np.asarray(radii, dtype=float).reshape(-1)
    reference_positions = np.asarray(reference_positions, dtype=float).reshape(-1, 3)
    reference_radii = np.asarray(reference_radii, dtype=float).reshape(-1)
    box = np.asarray(box, dtype=float)
    result = float("inf")
    for position, radius in zip(positions, radii):
        for reference_position, reference_radius in zip(
            reference_positions, reference_radii,
        ):
            squared_distance = 0.0
            for delta, box_length in zip(reference_position - position, box):
                if box_length > 0:
                    image_distances = [
                        abs(delta + image * box_length) for image in range(-20, 21)
                    ]
                    delta = min(image_distances)
                squared_distance += float(delta * delta)
            clearance = squared_distance**0.5 - scale * (radius + reference_radius)
            if clearance < result:
                result = clearance
    return result


def _copy_batch(batch):
    """Return independent copies of all arrays in one steric batch."""
    return tuple(
        value.copy()
        for value in (batch.positions, batch.radii, batch.block_atom_ids)
    )


def test_placement_helpers_remain_private():
    """Keep the numerical extraction out of the supported package API."""
    assert placement.__all__ == []
    for name in (
        "_positions_clearance",
        "_filtered_steric_batch_arrays",
        "_best_bridge_position",
        "_BridgeStericCache",
    ):
        assert not hasattr(sms, name)


@pytest.mark.parametrize(
    ("clearance", "expected"),
    (
        (0.1, True),
        (0.0, True),
        (-5e-13, True),
        (-2e-12, False),
        (float("inf"), True),
        (float("-inf"), False),
        (float("nan"), False),
    ),
)
def test_clearance_acceptance_uses_dimensioned_roundoff_tolerance(
    clearance, expected,
):
    """Accept only nonnegative and roundoff-negative clearances."""
    assert _clearance_is_acceptable(clearance) is expected


def _bridge_cache(
    local_positions=(),
    local_min_distances=(),
    global_positions=(),
    global_min_distances=(),
    box=(2.0, 3.0, 4.0),
):
    """Return one independently assembled bridge steric cache."""
    return _BridgeStericCache(
        box=np.asarray(box, dtype=float),
        local_positions=np.asarray(local_positions, dtype=float).reshape(-1, 3),
        local_min_distances=np.asarray(local_min_distances, dtype=float),
        global_positions=np.asarray(global_positions, dtype=float).reshape(-1, 3),
        global_min_distances=np.asarray(global_min_distances, dtype=float),
    )


def test_bridge_steric_cache_freezes_fields_but_not_owned_arrays():
    """Keep a frozen record without claiming deep array immutability."""
    cache = _bridge_cache()
    with pytest.raises(FrozenInstanceError):
        cache.box = np.ones(3)
    cache.box[0] = 5.0
    assert cache.box.tolist() == [5.0, 3.0, 4.0]


def test_bridge_pair_frame_wraps_periodic_midpoint_without_mutating_inputs():
    """Construct the pair frame across a seam in a non-cubic box."""
    position_a = np.array([1.875, 1.0, 1.0], dtype=np.float32)
    position_b = np.array([0.125, 1.0, 1.0], dtype=np.float32)
    box = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    source = tuple(value.copy() for value in (position_a, position_b, box))

    center, axis = _bridge_pair_frame(position_a, position_b, box)

    np.testing.assert_allclose(center, [0.0, 1.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(axis, [1.0, 0.0, 0.0], atol=1e-15)
    for actual, expected in zip((position_a, position_b, box), source):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("axis", "normals", "expected"),
    (
        ([1, 0, 0], ([0, 0, 1], [0, 0, 1]), [0, 0, 1]),
        ([1, 0, 0], ([1, 0, 0], [1, 0, 0]), [0, 0, 1]),
        ([0, 1, 0], ([0, 1, 0], [0, 1, 0]), [0, 0, -1]),
    ),
)
def test_bridge_base_direction_preserves_surface_and_fallback_rules(
    axis, normals, expected,
):
    """Use the projected normals or the deterministic fallback direction."""
    np.testing.assert_allclose(
        _bridge_base_direction(axis, normals),
        expected,
        atol=1e-15,
    )


def test_bridge_candidates_preserve_rotation_order_and_periodic_wrapping():
    """Generate every candidate in order without changing source arrays."""
    center = np.array([0.95, 1.0, 2.95], dtype=float)
    axis = np.array([1.0, 0.0, 0.0], dtype=float)
    normals = (
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    box = np.array([1.0, 2.0, 3.0], dtype=float)
    source = tuple(value.copy() for value in (center, axis, *normals, box))

    candidates = _bridge_candidate_positions(
        center,
        axis,
        normals,
        box,
        0.1,
        (0.0, 90.0, -90.0, 180.0),
    )

    np.testing.assert_allclose(
        candidates,
        (
            [0.95, 1.0, 0.05],
            [0.95, 0.9, 2.95],
            [0.95, 1.1, 2.95],
            [0.95, 1.0, 2.85],
        ),
        atol=1e-15,
    )
    for actual, expected in zip((center, axis, *normals, box), source):
        np.testing.assert_array_equal(actual, expected)


def test_bridge_clearance_handles_empty_distant_and_periodic_references():
    """Preserve the cutoff result and periodic distance calculation."""
    cutoff = 0.3
    empty = np.empty((0, 3), dtype=float)
    assert _bridge_clearance_from_arrays(
        [0, 0, 0], [2, 3, 4], empty, np.empty(0), cutoff,
    ) == cutoff
    assert _bridge_clearance_from_arrays(
        [0, 0, 0],
        [2, 3, 4],
        np.array([[0.8, 0, 0]]),
        np.array([0.1]),
        cutoff,
    ) == cutoff
    assert _bridge_clearance_from_arrays(
        [0, 0, 0],
        [2, 3, 4],
        np.array([[4.125, 0, 0]]),
        np.array([0.125]),
        cutoff,
    ) == pytest.approx(0.0, abs=1e-15)


def test_bridge_clearance_preserves_first_negative_reference_row():
    """Keep the established first-overlap reduction rather than the minimum."""
    clearance = _bridge_clearance_from_arrays(
        [0, 0, 0],
        [2, 3, 4],
        np.array([[0.125, 0, 0], [0.0625, 0, 0]]),
        np.array([0.25, 0.25]),
        0.3,
    )
    assert clearance == pytest.approx(-0.125, abs=1e-15)


def test_best_bridge_orders_locally_then_requires_global_acceptance():
    """Reject the locally best clash and accept the next-ranked candidate."""
    candidates = [
        [0.25, 0, 0],
        [0.2, 0, 0],
        [0.15, 0, 0],
    ]
    cache = _bridge_cache(
        local_positions=[[0, 0, 0]],
        local_min_distances=[0.1],
        global_positions=[[0.25, 0, 0]],
        global_min_distances=[0.05],
        box=[2, 2, 2],
    )
    selected = _best_bridge_position(candidates, cache, 0.3)
    assert selected is candidates[1]


def test_best_bridge_keeps_first_tie_and_accepts_roundoff_contact():
    """Keep stable tie ordering and share the exact-contact tolerance."""
    tied_candidates = ([0.4, 0, 0], [0.6, 0, 0])
    assert _best_bridge_position(
        tied_candidates, _bridge_cache(), 0.3,
    ) is tied_candidates[0]

    contact_candidate = [0, 0, 0]
    contact_cache = _bridge_cache(
        local_positions=[[0.2, 0, 0]],
        local_min_distances=[0.2],
        box=[2, 2, 2],
    )
    assert _best_bridge_position(
        [contact_candidate], contact_cache, 0.3,
    ) is contact_candidate


@pytest.mark.parametrize(
    "cache",
    (
        _bridge_cache(
            local_positions=[[0, 0, 0]],
            local_min_distances=[0.1],
        ),
        _bridge_cache(
            global_positions=[[0, 0, 0]],
            global_min_distances=[0.1],
        ),
    ),
)
def test_best_bridge_rejects_local_or_global_overlap(cache):
    """Return ``None`` when either screening stage finds a real overlap."""
    assert _best_bridge_position([[0, 0, 0]], cache, 0.3) is None


def test_batch_filter_normalizes_dtypes_without_mutating_source():
    """Normalize kernel arrays and preserve source values and dtypes."""
    batch = _StericAtomBatch(
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        np.array([0.1, 0.2], dtype=np.float32),
        np.array([7, -1], dtype=np.int16),
    )
    source = _copy_batch(batch)
    positions, radii, atom_ids = _filtered_steric_batch_arrays(batch)
    assert positions.dtype == radii.dtype == np.float64
    assert atom_ids.dtype == np.int64
    np.testing.assert_array_equal(positions, source[0])
    np.testing.assert_array_equal(radii, source[1])
    np.testing.assert_array_equal(atom_ids, source[2])
    for actual, expected in zip(_copy_batch(batch), source):
        np.testing.assert_array_equal(actual, expected)


def test_batch_filter_removes_repeated_ids_in_requested_order():
    """Remove all matching scaffold and sentinel rows without reordering."""
    batch = _StericAtomBatch(
        np.arange(15, dtype=float).reshape(5, 3),
        np.arange(5, dtype=float) / 10,
        np.array([7, -1, 2, 7, 9]),
    )
    positions, radii, atom_ids = _filtered_steric_batch_arrays(
        batch, np.array([99, 7, -1], dtype=np.int64),
    )
    np.testing.assert_array_equal(positions, [[6, 7, 8], [12, 13, 14]])
    np.testing.assert_array_equal(radii, [0.2, 0.4])
    np.testing.assert_array_equal(atom_ids, [2, 9])


@pytest.mark.parametrize("ignored_ids", (None, np.empty(0, dtype=np.int64)))
def test_batch_filter_reshapes_empty_positions(ignored_ids):
    """Return the existing native empty shapes after dtype normalization."""
    positions, radii, atom_ids = _filtered_steric_batch_arrays(
        _StericAtomBatch.empty(), ignored_ids,
    )
    assert positions.shape == (0, 3)
    assert radii.shape == atom_ids.shape == (0,)
    assert positions.dtype == radii.dtype == np.float64
    assert atom_ids.dtype == np.int64


@pytest.mark.parametrize(
    ("box", "positions", "reference_positions"),
    (
        ((2.0, 3.0, 4.0), [[0.05, 2.9, 0.1]], [[1.95, 0.1, 3.9]]),
        ((2.0, 3.0, 4.0), [[8.05, -6.1, 12.1]], [[-5.95, 9.1, -8.1]]),
        ((2.0, 0.0, -1.0), [[0.05, -2.0, 3.0]], [[1.95, 2.0, -1.0]]),
    ),
)
def test_full_clearance_matches_independent_periodic_reference(
    box, positions, reference_positions,
):
    """Match scalar image enumeration across periodic and unwrapped axes."""
    radii = [0.11]
    reference_radii = [0.17]
    batch = _StericAtomBatch(
        np.asarray(reference_positions),
        np.asarray(reference_radii),
        np.array([7]),
    )
    expected = _brute_force_clearance(
        positions, radii, reference_positions, reference_radii, box, 0.6,
    )
    assert _positions_clearance(
        positions, radii, batch, set(), 0.6, box,
    ) == pytest.approx(expected, abs=1e-14)


def test_full_clearance_preserves_candidate_and_reference_pair_order():
    """Reduce multiple candidates and references to the first numeric minimum."""
    positions = np.array([[0.1, 0.2, 0.3], [1.9, 2.8, 3.7]])
    radii = np.array([0.1, 0.2])
    reference_positions = np.array(
        [[1.95, 0.2, 0.3], [1.1, 1.2, 1.3], [1.9, 2.8, 3.7]],
    )
    reference_radii = np.array([0.1, 0.2, 0.05])
    batch = _StericAtomBatch(
        reference_positions, reference_radii, np.array([7, 8, -1]),
    )
    expected = _brute_force_clearance(
        positions,
        radii,
        reference_positions[[1, 2]],
        reference_radii[[1, 2]],
        [2, 3, 4],
        1.0,
    )
    assert _positions_clearance(
        positions, radii, batch, {7}, 1.0, [2, 3, 4],
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("distance", "scale", "expected"),
    ((0.2, 1.0, 0.0), (0.2, 0.5, 0.1), (0.19, 1.0, -0.01)),
)
def test_contact_threshold_sign_is_preserved(distance, scale, expected):
    """Keep exact contact at zero and the existing clearance sign convention."""
    batch = _StericAtomBatch(
        np.array([[distance, 0, 0]]), np.array([0.1]), np.array([7]),
    )
    assert _positions_clearance(
        [[0, 0, 0]], [0.1], batch, set(), scale, [1, 1, 1],
    ) == pytest.approx(expected, abs=1e-15)


def test_empty_candidates_references_and_exclusions_return_infinity():
    """Return infinity whenever no candidate-reference pair participates."""
    populated = _StericAtomBatch(
        np.array([[0.2, 0, 0]]), np.array([0.1]), np.array([7]),
    )
    assert np.isposinf(
        _positions_clearance([], [], populated, set(), 1.0, [1, 1, 1])
    )
    assert np.isposinf(
        _positions_clearance(
            [[0, 0, 0]], [0.1], _StericAtomBatch.empty(), set(), 1.0, [1, 1, 1],
        )
    )
    assert np.isposinf(
        _positions_clearance(
            [[0, 0, 0]], [0.1], populated, {7}, 1.0, [1, 1, 1],
        )
    )
    assert np.isposinf(
        _positions_clearance(
            [[0, 0, 0]], [0.1], _StericGrid((1, 1, 1)), set(), 1.0, [1, 1, 1],
        )
    )


def test_prefilter_and_kernel_keep_distinct_sentinel_exclusion_behavior():
    """Document sentinel removal by the wrapper but not the low-level kernel."""
    batch = _StericAtomBatch(
        np.array([[0.05, 0, 0], [0.5, 0, 0]]),
        np.array([0.1, 0.1]),
        np.array([-1, 7]),
    )
    wrapper_result = _positions_clearance(
        [[0, 0, 0]], [0.1], batch, {-1}, 1.0, [2, 2, 2],
    )
    kernel_result = minimum_clearance_against_batch(
        np.array([[0, 0, 0]], dtype=np.float64),
        np.array([0.1], dtype=np.float64),
        batch.positions.astype(np.float64),
        batch.radii.astype(np.float64),
        batch.block_atom_ids.astype(np.int64),
        np.array([-1], dtype=np.int64),
        np.array([2, 2, 2], dtype=np.float64),
        1.0,
    )
    assert wrapper_result == pytest.approx(0.3)
    assert kernel_result == pytest.approx(-0.15)


def test_local_grid_and_full_reference_modes_intentionally_differ():
    """Do not broaden fixed local neighborhoods into a global search."""
    grid = _StericGrid((1, 1, 1))
    grid.add_block_atoms(
        [7, 8],
        [[0.9, 0, 0], [0.5, 0, 0]],
        [0.05, 1.0],
    )
    full = _StericAtomBatch.concatenate(list(grid.cells.values()))
    local_result = _positions_clearance(
        [[0, 0, 0]], [0.05], grid, set(), 1.0, [1, 1, 1],
    )
    full_result = _positions_clearance(
        [[0, 0, 0]], [0.05], full, set(), 1.0, [1, 1, 1],
    )
    assert local_result == pytest.approx(0.0, abs=1e-15)
    assert full_result == pytest.approx(-0.55)


def test_local_grid_groups_cells_and_filters_scaffold_ids():
    """Match independently assembled local candidates across several cells."""
    grid = _StericGrid((1, 1.25, 1.5))
    references = []
    for index, key in enumerate(product(range(4), range(5), range(6))):
        position = np.asarray(key, dtype=float) * 0.25 + 0.05
        grid.add_block_atom(index, position, 0.05 + index * 0.0001)
        references.append((index, position, 0.05 + index * 0.0001))
    positions = np.array([[0.02, 0.02, 0.02], [0.20, 0.20, 0.20], [0.52, 0.52, 0.52]])
    radii = np.array([0.07, 0.08, 0.09])
    ignored = {0, 119}
    expected = float("inf")
    for position, radius in zip(positions, radii):
        center = grid._cell_key(position)
        selected = [
            (reference_position, reference_radius)
            for atom_id, reference_position, reference_radius in references
            if atom_id not in ignored
            and all(
                min(abs(cell - center_cell), size - abs(cell - center_cell)) <= 1
                for cell, center_cell, size in zip(
                    grid._cell_key(reference_position), center, grid.dims,
                )
            )
        ]
        candidate = _brute_force_clearance(
            [position],
            [radius],
            [item[0] for item in selected],
            [item[1] for item in selected],
            grid.box,
            0.6,
        )
        expected = min(expected, candidate)
    result = _positions_clearance(
        positions, radii, grid, ignored, 0.6, grid.box,
    )
    assert result == pytest.approx(expected, abs=1e-14)


def test_local_grid_keeps_first_equal_minimum_across_distinct_cells():
    """Exercise strict clearance reduction for equal results in cell order."""
    grid = _StericGrid((2.0, 0.5, 0.5))
    grid.add_block_atoms(
        [7, 8],
        [[0.25, 0.125, 0.125], [1.25, 0.125, 0.125]],
        [0.0625, 0.0625],
    )
    result = _positions_clearance(
        [[0.125, 0.125, 0.125], [1.125, 0.125, 0.125]],
        [0.0625, 0.0625],
        grid,
        set(),
        1.0,
        grid.box,
    )
    assert result == 0.0


def test_clearance_queries_do_not_mutate_inputs_or_reference_containers():
    """Leave supplied coordinate arrays and grid mappings unchanged."""
    positions = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    radii = np.array([0.1], dtype=np.float32)
    grid = _StericGrid((1, 1, 1))
    grid.add_block_atom(7, [0.2, 0.2, 0.3], 0.1)
    positions_copy = positions.copy()
    radii_copy = radii.copy()
    cells_before = {
        key: _copy_batch(batch) for key, batch in grid.cells.items()
    }
    block_cells_before = dict(grid.block_cells)
    _positions_clearance(positions, radii, grid, {99}, 0.6, grid.box)
    np.testing.assert_array_equal(positions, positions_copy)
    np.testing.assert_array_equal(radii, radii_copy)
    assert grid.block_cells == block_cells_before
    for key, expected_arrays in cells_before.items():
        for actual, expected in zip(_copy_batch(grid.cells[key]), expected_arrays):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("axis", "angle", "is_deg", "expected"),
    (
        (
            [2, 0, 0],
            90.0,
            True,
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        ),
        (
            [0, 0, 3],
            np.pi / 2,
            False,
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        ),
    ),
)
def test_rotation_matrix_preserves_existing_axis_normalization(
    axis, angle, is_deg, expected,
):
    """Match analytic rotations for non-unit axes in degrees and radians."""
    np.testing.assert_allclose(
        _rotation_matrix(axis, angle, is_deg=is_deg),
        expected,
        atol=1e-15,
    )


def test_zero_rotation_axis_keeps_existing_degenerate_matrix():
    """Characterize the existing zero-axis arithmetic without tightening it."""
    np.testing.assert_allclose(
        _rotation_matrix([0, 0, 0], 60),
        0.5 * np.eye(3),
        atol=1e-15,
    )


def test_rotation_around_fixed_origin_preserves_distances_and_inputs():
    """Rotate every coordinate about a supplied origin without mutation."""
    positions = np.array([[1, 2, 3], [2, 2, 3], [1, 4, 3]], dtype=np.float32)
    origin = np.array([1, 2, 3], dtype=np.float32)
    source = positions.copy()
    rotated = _rotate_positions_around_axis(positions, origin, [0, 0, 2], 90)
    np.testing.assert_allclose(
        rotated,
        [[1, 2, 3], [1, 3, 3], [-1, 2, 3]],
        atol=1e-15,
    )
    np.testing.assert_allclose(
        np.linalg.norm(rotated - origin, axis=1),
        np.linalg.norm(source - origin, axis=1),
        atol=1e-15,
    )
    np.testing.assert_array_equal(positions, source)


@pytest.mark.parametrize(
    ("step", "expected"),
    (
        (100.0, (0.0, 100.0, 200.0, 300.0)),
        (720.0, (0.0,)),
        (90.0, (0.0, 90.0, 180.0, 270.0)),
        (np.inf, (0.0,)),
        (np.nan, (0.0,)),
    ),
)
def test_rotation_angle_sequence_preserves_repeated_addition(step, expected):
    """Retain ordering, upper-bound exclusion, and existing nonfinite behavior."""
    assert _rotation_angles(step) == expected


def test_nondividing_rotation_step_is_rounded_after_each_append():
    """Keep repeated-addition samples rather than replacing them with linspace."""
    angles = _rotation_angles(17.0)
    assert len(angles) == 22
    assert angles[:4] == (0.0, 17.0, 34.0, 51.0)
    assert angles[-1] == 357.0
    assert _rotation_angles(0.1)[7] == 0.7


@pytest.mark.parametrize("step", (0.0, -1.0))
def test_rotation_angle_sequence_rejects_nonpositive_steps(step):
    """Preserve the private helper's established exception contract."""
    with pytest.raises(
        ValueError, match="Attachment rotation step must be greater than zero",
    ):
        _rotation_angles(step)


def test_best_pose_keeps_first_exactly_tied_rotation():
    """Use strict score improvement so equal candidates retain the first pose."""
    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    reference = _StericAtomBatch(
        np.array([[0, 0, 0]], dtype=float),
        np.array([0.1]),
        np.array([7]),
    )
    result = _best_pose_positions(
        positions,
        [1],
        [0.1],
        [0, 0, 0],
        [0, 0, 1],
        (90.0, 180.0, 270.0),
        reference,
        set(),
        1.0,
        [4, 4, 4],
    )
    np.testing.assert_allclose(result, [[0, 0, 0], [0, 1, 0]], atol=1e-15)


def test_best_pose_rotates_all_atoms_but_scores_selected_subset():
    """Transform unscored atoms when a later scored pose has greater clearance."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]], dtype=float)
    reference = _StericAtomBatch(
        np.array([[0.8, 0, 0]], dtype=float),
        np.array([0.1]),
        np.array([7]),
    )
    result = _best_pose_positions(
        positions,
        [1],
        [0.1],
        [0, 0, 0],
        [0, 0, 1],
        (0.0, 180.0),
        reference,
        set(),
        1.0,
        [6, 6, 6],
    )
    np.testing.assert_allclose(result, [[0, 0, 0], [-1, 0, 0], [0, -2, 0]], atol=1e-15)


@pytest.mark.parametrize(
    ("distance", "radius", "expected_none"),
    (
        (0.25, 0.125, False),
        (0.2, 0.1, False),
        (0.2 - 5e-13, 0.1, False),
        (0.2 - 2e-12, 0.1, True),
        (0.19, 0.1, True),
    ),
)
def test_best_pose_uses_numerical_tolerance_at_contact(
    distance, radius, expected_none,
):
    """Accept roundoff at contact without admitting larger overlaps."""
    reference = _StericAtomBatch(
        np.array([[distance, 0, 0]]), np.array([radius]), np.array([7]),
    )
    result = _best_pose_positions(
        [[0, 0, 0]],
        [0],
        [radius],
        [0, 0, 0],
        [0, 0, 1],
        (0.0,),
        reference,
        set(),
        1.0,
        [2, 2, 2],
    )
    assert (result is None) is expected_none


def test_empty_steric_subset_accepts_first_pose_and_transforms_unscored_atoms():
    """Use infinite clearance without requiring any reference atoms."""
    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    result = _best_pose_positions(
        positions,
        [],
        [],
        [0, 0, 0],
        [0, 0, 1],
        (90.0, 180.0),
        _StericAtomBatch.empty(),
        set(),
        1.0,
        [2, 2, 2],
    )
    np.testing.assert_allclose(result, [[0, 0, 0], [0, 1, 0]], atol=1e-15)
    np.testing.assert_array_equal(positions, [[0, 0, 0], [1, 0, 0]])
    assert result.dtype == np.dtype(float)


def test_pose_selection_without_angles_returns_none():
    """Preserve the existing negative-infinity result for no evaluated poses."""
    assert _best_pose_positions(
        [[0, 0, 0]],
        [0],
        [0.1],
        [0, 0, 0],
        [0, 0, 1],
        (),
        _StericAtomBatch.empty(),
        set(),
        1.0,
        [1, 1, 1],
    ) is None


def test_pose_selection_does_not_mutate_inputs_or_reference():
    """Return independent coordinates while preserving all supplied state."""
    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    atom_ids = np.array([1], dtype=np.int16)
    radii = np.array([0.1], dtype=np.float32)
    reference = _StericAtomBatch(
        np.array([[0, 0, 0]], dtype=np.float32),
        np.array([0.1], dtype=np.float32),
        np.array([7], dtype=np.int16),
    )
    inputs = (positions.copy(), atom_ids.copy(), radii.copy())
    reference_arrays = _copy_batch(reference)
    result = _best_pose_positions(
        positions,
        atom_ids,
        radii,
        [0, 0, 0],
        [0, 0, 1],
        (0.0, 90.0),
        reference,
        set(),
        1.0,
        [4, 4, 4],
    )
    for actual, expected in zip((positions, atom_ids, radii), inputs):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(_copy_batch(reference), reference_arrays):
        np.testing.assert_array_equal(actual, expected)
    assert result is not positions
    assert not np.shares_memory(result, positions)


def test_engine_pose_adapter_collects_one_reference_and_copies_on_success():
    """Keep chemistry collection and molecule ownership in the engine adapter."""
    molecule = generic.tms()
    source_positions = molecule.positions_view().copy()
    molecule_batch = _MoleculeStericBatch(
        atom_ids=np.array([0], dtype=int),
        positions=source_positions[[0]].copy(),
        radii=np.array([0.1]),
    )
    reference_calls = []
    engine = object.__new__(_SilicaChemistryEngine)
    engine._block = SimpleNamespace(get_box=lambda: [4, 4, 4])
    engine._molecule_steric_batch = lambda candidate: molecule_batch
    engine._reference_steric_batch = lambda: (
        reference_calls.append(True) or _StericAtomBatch.empty()
    )
    result = engine._optimize_attachment_pose(
        molecule,
        mount=0,
        surf_axis=[0, 0, 1],
        ignored_block_atoms=set(),
        steric_grid=None,
        is_rotate=False,
        rotate_step_deg=0.0,
        steric_clearance_scale=1.0,
    )
    assert reference_calls == [True]
    assert result is not molecule
    np.testing.assert_array_equal(result.positions_view(), source_positions)
    np.testing.assert_array_equal(molecule.positions_view(), source_positions)
    assert not np.shares_memory(result.positions_view(), molecule.positions_view())


def test_engine_pose_adapter_skips_reference_collection_for_empty_subset():
    """Accept the first pose without collecting live state for unscored atoms."""
    molecule = generic.tms()
    source_positions = molecule.positions_view().copy()
    engine = object.__new__(_SilicaChemistryEngine)
    engine._block = SimpleNamespace(get_box=lambda: [4, 4, 4])
    engine._molecule_steric_batch = lambda candidate: _MoleculeStericBatch(
        atom_ids=np.empty(0, dtype=int),
        positions=np.empty((0, 3), dtype=float),
        radii=np.empty(0, dtype=float),
    )
    engine._reference_steric_batch = lambda: pytest.fail(
        "empty steric subsets must not collect references"
    )
    result = engine._optimize_attachment_pose(
        molecule,
        mount=0,
        surf_axis=[0, 0, 1],
        ignored_block_atoms=set(),
        steric_grid=None,
        is_rotate=True,
        rotate_step_deg=90.0,
        steric_clearance_scale=1.0,
    )
    np.testing.assert_array_equal(result.positions_view(), source_positions)
    np.testing.assert_array_equal(molecule.positions_view(), source_positions)
