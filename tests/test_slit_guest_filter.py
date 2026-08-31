"""Whole-residue selection and periodic guest/slit clash regression tests."""

from __future__ import annotations

import ast
from dataclasses import replace
import inspect
from itertools import count, product

import numpy as np
import pytest

import silicams._gro_io as gro_io
import silicams._slit_guest_filter as filtering
from silicams.slit_geometry import PeriodicSlitGeometry


@pytest.fixture
def system_factory(tmp_path, write_gro):
    """Provide parsed GRO fixtures while retaining exact test coordinates."""

    serial = count()

    def build(atoms, box=(4.0, 5.0, 6.0)):
        """Build a system from atom records and orthorhombic box lengths.

        Parameters
        ----------
        atoms : list[tuple]
            Shared GRO-builder records with positions in nanometers.
        box : tuple[float, float, float], optional
            Box lengths in nanometers.

        Returns
        -------
        _GroSystem
            Parsed identifiers and spans with float64 coordinates restored
            before GRO rounding, allowing exact numerical-boundary tests.
        """

        path = tmp_path / f"system-{next(serial)}.gro"
        write_gro(path, atoms, box)
        return replace(
            gro_io._load_gro_system(path),
            coordinates=np.array(
                [atom[4:] for atom in atoms], dtype=np.float64
            ).reshape(-1, 3),
            box_lengths=np.array(box, dtype=np.float64),
        )

    return build


def _classify(guest, slit, selected=None, *, cutoff=0.1, include_hydrogen=True):
    """Invoke real filtering using supplied systems in one coordinate frame.

    Parameters
    ----------
    guest, slit : _GroSystem
        Guest and framework systems with positions and box lengths in nm.
    selected : ndarray or None, optional
        Residue mask; select all guest spans when omitted.
    cutoff : float, optional
        General all-atom distance cutoff in nm.
    include_hydrogen : bool, optional
        Whether guest covalent bonds to hydrogen enter forward ring checks.

    Returns
    -------
    tuple[_ClashSelection, _RingCheckCache]
        Real selection masks and cached geometry.
    """

    if selected is None:
        selected = np.ones(len(guest.residue_spans), dtype=bool)
    return filtering._identify_clashing_guest_residues(
        guest,
        slit,
        guest.coordinates,
        selected,
        slit.coordinates,
        slit.box_lengths,
        cutoff,
        "CA",
        0.04,
        0.02,
        include_hydrogen,
    )


@pytest.fixture
def analytic_ring():
    """Return an independently specified planar hexagon centered at the origin."""

    polygon = np.array(
        [
            [0.25, 0.0],
            [0.125, 0.25],
            [-0.125, 0.25],
            [-0.25, 0.0],
            [-0.125, -0.25],
            [0.125, -0.25],
        ]
    )
    return filtering._RingGeometry(
        0,
        "RING",
        np.zeros(3),
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        polygon,
        float(np.max(np.linalg.norm(polygon, axis=1))),
    )


def test_private_filter_has_no_workflow_or_io_dependencies():
    """Keep filtering private and independent of workflow/report ownership."""

    tree = ast.parse(inspect.getsource(filtering))
    imports = [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    imports += [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not any(
        token in name
        for name in imports
        for token in (
            "slit_fill",
            "slit_density",
            "_slit_report",
            "yaml",
            "argparse",
            "pathlib",
        )
    )
    assert filtering.__all__ == []
    assert not hasattr(filtering, "_shift_residue_near_reference")
    assert not hasattr(filtering, "SlitFillConfig")


def test_crop_keeps_whole_spans_repeated_ids_and_source_arrays(system_factory):
    """Reject an entire residue when one atom fails the crop, without mutation."""

    system = system_factory(
        [
            (1, "A", "C1", 1, 1.0, 2.0, 3.0),
            (1, "A", "C2", 2, 1.1, 2.0, 3.0),
            (2, "B", "C1", 3, 1.5, 2.0, 3.0),
            (2, "B", "C2", 4, 0.9, 2.0, 3.0),
            (1, "A", "C1", 5, 2.9, 4.9, 6.9),
        ],
        (4.0, 6.0, 8.0),
    )
    source = system.coordinates.copy()
    velocities = np.arange(15.0).reshape(5, 3)
    system = replace(system, velocities=velocities)
    system.coordinates.setflags(write=False)
    velocities.setflags(write=False)
    translated, selected, origin = filtering._center_crop_guest_residues(
        system, np.array([2.0, 4.0, 6.0])
    )
    np.testing.assert_array_equal(selected, [True, False, True])
    np.testing.assert_array_equal(origin, [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(translated[[0, 1, 4]], source[[0, 1, 4]] - origin)
    np.testing.assert_array_equal(translated[2:4], np.zeros((2, 3)))
    np.testing.assert_array_equal(system.coordinates, source)
    np.testing.assert_array_equal(system.velocities, np.arange(15.0).reshape(5, 3))
    assert selected.dtype == np.bool_
    assert translated.dtype == origin.dtype == np.float64
    assert not np.shares_memory(translated, system.coordinates)


@pytest.mark.parametrize(
    "position,expected",
    (
        (1.0 - 1e-6, True),
        (np.nextafter(1.0 - 1e-6, -np.inf), False),
        (np.nextafter(3.0 + 1e-6, -np.inf), True),
        (3.0 + 1e-6, False),
    ),
)
def test_crop_tolerance_keeps_inclusive_lower_exclusive_upper(
    system_factory, position, expected
):
    """Exercise sub-GRO-precision boundaries using exact in-memory coordinates."""

    system = system_factory([(1, "A", "C1", 1, position, 2.0, 2.0)], (4.0, 4.0, 4.0))
    translated, selected, _ = filtering._center_crop_guest_residues(
        system, np.full(3, 2.0)
    )
    assert selected.tolist() == [expected]
    if not expected:
        np.testing.assert_array_equal(translated, np.zeros((1, 3)))


def test_unwrapping_and_cropping_preserve_first_atom_image(system_factory):
    """Keep seam-spanning residues in the first atom's image, not a centered image."""

    coordinates = np.array([[3.9, 1.0, 1.0], [0.1, 1.0, 1.0]])
    coordinates.setflags(write=False)
    unwrapped = filtering._unwrap_residue_coordinates(coordinates, np.full(3, 4.0))
    np.testing.assert_allclose(unwrapped, [[3.9, 1.0, 1.0], [4.1, 1.0, 1.0]])
    system = system_factory(
        [
            (1, "A", "C1", 1, *coordinates[0]),
            (1, "A", "C2", 2, *coordinates[1]),
        ],
        (4.0, 4.0, 4.0),
    )
    _, selected, _ = filtering._center_crop_guest_residues(system, system.box_lengths)
    assert selected.tolist() == [False]


@pytest.mark.parametrize("axis", (0, 1, 2))
@pytest.mark.parametrize("wraps", (False, True))
@pytest.mark.parametrize("padding", (-0.05, 0.0, 0.05))
def test_plane_filter_checks_all_atoms_across_periodic_intervals(
    system_factory, axis, wraps, padding
):
    """Test independently positioned inside/outside atoms on both padded faces."""

    box = np.array([4.0, 5.0, 6.0])
    lower, upper = (box[axis] - 0.5, 0.5) if wraps else (0.5, box[axis] - 0.5)
    geometry = PeriodicSlitGeometry(tuple(box), axis, lower, upper)
    values = [
        lower + padding + 0.001,
        upper - padding - 0.001,
        lower + padding - 0.001,
        lower + padding + 0.001,
        upper - padding + 0.001,
        upper - padding - 0.001,
        lower + padding + 0.001,
    ]
    atoms = []
    for index, (resid, value) in enumerate(zip((1, 1, 2, 2, 3, 3, 4), values)):
        position = np.ones(3)
        position[axis] = value % box[axis]
        atoms.append((resid, "A", f"C{index}", index + 1, *position))
    system = system_factory(atoms, tuple(box))
    selected = np.array([True, True, True, False])
    selected.setflags(write=False)
    system.coordinates.setflags(write=False)
    result = filtering._apply_surface_plane_filter(
        system, system.coordinates, selected, geometry, padding
    )
    np.testing.assert_array_equal(
        result.selected_residue_mask, [True, False, False, False]
    )
    np.testing.assert_array_equal(
        result.removed_residue_mask, [False, True, True, False]
    )
    np.testing.assert_array_equal(selected, [True, True, True, False])
    assert result.slit_geometry is geometry
    assert (
        result.selected_residue_mask.dtype
        == result.removed_residue_mask.dtype
        == np.bool_
    )
    assert not np.shares_memory(result.selected_residue_mask, selected)


def test_kept_atom_mask_expands_only_surviving_whole_residues(system_factory):
    """Keep original atom ordering and never retain only part of a residue."""

    system = system_factory(
        [
            (1, "A", "C1", 1, 1.0, 1.0, 1.0),
            (1, "A", "C2", 2, 1.1, 1.0, 1.0),
            (2, "B", "O1", 3, 2.0, 2.0, 2.0),
            (1, "A", "C1", 4, 3.0, 3.0, 3.0),
        ]
    )
    selected = np.array([True, False, True])
    removed = np.array([False, True, True])
    selected.setflags(write=False)
    removed.setflags(write=False)
    kept = filtering._build_kept_guest_atom_mask(system, selected, removed)
    assert kept.dtype == np.bool_
    np.testing.assert_array_equal(kept, [True, True, False, False])


@pytest.mark.parametrize("cutoff", (0.05, 0.25))
@pytest.mark.parametrize("shifted", (False, True))
def test_general_clashes_match_exhaustive_periodic_images(
    system_factory, cutoff, shifted
):
    """Compare real KD-tree masks with every atom pair and all adjacent images."""

    box = np.array([4.0, 5.0, 6.0])
    rng = np.random.default_rng(59)
    framework = rng.uniform(0.0, box, size=(15, 3))
    positions = rng.uniform(0.0, box, size=(24, 3))
    framework[0] = 0.0
    positions[:2] = [[0.01, 0.0, 0.0], [3.99, 0.0, 0.0]]
    selected = np.arange(len(positions)) % 3 != 2
    if shifted:
        framework += box * [1, -2, 3]
        positions += box * [-2, 1, -1]
    slit = system_factory(
        [(i + 1, "SIL", "C1", i + 1, *xyz) for i, xyz in enumerate(framework)]
    )
    guest = system_factory(
        [
            (i + 1, "THY" if i % 2 else "ION", "C1", i + 1, *xyz)
            for i, xyz in enumerate(positions)
        ]
    )
    images = np.array(list(product((-1, 0, 1), repeat=3))) * box
    delta = (positions % box)[:, None, None, :] - (
        (framework % box)[None, :, None, :] + images[None, None, :, :]
    )
    expected = (
        np.any(np.sum(delta * delta, axis=-1) <= cutoff**2, axis=(1, 2)) & selected
    )
    slit.coordinates.setflags(write=False)
    guest.coordinates.setflags(write=False)
    selected.setflags(write=False)
    result, cache = _classify(guest, slit, selected, cutoff=cutoff)
    np.testing.assert_array_equal(result.removed_by_general_mask, expected)
    np.testing.assert_array_equal(result.removed_residue_mask, expected)
    assert not np.any(result.removed_by_any_ring_mask)
    assert len(cache.guest_bond_templates) == 2
    assert cache.guest_bond_geometries == cache.guest_ring_geometries == ()


@pytest.mark.parametrize(
    "distance,expected",
    (
        (np.nextafter(0.25, 0.0), True),
        (0.25, True),
        (np.nextafter(0.25, np.inf), False),
    ),
)
def test_general_clash_cutoff_is_inclusive(system_factory, distance, expected):
    """Keep the exact nearest-distance comparison at the general cutoff."""

    slit = system_factory([(1, "SIL", "C1", 1, 0.0, 0.0, 0.0)])
    guest = system_factory([(1, "ION", "C1", 1, distance, 0.0, 0.0)])
    result, _ = _classify(guest, slit, cutoff=0.25)
    assert result.removed_by_general_mask.tolist() == [expected]


@pytest.mark.parametrize("empty_system", (False, True))
def test_empty_selection_retains_slit_cache_and_zero_masks(
    system_factory, ring_residue, empty_system
):
    """Even no selected guests must still construct the existing slit caches."""

    guest = system_factory([] if empty_system else [(1, "A", "C1", 1, 1.0, 1.0, 1.0)])
    slit = system_factory(ring_residue(1, "RNG", 1, (2.0, 2.0, 2.0)))
    selected = np.zeros(len(guest.residue_spans), dtype=bool)
    result, cache = _classify(guest, slit, selected)
    assert not np.any(result.removed_residue_mask)
    assert result.removed_residue_mask.shape == selected.shape
    assert len(cache.slit_ring_geometries) == len(cache.slit_bond_templates) == 1
    assert len(cache.slit_bond_geometries) == 6
    assert (
        cache.guest_bond_templates
        == cache.guest_bond_geometries
        == cache.guest_ring_geometries
        == ()
    )
    translated, cropped, _ = filtering._center_crop_guest_residues(
        guest, slit.box_lengths
    )
    assert translated.shape == guest.coordinates.shape
    assert cropped.shape == selected.shape


def test_empty_selection_does_not_bypass_slit_lookup_failures(system_factory):
    """Preserve validation ordering instead of returning before slit construction."""

    guest = system_factory([])
    slit = system_factory([(1, "SIL", "ZZZ", 1, 0.0, 0.0, 0.0)])
    with pytest.raises(ValueError, match="Unsupported element"):
        _classify(guest, slit)


@pytest.mark.parametrize(
    "distance,expected",
    (
        (0.0, False),
        (0.05, False),
        (0.0500001, True),
        (0.25, True),
        (0.2500001, False),
    ),
)
def test_bond_inference_distance_bounds_and_radius_scale(
    monkeypatch, distance, expected
):
    """Use known radii to exercise strict lower and inclusive scaled upper bounds."""

    monkeypatch.setattr(filtering.db, "get_covalent_radius", lambda element: 0.1)
    definitions = filtering._guess_bond_definitions(
        ("C1", "C2"), np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]), True
    )
    assert len(definitions) == int(expected)
    if expected:
        assert definitions[0] == filtering._BondDefinition(0, 1, "C1", "C2")


def test_bond_pair_order_hydrogen_option_and_lookup_errors():
    """Keep pair order, covalent-hydrogen exclusions, and chained name failures."""

    coordinates = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.18, 0.0, 0.0]])
    all_bonds = filtering._guess_bond_definitions(("C1", "H1", "C2"), coordinates, True)
    heavy_bonds = filtering._guess_bond_definitions(
        ("C1", "H1", "C2"), coordinates, False
    )
    assert [(bond.start_atom_index, bond.stop_atom_index) for bond in all_bonds] == [
        (0, 1),
        (0, 2),
        (1, 2),
    ]
    assert heavy_bonds == (all_bonds[1],)
    with pytest.raises(
        ValueError, match="Unsupported element inferred from atom name 'ZZZ'"
    ) as caught:
        filtering._guess_bond_definitions(("ZZZ",), coordinates[:1], True)
    assert isinstance(caught.value.__cause__, ValueError)
    with pytest.raises(ValueError, match="Covalent radius not found"):
        filtering._guess_bond_definitions(("NA", "C1"), coordinates[:2], True)


def test_guest_template_cache_preserves_keys_first_occurrence_and_order(system_factory):
    """Reuse first-selected connectivity even when a later conformation differs."""

    system = system_factory(
        [
            (1, "ZZ", "C1", 1, 1.0, 1.0, 1.0),
            (1, "ZZ", "C2", 2, 1.1, 1.0, 1.0),
            (2, "ZZ", "C1", 3, 2.0, 1.0, 1.0),
            (2, "ZZ", "C2", 4, 2.8, 1.0, 1.0),
            (3, "ZZ", "C2", 5, 1.0, 2.0, 1.0),
            (3, "ZZ", "C1", 6, 1.1, 2.0, 1.0),
            (4, "AA", "C1", 7, 2.0, 3.0, 1.0),
            (5, "BAD", "C1", 8, 2.0, 4.0, 1.0),
            (5, "BAD", "C2", 9, 2.8, 4.0, 1.0),
        ]
    )
    selected = np.array([True, True, True, True, False])
    templates, bonds, rings = filtering._build_guest_filter_geometries(
        system, system.coordinates, selected, system.box_lengths, "CA", True
    )
    assert [(template.residue_name, template.atom_names) for template in templates] == [
        ("ZZ", ("C1", "C2")),
        ("ZZ", ("C2", "C1")),
        ("AA", ("C1",)),
    ]
    assert [bond.residue_index for bond in bonds] == [0, 1, 2]
    np.testing.assert_allclose(
        [bond.half_length_nm for bond in bonds], [0.05, 0.4, 0.05]
    )
    assert rings == ()
    selected[0] = False
    with pytest.raises(ValueError, match="No covalent bonds were guessed.*'ZZ'"):
        filtering._build_guest_filter_geometries(
            system, system.coordinates, selected, system.box_lengths, "CA", True
        )
    slit_templates, slit_bonds = filtering._build_slit_bond_geometries(
        system, system.coordinates, system.box_lengths
    )
    assert [
        (template.residue_name, template.atom_names) for template in slit_templates[:-1]
    ] == [(template.residue_name, template.atom_names) for template in templates]
    assert slit_templates[-1].residue_name == "BAD"
    assert slit_templates[-1].bond_definitions == ()
    assert [bond.residue_index for bond in slit_bonds] == [0, 1, 2]


def test_hydrogen_exclusion_is_guest_forward_only(system_factory):
    """The guest option must not remove slit hydrogen segments from the reverse cache."""

    slit = system_factory(
        [
            (1, "SUR", "O1", 1, 1.0, 1.0, 1.0),
            (1, "SUR", "H1", 2, 1.1, 1.0, 1.0),
        ]
    )
    guest = system_factory(
        [
            (1, "SOL", "O1", 1, 2.0, 2.0, 2.0),
            (1, "SOL", "H1", 2, 2.1, 2.0, 2.0),
        ]
    )
    _, cache = _classify(guest, slit, include_hydrogen=False)
    assert cache.guest_bond_geometries == ()
    assert len(cache.slit_bond_geometries) == 1
    assert cache.slit_bond_geometries[0].stop_atom_name == "H1"


def test_general_clash_does_not_suppress_guest_bond_inference_failure(system_factory):
    """A selected invalid multi-atom guest still fails after a general clash."""

    slit = system_factory([(1, "SIL", "C1", 1, 1.0, 1.0, 1.0)])
    guest = system_factory(
        [
            (1, "BAD", "C1", 1, 1.0, 1.0, 1.0),
            (1, "BAD", "C2", 2, 2.0, 1.0, 1.0),
        ]
    )
    with pytest.raises(ValueError, match="No covalent bonds were guessed.*'BAD'"):
        _classify(guest, slit)


@pytest.mark.parametrize("matched_count", (5, 6, 7))
def test_ring_template_uses_exactly_six_prefix_matches_in_source_order(matched_count):
    """Keep prefix counting and input order without introducing name validation."""

    names = ("C1",) + tuple(f"AR{i}" for i in range(matched_count, 0, -1)) + ("H1",)
    result = filtering._build_ring_template_from_atom_names("RNG", names, "AR")
    if matched_count != 6:
        assert result is None
    else:
        assert result == filtering._RingTemplate(
            "RNG", "AR", (1, 2, 3, 4, 5, 6), names[1:-1]
        )
    assert filtering._build_ring_template_from_atom_names("RNG", names, "CA") is None


@pytest.mark.parametrize("axes", ((0, 1, 2), (1, 2, 0), (2, 0, 1)))
def test_ring_geometry_unwraps_seams_without_prescribing_svd_signs(ring_residue, axes):
    """Recover the physical center, radius, and plane of a seam-spanning ring."""

    box = np.array([4.0, 5.0, 6.0])[list(axes)]
    center = np.array([0.02, 2.0, 3.0])[list(axes)]
    atoms = ring_residue(1, "RNG", 1, (0.02, 2.0, 3.0), radius=0.14)
    coordinates = np.array([atom[4:] for atom in atoms])[:, axes] % box
    original = coordinates.copy()
    coordinates.setflags(write=False)
    names = tuple(atom[2] for atom in atoms)
    template = filtering._build_ring_template_from_atom_names("RNG", names, "CA")
    geometry = filtering._build_ring_geometry(7, "RNG", coordinates, template, box)
    np.testing.assert_allclose(geometry.center, center, atol=1e-14)
    np.testing.assert_allclose(geometry.wrapped_center, center, atol=1e-14)
    normal = np.array([0.0, 0.0, 1.0])[list(axes)]
    assert abs(np.dot(geometry.normal, normal)) == pytest.approx(1.0)
    assert np.dot(geometry.basis_u, geometry.basis_v) == pytest.approx(0.0, abs=1e-14)
    assert geometry.polygon_2d.shape == (6, 2)
    assert geometry.max_radius_nm == pytest.approx(0.14)
    assert geometry.residue_index == 7
    angles = np.arctan2(geometry.polygon_2d[:, 1], geometry.polygon_2d[:, 0])
    assert np.all(np.diff(angles) >= 0)
    np.testing.assert_array_equal(coordinates, original)


@pytest.mark.parametrize(
    "point,padding,expected",
    (
        ((0.0, 0.0), 0.0, True),
        ((0.0, 0.25), 0.0, False),
        ((0.0, 0.25), 0.125, True),
        ((0.0, 0.375), 0.125, True),
        ((0.0, 0.375001), 0.125, False),
        ((0.2, 0.125), 0.0, False),
        ((0.2, 0.125), 0.02, True),
    ),
)
def test_polygon_padding_and_existing_zero_padding_boundary(
    analytic_ring, point, padding, expected
):
    """Check known planar locations and the inclusive positive-padding boundary."""

    assert (
        filtering._point_inside_polygon_with_padding(
            np.array(point), analytic_ring.polygon_2d, padding
        )
        == expected
    )


def test_polygon_padding_handles_repeated_vertices():
    """Zero-length edges retain the existing nearest-endpoint projection."""

    polygon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert filtering._point_inside_polygon_with_padding(
        np.array([1.125, 0.5]), polygon, 0.125
    )
    assert not filtering._point_inside_polygon_with_padding(
        np.array([1.126, 0.5]), polygon, 0.125
    )


@pytest.mark.parametrize(
    "start,stop,tolerance,padding,expected",
    (
        ((0, 0, -0.5), (0, 0, 0.5), 0.0, 0.0, True),
        ((0, 0, 0), (0, 0, 0.5), 0.0, 0.0, True),
        ((0, 0, 0.125), (0, 0, 0.5), 0.125, 0.0, True),
        ((0, 0, 0.125001), (0, 0, 0.5), 0.125, 0.0, False),
        ((1, 0, -0.5), (1, 0, 0.5), 0.125, 0.0, False),
        ((0.2, 0.125, -0.5), (0.2, 0.125, 0.5), 0.0, 0.0, False),
        ((0.2, 0.125, -0.5), (0.2, 0.125, 0.5), 0.0, 0.02, True),
        ((0, 0, 0.125), (0.1, 0, 0.125), 0.125, 0.0, True),
        ((-0.5, 0, 0), (0.5, 0, 0), 0.0, 0.0, False),
        ((-0.5, 0, 1e-13), (0, 0, 0), 0.0, 0.0, True),
    ),
)
def test_bond_ring_predicate_preserves_tolerance_and_parallel_semantics(
    analytic_ring, start, stop, tolerance, padding, expected
):
    """Preserve clamped endpoints and near-parallel endpoint choice, not strict intersection."""

    assert (
        filtering._bond_crosses_ring(
            np.array(start, dtype=float),
            np.array(stop, dtype=float),
            analytic_ring,
            tolerance,
            padding,
        )
        == expected
    )


@pytest.mark.parametrize(
    "midpoint,expected_shift",
    (
        ((2.0, 3.0, 4.0), (0.0, 0.0, 0.0)),
        ((6.0, 9.0, 12.0), (8.0, 12.0, 16.0)),
        ((-6.0, -9.0, -12.0), (-8.0, -12.0, -16.0)),
    ),
)
def test_bond_image_shift_preserves_half_box_ties_and_length(midpoint, expected_shift):
    """Use the unchanged nearest-midpoint NumPy rounding for whole-bond shifts."""

    start = np.array(midpoint) - [0, 0, 0.125]
    stop = np.array(midpoint) + [0, 0, 0.125]
    start.setflags(write=False)
    stop.setflags(write=False)
    shifted_start, shifted_stop = filtering._shift_bond_near_reference(
        start, stop, np.zeros(3), np.array([4.0, 6.0, 8.0])
    )
    np.testing.assert_array_equal(shifted_start, start - expected_shift)
    np.testing.assert_array_equal(shifted_stop, stop - expected_shift)
    np.testing.assert_array_equal(shifted_stop - shifted_start, stop - start)
    assert not np.shares_memory(shifted_start, start)


@pytest.mark.parametrize("axes", ((0, 1, 2), (1, 2, 0), (2, 0, 1)))
@pytest.mark.parametrize("shifted", (False, True))
def test_ring_spatial_search_matches_exhaustive_pairs(
    system_factory, ring_residue, bond_residue, axes, shifted
):
    """Validate spatial-tree pruning against every applicable bond/ring pair."""

    box = np.array([4.0, 5.0, 6.0])[list(axes)]
    slit_atoms = (
        ring_residue(1, "SR1", 1, (0.03, 1.0, 1.0), radius=0.12)
        + ring_residue(2, "SR2", 7, (2.0, 3.0, 3.0), radius=0.18)
        + bond_residue(3, "SB1", 13, (1.0, 1.0, 1.92), (1.0, 1.0, 2.08))
    )
    guest_atoms = (
        bond_residue(1, "BND", 1, (0.03, 1.0, 0.92), (0.03, 1.0, 1.08))
        + ring_residue(2, "RNG", 3, (1.0, 1.0, 2.0))
        + [(3, "ION", "C1", 9, 2.0, 1.0, 1.0)]
        + ring_residue(4, "FAR", 10, (3.0, 3.0, 5.0))
        + bond_residue(5, "BND", 16, (0.33, 1.0, 0.92), (0.33, 1.0, 1.08))
        + bond_residue(6, "BAD", 18, (1.0, 3.0, 3.0), (2.0, 3.0, 3.0))
    )
    slit = system_factory(
        [(*atom[:4], *(np.array(atom[4:])[list(axes)] % box)) for atom in slit_atoms],
        tuple(box),
    )
    image = box * [1, -1, 2] if shifted else np.zeros(3)
    guest = system_factory(
        [
            (*atom[:4], *(np.array(atom[4:])[list(axes)] + image))
            for atom in guest_atoms
        ],
        tuple(box),
    )
    selected = np.array([True, True, True, True, True, False])
    result, cache = _classify(guest, slit, selected, cutoff=0.04)
    forward = np.zeros(6, dtype=bool)
    reverse = np.zeros(6, dtype=bool)

    def crosses(bond, ring):
        """Check one pair without a tree or the production image-shift helper.

        Parameters
        ----------
        bond : _BondSegmentGeometry
            Bond endpoints in nanometers.
        ring : _RingGeometry
            Ring center and polygon in the same periodic box.

        Returns
        -------
        bool
            Crossing according to the independently tested narrow predicate.
        """

        # Crossing pairs here avoid half-box ties; exact tie behavior is
        # covered separately. The narrow predicate has analytical tests above;
        # this exhaustive oracle specifically tests broad-phase pruning.
        midpoint = (bond.start_point + bond.stop_point) / 2
        shift = box * np.floor((midpoint - ring.center) / box + 0.5)
        return filtering._bond_crosses_ring(
            bond.start_point - shift, bond.stop_point - shift, ring, 0.04, 0.02
        )

    for bond in cache.guest_bond_geometries:
        for ring in cache.slit_ring_geometries:
            forward[bond.residue_index] |= crosses(bond, ring)
    for ring in cache.guest_ring_geometries:
        for bond in cache.slit_bond_geometries:
            reverse[ring.residue_index] |= crosses(bond, ring)
    np.testing.assert_array_equal(result.removed_by_forward_ring_mask, forward)
    np.testing.assert_array_equal(result.removed_by_reverse_ring_mask, reverse)
    np.testing.assert_array_equal(forward, [True, False, False, False, False, False])
    np.testing.assert_array_equal(reverse, [False, True, False, False, False, False])
    assert len(cache.guest_bond_templates) == 4
    assert len(cache.guest_ring_geometries) == len(cache.slit_ring_geometries) == 2


def test_general_and_both_ring_clashes_are_evaluated_independently(
    system_factory, ring_residue, bond_residue
):
    """Real overlapping geometry must retain all three removal reasons."""

    atoms = ring_residue(1, "MIX", 1, (1.0, 1.0, 1.0), radius=0.12) + bond_residue(
        1, "MIX", 7, (1.0, 1.0, 0.92), (1.0, 1.0, 1.08)
    )
    system = system_factory(atoms)
    result, cache = _classify(system, system)
    for name in (
        "removed_residue_mask",
        "removed_by_general_mask",
        "removed_by_forward_ring_mask",
        "removed_by_reverse_ring_mask",
        "removed_by_any_ring_mask",
    ):
        np.testing.assert_array_equal(getattr(result, name), [True])
    assert len(cache.guest_ring_geometries) == len(cache.slit_ring_geometries) == 1
    assert cache.guest_bond_geometries and cache.slit_bond_geometries


def test_center_crop_guest_residues(module_workspace, write_guest_box) -> None:
    """The centered crop should keep only residues fully inside the slit box."""

    guest_path = module_workspace.root / "guest_crop.gro"
    write_guest_box(guest_path)

    guest_system = gro_io._load_gro_system(guest_path)
    translated_coordinates, selected_mask, crop_window_start = (
        filtering._center_crop_guest_residues(
            guest_system=guest_system,
            final_box_lengths=np.array([2.0, 2.0, 2.0], dtype=float),
        )
    )

    assert selected_mask.tolist() == [True, False]
    assert np.allclose(crop_window_start, np.array([0.5, 0.5, 0.5], dtype=float))
    assert np.allclose(
        translated_coordinates[0],
        guest_system.coordinates[0] - np.array([0.5, 0.5, 0.5], dtype=float),
    )


def test_identify_clashes_detects_forward_ring_crossing(
    module_workspace, write_basic_slit, write_guest_box
) -> None:
    """A guest bond through a slit aromatic ring should trigger the forward mask."""

    slit_path = module_workspace.root / "slit_forward.gro"
    guest_path = module_workspace.root / "guest_forward.gro"
    write_basic_slit(slit_path, include_ring=True, include_crossing_bond=False)
    write_guest_box(
        guest_path,
        include_inside_ring=True,
        include_outside_ring=False,
        include_crossing_bond=True,
    )

    slit_system = gro_io._load_gro_system(slit_path)
    guest_system = gro_io._load_gro_system(guest_path)
    clash_selection, _ = filtering._identify_clashing_guest_residues(
        guest_system=guest_system,
        slit_system=slit_system,
        translated_guest_coordinates=guest_system.coordinates.copy(),
        selected_residue_mask=np.array([True], dtype=bool),
        slit_coordinates=slit_system.coordinates.copy(),
        final_box_lengths=np.array([3.0, 3.0, 3.0], dtype=float),
        general_cutoff_nm=0.05,
        ring_atom_prefix="CA",
        ring_plane_tolerance_nm=0.04,
        ring_polygon_padding_nm=0.02,
        include_hydrogen_bonds_in_ring_check=True,
    )

    assert clash_selection.removed_by_forward_ring_mask.tolist() == [True]
    assert clash_selection.removed_by_reverse_ring_mask.tolist() == [False]


def test_identify_clashes_detects_reverse_ring_crossing(
    module_workspace, write_gro, ring_residue, write_basic_slit
) -> None:
    """A slit bond through a guest aromatic ring should trigger the reverse mask."""

    slit_path = module_workspace.root / "slit_reverse.gro"
    guest_path = module_workspace.root / "guest_reverse.gro"
    write_basic_slit(slit_path, include_ring=False, include_crossing_bond=True)
    write_gro(
        guest_path,
        ring_residue(1, "THY", 1, (1.0, 1.0, 1.0)),
        (3.0, 3.0, 3.0),
        title="guest-reverse",
    )

    slit_system = gro_io._load_gro_system(slit_path)
    guest_system = gro_io._load_gro_system(guest_path)
    clash_selection, _ = filtering._identify_clashing_guest_residues(
        guest_system=guest_system,
        slit_system=slit_system,
        translated_guest_coordinates=guest_system.coordinates.copy(),
        selected_residue_mask=np.array([True], dtype=bool),
        slit_coordinates=slit_system.coordinates.copy(),
        final_box_lengths=np.array([3.0, 3.0, 3.0], dtype=float),
        general_cutoff_nm=0.05,
        ring_atom_prefix="CA",
        ring_plane_tolerance_nm=0.04,
        ring_polygon_padding_nm=0.02,
        include_hydrogen_bonds_in_ring_check=True,
    )

    assert clash_selection.removed_by_forward_ring_mask.tolist() == [False]
    assert clash_selection.removed_by_reverse_ring_mask.tolist() == [True]


def test_guest_with_only_excluded_hydrogen_bonds_is_supported(
    module_workspace, write_gro, write_basic_slit
) -> None:
    """Allow water-like guests when hydrogen bonds are omitted from ring checks."""

    slit_path = module_workspace.root / "slit_water.gro"
    guest_path = module_workspace.root / "guest_water.gro"
    write_basic_slit(slit_path)
    write_gro(
        guest_path,
        [
            (1, "SOL", "OW", 1, 1.0, 1.0, 1.0),
            (1, "SOL", "HW1", 2, 1.096, 1.0, 1.0),
            (1, "SOL", "HW2", 3, 0.968, 1.091, 1.0),
        ],
        (2.0, 2.0, 2.0),
    )
    slit_system = gro_io._load_gro_system(slit_path)
    guest_system = gro_io._load_gro_system(guest_path)

    selection, cache = filtering._identify_clashing_guest_residues(
        guest_system=guest_system,
        slit_system=slit_system,
        translated_guest_coordinates=guest_system.coordinates.copy(),
        selected_residue_mask=np.array([True], dtype=bool),
        slit_coordinates=slit_system.coordinates.copy(),
        final_box_lengths=np.array([2.0, 2.0, 2.0]),
        general_cutoff_nm=0.05,
        ring_atom_prefix="CA",
        ring_plane_tolerance_nm=0.04,
        ring_polygon_padding_nm=0.02,
        include_hydrogen_bonds_in_ring_check=False,
    )

    assert not selection.removed_residue_mask[0]
    assert cache.guest_bond_geometries == ()
