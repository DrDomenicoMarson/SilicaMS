"""Tests for the shared periodic slit-geometry and PBC contracts."""

from __future__ import annotations

import numpy as np
import pytest

import silicams as sms
from silicams._numba_kernels import minimum_image_component


def test_reference_pbc_primitives_handle_seams_and_exact_half_boxes() -> None:
    """Reference PBC helpers should preserve direction at exact half boxes."""

    box = (2.0, 4.0, 6.0)
    reference = np.array([0.0, 0.0, 0.0])
    targets = np.array(
        [
            [1.0, -2.0, 3.0],
            [-1.0, 2.0, -3.0],
            [1.9, 3.9, 5.9],
        ]
    )

    displacements = sms.minimum_image_displacements(reference, targets, box)

    assert np.allclose(displacements[0], [1.0, -2.0, 3.0])
    assert np.allclose(displacements[1], [-1.0, 2.0, -3.0])
    assert np.allclose(displacements[2], [-0.1, -0.1, -0.1])
    assert np.allclose(
        sms.wrap_positions(np.array([[-0.1, 4.1, 12.2]]), box),
        [[1.9, 0.1, 0.2]],
    )


def test_pairwise_distances_use_minimum_images_across_box_seams() -> None:
    """Sites on opposite box faces should be periodic near neighbors."""

    positions = np.array([[0.05, 0.2, 0.3], [1.95, 0.2, 0.3], [1.0, 0.2, 0.3]])
    distances = sms.pairwise_minimum_image_distances(positions, (2.0, 2.0, 2.0))

    assert distances[0, 1] == pytest.approx(0.1)
    assert distances[0, 2] == pytest.approx(0.95)
    assert np.allclose(distances, distances.T)


def test_inferred_mean_planes_are_translation_and_permutation_consistent() -> None:
    """Periodic translation and axis permutation should preserve slit metrics."""

    box = np.array([4.0, 6.0, 8.0])
    surface_positions = np.array(
        [
            [0.2, 1.0, 0.4],
            [3.7, 1.2, 7.4],
            [0.4, 5.0, 7.7],
            [3.5, 5.2, 0.2],
        ]
    )
    geometry = sms.PeriodicSlitGeometry.from_surface_positions(
        box,
        surface_positions,
        normal_axis_index=1,
    )

    assert geometry.lower_plane_nm == pytest.approx(1.1)
    assert geometry.upper_plane_nm == pytest.approx(5.1)
    assert geometry.plane_separation_nm == pytest.approx(4.0)
    assert geometry.normal_roughness_rms_nm == pytest.approx(0.1)
    assert geometry.projected_area_per_face_nm2 == pytest.approx(32.0)
    assert geometry.total_projected_surface_area_nm2 == pytest.approx(64.0)
    assert geometry.geometric_slit_volume_nm3 == pytest.approx(128.0)

    translated = surface_positions.copy()
    translated[:, 1] += 2.0
    translated_geometry = sms.PeriodicSlitGeometry.from_surface_positions(
        box,
        translated,
        normal_axis_index=1,
    )
    assert translated_geometry.lower_plane_nm == pytest.approx(3.1)
    assert translated_geometry.upper_plane_nm == pytest.approx(1.1)
    assert translated_geometry.interval_wraps
    assert translated_geometry.plane_separation_nm == pytest.approx(
        geometry.plane_separation_nm
    )
    assert translated_geometry.geometric_slit_volume_nm3 == pytest.approx(
        geometry.geometric_slit_volume_nm3
    )

    complementary_geometry = sms.PeriodicSlitGeometry.from_surface_positions(
        box,
        surface_positions,
        normal_axis_index=1,
        interior_reference_position_nm=np.array([0.0, 0.0, 0.0]),
    )
    assert complementary_geometry.lower_plane_nm == pytest.approx(5.1)
    assert complementary_geometry.upper_plane_nm == pytest.approx(1.1)
    assert complementary_geometry.plane_separation_nm == pytest.approx(2.0)
    assert bool(complementary_geometry.contains_positions(np.array([0.0, 0.0, 0.0])))

    permuted = geometry.permute_axes((1, 2, 0))
    assert permuted.box_lengths_nm == pytest.approx((6.0, 8.0, 4.0))
    assert permuted.normal_axis_index == 0
    assert permuted.plane_separation_nm == pytest.approx(geometry.plane_separation_nm)
    assert permuted.projected_area_per_face_nm2 == pytest.approx(
        geometry.projected_area_per_face_nm2
    )


def test_padding_controls_containment_and_geometric_volume() -> None:
    """Signed padding should be validated and used consistently."""

    geometry = sms.PeriodicSlitGeometry(
        box_lengths_nm=(2.0, 3.0, 4.0),
        normal_axis_index=0,
        lower_plane_nm=1.8,
        upper_plane_nm=0.8,
    )
    coordinates = np.array(
        [
            [1.9, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.7, 0.0, 0.0],
            [0.9, 0.0, 0.0],
        ]
    )

    assert geometry.contains_positions(coordinates).tolist() == [True, True, False, False]
    assert geometry.contains_positions(coordinates, padding_nm=0.2).tolist() == [False, True, False, False]
    assert geometry.padded_width_nm(0.2) == pytest.approx(0.6)
    assert geometry.padded_geometric_volume_nm3(0.2) == pytest.approx(7.2)
    assert geometry.padded_width_nm(-0.5) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="positive interval"):
        geometry.padded_width_nm(0.5)
    with pytest.raises(ValueError, match="no wider"):
        geometry.padded_width_nm(-0.6)


@pytest.mark.parametrize(
    "delta, box_length",
    [
        (1.0, 2.0),
        (-1.0, 2.0),
        (1.9, 2.0),
        (-1.9, 2.0),
        (4.1, 2.0),
        (-4.1, 2.0),
    ],
)
def test_numba_minimum_image_component_matches_reference(
    delta: float,
    box_length: float,
) -> None:
    """The optimized scalar kernel should match the reference PBC contract."""

    expected = sms.minimum_image_displacements(
        np.zeros(3),
        np.array([delta, 0.0, 0.0]),
        np.array([box_length, 1.0, 1.0]),
    )[0]
    assert minimum_image_component(delta, box_length) == pytest.approx(expected)
