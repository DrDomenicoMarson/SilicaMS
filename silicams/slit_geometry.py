"""Periodic orthorhombic geometry contracts for silica slit workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
AXIS_NAMES = ("x", "y", "z")
_GEOMETRY_TOLERANCE_NM = 1.0e-12


def _validated_box_lengths(box_lengths_nm: ArrayLike) -> FloatArray:
    """Return one validated three-dimensional orthorhombic box.

    Parameters
    ----------
    box_lengths_nm : array-like
        Orthorhombic box lengths in nanometers.

    Returns
    -------
    numpy.ndarray
        Float64 array with shape ``(3,)``.

    Raises
    ------
    ValueError
        Raised when the box does not contain three finite positive lengths.
    """

    box = np.asarray(box_lengths_nm, dtype=np.float64)
    if box.shape != (3,):
        raise ValueError("Periodic slit geometry requires exactly three box lengths.")
    if not np.all(np.isfinite(box)) or np.any(box <= 0.0):
        raise ValueError("Periodic box lengths must be finite and strictly positive.")
    return box


def wrap_positions(coordinates_nm: ArrayLike, box_lengths_nm: ArrayLike) -> FloatArray:
    """Wrap Cartesian coordinates into an orthorhombic periodic box.

    Parameters
    ----------
    coordinates_nm : array-like
        Cartesian coordinates whose final dimension has length three, in
        nanometers.
    box_lengths_nm : array-like
        Orthorhombic box lengths in nanometers.

    Returns
    -------
    numpy.ndarray
        Newly allocated wrapped coordinates with the same shape as the input.

    Raises
    ------
    ValueError
        Raised when the coordinate shape or box is invalid.
    """

    box = _validated_box_lengths(box_lengths_nm)
    coordinates = np.asarray(coordinates_nm, dtype=np.float64)
    if coordinates.ndim == 0 or coordinates.shape[-1] != 3:
        raise ValueError("Cartesian coordinates must have a final dimension of length three.")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Cartesian coordinates must be finite.")
    return np.mod(coordinates, box)


def minimum_image_displacements(
    reference_positions_nm: ArrayLike,
    target_positions_nm: ArrayLike,
    box_lengths_nm: ArrayLike,
) -> FloatArray:
    """Return minimum-image vectors from reference positions to targets.

    Parameters
    ----------
    reference_positions_nm : array-like
        Reference Cartesian positions in nanometers. The final dimension must
        have length three and must broadcast with ``target_positions_nm``.
    target_positions_nm : array-like
        Target Cartesian positions in nanometers. The final dimension must have
        length three and must broadcast with ``reference_positions_nm``.
    box_lengths_nm : array-like
        Orthorhombic box lengths in nanometers.

    Returns
    -------
    numpy.ndarray
        Minimum-image vectors ``target - reference``. Exact positive half-box
        displacements remain positive; exact negative half-box displacements
        remain negative.

    Raises
    ------
    ValueError
        Raised when positions are non-finite, cannot broadcast, or do not have a
        final dimension of length three, or when the box is invalid.
    """

    box = _validated_box_lengths(box_lengths_nm)
    reference = np.asarray(reference_positions_nm, dtype=np.float64)
    target = np.asarray(target_positions_nm, dtype=np.float64)
    if (
        reference.ndim == 0
        or target.ndim == 0
        or reference.shape[-1] != 3
        or target.shape[-1] != 3
    ):
        raise ValueError("Cartesian positions must have a final dimension of length three.")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(target)):
        raise ValueError("Cartesian positions must be finite.")
    try:
        displacement = np.subtract(target, reference, dtype=np.float64)
    except ValueError as error:
        raise ValueError("Reference and target positions could not be broadcast together.") from error

    original_positive = displacement > 0.0
    half_box = 0.5 * box
    wrapped = np.mod(displacement + half_box, box) - half_box
    positive_half_box = np.isclose(
        wrapped,
        -half_box,
        rtol=0.0,
        atol=_GEOMETRY_TOLERANCE_NM,
    ) & original_positive
    wrapped[positive_half_box] = np.broadcast_to(half_box, wrapped.shape)[positive_half_box]
    return wrapped


def pairwise_minimum_image_distances(
    positions_nm: ArrayLike,
    box_lengths_nm: ArrayLike,
) -> FloatArray:
    """Return all pairwise minimum-image distances for Cartesian positions.

    Parameters
    ----------
    positions_nm : array-like
        Cartesian positions with shape ``(n, 3)`` in nanometers.
    box_lengths_nm : array-like
        Orthorhombic box lengths in nanometers.

    Returns
    -------
    numpy.ndarray
        Symmetric distance matrix with shape ``(n, n)``.

    Raises
    ------
    ValueError
        Raised when the positions or box are invalid.
    """

    positions = np.asarray(positions_nm, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Pairwise positions must have shape (n, 3).")
    displacement = minimum_image_displacements(
        positions[:, np.newaxis, :],
        positions[np.newaxis, :, :],
        box_lengths_nm,
    )
    return np.sqrt(np.einsum("ijk,ijk->ij", displacement, displacement))


def _minimum_image_scalar(reference: float, target: FloatArray, box_length: float) -> FloatArray:
    """Return one-dimensional minimum-image displacements for plane fitting."""

    displacement = target - reference
    original_positive = displacement > 0.0
    half_box = 0.5 * box_length
    wrapped = np.mod(displacement + half_box, box_length) - half_box
    wrapped[
        np.isclose(wrapped, -half_box, rtol=0.0, atol=_GEOMETRY_TOLERANCE_NM)
        & original_positive
    ] = half_box
    return wrapped


@dataclass(frozen=True)
class PeriodicSlitGeometry:
    """Physical geometry of one orthorhombic periodic slit.

    The two plane positions are mean surface-site planes. The accessible
    interval proceeds in the positive normal-axis direction from
    ``lower_plane_nm`` to ``upper_plane_nm`` and may cross the periodic box
    boundary.

    Parameters
    ----------
    box_lengths_nm : tuple[float, float, float]
        Orthorhombic periodic box lengths in nanometers.
    normal_axis_index : int
        Cartesian axis normal to the two confining surface planes.
    lower_plane_nm : float
        Wrapped mean position of the first surface plane in nanometers.
    upper_plane_nm : float
        Wrapped mean position of the second surface plane in nanometers.
    surface_support_count : int or None, optional
        Number of surface positions used to infer the planes. ``None`` denotes
        explicitly supplied planes without inference diagnostics.
    normal_roughness_rms_nm : float or None, optional
        Pooled root-mean-square normal displacement from the two fitted mean
        planes in nanometers. ``None`` denotes unavailable diagnostics.
    """

    box_lengths_nm: tuple[float, float, float]
    normal_axis_index: int
    lower_plane_nm: float
    upper_plane_nm: float
    surface_support_count: int | None = None
    normal_roughness_rms_nm: float | None = None

    def __post_init__(self) -> None:
        """Validate and canonicalize the periodic slit geometry.

        Raises
        ------
        TypeError
            Raised when ``normal_axis_index`` is not an integer.
        ValueError
            Raised when box lengths, planes, support diagnostics, or the
            resulting plane separation are invalid.
        """

        box = _validated_box_lengths(self.box_lengths_nm)
        if isinstance(self.normal_axis_index, bool) or not isinstance(self.normal_axis_index, int):
            raise TypeError("The slit normal-axis index must be an integer.")
        if self.normal_axis_index not in (0, 1, 2):
            raise ValueError("The slit normal-axis index must be 0, 1, or 2.")
        if not np.isfinite(self.lower_plane_nm) or not np.isfinite(self.upper_plane_nm):
            raise ValueError("Slit surface-plane positions must be finite.")

        normal_box_length = float(box[self.normal_axis_index])
        lower_plane = float(np.mod(self.lower_plane_nm, normal_box_length))
        upper_plane = float(np.mod(self.upper_plane_nm, normal_box_length))
        separation = float(np.mod(upper_plane - lower_plane, normal_box_length))
        if separation <= _GEOMETRY_TOLERANCE_NM:
            raise ValueError("The two slit surface planes must define a non-zero interval.")

        if self.surface_support_count is not None:
            if isinstance(self.surface_support_count, bool) or not isinstance(
                self.surface_support_count,
                int,
            ):
                raise TypeError("The surface support count must be an integer or None.")
            if self.surface_support_count < 2:
                raise ValueError("At least two surface support positions are required.")
        if self.normal_roughness_rms_nm is not None:
            if (
                not np.isfinite(self.normal_roughness_rms_nm)
                or self.normal_roughness_rms_nm < 0.0
            ):
                raise ValueError("The normal RMS roughness must be finite and non-negative.")

        object.__setattr__(self, "box_lengths_nm", tuple(float(value) for value in box))
        object.__setattr__(self, "lower_plane_nm", lower_plane)
        object.__setattr__(self, "upper_plane_nm", upper_plane)
        if self.normal_roughness_rms_nm is not None:
            object.__setattr__(
                self,
                "normal_roughness_rms_nm",
                float(self.normal_roughness_rms_nm),
            )

    @property
    def normal_axis_name(self) -> str:
        """Return the human-readable slit normal-axis name."""

        return AXIS_NAMES[self.normal_axis_index]

    @property
    def interval_wraps(self) -> bool:
        """Return whether the slit interval crosses the periodic boundary."""

        return self.upper_plane_nm < self.lower_plane_nm

    @property
    def plane_separation_nm(self) -> float:
        """Return the positive periodic separation between mean planes."""

        normal_length = self.box_lengths_nm[self.normal_axis_index]
        return float(np.mod(self.upper_plane_nm - self.lower_plane_nm, normal_length))

    @property
    def lateral_axis_indices(self) -> tuple[int, int]:
        """Return the two Cartesian indices parallel to the slit surfaces."""

        return tuple(index for index in range(3) if index != self.normal_axis_index)  # type: ignore[return-value]

    @property
    def projected_area_per_face_nm2(self) -> float:
        """Return the projected area of one periodic silica-fluid interface."""

        axis_a, axis_b = self.lateral_axis_indices
        return self.box_lengths_nm[axis_a] * self.box_lengths_nm[axis_b]

    @property
    def total_projected_surface_area_nm2(self) -> float:
        """Return the projected area of both confining interfaces."""

        return 2.0 * self.projected_area_per_face_nm2

    @property
    def geometric_slit_volume_nm3(self) -> float:
        """Return the mean-plane geometric slit volume without padding."""

        return self.projected_area_per_face_nm2 * self.plane_separation_nm

    def padded_width_nm(self, padding_nm: float) -> float:
        """Return the signed-padded slit width.

        Parameters
        ----------
        padding_nm : float
            Inward padding on each plane in nanometers. Negative values expand
            the interval.

        Returns
        -------
        float
            Padded interval width in nanometers.

        Raises
        ------
        ValueError
            Raised when padding is non-finite, removes the interval, or expands
            it beyond the full periodic normal dimension.
        """

        if not np.isfinite(padding_nm):
            raise ValueError("Surface-plane padding must be finite.")
        width = self.plane_separation_nm - 2.0 * float(padding_nm)
        normal_length = self.box_lengths_nm[self.normal_axis_index]
        if width <= _GEOMETRY_TOLERANCE_NM or width > normal_length + _GEOMETRY_TOLERANCE_NM:
            raise ValueError(
                "Surface-plane padding must leave a positive interval no wider "
                "than the periodic normal dimension."
            )
        return min(width, normal_length)

    def padded_geometric_volume_nm3(self, padding_nm: float) -> float:
        """Return the geometric slit volume after signed plane padding.

        Parameters
        ----------
        padding_nm : float
            Inward padding on each surface plane in nanometers. Negative values
            expand the interval.

        Returns
        -------
        float
            Signed-padded geometric slit volume in cubic nanometers.
        """

        return self.projected_area_per_face_nm2 * self.padded_width_nm(padding_nm)

    def contains_positions(self, coordinates_nm: ArrayLike, padding_nm: float = 0.0) -> BoolArray:
        """Return which positions lie inside the signed-padded slit interval.

        Parameters
        ----------
        coordinates_nm : array-like
            Cartesian positions whose final dimension has length three.
        padding_nm : float, optional
            Inward padding on each surface plane in nanometers. Negative values
            expand the interval.

        Returns
        -------
        numpy.ndarray
            Boolean array with the coordinate leading shape.
        """

        wrapped = wrap_positions(coordinates_nm, self.box_lengths_nm)
        width = self.padded_width_nm(padding_nm)
        normal_length = self.box_lengths_nm[self.normal_axis_index]
        if np.isclose(width, normal_length, rtol=0.0, atol=_GEOMETRY_TOLERANCE_NM):
            return np.ones(wrapped.shape[:-1], dtype=bool)
        start = float(np.mod(self.lower_plane_nm + padding_nm, normal_length))
        progress = np.mod(wrapped[..., self.normal_axis_index] - start, normal_length)
        return progress <= width + _GEOMETRY_TOLERANCE_NM

    def sample_uniform_positions(
        self,
        sample_count: int,
        random_number_generator: np.random.Generator,
        padding_nm: float = 0.0,
    ) -> FloatArray:
        """Sample positions uniformly inside the signed-padded slit interval.

        Parameters
        ----------
        sample_count : int
            Number of Cartesian positions to generate.
        random_number_generator : numpy.random.Generator
            Random-number generator used for sampling.
        padding_nm : float, optional
            Inward padding on each surface plane in nanometers. Negative values
            expand the interval.

        Returns
        -------
        numpy.ndarray
            Sample positions with shape ``(sample_count, 3)``.

        Raises
        ------
        ValueError
            Raised when ``sample_count`` is not strictly positive.
        """

        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
            raise ValueError("The sample count must be a strictly positive integer.")
        box = np.asarray(self.box_lengths_nm, dtype=np.float64)
        samples = random_number_generator.random((sample_count, 3)) * box
        width = self.padded_width_nm(padding_nm)
        normal_length = box[self.normal_axis_index]
        start = float(np.mod(self.lower_plane_nm + padding_nm, normal_length))
        samples[:, self.normal_axis_index] = np.mod(
            start + random_number_generator.random(sample_count) * width,
            normal_length,
        )
        return samples

    def permute_axes(self, axis_permutation: tuple[int, int, int]) -> PeriodicSlitGeometry:
        """Return geometry transformed by one Cartesian-axis permutation.

        Parameters
        ----------
        axis_permutation : tuple[int, int, int]
            Old Cartesian indices listed in the order used by the new
            coordinate representation.

        Returns
        -------
        PeriodicSlitGeometry
            Geometry expressed in the permuted coordinate frame.

        Raises
        ------
        ValueError
            Raised when the permutation does not contain each axis exactly once.
        """

        if tuple(sorted(axis_permutation)) != (0, 1, 2):
            raise ValueError("Axis permutation must contain 0, 1, and 2 exactly once.")
        new_normal_axis = axis_permutation.index(self.normal_axis_index)
        return PeriodicSlitGeometry(
            box_lengths_nm=tuple(self.box_lengths_nm[index] for index in axis_permutation),
            normal_axis_index=new_normal_axis,
            lower_plane_nm=self.lower_plane_nm,
            upper_plane_nm=self.upper_plane_nm,
            surface_support_count=self.surface_support_count,
            normal_roughness_rms_nm=self.normal_roughness_rms_nm,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the unit-explicit schema-v1 geometry mapping."""

        return {
            "box_lengths_nm": list(self.box_lengths_nm),
            "normal_axis_index": self.normal_axis_index,
            "normal_axis_name": self.normal_axis_name,
            "lower_mean_plane_nm": self.lower_plane_nm,
            "upper_mean_plane_nm": self.upper_plane_nm,
            "interval_wraps": self.interval_wraps,
            "plane_separation_nm": self.plane_separation_nm,
            "projected_area_per_face_nm2": self.projected_area_per_face_nm2,
            "total_projected_surface_area_nm2": self.total_projected_surface_area_nm2,
            "geometric_slit_volume_nm3": self.geometric_slit_volume_nm3,
            "surface_support_count": self.surface_support_count,
            "normal_roughness_rms_nm": self.normal_roughness_rms_nm,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PeriodicSlitGeometry:
        """Build geometry from one schema-v1 mapping.

        Parameters
        ----------
        payload : dict
            Unit-explicit ``slit_geometry`` mapping.

        Returns
        -------
        PeriodicSlitGeometry
            Parsed and validated geometry.

        Raises
        ------
        KeyError
            Raised when a required geometry field is missing.
        ValueError
            Raised when a stored derived field disagrees with the parsed
            geometry.
        """

        geometry = cls(
            box_lengths_nm=tuple(payload["box_lengths_nm"]),
            normal_axis_index=payload["normal_axis_index"],
            lower_plane_nm=payload["lower_mean_plane_nm"],
            upper_plane_nm=payload["upper_mean_plane_nm"],
            surface_support_count=payload.get("surface_support_count"),
            normal_roughness_rms_nm=payload.get("normal_roughness_rms_nm"),
        )
        expected_values = {
            "normal_axis_name": geometry.normal_axis_name,
            "interval_wraps": geometry.interval_wraps,
        }
        for key, expected in expected_values.items():
            if key in payload and payload[key] != expected:
                raise ValueError(f"Stored slit geometry field {key!r} is inconsistent.")
        for key, expected in (
            ("plane_separation_nm", geometry.plane_separation_nm),
            ("projected_area_per_face_nm2", geometry.projected_area_per_face_nm2),
            ("total_projected_surface_area_nm2", geometry.total_projected_surface_area_nm2),
            ("geometric_slit_volume_nm3", geometry.geometric_slit_volume_nm3),
        ):
            if key in payload and not np.isclose(
                float(payload[key]),
                expected,
                rtol=1.0e-10,
                atol=1.0e-12,
            ):
                raise ValueError(f"Stored slit geometry field {key!r} is inconsistent.")
        return geometry

    @classmethod
    def from_surface_positions(
        cls,
        box_lengths_nm: ArrayLike,
        surface_positions_nm: ArrayLike,
        normal_axis_index: int | None = None,
        interior_reference_position_nm: ArrayLike | None = None,
    ) -> PeriodicSlitGeometry:
        """Fit periodic mean planes from two sets of surface positions.

        Parameters
        ----------
        box_lengths_nm : array-like
            Orthorhombic periodic box lengths in nanometers.
        surface_positions_nm : array-like
            Surface-site Cartesian positions with shape ``(n, 3)``.
        normal_axis_index : int or None, optional
            Known slit-normal axis. When omitted, the axis with the largest
            circular gap between wrapped surface coordinates is selected.
        interior_reference_position_nm : array-like or None, optional
            Cartesian point known to lie inside the intended slit interval.
            When supplied, the fitted plane order is oriented so the interval
            contains this point. Without it, the interval follows the largest
            circular surface-position gap.

        Returns
        -------
        PeriodicSlitGeometry
            Fitted mean-plane geometry with support and roughness diagnostics.

        Raises
        ------
        TypeError
            Raised when an explicit axis is not an integer.
        ValueError
            Raised when the box, positions, optional reference point, axis, or
            fitted face clusters are invalid.
        """

        box = _validated_box_lengths(box_lengths_nm)
        positions = np.asarray(surface_positions_nm, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 2:
            raise ValueError("At least two surface positions with shape (n, 3) are required.")
        wrapped = wrap_positions(positions, box)

        if normal_axis_index is not None:
            if isinstance(normal_axis_index, bool) or not isinstance(normal_axis_index, int):
                raise TypeError("The slit normal-axis index must be an integer or None.")
            if normal_axis_index not in (0, 1, 2):
                raise ValueError("The slit normal-axis index must be 0, 1, or 2.")
            candidate_axes = (normal_axis_index,)
        else:
            candidate_axes = (0, 1, 2)

        selected_axis = candidate_axes[0]
        selected_gap_index = 0
        selected_values = np.empty(0, dtype=np.float64)
        largest_gap = -np.inf
        for axis_index in candidate_axes:
            axis_values = np.sort(wrapped[:, axis_index])
            periodic_gaps = np.diff(axis_values)
            wrap_gap = axis_values[0] + box[axis_index] - axis_values[-1]
            gaps = np.concatenate((periodic_gaps, np.array([wrap_gap], dtype=np.float64)))
            gap_index = int(np.argmax(gaps))
            if float(gaps[gap_index]) > largest_gap:
                largest_gap = float(gaps[gap_index])
                selected_axis = axis_index
                selected_gap_index = gap_index
                selected_values = axis_values

        normal_length = float(box[selected_axis])
        lower_seed = float(selected_values[selected_gap_index])
        upper_seed = float(selected_values[(selected_gap_index + 1) % selected_values.size])
        axis_coordinates = wrapped[:, selected_axis]
        distance_to_lower = np.abs(
            _minimum_image_scalar(lower_seed, axis_coordinates, normal_length)
        )
        distance_to_upper = np.abs(
            _minimum_image_scalar(upper_seed, axis_coordinates, normal_length)
        )
        lower_mask = distance_to_lower <= distance_to_upper
        upper_mask = ~lower_mask
        if not np.any(lower_mask) or not np.any(upper_mask):
            raise ValueError("Surface positions could not be separated into two slit faces.")

        lower_offsets = _minimum_image_scalar(
            lower_seed,
            axis_coordinates[lower_mask],
            normal_length,
        )
        upper_offsets = _minimum_image_scalar(
            upper_seed,
            axis_coordinates[upper_mask],
            normal_length,
        )
        lower_plane = float(np.mod(lower_seed + np.mean(lower_offsets), normal_length))
        upper_plane = float(np.mod(upper_seed + np.mean(upper_offsets), normal_length))
        residuals = np.concatenate(
            (
                _minimum_image_scalar(lower_plane, axis_coordinates[lower_mask], normal_length),
                _minimum_image_scalar(upper_plane, axis_coordinates[upper_mask], normal_length),
            )
        )
        roughness = float(np.sqrt(np.mean(residuals * residuals)))
        geometry = cls(
            box_lengths_nm=tuple(float(value) for value in box),
            normal_axis_index=selected_axis,
            lower_plane_nm=lower_plane,
            upper_plane_nm=upper_plane,
            surface_support_count=int(positions.shape[0]),
            normal_roughness_rms_nm=roughness,
        )
        if interior_reference_position_nm is None:
            return geometry

        reference = np.asarray(interior_reference_position_nm, dtype=np.float64)
        if reference.shape != (3,) or not np.all(np.isfinite(reference)):
            raise ValueError(
                "The interior reference position must contain three finite coordinates."
            )
        if bool(geometry.contains_positions(reference)):
            return geometry
        return cls(
            box_lengths_nm=geometry.box_lengths_nm,
            normal_axis_index=geometry.normal_axis_index,
            lower_plane_nm=geometry.upper_plane_nm,
            upper_plane_nm=geometry.lower_plane_nm,
            surface_support_count=geometry.surface_support_count,
            normal_roughness_rms_nm=geometry.normal_roughness_rms_nm,
        )


__all__ = [
    "PeriodicSlitGeometry",
    "minimum_image_displacements",
    "pairwise_minimum_image_distances",
    "wrap_positions",
]
