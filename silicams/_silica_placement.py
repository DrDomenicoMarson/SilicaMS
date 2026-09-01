"""Private numerical clearance and attachment-pose calculations for silica.

This module operates on explicit coordinate/radius arrays and the private
steric reference containers. It does not own chemical state, covalent-radius
lookup, molecule objects, construction policy, or surface mutation.
"""

import numpy as np

import silicams.geometry as geometry

from silicams._numba_kernels import minimum_clearance_against_batch
from silicams._silica_sterics import _StericAtomBatch, _StericGrid


__all__: list[str] = []

_DIMENSIONS = 3
_CLEARANCE_NUMERICAL_TOLERANCE_NM = 1e-12


def _filtered_steric_batch_arrays(reference_batch, ignored_block_atom_ids=None):
    """Return one reference batch as kernel-ready arrays.

    Parameters
    ----------
    reference_batch : _StericAtomBatch
        Reference positions, radii, and scaffold identifiers. Positions and
        radii use nanometers.
    ignored_block_atom_ids : np.ndarray or None, optional
        Scaffold identifiers removed before the clearance kernel runs. Values
        are consumed in iteration order; attached atoms use sentinel ``-1``.

    Returns
    -------
    arrays : tuple[np.ndarray, np.ndarray, np.ndarray]
        Reference positions with shape ``(n, 3)`` and dtype ``float64``, radii
        with shape ``(n,)`` and dtype ``float64``, and identifiers with shape
        ``(n,)`` and dtype ``int64``. Filtering returns independent arrays;
        unfiltered nonempty inputs may share memory after dtype normalization.

    Notes
    -----
    Empty reference positions are reshaped to ``(0, 3)``. The caller owns
    alignment of the three arrays; this helper preserves the existing NumPy
    conversion and indexing failure behavior rather than adding validation.
    """
    positions = np.asarray(reference_batch.positions, dtype=np.float64)
    radii = np.asarray(reference_batch.radii, dtype=np.float64)
    block_atom_ids = np.asarray(reference_batch.block_atom_ids, dtype=np.int64)

    if radii.size == 0:
        return positions.reshape(0, _DIMENSIONS), radii, block_atom_ids

    if ignored_block_atom_ids is not None and ignored_block_atom_ids.size > 0:
        keep_mask = np.ones(block_atom_ids.shape[0], dtype=bool)
        for atom_id in ignored_block_atom_ids:
            keep_mask &= block_atom_ids != atom_id
        positions = positions[keep_mask]
        radii = radii[keep_mask]
        block_atom_ids = block_atom_ids[keep_mask]

    return positions, radii, block_atom_ids


def _positions_clearance(
    positions,
    radii,
    reference,
    ignored_block_atoms,
    steric_clearance_scale,
    box,
):
    """Return the minimum clearance for explicit candidate arrays.

    Parameters
    ----------
    positions : array-like
        Candidate Cartesian positions in nanometers, reshaped to ``(n, 3)``
        and normalized to ``float64``.
    radii : array-like
        Candidate covalent radii in nanometers, reshaped to ``(n,)`` and
        normalized to ``float64``.
    reference : _StericAtomBatch or _StericGrid
        Full reference batch or periodic local-cell reference. Batch mode scans
        all supplied reference atoms. Grid mode scans only each query cell and
        its immediate wrapped neighbors.
    ignored_block_atoms : set[int]
        Scaffold identifiers excluded from the reference. Sentinel ``-1`` is
        treated like any other value during the existing prefilter step.
    steric_clearance_scale : float
        Multiplier applied to each sum of candidate and reference radii.
    box : array-like
        Orthorhombic box lengths in nanometers with shape ``(3,)``. Positive
        lengths are periodic; nonpositive components remain unwrapped in the
        existing numerical kernel.

    Returns
    -------
    clearance : float
        Minimum distance minus scaled summed radii, in nanometers. Return
        ``inf`` for an empty candidate set or when no references remain.

    Notes
    -----
    Inputs and reference containers are not mutated. Grid queries are grouped
    by cell in first-candidate order, with one neighbor batch assembled per cell.
    Clearance reductions and Numba-kernel arguments retain their existing
    ordering. Candidate/radius row alignment is a caller precondition.
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, _DIMENSIONS)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    if radii.size == 0:
        return float("inf")
    box = np.asarray(box, dtype=np.float64)
    ignored_block_atom_ids = (
        np.asarray(sorted(ignored_block_atoms), dtype=np.int64)
        if ignored_block_atoms
        else np.empty(0, dtype=np.int64)
    )
    empty_ignored_block_atom_ids = np.empty(0, dtype=np.int64)

    if isinstance(reference, _StericAtomBatch):
        reference_positions, reference_radii, reference_block_atom_ids = (
            _filtered_steric_batch_arrays(
                reference,
                ignored_block_atom_ids=ignored_block_atom_ids,
            )
        )
        if reference_radii.size == 0:
            return float("inf")
        return float(
            minimum_clearance_against_batch(
                positions,
                radii,
                reference_positions,
                reference_radii,
                reference_block_atom_ids,
                empty_ignored_block_atom_ids,
                box,
                steric_clearance_scale,
            )
        )

    cell_groups = {}
    for atom_index, position in enumerate(positions):
        cell_key = reference._cell_key(position)
        cell_groups.setdefault(cell_key, []).append(atom_index)

    min_clearance = float("inf")
    for cell_key, atom_indices in cell_groups.items():
        batch = reference.neighbor_batch(positions[atom_indices[0]])
        reference_positions, reference_radii, reference_block_atom_ids = (
            _filtered_steric_batch_arrays(
                batch,
                ignored_block_atom_ids=ignored_block_atom_ids,
            )
        )
        if reference_radii.size == 0:
            continue

        atom_indices = np.asarray(atom_indices, dtype=np.int64)
        clearance = float(
            minimum_clearance_against_batch(
                positions[atom_indices],
                radii[atom_indices],
                reference_positions,
                reference_radii,
                reference_block_atom_ids,
                empty_ignored_block_atom_ids,
                box,
                steric_clearance_scale,
            )
        )
        if clearance < min_clearance:
            min_clearance = clearance

    return min_clearance


def _rotation_matrix(axis, angle, is_deg=True):
    """Build the rotation matrix for an arbitrary three-dimensional axis.

    Parameters
    ----------
    axis : array-like
        Three-dimensional rotation-axis vector. The existing geometry helper
        normalizes nonzero axes and leaves a zero axis unchanged.
    angle : float
        Rotation angle in degrees when ``is_deg`` is true, otherwise radians.
    is_deg : bool, optional
        Whether ``angle`` is expressed in degrees.

    Returns
    -------
    matrix : np.ndarray
        Native-float rotation matrix with shape ``(3, 3)``.

    Notes
    -----
    The formula and NumPy operation order intentionally remain separate from
    :class:`silicams.Molecule`'s more general transformation implementation.
    """
    angle = np.deg2rad(angle) if is_deg else angle
    normal = np.asarray(geometry.unit(axis), dtype=float)
    n1, n2, n3 = normal.tolist()
    c = np.cos(angle)
    s = np.sin(angle)

    return np.asarray(
        [
            [n1 * n1 * (1.0 - c) + c, n1 * n2 * (1.0 - c) - n3 * s, n1 * n3 * (1.0 - c) + n2 * s],
            [n2 * n1 * (1.0 - c) + n3 * s, n2 * n2 * (1.0 - c) + c, n2 * n3 * (1.0 - c) - n1 * s],
            [n3 * n1 * (1.0 - c) - n2 * s, n3 * n2 * (1.0 - c) + n1 * s, n3 * n3 * (1.0 - c) + c],
        ],
        dtype=float,
    )


def _rotate_positions_around_axis(positions, origin, axis, angle):
    """Return positions rotated around an axis through one origin.

    Parameters
    ----------
    positions : array-like
        Cartesian coordinates in nanometers with shape ``(n, 3)``.
    origin : array-like
        Cartesian point in nanometers through which the axis passes.
    axis : array-like
        Three-dimensional rotation-axis vector.
    angle : float
        Rotation angle in degrees.

    Returns
    -------
    positions : np.ndarray
        New native-float coordinates with shape ``(n, 3)``. The input arrays
        are not mutated and coordinates remain in their original frame.
    """
    rotation = _rotation_matrix(axis, angle, is_deg=True)
    centered = np.asarray(positions, dtype=float) - np.asarray(origin, dtype=float)
    return centered @ rotation.T + np.asarray(origin, dtype=float)


def _rotation_angles(rotate_step_deg):
    """Return the sampled axis-rotation angles for one pose search.

    Parameters
    ----------
    rotate_step_deg : float
        Strictly positive angular increment in degrees.

    Returns
    -------
    angles : tuple[float, ...]
        Repeated-addition samples in ``[0, 360)``, beginning at zero. Each
        appended angle is rounded to ten decimal places.

    Raises
    ------
    ValueError
        If ``rotate_step_deg`` is not strictly positive. Public configuration
        validation rejects nonfinite values before this private helper runs.
    """
    if rotate_step_deg <= 0:
        raise ValueError("Attachment rotation step must be greater than zero.")

    angles = []
    angle = 0.0
    while angle < 360.0:
        angles.append(round(angle, 10))
        angle += rotate_step_deg

    return tuple(angles if angles else [0.0])


def _best_pose_positions(
    positions,
    steric_atom_ids,
    radii,
    origin,
    axis,
    angles,
    reference,
    ignored_block_atoms,
    steric_clearance_scale,
    box,
):
    """Return the highest-clearance coordinates from an explicit pose scan.

    Parameters
    ----------
    positions : array-like
        Full molecule coordinates in nanometers with shape ``(n, 3)``. Every
        atom is transformed for each sampled pose.
    steric_atom_ids : array-like
        Local integer indices selecting atoms that participate in clearance
        scoring. Unselected atoms still receive the chosen transformation.
    radii : array-like
        Covalent radii in nanometers aligned with ``steric_atom_ids``.
    origin : array-like
        Cartesian point in nanometers through which the rotation axis passes.
    axis : array-like
        Three-dimensional rotation-axis vector.
    angles : iterable[float]
        Candidate rotations in degrees and in evaluation order. Callers use
        ``(0.0,)`` when rotational scanning is disabled.
    reference : _StericAtomBatch or _StericGrid
        Full or local reference used for every candidate evaluation.
    ignored_block_atoms : set[int]
        Scaffold identifiers excluded from reference clearance calculations.
    steric_clearance_scale : float
        Multiplier applied to summed candidate and reference covalent radii.
    box : array-like
        Orthorhombic box lengths in nanometers with shape ``(3,)``.

    Returns
    -------
    positions : np.ndarray or None
        Independent native-float coordinates for the first pose attaining the
        greatest clearance. Return ``None`` only when every evaluated pose has
        clearance below the internal numerical tolerance or no angles are
        supplied.

    Notes
    -----
    Inputs and reference state are not mutated. Exact ties preserve the first
    evaluated pose. Positive, zero, and roundoff-negative clearances within
    ``1e-12`` nanometers are accepted. An empty scored subset has infinite
    clearance and therefore accepts the first pose.
    """
    base_positions = np.asarray(positions, dtype=float).copy()
    steric_atom_ids = np.asarray(steric_atom_ids, dtype=np.int64).reshape(-1)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    best_clearance = float("-inf")
    best_positions = None

    for angle in angles:
        candidate_positions = (
            base_positions
            if angle == 0
            else _rotate_positions_around_axis(
                base_positions,
                origin,
                axis,
                angle,
            )
        )
        clearance = _positions_clearance(
            candidate_positions[steric_atom_ids],
            radii,
            reference,
            ignored_block_atoms,
            steric_clearance_scale,
            box,
        )
        if clearance > best_clearance:
            best_clearance = clearance
            best_positions = candidate_positions.copy()

    # Periodic wrapping and distance reconstruction can make mathematically
    # exact contact infinitesimally negative in binary64 arithmetic.
    if best_clearance < -_CLEARANCE_NUMERICAL_TOLERANCE_NM:
        return None
    return best_positions
