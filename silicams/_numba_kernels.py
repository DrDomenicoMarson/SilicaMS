################################################################################
# Internal Numba Kernels                                                       #
#                                                                              #
"""Numba-accelerated numeric kernels used by performance-critical internals."""
################################################################################


from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True)
def minimum_image_component(delta, box_length):
    """Return the wrapped one-dimensional minimum-image displacement.

    Parameters
    ----------
    delta : float
        Signed displacement component.
    box_length : float
        Periodic box length for the same dimension.

    Returns
    -------
    delta : float
        Wrapped minimum-image displacement component.
    """
    if box_length <= 0.0:
        return delta

    half_box = box_length / 2.0
    wrapped = (delta + half_box) % box_length - half_box
    if abs(wrapped + half_box) <= 1e-12 and delta > 0.0:
        return half_box
    return wrapped


@nb.njit(cache=True)
def minimum_clearance_against_batch(
    positions,
    radii,
    ref_positions,
    ref_radii,
    ref_block_atom_ids,
    ignored_block_atom_ids,
    box,
    steric_clearance_scale,
):
    """Return the minimum steric clearance against one reference batch.

    Parameters
    ----------
    positions : np.ndarray
        Candidate atom positions with shape ``(n, 3)``.
    radii : np.ndarray
        Candidate covalent radii with shape ``(n,)``.
    ref_positions : np.ndarray
        Reference atom positions with shape ``(m, 3)``.
    ref_radii : np.ndarray
        Reference covalent radii with shape ``(m,)``.
    ref_block_atom_ids : np.ndarray
        Reference scaffold atom ids with shape ``(m,)``. Attached ligand atoms
        are encoded as ``-1``.
    ignored_block_atom_ids : np.ndarray
        Sorted scaffold atom ids that should be skipped.
    box : np.ndarray
        Periodic box lengths with shape ``(3,)``.
    steric_clearance_scale : float
        Multiplicative factor applied to the steric cutoff.

    Returns
    -------
    clearance : float
        Minimum distance minus steric cutoff across all checked pairs.
    """
    min_clearance = np.inf

    for atom_index in range(radii.shape[0]):
        position = positions[atom_index]
        radius = radii[atom_index]

        for reference_index in range(ref_radii.shape[0]):
            block_atom_id = ref_block_atom_ids[reference_index]
            if block_atom_id >= 0:
                skip = False
                for ignored_atom_id in ignored_block_atom_ids:
                    if block_atom_id == ignored_atom_id:
                        skip = True
                        break
                if skip:
                    continue

            dx = minimum_image_component(
                ref_positions[reference_index, 0] - position[0],
                box[0],
            )
            dy = minimum_image_component(
                ref_positions[reference_index, 1] - position[1],
                box[1],
            )
            dz = minimum_image_component(
                ref_positions[reference_index, 2] - position[2],
                box[2],
            )
            distance = (dx * dx + dy * dy + dz * dz) ** 0.5
            clearance = (
                distance
                - steric_clearance_scale * (radius + ref_radii[reference_index])
            )
            if clearance < min_clearance:
                min_clearance = clearance

    return min_clearance
