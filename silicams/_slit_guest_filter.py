"""Private whole-residue selection and clash geometry for slit guest filling.

These helpers operate on supplied systems and coordinate arrays. Configuration,
geometry resolution, reports, density analysis, and all file I/O belong to the
calling workflow. Selection masks are indexed by contiguous residue spans.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from . import database as db
from .database import _infer_element_from_atom_name
from ._gro_io import _GroSystem
from .slit_geometry import (
    PeriodicSlitGeometry,
    minimum_image_displacements,
    wrap_positions,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

__all__: list[str] = []


@dataclass(frozen=True)
class _SurfacePlaneSelection:
    """Result of applying the optional slit-plane filter to whole residues.

    Parameters
    ----------
    selected_residue_mask : ndarray
        Boolean array of shape ``(R,)`` selecting surviving contiguous residue
        spans.
    removed_residue_mask : ndarray
        Boolean array of shape ``(R,)`` marking previously selected residues
        rejected by the plane filter.
    slit_geometry : PeriodicSlitGeometry
        The supplied geometry, unchanged, in the same frame as the tested
        coordinates.

    Notes
    -----
    The masks are new arrays. Residues not initially selected remain false in
    both masks.
    """

    selected_residue_mask: BoolArray
    removed_residue_mask: BoolArray
    slit_geometry: PeriodicSlitGeometry


@dataclass(frozen=True)
class _BondDefinition:
    """One inferred covalent bond within a residue template.

    Parameters
    ----------
    start_atom_index, stop_atom_index : int
        Local zero-based indices within the residue's ordered atom-name
        signature.
    start_atom_name, stop_atom_name : str
        Atom names at the corresponding local indices.
    """

    start_atom_index: int
    stop_atom_index: int
    start_atom_name: str
    stop_atom_name: str


@dataclass(frozen=True)
class _ResidueBondTemplate:
    """Inferred bond connectivity for one residue-name and atom-name signature.

    Parameters
    ----------
    residue_name : str
        Residue name forming the first part of the template cache key.
    atom_names : tuple[str, ...]
        Ordered atom-name signature forming the rest of the cache key.
    bond_definitions : tuple[_BondDefinition, ...]
        Bonds inferred from the first encountered residue with this signature,
        in atom-pair iteration order.

    Notes
    -----
    Later residues with the same key reuse the first residue's connectivity;
    their coordinates still define their own bond segments.
    """

    residue_name: str
    atom_names: tuple[str, ...]
    bond_definitions: tuple[_BondDefinition, ...]


@dataclass(frozen=True)
class _RingTemplate:
    """Local atom-index template for a six-atom aromatic-ring candidate.

    Parameters
    ----------
    residue_name : str
        Name of the residue containing the candidate ring.
    atom_prefix : str
        Configured prefix matched against atom names.
    local_atom_indices : tuple[int, ...]
        Six matching local atom indices in source atom order.
    ring_atom_names : tuple[str, ...]
        Names at the selected indices, in the same order.
    """

    residue_name: str
    atom_prefix: str
    local_atom_indices: tuple[int, ...]
    ring_atom_names: tuple[str, ...]


@dataclass(frozen=True)
class _RingGeometry:
    """Periodic geometric model of one aromatic ring.

    Parameters
    ----------
    residue_index : int
        Zero-based contiguous residue-span index in the originating system.
    residue_name : str
        Name of the originating residue.
    center : ndarray
        Ring center in nanometers, shape ``(3,)``, shifted into the primary
        periodic box.
    wrapped_center : ndarray
        Periodic search center in nanometers, shape ``(3,)``.
    normal : ndarray
        Unit SVD normal, shape ``(3,)``; its sign is not prescribed.
    basis_u, basis_v : ndarray
        Unit in-plane SVD basis vectors, each shape ``(3,)``.
    polygon_2d : ndarray
        Angle-ordered ring vertices in the local basis, shape ``(6, 2)``, in
        nanometers relative to the center.
    max_radius_nm : float
        Largest center-to-vertex distance in the projected polygon, in
        nanometers.

    Notes
    -----
    Array fields are used read-only by filtering. A frozen dataclass does not
    itself make the arrays immutable.
    """

    residue_index: int
    residue_name: str
    center: FloatArray
    wrapped_center: FloatArray
    normal: FloatArray
    basis_u: FloatArray
    basis_v: FloatArray
    polygon_2d: FloatArray
    max_radius_nm: float


@dataclass(frozen=True)
class _BondSegmentGeometry:
    """Coordinates and periodic search metadata for one inferred bond.

    Parameters
    ----------
    residue_index : int
        Zero-based contiguous residue-span index in the originating system.
    residue_name : str
        Name of the originating residue.
    start_atom_name, stop_atom_name : str
        Names identifying the bond endpoints.
    start_point, stop_point : ndarray
        Endpoint positions in nanometers, each shape ``(3,)``, in one residue
        image.
    midpoint : ndarray
        Arithmetic endpoint midpoint in nanometers, shape ``(3,)``.
    wrapped_midpoint : ndarray
        Midpoint in the primary periodic box, shape ``(3,)``, in nanometers.
    half_length_nm : float
        Half the endpoint distance in nanometers.

    Notes
    -----
    Endpoint arrays can share memory with source coordinates. Consumers must
    treat the geometry as read-only.
    """

    residue_index: int
    residue_name: str
    start_atom_name: str
    stop_atom_name: str
    start_point: FloatArray
    stop_point: FloatArray
    midpoint: FloatArray
    wrapped_midpoint: FloatArray
    half_length_nm: float


@dataclass(frozen=True)
class _ClashSelection:
    """Independent whole-residue removal masks for the three clash checks.

    Parameters
    ----------
    removed_residue_mask : ndarray
        Boolean array of shape ``(R,)`` containing the union of all clash masks.
    removed_by_general_mask : ndarray
        Boolean array of shape ``(R,)`` for the all-atom distance cutoff.
    removed_by_forward_ring_mask : ndarray
        Boolean array of shape ``(R,)`` for guest bonds near or through slit
        rings.
    removed_by_reverse_ring_mask : ndarray
        Boolean array of shape ``(R,)`` for slit bonds near or through guest
        rings.
    removed_by_any_ring_mask : ndarray
        Boolean array of shape ``(R,)`` containing the union of both ring masks.

    Notes
    -----
    Masks index the input guest system's contiguous residue spans, including
    unselected spans. Unselected residues remain false. Mechanisms are evaluated
    independently, so removal masks may overlap.
    """

    removed_residue_mask: BoolArray
    removed_by_general_mask: BoolArray
    removed_by_forward_ring_mask: BoolArray
    removed_by_reverse_ring_mask: BoolArray
    removed_by_any_ring_mask: BoolArray


@dataclass(frozen=True)
class _RingCheckCache:
    """Templates and geometries retained for filling-report counts.

    Parameters
    ----------
    slit_ring_geometries : tuple[_RingGeometry, ...]
        Slit rings in source residue order.
    guest_bond_templates : tuple[_ResidueBondTemplate, ...]
        Selected-guest templates in first-seen cache-key order.
    guest_bond_geometries : tuple[_BondSegmentGeometry, ...]
        Bond segments from all selected guests, in residue and template bond
        order.
    guest_ring_geometries : tuple[_RingGeometry, ...]
        Rings from selected guest residues, in source order.
    slit_bond_templates : tuple[_ResidueBondTemplate, ...]
        Slit templates in first-seen cache-key order.
    slit_bond_geometries : tuple[_BondSegmentGeometry, ...]
        Slit bond segments in residue and template bond order.

    Notes
    -----
    Geometry is retained even when another clash mechanism already rejects a
    residue. Lengths of these tuples provide the existing report counts, not
    counts of candidate bond-ring pairs.
    """

    slit_ring_geometries: tuple[_RingGeometry, ...]
    guest_bond_templates: tuple[_ResidueBondTemplate, ...]
    guest_bond_geometries: tuple[_BondSegmentGeometry, ...]
    guest_ring_geometries: tuple[_RingGeometry, ...]
    slit_bond_templates: tuple[_ResidueBondTemplate, ...]
    slit_bond_geometries: tuple[_BondSegmentGeometry, ...]


def _apply_surface_plane_filter(
    guest_system: _GroSystem,
    translated_guest_coordinates: FloatArray,
    selected_residue_mask: BoolArray,
    slit_geometry: PeriodicSlitGeometry,
    padding_nm: float,
) -> _SurfacePlaneSelection:
    """Reject a selected residue if any atom lies outside the padded slit interval.

    Parameters
    ----------
    guest_system : _GroSystem
        Guest system defining ``R`` contiguous residue spans and ``N`` atom
        records.
    translated_guest_coordinates : ndarray
        Guest positions in the slit frame, shape ``(N, 3)``, in nanometers.
    selected_residue_mask : ndarray
        Boolean input selection of shape ``(R,)`` after center cropping.
    slit_geometry : PeriodicSlitGeometry
        Resolved periodic geometry in the same frame as the positions.
    padding_nm : float
        Signed padding at both planes in nanometers; positive values contract
        the interval.

    Returns
    -------
    _SurfacePlaneSelection
        New selection and rejection masks plus the original geometry; input
        arrays are not mutated.

    Raises
    ------
    ValueError
        If geometry padding is invalid or the position data fail geometry
        validation.
    """

    filtered_residue_mask = selected_residue_mask.copy()
    removed_residue_mask = np.zeros(len(guest_system.residue_spans), dtype=bool)

    for residue_index, residue_span in enumerate(guest_system.residue_spans):
        if not selected_residue_mask[residue_index]:
            continue

        residue_coordinates = translated_guest_coordinates[
            residue_span.start : residue_span.stop
        ]
        if np.all(slit_geometry.contains_positions(residue_coordinates, padding_nm)):
            continue

        filtered_residue_mask[residue_index] = False
        removed_residue_mask[residue_index] = True

    return _SurfacePlaneSelection(
        selected_residue_mask=filtered_residue_mask,
        removed_residue_mask=removed_residue_mask,
        slit_geometry=slit_geometry,
    )


def _unwrap_residue_coordinates(
    residue_coordinates: FloatArray,
    box_lengths: FloatArray,
) -> FloatArray:
    """Unwrap a residue relative to its first atom under orthorhombic PBC.

    Parameters
    ----------
    residue_coordinates : ndarray
        Nonempty residue positions, shape ``(N_residue, 3)``, in nanometers.
    box_lengths : ndarray
        Positive orthorhombic box lengths, shape ``(3,)``, in nanometers.

    Returns
    -------
    ndarray
        New position array in the first atom's image, using the existing
        minimum-image half-box tie convention.

    Notes
    -----
    This is first-atom-relative unwrapping, not connectivity-based
    reconstruction. Input positions are unchanged.
    """

    reference = residue_coordinates[0]
    return reference + minimum_image_displacements(
        reference,
        residue_coordinates,
        box_lengths,
    )


def _guess_bond_definitions(
    atom_names: tuple[str, ...],
    residue_coordinates: FloatArray,
    include_hydrogen_bonds: bool,
) -> tuple[_BondDefinition, ...]:
    """Infer a residue's covalent bonds from element radii and endpoint distances.

    Parameters
    ----------
    atom_names : tuple[str, ...]
        Ordered atom names used to infer supported chemical elements.
    residue_coordinates : ndarray
        Corresponding positions in a consistent residue image, shape
        ``(N_residue, 3)``, in nanometers.
    include_hydrogen_bonds : bool
        Whether bonds with at least one hydrogen endpoint are included. This
        refers to covalent bonds involving hydrogen, not intermolecular hydrogen
        bonding.

    Returns
    -------
    tuple[_BondDefinition, ...]
        Local bond records in ascending start-index and then stop-index order.

    Notes
    -----
    A bond is accepted when ``0.05 < distance <= 1.25 * (r_start + r_stop)`` in
    nanometers. The function does not perform periodic unwrapping.

    Raises
    ------
    ValueError
        If an atom name or its covalent radius cannot be resolved.
    """

    bond_definitions: list[_BondDefinition] = []
    element_symbols = [
        _infer_element_from_atom_name(atom_name) for atom_name in atom_names
    ]
    atom_count = len(atom_names)
    scale_factor = 1.25

    for start_atom_index in range(atom_count):
        for stop_atom_index in range(start_atom_index + 1, atom_count):
            start_element = element_symbols[start_atom_index]
            stop_element = element_symbols[stop_atom_index]
            if not include_hydrogen_bonds and (
                start_element == "H" or stop_element == "H"
            ):
                continue

            distance_nm = float(
                np.linalg.norm(
                    residue_coordinates[stop_atom_index]
                    - residue_coordinates[start_atom_index]
                )
            )
            cutoff_nm = scale_factor * (
                db.get_covalent_radius(start_element)
                + db.get_covalent_radius(stop_element)
            )
            if 0.05 < distance_nm <= cutoff_nm:
                bond_definitions.append(
                    _BondDefinition(
                        start_atom_index=start_atom_index,
                        stop_atom_index=stop_atom_index,
                        start_atom_name=atom_names[start_atom_index],
                        stop_atom_name=atom_names[stop_atom_index],
                    )
                )

    return tuple(bond_definitions)


def _build_ring_template_from_atom_names(
    residue_name: str,
    atom_names: tuple[str, ...],
    ring_atom_prefix: str,
) -> _RingTemplate | None:
    """Select an aromatic-ring template by exactly six prefix-matching atoms.

    Parameters
    ----------
    residue_name : str
        Name to retain in the template.
    atom_names : tuple[str, ...]
        Ordered residue atom names.
    ring_atom_prefix : str
        Prefix matched with ``str.startswith``.

    Returns
    -------
    _RingTemplate or None
        Template in original atom order when exactly six names match; otherwise
        ``None``. No additional ring-connectivity validation is performed.
    """

    local_atom_indices = tuple(
        atom_index
        for atom_index, atom_name in enumerate(atom_names)
        if atom_name.startswith(ring_atom_prefix)
    )
    if len(local_atom_indices) != 6:
        return None

    return _RingTemplate(
        residue_name=residue_name,
        atom_prefix=ring_atom_prefix,
        local_atom_indices=local_atom_indices,
        ring_atom_names=tuple(
            atom_names[atom_index] for atom_index in local_atom_indices
        ),
    )


def _build_ring_geometry(
    residue_index: int,
    residue_name: str,
    residue_coordinates: FloatArray,
    ring_template: _RingTemplate,
    final_box_lengths: FloatArray,
) -> _RingGeometry:
    """Fit an SVD ring plane and an angle-ordered polygon under periodic boundaries.

    Parameters
    ----------
    residue_index : int
        Contiguous residue-span index in the source system.
    residue_name : str
        Name of the source residue.
    residue_coordinates : ndarray
        Full residue positions, shape ``(N_residue, 3)``, in nanometers.
    ring_template : _RingTemplate
        Six local atom indices defining the ring candidate.
    final_box_lengths : ndarray
        Orthorhombic box lengths, shape ``(3,)``, in nanometers.

    Returns
    -------
    _RingGeometry
        Ring geometry shifted into the primary periodic box, without mutating
        source positions.

    Notes
    -----
    The selected atoms are unwrapped relative to the first selected atom before
    fitting. SVD sign choices and angular sorting are retained without
    additional normalization or degeneracy handling.
    """

    ring_coordinates = _unwrap_residue_coordinates(
        residue_coordinates[np.array(ring_template.local_atom_indices, dtype=np.int32)],
        final_box_lengths,
    )
    ring_center = np.mean(ring_coordinates, axis=0)
    wrapped_center = wrap_positions(ring_center[np.newaxis, :], final_box_lengths)[0]
    image_shift = ring_center - wrapped_center
    ring_coordinates = ring_coordinates - image_shift
    ring_center = ring_center - image_shift

    centered_coordinates = ring_coordinates - ring_center
    _, _, right_singular_vectors = np.linalg.svd(
        centered_coordinates, full_matrices=False
    )
    basis_u = right_singular_vectors[0] / np.linalg.norm(right_singular_vectors[0])
    basis_v = right_singular_vectors[1] / np.linalg.norm(right_singular_vectors[1])
    normal = right_singular_vectors[2] / np.linalg.norm(right_singular_vectors[2])

    ring_coordinates_2d = np.column_stack(
        (centered_coordinates @ basis_u, centered_coordinates @ basis_v)
    )
    angles = np.arctan2(ring_coordinates_2d[:, 1], ring_coordinates_2d[:, 0])
    order = np.argsort(angles)
    polygon_2d = ring_coordinates_2d[order]
    max_radius_nm = float(np.max(np.linalg.norm(polygon_2d, axis=1)))

    return _RingGeometry(
        residue_index=residue_index,
        residue_name=residue_name,
        center=ring_center,
        wrapped_center=wrapped_center,
        normal=normal,
        basis_u=basis_u,
        basis_v=basis_v,
        polygon_2d=polygon_2d,
        max_radius_nm=max_radius_nm,
    )


def _build_slit_ring_geometries(
    slit_system: _GroSystem,
    slit_coordinates: FloatArray,
    final_box_lengths: FloatArray,
    ring_atom_prefix: str,
) -> tuple[_RingGeometry, ...]:
    """Construct geometry for each slit residue with exactly six matching ring atoms.

    Parameters
    ----------
    slit_system : _GroSystem
        Slit atom names and contiguous residue spans.
    slit_coordinates : ndarray
        Slit positions, shape ``(N_slit, 3)``, in nanometers.
    final_box_lengths : ndarray
        Orthorhombic slit box lengths, shape ``(3,)``, in nanometers.
    ring_atom_prefix : str
        Atom-name prefix used to identify ring candidates.

    Returns
    -------
    tuple[_RingGeometry, ...]
        Ring geometries in source residue order; inapplicable residues are
        skipped.
    """

    ring_geometries: list[_RingGeometry] = []
    for residue_index, residue_span in enumerate(slit_system.residue_spans):
        atom_names = tuple(
            slit_system.atom_names[residue_span.start : residue_span.stop]
        )
        ring_template = _build_ring_template_from_atom_names(
            residue_name=residue_span.residue_name,
            atom_names=atom_names,
            ring_atom_prefix=ring_atom_prefix,
        )
        if ring_template is None:
            continue

        residue_coordinates = slit_coordinates[residue_span.start : residue_span.stop]
        ring_geometries.append(
            _build_ring_geometry(
                residue_index=residue_index,
                residue_name=residue_span.residue_name,
                residue_coordinates=residue_coordinates,
                ring_template=ring_template,
                final_box_lengths=final_box_lengths,
            )
        )

    return tuple(ring_geometries)


def _build_guest_filter_geometries(
    guest_system: _GroSystem,
    translated_guest_coordinates: FloatArray,
    selected_residue_mask: BoolArray,
    final_box_lengths: FloatArray,
    ring_atom_prefix: str,
    include_hydrogen_bonds: bool,
) -> tuple[
    tuple[_ResidueBondTemplate, ...],
    tuple[_BondSegmentGeometry, ...],
    tuple[_RingGeometry, ...],
]:
    """Build cached bond templates and optional ring geometry for selected guests.

    Parameters
    ----------
    guest_system : _GroSystem
        Guest atom names and ``R`` contiguous residue spans.
    translated_guest_coordinates : ndarray
        Guest positions in a consistent image per residue in the slit frame,
        shape ``(N_guest, 3)``, in nanometers.
    selected_residue_mask : ndarray
        Boolean array of shape ``(R,)`` after cropping and optional plane
        filtering.
    final_box_lengths : ndarray
        Periodic slit box lengths, shape ``(3,)``, in nanometers.
    ring_atom_prefix : str
        Atom-name prefix identifying six-atom ring candidates.
    include_hydrogen_bonds : bool
        Whether inferred bonds to hydrogen participate in forward ring checks.

    Returns
    -------
    tuple[tuple[_ResidueBondTemplate, ...], tuple[_BondSegmentGeometry, ...], tuple[_RingGeometry, ...]]
        Unique bond templates in first-seen key order, actual bond segments for
        selected guests, and their ring geometries.

    Notes
    -----
    Cache keys combine residue name with the ordered atom-name signature.
    Connectivity is inferred from the first selected occurrence and reused.
    Monatomic residues and residues with only explicitly excluded hydrogen bonds
    may contribute no bond segments.

    Raises
    ------
    ValueError
        If a selected multi-atom residue has no inferred bonds even with
        hydrogen included, or an element/radius lookup fails.
    """

    template_cache: dict[tuple[str, tuple[str, ...]], _ResidueBondTemplate] = {}
    bond_geometries: list[_BondSegmentGeometry] = []
    ring_geometries: list[_RingGeometry] = []
    for residue_index, residue_span in enumerate(guest_system.residue_spans):
        if not selected_residue_mask[residue_index]:
            continue

        residue_coordinates = translated_guest_coordinates[
            residue_span.start : residue_span.stop
        ]
        atom_names = tuple(
            guest_system.atom_names[residue_span.start : residue_span.stop]
        )
        template_key = (residue_span.residue_name, atom_names)
        bond_template = template_cache.get(template_key)
        if bond_template is None:
            bond_definitions = _guess_bond_definitions(
                atom_names=atom_names,
                residue_coordinates=residue_coordinates,
                include_hydrogen_bonds=include_hydrogen_bonds,
            )
            if len(atom_names) > 1 and not bond_definitions:
                all_bond_definitions = _guess_bond_definitions(
                    atom_names=atom_names,
                    residue_coordinates=residue_coordinates,
                    include_hydrogen_bonds=True,
                )
                if not all_bond_definitions:
                    raise ValueError(
                        "No covalent bonds were guessed for selected multi-atom "
                        f"guest residue {residue_span.residue_name!r} with atom "
                        f"signature {atom_names!r}."
                    )
            bond_template = _ResidueBondTemplate(
                residue_name=residue_span.residue_name,
                atom_names=atom_names,
                bond_definitions=bond_definitions,
            )
            template_cache[template_key] = bond_template

        for bond_definition in bond_template.bond_definitions:
            start_point = residue_coordinates[bond_definition.start_atom_index]
            stop_point = residue_coordinates[bond_definition.stop_atom_index]
            midpoint = 0.5 * (start_point + stop_point)
            bond_geometries.append(
                _BondSegmentGeometry(
                    residue_index=residue_index,
                    residue_name=residue_span.residue_name,
                    start_atom_name=bond_definition.start_atom_name,
                    stop_atom_name=bond_definition.stop_atom_name,
                    start_point=start_point,
                    stop_point=stop_point,
                    midpoint=midpoint,
                    wrapped_midpoint=wrap_positions(
                        midpoint[np.newaxis, :],
                        final_box_lengths,
                    )[0],
                    half_length_nm=0.5
                    * float(np.linalg.norm(stop_point - start_point)),
                )
            )

        ring_template = _build_ring_template_from_atom_names(
            residue_name=residue_span.residue_name,
            atom_names=atom_names,
            ring_atom_prefix=ring_atom_prefix,
        )
        if ring_template is not None:
            ring_geometries.append(
                _build_ring_geometry(
                    residue_index=residue_index,
                    residue_name=residue_span.residue_name,
                    residue_coordinates=residue_coordinates,
                    ring_template=ring_template,
                    final_box_lengths=final_box_lengths,
                )
            )

    return (
        tuple(template_cache.values()),
        tuple(bond_geometries),
        tuple(ring_geometries),
    )


def _build_slit_bond_geometries(
    slit_system: _GroSystem,
    slit_coordinates: FloatArray,
    final_box_lengths: FloatArray,
) -> tuple[tuple[_ResidueBondTemplate, ...], tuple[_BondSegmentGeometry, ...]]:
    """Build slit bond templates and periodic bond-segment geometry.

    Parameters
    ----------
    slit_system : _GroSystem
        Slit atom names and contiguous residue spans.
    slit_coordinates : ndarray
        Slit positions, shape ``(N_slit, 3)``, in nanometers.
    final_box_lengths : ndarray
        Orthorhombic box lengths, shape ``(3,)``, in nanometers.

    Returns
    -------
    tuple[tuple[_ResidueBondTemplate, ...], tuple[_BondSegmentGeometry, ...]]
        Templates in first-seen key order and bond segments in residue/template
        order.

    Notes
    -----
    Each residue is unwrapped relative to its first atom. Connectivity comes
    from the first residue-name/atom-name signature occurrence. Hydrogen
    endpoints are always included here, independently of the guest forward-check
    option. Empty bond templates are retained.

    Raises
    ------
    ValueError
        If an atom name or covalent radius lookup fails.
    """

    template_cache: dict[tuple[str, tuple[str, ...]], _ResidueBondTemplate] = {}
    bond_geometries: list[_BondSegmentGeometry] = []

    for residue_index, residue_span in enumerate(slit_system.residue_spans):
        atom_names = tuple(
            slit_system.atom_names[residue_span.start : residue_span.stop]
        )
        template_key = (residue_span.residue_name, atom_names)
        residue_coordinates = _unwrap_residue_coordinates(
            slit_coordinates[residue_span.start : residue_span.stop],
            final_box_lengths,
        )

        residue_bond_template = template_cache.get(template_key)
        if residue_bond_template is None:
            residue_bond_template = _ResidueBondTemplate(
                residue_name=residue_span.residue_name,
                atom_names=atom_names,
                bond_definitions=_guess_bond_definitions(
                    atom_names=atom_names,
                    residue_coordinates=residue_coordinates,
                    include_hydrogen_bonds=True,
                ),
            )
            template_cache[template_key] = residue_bond_template

        for bond_definition in residue_bond_template.bond_definitions:
            start_point = residue_coordinates[bond_definition.start_atom_index]
            stop_point = residue_coordinates[bond_definition.stop_atom_index]
            midpoint = 0.5 * (start_point + stop_point)
            wrapped_midpoint = wrap_positions(
                midpoint[np.newaxis, :], final_box_lengths
            )[0]
            bond_geometries.append(
                _BondSegmentGeometry(
                    residue_index=residue_index,
                    residue_name=residue_span.residue_name,
                    start_atom_name=bond_definition.start_atom_name,
                    stop_atom_name=bond_definition.stop_atom_name,
                    start_point=start_point,
                    stop_point=stop_point,
                    midpoint=midpoint,
                    wrapped_midpoint=wrapped_midpoint,
                    half_length_nm=0.5
                    * float(np.linalg.norm(stop_point - start_point)),
                )
            )

    return tuple(template_cache.values()), tuple(bond_geometries)


def _point_inside_polygon_with_padding(
    point_2d: FloatArray,
    polygon_2d: FloatArray,
    padding_nm: float,
) -> bool:
    """Test a projected point against a polygon and optional edge-distance padding.

    Parameters
    ----------
    point_2d : ndarray
        Point in local planar coordinates, shape ``(2,)``, in nanometers.
    polygon_2d : ndarray
        Ordered polygon vertices, shape ``(N_vertices, 2)``, in nanometers.
    padding_nm : float
        Non-negative extra edge-distance allowance in nanometers.

    Returns
    -------
    bool
        Whether ray casting accepts the point, or positive padding includes its
        minimum distance to a polygon edge.

    Notes
    -----
    With zero padding, only the existing ray-casting boundary convention
    applies. Positive padding uses an inclusive distance comparison. Degenerate
    edge projection and ray-casting numerical tolerances are preserved.
    """

    inside = False
    point_x = float(point_2d[0])
    point_y = float(point_2d[1])
    vertex_count = polygon_2d.shape[0]

    for vertex_index in range(vertex_count):
        next_index = (vertex_index + 1) % vertex_count
        x1, y1 = polygon_2d[vertex_index]
        x2, y2 = polygon_2d[next_index]
        intersects = ((y1 > point_y) != (y2 > point_y)) and (
            point_x < (x2 - x1) * (point_y - y1) / (y2 - y1 + 1.0e-12) + x1
        )
        if intersects:
            inside = not inside

    if inside or padding_nm <= 0.0:
        return inside

    minimum_distance_nm = np.inf
    for vertex_index in range(vertex_count):
        next_index = (vertex_index + 1) % vertex_count
        segment_start = polygon_2d[vertex_index]
        segment_stop = polygon_2d[next_index]
        segment = segment_stop - segment_start
        denominator = float(np.dot(segment, segment))
        if denominator <= 1.0e-12:
            projection = segment_start
        else:
            fraction = float(
                np.clip(
                    np.dot(point_2d - segment_start, segment) / denominator,
                    0.0,
                    1.0,
                )
            )
            projection = segment_start + fraction * segment
        minimum_distance_nm = min(
            minimum_distance_nm, float(np.linalg.norm(point_2d - projection))
        )

    return minimum_distance_nm <= padding_nm


def _shift_bond_near_reference(
    start_point: FloatArray,
    stop_point: FloatArray,
    reference_point: FloatArray,
    box_lengths: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Move both bond endpoints together to the periodic image nearest a reference.

    Parameters
    ----------
    start_point, stop_point : ndarray
        Endpoint positions, each shape ``(3,)``, in nanometers.
    reference_point : ndarray
        Reference position, shape ``(3,)``, in nanometers.
    box_lengths : ndarray
        Orthorhombic box lengths, shape ``(3,)``, in nanometers.

    Returns
    -------
    tuple[ndarray, ndarray]
        New endpoints shifted by the same whole-box vector, preserving the bond
        displacement.

    Notes
    -----
    The image is chosen from the arithmetic midpoint using NumPy rounding,
    including its half-box tie behavior. Inputs are not modified.
    """

    midpoint = 0.5 * (start_point + stop_point)
    image_shift = box_lengths * np.round((midpoint - reference_point) / box_lengths)
    return start_point - image_shift, stop_point - image_shift


def _bond_crosses_ring(
    bond_start: FloatArray,
    bond_stop: FloatArray,
    ring_geometry: _RingGeometry,
    plane_tolerance_nm: float,
    polygon_padding_nm: float,
) -> bool:
    """Apply the existing near-plane and padded-polygon bond/ring crossing test.

    Parameters
    ----------
    bond_start, bond_stop : ndarray
        Bond endpoints, each shape ``(3,)``, in nanometers, already shifted near
        the ring center.
    ring_geometry : _RingGeometry
        Ring center, plane, basis, and projected polygon in the same frame.
    plane_tolerance_nm : float
        Inclusive maximum distance to the ring plane in nanometers.
    polygon_padding_nm : float
        Additional in-plane polygon padding in nanometers.

    Returns
    -------
    bool
        Whether the selected closest point satisfies the plane, radial, and
        polygon tests.

    Notes
    -----
    For nonparallel bonds, the plane-intersection parameter is clamped to the
    segment. For parallel bonds, the nearer endpoint is used, with the start
    endpoint winning a tie. A sign-changing plane intersection is not required;
    this is a tolerance-based construction heuristic.
    """

    bond_direction = bond_stop - bond_start
    denominator = float(np.dot(ring_geometry.normal, bond_direction))
    if abs(denominator) > 1.0e-12:
        fraction = float(
            np.clip(
                np.dot(ring_geometry.normal, ring_geometry.center - bond_start)
                / denominator,
                0.0,
                1.0,
            )
        )
    else:
        start_distance = abs(
            float(np.dot(bond_start - ring_geometry.center, ring_geometry.normal))
        )
        stop_distance = abs(
            float(np.dot(bond_stop - ring_geometry.center, ring_geometry.normal))
        )
        fraction = 0.0 if start_distance <= stop_distance else 1.0

    closest_point = bond_start + fraction * bond_direction
    plane_distance_nm = abs(
        float(np.dot(closest_point - ring_geometry.center, ring_geometry.normal))
    )
    if plane_distance_nm > plane_tolerance_nm:
        return False

    projected_point = closest_point - ring_geometry.center
    projected_point_2d = np.array(
        [
            np.dot(projected_point, ring_geometry.basis_u),
            np.dot(projected_point, ring_geometry.basis_v),
        ],
        dtype=np.float64,
    )
    if np.linalg.norm(projected_point_2d) > (
        ring_geometry.max_radius_nm + polygon_padding_nm
    ):
        return False

    return _point_inside_polygon_with_padding(
        point_2d=projected_point_2d,
        polygon_2d=ring_geometry.polygon_2d,
        padding_nm=polygon_padding_nm,
    )


def _center_crop_guest_residues(
    guest_system: _GroSystem,
    final_box_lengths: FloatArray,
) -> tuple[FloatArray, BoolArray, FloatArray]:
    """Center-crop whole guest residues from a reservoir into the slit cell.

    Parameters
    ----------
    guest_system : _GroSystem
        Guest reservoir with ``N`` atoms, ``R`` contiguous residue spans, and an
        orthorhombic source box.
    final_box_lengths : ndarray
        Target slit box lengths, shape ``(3,)``, in nanometers; the workflow
        validates that they do not exceed the reservoir box.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        Translated positions of shape ``(N, 3)`` in nanometers, a boolean
        residue selection of shape ``(R,)``, and the crop-origin vector of shape
        ``(3,)`` in nanometers. Unselected atoms have zero coordinates in the
        returned array.

    Notes
    -----
    Each residue is unwrapped relative to its first atom in the reservoir box.
    Every atom must satisfy the lower inclusive and upper exclusive crop bounds
    after the existing ``1e-6 nm`` tolerance is applied. Source coordinates and
    velocities are unchanged.
    """

    crop_window_start = 0.5 * (guest_system.box_lengths - final_box_lengths)
    crop_window_stop = crop_window_start + final_box_lengths
    translated_coordinates = np.zeros_like(guest_system.coordinates)
    selected_residues = np.zeros(len(guest_system.residue_spans), dtype=bool)
    tolerance = 1.0e-6

    for residue_index, residue_span in enumerate(guest_system.residue_spans):
        residue_coordinates = _unwrap_residue_coordinates(
            guest_system.coordinates[residue_span.start : residue_span.stop],
            guest_system.box_lengths,
        )
        is_inside_crop = bool(
            np.all(residue_coordinates >= (crop_window_start - tolerance))
            and np.all(residue_coordinates < (crop_window_stop + tolerance))
        )
        if is_inside_crop:
            translated_coordinates[residue_span.start : residue_span.stop] = (
                residue_coordinates - crop_window_start
            )
            selected_residues[residue_index] = True

    return translated_coordinates, selected_residues, crop_window_start


def _identify_clashing_guest_residues(
    guest_system: _GroSystem,
    slit_system: _GroSystem,
    translated_guest_coordinates: FloatArray,
    selected_residue_mask: BoolArray,
    slit_coordinates: FloatArray,
    final_box_lengths: FloatArray,
    general_cutoff_nm: float,
    ring_atom_prefix: str,
    ring_plane_tolerance_nm: float,
    ring_polygon_padding_nm: float,
    include_hydrogen_bonds_in_ring_check: bool,
) -> tuple[_ClashSelection, _RingCheckCache]:
    """Independently classify all selected guests by general and ring clashes.

    Parameters
    ----------
    guest_system : _GroSystem
        Guest reservoir atom identities and contiguous residue indexing.
    slit_system : _GroSystem
        Slit framework atom identities and contiguous residue indexing.
    translated_guest_coordinates : ndarray
        Guest positions after cropping, shape ``(N_guest, 3)``, in nanometers.
    selected_residue_mask : ndarray
        Boolean guest residue selection, shape ``(R_guest,)``, after cropping
        and optional plane filtering.
    slit_coordinates : ndarray
        Slit positions, shape ``(N_slit, 3)``, in nanometers, in the same
        input-axis slit frame as the guest positions.
    final_box_lengths : ndarray
        Orthorhombic slit box lengths, shape ``(3,)``, in nanometers.
    general_cutoff_nm : float
        Inclusive nearest-atom guest-to-slit clash cutoff in nanometers.
    ring_atom_prefix : str
        Atom-name prefix identifying exactly six ring atoms per candidate
        residue.
    ring_plane_tolerance_nm : float
        Inclusive bond-to-ring-plane tolerance in nanometers.
    ring_polygon_padding_nm : float
        Additional in-plane polygon padding in nanometers.
    include_hydrogen_bonds_in_ring_check : bool
        Whether inferred guest bonds with hydrogen endpoints participate in
        forward ring checks; slit bonds always include them.

    Returns
    -------
    tuple[_ClashSelection, _RingCheckCache]
        Independent per-residue masks and the templates/geometries used for
        report counts.

    Notes
    -----
    All selected residue names are checked, not only the density target. Ring
    checks still run for general-clash residues so overlap accounting is
    preserved. Coordinates and masks are not mutated; cache endpoints may retain
    read-only-use views of input arrays.

    Raises
    ------
    ValueError
        If element/radius lookup, guest bond inference, or periodic-coordinate
        validation fails. Even an empty guest selection still constructs slit
        geometry, preserving validation order.
    """

    residue_count = len(guest_system.residue_spans)
    removed_by_general = np.zeros(residue_count, dtype=bool)
    removed_by_forward_ring = np.zeros(residue_count, dtype=bool)
    removed_by_reverse_ring = np.zeros(residue_count, dtype=bool)

    slit_wrapped = wrap_positions(slit_coordinates, final_box_lengths)
    slit_tree = cKDTree(slit_wrapped, boxsize=final_box_lengths)

    candidate_atoms = np.zeros(guest_system.atom_count, dtype=bool)
    for residue_index, residue_span in enumerate(guest_system.residue_spans):
        if selected_residue_mask[residue_index]:
            candidate_atoms[residue_span.start : residue_span.stop] = True

    if np.any(candidate_atoms):
        candidate_atom_indices = np.flatnonzero(candidate_atoms)
        wrapped_candidate_coordinates = wrap_positions(
            translated_guest_coordinates[candidate_atom_indices],
            final_box_lengths,
        )
        nearest_distances, _ = slit_tree.query(
            wrapped_candidate_coordinates, k=1, workers=-1
        )
        clashing_candidate_atoms = np.zeros(guest_system.atom_count, dtype=bool)
        clashing_candidate_atoms[candidate_atom_indices] = (
            nearest_distances <= general_cutoff_nm
        )
        if np.any(clashing_candidate_atoms):
            residue_indices = guest_system.atom_to_residue_index[
                clashing_candidate_atoms
            ]
            removed_by_general[np.unique(residue_indices)] = True

    guest_bond_templates, guest_bond_geometries, guest_ring_geometries = (
        _build_guest_filter_geometries(
            guest_system=guest_system,
            translated_guest_coordinates=translated_guest_coordinates,
            selected_residue_mask=selected_residue_mask,
            final_box_lengths=final_box_lengths,
            ring_atom_prefix=ring_atom_prefix,
            include_hydrogen_bonds=include_hydrogen_bonds_in_ring_check,
        )
    )
    slit_ring_geometries = _build_slit_ring_geometries(
        slit_system=slit_system,
        slit_coordinates=slit_coordinates,
        final_box_lengths=final_box_lengths,
        ring_atom_prefix=ring_atom_prefix,
    )
    slit_bond_templates, slit_bond_geometries = _build_slit_bond_geometries(
        slit_system=slit_system,
        slit_coordinates=slit_coordinates,
        final_box_lengths=final_box_lengths,
    )
    ring_check_cache = _RingCheckCache(
        slit_ring_geometries=slit_ring_geometries,
        guest_bond_templates=guest_bond_templates,
        guest_bond_geometries=guest_bond_geometries,
        guest_ring_geometries=guest_ring_geometries,
        slit_bond_templates=slit_bond_templates,
        slit_bond_geometries=slit_bond_geometries,
    )

    if slit_ring_geometries and guest_bond_geometries:
        slit_ring_center_tree = cKDTree(
            np.array(
                [ring_geometry.wrapped_center for ring_geometry in slit_ring_geometries]
            ),
            boxsize=final_box_lengths,
        )
        maximum_slit_ring_radius_nm = max(
            ring_geometry.max_radius_nm for ring_geometry in slit_ring_geometries
        )
        for bond_geometry in guest_bond_geometries:
            candidate_ring_indices = slit_ring_center_tree.query_ball_point(
                bond_geometry.wrapped_midpoint,
                r=(
                    bond_geometry.half_length_nm
                    + maximum_slit_ring_radius_nm
                    + ring_plane_tolerance_nm
                    + ring_polygon_padding_nm
                ),
            )
            for ring_index in candidate_ring_indices:
                ring_geometry = slit_ring_geometries[ring_index]
                shifted_start_point, shifted_stop_point = _shift_bond_near_reference(
                    start_point=bond_geometry.start_point,
                    stop_point=bond_geometry.stop_point,
                    reference_point=ring_geometry.center,
                    box_lengths=final_box_lengths,
                )
                if _bond_crosses_ring(
                    bond_start=shifted_start_point,
                    bond_stop=shifted_stop_point,
                    ring_geometry=ring_geometry,
                    plane_tolerance_nm=ring_plane_tolerance_nm,
                    polygon_padding_nm=ring_polygon_padding_nm,
                ):
                    removed_by_forward_ring[bond_geometry.residue_index] = True
                    break

    if guest_ring_geometries and slit_bond_geometries:
        slit_bond_midpoint_tree = cKDTree(
            np.array(
                [
                    bond_geometry.wrapped_midpoint
                    for bond_geometry in slit_bond_geometries
                ]
            ),
            boxsize=final_box_lengths,
        )
        maximum_slit_bond_half_length_nm = max(
            bond_geometry.half_length_nm for bond_geometry in slit_bond_geometries
        )

        for guest_ring_geometry in guest_ring_geometries:
            candidate_bond_indices = slit_bond_midpoint_tree.query_ball_point(
                guest_ring_geometry.wrapped_center,
                r=(
                    guest_ring_geometry.max_radius_nm
                    + maximum_slit_bond_half_length_nm
                    + ring_plane_tolerance_nm
                    + ring_polygon_padding_nm
                ),
            )
            for bond_index in candidate_bond_indices:
                bond_geometry = slit_bond_geometries[bond_index]
                shifted_start_point, shifted_stop_point = _shift_bond_near_reference(
                    start_point=bond_geometry.start_point,
                    stop_point=bond_geometry.stop_point,
                    reference_point=guest_ring_geometry.center,
                    box_lengths=final_box_lengths,
                )
                if _bond_crosses_ring(
                    bond_start=shifted_start_point,
                    bond_stop=shifted_stop_point,
                    ring_geometry=guest_ring_geometry,
                    plane_tolerance_nm=ring_plane_tolerance_nm,
                    polygon_padding_nm=ring_polygon_padding_nm,
                ):
                    removed_by_reverse_ring[guest_ring_geometry.residue_index] = True
                    break

    removed_by_any_ring = removed_by_forward_ring | removed_by_reverse_ring
    clash_selection = _ClashSelection(
        removed_residue_mask=removed_by_general | removed_by_any_ring,
        removed_by_general_mask=removed_by_general,
        removed_by_forward_ring_mask=removed_by_forward_ring,
        removed_by_reverse_ring_mask=removed_by_reverse_ring,
        removed_by_any_ring_mask=removed_by_any_ring,
    )
    return clash_selection, ring_check_cache


def _build_kept_guest_atom_mask(
    guest_system: _GroSystem,
    selected_residue_mask: BoolArray,
    removed_residue_mask: BoolArray,
) -> BoolArray:
    """Expand surviving whole-residue selections into an atom mask.

    Parameters
    ----------
    guest_system : _GroSystem
        Guest system defining ``N`` atoms and ``R`` contiguous residue spans.
    selected_residue_mask : ndarray
        Boolean array of shape ``(R,)`` after cropping and optional plane
        filtering.
    removed_residue_mask : ndarray
        Boolean array of shape ``(R,)`` marking residues rejected by clashes.

    Returns
    -------
    ndarray
        New boolean array of shape ``(N,)`` selecting every atom of selected,
        non-rejected residues. Input masks remain unchanged.
    """

    keep_mask = np.zeros(guest_system.atom_count, dtype=bool)
    for residue_index, residue_span in enumerate(guest_system.residue_spans):
        if (
            selected_residue_mask[residue_index]
            and not removed_residue_mask[residue_index]
        ):
            keep_mask[residue_span.start : residue_span.stop] = True
    return keep_mask
