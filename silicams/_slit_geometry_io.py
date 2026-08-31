"""Private geometry resolution and YAML I/O for GRO-based slit workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import yaml

from ._gro_io import BOX_LINE_TOLERANCE_NM, _GroSystem
from .database import _infer_element_from_atom_name
from .slit_geometry import PeriodicSlitGeometry

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]

__all__: list[str] = []


def _find_hydroxylated_surface_silicon_indices(slit_system: _GroSystem) -> IntArray:
    """Return slit Si atoms that belong to hydroxylated surface residues.

    Parameters
    ----------
    slit_system : _GroSystem
        Loaded slit system.

    Returns
    -------
    ndarray
        Integer atom indices of hydroxylated surface silicon atoms, shape
        ``(N_surface,)`` and dtype ``int32``, in file order.

    Raises
    ------
    ValueError
        Raised when no hydroxylated surface silicon atoms can be identified.
    """

    silicon_indices: list[int] = []
    for residue_span in slit_system.residue_spans:
        residue_elements = tuple(
            _infer_element_from_atom_name(atom_name)
            for atom_name in slit_system.atom_names[
                residue_span.start : residue_span.stop
            ]
        )
        has_oxygen = any(element == "O" for element in residue_elements)
        has_hydrogen = any(element == "H" for element in residue_elements)
        if not has_oxygen or not has_hydrogen:
            continue

        for local_atom_index, element in enumerate(residue_elements):
            if element == "Si":
                silicon_indices.append(residue_span.start + local_atom_index)

    if not silicon_indices:
        raise ValueError(
            "Could not identify any hydroxylated surface Si atoms in the slit "
            "structure, so the slit interval cannot be inferred."
        )

    return np.array(silicon_indices, dtype=np.int32)


def _infer_slit_geometry(
    slit_system: _GroSystem,
    slit_coordinates: FloatArray,
    box_lengths: FloatArray,
) -> PeriodicSlitGeometry:
    """Infer periodic mean slit planes and the lower-occupancy framework arc.

    Parameters
    ----------
    slit_system : _GroSystem
        Loaded slit system.
    slit_coordinates : ndarray
        Slit coordinates in nanometers, shape ``(N, 3)``, in the final slit
        reference frame.
    box_lengths : ndarray
        Orthorhombic box lengths in nanometers, shape ``(3,)``.

    Returns
    -------
    PeriodicSlitGeometry
        Inferred orthorhombic slit geometry.

    Notes
    -----
    Two periodic planes define complementary arcs. After fitting the planes,
    this function chooses the arc containing fewer framework atom centers. A
    tie retains the largest surface-position gap selected by the fitter.

    Raises
    ------
    ValueError
        Raised when too few surface Si atoms are available or their positions
        cannot be separated into two faces.
    """

    surface_silicon_indices = _find_hydroxylated_surface_silicon_indices(slit_system)
    if surface_silicon_indices.size < 2:
        raise ValueError(
            "At least two hydroxylated surface Si atoms are required to infer "
            "the slit planes."
        )

    candidate_geometry = PeriodicSlitGeometry.from_surface_positions(
        box_lengths_nm=box_lengths,
        surface_positions_nm=slit_coordinates[surface_silicon_indices],
    )
    complementary_geometry = PeriodicSlitGeometry(
        box_lengths_nm=candidate_geometry.box_lengths_nm,
        normal_axis_index=candidate_geometry.normal_axis_index,
        lower_plane_nm=candidate_geometry.upper_plane_nm,
        upper_plane_nm=candidate_geometry.lower_plane_nm,
        surface_support_count=candidate_geometry.surface_support_count,
        normal_roughness_rms_nm=candidate_geometry.normal_roughness_rms_nm,
    )
    candidate_framework_count = int(
        np.count_nonzero(candidate_geometry.contains_positions(slit_coordinates))
    )
    complementary_framework_count = int(
        np.count_nonzero(complementary_geometry.contains_positions(slit_coordinates))
    )
    if complementary_framework_count < candidate_framework_count:
        return complementary_geometry
    return candidate_geometry


def _load_slit_geometry(path: Path) -> PeriodicSlitGeometry:
    """Load one schema-v1 periodic slit geometry YAML file.

    Parameters
    ----------
    path : Path
        Explicit YAML file path.

    Returns
    -------
    PeriodicSlitGeometry
        Parsed and validated slit geometry.

    Raises
    ------
    ValueError
        Raised when the YAML root, schema version, or geometry mapping is
        invalid.
    OSError
        Raised when the explicitly named file cannot be read.
    """

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    if payload.get("schema_version") != 1:
        raise ValueError(f"{path} must declare schema_version: 1.")
    geometry_payload = payload.get("slit_geometry")
    if not isinstance(geometry_payload, dict):
        raise ValueError(f"{path} must contain a slit_geometry mapping.")
    try:
        return PeriodicSlitGeometry.from_dict(geometry_payload)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid slit geometry in {path}: {error}") from error


def _validate_geometry_box(
    slit_geometry: PeriodicSlitGeometry,
    box_lengths: FloatArray,
) -> None:
    """Validate that geometry and GRO orthorhombic box lengths agree.

    Parameters
    ----------
    slit_geometry : PeriodicSlitGeometry
        Geometry supplied for the GRO structure.
    box_lengths : ndarray
        GRO box lengths in nanometers, shape ``(3,)``.

    Raises
    ------
    ValueError
        Raised when the box lengths differ beyond the GRO precision tolerance.
    """

    if not np.allclose(
        np.asarray(slit_geometry.box_lengths_nm),
        box_lengths,
        rtol=0.0,
        atol=BOX_LINE_TOLERANCE_NM,
    ):
        raise ValueError(
            "The slit geometry box lengths do not match the GRO box lengths: "
            f"geometry={slit_geometry.box_lengths_nm}, GRO={tuple(box_lengths)}."
        )


def _resolve_slit_geometry(
    slit_system: _GroSystem,
    slit_coordinates: FloatArray,
    explicit_geometry: PeriodicSlitGeometry | None,
    geometry_path: Path | None,
    padding_nm: float,
) -> PeriodicSlitGeometry:
    """Resolve explicit, file-backed, or inferred periodic slit geometry.

    Parameters
    ----------
    slit_system : _GroSystem
        Framework system used when inference is required.
    slit_coordinates : ndarray
        Framework coordinates in nanometers, shape ``(N, 3)``, in the same
        frame as ``slit_system.box_lengths``.
    explicit_geometry : PeriodicSlitGeometry or None
        Geometry supplied directly through the Python API.
    geometry_path : Path or None
        Explicit schema-v1 geometry YAML file.
    padding_nm : float
        Signed plane padding to validate against the resolved geometry.

    Returns
    -------
    PeriodicSlitGeometry
        Resolved and validated geometry.

    Raises
    ------
    ValueError
        Raised when geometry inference/loading fails, its box disagrees with
        the GRO box, or the padded slit width is invalid.
    OSError
        Raised when an explicitly supplied geometry file cannot be read.
    """

    if explicit_geometry is not None:
        slit_geometry = explicit_geometry
    elif geometry_path is not None:
        slit_geometry = _load_slit_geometry(geometry_path)
    else:
        slit_geometry = _infer_slit_geometry(
            slit_system=slit_system,
            slit_coordinates=slit_coordinates,
            box_lengths=slit_system.box_lengths,
        )
    _validate_geometry_box(slit_geometry, slit_system.box_lengths)
    slit_geometry.padded_width_nm(padding_nm)
    return slit_geometry


def _write_slit_geometry_metadata(
    path: Path,
    slit_geometry: PeriodicSlitGeometry,
    padding_nm: float,
) -> None:
    """Write schema-v1 output geometry and analysis metadata.

    Parameters
    ----------
    path : Path
        YAML output path.
    slit_geometry : PeriodicSlitGeometry
        Geometry expressed in the output coordinate frame.
    padding_nm : float
        Signed plane padding used by filtering and density analysis.

    Raises
    ------
    ValueError
        Raised when the signed padding gives an invalid slit width.
    OSError
        Raised when the destination cannot be written. Staging and promotion
        remain the calling workflow's responsibility.
    """

    payload = {
        "schema_version": 1,
        "slit_geometry": slit_geometry.to_dict(),
        "slit_analysis": {
            "surface_plane_padding_nm": padding_nm,
            "padded_width_nm": slit_geometry.padded_width_nm(padding_nm),
            "padded_geometric_volume_nm3": slit_geometry.padded_geometric_volume_nm3(
                padding_nm
            ),
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
