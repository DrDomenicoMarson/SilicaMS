"""Private orthorhombic GRO data structures, parsing, and serialization.

Callers supply positions and optional velocities already expressed in the output
frame. Geometry transformations, residue selection, and output transactions
belong to the calling workflow, not to this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]
BoolArray = NDArray[np.bool_]

BOX_LINE_TOLERANCE_NM = 1.0e-6

__all__: list[str] = []


@dataclass(frozen=True)
class _ResidueSpan:
    """Contiguous atom range for one residue in a GRO file.

    Parameters
    ----------
    residue_id : int
        Residue identifier as stored in the file, including any rollover.
    residue_name : str
        Residue name read from the file.
    start : int
        Inclusive, zero-based first atom index.
    stop : int
        Exclusive, zero-based end atom index.
    """

    residue_id: int
    residue_name: str
    start: int
    stop: int


@dataclass(frozen=True)
class _GroSystem:
    """Internal atom arrays and residue indexing for an orthorhombic GRO file.

    Parameters
    ----------
    title : str
        File title without the trailing newline.
    residue_ids : ndarray
        Per-atom residue identifiers, shape ``(N,)`` and dtype ``int32``.
    residue_names : list[str]
        One residue name per atom in file order.
    atom_names : list[str]
        One atom name per atom in file order.
    atom_ids : ndarray
        Per-atom identifiers, shape ``(N,)`` and dtype ``int32``.
    coordinates : ndarray
        Cartesian positions in nanometers, shape ``(N, 3)`` and dtype
        ``float64``.
    velocities : ndarray or None
        Cartesian velocities in nm/ps in the same frame as the coordinates,
        shape ``(N, 3)`` and dtype ``float64``. ``None`` means no atom record
        supplied velocities; missing individual records are otherwise zeros.
    box_lengths : ndarray
        Orthorhombic box lengths in nanometers, shape ``(3,)`` and dtype
        ``float64``.
    residue_spans : tuple[_ResidueSpan, ...]
        Contiguous residue ranges in file order, distinguished by both id
        and name. Non-contiguous occurrences remain separate residues.
    atom_to_residue_index : ndarray
        Zero-based residue-span index per atom, shape ``(N,)`` and dtype
        ``int32``.

    Notes
    -----
    The dataclass prevents field reassignment but does not freeze or copy its
    arrays and lists, or perform additional input validation.
    """

    title: str
    residue_ids: IntArray
    residue_names: list[str]
    atom_names: list[str]
    atom_ids: IntArray
    coordinates: FloatArray
    velocities: FloatArray | None
    box_lengths: FloatArray
    residue_spans: tuple[_ResidueSpan, ...]
    atom_to_residue_index: IntArray

    @property
    def atom_count(self) -> int:
        """Return the number of coordinate rows as a Python integer."""

        return int(self.coordinates.shape[0])


def _load_gro_system(path: Path) -> _GroSystem:
    """Read fixed-width GRO atom records and an orthorhombic box.

    Parameters
    ----------
    path : Path
        Path to the GRO file.

    Returns
    -------
    _GroSystem
        Atom records in file order, positions and box lengths in nanometers,
        and optional velocities in nm/ps. Numeric arrays use ``float64`` for
        vectors and ``int32`` for identifiers and residue indices.

    Raises
    ------
    ValueError
        Raised by invalid numeric fields or an unsupported box line. Nine
        box values are accepted only when all six off-diagonal components
        have magnitude at most ``BOX_LINE_TOLERANCE_NM``.
    OSError
        Raised when the input cannot be opened or read.

    Notes
    -----
    A record with at least 68 characters supplies velocities. If any record
    supplies them, missing per-atom velocities are zero-filled; otherwise
    ``velocities`` is ``None``. Coordinates and velocities are not transformed.
    """

    with path.open("r", encoding="utf-8") as handle:
        title = handle.readline().rstrip("\n")
        atom_count = int(handle.readline().strip())

        residue_ids = np.empty(atom_count, dtype=np.int32)
        atom_ids = np.empty(atom_count, dtype=np.int32)
        coordinates = np.empty((atom_count, 3), dtype=np.float64)
        velocities = np.zeros((atom_count, 3), dtype=np.float64)
        has_velocities = False
        residue_names: list[str] = []
        atom_names: list[str] = []

        for atom_index in range(atom_count):
            line = handle.readline().rstrip("\n")
            residue_ids[atom_index] = int(line[0:5])
            residue_names.append(line[5:10].strip())
            atom_names.append(line[10:15].strip())
            atom_ids[atom_index] = int(line[15:20])
            coordinates[atom_index, 0] = float(line[20:28])
            coordinates[atom_index, 1] = float(line[28:36])
            coordinates[atom_index, 2] = float(line[36:44])

            if len(line) >= 68:
                velocities[atom_index, 0] = float(line[44:52])
                velocities[atom_index, 1] = float(line[52:60])
                velocities[atom_index, 2] = float(line[60:68])
                has_velocities = True

        box_values = [float(value) for value in handle.readline().split()]

    if len(box_values) == 3:
        orthorhombic_box = np.array(box_values, dtype=np.float64)
    elif len(box_values) == 9:
        off_diagonal_values = np.array(box_values[3:9], dtype=np.float64)
        if np.any(np.abs(off_diagonal_values) > BOX_LINE_TOLERANCE_NM):
            raise ValueError(
                f"{path} uses a non-orthorhombic 9-value GRO box with non-negligible "
                f"off-diagonal terms: {off_diagonal_values.tolist()}"
            )
        orthorhombic_box = np.array(box_values[:3], dtype=np.float64)
    else:
        raise ValueError(
            f"{path} uses a non-orthorhombic box with {len(box_values)} values; "
            "this workflow currently supports only orthorhombic GRO boxes."
        )

    residue_spans, atom_to_residue_index = _build_residue_spans(
        residue_ids, residue_names
    )
    return _GroSystem(
        title=title,
        residue_ids=residue_ids,
        residue_names=residue_names,
        atom_names=atom_names,
        atom_ids=atom_ids,
        coordinates=coordinates,
        velocities=velocities if has_velocities else None,
        box_lengths=orthorhombic_box,
        residue_spans=tuple(residue_spans),
        atom_to_residue_index=atom_to_residue_index,
    )


def _build_residue_spans(
    residue_ids: IntArray,
    residue_names: list[str],
) -> tuple[list[_ResidueSpan], IntArray]:
    """Build contiguous residue spans for atoms read from a GRO file.

    Parameters
    ----------
    residue_ids : ndarray
        Per-atom residue identifiers, shape ``(N,)`` and dtype ``int32``.
    residue_names : list[str]
        One residue name per atom in the same order as ``residue_ids``.

    Returns
    -------
    tuple[list[_ResidueSpan], ndarray]
        Residue spans in file order and a zero-based atom-to-residue index
        array of shape ``(N,)`` and dtype ``int32``. A span ends when either
        the id or name changes; repeated non-contiguous id/name pairs are
        not merged. Empty inputs produce an empty list and index array.
    """

    spans: list[_ResidueSpan] = []
    atom_to_residue_index = np.empty(residue_ids.shape[0], dtype=np.int32)

    start = 0
    span_index = -1
    while start < residue_ids.shape[0]:
        residue_id = int(residue_ids[start])
        residue_name = residue_names[start]
        stop = start + 1
        while stop < residue_ids.shape[0]:
            if residue_ids[stop] != residue_id or residue_names[stop] != residue_name:
                break
            stop += 1

        span_index += 1
        spans.append(
            _ResidueSpan(
                residue_id=residue_id,
                residue_name=residue_name,
                start=start,
                stop=stop,
            )
        )
        atom_to_residue_index[start:stop] = span_index
        start = stop

    return spans, atom_to_residue_index


def _format_gro_atom_line(
    residue_id: int,
    residue_name: str,
    atom_name: str,
    atom_id: int,
    coordinate: FloatArray,
    velocity: FloatArray | None,
) -> str:
    """Format one atom record with the existing fixed-width GRO precision.

    Parameters
    ----------
    residue_id : int
        Residue identifier, written modulo 100000.
    residue_name : str
        Residue name, truncated to five characters and left-aligned.
    atom_name : str
        Atom name, truncated to five characters and right-aligned.
    atom_id : int
        Atom identifier, written modulo 100000.
    coordinate : ndarray
        Cartesian position in nanometers, shape ``(3,)``.
    velocity : ndarray or None
        Velocity in nm/ps in the same frame, shape ``(3,)``. ``None`` omits
        velocity columns.

    Returns
    -------
    str
        Newline-terminated record with three decimal places for coordinates
        and four for velocities. Values are formatted, not transformed.
    """

    line = (
        f"{residue_id % 100000:5d}"
        f"{residue_name[:5]:<5}"
        f"{atom_name[:5]:>5}"
        f"{atom_id % 100000:5d}"
        f"{coordinate[0]:8.3f}"
        f"{coordinate[1]:8.3f}"
        f"{coordinate[2]:8.3f}"
    )
    if velocity is not None:
        line += f"{velocity[0]:8.4f}{velocity[1]:8.4f}{velocity[2]:8.4f}"
    return line + "\n"


def _write_system_atoms(
    handle: TextIO,
    system: _GroSystem,
    coordinates: FloatArray,
    velocities: FloatArray | None,
    keep_atom_mask: BoolArray | None,
    write_velocities: bool,
    starting_residue_id: int,
    starting_atom_id: int,
) -> tuple[int, int]:
    """Write complete selected residues in the supplied output frame.

    Parameters
    ----------
    handle : TextIO
        Open text destination for the atom records.
    system : _GroSystem
        Source atom names and contiguous residue spans in input order.
    coordinates : ndarray
        Output-frame positions in nanometers, shape ``(N, 3)``.
    velocities : ndarray or None
        Output-frame velocities in nm/ps, shape ``(N, 3)``. Missing velocities
        are zero-filled when velocity columns are requested.
    keep_atom_mask : ndarray or None
        Boolean selection of shape ``(N,)``. A residue is written only when
        all its atoms are selected; ``None`` selects every residue.
    write_velocities : bool
        Whether to include velocity columns for every written atom.
    starting_residue_id : int
        First sequential output residue identifier.
    starting_atom_id : int
        First sequential output atom identifier.

    Returns
    -------
    tuple[int, int]
        Next unused residue and atom identifiers, before GRO field rollover.

    Raises
    ------
    OSError
        Raised when writing the atom records fails.
    """

    residue_id = starting_residue_id
    atom_id = starting_atom_id

    for residue_span in system.residue_spans:
        if keep_atom_mask is not None and not np.all(
            keep_atom_mask[residue_span.start : residue_span.stop]
        ):
            continue

        for atom_index in range(residue_span.start, residue_span.stop):
            velocity = None
            if write_velocities:
                if velocities is None:
                    velocity = np.zeros(3, dtype=np.float64)
                else:
                    velocity = velocities[atom_index]

            handle.write(
                _format_gro_atom_line(
                    residue_id=residue_id,
                    residue_name=system.residue_names[atom_index],
                    atom_name=system.atom_names[atom_index],
                    atom_id=atom_id,
                    coordinate=coordinates[atom_index],
                    velocity=velocity,
                )
            )
            atom_id += 1

        residue_id += 1

    return residue_id, atom_id


def _write_merged_gro(
    output_path: Path,
    slit_system: _GroSystem,
    slit_coordinates: FloatArray,
    slit_velocities: FloatArray | None,
    guest_system: _GroSystem,
    guest_coordinates: FloatArray,
    guest_velocities: FloatArray | None,
    kept_guest_mask: BoolArray,
    final_box_lengths: FloatArray,
) -> tuple[int, int]:
    """Write slit and selected guest residues in one consistent output frame.

    Parameters
    ----------
    output_path : Path
        GRO destination, normally a staging path supplied by the caller.
    slit_system : _GroSystem
        Source slit atom metadata and contiguous residue spans.
    slit_coordinates : ndarray
        Output-frame slit positions in nanometers, shape ``(N_slit, 3)``.
    slit_velocities : ndarray or None
        Output-frame slit velocities in nm/ps, shape ``(N_slit, 3)``.
    guest_system : _GroSystem
        Source guest atom metadata and contiguous residue spans.
    guest_coordinates : ndarray
        Output-frame guest positions in nanometers, shape ``(N_guest, 3)``.
    guest_velocities : ndarray or None
        Output-frame guest velocities in nm/ps, shape ``(N_guest, 3)``.
    kept_guest_mask : ndarray
        Boolean mask of shape ``(N_guest,)`` selecting complete residues. Each
        residue must have either all or none of its atoms selected.
    final_box_lengths : ndarray
        Orthorhombic output-frame box lengths in nanometers, shape ``(3,)``.

    Returns
    -------
    tuple[int, int]
        Written atom and residue counts. Slit atoms precede retained guests,
        with sequential identifiers starting at one. Velocity columns are
        omitted only when both velocity arrays are absent; a missing array
        is otherwise zero-filled.

    Raises
    ------
    OSError
        Raised when the destination cannot be opened or written. Cleanup and
        transactional promotion are the caller's responsibility.
    """

    final_atom_count = slit_system.atom_count + int(np.count_nonzero(kept_guest_mask))
    write_velocities = slit_velocities is not None or guest_velocities is not None

    final_residue_count = len(slit_system.residue_spans)
    for residue_span in guest_system.residue_spans:
        if np.all(kept_guest_mask[residue_span.start : residue_span.stop]):
            final_residue_count += 1

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Merged slit + ring-check filtered guest\n")
        handle.write(f"{final_atom_count}\n")

        next_residue_id, next_atom_id = _write_system_atoms(
            handle=handle,
            system=slit_system,
            coordinates=slit_coordinates,
            velocities=slit_velocities,
            keep_atom_mask=None,
            write_velocities=write_velocities,
            starting_residue_id=1,
            starting_atom_id=1,
        )
        _write_system_atoms(
            handle=handle,
            system=guest_system,
            coordinates=guest_coordinates,
            velocities=guest_velocities,
            keep_atom_mask=kept_guest_mask,
            write_velocities=write_velocities,
            starting_residue_id=next_residue_id,
            starting_atom_id=next_atom_id,
        )
        handle.write(
            f"{final_box_lengths[0]:10.5f}{final_box_lengths[1]:10.5f}{final_box_lengths[2]:10.5f}\n"
        )

    return final_atom_count, final_residue_count
