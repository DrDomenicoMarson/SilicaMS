"""Validated full-slit ITP connectivity for coordinate-based filling."""

from pathlib import Path

from ._gro_io import _GroSystem
from .topology import (
    _CHEMICAL_BOND_FUNCTIONS,
    _parse_atoms,
    _parse_bonds,
    _read_section_rows,
    _validate_interaction_atom_references,
)


def load_slit_bonds(path: Path, system: _GroSystem) -> tuple[tuple[int, int], ...]:
    """Read chemical bonds from one flat ITP matching the entire slit GRO.

    Parameters
    ----------
    path : Path
        Self-contained ITP with one molecule type in full-slit atom order.
    system : _GroSystem
        Slit coordinates and identities used to check topology correspondence.

    Returns
    -------
    tuple[tuple[int, int], ...]
        Unique sorted zero-based chemical edges, including residue junctions.
        Non-chemical distance potentials (bond types 6, 9, 10) are excluded.

    Raises
    ------
    ValueError
        If the input is not a flat single-molecule ITP, uses unsupported
        sections or bond functions, has dangling/self bonds, or its atom order,
        names, residue names, or residue boundaries disagree with the GRO.
    OSError
        If the topology cannot be read.

    Notes
    -----
    Repeated atom names in different residues are valid here. This reader
    extracts connectivity; it does not validate force-field parameters.
    """
    document = _read_section_rows(path)
    rows = document.rows_by_section
    if document.header_count_by_section.get("moleculetype") != 1 or len(rows["moleculetype"]) != 1:
        raise ValueError("Slit topology must contain exactly one molecule type.")
    unsupported = set(rows) - {"atomtypes", "moleculetype", "atoms", "bonds", "pairs", "angles", "dihedrals"}
    if unsupported:
        raise ValueError(f"Unsupported slit topology sections: {sorted(unsupported)}.")
    atoms = _parse_atoms(path, rows.get("atoms", []))
    if tuple(atom.index for atom in atoms) != tuple(range(1, system.atom_count + 1)):
        raise ValueError("Slit topology atom indices/count must match the complete GRO order.")
    previous_residue = None
    previous_gro_residue = None
    for index, atom in enumerate(atoms):
        residue = (atom.residue_number, atom.residue_name)
        gro_residue = int(system.atom_to_residue_index[index])
        if (
            atom.atom_name != system.atom_names[index]
            or atom.residue_name != system.residue_names[index]
            or (residue != previous_residue) != (gro_residue != previous_gro_residue)
        ):
            raise ValueError(f"Slit topology atom/residue identity disagrees with GRO at atom {index + 1}.")
        previous_residue, previous_gro_residue = residue, gro_residue
    bonds = _parse_bonds(path, rows.get("bonds", []))
    _validate_interaction_atom_references(
        source_path=str(path), moleculetype_name=rows["moleculetype"][0][0],
        section_name="bonds", reference_rows=tuple((b.atom_a, b.atom_b) for b in bonds),
        defined_atom_indices={atom.index for atom in atoms},
    )
    edges = set()
    for bond in bonds:
        if bond.atom_a == bond.atom_b or bond.parameters.function not in range(1, 11):
            raise ValueError("Slit topology contains a self bond or unsupported bond function.")
        if bond.parameters.function in _CHEMICAL_BOND_FUNCTIONS:
            edges.add(tuple(sorted((bond.atom_a - 1, bond.atom_b - 1))))
    return tuple(sorted(edges))
