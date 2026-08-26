################################################################################
# Molecule Class                                                               #
#                                                                              #
"""Tools for creating, transforming, and inspecting molecular structures."""
################################################################################


from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import silicams.database as db
import silicams.geometry as geometry

from silicams.atom import Atom


@dataclass
class _AtomTable:
    """Array-backed atom storage used as the authoritative molecule state.

    Parameters
    ----------
    positions : np.ndarray
        Cartesian atom positions with shape ``(n, 3)``.
    atom_types : np.ndarray
        Atom-type labels with shape ``(n,)``.
    names : np.ndarray
        Atom-name labels with shape ``(n,)``.
    residues : np.ndarray
        Residue indices with shape ``(n,)``.
    source_ids : np.ndarray
        Optional source atom identifiers with shape ``(n,)``.
    """

    positions: np.ndarray
    atom_types: np.ndarray
    names: np.ndarray
    residues: np.ndarray
    source_ids: np.ndarray

    def __post_init__(self):
        """Normalize the storage arrays and validate their shared length."""
        positions = np.asarray(self.positions, dtype=float)
        if positions.size == 0:
            positions = np.empty((0, 3), dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Atom positions must be stored as an (n, 3) array.")

        atom_types = np.asarray(self.atom_types, dtype=object)
        names = np.asarray(self.names, dtype=object)
        residues = np.asarray(self.residues, dtype=int)
        source_ids = np.asarray(self.source_ids, dtype=object)

        num_atoms = positions.shape[0]
        for field_name, values in (
            ("atom_types", atom_types),
            ("names", names),
            ("residues", residues),
            ("source_ids", source_ids),
        ):
            if values.shape != (num_atoms,):
                raise ValueError(
                    f"{field_name} must contain exactly one entry per atom."
                )

        self.positions = positions
        self.atom_types = atom_types
        self.names = names
        self.residues = residues
        self.source_ids = source_ids

    @classmethod
    def empty(cls):
        """Return an empty atom table."""
        return cls(
            positions=np.empty((0, 3), dtype=float),
            atom_types=np.empty(0, dtype=object),
            names=np.empty(0, dtype=object),
            residues=np.empty(0, dtype=int),
            source_ids=np.empty(0, dtype=object),
        )

    @classmethod
    def from_atoms(cls, atoms):
        """Build an atom table from compatibility :class:`Atom` objects.

        Parameters
        ----------
        atoms : list[Atom]
            Atom objects whose data should be copied into array storage.

        Returns
        -------
        atom_table : _AtomTable
            Array-backed atom storage populated from ``atoms``.
        """
        if not atoms:
            return cls.empty()

        return cls(
            positions=np.asarray([atom.get_pos() for atom in atoms], dtype=float),
            atom_types=np.asarray([atom.get_atom_type() for atom in atoms], dtype=object),
            names=np.asarray([atom.get_name() for atom in atoms], dtype=object),
            residues=np.asarray([atom.get_residue() for atom in atoms], dtype=int),
            source_ids=np.asarray([atom.get_source_id() for atom in atoms], dtype=object),
        )

    @classmethod
    def from_records(
        cls,
        positions,
        atom_types,
        names=None,
        residues=None,
        source_ids=None,
    ):
        """Build an atom table from column-like record arrays.

        Parameters
        ----------
        positions : list
            Cartesian atom positions.
        atom_types : list
            Atom-type labels.
        names : list or None, optional
            Atom names. Empty strings are used when omitted.
        residues : list or None, optional
            Residue indices. Zeros are used when omitted.
        source_ids : list or None, optional
            Optional source atom identifiers. ``None`` is used when omitted.

        Returns
        -------
        atom_table : _AtomTable
            Array-backed atom storage populated from the record arrays.
        """
        num_atoms = len(atom_types)
        return cls(
            positions=np.asarray(positions, dtype=float),
            atom_types=np.asarray(atom_types, dtype=object),
            names=np.asarray([""] * num_atoms if names is None else names, dtype=object),
            residues=np.asarray([0] * num_atoms if residues is None else residues, dtype=int),
            source_ids=np.asarray([None] * num_atoms if source_ids is None else source_ids, dtype=object),
        )

    @classmethod
    def concatenate(cls, tables):
        """Concatenate several atom tables into one.

        Parameters
        ----------
        tables : list[_AtomTable]
            Atom tables to concatenate in order.

        Returns
        -------
        atom_table : _AtomTable
            Concatenated atom storage.
        """
        tables = [table for table in tables if table is not None]
        if not tables:
            return cls.empty()

        return cls(
            positions=np.concatenate([table.positions for table in tables], axis=0),
            atom_types=np.concatenate([table.atom_types for table in tables]),
            names=np.concatenate([table.names for table in tables]),
            residues=np.concatenate([table.residues for table in tables]),
            source_ids=np.concatenate([table.source_ids for table in tables]),
        )

    def copy(self):
        """Return a deep copy of the atom table.

        Returns
        -------
        atom_table : _AtomTable
            Copied atom storage.
        """
        return _AtomTable(
            positions=self.positions.copy(),
            atom_types=self.atom_types.copy(),
            names=self.names.copy(),
            residues=self.residues.copy(),
            source_ids=self.source_ids.copy(),
        )

    def subset(self, indices):
        """Return a copied subset of the stored atoms.

        Parameters
        ----------
        indices : list or np.ndarray
            Atom indices that should be copied into the subset.

        Returns
        -------
        atom_table : _AtomTable
            Atom table containing only the selected atoms.
        """
        idx = np.asarray(indices, dtype=int)
        return _AtomTable(
            positions=self.positions[idx].copy(),
            atom_types=self.atom_types[idx].copy(),
            names=self.names[idx].copy(),
            residues=self.residues[idx].copy(),
            source_ids=self.source_ids[idx].copy(),
        )

    def delete(self, indices):
        """Return a copy with the selected atoms removed.

        Parameters
        ----------
        indices : list or np.ndarray
            Atom indices that should be deleted.

        Returns
        -------
        atom_table : _AtomTable
            Atom table without the selected atoms.
        """
        idx = np.asarray(indices, dtype=int)
        if idx.size == 0:
            return self.copy()

        mask = np.ones(self.positions.shape[0], dtype=bool)
        mask[idx] = False
        return _AtomTable(
            positions=self.positions[mask].copy(),
            atom_types=self.atom_types[mask].copy(),
            names=self.names[mask].copy(),
            residues=self.residues[mask].copy(),
            source_ids=self.source_ids[mask].copy(),
        )

    def swap(self, atom_a, atom_b):
        """Swap two atom rows in place.

        Parameters
        ----------
        atom_a : int
            Index of the first atom.
        atom_b : int
            Index of the second atom.
        """
        for values in (
            self.positions,
            self.atom_types,
            self.names,
            self.residues,
            self.source_ids,
        ):
            values[[atom_a, atom_b]] = values[[atom_b, atom_a]]

    def append_atom(self, position, atom_type, name="", residue=0, source_id=None):
        """Append one atom record in place.

        Parameters
        ----------
        position : list or np.ndarray
            Cartesian atom position.
        atom_type : str
            Atom-type label.
        name : str, optional
            Atom name.
        residue : int, optional
            Residue index.
        source_id : int or None, optional
            Optional source atom identifier.
        """
        position_row = np.asarray(position, dtype=float).reshape(1, 3)
        self.positions = (
            position_row
            if self.positions.size == 0
            else np.concatenate([self.positions, position_row], axis=0)
        )
        self.atom_types = np.append(self.atom_types, atom_type)
        self.names = np.append(self.names, name)
        self.residues = np.append(self.residues, int(residue))
        self.source_ids = np.append(self.source_ids, source_id)

    def to_atoms(self):
        """Materialize compatibility :class:`Atom` objects from the arrays.

        Returns
        -------
        atom_list : list[Atom]
            Snapshot atom list detached from the stored array state.
        """
        atom_list = []
        for atom_id in range(self.positions.shape[0]):
            source_id = self.source_ids[atom_id]
            atom_list.append(
                Atom(
                    self.positions[atom_id].tolist(),
                    self.atom_types[atom_id],
                    self.names[atom_id],
                    int(self.residues[atom_id]),
                    source_id=None if source_id is None else source_id,
                )
            )
        return atom_list

    def get_num(self):
        """Return the number of stored atoms.

        Returns
        -------
        num : int
            Number of atoms stored in the table.
        """
        return self.positions.shape[0]


class Molecule:
    """Mutable molecule with array-backed atomic coordinates and metadata.

    The class keeps Cartesian coordinates and per-atom metadata in NumPy
    arrays for numeric work. Compatibility :class:`silicams.atom.Atom` objects
    are materialized only on demand for inspection or serialization helpers.

    Parameters
    ----------
    name : str, optional
        Molecule name.
    short : str, optional
        Short residue-style identifier used in exported structure files.
    inp : None, str, list, optional
        Optional initial content. Use ``None`` for an empty molecule, a file
        path string to read a structure, a list of :class:`Molecule` objects to
        concatenate them, or a list of :class:`silicams.atom.Atom` objects to use
        as a compatibility input.

    Examples
    --------
    Following example generates a benzene molecule without hydrogen atoms

    .. code-block:: python

        import silicams as sms

        mol = sms.Molecule("benzene", "BEN")
        mol.add("C", [0,0,0])
        mol.add("C", 0, r=0.1375, theta= 60)
        mol.add("C", 1, r=0.1375, theta=120)
        mol.add("C", 2, r=0.1375, theta=180)
        mol.add("C", 3, r=0.1375, theta=240)
        mol.add("C", 4, r=0.1375, theta=300)

    """

    def __init__(self, name="molecule", short="MOL", inp=None):
        """Initialize the molecule and optional input content.

        Parameters
        ----------
        name : str, optional
            Molecule name.
        short : str, optional
            Short residue-style identifier used in exported structure files.
        inp : None, str, list, optional
            Optional initial content. Use ``None`` for an empty molecule, a
            file path string to read a structure, a list of molecules to
            concatenate, or a list of compatibility :class:`Atom` objects to
            copy into the internal array storage.
        """
        self._dim = 3

        self._name = name
        self._short = short

        self._box = []
        self._charge = 0
        self._masses = None
        self._mass = None
        self._bonds = set()
        self._atoms = _AtomTable.empty()

        if inp is None:
            return

        if isinstance(inp, str):
            self._atoms, self._bonds = self._read(inp, inp.split(".")[-1].upper())
            return

        if isinstance(inp, list):
            if not inp:
                return
            if isinstance(inp[0], Atom):
                self._atoms = _AtomTable.from_atoms(inp)
            else:
                self._atoms, self._bonds = self._concat(inp)

    ##################
    # Representation #
    ##################
    def __repr__(self):
        """Create a pandas table of the molecule data.

        Returns
        -------
        repr : str
            Pandas-formatted string representation of the molecule.
        """
        columns = ["Residue", "Name", "Type", "x", "y", "z"]
        if self.get_num() == 0:
            return pd.DataFrame([], columns=columns).to_string()

        data = np.column_stack(
            [
                self._atoms.residues,
                self._atoms.names,
                self._atoms.atom_types,
                self._atoms.positions,
            ]
        ).tolist()
        return pd.DataFrame(data, columns=columns).to_string()

    ##############
    # Management #
    ##############
    def _invalidate_mass_cache(self):
        """Invalidate cached mass-derived properties after atom changes."""
        self._masses = None
        self._mass = None

    def _materialize_atom(self, atom_id):
        """Materialize one compatibility :class:`Atom` snapshot.

        Parameters
        ----------
        atom_id : int
            Atom index to materialize.

        Returns
        -------
        atom : Atom
            Detached compatibility atom snapshot.
        """
        source_id = self._atoms.source_ids[atom_id]
        return Atom(
            self._atoms.positions[atom_id].tolist(),
            self._atoms.atom_types[atom_id],
            self._atoms.names[atom_id],
            int(self._atoms.residues[atom_id]),
            source_id=None if source_id is None else source_id,
        )

    def _normalize_atom_ids(self, atoms):
        """Normalize atom-id input to an integer NumPy array.

        Parameters
        ----------
        atoms : int or list
            Atom index or collection of atom indices.

        Returns
        -------
        atom_ids : np.ndarray
            Normalized integer atom-index array.
        """
        if isinstance(atoms, int):
            return np.asarray([atoms], dtype=int)
        return np.asarray(list(atoms), dtype=int)

    def _rotation_matrix(self, axis, angle, is_deg=True):
        """Build the rotation matrix used for whole-molecule transforms.

        Parameters
        ----------
        axis : int or str or list
            Rotation axis accepted by :func:`silicams.geometry.rotate`.
        angle : float
            Rotation angle.
        is_deg : bool, optional
            True when ``angle`` is given in degrees.

        Returns
        -------
        matrix : np.ndarray
            Rotation matrix with shape ``(3, 3)``.

        Raises
        ------
        ValueError
            Raised when the rotation axis input is invalid.
        """
        angle = angle * math.pi / 180 if is_deg else angle

        if isinstance(axis, np.ndarray):
            axis = axis.tolist()
        if isinstance(axis, tuple):
            axis = list(axis)

        if isinstance(axis, list):
            if len(axis) == self._dim and not any(
                isinstance(entry, (list, tuple, np.ndarray)) for entry in axis
            ):
                normal = axis
            elif len(axis) == 2:
                axis_a = axis[0].tolist() if isinstance(axis[0], np.ndarray) else list(axis[0])
                axis_b = axis[1].tolist() if isinstance(axis[1], np.ndarray) else list(axis[1])
                try:
                    normal = geometry.vector(axis_a, axis_b)
                except ValueError as error:
                    raise ValueError("Rotate: Wrong vector dimensions.") from error
            else:
                raise ValueError("Rotate: Wrong vector dimensions.")
        else:
            try:
                normal = geometry.main_axis(axis, dim=self._dim)
            except ValueError as error:
                raise ValueError(f"Rotate: {error}") from error

        normal = np.asarray(geometry.unit(normal), dtype=float)
        n1, n2, n3 = normal.tolist()
        c = math.cos(angle)
        s = math.sin(angle)

        return np.asarray(
            [
                [n1 * n1 * (1.0 - c) + c, n1 * n2 * (1.0 - c) - n3 * s, n1 * n3 * (1.0 - c) + n2 * s],
                [n2 * n1 * (1.0 - c) + n3 * s, n2 * n2 * (1.0 - c) + c, n2 * n3 * (1.0 - c) - n1 * s],
                [n3 * n1 * (1.0 - c) - n2 * s, n3 * n2 * (1.0 - c) + n1 * s, n3 * n3 * (1.0 - c) + c],
            ],
            dtype=float,
        )

    def _apply_rotation(self, atom_ids, axis, angle, is_deg=True):
        """Rotate the selected atoms in place.

        Parameters
        ----------
        atom_ids : list or np.ndarray
            Atom indices to rotate.
        axis : int or str or list
            Rotation axis accepted by :func:`silicams.geometry.rotate`.
        angle : float
            Rotation angle.
        is_deg : bool, optional
            True when ``angle`` is given in degrees.
        """
        idx = self._normalize_atom_ids(atom_ids)
        if idx.size == 0:
            return

        rotation = self._rotation_matrix(axis, angle, is_deg=is_deg)
        self._atoms.positions[idx] = self._atoms.positions[idx] @ rotation.T

    def _read(self, file_path, file_type):
        """Read a molecule from a supported structure file.

        Parameters
        ----------
        file_path : str
            Path to the requested file.
        file_type : str
            File extension name in upper case.

        Returns
        -------
        atom_table : _AtomTable
            Array-backed atom storage built from the file.
        bonds : set[tuple[int, int]]
            Explicit bond pairs read from the input file.

        Raises
        ------
        ValueError
            Raised when the requested file type is unsupported.
        """
        if file_type not in ["GRO", "PDB", "MOL2"]:
            raise ValueError("Unsupported filetype.")

        positions = []
        atom_types = []
        names = []
        residues = []
        bonds = set()

        if file_type == "GRO":
            with open(file_path, "r") as file_in:
                for line_idx, line in enumerate(file_in):
                    line_val = line.split()
                    if line_idx > 0 and len(line_val) > 3:
                        residues.append(int(line[0:5]) - 1)
                        positions.append([float(line_val[i]) for i in range(3, 6)])
                        names.append(line_val[1])
                        atom_types.append("".join(char for char in line_val[1] if not char.isdigit()))

            return _AtomTable.from_records(positions, atom_types, names, residues), bonds

        if file_type == "PDB":
            serial_to_index = {}
            with open(file_path, "r") as file_in:
                for line in file_in:
                    record = line[0:6].strip()
                    if record in ["ATOM", "HETATM"]:
                        serial = int(line[6:11])
                        serial_to_index[serial] = len(atom_types)
                        residues.append(int(line[22:26]) - 1)
                        positions.append(
                            [
                                float(line[30:38]) / 10,
                                float(line[38:46]) / 10,
                                float(line[46:54]) / 10,
                            ]
                        )
                        name = line[12:16].strip()
                        atom_token = line[76:78].strip() if len(line) >= 78 else ""
                        names.append(name)
                        atom_types.append(db.get_pdb_element(name, atom_token))
                    elif record == "CONECT":
                        serials = [
                            int(line[start : start + 5])
                            for start in range(6, len(line.rstrip("\n")), 5)
                            if line[start : start + 5].strip()
                        ]
                        if len(serials) < 2 or serials[0] not in serial_to_index:
                            continue
                        atom_a = serial_to_index[serials[0]]
                        for serial_b in serials[1:]:
                            if serial_b in serial_to_index:
                                bonds.add(
                                    self._normalize_bond(atom_a, serial_to_index[serial_b])
                                )

            return _AtomTable.from_records(positions, atom_types, names, residues), bonds

        section = None
        with open(file_path, "r") as file_in:
            for line in file_in:
                line_val = line.split()
                if not line_val:
                    continue
                if line.startswith("@<TRIPOS>"):
                    section = line.strip()
                    continue

                if section == "@<TRIPOS>ATOM" and len(line_val) > 5:
                    residues.append(int(line_val[6]) - 1 if len(line_val) > 6 else 0)
                    positions.append([float(line_val[i]) / 10 for i in range(2, 5)])
                    names.append(line_val[1])
                    try:
                        atom_types.append(db.get_element(line_val[5]))
                    except ValueError:
                        atom_types.append("".join(char for char in line_val[1] if not char.isdigit()))
                elif section == "@<TRIPOS>BOND" and len(line_val) >= 4:
                    bonds.add(
                        self._normalize_bond(int(line_val[1]) - 1, int(line_val[2]) - 1)
                    )

        return _AtomTable.from_records(positions, atom_types, names, residues), bonds

    def _concat(self, mol_list):
        """Concatenate a molecule list into one molecule object.

        Parameters
        ----------
        mol_list : list[Molecule]
            Molecule objects to concatenate.

        Returns
        -------
        atom_table : _AtomTable
            Concatenated array-backed atom storage.
        bonds : set[tuple[int, int]]
            Bond pairs with concatenated atom indexing.
        """
        atom_tables = []
        bonds = set()
        atom_offset = 0
        for mol in mol_list:
            atom_tables.append(mol._atoms.copy())
            bonds.update(
                self._normalize_bond(atom_a + atom_offset, atom_b + atom_offset)
                for atom_a, atom_b in mol.get_bonds()
            )
            atom_offset += mol.get_num()

        return _AtomTable.concatenate(atom_tables), bonds

    def _normalize_bond(self, atom_a, atom_b):
        """Normalize one bond definition to a sorted atom pair.

        Parameters
        ----------
        atom_a : int
            First atom index.
        atom_b : int
            Second atom index.

        Returns
        -------
        bond : tuple[int, int]
            Sorted bond tuple.

        Raises
        ------
        ValueError
            Raised when the two atom indices are identical.
        """
        if atom_a == atom_b:
            raise ValueError("Molecule: Cannot create a bond from one atom to itself.")
        return (atom_a, atom_b) if atom_a < atom_b else (atom_b, atom_a)

    def _temp(self, atoms):
        """Create a detached temporary molecule of selected atom ids.

        Parameters
        ----------
        atoms : list
            Atom ids to copy into the temporary molecule.

        Returns
        -------
        mol : Molecule
            Temporary molecule containing compatibility atom snapshots.
        """
        atom_ids = self._normalize_atom_ids(atoms)
        return Molecule(inp=[self._materialize_atom(atom_id) for atom_id in atom_ids])

    def append(self, mol):
        """Append all atoms from another molecule.

        Parameters
        ----------
        mol : Molecule
            Molecule whose atoms will be appended in their current order.
        """
        atom_offset = self.get_num()
        self._atoms = _AtomTable.concatenate([self._atoms, mol._atoms.copy()])
        for atom_a, atom_b in mol.get_bonds():
            self._bonds.add(self._normalize_bond(atom_a + atom_offset, atom_b + atom_offset))
        if mol.get_num() > 0:
            self._invalidate_mass_cache()

    def positions_view(self):
        """Return the live Cartesian coordinate array.

        Returns
        -------
        positions : np.ndarray
            Live ``(n, 3)`` coordinate array backing the molecule.
        """
        return self._atoms.positions

    def atom_types_view(self):
        """Return the live atom-type array.

        Returns
        -------
        atom_types : np.ndarray
            Live ``(n,)`` atom-type array backing the molecule.
        """
        return self._atoms.atom_types

    def atom_names_view(self):
        """Return the live atom-name array.

        Returns
        -------
        names : np.ndarray
            Live ``(n,)`` atom-name array backing the molecule.
        """
        return self._atoms.names

    def residues_view(self):
        """Return the live residue-index array.

        Returns
        -------
        residues : np.ndarray
            Live ``(n,)`` residue-index array backing the molecule.
        """
        return self._atoms.residues

    def source_ids_view(self):
        """Return the live optional source-id array.

        Returns
        -------
        source_ids : np.ndarray
            Live ``(n,)`` source-id array backing the molecule.
        """
        return self._atoms.source_ids

    def column_pos(self):
        """Return Cartesian coordinates grouped by dimension.

        Returns
        -------
        column : list
            Position columns ordered as ``[x_values, y_values, z_values]``.
        """
        if self.get_num() == 0:
            return [[] for _ in range(self._dim)]
        return self._atoms.positions.T.tolist()

    ############
    # Geometry #
    ############
    def _vector(self, pos_a, pos_b):
        """Calculate the vector between two positions or atom ids.

        Parameters
        ----------
        pos_a : int or list or tuple or np.ndarray
            First atom id or Cartesian position :math:`\\boldsymbol{a}`.
        pos_b : int or list or tuple or np.ndarray
            Second atom id or Cartesian position :math:`\\boldsymbol{b}`.

        Returns
        -------
        vector : list
            Bond vector from ``pos_a`` to ``pos_b``.

        Raises
        ------
        ValueError
            Raised when the inputs are not atom ids or three-dimensional
            position vectors.
        """
        if isinstance(pos_a, int) and isinstance(pos_b, int):
            vec_a = self._atoms.positions[pos_a]
            vec_b = self._atoms.positions[pos_b]
        elif isinstance(pos_a, (list, tuple, np.ndarray)) and isinstance(
            pos_b, (list, tuple, np.ndarray)
        ):
            vec_a = pos_a
            vec_b = pos_b
        else:
            raise ValueError("Vector: Wrong input...")

        if len(vec_a) != self._dim or len(vec_b) != self._dim:
            raise ValueError("Vector: Wrong dimensions...")

        return geometry.vector(list(vec_a), list(vec_b))

    def _box_size(self):
        """Calculate the box size of the current molecule.

        Returns
        -------
        box : list
            Box lengths derived from the current coordinate maxima.
        """
        if self.get_num() == 0:
            return [0.001 for _ in range(self._dim)]

        maxima = self._atoms.positions.max(axis=0)
        return [value if value > 0 else 0.001 for value in maxima.tolist()]

    ##############
    # Properties #
    ##############
    def pos(self, atom):
        """Return the Cartesian position of one atom.

        Parameters
        ----------
        atom : int
            Atom index.

        Returns
        -------
        pos : list
            Three-dimensional position vector of the selected atom.
        """
        return self._atoms.positions[atom].tolist()

    def bond(self, inp_a, inp_b):
        """Return the vector between two atoms or positions.

        Parameters
        ----------
        inp_a : int or list
            First atom index or Cartesian position.
        inp_b : int or list
            Second atom index or Cartesian position.

        Returns
        -------
        bond : list
            Vector from ``inp_a`` to ``inp_b``.
        """
        return self._vector(inp_a, inp_b)

    def centroid(self):
        """Return the geometric centroid of all atom positions.

        Returns
        -------
        centroid : list
            Arithmetic mean of all atomic coordinates.
        """
        if self.get_num() == 0:
            return [0.0, 0.0, 0.0]
        return self._atoms.positions.mean(axis=0).tolist()

    def com(self):
        """Return the mass-weighted center of mass.

        Returns
        -------
        com : list
            Mass-weighted center of mass.
        """
        if self.get_num() == 0:
            return [0.0, 0.0, 0.0]

        masses = np.asarray(self.get_masses(), dtype=float)
        if masses.size < self.get_num():
            raise IndexError("Molecule: Not enough masses defined for all atoms.")

        weights = masses[: self.get_num()]
        center = (self._atoms.positions * weights[:, None]).sum(axis=0) / weights.sum()
        return center.tolist()

    #################
    # Basic Editing #
    #################
    def translate(self, vec):
        """Translate every atom by the same vector.

        Parameters
        ----------
        vec : list
            Translation vector.
        """
        if self.get_num() == 0:
            return
        self._atoms.positions += np.asarray(vec, dtype=float)

    def rotate(self, axis, angle, is_deg=True):
        """Rotate all atom positions around a common axis.

        Parameters
        ----------
        axis : int or str or list
            Rotation axis accepted by :func:`silicams.geometry.rotate`.
        angle : float
            Rotation angle.
        is_deg : bool, optional
            True if ``angle`` is given in degrees.
        """
        self._apply_rotation(np.arange(self.get_num(), dtype=int), axis, angle, is_deg=is_deg)

    def move(self, atom, pos):
        """Translate the molecule so one atom reaches a target position.

        Parameters
        ----------
        atom : int
            Anchor atom index.
        pos : list
            Target position of the anchor atom.
        """
        self.translate(self._vector(self.pos(atom), pos))

    def zero(self, pos=None):
        """Translate the molecule so its minimum coordinate matches ``pos``.

        Parameters
        ----------
        pos : list, optional
            New lower coordinate corner. Defaults to the origin.

        Returns
        -------
        vec : list
            Translation vector applied to the molecule.
        """
        pos = [0, 0, 0] if pos is None else pos
        if self.get_num() == 0:
            return list(pos)

        vec = np.asarray(pos, dtype=float) - self._atoms.positions.min(axis=0)
        self._box = []
        self.translate(vec.tolist())
        return vec.tolist()

    def put(self, atom, pos):
        """Set the Cartesian position of one atom directly.

        Parameters
        ----------
        atom : int
            Atom index to update.
        pos : list
            New position vector.
        """
        self._atoms.positions[atom] = np.asarray(pos, dtype=float)

    ####################
    # Advanced Editing #
    ####################
    def part_move(self, bond, atoms, length, vec=None):
        """Translate part of a molecule to adjust a bond length.

        Parameters
        ----------
        bond : list
            Two atom ids defining the bond to adjust.
        atoms : int or list
            Atom id or atom ids that are translated as one fragment.
        length : float
            Requested final bond length.
        vec : list, optional
            Explicit translation direction vector.
        """
        atom_ids = self._normalize_atom_ids(atoms)
        length = abs(length - geometry.length(self.bond(*bond)))

        direction = self._vector(bond[0], bond[1]) if vec is None else vec
        displacement = np.asarray(geometry.unit(direction), dtype=float) * length
        self._atoms.positions[atom_ids] += displacement

    def part_rotate(self, bond, atoms, angle, zero):
        """Rotate part of the molecule around a bond axis.

        Parameters
        ----------
        bond : list
            Two atom ids defining the rotation axis.
        atoms : int or list
            Atom id or atom ids to rotate.
        angle : float
            Rotation angle in degrees.
        zero : int
            Atom id that is translated to the origin before the rotation.
        """
        self.move(zero, [0, 0, 0])
        atom_ids = self._normalize_atom_ids(atoms)
        axis = [self.pos(bond[0]), self.pos(bond[1])]
        self._apply_rotation(atom_ids, axis, angle, is_deg=True)

    def part_angle(self, bond_a, bond_b, atoms, angle, zero):
        """Rotate a fragment to change the angle between two bonds.

        Parameters
        ----------
        bond_a : list
            First bond definition, either two atom ids or an explicit vector.
        bond_b : list
            Second bond definition, either two atom ids or an explicit vector.
        atoms : int or list
            Atom id or atom ids to rotate.
        angle : float
            Rotation angle in degrees.
        zero : int
            Atom id translated to the origin before the rotation.

        Raises
        ------
        ValueError
            Raised when the provided bond definitions do not have compatible
            dimensions.
        """
        if len(bond_a) != len(bond_b):
            raise ValueError("Part_Angle : Wrong bond dimensions...")
        if len(bond_a) not in [2, self._dim]:
            raise ValueError("Part_Angle : Wrong bond input...")

        self.move(zero, [0, 0, 0])
        atom_ids = self._normalize_atom_ids(atoms)

        if len(bond_a) == 2:
            axis = geometry.cross_product(self._vector(*bond_a), self._vector(*bond_b))
        else:
            axis = geometry.cross_product(bond_a, bond_b)

        self._apply_rotation(atom_ids, axis, angle, is_deg=True)

    #########
    # Atoms #
    #########
    def add(
        self,
        atom_type,
        pos,
        bond=None,
        r=0,
        theta=0,
        phi=0,
        is_deg=True,
        name="",
        residue=0,
        source_id=None,
    ):
        """Add a new atom using spherical coordinates relative to a reference.

        Parameters
        ----------
        atom_type : str
            Atom type of the new atom.
        pos : int or list
            Reference atom id or Cartesian origin.
        bond : list, optional
            Optional two-atom bond defining the local axis orientation.
        r : float, optional
            Radial distance from ``pos``.
        theta : float, optional
            Polar rotation angle.
        phi : float, optional
            Azimuthal rotation angle.
        is_deg : bool, optional
            True if ``theta`` and ``phi`` are given in degrees.
        name : str, optional
            Optional explicit atom name.
        residue : int, optional
            Residue index stored on the new atom.
        source_id : int or None, optional
            Optional identifier of the source atom in a parent structure.
        """
        bond_atom = pos if isinstance(pos, int) else None
        pos_vec = self._atoms.positions[pos].copy() if isinstance(pos, int) else np.asarray(pos, dtype=float)
        vec = self._vector(*bond) if bond else geometry.main_axis("z")

        phi += geometry.angle_polar(vec, is_deg)
        theta += geometry.angle_azi(vec, is_deg)

        phi = phi * math.pi / 180 if is_deg else phi
        theta = theta * math.pi / 180 if is_deg else theta

        coord = np.asarray(
            [
                r * math.sin(theta) * math.cos(phi),
                r * math.sin(theta) * math.sin(phi),
                r * math.cos(theta),
            ],
            dtype=float,
        )
        self._atoms.append_atom(pos_vec + coord, atom_type, name, residue, source_id=source_id)
        self._invalidate_mass_cache()

        if bond_atom is not None:
            self.add_bond(bond_atom, self.get_num() - 1)

    def delete(self, atoms):
        """Delete one atom or several atoms by index.

        Parameters
        ----------
        atoms : int or list
            Atom index or indices to remove.
        """
        atom_ids = sorted(set(self._normalize_atom_ids(atoms).tolist()))
        if not atom_ids:
            return

        old_to_new = {}
        new_index = 0
        deleted = set(atom_ids)
        for atom_index in range(self.get_num()):
            if atom_index in deleted:
                continue
            old_to_new[atom_index] = new_index
            new_index += 1

        self._atoms = self._atoms.delete(atom_ids)
        self._bonds = {
            self._normalize_bond(old_to_new[atom_a], old_to_new[atom_b])
            for atom_a, atom_b in self._bonds
            if atom_a in old_to_new and atom_b in old_to_new
        }
        self._invalidate_mass_cache()

    def overlap(self, error=0.005):
        """Return groups of atoms whose coordinates overlap within a tolerance.

        Parameters
        ----------
        error : float, optional
            Maximum absolute coordinate difference per dimension.

        Returns
        -------
        duplicate : dict
            Mapping from reference atom id to overlapping atom ids.
        """
        num_atoms = self.get_num()
        if num_atoms < 2:
            return {}

        delta = self._atoms.positions[:, None, :] - self._atoms.positions[None, :, :]
        overlap_mask = np.all(np.abs(delta) < error, axis=2)

        seen = np.zeros(num_atoms, dtype=bool)
        duplicates = {}
        for atom_a in range(num_atoms):
            if seen[atom_a]:
                continue

            partners = np.where(overlap_mask[atom_a, atom_a + 1 :])[0] + atom_a + 1
            if partners.size == 0:
                continue

            duplicates[atom_a] = partners.tolist()
            seen[atom_a] = True
            seen[partners] = True

        return duplicates

    def switch_atom_order(self, atom_a, atom_b):
        """Swap two atoms in the internal atom order.

        Parameters
        ----------
        atom_a : int
            Index of the first atom.
        atom_b : int
            Index of the second atom.
        """
        self._atoms.swap(atom_a, atom_b)
        remapped_bonds = set()
        for bond_a, bond_b in self._bonds:
            bond_a = atom_b if bond_a == atom_a else atom_a if bond_a == atom_b else bond_a
            bond_b = atom_b if bond_b == atom_a else atom_a if bond_b == atom_b else bond_b
            remapped_bonds.add(self._normalize_bond(bond_a, bond_b))
        self._bonds = remapped_bonds

    def add_bond(self, atom_a, atom_b):
        """Add one explicit bond to the molecule graph.

        Parameters
        ----------
        atom_a : int
            Index of the first bonded atom.
        atom_b : int
            Index of the second bonded atom.

        Raises
        ------
        IndexError
            Raised when one atom index is outside the molecule.
        ValueError
            Raised when a self-bond is requested.
        """
        if atom_a < 0 or atom_a >= self.get_num() or atom_b < 0 or atom_b >= self.get_num():
            raise IndexError("Molecule: Bond atom index out of range.")
        self._bonds.add(self._normalize_bond(atom_a, atom_b))

    def get_bonds(self):
        """Return explicit molecule bonds.

        Returns
        -------
        bonds : list[tuple[int, int]]
            Sorted explicit atom-index pairs.
        """
        return sorted(self._bonds)

    def infer_bonds(self, cutoff_scale=1.20):
        """Infer missing bonds from covalent radii and interatomic distances.

        Parameters
        ----------
        cutoff_scale : float, optional
            Multiplicative tolerance applied to the sum of covalent radii.

        Returns
        -------
        bonds : list[tuple[int, int]]
            Sorted inferred bond pairs that are not already explicit bonds.
        """
        num_atoms = self.get_num()
        if num_atoms < 2:
            return []

        atom_types = self._atoms.atom_types.tolist()
        elements = np.empty(num_atoms, dtype=object)
        radii = np.empty(num_atoms, dtype=float)
        valid = np.ones(num_atoms, dtype=bool)

        for atom_id, atom_type in enumerate(atom_types):
            try:
                elements[atom_id] = db.get_element(atom_type)
                radii[atom_id] = db.get_covalent_radius(atom_type)
            except ValueError:
                valid[atom_id] = False

        inferred_bonds = set()
        for atom_a in range(num_atoms):
            if not valid[atom_a]:
                continue

            candidate_ids = np.arange(atom_a + 1, num_atoms, dtype=int)
            if candidate_ids.size == 0:
                continue

            candidate_ids = candidate_ids[valid[candidate_ids]]
            if candidate_ids.size == 0:
                continue

            if elements[atom_a] == "H":
                candidate_ids = candidate_ids[elements[candidate_ids] != "H"]
                if candidate_ids.size == 0:
                    continue

            candidate_ids = np.asarray(
                [atom_b for atom_b in candidate_ids.tolist() if (atom_a, atom_b) not in self._bonds],
                dtype=int,
            )
            if candidate_ids.size == 0:
                continue

            delta = self._atoms.positions[candidate_ids] - self._atoms.positions[atom_a]
            distances = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            cutoffs = cutoff_scale * (radii[atom_a] + radii[candidate_ids])
            for atom_b in candidate_ids[distances <= cutoffs].tolist():
                inferred_bonds.add((atom_a, atom_b))

        return sorted(inferred_bonds)

    def set_atom_type(self, atom, atom_type):
        """Set the atom type of one atom.

        Parameters
        ----------
        atom : int
            Atom index.
        atom_type : str
            New atom type label.
        """
        self._atoms.atom_types[atom] = atom_type
        self._invalidate_mass_cache()

    def set_atom_name(self, atom, name):
        """Set the name of one atom.

        Parameters
        ----------
        atom : int
            Atom index.
        name : str
            New atom name.
        """
        self._atoms.names[atom] = name

    def set_atom_residue(self, atom, residue):
        """Set the residue index of one atom.

        Parameters
        ----------
        atom : int
            Atom index.
        residue : int
            New residue index.
        """
        self._atoms.residues[atom] = residue

    def get_atom_type(self, atom):
        """Return the atom type stored for one atom.

        Parameters
        ----------
        atom : int
            Atom index.

        Returns
        -------
        atom_type : str
            Atom type label.
        """
        return self._atoms.atom_types[atom]

    def get_atom_list(self):
        """Return compatibility atom snapshots for the current molecule.

        Returns
        -------
        atom_list : list[Atom]
            Detached :class:`silicams.atom.Atom` snapshots materialized from the
            internal array storage. Mutating the returned atoms does not update
            the molecule itself.
        """
        return self._atoms.to_atoms()

    ##################
    # Setter Methods #
    ##################
    def set_name(self, name):
        """Set the molecule name.

        Parameters
        ----------
        name : str
            Molecule name.
        """
        self._name = name

    def set_short(self, short):
        """Set the molecule short identifier.

        Parameters
        ----------
        short : str
            Short name used in exported structure records.
        """
        self._short = short

    def set_box(self, box):
        """Set the simulation box dimensions.

        Parameters
        ----------
        box : list
            Box lengths in all dimensions.
        """
        self._box = list(box)

    def set_charge(self, charge):
        """Set the total molecular charge.

        Parameters
        ----------
        charge : float
            Total charge of the molecule.
        """
        self._charge = charge

    def set_masses(self, masses=None):
        """Set per-atom molar masses.

        Parameters
        ----------
        masses : list, optional
            Explicit molar masses in :math:`\\frac{g}{mol}`. When omitted, the
            masses are inferred from the stored atom types using
            :mod:`silicams.database`.
        """
        if masses is not None and len(masses) > 0:
            self._masses = np.asarray(masses, dtype=float)
        else:
            self._masses = np.asarray(
                [db.get_mass(atom_type) for atom_type in self._atoms.atom_types.tolist()],
                dtype=float,
            )
        self._mass = None

    def set_mass(self, mass=0):
        """Set or derive the total molar mass of the molecule.

        Parameters
        ----------
        mass : float, optional
            Explicit total molar mass in :math:`\\frac{g}{mol}`. When zero, the
            value is derived from :meth:`get_masses`.
        """
        self._mass = float(mass) if mass else float(np.sum(self.get_masses()))

    ##################
    # Getter Methods #
    ##################
    def get_name(self):
        """Return the molecule name.

        Returns
        -------
        name : str
            Molecule name.
        """
        return self._name

    def get_short(self):
        """Return the molecule short identifier.

        Returns
        -------
        short : str
            Short name used in exported structure records.
        """
        return self._short

    def get_box(self):
        """Return the simulation box dimensions.

        Returns
        -------
        box : list
            Box lengths in all dimensions. When no explicit box was set, a
            bounding box derived from the current coordinates is returned.
        """
        return self._box if self._box else self._box_size()

    def get_num(self):
        """Return the number of atoms.

        Returns
        -------
        num : int
            Number of atoms currently stored in the molecule.
        """
        return self._atoms.get_num()

    def get_charge(self):
        """Return the total molecular charge.

        Returns
        -------
        charge : float
            Total charge of the molecule.
        """
        return self._charge

    def get_masses(self):
        """Return per-atom molar masses.

        Returns
        -------
        masses : list
            Atom masses in :math:`\\frac{g}{mol}`.
        """
        if self._masses is None:
            self.set_masses()
        return self._masses.tolist()

    def get_mass(self):
        """Return the total molar mass of the molecule.

        Returns
        -------
        mass : float
            Total molar mass in :math:`\\frac{g}{mol}`.
        """
        if self._mass is None:
            self.set_mass()
        return self._mass
