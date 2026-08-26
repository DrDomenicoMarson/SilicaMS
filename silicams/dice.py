################################################################################
# Dice Class                                                                   #
#                                                                              #
"""Cube-based neighbor-search helpers for molecular structures."""
################################################################################

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class _DiceSearchCache:
    """Cached array views used by the cube-based neighbor search.

    Parameters
    ----------
    positions : np.ndarray
        Live Cartesian coordinates with shape ``(n, 3)``.
    atom_types : np.ndarray
        Live atom-type array with shape ``(n,)``.
    box : np.ndarray
        Simulation-box lengths used by the minimum-image correction.
    cube_atoms : dict
        Mapping from cube ids to atom-id arrays stored inside each cube.
    neighbor_atoms : dict
        Mapping from cube ids to concatenated atom-id arrays from the cube and
        its neighbors.
    """

    positions: np.ndarray
    atom_types: np.ndarray
    box: np.ndarray
    cube_atoms: dict
    neighbor_atoms: dict


class Dice:
    """Partition a molecule into cubes for local pair searches.

    Parameters
    ----------
    mol : Molecule
        Molecule to be divided into cubes.
    size : float
        Cube edge size.
    is_pbc : bool
        True when periodic boundary conditions should be applied during
        neighbor lookups.
    """

    def __init__(self, mol, size, is_pbc):
        """Initialize the cube-based search structure.

        Parameters
        ----------
        mol : Molecule
            Molecule to be divided into cubes.
        size : float
            Edge length of each search cube.
        is_pbc : bool
            True when periodic boundary conditions should be applied during
            neighbor lookups.
        """
        self._dim = 3
        self._mol = mol
        self._size = size
        self._is_pbc = is_pbc

        self._positions = np.asarray(self._mol.positions_view(), dtype=float)
        self._atom_types = self._mol.atom_types_view()
        self._mol_box = np.asarray(self._mol.get_box(), dtype=float)

        self._split()
        self._fill()
        self._cache = _DiceSearchCache(
            positions=self._positions,
            atom_types=self._atom_types,
            box=self._mol_box,
            cube_atoms={
                cube_id: np.asarray(atom_ids, dtype=int)
                for cube_id, atom_ids in self._pointer.items()
            },
            neighbor_atoms=self._build_neighbor_atom_cache(),
        )

    ##############
    # Management #
    ##############
    def _split(self):
        """Calculate the cube lattice and initialize cube storage."""
        self._count = [max(1, math.floor(box / self._size)) for box in self._mol_box.tolist()]

        self._origin = {}
        self._pointer = {}
        for i in range(self._count[0]):
            for j in range(self._count[1]):
                for k in range(self._count[2]):
                    cube_id = (i, j, k)
                    self._origin[cube_id] = [self._size * value for value in cube_id]
                    self._pointer[cube_id] = []

    def _pos_to_index(self, position):
        """Calculate the cube index for a given position.

        Parameters
        ----------
        position : list
            Three-dimensional coordinates.

        Returns
        -------
        index : tuple
            Cube index.
        """
        index = []
        for dim, pos in enumerate(position):
            pos_index = math.floor(pos / self._size)
            if pos_index < 0:
                pos_index = 0
            elif pos_index >= self._count[dim]:
                pos_index = self._count[dim] - 1
            index.append(pos_index)

        return tuple(index)

    def _fill(self):
        """Assign every atom id to its cube."""
        for atom_id, position in enumerate(self._positions):
            self._pointer[self._pos_to_index(position)].append(atom_id)

    def _build_neighbor_atom_cache(self):
        """Cache concatenated neighbor atom ids for every cube.

        Returns
        -------
        neighbor_atoms : dict
            Mapping from cube ids to concatenated atom-id arrays from the cube
            and its surrounding cubes.
        """
        neighbor_atoms = {}
        for cube_id in self._pointer:
            atoms = []
            for neighbor_id in self.neighbor(cube_id):
                if neighbor_id is not None and None not in neighbor_id:
                    atoms.extend(self._pointer[neighbor_id])
            neighbor_atoms[cube_id] = np.asarray(atoms, dtype=int)
        return neighbor_atoms

    ############
    # Iterator #
    ############
    def _step(self, dim, step, index):
        """Helper function for iterating through the cubes.

        Parameters
        ----------
        dim : int
            Stepping dimension.
        step : int
            Step to move.
        index : list
            Cube index.

        Returns
        -------
        index : tuple
            New cube index.
        """
        index = list(index)
        index[dim] += step

        if index[dim] >= self._count[dim]:
            index[dim] = 0 if self._is_pbc else None
        elif index[dim] < 0:
            index[dim] = self._count[dim] - 1 if self._is_pbc else None

        return tuple(index)

    def _right(self, index):
        """Step one cube to the right considering the x-axis."""
        return self._step(0, 1, index)

    def _left(self, index):
        """Step one cube to the left considering the x-axis."""
        return self._step(0, -1, index)

    def _top(self, index):
        """Step one cube to the top considering the y-axis."""
        return self._step(1, 1, index)

    def _bot(self, index):
        """Step one cube to the bottom considering the y-axis."""
        return self._step(1, -1, index)

    def _front(self, index):
        """Step one cube to the front considering the z-axis."""
        return self._step(2, 1, index)

    def _back(self, index):
        """Step one cube to the back considering the z-axis."""
        return self._step(2, -1, index)

    def neighbor(self, cube_id, is_self=True):
        """Get the ids of the cubes surrounding the given one.

        Parameters
        ----------
        cube_id : list
            Main cube index.
        is_self : bool, optional
            True to add the main cube to the output.

        Returns
        -------
        neighbor : list
            Surrounding cube ids, optionally including ``cube_id``.
        """
        neighbor = []

        z = [self._back(cube_id), cube_id, self._front(cube_id)]
        y = [[self._top(i), i, self._bot(i)] for i in z]

        for z_index in range(len(z)):
            for y_index in range(len(y[z_index])):
                neighbor.append(self._left(y[z_index][y_index]))
                neighbor.append(y[z_index][y_index])
                neighbor.append(self._right(y[z_index][y_index]))

        if not is_self:
            neighbor.pop(13)

        return [neighbor_id for neighbor_id in neighbor if neighbor_id is not None]

    ##########
    # Search #
    ##########
    def _minimum_image_delta(self, atom_id, partner_positions):
        """Return minimum-image vectors from one atom to many partner positions.

        Parameters
        ----------
        atom_id : int
            Atom id used as the reference position.
        partner_positions : np.ndarray
            Partner positions with shape ``(m, 3)``.

        Returns
        -------
        delta : np.ndarray
            Minimum-image displacement vectors with shape ``(m, 3)``.
        """
        delta = self._cache.positions[atom_id] - partner_positions
        wrap_mask = np.abs(delta) > (3 * self._size)
        if np.any(wrap_mask):
            box_rows = np.broadcast_to(self._cache.box, delta.shape)
            delta = delta.copy()
            delta[wrap_mask] -= box_rows[wrap_mask] * np.round(
                delta[wrap_mask] / box_rows[wrap_mask]
            )
        return delta

    def _find_bond(self, cube_list, atom_type, distance):
        """Search for atom pairs in the given cubes.

        Parameters
        ----------
        cube_list : list
            List of cube indices to search in. Use an empty list for all cubes.
        atom_type : list
            List of two atom types.
        distance : list
            Bounds of the allowed distance ``[lower, upper]``.

        Returns
        -------
        bond_list : list
            Bond array containing lists of two atom ids.
        """
        cube_list = cube_list if cube_list else list(self._pointer.keys())
        lower, upper = distance

        bond_list = []
        for cube_id in cube_list:
            cube_atoms = self._cache.cube_atoms[cube_id]
            if cube_atoms.size == 0:
                continue

            neighbor_atoms = self._cache.neighbor_atoms[cube_id]
            partner_ids = neighbor_atoms[self._cache.atom_types[neighbor_atoms] == atom_type[1]]
            partner_positions = self._cache.positions[partner_ids] if partner_ids.size else None

            for atom_id_a in cube_atoms.tolist():
                if self._cache.atom_types[atom_id_a] != atom_type[0]:
                    continue

                if partner_ids.size == 0:
                    bond_list.append([int(atom_id_a), []])
                    continue

                delta = self._minimum_image_delta(atom_id_a, partner_positions)
                lengths = np.sqrt(np.einsum("ij,ij->i", delta, delta))
                partners = partner_ids[
                    (lengths >= lower)
                    & (lengths <= upper)
                    & (partner_ids != atom_id_a)
                ]
                bond_list.append([int(atom_id_a), partners.astype(int).tolist()])

        return bond_list

    def find(self, cube_list=None, atom_type=None, distance=None):
        """Search for atom pairs in serial.

        Parameters
        ----------
        cube_list : list, optional
            List of cube indices to search in. Use an empty list for all cubes.
        atom_type : list, optional
            List of two atom types.
        distance : list, optional
            Bounds of allowed distance ``[lower, upper]``.

        Returns
        -------
        bond_list : list
            Bond array containing lists of two atom ids.

        Raises
        ------
        TypeError
            If ``atom_type`` or ``distance`` is missing.
        """
        if atom_type is None or distance is None:
            raise TypeError("Dice.find requires atom_type and distance arguments.")
        return self._find_bond(cube_list, atom_type, distance)

    ##################
    # Setter methods #
    ##################
    def set_pbc(self, pbc):
        """Turn the periodic boundary conditions on or off.

        Parameters
        ----------
        pbc : bool
            True to enable periodic boundary conditions for neighbor lookups.
        """
        self._is_pbc = pbc
        self._cache.neighbor_atoms = self._build_neighbor_atom_cache()

    ##################
    # Getter methods #
    ##################
    def get_origin(self):
        """Return the origin positions of the cubes.

        Returns
        -------
        origin : dict
            Mapping from cube index tuples to origin positions.
        """
        return self._origin

    def get_pointer(self):
        """Return the list of atoms in each cube.

        Returns
        -------
        pointer : dict
            Mapping from cube index tuples to lists of atom ids.
        """
        return self._pointer

    def get_count(self):
        """Return the number of cubes in each dimension.

        Returns
        -------
        count : list
            Number of cubes along ``x``, ``y``, and ``z``.
        """
        return self._count

    def get_size(self):
        """Return the cube size.

        Returns
        -------
        size : float
            Edge length of each search cube.
        """
        return self._size

    def get_mol(self):
        """Return the wrapped molecule.

        Returns
        -------
        mol : Molecule
            Wrapped molecule instance.
        """
        return self._mol
