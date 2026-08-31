"""Private construction steric batches and periodic local candidate indexing.

These containers store supplied coordinates and covalent radii in nanometers.
The grid selects the query cell and its immediate wrapped neighbors; it does
not calculate clearances or guarantee a global nearest-atom search. Chemistry,
radius lookup, contact thresholds, and attachment decisions belong to the
calling engine.
"""

from dataclasses import dataclass, field

import numpy as np


__all__: list[str] = []

_STERIC_GRID_CELL_SIZE_NM = 0.25


@dataclass
class _StericAtomBatch:
    """Array-backed steric coordinates and radii for one atom collection.

    Parameters
    ----------
    positions : np.ndarray, optional
        Cartesian atom positions with shape ``(n, 3)`` in nanometers.
    radii : np.ndarray, optional
        Covalent radii in nanometers with shape ``(n,)`` used for steric cutoffs.
    block_atom_ids : np.ndarray, optional
        Source block atom ids with shape ``(n,)``. Attached ligand atoms use
        ``-1`` because they do not map back to the live scaffold.

    Notes
    -----
    Direct construction retains the supplied arrays without copying, coercion,
    or validation. Defaults are independent empty arrays with native ``float``
    positions/radii and ``int`` identifiers. Methods mutate this record by
    replacing its arrays; corresponding rows must describe the same atoms.
    """

    positions: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=float)
    )
    radii: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    block_atom_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))

    @classmethod
    def empty(cls):
        """Return an empty steric batch.

        Returns
        -------
        batch : _StericAtomBatch
            New batch with independent native-float arrays of shapes ``(0, 3)``
            and ``(0,)``, and a native-int identifier array of shape ``(0,)``.
        """
        return cls()

    @classmethod
    def concatenate(cls, batches):
        """Concatenate several steric batches into one.

        Parameters
        ----------
        batches : iterable[_StericAtomBatch or None]
            Batches to concatenate in iteration order. ``None`` entries and
            batches whose radius arrays are empty are skipped.

        Returns
        -------
        batch : _StericAtomBatch
            New arrays preserving batch order and row order within each batch.
            NumPy concatenation determines the output dtypes. With no nonempty
            batches, return the same empty-array shapes/dtypes as ``empty()``.

        Raises
        ------
        ValueError
            If participating array shapes cannot be concatenated by NumPy.
        """
        batches = [batch for batch in batches if batch is not None and batch.radii.size > 0]
        if not batches:
            return cls.empty()

        return cls(
            positions=np.concatenate([batch.positions for batch in batches], axis=0),
            radii=np.concatenate([batch.radii for batch in batches]),
            block_atom_ids=np.concatenate([batch.block_atom_ids for batch in batches]),
        )

    def append(self, positions, radii, block_atom_ids):
        """Append one or more atoms to the batch.

        Parameters
        ----------
        positions : array-like
            Cartesian positions in nanometers, reshaped to ``(n, 3)`` and
            converted to native ``float``.
        radii : array-like
            Covalent radii in nanometers, reshaped to ``(n,)`` and converted to
            native ``float``.
        block_atom_ids : array-like
            Source block atom ids, reshaped to ``(n,)`` and converted to native
            ``int``. Use ``-1`` for attached atoms without scaffold identifiers.

        Returns
        -------
        None
            Update the batch in place, preserving existing rows before new rows.

        Raises
        ------
        TypeError or ValueError
            If NumPy cannot convert, reshape, or concatenate the supplied arrays.

        Notes
        -----
        New rows are copied, so subsequent input changes do not affect the
        batch. Zero position rows leave the batch unchanged after conversion.
        Matching row counts are a caller precondition, not explicitly validated.
        """
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        radii = np.asarray(radii, dtype=float).reshape(-1)
        block_atom_ids = np.asarray(block_atom_ids, dtype=int).reshape(-1)
        if positions.shape[0] == 0:
            return

        if self.radii.size == 0:
            self.positions = positions.copy()
            self.radii = radii.copy()
            self.block_atom_ids = block_atom_ids.copy()
            return

        self.positions = np.concatenate([self.positions, positions], axis=0)
        self.radii = np.concatenate([self.radii, radii])
        self.block_atom_ids = np.concatenate([self.block_atom_ids, block_atom_ids])

    def remove_block_atoms(self, atom_ids):
        """Remove scaffold atoms from the batch by source atom id.

        Parameters
        ----------
        atom_ids : list[int] or np.ndarray
            Identifiers to remove, flattened and converted to native ``int``.
            Every matching row is removed, including sentinel rows if ``-1`` is
            explicitly supplied. Missing identifiers are ignored.

        Returns
        -------
        None
            Replace arrays with surviving rows in their original order. Empty
            batches or empty removal requests are unchanged.

        Raises
        ------
        TypeError or ValueError
            If NumPy cannot convert the removal identifiers to integers.
        """
        atom_ids = np.asarray(atom_ids, dtype=int).reshape(-1)
        if self.radii.size == 0 or atom_ids.size == 0:
            return

        keep_mask = ~np.isin(self.block_atom_ids, atom_ids)
        self.positions = self.positions[keep_mask]
        self.radii = self.radii[keep_mask]
        self.block_atom_ids = self.block_atom_ids[keep_mask]


@dataclass(frozen=True)
class _MoleculeStericBatch:
    """Array-backed steric subset extracted from one molecule.

    Parameters
    ----------
    atom_ids : np.ndarray
        Local atom ids with shape ``(n,)`` that participate in steric checks.
    positions : np.ndarray
        Cartesian positions in nanometers with shape ``(n, 3)``.
    radii : np.ndarray
        Covalent radii in nanometers with shape ``(n,)``.

    Notes
    -----
    The record is frozen, but its arrays remain mutable. Construction retains
    supplied array references and dtypes without copying, coercion, or validation.
    """

    atom_ids: np.ndarray
    positions: np.ndarray
    radii: np.ndarray


@dataclass
class _StericGrid:
    """Local spatial index for attachment steric checks.

    Parameters
    ----------
    box : tuple[float, float, float]
        Orthorhombic periodic box lengths in nanometers.
    cell_size_nm : float, optional
        Nominal cell size in nanometers, defaulting to ``0.25``. Each dimension
        has at least one cell; the last cell absorbs any remainder in box length.
    cells : dict[tuple[int, int, int], _StericAtomBatch], optional
        Cell keys mapped to atom batches in insertion order. A supplied mapping
        is retained and mutated directly; the default is an independent mapping.
    block_cells : dict[int, tuple[int, int, int]], optional
        Live scaffold identifiers mapped to their cells for removal. Attached
        atoms are not registered here. A supplied mapping is retained directly
        and must agree with ``cells``; the default is an independent mapping.

    Attributes
    ----------
    dims : tuple[int, int, int]
        Cell counts derived during initialization as
        ``max(1, int(box_length / cell_size_nm))``. Not a dataclass/constructor field.

    Notes
    -----
    The caller supplies aligned arrays and unique live scaffold identifiers.
    Inputs are not explicitly validated. Coordinates are wrapped only to choose
    cells; stored coordinates are not translated or wrapped. Query results are
    local candidates, not distance-filtered neighbors or global nearest atoms.
    """

    box: tuple[float, float, float]
    cell_size_nm: float = _STERIC_GRID_CELL_SIZE_NM
    cells: dict[tuple[int, int, int], _StericAtomBatch] = field(default_factory=dict)
    block_cells: dict[int, tuple[int, int, int]] = field(default_factory=dict)

    def __post_init__(self):
        """Derive cell counts without copying or rebuilding supplied mappings.

        Returns
        -------
        None
            Set ``dims`` on this grid, keeping at least one cell per dimension.

        Raises
        ------
        ZeroDivisionError
            If the cell size is zero.
        ValueError or OverflowError
            If a dimension-to-cell-size ratio cannot be converted to an integer.
        """
        self.dims = tuple(
            max(1, int(length / self.cell_size_nm))
            for length in self.box
        )

    def _wrap_component(self, value, dim):
        """Wrap one Cartesian component into the periodic simulation box.

        Parameters
        ----------
        value : float
            Cartesian component in nanometers.
        dim : int
            Cartesian dimension index, ``0``, ``1``, or ``2``.

        Returns
        -------
        value : float
            Coordinate in ``[0, box[dim])`` for a positive box length. For a
            nonpositive box length, return the supplied component unchanged.
        """
        box_length = self.box[dim]
        if box_length <= 0:
            return value
        return value % box_length

    def _cell_key(self, position):
        """Return the grid-cell key for one Cartesian position.

        Parameters
        ----------
        position : array-like
            Cartesian position of shape ``(3,)`` in nanometers.

        Returns
        -------
        key : tuple[int, int, int]
            Three-dimensional cell index after periodic wrapping. Components
            beyond the nominal last-cell boundary are assigned to the last cell.
        """
        key = []
        for dim in range(3):
            wrapped = self._wrap_component(position[dim], dim)
            dim_size = self.dims[dim]
            key.append(min(dim_size - 1, int(wrapped / self.cell_size_nm)))
        return tuple(key)

    def add_block_atom(self, atom_id, position, radius):
        """Add one live scaffold atom to the steric grid.

        Parameters
        ----------
        atom_id : int
            Unique live scaffold atom id; ``-1`` is reserved for attached atoms.
        position : array-like
            Cartesian atom position of shape ``(3,)`` in nanometers.
        radius : float
            Covalent radius in nanometers used for steric cutoff estimates.

        Returns
        -------
        None
            Append copied coordinates/radius/id to the cell and register its key
            in ``block_cells``. The stored position retains its original frame.
        """
        key = self._cell_key(position)
        self.cells.setdefault(key, _StericAtomBatch.empty()).append(
            np.asarray([position], dtype=float),
            np.asarray([radius], dtype=float),
            np.asarray([atom_id], dtype=int),
        )
        self.block_cells[atom_id] = key

    def add_block_atoms(self, atom_ids, positions, radii):
        """Add several live scaffold atoms to the steric grid.

        Parameters
        ----------
        atom_ids : array-like
            Unique live scaffold ids, converted to native ``int`` with shape
            ``(n,)``. The sentinel ``-1`` is reserved for attached atoms.
        positions : array-like
            Cartesian positions in nanometers, converted to native ``float``
            with shape ``(n, 3)``.
        radii : array-like
            Covalent radii in nanometers, converted to native ``float`` with
            shape ``(n,)``.

        Returns
        -------
        None
            Add copied atoms in input order, preserving order within each cell.

        Notes
        -----
        Matching row counts are a caller precondition. Iteration stops at the
        shortest input array; no explicit length or uniqueness check is made.
        """
        atom_ids = np.asarray(atom_ids, dtype=int).reshape(-1)
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        radii = np.asarray(radii, dtype=float).reshape(-1)

        for atom_id, position, radius in zip(atom_ids, positions, radii):
            self.add_block_atom(int(atom_id), position, float(radius))

    def add_attached_atoms(self, positions, radii):
        """Add several already attached ligand atoms to the steric grid.

        Parameters
        ----------
        positions : array-like
            Cartesian positions in nanometers, converted to native ``float``
            with shape ``(n, 3)``.
        radii : array-like
            Covalent radii in nanometers, converted to native ``float`` with
            shape ``(n,)``.

        Returns
        -------
        None
            Append copied atoms to cells in input order with scaffold identifier
            ``-1``. Do not add entries to ``block_cells``.

        Notes
        -----
        Matching row counts are a caller precondition. Iteration stops at the
        shorter input array; no explicit length check is made.
        """
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        radii = np.asarray(radii, dtype=float).reshape(-1)

        for position, radius in zip(positions, radii):
            key = self._cell_key(position)
            self.cells.setdefault(key, _StericAtomBatch.empty()).append(
                np.asarray([position], dtype=float),
                np.asarray([radius], dtype=float),
                np.asarray([-1], dtype=int),
            )

    def remove_block_atoms(self, atom_ids):
        """Remove scaffold atoms from the steric grid by source atom id.

        Parameters
        ----------
        atom_ids : iterable[int]
            Live scaffold identifiers to remove. Missing identifiers and cells
            are ignored. Attached atoms are not indexed in ``block_cells`` and
            therefore survive scaffold removal.

        Returns
        -------
        None
            Remove mappings and matching rows, preserving surviving row order.
            Delete cells that become empty.
        """
        for atom_id in atom_ids:
            key = self.block_cells.pop(atom_id, None)
            if key is None or key not in self.cells:
                continue
            self.cells[key].remove_block_atoms([atom_id])
            if self.cells[key].radii.size == 0:
                del self.cells[key]

    def neighbor_batch(self, position):
        """Return steric atoms from the local neighboring cells.

        Parameters
        ----------
        position : array-like
            Cartesian query position of shape ``(3,)`` in nanometers.

        Returns
        -------
        batch : _StericAtomBatch
            New arrays containing atoms from the query cell and its 26 wrapped
            neighbors, omitting absent/empty cells. Visit offsets ``-1, 0, 1``
            with x outermost and z innermost, taking each wrapped cell only on
            its first encounter. Preserve atom order within cells. Dimensions
            of one or two cells do not duplicate atoms in the result.

        Notes
        -----
        No distance or radius filtering is performed. Returned coordinates
        retain their stored values, not wrapped images relative to the query.
        """
        center = self._cell_key(position)
        seen_keys = set()
        batches = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    key = (
                        (center[0] + dx) % self.dims[0],
                        (center[1] + dy) % self.dims[1],
                        (center[2] + dz) % self.dims[2],
                    )
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    batch = self.cells.get(key)
                    if batch is not None and batch.radii.size > 0:
                        batches.append(batch)

        return _StericAtomBatch.concatenate(batches)
