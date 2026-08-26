################################################################################
# Private silica chemistry engine                                              #
################################################################################

"""Private surface-preparation and functionalization engine for silica."""


import copy
import random

from collections.abc import Callable
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

import silicams.database as db
import silicams.geometry as geometry
import silicams.generic as generic

from silicams._numba_kernels import minimum_clearance_against_batch
from silicams.connectivity import AttachmentRecord
from silicams.dice import Dice
from silicams.molecule import Molecule
from silicams.slit_system import SurfaceEditRecord, SurfacePreparationDiagnostics


_VALID_SITE_TYPES = {"in", "ex"}
_STERIC_CLEARANCE_SCALE = 0.85
_STERIC_GRID_CELL_SIZE_NM = 0.25


@dataclass
class BindingSite:
    """Surface silicon binding site with its remaining oxygen handles.

    Parameters
    ----------
    oxygen_ids : list[int], optional
        Surface oxygen atom identifiers currently bound to the silicon site.
    site_type : str, optional
        Surface family identifier, ``"in"`` for interior or ``"ex"`` for
        exterior.
    is_available : bool, optional
        True when the site can still accept a new attachment.
    normal : callable, optional
        Surface-normal callback used to orient attached molecules at this site.
    """

    oxygen_ids: list[int] = field(default_factory=list)
    site_type: str = "in"
    is_available: bool = True
    normal: Callable[[list], list] | None = None

    @property
    def oxygen_count(self):
        """Return the number of remaining oxygen handles on the site.

        Returns
        -------
        count : int
            Number of oxygen atoms still bound to the silicon site.
        """
        return len(self.oxygen_ids)

    @property
    def is_geminal(self):
        """Return whether the site is geminal.

        Returns
        -------
        is_geminal : bool
            True when two oxygen handles remain on the silicon site.
        """
        return self.oxygen_count == 2


@dataclass(frozen=True)
class _ScaffoldClassificationCache:
    """Cached scaffold and surface classifications for the active matrix.

    Parameters
    ----------
    attachment_scaffold_oxygen_ids : frozenset[int]
        Active scaffold oxygen ids referenced by attachment records.
    surface_handle_oxygen_ids : tuple[int, ...]
        Degree-one surface oxygen ids currently available as handles.
    framework_oxygen_ids : tuple[int, ...]
        Degree-two oxygen ids bonded to two silicon atoms.
    final_scaffold_oxygen_ids : tuple[int, ...]
        Oxygen ids exported as scaffold atoms in the finalized state,
        including attachment-referenced retained oxygens whose missing silica
        bonds are reconstructed later during topology export.
    final_scaffold_silicon_ids : tuple[int, ...]
        Silicon ids exported as scaffold atoms in the finalized state.
    """

    attachment_scaffold_oxygen_ids: frozenset[int]
    surface_handle_oxygen_ids: tuple[int, ...]
    framework_oxygen_ids: tuple[int, ...]
    final_scaffold_oxygen_ids: tuple[int, ...]
    final_scaffold_silicon_ids: tuple[int, ...]


@dataclass
class _StericAtomBatch:
    """Array-backed steric coordinates and radii for one atom collection.

    Parameters
    ----------
    positions : np.ndarray, optional
        Cartesian atom positions with shape ``(n, 3)`` in nanometers.
    radii : np.ndarray, optional
        Covalent radii with shape ``(n,)`` used for steric cutoffs.
    block_atom_ids : np.ndarray, optional
        Source block atom ids with shape ``(n,)``. Attached ligand atoms use
        ``-1`` because they do not map back to the live scaffold.
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
            Empty batch with zero rows.
        """
        return cls()

    @classmethod
    def concatenate(cls, batches):
        """Concatenate several steric batches into one.

        Parameters
        ----------
        batches : list[_StericAtomBatch]
            Batches that should be concatenated in order.

        Returns
        -------
        batch : _StericAtomBatch
            Concatenated steric batch.
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
        positions : np.ndarray
            Cartesian positions with shape ``(n, 3)``.
        radii : np.ndarray
            Covalent radii with shape ``(n,)``.
        block_atom_ids : np.ndarray
            Source block atom ids with shape ``(n,)``.
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
            Source block atom ids that should be removed.
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
        Local atom ids that participate in steric checks.
    positions : np.ndarray
        Cartesian positions of the participating atoms with shape ``(n, 3)``.
    radii : np.ndarray
        Covalent radii for the participating atoms with shape ``(n,)``.
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
        Periodic box lengths in nanometers.
    cell_size_nm : float, optional
        Cubic cell size used to bin atoms for local clash queries.
    """

    box: tuple[float, float, float]
    cell_size_nm: float = _STERIC_GRID_CELL_SIZE_NM
    cells: dict[tuple[int, int, int], _StericAtomBatch] = field(default_factory=dict)
    block_cells: dict[int, tuple[int, int, int]] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize wrapped grid dimensions from the box and cell size."""
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
            Box dimension index.

        Returns
        -------
        value : float
            Wrapped coordinate component.
        """
        box_length = self.box[dim]
        if box_length <= 0:
            return value
        return value % box_length

    def _cell_key(self, position):
        """Return the grid-cell key for one Cartesian position.

        Parameters
        ----------
        position : list[float] or tuple[float, float, float]
            Cartesian position in nanometers.

        Returns
        -------
        key : tuple[int, int, int]
            Wrapped three-dimensional cell index.
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
            Source block atom id.
        position : list[float] or tuple[float, float, float]
            Cartesian atom position in nanometers.
        radius : float
            Covalent radius used for steric cutoff estimates.
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
        atom_ids : list[int] or np.ndarray
            Source block atom ids.
        positions : np.ndarray
            Cartesian atom positions with shape ``(n, 3)`` in nanometers.
        radii : np.ndarray
            Covalent radii with shape ``(n,)``.
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
        positions : np.ndarray
            Cartesian atom positions with shape ``(n, 3)`` in nanometers.
        radii : np.ndarray
            Covalent radii with shape ``(n,)``.
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
        atom_ids : list[int] or tuple[int, ...]
            Source block atom ids to remove.
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
        position : list[float] or tuple[float, float, float]
            Cartesian query position in nanometers.

        Returns
        -------
        batch : _StericAtomBatch
            Steric atoms from the query cell and its 26 wrapped neighbors.
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


class _SilicaChemistryEngine:
    """Prepared silica block with editable binding sites.

    The class wraps a silica block and its connectivity matrix, prepares
    chemically valid surface sites, tracks interior and exterior attachment
    states, and attaches silanol, siloxane, or user-provided molecules to the
    resulting binding sites.

    Parameters
    ----------
    block : Molecule
        Block molecule that will be interpreted as a silica scaffold.
    matrix : Matrix
        Connectivity matrix describing the bonds in ``block``.
    """
    def __init__(self, block, matrix):
        # Initialize
        self._dim = 3

        self._name = ""
        self._box = []

        self._block = block
        self._matrix = matrix

        self._num_in_ex = 0
        self._oxygen_ex = []
        self._sites = {}
        self.sites_attach_mol = {}
        self._objectified_atoms = set()
        self._surface_edit_history = []
        self._surface_preparation_diagnostics = SurfacePreparationDiagnostics()
        self._attachment_records = []
        self._scaffold_cache = None
        self._is_finalized = False

        self._mol_dict = {"block": {}, "in": {}, "ex": {}}

    def _reset_surface_preparation_tracking(self):
        """Reset internal surface-edit provenance and diagnostics."""
        self._surface_edit_history = []
        self._surface_preparation_diagnostics = SurfacePreparationDiagnostics()

    def _invalidate_finalized_export_state(self):
        """Mark the current silica representation as not finalized."""
        self._is_finalized = False
        self._invalidate_scaffold_cache()

    def _invalidate_scaffold_cache(self):
        """Drop cached scaffold and surface classifications."""
        self._scaffold_cache = None

    def _clear_block_objectification(self):
        """Drop the current block-residue snapshot built from objectification."""
        self._mol_dict["block"] = {}
        self._objectified_atoms = set()

    def _attachment_scaffold_oxygen_ids(self):
        """Return scaffold oxygens already committed to silica-ligand junctions.

        Returns
        -------
        oxygen_ids : set[int]
            Active scaffold oxygen ids referenced by attachment records.
        """
        return set(self._get_scaffold_cache().attachment_scaffold_oxygen_ids)

    def _build_scaffold_cache(self):
        """Build cached scaffold and surface classifications for the matrix.

        Returns
        -------
        cache : _ScaffoldClassificationCache
            Cached classifications derived from the current matrix and
            attachment state.
        """
        matrix = self._matrix.get_matrix()
        atom_types = self._block.atom_types_view()

        attachment_scaffold_oxygen_ids = frozenset(
            oxygen_id
            for record in self._attachment_records
            for oxygen_id in record.scaffold_oxygen_source_ids
            if oxygen_id in matrix
        )

        surface_handle_oxygen_ids = []
        framework_oxygen_ids = []
        final_scaffold_oxygen_ids = []
        final_scaffold_silicon_ids = []

        for atom_id, props in matrix.items():
            atom_type = atom_types[atom_id]
            neighbors = props["atoms"]
            degree = len(neighbors)

            if atom_type == "O":
                neighbor_types = [atom_types[neighbor] for neighbor in neighbors]
                is_framework_oxygen = (
                    degree == 2 and all(neighbor_type == "Si" for neighbor_type in neighbor_types)
                )
                if is_framework_oxygen:
                    framework_oxygen_ids.append(atom_id)
                    final_scaffold_oxygen_ids.append(atom_id)
                elif atom_id in attachment_scaffold_oxygen_ids:
                    final_scaffold_oxygen_ids.append(atom_id)
                elif (
                    degree == 1
                    and neighbor_types == ["Si"]
                ):
                    surface_handle_oxygen_ids.append(atom_id)

            elif atom_type == "Si" and degree > 0:
                final_scaffold_silicon_ids.append(atom_id)

        return _ScaffoldClassificationCache(
            attachment_scaffold_oxygen_ids=attachment_scaffold_oxygen_ids,
            surface_handle_oxygen_ids=tuple(surface_handle_oxygen_ids),
            framework_oxygen_ids=tuple(sorted(framework_oxygen_ids)),
            final_scaffold_oxygen_ids=tuple(sorted(final_scaffold_oxygen_ids)),
            final_scaffold_silicon_ids=tuple(sorted(final_scaffold_silicon_ids)),
        )

    def _get_scaffold_cache(self):
        """Return cached scaffold classifications for the current matrix.

        Returns
        -------
        cache : _ScaffoldClassificationCache
            Cached scaffold and surface classifications.
        """
        if self._scaffold_cache is None:
            self._scaffold_cache = self._build_scaffold_cache()
        return self._scaffold_cache

    def _record_surface_edit(self, atom_id, reason):
        """Store one surface-edit event and update diagnostics counters.

        Parameters
        ----------
        atom_id : int
            Atom identifier affected by the edit.
        reason : str
            Reason code describing the edit.
        """
        atom_type = self._block.get_atom_type(atom_id)
        neighbor_ids = ()
        if atom_id in self._matrix.get_matrix():
            neighbor_ids = tuple(self._matrix.get_matrix()[atom_id]["atoms"])

        self._surface_edit_history.append(
            SurfaceEditRecord(
                atom_id=atom_id,
                atom_type=atom_type,
                reason=reason,
                neighbor_ids=neighbor_ids,
            )
        )

        diagnostics = self._surface_preparation_diagnostics
        if reason == "undercoordinated_si":
            diagnostics.stripped_undercoordinated_si += 1
        elif reason == "excess_surface_oxygen_si":
            diagnostics.stripped_excess_surface_oxygen_si += 1
        elif reason == "orphan_oxygen":
            diagnostics.removed_orphan_oxygen += 1
        elif reason == "invalid_oxygen":
            diagnostics.removed_invalid_oxygen += 1
        elif reason == "orphan_silicon":
            diagnostics.removed_orphan_silicon += 1
        elif reason == "inserted_bridge_oxygen":
            diagnostics.inserted_bridge_oxygen += 1

    def _remove_atoms(self, atoms, reason):
        """Remove active atoms from the matrix and record the change.

        Parameters
        ----------
        atoms : list[int]
            Atom identifiers that should be removed from the active matrix.
        reason : str
            Reason code recorded for each removed atom.
        """
        atoms = sorted({atom for atom in atoms if atom in self._matrix.get_matrix()})
        if not atoms:
            return
        for atom_id in atoms:
            self._record_surface_edit(atom_id, reason)
        self._matrix.remove(atoms)
        self._invalidate_scaffold_cache()

    def active_atom_ids(self):
        """Return active scaffold atom identifiers in connectivity order.

        Returns
        -------
        atom_ids : tuple[int, ...]
            Atom identifiers currently present in the live connectivity
            matrix.
        """
        return tuple(self._matrix.get_matrix())

    def atom_neighbors(self, atom_id):
        """Return active neighbors for one scaffold atom.

        Parameters
        ----------
        atom_id : int
            Scaffold atom identifier.

        Returns
        -------
        neighbors : tuple[int, ...]
            Neighbor identifiers in connectivity order.
        """
        return tuple(self._matrix.get_matrix()[atom_id]["atoms"])

    def atom_position(self, atom_id):
        """Return a copy of one scaffold atom position.

        Parameters
        ----------
        atom_id : int
            Scaffold atom identifier.

        Returns
        -------
        position : list[float]
            Cartesian position in nanometers.
        """
        return list(self._block.pos(atom_id))

    def atom_positions(self, atom_ids):
        """Return positions for selected scaffold atoms.

        Parameters
        ----------
        atom_ids : iterable[int]
            Scaffold atom identifiers.

        Returns
        -------
        positions : numpy.ndarray
            Cartesian positions with shape ``(n, 3)``.
        """
        atom_ids = np.asarray(tuple(atom_ids), dtype=int)
        if atom_ids.size == 0:
            return np.empty((0, self._dim), dtype=float)
        return self._block.positions_view()[atom_ids].copy()

    def atom_type(self, atom_id):
        """Return the element/type label for one scaffold atom.

        Parameters
        ----------
        atom_id : int
            Scaffold atom identifier.

        Returns
        -------
        atom_type : str
            Stored atom type.
        """
        return self._block.get_atom_type(atom_id)

    def surface_handle_oxygen_ids(self):
        """Return exposed oxygen identifiers available for surface chemistry.

        Returns
        -------
        oxygen_ids : tuple[int, ...]
            Active one-coordinate surface oxygen identifiers.
        """
        return tuple(self._surface_handle_oxygen_ids())

    def connectivity_bonds(self):
        """Return active scaffold bonds as source-identifier pairs.

        Returns
        -------
        bonds : tuple[tuple[int, int], ...]
            Sorted unique atom-id pairs from the live connectivity matrix.
        """
        bonds = set()
        for atom_id, props in self._matrix.get_matrix().items():
            for neighbor_id in props["atoms"]:
                bonds.add(tuple(sorted((atom_id, neighbor_id))))
        return tuple(sorted(bonds))

    def insert_siloxane_bridge(self, site_pair, position):
        """Insert one deterministic bridge oxygen between two surface sites.

        Parameters
        ----------
        site_pair : tuple[int, int]
            Silicon surface-site identifiers to condense.
        position : sequence[float]
            Cartesian position of the new bridge oxygen in nanometers.

        Returns
        -------
        bridge_atom_id : int
            Source identifier assigned to the inserted oxygen atom.

        Raises
        ------
        ValueError
            Raised when either surface site is absent or no longer has an
            oxygen handle that can be consumed.
        """
        site_a, site_b = site_pair
        if site_a not in self._sites or site_b not in self._sites:
            raise ValueError(
                "Cannot bridge a silicon pair that is no longer present in "
                "the site registry."
            )
        if not self._sites[site_a].oxygen_ids or not self._sites[site_b].oxygen_ids:
            raise ValueError("Both silicon sites require an oxygen handle for condensation.")

        self._invalidate_finalized_export_state()
        bridge_atom_id = self._block.get_num()
        self._block.add("O", list(position), name="OM1")
        self._matrix.add(site_a, bridge_atom_id)
        self._matrix.add(site_b, bridge_atom_id)
        self._matrix.get_matrix()[bridge_atom_id]["bonds"] = 2
        self._record_surface_edit(bridge_atom_id, "inserted_bridge_oxygen")
        self.objectify([bridge_atom_id])

        newly_condensed_si = []
        for site_id in site_pair:
            oxygen_id = self._sites[site_id].oxygen_ids[0]
            self._matrix.remove(oxygen_id)
            if self._sites[site_id].is_geminal:
                self._sites[site_id].oxygen_ids.pop(0)
            else:
                newly_condensed_si.append(site_id)
                del self._sites[site_id]

        self._invalidate_scaffold_cache()
        if newly_condensed_si:
            self.objectify(newly_condensed_si)
        return bridge_atom_id

    def _retained_scaffold_oxygen_ids(self, site_id, oxygen_ids):
        """Return scaffold oxygens that stay bonded to an attached mount atom.

        Parameters
        ----------
        site_id : int
            Silicon site id that is about to be consumed.
        oxygen_ids : list[int]
            Surface-handle oxygen ids removed together with the silicon site.

        Returns
        -------
        oxygen_ids : tuple[int, ...]
            Source ids of retained scaffold oxygens bonded to ``site_id``.
        """
        if site_id not in self._matrix.get_matrix():
            return ()

        return tuple(
            sorted(
                atom_id
                for atom_id in self._matrix.get_matrix()[site_id]["atoms"]
                if atom_id not in oxygen_ids and self._block.get_atom_type(atom_id) == "O"
            )
        )

    def _minimum_image_vector(self, pos_a, pos_b):
        """Return the shortest periodic vector from ``pos_a`` to ``pos_b``.

        Parameters
        ----------
        pos_a : list[float]
            First position vector.
        pos_b : list[float]
            Second position vector.

        Returns
        -------
        vector : list[float]
            Minimum-image displacement vector.
        """
        box = self._block.get_box()
        vector = geometry.vector(pos_a, pos_b)
        for dim, box_length in enumerate(box):
            if box_length <= 0:
                continue
            half_box = box_length / 2
            while vector[dim] > half_box:
                vector[dim] -= box_length
            while vector[dim] < -half_box:
                vector[dim] += box_length
        return vector

    def _minimum_image_displacements(self, pos_a, positions_b, box=None):
        """Return shortest periodic vectors from one point to many targets.

        Parameters
        ----------
        pos_a : list[float] or np.ndarray
            Reference position.
        positions_b : np.ndarray
            Target positions with shape ``(n, 3)``.
        box : list[float] or np.ndarray or None, optional
            Periodic box lengths. When ``None``, the current block box is used.

        Returns
        -------
        vectors : np.ndarray
            Minimum-image displacement vectors with shape ``(n, 3)``.
        """
        positions_b = np.asarray(positions_b, dtype=float).reshape(-1, self._dim)
        if positions_b.size == 0:
            return np.empty((0, self._dim), dtype=float)

        vectors = positions_b - np.asarray(pos_a, dtype=float)
        box = np.asarray(self._block.get_box() if box is None else box, dtype=float)
        for dim, box_length in enumerate(box):
            if box_length <= 0:
                continue
            half_box = box_length / 2
            component = vectors[:, dim]
            original_positive = component > 0
            component[:] = np.mod(component + half_box, box_length) - half_box
            boundary_mask = (np.abs(component + half_box) <= 1e-12) & original_positive
            component[boundary_mask] = half_box

        return vectors

    def _steric_radii(self, atom_types):
        """Return covalent radii for atom types participating in sterics.

        Parameters
        ----------
        atom_types : np.ndarray
            Atom-type array with shape ``(n,)``.

        Returns
        -------
        result : tuple[np.ndarray, np.ndarray]
            Tuple ``(radii, valid_mask)`` where ``radii`` contains the radii of
            all valid atom types in input order and ``valid_mask`` marks the
            entries for which a covalent radius is available.
        """
        atom_types = np.asarray(atom_types)
        valid_mask = np.zeros(atom_types.shape[0], dtype=bool)
        radii = np.empty(atom_types.shape[0], dtype=float)

        for atom_type in np.unique(atom_types):
            type_mask = atom_types == atom_type
            try:
                radii[type_mask] = db.get_covalent_radius(str(atom_type))
            except ValueError:
                continue
            valid_mask[type_mask] = True

        return radii[valid_mask], valid_mask

    def _molecule_steric_batch(self, mol):
        """Return the steric subset of one molecule as arrays.

        Parameters
        ----------
        mol : Molecule
            Molecule whose steric atoms should be extracted.

        Returns
        -------
        batch : _MoleculeStericBatch
            Local atom ids, positions, and radii for all atoms with known
            covalent radii.
        """
        atom_types = mol.atom_types_view()
        radii, valid_mask = self._steric_radii(atom_types)
        atom_ids = np.flatnonzero(valid_mask).astype(int)

        return _MoleculeStericBatch(
            atom_ids=atom_ids,
            positions=np.asarray(mol.positions_view(), dtype=float)[valid_mask].copy(),
            radii=radii,
        )

    def _reference_steric_batch(self):
        """Return array-backed steric references for the current silica state.

        Returns
        -------
        batch : _StericAtomBatch
            Concatenated scaffold and already attached ligand atoms used for
            global steric scans.
        """
        batches = []

        matrix_ids = np.asarray(list(self._matrix.get_matrix().keys()), dtype=int)
        if matrix_ids.size > 0:
            radii, valid_mask = self._steric_radii(self._block.atom_types_view()[matrix_ids])
            valid_ids = matrix_ids[valid_mask]
            if valid_ids.size > 0:
                batches.append(
                    _StericAtomBatch(
                        positions=np.asarray(self._block.positions_view(), dtype=float)[valid_ids].copy(),
                        radii=radii,
                        block_atom_ids=valid_ids.copy(),
                    )
                )

        for site_group in ("in", "ex"):
            for molecules in self._mol_dict[site_group].values():
                for attached_molecule in molecules:
                    molecule_batch = self._molecule_steric_batch(attached_molecule)
                    if molecule_batch.radii.size == 0:
                        continue
                    batches.append(
                        _StericAtomBatch(
                            positions=molecule_batch.positions,
                            radii=molecule_batch.radii,
                            block_atom_ids=np.full(molecule_batch.radii.size, -1, dtype=int),
                        )
                    )

        return _StericAtomBatch.concatenate(batches)

    def _filtered_steric_batch_arrays(
        self,
        reference_batch,
        ignored_block_atom_ids=None,
    ):
        """Return steric-batch arrays filtered for one clearance query.

        Parameters
        ----------
        reference_batch : _StericAtomBatch
            Reference steric atoms used during one clearance evaluation.
        ignored_block_atom_ids : np.ndarray or None, optional
            Sorted scaffold atom ids that should be removed from the reference
            batch before the numeric clearance kernel runs.

        Returns
        -------
        arrays : tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple ``(positions, radii, block_atom_ids)`` ready for the
            numba-backed clearance kernel.
        """
        positions = np.asarray(reference_batch.positions, dtype=np.float64)
        radii = np.asarray(reference_batch.radii, dtype=np.float64)
        block_atom_ids = np.asarray(reference_batch.block_atom_ids, dtype=np.int64)

        if radii.size == 0:
            return positions.reshape(0, self._dim), radii, block_atom_ids

        if ignored_block_atom_ids is not None and ignored_block_atom_ids.size > 0:
            keep_mask = np.ones(block_atom_ids.shape[0], dtype=bool)
            for atom_id in ignored_block_atom_ids:
                keep_mask &= block_atom_ids != atom_id
            positions = positions[keep_mask]
            radii = radii[keep_mask]
            block_atom_ids = block_atom_ids[keep_mask]

        return positions, radii, block_atom_ids

    def _point_clearance_against_batch(
        self,
        position,
        radius,
        reference_batch,
        ignored_block_atoms,
        steric_clearance_scale,
        box=None,
        ignored_block_atom_ids=None,
    ):
        """Return the minimum clearance from one atom to one reference batch.

        Parameters
        ----------
        position : list[float] or np.ndarray
            Cartesian query position.
        radius : float
            Covalent radius of the query atom.
        reference_batch : _StericAtomBatch
            Reference atoms checked against the query position.
        ignored_block_atoms : set[int]
            Scaffold atom ids that should be ignored.
        steric_clearance_scale : float
            Multiplicative factor applied to steric cutoffs.
        box : np.ndarray or None, optional
            Periodic box lengths used for minimum-image corrections.
        ignored_block_atom_ids : np.ndarray or None, optional
            Sorted integer array of ignored scaffold atom ids. When ``None``,
            it is derived from ``ignored_block_atoms``.

        Returns
        -------
        clearance : float
            Minimum distance minus steric cutoff. ``inf`` when no reference
            atoms remain after filtering.
        """
        if ignored_block_atom_ids is None:
            ignored_block_atom_ids = (
                np.asarray(sorted(ignored_block_atoms), dtype=np.int64)
                if ignored_block_atoms
                else np.empty(0, dtype=np.int64)
            )

        box = np.asarray(self._block.get_box() if box is None else box, dtype=np.float64)
        positions, radii, block_atom_ids = self._filtered_steric_batch_arrays(
            reference_batch,
            ignored_block_atom_ids=ignored_block_atom_ids,
        )
        if radii.size == 0:
            return float("inf")

        return float(
            minimum_clearance_against_batch(
                np.asarray(position, dtype=np.float64).reshape(1, self._dim),
                np.asarray([radius], dtype=np.float64),
                positions,
                radii,
                block_atom_ids,
                np.empty(0, dtype=np.int64),
                box,
                steric_clearance_scale,
            )
        )

    def _positions_clearance(
        self,
        positions,
        radii,
        steric_grid,
        ignored_block_atoms,
        steric_clearance_scale,
        box=None,
    ):
        """Return the minimum steric clearance for array-backed positions.

        Parameters
        ----------
        positions : np.ndarray
            Candidate atom positions with shape ``(n, 3)``.
        radii : np.ndarray
            Candidate covalent radii with shape ``(n,)``.
        steric_grid : _StericGrid or None
            Spatial index used to restrict local steric checks. When ``None``,
            the full silica state is scanned.
        ignored_block_atoms : set[int]
            Scaffold atom ids that should be ignored.
        steric_clearance_scale : float
            Multiplicative factor applied to steric cutoffs.
        box : np.ndarray or None, optional
            Periodic box lengths used for minimum-image corrections. When
            ``None``, the current block box is used.

        Returns
        -------
        clearance : float
            Minimum distance minus steric cutoff across all checked pairs.
        """
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, self._dim)
        radii = np.asarray(radii, dtype=np.float64).reshape(-1)
        if radii.size == 0:
            return float("inf")
        box = np.asarray(self._block.get_box() if box is None else box, dtype=np.float64)
        ignored_block_atom_ids = (
            np.asarray(sorted(ignored_block_atoms), dtype=np.int64)
            if ignored_block_atoms
            else np.empty(0, dtype=np.int64)
        )
        empty_ignored_block_atom_ids = np.empty(0, dtype=np.int64)

        if steric_grid is None:
            reference_positions, reference_radii, reference_block_atom_ids = (
                self._filtered_steric_batch_arrays(
                    self._reference_steric_batch(),
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
            cell_key = steric_grid._cell_key(position)
            cell_groups.setdefault(cell_key, []).append(atom_index)

        min_clearance = float("inf")
        neighbor_cache = {}
        for cell_key, atom_indices in cell_groups.items():
            if cell_key not in neighbor_cache:
                batch = steric_grid.neighbor_batch(positions[atom_indices[0]])
                neighbor_cache[cell_key] = self._filtered_steric_batch_arrays(
                    batch,
                    ignored_block_atom_ids=ignored_block_atom_ids,
                )
            reference_positions, reference_radii, reference_block_atom_ids = neighbor_cache[cell_key]
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

    def _rotation_matrix(self, axis, angle, is_deg=True):
        """Build the rotation matrix for an arbitrary three-dimensional axis.

        Parameters
        ----------
        axis : list[float] or tuple[float, float, float]
            Rotation-axis vector.
        angle : float
            Rotation angle.
        is_deg : bool, optional
            True when ``angle`` is given in degrees.

        Returns
        -------
        matrix : np.ndarray
            Rotation matrix with shape ``(3, 3)``.
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

    def _rotate_positions_around_axis(self, positions, origin, axis, angle):
        """Return positions rotated around one axis passing through ``origin``.

        Parameters
        ----------
        positions : np.ndarray
            Cartesian coordinates with shape ``(n, 3)``.
        origin : list[float] or np.ndarray
            One point on the rotation axis.
        axis : list[float] or tuple[float, float, float]
            Rotation-axis vector.
        angle : float
            Rotation angle in degrees.

        Returns
        -------
        positions : np.ndarray
            Rotated coordinates with shape ``(n, 3)``.
        """
        rotation = self._rotation_matrix(axis, angle, is_deg=True)
        centered = np.asarray(positions, dtype=float) - np.asarray(origin, dtype=float)
        return centered @ rotation.T + np.asarray(origin, dtype=float)

    def _build_steric_grid(self):
        """Build a local steric-search grid for the current silica state.

        Returns
        -------
        grid : _StericGrid
            Spatial index containing active scaffold atoms and already attached
            ligand atoms.
        """
        grid = _StericGrid(tuple(self._block.get_box()))
        matrix_ids = np.asarray(list(self._matrix.get_matrix().keys()), dtype=int)
        if matrix_ids.size > 0:
            radii, valid_mask = self._steric_radii(self._block.atom_types_view()[matrix_ids])
            valid_ids = matrix_ids[valid_mask]
            if valid_ids.size > 0:
                grid.add_block_atoms(
                    valid_ids,
                    np.asarray(self._block.positions_view(), dtype=float)[valid_ids],
                    radii,
                )

        for site_group in ("in", "ex"):
            for molecules in self._mol_dict[site_group].values():
                for attached_molecule in molecules:
                    molecule_batch = self._molecule_steric_batch(attached_molecule)
                    if molecule_batch.radii.size > 0:
                        grid.add_attached_atoms(molecule_batch.positions, molecule_batch.radii)

        return grid

    def _placement_clearance(
        self,
        mol,
        steric_grid=None,
        ignored_block_atoms=None,
        steric_clearance_scale=_STERIC_CLEARANCE_SCALE,
        box=None,
    ):
        """Return the minimum steric clearance of a candidate attached pose.

        Parameters
        ----------
        mol : Molecule
            Candidate attached molecule.
        steric_grid : _StericGrid or None, optional
            Spatial index used to restrict the steric check to local
            neighboring atoms. When ``None``, the full current structure is
            scanned directly.
        ignored_block_atoms : set[int] or None, optional
            Scaffold atom ids that should be ignored during the clash check.
        steric_clearance_scale : float, optional
            Multiplicative factor applied to the sum of covalent radii when
            estimating the steric cutoff.
        box : np.ndarray or None, optional
            Periodic box lengths used for minimum-image corrections. When
            ``None``, the current block box is used.

        Returns
        -------
        clearance : float
            Minimum distance minus the steric cutoff across all checked pairs.
            Positive values indicate a clash-free pose.
        """
        ignored_block_atoms = set() if ignored_block_atoms is None else set(ignored_block_atoms)
        molecule_batch = self._molecule_steric_batch(mol)
        return self._positions_clearance(
            molecule_batch.positions,
            molecule_batch.radii,
            steric_grid,
            ignored_block_atoms,
            steric_clearance_scale,
            box=box,
        )

    def _rotation_angles(self, rotate_step_deg):
        """Return the sampled axis-rotation angles for one pose search.

        Parameters
        ----------
        rotate_step_deg : float
            Angular step in degrees used to sample rotations around the mount
            axis.

        Returns
        -------
        angles : tuple[float, ...]
            Rotation angles covering one full turn, including ``0``.

        Raises
        ------
        ValueError
            Raised when ``rotate_step_deg`` is not strictly positive.
        """
        if rotate_step_deg <= 0:
            raise ValueError("Attachment rotation step must be greater than zero.")

        angles = []
        angle = 0.0
        while angle < 360.0:
            angles.append(round(angle, 10))
            angle += rotate_step_deg

        return tuple(angles if angles else [0.0])

    def _optimize_attachment_pose(
        self,
        mol,
        mount,
        surf_axis,
        ignored_block_atoms,
        steric_grid,
        is_rotate,
        rotate_step_deg,
        steric_clearance_scale,
        box=None,
    ):
        """Rotate one candidate pose around the mount axis to reduce clashes.

        Parameters
        ----------
        mol : Molecule
            Candidate attached molecule already aligned to ``surf_axis`` and
            translated to the target site.
        mount : int
            Mount atom index.
        surf_axis : list[float]
            Surface-normal vector used as the rotation axis.
        ignored_block_atoms : set[int]
            Scaffold atom ids ignored during the steric check.
        steric_grid : _StericGrid or None
            Spatial index used to evaluate only local steric clashes.
        is_rotate : bool
            True to scan several rotations around the mount axis.
        rotate_step_deg : float
            Angular step in degrees used during the rotation scan.
        steric_clearance_scale : float
            Multiplicative factor applied to the sum of covalent radii when
            estimating the steric cutoff.
        box : np.ndarray or None, optional
            Periodic box lengths used for minimum-image corrections. When
            ``None``, the current block box is used.

        Returns
        -------
        mol : Molecule or None
            Best non-clashing pose, or ``None`` when every sampled pose clashes.
        """
        angles = self._rotation_angles(rotate_step_deg) if is_rotate else (0.0,)
        best_clearance = float("-inf")
        best_positions = None
        base_positions = np.asarray(mol.positions_view(), dtype=float).copy()
        molecule_batch = self._molecule_steric_batch(mol)
        mount_position = base_positions[mount].copy()
        box = np.asarray(self._block.get_box() if box is None else box, dtype=float)

        for angle in angles:
            candidate_positions = (
                base_positions
                if angle == 0
                else self._rotate_positions_around_axis(
                    base_positions,
                    mount_position,
                    surf_axis,
                    angle,
                )
            )
            clearance = self._positions_clearance(
                candidate_positions[molecule_batch.atom_ids],
                molecule_batch.radii,
                steric_grid,
                ignored_block_atoms,
                steric_clearance_scale,
                box=box,
            )
            if clearance > best_clearance:
                best_clearance = clearance
                best_positions = candidate_positions.copy()

        if best_clearance < 0:
            return None
        best_molecule = copy.deepcopy(mol)
        best_molecule.positions_view()[:] = best_positions
        return best_molecule

    def _surface_handle_oxygen_ids(self):
        """Return oxygen atoms that currently behave as surface handles.

        Returns
        -------
        oxygen_ids : list[int]
            Degree-one oxygen atoms bonded to exactly one silicon atom and not
            already committed to an existing silica-ligand junction.
        """
        return list(self._get_scaffold_cache().surface_handle_oxygen_ids)

    def _final_scaffold_oxygen_ids(self):
        """Return oxygen atoms that belong to the finalized scaffold export.

        Returns
        -------
        oxygen_ids : list[int]
            Matrix oxygen ids exported as ``OM`` after finalization, including
            framework oxygens and every attachment-referenced retained scaffold
            oxygen, even when the live matrix no longer preserves both silica
            bonds that will be reconstructed during topology export.
        """
        return list(self._get_scaffold_cache().final_scaffold_oxygen_ids)

    def _final_scaffold_silicon_ids(self):
        """Return silicon atoms that belong to the finalized scaffold export.

        Returns
        -------
        silicon_ids : list[int]
            Active silicon ids that remain part of the finalized scaffold.
        """
        return list(self._get_scaffold_cache().final_scaffold_silicon_ids)

    def _framework_oxygen_ids(self):
        """Return oxygen atoms that qualify as framework oxygens.

        Returns
        -------
        oxygen_ids : list[int]
            Degree-two oxygen atoms bonded to two silicon atoms.
        """
        return list(self._get_scaffold_cache().framework_oxygen_ids)

    def _invalid_oxygen_ids(self):
        """Return oxygen atoms whose active connectivity is chemically invalid.

        Returns
        -------
        invalid_groups : tuple[list[int], list[int]]
            Two lists containing orphan oxygens and other invalid oxygens.
        """
        orphan_oxygen = []
        invalid_oxygen = []
        for atom_id, props in self._matrix.get_matrix().items():
            if self._block.get_atom_type(atom_id) != "O":
                continue

            degree = len(props["atoms"])
            neighbor_types = [self._block.get_atom_type(neighbor) for neighbor in props["atoms"]]
            if degree == 0:
                orphan_oxygen.append(atom_id)
            elif degree == 1 and neighbor_types == ["Si"]:
                continue
            elif degree == 2 and all(atom_type == "Si" for atom_type in neighbor_types):
                continue
            else:
                invalid_oxygen.append(atom_id)

        return orphan_oxygen, invalid_oxygen

    def _orphan_silicon_ids(self):
        """Return silicon atoms that no longer have active bonded neighbors.

        Returns
        -------
        silicon_ids : list[int]
            Zero-bond silicon atoms present in the active matrix.
        """
        return [
            atom_id
            for atom_id, props in self._matrix.get_matrix().items()
            if self._block.get_atom_type(atom_id) == "Si" and len(props["atoms"]) == 0
        ]

    def _silicon_with_excess_surface_oxygen(self):
        """Return silicon atoms carrying three or more surface oxygen handles.

        Returns
        -------
        silicon_ids : list[int]
            Silicon identifiers that currently own at least three degree-one
            oxygen handles.
        """
        if not self._surface_handle_oxygen_ids():
            return []

        si_count = Counter(
            sum(
                [self._matrix.get_matrix()[atom]["atoms"] for atom in self._surface_handle_oxygen_ids()],
                [],
            )
        )
        return sorted(si for si, count in si_count.items() if count >= 3)

    def refresh_surface_preparation_diagnostics(self):
        """Refresh the final surface/scaffold counts stored in diagnostics."""
        diagnostics = self._surface_preparation_diagnostics
        diagnostics.final_surface_oxygen_handles = len(self._surface_handle_oxygen_ids())
        diagnostics.final_framework_oxygen = len(self._final_scaffold_oxygen_ids())

    def validate_scaffold_atoms(self, atoms, allow_attachment_oxygen=False):
        """Validate scaffold atoms before they are exported as one-atom residues.

        Parameters
        ----------
        atoms : list[int]
            Active scaffold atom identifiers that are about to be objectified.
        allow_attachment_oxygen : bool, optional
            When ``True``, also accept retained silica-ligand junction oxygens
            as valid ``OM`` export atoms.

        Raises
        ------
        ValueError
            Raised when an atom does not match a valid scaffold role.
        """
        framework_oxygen = (
            set(self._final_scaffold_oxygen_ids())
            if allow_attachment_oxygen
            else set(self._framework_oxygen_ids())
        )
        for atom_id in atoms:
            if atom_id not in self._matrix.get_matrix():
                raise ValueError(f"Scaffold atom {atom_id} is not present in the active matrix.")

            atom_type = self._block.get_atom_type(atom_id)
            if atom_type == "O":
                if atom_id not in framework_oxygen:
                    detail = (
                        "a valid framework oxygen or retained attachment oxygen"
                        if allow_attachment_oxygen
                        else "a valid two-coordinate framework oxygen"
                    )
                    raise ValueError(
                        f"Scaffold oxygen {atom_id} cannot be exported as OM because it is not {detail}."
                    )
            elif atom_type != "Si":
                raise ValueError(
                    f"Unsupported scaffold atom type '{atom_type}' for objectification."
                )


    ###########
    # Surface #
    ###########
    def prepare(self):
        """Prepare the carved silica surface for later functionalization.

        The preparation follows the silica-surface cleanup rules used throughout
        SilicaMS: undercoordinated silicon atoms are removed, silicon atoms with
        excessive dangling oxygen neighbors are stripped, and the remaining
        unsaturated oxygen atoms become attachment handles for later surface
        chemistry.
        """
        self._invalidate_finalized_export_state()
        self._reset_surface_preparation_tracking()

        while True:
            changed = False

            undercoordinated_si = [
                atom_id
                for atom_id, props in self._matrix.get_matrix().items()
                if self._block.get_atom_type(atom_id) == "Si"
                and len(props["atoms"]) < props["bonds"]
            ]
            if undercoordinated_si:
                self._remove_atoms(undercoordinated_si, "undercoordinated_si")
                changed = True

            excess_surface_oxygen_si = self._silicon_with_excess_surface_oxygen()
            if excess_surface_oxygen_si:
                self._remove_atoms(excess_surface_oxygen_si, "excess_surface_oxygen_si")
                changed = True

            orphan_oxygen, invalid_oxygen = self._invalid_oxygen_ids()
            if orphan_oxygen:
                self._remove_atoms(orphan_oxygen, "orphan_oxygen")
                changed = True
            if invalid_oxygen:
                self._remove_atoms(invalid_oxygen, "invalid_oxygen")
                changed = True

            orphan_silicon = self._orphan_silicon_ids()
            if orphan_silicon:
                self._remove_atoms(orphan_silicon, "orphan_silicon")
                changed = True

            if not changed:
                break

        self.refresh_surface_preparation_diagnostics()

    def amorph(self, dist=0.05, accept=None, trials=100):
        """Randomly displace bonded atoms to roughen the local structure.

        Parameters
        ----------
        dist : float, optional
            Maximum displacement per Cartesian component.
        accept : list, optional
            Accepted bond-length interval after displacement.
        trials : int, optional
            Maximum number of displacement attempts per atom.
        """
        accept = [0.1, 0.2] if accept is None else accept

        # Get connectivity matrix
        connect = self._matrix.get_matrix()

        # Run through atoms that have bond partners
        for atom_id in self._matrix.bound(0, "gt"):
            atom_temp_pos = self._block.pos(atom_id)

            # Run through trials
            for i in range(trials):
                # Create random displacement vector
                disp_vec = [random.uniform(-dist, dist) for x in range(self._dim)]
                disp_pos = [atom_temp_pos[x]+disp_vec[x] for x in range(self._dim)]

                # Calculate new bond lengths
                is_disp = True
                for bond in connect[atom_id]["atoms"]:
                    bond_length = geometry.length(geometry.vector(disp_pos, self._block.pos(bond)))
                    if bond_length < accept[0] or bond_length > accept[1]:
                        is_disp = False
                        break

                # Displace if new bond length is in acceptance range
                if is_disp:
                    self._block.put(atom_id, disp_pos)
                    break

    def exterior(self):
        """Create exterior surface sites for reservoir-facing functionalization.

        Periodic Si-O bonds crossing the box boundary are split, replacement
        oxygen atoms are added on the exposed silicon atoms, and the block is
        shifted so the newly exposed outer surfaces can later be populated with
        molecules.
        """
        # Initialize
        box = self._block.get_box()
        bound_list = self._matrix.get_matrix()

        # Get gap
        gap = [-2*x for x in self._block.zero()]

        # Get list of all si atoms
        si_list = np.where(self._block.atom_types_view() == "Si")[0].tolist()

        # Run through silicon atoms
        add_list = []
        break_list = []
        for si in si_list:
            if si not in bound_list:
                continue
            # Run through bound oxygen atoms
            for o in bound_list[si]["atoms"]:
                # Calculate bond vector
                bond_vector = [
                    self._block.pos(si)[dim] - self._block.pos(o)[dim]
                    for dim in range(3)
                ]

                # Check if z dimension of bond - after rotation in pattern class - goes over boundary
                if abs(bond_vector[2]) > box[2]/2:
                    # Get bond direction
                    r = -0.155 if bond_vector[2] < 0 else 0.155

                    # Get bond angle
                    theta = geometry.angle_azi(bond_vector, is_deg=True)
                    phi = geometry.angle_polar(bond_vector, is_deg=True)
                    theta = theta-180 if r < 0 else theta

                    # Add new oxygen atom
                    self._block.add("O", si, r=r, theta=theta, phi=phi)

                    # Add list for new atoms
                    add_list.append([si, self._block.get_num()-1])

                    # Add to break bond list
                    break_list.append([si, o])

        # Translate initial gap
        self._block.zero()
        self._block.translate(gap)

        # Break periodic bonds
        for bond in break_list:
            self._matrix.split(bond[0], bond[1])

        # Add bonds to new oxygens
        for bond in add_list:
            self._matrix.add(bond[0], bond[1])
        if break_list or add_list:
            self._invalidate_scaffold_cache()

        # Add atoms to exterior oxygen list
        self.prepare()
        self._oxygen_ex = self._surface_handle_oxygen_ids()

    def _site_search_molecule(self, sites):
        """Create a zero-based temporary molecule for local site searches.

        Parameters
        ----------
        sites : list
            Silicon site identifiers used for a local proximity search.

        Returns
        -------
        site_molecule : Molecule
            Temporary molecule containing deep-copied silicon atoms translated
            so that all coordinates are non-negative.
        """
        site_atoms = [self._block._materialize_atom(atom) for atom in sites]
        site_molecule = Molecule(inp=site_atoms)
        site_molecule.zero()

        return site_molecule

    def sites(self):
        """Build the internal binding-site registry.

        Each silicon surface site is mapped to a :class:`BindingSite`
        containing the currently exposed oxygen atoms, the site type
        (``"in"`` or ``"ex"``), and the availability flag updated during
        later attachment steps.
        """
        # Get list of surface oxygen atoms
        oxygen_list = self._surface_handle_oxygen_ids()
        connect = self._matrix.get_matrix()

        # Create binding site dictionary
        self._num_in_ex = 0
        self._sites = {}
        for o in oxygen_list:
            site_id = connect[o]["atoms"][0]
            if site_id not in self._sites:
                self._sites[site_id] = BindingSite()
            self._sites[site_id].oxygen_ids.append(o)

        # Fill other information
        for si, data in self._sites.items():
            # Site type
            is_in = False
            is_ex = False
            for o in data.oxygen_ids:
                if o in self._oxygen_ex:
                    is_ex = True
                else:
                    is_in = True

            data.site_type = "ex" if is_ex else "in"

            if is_in and is_ex:
                self._num_in_ex += 1

            # State
            data.is_available = True


    #######################
    # Molecule Attachment #
    #######################
    def _validate_site_type(self, site_type):
        """Validate a silica surface-site type.

        Parameters
        ----------
        site_type : str
            Requested site type.

        Raises
        ------
        ValueError
            Raised when the site type is not supported.
        """
        if site_type not in _VALID_SITE_TYPES:
            raise ValueError(
                f"Unsupported site_type '{site_type}'. Expected one of: {sorted(_VALID_SITE_TYPES)}."
            )

    def attach(
        self,
        mol,
        mount,
        axis,
        sites,
        amount,
        scale=1,
        trials=1000,
        pos_list=None,
        site_type="in",
        is_proxi=True,
        is_random=True,
        is_rotate=False,
        rotate_step_deg=30,
        is_g=True,
        check_sterics=True,
        steric_clearance_scale=_STERIC_CLEARANCE_SCALE,
        _steric_grid=None,
        _progress_callback=None,
        _site_attempt_callback=None,
    ):
        """Attach molecules to available silica surface sites.

        Parameters
        ----------
        mol : Molecule
            Molecule to attach.
        mount : int
            Atom id on ``mol`` placed onto the selected silicon site.
        axis : list
            Two atom ids defining the molecule orientation axis.
        sites : list
            Silicon site ids from which attachment positions are chosen.
        amount : int
            Number of molecules to attach.
        scale : float, optional
            Effective lateral spacing multiplier for proximity searches.
        trials : int, optional
            Number of random site-selection attempts per molecule.
        pos_list : list, optional
            Explicit Cartesian target positions used to pick nearest free sites.
        site_type : str, optional
            Site family, either interior ``"in"`` or exterior ``"ex"``.
        is_proxi : bool, optional
            True to consume nearby sites with silanol fills after attachment.
        is_random : bool, optional
            True to choose sites randomly from ``sites``.
        is_rotate : bool, optional
            True to scan several rotations around the molecule axis and keep
            the least crowded pose before accepting one placement.
        rotate_step_deg : float, optional
            Angular step in degrees used when ``is_rotate`` is enabled.
        is_g : bool, optional
            True to allow geminal surface sites as mounting positions.
        check_sterics : bool, optional
            True to reject placements whose final pose clashes with the current
            silica scaffold or already attached molecules. False to place the
            molecule directly after geometric alignment without steric
            rejection. Final silanol saturation uses ``False`` so every
            remaining free site is consumed.
        steric_clearance_scale : float, optional
            Multiplicative factor applied to the sum of covalent radii when
            estimating the steric cutoff for clash rejection.
        _steric_grid : _StericGrid or None, optional
            Internal steric-grid cache reused across recursive proximity-fill
            calls. External callers should keep the default ``None``.
        _progress_callback : callable or None, optional
            Internal callback invoked once per requested attachment slot with
            keyword arguments ``requested_count``, ``attached_count``, and
            ``success``. External callers should keep the default ``None``.
        _site_attempt_callback : callable or None, optional
            Internal callback invoked for every sterically evaluated candidate
            site with ``site_id`` and ``success`` keyword arguments. External
            callers should keep the default ``None``.

        Returns
        -------
        mol_list : list
            Attached molecule copies in placement order.

        Raises
        ------
        ValueError
            Raised when the requested site type is not supported or when the
            steric clearance scale is not strictly positive.
        """
        self._validate_site_type(site_type)
        if steric_clearance_scale <= 0:
            raise ValueError("Attachment steric clearance scale must be greater than zero.")
        pos_list = [] if pos_list is None else pos_list

        # Rotate molecule towards z-axis
        mol_axis = mol.bond(*axis)
        mol.rotate(geometry.cross_product(mol_axis, [0, 0, 1]), geometry.angle(mol_axis, [0, 0, 1]))
        mol.zero()

        # Search for overlapping placements - Calculate diameter and add carbon VdW-raidus (Wiki)
        if is_proxi:
            mol_diam = (max(mol.get_box()[:2])+0.17)*scale
            si_dice = Dice(self._site_search_molecule(sites), mol_diam, True)
            si_proxi = si_dice.find(None, ["Si", "Si"], [-mol_diam, mol_diam])
            si_matrix = {x[0]: x[1] for x in si_proxi}

        steric_grid = (
            _steric_grid
            if _steric_grid is not None
            else (self._build_steric_grid() if check_sterics else None)
        )
        box = np.asarray(self._block.get_box(), dtype=float)

        # Run through number of binding sites to add
        mol_list = []
        attached_count = 0
        for i in range(amount):
            slot_succeeded = False
            if pos_list:
                pos = pos_list[i]
                candidate_sites = [
                    site
                    for site in sites
                    if self._sites[site].is_available
                    and (is_g or not self._sites[site].is_geminal)
                ]
                candidate_sites.sort(
                    key=lambda site: geometry.length(
                        geometry.vector(self._block.pos(site), pos)
                    )
                )
            elif is_random:
                candidate_sites = [
                    site
                    for site in sites
                    if self._sites[site].is_available
                    and (is_g or not self._sites[site].is_geminal)
                ]
                random.shuffle(candidate_sites)
            else:
                candidate_sites = [
                    site
                    for site in sites[i:]
                    if self._sites[site].is_available
                    and (is_g or not self._sites[site].is_geminal)
                ]

            if trials > 0:
                candidate_sites = candidate_sites[:trials]

            # Place molecule on surface
            for si in candidate_sites:
                # Disable binding site
                self._sites[si].is_available = False

                # Create a copy of the molecule
                mol_temp = copy.deepcopy(mol)

                # Check if geminal
                if self._sites[si].is_geminal:
                    mol_temp.add("O", mount, r=0.164, theta=45)
                    mol_temp.add("H", mol_temp.get_num() - 1, r=0.098)
                    mol_temp.set_name(mol.get_name() + "g")
                    mol_temp.set_short(mol.get_short() + "G")

                # Rotate molecule towards surface normal vector
                surf_axis = self._sites[si].normal(self._block.pos(si))
                mol_temp.rotate(
                    geometry.cross_product([0, 0, 1], surf_axis),
                    -geometry.angle([0, 0, 1], surf_axis),
                )

                # Move molecule to mounting position
                mol_temp.move(mount, self._block.pos(si))

                if check_sterics:
                    mol_temp = self._optimize_attachment_pose(
                        mol_temp,
                        mount,
                        surf_axis,
                        {si, *self._sites[si].oxygen_ids},
                        steric_grid,
                        is_rotate,
                        rotate_step_deg,
                        steric_clearance_scale,
                        box=box,
                    )
                    if mol_temp is None:
                        self._sites[si].is_available = True
                        if _site_attempt_callback is not None:
                            _site_attempt_callback(site_id=si, success=False)
                        continue

                # Add molecule to molecule list and global dictionary
                mol_list.append(mol_temp)
                if not mol_temp.get_short() in self._mol_dict[site_type]:
                    self._mol_dict[site_type][mol_temp.get_short()] = []
                self._mol_dict[site_type][mol_temp.get_short()].append(mol_temp)

                scaffold_oxygen_source_ids = self._retained_scaffold_oxygen_ids(
                    si,
                    self._sites[si].oxygen_ids,
                )
                self._attachment_records.append(
                    AttachmentRecord(
                        site_id=si,
                        site_type=site_type,
                        mount_atom_local_id=mount,
                        is_geminal=self._sites[si].is_geminal,
                        scaffold_oxygen_source_ids=scaffold_oxygen_source_ids,
                        surface_oxygen_source_ids=tuple(self._sites[si].oxygen_ids),
                        molecule=mol_temp,
                    )
                )

                # Remove bonds of occupied binding site
                self._invalidate_finalized_export_state()
                if steric_grid is not None:
                    steric_grid.remove_block_atoms([si] + self._sites[si].oxygen_ids)
                    molecule_batch = self._molecule_steric_batch(mol_temp)
                    if molecule_batch.radii.size > 0:
                        steric_grid.add_attached_atoms(
                            molecule_batch.positions,
                            molecule_batch.radii,
                        )
                self._matrix.remove([si] + self._sites[si].oxygen_ids)
                self._invalidate_scaffold_cache()

                # Recursively fill sites in proximity with silanol and geminal silanol
                if is_proxi:
                    proxi_list = [sites[x] for x in si_matrix[sites.index(si)]]
                    if len(proxi_list) > 0:
                        mol_list.extend(
                            self.attach(
                                generic.silanol(),
                                0,
                                [0, 1],
                                proxi_list,
                                len(proxi_list),
                                site_type=site_type,
                                is_proxi=False,
                                is_random=False,
                                check_sterics=False,
                                _steric_grid=steric_grid,
                            )
                        )
                attached_count += 1
                slot_succeeded = True
                if _site_attempt_callback is not None:
                    _site_attempt_callback(site_id=si, success=True)
                break

            if _progress_callback is not None:
                _progress_callback(
                    requested_count=i + 1,
                    attached_count=attached_count,
                    success=slot_succeeded,
                )
        return mol_list

    def siloxane(self, sites, amount, slx_dist=None, trials=1000, site_type="in"):
        """Convert neighboring silanol sites into siloxane bridges.

        Candidate silicon pairs are searched within ``slx_dist`` and, when
        available, converted from two hydroxylated sites into one bridging
        siloxane oxygen placed between them.

        Parameters
        ----------
        sites : list
            Silicon site ids considered for bridge formation.
        amount : int
            Number of bridges to attempt.
        slx_dist : list
            Accepted silicon-silicon distance interval ``[lower, upper]``.
        trials : int, optional
            Number of random pair-selection attempts per bridge.
        site_type : str, optional
            Site family, either interior ``"in"`` or exterior ``"ex"``.

        Returns
        -------
        mol_list : list
            Added siloxane bridge molecules.

        Raises
        ------
        ValueError
            Raised when the requested site type is not supported.
        """
        self._validate_site_type(site_type)
        slx_dist = [0.507-1e-2, 0.507+1e-2] if slx_dist is None else slx_dist

        # Create siloxane molecule
        mol = Molecule("siloxane", "SLX")
        mol.add("O", [0, 0, 0], name="OM1")
        mol.add("O", 0, r=0.09, name="OM1")
        mount = 0
        axis = [0, 1]

        # Rotate molecule towards z-axis
        mol_axis = mol.bond(*axis)
        mol.rotate(geometry.cross_product(mol_axis, [0, 0, 1]), geometry.angle(mol_axis, [0, 0, 1]))
        mol.zero()

        # Search for silicon atoms near each other
        si_dice = Dice(self._site_search_molecule(sites), slx_dist[1], False)
        si_proxi = si_dice.find(None, ["Si", "Si"], slx_dist)
        si_matrix = {x[0]: x[1] for x in si_proxi}

        # Run through number of siloxan bridges to add
        bond_matrix = self._matrix.get_matrix()
        mol_list = []
        for i in range(amount):
            # Randomly pick an available site pair
            si = []
            for j in range(trials):
                si_rand = random.choice(sites)
                # Check if binding site in local si-si matrix and if it contains binding partners
                if sites.index(si_rand) in si_matrix and si_matrix[sites.index(si_rand)]:
                    # Choose first binding partner in list
                    si_rand_proxi = sites[si_matrix[sites.index(si_rand)][0]]
                    # Check if binding partner is in local si-si matrix
                    if sites.index(si_rand_proxi) in si_matrix:
                        # Check if unbound states
                        if self._sites[si_rand].is_available and self._sites[si_rand_proxi].is_available:
                            # Check if binding site silicon atoms are already connected with an oxygen
                            is_connected = False
                            for atom_o in bond_matrix[si_rand]["atoms"]:
                                if atom_o in bond_matrix[si_rand_proxi]["atoms"]:
                                    is_connected = True
                            # Add to siloxane list if not connected
                            if not is_connected:
                                si = [si_rand, si_rand_proxi]
                                # End trials if appended
                                break

            # Place molecule on surface
            if si and self._sites[si[0]].is_available and self._sites[si[1]].is_available:
                # Create a copy of the molecule
                mol_temp = copy.deepcopy(mol)

                # Calculate center position
                pos_vec_halve = [x/2 for x in geometry.vector(self._block.pos(si[0]), self._block.pos(si[1]))]
                center_pos = [pos_vec_halve[x]+self._block.pos(si[0])[x] for x in range(self._dim)]

                # Rotate molecule towards surface normal vector
                surf_axis = self._sites[si[0]].normal(center_pos)
                mol_temp.rotate(geometry.cross_product([0, 0, 1], surf_axis), -geometry.angle([0, 0, 1], surf_axis))

                # Move molecule to mounting position and remove temporary atom
                mol_temp.move(mount, center_pos)
                mol_temp.delete(0)

                # Add molecule to molecule list and global dictionary
                mol_list.append(mol_temp)
                if not mol_temp.get_short() in self._mol_dict[site_type]:
                    self._mol_dict[site_type][mol_temp.get_short()] = []
                self._mol_dict[site_type][mol_temp.get_short()].append(mol_temp)

                # Remove oxygen atom and if not geminal delete site
                self._invalidate_finalized_export_state()
                for si_id in si:
                    self._matrix.remove(self._sites[si_id].oxygen_ids[0])
                    if self._sites[si_id].is_geminal:
                        self._sites[si_id].oxygen_ids.pop(0)
                    else:
                        del self._sites[si_id]
                    del si_matrix[sites.index(si_id)]
                self._invalidate_scaffold_cache()

        return mol_list

    def fill_sites(self, sites, site_type):
        """Fill remaining free sites with silanol or geminal silanol groups.

        Parameters
        ----------
        sites : list
            Silicon site ids to fill.
        site_type : str
            Site family, either interior ``"in"`` or exterior ``"ex"``.

        Returns
        -------
        mol_list : list
            Added silanol molecules. Final silanol saturation does not apply
            steric rejection, so every remaining free handle is consumed.
        """
        mol_list = self.attach(
            generic.silanol(),
            0,
            [0, 1],
            sites,
            len(sites),
            site_type=site_type,
            is_proxi=False,
            is_random=False,
            check_sterics=False,
        )
        
        return mol_list


    ###############
    # Final Edits #
    ###############
    def objectify(self, atoms, allow_attachment_oxygen=False):
        """Convert standalone scaffold atoms into one-atom molecule objects.

        Parameters
        ----------
        atoms : list
            Atom ids to convert.
        allow_attachment_oxygen : bool, optional
            When ``True``, allow retained silica-ligand junction oxygens to be
            exported as ``OM`` atoms in addition to pure framework oxygens.

        Returns
        -------
        mol_list : list
            One-atom molecule objects created from the selected atoms.
        """
        self.validate_scaffold_atoms(atoms, allow_attachment_oxygen=allow_attachment_oxygen)

        # Initialize
        mol_list = []

        # Run through all remaining atoms with a bond or more
        for atom_id in atoms:
            if atom_id in self._objectified_atoms:
                continue

            atom_type = self._block.get_atom_type(atom_id)
            atom_pos = self._block.pos(atom_id)

            # Create molecule object
            if atom_type == "O":
                mol = Molecule("om", "OM")
                mol.add("O", atom_pos, name="OM1", source_id=atom_id)
            elif atom_type == "Si":
                mol = Molecule("si", "SI")
                mol.add("Si", atom_pos, name="SI1", source_id=atom_id)

            # Add to molecule list and global dictionary
            mol_list.append(mol)
            if not mol.get_short() in self._mol_dict["block"]:
                self._mol_dict["block"][mol.get_short()] = []
            self._mol_dict["block"][mol.get_short()].append(mol)
            self._objectified_atoms.add(atom_id)

        # Output
        return mol_list

    def rebuild_final_scaffold_state(self):
        """Rebuild the finalized scaffold snapshot from the live matrix.

        Returns
        -------
        mol_list : list
            Rebuilt one-atom scaffold molecules in the finalized export state.
        """
        self._clear_block_objectification()
        scaffold_atoms = self._final_scaffold_oxygen_ids() + self._final_scaffold_silicon_ids()
        mol_list = self.objectify(scaffold_atoms, allow_attachment_oxygen=True)
        self.refresh_surface_preparation_diagnostics()
        self._is_finalized = True
        return mol_list

    def reservoir(self, size):
        """Translate the silica-system content into a box with solvent reservoirs.

        Parameters
        ----------
        size : float
            Reservoir length added on each side along ``z`` in nanometers.
        """
        # Convert molecule dict into list
        mol_list = sum([x for x in self.get_mol_dict().values()], [])

        # Get zero translation
        min_z = 1000000
        max_z = 0
        for mol in mol_list:
            col_z = mol.column_pos()[2]
            min_z = min(col_z) if min(col_z) < min_z else min_z
            max_z = max(col_z) if max(col_z) > max_z else max_z

        # Translate all molecules
        for mol in mol_list:
            mol.translate([0, 0, -min_z+size])

        # Set new box size
        box = self._block.get_box()
        self.set_box([box[0], box[1], max_z-min_z+2*size])


    ##################
    # Setter Methods #
    ##################
    def set_name(self, name):
        """Set the silica-system name.

        Parameters
        ----------
        name : str
            Silica-system name.
        """
        self._name = name

    def set_box(self, box):
        """Set the silica-system simulation box dimensions.

        Parameters
        ----------
        box : list
            Box lengths in all dimensions.
        """
        self._box = box


    ##################
    # Getter Methods #
    ##################
    def get_name(self):
        """Return the silica-system name.

        Returns
        -------
        name : str
            Silica-system name.
        """
        return self._name

    def get_box(self):
        """Return the silica-system box size.

        Returns
        -------
        box : list
            Box lengths in all dimensions.
        """
        return self._box

    def get_block(self):
        """Return the block molecule.

        Returns
        -------
        block : Molecule
            Underlying block molecule.
        """
        return self._block

    def get_sites(self):
        """Return the binding-site registry.

        Returns
        -------
        sites : dict[int, BindingSite]
            Binding sites keyed by silicon atom id.
        """
        return self._sites

    def get_mol_dict(self):
        """Return all molecules grouped by residue short name.

        Returns
        -------
        mol_dict : dict
            Molecule dictionary merged across interior, exterior, and block
            groups.
        """
        mol_dict = {}
        for site_type in self._mol_dict.keys():
            for key, item in self._mol_dict[site_type].items():
                if not key in mol_dict.keys():
                    mol_dict[key] = []
                mol_dict[key].extend(item)

        return mol_dict

    def get_surface_preparation_diagnostics(self):
        """Return a snapshot of the surface-preparation diagnostics.

        Returns
        -------
        diagnostics : SurfacePreparationDiagnostics
            Surface-cleanup and scaffold-validation counters collected for the
            current silica system.
        """
        return copy.deepcopy(self._surface_preparation_diagnostics)

    def get_surface_edit_history(self):
        """Return the recorded surface-edit provenance entries.

        Returns
        -------
        history : list[SurfaceEditRecord]
            Chronological list of surface-edit records collected during
            preparation and later bridge insertion.
        """
        return list(self._surface_edit_history)

    def get_attachment_records(self):
        """Return metadata for molecules attached to the prepared silica surface.

        Returns
        -------
        records : list[AttachmentRecord]
            Attachment records in placement order.
        """
        return list(self._attachment_records)

    def is_finalized(self):
        """Return whether the current silica system has a rebuilt final scaffold state.

        Returns
        -------
        is_finalized : bool
            True when the block scaffold snapshot matches the finalized live
            matrix state.
        """
        return self._is_finalized

    def get_site_dict(self):
        """Return molecules grouped by site family.

        Returns
        -------
        site_dict : dict
            Molecule dictionary split into ``"block"``, ``"in"``, and ``"ex"``.
        """
        return self._mol_dict

    def get_num_in_ex(self):
        """Return the number of mixed interior/exterior geminal sites.

        Returns
        -------
        num_in_ex : int
            Number of geminal silicon sites spanning both interior and exterior
            oxygen assignments.
        """
        return self._num_in_ex
