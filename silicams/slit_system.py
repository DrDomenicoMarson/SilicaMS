################################################################################
# Periodic silica slit domain model                                            #
################################################################################

"""Attach-ready domain model for periodic amorphous silica slits."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from types import MappingProxyType

from .molecule import Molecule
from .shape import Cuboid, CuboidConfig


@dataclass(frozen=True)
class SurfaceEditRecord:
    """One tracked silica-surface edit.

    Parameters
    ----------
    atom_id : int
        Atom identifier affected by the edit.
    atom_type : str
        Atom type of the affected atom.
    reason : str
        Reason code describing why the atom was removed or inserted.
    neighbor_ids : tuple[int, ...], optional
        Atom identifiers bonded to the edited atom at the time of the event.
    """

    atom_id: int
    atom_type: str
    reason: str
    neighbor_ids: tuple[int, ...] = ()


@dataclass
class SurfacePreparationDiagnostics:
    """Surface-cleanup and scaffold-validation counters.

    Parameters
    ----------
    stripped_undercoordinated_si : int, optional
        Silicon atoms removed after losing original scaffold bonds.
    stripped_excess_surface_oxygen_si : int, optional
        Silicon atoms removed after retaining too many exposed oxygens.
    removed_orphan_oxygen : int, optional
        Zero-bond oxygen atoms removed from the active scaffold.
    removed_invalid_oxygen : int, optional
        Oxygen atoms removed because their connectivity was chemically invalid.
    removed_orphan_silicon : int, optional
        Zero-bond silicon atoms removed from the active scaffold.
    inserted_bridge_oxygen : int, optional
        Bridge oxygens inserted during siloxane editing.
    final_surface_oxygen_handles : int, optional
        Free one-coordinate surface oxygen handles in the current scaffold.
    final_framework_oxygen : int, optional
        Oxygen atoms exported as framework ``OM`` atoms.
    """

    stripped_undercoordinated_si: int = 0
    stripped_excess_surface_oxygen_si: int = 0
    removed_orphan_oxygen: int = 0
    removed_invalid_oxygen: int = 0
    removed_orphan_silicon: int = 0
    inserted_bridge_oxygen: int = 0
    final_surface_oxygen_handles: int = 0
    final_framework_oxygen: int = 0

    @property
    def stripped_silicon_total(self):
        """Return the total number of stripped silicon atoms."""
        return (
            self.stripped_undercoordinated_si
            + self.stripped_excess_surface_oxygen_si
            + self.removed_orphan_silicon
        )


from ._silica_engine import _SilicaChemistryEngine


@dataclass(frozen=True)
class SlitBindingSite:
    """Read-only snapshot of one silica-slit binding site.

    Parameters
    ----------
    site_id : int
        Silicon source identifier.
    oxygen_ids : tuple[int, ...]
        Exposed oxygen source identifiers attached to the silicon site.
    position_nm : tuple[float, float, float]
        Silicon position in nanometers.
    site_type : str
        Surface family. Periodic slits use ``"in"``.
    is_available : bool
        Whether the site remains available for attachment.
    is_geminal : bool
        Whether the site currently exposes two oxygen handles.
    """

    site_id: int
    oxygen_ids: tuple[int, ...]
    position_nm: tuple[float, float, float]
    site_type: str
    is_available: bool
    is_geminal: bool

    @property
    def oxygen_count(self):
        """Return the number of exposed oxygen handles."""
        return len(self.oxygen_ids)


@dataclass(frozen=True)
class LigandAttachmentResult:
    """Outcome of one explicit slit-ligand attachment batch.

    Parameters
    ----------
    requested_site_ids : tuple[int, ...]
        Ordered surface-site candidates supplied by the caller.
    attached_site_ids : tuple[int, ...]
        Sites consumed by successful ligand placements.
    rejected_site_ids : tuple[int, ...]
        Candidate sites that did not produce a ligand placement.
    molecules : tuple[Molecule, ...]
        Attached molecule instances in placement order.
    """

    requested_site_ids: tuple[int, ...]
    attached_site_ids: tuple[int, ...]
    rejected_site_ids: tuple[int, ...]
    molecules: tuple[Molecule, ...]


@dataclass(frozen=True)
class SlitAttachmentRecord:
    """Read-only snapshot of one graft attachment.

    Parameters
    ----------
    site_id : int
        Consumed scaffold-silicon source identifier.
    site_type : str
        Surface family. Periodic slits use ``"in"``.
    mount_atom_local_id : int
        Zero-based ligand mount-atom index.
    is_geminal : bool
        Whether the attachment consumed a geminal site.
    scaffold_oxygen_source_ids : tuple[int, ...]
        Retained scaffold-oxygen source identifiers at the junction.
    surface_oxygen_source_ids : tuple[int, ...]
        Removed surface-handle oxygen source identifiers.
    molecule_short : str
        Residue short name of the attached ligand instance.
    """

    site_id: int
    site_type: str
    mount_atom_local_id: int
    is_geminal: bool
    scaffold_oxygen_source_ids: tuple[int, ...]
    surface_oxygen_source_ids: tuple[int, ...]
    molecule_short: str


class SilicaSlit:
    """Mutable, attach-ready periodic amorphous silica slit.

    The class owns the low-level silica chemistry engine and exposes controlled
    read/query operations plus explicit surface edits. Callers never need the
    underlying block molecule or connectivity matrix.

    Parameters
    ----------
    surface : _SilicaChemistryEngine
        Prepared low-level silica surface engine.
    interior_site_ids : iterable[int]
        Initial periodic slit surface-site identifiers.
    centroid_nm : sequence[float]
        Slit centroid in nanometers.
    slit_width_nm : float
        Requested geometric slit width in nanometers.
    original_box_nm : sequence[float]
        Periodic box dimensions before finalization.
    """

    def __init__(
        self,
        surface,
        interior_site_ids,
        centroid_nm,
        slit_width_nm,
        original_box_nm,
    ):
        """Initialize a slit around its private silica chemistry engine.

        Parameters
        ----------
        surface : _SilicaChemistryEngine
            Prepared low-level silica surface engine.
        interior_site_ids : iterable[int]
            Initial periodic slit surface-site identifiers.
        centroid_nm : sequence[float]
            Slit centroid in nanometers.
        slit_width_nm : float
            Requested geometric slit width in nanometers.
        original_box_nm : sequence[float]
            Periodic box dimensions before finalization.

        Raises
        ------
        TypeError
            Raised when ``surface`` is not a prepared internal silica engine.
        """
        if not isinstance(surface, _SilicaChemistryEngine):
            raise TypeError("surface must be a prepared internal silica engine.")
        self.__surface = surface
        self.__interior_site_ids = list(interior_site_ids)
        self.__centroid_nm = tuple(float(value) for value in centroid_nm)
        self.__slit_width_nm = float(slit_width_nm)
        self.__original_box_nm = tuple(float(value) for value in original_box_nm)
        self.__sort_order = ["OM", "SI"]
        self.__total_surface_si = len(self.__interior_site_ids)
        self.__surface_positions_nm = tuple(
            tuple(self.__surface.atom_position(site_id))
            for site_id in self.__interior_site_ids
        )

    @classmethod
    def _from_block(cls, block, connectivity, slit_width_nm, name):
        """Carve and prepare a periodic slit for the public builder.

        Parameters
        ----------
        block : Molecule
            Replicated silica template.
        connectivity : Matrix
            Si-O connectivity for ``block``.
        slit_width_nm : float
            Requested slit width in nanometers.
        name : str
            System name used for reports and exports.

        Returns
        -------
        slit : SilicaSlit
            Prepared attach-ready slit before Q-state editing.

        Raises
        ------
        ValueError
            Raised when the prepared periodic slit contains exterior sites.
        """
        box = tuple(float(value) for value in block.get_box())
        centroid = tuple(float(value) for value in block.centroid())
        shape = Cuboid(
            CuboidConfig(
                centroid=centroid,
                central=(0, 0, 1),
                length=box[2],
                width=box[0],
                height=slit_width_nm - 0.5,
            )
        )
        deleted_atom_ids = [
            atom_id
            for atom_id, position in enumerate(block.positions_view())
            if shape.is_in(position.tolist())
        ]
        connectivity.strip(deleted_atom_ids)

        surface = _SilicaChemistryEngine(block, connectivity)
        surface.set_name(name)
        surface.set_box(list(box))
        surface.prepare()
        surface.sites()

        sites = surface.get_sites()
        exterior_site_ids = [
            site_id for site_id, site in sites.items() if site.site_type == "ex"
        ]
        if exterior_site_ids:
            raise ValueError("The periodic slit preparation requires zero exterior sites.")

        interior_site_ids = sorted(
            site_id for site_id, site in sites.items() if site.site_type == "in"
        )
        for site_id in interior_site_ids:
            sites[site_id].normal = shape.normal

        non_grid = set(surface.surface_handle_oxygen_ids()) | set(sites)
        grid_atom_ids = [
            atom_id
            for atom_id in connectivity.bound(0, "gt")
            if atom_id not in non_grid
        ]
        surface.validate_scaffold_atoms(grid_atom_ids)
        surface.objectify(grid_atom_ids)
        surface.refresh_surface_preparation_diagnostics()

        return cls(
            surface=surface,
            interior_site_ids=interior_site_ids,
            centroid_nm=centroid,
            slit_width_nm=slit_width_nm,
            original_box_nm=box,
        )

    @property
    def name(self):
        """Return the slit name."""
        return self.__surface.get_name()

    @property
    def box_nm(self):
        """Return periodic box dimensions in nanometers."""
        return tuple(float(value) for value in self.__surface.get_box())

    @property
    def centroid_nm(self):
        """Return the original slit centroid in nanometers."""
        return self.__centroid_nm

    @property
    def slit_width_nm(self):
        """Return the requested geometric slit width in nanometers."""
        return self.__slit_width_nm

    @property
    def is_finalized(self):
        """Return whether free surface sites have been saturated."""
        return self.__surface.is_finalized()

    @property
    def binding_sites(self):
        """Return read-only binding-site snapshots keyed by silicon id."""
        snapshots = {
            site_id: self._site_snapshot(site_id, site)
            for site_id, site in self.__surface.get_sites().items()
        }
        return MappingProxyType(snapshots)

    @property
    def interior_site_ids(self):
        """Return currently available interior surface-site identifiers."""
        return tuple(self.__interior_site_ids)

    @property
    def molecule_counts(self):
        """Return finalized/current molecule counts by exported short name."""
        return MappingProxyType(
            {
                short_name: len(molecules)
                for short_name, molecules in self.__surface.get_mol_dict().items()
            }
        )

    @property
    def attachment_records(self):
        """Return read-only graft attachment snapshots in placement order."""
        return tuple(
            SlitAttachmentRecord(
                site_id=record.site_id,
                site_type=record.site_type,
                mount_atom_local_id=record.mount_atom_local_id,
                is_geminal=record.is_geminal,
                scaffold_oxygen_source_ids=tuple(
                    record.scaffold_oxygen_source_ids
                ),
                surface_oxygen_source_ids=tuple(record.surface_oxygen_source_ids),
                molecule_short=record.molecule.get_short(),
            )
            for record in self.__surface.get_attachment_records()
        )

    @property
    def surface_edit_history(self):
        """Return surface-edit provenance entries in chronological order."""
        return tuple(self.__surface.get_surface_edit_history())

    @property
    def preparation_diagnostics(self):
        """Return a copy of surface-preparation diagnostics."""
        return self.__surface.get_surface_preparation_diagnostics()

    @property
    def residue_order(self):
        """Return residue short names in deterministic export order."""
        available = self.__surface.get_mol_dict()
        return tuple(short for short in self.__sort_order if short in available)

    def _site_snapshot(self, site_id, site):
        """Build one immutable public binding-site snapshot."""
        return SlitBindingSite(
            site_id=site_id,
            oxygen_ids=tuple(site.oxygen_ids),
            position_nm=tuple(self.__surface.atom_position(site_id)),
            site_type=site.site_type,
            is_available=bool(site.is_available),
            is_geminal=site.is_geminal,
        )

    def clone(self):
        """Return an independent deep copy of the current slit."""
        return copy.deepcopy(self)

    def active_silicon_count(self):
        """Return the number of active scaffold silicon atoms."""
        return sum(
            self.__surface.atom_type(atom_id) == "Si"
            for atom_id in self.__surface.active_atom_ids()
        )

    def active_atom_ids(self):
        """Return active scaffold atom identifiers."""
        return self.__surface.active_atom_ids()

    def atom_neighbors(self, atom_id):
        """Return active neighbors for one scaffold atom.

        Parameters
        ----------
        atom_id : int
            Scaffold source identifier.
        """
        return self.__surface.atom_neighbors(atom_id)

    def atom_position(self, atom_id):
        """Return the position of one scaffold atom.

        Parameters
        ----------
        atom_id : int
            Scaffold source identifier.
        """
        return self.__surface.atom_position(atom_id)

    def atom_positions(self, atom_ids):
        """Return positions for selected scaffold atoms.

        Parameters
        ----------
        atom_ids : iterable[int]
            Scaffold source identifiers.
        """
        return self.__surface.atom_positions(atom_ids)

    def atom_type(self, atom_id):
        """Return the atom type for one scaffold source identifier.

        Parameters
        ----------
        atom_id : int
            Scaffold source identifier.
        """
        return self.__surface.atom_type(atom_id)

    def site_normal(self, site_id, position_nm):
        """Return the slit-facing normal for one binding site.

        Parameters
        ----------
        site_id : int
            Surface silicon identifier.
        position_nm : sequence[float]
            Position at which to evaluate the stored normal function.
        """
        site = self.__surface.get_sites()[site_id]
        return list(site.normal(list(position_nm)))

    def available_site_ids(self, oxygen_count=None):
        """Return sorted available interior sites, optionally by OH count.

        Parameters
        ----------
        oxygen_count : int or None, optional
            Required number of oxygen handles. ``None`` accepts either
            single or geminal sites.
        """
        return tuple(
            sorted(
                site_id
                for site_id, site in self.__surface.get_sites().items()
                if site.site_type == "in"
                and site.is_available
                and (oxygen_count is None or site.oxygen_count == oxygen_count)
            )
        )

    def insert_siloxane_bridge(self, site_pair, position_nm):
        """Condense two surface sites using a preselected bridge position.

        Parameters
        ----------
        site_pair : tuple[int, int]
            Silicon surface-site identifiers.
        position_nm : sequence[float]
            New bridge-oxygen position in nanometers.
        """
        if self.is_finalized:
            raise ValueError("A finalized slit cannot be edited.")
        bridge_atom_id = self.__surface.insert_siloxane_bridge(
            tuple(site_pair),
            list(position_nm),
        )
        self.refresh_site_tracking()
        return bridge_atom_id

    def attach_ligands(
        self,
        molecule,
        mount,
        axis,
        site_ids,
        requested_count=None,
        allow_geminal=True,
        rotate_about_axis=True,
        rotate_step_deg=10.0,
        check_sterics=True,
        steric_clearance_scale=0.60,
        progress_callback=None,
    ):
        """Attach a ligand to an ordered batch of candidate slit sites.

        Parameters
        ----------
        molecule : Molecule
            Post-condensation ligand fragment.
        mount : int
            Local ligand atom placed at the consumed surface silicon.
        axis : sequence[int]
            Two local atom identifiers defining the ligand orientation axis.
        site_ids : iterable[int]
            Ordered candidate surface-site identifiers.
        requested_count : int or None, optional
            Number of attachments requested. Defaults to ``len(site_ids)``.
        allow_geminal : bool, optional
            Whether sites with two oxygen handles may be consumed.
        rotate_about_axis : bool, optional
            Whether to scan rotations around the attachment axis.
        rotate_step_deg : float, optional
            Angular step used by the rotational steric search.
        check_sterics : bool, optional
            Whether to reject sterically clashing poses.
        steric_clearance_scale : float, optional
            Scale applied to covalent-radius steric cutoffs.
        progress_callback : callable or None, optional
            Callback invoked after every requested attachment slot.

        Returns
        -------
        result : LigandAttachmentResult
            Successful/rejected site identifiers and attached molecules.
        """
        if self.is_finalized:
            raise ValueError("A finalized slit cannot accept new ligands.")
        if not isinstance(molecule, Molecule):
            raise TypeError("molecule must be a Molecule instance.")

        candidate_ids = tuple(int(site_id) for site_id in site_ids)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("site_ids must not contain duplicates.")
        sites = self.__surface.get_sites()
        invalid_ids = [
            site_id
            for site_id in candidate_ids
            if site_id not in sites
            or sites[site_id].site_type != "in"
            or not sites[site_id].is_available
        ]
        if invalid_ids:
            raise ValueError(f"Unavailable or invalid slit site ids: {invalid_ids}.")

        amount = len(candidate_ids) if requested_count is None else int(requested_count)
        if amount < 0 or amount > len(candidate_ids):
            raise ValueError("requested_count must be between zero and len(site_ids).")

        record_start = len(self.__surface.get_attachment_records())
        attempted_sites = []

        def record_site_attempt(site_id, success):
            """Record one internal placement attempt.

            Parameters
            ----------
            site_id : int
                Candidate surface-site identifier.
            success : bool
                Whether the candidate accepted the placement.
            """
            attempted_sites.append((int(site_id), bool(success)))

        molecules = self.__surface.attach(
            copy.deepcopy(molecule),
            int(mount),
            list(axis),
            list(candidate_ids),
            amount,
            pos_list=[],
            site_type="in",
            is_proxi=False,
            is_random=False,
            is_rotate=rotate_about_axis,
            rotate_step_deg=rotate_step_deg,
            is_g=allow_geminal,
            check_sterics=check_sterics,
            steric_clearance_scale=steric_clearance_scale,
            _progress_callback=progress_callback,
            _site_attempt_callback=record_site_attempt,
        )
        new_records = self.__surface.get_attachment_records()[record_start:]
        attached_site_ids = tuple(record.site_id for record in new_records)
        rejected_site_ids = tuple(
            site_id
            for site_id in dict.fromkeys(
                site_id for site_id, success in attempted_sites if not success
            )
            if site_id not in attached_site_ids
        )
        for attached_molecule in molecules:
            short_name = attached_molecule.get_short()
            if short_name not in self.__sort_order:
                self.__sort_order.append(short_name)
        self.refresh_site_tracking()
        return LigandAttachmentResult(
            requested_site_ids=candidate_ids,
            attached_site_ids=attached_site_ids,
            rejected_site_ids=rejected_site_ids,
            molecules=tuple(molecules),
        )

    def refresh_site_tracking(self):
        """Refresh available-site identifiers and preparation diagnostics."""
        self.__surface.refresh_surface_preparation_diagnostics()
        self.__interior_site_ids = list(self.available_site_ids())

    def attached_molecule_counts(self):
        """Return non-silanol interior molecule counts by short name."""
        return {
            short_name: len(molecules)
            for short_name, molecules in self.__surface.get_site_dict()["in"].items()
            if short_name not in {"SL", "SLG", "SLX"}
        }

    def attached_state_counts(self, base_short):
        """Return attached T2/T3 counts for one ligand family.

        Parameters
        ----------
        base_short : str
            Base T3 residue short name. The T2 geminal variant is expected to
            use the same name with a trailing ``G``.
        """
        site_dict = self.__surface.get_site_dict()["in"]
        return (
            len(site_dict.get(base_short + "G", [])),
            len(site_dict.get(base_short, [])),
        )

    def finalize(self):
        """Saturate remaining sites and rebuild the final scaffold snapshot."""
        if self.is_finalized:
            return
        molecules = self.__surface.fill_sites(list(self.available_site_ids()), "in")
        for molecule in molecules:
            short_name = molecule.get_short()
            if short_name not in self.__sort_order:
                self.__sort_order.append(short_name)
        self.__surface.rebuild_final_scaffold_state()
        self.__surface.set_box(list(self.__original_box_nm))
        self.refresh_site_tracking()

    def _export_components(self):
        """Return copied data required to build an immutable export snapshot.

        Returns
        -------
        components : dict
            Molecules, residue order, scaffold bonds, attachments, provenance,
            name, box, and finalization state. The returned collections do not
            expose mutable connectivity or binding-site objects.
        """
        molecule_dict = self.__surface.get_mol_dict()
        molecules = [
            molecule
            for short_name in self.residue_order
            for molecule in molecule_dict[short_name]
        ]
        molecule_index = {id(molecule): index for index, molecule in enumerate(molecules)}
        bridge_atom_ids = {
            record.atom_id
            for record in self.surface_edit_history
            if record.reason == "inserted_bridge_oxygen"
        }
        scaffold_bonds = []
        for atom_a, atom_b in self.__surface.connectivity_bonds():
            provenance = (
                "siloxane_bridge"
                if atom_a in bridge_atom_ids or atom_b in bridge_atom_ids
                else "scaffold"
            )
            scaffold_bonds.append((atom_a, atom_b, provenance))
        attachments = []
        for record in self.__surface.get_attachment_records():
            if id(record.molecule) not in molecule_index:
                continue
            attachments.append(
                (
                    molecule_index[id(record.molecule)],
                    record.site_id,
                    record.site_type,
                    record.mount_atom_local_id,
                    record.is_geminal,
                    tuple(record.scaffold_oxygen_source_ids),
                    tuple(record.surface_oxygen_source_ids),
                )
            )
        return {
            "name": self.name,
            "box_nm": self.box_nm,
            "molecules": tuple(molecules),
            "residue_order": self.residue_order,
            "scaffold_bonds": tuple(scaffold_bonds),
            "attachments": tuple(attachments),
            "is_finalized": self.is_finalized,
            "source_kind": "silica_slit",
        }

    def export_snapshot(self):
        """Return an immutable writer snapshot of the finalized slit.

        Returns
        -------
        snapshot : StructureSnapshot
            Shared structure/topology export input.

        Raises
        ------
        ValueError
            Raised when the slit has not been finalized.
        """
        if not self.is_finalized:
            raise ValueError("Slit export requires finalize() first.")
        from .writers.common import StructureSnapshot
        from .writers.structure import StructureWriter

        snapshot = StructureSnapshot.from_components(**self._export_components())
        return StructureWriter(snapshot).complete_snapshot()

    def metadata(self):
        """Return YAML-ready geometric metadata for the periodic slit."""
        radii = [
            abs(position[1] - self.__centroid_nm[1])
            for position in self.__surface_positions_nm
        ]
        radius_mean = sum(radii) / len(radii) if radii else 0.0
        effective_width = 2 * radius_mean
        roughness = (
            math.sqrt(sum((radius - radius_mean) ** 2 for radius in radii) / len(radii))
            if radii
            else 0.0
        )
        length = self.__original_box_nm[2]
        width = self.__original_box_nm[0]
        volume = width * effective_width * length
        surface = 2 * (
            length * width + length * effective_width + width * effective_width
        )
        return {
            "shape_00": {
                "diameter": effective_width,
                "parameter": {
                    "central": [0, 0, 1],
                    "centroid": list(self.__centroid_nm),
                    "height": self.__slit_width_nm,
                    "length": length,
                    "width": width,
                },
                "roughness": roughness,
                "shape": "SLIT",
                "surface": surface,
                "volume": volume,
            },
            "system": {
                "centroid": list(self.__centroid_nm),
                "dimensions": list(self.box_nm),
                "reservoir": 0,
                "surface": {"in": surface, "ex": 0.0},
                "volume": volume,
            },
        }
