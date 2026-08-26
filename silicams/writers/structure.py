################################################################################
# Structure writer                                                             #
################################################################################

"""Coordinate, object, graph, and connectivity-validation writer."""

import os
import warnings

import silicams.database as db
import silicams.utils as utils
from silicams.connectivity import (
    AssembledStructureGraph,
    ConnectivityValidationFinding,
    ConnectivityValidationReport,
    GraphBond,
)
from silicams.molecule import Molecule

from .common import (
    StructureSnapshot,
    _CIF_LABEL_SEQ_ID,
    _CifEntityRecord,
    _CifStructAsymRecord,
    _ExportMoleculeCount,
    _HYBRID36_DIGITS_UPPER,
    _PDB_CHAIN_ID,
    _PdbResidueAliasRecord,
    _ResidueExportIdentifiers,
    _StructureAtomRecord,
    _StructureExportCache,
    _cif_token,
    _encode_hybrid36,
    _encode_pure,
    _hybrid36_max_value,
    _normalize_pdb_identifier,
    _sanitize_pdb_token,
)


class StructureWriter:
    """Write coordinates and connectivity for molecules or structure snapshots.

    Parameters
    ----------
    inp : Molecule or StructureSnapshot
        Molecule or immutable export snapshot to serialize.
    link : str, optional
        Output directory for generated files.

    Examples
    --------
    Structure files can be generated from a molecule or from a finalized slit
    snapshot.

    .. code-block:: python

        StructureWriter(mol).write_pdb()
        StructureWriter(snapshot, "output").write_gro("slit.gro")
    """
    def __init__(self, inp, link="./"):
        """Initialize a structure writer from a molecule or export snapshot.

        Parameters
        ----------
        inp : Molecule or StructureSnapshot
            Molecule or immutable domain export snapshot to serialize.
        link : str or os.PathLike, optional
            Output directory for generated files.

        Raises
        ------
        TypeError
            Raised when the input is not supported.
        """
        self._dim = 3
        self._link = os.fspath(link)
        self._link = self._link if self._link.endswith("/") else self._link + "/"
        if isinstance(inp, Molecule):
            snapshot = StructureSnapshot.from_molecule(inp)
        elif isinstance(inp, StructureSnapshot):
            snapshot = inp
        else:
            raise TypeError(
                "StructureWriter input must be a Molecule or StructureSnapshot."
            )
        self._snapshot = snapshot
        self._inp = snapshot
        self._mols = list(snapshot.molecules)
        self._short_list = list(snapshot.residue_order)
        self._name = snapshot.name or "molecule"
        self._box = list(snapshot.box_nm)
        utils.mkdirp(self._link)
        self._structure_export_cache = {}

    def complete_snapshot(self):
        """Return the shared snapshot with final numbering and connectivity.

        Returns
        -------
        snapshot : StructureSnapshot
            Immutable snapshot populated with atom/residue order, molecule
            membership, assembled bonds and angles, graft-junction serials,
            and residue counts.
        """
        if self._snapshot.has_assembled_export:
            return self._snapshot
        cache = self._collect_structure_records(use_atom_names=True)
        graph = self.assembled_graph(use_atom_names=True)
        if self._snapshot.source_kind == "silica_slit":
            residue_counts = tuple(
                (record.residue_name, record.count)
                for record in self._export_molecule_counts()
            )
        else:
            counts = {}
            for molecule in self._mols:
                residue_name = self._export_residue_name(molecule.get_short())
                counts[residue_name] = counts.get(residue_name, 0) + 1
            residue_counts = tuple(counts.items())
        return self._snapshot.with_assembled_export(
            atom_records=cache.atom_records,
            molecule_serials=cache.molecule_serials,
            graph=graph,
            residue_counts=residue_counts,
        )

    #############
    # Structure #
    #############
    def write_object(self, name=""):
        """Serialize the wrapped object with pickle.

        Parameters
        ----------
        name : str, optional
            Output filename. Defaults to ``<name>.obj``.
        """
        # Initialize
        link = self._link
        link += name if name else self._name+".obj"

        # Save one independent snapshot with final writer numbering.
        utils.save(self.complete_snapshot(), link)

    def _export_residue_name(self, residue_name):
        """Return the normalized residue name written by export writers.

        Parameters
        ----------
        residue_name : str
            Internal residue identifier stored on molecules or slit molecule
            dictionaries.

        Returns
        -------
        export_name : str
            Residue identifier written to structure and helper topology
            outputs. Internal siloxane bridge oxygens normalize to ``"OM"``
            while every other residue keeps its native name.
        """
        return "OM" if residue_name == "SLX" else residue_name

    def _export_molecule_counts(self):
        """Return slit helper-topology counts in exported residue space.

        Returns
        -------
        exported_counts : list[_ExportMoleculeCount]
            Exported residue counts in first-appearance order after applying
            residue-name normalization.

        Raises
        ------
        TypeError
            Raised when helper-topology counting is requested for a non-slit
            input.
        """
        if self._snapshot.source_kind != "silica_slit":
            raise TypeError(
                "Helper-topology residue counts require a silica-slit snapshot."
            )

        if self._snapshot.has_assembled_export:
            return [
                _ExportMoleculeCount(residue_name=name, count=count)
                for name, count in self._snapshot.residue_counts
            ]

        counts_by_name = {}
        ordered_names = []
        for residue_name in self._short_list:
            export_name = self._export_residue_name(residue_name)
            if export_name not in counts_by_name:
                counts_by_name[export_name] = 0
                ordered_names.append(export_name)
            counts_by_name[export_name] += sum(
                molecule.get_short() == residue_name
                for molecule in self._mols
            )

        return [
            _ExportMoleculeCount(residue_name=name, count=counts_by_name[name])
            for name in ordered_names
        ]

    def _residue_names_in_order(self):
        """Return exported residue identifiers in first-appearance order.

        Returns
        -------
        residue_names : list[str]
            Unique residue identifiers encountered while iterating over the
            serialized molecule list in writer order after applying export
            normalization such as ``SLX -> OM``.
        """
        residue_names = []
        seen = set()
        for mol in self._mols:
            residue_name = self._export_residue_name(mol.get_short())
            if residue_name in seen:
                continue
            residue_names.append(residue_name)
            seen.add(residue_name)
        return residue_names

    def _pdb_residue_alias_records(self, residue_names):
        """Return deterministic PDB-safe aliases for residue identifiers.

        Parameters
        ----------
        residue_names : list[str]
            Unique residue identifiers in first-appearance order.

        Returns
        -------
        alias_records : list[_PdbResidueAliasRecord]
            Full-name to three-character alias mappings.

        Raises
        ------
        ValueError
            Raised when more than ``36**2`` colliding long-name aliases would
            be required for the same initial letter.
        """
        alias_records = []
        used_aliases = set()

        for residue_name in residue_names:
            normalized = _normalize_pdb_identifier(residue_name, fallback="UNK")
            alias = None

            if len(normalized) <= 3 and normalized not in used_aliases:
                alias = normalized
            else:
                for candidate in (
                    (normalized[:3] if len(normalized) >= 3 else (normalized + "XXX")[:3]),
                    (normalized[:2] + normalized[-1]) if len(normalized) >= 3 else None,
                    (normalized[0] + normalized[-2:]) if len(normalized) >= 3 else None,
                ):
                    if candidate is None:
                        continue
                    candidate = candidate[:3]
                    if candidate not in used_aliases:
                        alias = candidate
                        break

            if alias is None:
                prefix = normalized[0]
                for suffix_value in range(36 ** 2):
                    candidate = prefix + _encode_pure(
                        _HYBRID36_DIGITS_UPPER,
                        suffix_value,
                        2,
                    )
                    if candidate not in used_aliases:
                        alias = candidate
                        break
                if alias is None:
                    raise ValueError(
                        "Could not assign a unique 3-character PDB residue "
                        f"alias for residue {residue_name!r}."
                    )

            used_aliases.add(alias)
            alias_records.append(
                _PdbResidueAliasRecord(
                    full_name=residue_name,
                    pdb_name=alias,
                )
            )

        return alias_records

    def _pdb_alias_remark_lines(self, residue_alias_records):
        """Return PDB remark lines describing residue-name aliasing.

        Parameters
        ----------
        residue_alias_records : list[_PdbResidueAliasRecord]
            Full-name to PDB-alias mappings for the current export.

        Returns
        -------
        lines : list[str]
            ``REMARK`` lines describing aliases that differ from the original
            residue name.
        """
        lines = []
        for record in residue_alias_records:
            if record.full_name == record.pdb_name:
                continue
            lines.append(
                f"REMARK 250 RESIDUE_ALIAS {record.full_name} -> {record.pdb_name}\n"
            )
        return lines

    def _pdb_atom_name_field(self, atom_name, atom_type):
        """Return one fixed-width PDB atom-name field.

        Parameters
        ----------
        atom_name : str
            Atom name to serialize.
        atom_type : str
            Atom type used to infer the element alignment rule.

        Returns
        -------
        field : str
            Exactly four characters suitable for columns 13-16 of a PDB
            ``ATOM`` or ``HETATM`` record.
        """
        sanitized = _sanitize_pdb_token(atom_name, width=4, fallback="X")
        try:
            element = db.get_element(atom_type)
        except ValueError:
            element = atom_type

        if len(sanitized) == 4:
            return sanitized
        if sanitized[0].isdigit() or len(element) == 2:
            return f"{sanitized:<4s}"
        return f"{sanitized:>4s}"

    def _pdb_cryst1_record(self):
        """Return the CRYST1 record for the current periodic box.

        Returns
        -------
        line : str
            PDB ``CRYST1`` record or an empty string when no periodic box is
            available.
        """
        if not any(self._box):
            return ""

        return (
            "CRYST1"
            f"{self._box[0] * 10:9.3f}"
            f"{self._box[1] * 10:9.3f}"
            f"{self._box[2] * 10:9.3f}"
            f"{90.0:7.2f}"
            f"{90.0:7.2f}"
            f"{90.0:7.2f}"
            " P 1           1\n"
        )

    def _collect_structure_records(self, use_atom_names=False):
        """Collect structure-writer metadata in the current writer order.

        Parameters
        ----------
        use_atom_names : bool, optional
            True to preserve explicit atom names when available. False to
            enumerate atom names from atom types.

        Returns
        -------
        cache : _StructureExportCache
            Structure-export metadata, alias tables, and mmCIF support tables
            in writer order. Residue names are normalized to the exported
            chemistry view, so internal siloxane bridge ``SLX`` residues are
            serialized as ``OM``.
        """
        residue_names = self._residue_names_in_order()
        residue_alias_records = self._pdb_residue_alias_records(residue_names)
        pdb_alias_by_name = {
            record.full_name: record.pdb_name
            for record in residue_alias_records
        }
        entity_records = [
            _CifEntityRecord(entity_id=str(entity_index), residue_name=residue_name)
            for entity_index, residue_name in enumerate(residue_names, start=1)
        ]
        entity_id_by_name = {
            record.residue_name: record.entity_id
            for record in entity_records
        }

        atom_records = []
        molecule_serials = []
        struct_asym_records = []
        atom_serial = 1
        residue_serial = 1

        for molecule_index, mol in enumerate(self._mols):
            atom_types = {}
            molecule_serial = []
            residue_marker = None
            residue_identifiers = None

            for local_atom_index, atom in enumerate(mol.get_atom_list()):
                if residue_identifiers is None or atom.get_residue() != residue_marker:
                    if residue_identifiers is not None:
                        residue_serial += 1
                    residue_marker = atom.get_residue()
                    residue_name = self._export_residue_name(mol.get_short())
                    residue_identifiers = _ResidueExportIdentifiers(
                        residue_id=residue_serial,
                        residue_name=residue_name,
                        pdb_residue_name=pdb_alias_by_name[residue_name],
                        pdb_chain_id=_PDB_CHAIN_ID,
                        pdb_residue_id_token=_encode_hybrid36(4, residue_serial),
                        cif_entity_id=entity_id_by_name[residue_name],
                        cif_asym_id=f"A{residue_serial}",
                        cif_label_seq_id=_CIF_LABEL_SEQ_ID,
                    )
                    struct_asym_records.append(
                        _CifStructAsymRecord(
                            asym_id=residue_identifiers.cif_asym_id,
                            entity_id=residue_identifiers.cif_entity_id,
                            residue_id=residue_identifiers.residue_id,
                            residue_name=residue_identifiers.residue_name,
                        )
                    )

                atom_type = atom.get_atom_type()
                if atom_type not in atom_types:
                    atom_types[atom_type] = 1

                if use_atom_names and atom.get_name():
                    atom_name = atom.get_name()
                else:
                    atom_name = atom_type + str(atom_types[atom_type])

                atom_records.append(
                    _StructureAtomRecord(
                        serial=atom_serial,
                        pdb_serial_token=_encode_hybrid36(5, atom_serial),
                        molecule_index=molecule_index,
                        local_atom_index=local_atom_index,
                        residue_name=residue_identifiers.residue_name,
                        pdb_residue_name=residue_identifiers.pdb_residue_name,
                        residue_id=residue_identifiers.residue_id,
                        pdb_residue_id_token=residue_identifiers.pdb_residue_id_token,
                        atom_name=atom_name,
                        atom_type=atom_type,
                        position=tuple(atom.get_pos()),
                        source_id=atom.get_source_id(),
                        pdb_chain_id=residue_identifiers.pdb_chain_id,
                        cif_entity_id=residue_identifiers.cif_entity_id,
                        cif_asym_id=residue_identifiers.cif_asym_id,
                        cif_label_seq_id=residue_identifiers.cif_label_seq_id,
                    )
                )
                molecule_serial.append(atom_serial)

                atom_serial += 1
                atom_types[atom_type] = atom_types[atom_type] + 1 if atom_types[atom_type] < 99 else 1

            molecule_serials.append(molecule_serial)
            if residue_identifiers is not None:
                residue_serial += 1

        return _StructureExportCache(
            atom_records=atom_records,
            molecule_serials=molecule_serials,
            residue_alias_records=residue_alias_records,
            entity_records=entity_records,
            struct_asym_records=struct_asym_records,
        )

    def _export_cache(self, use_atom_names=False):
        """Return cached structure-export data for one atom-name mode.

        Parameters
        ----------
        use_atom_names : bool, optional
            True to preserve explicit atom names when available. False to
            enumerate atom names from atom types.

        Returns
        -------
        cache : _StructureExportCache
            Cached structure records and lazily populated graph/validation
            data for the requested atom-name mode.
        """
        cache = self._structure_export_cache.get(use_atom_names)
        if cache is not None:
            return cache

        cache = self._collect_structure_records(use_atom_names)
        self._structure_export_cache[use_atom_names] = cache
        return cache

    def _export_graph(self, use_atom_names=False):
        """Return the cached assembled bond graph for one atom-name mode.

        Parameters
        ----------
        use_atom_names : bool, optional
            True to preserve explicit atom names when available. False to
            enumerate atom names from atom types.

        Returns
        -------
        graph : AssembledStructureGraph
            Assembled graph matching the current structure-writer order.
        """
        cache = self._export_cache(use_atom_names)
        if cache.graph is None:
            cache.graph = self._assembled_structure_graph(
                cache.atom_records,
                cache.molecule_serials,
            )
        return cache.graph

    def _validation_report(self, use_atom_names=False):
        """Return the cached connectivity-validation report.

        Parameters
        ----------
        use_atom_names : bool, optional
            True to preserve explicit atom names when available. False to
            enumerate atom names from atom types.

        Returns
        -------
        report : ConnectivityValidationReport
            Cached validation report for the assembled structure.
        """
        cache = self._export_cache(use_atom_names)
        if cache.validation_report is None:
            graph = self._export_graph(use_atom_names)
            cache.validation_report = ConnectivityValidationReport(
                atom_count=len(cache.atom_records),
                bond_count=len(graph.bonds),
                findings=self._connectivity_validation_findings(cache.atom_records, graph),
            )
        return cache.validation_report

    def _validate_pdb_hybrid36_limits(self, atom_records):
        """Validate that PDB identifiers fit inside hybrid-36 fields.

        Parameters
        ----------
        atom_records : list[_StructureAtomRecord]
            Serialized atom metadata in file order.

        Raises
        ------
        ValueError
            Raised when any atom serial or residue identifier exceeds the
            hybrid-36 representable range of the corresponding PDB field.
        """
        if not atom_records:
            return

        max_atom_serial = max(record.serial for record in atom_records)
        max_residue_serial = max(record.residue_id for record in atom_records)

        if max_atom_serial > _hybrid36_max_value(5):
            raise ValueError(
                "PDB atom serials exceed the hybrid-36 range "
                f"(max atom serial={max_atom_serial}, max {_hybrid36_max_value(5)})."
            )
        if max_residue_serial > _hybrid36_max_value(4):
            raise ValueError(
                "PDB residue ids exceed the hybrid-36 range "
                f"(max residue id={max_residue_serial}, max {_hybrid36_max_value(4)})."
            )

    def _write_cif_entity_loop(self, file_out, entity_records):
        """Write the mmCIF entity loop for the current export.

        Parameters
        ----------
        file_out : TextIO
            Open output stream.
        entity_records : list[_CifEntityRecord]
            Entity rows declared for the current export.
        """
        if not entity_records:
            return

        file_out.write("loop_\n")
        file_out.write("_entity.id\n")
        file_out.write("_entity.type\n")
        file_out.write("_entity.pdbx_description\n")
        for record in entity_records:
            file_out.write(
                " ".join(
                    (
                        _cif_token(record.entity_id),
                        _cif_token(record.entity_type),
                        _cif_token(record.residue_name),
                    )
                )
                + "\n"
            )
        file_out.write("#\n")

    def _write_cif_struct_asym_loop(self, file_out, struct_asym_records):
        """Write the mmCIF structural-asymmetry loop for the export.

        Parameters
        ----------
        file_out : TextIO
            Open output stream.
        struct_asym_records : list[_CifStructAsymRecord]
            Asymmetry rows declared for the current export.
        """
        if not struct_asym_records:
            return

        file_out.write("loop_\n")
        file_out.write("_struct_asym.id\n")
        file_out.write("_struct_asym.entity_id\n")
        for record in struct_asym_records:
            file_out.write(
                f"{_cif_token(record.asym_id)} {_cif_token(record.entity_id)}\n"
            )
        file_out.write("#\n")

    def _assembled_structure_graph(self, atom_records, molecule_serials):
        """Build a bonded graph for the serialized structure.

        Parameters
        ----------
        atom_records : list[_StructureAtomRecord]
            All atom records written to the current structure file.
        molecule_serials : list[list[int]]
            Atom serial numbers grouped per written molecule.

        Returns
        -------
        graph : AssembledStructureGraph
            Assembled bond graph in writer atom order.
        """
        if self._snapshot.has_assembled_export:
            atom_ids = tuple(record.serial for record in atom_records)
            snapshot_atom_ids = tuple(
                atom.serial for atom in self._snapshot.atom_order
            )
            if atom_ids != snapshot_atom_ids:
                raise ValueError(
                    "Structure snapshot atom order does not match writer order."
                )
            return self._snapshot.assembled_graph

        bonds = []

        source_serials = {
            record.source_id: record.serial
            for record in atom_records
            if record.source_id is not None
        }
        for source_bond in self._snapshot.scaffold_bonds:
            if (
                source_bond.source_atom_a in source_serials
                and source_bond.source_atom_b in source_serials
            ):
                bonds.append(
                    GraphBond(
                        source_serials[source_bond.source_atom_a],
                        source_serials[source_bond.source_atom_b],
                        source_bond.provenance,
                    )
                )

        for molecule_index, serials in enumerate(molecule_serials):
            mol = self._mols[molecule_index]
            explicit_bonds = mol.get_bonds()
            for atom_a, atom_b in explicit_bonds:
                bonds.append(
                    GraphBond(serials[atom_a], serials[atom_b], "ligand_explicit")
                )
            if not explicit_bonds:
                for atom_a, atom_b in mol.infer_bonds():
                    bonds.append(
                        GraphBond(serials[atom_a], serials[atom_b], "ligand_inferred")
                    )

        if self._snapshot.source_kind == "silica_slit":
            for record in self._snapshot.attachments:
                if record.molecule_index >= len(molecule_serials):
                    continue
                serials = molecule_serials[record.molecule_index]
                mount_serial = serials[record.mount_atom_local_id]
                for oxygen_source_id in record.scaffold_oxygen_source_ids:
                    if oxygen_source_id in source_serials:
                        bonds.append(
                            GraphBond(
                                mount_serial,
                                source_serials[oxygen_source_id],
                                "graft_junction",
                            )
                        )

        return AssembledStructureGraph.from_bonds(
            (record.serial for record in atom_records),
            bonds,
        )

    def assembled_graph(self, use_atom_names=False):
        """Return the assembled bonded graph in structure-writer atom order.

        Parameters
        ----------
        use_atom_names : bool, optional
            True to preserve explicit atom names when available while matching
            the graph to the same atom order used by structure writers.

        Returns
        -------
        graph : AssembledStructureGraph
            Bond graph and derived angle hooks for the current object.
        """
        return self._export_graph(use_atom_names)

    def _write_pdb_conect_records(self, file_out, bond_pairs):
        """Write ``CONECT`` records for previously collected bond pairs.

        Parameters
        ----------
        file_out : TextIO
            Open output stream.
        bond_pairs : list[tuple[int, int]]
            Sorted unique atom-serial pairs.
        """
        neighbors = {}
        for serial_a, serial_b in bond_pairs:
            neighbors.setdefault(serial_a, []).append(serial_b)
            neighbors.setdefault(serial_b, []).append(serial_a)

        for serial in sorted(neighbors):
            bonded = sorted(set(neighbors[serial]))
            for start in range(0, len(bonded), 4):
                chunk = bonded[start:start + 4]
                file_out.write(
                    "CONECT"
                    + _encode_hybrid36(5, serial)
                    + "".join(_encode_hybrid36(5, neighbor) for neighbor in chunk)
                    + "\n"
                )

    def _write_cif_struct_conn_loop(self, file_out, atom_records, graph):
        """Write an mmCIF bond loop for the assembled bonded structure.

        Parameters
        ----------
        file_out : TextIO
            Open output stream.
        atom_records : list[_StructureAtomRecord]
            Serialized atom metadata in file order.
        graph : AssembledStructureGraph
            Assembled graph matching ``atom_records``.
        """
        if not graph.bonds:
            return

        record_by_serial = {record.serial: record for record in atom_records}
        file_out.write("#\n")
        file_out.write("loop_\n")
        file_out.write("_struct_conn.id\n")
        file_out.write("_struct_conn.conn_type_id\n")
        file_out.write("_struct_conn.ptnr1_label_asym_id\n")
        file_out.write("_struct_conn.ptnr1_label_comp_id\n")
        file_out.write("_struct_conn.ptnr1_label_seq_id\n")
        file_out.write("_struct_conn.ptnr1_label_atom_id\n")
        file_out.write("_struct_conn.ptnr2_label_asym_id\n")
        file_out.write("_struct_conn.ptnr2_label_comp_id\n")
        file_out.write("_struct_conn.ptnr2_label_seq_id\n")
        file_out.write("_struct_conn.ptnr2_label_atom_id\n")

        for conn_index, bond in enumerate(graph.bonds, start=1):
            record_a = record_by_serial[bond.atom_a]
            record_b = record_by_serial[bond.atom_b]
            file_out.write(
                " ".join(
                    (
                        _cif_token(f"conn{conn_index}"),
                        "covale",
                        _cif_token(record_a.cif_asym_id),
                        _cif_token(record_a.residue_name),
                        _cif_token(record_a.cif_label_seq_id),
                        _cif_token(record_a.atom_name),
                        _cif_token(record_b.cif_asym_id),
                        _cif_token(record_b.residue_name),
                        _cif_token(record_b.cif_label_seq_id),
                        _cif_token(record_b.atom_name),
                    )
                )
                + "\n"
            )

    def _connectivity_validation_neighbors(self, graph):
        """Build an adjacency map for one assembled graph.

        Parameters
        ----------
        graph : AssembledStructureGraph
            Graph whose neighbor lists should be assembled.

        Returns
        -------
        neighbors : dict[int, set[int]]
            One-based neighbor ids grouped by central atom id.
        """
        neighbors = {atom_id: set() for atom_id in graph.atom_ids}
        for bond in graph.bonds:
            if bond.atom_a in neighbors:
                neighbors[bond.atom_a].add(bond.atom_b)
            if bond.atom_b in neighbors:
                neighbors[bond.atom_b].add(bond.atom_a)
        return neighbors

    def _connectivity_validation_findings(self, atom_records, graph):
        """Collect connectivity-validation findings for one assembled graph.

        Parameters
        ----------
        atom_records : list[_StructureAtomRecord]
            Serialized atom metadata in writer order.
        graph : AssembledStructureGraph
            Assembled bond graph matching ``atom_records``.

        Returns
        -------
        findings : tuple[ConnectivityValidationFinding, ...]
            Sorted validation findings for the assembled structure. Generic
            element-degree checks are applied first, followed by stricter
            residue-aware silica checks on assembled slit exports. The
            validation runs on exported residue identities, so internal
            siloxane bridge ``SLX`` residues are treated as exported
            bridging ``OM`` oxygens.
        """
        record_by_serial = {record.serial: record for record in atom_records}
        neighbors = self._connectivity_validation_neighbors(graph)
        findings = {}
        is_finalized_pore = (
            self._snapshot.source_kind != "silica_slit"
            or self._snapshot.is_finalized
        )

        allowed_degrees = {
            "H": {1},
            "B": {3, 4},
            "C": {1, 2, 3, 4},
            "N": {1, 2, 3, 4},
            "O": {1, 2},
            "F": {1},
            "P": {1, 2, 3, 4, 5},
            "S": {1, 2, 3, 4, 5, 6},
            "Cl": {1},
            "Br": {1},
            "I": {1},
            "Si": {1, 2, 3, 4},
        }

        def add_finding(code, message, atom_ids, is_error=True):
            atom_ids = tuple(atom_ids)
            key = (code, atom_ids)
            if key in findings:
                return
            findings[key] = ConnectivityValidationFinding(
                code=code,
                message=message,
                atom_ids=atom_ids,
                atom_types=tuple(record_by_serial[atom_id].atom_type for atom_id in atom_ids if atom_id in record_by_serial),
                residue_shorts=tuple(record_by_serial[atom_id].residue_name for atom_id in atom_ids if atom_id in record_by_serial),
                degrees=tuple(len(neighbors.get(atom_id, ())) for atom_id in atom_ids),
                is_error=is_error,
            )

        resolved_elements = {}
        for atom_id, record in record_by_serial.items():
            try:
                resolved_elements[atom_id] = db.get_element(record.atom_type)
            except ValueError:
                resolved_elements[atom_id] = None

        def neighbor_ids_for(atom_id):
            return tuple(sorted(neighbors.get(atom_id, ())))

        def element_token_for(atom_id):
            element = resolved_elements.get(atom_id)
            if element is not None:
                return element
            if atom_id in record_by_serial:
                return record_by_serial[atom_id].atom_type
            return "?"

        def neighbor_elements_for(atom_id):
            return [element_token_for(neighbor_id) for neighbor_id in neighbor_ids_for(atom_id)]

        def silica_oxygen_environment(atom_id):
            if resolved_elements.get(atom_id) != "O":
                return "other"
            neighbor_elements = sorted(neighbor_elements_for(atom_id))
            if len(neighbor_elements) != 2:
                return "other"
            if neighbor_elements == ["H", "Si"]:
                return "hydroxyl"
            if neighbor_elements == ["Si", "Si"]:
                return "bridge"
            return "other"

        def silica_silicon_neighbor_summary(atom_id):
            oxygen_neighbor_ids = [
                neighbor_id
                for neighbor_id in neighbor_ids_for(atom_id)
                if resolved_elements.get(neighbor_id) == "O"
            ]
            hydroxyl_oxygen_ids = [
                neighbor_id
                for neighbor_id in oxygen_neighbor_ids
                if silica_oxygen_environment(neighbor_id) == "hydroxyl"
            ]
            bridge_oxygen_ids = [
                neighbor_id
                for neighbor_id in oxygen_neighbor_ids
                if silica_oxygen_environment(neighbor_id) == "bridge"
            ]
            return oxygen_neighbor_ids, hydroxyl_oxygen_ids, bridge_oxygen_ids

        for atom_id in graph.atom_ids:
            if atom_id not in record_by_serial:
                continue
            record = record_by_serial[atom_id]
            degree = len(neighbors.get(atom_id, ()))
            is_pre_finalized_scaffold = (
                self._snapshot.source_kind == "silica_slit"
                and not is_finalized_pore
                and record.residue_name in {"OM", "SI"}
            )
            element = resolved_elements.get(atom_id)
            if element is None:
                add_finding(
                    "unknown_element",
                    f"Atom {atom_id} has unsupported atom type '{record.atom_type}' for connectivity validation.",
                    (atom_id,),
                )
                continue

            if (
                not is_pre_finalized_scaffold
                and element in allowed_degrees
                and degree not in allowed_degrees[element]
            ):
                add_finding(
                    "unexpected_degree",
                    f"Atom {atom_id} ({record.atom_type}) has degree {degree}, which is outside the supported range for {element}.",
                    (atom_id,),
                )

            neighbor_ids = neighbor_ids_for(atom_id)
            neighbor_elements = neighbor_elements_for(atom_id)

            if element == "H":
                if degree != 1:
                    add_finding(
                        "hydrogen_degree",
                        f"Hydrogen atom {atom_id} must have degree 1, found {degree}.",
                        (atom_id,),
                    )
                elif neighbor_elements and neighbor_elements[0] == "H":
                    add_finding(
                        "hydrogen_neighbor",
                        f"Hydrogen atom {atom_id} is bonded to another hydrogen atom.",
                        (atom_id, next(iter(sorted(neighbors[atom_id])))),
                    )

            if record.residue_name == "SL":
                if element == "Si":
                    oxygen_neighbor_ids, hydroxyl_oxygen_ids, bridge_oxygen_ids = (
                        silica_silicon_neighbor_summary(atom_id)
                    )
                    if (
                        degree != 4
                        or len(oxygen_neighbor_ids) != 4
                        or len(hydroxyl_oxygen_ids) != 1
                        or len(bridge_oxygen_ids) != 3
                    ):
                        add_finding(
                            "silanol_silicon_environment",
                            f"Silanol silicon atom {atom_id} must be bonded to exactly one hydroxyl oxygen and three bridging oxygen atoms.",
                            (atom_id, *neighbor_ids),
                        )
                elif element == "O":
                    if silica_oxygen_environment(atom_id) != "hydroxyl":
                        add_finding(
                            "silanol_oxygen_environment",
                            f"Silanol oxygen atom {atom_id} must be bonded to exactly one silicon atom and one hydrogen atom.",
                            (atom_id, *neighbor_ids),
                        )
                elif element == "H":
                    if degree != 1 or neighbor_elements != ["O"]:
                        add_finding(
                            "silanol_hydrogen_environment",
                            f"Silanol hydrogen atom {atom_id} must be bonded to exactly one oxygen atom.",
                            (atom_id, *neighbor_ids),
                        )

            if record.residue_name == "SLG":
                if element == "Si":
                    oxygen_neighbor_ids, hydroxyl_oxygen_ids, bridge_oxygen_ids = (
                        silica_silicon_neighbor_summary(atom_id)
                    )
                    if (
                        degree != 4
                        or len(oxygen_neighbor_ids) != 4
                        or len(hydroxyl_oxygen_ids) != 2
                        or len(bridge_oxygen_ids) != 2
                    ):
                        add_finding(
                            "geminal_silicon_environment",
                            f"Geminal silanol silicon atom {atom_id} must be bonded to exactly two hydroxyl oxygen and two bridging oxygen atoms.",
                            (atom_id, *neighbor_ids),
                        )
                elif element == "O":
                    if silica_oxygen_environment(atom_id) != "hydroxyl":
                        add_finding(
                            "geminal_oxygen_environment",
                            f"Geminal silanol oxygen atom {atom_id} must be bonded to exactly one silicon atom and one hydrogen atom.",
                            (atom_id, *neighbor_ids),
                        )
                elif element == "H":
                    if degree != 1 or neighbor_elements != ["O"]:
                        add_finding(
                            "geminal_hydrogen_environment",
                            f"Geminal silanol hydrogen atom {atom_id} must be bonded to exactly one oxygen atom.",
                            (atom_id, *neighbor_ids),
                        )

            if is_pre_finalized_scaffold:
                continue

            if record.residue_name in {"OM", "SLX"}:
                if silica_oxygen_environment(atom_id) != "bridge":
                    add_finding(
                        "framework_oxygen_environment",
                        f"Bridging silica oxygen atom {atom_id} must be bonded to exactly two silicon atoms after export.",
                        (atom_id, *neighbor_ids),
                    )

            if record.residue_name == "SI":
                oxygen_neighbor_ids, hydroxyl_oxygen_ids, bridge_oxygen_ids = (
                    silica_silicon_neighbor_summary(atom_id)
                )
                if (
                    degree != 4
                    or len(oxygen_neighbor_ids) != 4
                    or len(hydroxyl_oxygen_ids) != 0
                    or len(bridge_oxygen_ids) != 4
                ):
                    add_finding(
                        "framework_silicon_environment",
                        f"Framework silicon atom {atom_id} must be bonded to exactly four bridging oxygen atoms after export.",
                        (atom_id, *neighbor_ids),
                    )

        for bond in graph.bonds:
            if bond.atom_a not in record_by_serial or bond.atom_b not in record_by_serial:
                continue
            try:
                element_a = db.get_element(record_by_serial[bond.atom_a].atom_type)
                element_b = db.get_element(record_by_serial[bond.atom_b].atom_type)
            except ValueError:
                continue

            if bond.provenance in {"scaffold", "siloxane_bridge", "graft_junction"}:
                if sorted((element_a, element_b)) != ["O", "Si"]:
                    add_finding(
                        "invalid_silica_bond",
                        f"{bond.provenance.replace('_', ' ')} bond {bond.atom_a}-{bond.atom_b} must connect silicon and oxygen atoms.",
                        (bond.atom_a, bond.atom_b),
                    )

        return tuple(sorted(findings.values(), key=lambda finding: (finding.code, finding.atom_ids)))

    def validate_connectivity(self, use_atom_names=False):
        """Validate the assembled bond graph against chemistry rules.

        Parameters
        ----------
        use_atom_names : bool, optional
            True to preserve explicit atom names while matching the validation
            atom order to the same order used by structure writers.

        Returns
        -------
        report : ConnectivityValidationReport
            Structured validation report for the current assembled structure.
            Generic element-degree checks are always applied, and silica
            residues are additionally checked against stricter residue-aware
            local-environment rules on the exported chemistry view. Internal
            siloxane bridge ``SLX`` residues therefore validate as exported
            bridging ``OM`` oxygens. Full scaffold ``OM/SI`` checks are
            applied only to finalized slit states.
        """
        return self._validation_report(use_atom_names)

    def _handle_connectivity_validation(self, use_atom_names, validate_connectivity):
        """Validate one structure before writing and handle the chosen mode.

        Parameters
        ----------
        use_atom_names : bool
            Forwarded atom-name handling mode used for structure serialization.
        validate_connectivity : str
            Validation mode: ``"off"``, ``"warn"``, or ``"strict"``.

        Returns
        -------
        report : ConnectivityValidationReport or None
            Validation report when the mode is not ``"off"``, otherwise
            ``None``.

        Raises
        ------
        ValueError
            Raised when ``validate_connectivity`` is unsupported or when the
            mode is ``"strict"`` and the validation report contains errors.
        """
        if validate_connectivity not in {"off", "warn", "strict"}:
            raise ValueError(
                "Unsupported connectivity validation mode. "
                "Expected one of: ['off', 'strict', 'warn']."
            )

        if validate_connectivity == "off":
            return None

        report = self.validate_connectivity(use_atom_names=use_atom_names)
        if report.is_valid:
            return report

        summary = (
            "Connectivity validation found "
            f"{report.error_count} error(s) and {report.warning_count} warning(s)."
        )
        preview = "; ".join(
            finding.message for finding in report.findings[:3]
        )
        message = summary + (" " + preview if preview else "")

        if validate_connectivity == "strict":
            raise ValueError(message)

        warnings.warn(message, UserWarning, stacklevel=3)
        return report

    def write_cif(
        self,
        name="",
        use_atom_names=False,
        write_bonds=True,
        validate_connectivity="warn",
    ):
        """Write the current structure in mmCIF format.

        Parameters
        ----------
        name : str, optional
            Output filename. Defaults to ``<name>.cif``.
        use_atom_names : bool, optional
            True to preserve explicit atom names when available. False to
            enumerate atom names from atom types.
        write_bonds : bool, optional
            When ``True`` (the default), also emit an ``_struct_conn`` loop
            for the full assembled bond graph, including silica scaffold,
            siloxane bridges, ligand-internal bonds, and graft junctions.
            Full residue names are preserved in mmCIF, and the writer emits
            matching ``_entity`` and ``_struct_asym`` loops so that
            ``_atom_site.label_entity_id`` and ``_atom_site.label_asym_id``
            reference declared rows.
        validate_connectivity : str, optional
            Connectivity validation mode: ``"off"``, ``"warn"``, or
            ``"strict"``. The default warns on invalid assembled local
            chemistry before writing the file. Internal siloxane bridge
            residues are serialized as exported ``OM`` bridging oxygens.
        """
        self._handle_connectivity_validation(use_atom_names, validate_connectivity)
        link = self._link
        link += name if name else self._name + ".cif"
        cache = self._export_cache(use_atom_names)
        atom_records = cache.atom_records
        entity_records = cache.entity_records
        struct_asym_records = cache.struct_asym_records
        graph = self._export_graph(use_atom_names)
        data_name = self._name.replace(" ", "_")

        with open(link, "w") as file_out:
            file_out.write(f"data_{data_name}\n")
            file_out.write("#\n")
            file_out.write("_symmetry.space_group_name_H-M 'P 1'\n")
            file_out.write("_symmetry.Int_Tables_number 1\n")
            file_out.write(f"_cell.length_a {self._box[0] * 10:.3f}\n")
            file_out.write(f"_cell.length_b {self._box[1] * 10:.3f}\n")
            file_out.write(f"_cell.length_c {self._box[2] * 10:.3f}\n")
            file_out.write("_cell.angle_alpha 90.000\n")
            file_out.write("_cell.angle_beta 90.000\n")
            file_out.write("_cell.angle_gamma 90.000\n")
            file_out.write("#\n")
            self._write_cif_entity_loop(file_out, entity_records)
            self._write_cif_struct_asym_loop(file_out, struct_asym_records)
            file_out.write("loop_\n")
            file_out.write("_atom_site.group_PDB\n")
            file_out.write("_atom_site.id\n")
            file_out.write("_atom_site.type_symbol\n")
            file_out.write("_atom_site.label_atom_id\n")
            file_out.write("_atom_site.label_alt_id\n")
            file_out.write("_atom_site.label_comp_id\n")
            file_out.write("_atom_site.label_asym_id\n")
            file_out.write("_atom_site.label_entity_id\n")
            file_out.write("_atom_site.label_seq_id\n")
            file_out.write("_atom_site.pdbx_PDB_ins_code\n")
            file_out.write("_atom_site.Cartn_x\n")
            file_out.write("_atom_site.Cartn_y\n")
            file_out.write("_atom_site.Cartn_z\n")
            file_out.write("_atom_site.occupancy\n")
            file_out.write("_atom_site.B_iso_or_equiv\n")
            file_out.write("_atom_site.pdbx_formal_charge\n")
            file_out.write("_atom_site.auth_seq_id\n")
            file_out.write("_atom_site.auth_comp_id\n")
            file_out.write("_atom_site.auth_asym_id\n")
            file_out.write("_atom_site.auth_atom_id\n")
            file_out.write("_atom_site.pdbx_PDB_model_num\n")

            for record in atom_records:
                file_out.write(
                    " ".join(
                        [
                            "HETATM",
                            _cif_token(record.serial),
                            _cif_token(db.get_element(record.atom_type)),
                            _cif_token(record.atom_name),
                            ".",
                            _cif_token(record.residue_name),
                            _cif_token(record.cif_asym_id),
                            _cif_token(record.cif_entity_id),
                            _cif_token(record.cif_label_seq_id),
                            "?",
                            _cif_token(f"{record.position[0] * 10:.3f}"),
                            _cif_token(f"{record.position[1] * 10:.3f}"),
                            _cif_token(f"{record.position[2] * 10:.3f}"),
                            "1.00",
                            "0.00",
                            "?",
                            _cif_token(record.residue_id),
                            _cif_token(record.residue_name),
                            _cif_token(record.pdb_chain_id),
                            _cif_token(record.atom_name),
                            "1",
                        ]
                    )
                    + "\n"
                )

            if write_bonds:
                self._write_cif_struct_conn_loop(file_out, atom_records, graph)

            file_out.write("#\n")

    def write_pdb(
        self,
        name="",
        use_atom_names=False,
        write_conect=True,
        validate_connectivity="warn",
    ):
        """Write the current structure in PDB format.

        Parameters
        ----------
        name : str, optional
            Output filename. Defaults to ``<name>.pdb``.
        use_atom_names : bool, optional
            True to preserve explicit atom names when available. False to
            enumerate atom names from atom types.
        write_conect : bool, optional
            When ``True`` (the default), also emit inspection-oriented
            ``CONECT`` records for the full assembled bond graph, including
            silica scaffold, siloxane bridges, ligand-internal bonds, and
            graft junctions. Residue names longer than three characters are
            written as deterministic three-character aliases, atom serials and
            residue ids automatically switch to hybrid-36 on overflow, and a
            ``CRYST1`` record is emitted when periodic box lengths are known.
        validate_connectivity : str, optional
            Connectivity validation mode: ``"off"``, ``"warn"``, or
            ``"strict"``. The default warns on invalid assembled local
            chemistry before writing the file. Internal siloxane bridge
            residues are serialized as exported ``OM`` bridging oxygens.
        """
        self._handle_connectivity_validation(use_atom_names, validate_connectivity)
        # Initialize
        link = self._link
        link += name if name else self._name+".pdb"
        cache = self._export_cache(use_atom_names)
        atom_records = cache.atom_records
        residue_alias_records = cache.residue_alias_records
        graph = self._export_graph(use_atom_names)
        self._validate_pdb_hybrid36_limits(atom_records)

        # Open file
        with open(link, "w") as file_out:
            for line in self._pdb_alias_remark_lines(residue_alias_records):
                file_out.write(line)

            cryst1_record = self._pdb_cryst1_record()
            if cryst1_record:
                file_out.write(cryst1_record)

            for record in atom_records:
                element = _sanitize_pdb_token(
                    db.get_element(record.atom_type),
                    width=2,
                    fallback="X",
                )
                out_string = "HETATM"                       #  1- 6 (6)    Record name
                out_string += record.pdb_serial_token      #  7-11 (5)    Atom serial number
                out_string += " "                          # 12    (1)    -
                out_string += self._pdb_atom_name_field(
                    record.atom_name,
                    record.atom_type,
                )                                          # 13-16 (4)    Atom name
                out_string += " "                          # 17    (1)    Alternate location indicator
                out_string += f"{record.pdb_residue_name:>3s}"# 18-20 (3)  Residue name
                out_string += " "                          # 21    (1)    -
                out_string += record.pdb_chain_id          # 22    (1)    Chain identifier
                out_string += record.pdb_residue_id_token  # 23-26 (4)    Residue sequence number
                out_string += " "                          # 27    (1)    Code for insertion of residues
                out_string += "   "                        # 28-30 (3)    -
                for coord in record.position:              # 31-54 (3*8)  Coordinates
                    out_string += f"{coord*10:8.3f}"
                out_string += f"{1:6.2f}"                  # 55-60 (6)    Occupancy
                out_string += f"{0:6.2f}"                  # 61-66 (6)    Temperature factor
                out_string += "          "                 # 67-76 (10)   -
                out_string += f"{element:>2s}"             # 77-78 (2)    Element symbol
                out_string += "  "                         # 79-80 (2)    Charge on the atom

                file_out.write(out_string+"\n")

            if write_conect:
                self._write_pdb_conect_records(
                    file_out,
                    [(bond.atom_a, bond.atom_b) for bond in graph.bonds],
                )

            # End statement
            file_out.write("TER\nEND\n")

    def write_gro(self, name="", use_atom_names=False, validate_connectivity="warn"):
        """Write the current structure in GROMACS GRO format.

        Parameters
        ----------
        name : str, optional
            Output filename. Defaults to ``<name>.gro``.
        use_atom_names : bool, optional
            True to preserve explicit atom names when available. False to
            enumerate atom names from atom types.
        validate_connectivity : str, optional
            Connectivity validation mode: ``"off"``, ``"warn"``, or
            ``"strict"``. The default warns on invalid assembled local
            chemistry before writing the file. Internal siloxane bridge
            residues are serialized as exported ``OM`` bridging oxygens.
        """
        self._handle_connectivity_validation(use_atom_names, validate_connectivity)
        # Initialize
        link = self._link
        link += name if name else self._name+".gro"

        # Open file
        with open(link, "w") as file_out:
            # Set title
            file_out.write("Structure generated by SilicaMS\n")

            # Number of atoms
            file_out.write("%i" % sum([x.get_num() for x in self._mols])+"\n")

            # Set counter
            num_a = 1
            num_m = 1

            # Run through molecules
            for mol in self._mols:
                atom_types = {}
                temp_res_id = 0
                residue_name = self._export_residue_name(mol.get_short())
                # Run through atoms
                for atom in mol.get_atom_list():
                    # Process residue index
                    if not atom.get_residue() == temp_res_id:
                        num_m = num_m+1 if num_m < 99999 else 0
                        temp_res_id = atom.get_residue()

                    # Get atom type
                    atom_type = atom.get_atom_type()

                    # Create type dictionary
                    if atom_type not in atom_types:
                        atom_types[atom_type] = 1

                    # Set atom name
                    if use_atom_names and atom.get_name():
                        atom_name = atom.get_name()
                    else:
                        atom_name = atom_type+str(atom_types[atom_type])

                    # Write file
                    out_string = "%5i" % num_m              #  1- 5 (5)    Residue number
                    out_string += "%-5s" % residue_name     #  6-10 (5)    Residue short name
                    out_string += "%5s" % atom_name         # 11-15 (5)    Atom name
                    out_string += "%5i" % num_a             # 16-20 (5)    Atom number
                    for i in range(self._dim):                    # 21-44 (3*8)  Coordinates
                        out_string += "%8.3f" % atom.get_pos()[i]

                    file_out.write(out_string+"\n")

                    # Process counter
                    num_a = num_a+1 if num_a < 99999 else 0
                    atom_types[atom_type] = atom_types[atom_type]+1 if atom_types[atom_type] < 999 else 0
                num_m = num_m+1 if num_m < 99999 else 0

            # Box
            out_string = ""
            for i in range(self._dim):
                out_string += "%.3f" % self._box[i]
                out_string += " " if i < self._dim-1 else "\n"

            file_out.write(out_string)

    def write_xyz(self, name="", use_atom_names=False):
        """Write the current structure in XYZ format.

        Parameters
        ----------
        name : str, optional
            Output filename. Defaults to ``<name>.xyz``.
        use_atom_names : bool, optional
            Retained for API compatibility. XYZ output always uses atom types as
            element labels.
        """
        # Initialize
        link = self._link
        link += name if name else self._name+".xyz"

        # Open output file and set title
        with open(link, "w") as file_out:
            # Header
            file_out.write("%i" % sum([x.get_num() for x in self._mols])+"\n"+"Energy = \n")

            # Run through molecules
            for mol in self._mols:
                # Run through atoms
                for atom in mol.get_atom_list():
                    # Write file
                    out_string = "%-2s" % atom.get_atom_type()  # 1- 2 (2)     Atom name
                    for i in range(self._dim):                  # 3-41 (3*13)  Coordinates
                        out_string += "%13.7f" % (atom.get_pos()[i]*10)

                    file_out.write(out_string+"\n")

    def write_lammps(self, name=""):
        """Write the current structure in LAMMPS data format.

        Coordinates are written in Angstrom assuming ``real`` units.

        Parameters
        ----------
        name : str, optional
            Output filename. Defaults to ``<name>.lmp``.
        """
        # Initialize
        link = self._link
        link += name if name else self._name+".lmp"

        # Atom types
        atom_types = list(set(sum([[x.get_atom_type(i) for i in range(x.get_num())] for x in self._mols], [])))

        # Open file
        with open(link, "w") as file_out:
            # Set title
            file_out.write("Structure generated by SilicaMS\n\n")

            # Porperties section
            file_out.write("%i" % sum([x.get_num() for x in self._mols])+" atoms\n")
            file_out.write("%i" % len(atom_types)+" atom types\n")
            file_out.write("\n")

            # Box size - Periodic boundary conditions
            file_out.write("0.000 "+"%.3f" % (self._box[0]*10)+" xlo xhi\n")
            file_out.write("0.000 "+"%.3f" % (self._box[1]*10)+" ylo yhi\n")
            file_out.write("0.000 "+"%.3f" % (self._box[2]*10)+" zlo zhi\n")
            file_out.write("\n")

            # Masses
            file_out.write("Masses\n\n")
            for i, at in enumerate(atom_types):
                file_out.write("%i"%(i+1)+" "+"%8.3f"%db.get_mass(at)+"\n")
            file_out.write("\n")

            # Atoms
            file_out.write("Atoms\n\n")

            # Set counter
            num_a = 1
            num_m = 1

            # Run through molecules
            for mol in self._mols:
                temp_res_id = 0
                # Run through atoms
                for atom in mol.get_atom_list():
                    # Process residue index
                    if not atom.get_residue() == temp_res_id:
                        num_m = num_m+1
                        temp_res_id = atom.get_residue()

                    # Get atom type
                    atom_type_id = atom_types.index(atom.get_atom_type())+1

                    # Write atom line
                    out_string  = "%5i" % num_a + " "        #  Atom number
                    out_string += "%5i" % num_m + " "        #  Residue number
                    out_string += "%3i" % atom_type_id + " " #  Atom type
                    out_string += "%5i" % 0 + " "            #  Charge
                    for i in range(self._dim):               #  Coordinates
                        out_string += "%8.3f" % (atom.get_pos()[i]*10)
                        out_string += " " if i<self._dim-1 else ""
                    file_out.write(out_string+"\n")

                    # Process counter
                    num_a = num_a+1
                num_m = num_m+1


    ############
    # Topology #
    ############
