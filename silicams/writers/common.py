################################################################################
# Shared writer models                                                                  #
#                                                                              #
"""Shared immutable models and formatting helpers for output writers."""
################################################################################


import copy
from dataclasses import dataclass, replace

import silicams.database as db
from silicams.connectivity import (
    AssembledStructureGraph,
    ConnectivityValidationReport,
    GraphAngle,
    GraphBond,
)
from silicams.molecule import Molecule


_HYBRID36_DIGITS_UPPER = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_HYBRID36_DIGITS_LOWER = "0123456789abcdefghijklmnopqrstuvwxyz"
_PDB_CHAIN_ID = "A"
_CIF_LABEL_SEQ_ID = "1"
_FULL_SLIT_NREXCL = 3
_FUNCTIONALIZED_CHARGE_TOLERANCE = 1e-6


def _full_slit_mount_ligand_cross_angle_role(
    record_a,
    record_b,
    record_c,
    *,
    ligand_shorts,
    mount_atom_name,
    first_ligand_atom_name,
    is_generated_geminal_record,
):
    """Classify one mount-silicon cross-angle environment for slit export.

    Parameters
    ----------
    record_a : object
        First outer atom record. The record must expose ``residue_name``,
        ``atom_name``, and ``atom_type`` attributes.
    record_b : object
        Center atom record. The record must expose ``residue_name``,
        ``atom_name``, and ``atom_type`` attributes.
    record_c : object
        Second outer atom record. The record must expose ``residue_name``,
        ``atom_name``, and ``atom_type`` attributes.
    ligand_shorts : set[str]
        Allowed ligand residue short names, including geminal variants.
    mount_atom_name : str or None
        Atom name of the ligand mount silicon.
    first_ligand_atom_name : str
        Atom name of the first ligand atom directly bound to the mount
        silicon.
    is_generated_geminal_record : callable
        Predicate returning ``True`` for internally generated geminal oxygen
        records.

    Returns
    -------
    role : str or None
        ``"scaffold"`` for ``O(scaffold)-Si(mount)-first_ligand_atom``,
        ``"geminal"`` for ``O(geminal)-Si(mount)-first_ligand_atom``, or
        ``None`` when the angle does not match either cross-angle family.
    """
    if mount_atom_name is None:
        return None
    if record_b.residue_name not in ligand_shorts or record_b.atom_name != mount_atom_name:
        return None
    if db.get_element(record_b.atom_type) != "Si":
        return None

    for oxygen_record, ligand_record in ((record_a, record_c), (record_c, record_a)):
        if ligand_record.atom_name != first_ligand_atom_name:
            continue
        if oxygen_record.residue_name == "OM" and db.get_element(oxygen_record.atom_type) == "O":
            return "scaffold"
        if is_generated_geminal_record(oxygen_record, "O"):
            return "geminal"
    return None


@dataclass(frozen=True)
class _PdbResidueAliasRecord:
    """Mapping from one native residue name to its PDB-safe alias.

    Parameters
    ----------
    full_name : str
        Full residue identifier used internally and in mmCIF.
    pdb_name : str
        Three-character residue alias written to the PDB residue-name field.
    """

    full_name: str
    pdb_name: str


@dataclass(frozen=True)
class _CifEntityRecord:
    """One mmCIF entity row used by the structure writer.

    Parameters
    ----------
    entity_id : str
        mmCIF entity identifier referenced from ``_atom_site``.
    residue_name : str
        Full residue identifier represented by the entity.
    entity_type : str, optional
        mmCIF entity type. Functionalized slit exports use ``"non-polymer"``.
    """

    entity_id: str
    residue_name: str
    entity_type: str = "non-polymer"


@dataclass(frozen=True)
class _CifStructAsymRecord:
    """One mmCIF structural-asymmetry row used by the writer.

    Parameters
    ----------
    asym_id : str
        Unique label asymmetry identifier for one residue instance.
    entity_id : str
        Entity identifier referenced by ``asym_id``.
    residue_id : int
        One-based residue identifier in writer order.
    residue_name : str
        Full residue identifier represented by the asymmetry row.
    """

    asym_id: str
    entity_id: str
    residue_id: int
    residue_name: str


@dataclass(frozen=True)
class _ExportMoleculeCount:
    """One exported residue count written to helper topologies.

    Parameters
    ----------
    residue_name : str
        Residue identifier written to the exported topology.
    count : int
        Number of exported residue instances.
    """

    residue_name: str
    count: int


@dataclass(frozen=True)
class _ResidueExportIdentifiers:
    """Format-specific identifiers assigned to one residue instance.

    Parameters
    ----------
    residue_id : int
        One-based residue identifier in writer order.
    residue_name : str
        Full residue identifier preserved in mmCIF.
    pdb_residue_name : str
        Three-character PDB-safe residue alias.
    pdb_chain_id : str
        PDB chain identifier written to column 22.
    pdb_residue_id_token : str
        Four-character hybrid-36 residue-sequence token.
    cif_entity_id : str
        Entity identifier referenced from ``_atom_site.label_entity_id``.
    cif_asym_id : str
        Unique asymmetry identifier for ``_atom_site.label_asym_id``.
    cif_label_seq_id : str
        Label sequence token used by the mmCIF writer.
    """

    residue_id: int
    residue_name: str
    pdb_residue_name: str
    pdb_chain_id: str
    pdb_residue_id_token: str
    cif_entity_id: str
    cif_asym_id: str
    cif_label_seq_id: str


def _encode_pure(digits, value, width):
    """Encode one non-negative integer in a fixed-width positional alphabet.

    Parameters
    ----------
    digits : str
        Digits used by the positional numeral system.
    value : int
        Non-negative integer to encode.
    width : int
        Output width in characters.

    Returns
    -------
    token : str
        Encoded value padded to ``width`` characters.

    Raises
    ------
    ValueError
        Raised when ``value`` is negative or does not fit in ``width``
        characters for the selected alphabet.
    """
    if value < 0:
        raise ValueError("Encoded values must be non-negative.")

    base = len(digits)
    buffer = ["0"] * width
    remainder = value
    for index in range(width - 1, -1, -1):
        buffer[index] = digits[remainder % base]
        remainder //= base

    if remainder != 0:
        raise ValueError(f"Value {value} does not fit in width {width}.")

    return "".join(buffer)


def _decode_pure(digits, token):
    """Decode one positional token produced by :func:`_encode_pure`.

    Parameters
    ----------
    digits : str
        Digits used by the positional numeral system.
    token : str
        Fixed-width encoded token.

    Returns
    -------
    value : int
        Decoded integer value.

    Raises
    ------
    ValueError
        Raised when ``token`` contains a character outside ``digits``.
    """
    base = len(digits)
    value = 0
    for char in token:
        try:
            digit = digits.index(char)
        except ValueError as exc:
            raise ValueError(f"Invalid digit {char!r} in token {token!r}.") from exc
        value = value * base + digit
    return value


def _hybrid36_max_value(width):
    """Return the maximum non-negative integer representable in hybrid-36.

    Parameters
    ----------
    width : int
        Field width in characters.

    Returns
    -------
    value : int
        Largest representable non-negative integer.
    """
    return (10 ** width) + (2 * 26 * (36 ** (width - 1))) - 1


def _encode_hybrid36(width, value):
    """Encode one non-negative integer using canonical hybrid-36.

    Parameters
    ----------
    width : int
        Field width in characters. PDB uses ``4`` for residue ids and ``5``
        for atom serials.
    value : int
        Non-negative integer to encode.

    Returns
    -------
    token : str
        Fixed-width hybrid-36 token.

    Raises
    ------
    ValueError
        Raised when ``value`` is negative or exceeds the hybrid-36 range for
        ``width``.
    """
    if value < 0:
        raise ValueError("Hybrid-36 values must be non-negative.")

    decimal_limit = 10 ** width
    if value < decimal_limit:
        return f"{value:>{width}d}"

    base36_block = 26 * (36 ** (width - 1))
    offset = value - decimal_limit
    if offset < base36_block:
        return _encode_pure(
            _HYBRID36_DIGITS_UPPER,
            offset + (10 * (36 ** (width - 1))),
            width,
        )

    offset -= base36_block
    if offset < base36_block:
        return _encode_pure(
            _HYBRID36_DIGITS_LOWER,
            offset + (10 * (36 ** (width - 1))),
            width,
        )

    raise ValueError(
        f"Value {value} exceeds the hybrid-36 range for width {width} "
        f"(max {_hybrid36_max_value(width)})."
    )


def _decode_hybrid36(width, token):
    """Decode one canonical hybrid-36 token.

    Parameters
    ----------
    width : int
        Field width in characters.
    token : str
        Fixed-width hybrid-36 token.

    Returns
    -------
    value : int
        Decoded non-negative integer.

    Raises
    ------
    ValueError
        Raised when ``token`` does not match a supported hybrid-36 pattern.
    """
    if len(token) != width:
        raise ValueError(
            f"Hybrid-36 token {token!r} does not match width {width}."
        )

    first = token[0]
    if first == " " or first.isdigit():
        return int(token)
    if first.isupper():
        return (
            _decode_pure(_HYBRID36_DIGITS_UPPER, token)
            - (10 * (36 ** (width - 1)))
            + (10 ** width)
        )
    if first.islower():
        return (
            _decode_pure(_HYBRID36_DIGITS_LOWER, token)
            + (10 ** width)
            + (16 * (36 ** (width - 1)))
        )

    raise ValueError(f"Unsupported hybrid-36 token {token!r}.")


def _normalize_pdb_identifier(value, fallback):
    """Return an uppercase alphanumeric identifier for PDB-safe aliases.

    Parameters
    ----------
    value : str
        Identifier to normalize.
    fallback : str
        Fallback token used when normalization removes all characters.

    Returns
    -------
    token : str
        Uppercase alphanumeric identifier.
    """
    token = "".join(
        character
        for character in str(value).upper()
        if character.isascii() and character.isalnum()
    )
    return token or fallback


def _cif_token(value):
    """Return one mmCIF-safe token for a simple loop writer.

    Parameters
    ----------
    value : object
        Value written to the mmCIF file.

    Returns
    -------
    token : str
        Plain token or single-quoted value when quoting is required.
    """
    token = str(value)
    if not token:
        return "."
    if any(character.isspace() for character in token) or "'" in token:
        return "'" + token.replace("'", "''") + "'"
    return token


def _sanitize_pdb_token(value, width, fallback=""):
    """Return an ASCII token that cannot overflow a fixed-width PDB field.

    Parameters
    ----------
    value : str
        Token to sanitize.
    width : int
        Maximum field width.
    fallback : str, optional
        Replacement token used when sanitization removes all characters.

    Returns
    -------
    token : str
        Sanitized token truncated to ``width`` characters.
    """
    token = "".join(
        character
        for character in str(value)
        if character.isascii() and character.isprintable() and not character.isspace()
    )
    token = token or fallback
    return token[:width]


def _format_decimal_token(value, places=6):
    """Return one fixed-precision decimal token.

    Parameters
    ----------
    value : float or str
        Value to format.
    places : int, optional
        Number of decimal places used for float inputs.

    Returns
    -------
    token : str
        String token suitable for topology output.
    """
    if isinstance(value, str):
        return value
    return f"{value:.{places}f}"


def _sanitize_gromacs_identifier(value, fallback="SLIT"):
    """Return one simple GROMACS-safe identifier token.

    Parameters
    ----------
    value : str
        Candidate identifier.
    fallback : str, optional
        Fallback token used when sanitization removes every character.

    Returns
    -------
    token : str
        Uppercase alphanumeric identifier allowing underscores.
    """
    token = "".join(
        character
        for character in str(value)
        if character.isascii() and (character.isalnum() or character == "_")
    ).upper()
    return token or fallback


def _silica_atomtypes_in_order(silica_topology):
    """Return silica atom types in the exported deterministic order.

    Parameters
    ----------
    silica_topology : SilicaTopologyModel
        Resolved silica topology model.

    Returns
    -------
    atomtypes : tuple[GromacsAtomType, ...]
        Silica atom types converted into immutable GROMACS records.
    """
    return (
        silica_topology.atomtypes.framework_silicon.to_gromacs_atomtype(),
        silica_topology.atomtypes.framework_oxygen.to_gromacs_atomtype(),
        silica_topology.atomtypes.silanol_oxygen.to_gromacs_atomtype(),
        silica_topology.atomtypes.silanol_hydrogen.to_gromacs_atomtype(),
    )


def _silica_atomtype_lookup(silica_topology):
    """Return silica atom types keyed by their exported atom-type names.

    Parameters
    ----------
    silica_topology : SilicaTopologyModel
        Resolved silica topology model.

    Returns
    -------
    atomtypes_by_name : dict[str, GromacsAtomType]
        Mapping from atom-type name to immutable GROMACS atom-type record.

    Raises
    ------
    ValueError
        Raised when the silica model contains duplicate atom-type names.
    """
    atomtypes_by_name = {}
    for atomtype in _silica_atomtypes_in_order(silica_topology):
        if atomtype.name in atomtypes_by_name:
            raise ValueError(
                "Silica topology model defines duplicate atomtype name "
                f"{atomtype.name!r}."
            )
        atomtypes_by_name[atomtype.name] = atomtype
    return atomtypes_by_name


def _silica_assignment_role_name(record):
    """Return the silica-assignment role used for one exported atom.

    Parameters
    ----------
    record : _StructureAtomRecord
        Serialized atom record in writer order.

    Returns
    -------
    role_name : str
        Attribute name on :class:`silicams.topology.SilicaAtomAssignmentSet`
        describing the matching silica role.

    Raises
    ------
    ValueError
        Raised when ``record`` does not map to a supported silica export
        role.
    """
    if record.residue_name == "OM":
        return "framework_oxygen"
    if record.residue_name == "SI":
        return "framework_silicon"
    if record.residue_name == "SL":
        if record.atom_type == "Si":
            return "silanol_silicon"
        if record.atom_type == "O":
            return "silanol_oxygen"
        if record.atom_type == "H":
            return "silanol_hydrogen"
    if record.residue_name == "SLG":
        if record.atom_type == "Si":
            return "geminal_silicon"
        if record.atom_type == "O":
            return "geminal_oxygen"
        if record.atom_type == "H":
            return "geminal_hydrogen"

    raise ValueError(
        "Unsupported silica residue/atom combination for full slit topology "
        f"export: {(record.residue_name, record.atom_type, record.atom_name)!r}."
    )


@dataclass(frozen=True)
class _StructureAtomRecord:
    """One serialized atom record used by structure writers.

    Parameters
    ----------
    serial : int
        One-based atom serial number in writer order.
    pdb_serial_token : str
        Five-character hybrid-36 atom-serial token for PDB output.
    molecule_index : int
        Zero-based molecule index in the writer output order.
    local_atom_index : int
        Zero-based atom index inside the source molecule.
    residue_name : str
        Full residue identifier of the source molecule.
    pdb_residue_name : str
        Three-character PDB-safe residue alias.
    residue_id : int
        One-based residue identifier in writer order.
    pdb_residue_id_token : str
        Four-character hybrid-36 residue-sequence token for PDB output.
    atom_name : str
        Final atom name written by the structure writer.
    atom_type : str
        Element or atom-type token of the source atom.
    position : tuple[float, float, float]
        Cartesian position in nanometers.
    source_id : int or None
        Optional source atom identifier from the originating silica block.
    pdb_chain_id : str
        Single-character PDB chain identifier.
    cif_entity_id : str
        Entity identifier referenced by ``_atom_site.label_entity_id``.
    cif_asym_id : str
        Asymmetry identifier referenced by ``_atom_site.label_asym_id``.
    cif_label_seq_id : str
        Label sequence identifier referenced by ``_atom_site.label_seq_id``.
    """

    serial: int
    pdb_serial_token: str
    molecule_index: int
    local_atom_index: int
    residue_name: str
    pdb_residue_name: str
    residue_id: int
    pdb_residue_id_token: str
    atom_name: str
    atom_type: str
    position: tuple[float, float, float]
    source_id: int | None
    pdb_chain_id: str
    cif_entity_id: str
    cif_asym_id: str
    cif_label_seq_id: str


@dataclass
class _StructureExportCache:
    """Cached assembled export data for one atom-name mode.

    Parameters
    ----------
    atom_records : list[_StructureAtomRecord]
        Serialized atom metadata in structure-writer order.
    molecule_serials : list[list[int]]
        Atom serial numbers grouped by written molecule.
    residue_alias_records : list[_PdbResidueAliasRecord]
        Full-name to PDB-alias mappings used by the current export.
    entity_records : list[_CifEntityRecord]
        Declared mmCIF entity rows referenced by ``atom_records``.
    struct_asym_records : list[_CifStructAsymRecord]
        Declared mmCIF structural-asymmetry rows referenced by ``atom_records``.
    graph : AssembledStructureGraph or None, optional
        Cached assembled bond graph matching ``atom_records``.
    validation_report : ConnectivityValidationReport or None, optional
        Cached connectivity-validation report for the assembled structure.
    """

    atom_records: list[_StructureAtomRecord]
    molecule_serials: list[list[int]]
    residue_alias_records: list[_PdbResidueAliasRecord]
    entity_records: list[_CifEntityRecord]
    struct_asym_records: list[_CifStructAsymRecord]
    graph: AssembledStructureGraph | None = None
    validation_report: ConnectivityValidationReport | None = None



@dataclass(frozen=True)
class SnapshotScaffoldBond:
    """One scaffold bond expressed in source-atom identifiers.

    Parameters
    ----------
    source_atom_a : int
        First scaffold source identifier.
    source_atom_b : int
        Second scaffold source identifier.
    provenance : str
        Connectivity provenance label, such as ``"scaffold"`` or
        ``"siloxane_bridge"``.
    """

    source_atom_a: int
    source_atom_b: int
    provenance: str


@dataclass(frozen=True)
class SnapshotAtom:
    """One atom in deterministic shared export order.

    Parameters
    ----------
    serial : int
        One-based atom serial shared by all writers.
    molecule_index : int
        Zero-based molecule membership index.
    local_atom_index : int
        Zero-based index inside the source molecule.
    residue_name : str
        Exported residue identifier.
    residue_id : int
        One-based residue serial.
    atom_name : str
        Explicit export atom name.
    atom_type : str
        Source atom type or element token.
    position_nm : tuple[float, float, float]
        Cartesian position in nanometers.
    source_id : int or None
        Optional scaffold source identifier.
    """

    serial: int
    molecule_index: int
    local_atom_index: int
    residue_name: str
    residue_id: int
    atom_name: str
    atom_type: str
    position_nm: tuple[float, float, float]
    source_id: int | None


@dataclass(frozen=True)
class SnapshotAttachment:
    """Writer-ready graft attachment metadata.

    Parameters
    ----------
    molecule_index : int
        Index of the attached molecule in snapshot order.
    site_id : int
        Consumed scaffold-silicon source identifier.
    site_type : str
        Surface family identifier.
    mount_atom_local_id : int
        Zero-based ligand mount-atom index.
    is_geminal : bool
        Whether the attachment consumed a geminal site.
    scaffold_oxygen_source_ids : tuple[int, ...]
        Retained scaffold-oxygen identifiers at the graft junction.
    surface_oxygen_source_ids : tuple[int, ...]
        Removed surface-handle oxygen identifiers.
    """

    molecule_index: int
    site_id: int
    site_type: str
    mount_atom_local_id: int
    is_geminal: bool
    scaffold_oxygen_source_ids: tuple[int, ...]
    surface_oxygen_source_ids: tuple[int, ...]


@dataclass(frozen=True)
class StructureSnapshot:
    """Immutable shared input for structure and topology writers.

    Parameters
    ----------
    name : str
        Export basename and system title.
    box_nm : tuple[float, float, float]
        Orthorhombic box dimensions in nanometers.
    molecules : tuple[Molecule, ...]
        Independent molecule copies in deterministic export order.
    residue_order : tuple[str, ...]
        Residue short names in deterministic group order.
    scaffold_bonds : tuple[SnapshotScaffoldBond, ...], optional
        Scaffold connectivity expressed in source identifiers.
    attachments : tuple[SnapshotAttachment, ...], optional
        Graft-junction metadata expressed against molecule indices.
    atom_order : tuple[SnapshotAtom, ...], optional
        Final atom and residue numbering shared by all writers.
    molecule_serials : tuple[tuple[int, ...], ...], optional
        Atom serials grouped by molecule membership.
    assembled_bonds : tuple[GraphBond, ...], optional
        Complete scaffold, ligand, bridge, and graft-junction bonds.
    assembled_angles : tuple[GraphAngle, ...], optional
        Angles derived from ``assembled_bonds``.
    graft_junction_serials : tuple[tuple[int, int], ...], optional
        Export-serial pairs for graft-junction bonds.
    residue_counts : tuple[tuple[str, int], ...], optional
        Exported residue counts in first-appearance order.
    has_assembled_export : bool, optional
        Whether final numbering and connectivity fields are populated.
    is_finalized : bool, optional
        Whether the source domain model was finalized.
    source_kind : str, optional
        Source-domain discriminator used by specialized writers.
    """

    name: str
    box_nm: tuple[float, float, float]
    molecules: tuple[Molecule, ...]
    residue_order: tuple[str, ...]
    scaffold_bonds: tuple[SnapshotScaffoldBond, ...] = ()
    attachments: tuple[SnapshotAttachment, ...] = ()
    atom_order: tuple[SnapshotAtom, ...] = ()
    molecule_serials: tuple[tuple[int, ...], ...] = ()
    assembled_bonds: tuple[GraphBond, ...] = ()
    assembled_angles: tuple[GraphAngle, ...] = ()
    graft_junction_serials: tuple[tuple[int, int], ...] = ()
    residue_counts: tuple[tuple[str, int], ...] = ()
    has_assembled_export: bool = False
    is_finalized: bool = True
    source_kind: str = "molecule"

    @classmethod
    def from_components(
        cls,
        name,
        box_nm,
        molecules,
        residue_order,
        scaffold_bonds,
        attachments,
        is_finalized,
        source_kind,
    ):
        """Build an independent snapshot from domain export components.

        Parameters
        ----------
        name : str
            Export basename and system title.
        box_nm : sequence[float]
            Orthorhombic box dimensions in nanometers.
        molecules : iterable[Molecule]
            Molecules in deterministic export order.
        residue_order : iterable[str]
            Residue short names in deterministic group order.
        scaffold_bonds : iterable[tuple]
            Source-atom bond pairs with provenance labels.
        attachments : iterable[tuple]
            Writer-ready graft attachment tuples.
        is_finalized : bool
            Finalization state of the source domain model.
        source_kind : str
            Source-domain discriminator.

        Returns
        -------
        snapshot : StructureSnapshot
            Independent immutable writer input.
        """
        return cls(
            name=str(name),
            box_nm=tuple(float(value) for value in box_nm),
            molecules=tuple(copy.deepcopy(tuple(molecules))),
            residue_order=tuple(residue_order),
            scaffold_bonds=tuple(SnapshotScaffoldBond(*bond) for bond in scaffold_bonds),
            attachments=tuple(SnapshotAttachment(*item) for item in attachments),
            is_finalized=bool(is_finalized),
            source_kind=str(source_kind),
        )

    def with_assembled_export(
        self,
        atom_records,
        molecule_serials,
        graph,
        residue_counts,
    ):
        """Return a snapshot populated with final numbering and connectivity.

        Parameters
        ----------
        atom_records : iterable[_StructureAtomRecord]
            Final structure records in writer order.
        molecule_serials : iterable[iterable[int]]
            Atom serials grouped by molecule membership.
        graph : AssembledStructureGraph
            Complete assembled graph for those serials.
        residue_counts : iterable[tuple[str, int]]
            Exported residue names and counts.

        Returns
        -------
        snapshot : StructureSnapshot
            New immutable snapshot with shared export data.
        """
        atoms = tuple(
            SnapshotAtom(
                serial=record.serial,
                molecule_index=record.molecule_index,
                local_atom_index=record.local_atom_index,
                residue_name=record.residue_name,
                residue_id=record.residue_id,
                atom_name=record.atom_name,
                atom_type=record.atom_type,
                position_nm=tuple(record.position),
                source_id=record.source_id,
            )
            for record in atom_records
        )
        junctions = tuple(
            (bond.atom_a, bond.atom_b)
            for bond in graph.bonds
            if bond.provenance == "graft_junction"
        )
        return replace(
            self,
            atom_order=atoms,
            molecule_serials=tuple(
                tuple(int(serial) for serial in serials)
                for serials in molecule_serials
            ),
            assembled_bonds=tuple(graph.bonds),
            assembled_angles=tuple(graph.angles),
            graft_junction_serials=junctions,
            residue_counts=tuple(
                (str(residue_name), int(count))
                for residue_name, count in residue_counts
            ),
            has_assembled_export=True,
        )

    @property
    def assembled_graph(self):
        """Return the complete graph stored in final export order."""
        if not self.has_assembled_export:
            raise ValueError("The snapshot does not contain assembled export data.")
        return AssembledStructureGraph(
            atom_ids=tuple(atom.serial for atom in self.atom_order),
            bonds=self.assembled_bonds,
            angles=self.assembled_angles,
        )

    @classmethod
    def from_molecule(cls, molecule):
        """Build an independent structure snapshot from one molecule.

        Parameters
        ----------
        molecule : Molecule
            Standalone molecule to copy into the snapshot.

        Returns
        -------
        snapshot : StructureSnapshot
            Independent immutable writer input.
        """
        if not isinstance(molecule, Molecule):
            raise TypeError("molecule must be a Molecule instance.")
        return cls(
            name=molecule.get_name() or "molecule",
            box_nm=tuple(float(value) for value in (molecule.get_box() or [0.0, 0.0, 0.0])),
            molecules=(copy.deepcopy(molecule),),
            residue_order=(molecule.get_short(),),
        )
