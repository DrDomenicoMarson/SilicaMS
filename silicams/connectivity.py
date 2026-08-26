################################################################################
# Connectivity Models                                                          #
#                                                                              #
"""Shared connectivity dataclasses for bonded structure export and validation."""
################################################################################


from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class GraphBond:
    """One bonded pair in an assembled structure graph.

    Parameters
    ----------
    atom_a : int
        One-based atom id in assembled export order.
    atom_b : int
        One-based atom id in assembled export order.
    provenance : str
        Provenance label describing where the bond originates.
    """

    atom_a: int
    atom_b: int
    provenance: str

    def __post_init__(self):
        """Normalize bond ordering and reject degenerate pairs."""
        if self.atom_a == self.atom_b:
            raise ValueError("GraphBond requires two distinct atom ids.")
        if self.atom_a > self.atom_b:
            atom_a = self.atom_a
            atom_b = self.atom_b
            object.__setattr__(self, "atom_a", atom_b)
            object.__setattr__(self, "atom_b", atom_a)


@dataclass(frozen=True)
class GraphAngle:
    """One angle triplet derived from an assembled structure graph.

    Parameters
    ----------
    atom_a : int
        One-based first outer atom id.
    atom_b : int
        One-based central atom id.
    atom_c : int
        One-based second outer atom id.
    """

    atom_a: int
    atom_b: int
    atom_c: int

    def __post_init__(self):
        """Normalize angle ordering and reject degenerate triplets."""
        if len({self.atom_a, self.atom_b, self.atom_c}) != 3:
            raise ValueError("GraphAngle requires three distinct atom ids.")
        if self.atom_a > self.atom_c:
            atom_a = self.atom_a
            atom_c = self.atom_c
            object.__setattr__(self, "atom_a", atom_c)
            object.__setattr__(self, "atom_c", atom_a)


@dataclass(frozen=True)
class AttachmentRecord:
    """Metadata describing one grafted molecule placement.

    Parameters
    ----------
    site_id : int
        Silicon site id that was consumed during attachment.
    site_type : str
        Site family identifier, usually ``"in"`` or ``"ex"``.
    mount_atom_local_id : int
        Zero-based local atom index of the ligand mount atom.
    is_geminal : bool
        True when the attachment consumed a geminal surface site.
    scaffold_oxygen_source_ids : tuple[int, ...]
        Source ids of retained scaffold oxygens that define the silica-ligand
        junction.
    surface_oxygen_source_ids : tuple[int, ...], optional
        Source ids of the removed surface-handle oxygens originally present on
        the consumed site.
    molecule : object or None, optional
        Attached molecule instance stored in the silica-system molecule data.
    """

    site_id: int
    site_type: str
    mount_atom_local_id: int
    is_geminal: bool
    scaffold_oxygen_source_ids: tuple[int, ...]
    surface_oxygen_source_ids: tuple[int, ...] = ()
    molecule: object | None = None


@dataclass(frozen=True)
class ConnectivityValidationFinding:
    """One connectivity-validation issue found on an assembled structure.

    Parameters
    ----------
    code : str
        Stable issue code describing the validation rule that failed.
    message : str
        Human-readable explanation of the issue.
    atom_ids : tuple[int, ...]
        One-based atom ids involved in the issue.
    atom_types : tuple[str, ...]
        Atom types of the involved atoms in the same order as ``atom_ids``.
    residue_shorts : tuple[str, ...], optional
        Residue short names of the involved atoms.
    degrees : tuple[int, ...], optional
        Bond degrees of the involved atoms in the assembled graph.
    is_error : bool, optional
        True when the finding should make strict validation fail.
    """

    code: str
    message: str
    atom_ids: tuple[int, ...]
    atom_types: tuple[str, ...]
    residue_shorts: tuple[str, ...] = ()
    degrees: tuple[int, ...] = ()
    is_error: bool = True


@dataclass(frozen=True)
class ConnectivityValidationReport:
    """Structured connectivity-validation result for one assembled structure.

    Parameters
    ----------
    atom_count : int
        Number of atoms in the validated assembled structure.
    bond_count : int
        Number of bonds in the validated assembled graph.
    findings : tuple[ConnectivityValidationFinding, ...]
        Validation findings collected for the structure.
    """

    atom_count: int
    bond_count: int
    findings: tuple[ConnectivityValidationFinding, ...]

    @property
    def error_count(self):
        """Return the number of error-level findings."""
        return sum(1 for finding in self.findings if finding.is_error)

    @property
    def warning_count(self):
        """Return the number of warning-level findings."""
        return sum(1 for finding in self.findings if not finding.is_error)

    @property
    def is_valid(self):
        """Return whether no error-level findings were collected."""
        return self.error_count == 0


@dataclass(frozen=True)
class AssembledStructureGraph:
    """Bond graph assembled in the same atom order as structure export.

    Parameters
    ----------
    atom_ids : tuple[int, ...]
        One-based atom ids in writer order.
    bonds : tuple[GraphBond, ...]
        Bonded atom pairs present in the assembled system.
    angles : tuple[GraphAngle, ...]
        Unique angles derived from ``bonds``.
    """

    atom_ids: tuple[int, ...]
    bonds: tuple[GraphBond, ...]
    angles: tuple[GraphAngle, ...]

    @classmethod
    def from_bonds(cls, atom_ids, bonds):
        """Build an assembled graph and derive unique angles.

        Parameters
        ----------
        atom_ids : iterable[int]
            One-based atom ids in writer order.
        bonds : iterable[GraphBond]
            Bond definitions to include in the graph.

        Returns
        -------
        graph : AssembledStructureGraph
            Graph with deduplicated bonds and derived angles.
        """
        unique_bonds = {}
        for bond in bonds:
            key = (bond.atom_a, bond.atom_b)
            if key not in unique_bonds:
                unique_bonds[key] = bond

        ordered_atom_ids = tuple(atom_ids)
        ordered_bonds = tuple(sorted(unique_bonds.values(), key=lambda bond: (bond.atom_a, bond.atom_b, bond.provenance)))
        return cls(
            atom_ids=ordered_atom_ids,
            bonds=ordered_bonds,
            angles=cls.enumerate_angles(ordered_atom_ids, ordered_bonds),
        )

    @staticmethod
    def enumerate_angles(atom_ids, bonds):
        """Derive all unique angles implied by a set of bonds.

        Parameters
        ----------
        atom_ids : iterable[int]
            One-based atom ids in assembled order.
        bonds : iterable[GraphBond]
            Bond definitions used to derive adjacency.

        Returns
        -------
        angles : tuple[GraphAngle, ...]
            Unique angle triplets ordered around their central atom.
        """
        atom_set = set(atom_ids)
        neighbors = {atom_id: set() for atom_id in atom_set}
        for bond in bonds:
            if bond.atom_a not in atom_set or bond.atom_b not in atom_set:
                continue
            neighbors[bond.atom_a].add(bond.atom_b)
            neighbors[bond.atom_b].add(bond.atom_a)

        angles = set()
        for atom_b in sorted(neighbors):
            for atom_a, atom_c in combinations(sorted(neighbors[atom_b]), 2):
                angles.add(GraphAngle(atom_a, atom_b, atom_c))

        return tuple(sorted(angles, key=lambda angle: (angle.atom_b, angle.atom_a, angle.atom_c)))
