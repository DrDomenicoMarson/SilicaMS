################################################################################
# GROMACS Topology Models                                                      #
#                                                                              #
"""Dataclasses and helpers for flat GROMACS topology parsing and writing."""
################################################################################


from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import yaml


@dataclass(frozen=True)
class GromacsBondParameters:
    """Parameter payload for one GROMACS bond definition.

    Parameters
    ----------
    function : int
        GROMACS bond function type.
    parameters : tuple[str, ...]
        Raw parameter tokens written after ``function``.
    """

    function: int
    parameters: tuple[str, ...]

    @classmethod
    def harmonic(cls, length_nm, force_constant):
        """Build one harmonic bond-parameter record.

        Parameters
        ----------
        length_nm : float
            Equilibrium bond length in nanometers.
        force_constant : float
            Harmonic force constant in the GROMACS bond units.

        Returns
        -------
        parameters : GromacsBondParameters
            Harmonic bond-parameter record.
        """
        return cls(
            function=1,
            parameters=(f"{length_nm:.5f}", f"{force_constant:.6f}"),
        )


@dataclass(frozen=True)
class GromacsAngleParameters:
    """Parameter payload for one GROMACS angle definition.

    Parameters
    ----------
    function : int
        GROMACS angle function type.
    parameters : tuple[str, ...]
        Raw parameter tokens written after ``function``.
    """

    function: int
    parameters: tuple[str, ...]

    @classmethod
    def harmonic(cls, angle_deg, force_constant):
        """Build one harmonic angle-parameter record.

        Parameters
        ----------
        angle_deg : float
            Equilibrium angle in degrees.
        force_constant : float
            Harmonic force constant in the GROMACS angle units.

        Returns
        -------
        parameters : GromacsAngleParameters
            Harmonic angle-parameter record.
        """
        return cls(
            function=1,
            parameters=(f"{angle_deg:.5f}", f"{force_constant:.6f}"),
        )


@dataclass
class SilicaAtomTypeModel:
    """One editable silica atom-type definition with provenance metadata.

    Parameters
    ----------
    name : str
        GROMACS atom-type identifier.
    atomic_number : int or None
        Optional atomic number token.
    mass : str
        Mass token written in the atom-type row.
    charge : str
        Charge token written in the atom-type row.
    particle_type : str
        Particle-type token such as ``"A"``.
    sigma : str
        Lennard-Jones sigma token.
    epsilon : str
        Lennard-Jones epsilon token.
    origin : str
        Human-readable provenance string describing where the default came
        from.
    """

    name: str
    atomic_number: int | None
    mass: str
    charge: str
    particle_type: str
    sigma: str
    epsilon: str
    origin: str

    def to_gromacs_atomtype(self):
        """Convert the editable model into one GROMACS atom-type record.

        Returns
        -------
        atomtype : GromacsAtomType
            Equivalent immutable atom-type payload used during topology
            rendering.
        """
        return GromacsAtomType(
            name=self.name,
            atomic_number=self.atomic_number,
            mass=self.mass,
            charge=self.charge,
            particle_type=self.particle_type,
            sigma=self.sigma,
            epsilon=self.epsilon,
        )


@dataclass
class SilicaAtomTypeSet:
    """Editable silica atom-type bundle used by slit topology export.

    Parameters
    ----------
    framework_silicon : SilicaAtomTypeModel
        Atom-type definition used for silica scaffold silicon atoms.
    framework_oxygen : SilicaAtomTypeModel
        Atom-type definition used for retained framework oxygen atoms.
    silanol_oxygen : SilicaAtomTypeModel
        Atom-type definition used for silanol and geminal hydroxyl oxygen
        atoms.
    silanol_hydrogen : SilicaAtomTypeModel
        Atom-type definition used for silanol and geminal hydroxyl hydrogen
        atoms.
    """

    framework_silicon: SilicaAtomTypeModel
    framework_oxygen: SilicaAtomTypeModel
    silanol_oxygen: SilicaAtomTypeModel
    silanol_hydrogen: SilicaAtomTypeModel


@dataclass
class SilicaAtomAssignment:
    """One editable silica atom assignment with provenance metadata.

    Parameters
    ----------
    atom_type_name : str
        Atom-type name written into the exported ``[ atoms ]`` row.
    charge : str
        Charge token written into the exported atom row.
    mass : str
        Mass token written into the exported atom row.
    origin : str
        Human-readable provenance string describing where the default came
        from.
    """

    atom_type_name: str
    charge: str
    mass: str
    origin: str


@dataclass
class SilicaAtomAssignmentSet:
    """Editable silica charge and mass assignments used during slit export.

    Parameters
    ----------
    framework_oxygen : SilicaAtomAssignment
        Assignment used for ``OM`` framework oxygen residues.
    framework_silicon : SilicaAtomAssignment
        Assignment used for ``SI`` framework silicon residues.
    silanol_silicon : SilicaAtomAssignment
        Assignment used for silicon atoms inside ``SL`` residues.
    silanol_oxygen : SilicaAtomAssignment
        Assignment used for oxygen atoms inside ``SL`` residues.
    silanol_hydrogen : SilicaAtomAssignment
        Assignment used for hydrogen atoms inside ``SL`` residues.
    geminal_silicon : SilicaAtomAssignment
        Assignment used for silicon atoms inside ``SLG`` residues.
    geminal_oxygen : SilicaAtomAssignment
        Assignment used for oxygen atoms inside ``SLG`` residues.
    geminal_hydrogen : SilicaAtomAssignment
        Assignment used for hydrogen atoms inside ``SLG`` residues.
    """

    framework_oxygen: SilicaAtomAssignment
    framework_silicon: SilicaAtomAssignment
    silanol_silicon: SilicaAtomAssignment
    silanol_oxygen: SilicaAtomAssignment
    silanol_hydrogen: SilicaAtomAssignment
    geminal_silicon: SilicaAtomAssignment
    geminal_oxygen: SilicaAtomAssignment
    geminal_hydrogen: SilicaAtomAssignment


@dataclass
class SilicaBondTerm:
    """One editable harmonic silica bond term with provenance metadata.

    Parameters
    ----------
    length_nm : float
        Equilibrium bond length in nanometers.
    force_constant : float
        Harmonic force constant in the GROMACS bond units.
    origin : str
        Human-readable provenance string describing where the default came
        from.
    """

    length_nm: float
    force_constant: float
    origin: str

    @classmethod
    def from_gromacs_parameters(cls, parameters, origin):
        """Build one editable bond term from a GROMACS parameter payload.

        Parameters
        ----------
        parameters : GromacsBondParameters
            Source harmonic bond parameters.
        origin : str
            Provenance string stored on the returned term.

        Returns
        -------
        term : SilicaBondTerm
            Editable bond term carrying the same numerical values.

        Raises
        ------
        ValueError
            Raised when ``parameters`` is not a two-token harmonic bond term.
        """
        if parameters.function != 1 or len(parameters.parameters) != 2:
            raise ValueError(
                "SilicaBondTerm only supports harmonic two-parameter GROMACS "
                f"bond records. Received function={parameters.function} "
                f"parameters={parameters.parameters!r}."
            )
        return cls(
            length_nm=float(parameters.parameters[0]),
            force_constant=float(parameters.parameters[1]),
            origin=origin,
        )

    def to_gromacs_parameters(self):
        """Convert the editable bond term into a GROMACS parameter payload.

        Returns
        -------
        parameters : GromacsBondParameters
            Immutable harmonic bond parameters for topology rendering.
        """
        return GromacsBondParameters.harmonic(
            length_nm=self.length_nm,
            force_constant=self.force_constant,
        )


@dataclass
class SilicaBondTermSet:
    """Editable silica bond-term bundle used by slit topology export.

    Parameters
    ----------
    framework_si_o : SilicaBondTerm
        Harmonic bond term used for ordinary silica ``Si-O`` bonds.
    silanol_o_h : SilicaBondTerm
        Harmonic bond term used for hydroxyl ``O-H`` bonds.
    graft_mount_scaffold_si_o : SilicaBondTerm
        Harmonic bond term used for retained scaffold oxygen atoms bound to
        the ligand mount silicon during grafting.
    """

    framework_si_o: SilicaBondTerm
    silanol_o_h: SilicaBondTerm
    graft_mount_scaffold_si_o: SilicaBondTerm


@dataclass
class SilicaAngleTerm:
    """One editable harmonic silica angle term with provenance metadata.

    Parameters
    ----------
    angle_deg : float
        Equilibrium angle in degrees.
    force_constant : float
        Harmonic force constant in the GROMACS angle units.
    origin : str
        Human-readable provenance string describing where the default came
        from.
    """

    angle_deg: float
    force_constant: float
    origin: str

    @classmethod
    def from_gromacs_parameters(cls, parameters, origin):
        """Build one editable angle term from a GROMACS parameter payload.

        Parameters
        ----------
        parameters : GromacsAngleParameters
            Source harmonic angle parameters.
        origin : str
            Provenance string stored on the returned term.

        Returns
        -------
        term : SilicaAngleTerm
            Editable angle term carrying the same numerical values.

        Raises
        ------
        ValueError
            Raised when ``parameters`` is not a two-token harmonic angle term.
        """
        if parameters.function != 1 or len(parameters.parameters) != 2:
            raise ValueError(
                "SilicaAngleTerm only supports harmonic two-parameter GROMACS "
                f"angle records. Received function={parameters.function} "
                f"parameters={parameters.parameters!r}."
            )
        return cls(
            angle_deg=float(parameters.parameters[0]),
            force_constant=float(parameters.parameters[1]),
            origin=origin,
        )

    def to_gromacs_parameters(self):
        """Convert the editable angle term into a GROMACS parameter payload.

        Returns
        -------
        parameters : GromacsAngleParameters
            Immutable harmonic angle parameters for topology rendering.
        """
        return GromacsAngleParameters.harmonic(
            angle_deg=self.angle_deg,
            force_constant=self.force_constant,
        )


@dataclass
class SilicaAngleTermSet:
    """Editable silica angle-term bundle used by slit topology export.

    Parameters
    ----------
    framework_si_o_si : SilicaAngleTerm
        Harmonic angle term used for ordinary silica ``Si-O-Si`` bridge
        angles.
    silanol_o_si_o : SilicaAngleTerm
        Harmonic angle term used for ``O-Si-O`` angles around silanol or
        geminal silicon atoms.
    silanol_si_o_h : SilicaAngleTerm
        Harmonic angle term used for hydroxyl ``Si-O-H`` angles.
    graft_scaffold_si_scaffold_o_mount : SilicaAngleTerm
        Harmonic angle term used for ``Si(scaffold)-O(scaffold)-Si(mount)``
        graft-junction angles.
    graft_oxygen_mount_oxygen : SilicaAngleTerm
        Harmonic angle term used for ``O-Si(mount)-O`` angles around the
        grafted ligand silicon.
    """

    framework_si_o_si: SilicaAngleTerm
    silanol_o_si_o: SilicaAngleTerm
    silanol_si_o_h: SilicaAngleTerm
    graft_scaffold_si_scaffold_o_mount: SilicaAngleTerm
    graft_oxygen_mount_oxygen: SilicaAngleTerm


@dataclass
class SilicaTopologyModel:
    """Editable silica force-field model used by slit topology export.

    Parameters
    ----------
    atomtypes : SilicaAtomTypeSet
        Silica atom-type definitions emitted ahead of the generated
        ``[ moleculetype ]`` section.
    atom_assignments : SilicaAtomAssignmentSet
        Charge and mass assignments applied to finalized silica atoms during
        full-slab export.
    bond_terms : SilicaBondTermSet
        Harmonic bond terms used for silica scaffold, hydroxyl, and graft
        junction bonds.
    angle_terms : SilicaAngleTermSet
        Harmonic angle terms used for silica scaffold, hydroxyl, and graft
        junction angles.
    """

    atomtypes: SilicaAtomTypeSet
    atom_assignments: SilicaAtomAssignmentSet
    bond_terms: SilicaBondTermSet
    angle_terms: SilicaAngleTermSet

    def to_dict(self):
        """Return the silica model as one plain nested dictionary.

        Returns
        -------
        data : dict[str, object]
            Recursive dictionary/list/scalar representation of the full silica
            topology model, including provenance strings.
        """
        return asdict(self)

    def to_json(self, indent=2):
        """Return the silica model as formatted JSON text.

        Parameters
        ----------
        indent : int, optional
            JSON indentation level forwarded to :func:`json.dumps`.

        Returns
        -------
        text : str
            Human-readable JSON serialization of the full silica topology
            model.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def to_yaml(self):
        """Return the silica model as human-readable YAML text.

        Returns
        -------
        text : str
            YAML serialization of the full silica topology model.
        """
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


@dataclass(frozen=True)
class BareSilicaChargeContribution:
    """One charge contribution used by the bare-slit neutrality audit.

    Parameters
    ----------
    atom_role : str
        Human-readable silica role name such as ``"framework_oxygen"``.
    atom_type_name : str
        GROMACS atom-type identifier written for this role.
    atom_count : int
        Number of exported atoms that currently use this role.
    charge_per_atom : float
        Partial charge assigned to each atom in this role.
    total_charge : float
        Total charge contribution of the role, equal to
        ``atom_count * charge_per_atom``.
    """

    atom_role: str
    atom_type_name: str
    atom_count: int
    charge_per_atom: float
    total_charge: float


@dataclass(frozen=True)
class BareSilicaChargeDiagnostics:
    """Charge-neutrality diagnostics for one finalized bare silica slit.

    Parameters
    ----------
    framework_silicon : BareSilicaChargeContribution
        Contribution of exported ``SI`` scaffold silicon atoms.
    framework_oxygen : BareSilicaChargeContribution
        Contribution of exported ``OM`` scaffold oxygen atoms.
    silanol_silicon : BareSilicaChargeContribution
        Contribution of silicon atoms inside ``SL`` residues.
    silanol_oxygen : BareSilicaChargeContribution
        Contribution of oxygen atoms inside ``SL`` residues.
    silanol_hydrogen : BareSilicaChargeContribution
        Contribution of hydrogen atoms inside ``SL`` residues.
    geminal_silicon : BareSilicaChargeContribution
        Contribution of silicon atoms inside ``SLG`` residues.
    geminal_oxygen : BareSilicaChargeContribution
        Contribution of oxygen atoms inside ``SLG`` residues.
    geminal_hydrogen : BareSilicaChargeContribution
        Contribution of hydrogen atoms inside ``SLG`` residues.
    silanol_site_count : int
        Number of exported single-silanol ``SL`` residues.
    geminal_site_count : int
        Number of exported geminal-silanol ``SLG`` residues.
    total_silicon_count : int
        Total number of exported silica silicon atoms across scaffold and
        hydroxylated surface residues.
    total_hydroxyl_count : int
        Total number of hydroxyl groups, counted via exported hydroxyl
        hydrogen atoms.
    coordination_identity_left : int
        Left-hand side of the silica coordination identity ``4*N_Si``.
    coordination_identity_right : int
        Right-hand side of the silica coordination identity
        ``2*N_OM + N_OH``.
    coordination_identity_delta : int
        Difference ``coordination_identity_left - coordination_identity_right``.
    total_charge : float
        Summed total charge of the exported bare slit under the active silica
        assignments.
    """

    framework_silicon: BareSilicaChargeContribution
    framework_oxygen: BareSilicaChargeContribution
    silanol_silicon: BareSilicaChargeContribution
    silanol_oxygen: BareSilicaChargeContribution
    silanol_hydrogen: BareSilicaChargeContribution
    geminal_silicon: BareSilicaChargeContribution
    geminal_oxygen: BareSilicaChargeContribution
    geminal_hydrogen: BareSilicaChargeContribution
    silanol_site_count: int
    geminal_site_count: int
    total_silicon_count: int
    total_hydroxyl_count: int
    coordination_identity_left: int
    coordination_identity_right: int
    coordination_identity_delta: int
    total_charge: float

    @property
    def is_neutral(self):
        """Return whether the audited bare slit is charge neutral."""
        return abs(self.total_charge) <= 1e-8

    @property
    def coordination_identity_holds(self):
        """Return whether ``4*N_Si = 2*N_OM + N_OH`` holds exactly."""
        return self.coordination_identity_delta == 0


@dataclass(frozen=True)
class FunctionalizedSlitChargeDiagnostics:
    """Charge audit for one functionalized full-slit topology export.

    Parameters
    ----------
    expected_t3_fragment_charge : float
        Expected total charge of the user-supplied base ``T3`` fragment under
        the resolved silica model.
    observed_t3_fragment_charge : float
        Total charge parsed from the user-supplied flat ligand topology
        bundle.
    t3_fragment_charge_delta : float
        Difference ``observed_t3_fragment_charge - expected_t3_fragment_charge``.
    derived_t2_fragment_charge : float
        Expected total charge of the internally generated geminal ``T2`` form.
    geminal_added_oh_charge : float
        Charge contribution of the internally added silica ``OH`` pair.
    t2_site_count : int
        Number of exported geminal ``T2`` residues.
    t3_site_count : int
        Number of exported ``T3`` residues.
    final_total_charge : float
        Summed total charge of the final assembled functionalized slit
        topology.
    tolerance : float, optional
        Absolute tolerance used for the fragment-charge and final-neutrality
        checks.
    """

    expected_t3_fragment_charge: float
    observed_t3_fragment_charge: float
    t3_fragment_charge_delta: float
    derived_t2_fragment_charge: float
    geminal_added_oh_charge: float
    t2_site_count: int
    t3_site_count: int
    final_total_charge: float
    tolerance: float = 1e-6

    @property
    def is_base_fragment_charge_valid(self):
        """Return whether the base ``T3`` fragment charge matches the target."""
        return abs(self.t3_fragment_charge_delta) <= self.tolerance

    @property
    def is_final_topology_neutral(self):
        """Return whether the final assembled slit topology is neutral."""
        return abs(self.final_total_charge) <= self.tolerance

    @property
    def is_valid(self):
        """Return whether both functionalized-slit charge invariants hold."""
        return (
            self.is_base_fragment_charge_valid
            and self.is_final_topology_neutral
        )


def _build_default_silica_topology():
    """Build the package-default editable silica topology model.

    Returns
    -------
    model : SilicaTopologyModel
        Fresh default silica topology model populated from the package's
        standard slit-topology parameters.
    """
    return SilicaTopologyModel(
        atomtypes=SilicaAtomTypeSet(
            framework_silicon=SilicaAtomTypeModel(
                name="SI",
                atomic_number=14,
                mass="28.08600",
                charge="0.000000",
                particle_type="A",
                sigma="0.4150",
                epsilon="0.3891120",
                origin="doi:10.1021/cm500365c",
            ),
            framework_oxygen=SilicaAtomTypeModel(
                name="OM",
                atomic_number=8,
                mass="15.99940",
                charge="0.000000",
                particle_type="A",
                sigma="0.3470",
                epsilon="0.2259360",
                origin="doi:10.1021/cm500365c",
            ),
            silanol_oxygen=SilicaAtomTypeModel(
                name="OA",
                atomic_number=8,
                mass="15.99940",
                charge="0.000000",
                particle_type="A",
                sigma="0.3470",
                epsilon="0.5104480",
                origin="doi:10.1021/cm500365c",
            ),
            silanol_hydrogen=SilicaAtomTypeModel(
                name="HG",
                atomic_number=1,
                mass="2.01600",
                charge="0.000000",
                particle_type="A",
                sigma="0.1085",
                epsilon="0.0627600",
                origin="doi:10.1021/cm500365c",
            ),
        ),
        atom_assignments=SilicaAtomAssignmentSet(
            framework_oxygen=SilicaAtomAssignment(
                atom_type_name="OM",
                charge="-0.550000",
                mass="15.99940",
                origin="doi:10.1021/cm500365c",
            ),
            framework_silicon=SilicaAtomAssignment(
                atom_type_name="SI",
                charge="1.100000",
                mass="28.08600",
                origin="doi:10.1021/cm500365c",
            ),
            silanol_silicon=SilicaAtomAssignment(
                atom_type_name="SI",
                charge="1.100000",
                mass="28.08600",
                origin="doi:10.1021/cm500365c",
            ),
            silanol_oxygen=SilicaAtomAssignment(
                atom_type_name="OA",
                charge="-0.675000",
                mass="15.99940",
                origin="doi:10.1021/cm500365c",
            ),
            silanol_hydrogen=SilicaAtomAssignment(
                atom_type_name="HG",
                charge="0.400000",
                mass="2.01600",
                origin="doi:10.1021/cm500365c",
            ),
            geminal_silicon=SilicaAtomAssignment(
                atom_type_name="SI",
                charge="1.100000",
                mass="28.08600",
                origin="doi:10.1021/cm500365c",
            ),
            geminal_oxygen=SilicaAtomAssignment(
                atom_type_name="OA",
                charge="-0.675000",
                mass="15.99940",
                origin="doi:10.1021/cm500365c",
            ),
            geminal_hydrogen=SilicaAtomAssignment(
                atom_type_name="HG",
                charge="0.400000",
                mass="2.01600",
                origin="doi:10.1021/cm500365c",
            ),
        ),
        bond_terms=SilicaBondTermSet(
            framework_si_o=SilicaBondTerm(
                length_nm=0.16500,
                force_constant=119244.0,
                origin="doi:10.1021/cm500365c",
            ),
            silanol_o_h=SilicaBondTerm(
                length_nm=0.09450,
                force_constant=207108.0,
                origin="doi:10.1021/cm500365c",
            ),
            graft_mount_scaffold_si_o=SilicaBondTerm(
                length_nm=0.16500,
                force_constant=119244.0,
                origin="doi:10.1021/cm500365c",
            ),
        ),
        angle_terms=SilicaAngleTermSet(
            framework_si_o_si=SilicaAngleTerm(
                angle_deg=149.0,
                force_constant=418.4,
                origin="doi:10.1021/cm500365c",
            ),
            silanol_o_si_o=SilicaAngleTerm(
                angle_deg=109.5,
                force_constant=418.4,
                origin="doi:10.1021/cm500365c",
            ),
            silanol_si_o_h=SilicaAngleTerm(
                angle_deg=115.0,
                force_constant=209.2,
                origin="doi:10.1021/cm500365c",
            ),
            graft_scaffold_si_scaffold_o_mount=SilicaAngleTerm(
                angle_deg=149.0,
                force_constant=418.4,
                origin="doi:10.1021/cm500365c",
            ),
            graft_oxygen_mount_oxygen=SilicaAngleTerm(
                angle_deg=109.5,
                force_constant=418.4,
                origin="doi:10.1021/cm500365c",
            ),
        ),
    )


_DEFAULT_SILICA_TOPOLOGY_MODEL = _build_default_silica_topology()


def default_silica_topology():
    """Return a fresh editable copy of the package-default silica model.

    Returns
    -------
    model : SilicaTopologyModel
        Deep-copied editable silica topology model populated from the
        package's current slit-topology defaults.

    Examples
    --------
    >>> import silicams as sms
    >>> model = sms.default_silica_topology()
    >>> _ = model.to_yaml()
    >>> model.atom_assignments.silanol_oxygen.charge = "-0.750000"
    """
    return deepcopy(_DEFAULT_SILICA_TOPOLOGY_MODEL)


@dataclass(frozen=True)
class GromacsAtomType:
    """One ``[ atomtypes ]`` row.

    Parameters
    ----------
    name : str
        GROMACS atom-type identifier.
    atomic_number : int or None
        Optional atomic number token.
    mass : str
        Mass token written in the atom-type row.
    charge : str
        Charge token written in the atom-type row.
    particle_type : str
        Particle-type token such as ``"A"``.
    sigma : str
        Lennard-Jones sigma token.
    epsilon : str
        Lennard-Jones epsilon token.
    """

    name: str
    atomic_number: int | None
    mass: str
    charge: str
    particle_type: str
    sigma: str
    epsilon: str

    def is_compatible_with(self, other):
        """Return whether two atom-type definitions are identical.

        Parameters
        ----------
        other : GromacsAtomType
            Atom-type definition to compare.

        Returns
        -------
        is_compatible : bool
            True when all fields except object identity match exactly.
        """
        return (
            self.name == other.name
            and self.atomic_number == other.atomic_number
            and self.mass == other.mass
            and self.charge == other.charge
            and self.particle_type == other.particle_type
            and self.sigma == other.sigma
            and self.epsilon == other.epsilon
        )


@dataclass(frozen=True)
class GromacsAtom:
    """One ``[ atoms ]`` row inside a ``moleculetype``.

    Parameters
    ----------
    index : int
        One-based atom index inside the molecule type.
    atom_type : str
        GROMACS atom-type identifier.
    residue_number : int
        One-based residue number written in the atom row.
    residue_name : str
        Residue name written in the atom row.
    atom_name : str
        Atom name written in the atom row.
    charge_group : int
        Charge-group identifier.
    charge : str
        Charge token written in the atom row.
    mass : str or None
        Optional mass token.
    """

    index: int
    atom_type: str
    residue_number: int
    residue_name: str
    atom_name: str
    charge_group: int
    charge: str
    mass: str | None = None


@dataclass(frozen=True)
class GromacsBond:
    """One ``[ bonds ]`` row.

    Parameters
    ----------
    atom_a : int
        One-based first atom index.
    atom_b : int
        One-based second atom index.
    parameters : GromacsBondParameters
        Bond function and parameter tokens.
    """

    atom_a: int
    atom_b: int
    parameters: GromacsBondParameters


@dataclass(frozen=True)
class GromacsPair:
    """One ``[ pairs ]`` row.

    Parameters
    ----------
    atom_a : int
        One-based first atom index.
    atom_b : int
        One-based second atom index.
    function : int
        GROMACS pair function type.
    parameters : tuple[str, ...], optional
        Optional parameter tokens written after ``function``.
    """

    atom_a: int
    atom_b: int
    function: int
    parameters: tuple[str, ...] = ()


@dataclass(frozen=True)
class GromacsAngle:
    """One ``[ angles ]`` row.

    Parameters
    ----------
    atom_a : int
        One-based first outer atom index.
    atom_b : int
        One-based central atom index.
    atom_c : int
        One-based second outer atom index.
    parameters : GromacsAngleParameters
        Angle function and parameter tokens.
    """

    atom_a: int
    atom_b: int
    atom_c: int
    parameters: GromacsAngleParameters


@dataclass(frozen=True)
class GromacsDihedral:
    """One ``[ dihedrals ]`` row.

    Parameters
    ----------
    atom_a : int
        One-based first atom index.
    atom_b : int
        One-based second atom index.
    atom_c : int
        One-based third atom index.
    atom_d : int
        One-based fourth atom index.
    function : int
        GROMACS dihedral function type.
    parameters : tuple[str, ...], optional
        Optional parameter tokens written after ``function``.
    """

    atom_a: int
    atom_b: int
    atom_c: int
    atom_d: int
    function: int
    parameters: tuple[str, ...] = ()


@dataclass(frozen=True)
class GromacsMoleculeType:
    """Parsed or generated GROMACS molecule-type payload.

    Parameters
    ----------
    name : str
        Molecule-type name.
    nrexcl : int
        ``nrexcl`` value written in ``[ moleculetype ]``.
    atoms : tuple[GromacsAtom, ...]
        Atom rows in molecule-local order.
    bonds : tuple[GromacsBond, ...], optional
        Bond rows.
    pairs : tuple[GromacsPair, ...], optional
        Pair rows.
    angles : tuple[GromacsAngle, ...], optional
        Angle rows.
    dihedrals : tuple[GromacsDihedral, ...], optional
        Dihedral rows.
    """

    name: str
    nrexcl: int
    atoms: tuple[GromacsAtom, ...]
    bonds: tuple[GromacsBond, ...] = ()
    pairs: tuple[GromacsPair, ...] = ()
    angles: tuple[GromacsAngle, ...] = ()
    dihedrals: tuple[GromacsDihedral, ...] = ()


@dataclass(frozen=True)
class ParsedTopologyBundle:
    """One parsed self-contained flat GROMACS topology bundle.

    Parameters
    ----------
    source_path : str
        Source path from which the topology was parsed.
    atomtypes : tuple[GromacsAtomType, ...]
        Parsed atom-type definitions.
    moleculetype : GromacsMoleculeType
        Parsed molecule-type payload.
    """

    source_path: str
    atomtypes: tuple[GromacsAtomType, ...]
    moleculetype: GromacsMoleculeType
    atom_index_by_name: dict[str, int] = field(init=False, repr=False)
    bond_lookup: dict[tuple[str, str], GromacsBond] = field(init=False, repr=False)
    angle_lookup: dict[tuple[str, str, str], GromacsAngle] = field(init=False, repr=False)

    def __post_init__(self):
        """Build local lookup tables and validate unique atom names.

        Raises
        ------
        ValueError
            Raised when atom names are duplicated inside the parsed molecule.
        """
        atom_index_by_name = {}
        for atom in self.moleculetype.atoms:
            if atom.atom_name in atom_index_by_name:
                raise ValueError(
                    "Flat ligand topologies require unique atom names. "
                    f"Found duplicate atom name {atom.atom_name!r} in "
                    f"{self.source_path!r}."
                )
            atom_index_by_name[atom.atom_name] = atom.index
        object.__setattr__(self, "atom_index_by_name", atom_index_by_name)

        atoms_by_index = {
            atom.index: atom
            for atom in self.moleculetype.atoms
        }

        bond_lookup = {}
        for bond in self.moleculetype.bonds:
            name_a = atoms_by_index[bond.atom_a].atom_name
            name_b = atoms_by_index[bond.atom_b].atom_name
            bond_lookup[tuple(sorted((name_a, name_b)))] = bond
        object.__setattr__(self, "bond_lookup", bond_lookup)

        angle_lookup = {}
        for angle in self.moleculetype.angles:
            name_a = atoms_by_index[angle.atom_a].atom_name
            name_b = atoms_by_index[angle.atom_b].atom_name
            name_c = atoms_by_index[angle.atom_c].atom_name
            key = (
                name_a if name_a <= name_c else name_c,
                name_b,
                name_c if name_a <= name_c else name_a,
            )
            angle_lookup[key] = angle
        object.__setattr__(self, "angle_lookup", angle_lookup)

    def atom_by_name(self, atom_name):
        """Return one parsed atom row by atom name.

        Parameters
        ----------
        atom_name : str
            Atom name to look up.

        Returns
        -------
        atom : GromacsAtom
            Matching parsed atom row.

        Raises
        ------
        KeyError
            Raised when ``atom_name`` is not present in the bundle.
        """
        atom_index = self.atom_index_by_name[atom_name]
        for atom in self.moleculetype.atoms:
            if atom.index == atom_index:
                return atom
        raise KeyError(atom_name)

    def has_atom_name(self, atom_name):
        """Return whether the bundle contains one atom name.

        Parameters
        ----------
        atom_name : str
            Atom name to check.

        Returns
        -------
        contains : bool
            True when ``atom_name`` is present in the bundle.
        """
        return atom_name in self.atom_index_by_name

    def total_charge(self):
        """Return the total molecular charge parsed from the bundle.

        Returns
        -------
        total_charge : float
            Sum of the parsed ``[ atoms ]`` charge tokens converted to
            floating-point values.
        """
        return sum(float(atom.charge) for atom in self.moleculetype.atoms)

    def bond_by_names(self, atom_name_a, atom_name_b):
        """Return one bond definition by atom names.

        Parameters
        ----------
        atom_name_a : str
            First atom name.
        atom_name_b : str
            Second atom name.

        Returns
        -------
        bond : GromacsBond or None
            Matching bond row when present.
        """
        return self.bond_lookup.get(tuple(sorted((atom_name_a, atom_name_b))))

    def angle_by_names(self, atom_name_a, atom_name_b, atom_name_c):
        """Return one angle definition by atom names.

        Parameters
        ----------
        atom_name_a : str
            First outer atom name.
        atom_name_b : str
            Central atom name.
        atom_name_c : str
            Second outer atom name.

        Returns
        -------
        angle : GromacsAngle or None
            Matching angle row when present.
        """
        key = (
            atom_name_a if atom_name_a <= atom_name_c else atom_name_c,
            atom_name_b,
            atom_name_c if atom_name_a <= atom_name_c else atom_name_a,
        )
        return self.angle_lookup.get(key)


def _strip_comment(line):
    """Return one topology line without trailing semicolon comments.

    Parameters
    ----------
    line : str
        Raw topology line.

    Returns
    -------
    stripped : str
        Line content before the first semicolon.
    """
    return line.split(";", 1)[0].strip()


def _read_section_rows(path):
    """Parse a flat topology document into section-token rows.

    Parameters
    ----------
    path : str or Path
        Topology file path.

    Returns
    -------
    sections : dict[str, list[list[str]]]
        Mapping from lowercase section names to tokenized rows.

    Raises
    ------
    ValueError
        Raised when unsupported preprocessor lines are encountered or when a
        data row appears before the first section header.
    """
    sections = {}
    section_name = None
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = _strip_comment(raw_line)
        if not stripped:
            continue

        if stripped.startswith("#"):
            raise ValueError(
                "Flat slit ligand topology input does not support "
                f"preprocessor lines. Found {stripped!r} in {path!r} "
                f"at line {line_number}."
            )

        if stripped.startswith("[") and stripped.endswith("]"):
            section_name = stripped[1:-1].strip().lower()
            sections.setdefault(section_name, [])
            continue

        if section_name is None:
            raise ValueError(
                f"Topology row {stripped!r} in {path!r} appears before any "
                "section header."
            )

        sections[section_name].append(stripped.split())

    return sections


def _parse_atomtypes(path, rows):
    """Parse ``[ atomtypes ]`` rows.

    Parameters
    ----------
    path : str or Path
        Source path used for error messages.
    rows : list[list[str]]
        Tokenized atom-type rows.

    Returns
    -------
    atomtypes : tuple[GromacsAtomType, ...]
        Parsed atom-type definitions.
    """
    atomtypes = []
    for row in rows:
        if len(row) == 6:
            name, mass, charge, particle_type, sigma, epsilon = row
            atomtypes.append(
                GromacsAtomType(
                    name=name,
                    atomic_number=None,
                    mass=mass,
                    charge=charge,
                    particle_type=particle_type,
                    sigma=sigma,
                    epsilon=epsilon,
                )
            )
            continue
        if len(row) >= 7:
            atomtypes.append(
                GromacsAtomType(
                    name=row[0],
                    atomic_number=int(row[1]),
                    mass=row[2],
                    charge=row[3],
                    particle_type=row[4],
                    sigma=row[5],
                    epsilon=row[6],
                )
            )
            continue

        raise ValueError(
            f"Unsupported [ atomtypes ] row {row!r} in {path!r}. "
            "Expected 6 or 7 tokens."
        )

    return tuple(atomtypes)


def _parse_atoms(path, rows):
    """Parse ``[ atoms ]`` rows.

    Parameters
    ----------
    path : str or Path
        Source path used for error messages.
    rows : list[list[str]]
        Tokenized atom rows.

    Returns
    -------
    atoms : tuple[GromacsAtom, ...]
        Parsed atom rows.
    """
    atoms = []
    for row in rows:
        if len(row) < 7:
            raise ValueError(
                f"Unsupported [ atoms ] row {row!r} in {path!r}. "
                "Expected at least 7 tokens."
            )
        atoms.append(
            GromacsAtom(
                index=int(row[0]),
                atom_type=row[1],
                residue_number=int(row[2]),
                residue_name=row[3],
                atom_name=row[4],
                charge_group=int(row[5]),
                charge=row[6],
                mass=row[7] if len(row) >= 8 else None,
            )
        )
    return tuple(atoms)


def _parse_bonds(path, rows):
    """Parse ``[ bonds ]`` rows.

    Parameters
    ----------
    path : str or Path
        Source path used for error messages.
    rows : list[list[str]]
        Tokenized bond rows.

    Returns
    -------
    bonds : tuple[GromacsBond, ...]
        Parsed bond rows.
    """
    bonds = []
    for row in rows:
        if len(row) < 3:
            raise ValueError(
                f"Unsupported [ bonds ] row {row!r} in {path!r}. "
                "Expected at least 3 tokens."
            )
        bonds.append(
            GromacsBond(
                atom_a=int(row[0]),
                atom_b=int(row[1]),
                parameters=GromacsBondParameters(
                    function=int(row[2]),
                    parameters=tuple(row[3:]),
                ),
            )
        )
    return tuple(bonds)


def _parse_pairs(path, rows):
    """Parse ``[ pairs ]`` rows.

    Parameters
    ----------
    path : str or Path
        Source path used for error messages.
    rows : list[list[str]]
        Tokenized pair rows.

    Returns
    -------
    pairs : tuple[GromacsPair, ...]
        Parsed pair rows.
    """
    pairs = []
    for row in rows:
        if len(row) < 3:
            raise ValueError(
                f"Unsupported [ pairs ] row {row!r} in {path!r}. "
                "Expected at least 3 tokens."
            )
        pairs.append(
            GromacsPair(
                atom_a=int(row[0]),
                atom_b=int(row[1]),
                function=int(row[2]),
                parameters=tuple(row[3:]),
            )
        )
    return tuple(pairs)


def _parse_angles(path, rows):
    """Parse ``[ angles ]`` rows.

    Parameters
    ----------
    path : str or Path
        Source path used for error messages.
    rows : list[list[str]]
        Tokenized angle rows.

    Returns
    -------
    angles : tuple[GromacsAngle, ...]
        Parsed angle rows.
    """
    angles = []
    for row in rows:
        if len(row) < 4:
            raise ValueError(
                f"Unsupported [ angles ] row {row!r} in {path!r}. "
                "Expected at least 4 tokens."
            )
        angles.append(
            GromacsAngle(
                atom_a=int(row[0]),
                atom_b=int(row[1]),
                atom_c=int(row[2]),
                parameters=GromacsAngleParameters(
                    function=int(row[3]),
                    parameters=tuple(row[4:]),
                ),
            )
        )
    return tuple(angles)


def _parse_dihedrals(path, rows):
    """Parse ``[ dihedrals ]`` rows.

    Parameters
    ----------
    path : str or Path
        Source path used for error messages.
    rows : list[list[str]]
        Tokenized dihedral rows.

    Returns
    -------
    dihedrals : tuple[GromacsDihedral, ...]
        Parsed dihedral rows.
    """
    dihedrals = []
    for row in rows:
        if len(row) < 5:
            raise ValueError(
                f"Unsupported [ dihedrals ] row {row!r} in {path!r}. "
                "Expected at least 5 tokens."
            )
        dihedrals.append(
            GromacsDihedral(
                atom_a=int(row[0]),
                atom_b=int(row[1]),
                atom_c=int(row[2]),
                atom_d=int(row[3]),
                function=int(row[4]),
                parameters=tuple(row[5:]),
            )
        )
    return tuple(dihedrals)


def parse_flat_itp(path, moleculetype_name=""):
    """Parse one simple self-contained flat GROMACS ``.itp`` file.

    Parameters
    ----------
    path : str or Path
        Input ``.itp`` file path.
    moleculetype_name : str, optional
        Optional explicit molecule-type name. When provided, the parsed
        ``[ moleculetype ]`` section must match this name.

    Returns
    -------
    bundle : ParsedTopologyBundle
        Parsed self-contained topology bundle.

    Raises
    ------
    ValueError
        Raised when required sections are missing, unsupported sections are
        present, or the flat-input constraints are violated.
    """
    sections = _read_section_rows(path)

    unsupported_sections = sorted(
        section_name
        for section_name in sections
        if section_name not in {
            "atomtypes",
            "moleculetype",
            "atoms",
            "bonds",
            "pairs",
            "angles",
            "dihedrals",
        }
    )
    if unsupported_sections:
        raise ValueError(
            "Flat slit ligand topology input supports only [ atomtypes ], "
            "[ moleculetype ], [ atoms ], [ bonds ], [ pairs ], [ angles ], "
            f"and [ dihedrals ]. Unsupported sections in {path!r}: "
            f"{unsupported_sections}."
        )

    if "moleculetype" not in sections or not sections["moleculetype"]:
        raise ValueError(f"Missing [ moleculetype ] section in {path!r}.")
    if "atoms" not in sections or not sections["atoms"]:
        raise ValueError(f"Missing [ atoms ] section in {path!r}.")

    moleculetype_row = sections["moleculetype"][0]
    if len(moleculetype_row) < 2:
        raise ValueError(
            f"Unsupported [ moleculetype ] row {moleculetype_row!r} in {path!r}. "
            "Expected at least 2 tokens."
        )

    parsed_name = moleculetype_row[0]
    if moleculetype_name and parsed_name != moleculetype_name:
        raise ValueError(
            f"Expected moleculetype {moleculetype_name!r} in {path!r}, "
            f"found {parsed_name!r}."
        )

    bundle = ParsedTopologyBundle(
        source_path=str(path),
        atomtypes=_parse_atomtypes(path, sections.get("atomtypes", [])),
        moleculetype=GromacsMoleculeType(
            name=parsed_name,
            nrexcl=int(moleculetype_row[1]),
            atoms=_parse_atoms(path, sections["atoms"]),
            bonds=_parse_bonds(path, sections.get("bonds", [])),
            pairs=_parse_pairs(path, sections.get("pairs", [])),
            angles=_parse_angles(path, sections.get("angles", [])),
            dihedrals=_parse_dihedrals(path, sections.get("dihedrals", [])),
        ),
    )

    return bundle


def _render_bond_parameters(parameters):
    """Render one bond-parameter payload.

    Parameters
    ----------
    parameters : GromacsBondParameters
        Parameter payload to render.

    Returns
    -------
    text : str
        Rendered function and parameter tokens.
    """
    return " ".join((str(parameters.function), *parameters.parameters))


def _render_angle_parameters(parameters):
    """Render one angle-parameter payload.

    Parameters
    ----------
    parameters : GromacsAngleParameters
        Parameter payload to render.

    Returns
    -------
    text : str
        Rendered function and parameter tokens.
    """
    return " ".join((str(parameters.function), *parameters.parameters))


def render_itp(atomtypes, moleculetype):
    """Render one self-contained GROMACS ``.itp`` document.

    Parameters
    ----------
    atomtypes : list[GromacsAtomType]
        Atom-type definitions written before ``[ moleculetype ]``.
    moleculetype : GromacsMoleculeType
        Molecule-type payload to serialize.

    Returns
    -------
    text : str
        Serialized ``.itp`` document text.
    """
    lines = []

    if atomtypes:
        lines.extend(
            [
                "[ atomtypes ]",
                "; name at.num mass charge ptype sigma epsilon",
            ]
        )
        for atomtype in atomtypes:
            atomic_number = (
                str(atomtype.atomic_number)
                if atomtype.atomic_number is not None
                else ""
            )
            row = [
                atomtype.name,
                atomic_number,
                atomtype.mass,
                atomtype.charge,
                atomtype.particle_type,
                atomtype.sigma,
                atomtype.epsilon,
            ]
            lines.append(" ".join(token for token in row if token != ""))
        lines.append("")

    lines.extend(
        [
            "[ moleculetype ]",
            "; name nrexcl",
            f"{moleculetype.name} {moleculetype.nrexcl}",
            "",
            "[ atoms ]",
            "; nr type resnr resid atom cgnr charge mass",
        ]
    )
    for atom in moleculetype.atoms:
        row = [
            str(atom.index),
            atom.atom_type,
            str(atom.residue_number),
            atom.residue_name,
            atom.atom_name,
            str(atom.charge_group),
            atom.charge,
        ]
        if atom.mass is not None:
            row.append(atom.mass)
        lines.append(" ".join(row))

    if moleculetype.bonds:
        lines.extend(["", "[ bonds ]", "; ai aj funct params"])
        for bond in moleculetype.bonds:
            lines.append(
                f"{bond.atom_a} {bond.atom_b} {_render_bond_parameters(bond.parameters)}"
            )

    if moleculetype.pairs:
        lines.extend(["", "[ pairs ]", "; ai aj funct params"])
        for pair in moleculetype.pairs:
            row = [str(pair.atom_a), str(pair.atom_b), str(pair.function), *pair.parameters]
            lines.append(" ".join(row))

    if moleculetype.angles:
        lines.extend(["", "[ angles ]", "; ai aj ak funct params"])
        for angle in moleculetype.angles:
            lines.append(
                f"{angle.atom_a} {angle.atom_b} {angle.atom_c} "
                f"{_render_angle_parameters(angle.parameters)}"
            )

    if moleculetype.dihedrals:
        lines.extend(["", "[ dihedrals ]", "; ai aj ak al funct params"])
        for dihedral in moleculetype.dihedrals:
            row = [
                str(dihedral.atom_a),
                str(dihedral.atom_b),
                str(dihedral.atom_c),
                str(dihedral.atom_d),
                str(dihedral.function),
                *dihedral.parameters,
            ]
            lines.append(" ".join(row))

    lines.append("")
    return "\n".join(lines)


def render_top(include_filename, system_name, molecule_name):
    """Render one simple master ``.top`` document.

    Parameters
    ----------
    include_filename : str
        Included ``.itp`` filename.
    system_name : str
        Human-readable system label.
    molecule_name : str
        Molecule-type name listed in ``[ molecules ]``.

    Returns
    -------
    text : str
        Serialized ``.top`` document text.
    """
    return "\n".join(
        [
            "[ defaults ]",
            "; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ",
            "1 2 yes 0.5 0.833333",
            "",
            f"#include \"{include_filename}\"",
            "",
            "[ system ]",
            system_name,
            "",
            "[ molecules ]",
            f"{molecule_name} 1",
            "",
        ]
    )
