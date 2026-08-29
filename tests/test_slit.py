from dataclasses import asdict, dataclass, replace
import importlib.util
import inspect
import json
import os
from pathlib import Path
import tomllib

import numpy as np
import pytest
import yaml

import silicams as sms
import silicams.generic as generic
import silicams.geometry as geometry
import silicams.slit as slit_mod
import silicams.utils as utils
import silicams.writers.common as snapshot_mod
from silicams.connectivity import AssembledStructureGraph, ConnectivityValidationReport
import silicams.topology as topo_mod
from silicams._version import __version__ as EXPECTED_VERSION


def experimental_target_from_surface(surface_target, surface_silicon_fraction):
    """Build an experimental all-silicon target from surface-only fractions.

    Parameters
    ----------
    surface_target : SiliconStateFractions
        Desired surface-only silicon-state fractions.
    surface_silicon_fraction : float
        Physical surface-to-total silicon fraction used for the
        back-conversion and stored on the experimental target.

    Returns
    -------
    target : ExperimentalSiliconStateTarget
        Experimental all-silicon target that maps back to ``surface_target``
        when the same physical surface-silicon fraction is applied.
    """
    alpha = surface_silicon_fraction
    return sms.ExperimentalSiliconStateTarget(
        q2_fraction=alpha * surface_target.q2_fraction,
        q3_fraction=alpha * surface_target.q3_fraction,
        q4_fraction=alpha * surface_target.q4_fraction + (1.0 - alpha),
        t2_fraction=alpha * surface_target.t2_fraction,
        t3_fraction=alpha * surface_target.t3_fraction,
        surface_silicon_fraction=surface_silicon_fraction,
    )


def test_topology_parameter_helpers_are_exported_from_package_root():
    assert sms.GromacsAngleParameters is topo_mod.GromacsAngleParameters
    assert sms.GromacsBondParameters is topo_mod.GromacsBondParameters


@pytest.mark.parametrize(
    "symbol",
    (
        "Store",
        "Pore",
        "PoreKit",
        "PoreCylinder",
        "PoreSlit",
        "PoreCapsule",
        "PoreAmorphCylinder",
    ),
)
def test_removed_domain_and_writer_symbols_are_not_public(symbol):
    assert not hasattr(sms, symbol)


def test_removed_modules_and_generated_documentation_are_absent():
    assert importlib.util.find_spec("porems") is None
    assert not Path("silicams/system.py").exists()
    assert not Path("silicams/store.py").exists()
    assert not Path("silicams/pore.py").exists()
    assert Path("silicams/_silica_engine.py").exists()
    assert not Path("fill_pore").exists()
    assert not Path("setup.py").exists()
    assert not Path("docs/generated").exists()
    assert not Path("docs/pore.rst").exists()


def test_removed_writer_name_has_no_compatibility_alias():
    assert hasattr(sms.GromacsTopologyWriter, "write_helper_topology")
    assert not hasattr(sms.GromacsTopologyWriter, "write_legacy_topology")


def test_only_silicams_cli_entry_points_are_packaged():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["scripts"] == {
        "silicams-fill-slit": "silicams.slit_fill:_fill_slit_console_main",
        "silicams-slit-density": (
            "silicams.slit_fill:_estimate_guest_density_console_main"
        ),
    }
    assert not any(name.startswith("porems-") for name in project["scripts"])


@dataclass(frozen=True)
class _AngleRecordProbe:
    residue_name: str
    atom_name: str
    atom_type: str


def itp_atom_rows(itp_path):
    """Return parsed ``[ atoms ]`` rows from one topology file.

    Parameters
    ----------
    itp_path : str or Path
        Path to the generated ``.itp`` file.

    Returns
    -------
    rows : list[tuple[str, str, str, float]]
        Parsed rows as ``(atom_type, residue_name, atom_name, charge)``.
    """
    rows = []
    section = None
    with open(itp_path, "r") as file_in:
        for line in file_in:
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                section = stripped.strip("[]").strip().lower()
                continue
            if section == "atoms":
                fields = stripped.split()
                if len(fields) >= 7:
                    rows.append((fields[1], fields[3], fields[4], float(fields[6])))
    return rows


def graph_without_bond(graph, atom_a, atom_b):
    """Return a copy of ``graph`` with one bond removed.

    Parameters
    ----------
    graph : AssembledStructureGraph
        Source graph whose bond list should be filtered.
    atom_a : int
        First atom id of the bond to remove.
    atom_b : int
        Second atom id of the bond to remove.

    Returns
    -------
    graph : AssembledStructureGraph
        New graph with every matching bond removed.
    """
    bond_key = tuple(sorted((atom_a, atom_b)))
    filtered_bonds = [
        bond
        for bond in graph.bonds
        if (bond.atom_a, bond.atom_b) != bond_key
    ]
    return AssembledStructureGraph.from_bonds(graph.atom_ids, filtered_bonds)


def bundled_tms_template_path():
    """Return the checked-in TMS flat-topology template path.

    Returns
    -------
    itp_path : Path
        Package path to the checked-in TMS flat-topology template.
    """
    return Path(sms.__file__).resolve().parent / "templates" / "tms_slit.itp"


def explicit_tms_geminal_cross_terms():
    """Return explicit generated geminal cross terms for TMS export tests.

    Returns
    -------
    cross_terms : SilaneGeminalCrossTerms
        Deterministic geminal cross terms used by the functionalized export
        tests.
    """
    return sms.SilaneGeminalCrossTerms(
        first_ligand_atom_name="O1",
        geminal_oxygen_mount_ligand_angle=topo_mod.GromacsAngleParameters.harmonic(
            angle_deg=117.65432,
            force_constant=432.123456,
        ),
        geminal_dihedrals=(
            sms.GeminalMountDihedralSpec(
                fourth_atom_name="Si2",
                function=1,
                parameters=("12.34567", "0.98765", "2"),
            ),
        ),
    )


def explicit_tms_topology_config(
    tmp_path,
    total_charge=0.825,
    include_geminal_terms=True,
    source_itp_path=None,
):
    """Return an explicit TMS topology config for full-slab export tests.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory used to materialize corrected flat topology files.
    total_charge : float, optional
        Target total charge written onto the base parsed TMS fragment.
    include_geminal_terms : bool, optional
        True to include explicit generated geminal cross terms on the returned
        config.
    source_itp_path : Path or None, optional
        Optional explicit source ``.itp`` path. When omitted, the checked-in
        TMS template is used as the source text.

    Returns
    -------
    topology : SilaneTopologyConfig
        Explicit TMS topology config suitable for the functionalized slit
        exporter.
    """
    source_itp_path = (
        bundled_tms_template_path()
        if source_itp_path is None
        else Path(source_itp_path)
    )
    bundle = topo_mod.parse_flat_itp(
        source_itp_path,
        moleculetype_name="TMS",
    )
    charge_delta = total_charge - bundle.total_charge()
    corrected_atoms = []
    for atom in bundle.moleculetype.atoms:
        charge = float(atom.charge)
        if atom.atom_name == "Si1":
            charge += charge_delta
        corrected_atoms.append(
            replace(
                atom,
                charge=f"{charge:.6f}",
            )
        )

    corrected_bundle_path = Path(tmp_path) / "tms_explicit_charge_target.itp"
    with open(corrected_bundle_path, "w") as file_out:
        file_out.write(
            topo_mod.render_itp(
                bundle.atomtypes,
                replace(bundle.moleculetype, atoms=tuple(corrected_atoms)),
            )
        )

    topology_kwargs = {
        "itp_path": str(corrected_bundle_path),
        "moleculetype_name": "TMS",
        "geminal_cross_terms": (
            explicit_tms_geminal_cross_terms()
            if include_geminal_terms
            else None
        ),
    }
    return sms.SilaneTopologyConfig(**topology_kwargs)


def cif_loop_rows(cif_text, first_tag):
    """Return one simple whitespace-delimited mmCIF loop.

    Parameters
    ----------
    cif_text : str
        Full mmCIF document text.
    first_tag : str
        First tag of the loop that should be extracted.

    Returns
    -------
    result : tuple[list[str], list[list[str]]]
        Tuple ``(tags, rows)`` for the requested loop.

    Raises
    ------
    AssertionError
        Raised when the requested loop is not present in ``cif_text``.
    """
    lines = [line.rstrip() for line in cif_text.splitlines()]
    index = 0
    while index < len(lines):
        if lines[index] != "loop_":
            index += 1
            continue

        index += 1
        tags = []
        while index < len(lines) and lines[index].startswith("_"):
            tags.append(lines[index])
            index += 1

        rows = []
        while index < len(lines):
            line = lines[index]
            if not line or line == "#" or line == "loop_":
                break
            rows.append(line.split())
            index += 1

        if tags and tags[0] == first_tag:
            return tags, rows

    raise AssertionError(f"mmCIF loop starting with {first_tag!r} was not found.")


def naive_slit_adjacency(kit, site_ids, distance_range):
    """Return an independently vectorized periodic adjacency reference.

    The reference applies the orthorhombic minimum-image equation directly
    instead of calling the production pairwise-distance helper. Only pairs in
    the requested distance shell are inspected for shared bonded neighbors.

    Parameters
    ----------
    kit : SilicaSlit
        Slit system providing coordinates and bonded neighbors.
    site_ids : list[int]
        Silicon site identifiers included in the reference graph.
    distance_range : tuple[float, float]
        Inclusive silicon-pair distance shell in nanometers.

    Returns
    -------
    adjacency : dict[int, list[tuple[int, float]]]
        Sorted periodic adjacency entries keyed by site identifier.
    """

    adjacency = {site: [] for site in site_ids}
    positions = np.asarray([kit.atom_position(site) for site in site_ids], dtype=float)
    box = np.asarray(kit.box_nm, dtype=float)
    displacement = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    displacement = np.mod(displacement + 0.5 * box, box) - 0.5 * box
    distances = np.sqrt(np.einsum("ijk,ijk->ij", displacement, displacement))
    distance_mask = (
        (distances >= distance_range[0])
        & (distances <= distance_range[1])
        & np.triu(np.ones(distances.shape, dtype=bool), k=1)
    )
    neighbor_sets = {
        site_id: set(kit.atom_neighbors(site_id))
        for site_id in site_ids
    }

    row_indices, column_indices = np.where(distance_mask)
    for row_index, column_index in zip(row_indices.tolist(), column_indices.tolist()):
        site_a = site_ids[row_index]
        site_b = site_ids[column_index]
        if neighbor_sets[site_a] & neighbor_sets[site_b]:
            continue
        distance = float(distances[row_index, column_index])
        adjacency[site_a].append((site_b, distance))
        adjacency[site_b].append((site_a, distance))

    for site in adjacency:
        adjacency[site].sort(key=lambda item: (item[1], item[0]))

    return adjacency


def naive_bridge_local_ids(kit, pair):
    """Return the local steric graph used by the original bridge scorer."""

    frontier = list(pair)
    local_ids = set(pair)
    for _depth in range(slit_mod._BRIDGE_STERIC_GRAPH_DEPTH):
        next_frontier = []
        for atom_id in frontier:
            for neighbor_id in kit.atom_neighbors(atom_id):
                if neighbor_id not in local_ids:
                    local_ids.add(neighbor_id)
                    next_frontier.append(neighbor_id)
        if not next_frontier:
            break
        frontier = next_frontier
    return local_ids


def naive_bridge_clearance(kit, pair, bridge_position, local_only):
    """Return an independently assembled vectorized clearance reference.

    Parameters
    ----------
    kit : SilicaSlit
        Slit system providing atom positions, types, and bonding.
    pair : tuple[int, int]
        Silicon sites joined by the candidate bridge oxygen.
    bridge_position : sequence of float
        Candidate bridge position in nanometers.
    local_only : bool
        Whether to inspect only the bonded local environment.

    Returns
    -------
    clearance : float
        Minimum candidate-to-atom clearance in nanometers.
    """

    box = kit.box_nm
    sites = kit.binding_sites
    consumed_oxygen_ids = {
        sites[pair[0]].oxygen_ids[0],
        sites[pair[1]].oxygen_ids[0],
    }
    excluded_ids = {
        pair[0],
        pair[1],
        *consumed_oxygen_ids,
    }

    selected_ids = naive_bridge_local_ids(kit, pair) if local_only else kit.active_atom_ids()
    atom_ids = [atom_id for atom_id in selected_ids if atom_id not in excluded_ids]
    if not atom_ids:
        return slit_mod._BRIDGE_STERIC_DISTANCE_CUTOFF_NM

    positions = np.asarray([kit.atom_position(atom_id) for atom_id in atom_ids], dtype=float)
    minimum_distances = np.asarray(
        [
            slit_mod._BRIDGE_MIN_CLEARANCE_BY_TYPE_NM.get(
                kit.atom_type(atom_id),
                0.18,
            )
            for atom_id in atom_ids
        ],
        dtype=float,
    )
    displacement = sms.minimum_image_displacements(bridge_position, positions, box)
    local_mask = np.all(
        np.abs(displacement) <= slit_mod._BRIDGE_STERIC_DISTANCE_CUTOFF_NM,
        axis=1,
    )
    if not np.any(local_mask):
        return slit_mod._BRIDGE_STERIC_DISTANCE_CUTOFF_NM

    local_displacement = displacement[local_mask]
    clearances = np.sqrt(
        np.einsum("ij,ij->i", local_displacement, local_displacement)
    )
    clearances -= minimum_distances[local_mask]
    negative_clearances = clearances[clearances < 0.0]
    if negative_clearances.size:
        return float(negative_clearances[0])
    return float(np.min(clearances))


def export_shared_functionalized_result(
    context,
    output_dir,
    config,
    write_object_files=False,
    write_pdb=False,
    write_pdb_conect=True,
    write_cif=False,
    write_cif_bonds=True,
    validate_connectivity="warn",
):
    """Export an independent clone of the shared functionalized result.

    Parameters
    ----------
    context : FunctionalizedSlitContext
        Session context providing the prepared result and clone operation.
    output_dir : str or os.PathLike
        Directory receiving the exported files.
    config : FunctionalizedAmorphousSlitConfig
        Ligand-topology settings used for export.
    write_object_files : bool, optional
        Whether to write serialized object files.
    write_pdb : bool, optional
        Whether to write PDB coordinates.
    write_pdb_conect : bool, optional
        Whether PDB output includes connectivity records.
    write_cif : bool, optional
        Whether to write mmCIF coordinates.
    write_cif_bonds : bool, optional
        Whether mmCIF output includes bond records.
    validate_connectivity : str, optional
        Connectivity-validation mode used by structure writers.

    Returns
    -------
    result : FunctionalizedSlitResult
        Independently finalized and exported functionalized result.
    """

    progress_tracker = slit_mod._FunctionalizedProgressTracker(
        total_stages=2,
        progress_config=sms.FunctionalizedSlitProgressConfig(enabled=False),
    )
    result = replace(
        context.copy_finalized_result(),
        silica_topology=slit_mod.resolve_silica_topology(config.slit_config),
    )
    try:
        return slit_mod._write_prepared_functionalized_result(
            result=result,
            output_dir=output_dir,
            config=config,
            progress_tracker=progress_tracker,
            write_object_files=write_object_files,
            write_pdb=write_pdb,
            write_pdb_conect=write_pdb_conect,
            write_cif=write_cif,
            write_cif_bonds=write_cif_bonds,
            validate_connectivity=validate_connectivity,
        )
    finally:
        progress_tracker.close()


def build_uncondensed_slit(config):
    """Build a slit before custom Q-state enforcement.

    Parameters
    ----------
    config : AmorphousSlitConfig
        Slit preparation settings used to build the raw silanol surface.

    Returns
    -------
    system : SilicaSlit
        Prepared slit system before target-specific condensation.
    """

    return slit_mod._build_base_slit_system(config).system


class RecordingProgressBar:
    """Simple test double for slit progress-bar instrumentation tests."""

    def __init__(self, total, desc, unit, leave):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.leave = leave
        self.n = 0
        self.closed = False
        self.descriptions = [desc]

    def update(self, value=1):
        self.n += value

    def set_description_str(self, desc, refresh=True):
        del refresh
        self.desc = desc
        self.descriptions.append(desc)

    def close(self):
        self.closed = True


@pytest.fixture(scope="class")
def _bare_slit_case_context(request, bare_slit_context):
    """Expose the shared bare-slit context as class attributes."""

    if request.cls is None:
        return

    request.cls.output_dir = str(bare_slit_context.output_dir)
    request.cls.surface_target = bare_slit_context.surface_target
    request.cls.config = bare_slit_context.config
    request.cls.prepared_result = bare_slit_context.prepared_result
    request.cls.prepared_report = bare_slit_context.prepared_report
    request.cls.stored_result = bare_slit_context.stored_result
    request.cls.stored_report = bare_slit_context.stored_report


@pytest.mark.usefixtures("_bare_slit_case_context")
@pytest.mark.xdist_group("large_bare")
class TestAmorphousSlitPreparation:
    def test_periodic_slit_geometry(self):
        assert self.prepared_report.site_ex == 0
        assert isinstance(self.prepared_result.system, sms.SilicaSlit)
        assert isinstance(
            next(iter(self.prepared_result.system.binding_sites.values())),
            sms.SlitBindingSite,
        )

        expected_box = [9.605, 19.210, 9.605]
        for actual, expected in zip(
            self.prepared_report.slit_geometry.box_lengths_nm,
            expected_box,
        ):
            assert actual == pytest.approx(expected, abs=10 ** (-(3)))

        assert self.prepared_report.requested_slit_width_nm == pytest.approx(
            7.0,
            abs=10 ** (-(3)),
        )
        assert self.prepared_report.slit_geometry.normal_axis_index == 1
        assert self.prepared_report.slit_geometry.plane_separation_nm == pytest.approx(
            self.prepared_report.requested_slit_width_nm,
            abs=1.0,
        )
        assert bool(
            self.prepared_report.slit_geometry.contains_positions(
                self.prepared_result.system.centroid_nm
            )
        )
        metadata = self.prepared_result.system.metadata()
        assert metadata["schema_version"] == 1
        assert sms.PeriodicSlitGeometry.from_dict(metadata["slit_geometry"]) == (
            self.prepared_report.slit_geometry
        )
        assert metadata["slit_geometry"]["projected_area_per_face_nm2"] == pytest.approx(
            expected_box[0] * expected_box[2]
        )
        assert metadata["slit_geometry"]["total_projected_surface_area_nm2"] == pytest.approx(
            2.0 * expected_box[0] * expected_box[2]
        )
        assert self.prepared_report.siloxane_distance_range_nm == (0.4, 0.65)

    def test_default_silica_topology_returns_independent_copies_with_provenance(self):
        model_a = sms.default_silica_topology()
        model_b = sms.default_silica_topology()

        assert isinstance(model_a, sms.SilicaTopologyModel)
        assert isinstance(model_b, sms.SilicaTopologyModel)
        assert model_a is not model_b
        assert model_a.bond_terms.framework_si_o is not model_b.bond_terms.framework_si_o

        model_a.bond_terms.framework_si_o.force_constant = 123456.0

        assert model_b.bond_terms.framework_si_o.force_constant == pytest.approx(119244.0)
        assert model_b.atomtypes.framework_silicon.origin == "doi:10.1021/cm500365c"
        assert model_b.angle_terms.graft_oxygen_mount_oxygen.origin == "doi:10.1021/cm500365c"
        assert model_b.angle_terms.graft_scaffold_si_scaffold_o_mount.angle_deg == pytest.approx(149.0)
        assert model_b.angle_terms.graft_oxygen_mount_oxygen.angle_deg == pytest.approx(109.5)

    def test_silica_topology_serialization_helpers_return_readable_structures(self):
        model = sms.default_silica_topology()

        dict_data = model.to_dict()
        json_data = json.loads(model.to_json())
        yaml_data = yaml.safe_load(model.to_yaml())

        assert dict_data["atomtypes"]["framework_silicon"]["name"] == "SI"
        assert dict_data["bond_terms"]["framework_si_o"]["force_constant"] == pytest.approx(119244.0)
        assert "origin" in dict_data["angle_terms"]["graft_oxygen_mount_oxygen"]
        assert json_data == dict_data
        assert yaml_data == dict_data

    def test_bare_results_expose_resolved_silica_topology(self):
        assert isinstance(self.prepared_result.silica_topology, sms.SilicaTopologyModel)
        assert isinstance(self.stored_result.silica_topology, sms.SilicaTopologyModel)
        assert self.prepared_result.bare_charge_diagnostics is None
        assert isinstance(self.stored_result.bare_charge_diagnostics, sms.BareSilicaChargeDiagnostics)
        assert self.prepared_result.silica_topology.bond_terms.framework_si_o.force_constant == pytest.approx(119244.0)

    def test_bare_charge_diagnostics_match_surface_roles_and_are_neutral(self):
        diagnostics = self.stored_result.bare_charge_diagnostics

        assert diagnostics is not None
        assert diagnostics.is_neutral
        assert diagnostics.coordination_identity_holds
        assert diagnostics.total_charge == pytest.approx(0.0, abs=1e-8)
        assert diagnostics.coordination_identity_delta == 0
        assert diagnostics.silanol_site_count == self.stored_report.final_surface.q3_sites
        assert diagnostics.geminal_site_count == self.stored_report.final_surface.q2_sites
        assert diagnostics.silanol_silicon.atom_count == self.stored_report.final_surface.q3_sites
        assert diagnostics.silanol_oxygen.atom_count == self.stored_report.final_surface.q3_sites
        assert diagnostics.silanol_hydrogen.atom_count == self.stored_report.final_surface.q3_sites
        assert diagnostics.geminal_silicon.atom_count == self.stored_report.final_surface.q2_sites
        assert diagnostics.geminal_oxygen.atom_count == 2 * self.stored_report.final_surface.q2_sites
        assert diagnostics.geminal_hydrogen.atom_count == 2 * self.stored_report.final_surface.q2_sites

    def test_gromacs_writer_bare_charge_diagnostics_matches_stored_result(self):
        diagnostics = sms.GromacsTopologyWriter(
            self.stored_result.system.export_snapshot(),
        ).bare_charge_diagnostics(
            silica_topology=self.stored_result.silica_topology,
        )

        assert diagnostics == self.stored_result.bare_charge_diagnostics

    def test_bare_charge_diagnostics_satisfy_coordination_identity(self):
        diagnostics = self.stored_result.bare_charge_diagnostics

        assert diagnostics is not None
        assert diagnostics.coordination_identity_left == 4 * diagnostics.total_silicon_count
        assert diagnostics.coordination_identity_right == (
            2 * diagnostics.framework_oxygen.atom_count
            + diagnostics.total_hydroxyl_count
        )
        assert diagnostics.coordination_identity_left == diagnostics.coordination_identity_right

    def test_prepared_surface_composition_matches_target(self):
        assert self.prepared_report.final_surface == self.prepared_report.target_surface
        assert self.prepared_report.prepared_surface == self.prepared_report.target_surface
        assert self.prepared_report.prepared_surface.total_surface_si == 954
        assert self.prepared_report.prepared_surface.q2_sites == 66
        assert self.prepared_report.prepared_surface.q3_sites == 650
        assert self.prepared_report.prepared_surface.q4_sites == 238
        assert self.prepared_report.prepared_surface.t2_sites == 0
        assert self.prepared_report.prepared_surface.t3_sites == 0
        assert not (self.prepared_report.used_surface_tolerance)
        assert self.prepared_report.surface_fraction_tolerance == pytest.approx(0.005, abs=1e-7)
        assert self.prepared_report.experimental_target.surface_silicon_fraction == 1.0
        assert self.prepared_report.functionalization_steric_settings is None
        assert self.prepared_report.derived_surface_target == sms.SiliconStateFractions(0.069, 0.681, 0.25)
        assert self.prepared_report.final_surface.q2_fraction == pytest.approx(self.prepared_report.derived_surface_target.q2_fraction, abs=1e-3)
        assert self.prepared_report.final_surface.q3_fraction == pytest.approx(self.prepared_report.derived_surface_target.q3_fraction, abs=1e-3)
        assert self.prepared_report.final_surface.q4_fraction == pytest.approx(self.prepared_report.derived_surface_target.q4_fraction, abs=1e-3)
        assert (
            self.prepared_report.preparation_diagnostics.final_surface_oxygen_handles
            == self.prepared_report.final_surface.q3_sites
            + 2 * self.prepared_report.final_surface.q2_sites
        )
        assert (
            self.prepared_report.preparation_diagnostics.final_framework_oxygen
            == self.prepared_result.system.molecule_counts["OM"]
        )
        assert self.prepared_report.preparation_diagnostics.stripped_silicon_total > 0
        assert self.prepared_report.preparation_diagnostics.removed_orphan_oxygen > 0
        assert self.prepared_report.preparation_diagnostics.inserted_bridge_oxygen == 235
        assert "SLX" not in self.prepared_result.system.molecule_counts

    def test_inserted_bridge_oxygen_respects_local_clearance_threshold(self):
        history = self.prepared_result.system.surface_edit_history
        bridge_ids = [
            record.atom_id
            for record in history
            if record.reason == "inserted_bridge_oxygen"
        ]
        system = self.prepared_result.system
        box = np.asarray(system.box_nm, dtype=float)
        active_atom_ids = np.asarray(system.active_atom_ids(), dtype=int)
        active_positions = np.asarray(
            [system.atom_position(int(atom_id)) for atom_id in active_atom_ids],
            dtype=float,
        )
        minimum_distances = np.asarray(
            [
                slit_mod._BRIDGE_MIN_CLEARANCE_BY_TYPE_NM.get(
                    system.atom_type(int(atom_id)),
                    0.18,
                )
                for atom_id in active_atom_ids
            ],
            dtype=float,
        )

        assert len(bridge_ids) == self.prepared_report.preparation_diagnostics.inserted_bridge_oxygen

        for bridge_id in bridge_ids:
            bonded_ids = set(system.atom_neighbors(bridge_id)) | {bridge_id}
            included_mask = ~np.isin(active_atom_ids, tuple(bonded_ids))
            included_ids = active_atom_ids[included_mask]
            displacement = sms.minimum_image_displacements(
                system.atom_position(bridge_id),
                active_positions[included_mask],
                box,
            )
            local_mask = np.all(
                np.abs(displacement) <= slit_mod._BRIDGE_STERIC_DISTANCE_CUTOFF_NM,
                axis=1,
            )
            if not np.any(local_mask):
                continue

            local_displacement = displacement[local_mask]
            clearances = np.sqrt(
                np.einsum("ij,ij->i", local_displacement, local_displacement)
            )
            clearances -= minimum_distances[included_mask][local_mask]
            minimum_index = int(np.argmin(clearances))
            assert clearances[minimum_index] >= -1e-9, (
                f"Bridge oxygen {bridge_id} is too close to atom "
                f"{int(included_ids[local_mask][minimum_index])}."
            )

    def test_slit_adjacency_matches_naive_reference(self):
        site_ids = sorted(self.prepared_result.system.interior_site_ids)
        adjacency = slit_mod._build_slit_site_adjacency(
            self.prepared_result.system,
            site_ids,
            self.config.siloxane_distance_range_nm,
        )
        reference = naive_slit_adjacency(
            self.prepared_result.system,
            site_ids,
            self.config.siloxane_distance_range_nm,
        )

        adjacency_pairs = {
            tuple(sorted((site_a, site_b)))
            for site_a, neighbors in adjacency.items()
            for site_b, _distance in neighbors
        }
        reference_pairs = {
            tuple(sorted((site_a, site_b)))
            for site_a, neighbors in reference.items()
            for site_b, _distance in neighbors
        }
        assert adjacency_pairs == reference_pairs

    def test_bridge_clearance_matches_naive_reference_for_valid_candidate(self):
        system = self.prepared_result.system
        adjacency = slit_mod._build_slit_site_adjacency(
            system,
            sorted(system.interior_site_ids),
            self.config.siloxane_distance_range_nm,
        )

        pair = None
        bridge_position = None
        for site_a, neighbors in adjacency.items():
            for site_b, _distance in neighbors:
                candidate_pair = (site_a, site_b)
                candidate_position = slit_mod._siloxane_bridge_position(system, candidate_pair)
                if candidate_position is not None:
                    pair = candidate_pair
                    bridge_position = candidate_position
                    break
            if pair is not None:
                break

        assert pair is not None
        assert bridge_position is not None
        assert slit_mod._bridge_steric_score(system, pair, bridge_position) == pytest.approx(
            naive_bridge_clearance(system, pair, bridge_position, local_only=True),
            abs=1e-12,
        )
        assert slit_mod._bridge_global_clearance(system, pair, bridge_position) == pytest.approx(
            naive_bridge_clearance(system, pair, bridge_position, local_only=False),
            abs=1e-12,
        )

    def test_bridge_clearance_matches_naive_reference_for_invalid_candidate(self):
        system = self.prepared_result.system
        adjacency = slit_mod._build_slit_site_adjacency(
            system,
            sorted(system.interior_site_ids),
            self.config.siloxane_distance_range_nm,
        )

        pair = None
        for site_a, neighbors in adjacency.items():
            if neighbors:
                pair = (site_a, neighbors[0][0])
                break

        assert pair is not None
        sites = system.binding_sites
        excluded_ids = {
            pair[0],
            pair[1],
            sites[pair[0]].oxygen_ids[0],
            sites[pair[1]].oxygen_ids[0],
        }
        local_ids = naive_bridge_local_ids(system, pair)
        reference_atom_id = next(
            atom_id
            for atom_id in sorted(local_ids)
            if atom_id not in excluded_ids
        )
        invalid_position = system.atom_position(reference_atom_id)

        local_score = slit_mod._bridge_steric_score(system, pair, invalid_position)
        global_score = slit_mod._bridge_global_clearance(system, pair, invalid_position)
        assert local_score < 0
        assert global_score < 0
        assert local_score == pytest.approx(
            naive_bridge_clearance(system, pair, invalid_position, local_only=True),
            abs=1e-12,
        )
        assert global_score == pytest.approx(
            naive_bridge_clearance(system, pair, invalid_position, local_only=False),
            abs=1e-12,
        )

@pytest.mark.xdist_group("small_bare")
class TestAmorphousSlitBehavior:
    def test_repeat_y_one_reaches_requested_surface_target(self, small_bare_slit_context):
        result = small_bare_slit_context.prepared_result

        assert result.report.site_ex == 0
        assert result.report.requested_slit_width_nm == pytest.approx(7.0)
        assert result.report.slit_geometry.normal_axis_index == 1
        assert result.report.slit_geometry.plane_separation_nm == pytest.approx(7.0, abs=1.0)
        assert bool(
            result.report.slit_geometry.contains_positions(result.system.centroid_nm)
        )
        assert not (result.report.used_surface_tolerance)
        assert result.report.prepared_surface == result.report.target_surface
        assert result.report.final_surface == result.report.target_surface
        assert result.report.prepared_surface.total_surface_si == 957
        assert result.report.prepared_surface.q2_sites == 66
        assert result.report.prepared_surface.q3_sites == 652
        assert result.report.prepared_surface.q4_sites == 239

    def test_physical_surface_fraction_uses_unified_conversion(self):
        reference_surface = sms.SiliconStateFractions(0.069, 0.681, 0.25)
        experimental_target = experimental_target_from_surface(
            reference_surface,
            0.2,
        )
        converted = slit_mod._surface_target_from_experimental(experimental_target)

        assert converted.q2_fraction == pytest.approx(reference_surface.q2_fraction)
        assert converted.q3_fraction == pytest.approx(reference_surface.q3_fraction)
        assert converted.q4_fraction == pytest.approx(reference_surface.q4_fraction)

    def test_surface_conversion_example_matches_expected_q2_enrichment(self):
        target = sms.ExperimentalSiliconStateTarget(
            q2_fraction=0.5,
            q3_fraction=0.0,
            surface_silicon_fraction=0.5,
        )
        surface_target = slit_mod._surface_target_from_experimental(target)

        assert surface_target == sms.SiliconStateFractions(1.0, 0.0, 0.0)

    def test_surface_silicon_fraction_is_mandatory(self):
        with pytest.raises(TypeError, match="surface_silicon_fraction"):
            sms.ExperimentalSiliconStateTarget(
                q2_fraction=0.02,
                q3_fraction=0.03,
            )

    @pytest.mark.parametrize("surface_silicon_fraction", (0.0, -0.1, 1.1, np.nan))
    def test_surface_silicon_fraction_must_be_finite_and_physical(
        self,
        surface_silicon_fraction,
    ):
        with pytest.raises(ValueError, match="surface-silicon fraction"):
            sms.ExperimentalSiliconStateTarget(
                q2_fraction=0.02,
                q3_fraction=0.03,
                surface_silicon_fraction=surface_silicon_fraction,
            )

    def test_random_seed_reproducibly_changes_bare_surface_realization(self):
        target = sms.ExperimentalSiliconStateTarget(
            q2_fraction=307 / 957,
            q3_fraction=650 / 957,
            q4_fraction=0.0,
            surface_silicon_fraction=1.0,
        )

        def bridge_pairs(seed):
            """Return inserted bridge-neighbor pairs for one seeded variant."""
            result = sms.prepare_amorphous_slit_surface(
                sms.AmorphousSlitConfig(
                    name=f"seeded_bare_{seed}",
                    repeat_y=1,
                    surface_target=target,
                    random_seed=seed,
                )
            )
            assert result.report.random_seed == seed
            assert result.report.final_surface == sms.SiliconStateComposition(
                957,
                307,
                650,
                0,
            )
            return tuple(
                record.neighbor_ids
                for record in result.system.surface_edit_history
                if record.reason == "inserted_bridge_oxygen"
            )

        seed_a_pairs = bridge_pairs(11)
        repeated_seed_a_pairs = bridge_pairs(11)
        seed_b_pairs = bridge_pairs(12)

        assert seed_a_pairs == repeated_seed_a_pairs
        assert seed_a_pairs != seed_b_pairs

    def test_q4_fraction_is_derived_when_omitted(self):
        target = sms.ExperimentalSiliconStateTarget(
            q2_fraction=0.02,
            q3_fraction=0.03,
            surface_silicon_fraction=0.5,
            t2_fraction=0.04,
            t3_fraction=0.01,
        )

        assert target.q4_fraction == pytest.approx(0.9, abs=1e-12)
        assert asdict(target)["q4_fraction"] == pytest.approx(0.9, abs=1e-12)

    def test_explicit_q4_fraction_must_match_remainder(self):
        with pytest.raises(ValueError, match="q4 fraction"):
            sms.ExperimentalSiliconStateTarget(
                q2_fraction=0.02,
                q3_fraction=0.03,
                surface_silicon_fraction=1.0,
                q4_fraction=0.89,
                t2_fraction=0.04,
                t3_fraction=0.01,
            )

    def test_invalid_alpha_target_combinations_raise(self):
        with pytest.raises(
            ValueError,
            match=(
                r"Minimum required fraction is 0\.500000, observed 0\.400000"
            ),
        ):
            slit_mod._surface_target_from_experimental(
                sms.ExperimentalSiliconStateTarget(
                    q2_fraction=0.5,
                    q3_fraction=0.0,
                    surface_silicon_fraction=0.4,
                ),
            )

    def test_bridge_algebra_matches_q_state_changes(self, small_bare_slit_context):
        cases = [
            ((2, 2), (-2, 2, 0), (1, 0)),
            ((2, 1), (-1, 0, 1), (1, 1)),
            ((1, 1), (0, -2, 2), (1, 2)),
        ]
        config = sms.AmorphousSlitConfig(
            name="bridge_algebra_case",
            repeat_y=1,
            surface_target=small_bare_slit_context.config.surface_target,
        )
        uncondensed_system = build_uncondensed_slit(config)

        for pair_counts, expected_delta, expected_objectified in cases:
            system = uncondensed_system.clone()
            total_surface_si = len(system.interior_site_ids)
            sites = system.binding_sites
            before = slit_mod._surface_composition(total_surface_si, sites)
            adjacency = slit_mod._build_slit_site_adjacency(
                system,
                sorted(system.interior_site_ids),
                config.siloxane_distance_range_nm,
            )
            pair, bridge_position = slit_mod._find_placeable_pair(
                system,
                sites,
                adjacency,
                *pair_counts,
            )

            assert pair is not None, pair_counts
            assert bridge_position is not None, pair_counts
            om_before = system.molecule_counts.get("OM", 0)
            si_before = system.molecule_counts.get("SI", 0)
            slit_mod._bridge_pair(system, pair, bridge_position=bridge_position)
            slit_mod._consume_pair(adjacency, pair)

            after = slit_mod._surface_composition(
                total_surface_si,
                system.binding_sites,
            )
            assert after.q2_sites - before.q2_sites == expected_delta[0], pair_counts
            assert after.q3_sites - before.q3_sites == expected_delta[1], pair_counts
            assert after.q4_sites - before.q4_sites == expected_delta[2], pair_counts
            assert system.molecule_counts.get("OM", 0) - om_before == expected_objectified[0], pair_counts
            assert system.molecule_counts.get("SI", 0) - si_before == expected_objectified[1], pair_counts
            assert "SLX" not in system.molecule_counts

    def test_tolerance_fallback_selects_nearest_realizable_target(
        self,
        small_bare_slit_context,
    ):
        config = sms.AmorphousSlitConfig(
            name="tolerance_case",
            repeat_y=1,
            surface_target=small_bare_slit_context.config.surface_target,
        )
        system = build_uncondensed_slit(config)
        total_surface_si = len(system.interior_site_ids)
        initial_surface = slit_mod._surface_composition(
            total_surface_si,
            system.binding_sites,
        )
        requested_target = sms.SiliconStateFractions(66 / 957, 653 / 957, 238 / 957)
        exact_target = sms.SiliconStateComposition(
            total_surface_si=total_surface_si,
            q2_sites=66,
            q3_sites=653,
            q4_sites=238,
        )

        attempt = slit_mod._realize_surface_target(
            system,
            total_surface_si,
            initial_surface,
            requested_target,
            exact_target,
            config.surface_fraction_tolerance,
            config.siloxane_distance_range_nm,
            ligand=None,
        )
        errors = slit_mod._surface_fraction_errors(
            attempt.final_surface,
            requested_target,
        )

        assert attempt.used_surface_tolerance
        assert attempt.target_surface == sms.SiliconStateComposition(
                total_surface_si=total_surface_si,
                q2_sites=65,
                q3_sites=654,
                q4_sites=238,
            )
        assert attempt.prepared_surface == attempt.target_surface
        assert attempt.final_surface == attempt.target_surface
        assert all(error <= config.surface_fraction_tolerance for error in errors)

    def test_prepared_slit_remains_attachable(self, small_bare_slit_context):
        system = small_bare_slit_context.clone_result().system
        attachment = system.attach_ligands(
            molecule=generic.tms(),
            mount=0,
            axis=[0, 1],
            site_ids=system.available_site_ids(oxygen_count=1)[:1],
            requested_count=1,
            allow_geminal=False,
            check_sterics=False,
        )

        assert len(attachment.attached_site_ids) == 1
        assert system.molecule_counts["TMS"] == 1

    def test_slit_clone_is_independent_and_binding_sites_are_read_only(
        self,
        small_bare_slit_context,
    ):
        original = small_bare_slit_context.prepared_result.system
        cloned = original.clone()
        site_id = cloned.available_site_ids(oxygen_count=1)[0]

        attachment = cloned.attach_ligands(
            molecule=generic.tms(),
            mount=0,
            axis=(0, 1),
            site_ids=(site_id,),
            check_sterics=False,
        )

        assert attachment.attached_site_ids == (site_id,)
        assert cloned.molecule_counts["TMS"] == 1
        assert "TMS" not in original.molecule_counts
        assert site_id in original.available_site_ids(oxygen_count=1)
        with pytest.raises(TypeError):
            original.binding_sites[site_id] = None

    def test_slit_attachment_reports_rejected_sites(self, small_bare_slit_context):
        system = small_bare_slit_context.clone_result().system
        site_id = system.available_site_ids(oxygen_count=1)[0]

        attachment = system.attach_ligands(
            molecule=generic.tms(),
            mount=0,
            axis=(0, 1),
            site_ids=(site_id,),
            rotate_about_axis=False,
            steric_clearance_scale=1000.0,
        )

        assert attachment.attached_site_ids == ()
        assert attachment.rejected_site_ids == (site_id,)
        assert attachment.molecules == ()

    def test_slit_finalization_is_idempotent_and_blocks_mutation(
        self,
        small_bare_slit_context,
    ):
        system = small_bare_slit_context.clone_result().system
        system.finalize()
        first_snapshot = system.export_snapshot()
        first_counts = dict(system.molecule_counts)

        system.finalize()
        second_snapshot = system.export_snapshot()

        assert system.is_finalized
        assert dict(system.molecule_counts) == first_counts
        assert first_snapshot.name == second_snapshot.name
        assert len(first_snapshot.molecules) == len(second_snapshot.molecules)
        with pytest.raises(ValueError, match="finalized slit"):
            system.attach_ligands(
                molecule=generic.tms(),
                mount=0,
                axis=(0, 1),
                site_ids=(),
            )
        with pytest.raises(ValueError, match="finalized slit"):
            system.insert_siloxane_bridge((0, 1), (0.0, 0.0, 0.0))

    def test_bare_builder_rejects_non_zero_t_states(self):
        with pytest.raises(ValueError):
            sms.prepare_amorphous_slit_surface(
                config=sms.AmorphousSlitConfig(
                    surface_target=sms.ExperimentalSiliconStateTarget(
                        q2_fraction=0.05,
                        q3_fraction=0.05,
                        surface_silicon_fraction=1.0,
                        q4_fraction=0.85,
                        t2_fraction=0.05,
                    )
                )
            )

@pytest.mark.usefixtures("_bare_slit_case_context")
@pytest.mark.xdist_group("large_bare")
class TestStoredAmorphousSlit:
    def test_bare_slit_files_are_written(self):
        expected_files = [
            "test_bare_amorphous_slit.gro",
            "test_bare_amorphous_slit.itp",
            "test_bare_amorphous_slit.top",
            "test_bare_amorphous_slit.yml",
            "test_bare_amorphous_slit_report.json",
        ]
        for file_name in expected_files:
            assert os.path.isfile(os.path.join(self.output_dir, file_name))
        assert not os.path.exists(os.path.join(self.output_dir, "grid.itp"))

        assert not (os.path.exists(os.path.join(self.output_dir, "test_bare_amorphous_slit.obj")))
        assert not (os.path.exists(
                os.path.join(self.output_dir, "test_bare_amorphous_slit_system.obj")
            ))
        with open(os.path.join(self.output_dir, "test_bare_amorphous_slit.top"), "r") as file_in:
            top_text = file_in.read()
        with open(os.path.join(self.output_dir, "test_bare_amorphous_slit.itp"), "r") as file_in:
            itp_text = file_in.read()

        assert '#include "test_bare_amorphous_slit.itp"' in top_text
        assert "TEST_BARE_AMORPHOUS_SLIT 1" in top_text
        assert "[ atomtypes ]" in itp_text
        assert "[ moleculetype ]" in itp_text

        atom_rows = itp_atom_rows(
            os.path.join(self.output_dir, "test_bare_amorphous_slit.itp")
        )
        total_charge = sum(row[3] for row in atom_rows)
        residue_counts = {
            residue_name: sum(1 for _atom_type, residue, _atom_name, _charge in atom_rows if residue == residue_name)
            for residue_name in {"OM", "SI", "SL", "SLG"}
        }

        diagnostics = self.stored_result.bare_charge_diagnostics
        assert diagnostics is not None
        assert total_charge == pytest.approx(0.0, abs=1e-8)
        assert residue_counts["OM"] == diagnostics.framework_oxygen.atom_count
        assert residue_counts["SI"] == diagnostics.framework_silicon.atom_count
        assert residue_counts["SL"] == (
            diagnostics.silanol_silicon.atom_count
            + diagnostics.silanol_oxygen.atom_count
            + diagnostics.silanol_hydrogen.atom_count
        )
        assert residue_counts["SLG"] == (
            diagnostics.geminal_silicon.atom_count
            + diagnostics.geminal_oxygen.atom_count
            + diagnostics.geminal_hydrogen.atom_count
        )

        report_path = os.path.join(
            self.output_dir,
            "test_bare_amorphous_slit_report.json",
        )
        with open(report_path, "r") as file_in:
            data = json.load(file_in)

        assert data["site_ex"] == 0
        assert data["siloxane_distance_range_nm"] == [0.4, 0.65]
        assert data["surface_fraction_tolerance"] == 0.005
        assert data["random_seed"] is None
        assert not (data["used_surface_tolerance"])
        assert data["experimental_target"]["surface_silicon_fraction"] == 1.0
        assert data["functionalization_steric_settings"] is None
        assert data["final_surface"]["q2_sites"] == self.stored_report.final_surface.q2_sites
        assert data["final_surface"]["q3_sites"] == self.stored_report.final_surface.q3_sites
        assert data["final_surface"]["q4_sites"] == self.stored_report.final_surface.q4_sites
        assert data["preparation_diagnostics"]["inserted_bridge_oxygen"] == self.stored_report.preparation_diagnostics.inserted_bridge_oxygen
        assert not (os.path.exists(
                os.path.join(self.output_dir, "test_bare_amorphous_slit_next_steps.md")
            ))

    def test_public_bare_writer_delegates_prepared_result_to_export(self, monkeypatch):
        """The public workflow should compose preparation and export unchanged."""

        prepared_sentinel = object()
        exported_sentinel = object()
        observed = {}

        def fake_prepare(config):
            """Return the prepared-result sentinel and record its config.

            Parameters
            ----------
            config : AmorphousSlitConfig
                Configuration forwarded by the public writer.

            Returns
            -------
            result : object
                Prepared-result sentinel.
            """

            observed["config"] = config
            return prepared_sentinel

        def fake_export(**kwargs):
            """Record export arguments and return the export sentinel.

            Parameters
            ----------
            **kwargs : object
                Export arguments forwarded by the public writer.

            Returns
            -------
            result : object
                Export-result sentinel.
            """

            observed["export"] = kwargs
            return exported_sentinel

        monkeypatch.setattr(slit_mod, "prepare_amorphous_slit_surface", fake_prepare)
        monkeypatch.setattr(slit_mod, "_write_prepared_bare_result", fake_export)
        config = sms.AmorphousSlitConfig(surface_target=self.surface_target)

        result = sms.write_bare_amorphous_slit(
            "unused",
            config=config,
            write_object_files=True,
            write_pdb=True,
            write_pdb_conect=False,
            write_cif=True,
            write_cif_bonds=False,
            validate_connectivity="strict",
        )

        assert result is exported_sentinel
        assert observed["config"] is config
        assert observed["export"] == {
            "result": prepared_sentinel,
            "output_dir": "unused",
            "write_object_files": True,
            "write_pdb": True,
            "write_pdb_conect": False,
            "write_cif": True,
            "write_cif_bonds": False,
            "validate_connectivity": "strict",
        }

    def test_finalized_bare_slit_connectivity_is_valid(self):
        snapshot = self.stored_result.system.export_snapshot()
        report = sms.StructureWriter(snapshot).validate_connectivity(
            use_atom_names=True
        )

        assert report.is_valid

@pytest.mark.xdist_group("small_bare")
class TestStoredSlitFormats:
    def test_shared_snapshot_contains_final_ordering_and_connectivity(
        self,
        small_bare_slit_context,
    ):
        system = small_bare_slit_context.finalized_result.system
        snapshot = system.export_snapshot()
        graph = sms.StructureWriter(snapshot).assembled_graph(use_atom_names=True)

        assert snapshot.has_assembled_export
        assert tuple(atom.serial for atom in snapshot.atom_order) == tuple(
            range(1, len(snapshot.atom_order) + 1)
        )
        assert tuple(
            serial
            for molecule_serials in snapshot.molecule_serials
            for serial in molecule_serials
        ) == tuple(range(1, len(snapshot.atom_order) + 1))
        assert snapshot.assembled_graph == graph
        assert snapshot.assembled_bonds == graph.bonds
        assert snapshot.assembled_angles == graph.angles
        assert any(
            bond.provenance == "siloxane_bridge"
            for bond in snapshot.assembled_bonds
        )
        assert dict(snapshot.residue_counts) == dict(
            system.molecule_counts
        )

    def test_validation_flags_broken_silanol_silicon_environment(
        self,
        small_bare_slit_context,
    ):
        system = small_bare_slit_context.finalized_result.system
        store = sms.StructureWriter(system.export_snapshot())
        cache = store._collect_structure_records(use_atom_names=True)
        graph = store.assembled_graph(use_atom_names=True)
        neighbors = store._connectivity_validation_neighbors(graph)
        record_by_serial = {record.serial: record for record in cache.atom_records}

        silanol_si_serial = next(
            record.serial
            for record in cache.atom_records
            if record.residue_name == "SL" and record.atom_type == "Si"
        )
        broken_neighbor = next(
            neighbor_id
            for neighbor_id in sorted(neighbors[silanol_si_serial])
            if record_by_serial[neighbor_id].atom_type == "O"
            and record_by_serial[neighbor_id].residue_name == "OM"
        )
        broken_graph = graph_without_bond(graph, silanol_si_serial, broken_neighbor)
        findings = store._connectivity_validation_findings(
            cache.atom_records,
            broken_graph,
        )

        assert any(
            finding.code == "silanol_silicon_environment"
            for finding in findings
        )

    def test_validation_flags_broken_geminal_silicon_environment(
        self,
        small_bare_slit_context,
    ):
        system = small_bare_slit_context.finalized_result.system
        store = sms.StructureWriter(system.export_snapshot())
        cache = store._collect_structure_records(use_atom_names=True)
        graph = store.assembled_graph(use_atom_names=True)
        neighbors = store._connectivity_validation_neighbors(graph)
        record_by_serial = {record.serial: record for record in cache.atom_records}

        geminal_si_serial = next(
            record.serial
            for record in cache.atom_records
            if record.residue_name == "SLG" and record.atom_type == "Si"
        )
        broken_neighbor = next(
            neighbor_id
            for neighbor_id in sorted(neighbors[geminal_si_serial])
            if record_by_serial[neighbor_id].atom_type == "O"
            and record_by_serial[neighbor_id].residue_name == "OM"
        )
        broken_graph = graph_without_bond(graph, geminal_si_serial, broken_neighbor)
        findings = store._connectivity_validation_findings(
            cache.atom_records,
            broken_graph,
        )

        assert any(
            finding.code == "geminal_silicon_environment"
            for finding in findings
        )

    def test_validation_flags_broken_silanol_hydroxyl_environment(
        self,
        small_bare_slit_context,
    ):
        system = small_bare_slit_context.finalized_result.system
        store = sms.StructureWriter(system.export_snapshot())
        cache = store._collect_structure_records(use_atom_names=True)
        graph = store.assembled_graph(use_atom_names=True)

        silanol_oxygen_serial = next(
            record.serial
            for record in cache.atom_records
            if record.residue_name == "SL" and record.atom_type == "O"
        )
        silanol_hydrogen_serial = next(
            record.serial
            for record in cache.atom_records
            if record.residue_name == "SL" and record.atom_type == "H"
        )
        broken_graph = graph_without_bond(
            graph,
            silanol_oxygen_serial,
            silanol_hydrogen_serial,
        )
        findings = store._connectivity_validation_findings(
            cache.atom_records,
            broken_graph,
        )

        finding_codes = {finding.code for finding in findings}
        assert "silanol_oxygen_environment" in finding_codes
        assert "silanol_hydrogen_environment" in finding_codes

    def test_bare_slit_object_files_are_opt_in(self, tmp_path, small_bare_slit_context):
        assert inspect.signature(sms.write_bare_amorphous_slit).parameters[
            "write_object_files"
        ].default is False
        output_dir = tmp_path / "bare_amorphous_slit_preparation_with_objects"
        system = small_bare_slit_context.finalized_result.system
        slit_mod._write_slit_structure_outputs(
            system=system,
            output_dir=str(output_dir),
            write_object_files=True,
            write_pdb=False,
            write_pdb_conect=True,
            write_cif=False,
            write_cif_bonds=True,
            validate_connectivity="warn",
        )

        name = small_bare_slit_context.config.name
        assert (output_dir / f"{name}.obj").is_file()
        assert (output_dir / f"{name}_system.obj").is_file()
        snapshot = utils.load(output_dir / f"{name}.obj")
        system = utils.load(output_dir / f"{name}_system.obj")
        assert isinstance(snapshot, snapshot_mod.StructureSnapshot)
        assert isinstance(system, sms.SilicaSlit)
        restored_snapshot = system.export_snapshot()
        assert snapshot.name == restored_snapshot.name
        assert snapshot.atom_order == restored_snapshot.atom_order
        assert snapshot.assembled_bonds == restored_snapshot.assembled_bonds
        assert snapshot.residue_counts == restored_snapshot.residue_counts

    def test_bare_slit_pdb_writes_conect_by_default(
        self,
        tmp_path,
        small_bare_slit_context,
    ):
        assert inspect.signature(sms.write_bare_amorphous_slit).parameters[
            "write_pdb_conect"
        ].default is True
        output_dir = tmp_path / "bare_amorphous_slit_preparation_with_pdb"
        system = small_bare_slit_context.finalized_result.system
        slit_mod._write_slit_structure_outputs(
            system=system,
            output_dir=str(output_dir),
            write_object_files=False,
            write_pdb=True,
            write_pdb_conect=True,
            write_cif=False,
            write_cif_bonds=True,
            validate_connectivity="warn",
        )

        pdb_path = output_dir / f"{small_bare_slit_context.config.name}.pdb"
        assert pdb_path.is_file()

        with open(pdb_path, "r") as file_in:
            pdb_lines = file_in.readlines()

        assert any(line.startswith("HETATM") for line in pdb_lines)
        assert any(line.startswith("CONECT") for line in pdb_lines)

    def test_bare_slit_cif_writes_bonds_by_default(
        self,
        tmp_path,
        small_bare_slit_context,
    ):
        assert inspect.signature(sms.write_bare_amorphous_slit).parameters[
            "write_cif_bonds"
        ].default is True
        output_dir = tmp_path / "bare_amorphous_slit_preparation_with_cif"
        system = small_bare_slit_context.finalized_result.system
        slit_mod._write_slit_structure_outputs(
            system=system,
            output_dir=str(output_dir),
            write_object_files=False,
            write_pdb=False,
            write_pdb_conect=True,
            write_cif=True,
            write_cif_bonds=True,
            validate_connectivity="warn",
        )

        cif_path = output_dir / f"{small_bare_slit_context.config.name}.cif"
        assert cif_path.is_file()

        with open(cif_path, "r") as file_in:
            cif_text = file_in.read()

        assert "_atom_site.Cartn_x" in cif_text
        assert "_struct_conn.id" in cif_text

    def test_explicit_silica_topology_override_changes_bare_itp_terms(
        self,
        tmp_path,
        small_bare_slit_context,
    ):
        output_dir = tmp_path / "bare_amorphous_slit_custom_silica"
        silica_topology = sms.default_silica_topology()
        silica_topology.bond_terms.framework_si_o.force_constant = 123456.0
        resolved_topology = slit_mod.resolve_silica_topology(
            sms.AmorphousSlitConfig(
                surface_target=small_bare_slit_context.config.surface_target,
                silica_topology=silica_topology,
            )
        )
        system = small_bare_slit_context.finalized_result.system
        sms.GromacsTopologyWriter(system.export_snapshot(), str(output_dir)).write_full_slit(
            silica_topology=resolved_topology,
        )

        name = small_bare_slit_context.config.name
        with open(output_dir / f"{name}.itp", "r") as file_in:
            itp_text = file_in.read()

        assert resolved_topology is not silica_topology
        assert resolved_topology.bond_terms.framework_si_o.force_constant == pytest.approx(
            123456.0
        )
        assert "0.16500 123456.000000" in itp_text

    def test_top_level_exports_and_version(self, small_bare_slit_context):
        context = small_bare_slit_context
        assert sms.__version__ == EXPECTED_VERSION
        assert callable(sms.prepare_amorphous_slit_surface)
        assert callable(sms.write_bare_amorphous_slit)
        assert callable(sms.prepare_functionalized_amorphous_slit_surface)
        assert callable(sms.write_functionalized_amorphous_slit)
        assert callable(sms.default_silica_topology)
        assert isinstance(context.config, sms.AmorphousSlitConfig)
        assert isinstance(
            context.config.surface_target,
            sms.ExperimentalSiliconStateTarget,
        )
        assert isinstance(context.prepared_result, sms.SlitPreparationResult)
        assert isinstance(context.prepared_result.report, sms.SlitPreparationReport)
        assert isinstance(
            context.prepared_result.report.prepared_surface,
            sms.SiliconStateComposition,
        )
        assert isinstance(
            context.prepared_result.report.preparation_diagnostics,
            sms.SurfacePreparationDiagnostics,
        )
        assert hasattr(sms, "SiliconStateFractions")
        assert hasattr(sms, "SurfacePreparationDiagnostics")
        assert hasattr(sms, "BareSilicaChargeContribution")
        assert hasattr(sms, "BareSilicaChargeDiagnostics")
        assert hasattr(sms, "FunctionalizedSlitChargeDiagnostics")
        assert hasattr(sms, "SilaneAttachmentConfig")
        assert hasattr(sms, "SlitTimingSummary")
        assert hasattr(sms, "FunctionalizedSlitProgressConfig")
        assert hasattr(sms, "FunctionalizedSlitStericConfig")
        assert hasattr(sms, "FunctionalizedAmorphousSlitConfig")
        assert hasattr(sms, "FunctionalizedSlitResult")
        assert hasattr(sms, "GeminalMountDihedralSpec")
        assert hasattr(sms, "SilaneGeminalCrossTerms")
        assert hasattr(sms, "SilicaAtomTypeModel")
        assert hasattr(sms, "SilicaAtomTypeSet")
        assert hasattr(sms, "SilicaAtomAssignment")
        assert hasattr(sms, "SilicaAtomAssignmentSet")
        assert hasattr(sms, "SilicaBondTerm")
        assert hasattr(sms, "SilicaBondTermSet")
        assert hasattr(sms, "SilicaAngleTerm")
        assert hasattr(sms, "SilicaAngleTermSet")
        assert hasattr(sms, "SilicaTopologyModel")
        assert hasattr(sms, "SilaneTopologyConfig")
        for primitive_name in (
            "Atom",
            "GraphBond",
            "GraphAngle",
            "AttachmentRecord",
            "AssembledStructureGraph",
            "Dice",
            "Matrix",
            "AlphaCristobalit",
            "BetaCristobalit",
            "Cylinder",
            "Sphere",
            "Cuboid",
            "Cone",
            "db",
            "gen",
            "geom",
            "utils",
            "SlitJunctionParameters",
        ):
            assert primitive_name not in sms.__all__


@pytest.mark.xdist_group("functionalized")
class TestFunctionalizedAmorphousSlit:
    def test_resolve_silane_topology_config_requires_explicit_topology_input(self):
        assert slit_mod.resolve_silane_topology_config(None) is None
        assert slit_mod.resolve_silane_topology_config(
            sms.SilaneAttachmentConfig(
                molecule=generic.tms(),
                mount=0,
                axis=(0, 1),
            )
        ) is None

    def test_silane_attachment_config_defaults_to_ten_degree_rotation_scan(self):
        ligand = sms.SilaneAttachmentConfig(
            molecule=generic.tms(),
            mount=0,
            axis=(0, 1),
        )

        assert ligand.rotate_about_axis
        assert ligand.rotate_step_deg == 10.0

    def test_functionalized_steric_config_defaults_to_relaxed_slit_scale(self):
        sterics = sms.FunctionalizedSlitStericConfig()

        assert sterics.enabled
        assert sterics.clearance_scale == pytest.approx(0.60)

    @pytest.mark.parametrize("clearance_scale", (0.0, -0.1, np.nan))
    def test_functionalized_steric_scale_must_be_finite_and_positive(
        self,
        clearance_scale,
    ):
        with pytest.raises(ValueError, match="finite and greater than zero"):
            sms.FunctionalizedSlitStericConfig(clearance_scale=clearance_scale)

    def test_functionalized_progress_config_defaults_to_auto_quiet_leave_false(self):
        progress = sms.FunctionalizedSlitProgressConfig()

        assert progress.enabled is None
        assert not (progress.leave)

    def test_functionalized_config_rejects_tuple_ligand_payload(self):
        with pytest.raises(TypeError, match="SilaneAttachmentConfig"):
            sms.FunctionalizedAmorphousSlitConfig(
                slit_config=sms.AmorphousSlitConfig(
                    surface_target=sms.ExperimentalSiliconStateTarget(
                        q2_fraction=0.05,
                        q3_fraction=0.05,
                        surface_silicon_fraction=1.0,
                    )
                ),
                ligand=(
                    sms.SilaneAttachmentConfig(
                        molecule=generic.tms(),
                        mount=0,
                        axis=(0, 1),
                    ),
                ),
            )

    def test_silane_topology_config_rejects_tuple_geminal_cross_terms(self):
        with pytest.raises(TypeError, match="SilaneGeminalCrossTerms"):
            sms.SilaneTopologyConfig(
                itp_path="demo.itp",
                geminal_cross_terms=(
                    sms.SilaneGeminalCrossTerms(
                        first_ligand_atom_name="C1",
                        geminal_oxygen_mount_ligand_angle=(
                            sms.GromacsAngleParameters.harmonic(
                                angle_deg=109.5,
                                force_constant=418.4,
                            )
                        ),
                    ),
                ),
            )

    def test_mount_ligand_cross_angle_role_classifies_scaffold_and_geminal(self):
        mount = _AngleRecordProbe(residue_name="TPSG", atom_name="Si", atom_type="Si")
        first = _AngleRecordProbe(residue_name="TPSG", atom_name="CA1", atom_type="ca")
        scaffold_oxygen = _AngleRecordProbe(residue_name="OM", atom_name="OM1", atom_type="O")
        geminal_oxygen = _AngleRecordProbe(residue_name="TPSG", atom_name="O1", atom_type="O")

        def is_generated_geminal_record(record, atom_type):
            return record.atom_name == "O1" and atom_type == "O"

        assert snapshot_mod._full_slit_mount_ligand_cross_angle_role(
            scaffold_oxygen,
            mount,
            first,
            ligand_shorts={"TPS", "TPSG"},
            mount_atom_name="Si",
            first_ligand_atom_name="CA1",
            is_generated_geminal_record=is_generated_geminal_record,
        ) == "scaffold"
        assert snapshot_mod._full_slit_mount_ligand_cross_angle_role(
            geminal_oxygen,
            mount,
            first,
            ligand_shorts={"TPS", "TPSG"},
            mount_atom_name="Si",
            first_ligand_atom_name="CA1",
            is_generated_geminal_record=is_generated_geminal_record,
        ) == "geminal"

    def test_progress_auto_mode_is_quiet_under_pytest(self):
        bar = slit_mod._create_progress_bar(
            total=3,
            desc="demo",
            progress_config=sms.FunctionalizedSlitProgressConfig(),
            unit="step",
        )

        assert isinstance(bar, slit_mod._NullProgressBar)

    def test_progress_can_be_forced_off_even_in_interactive_mode(self, monkeypatch):
        monkeypatch.setattr(slit_mod, "_is_interactive_progress_environment", lambda: True)

        bar = slit_mod._create_progress_bar(
            total=3,
            desc="demo",
            progress_config=sms.FunctionalizedSlitProgressConfig(enabled=False),
            unit="step",
        )

        assert isinstance(bar, slit_mod._NullProgressBar)

    def test_progress_can_be_forced_on_in_non_interactive_mode(self, monkeypatch):
        created = []

        def fake_tqdm(**kwargs):
            bar = RecordingProgressBar(
                total=kwargs["total"],
                desc=kwargs["desc"],
                unit=kwargs["unit"],
                leave=kwargs["leave"],
            )
            created.append(bar)
            return bar

        monkeypatch.setattr(slit_mod, "_is_interactive_progress_environment", lambda: False)
        monkeypatch.setattr(slit_mod, "_tqdm_auto", fake_tqdm)

        bar = slit_mod._create_progress_bar(
            total=4,
            desc="forced",
            progress_config=sms.FunctionalizedSlitProgressConfig(enabled=True, leave=True),
            unit="stage",
        )

        assert created
        assert bar is created[0]
        assert created[0].leave
        assert created[0].total == 4

    def test_slit_attachment_default_steric_scale_matches_workflow_default(self):
        steric_parameter = inspect.signature(
            sms.SilicaSlit.attach_ligands
        ).parameters["steric_clearance_scale"]

        assert steric_parameter.default == pytest.approx(0.60)

    def test_functionalized_path_uses_configured_steric_clearance_scale(
        self,
        functionalized_slit_context,
    ):
        recorded_scales = [
            call.steric_clearance_scale
            for call in functionalized_slit_context.attachment_calls
        ]
        assert recorded_scales
        assert 0.55 in recorded_scales

    def test_functionalized_path_batches_attachment_slots(
        self,
        functionalized_slit_context,
    ):
        batched_calls = [
            call
            for call in functionalized_slit_context.attachment_calls
            if call.has_progress_callback
        ]
        assert any(
            call.requested_count == 3
            and call.site_count >= 3
            and call.allow_geminal
            for call in batched_calls
        )
        assert any(
            call.requested_count == 4
            and call.site_count >= 4
            and not call.allow_geminal
            for call in batched_calls
        )

    def test_random_seed_reproducibly_changes_functionalized_graft_sites(self):
        target = sms.ExperimentalSiliconStateTarget(
            q2_fraction=308 / 957,
            q3_fraction=648 / 957,
            q4_fraction=0.0,
            t2_fraction=1 / 957,
            t3_fraction=0.0,
            surface_silicon_fraction=1.0,
        )

        def attachment_sites(seed):
            """Return grafted site ids for one seeded functionalized variant."""
            config = sms.FunctionalizedAmorphousSlitConfig(
                slit_config=sms.AmorphousSlitConfig(
                    name=f"seeded_functionalized_{seed}",
                    repeat_y=1,
                    surface_target=target,
                    random_seed=seed,
                ),
                ligand=sms.SilaneAttachmentConfig(
                    molecule=generic.tms(),
                    mount=0,
                    axis=(0, 1),
                    rotate_about_axis=False,
                ),
            )
            result = sms.prepare_functionalized_amorphous_slit_surface(config)
            assert result.report.random_seed == seed
            assert result.report.final_surface == sms.SiliconStateComposition(
                957,
                308,
                648,
                0,
                1,
                0,
            )
            return tuple(
                record.site_id
                for record in result.system.attachment_records
            )

        seed_a_sites = attachment_sites(21)
        repeated_seed_a_sites = attachment_sites(21)
        seed_b_sites = attachment_sites(22)

        assert seed_a_sites == repeated_seed_a_sites
        assert seed_a_sites != seed_b_sites

    def test_attachment_progress_description_includes_candidate_context(self):
        context = slit_mod._AttachmentPhaseProgressContext(
            phase_name="T2 attachment",
            requested_count=5,
            candidate_index=2,
            total_candidates=7,
        )

        assert (
            slit_mod._attachment_progress_description(context, attached_count=3)
            == "T2 attachment 3/5 attached [candidate 2/7]"
        )

    def test_prepare_progress_creates_outer_and_inner_bars(
        self,
        functionalized_slit_context,
    ):
        result = functionalized_slit_context.prepared_result

        assert result.report.final_surface == sms.SiliconStateComposition(
            957,
            63,
            648,
            239,
            3,
            4,
        )
        stage_bars = [
            bar for bar in functionalized_slit_context.progress_bars
            if bar.unit == "stage"
        ]
        site_bars = [
            bar for bar in functionalized_slit_context.progress_bars
            if bar.unit == "site"
        ]
        assert len(stage_bars) == 1
        assert stage_bars[0].total == 4
        assert stage_bars[0].n == 4
        assert any(desc == "Base slit build" for desc in stage_bars[0].descriptions)
        assert any(desc == "Q-state preparation" for desc in stage_bars[0].descriptions)
        assert any(desc == "T2 attachment" for desc in stage_bars[0].descriptions)
        assert any(desc == "T3 attachment" for desc in stage_bars[0].descriptions)
        assert sorted(bar.total for bar in site_bars) == [3, 4]

    def test_write_progress_includes_finalize_and_export_stages(
        self,
        monkeypatch,
        tmp_path,
        functionalized_slit_context,
    ):
        created = []

        def fake_create_progress_bar(total, desc, progress_config, unit="it"):
            """Create a recording progress bar for public-writer assertions.

            Parameters
            ----------
            total : int
                Expected number of workflow updates.
            desc : str
                Initial progress description.
            progress_config : FunctionalizedSlitProgressConfig
                Accepted for interface parity and otherwise unused.
            unit : str, optional
                Unit label for the progress counter.

            Returns
            -------
            bar : RecordingProgressBar
                Newly retained progress-bar test double.
            """

            del progress_config
            bar = RecordingProgressBar(total=total, desc=desc, unit=unit, leave=False)
            created.append(bar)
            return bar

        monkeypatch.setattr(slit_mod, "_create_progress_bar", fake_create_progress_bar)

        config = replace(
            functionalized_slit_context.config,
            progress_settings=sms.FunctionalizedSlitProgressConfig(enabled=True),
        )

        def fake_prepare(config, progress_tracker):
            """Simulate completed scientific stages using a validated fixture.

            Parameters
            ----------
            config : FunctionalizedAmorphousSlitConfig
                Configuration forwarded by the public writer.
            progress_tracker : _FunctionalizedProgressTracker
                Tracker receiving the four already-validated scientific stages.

            Returns
            -------
            result : FunctionalizedSlitResult
                Independent prepared result from the scientific fixture.
            """

            assert config is not None
            for description in (
                "Base slit build",
                "Q-state preparation",
                "T2 attachment",
                "T3 attachment",
            ):
                progress_tracker.set_stage(description)
                progress_tracker.update_stage(1)
            return functionalized_slit_context.clone_result()

        monkeypatch.setattr(
            slit_mod,
            "_prepare_functionalized_amorphous_slit_surface",
            fake_prepare,
        )

        result = sms.write_functionalized_amorphous_slit(
            str(tmp_path / "functionalized_progress_write"),
            config,
            write_pdb=False,
            write_cif=False,
        )

        assert result.charge_diagnostics is None
        assert result.report.timing_summary.finalize_s > 0
        assert result.report.timing_summary.export_s > 0
        name = functionalized_slit_context.config.slit_config.name
        assert not (tmp_path / "functionalized_progress_write" / f"{name}.top").exists()
        assert not (tmp_path / "functionalized_progress_write" / f"{name}.itp").exists()
        stage_bars = [bar for bar in created if bar.unit == "stage"]
        assert len(stage_bars) == 1
        assert stage_bars[0].total == 6
        assert stage_bars[0].n == 6
        assert any(desc == "Finalize" for desc in stage_bars[0].descriptions)
        assert any(desc == "Finalize/export" for desc in stage_bars[0].descriptions)

    def test_exact_functionalized_target_is_realized(self, functionalized_slit_context):
        result = functionalized_slit_context.prepared_result

        assert isinstance(result.silica_topology, sms.SilicaTopologyModel)
        assert not (result.report.used_surface_tolerance)
        assert result.report.prepared_surface == sms.SiliconStateComposition(957, 66, 652, 239)
        assert result.report.final_surface == sms.SiliconStateComposition(957, 63, 648, 239, 3, 4)
        assert result.report.final_surface == result.report.target_surface
        assert isinstance(result.report.timing_summary, sms.SlitTimingSummary)
        assert result.report.timing_summary.base_slit_build_s > 0
        assert result.report.timing_summary.q_state_preparation_s > 0
        assert result.report.timing_summary.t2_attachment_s > 0
        assert result.report.timing_summary.t3_attachment_s > 0
        assert result.report.functionalization_steric_settings == (
            functionalized_slit_context.config.steric_settings
        )
        assert result.system.attached_state_counts("TMS") == (3, 4)
        assert "SLX" not in result.system.molecule_counts

    def test_functionalized_tolerance_fallback_selects_nearest_realizable_target(self):
        target = sms.ExperimentalSiliconStateTarget(
            q2_fraction=63 / 957,
            q3_fraction=649 / 957,
            q4_fraction=238 / 957,
            t2_fraction=3 / 957,
            t3_fraction=4 / 957,
            surface_silicon_fraction=1.0,
        )
        config = sms.FunctionalizedAmorphousSlitConfig(
            slit_config=sms.AmorphousSlitConfig(
                name="functionalized_tolerance_slit",
                repeat_y=1,
                surface_target=target,
            ),
            ligand=sms.SilaneAttachmentConfig(
                molecule=generic.tms(),
                mount=0,
                axis=(0, 1),
            ),
        )

        result = sms.prepare_functionalized_amorphous_slit_surface(config)

        assert result.report.used_surface_tolerance
        assert result.report.final_surface == sms.SiliconStateComposition(957, 63, 648, 239, 3, 4)
        errors = slit_mod._surface_fraction_errors(
            result.report.final_surface,
            result.report.derived_surface_target,
        )
        assert all(error <= result.report.surface_fraction_tolerance for error in errors)

    def test_functionalized_assembled_graph_contains_graft_junctions_and_angles(
        self,
        functionalized_slit_context,
    ):
        finalized_system = functionalized_slit_context.finalized_result.system
        snapshot = finalized_system.export_snapshot()
        store = sms.StructureWriter(snapshot)
        graph = store.assembled_graph(use_atom_names=True)
        report = store.validate_connectivity(use_atom_names=True)
        cache = store._collect_structure_records(use_atom_names=True)
        atom_records = cache.atom_records
        molecule_serials = cache.molecule_serials

        assert isinstance(graph, AssembledStructureGraph)
        assert isinstance(report, ConnectivityValidationReport)
        assert not (any(
                finding.code in {"framework_oxygen_environment", "framework_silicon_environment"}
                for finding in report.findings
            ))
        assert any(bond.provenance == "graft_junction" for bond in graph.bonds)
        assert snapshot.graft_junction_serials == tuple(
            (bond.atom_a, bond.atom_b)
            for bond in graph.bonds
            if bond.provenance == "graft_junction"
        )
        assert any(bond.provenance == "ligand_explicit" for bond in graph.bonds)
        assert not (any(bond.provenance == "ligand_inferred" for bond in graph.bonds))

        attachment_record = snapshot.attachments[0]
        mount_serial = molecule_serials[attachment_record.molecule_index][
            attachment_record.mount_atom_local_id
        ]
        assert any(
                bond.provenance == "graft_junction"
                and mount_serial in (bond.atom_a, bond.atom_b)
                for bond in graph.bonds
            )
        assert any(angle.atom_b == mount_serial for angle in graph.angles)

        record_by_serial = {record.serial: record for record in atom_records}
        graft_neighbors = {
            bond.atom_b if bond.atom_a == mount_serial else bond.atom_a
            for bond in graph.bonds
            if bond.provenance == "graft_junction"
            and mount_serial in (bond.atom_a, bond.atom_b)
        }
        assert all(record_by_serial[serial].atom_type == "O" for serial in graft_neighbors)

    def test_finalized_functionalized_connectivity_is_valid(
        self,
        functionalized_slit_context,
    ):
        result = functionalized_slit_context.finalized_result
        report = sms.StructureWriter(
            result.system.export_snapshot()
        ).validate_connectivity(use_atom_names=True)

        assert report.is_valid

    def test_functionalized_bonded_exports_include_graft_connectivity(
        self,
        tmp_path,
        functionalized_slit_context,
    ):
        output_dir = tmp_path / "functionalized_amorphous_slit_bonded_export"
        config = replace(
            functionalized_slit_context.config,
            ligand=replace(
                functionalized_slit_context.config.ligand,
                topology=explicit_tms_topology_config(tmp_path),
            ),
        )

        result = export_shared_functionalized_result(
            functionalized_slit_context,
            str(output_dir),
            config,
            write_pdb=True,
            write_cif=True,
        )

        name = functionalized_slit_context.config.slit_config.name
        with open(output_dir / f"{name}.pdb", "r") as file_in:
            pdb_text = file_in.read()
        with open(output_dir / f"{name}.cif", "r") as file_in:
            cif_text = file_in.read()
        with open(output_dir / f"{name}.top", "r") as file_in:
            top_text = file_in.read()
        with open(output_dir / f"{name}.itp", "r") as file_in:
            itp_text = file_in.read()

        assert "CONECT" in pdb_text
        assert "_struct_conn.id" in cif_text
        assert " TMS " in pdb_text
        assert f'#include "{name}.itp"' in top_text
        assert f"{name.upper()} 1" in top_text
        assert "[ atomtypes ]" in itp_text
        assert "[ dihedrals ]" in itp_text
        assert "si 14 28.08600" in itp_text
        assert "117.65432 432.123456" in itp_text
        assert "12.34567 0.98765 2" in itp_text
        assert "OM O1" not in top_text
        assert result.charge_diagnostics is not None
        assert result.charge_diagnostics.is_valid
        assert result.charge_diagnostics.expected_t3_fragment_charge == pytest.approx(0.825)
        assert result.charge_diagnostics.observed_t3_fragment_charge == pytest.approx(0.825, abs=1e-6)
        assert result.charge_diagnostics.derived_t2_fragment_charge == pytest.approx(0.55)
        assert result.charge_diagnostics.t2_site_count == 3
        assert result.charge_diagnostics.t3_site_count == 4
        assert result.charge_diagnostics.final_total_charge == pytest.approx(0.0, abs=1e-6)
        assert sum(row[3] for row in itp_atom_rows(output_dir / f"{name}.itp")) == pytest.approx(
            0.0,
            abs=1e-6,
        )
        assert result.report.timing_summary.finalize_s > 0
        assert result.report.timing_summary.export_s > 0

    def test_explicit_silica_topology_override_changes_functionalized_junction_angles(
        self,
        tmp_path,
        functionalized_slit_context,
    ):
        output_dir = tmp_path / "functionalized_amorphous_slit_custom_silica"
        silica_topology = sms.default_silica_topology()
        silica_topology.angle_terms.graft_oxygen_mount_oxygen.angle_deg = 111.11111
        silica_topology.angle_terms.graft_oxygen_mount_oxygen.force_constant = 222.222222
        config = replace(
            functionalized_slit_context.config,
            slit_config=replace(
                functionalized_slit_context.config.slit_config,
                silica_topology=silica_topology,
            ),
            ligand=replace(
                functionalized_slit_context.config.ligand,
                topology=explicit_tms_topology_config(tmp_path),
            ),
        )

        result = export_shared_functionalized_result(
            functionalized_slit_context,
            str(output_dir),
            config,
            write_pdb=False,
            write_cif=False,
        )

        name = functionalized_slit_context.config.slit_config.name
        with open(output_dir / f"{name}.itp", "r") as file_in:
            itp_text = file_in.read()

        assert result.silica_topology is not silica_topology
        assert result.silica_topology.angle_terms.graft_oxygen_mount_oxygen.angle_deg == pytest.approx(111.11111)
        assert "111.11111 222.222222" in itp_text

    def test_functionalized_write_skips_topology_without_explicit_silane_itp(
        self,
        tmp_path,
        functionalized_slit_context,
    ):
        output_dir = tmp_path / "functionalized_no_explicit_topology"
        result = export_shared_functionalized_result(
            functionalized_slit_context,
            str(output_dir),
            functionalized_slit_context.config,
            write_pdb=False,
            write_cif=False,
        )

        assert result.charge_diagnostics is None
        name = functionalized_slit_context.config.slit_config.name
        assert not (output_dir / f"{name}.top").exists()
        assert not (output_dir / f"{name}.itp").exists()

    def test_functionalized_export_rejects_charge_mismatched_t3_topology(
        self,
        tmp_path,
        functionalized_slit_context,
    ):
        output_dir = tmp_path / "functionalized_invalid_t3_charge"
        config = replace(
            functionalized_slit_context.config,
            ligand=replace(
                functionalized_slit_context.config.ligand,
                topology=sms.SilaneTopologyConfig(
                    itp_path=str(bundled_tms_template_path()),
                    moleculetype_name="TMS",
                    geminal_cross_terms=explicit_tms_geminal_cross_terms(),
                ),
            ),
        )

        with pytest.raises(ValueError, match="base T3 fragment topology"):
            export_shared_functionalized_result(
                functionalized_slit_context,
                str(output_dir),
                config,
                write_pdb=False,
                write_cif=False,
            )

    def test_functionalized_export_rejects_missing_geminal_cross_terms(
        self,
        tmp_path,
        functionalized_slit_context,
    ):
        output_dir = tmp_path / "functionalized_missing_geminal_terms"
        config = replace(
            functionalized_slit_context.config,
            ligand=replace(
                functionalized_slit_context.config.ligand,
                topology=explicit_tms_topology_config(
                    tmp_path,
                    include_geminal_terms=False,
                ),
            ),
        )

        with pytest.raises(ValueError, match="geminal_cross_terms"):
            export_shared_functionalized_result(
                functionalized_slit_context,
                str(output_dir),
                config,
                write_pdb=False,
                write_cif=False,
            )

    def test_teps_exports_use_pdb_aliases_and_consistent_mmcif_metadata(
        self,
        tmp_path,
        teps_slit_context,
    ):
        output_dir = tmp_path / "functionalized_amorphous_slit_teps_export"
        export_shared_functionalized_result(
            teps_slit_context,
            str(output_dir),
            teps_slit_context.config,
            write_pdb=True,
            write_cif=True,
        )

        name = teps_slit_context.config.slit_config.name
        with open(output_dir / f"{name}.pdb", "r") as file_in:
            pdb_text = file_in.read()
        with open(output_dir / f"{name}.cif", "r") as file_in:
            cif_text = file_in.read()

        alias_map = {}
        for line in pdb_text.splitlines():
            if line.startswith("REMARK 250 RESIDUE_ALIAS "):
                parts = line.split()
                alias_map[parts[3]] = parts[5]

        assert "CRYST1" in pdb_text
        assert alias_map["TEPS"] != "TEPS"
        assert alias_map["TEPSG"] != "TEPSG"
        assert len(alias_map["TEPS"]) == 3
        assert len(alias_map["TEPSG"]) == 3

        atom_lines = [
            line for line in pdb_text.splitlines()
            if line.startswith("HETATM")
        ]
        teps_atom_lines = [
            line for line in atom_lines
            if line[17:20].strip() == alias_map["TEPS"]
        ]
        tepsg_atom_lines = [
            line for line in atom_lines
            if line[17:20].strip() == alias_map["TEPSG"]
        ]

        assert teps_atom_lines
        assert tepsg_atom_lines
        for line in teps_atom_lines[:3] + tepsg_atom_lines[:3]:
            assert line[21] == "A"
            assert line[26] == " "
            assert snapshot_mod._decode_hybrid36(4, line[22:26]) >= 1

        entity_tags, entity_rows = cif_loop_rows(cif_text, "_entity.id")
        asym_tags, asym_rows = cif_loop_rows(cif_text, "_struct_asym.id")
        atom_tags, atom_rows = cif_loop_rows(cif_text, "_atom_site.group_PDB")
        conn_tags, conn_rows = cif_loop_rows(cif_text, "_struct_conn.id")

        entity_id_index = entity_tags.index("_entity.id")
        asym_id_index = asym_tags.index("_struct_asym.id")
        atom_entity_index = atom_tags.index("_atom_site.label_entity_id")
        atom_asym_index = atom_tags.index("_atom_site.label_asym_id")
        atom_comp_index = atom_tags.index("_atom_site.label_comp_id")
        conn_asym_1_index = conn_tags.index("_struct_conn.ptnr1_label_asym_id")
        conn_asym_2_index = conn_tags.index("_struct_conn.ptnr2_label_asym_id")

        entity_ids = {row[entity_id_index] for row in entity_rows}
        asym_ids = {row[asym_id_index] for row in asym_rows}

        assert any(row[atom_comp_index] == "TEPS" for row in atom_rows)
        assert any(row[atom_comp_index] == "TEPSG" for row in atom_rows)
        assert {row[atom_entity_index] for row in atom_rows} <= entity_ids
        assert {row[atom_asym_index] for row in atom_rows} <= asym_ids
        assert {row[conn_asym_1_index] for row in conn_rows} <= asym_ids
        assert {row[conn_asym_2_index] for row in conn_rows} <= asym_ids
        assert not (output_dir / f"{name}.top").exists()
        assert not (output_dir / f"{name}.itp").exists()

    def test_small_teps_functionalized_smoke_populates_timing_summary(
        self,
        teps_slit_context,
    ):
        result = teps_slit_context.prepared_result

        assert result.report.final_surface == sms.SiliconStateComposition(957, 65, 651, 239, 1, 1)
        assert result.report.timing_summary.base_slit_build_s > 0
        assert result.report.timing_summary.q_state_preparation_s > 0
        assert result.report.timing_summary.t2_attachment_s > 0
        assert result.report.timing_summary.t3_attachment_s > 0
