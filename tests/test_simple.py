import os
import copy
import warnings

import matplotlib.pyplot as plt
import pytest

import silicams as sms
import silicams.database as database
import silicams.generic as generic
import silicams.geometry as geometry
import silicams.utils as utils
import silicams.writers.common as snapshot_mod
from silicams.atom import Atom
from silicams.connectivity import (
    AssembledStructureGraph,
    ConnectivityValidationReport,
    GraphBond,
)
from silicams.dice import Dice
from silicams.matrix import Matrix
from silicams.pattern import AlphaCristobalit, BetaCristobalit
from silicams.shape import (
    Cone,
    ConeConfig,
    Cuboid,
    CuboidConfig,
    Cylinder,
    CylinderConfig,
    Sphere,
    SphereConfig,
)


pytestmark = pytest.mark.usefixtures("module_workspace")


class TestUserModel:


    #########
    # Utils #
    #########
    def test_utils(self):
        text_link = "output/test/test.txt"
        text_copy_link = "output/test/test_copy.txt"
        pickle_link = "output/test/test.pkl"

        utils.mkdirp("output/test")

        with open(text_link, "w") as file_out:
            file_out.write("TEST")
        utils.copy(text_link, text_copy_link)
        utils.replace(text_copy_link, "TEST", "DOTA")
        with open(text_copy_link, "r") as file_in:
            for line in file_in:
                assert line == "DOTA\n"

        assert utils.column([[1, 1, 1], [2, 2, 2]]) == [[1, 2], [1, 2], [1, 2]]

        utils.save([1, 1, 1], pickle_link)
        assert utils.load(pickle_link) == [1, 1, 1]

        assert round(utils.mumol_m2_to_mols(3, 100), 4) == 180.66
        assert round(utils.mols_to_mumol_m2(180, 100), 4) == 2.989
        assert round(utils.mmol_g_to_mumol_m2(0.072, 512), 2) == 0.14
        assert round(utils.mmol_l_to_mols(30, 1000), 4) == 18.066
        assert round(utils.mols_to_mmol_l(18, 1000), 4) == 29.8904

        print()
        utils.toc(utils.tic(), message="Test", is_print=True)
        assert round(utils.toc(utils.tic(), is_print=True)) == 0


    ############
    # Geometry #
    ############
    def test_geometry(self):
        vec_a = [1, 1, 2]
        vec_b = [0, 3, 2]

        print()

        assert round(geometry.dot_product(vec_a, vec_b), 4) == 7
        assert round(geometry.length(vec_a), 4) == 2.4495
        assert [round(x, 4) for x in geometry.vector(vec_a, vec_b)] == [-1, 2, 0]
        with pytest.raises(ValueError, match="Wrong dimensions"):
            geometry.vector([0, 1], [0, 0, 0])
        assert [round(x, 4) for x in geometry.unit(vec_a)] == [0.4082, 0.4082, 0.8165]
        assert [round(x, 4) for x in geometry.cross_product(vec_a, vec_b)] == [-4, -2, 3]
        assert round(geometry.angle(vec_a, vec_b), 4) == 37.5714
        assert round(geometry.angle_polar(vec_a), 4) == 0.7854
        assert round(geometry.angle_azi(vec_b), 4) == 0.9828
        assert round(geometry.angle_azi([0, 0, 0]), 4) == 1.5708
        assert [round(x, 4) for x in geometry.main_axis(1)] == [1, 0, 0]
        assert [round(x, 4) for x in geometry.main_axis(2)] == [0, 1, 0]
        assert [round(x, 4) for x in geometry.main_axis(3)] == [0, 0, 1]
        assert [round(x, 4) for x in geometry.main_axis("x")] == [1, 0, 0]
        assert [round(x, 4) for x in geometry.main_axis("y")] == [0, 1, 0]
        assert [round(x, 4) for x in geometry.main_axis("z")] == [0, 0, 1]
        with pytest.raises(ValueError, match="Wrong axis definition"):
            geometry.main_axis("h")
        with pytest.raises(ValueError, match="Wrong axis definition"):
            geometry.main_axis(100)
        with pytest.raises(ValueError, match="Wrong axis definition"):
            geometry.main_axis(0.1)
        assert [round(x, 4) for x in geometry.rotate(vec_a, "x", 90, True)] == [1.0, -2.0, 1.0]
        with pytest.raises(ValueError, match="Wrong vector dimensions"):
            geometry.rotate(vec_a, [0, 1, 2, 3], 90, True)
        with pytest.raises(ValueError, match="Wrong axis definition"):
            geometry.rotate(vec_a, "h", 90, True)


    ############
    # Database #
    ############
    def test_database(self):
        print()
        assert database.get_mass("H") == 1.0079
        assert database.get_element("Si") == "Si"
        assert database.get_element("Ci") == "C"
        assert database.get_pdb_element("CA") == "C"
        assert database.get_pdb_element("CD1") == "C"
        assert database.get_pdb_element("SI1") == "Si"
        assert database.get_covalent_radius("OM1") == pytest.approx(0.066, abs=1e-7)
        with pytest.raises(ValueError, match="Atom name not found"):
            database.get_mass("DOTA")


    ########
    # Atom #
    ########
    def test_atom(self):
        atom = Atom([0, 0, 0], "O", "O", 5)

        atom.set_pos([0.0, 0.1, 0.2])
        atom.set_atom_type("H")
        atom.set_name("HO1")
        atom.set_residue(0)

        assert atom.get_pos() == [0.0, 0.1, 0.2]
        assert atom.get_atom_type() == "H"
        assert atom.get_name() == "HO1"
        assert atom.get_residue() == 0

        assert atom.__str__() == "   Residue Name Type    x    y    z\n0        0  HO1    H  0.0  0.1  0.2"


    ############
    # Molecule #
    ############
    def test_molecule_loading(self):
        mol_gro = sms.Molecule(inp="data/benzene.gro")
        mol_pdb = sms.Molecule(inp="data/benzene.pdb")
        mol_mol2 = sms.Molecule(inp="data/benzene.mol2")

        mol_atom = sms.Molecule(inp=mol_mol2.get_atom_list())
        mol_concat = sms.Molecule(inp=[mol_gro, mol_pdb])

        mol_append = sms.Molecule(inp="data/benzene.gro")
        mol_append.append(mol_gro)

        pos_gro = [[round(x, 4) for x in col] for col in mol_gro.column_pos()]
        pos_pdb = [[round(x, 4) for x in col] for col in mol_pdb.column_pos()]
        pos_mol2 = [[round(x, 4) for x in col] for col in mol_mol2.column_pos()]
        pos_atom = [[round(x, 4) for x in col] for col in mol_atom.column_pos()]
        pos_concat = [[round(x, 4) for x in col] for col in mol_concat.column_pos()]
        pos_append = [[round(x, 4) for x in col] for col in mol_append.column_pos()]

        assert pos_gro == pos_pdb
        assert pos_gro == pos_mol2
        assert pos_gro == pos_atom
        assert [col+col for col in pos_gro] == pos_concat
        assert [col+col for col in pos_gro] == pos_append
        assert mol_gro.get_bonds() == []
        assert mol_pdb.get_bonds() == []
        assert len(mol_mol2.get_bonds()) == 12
        assert len(mol_gro.infer_bonds()) == 12

        print()
        with pytest.raises(ValueError, match="Unsupported filetype"):
            sms.Molecule(inp="data/benzene.DOTA")

    def test_molecule_loads_pdb_conect_and_preserves_bonds(self):
        pdb_path = os.path.join("output", "bonded_probe.pdb")
        with open(pdb_path, "w") as file_out:
            file_out.write(
                "HETATM    1 SI1 TMS A   1       0.000   0.000   0.000  1.00  0.00          Si  \n"
                "HETATM    2  O1 TMS A   1       1.640   0.000   0.000  1.00  0.00           O  \n"
                "HETATM    3  C1 TMS A   1       2.800   0.000   0.000  1.00  0.00           C  \n"
                "CONECT    1    2\n"
                "CONECT    2    1    3\n"
                "CONECT    3    2\n"
                "TER\nEND\n"
            )

        mol = sms.Molecule(inp=pdb_path)
        assert mol.get_bonds() == [(0, 1), (1, 2)]

        graph = sms.StructureWriter(mol, "output").assembled_graph(use_atom_names=True)
        assert len(graph.bonds) == 2
        assert all(bond.provenance == "ligand_explicit" for bond in graph.bonds)

    def test_molecule_pdb_name_fallback_keeps_phenyl_atoms_as_carbon(self):
        pdb_path = os.path.join("output", "phenyl_silane_probe.pdb")
        with open(pdb_path, "w") as file_out:
            file_out.write(
                "HETATM    1 SI1 PHS A   1       0.000   0.000   0.000  1.00  0.00              \n"
                "HETATM    2  CA PHS A   1       1.860   0.000   0.000  1.00  0.00              \n"
                "HETATM    3 CD1 PHS A   1       2.560   1.210   0.000  1.00  0.00              \n"
                "HETATM    4 CE1 PHS A   1       3.960   1.210   0.000  1.00  0.00              \n"
                "CONECT    1    2\n"
                "CONECT    2    1    3\n"
                "CONECT    3    2    4\n"
                "CONECT    4    3\n"
                "TER\nEND\n"
            )

        mol = sms.Molecule(inp=pdb_path)

        assert [mol.get_atom_type(index) for index in range(mol.get_num())] == ["Si", "C", "C", "C"]
        assert mol.get_bonds() == [(0, 1), (1, 2), (2, 3)]

    def test_molecule_bond_graph_is_preserved_through_edits(self):
        mol = sms.Molecule()
        mol.add("Si", [0.0, 0.0, 0.0], name="SI1")
        mol.add("O", 0, r=0.164, name="O1")
        mol.add("H", 1, r=0.098, name="H1")

        assert mol.get_bonds() == [(0, 1), (1, 2)]

        mol_copy = copy.deepcopy(mol)
        assert mol_copy.get_bonds() == [(0, 1), (1, 2)]

        mol.switch_atom_order(0, 2)
        assert mol.get_bonds() == [(0, 1), (1, 2)]

        mol.delete(2)
        assert mol.get_bonds() == [(0, 1)]

    def test_molecule_properties(self):
        mol = sms.Molecule(inp="data/benzene.gro")

        assert mol.pos(0) == [0.0935, 0.0000, 0.3143]
        assert [round(x, 4) for x in mol.bond(0, 1)] == [0.1191, 0.0, 0.0687]
        assert mol.bond([1, 0, 0], [0, 0, 0]) == [-1, 0, 0]
        assert mol.get_box() == [0.4252, 0.001, 0.491]
        assert [round(x, 4) for x in mol.centroid()] == [0.2126, 0.0, 0.2455]
        assert [round(x, 4) for x in mol.com()] == [0.2126, 0.0, 0.2455]

    def test_molecule_editing(self):
        mol = sms.Molecule(inp="data/benzene.gro")

        mol.translate([0, 0.1, 0.2])
        assert [round(x, 4) for x in mol.pos(3)] == [0.3317, 0.1000, 0.3768]
        mol.rotate("x", 45)
        assert [round(x, 4) for x in mol.pos(3)] == [0.3317, -0.1957, 0.3371]
        mol.move(0, [1, 1, 1])
        assert [round(x, 4) for x in mol.pos(3)] == [1.2382, 1.0972, 0.9028]
        mol.zero()
        assert [round(x, 4) for x in mol.pos(3)] == [0.3317, 0.2222, 0.1250]
        mol.put(3, [0, 0, 0])
        assert [round(x, 4) for x in mol.pos(3)] == [0.0000, 0.0000, 0.0000]
        mol.part_move([0, 1], [2, 3, 4], 0.5)
        mol.part_move([0, 1], 1, 0.5)
        assert [round(x, 4) for x in mol.pos(3)] == [0.3140, -0.1281, 0.1281]
        mol.part_rotate([0, 1], [2, 3, 4], 45, 1)
        mol.part_rotate([0, 1], 1, 45, 1)
        assert [round(x, 4) for x in mol.pos(3)] == [-0.1277, 0.0849, -0.3176]
        mol.part_angle([0, 1], [1, 2], [1, 2, 3, 4], 45, 1)
        assert [round(x, 4) for x in mol.pos(3)] == [-0.1360, -0.1084, -0.3068]
        mol.part_angle([0, 0, 1], [0, 1, 0], 1, 45, 1)
        assert [round(x, 4) for x in mol.pos(3)] == [-0.1360, -0.1084, -0.3068]

        print()

        with pytest.raises(ValueError, match="Wrong input"):
            mol._vector(0.1, 0.1)
        with pytest.raises(ValueError, match="Wrong dimensions"):
            mol._vector([0, 0], [0, 0])

        with pytest.raises(ValueError, match="Wrong bond input"):
            mol.part_angle([0, 0, 1, 0], [0, 1, 0, 0], 1, 45, 1)
        with pytest.raises(ValueError, match="Wrong bond dimensions"):
            mol.part_angle([0, 0], [0, 1, 2], 1, 45, 1)

    def test_molecule_creation(self):
        mol = sms.Molecule()

        mol.add("C", [0, 0.1, 0.2])
        mol.add("C", 0, r=0.1, theta=90)
        mol.add("C", 1, [0, 1], r=0.1, theta=90)
        mol.add("C", 2, [0, 2], r=0.1, theta=90, phi=45)
        assert [round(x, 4) for x in mol.pos(3)] == [0.0500, 0.0500, 0.0293]
        mol.delete(2)
        assert [round(x, 4) for x in mol.pos(2)] == [0.0500, 0.0500, 0.0293]
        mol.add("C", [0, 0.1, 0.2])
        assert mol.overlap() == {0: [3]}
        mol.switch_atom_order(0, 2)
        assert [round(x, 4) for x in mol.pos(0)] == [0.0500, 0.0500, 0.0293]
        mol.set_atom_type(0, "R")
        assert mol.get_atom_list()[0].get_atom_type() == "R"
        assert mol.get_atom_type(0) == "R"
        mol.set_atom_name(0, "RuX")
        assert mol.get_atom_list()[0].get_name() == "RuX"
        mol.set_atom_residue(0, 1)
        assert mol.get_atom_list()[0].get_residue() == 1

    def test_molecule_set_get(self):
        mol = sms.Molecule()

        mol.set_name("test_mol")
        mol.set_short("TMOL")
        mol.set_box([1, 1, 1])
        mol.set_charge(1.5)
        mol.set_masses([1, 2, 3])

        assert mol.get_name() == "test_mol"
        assert mol.get_short() == "TMOL"
        assert mol.get_box() == [1, 1, 1]
        assert mol.get_num() == 0
        assert mol.get_charge() == 1.5
        assert mol.get_masses() == [1, 2, 3]
        assert mol.get_mass() == 6

    def test_molecule_representation(self):
        mol = sms.Molecule()
        mol.add("H", [0.0, 0.1, 0.2], name="HO1")

        assert mol.__str__() == "   Residue Name Type    x    y    z\n0        0  HO1    H  0.0  0.1  0.2"


    ###########
    # Generic #
    ###########
    def test_generic(self):
        assert [round(x, 4) for x in generic.alkane(10, "decane", "DEC").pos(5)] == [0.0472, 0.1028, 0.7170]
        assert [round(x, 4) for x in generic.alkane(1, "methane", "MET").pos(0)] == [0.0514, 0.0890, 0.0363]
        assert [round(x, 4) for x in generic.alcohol(10, "decanol", "DCOL").pos(5)] == [0.0363, 0.1028, 0.7170]
        assert [round(x, 4) for x in generic.alcohol(1, "methanol", "MEOL").pos(0)] == [0.0715, 0.0890, 0.0363]
        assert [round(x, 4) for x in generic.ketone(10, 5, "decanone", "DCON").pos(5)] == [0.0472, 0.1028, 0.7170]
        assert [round(x, 4) for x in  generic.tms(separation=30).pos(5)] == [0.0273, 0.0472, 0.4525]
        assert [round(x, 4) for x in  generic.tms(is_si=False).pos(5)] == [0.0273, 0.0472, 0.4976]
        assert [round(x, 4) for x in  generic.silanol().pos(0)] == [0.000, 0.000, 0.000]

        print()
        with pytest.raises(ValueError, match="too small for ketones"):
            generic.ketone(2, 0)


    #########
    # Writers #
    #########
    def test_hybrid36_helpers(self):
        reference_vectors = (
            (4, 9999, "9999"),
            (4, 10000, "A000"),
            (4, 10035, "A00Z"),
            (4, 10036, "A010"),
            (4, 10000 + (26 * (36 ** 3)), "a000"),
            (5, 99999, "99999"),
            (5, 100000, "A0000"),
            (5, 100035, "A000Z"),
            (5, 100036, "A0010"),
            (5, 100000 + (26 * (36 ** 4)), "a0000"),
        )

        for width, value, token in reference_vectors:
            assert snapshot_mod._encode_hybrid36(width, value) == token
            assert snapshot_mod._decode_hybrid36(width, token) == value

        with pytest.raises(ValueError, match="exceeds the hybrid-36 range"):
            snapshot_mod._encode_hybrid36(4, snapshot_mod._hybrid36_max_value(4) + 1)

    def test_pdb_writer_uses_hybrid36_for_overflowed_ids(self):
        mol = sms.Molecule("overflow_writer", "OVF")
        mol.add("C", [0.0, 0.0, 0.0], name="C1")
        mol.add("C", [0.1, 0.0, 0.0], name="C2")

        store = sms.StructureWriter(mol, "output")
        atom_records = [
            snapshot_mod._StructureAtomRecord(
                serial=99999,
                pdb_serial_token=snapshot_mod._encode_hybrid36(5, 99999),
                molecule_index=0,
                local_atom_index=0,
                residue_name="TEPS",
                pdb_residue_name="TEP",
                residue_id=9999,
                pdb_residue_id_token=snapshot_mod._encode_hybrid36(4, 9999),
                atom_name="C1",
                atom_type="C",
                position=(0.0, 0.0, 0.0),
                source_id=None,
                pdb_chain_id="A",
                cif_entity_id="1",
                cif_asym_id="A9999",
                cif_label_seq_id="1",
            ),
            snapshot_mod._StructureAtomRecord(
                serial=100000,
                pdb_serial_token=snapshot_mod._encode_hybrid36(5, 100000),
                molecule_index=0,
                local_atom_index=1,
                residue_name="TEPS",
                pdb_residue_name="TEP",
                residue_id=10000,
                pdb_residue_id_token=snapshot_mod._encode_hybrid36(4, 10000),
                atom_name="C2",
                atom_type="C",
                position=(0.1, 0.0, 0.0),
                source_id=None,
                pdb_chain_id="A",
                cif_entity_id="1",
                cif_asym_id="A10000",
                cif_label_seq_id="1",
            ),
        ]
        graph = AssembledStructureGraph.from_bonds(
            (99999, 100000),
            [GraphBond(99999, 100000, "ligand_explicit")],
        )
        store._structure_export_cache[True] = snapshot_mod._StructureExportCache(
            atom_records=atom_records,
            molecule_serials=[[99999, 100000]],
            residue_alias_records=[
                snapshot_mod._PdbResidueAliasRecord(full_name="TEPS", pdb_name="TEP")
            ],
            entity_records=[
                snapshot_mod._CifEntityRecord(entity_id="1", residue_name="TEPS")
            ],
            struct_asym_records=[
                snapshot_mod._CifStructAsymRecord(
                    asym_id="A9999",
                    entity_id="1",
                    residue_id=9999,
                    residue_name="TEPS",
                ),
                snapshot_mod._CifStructAsymRecord(
                    asym_id="A10000",
                    entity_id="1",
                    residue_id=10000,
                    residue_name="TEPS",
                ),
            ],
            graph=graph,
        )

        store.write_pdb("store_hybrid36_overflow.pdb", use_atom_names=True)

        with open("output/store_hybrid36_overflow.pdb", "r") as file_in:
            pdb_lines = [
                line.rstrip("\n")
                for line in file_in
                if line.startswith(("HETATM", "CONECT"))
            ]

        assert pdb_lines[0][6:11] == "99999"
        assert pdb_lines[1][6:11] == "A0000"
        assert pdb_lines[0][22:26] == "9999"
        assert pdb_lines[1][22:26] == "A000"
        assert snapshot_mod._decode_hybrid36(5, pdb_lines[1][6:11]) == 100000
        assert snapshot_mod._decode_hybrid36(4, pdb_lines[1][22:26]) == 10000
        assert pdb_lines[2].startswith("CONECT")
        assert "A0000" in pdb_lines[2]

    def test_writers(self):
        mol = sms.Molecule(inp="data/benzene.gro")

        mol.set_atom_residue(1, 1)

        sms.AntechamberWriter(mol, "output").write(
            "store_job",
            "store_master.job",
        )
        sms.StructureWriter(mol, "output").write_object("store_obj.obj")
        sms.StructureWriter(mol, "output").write_gro("store_gro.gro", True)
        sms.StructureWriter(mol, "output").write_pdb("store_pdb.pdb", True)
        sms.StructureWriter(mol, "output").write_cif("store_cif.cif", True)
        sms.StructureWriter(mol, "output").write_xyz("store_xyz.xyz")
        sms.StructureWriter(mol, "output").write_lammps("store_lmp.lmp")
        sms.GromacsTopologyWriter(mol, "output").write_grid_itp(
            "store_grid.itp"
        )

        object_snapshot = utils.load("output/store_obj.obj")
        assert isinstance(object_snapshot, snapshot_mod.StructureSnapshot)
        assert object_snapshot.has_assembled_export
        assert len(object_snapshot.atom_order) == mol.get_num()

        with open("output/store_cif.cif", "r") as file_in:
            cif_text = file_in.read()
        assert "_atom_site.Cartn_x" in cif_text
        assert "_struct_conn.id" in cif_text

        with open("output/store_pdb.pdb", "r") as file_in:
            pdb_text = file_in.read()
        assert "CONECT" in pdb_text

        graph = sms.StructureWriter(mol, "output").assembled_graph(use_atom_names=True)
        assert isinstance(graph, AssembledStructureGraph)
        assert len(graph.bonds) == 12
        assert all(bond.provenance == "ligand_inferred" for bond in graph.bonds)
        assert len(graph.angles) > 0

        report = sms.StructureWriter(mol, "output").validate_connectivity(use_atom_names=True)
        assert isinstance(report, ConnectivityValidationReport)
        assert report.is_valid

        print()
        with pytest.raises(TypeError, match="Molecule or StructureSnapshot"):
            sms.StructureWriter({})
        with pytest.raises(TypeError, match="silica-slit snapshot"):
            sms.GromacsTopologyWriter(mol).write_helper_topology()

    def test_connectivity_validation_reports_invalid_local_valence(self):
        mol = sms.Molecule("invalid_valence", "IVL")
        mol.add("O", [0.0, 0.0, 0.0], name="O1")
        mol.add("H", 0, r=0.098, name="H1")
        mol.add("H", 0, r=0.098, theta=120, name="H2")
        mol.add("H", 0, r=0.098, theta=240, name="H3")

        store = sms.StructureWriter(mol, "output")
        report = store.validate_connectivity(use_atom_names=True)

        assert not (report.is_valid)
        assert report.error_count > 0
        assert any(finding.code == "unexpected_degree" for finding in report.findings)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            store.write_gro("invalid_valence_warn.gro", use_atom_names=True, validate_connectivity="warn")
        assert any("Connectivity validation found" in str(warning.message) for warning in caught)

        with pytest.raises(ValueError, match="Connectivity validation found"):
            store.write_gro("invalid_valence_strict.gro", use_atom_names=True, validate_connectivity="strict")

    def test_connectivity_validation_allows_stretched_but_element_sane_bonds(self):
        mol = sms.Molecule("stretched_silica_fragment", "SSF")
        mol.add("Si", [0.0, 0.0, 0.0], name="SI1")
        mol.add("O", [0.32, 0.0, 0.0], name="O1")
        mol.add("H", [0.50, 0.0, 0.0], name="H1")
        mol.add_bond(0, 1)
        mol.add_bond(1, 2)

        report = sms.StructureWriter(mol, "output").validate_connectivity(use_atom_names=True)

        assert report.is_valid
        assert not any(
            finding.code in {"invalid_silica_bond", "hydrogen_degree", "unexpected_degree"}
            for finding in report.findings
        )

    def test_structure_writers_normalize_slx_to_exported_om(self):
        mol = sms.Molecule("siloxane_bridge_probe", "SLX")
        mol.set_box([1.0, 1.0, 1.0])
        mol.add("O", [0.0, 0.0, 0.0], name="OM1")

        store = sms.StructureWriter(mol, "output")
        store.write_gro(
            "siloxane_bridge_probe.gro",
            use_atom_names=True,
            validate_connectivity="off",
        )
        store.write_pdb(
            "siloxane_bridge_probe.pdb",
            use_atom_names=True,
            validate_connectivity="off",
        )
        store.write_cif(
            "siloxane_bridge_probe.cif",
            use_atom_names=True,
            validate_connectivity="off",
        )

        with open("output/siloxane_bridge_probe.gro", "r") as file_in:
            gro_text = file_in.read()
        with open("output/siloxane_bridge_probe.pdb", "r") as file_in:
            pdb_text = file_in.read()
        with open("output/siloxane_bridge_probe.cif", "r") as file_in:
            cif_text = file_in.read()

        assert "SLX" not in gro_text
        assert "SLX" not in pdb_text
        assert "SLX" not in cif_text
        assert "OM" in gro_text
        assert " OM " in pdb_text
        assert " OM " in cif_text

    def test_connectivity_validation_normalizes_siloxane_bridges_to_framework_oxygen_rules(self):
        mol = sms.Molecule("invalid_siloxane_bridge", "SLX")
        mol.add("O", [0.0, 0.0, 0.0], name="OM1")
        mol.add("Si", [0.16, 0.0, 0.0], name="SI1")
        mol.add("H", [-0.10, 0.0, 0.0], name="H1")
        mol.add_bond(0, 1)
        mol.add_bond(0, 2)

        report = sms.StructureWriter(mol, "output").validate_connectivity(use_atom_names=True)

        assert not report.is_valid
        assert any(
            finding.code == "framework_oxygen_environment"
            for finding in report.findings
        )

    def test_pdb_uses_hybrid36_when_fixed_width_limits_are_exceeded(self):
        atom = sms.Molecule("single_atom", "SIN")
        atom.add("H", [0.0, 0.0, 0.0], name="H1")
        store = sms.StructureWriter(atom, "output")
        store._mols = [atom] * 100000

        cache = store._collect_structure_records(use_atom_names=True)
        records = cache.atom_records
        store._validate_pdb_hybrid36_limits(records)

        assert records[9998].pdb_residue_id_token == "9999"
        assert records[9999].pdb_residue_id_token == "A000"
        assert snapshot_mod._decode_hybrid36(4, records[-1].pdb_residue_id_token) == records[-1].residue_id

    ###########
    # Pattern #
    ###########
    def test_pattern_beta_cristobalit(self):
        # Initialize
        beta_cristobalit = BetaCristobalit()

        # Pattern and output
        pattern = beta_cristobalit.pattern()
        pattern.set_name("pattern_beta_cbt_minimal")
        assert pattern.get_num() == 36
        sms.StructureWriter(pattern, "output").write_gro()

        # Generation and Orientation
        beta_cristobalit = BetaCristobalit()
        beta_cristobalit.generate([2, 2, 2], "x")
        beta_cristobalit.get_block().set_name("pattern_beta_cbt_x")
        assert beta_cristobalit.get_size() == [2.480, 1.754, 2.024]
        assert [round(x, 3) for x in beta_cristobalit.get_block().get_box()] == [2.480, 1.754, 2.024]
        sms.StructureWriter(beta_cristobalit.get_block(), "output").write_gro()

        beta_cristobalit = BetaCristobalit()
        beta_cristobalit.generate([2, 2, 2], "y")
        beta_cristobalit.get_block().set_name("pattern_beta_cbt_y")
        assert beta_cristobalit.get_size() == [2.024, 2.480, 1.754]
        assert [round(x, 3) for x in beta_cristobalit.get_block().get_box()] == [2.024, 2.480, 1.754]
        sms.StructureWriter(beta_cristobalit.get_block(), "output").write_gro()

        beta_cristobalit = BetaCristobalit()
        beta_cristobalit.generate([2, 2, 2], "z")
        beta_cristobalit.get_block().set_name("pattern_beta_cbt_z")
        assert beta_cristobalit.get_size() == [2.024, 1.754, 2.480]
        assert [round(x, 3) for x in beta_cristobalit.get_block().get_box()] == [2.024, 1.754, 2.480]
        sms.StructureWriter(beta_cristobalit.get_block(), "output").write_gro()
        sms.StructureWriter(beta_cristobalit.get_block(), "output").write_lammps()

        # Misc
        beta_cristobalit = BetaCristobalit()
        beta_cristobalit.generate([2, 2, 2], "z")
        beta_cristobalit.get_block().set_name("DOTA")

        # Overlap and output
        assert beta_cristobalit.get_block().get_num() == 576
        assert beta_cristobalit.get_block().overlap() == {}

        # Getter
        assert beta_cristobalit.get_repeat() == [0.506, 0.877, 1.240]
        assert beta_cristobalit.get_gap() == [0.126, 0.073, 0.155]
        assert beta_cristobalit.get_orient() == "z"
        assert beta_cristobalit.get_block().get_name() == "DOTA"

    def test_alpha_cristobalit(self):
        # Initialize
        alpha_cristobalit = AlphaCristobalit()

        # Pattern and output
        pattern = alpha_cristobalit.pattern()
        pattern.set_name("pattern_alpha_cbt_minimal")
        assert pattern.get_num() == 12
        sms.StructureWriter(pattern, "output").write_gro()

        # Generation and Orientation
        alpha_cristobalit = AlphaCristobalit()
        alpha_cristobalit.generate([2, 2, 2], "x")
        alpha_cristobalit.get_block().set_name("pattern_alpha_cbt_x")
        assert alpha_cristobalit.get_size() == [2.0844, 1.9912, 1.9912]
        assert [round(x, 3) for x in alpha_cristobalit.get_block().get_box()] == [2.084, 1.991, 1.991]
        sms.StructureWriter(alpha_cristobalit.get_block(), "output").write_gro()

        alpha_cristobalit = AlphaCristobalit()
        alpha_cristobalit.generate([2, 2, 2], "y")
        alpha_cristobalit.get_block().set_name("pattern_alpha_cbt_y")
        assert alpha_cristobalit.get_size() == [1.9912, 2.0844, 1.9912]
        assert [round(x, 3) for x in alpha_cristobalit.get_block().get_box()] == [1.991, 2.084, 1.991]
        sms.StructureWriter(alpha_cristobalit.get_block(), "output").write_gro()

        alpha_cristobalit = AlphaCristobalit()
        alpha_cristobalit.generate([2, 2, 2], "z")
        alpha_cristobalit.get_block().set_name("pattern_alpha_cbt_z")
        assert alpha_cristobalit.get_size() == [1.9912, 1.9912, 2.0844]
        assert [round(x, 3) for x in alpha_cristobalit.get_block().get_box()] == [1.991, 1.991, 2.084]
        sms.StructureWriter(alpha_cristobalit.get_block(), "output").write_gro()
        sms.StructureWriter(alpha_cristobalit.get_block(), "output").write_lammps()

        # Misc
        alpha_cristobalit = AlphaCristobalit()
        alpha_cristobalit.generate([2, 2, 2], "z")
        alpha_cristobalit.get_block().set_name("DOTA")

        # Overlap and output
        assert alpha_cristobalit.get_block().get_num() == 576
        assert alpha_cristobalit.get_block().overlap() == {}

        # Getter
        assert alpha_cristobalit.get_repeat() == [0.4978, 0.4978, 0.6948]
        assert alpha_cristobalit.get_orient() == "z"
        assert alpha_cristobalit.get_block().get_name() == "DOTA"


    ########
    # Dice #
    ########
    def test_dice(self):
        block = BetaCristobalit().generate([2, 2, 2], "z")
        block.set_name("dice")
        sms.StructureWriter(block, "output").write_gro()
        dice = Dice(block, 0.4, True)

        # Splitting and filling
        assert len(dice.get_origin()) == 120
        assert dice.get_origin()[(1, 1, 1)] == [0.4, 0.4, 0.4]
        assert dice.get_pointer()[(1, 1, 1)] == [14, 51, 52, 64, 65, 67]

        # Iterator
        assert dice._right((1, 1, 1)) == (2, 1, 1)
        assert dice._left((1, 1, 1)) == (0, 1, 1)
        assert dice._top((1, 1, 1)) == (1, 2, 1)
        assert dice._bot((1, 1, 1)) == (1, 0, 1)
        assert dice._front((1, 1, 1)) == (1, 1, 2)
        assert dice._back((1, 1, 1)) == (1, 1, 0)
        assert len(dice.neighbor((1, 1, 1))) == 27
        assert len(dice.neighbor((1, 1, 1), False)) == 26

        # Search
        assert dice.find([(1, 1, 1)], ["Si", "O"], [0.155-0.005, 0.155+0.005]) == [[51, [46, 14, 52, 65]], [64, [26, 63, 65, 67]]]
        assert dice.find([(1, 1, 1)], ["O", "Si"], [0.155-0.005, 0.155+0.005]) == [[14, [51, 13]], [52, [51, 49]], [65, [51, 64]], [67, [64, 69]]]
        assert dice.find([(0, 0, 0)], ["Si", "O"], [0.155-0.005, 0.155+0.005]) == [[3, [4, 9, 2, 174]], [5, [306, 110, 4, 6]]]
        assert dice.find([(0, 0, 0)], ["O", "Si"], [0.155-0.005, 0.155+0.005]) == [[4, [3, 5]], [6, [7, 5]], [9, [3, 11]]]

        # Full search
        assert len(dice.find(None, ["Si", "O"], [0.155-0.005, 0.155+0.005])) == 192
        assert len(dice.find(None, ["O", "Si"], [0.155-0.005, 0.155+0.005])) == 384

        # Setter Getter
        dice.set_pbc(True)
        assert dice.get_count() == [5, 4, 6]
        assert dice.get_size() == 0.4
        assert dice.get_mol().get_name() == "dice"


    ##########
    # Matrix #
    ##########
    def test_matrix(self):
        orient = "z"
        block = BetaCristobalit().generate([1, 1, 1], orient)
        block.set_name("matrix")
        sms.StructureWriter(block, "output").write_gro()
        dice = Dice(block, 0.2, True)
        bonds = dice.find(None, ["Si", "O"], [0.155-1e-2, 0.155+1e-2])

        matrix = Matrix(bonds)
        connect = matrix.get_matrix()
        matrix.split(0, 17)
        assert connect[0]["atoms"] == [30, 8, 1]
        assert connect[17]["atoms"] == [19]
        matrix.strip(0)
        assert connect[0]["atoms"] == []
        assert connect[1]["atoms"] == [43]
        assert connect[8]["atoms"] == [7]
        assert connect[30]["atoms"] == [3]
        assert matrix.bound(0) == [0]
        assert matrix.bound(1, "lt") == [0]
        assert matrix.bound(4, "gt") == []
        matrix.add(0, 17)
        assert connect[0]["atoms"] == [17]
        assert connect[17]["atoms"] == [19, 0]

        print()
        with pytest.raises(ValueError, match="Wrong logic statement"):
            matrix.bound(4, "test")


    #########
    # Shape #
    #########
    def test_shape_cylinder(self):
        block = BetaCristobalit().generate([6, 6, 6], "z")
        block.set_name("shape_cylinder")
        dice = Dice(block, 0.4, True)
        matrix = Matrix(dice.find(None, ["Si", "O"], [0.155-1e-2, 0.155+1e-2]))
        centroid = block.centroid()
        central = geometry.unit(geometry.rotate([0, 0, 1], [1, 0, 0], 45, True))

        cylinder = Cylinder(
            CylinderConfig(
                centroid=tuple(centroid),
                central=tuple(central),
                length=3,
                diameter=4,
            )
        )
        assert isinstance(cylinder.get_config(), CylinderConfig)

        # Properties
        assert round(cylinder.volume(), 4) == 37.6991
        assert round(cylinder.surface(), 4) == 37.6991

        # Test vector
        vec = [3.6086, 4.4076, 0.2065]

        # Surface
        assert [round(x[0][20], 4) for x in cylinder.surf(num=100)] == vec
        assert [round(x[0][20], 4) for x in cylinder.rim(0, num=100)] == vec

        # Normal
        assert [round(x, 4) for x in cylinder.convert([0, 0, 0], False)] == [3.0147, 3.0572, 1.5569]
        assert [round(x, 4) for x in cylinder.normal(vec)] == [0.5939, 2.9704, 0.0000]

        # Positioning
        del_list = [atom_id for atom_id, atom in enumerate(block.get_atom_list()) if cylinder.is_in(atom.get_pos())]
        matrix.strip(del_list)
        block.delete(matrix.bound(0))
        assert block.get_num() == 12650

        # Write molecule
        sms.StructureWriter(block, "output").write_gro()

        # Plot surface
        plt.figure()
        cylinder.plot(vec=[3.17290646, 4.50630614, 0.22183271])
        # plt.show()

    def test_shape_sphere(self):
        block = BetaCristobalit().generate([6, 6, 6], "z")
        block.set_name("shape_sphere")
        dice = Dice(block, 0.4, True)
        matrix = Matrix(dice.find(None, ["Si", "O"], [0.155-1e-2, 0.155+1e-2]))
        centroid = block.centroid()
        central = geometry.unit(geometry.rotate([0, 0, 1], [1, 0, 0], 0, True))

        sphere = Sphere(
            SphereConfig(
                centroid=tuple(centroid),
                central=tuple(central),
                diameter=4,
            )
        )
        assert isinstance(sphere.get_config(), SphereConfig)

        # Properties
        assert round(sphere.volume(), 4) == 33.5103
        assert round(sphere.surface(), 4) == 50.2655

        # Surface
        assert [round(x[0][20], 4) for x in sphere.surf(num=100)] == [4.2006, 3.0572, 4.6675]
        assert [round(x[0][20], 4) for x in sphere.rim(0, num=100)] == [4.9245, 3.0572, 3.6508]

        # Normal
        assert [round(x, 4) for x in sphere.convert([0, 0, 0], False)] == [3.0147, 3.0572, 3.0569]
        assert [round(x, 4) for x in sphere.normal([4.2006, 3.0572, 4.6675])] == [1.4063, 0.0000, 1.9099]

        # Positioning
        del_list = [atom_id for atom_id, atom in enumerate(block.get_atom_list()) if sphere.is_in(atom.get_pos())]
        matrix.strip(del_list)
        block.delete(matrix.bound(0))
        assert block.get_num() == 12934

        # Write molecule
        sms.StructureWriter(block, "output").write_gro()

        # Plot surface
        sphere.plot(inp=3.14, vec=[1.08001048, 3.09687610, 1.72960828])
        # plt.show()

    def test_shape_cuboid(self):
        block = BetaCristobalit().generate([6, 6, 6], "z")
        block.set_name("shape_cuboid")
        dice = Dice(block, 0.4, True)
        matrix = Matrix(dice.find(None, ["Si", "O"], [0.155-1e-2, 0.155+1e-2]))
        centroid = block.centroid()
        central = geometry.unit(geometry.rotate([0, 0, 1], [1, 0, 0], 0, True))

        cuboid = Cuboid(
            CuboidConfig(
                centroid=tuple(centroid),
                central=tuple(central),
                length=10,
                width=6,
                height=4,
            )
        )
        assert isinstance(cuboid.get_config(), CuboidConfig)

        # Properties
        assert round(cuboid.volume(), 4) == 240
        assert round(cuboid.surface(), 4) == 248

        # Normal
        assert [round(x, 4) for x in cuboid.convert([0, 0, 0], False)] == [0.0147, 1.0572, -1.9431]
        assert [round(x, 4) for x in cuboid.normal([4.2636, 3.0937, 4.745])] == [0, 1, 0]

        # Positioning
        del_list = [atom_id for atom_id, atom in enumerate(block.get_atom_list()) if cuboid.is_in(atom.get_pos())]
        matrix.strip(del_list)
        block.delete(matrix.bound(0))
        assert block.get_num() == 5160

        # Write molecule
        sms.StructureWriter(block, "output").write_gro()

        # Plot surface
        cuboid.plot()
        # plt.show()

    def test_shape_cone(self):
        block = BetaCristobalit().generate([6, 6, 6], "z")
        block.set_name("shape_cone")
        dice = Dice(block, 0.4, True)
        matrix = Matrix(dice.find(None, ["Si", "O"], [0.155-1e-2, 0.155+1e-2]))
        centroid = block.centroid()
        central = geometry.unit(geometry.rotate([0, 0, 1], [1, 0, 0], 45, True))

        cone = Cone(
            ConeConfig(
                centroid=tuple(centroid),
                central=tuple(central),
                length=6,
                diameter_1=4,
                diameter_2=1,
            )
        )
        assert isinstance(cone.get_config(), ConeConfig)

        # Properties
        assert round(cone.volume(), 4) == 32.9867
        assert round(cone.surface(), 4) == 48.5742

        # Test vector
        vec = [3.6977, 4.6102, -1.4961]

        # Surface
        assert [round(x[0][20], 4) for x in cone.surf(num=100)] == vec
        assert [round(x[0][20], 4) for x in cone.rim(0, num=100)] == vec

        # Normal
        assert [round(x, 4) for x in cone.convert([0, 0, 0], False)] == [3.0147, 3.0572, 0.0569]
        assert [round(x, 4) for x in cone.normal(vec)] == [0.3182, 2.0114, 0.6109]

        # Positioning
        del_list = [atom_id for atom_id, atom in enumerate(block.get_atom_list()) if cone.is_in(atom.get_pos())]
        matrix.strip(del_list)
        block.delete(matrix.bound(0))
        assert block.get_num() == 12486

        # Write molecule
        sms.StructureWriter(block, "output").write_gro()

        # Plot surface
        plt.figure()
        cone.plot(vec=[3.17290646, 4.50630614, 0.22183271])
        # plt.show()
