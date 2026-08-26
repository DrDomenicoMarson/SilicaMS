import numpy as np
import pytest

import silicams as sms


pytestmark = pytest.mark.usefixtures("module_workspace")


class TestArrayBackedMolecule:
    """Validate the array-backed molecule storage and compatibility snapshots."""

    def test_positions_view_tracks_coordinate_edits(self):
        mol = sms.Molecule(inp="data/benzene.gro")

        initial_positions = mol.positions_view().copy()
        np.testing.assert_allclose(initial_positions[0], mol.pos(0))

        mol.translate([0.0, 0.1, 0.2])
        np.testing.assert_allclose(mol.positions_view()[3], mol.pos(3))

        mol.rotate("x", 45)
        np.testing.assert_allclose(mol.positions_view()[3], mol.pos(3), atol=1e-12)

        mol.move(0, [1.0, 1.0, 1.0])
        np.testing.assert_allclose(mol.positions_view()[0], mol.pos(0), atol=1e-12)

        mol.zero()
        np.testing.assert_allclose(
            mol.positions_view().min(axis=0),
            np.zeros(3),
            atol=1e-12,
        )

        assert len(mol.infer_bonds()) == 12

        overlap = sms.Molecule()
        overlap.add("C", [0.0, 0.0, 0.0])
        overlap.add("C", [0.0, 0.0, 0.0])
        assert overlap.overlap() == {0: [1]}

    def test_get_atom_list_returns_detached_snapshots(self):
        mol = sms.Molecule()
        mol.add("C", [0.0, 0.0, 0.0], name="C1")

        snapshot = mol.get_atom_list()
        snapshot[0].set_pos([9.0, 9.0, 9.0])
        snapshot[0].set_atom_type("O")
        snapshot[0].set_name("O1")
        snapshot[0].set_residue(5)

        assert mol.pos(0) == [0.0, 0.0, 0.0]
        assert mol.get_atom_type(0) == "C"
        assert mol.get_atom_list()[0].get_name() == "C1"
        assert mol.get_atom_list()[0].get_residue() == 0
