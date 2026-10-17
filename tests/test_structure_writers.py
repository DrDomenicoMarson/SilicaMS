"""Cross-format numbering and coordinate contracts for shared writer records."""

import numpy as np
import pytest

from silicams.atom import Atom
from silicams.molecule import Molecule
from silicams.writers.structure import StructureWriter
from silicams._gro_io import _load_gro_system


@pytest.mark.parametrize("first_label", (0, 5, 80000))
@pytest.mark.parametrize("use_atom_names", (False, True))
def test_gro_uses_shared_residue_and_atom_records(tmp_path, first_label, use_atom_names):
    """Keep cross-format IDs consistent for arbitrary and recurring residue labels."""
    atoms = []
    for group, label in enumerate((first_label, first_label + 2, first_label)):
        atoms.extend([
            Atom([.5 * group, 0., 0.], "C", "CX", label),
            Atom([.5 * group + .109, 0., 0.], "H", "HX", label),
        ])
    molecule = Molecule(short="LIG", inp=atoms)
    molecule.set_box([2., 2., 2.])
    for start in (0, 2, 4):
        molecule.add_bond(start, start + 1)
    original = StructureWriter(molecule, tmp_path)
    writer = StructureWriter(original.complete_snapshot(), tmp_path)
    writer.write_gro("shared.gro", use_atom_names=use_atom_names)
    writer.write_pdb("shared.pdb", use_atom_names=use_atom_names)
    writer.write_cif("shared.cif", use_atom_names=use_atom_names)
    records = writer._export_cache(use_atom_names).atom_records
    gro = _load_gro_system(tmp_path / "shared.gro")
    pdb = [line for line in (tmp_path / "shared.pdb").read_text().splitlines() if line.startswith("HETATM")]
    cif = [line.split() for line in (tmp_path / "shared.cif").read_text().splitlines() if line.startswith("HETATM")]
    expected = [1, 1, 2, 2, 3, 3]
    assert gro.residue_ids.tolist() == [r.residue_id for r in records] == expected
    assert [int(line[22:26]) for line in pdb] == expected
    assert gro.atom_ids.tolist() == [r.serial for r in records] == list(range(1, 7))
    assert [int(row[1]) for row in cif] == list(range(1, 7))
    assert gro.atom_names == [r.atom_name for r in records]
    assert [line[12:16].strip() for line in pdb] == gro.atom_names
    np.testing.assert_allclose(gro.coordinates, [r.position for r in records], atol=.0005)
