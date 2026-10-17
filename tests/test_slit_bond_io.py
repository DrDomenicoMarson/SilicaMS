"""Full-slit topology/coordinate correspondence and chemical-edge contracts."""

import pytest

from silicams._gro_io import _load_gro_system
from silicams._slit_bond_io import load_slit_bonds


@pytest.fixture
def slit_bond_input(tmp_path, write_gro):
    """Provide matching repeated-name GRO and flat ITP inputs."""
    gro = tmp_path / "slit.gro"
    write_gro(gro, [
        (1, "SI", "Si1", 1, .1, .1, .1),
        (2, "OM", "O1", 2, .2, .1, .1),
        (3, "SI", "Si1", 3, .3, .1, .1),
    ], (2., 2., 2.))
    itp = tmp_path / "slit.itp"
    itp.write_text(
        "[ moleculetype ]\nSIL 3\n[ atoms ]\n"
        "1 SI 1 SI Si1 1 0 28\n2 OM 2 OM O1 2 0 16\n3 SI 3 SI Si1 3 0 28\n"
        "[ bonds ]\n1 2 1 .165 1000\n1 2 6 .165 500\n2 3 5\n1 3 6 .33 500\n"
    )
    return itp, _load_gro_system(gro)


def test_slit_itp_keeps_chemical_edges_and_repeated_names(slit_bond_input):
    """Retain junctions, deduplicate chemical terms, and omit restraint-only edges."""
    assert load_slit_bonds(*slit_bond_input) == ((0, 1), (1, 2))


@pytest.mark.parametrize("before,after,match", (
    ("3 SI 3", "4 SI 3", "indices/count"),
    ("2 OM 2 OM O1", "2 OM 2 OM O2", "identity"),
    ("3 SI 3 SI", "3 SI 3 LIG", "identity"),
    ("1 2 1 .165", "1 9 1 .165", "undefined atom indices"),
    ("1 2 1 .165", "1 1 1 .165", "self bond"),
    ("1 2 1 .165", "1 2 11 .165", "unsupported bond function"),
    ("[ moleculetype ]", "[ constraints ]", "exactly one"),
    ("[ bonds ]", "[ constraints ]", "Unsupported slit topology sections"),
    ("[ bonds ]", "#include \"other.itp\"\n[ bonds ]", "preprocessor"),
))
def test_slit_itp_rejects_ambiguous_or_mismatched_input(slit_bond_input, before, after, match):
    """Fail before filtering when the supplied topology cannot identify exact edges."""
    path, system = slit_bond_input
    path.write_text(path.read_text().replace(before, after))
    with pytest.raises(ValueError, match=match):
        load_slit_bonds(path, system)
