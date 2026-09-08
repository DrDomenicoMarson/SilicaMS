"""Strict flat-ITP parser and topology-bundle validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import silicams.topology as topology_mod


def _write_itp(path: Path, text: str) -> Path:
    """Write one flat topology fixture and return its path.

    Parameters
    ----------
    path : Path
        Fixture destination.
    text : str
        Complete topology text.

    Returns
    -------
    Path
        The written fixture path.
    """

    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def _single_molecule_itp(interaction_text: str = "") -> str:
    """Return a four-atom flat topology with optional interaction sections.

    Parameters
    ----------
    interaction_text : str, optional
        Interaction sections appended after ``[ atoms ]``.

    Returns
    -------
    str
        Complete flat topology text.
    """

    return f"""
[ moleculetype ]
MOL 3

[ atoms ]
1 c3 1 MOL C1 1 0.0 12.01
2 c3 1 MOL C2 2 0.0 12.01
3 c3 1 MOL C3 3 0.0 12.01
4 c3 1 MOL C4 4 0.0 12.01

{interaction_text}
"""


def test_flat_itp_rejects_multiple_molecule_definition_headers(tmp_path: Path) -> None:
    """Reject the reproduced two-molecule input before aggregating its atoms."""

    path = _write_itp(
        tmp_path / "two_molecules.itp",
        """
[ moleculetype ]
FIRST 3
[ atoms ]
1 c3 1 FIRST A1 1 0.0 12.01

[ moleculetype ]
SECOND 3
[ atoms ]
1 c3 1 SECOND B1 1 0.0 12.01
""",
    )

    with pytest.raises(ValueError, match=r"exactly one molecule definition.*2"):
        topology_mod.parse_flat_itp(path)


def test_flat_itp_rejects_multiple_moleculetype_rows(tmp_path: Path) -> None:
    """Reject multiple molecule declarations even under a single header."""

    path = _write_itp(
        tmp_path / "two_rows.itp",
        """
[ moleculetype ]
FIRST 3
SECOND 3
[ atoms ]
1 c3 1 FIRST A1 1 0.0 12.01
""",
    )

    with pytest.raises(ValueError, match=r"exactly one \[ moleculetype \] row.*2"):
        topology_mod.parse_flat_itp(path)


def test_flat_itp_rejects_duplicate_atom_indices(tmp_path: Path) -> None:
    """Reject ambiguous atom identities inside the single molecule type."""

    path = _write_itp(
        tmp_path / "duplicate_index.itp",
        """
[ moleculetype ]
MOL 3
[ atoms ]
1 c3 1 MOL C1 1 0.0 12.01
1 hc 1 MOL H1 2 0.0 1.008
""",
    )

    with pytest.raises(
        ValueError,
        match=r"duplicate atom index 1.*molecule type 'MOL'",
    ):
        topology_mod.parse_flat_itp(path)


@pytest.mark.parametrize(
    "section_name,interaction_row",
    (
        ("bonds", "1 9 1"),
        ("pairs", "1 9 1"),
        ("angles", "1 2 9 1"),
        ("dihedrals", "1 2 3 9 1"),
    ),
)
def test_flat_itp_rejects_undefined_interaction_atom_indices(
    tmp_path: Path,
    section_name: str,
    interaction_row: str,
) -> None:
    """Validate atom references in every supported interaction section."""

    path = _write_itp(
        tmp_path / f"invalid_{section_name}.itp",
        _single_molecule_itp(
            f"""
[ {section_name} ]
{interaction_row}
"""
        ),
    )

    with pytest.raises(
        ValueError,
        match=rf"\[ {section_name} \] row 1.*'MOL'.*undefined atom indices \[9\]",
    ):
        topology_mod.parse_flat_itp(path)


def test_topology_bundle_direct_construction_validates_references() -> None:
    """Enforce index integrity when callers bypass the text parser."""

    atom = topology_mod.GromacsAtom(
        index=1,
        atom_type="c3",
        residue_number=1,
        residue_name="MOL",
        atom_name="C1",
        charge_group=1,
        charge="0.0",
        mass="12.01",
    )
    molecule = topology_mod.GromacsMoleculeType(
        name="MOL",
        nrexcl=3,
        atoms=(atom,),
        pairs=(topology_mod.GromacsPair(1, 2, 1),),
    )

    with pytest.raises(ValueError, match=r"\[ pairs \].*undefined atom indices \[2\]"):
        topology_mod.ParsedTopologyBundle(
            source_path="direct-construction",
            atomtypes=(),
            moleculetype=molecule,
        )


def test_flat_itp_accepts_repeated_interaction_sections_for_one_molecule(
    tmp_path: Path,
) -> None:
    """Preserve concatenation of repeated interaction blocks within one molecule."""

    path = _write_itp(
        tmp_path / "repeated_dihedrals.itp",
        _single_molecule_itp(
            """
[ dihedrals ]
1 2 3 4 1

[ dihedrals ]
4 3 2 1 1
"""
        ),
    )

    bundle = topology_mod.parse_flat_itp(path, moleculetype_name="MOL")

    assert len(bundle.moleculetype.dihedrals) == 2
    assert bundle.atom_by_name("C4") is bundle.atom_by_index[4]
