#!/usr/bin/env python3
"""Build the TEPS-example slit series."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import silicams as sms

SCRIPT_DIR = Path(__file__).resolve().parent
SLIT_WIDTH = 7.0
SYSTEM_ORDER = ("0_0", "9_1", "8_2", "7_3", "6_4")

systems = {
    "0_0": sms.ExperimentalSiliconStateTarget(
        surface_silicon_fraction=0.60,
        q2_fraction=0.0170,
        q3_fraction=0.1675,
    ),
    "9_1": sms.ExperimentalSiliconStateTarget(
        surface_silicon_fraction=0.571,
        q2_fraction=0.0133,
        q3_fraction=0.1735,
        t2_fraction=0.0195,
        t3_fraction=0.0367,
    ),
    "8_2": sms.ExperimentalSiliconStateTarget(
        surface_silicon_fraction=0.563,
        q2_fraction=0.0158,
        q3_fraction=0.1637,
        t2_fraction=0.0508,
        t3_fraction=0.0818,
    ),
    "7_3": sms.ExperimentalSiliconStateTarget(
        surface_silicon_fraction=0.472,
        q2_fraction=0.0101,
        q3_fraction=0.1275,
        t2_fraction=0.0622,
        t3_fraction=0.1144,
    ),
    "6_4": sms.ExperimentalSiliconStateTarget(
        surface_silicon_fraction=0.398,
        q2_fraction=0.0149,
        q3_fraction=0.0976,
        t2_fraction=0.0605,
        t3_fraction=0.1836,
    ),
}


def _random_seed_for_system(select, seed_base):
    """Return the per-system variant seed for one TEPS-example build.

    Parameters
    ----------
    select : str
        System selector such as ``"9_1"``.
    seed_base : int or None
        Optional base seed from the command line. When omitted, the slit
        builder keeps its deterministic legacy ordering. When supplied, each
        system receives ``seed_base + index`` according to ``SYSTEM_ORDER``.

    Returns
    -------
    seed : int or None
        Per-system seed forwarded to :class:`silicams.AmorphousSlitConfig`.
    """
    if seed_base is None:
        return None
    return seed_base + SYSTEM_ORDER.index(select)


def do_stuff(select, seed_base=None):
    """Build one bare or TEPS-functionalized slit example.

    Parameters
    ----------
    select : str
        System selector. Supported values are listed in ``SYSTEM_ORDER``.
    seed_base : int or None, optional
        Optional base seed used to generate alternative slit variants with the
        same requested ``Q2/Q3/Q4/T2/T3`` target.
    """
    NAME = f"msn_{select}"
    slit_config = sms.AmorphousSlitConfig(
        name=NAME,
        slit_width_nm=SLIT_WIDTH,
        repeat_y=1,
        surface_target=systems[select],
        random_seed=_random_seed_for_system(select, seed_base),
    )

    if select == "0_0":
        result = sms.prepare_amorphous_slit_surface(slit_config)
        print(result.report.final_surface)
        result = sms.write_bare_amorphous_slit(
            str(SCRIPT_DIR / f"msn_{select}"),
            slit_config,
            write_pdb=True,
            write_cif=True,
        )
        print(result.bare_charge_diagnostics.is_neutral)
    else:
        geminal_cross_terms = sms.SilaneGeminalCrossTerms(
            first_ligand_atom_name="CA1",
            scaffold_oxygen_mount_ligand_angle=sms.GromacsAngleParameters.harmonic(
                angle_deg=103.7000444,
                force_constant=836.8,
            ),
            geminal_oxygen_mount_ligand_angle=sms.GromacsAngleParameters.harmonic(
                angle_deg=103.7000444,
                force_constant=1034.033760,
            ),
            geminal_dihedrals=(),
        )

        topology = sms.SilaneTopologyConfig(
            itp_path=str(SCRIPT_DIR / "TEPS_T2.itp"),
            moleculetype_name="TPS",
            geminal_cross_terms=geminal_cross_terms,
        )

        ligand = sms.SilaneAttachmentConfig(
            molecule=sms.Molecule("TPS", "TPS", str(SCRIPT_DIR / "TEPS.pdb")),
            mount=3,        # 0-based atom indexes
            axis=(3, 2),    # 0-based atom indexes
            rotate_about_axis=True,
            rotate_step_deg=20.0,
            topology=topology,
        )

        functionalized_config = sms.FunctionalizedAmorphousSlitConfig(
            slit_config=slit_config,
            ligand=ligand,
        )

        result = sms.write_functionalized_amorphous_slit(
            str(SCRIPT_DIR / f"msn_{select}"),
            functionalized_config,
            write_pdb=True,
            write_cif=True,
        )
        print(result.report.final_surface)
        print(result.charge_diagnostics.is_valid)


def main(argv=None):
    """Parse command-line options and build the TEPS-example slit series.

    Parameters
    ----------
    argv : list[str] or None, optional
        Optional argument vector. When omitted, arguments are read from the
        process command line.
    """
    parser = argparse.ArgumentParser(
        description="Build the TEPS-example bare and functionalized slit series."
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help=(
            "Optional base seed for variant generation. When omitted, the slit "
            "builder keeps its deterministic legacy ordering."
        ),
    )
    args = parser.parse_args(argv)

    for sel in SYSTEM_ORDER:
        do_stuff(sel, seed_base=args.seed_base)


if __name__ == "__main__":
    main()
