"""Build a small TMS-functionalized amorphous silica slit."""

from pathlib import Path

import silicams as sms
from silicams.generic import tms


def main(
    output_dir: str | Path = "output/functionalized_slit",
) -> sms.FunctionalizedSlitResult:
    """Build and write one coordinate-only functionalized slit example.

    Parameters
    ----------
    output_dir : str or Path, optional
        Directory receiving the generated coordinates and reports.

    Returns
    -------
    result : FunctionalizedSlitResult
        Finalized functionalized-slit result and diagnostics.

    Notes
    -----
    No ligand ITP is supplied, so this example intentionally writes
    coordinates and reports without a functionalized GROMACS topology.
    """

    config = sms.FunctionalizedAmorphousSlitConfig(
        slit_config=sms.AmorphousSlitConfig(
            name="tms_functionalized_silica_slit",
            repeat_y=1,
            surface_target=sms.ExperimentalSiliconStateTarget(
                q2_fraction=65 / 957,
                q3_fraction=651 / 957,
                q4_fraction=239 / 957,
                t2_fraction=1 / 957,
                t3_fraction=1 / 957,
                surface_silicon_fraction=1.0,
            ),
        ),
        ligand=sms.SilaneAttachmentConfig(
            molecule=tms(),
            mount=0,
            axis=(0, 1),
            rotate_about_axis=False,
        ),
        progress_settings=sms.FunctionalizedSlitProgressConfig(enabled=True),
    )
    return sms.write_functionalized_amorphous_slit(
        str(output_dir),
        config,
        write_pdb=True,
        write_cif=True,
    )


if __name__ == "__main__":
    main()
