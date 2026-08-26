"""Build a small deterministic bare amorphous silica slit."""

from pathlib import Path

import silicams as sms


def main(output_dir: str | Path = "output/bare_slit") -> sms.SlitPreparationResult:
    """Build and write one bare slit example.

    Parameters
    ----------
    output_dir : str or Path, optional
        Directory receiving the generated structure, topology, and reports.

    Returns
    -------
    result : SlitPreparationResult
        Finalized bare-slit result and diagnostics.
    """

    config = sms.AmorphousSlitConfig(
        name="bare_silica_slit",
        repeat_y=1,
        surface_target=sms.ExperimentalSiliconStateTarget(
            q2_fraction=65 / 957,
            q3_fraction=651 / 957,
            q4_fraction=241 / 957,
            alpha_override=1.0,
        ),
    )
    return sms.write_bare_amorphous_slit(
        str(output_dir),
        config,
        write_pdb=True,
        write_cif=True,
    )


if __name__ == "__main__":
    main()
