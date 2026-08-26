"""Fill a generated silica slit from a larger guest reservoir."""

import argparse
from pathlib import Path
from collections.abc import Sequence

import silicams as sms


def main(argv: Sequence[str] | None = None) -> sms.SlitFillReport:
    """Fill one slit using explicit guest, slit, and output paths.

    Parameters
    ----------
    argv : sequence[str] or None, optional
        Optional command-line arguments. When omitted, use ``sys.argv``.

    Returns
    -------
    report : SlitFillReport
        Structured filtering and density report.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("guest", type=Path, help="Guest-reservoir GRO file")
    parser.add_argument("slit", type=Path, help="Generated silica-slit GRO file")
    parser.add_argument("output", type=Path, help="Filled output GRO file")
    parser.add_argument("--resname", default="THY", help="Guest residue name")
    args = parser.parse_args(argv)

    return sms.fill_slit(
        sms.SlitFillConfig(
            guest_path=args.guest,
            slit_path=args.slit,
            output_path=args.output,
            target_resname=args.resname,
        )
    )


if __name__ == "__main__":
    main()
