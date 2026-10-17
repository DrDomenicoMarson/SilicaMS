"""Fill a generated silica slit from a larger guest reservoir."""

import argparse
from pathlib import Path
from collections.abc import Sequence

import silicams as sms


def main(argv: Sequence[str] | None = None) -> sms.SlitFillReport:
    """Fill one slit with explicit paths and optional exact ITP connectivity.

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
    parser.add_argument(
        "--slit-topology", type=Path, default=None,
        help="Flat full-slit ITP matching the GRO; recommended for stretched bonds",
    )
    parser.add_argument(
        "--slit-geometry",
        type=Path,
        default=None,
        help=(
            "Explicit schema-v1 geometry YAML from slit construction; use this "
            "for a fully functionalized framework that cannot be inferred"
        ),
    )
    parser.add_argument("--resname", default="THY", help="Guest residue name")
    args = parser.parse_args(argv)

    return sms.fill_slit(
        sms.SlitFillConfig(
            guest_path=args.guest,
            slit_path=args.slit,
            output_path=args.output,
            slit_geometry_path=args.slit_geometry,
            slit_topology_path=args.slit_topology,
            target_resname=args.resname,
        )
    )


if __name__ == "__main__":
    main()
