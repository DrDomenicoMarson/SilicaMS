"""Private common text sections for slit filling and density reports."""

from __future__ import annotations

from typing import Sequence

from .slit_geometry import PeriodicSlitGeometry

__all__: list[str] = []


def _format_value_lines(title: str, rows: Sequence[tuple[str, str]]) -> str:
    """Format one aligned report section from label/value rows.

    Parameters
    ----------
    title : str
        Section heading.
    rows : sequence[tuple[str, str]]
        Label/value pairs rendered in the section body.

    Returns
    -------
    str
        Formatted text section.
    """

    width = max((len(label) for label, _ in rows), default=0)
    body = "\n".join(f"  {label:<{width}} : {value}" for label, value in rows)
    return (
        f"{title}\n{'-' * len(title)}\n{body}"
        if body
        else f"{title}\n{'-' * len(title)}"
    )


def _format_geometry_block(
    title: str,
    slit_geometry: PeriodicSlitGeometry,
    padding_nm: float,
) -> str:
    """Format one periodic slit geometry report section.

    Parameters
    ----------
    title : str
        Section heading.
    slit_geometry : PeriodicSlitGeometry
        Geometry to report.
    padding_nm : float
        Signed plane padding in nanometers used by the workflow.

    Returns
    -------
    str
        Human-readable geometry section.
    """

    roughness = (
        "not available"
        if slit_geometry.normal_roughness_rms_nm is None
        else f"{slit_geometry.normal_roughness_rms_nm:.5f} nm"
    )
    support_count = (
        "not available"
        if slit_geometry.surface_support_count is None
        else str(slit_geometry.surface_support_count)
    )
    return _format_value_lines(
        title,
        [
            ("Normal axis", slit_geometry.normal_axis_name),
            ("Axis index", str(slit_geometry.normal_axis_index)),
            ("Lower mean plane", f"{slit_geometry.lower_plane_nm:.5f} nm"),
            ("Upper mean plane", f"{slit_geometry.upper_plane_nm:.5f} nm"),
            ("Interval wraps", str(slit_geometry.interval_wraps)),
            ("Mean-plane separation", f"{slit_geometry.plane_separation_nm:.5f} nm"),
            ("Signed padding", f"{padding_nm:.5f} nm"),
            ("Padded width", f"{slit_geometry.padded_width_nm(padding_nm):.5f} nm"),
            (
                "Projected area per face",
                f"{slit_geometry.projected_area_per_face_nm2:.5f} nm^2",
            ),
            (
                "Total projected surface area",
                f"{slit_geometry.total_projected_surface_area_nm2:.5f} nm^2",
            ),
            (
                "Geometric slit volume",
                f"{slit_geometry.geometric_slit_volume_nm3:.5f} nm^3",
            ),
            (
                "Padded geometric volume",
                f"{slit_geometry.padded_geometric_volume_nm3(padding_nm):.5f} nm^3",
            ),
            ("Surface support positions", support_count),
            ("Normal RMS roughness", roughness),
        ],
    )
