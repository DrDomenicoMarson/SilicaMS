from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

import silicams.slit_fill as slit_fill_mod


def _gro_atom_line(
    residue_id: int,
    residue_name: str,
    atom_name: str,
    atom_id: int,
    x: float,
    y: float,
    z: float,
) -> str:
    """Return one minimal GRO atom line."""

    return (
        f"{residue_id % 100000:5d}"
        f"{residue_name[:5]:<5}"
        f"{atom_name[:5]:>5}"
        f"{atom_id % 100000:5d}"
        f"{x:8.3f}"
        f"{y:8.3f}"
        f"{z:8.3f}\n"
    )


def _write_gro(
    path: Path,
    atoms: list[tuple[int, str, str, int, float, float, float]],
    box_values: tuple[float, ...],
    title: str = "test",
) -> None:
    """Write a simple GRO file for tests."""

    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{title}\n")
        handle.write(f"{len(atoms)}\n")
        for atom in atoms:
            handle.write(_gro_atom_line(*atom))
        if len(box_values) == 3:
            handle.write(
                f"{box_values[0]:10.5f}{box_values[1]:10.5f}{box_values[2]:10.5f}\n"
            )
        else:
            handle.write(" ".join(f"{value:10.5f}" for value in box_values) + "\n")


def _surface_residue(
    residue_id: int,
    atom_id_start: int,
    x: float,
    y: float,
    z: float,
    residue_name: str = "SUR",
) -> list[tuple[int, str, str, int, float, float, float]]:
    """Return one hydroxylated surface residue used for slit-plane inference."""

    return [
        (residue_id, residue_name, "SI1", atom_id_start, x, y, z),
        (residue_id, residue_name, "O1", atom_id_start + 1, x + 0.05, y, z),
        (residue_id, residue_name, "H1", atom_id_start + 2, x + 0.08, y, z),
    ]


def _ring_residue(
    residue_id: int,
    residue_name: str,
    atom_id_start: int,
    center: tuple[float, float, float],
    radius: float = 0.14,
) -> list[tuple[int, str, str, int, float, float, float]]:
    """Return one six-atom aromatic ring residue in the xy plane."""

    atoms = []
    center_x, center_y, center_z = center
    for index, angle_deg in enumerate(range(0, 360, 60), start=1):
        angle_rad = math.radians(angle_deg)
        atoms.append(
            (
                residue_id,
                residue_name,
                f"CA{index}",
                atom_id_start + index - 1,
                center_x + radius * math.cos(angle_rad),
                center_y + radius * math.sin(angle_rad),
                center_z,
            )
        )
    return atoms


def _bond_residue(
    residue_id: int,
    residue_name: str,
    atom_id_start: int,
    start: tuple[float, float, float],
    stop: tuple[float, float, float],
) -> list[tuple[int, str, str, int, float, float, float]]:
    """Return one simple two-carbon bond residue."""

    return [
        (residue_id, residue_name, "C1", atom_id_start, *start),
        (residue_id, residue_name, "C2", atom_id_start + 1, *stop),
    ]


def _write_basic_slit(path: Path, include_ring: bool = False, include_crossing_bond: bool = False) -> None:
    """Write one small slit file with hydroxylated surface residues."""

    atoms: list[tuple[int, str, str, int, float, float, float]] = []
    atoms.extend(_surface_residue(1, 1, 0.2, 0.5, 0.5))
    atoms.extend(_surface_residue(2, 4, 1.8, 1.5, 1.5))
    next_atom_id = 7
    if include_ring:
        atoms.extend(_ring_residue(3, "SLR", next_atom_id, (1.0, 1.0, 1.0), radius=0.12))
        next_atom_id += 6
    if include_crossing_bond:
        atoms.extend(
            _bond_residue(
                4,
                "BND",
                next_atom_id,
                (1.0, 1.0, 0.92),
                (1.0, 1.0, 1.08),
            )
        )
    _write_gro(path, atoms, (2.0, 2.0, 2.0), title="slit")


def _write_guest_box(
    path: Path,
    include_inside_ring: bool = True,
    include_outside_ring: bool = True,
    include_crossing_bond: bool = False,
    ring_atom_count: int = 6,
    residue_name: str = "THY",
) -> None:
    """Write one small guest reservoir box."""

    atoms: list[tuple[int, str, str, int, float, float, float]] = []
    atom_id = 1
    if include_inside_ring:
        ring_atoms = _ring_residue(1, residue_name, atom_id, (1.5, 1.5, 1.5))
        if ring_atom_count < 6:
            ring_atoms = ring_atoms[:ring_atom_count]
        atoms.extend(ring_atoms)
        atom_id += len(ring_atoms)
        if include_crossing_bond:
            atoms.extend(
                _bond_residue(
                    1,
                    residue_name,
                    atom_id,
                    (1.0, 1.0, 0.92),
                    (1.0, 1.0, 1.08),
                )
            )
            atom_id += 2
    if include_outside_ring:
        atoms.extend(_ring_residue(2, residue_name, atom_id, (0.2, 1.5, 1.5)))
    _write_gro(path, atoms, (3.0, 3.0, 3.0), title="guest")


def test_center_crop_guest_residues(module_workspace) -> None:
    """The centered crop should keep only residues fully inside the slit box."""

    guest_path = module_workspace.root / "guest_crop.gro"
    _write_guest_box(guest_path)

    guest_system = slit_fill_mod._load_gro_system(guest_path)
    translated_coordinates, selected_mask, crop_window_start = slit_fill_mod._center_crop_guest_residues(
        guest_system=guest_system,
        final_box_lengths=np.array([2.0, 2.0, 2.0], dtype=float),
    )

    assert selected_mask.tolist() == [True, False]
    assert np.allclose(crop_window_start, np.array([0.5, 0.5, 0.5], dtype=float))
    assert np.allclose(
        translated_coordinates[0],
        guest_system.coordinates[0] - np.array([0.5, 0.5, 0.5], dtype=float),
    )


def test_infer_surface_plane_region_detects_x_axis(module_workspace) -> None:
    """Hydroxylated surface Si atoms should define the slit-normal axis."""

    slit_path = module_workspace.root / "slit_plane.gro"
    _write_basic_slit(slit_path)
    slit_system = slit_fill_mod._load_gro_system(slit_path)

    plane_region = slit_fill_mod._infer_surface_plane_region(
        slit_system=slit_system,
        slit_coordinates=slit_system.coordinates,
        box_lengths=slit_system.box_lengths,
        padding_nm=0.0,
    )

    assert plane_region.axis_index == 0
    assert plane_region.axis_name == "x"
    assert plane_region.interval_wraps is False
    assert plane_region.lower_plane_nm == pytest.approx(0.2)
    assert plane_region.upper_plane_nm == pytest.approx(1.8)
    assert plane_region.accessible_width_nm == pytest.approx(1.6)


def test_identify_clashes_detects_forward_ring_crossing(module_workspace) -> None:
    """A target bond through a slit aromatic ring should trigger the forward mask."""

    slit_path = module_workspace.root / "slit_forward.gro"
    guest_path = module_workspace.root / "guest_forward.gro"
    _write_basic_slit(slit_path, include_ring=True, include_crossing_bond=False)
    _write_guest_box(
        guest_path,
        include_inside_ring=True,
        include_outside_ring=False,
        include_crossing_bond=True,
    )

    slit_system = slit_fill_mod._load_gro_system(slit_path)
    guest_system = slit_fill_mod._load_gro_system(guest_path)
    clash_selection, _ = slit_fill_mod._identify_clashing_target_residues(
        guest_system=guest_system,
        slit_system=slit_system,
        translated_guest_coordinates=guest_system.coordinates.copy(),
        selected_residue_mask=np.array([True], dtype=bool),
        slit_coordinates=slit_system.coordinates.copy(),
        final_box_lengths=np.array([3.0, 3.0, 3.0], dtype=float),
        target_resname="THY",
        general_cutoff_nm=0.05,
        ring_atom_prefix="CA",
        ring_plane_tolerance_nm=0.04,
        ring_polygon_padding_nm=0.02,
        include_hydrogen_bonds_in_ring_check=True,
    )

    assert clash_selection.removed_by_forward_ring_mask.tolist() == [True]
    assert clash_selection.removed_by_reverse_ring_mask.tolist() == [False]


def test_identify_clashes_detects_reverse_ring_crossing(module_workspace) -> None:
    """A slit bond through a target aromatic ring should trigger the reverse mask."""

    slit_path = module_workspace.root / "slit_reverse.gro"
    guest_path = module_workspace.root / "guest_reverse.gro"
    _write_basic_slit(slit_path, include_ring=False, include_crossing_bond=True)
    _write_gro(
        guest_path,
        _ring_residue(1, "THY", 1, (1.0, 1.0, 1.0)),
        (3.0, 3.0, 3.0),
        title="guest-reverse",
    )

    slit_system = slit_fill_mod._load_gro_system(slit_path)
    guest_system = slit_fill_mod._load_gro_system(guest_path)
    clash_selection, _ = slit_fill_mod._identify_clashing_target_residues(
        guest_system=guest_system,
        slit_system=slit_system,
        translated_guest_coordinates=guest_system.coordinates.copy(),
        selected_residue_mask=np.array([True], dtype=bool),
        slit_coordinates=slit_system.coordinates.copy(),
        final_box_lengths=np.array([3.0, 3.0, 3.0], dtype=float),
        target_resname="THY",
        general_cutoff_nm=0.05,
        ring_atom_prefix="CA",
        ring_plane_tolerance_nm=0.04,
        ring_polygon_padding_nm=0.02,
        include_hydrogen_bonds_in_ring_check=True,
    )

    assert clash_selection.removed_by_forward_ring_mask.tolist() == [False]
    assert clash_selection.removed_by_reverse_ring_mask.tolist() == [True]


def test_fill_slit_writes_merged_gro_and_human_report(module_workspace, capsys) -> None:
    """The packaged API should write the merged GRO report without stdout output."""

    guest_path = module_workspace.root / "guest_fill.gro"
    slit_path = module_workspace.root / "slit_fill.gro"
    output_path = module_workspace.root / "merged_fill.gro"
    log_path = module_workspace.root / "merged_fill.log"
    _write_guest_box(guest_path)
    _write_basic_slit(slit_path)

    report = slit_fill_mod.fill_slit(
        slit_fill_mod.SlitFillConfig(
            guest_path=guest_path,
            slit_path=slit_path,
            output_path=output_path,
            log_path=log_path,
            density_sample_count=500,
            density_seed_count=1,
            density_probe_radii_nm=(0.0,),
            random_seed=7,
        )
    )
    captured = capsys.readouterr()

    merged_system = slit_fill_mod._load_gro_system(output_path)
    log_text = log_path.read_text(encoding="utf-8")

    assert captured.out == ""
    assert captured.err == ""
    assert report.remaining_guest_molecules == 1
    assert report.final_atom_count == 12
    assert report.final_residue_count == 3
    assert len(merged_system.residue_spans) == 3
    assert output_path.is_file()
    assert "Slit Fill Report" in log_text
    assert "Inputs" in log_text
    assert "Selection" in log_text
    assert "Surface planes" in log_text
    assert "Clash filters" in log_text
    assert "Ring checks" in log_text
    assert "Density" in log_text
    assert "Probe details" in log_text
    assert "Probe 0.00 nm" in log_text
    assert "Output" in log_text
    assert "General cutoff" in log_text
    assert "0.100 nm" in log_text


def test_density_analysis_is_reproducible_and_cli_helpers_accept_argv(
    module_workspace,
    capsys,
) -> None:
    """Density analysis CLI helpers should be reproducible and stay silent."""

    guest_path = module_workspace.root / "guest_cli.gro"
    slit_path = module_workspace.root / "slit_cli.gro"
    merged_path = module_workspace.root / "merged_cli.gro"
    fill_log_path = module_workspace.root / "merged_cli.log"
    density_log_path = module_workspace.root / "merged_cli_density.log"
    _write_guest_box(guest_path)
    _write_basic_slit(slit_path)

    fill_report = slit_fill_mod.fill_slit_main(
        [
            "--guest",
            str(guest_path),
            "--slit",
            str(slit_path),
            "--output",
            str(merged_path),
            "--log",
            str(fill_log_path),
            "--density-samples",
            "400",
            "--density-seed-count",
            "1",
            "--density-probe-radius",
            "0.0",
            "--random-seed",
            "5",
        ]
    )
    assert fill_report.remaining_guest_molecules == 1

    config = slit_fill_mod.SlitDensityConfig(
        input_path=merged_path,
        log_path=density_log_path,
        density_probe_radii_nm=(0.0,),
        density_sample_count=600,
        density_seed_count=2,
        random_seed=11,
    )
    report_a = slit_fill_mod.estimate_guest_density(config)
    report_b = slit_fill_mod.estimate_guest_density(config)
    cli_report = slit_fill_mod.estimate_guest_density_main(
        [
            "--input",
            str(merged_path),
            "--log",
            str(density_log_path),
            "--density-probe-radius",
            "0.0",
            "--density-samples",
            "600",
            "--density-seed-count",
            "2",
            "--random-seed",
            "11",
        ]
    )

    probe_a = report_a.density_estimate.probe_estimates[0]
    probe_b = report_b.density_estimate.probe_estimates[0]
    probe_cli = cli_report.density_estimate.probe_estimates[0]

    assert report_a.guest_molecule_count == 1
    assert report_a.guest_atom_count == 6
    assert report_a.framework_atom_count == 6
    assert probe_a.seed_values == probe_b.seed_values == probe_cli.seed_values
    assert probe_a.accessible_volumes_nm3 == probe_b.accessible_volumes_nm3 == probe_cli.accessible_volumes_nm3
    density_log_text = density_log_path.read_text(encoding="utf-8")

    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == ""
    assert "Slit Density Report" in density_log_text
    assert "Probe 0.00 nm" in density_log_text
    assert density_log_text.index("Probe details") < density_log_text.index("Density summary")


def test_fill_slit_raises_when_target_residue_missing(module_workspace) -> None:
    """The fill workflow should fail early when the target residue is absent."""

    guest_path = module_workspace.root / "guest_missing_target.gro"
    slit_path = module_workspace.root / "slit_missing_target.gro"
    _write_guest_box(guest_path, residue_name="SOL")
    _write_basic_slit(slit_path)

    with pytest.raises(ValueError, match="Residue name 'THY' was not found"):
        slit_fill_mod.fill_slit(
            slit_fill_mod.SlitFillConfig(
                guest_path=guest_path,
                slit_path=slit_path,
                output_path=module_workspace.root / "unused.gro",
            )
        )


def test_fill_slit_raises_when_slit_box_is_larger_than_guest_box(module_workspace) -> None:
    """The slit box must fit inside the guest reservoir box for center cropping."""

    guest_path = module_workspace.root / "guest_small_box.gro"
    slit_path = module_workspace.root / "slit_large_box.gro"
    _write_guest_box(guest_path, include_outside_ring=False)
    _write_gro(
        guest_path,
        _ring_residue(1, "THY", 1, (0.5, 0.5, 0.5)),
        (1.0, 1.0, 1.0),
        title="small-guest",
    )
    _write_basic_slit(slit_path)

    with pytest.raises(ValueError, match="slit box is larger than the guest box"):
        slit_fill_mod.fill_slit(
            slit_fill_mod.SlitFillConfig(
                guest_path=guest_path,
                slit_path=slit_path,
                output_path=module_workspace.root / "unused_large_box.gro",
            )
        )


def test_fill_slit_raises_for_non_orthorhombic_box(module_workspace) -> None:
    """A non-orthorhombic GRO box should be rejected."""

    guest_path = module_workspace.root / "guest_ortho.gro"
    slit_path = module_workspace.root / "slit_non_ortho.gro"
    _write_guest_box(guest_path, include_outside_ring=False)
    _write_gro(
        slit_path,
        _surface_residue(1, 1, 0.2, 0.5, 0.5) + _surface_residue(2, 4, 1.8, 1.5, 1.5),
        (2.0, 2.0, 2.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
        title="non-ortho",
    )

    with pytest.raises(ValueError, match="non-orthorhombic"):
        slit_fill_mod.fill_slit(
            slit_fill_mod.SlitFillConfig(
                guest_path=guest_path,
                slit_path=slit_path,
                output_path=module_workspace.root / "unused_non_ortho.gro",
            )
        )


def test_fill_slit_raises_when_target_ring_is_missing(module_workspace) -> None:
    """Ring checks require exactly six aromatic atoms on the target residue."""

    guest_path = module_workspace.root / "guest_missing_ring.gro"
    slit_path = module_workspace.root / "slit_missing_ring.gro"
    _write_guest_box(guest_path, include_outside_ring=False, ring_atom_count=5)
    _write_basic_slit(slit_path)

    with pytest.raises(ValueError, match="does not contain exactly six atoms"):
        slit_fill_mod.fill_slit(
            slit_fill_mod.SlitFillConfig(
                guest_path=guest_path,
                slit_path=slit_path,
                output_path=module_workspace.root / "unused_missing_ring.gro",
            )
        )


def test_density_analysis_raises_without_framework(module_workspace) -> None:
    """Merged systems that contain only target residues should be rejected."""

    merged_path = module_workspace.root / "merged_no_framework.gro"
    _write_gro(
        merged_path,
        _ring_residue(1, "THY", 1, (1.0, 1.0, 1.0)),
        (2.0, 2.0, 2.0),
        title="merged-no-framework",
    )

    with pytest.raises(ValueError, match="does not contain any non-target atoms"):
        slit_fill_mod.estimate_guest_density(
            slit_fill_mod.SlitDensityConfig(input_path=merged_path)
        )


def test_density_analysis_skips_nominal_box_warning(module_workspace) -> None:
    """Density analysis should not warn for framework atoms outside the nominal box."""

    merged_path = module_workspace.root / "merged_outside_box.gro"
    atoms: list[tuple[int, str, str, int, float, float, float]] = []
    atoms.extend(_surface_residue(1, 1, 2.2, 0.5, 0.5))
    atoms.extend(_surface_residue(2, 4, 1.8, 1.5, 1.5))
    atoms.extend(_ring_residue(3, "THY", 7, (1.0, 1.0, 1.0)))
    _write_gro(merged_path, atoms, (2.0, 2.0, 2.0), title="merged-outside-box")

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        report = slit_fill_mod.estimate_guest_density(
            slit_fill_mod.SlitDensityConfig(
                input_path=merged_path,
                density_probe_radii_nm=(0.0,),
                density_sample_count=200,
                density_seed_count=1,
                random_seed=3,
            )
        )

    assert report.framework_atom_count == 6
    assert not any(
        "fall outside the nominal slit box range" in str(warning.message)
        for warning in caught_warnings
    )
