from __future__ import annotations

from pathlib import Path
from itertools import product

import numpy as np
import pytest
import yaml

import silicams._gro_io as gro_io_mod
import silicams.slit_fill as slit_fill_mod
import silicams.slit_density as density_mod
import silicams._slit_geometry_io as geometry_io
import silicams._slit_guest_filter as filtering
from silicams import PeriodicSlitGeometry


@pytest.mark.parametrize("general_cutoff_nm", (0.0, -0.1, np.nan))
def test_general_cutoff_must_be_finite_and_positive(general_cutoff_nm: float) -> None:
    """Reject non-physical all-atom construction cutoffs."""

    with pytest.raises(ValueError, match="finite|strictly positive"):
        slit_fill_mod.SlitFillConfig(general_cutoff_nm=general_cutoff_nm)


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("ring_plane_tolerance_nm", np.nan),
        ("ring_polygon_padding_nm", np.inf),
        ("density_probe_radii_nm", (np.nan,)),
    ),
)
def test_fill_config_rejects_non_finite_scientific_values(
    field_name: str,
    value: object,
) -> None:
    """Reject non-finite tolerances, paddings, and density probe radii."""

    with pytest.raises(ValueError, match="finite"):
        slit_fill_mod.SlitFillConfig(**{field_name: value})


@pytest.mark.parametrize(
    "field_name", ("density_sample_count", "density_seed_count", "random_seed")
)
@pytest.mark.parametrize("value", (1.5, True))
def test_fill_config_requires_actual_integer_fields(
    field_name: str,
    value: object,
) -> None:
    """Reject floats and booleans for integer counts and seeds."""

    with pytest.raises(TypeError, match="integer"):
        slit_fill_mod.SlitFillConfig(**{field_name: value})


def test_fill_config_normalizes_numpy_integer_fields() -> None:
    """Accept NumPy integral values while storing ordinary Python integers."""

    config = slit_fill_mod.SlitFillConfig(
        density_sample_count=np.int64(10),
        density_seed_count=np.int64(2),
        random_seed=np.int64(3),
    )

    assert type(config.density_sample_count) is int
    assert type(config.density_seed_count) is int
    assert type(config.random_seed) is int


def test_fill_paths_must_be_distinct() -> None:
    """Reject a report path that would overwrite another member of the set."""

    config = slit_fill_mod.SlitFillConfig(
        output_path=Path("filled.gro"),
        log_path=Path("filled.yml"),
    )
    with pytest.raises(ValueError, match="must be distinct"):
        slit_fill_mod._resolve_fill_config(config)


def test_infer_slit_geometry_detects_x_axis(module_workspace, write_basic_slit) -> None:
    """Hydroxylated surface Si atoms should define the slit-normal axis."""

    slit_path = module_workspace.root / "slit_plane.gro"
    write_basic_slit(slit_path)
    slit_system = gro_io_mod._load_gro_system(slit_path)

    slit_geometry = geometry_io._infer_slit_geometry(
        slit_system=slit_system,
        slit_coordinates=slit_system.coordinates,
        box_lengths=slit_system.box_lengths,
    )

    assert slit_geometry.normal_axis_index == 0
    assert slit_geometry.normal_axis_name == "x"
    assert slit_geometry.interval_wraps is False
    assert slit_geometry.lower_plane_nm == pytest.approx(0.2)
    assert slit_geometry.upper_plane_nm == pytest.approx(1.8)
    assert slit_geometry.plane_separation_nm == pytest.approx(1.6)


def test_fill_slit_writes_merged_gro_and_human_report(
    module_workspace, capsys, write_basic_slit, write_guest_box
) -> None:
    """The packaged API should write the merged GRO report without stdout output."""

    guest_path = module_workspace.root / "guest_fill.gro"
    slit_path = module_workspace.root / "slit_fill.gro"
    output_path = module_workspace.root / "merged_fill.gro"
    log_path = module_workspace.root / "merged_fill.log"
    write_guest_box(guest_path)
    write_basic_slit(slit_path)

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

    merged_system = gro_io_mod._load_gro_system(output_path)
    log_text = log_path.read_text(encoding="utf-8")

    assert captured.out == ""
    assert captured.err == ""
    assert report.remaining_guest_molecules == 1
    assert report.final_atom_count == 12
    assert report.final_residue_count == 3
    assert len(merged_system.residue_spans) == 3
    assert output_path.is_file()
    metadata_path = output_path.with_suffix(".yml")
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["slit_geometry"]["normal_axis_name"] == "z"
    assert metadata["slit_components"] == {
        "selection_basis": "separate_slit_and_guest_inputs",
        "framework_resnames": ["SUR"],
        "mobile_resnames": ["THY"],
        "target_resname": "THY",
        "framework_atom_count": 6,
        "framework_residue_count": 2,
    }
    assert PeriodicSlitGeometry.from_dict(metadata["slit_geometry"]) == (
        report.output_slit_geometry
    )
    assert "Slit Fill Report" in log_text
    assert "Inputs" in log_text
    assert "Selection" in log_text
    assert "Input slit geometry" in log_text
    assert "Output slit geometry" in log_text
    assert "Clash filters" in log_text
    assert "Ring checks" in log_text
    assert "Density" in log_text
    assert "Probe details" in log_text
    assert "Probe 0.00 nm" in log_text
    assert "Output" in log_text
    assert "General cutoff" in log_text
    assert "0.100 nm" in log_text
    assert "Framework residue names" in log_text
    assert "Mobile residue names" in log_text


@pytest.mark.parametrize("normal_axis", (0, 1, 2))
@pytest.mark.parametrize(
    "slit_has_velocities,guest_has_velocities",
    (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ),
)
@pytest.mark.parametrize("wrap_output", (False, True))
def test_fill_slit_keeps_positions_and_velocities_in_the_same_output_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    normal_axis: int,
    slit_has_velocities: bool,
    guest_has_velocities: bool,
    wrap_output: bool,
    write_gro,
) -> None:
    """Reframe retained vectors together without translating velocities or inputs."""

    box = np.array([2.0, 3.0, 4.0])
    slit_coordinates = np.array([[0.1, 0.2, 0.3], [1.9, 2.8, 3.7]])
    boundary_position = box / 2.0
    tangential_axis = (normal_axis + 1) % 3
    boundary_position[tangential_axis] = box[tangential_axis]
    # A retained dimer, a clashing residue, a retained boundary residue, and
    # a residue outside the crop. Distinct velocities detect indexing errors.
    guest_coordinates = (
        np.array(
            [
                [1.0, 1.5, 2.0],
                [1.1, 1.5, 2.0],
                slit_coordinates[0],
                boundary_position,
                [-0.75, -0.75, -0.75],
            ]
        )
        + 1.0
    )
    slit_velocities = np.array([[1.1, -2.2, 3.3], [-4.4, 5.5, -6.6]])
    guest_velocities = np.arange(1.0, 16.0).reshape(5, 3) / 10.0
    guest_velocities[:, 1] *= -1
    slit_path = tmp_path / "slit.gro"
    guest_path = tmp_path / "guest.gro"
    output_path = tmp_path / "filled.gro"
    write_gro(
        slit_path,
        [
            (index + 1, "SIL", "SI1", index + 1, *position)
            for index, position in enumerate(slit_coordinates)
        ],
        tuple(box),
        velocities=slit_velocities if slit_has_velocities else None,
    )
    write_gro(
        guest_path,
        [
            (residue_id, "GAS", atom_name, index + 1, *position)
            for index, (residue_id, atom_name, position) in enumerate(
                zip(
                    (9, 9, 4, 22, 24),
                    ("C1", "O1", "C2", "C3", "C4"),
                    guest_coordinates,
                    strict=True,
                )
            )
        ],
        tuple(box + 2.0),
        velocities=guest_velocities if guest_has_velocities else None,
    )
    load_gro = gro_io_mod._load_gro_system
    sources = {path: load_gro(path) for path in (slit_path, guest_path)}
    source_bytes = {path: path.read_bytes() for path in sources}
    source_arrays = []
    for system in sources.values():
        for array in (system.coordinates, system.velocities, system.box_lengths):
            if array is not None:
                source_arrays.append((array, array.copy()))
                array.setflags(write=False)
    monkeypatch.setattr(slit_fill_mod, "_load_gro_system", sources.__getitem__)

    report = slit_fill_mod.fill_slit(
        slit_fill_mod.SlitFillConfig(
            slit_path=slit_path,
            guest_path=guest_path,
            output_path=output_path,
            target_resname="GAS",
            slit_geometry=PeriodicSlitGeometry(
                box_lengths_nm=tuple(box),
                normal_axis_index=normal_axis,
                lower_plane_nm=0.2,
                upper_plane_nm=float(box[normal_axis] - 0.2),
            ),
            use_surface_plane_filter=False,
            wrap_output=wrap_output,
            density_sample_count=100,
            density_seed_count=1,
            density_probe_radii_nm=(0.0,),
            random_seed=17,
        )
    )

    permutation = ((2, 1, 0), (0, 2, 1), (0, 1, 2))[normal_axis]
    kept_guest_indices = [0, 1, 3]
    expected_guest_coordinates = (guest_coordinates[kept_guest_indices] - 1.0).copy()
    if wrap_output:
        expected_guest_coordinates[-1, tangential_axis] = 0.0
    expected_coordinates = np.vstack((slit_coordinates, expected_guest_coordinates))
    expected_coordinates = expected_coordinates[:, permutation]
    output_system = load_gro(output_path)
    np.testing.assert_allclose(
        output_system.coordinates, expected_coordinates, atol=1e-12
    )
    np.testing.assert_array_equal(output_system.box_lengths, box[list(permutation)])
    assert output_system.atom_names == ["SI1", "SI1", "C1", "O1", "C3"]
    assert output_system.residue_ids.tolist() == [1, 2, 3, 3, 4]
    assert output_system.atom_ids.tolist() == [1, 2, 3, 4, 5]
    assert report.removed_outside_crop_guest_molecules == 1
    assert report.removed_by_general_cutoff_guest_molecules == 1
    assert report.remaining_guest_molecules == 2
    assert report.final_atom_count == 5
    assert report.final_residue_count == 4
    assert report.output_axis_permutation == permutation
    assert report.output_slit_geometry.normal_axis_index == 2
    metadata = yaml.safe_load(output_path.with_suffix(".yml").read_text())
    assert PeriodicSlitGeometry.from_dict(metadata["slit_geometry"]) == (
        report.output_slit_geometry
    )

    atom_lines = output_path.read_text().splitlines()[2:-1]
    if slit_has_velocities or guest_has_velocities:
        expected_velocities = np.vstack(
            (
                slit_velocities if slit_has_velocities else np.zeros((2, 3)),
                guest_velocities[kept_guest_indices]
                if guest_has_velocities
                else np.zeros((3, 3)),
            )
        )[:, permutation]
        np.testing.assert_array_equal(output_system.velocities, expected_velocities)
        assert all(len(line) == 68 for line in atom_lines)
    else:
        assert output_system.velocities is None
        assert all(len(line) == 44 for line in atom_lines)
    for array, original in source_arrays:
        np.testing.assert_array_equal(array, original)
    assert {path: path.read_bytes() for path in sources} == source_bytes


def test_fill_slit_filters_every_residue_type_and_reports_each_one(
    module_workspace, write_gro, ring_residue, write_basic_slit
) -> None:
    """Screen mixed reservoirs while keeping density reporting target-specific."""

    guest_path = module_workspace.root / "guest_mixed.gro"
    slit_path = module_workspace.root / "slit_mixed.gro"
    output_path = module_workspace.root / "merged_mixed.gro"
    target_atoms = ring_residue(1, "THY", 1, (1.5, 1.5, 1.5))
    mixed_atoms = target_atoms + [
        (2, "NA", "NA", 7, 0.75, 1.00, 1.00),
        (3, "CL", "CL", 8, 0.60, 1.20, 1.20),
        (4, "K", "K", 9, 1.30, 1.30, 1.30),
    ]
    write_gro(guest_path, mixed_atoms, (3.0, 3.0, 3.0), title="mixed guest")
    write_basic_slit(slit_path)

    report = slit_fill_mod.fill_slit(
        slit_fill_mod.SlitFillConfig(
            guest_path=guest_path,
            slit_path=slit_path,
            output_path=output_path,
            density_probe_radii_nm=(0.0,),
            density_sample_count=100,
            density_seed_count=1,
            random_seed=9,
        )
    )

    summaries = {
        summary.residue_name: summary for summary in report.residue_filter_summaries
    }
    output_system = gro_io_mod._load_gro_system(output_path)

    assert summaries["NA"].removed_by_general_cutoff_residues == 1
    assert summaries["CL"].removed_by_surface_plane_residues == 1
    assert summaries["K"].remaining_residues == 1
    assert summaries["THY"].remaining_residues == 1
    assert report.remaining_guest_molecules == 1
    assert set(output_system.residue_names) == {"SUR", "THY", "K"}
    density_report = density_mod.estimate_guest_density(
        density_mod.SlitDensityConfig(
            input_path=output_path,
            slit_geometry_path=output_path.with_suffix(".yml"),
            target_resname="THY",
            framework_resnames=("SUR",),
            mobile_resnames=("K",),
            density_probe_radii_nm=(0.0,),
            density_sample_count=100,
            density_seed_count=1,
            random_seed=9,
        )
    )
    assert density_report.framework_resnames == ("SUR",)
    assert density_report.mobile_resnames == ("K", "THY")
    assert (
        density_report.density_estimate.probe_estimates
        == report.density_estimate.probe_estimates
    )
    assert "Residue filtering: NA" in output_path.with_suffix(".log").read_text(
        encoding="utf-8"
    )


def test_fill_reports_preserve_all_overlap_categories(
    tmp_path, write_gro, write_basic_slit, monkeypatch
):
    """Keep target and per-type accounting for every combination of clash masks."""

    guest_path, slit_path = tmp_path / "guest.gro", tmp_path / "slit.gro"
    output_path = tmp_path / "filled.gro"
    write_basic_slit(slit_path)
    atoms = []
    mask_rows = []
    selected_rows = []
    for name in ("THY", "ION"):
        for combination in product((False, True), repeat=3):
            identifier = len(atoms) + 1
            atoms.append((identifier, name, "C1", identifier, 1.5, 1.5, 1.5))
            mask_rows.append(combination)
            selected_rows.append(True)
        for x in (0.2, 0.6):
            identifier = len(atoms) + 1
            atoms.append((identifier, name, "C1", identifier, x, 1.5, 1.5))
            mask_rows.append((False, False, False))
            selected_rows.append(False)
    write_gro(guest_path, atoms, (3.0, 3.0, 3.0))
    general, forward, reverse = np.array(mask_rows, dtype=bool).T
    selection = filtering._ClashSelection(
        general | forward | reverse, general, forward, reverse, forward | reverse
    )
    cache = filtering._RingCheckCache((), (), (), (), (), ())

    def classify(**kwargs):
        """Isolate report accounting using known masks for all overlap categories.

        Parameters
        ----------
        **kwargs : object
            Workflow-supplied filtering inputs, including the real crop/plane mask.

        Returns
        -------
        tuple[_ClashSelection, _RingCheckCache]
            Explicit removal masks and empty report geometry.
        """

        np.testing.assert_array_equal(kwargs["selected_residue_mask"], selected_rows)
        return selection, cache

    monkeypatch.setattr(slit_fill_mod, "_identify_clashing_guest_residues", classify)
    report = slit_fill_mod.fill_slit(
        slit_fill_mod.SlitFillConfig(
            guest_path=guest_path,
            slit_path=slit_path,
            output_path=output_path,
            density_sample_count=100,
            density_seed_count=1,
            density_probe_radii_nm=(0.0,),
            random_seed=11,
        )
    )
    assert report.initial_guest_molecules == 10
    assert report.cropped_guest_molecules == 9
    assert report.removed_outside_crop_guest_molecules == 1
    assert report.surface_plane_filtered_guest_molecules == 8
    assert report.removed_by_surface_plane_guest_molecules == 1
    assert report.removed_by_general_cutoff_guest_molecules == 4
    assert report.removed_by_forward_ring_guest_molecules == 4
    assert report.removed_by_reverse_ring_guest_molecules == 4
    assert report.removed_by_any_ring_guest_molecules == 6
    assert report.removed_by_clash_guest_molecules == 7
    assert report.removed_guest_molecules == 9
    assert report.remaining_guest_molecules == 1
    for field in (
        "removed_by_general_only_guest_molecules",
        "removed_by_forward_ring_only_guest_molecules",
        "removed_by_reverse_ring_only_guest_molecules",
        "removed_by_general_and_forward_ring_only_guest_molecules",
        "removed_by_general_and_reverse_ring_only_guest_molecules",
        "removed_by_forward_and_reverse_ring_only_guest_molecules",
        "removed_by_general_and_forward_and_reverse_ring_guest_molecules",
    ):
        assert getattr(report, field) == 1
    assert [summary.residue_name for summary in report.residue_filter_summaries] == [
        "ION",
        "THY",
    ]
    for summary in report.residue_filter_summaries:
        assert summary.initial_residues == 10
        assert summary.cropped_residues == 9
        assert summary.removed_outside_crop_residues == 1
        assert summary.surface_plane_filtered_residues == 8
        assert summary.removed_by_surface_plane_residues == 1
        assert summary.removed_by_general_cutoff_residues == 4
        assert summary.removed_by_forward_ring_residues == 4
        assert summary.removed_by_reverse_ring_residues == 4
        assert summary.removed_by_any_ring_residues == 6
        assert summary.removed_by_any_clash_residues == 7
        assert summary.removed_total_residues == 9
        assert summary.remaining_residues == 1
    output = gro_io_mod._load_gro_system(output_path)
    assert report.final_atom_count == output.atom_count == 8
    assert report.final_residue_count == len(output.residue_spans) == 4
    assert output.residue_names[-2:] == ["THY", "ION"]


@pytest.mark.parametrize("existing_outputs", (False, True))
@pytest.mark.parametrize("failure_stage", ("gro", "metadata", "counts"))
def test_fill_slit_late_failure_preserves_output_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_outputs: bool,
    failure_stage: str,
    write_basic_slit,
    write_guest_box,
) -> None:
    """Keep old outputs or expose no new set when staged writing/count checks fail."""

    guest_path = tmp_path / "guest_transaction.gro"
    slit_path = tmp_path / "slit_transaction.gro"
    output_path = tmp_path / "merged_transaction.gro"
    metadata_path = output_path.with_suffix(".yml")
    log_path = output_path.with_suffix(".log")
    write_guest_box(guest_path, include_outside_ring=False)
    write_basic_slit(slit_path)
    old_contents = {
        output_path: "old gro",
        metadata_path: "old metadata",
        log_path: "old log",
    }
    if existing_outputs:
        for path, content in old_contents.items():
            path.write_text(content, encoding="utf-8")

    def fail_metadata_write(*args, **kwargs):
        """Inject a failure after the merged GRO has been staged."""

        raise RuntimeError("injected metadata failure")

    def fail_gro_write(*args, **kwargs):
        """Write a partial staged GRO before raising a filesystem failure."""

        staged_path = kwargs["output_path"]
        assert staged_path != output_path
        assert staged_path.parent == output_path.parent
        staged_path.write_text("partial staged gro", encoding="utf-8")
        raise OSError("injected GRO failure")

    real_gro_writer = slit_fill_mod._write_merged_gro

    def write_inconsistent_counts(*args, **kwargs):
        """Exercise the workflow's independent staged atom-count validation."""

        atoms, residues = real_gro_writer(*args, **kwargs)
        return atoms + 1, residues

    if failure_stage == "metadata":
        monkeypatch.setattr(
            slit_fill_mod, "_write_slit_geometry_metadata", fail_metadata_write
        )
        expected_error, message = RuntimeError, "injected metadata failure"
    elif failure_stage == "gro":
        monkeypatch.setattr(slit_fill_mod, "_write_merged_gro", fail_gro_write)
        expected_error, message = OSError, "injected GRO failure"
    else:
        monkeypatch.setattr(
            slit_fill_mod, "_write_merged_gro", write_inconsistent_counts
        )
        expected_error, message = RuntimeError, "Staged GRO counts diverged"

    with pytest.raises(expected_error, match=message):
        slit_fill_mod.fill_slit(
            slit_fill_mod.SlitFillConfig(
                guest_path=guest_path,
                slit_path=slit_path,
                output_path=output_path,
                log_path=log_path,
                density_probe_radii_nm=(0.0,),
                density_sample_count=100,
                density_seed_count=1,
                random_seed=5,
            )
        )

    for path, content in old_contents.items():
        if existing_outputs:
            assert path.read_text(encoding="utf-8") == content
        else:
            assert not path.exists()
    assert not list(tmp_path.glob("*.silicams-stage-*"))
    assert not list(tmp_path.glob("*.silicams-backup-*"))


def test_density_analysis_is_reproducible_and_cli_helpers_accept_argv(
    module_workspace, capsys, write_basic_slit, write_guest_box
) -> None:
    """Density analysis CLI helpers should be reproducible and stay silent."""

    guest_path = module_workspace.root / "guest_cli.gro"
    slit_path = module_workspace.root / "slit_cli.gro"
    merged_path = module_workspace.root / "merged_cli.gro"
    fill_log_path = module_workspace.root / "merged_cli.log"
    density_log_path = module_workspace.root / "merged_cli_density.log"
    write_guest_box(guest_path)
    write_basic_slit(slit_path)

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

    config = density_mod.SlitDensityConfig(
        input_path=merged_path,
        log_path=density_log_path,
        framework_resnames=("SUR",),
        density_probe_radii_nm=(0.0,),
        density_sample_count=600,
        density_seed_count=2,
        random_seed=11,
    )
    report_a = density_mod.estimate_guest_density(config)
    report_b = density_mod.estimate_guest_density(config)
    cli_report = density_mod.estimate_guest_density_main(
        [
            "--input",
            str(merged_path),
            "--log",
            str(density_log_path),
            "--framework-resname",
            "SUR",
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
    assert (
        probe_a.probe_free_volumes_nm3
        == probe_b.probe_free_volumes_nm3
        == probe_cli.probe_free_volumes_nm3
    )
    density_log_text = density_log_path.read_text(encoding="utf-8")

    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == ""
    assert "Slit Density Report" in density_log_text
    assert "Probe 0.00 nm" in density_log_text
    assert density_log_text.index("Probe details") < density_log_text.index(
        "Density summary"
    )


def test_fill_slit_raises_when_target_residue_missing(
    module_workspace, write_basic_slit, write_guest_box
) -> None:
    """The fill workflow should fail early when the target residue is absent."""

    guest_path = module_workspace.root / "guest_missing_target.gro"
    slit_path = module_workspace.root / "slit_missing_target.gro"
    write_guest_box(guest_path, residue_name="SOL")
    write_basic_slit(slit_path)

    with pytest.raises(ValueError, match="Residue name 'THY' was not found"):
        slit_fill_mod.fill_slit(
            slit_fill_mod.SlitFillConfig(
                guest_path=guest_path,
                slit_path=slit_path,
                output_path=module_workspace.root / "unused.gro",
            )
        )


def test_fill_slit_rejects_residue_names_shared_by_both_sources(
    module_workspace, write_basic_slit, write_guest_box
) -> None:
    """Reject component names that become ambiguous in the merged GRO."""

    guest_path = module_workspace.root / "guest_ambiguous.gro"
    slit_path = module_workspace.root / "slit_ambiguous.gro"
    write_guest_box(guest_path, residue_name="SUR")
    write_basic_slit(slit_path)

    with pytest.raises(ValueError, match="both the slit and guest inputs.*SUR"):
        slit_fill_mod.fill_slit(
            slit_fill_mod.SlitFillConfig(
                guest_path=guest_path,
                slit_path=slit_path,
                output_path=module_workspace.root / "unused_ambiguous.gro",
                target_resname="SUR",
            )
        )


def test_fill_slit_raises_when_slit_box_is_larger_than_guest_box(
    module_workspace, write_gro, ring_residue, write_basic_slit, write_guest_box
) -> None:
    """The slit box must fit inside the guest reservoir box for center cropping."""

    guest_path = module_workspace.root / "guest_small_box.gro"
    slit_path = module_workspace.root / "slit_large_box.gro"
    write_guest_box(guest_path, include_outside_ring=False)
    write_gro(
        guest_path,
        ring_residue(1, "THY", 1, (0.5, 0.5, 0.5)),
        (1.0, 1.0, 1.0),
        title="small-guest",
    )
    write_basic_slit(slit_path)

    with pytest.raises(ValueError, match="slit box is larger than the guest box"):
        slit_fill_mod.fill_slit(
            slit_fill_mod.SlitFillConfig(
                guest_path=guest_path,
                slit_path=slit_path,
                output_path=module_workspace.root / "unused_large_box.gro",
            )
        )


def test_fill_slit_raises_for_non_orthorhombic_box(
    module_workspace, write_gro, surface_residue, write_guest_box
) -> None:
    """A non-orthorhombic GRO box should be rejected."""

    guest_path = module_workspace.root / "guest_ortho.gro"
    slit_path = module_workspace.root / "slit_non_ortho.gro"
    write_guest_box(guest_path, include_outside_ring=False)
    write_gro(
        slit_path,
        surface_residue(1, 1, 0.2, 0.5, 0.5) + surface_residue(2, 4, 1.8, 1.5, 1.5),
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


def test_fill_slit_allows_non_aromatic_target_residues(
    module_workspace, write_basic_slit, write_guest_box
) -> None:
    """Skip reverse ring checks when a target has no complete aromatic ring."""

    guest_path = module_workspace.root / "guest_missing_ring.gro"
    slit_path = module_workspace.root / "slit_missing_ring.gro"
    write_guest_box(guest_path, include_outside_ring=False, ring_atom_count=5)
    write_basic_slit(slit_path)

    report = slit_fill_mod.fill_slit(
        slit_fill_mod.SlitFillConfig(
            guest_path=guest_path,
            slit_path=slit_path,
            output_path=module_workspace.root / "merged_missing_ring.gro",
            density_sample_count=100,
            density_seed_count=1,
            density_probe_radii_nm=(0.0,),
            random_seed=4,
        )
    )

    assert report.cropped_guest_ring_count == 0
    assert report.remaining_guest_molecules == 1


def test_geometry_input_validation_rejects_conflicts_legacy_schema_and_box_mismatch(
    module_workspace,
) -> None:
    """Geometry inputs should be unambiguous, versioned, and box-consistent."""

    geometry = PeriodicSlitGeometry(
        box_lengths_nm=(2.0, 2.0, 2.0),
        normal_axis_index=0,
        lower_plane_nm=0.2,
        upper_plane_nm=1.8,
    )
    geometry_path = module_workspace.root / "geometry.yml"
    geometry_path.write_text(
        yaml.safe_dump({"shape_00": {"system": {"surface": 1.0}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not both"):
        density_mod.SlitDensityConfig(
            slit_geometry=geometry,
            slit_geometry_path=geometry_path,
        )
    with pytest.raises(ValueError, match="schema_version: 1"):
        geometry_io._load_slit_geometry(geometry_path)

    mismatched_geometry = PeriodicSlitGeometry(
        box_lengths_nm=(3.0, 2.0, 2.0),
        normal_axis_index=0,
        lower_plane_nm=0.2,
        upper_plane_nm=1.8,
    )
    with pytest.raises(ValueError, match="do not match"):
        geometry_io._validate_geometry_box(
            mismatched_geometry,
            np.array([2.0, 2.0, 2.0]),
        )
