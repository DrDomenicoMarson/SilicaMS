"""Density workflow and probe-free-volume regression tests."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, replace
import inspect
from pathlib import Path
import pickle
import warnings

import numpy as np
import pytest
import yaml

import silicams as sms
import silicams._gro_io as gro_io
import silicams._slit_geometry_io as geometry_io
import silicams._slit_report as report_sections
import silicams.database as db
import silicams.slit_density as density_mod
import silicams.slit_fill as fill_mod
from silicams import PeriodicSlitGeometry


@dataclass(frozen=True)
class DensityCase:
    """Small density inputs with independently specified framework membership.

    Parameters
    ----------
    input_path : Path
        Merged GRO input for public workflow tests.
    merged : _GroSystem
        Two target residues and three non-target residues in file order.
    framework : _GroSystem
        Independently selected non-target atom records.
    geometry : PeriodicSlitGeometry
        Explicit geometry for the non-cubic input box.
    """

    input_path: Path
    merged: gro_io._GroSystem
    framework: gro_io._GroSystem
    geometry: PeriodicSlitGeometry


@pytest.fixture
def density_case(tmp_path, write_gro) -> DensityCase:
    """Build a tiny real GRO system without using the density extraction helper."""

    atoms = [
        (1, "SIL", "SI1", 1, 0.15, 0.2, 0.3),
        (2, "ADS", "O1", 2, 1.90, 0.25, 0.25),
        (3, "SOL", "C1", 3, 0.6, 1.0, 1.0),
        (3, "SOL", "H1", 4, 0.6, 1.0, 1.1),
        (4, "ION", "CL", 5, 1.0, 1.5, 2.0),
        (3, "SOL", "O1", 6, 1.4, 2.0, 3.0),
        (3, "SOL", "H2", 7, 1.4, 2.0, 3.1),
    ]
    velocities = np.arange(21.0).reshape(7, 3) / 10.0
    merged_path = tmp_path / "merged.gro"
    framework_path = tmp_path / "framework.gro"
    write_gro(merged_path, atoms, (2.0, 3.0, 4.0), velocities=velocities)
    indices = [0, 1, 4]
    write_gro(
        framework_path,
        [atoms[index] for index in indices],
        (2.0, 3.0, 4.0),
        velocities=velocities[indices],
    )
    return DensityCase(
        merged_path,
        gro_io._load_gro_system(merged_path),
        gro_io._load_gro_system(framework_path),
        PeriodicSlitGeometry((2.0, 3.0, 4.0), 0, 0.2, 1.8),
    )


def test_density_public_exports_and_dataclass_serialization(density_case) -> None:
    """Keep package-root identities and value/serialization contracts intact."""

    config = density_mod.SlitDensityConfig(
        input_path=density_case.input_path,
        slit_geometry=density_case.geometry,
        target_resname="SOL",
        framework_resnames=("SIL", "ADS", "ION"),
        density_sample_count=100,
        density_seed_count=2,
        density_probe_radii_nm=(0.0, 0.14),
        random_seed=11,
    )
    report = sms.estimate_guest_density(config)
    assert density_mod.__all__ == [
        "DEFAULT_DENSITY_PROBE_RADII_NM",
        "DEFAULT_FRAMEWORK_RESNAMES",
        "DensityProbeEstimate",
        "DensityEstimate",
        "SlitDensityConfig",
        "SlitDensityReport",
        "estimate_guest_density",
    ]
    for name in density_mod.__all__[2:]:
        assert getattr(sms, name) is getattr(density_mod, name)
        assert name not in fill_mod.__all__
    assert not hasattr(fill_mod, "SlitDensityConfig")
    assert not hasattr(fill_mod, "estimate_guest_density")
    assert not hasattr(fill_mod, "_estimate_guest_density_console_main")
    for value in (
        config,
        report,
        report.density_estimate,
        report.density_estimate.probe_estimates[0],
    ):
        restored = pickle.loads(pickle.dumps(value))
        assert type(restored) is type(value)
        assert restored == value
        assert hash(restored) == hash(value)
        assert asdict(restored) == asdict(value)


@pytest.mark.parametrize(
    "field,value,message",
    (
        ("density_probe_radii_nm", (), "At least one"),
        ("density_probe_radii_nm", (-0.1,), "non-negative"),
        ("density_probe_radii_nm", (np.nan,), "finite"),
        ("surface_plane_padding_nm", np.inf, "finite"),
        ("density_sample_count", 0, "strictly positive"),
        ("density_seed_count", 0, "strictly positive"),
        ("random_seed", -1, "non-negative"),
    ),
)
def test_density_config_preserves_validation(field, value, message) -> None:
    """Retain finite-value, count, and radius validation after relocation."""

    with pytest.raises(ValueError, match=message):
        density_mod.SlitDensityConfig(**{field: value})


def test_density_defaults_and_numpy_integer_normalization() -> None:
    """Keep documented defaults and normalized native integer fields."""

    default = density_mod.SlitDensityConfig()
    assert default.input_path == Path("merged_guest_slit_ring_check.gro")
    assert default.log_path is default.random_seed is None
    assert default.density_probe_radii_nm == (0.0, 0.14, 0.2)
    assert default.framework_resnames == ("OM", "SI", "SL", "SLG")
    assert default.mobile_resnames == ()
    assert default.density_sample_count == 200000
    assert default.density_seed_count == 5
    assert default.surface_plane_padding_nm == 0.0
    config = replace(
        default,
        density_sample_count=np.int64(100),
        density_seed_count=np.int64(2),
        random_seed=np.int64(0),
    )
    for name in ("density_sample_count", "density_seed_count", "random_seed"):
        assert type(getattr(config, name)) is int


@pytest.mark.parametrize(
    "field,value,error",
    (
        ("framework_resnames", ("SI", "SI"), ValueError),
        ("framework_resnames", (" SI",), ValueError),
        ("framework_resnames", "SI", TypeError),
        ("mobile_resnames", (1,), TypeError),
    ),
)
def test_density_component_selectors_are_exact_and_unique(
    field, value, error
) -> None:
    """Reject malformed residue selectors before reading an input file."""

    with pytest.raises(error):
        density_mod.SlitDensityConfig(**{field: value})


def test_density_requires_exhaustive_disjoint_component_selectors(
    density_case,
) -> None:
    """Keep co-guests mobile and fail rather than guessing ambiguous roles."""

    config = density_mod.SlitDensityConfig(
        input_path=density_case.input_path,
        slit_geometry=density_case.geometry,
        target_resname="SOL",
        framework_resnames=("SIL",),
        mobile_resnames=("ADS", "ION"),
        density_probe_radii_nm=(0.0,),
        density_sample_count=100,
        density_seed_count=1,
        random_seed=7,
    )
    report = density_mod.estimate_guest_density(config)
    assert report.framework_atom_count == report.framework_residue_count == 1
    assert report.framework_resnames == ("SIL",)
    assert report.mobile_resnames == ("ADS", "ION", "SOL")

    with pytest.raises(ValueError, match=r"ION \(1 residues\)"):
        density_mod.estimate_guest_density(
            replace(config, mobile_resnames=("ADS",))
        )
    with pytest.raises(ValueError, match="both framework and mobile: SOL"):
        density_mod.estimate_guest_density(
            replace(config, framework_resnames=("SIL", "SOL"))
        )


@pytest.mark.parametrize(
    "field_name", ("density_sample_count", "density_seed_count", "random_seed")
)
@pytest.mark.parametrize("value", (1.5, True))
def test_density_config_requires_actual_integer_fields(field_name, value) -> None:
    """Reject floats and booleans for integer counts and seeds."""

    with pytest.raises(TypeError, match="integer"):
        density_mod.SlitDensityConfig(**{field_name: value})


@pytest.mark.parametrize("with_velocities", (False, True))
def test_framework_and_target_selection_preserve_order_and_independence(
    density_case,
    with_velocities,
) -> None:
    """Copy only selected framework atoms and count repeated target ids."""

    source = density_case.merged
    if not with_velocities:
        source = replace(source, velocities=None)
    framework = density_mod._build_framework_system(
        source, ("SIL", "ADS", "ION")
    )
    assert density_mod._count_target_molecules(source, "SOL") == (2, 4)
    assert density_mod._count_target_molecules(source, "MISSING") == (0, 0)
    assert framework.residue_names == ["SIL", "ADS", "ION"]
    assert framework.atom_names == ["SI1", "O1", "CL"]
    np.testing.assert_array_equal(framework.atom_ids, [1, 2, 5])
    np.testing.assert_array_equal(framework.residue_ids, [1, 2, 4])
    np.testing.assert_array_equal(framework.atom_to_residue_index, [0, 1, 2])
    assert framework.residue_spans == density_case.framework.residue_spans
    assert framework.title == source.title + " [framework only]"
    for name in ("coordinates", "box_lengths", "atom_ids", "residue_ids"):
        np.testing.assert_array_equal(
            getattr(framework, name), getattr(density_case.framework, name)
        )
        assert not np.shares_memory(getattr(framework, name), getattr(source, name))
    if with_velocities:
        np.testing.assert_array_equal(
            framework.velocities, source.velocities[[0, 1, 4]]
        )
        assert not np.shares_memory(framework.velocities, source.velocities)
    else:
        assert framework.velocities is None
    assert framework.atom_names is not source.atom_names
    assert framework.residue_names is not source.residue_names


def test_representative_mass_and_lookup_failures(density_case) -> None:
    """Use the first target span's mass without adding composition validation."""

    assert density_mod._compute_target_residue_mass_da(density_case.merged, "SOL") == (
        12.0107 + 1.0079
    )
    with pytest.raises(ValueError, match="No residue named 'ABSENT'"):
        density_mod._compute_target_residue_mass_da(density_case.merged, "ABSENT")
    names = density_case.merged.atom_names.copy()
    names[2] = "ZZZ"
    with pytest.raises(
        ValueError, match="Unsupported element inferred from atom name 'ZZZ'"
    ) as caught:
        density_mod._compute_target_residue_mass_da(
            replace(density_case.merged, atom_names=names), "SOL"
        )
    assert isinstance(caught.value.__cause__, ValueError)
    for name, element in (("CA1", "C"), ("SI1", "Si"), ("NA", "Na"), ("CL", "Cl")):
        assert db._infer_element_from_atom_name(name) == element
    unsupported = replace(density_case.framework, atom_names=["NA", "O1", "CL"])
    with pytest.raises(ValueError, match="DB: Atom name not found"):
        density_mod._estimate_probe_free_volume_nm3(
            unsupported,
            unsupported.coordinates,
            density_case.geometry,
            0.0,
            0.0,
            10,
            11,
        )


def test_seed_sequence_matches_recorded_default_generator_values(monkeypatch) -> None:
    """Keep deterministic draw order and the 63-bit entropy-backed seed path."""

    assert density_mod._seed_values(3, 11) == (
        1185850812994184743,
        4605025475050782003,
        5547843131917349438,
    )
    entropy = iter((0, 2**63 - 1, 17))
    bits_requested = []

    def draw(bits):
        """Supply controlled entropy while checking the requested bit width."""
        bits_requested.append(bits)
        return next(entropy)

    monkeypatch.setattr(density_mod.secrets, "randbits", draw)
    assert density_mod._seed_values(3, None) == (0, 2**63 - 1, 17)
    assert bits_requested == [63, 63, 63]


@pytest.mark.parametrize(
    "values,expected",
    (
        ((), 0.0),
        ((7.0,), 0.0),
        ((np.inf,), 0.0),
        ((np.nan,), 0.0),
        ((1.0, 3.0), np.sqrt(2.0)),
        ((1.0, 2.0, 3.0), 1.0),
        ((1.0, np.inf), np.inf),
        ((np.nan, 1.0), np.inf),
    ),
)
def test_repeated_standard_deviation_contract(values, expected) -> None:
    """Retain sample statistics and the established degenerate-value policies."""

    assert density_mod._compute_repeated_std(values) == pytest.approx(expected)


@pytest.mark.parametrize("random_seed", (11, None))
@pytest.mark.parametrize("guest_count", (0, 2))
def test_density_aggregation_preserves_probe_seed_and_arithmetic_order(
    density_case,
    monkeypatch,
    random_seed,
    guest_count,
) -> None:
    """Average repeat densities individually and advance the seed source per probe."""

    calls = []
    entropy = iter(range(6))
    monkeypatch.setattr(density_mod.secrets, "randbits", lambda bits: next(entropy))
    volume = density_case.geometry.padded_geometric_volume_nm3(0.05)

    def estimate(**kwargs):
        """Isolate repeat aggregation with two exactly known, unequal volumes."""
        calls.append((kwargs["probe_radius_nm"], kwargs["random_seed"]))
        repeat_volume = 2.0 if len(calls) % 2 else 4.0
        return repeat_volume, repeat_volume / volume

    monkeypatch.setattr(density_mod, "_estimate_probe_free_volume_nm3", estimate)
    result = density_mod._compute_density_estimate(
        density_case.merged,
        density_case.framework,
        density_case.framework.coordinates,
        density_case.geometry,
        0.05,
        "SOL",
        guest_count,
        (0.2, 0.0, 0.2),
        100,
        2,
        random_seed,
    )
    seeds = (
        (
            (1185850812994184743, 4605025475050782003),
            (2313447293076694940, 8732254618979584852),
            (7976349881628269150, 7888773299275917823),
        )
        if random_seed is not None
        else ((0, 1), (2, 3), (4, 5))
    )
    assert calls == [
        (radius, seed) for radius, pair in zip((0.2, 0.0, 0.2), seeds) for seed in pair
    ]
    mass = guest_count * (12.0107 + 1.0079)
    factor = mass * 1.66053906660e-24 / 1.0e-21
    assert result.total_guest_mass_da == mass
    assert result.box_volume_nm3 == 24.0
    assert result.box_average_density_g_cm3 == pytest.approx(factor / 24.0)
    assert result.geometric_slit_volume_nm3 == volume
    for probe, pair in zip(result.probe_estimates, seeds):
        assert probe.seed_values == pair
        assert probe.probe_free_volumes_nm3 == (2.0, 4.0)
        assert probe.probe_free_fractions == (2.0 / volume, 4.0 / volume)
        assert probe.probe_free_volume_mean_nm3 == 3.0
        assert probe.probe_free_volume_std_nm3 == pytest.approx(np.sqrt(2.0))
        assert probe.probe_free_density_mean_g_cm3 == pytest.approx(
            (factor / 2 + factor / 4) / 2
        )
        assert probe.probe_free_density_std_g_cm3 == pytest.approx(
            abs(factor / 2 - factor / 4) / np.sqrt(2)
        )


@pytest.mark.parametrize(
    "normal_axis,wraps,padding,sample_count",
    (
        (0, False, 0.05, 257),
        (1, False, -0.05, 257),
        (2, False, 0.0, 257),
        (0, True, -0.05, 257),
        (1, True, 0.05, 257),
        (2, True, 0.0, 257),
        (0, False, 0.05, 10001),
        (2, True, -0.05, 10001),
    ),
)
def test_probe_volume_matches_brute_force_periodic_exclusion(
    density_case,
    normal_axis,
    wraps,
    padding,
    sample_count,
) -> None:
    """Check real sampling and KD-tree results against all atom-point distances."""

    box = np.array([2.0, 3.0, 4.0])
    geometry = PeriodicSlitGeometry(
        tuple(box),
        normal_axis,
        float(box[normal_axis] - 0.4) if wraps else 0.2,
        0.4 if wraps else float(box[normal_axis] - 0.2),
    )
    framework_coordinates = density_case.framework.coordinates + box * [1, -1, 2]
    original = framework_coordinates.copy()
    framework_coordinates.setflags(write=False)
    rng = np.random.default_rng(91)
    free_count = 0
    # Known canonical elements give independent radius assignment. The oracle
    # uses no KD-tree or production minimum-image/exclusion helper.
    radii = np.array([0.210, 0.152, 0.175]) + 0.14
    for start in range(0, sample_count, 10000):
        points = geometry.sample_uniform_positions(
            min(10000, sample_count - start), rng, padding
        )
        delta = points[:, None, :] - framework_coordinates[None, :, :]
        delta -= box * np.rint(delta / box)
        excluded = np.any(np.sum(delta * delta, axis=2) <= radii**2, axis=1)
        free_count += int(np.count_nonzero(~excluded))
    expected_fraction = free_count / sample_count
    result_volume, result_fraction = density_mod._estimate_probe_free_volume_nm3(
        density_case.framework,
        framework_coordinates,
        geometry,
        padding,
        0.14,
        sample_count,
        91,
    )
    assert result_fraction == expected_fraction
    assert result_volume == expected_fraction * geometry.padded_geometric_volume_nm3(
        padding
    )
    np.testing.assert_array_equal(framework_coordinates, original)


def test_probe_exclusion_includes_exact_radius_and_periodic_seam(
    density_case, monkeypatch
) -> None:
    """Exclude points on the radius and periodic images, but not just outside it."""

    framework = replace(density_case.framework, atom_names=["H1", "H2", "H3"])
    coordinates = np.zeros((3, 3))
    # H radius + probe radius is exactly representable as 0.25 nm.
    probe = 0.25 - db.get_vdw_radius("H")
    points = np.array([[0.25, 0, 0], [1.75, 0, 0], [0.26, 0, 0], [1.0, 1.5, 2.0]])

    def sample(self, count, rng, padding):
        """Supply exact test points while exercising the real exclusion search."""
        assert count == len(points)
        return points.copy()

    monkeypatch.setattr(PeriodicSlitGeometry, "sample_uniform_positions", sample)
    volume, fraction = density_mod._estimate_probe_free_volume_nm3(
        framework,
        coordinates,
        density_case.geometry,
        0.0,
        probe,
        4,
        11,
    )
    assert fraction == 0.5
    assert volume == 0.5 * density_case.geometry.geometric_slit_volume_nm3


@pytest.mark.parametrize("seed_count", (1, 2))
@pytest.mark.parametrize("guest_count", (0, 2))
def test_zero_free_volume_retains_warnings_and_infinite_density(
    density_case,
    seed_count,
    guest_count,
) -> None:
    """A fully excluded real sample keeps the established infinity/std policy."""

    with pytest.warns(
        UserWarning, match="reporting infinite probe-free density"
    ) as caught:
        estimate = density_mod._compute_density_estimate(
            density_case.merged,
            density_case.framework,
            density_case.framework.coordinates,
            density_case.geometry,
            0.0,
            "SOL",
            guest_count,
            (10.0,),
            100,
            seed_count,
            11,
        )
    assert len(caught) == seed_count
    probe = estimate.probe_estimates[0]
    assert probe.probe_free_volumes_nm3 == (0.0,) * seed_count
    assert probe.probe_free_fractions == (0.0,) * seed_count
    assert probe.probe_free_densities_g_cm3 == (np.inf,) * seed_count
    assert probe.probe_free_density_mean_g_cm3 == np.inf
    assert probe.probe_free_density_std_g_cm3 == (0.0 if seed_count == 1 else np.inf)
    assert "inf g/cm^3" in density_mod._format_probe_block(probe)


def test_standalone_density_api_cli_defaults_and_report(
    density_case, capsys, monkeypatch
) -> None:
    """Keep the standalone API, argv adapter, console status, and report paths aligned."""

    metadata = density_case.input_path.with_suffix(".yml")
    geometry_io._write_slit_geometry_metadata(metadata, density_case.geometry, 0.05)
    config = density_mod.SlitDensityConfig(
        input_path=density_case.input_path,
        slit_geometry_path=metadata,
        target_resname="SOL",
        framework_resnames=("SIL", "ADS", "ION"),
        density_sample_count=100,
        density_seed_count=2,
        density_probe_radii_nm=(0.0, 0.14),
        surface_plane_padding_nm=0.05,
        random_seed=11,
    )
    report = density_mod.estimate_guest_density(config)
    log_path = density_case.input_path.with_name("merged_density.log")
    report_text = log_path.read_text()
    argv = [
        "--input",
        str(config.input_path),
        "--slit-geometry",
        str(metadata),
        "--target-resname",
        "SOL",
        "--framework-resname",
        "SIL",
        "--framework-resname",
        "ADS",
        "--framework-resname",
        "ION",
        "--density-samples",
        "100",
        "--density-seed-count",
        "2",
        "--density-probe-radius",
        "0",
        "--density-probe-radius",
        "0.14",
        "--surface-plane-padding",
        "0.05",
        "--random-seed",
        "11",
    ]
    assert density_mod.estimate_guest_density_main(argv) == report
    monkeypatch.setattr("sys.argv", ["silicams-slit-density", *argv])
    assert density_mod._estimate_guest_density_console_main() == 0
    assert log_path.read_text() == report_text
    assert report_text.index("Probe details") < report_text.index("Density summary")
    assert "g/cm^3" in report_text and "nm^3" in report_text
    assert report.guest_molecule_count == 2 and report.guest_atom_count == 4
    assert report.framework_atom_count == report.framework_residue_count == 3
    assert report.framework_resnames == ("ADS", "ION", "SIL")
    assert report.mobile_resnames == ("SOL",)
    explicit = replace(config, log_path=log_path)
    assert density_mod._resolve_density_config(explicit) is explicit
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_standalone_density_rejects_missing_target_before_writing(density_case) -> None:
    """Report available residues and leave the report unwritten on invalid input."""

    log_path = density_case.input_path.with_name("invalid_density.log")
    config = density_mod.SlitDensityConfig(
        input_path=density_case.input_path,
        target_resname="ABSENT",
        log_path=log_path,
    )
    with pytest.raises(ValueError, match="Available residue names: ADS, ION, SIL, SOL"):
        density_mod.estimate_guest_density(config)
    assert not log_path.exists()


def test_common_report_sections_preserve_alignment_and_optional_fields(
    density_case,
) -> None:
    """Keep shared section layout, units, and unavailable geometry diagnostics."""

    assert report_sections._format_value_lines("Empty", ()) == "Empty\n-----"
    assert report_sections._format_value_lines(
        "Values", (("A", "1"), ("Long", "2"))
    ) == ("Values\n------\n  A    : 1\n  Long : 2")
    section = report_sections._format_geometry_block(
        "Geometry", density_case.geometry, -0.05
    )
    assert section.count("not available") == 2
    assert "-0.05000 nm" in section
    measured = replace(
        density_case.geometry, surface_support_count=4, normal_roughness_rms_nm=0.012
    )
    measured_section = report_sections._format_geometry_block("Geometry", measured, 0.0)
    assert "not available" not in measured_section
    assert "0.01200 nm" in measured_section


def test_density_and_support_modules_have_one_way_dependencies() -> None:
    """Keep shared support independent of either workflow's classes or code."""

    for module in (density_mod, geometry_io, report_sections):
        tree = ast.parse(inspect.getsource(module))
        imported_modules = [
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ]
        imported_modules += [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        assert not any("slit_fill" in name for name in imported_modules)
        if module is not density_mod:
            assert not any("slit_density" in name for name in imported_modules)
            assert module.__all__ == []


def test_density_analysis_raises_without_framework(
    module_workspace, write_gro, ring_residue
) -> None:
    """Merged systems that contain only target residues should be rejected."""

    merged_path = module_workspace.root / "merged_no_framework.gro"
    write_gro(
        merged_path,
        ring_residue(1, "THY", 1, (1.0, 1.0, 1.0)),
        (2.0, 2.0, 2.0),
        title="merged-no-framework",
    )

    with pytest.raises(ValueError, match="does not contain any residues selected"):
        density_mod.estimate_guest_density(
            density_mod.SlitDensityConfig(input_path=merged_path)
        )


def test_density_analysis_skips_nominal_box_warning(
    module_workspace, write_gro, surface_residue, ring_residue
) -> None:
    """Density analysis should not warn for framework atoms outside the nominal box."""

    merged_path = module_workspace.root / "merged_outside_box.gro"
    atoms: list[tuple[int, str, str, int, float, float, float]] = []
    atoms.extend(surface_residue(1, 1, 2.2, 0.5, 0.5))
    atoms.extend(surface_residue(2, 4, 1.8, 1.5, 1.5))
    atoms.extend(ring_residue(3, "THY", 7, (1.0, 1.0, 1.0)))
    write_gro(merged_path, atoms, (2.0, 2.0, 2.0), title="merged-outside-box")

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        report = density_mod.estimate_guest_density(
            density_mod.SlitDensityConfig(
                input_path=merged_path,
                framework_resnames=("SUR",),
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


def test_density_requires_explicit_geometry_for_nonhydroxylated_framework(
    module_workspace, write_gro, ring_residue
) -> None:
    """Neighboring metadata is ignored unless its path is supplied explicitly."""

    merged_path = module_workspace.root / "functionalized_merged.gro"
    geometry_path = merged_path.with_suffix(".yml")
    atoms = [
        (1, "SIL", "SI1", 1, 0.2, 0.5, 0.5),
        (2, "SIL", "SI1", 2, 1.8, 1.5, 1.5),
        *ring_residue(3, "THY", 3, (1.0, 1.0, 1.0)),
    ]
    write_gro(merged_path, atoms, (2.0, 2.0, 2.0), title="functionalized")
    slit_geometry = PeriodicSlitGeometry(
        box_lengths_nm=(2.0, 2.0, 2.0),
        normal_axis_index=0,
        lower_plane_nm=0.2,
        upper_plane_nm=1.8,
    )
    geometry_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "slit_geometry": slit_geometry.to_dict(),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Could not identify any hydroxylated"):
        density_mod.estimate_guest_density(
            density_mod.SlitDensityConfig(
                input_path=merged_path,
                framework_resnames=("SIL",),
                density_probe_radii_nm=(0.0,),
                density_sample_count=100,
                density_seed_count=1,
                random_seed=4,
            )
        )

    report = density_mod.estimate_guest_density(
        density_mod.SlitDensityConfig(
            input_path=merged_path,
            slit_geometry_path=geometry_path,
            framework_resnames=("SIL",),
            density_probe_radii_nm=(0.0,),
            density_sample_count=300,
            density_seed_count=1,
            random_seed=4,
        )
    )

    probe = report.density_estimate.probe_estimates[0]
    assert report.slit_geometry == slit_geometry
    assert report.density_estimate.geometric_slit_volume_nm3 == pytest.approx(6.4)
    assert 0.0 <= probe.probe_free_fractions[0] <= 1.0
    assert 0.0 <= probe.probe_free_volumes_nm3[0] <= 6.4
