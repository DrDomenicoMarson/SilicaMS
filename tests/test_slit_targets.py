"""Unit tests for silicon-state target models and pure realization math."""

from dataclasses import asdict
from itertools import product
import pickle

import numpy as np
import pytest

import silicams as sms
import silicams.slit_targets as target_mod


def _experimental_target_from_surface(
    surface_target: sms.SiliconStateFractions,
    surface_silicon_fraction: float,
) -> sms.ExperimentalSiliconStateTarget:
    """Build an all-silicon target that maps to given surface fractions.

    Parameters
    ----------
    surface_target : SiliconStateFractions
        Desired surface-only silicon-state fractions.
    surface_silicon_fraction : float
        Physical surface-to-total silicon fraction used for conversion.

    Returns
    -------
    target : ExperimentalSiliconStateTarget
        Experimental target that maps back to ``surface_target``.
    """
    alpha = surface_silicon_fraction
    return sms.ExperimentalSiliconStateTarget(
        q2_fraction=alpha * surface_target.q2_fraction,
        q3_fraction=alpha * surface_target.q3_fraction,
        q4_fraction=alpha * surface_target.q4_fraction + (1.0 - alpha),
        t2_fraction=alpha * surface_target.t2_fraction,
        t3_fraction=alpha * surface_target.t3_fraction,
        surface_silicon_fraction=surface_silicon_fraction,
    )


def test_target_types_are_exported_from_package_root():
    """Keep the supported package-root target API bound to the new module."""
    assert sms.SiliconStateFractions is target_mod.SiliconStateFractions
    assert sms.ExperimentalSiliconStateTarget is target_mod.ExperimentalSiliconStateTarget
    assert sms.SiliconStateComposition is target_mod.SiliconStateComposition
    assert target_mod.__all__ == [
        "ExperimentalSiliconStateTarget",
        "SiliconStateComposition",
        "SiliconStateFractions",
    ]
    assert not any(name.startswith("_") for name in target_mod.__all__)


@pytest.mark.parametrize(
    "value",
    (
        sms.SiliconStateFractions(0.25, 0.25, 0.5),
        sms.ExperimentalSiliconStateTarget(0.125, 0.125, 0.5),
        sms.SiliconStateComposition(8, 2, 2, 4),
    ),
)
def test_target_dataclass_value_and_serialization_contracts(value):
    """Preserve class identity, equality, hashes, and serialized field values."""
    restored = pickle.loads(pickle.dumps(value))

    assert type(restored) is type(value)
    assert restored == value
    assert hash(restored) == hash(value)
    assert asdict(restored) == asdict(value)


@pytest.mark.parametrize(
    ("fractions", "match"),
    (
        ((np.nan, 0.0, 1.0), "finite"),
        ((np.inf, 0.0, 0.0), "finite"),
        ((-0.1, 0.1, 1.0), "non-negative"),
        ((0.2, 0.2, 0.2), "add up"),
    ),
)
def test_surface_fractions_reject_invalid_payloads(fractions, match):
    """Reject non-finite, negative, or non-normalized surface fractions."""
    with pytest.raises(ValueError, match=match):
        sms.SiliconStateFractions(*fractions)


def test_surface_fractions_preserve_five_state_values():
    """Store and compare all five normalized state fractions exactly."""
    fractions = sms.SiliconStateFractions(0.1, 0.2, 0.3, 0.15, 0.25)

    assert asdict(fractions) == {
        "q2_fraction": 0.1,
        "q3_fraction": 0.2,
        "q4_fraction": 0.3,
        "t2_fraction": 0.15,
        "t3_fraction": 0.25,
    }
    assert hash(fractions) == hash(
        sms.SiliconStateFractions(0.1, 0.2, 0.3, 0.15, 0.25)
    )


def test_physical_surface_fraction_uses_unified_conversion():
    """Recover a surface target from its physically scaled all-Si target."""
    reference_surface = sms.SiliconStateFractions(0.069, 0.681, 0.25)
    experimental_target = _experimental_target_from_surface(reference_surface, 0.2)

    converted = target_mod._surface_target_from_experimental(experimental_target)

    assert converted.q2_fraction == pytest.approx(reference_surface.q2_fraction)
    assert converted.q3_fraction == pytest.approx(reference_surface.q3_fraction)
    assert converted.q4_fraction == pytest.approx(reference_surface.q4_fraction)


def test_conversion_preserves_all_five_surface_states():
    """Scale both organosilicon states with the same physical surface fraction."""
    reference = sms.SiliconStateFractions(0.125, 0.25, 0.375, 0.125, 0.125)
    experimental = _experimental_target_from_surface(reference, 0.5)

    assert target_mod._surface_target_from_experimental(experimental) == reference


def test_surface_conversion_example_matches_expected_q2_enrichment():
    """Convert an all-Si target whose modeled surface is entirely Q2."""
    target = sms.ExperimentalSiliconStateTarget(
        q2_fraction=0.5,
        q3_fraction=0.0,
        surface_silicon_fraction=0.5,
    )

    assert target_mod._surface_target_from_experimental(target) == (
        sms.SiliconStateFractions(1.0, 0.0, 0.0)
    )


def test_surface_silicon_fraction_is_mandatory():
    """Require the physical all-Si to modeled-surface mapping factor."""
    with pytest.raises(TypeError, match="surface_silicon_fraction"):
        sms.ExperimentalSiliconStateTarget(
            q2_fraction=0.02,
            q3_fraction=0.03,
        )


@pytest.mark.parametrize("surface_silicon_fraction", (0.0, -0.1, 1.1, np.nan))
def test_surface_silicon_fraction_must_be_finite_and_physical(
    surface_silicon_fraction,
):
    """Reject non-finite or out-of-domain physical surface fractions."""
    with pytest.raises(ValueError, match="surface-silicon fraction"):
        sms.ExperimentalSiliconStateTarget(
            q2_fraction=0.02,
            q3_fraction=0.03,
            surface_silicon_fraction=surface_silicon_fraction,
        )


@pytest.mark.parametrize(
    "field_name",
    ("q2_fraction", "q3_fraction", "q4_fraction", "t2_fraction", "t3_fraction"),
)
@pytest.mark.parametrize("value", (np.nan, np.inf))
def test_experimental_fractions_must_be_finite(field_name, value):
    """Reject non-finite inputs in every experimental silicon-state field."""
    parameters = {
        "q2_fraction": 0.125,
        "q3_fraction": 0.125,
        "surface_silicon_fraction": 0.5,
        field_name: value,
    }

    with pytest.raises(ValueError, match="finite"):
        sms.ExperimentalSiliconStateTarget(**parameters)


def test_q4_fraction_is_derived_when_omitted():
    """Store the all-Si Q4 remainder explicitly on a frozen target."""
    target = sms.ExperimentalSiliconStateTarget(
        q2_fraction=0.02,
        q3_fraction=0.03,
        surface_silicon_fraction=0.5,
        t2_fraction=0.04,
        t3_fraction=0.01,
    )

    assert target.q4_fraction == pytest.approx(0.9, abs=1e-12)
    assert asdict(target)["q4_fraction"] == pytest.approx(0.9, abs=1e-12)


def test_explicit_q4_fraction_must_match_remainder():
    """Reject an explicit Q4 fraction inconsistent with the other states."""
    with pytest.raises(ValueError, match="q4 fraction"):
        sms.ExperimentalSiliconStateTarget(
            q2_fraction=0.02,
            q3_fraction=0.03,
            surface_silicon_fraction=1.0,
            q4_fraction=0.89,
            t2_fraction=0.04,
            t3_fraction=0.01,
        )


def test_incompatible_surface_fraction_and_non_q4_target_raise():
    """Reject all-Si targets that imply a negative modeled-surface Q4 state."""
    with pytest.raises(
        ValueError,
        match=r"Minimum required fraction is 0\.500000, observed 0\.400000",
    ):
        target_mod._surface_target_from_experimental(
            sms.ExperimentalSiliconStateTarget(
                q2_fraction=0.5,
                q3_fraction=0.0,
                surface_silicon_fraction=0.4,
            )
        )


@pytest.mark.parametrize("value", (1.5, True))
def test_silicon_state_composition_requires_integer_counts(value):
    """Reject non-integer silicon-state population counts."""
    with pytest.raises(TypeError, match="integer"):
        sms.SiliconStateComposition(value, value, 0, 0)


def test_silicon_state_composition_normalizes_counts_and_reports_fractions():
    """Normalize integral scalars and expose all five population fractions."""
    composition = sms.SiliconStateComposition(
        np.int64(10),
        np.int64(1),
        np.int64(2),
        np.int64(3),
        np.int64(2),
        np.int64(2),
    )

    assert all(type(value) is int for value in asdict(composition).values())
    assert (
        composition.q2_fraction,
        composition.q3_fraction,
        composition.q4_fraction,
        composition.t2_fraction,
        composition.t3_fraction,
    ) == pytest.approx((0.1, 0.2, 0.3, 0.2, 0.2))


@pytest.mark.parametrize(
    "composition",
    (
        (1, -1, 1, 1),
        (4, 1, 1, 1),
    ),
)
def test_silicon_state_composition_rejects_invalid_totals(composition):
    """Reject negative populations and counts that do not sum to the total."""
    with pytest.raises(ValueError):
        sms.SiliconStateComposition(*composition)


def test_zero_silicon_composition_has_zero_fractions():
    """Avoid division by zero for an empty validated composition."""
    composition = sms.SiliconStateComposition(0, 0, 0, 0)

    assert (
        composition.q2_fraction,
        composition.q3_fraction,
        composition.q4_fraction,
        composition.t2_fraction,
        composition.t3_fraction,
    ) == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_surface_fraction_errors_follow_q2_q3_q4_t2_t3_order():
    """Return independent absolute deviations in the documented state order."""
    composition = sms.SiliconStateComposition(8, 1, 1, 4, 1, 1)
    target = sms.SiliconStateFractions(0.25, 0.375, 0.0, 0.0, 0.375)

    assert target_mod._surface_fraction_errors(composition, target) == (
        0.125, 0.25, 0.5, 0.125, 0.25
    )


def test_nearest_integer_composition_conserves_total_and_tie_order():
    """Preserve largest-remainder allocation and deterministic key ties."""
    target = sms.SiliconStateFractions(0.2, 0.2, 0.2, 0.2, 0.2)

    composition = target_mod._nearest_integer_composition(3, target)

    assert composition == sms.SiliconStateComposition(3, 0, 0, 1, 1, 1)


def test_surface_target_candidates_are_unique_bounded_and_ordered():
    """Keep deterministic tolerance candidates without repeating the exact target."""
    target = sms.SiliconStateFractions(0.25, 0.35, 0.2, 0.1, 0.1)
    exact = target_mod._nearest_integer_composition(10, target)

    candidates = target_mod._surface_target_candidates(10, target, exact, 0.1)
    repeated = target_mod._surface_target_candidates(10, target, exact, 0.1)
    compositions = [candidate.composition for candidate in candidates]

    assert candidates == repeated
    assert exact not in compositions
    assert len(compositions) == len(set(compositions))
    assert candidates[0].composition == sms.SiliconStateComposition(10, 3, 3, 2, 1, 1)
    assert [candidate.total_fraction_error for candidate in candidates] == sorted(
        candidate.total_fraction_error for candidate in candidates
    )
    assert all(
        all(error <= 0.1 for error in target_mod._surface_fraction_errors(item, target))
        for item in compositions
    )


def test_tolerance_candidates_match_exhaustive_integer_reference():
    """Include exactly all integer compositions at the tolerance boundary."""
    total = 8
    fractions = (0.25, 0.25, 0.25, 0.125, 0.125)
    target = sms.SiliconStateFractions(*fractions)
    exact = target_mod._nearest_integer_composition(total, target)
    exact_counts = (2, 2, 2, 1, 1)
    expected = []
    for first_four in product(range(total + 1), repeat=4):
        counts = (*first_four, total - sum(first_four))
        if min(counts) < 0 or counts == exact_counts:
            continue
        errors = tuple(
            abs(count / total - fraction)
            for count, fraction in zip(counts, fractions, strict=True)
        )
        if max(errors) <= 0.125:
            expected.append((sum(errors), counts))
    expected.sort(key=lambda row: (row[0], row[1][2], row[1][1], row[1][0], row[1][4], row[1][3]))

    candidates = target_mod._surface_target_candidates(total, target, exact, 0.125)
    actual = [
        (
            candidate.total_fraction_error,
            (
                candidate.composition.q2_sites,
                candidate.composition.q3_sites,
                candidate.composition.q4_sites,
                candidate.composition.t2_sites,
                candidate.composition.t3_sites,
            ),
        )
        for candidate in candidates
    ]

    assert actual == expected
    assert expected
    assert target_mod._surface_target_candidates(total, target, exact, 0.0) == []


def test_prepared_target_restores_q_states_consumed_by_attachment():
    """Map final T2/T3 populations back to the required bare Q2/Q3 counts."""
    final_surface = sms.SiliconStateComposition(10, 2, 2, 2, 2, 2)

    prepared = target_mod._prepared_target_from_final(final_surface)

    assert prepared == sms.SiliconStateComposition(10, 4, 4, 2)


@pytest.mark.parametrize(
    ("prepared", "expected"),
    (
        (sms.SiliconStateComposition(10, 4, 4, 2), True),
        (sms.SiliconStateComposition(10, 3, 4, 2, 1), False),
        (sms.SiliconStateComposition(10, 3, 4, 2, 0, 1), False),
        (sms.SiliconStateComposition(10, 5, 2, 3), False),
        (sms.SiliconStateComposition(10, 3, 6, 1), False),
        (sms.SiliconStateComposition(10, 3, 5, 2), False),
    ),
)
def test_prepared_target_compatibility_rules(prepared, expected):
    """Enforce Q2 reachability, Q4 direction, Q3 parity, and no T states."""
    initial = sms.SiliconStateComposition(10, 4, 4, 2)

    assert target_mod._prepared_target_is_compatible(initial, prepared) is expected
