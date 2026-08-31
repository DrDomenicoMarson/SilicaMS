"""Silicon-state targets and pure surface-composition mathematics."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ._validation import integral

__all__ = [
    "ExperimentalSiliconStateTarget",
    "SiliconStateComposition",
    "SiliconStateFractions",
]


@dataclass(frozen=True)
class SiliconStateFractions:
    """Five-state silicon surface fractions.

    Parameters
    ----------
    q2_fraction : float
        Fraction of geminal silanol sites ``Q2`` over the modeled surface
        silicon population.
    q3_fraction : float
        Fraction of single silanol sites ``Q3`` over the modeled surface
        silicon population.
    q4_fraction : float
        Fraction of fully condensed surface silicon sites ``Q4`` over the
        modeled surface silicon population.
    t2_fraction : float, optional
        Fraction of geminally attached organosilicon sites ``T2`` over the
        modeled surface silicon population.
    t3_fraction : float, optional
        Fraction of singly attached organosilicon sites ``T3`` over the
        modeled surface silicon population.
    """

    q2_fraction: float
    q3_fraction: float
    q4_fraction: float
    t2_fraction: float = 0.0
    t3_fraction: float = 0.0

    def __post_init__(self):
        """Validate the fraction payload.

        Raises
        ------
        ValueError
            Raised when any fraction is negative or when the total fraction
            does not add up to one.
        """
        fractions = (
            self.q2_fraction,
            self.q3_fraction,
            self.q4_fraction,
            self.t2_fraction,
            self.t3_fraction,
        )
        if not all(math.isfinite(fraction) for fraction in fractions):
            raise ValueError("Silicon-state fractions must be finite.")
        if min(fractions) < 0:
            raise ValueError("Silicon-state fractions must be non-negative.")
        if abs(sum(fractions) - 1.0) > 1e-6:
            raise ValueError("Silicon-state fractions must add up to 1.0.")


@dataclass(frozen=True)
class ExperimentalSiliconStateTarget:
    """Experimental silicon-state fractions over all Si atoms in the sample.

    Parameters
    ----------
    q2_fraction : float
        Experimental ``Q2`` fraction over all Si atoms in the sample.
    q3_fraction : float
        Experimental ``Q3`` fraction over all Si atoms in the sample.
    q4_fraction : float or None, optional
        Experimental ``Q4`` fraction over all Si atoms in the sample. When
        omitted, the value is derived as
        ``1.0 - (q2_fraction + q3_fraction + t2_fraction + t3_fraction)``.
        When provided explicitly, it must match that derived remainder within
        the class validation tolerance.
    t2_fraction : float, optional
        Experimental ``T2`` fraction over all Si atoms in the sample.
    t3_fraction : float, optional
        Experimental ``T3`` fraction over all Si atoms in the sample.
    surface_silicon_fraction : float
        Physically justified fraction of all silicon atoms represented by the
        modeled surface population. This is the experimental-to-model mapping
        factor commonly denoted ``alpha``. SilicaMS does not estimate it from
        the simulated wall geometry.
    """

    q2_fraction: float
    q3_fraction: float
    surface_silicon_fraction: float
    q4_fraction: float | None = None
    t2_fraction: float = 0.0
    t3_fraction: float = 0.0

    def __post_init__(self):
        """Validate the experimental target.

        Raises
        ------
        ValueError
            Raised when the fractions are invalid, when an explicit
            ``q4_fraction`` does not match the derived remainder, or when the
            physical surface-silicon fraction lies outside ``(0, 1]``.
        """
        input_fractions = (
            self.q2_fraction,
            self.q3_fraction,
            self.t2_fraction,
            self.t3_fraction,
        )
        if not all(math.isfinite(fraction) for fraction in input_fractions):
            raise ValueError("Experimental silicon-state fractions must be finite.")
        if self.q4_fraction is not None and not math.isfinite(self.q4_fraction):
            raise ValueError("The q4 fraction must be finite when supplied.")
        if not math.isfinite(self.surface_silicon_fraction):
            raise ValueError("The surface-silicon fraction must be finite.")

        derived_q4 = 1.0 - (
            self.q2_fraction
            + self.q3_fraction
            + self.t2_fraction
            + self.t3_fraction
        )
        if abs(derived_q4) < 1e-12:
            derived_q4 = 0.0
        if self.q4_fraction is None:
            object.__setattr__(self, "q4_fraction", derived_q4)
        elif abs(self.q4_fraction - derived_q4) > 1e-6:
            raise ValueError(
                "The q4 fraction must match 1.0 - (q2 + q3 + t2 + t3)."
            )
        SiliconStateFractions(
            self.q2_fraction,
            self.q3_fraction,
            self.q4_fraction,
            self.t2_fraction,
            self.t3_fraction,
        )
        if not (0.0 < self.surface_silicon_fraction <= 1.0):
            raise ValueError(
                "The surface-silicon fraction must be in the interval (0, 1]."
            )


@dataclass(frozen=True)
class SiliconStateComposition:
    """Integer five-state silicon surface composition.

    Parameters
    ----------
    total_surface_si : int
        Total number of tracked surface silicon atoms.
    q2_sites : int
        Number of residual ``Q2`` surface silicon sites.
    q3_sites : int
        Number of residual ``Q3`` surface silicon sites.
    q4_sites : int
        Number of residual ``Q4`` surface silicon sites.
    t2_sites : int, optional
        Number of attached ``T2`` sites.
    t3_sites : int, optional
        Number of attached ``T3`` sites.
    """

    total_surface_si: int
    q2_sites: int
    q3_sites: int
    q4_sites: int
    t2_sites: int = 0
    t3_sites: int = 0

    def __post_init__(self):
        """Validate the integer silicon-state counts.

        Raises
        ------
        TypeError
            Raised when any population is not an integer.
        ValueError
            Raised when the counts are invalid or do not add up to the tracked
            surface silicon count.
        """
        counts = tuple(
            integral(name, value)
            for name, value in (
                ("total_surface_si", self.total_surface_si),
                ("q2_sites", self.q2_sites),
                ("q3_sites", self.q3_sites),
                ("q4_sites", self.q4_sites),
                ("t2_sites", self.t2_sites),
                ("t3_sites", self.t3_sites),
            )
        )
        if min(counts) < 0:
            raise ValueError("Silicon-state counts must be non-negative.")
        if (
            self.q2_sites
            + self.q3_sites
            + self.q4_sites
            + self.t2_sites
            + self.t3_sites
            != self.total_surface_si
        ):
            raise ValueError(
                "Silicon-state counts must add up to the total surface silicon count."
            )
        for field_name, value in zip(
            (
                "total_surface_si",
                "q2_sites",
                "q3_sites",
                "q4_sites",
                "t2_sites",
                "t3_sites",
            ),
            counts,
            strict=True,
        ):
            object.__setattr__(self, field_name, value)

    @property
    def q2_fraction(self):
        """Return the residual ``Q2`` fraction."""
        return self.q2_sites / self.total_surface_si if self.total_surface_si else 0.0

    @property
    def q3_fraction(self):
        """Return the residual ``Q3`` fraction."""
        return self.q3_sites / self.total_surface_si if self.total_surface_si else 0.0

    @property
    def q4_fraction(self):
        """Return the residual ``Q4`` fraction."""
        return self.q4_sites / self.total_surface_si if self.total_surface_si else 0.0

    @property
    def t2_fraction(self):
        """Return the ``T2`` fraction."""
        return self.t2_sites / self.total_surface_si if self.total_surface_si else 0.0

    @property
    def t3_fraction(self):
        """Return the ``T3`` fraction."""
        return self.t3_sites / self.total_surface_si if self.total_surface_si else 0.0


@dataclass(frozen=True)
class _SurfaceTargetCandidate:
    """Candidate final surface composition ranked against target fractions.

    Parameters
    ----------
    composition : SiliconStateComposition
        Candidate integer five-state surface composition.
    total_fraction_error : float
        Sum of the absolute fraction deviations from the requested five-state
        target.
    """

    composition: SiliconStateComposition
    total_fraction_error: float


def _surface_fraction_errors(
    composition: SiliconStateComposition,
    target: SiliconStateFractions,
) -> tuple[float, float, float, float, float]:
    """Return absolute fraction deviations from a five-state target.

    Parameters
    ----------
    composition : SiliconStateComposition
        Candidate or realized surface composition.
    target : SiliconStateFractions
        Requested surface-only five-state fractions.

    Returns
    -------
    errors : tuple[float, float, float, float, float]
        Absolute fraction deviations for ``Q2``, ``Q3``, ``Q4``, ``T2``, and
        ``T3``.
    """
    return (
        abs(composition.q2_fraction - target.q2_fraction),
        abs(composition.q3_fraction - target.q3_fraction),
        abs(composition.q4_fraction - target.q4_fraction),
        abs(composition.t2_fraction - target.t2_fraction),
        abs(composition.t3_fraction - target.t3_fraction),
    )


def _surface_target_from_experimental(
    target: ExperimentalSiliconStateTarget,
) -> SiliconStateFractions:
    """Convert experimental all-silicon ratios into surface-only fractions.

    Parameters
    ----------
    target : ExperimentalSiliconStateTarget
        Experimental all-silicon ratios supplied by the caller.

    Returns
    -------
    surface_target : SiliconStateFractions
        Modeled surface-only fractions.

    Raises
    ------
    ValueError
        Raised when the experimental ratios and physical surface-silicon
        fraction are incompatible with a surface-only interpretation. In
        particular, the physical fraction must be at least the experimental
        non-``Q4`` fraction so the converted surface-only ``Q4`` fraction
        remains non-negative.
    """
    surface_silicon_fraction = target.surface_silicon_fraction
    non_q4_fraction = (
        target.q2_fraction
        + target.q3_fraction
        + target.t2_fraction
        + target.t3_fraction
    )
    if surface_silicon_fraction + 1e-9 < non_q4_fraction:
        raise ValueError(
            "The physical surface-silicon fraction is too small for the "
            "requested experimental non-Q4 silicon-state fractions. Minimum "
            f"required fraction is {non_q4_fraction:.6f}, observed "
            f"{surface_silicon_fraction:.6f}."
        )

    surface_target = SiliconStateFractions(
        q2_fraction=target.q2_fraction / surface_silicon_fraction,
        q3_fraction=target.q3_fraction / surface_silicon_fraction,
        q4_fraction=(
            target.q4_fraction - (1.0 - surface_silicon_fraction)
        ) / surface_silicon_fraction,
        t2_fraction=target.t2_fraction / surface_silicon_fraction,
        t3_fraction=target.t3_fraction / surface_silicon_fraction,
    )

    return surface_target


def _nearest_integer_composition(
    total_surface_si: int,
    target: SiliconStateFractions,
) -> SiliconStateComposition:
    """Convert target fractions into one nearest integer surface composition.

    Parameters
    ----------
    total_surface_si : int
        Number of tracked surface silicon atoms.
    target : SiliconStateFractions
        Requested surface-only target fractions.

    Returns
    -------
    composition : SiliconStateComposition
        Nearest integer surface composition whose counts add up to
        ``total_surface_si``.
    """
    ideals = {
        "q2_sites": total_surface_si * target.q2_fraction,
        "q3_sites": total_surface_si * target.q3_fraction,
        "q4_sites": total_surface_si * target.q4_fraction,
        "t2_sites": total_surface_si * target.t2_fraction,
        "t3_sites": total_surface_si * target.t3_fraction,
    }
    counts = {key: math.floor(value) for key, value in ideals.items()}
    missing = total_surface_si - sum(counts.values())
    ranked = sorted(
        ideals,
        key=lambda key: (ideals[key] - counts[key], ideals[key], key),
        reverse=True,
    )
    for key in ranked[:missing]:
        counts[key] += 1

    return SiliconStateComposition(
        total_surface_si=total_surface_si,
        **counts,
    )


def _surface_target_candidates(
    total_surface_si: int,
    target: SiliconStateFractions,
    exact_target: SiliconStateComposition,
    tolerance: float,
) -> list[_SurfaceTargetCandidate]:
    """Enumerate nearby five-state targets inside the allowed tolerance.

    Parameters
    ----------
    total_surface_si : int
        Number of tracked surface silicon atoms.
    target : SiliconStateFractions
        Requested surface-only target fractions.
    exact_target : SiliconStateComposition
        Preferred nearest-integer target.
    tolerance : float
        Allowed absolute fraction deviation per silicon state.

    Returns
    -------
    candidates : list[_SurfaceTargetCandidate]
        Nearby candidate targets sorted by increasing total fraction error.
    """
    ranges = {}
    for state_name, state_fraction in (
        ("q2_sites", target.q2_fraction),
        ("q3_sites", target.q3_fraction),
        ("t2_sites", target.t2_fraction),
        ("t3_sites", target.t3_fraction),
    ):
        lower = max(
            0,
            math.ceil(max(0.0, state_fraction - tolerance) * total_surface_si - 1e-9),
        )
        upper = min(
            total_surface_si,
            math.floor(min(1.0, state_fraction + tolerance) * total_surface_si + 1e-9),
        )
        ranges[state_name] = (lower, upper)

    q4_lower = max(
        0,
        math.ceil(max(0.0, target.q4_fraction - tolerance) * total_surface_si - 1e-9),
    )
    q4_upper = min(
        total_surface_si,
        math.floor(min(1.0, target.q4_fraction + tolerance) * total_surface_si + 1e-9),
    )

    seen = {exact_target}
    candidates = []
    for q2_sites in range(ranges["q2_sites"][0], ranges["q2_sites"][1] + 1):
        for q3_sites in range(ranges["q3_sites"][0], ranges["q3_sites"][1] + 1):
            for t2_sites in range(ranges["t2_sites"][0], ranges["t2_sites"][1] + 1):
                for t3_sites in range(ranges["t3_sites"][0], ranges["t3_sites"][1] + 1):
                    q4_sites = (
                        total_surface_si - q2_sites - q3_sites - t2_sites - t3_sites
                    )
                    if not (q4_lower <= q4_sites <= q4_upper):
                        continue

                    composition = SiliconStateComposition(
                        total_surface_si=total_surface_si,
                        q2_sites=q2_sites,
                        q3_sites=q3_sites,
                        q4_sites=q4_sites,
                        t2_sites=t2_sites,
                        t3_sites=t3_sites,
                    )
                    if composition in seen:
                        continue

                    errors = _surface_fraction_errors(composition, target)
                    if any(error > tolerance for error in errors):
                        continue

                    candidates.append(
                        _SurfaceTargetCandidate(
                            composition=composition,
                            total_fraction_error=sum(errors),
                        )
                    )
                    seen.add(composition)

    candidates.sort(
        key=lambda candidate: (
            candidate.total_fraction_error,
            candidate.composition.q4_sites,
            candidate.composition.q3_sites,
            candidate.composition.q2_sites,
            candidate.composition.t3_sites,
            candidate.composition.t2_sites,
        )
    )
    return candidates


def _prepared_target_from_final(
    final_surface: SiliconStateComposition,
) -> SiliconStateComposition:
    """Return the bare pre-grafting target required by a final goal.

    Parameters
    ----------
    final_surface : SiliconStateComposition
        Selected final five-state surface composition.

    Returns
    -------
    prepared_surface : SiliconStateComposition
        Bare pre-grafting Q-state composition required to realize the final
        surface after exact ``T2/T3`` attachment.
    """
    return SiliconStateComposition(
        total_surface_si=final_surface.total_surface_si,
        q2_sites=final_surface.q2_sites + final_surface.t2_sites,
        q3_sites=final_surface.q3_sites + final_surface.t3_sites,
        q4_sites=final_surface.q4_sites,
    )


def _prepared_target_is_compatible(
    initial_surface: SiliconStateComposition,
    prepared_surface: SiliconStateComposition,
) -> bool:
    """Return whether a prepared Q-state target is chemically reachable.

    Parameters
    ----------
    initial_surface : SiliconStateComposition
        Initial surface composition before custom condensation.
    prepared_surface : SiliconStateComposition
        Candidate pre-grafting Q-state composition.

    Returns
    -------
    is_compatible : bool
        ``True`` when the current siloxane-editing algebra can in principle
        reach the requested prepared surface.
    """
    return (
        prepared_surface.t2_sites == 0
        and prepared_surface.t3_sites == 0
        and prepared_surface.q2_sites <= initial_surface.q2_sites
        and prepared_surface.q4_sites >= initial_surface.q4_sites
        and prepared_surface.q3_sites % 2 == initial_surface.q3_sites % 2
    )
