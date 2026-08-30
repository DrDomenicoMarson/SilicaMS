"""Shared validation helpers for public scientific configuration objects."""

from __future__ import annotations

import math
from numbers import Integral, Real


def finite_real(name: str, value: Real) -> float:
    """Return one finite real-valued configuration field as ``float``.

    Parameters
    ----------
    name : str
        Human-readable field name used in error messages.
    value : numbers.Real
        Value to validate. Boolean values are not accepted as real-valued
        scientific parameters.

    Returns
    -------
    normalized : float
        Validated finite value.

    Raises
    ------
    TypeError
        Raised when ``value`` is not a non-boolean real number.
    ValueError
        Raised when ``value`` is NaN or infinite.
    """

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def integral(name: str, value: Integral) -> int:
    """Return one integer-valued configuration field as ``int``.

    Parameters
    ----------
    name : str
        Human-readable field name used in error messages.
    value : numbers.Integral
        Integer value to validate. Boolean values are rejected explicitly.

    Returns
    -------
    normalized : int
        Validated Python integer.

    Raises
    ------
    TypeError
        Raised when ``value`` is not an integer or is a boolean.
    """

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    return int(value)


def boolean(name: str, value: bool) -> bool:
    """Return one strictly boolean configuration field.

    Parameters
    ----------
    name : str
        Human-readable field name used in error messages.
    value : bool
        Value to validate.

    Returns
    -------
    normalized : bool
        The validated value.

    Raises
    ------
    TypeError
        Raised when ``value`` is not exactly a boolean.
    """

    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def finite_range(
    name: str,
    value: tuple[Real, Real],
    *,
    positive: bool = False,
) -> tuple[float, float]:
    """Validate a finite, strictly increasing two-value range.

    Parameters
    ----------
    name : str
        Human-readable field name used in error messages.
    value : tuple[numbers.Real, numbers.Real]
        Lower and upper bounds.
    positive : bool, optional
        When ``True``, require both bounds to be strictly positive.

    Returns
    -------
    normalized : tuple[float, float]
        Validated lower and upper bounds.

    Raises
    ------
    TypeError
        Raised when ``value`` is not a tuple or a bound is not real-valued.
    ValueError
        Raised when the tuple length, finiteness, positivity, or ordering is
        invalid.
    """

    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple.")
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two bounds.")
    lower = finite_real(f"{name} lower bound", value[0])
    upper = finite_real(f"{name} upper bound", value[1])
    if positive and (lower <= 0.0 or upper <= 0.0):
        raise ValueError(f"{name} bounds must be strictly positive.")
    if lower >= upper:
        raise ValueError(f"{name} lower bound must be smaller than its upper bound.")
    return lower, upper
