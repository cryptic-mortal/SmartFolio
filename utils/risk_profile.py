"""Utility helpers for mapping a user risk score into concrete knobs."""

from __future__ import annotations


def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, float(value)))


def _lerp(low: float, high: float, weight: float) -> float:
    return low + (high - low) * weight


def build_risk_profile(risk_score: float) -> dict:
    """Create a lightweight profile dict from a normalized risk score."""
    rho = _clamp(risk_score)
    conservative = 1.0 - rho

    markowitz_range = (
        _lerp(3.5, 0.7, rho),  # higher aversion for conservative users
        _lerp(5.0, 1.5, rho),
    )
    bl_range = (
        _lerp(2.5, 0.5, rho),
        _lerp(4.0, 2.0, rho),
    )

    max_weight = _lerp(0.10, 0.60, rho)
    min_weight = 0.005

    # Bias ensemble weights toward defensive (risk parity/HRP) when conservative
    ensemble_weights = {
        "robust_markowitz": 0.25 + 0.15 * rho,
        "black_litterman": 0.20 + 0.10 * rho,
        "risk_parity": 0.25 + 0.15 * conservative,
        "hrp": 0.15 + 0.10 * conservative,
    }
    total = sum(ensemble_weights.values())
    ensemble_weights = {k: v / total for k, v in ensemble_weights.items()}

    return {
        "risk_score": rho,
        "markowitz_ra_range": markowitz_range,
        "black_litterman_ra_range": bl_range,
        "max_weight": max_weight,
        "min_weight": min_weight,
        "ensemble_weights": ensemble_weights,
    }


def get_risk_score(profile: dict | None, default: float = 0.5) -> float:
    """Helper to pull the risk score value out of a profile dict."""
    if not profile:
        return default
    return float(profile.get("risk_score", default))
