"""
Human model for cost-sensitive clarification policy.

Each profile encodes:
- Ca_base: baseline cost of a single clarification request
- fatigue_rate: additional cost per prior interruption in the session

The effective clarification cost grows with n_asks:
    Ca(t) = Ca_base + fatigue_rate * n_asks

Profiles:
- patient: tolerant of questions, minimal fatigue
- busy: high baseline cost, moderate fatigue
- interruption_averse: moderate baseline, rapid fatigue
"""

PROFILES = {
    "patient":             {"Ca_base": 1.0, "fatigue_rate": 0.1},
    "busy":                {"Ca_base": 5.0, "fatigue_rate": 0.5},
    "interruption_averse": {"Ca_base": 3.0, "fatigue_rate": 1.5},
}


def effective_ca(profile_name: str, n_asks: int) -> float:
    """
    Compute the effective clarification cost given a human profile and
    the number of prior ASK actions in the current session.

    Args:
        profile_name: one of 'patient', 'busy', 'interruption_averse'
        n_asks: number of times the robot has already asked in this session

    Returns:
        Effective Ca to use in Q(ASK) = -Ca
    """
    profile = PROFILES[profile_name]
    return profile["Ca_base"] + profile["fatigue_rate"] * n_asks
