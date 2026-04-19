"""
Policy module for Algorithmic HRI project

We operate in belief space:
- belief[g] = P(g | instruction)

Actions:
- ACT: choose most likely object
- ASK: ask clarification

We implement:
1. Baselines (always ask, never ask, threshold)
2. Cost-sensitive optimal policy (derived from expected utility)

Core idea:
Choose action that maximizes expected utility.
"""

import math


# =========================
# Utility Functions
# =========================

def normalize_belief(belief):
    """
    Return a probability distribution over the same keys as ``belief``.

    Raises ``ValueError`` if the belief is empty, contains a negative weight,
    or has zero total weight.
    """
    if not belief:
        raise ValueError("belief must contain at least one entry")

    if any(v < 0 for v in belief.values()):
        raise ValueError("belief weights must be non-negative")

    total = sum(belief.values())
    if total <= 0:
        raise ValueError("belief weights must sum to a positive number")

    return {k: v / total for k, v in belief.items()}


def entropy(belief):
    """
    Measures uncertainty in belief distribution.
    Higher entropy = more ambiguity.

    Operates on a normalized copy of ``belief``.
    """
    eps = 1e-9
    normalized = normalize_belief(belief)
    return -sum(p * math.log(p + eps) for p in normalized.values() if p > 0)


def best_object(belief):
    """
    Returns most likely object (argmax).
    """
    return max(belief, key=belief.get)


def max_prob(belief):
    """
    Returns p* = max probability from the normalized belief.
    """
    return max(normalize_belief(belief).values())


# =========================
# Expected Utility Functions
# =========================

def expected_utility_act(belief, reward_correct=10, cost_wrong=12):
    """
    Q(b, ACT) = p* * R_c - (1 - p*) * C_w
    """
    p_star = max_prob(belief)
    return p_star * reward_correct - (1 - p_star) * cost_wrong


def expected_utility_ask(cost_ask=2):
    """
    Q(b, ASK) = -C_a
    (simplified one-step version)
    """
    return -cost_ask


# =========================
# Baseline Policies
# =========================

def never_ask_policy(belief, **kwargs):
    """
    Always act immediately.
    """
    return "ACT"


def always_ask_policy(belief, **kwargs):
    """
    Always ask before acting.
    """
    return "ASK"


def threshold_policy(belief, threshold=0.75, **kwargs):
    """
    Heuristic baseline:
    If confidence > threshold -> ACT
    else -> ASK
    """
    if max_prob(belief) >= threshold:
        return "ACT"
    return "ASK"


# =========================
# Cost-Sensitive Policy (MAIN)
# =========================

def cost_sensitive_policy(
    belief,
    reward_correct=10,
    cost_wrong=12,
    cost_ask=2,
):
    """
    Optimal policy in belief space:

    pi(b) = argmax_a Q(b, a)

    Where:
    Q(ACT) = p* R_c - (1-p*) C_w
    Q(ASK) = -C_a
    """

    u_act = expected_utility_act(belief, reward_correct, cost_wrong)
    u_ask = expected_utility_ask(cost_ask)

    utilities = {
        "ACT": u_act,
        "ASK": u_ask,
    }

    return max(utilities, key=utilities.get)


# =========================
# Derived Threshold Policy (IMPORTANT)
# =========================

def derived_threshold_policy(
    belief,
    reward_correct=10,
    cost_wrong=12,
    cost_ask=2,
):
    """
    Instead of an arbitrary threshold, derive it from costs:

    p* >= (C_w - C_a) / (R_c + C_w)
    """

    p_star = max_prob(belief)

    threshold = (cost_wrong - cost_ask) / (reward_correct + cost_wrong)

    if p_star >= threshold:
        return "ACT"
    return "ASK"


# =========================
# Debug / Explain Helper
# =========================

def explain_decision(
    belief,
    reward_correct=10,
    cost_wrong=12,
    cost_ask=2,
):
    """
    Returns detailed breakdown of decision for debugging or UI.
    """

    p_star = max_prob(belief)

    u_act = expected_utility_act(belief, reward_correct, cost_wrong)
    u_ask = expected_utility_ask(cost_ask)

    threshold = (cost_wrong - cost_ask) / (reward_correct + cost_wrong)

    return {
        "p_star": p_star,
        "u_act": u_act,
        "u_ask": u_ask,
        "derived_threshold": threshold,
        "entropy": entropy(belief),
    }
