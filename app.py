import json
import os
import re

import pandas as pd
import requests
import streamlit as st

from simulator import sample_trial, resolve_action
from policy import (
    never_ask_policy,
    always_ask_policy,
    threshold_policy,
    cost_sensitive_policy,
    derived_threshold_policy,
    explain_decision,
)
from human_model import PROFILES, effective_ca

POLICIES = {
    "Never Ask": never_ask_policy,
    "Always Ask": always_ask_policy,
    "Threshold": threshold_policy,
    "Cost Sensitive": cost_sensitive_policy,
    "Derived Threshold": derived_threshold_policy,
}

st.set_page_config(page_title="Algorithmic HRI Project", layout="wide")
st.title("When Should a Robot Ask?")
st.subheader("Cost-Sensitive Clarification in Human-Robot Collaboration")

st.markdown(
    "### Optional: Provide an OpenAI API key for better belief updates (overrides env/secrets)"
)
st.text_input(
    "OpenAI API key",
    value=st.session_state.get("openai_api_key", ""),
    type="password",
    key="openai_api_key",
)


def _get_openai_key_status() -> tuple[str, bool]:
    """Return (source, found) for the OpenAI key.

    source is 'session', 'env', 'secrets', or 'none'.
    """

    if st.session_state.get("openai_api_key"):
        return "session", True

    if os.getenv("OPENAI_API_KEY"):
        return "env", True

    try:
        if st.secrets.get("openai_api_key"):
            return "secrets", True
    except Exception:
        pass

    return "none", False


key_source, has_key = _get_openai_key_status()
if not has_key:
    st.warning(
        "No OpenAI API key detected. The app will use a uniform belief distribution (no LLM). "
        "Set OPENAI_API_KEY, enter a key above, or add a .streamlit/secrets.toml entry to enable the LLM."
    )
else:
    st.info(f"OpenAI key detected via: {key_source}. (Key is not displayed.)")


def _extract_json(text: str) -> dict | None:
    """Try to find a JSON object in model output."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _normalize_belief(belief: dict[str, float]) -> dict[str, float]:
    total = sum(belief.values())
    if total <= 0:
        return {k: 1 / len(belief) for k in belief}
    return {k: v / total for k, v in belief.items()}


def _update_belief_from_llm(trial: dict) -> None:
    """Update trial['belief'] using the LLM and show the model output."""

    try:
        updated_belief, model_output = llm_update_belief(
            st.session_state.instruction, list(trial["belief"].keys())
        )
        if updated_belief:
            trial["belief"] = updated_belief
            st.session_state.trial = trial
            st.success("Robot belief updated from LLM output.")
            st.code(model_output, language="text")
        else:
            st.error("LLM response did not include a valid JSON belief distribution.")
            st.code(model_output, language="text")
    except Exception as e:
        st.error(f"Failed to update belief via LLM: {e}")


def get_ask_options(belief: dict[str, float], delta: float = 0.05) -> list[str]:
    """Return candidate objects to ask about based on current belief.

    If multiple objects have probabilities within `delta` of the top belief, include them all.
    """

    if not belief:
        return []

    max_prob = max(belief.values())
    threshold = max_prob - delta

    # Include any object whose probability is within delta of the max.
    return [obj for obj, p in sorted(belief.items(), key=lambda kv: -kv[1]) if p >= threshold]


def llm_update_belief(instruction: str, objects: list[str]) -> tuple[dict[str, float], str]:
    """Query an LLM to return an updated belief distribution over objects.

    Returns (updated_belief, raw_model_text).
    """

    api_key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("openai_api_key")
        except Exception:
            api_key = None

    if not api_key:
        # Without an API key, return uniform belief (no LLM inference).
        return _normalize_belief({obj: 1.0 for obj in objects}), (
            "No OPENAI_API_KEY found; returning uniform belief distribution."
        )

    system_prompt = (
        "You are an assistant that outputs only a single JSON object. "
        "The keys must exactly match the candidate objects provided, and the values must be probabilities (numbers) that sum to 1. "
        "Do not include any extra text, commentary, or markdown formatting."
    )

    user_prompt = (
        f"Instruction: {instruction}\n"
        f"Candidate objects: {objects}\n"
        "Output must be a JSON object with one key for each candidate object. "
        "Example output (JSON only): {\"red mug\": 0.5, \"blue mug\": 0.5, \"bottle\": 0.0, \"notebook\": 0.0}"
    )

    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 256,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()

    parsed = _extract_json(text)
    if not parsed:
        return _normalize_belief({obj: 1.0 for obj in objects}), text

    # Keep only known objects and fill missing ones with 0
    parsed = {k: float(parsed.get(k, 0)) for k in objects}
    return _normalize_belief(parsed), text

if "trial" not in st.session_state:
    st.session_state.trial = sample_trial()
    st.session_state.true_object = st.session_state.trial["true_object"]
if "history" not in st.session_state:
    st.session_state.history = []
if "instruction" not in st.session_state:
    st.session_state.instruction = ""
if "true_object" not in st.session_state:
    st.session_state.true_object = st.session_state.trial["true_object"]

trial = st.session_state.trial

sel_col1, sel_col2 = st.columns(2)

with sel_col1:
    policy_name = st.selectbox("Choose policy", list(POLICIES.keys()))

with sel_col2:
    profile_name = st.selectbox(
        "Human profile",
        list(PROFILES.keys()),
        help="patient: tolerant of questions, low fatigue | busy: high baseline interruption cost | interruption_averse: cost grows rapidly with repeated asks",
    )

# --- Dynamic clarification cost ----------------------------------------------

n_asks = sum(1 for h in st.session_state.history if h.get("action") == "ASK")
dynamic_ca = effective_ca(profile_name, n_asks)

profile_info = PROFILES[profile_name]
st.info(
    f"**Human profile:** {profile_name} | "
    f"Ca_base = {profile_info['Ca_base']} | "
    f"fatigue_rate = {profile_info['fatigue_rate']} | "
    f"ASKs so far = {n_asks} | "
    f"**Effective Ca = {dynamic_ca:.2f}**"
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Human instruction")
    st.text_input(
        "Type an instruction",
        value=st.session_state.instruction,
        key="instruction",
    )

    st.markdown("### True (ground-truth) object")
    st.selectbox(
        "Select the true object (used for evaluation)",
        list(trial["belief"].keys()),
        index=list(trial["belief"].keys()).index(st.session_state.true_object),
        key="true_object",
    )

    if st.button("Update beliefs (LLM)"):
        try:
            with st.spinner("Calling the LLM to update robot belief..."):
                updated_belief, model_output = llm_update_belief(
                    st.session_state.instruction, list(trial["belief"].keys())
                )

            if updated_belief:
                trial["belief"] = updated_belief
                st.session_state.trial = trial
                st.success("Robot belief updated from LLM output.")
                st.code(model_output, language="text")
            else:
                st.error(
                    "LLM response did not include a valid JSON belief distribution."
                )
                st.code(model_output, language="text")
        except Exception as e:
            st.error(f"Failed to update belief via LLM: {e}")

    st.markdown("### Candidate objects")
    for obj in trial["belief"]:
        st.write(f"- {obj}")

with col2:
    st.markdown("### Robot belief")
    belief_df = pd.DataFrame(
        {"object": list(trial["belief"].keys()), "probability": list(trial["belief"].values())}
    )
    st.dataframe(belief_df, use_container_width=True)

    st.markdown("### Edit candidate objects")
    if "new_object" not in st.session_state:
        st.session_state.new_object = ""

    new_object = st.text_input("Add a candidate object", key="new_object")
    if st.button("Add object"):
        name = new_object.strip()
        if not name:
            st.warning("Enter a non-empty object name.")
        elif name in trial["belief"]:
            st.warning(f"Object '{name}' already exists.")
        else:
            trial["belief"][name] = 0.0
            st.session_state.trial = trial
            _update_belief_from_llm(trial)
            st.experimental_rerun()

    to_remove = st.multiselect(
        "Remove candidate objects",
        options=list(trial["belief"].keys()),
        key="remove_objects",
    )
    if st.button("Remove selected"):
        if not to_remove:
            st.warning("Select at least one object to remove.")
        else:
            for obj in to_remove:
                trial["belief"].pop(obj, None)
            if st.session_state.true_object not in trial["belief"]:
                st.session_state.true_object = next(iter(trial["belief"].keys()), "")
            st.session_state.trial = trial
            _update_belief_from_llm(trial)

if st.button("Run policy"):
    if not st.session_state.instruction.strip():
        st.warning("Please type an instruction before running the policy.")
    else:
        policy_fn = POLICIES[policy_name]
        action = policy_fn(trial["belief"], cost_ask=dynamic_ca)
        result = resolve_action(action, trial["belief"], st.session_state.true_object)

        st.session_state.history.append(
            {
                "instruction": st.session_state.instruction,
                "true_object": st.session_state.true_object,
                "profile": profile_name,
                "policy": policy_name,
                "action": action,
                "ca_used": round(dynamic_ca, 2),
                "utility": result["utility"],
                "correct": result["correct"],
                "message": result["message"],
            }
        )

        st.success(f"Action: {action}")
        st.write(result["message"])

        with st.expander("Decision breakdown"):
            explanation = explain_decision(trial["belief"], cost_ask=dynamic_ca)
            explanation["ca_used"] = round(dynamic_ca, 2)
            explanation["profile"] = profile_name
            explanation["n_asks"] = n_asks
            st.json(explanation)

        if action == "ASK":
            ask_options = get_ask_options(trial["belief"])
            if not ask_options:
                st.warning("No close alternatives to ask about.")
            else:
                answer = st.radio("Clarification answer", ask_options, index=0)
                if st.button("Submit clarification"):
                    # Clarification is not a final retrieval action, so we do not mark this step as correct.
                    correct = None
                    utility = 10 if answer == st.session_state.true_object else -12
                    st.session_state.history.append(
                        {
                            "instruction": st.session_state.instruction,
                            "true_object": st.session_state.true_object,
                            "profile": profile_name,
                            "policy": policy_name,
                            "action": f"ANSWER:{answer}",
                            "ca_used": round(dynamic_ca, 2),
                            "utility": utility,
                            "correct": correct,
                            "message": f"User answered {answer}",
                        }
                    )
                    st.write(f"Clarified target: {answer}")
                    st.write(f"Correct: {answer == st.session_state.true_object}")

nav_col1, nav_col2 = st.columns(2)

with nav_col1:
    if st.button("Next trial"):
        st.session_state.trial = sample_trial()
        st.session_state.true_object = st.session_state.trial["true_object"]
        st.session_state.instruction = ""
        st.rerun()

with nav_col2:
    if st.button("Reset session"):
        st.session_state.history = []
        st.session_state.trial = sample_trial()
        st.session_state.true_object = st.session_state.trial["true_object"]
        st.session_state.instruction = ""
        st.rerun()

st.markdown("### History")
if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    # Display correct more clearly: True/False become explicit labels, None becomes blank.
    history_df["correct"] = history_df["correct"].apply(
        lambda v: "Yes" if v is True else "No" if v is False else ""
    )
    st.dataframe(history_df, use_container_width=True)