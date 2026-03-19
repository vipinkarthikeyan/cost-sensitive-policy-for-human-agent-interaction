import random

OBJECTS = ["red mug", "blue mug", "bottle", "notebook"]

TRIALS = [
    {
        "instruction": "get me the mug",
        "true_object": random.choice(["red mug", "blue mug"]),
        "belief": {"red mug": 0.45, "blue mug": 0.45, "bottle": 0.05, "notebook": 0.05},
        "ask_options": ["red mug", "blue mug"],
    },
    {
        "instruction": "pass me the drink",
        "true_object": random.choice(["red mug", "blue mug", "bottle"]),
        "belief": {"red mug": 0.25, "blue mug": 0.25, "bottle": 0.45, "notebook": 0.05},
        "ask_options": ["bottle", "red mug", "blue mug"],
    },
    {
        "instruction": "get me the reading item",
        "true_object": "notebook",
        "belief": {"red mug": 0.05, "blue mug": 0.05, "bottle": 0.10, "notebook": 0.80},
        "ask_options": ["notebook", "bottle"],
    },
]

def sample_trial():
    return random.choice(TRIALS)

def resolve_action(action, belief, true_object, reward_correct=10, cost_wrong=12, cost_ask=2):
    if action == "ACT":
        chosen = max(belief, key=belief.get)
        correct = chosen == true_object
        utility = reward_correct if correct else -cost_wrong
        return {
            "chosen": chosen,
            "correct": correct,
            "utility": utility,
            "done": True,
            "message": f"Robot acted and chose: {chosen}",
        }

    if action == "ASK":
        return {
            "chosen": None,
            "correct": None,
            "utility": -cost_ask,
            "done": False,
            "message": "Robot asked for clarification.",
        }

    raise ValueError(f"Unknown action: {action}")
