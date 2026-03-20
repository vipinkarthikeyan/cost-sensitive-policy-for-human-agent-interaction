import unittest

from policy import (
    best_object,
    cost_sensitive_policy,
    derived_threshold_policy,
    entropy,
    explain_decision,
    max_prob,
    normalize_belief,
)


class PolicyNormalizationTests(unittest.TestCase):
    def test_normalize_belief_scales_positive_weights(self):
        belief = {"red mug": 3.0, "blue mug": 1.0}

        normalized = normalize_belief(belief)

        self.assertAlmostEqual(normalized["red mug"], 0.75)
        self.assertAlmostEqual(normalized["blue mug"], 0.25)
        self.assertAlmostEqual(sum(normalized.values()), 1.0)

    def test_normalize_belief_rejects_empty_belief(self):
        with self.assertRaises(ValueError):
            normalize_belief({})

    def test_normalize_belief_rejects_negative_weights(self):
        with self.assertRaises(ValueError):
            normalize_belief({"red mug": 0.8, "blue mug": -0.2})

    def test_normalize_belief_rejects_zero_total_weight(self):
        with self.assertRaises(ValueError):
            normalize_belief({"red mug": 0.0, "blue mug": 0.0})

    def test_max_prob_and_best_object_work_on_unnormalized_input(self):
        belief = {"red mug": 6.0, "blue mug": 4.0}

        self.assertEqual(best_object(belief), "red mug")
        self.assertAlmostEqual(max_prob(belief), 0.6)

    def test_cost_sensitive_and_derived_threshold_policies_remain_equivalent(self):
        belief = {"red mug": 9.0, "blue mug": 11.0}

        action_cost_sensitive = cost_sensitive_policy(
            belief,
            reward_correct=10,
            cost_wrong=12,
            cost_ask=2,
        )
        action_threshold = derived_threshold_policy(
            belief,
            reward_correct=10,
            cost_wrong=12,
            cost_ask=2,
        )

        self.assertEqual(action_cost_sensitive, action_threshold)
        self.assertEqual(action_cost_sensitive, "ACT")

    def test_explain_decision_reports_normalized_probability(self):
        belief = {"red mug": 2.0, "blue mug": 3.0}

        explanation = explain_decision(belief)

        self.assertAlmostEqual(explanation["p_star"], 0.6)
        self.assertIn("entropy", explanation)

    def test_entropy_uses_normalized_belief(self):
        normalized_entropy = entropy({"red mug": 0.5, "blue mug": 0.5})
        raw_entropy = entropy({"red mug": 5.0, "blue mug": 5.0})

        self.assertAlmostEqual(normalized_entropy, raw_entropy)


if __name__ == "__main__":
    unittest.main()
