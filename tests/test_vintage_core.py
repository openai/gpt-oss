import unittest

from boolelot_vintage.config import VintageConfig, MODEL_PROFILES
from boolelot_vintage.engine import VintageEngine
from boolelot_vintage.prompts import build_system_prompt


class VintageCoreTests(unittest.TestCase):
    def test_default_period_is_nineteenth_century(self):
        config = VintageConfig()
        self.assertEqual((config.period_start, config.period_end), (1800, 1899))

    def test_prompt_marks_temporal_horizon(self):
        prompt = build_system_prompt(VintageConfig(period_start=1837, period_end=1901))
        self.assertIn("1837-1901", prompt)
        self.assertIn("historical horizon", prompt)

    def test_empty_prompt_rejected_before_loading_weights(self):
        with self.assertRaises(ValueError):
            VintageEngine().generate("  ")

    def test_invalid_period_rejected_before_loading_weights(self):
        with self.assertRaises(ValueError):
            VintageEngine(VintageConfig(period_start=1900, period_end=1800)).generate("Hello")

    def test_profiles_are_explicit_about_model_roles(self):
        self.assertIn("victorian-small", MODEL_PROFILES)
        self.assertIn("victorian-research", MODEL_PROFILES)
        self.assertEqual(MODEL_PROFILES["historical-retrieval"]["backend"], "retrieval")


if __name__ == "__main__":
    unittest.main()
