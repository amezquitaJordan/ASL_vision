import unittest
from collections import deque

import numpy as np

from src.app import PredictionGateConfig, accept_stable_prediction


class PredictionGateTests(unittest.TestCase):
    def test_rejects_low_confidence_prediction(self):
        history = deque(maxlen=4)
        config = PredictionGateConfig(confidence_threshold=0.80, margin_threshold=0.15, stable_frames=3)
        probabilities = np.array([0.79, 0.10, 0.11], dtype=np.float32)

        result = accept_stable_prediction(probabilities, ["A", "B", "C"], history, config)

        self.assertIsNone(result.letter)
        self.assertEqual(len(history), 0)

    def test_rejects_prediction_with_small_margin(self):
        history = deque(maxlen=4)
        config = PredictionGateConfig(confidence_threshold=0.70, margin_threshold=0.15, stable_frames=3)
        probabilities = np.array([0.76, 0.68, 0.02], dtype=np.float32)

        result = accept_stable_prediction(probabilities, ["A", "B", "C"], history, config)

        self.assertIsNone(result.letter)
        self.assertEqual(len(history), 0)

    def test_accepts_only_after_stable_repeated_frames(self):
        history = deque(maxlen=4)
        config = PredictionGateConfig(confidence_threshold=0.70, margin_threshold=0.15, stable_frames=3)
        probabilities = np.array([0.91, 0.04, 0.05], dtype=np.float32)

        first = accept_stable_prediction(probabilities, ["A", "B", "C"], history, config)
        second = accept_stable_prediction(probabilities, ["A", "B", "C"], history, config)
        third = accept_stable_prediction(probabilities, ["A", "B", "C"], history, config)

        self.assertIsNone(first.letter)
        self.assertIsNone(second.letter)
        self.assertEqual(third.letter, "A")
        self.assertGreaterEqual(third.confidence, 0.90)


if __name__ == "__main__":
    unittest.main()
