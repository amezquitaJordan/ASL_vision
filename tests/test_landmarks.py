import unittest
from types import SimpleNamespace

import numpy as np

from src.landmarks import LANDMARK_VECTOR_SIZE, landmarks_to_feature_vector


def fake_landmarks(points):
    landmarks = [SimpleNamespace(x=x, y=y) for x, y in points]
    return SimpleNamespace(landmark=landmarks)


class LandmarkFeatureTests(unittest.TestCase):
    def test_landmarks_to_feature_vector_has_expected_shape(self):
        points = [(index / 100, (index + 1) / 100) for index in range(21)]

        features = landmarks_to_feature_vector(fake_landmarks(points))

        self.assertEqual(features.shape, (LANDMARK_VECTOR_SIZE,))
        self.assertEqual(features.dtype, np.float32)

    def test_landmarks_to_feature_vector_is_translation_invariant(self):
        base_points = [(index / 100, (index + 1) / 100) for index in range(21)]
        moved_points = [(x + 0.25, y + 0.35) for x, y in base_points]

        base = landmarks_to_feature_vector(fake_landmarks(base_points))
        moved = landmarks_to_feature_vector(fake_landmarks(moved_points))

        np.testing.assert_allclose(base, moved, atol=1e-6)

    def test_landmarks_to_feature_vector_rejects_incomplete_hands(self):
        points = [(index / 100, (index + 1) / 100) for index in range(20)]

        with self.assertRaises(ValueError):
            landmarks_to_feature_vector(fake_landmarks(points))


if __name__ == "__main__":
    unittest.main()
