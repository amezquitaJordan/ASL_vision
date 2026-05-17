import unittest

import numpy as np

from src.labels import LABEL_TO_LETTER, letter_from_label, static_class_indices
from src.preprocessing import normalize_pixels, reshape_flat_image


class LabelAndPreprocessingTests(unittest.TestCase):
    def test_sign_mnist_labels_skip_dynamic_letters(self):
        self.assertEqual(letter_from_label(0), "A")
        self.assertEqual(letter_from_label(8), "I")
        self.assertEqual(letter_from_label(10), "K")
        self.assertEqual(letter_from_label(24), "Y")
        self.assertNotIn(9, LABEL_TO_LETTER)
        self.assertNotIn(25, LABEL_TO_LETTER)
        self.assertEqual(len(static_class_indices()), 24)

    def test_reshape_flat_image_returns_cnn_shape_and_normalized_values(self):
        flat = np.arange(784, dtype=np.float32)
        image = reshape_flat_image(flat)

        self.assertEqual(image.shape, (28, 28, 1))
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)

    def test_normalize_pixels_handles_uint8_images(self):
        image = np.array([[0, 127, 255]], dtype=np.uint8)
        normalized = normalize_pixels(image)

        self.assertEqual(normalized.dtype, np.float32)
        self.assertAlmostEqual(float(normalized[0, 0]), 0.0)
        self.assertAlmostEqual(float(normalized[0, 2]), 1.0)


if __name__ == "__main__":
    unittest.main()
