import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.real_images import load_real_image_dataset


class RealImageTrainingTests(unittest.TestCase):
    def test_load_real_image_dataset_uses_letter_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for letter, value in [("A", 40), ("B", 180), ("J", 255)]:
                folder = root / letter
                folder.mkdir()
                image = np.full((32, 32, 3), value, dtype=np.uint8)
                cv2.imwrite(str(folder / f"{letter}.jpg"), image)

            x, y = load_real_image_dataset(root, cropper=lambda frame: frame)

        self.assertEqual(x.shape, (2, 28, 28, 1))
        self.assertEqual(y.tolist(), [0, 1])
        self.assertGreaterEqual(float(x.min()), 0.0)
        self.assertLessEqual(float(x.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
