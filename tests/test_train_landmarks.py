import tempfile
import unittest
from pathlib import Path

from src.train_landmarks import find_conflicting_hashes


class TrainLandmarkDataQualityTests(unittest.TestCase):
    def test_find_conflicting_hashes_detects_same_file_in_different_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.jpg"
            b = root / "b.jpg"
            c = root / "c.jpg"
            a.write_bytes(b"same-image")
            b.write_bytes(b"same-image")
            c.write_bytes(b"different-image")

            path_hashes, conflicts = find_conflicting_hashes([(a, 0), (b, 1), (c, 1)])

        self.assertEqual(len(path_hashes), 3)
        self.assertEqual(len(conflicts), 1)
        self.assertIn(path_hashes[a], conflicts)


if __name__ == "__main__":
    unittest.main()
