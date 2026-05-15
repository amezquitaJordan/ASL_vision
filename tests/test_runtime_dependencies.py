import unittest


class RuntimeDependencyTests(unittest.TestCase):
    def test_mediapipe_exposes_classic_hands_api(self):
        import mediapipe as mp

        self.assertTrue(hasattr(mp, "solutions"))
        self.assertTrue(hasattr(mp.solutions, "hands"))


if __name__ == "__main__":
    unittest.main()
