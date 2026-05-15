import unittest

from src.motion_letters import MotionLetterDetector


class MotionLetterDetectorTests(unittest.TestCase):
    def test_detects_j_like_hook_motion(self):
        detector = MotionLetterDetector(min_points=5)
        path = [(0.60, 0.20), (0.58, 0.36), (0.55, 0.52), (0.48, 0.68), (0.38, 0.74)]

        detected = None
        for point in path:
            detected = detector.update(point)

        self.assertEqual(detected, "J")

    def test_detects_z_like_three_segment_motion(self):
        detector = MotionLetterDetector(min_points=6)
        path = [
            (0.25, 0.25),
            (0.48, 0.25),
            (0.70, 0.25),
            (0.50, 0.48),
            (0.30, 0.70),
            (0.55, 0.70),
            (0.78, 0.70),
        ]

        detections = []
        for point in path:
            detections.append(detector.update(point))

        self.assertIn("Z", detections)

    def test_static_hand_does_not_trigger_motion_letter(self):
        detector = MotionLetterDetector(min_points=5)
        detected = None
        for _ in range(8):
            detected = detector.update((0.50, 0.50))

        self.assertIsNone(detected)


if __name__ == "__main__":
    unittest.main()
