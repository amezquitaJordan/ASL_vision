import unittest

from src.hand_tracking import fixed_roi_bbox


class HandTrackingTests(unittest.TestCase):
    def test_fixed_roi_bbox_stays_inside_frame(self):
        self.assertEqual(fixed_roi_bbox(640, 480), (200, 82, 440, 322))

    def test_fixed_roi_bbox_uses_square_region(self):
        x1, y1, x2, y2 = fixed_roi_bbox(800, 600)

        self.assertEqual(x2 - x1, y2 - y1)


if __name__ == "__main__":
    unittest.main()
