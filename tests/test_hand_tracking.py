import unittest

import numpy as np

from src.hand_tracking import fixed_roi_bbox, fingertip_from_mask


class HandTrackingTests(unittest.TestCase):
    def test_fixed_roi_bbox_stays_inside_frame(self):
        self.assertEqual(fixed_roi_bbox(640, 480), (200, 82, 440, 322))

    def test_fingertip_from_mask_returns_topmost_contour_point(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 40:70] = 255
        mask[10:25, 50:55] = 255

        point = fingertip_from_mask(mask, roi_origin=(100, 200), frame_size=(300, 400))

        self.assertIsNotNone(point)
        self.assertAlmostEqual(point[0], 150 / 300, places=2)
        self.assertAlmostEqual(point[1], 210 / 400, places=2)


if __name__ == "__main__":
    unittest.main()
