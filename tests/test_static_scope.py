import json
import unittest
from pathlib import Path

from src.labels import class_names
from src.real_images import iter_labeled_image_paths


class StaticScopeTests(unittest.TestCase):
    def test_static_classes_exclude_dynamic_letters(self):
        names = class_names()

        self.assertEqual(len(names), 24)
        self.assertNotIn("J", names)
        self.assertNotIn("Z", names)

    def test_real_dataset_loader_ignores_dynamic_letter_folders(self):
        dataset_root = Path("dataset") / "senas_reales_entrenamiento"
        paths = [path for path, _ in iter_labeled_image_paths(dataset_root)]
        folder_names = {path.parent.name.upper() for path in paths}

        self.assertNotIn("J", folder_names)
        self.assertNotIn("Z", folder_names)

    def test_class_map_does_not_publish_dynamic_letters(self):
        class_map_path = Path("models") / "class_map.json"
        if not class_map_path.exists():
            self.skipTest("No existe models/class_map.json")

        data = json.loads(class_map_path.read_text(encoding="utf-8"))

        self.assertNotIn("motion_letters", data)
        self.assertNotIn("J", data.get("class_names", []))
        self.assertNotIn("Z", data.get("class_names", []))


if __name__ == "__main__":
    unittest.main()
