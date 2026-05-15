import tempfile
import unittest
from pathlib import Path

from src.tts import TTSConfig, cached_audio_path, should_speak


class TTSTests(unittest.TestCase):
    def test_cached_audio_path_uses_letter_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = cached_audio_path("A", Path(tmp))

        self.assertEqual(path.name, "A.mp3")

    def test_should_speak_waits_for_stable_letter(self):
        config = TTSConfig(stable_frames=3, cooldown_seconds=1.0)
        state = {"letter": None, "count": 0, "last_spoken": {}, "now": 10.0}

        self.assertFalse(should_speak("B", state, config))
        state["now"] = 10.1
        self.assertFalse(should_speak("B", state, config))
        state["now"] = 10.2
        self.assertTrue(should_speak("B", state, config))
        state["now"] = 10.3
        self.assertFalse(should_speak("B", state, config))
        state["now"] = 11.4
        self.assertTrue(should_speak("B", state, config))


if __name__ == "__main__":
    unittest.main()
