import tempfile
import unittest
from pathlib import Path

from gpt_oss.tools.apply_patch import apply_patch


class ApplyPatchMoveTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.source = Path(directory.name) / "source.txt"
        self.source.write_text("before\n", encoding="utf-8")

    def update(self, move_to=None):
        move = f"*** Move to: {move_to}\n" if move_to is not None else ""
        patch = (
            f"*** Begin Patch\n*** Update File: {self.source}\n"
            f"{move}@@\n-before\n+after\n*** End Patch"
        )
        self.assertEqual(apply_patch(patch), "Done!")

    def test_move_to_same_path_preserves_updated_file(self):
        self.update(move_to=self.source)

        self.assertTrue(self.source.exists())
        self.assertEqual(self.source.read_text(encoding="utf-8"), "after\n")

    def test_move_to_different_path_removes_source(self):
        destination = self.source.with_name("destination.txt")
        self.update(move_to=destination)

        self.assertFalse(self.source.exists())
        self.assertEqual(destination.read_text(encoding="utf-8"), "after\n")

    def test_update_without_move_preserves_updated_file(self):
        self.update()

        self.assertEqual(self.source.read_text(encoding="utf-8"), "after\n")


if __name__ == "__main__":
    unittest.main()
