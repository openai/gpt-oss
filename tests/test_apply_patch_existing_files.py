"""Offline checks for add-file collisions: python -m unittest discover -s tests -p test_apply_patch_existing_files.py."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from gpt_oss.tools.apply_patch import DiffError, apply_patch


def add_patch(path, content="replacement"):
    return f"*** Begin Patch\n*** Add File: {path}\n+{content}\n*** End Patch"


class ExistingFileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_existing_text_file_is_preserved(self):
        path = self.root / "existing.txt"
        path.write_text("original\n", encoding="utf-8")
        with self.assertRaisesRegex(DiffError, "already exists"):
            apply_patch(add_patch(path))
        self.assertEqual(path.read_text(encoding="utf-8"), "original\n")

    def test_existing_empty_file_is_preserved(self):
        path = self.root / "empty.txt"
        path.touch()
        with self.assertRaisesRegex(DiffError, "already exists"):
            apply_patch(add_patch(path))
        self.assertEqual(path.read_bytes(), b"")

    def test_collision_aborts_before_any_patch_writes(self):
        old = self.root / "update.txt"
        collision = self.root / "collision.txt"
        new = self.root / "new.txt"
        old.write_text("before", encoding="utf-8")
        collision.write_text("keep", encoding="utf-8")
        patch = (f"*** Begin Patch\n*** Update File: {old}\n@@\n-before\n+after\n"
                 f"*** Add File: {new}\n+new\n*** Add File: {collision}\n+bad\n*** End Patch")
        with self.assertRaisesRegex(DiffError, "already exists"):
            apply_patch(patch)
        self.assertEqual(old.read_text(), "before")
        self.assertEqual(collision.read_text(), "keep")
        self.assertFalse(new.exists())

    def test_new_file_and_parent_directories_are_created(self):
        path = self.root / "new" / "nested.txt"
        self.assertEqual(apply_patch(add_patch(path, "café")), "Done!")
        self.assertEqual(path.read_text(encoding="utf-8"), "café")

    def test_mapping_callbacks_reject_existing_files(self):
        files = {"existing.txt": "original"}
        with self.assertRaisesRegex(DiffError, "already exists"):
            apply_patch(add_patch("existing.txt"), files.__getitem__, files.__setitem__, files.__delitem__)
        self.assertEqual(files, {"existing.txt": "original"})

    def test_mapping_callbacks_allow_missing_files(self):
        files = {}
        apply_patch(add_patch("new.txt"), files.__getitem__, files.__setitem__, files.__delitem__)
        self.assertEqual(files, {"new.txt": "replacement"})

    def test_file_not_found_callback_allows_creation(self):
        write = Mock()
        apply_patch(add_patch("new.txt"), Mock(side_effect=FileNotFoundError), write, Mock())
        write.assert_called_once_with("new.txt", "replacement")

    def test_read_errors_do_not_authorize_overwriting(self):
        write = Mock()
        failure = PermissionError("synthetic read denial")
        with self.assertRaises(PermissionError) as raised:
            apply_patch(add_patch("unreadable.txt"), Mock(side_effect=failure), write, Mock())
        self.assertIs(raised.exception, failure)
        write.assert_not_called()

    def test_directory_target_is_not_replaced(self):
        path = self.root / "directory"
        path.mkdir()
        with self.assertRaises(IsADirectoryError):
            apply_patch(add_patch(path))
        self.assertTrue(path.is_dir())

    def test_update_and_delete_keep_existing_behavior(self):
        files = {"update.txt": "old", "delete.txt": "remove"}
        patch = ("*** Begin Patch\n*** Update File: update.txt\n@@\n-old\n+new\n"
                 "*** Delete File: delete.txt\n*** End Patch")
        apply_patch(patch, files.__getitem__, files.__setitem__, files.__delitem__)
        self.assertEqual(files, {"update.txt": "new"})

    def test_duplicate_add_sections_still_fail_without_writes(self):
        files = {}
        patch = "*** Begin Patch\n*** Add File: new.txt\n+one\n*** Add File: new.txt\n+two\n*** End Patch"
        with self.assertRaisesRegex(DiffError, "Duplicate add"):
            apply_patch(patch, files.__getitem__, files.__setitem__, files.__delitem__)
        self.assertEqual(files, {})


if __name__ == "__main__":
    unittest.main()
