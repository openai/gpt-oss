"""Run offline with python -m unittest discover -s tests -p test_apply_patch_unicode.py."""
import unittest

from gpt_oss.tools.apply_patch import (
    DiffError,
    apply_patch,
    identify_files_added,
    identify_files_needed,
)


class PatchUnicodeTests(unittest.TestCase):
    separators = ("\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029")

    def apply(self, text, files):
        return apply_patch(text, files.__getitem__, files.__setitem__, files.__delitem__)

    def test_added_content_preserves_non_newline_separators(self):
        for char in self.separators:
            with self.subTest(char=repr(char)):
                files = {}
                content = f"before{char}+after"
                self.apply(f"*** Begin Patch\n*** Add File: fixture.txt\n+{content}\n*** End Patch", files)
                self.assertEqual(files, {"fixture.txt": content})

    def test_update_matches_original_unicode_content(self):
        for char in self.separators:
            with self.subTest(char=repr(char)):
                old, new = f"before{char}after", f"changed{char}value"
                files = {"fixture.txt": old}
                self.apply(f"*** Begin Patch\n*** Update File: fixture.txt\n@@\n-{old}\n+{new}\n*** End Patch", files)
                self.assertEqual(files["fixture.txt"], new)

    def test_file_discovery_preserves_unicode_path_characters(self):
        for char in self.separators:
            with self.subTest(char=repr(char)):
                path = f"before{char}after.txt"
                update = f"*** Begin Patch\n*** Update File: {path}\n@@\n-old\n+new\n*** End Patch"
                self.assertEqual(identify_files_needed(update), [path])
                files = {path: "old"}
                self.apply(update, files)
                self.assertEqual(files, {path: "new"})
                delete = f"*** Begin Patch\n*** Delete File: {path}\n*** End Patch"
                self.assertEqual(identify_files_needed(delete), [path])
                self.apply(delete, files)
                self.assertEqual(files, {})
                add = f"*** Begin Patch\n*** Add File: {path}\n+value\n*** End Patch"
                self.assertEqual(identify_files_added(add), [path])
                self.apply(add, files)
                self.assertEqual(files, {path: "value"})

    def test_lf_crlf_and_cr_patches_preserve_blank_content_lines(self):
        for newline in ("\n", "\r\n", "\r"):
            for terminal_newline in (False, True):
                with self.subTest(newline=repr(newline), terminal_newline=terminal_newline):
                    lines = ["*** Begin Patch", "*** Add File: fixture.txt", "+first", "+", "+last", "*** End Patch"]
                    text = newline.join(lines) + (newline if terminal_newline else "")
                    files = {}
                    self.apply(text, files)
                    self.assertEqual(files, {"fixture.txt": "first\n\nlast"})

    def test_invalid_sentinels_and_duplicate_sections_still_fail(self):
        for text in ("", "*** Begin Patch\n+bad", "*** Begin Patch\n*** End Patch\n\n",
                     "*** Begin Patch\n*** Add File: a\n+x\n*** Add File: a\n+y\n*** End Patch"):
            with self.subTest(text=text):
                with self.assertRaises(DiffError):
                    self.apply(text, {})

    def test_real_text_file_roundtrip(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.txt"
            path.write_text("original\u2028value\n", encoding="utf-8")
            apply_patch(f"*** Begin Patch\n*** Update File: {path}\n@@\n-original\u2028value\n+new\u2028value\n*** End Patch")
            self.assertEqual(path.read_text(encoding="utf-8"), "new\u2028value\n")


if __name__ == "__main__":
    unittest.main()
