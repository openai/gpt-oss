import pytest

from gpt_oss.tools.apply_patch import (
    DiffError,
    apply_commit,
    apply_patch,
    patch_to_commit,
    text_to_patch,
)


def test_duplicate_move_destinations_raise_differror():
    orig = {"a.txt": "content a\n", "b.txt": "content b\n"}
    patch_text = """*** Begin Patch
*** Update File: a.txt
*** Move to: dest.txt
@@ -1,1 +1,1 @@
-content a
+updated a
*** Update File: b.txt
*** Move to: dest.txt
@@ -1,1 +1,1 @@
-content b
+updated b
*** End Patch"""

    with pytest.raises(
        DiffError, match=r"Duplicate destination path: dest\.txt"
    ):
        text_to_patch(patch_text, orig)


def test_update_and_move_destination_collision_raise_differror():
    orig = {"a.txt": "content a\n", "b.txt": "content b\n"}
    patch_text = """*** Begin Patch
*** Update File: a.txt
@@ -1,1 +1,1 @@
-content a
+updated a
*** Update File: b.txt
*** Move to: a.txt
@@ -1,1 +1,1 @@
-content b
+updated b
*** End Patch"""

    with pytest.raises(DiffError, match=r"Duplicate destination path: a\.txt"):
        text_to_patch(patch_text, orig)


def test_add_and_move_destination_collision_raise_differror():
    orig = {"b.txt": "content b\n"}
    patch_text = """*** Begin Patch
*** Add File: dest.txt
+new file content
*** Update File: b.txt
*** Move to: dest.txt
@@ -1,1 +1,1 @@
-content b
+updated b
*** End Patch"""

    with pytest.raises(
        DiffError, match=r"Duplicate destination path: dest\.txt"
    ):
        text_to_patch(patch_text, orig)


def test_self_move_preserves_file():
    orig = {"a.txt": "hello world"}
    patch_text = """*** Begin Patch
*** Update File: a.txt
*** Move to: a.txt
@@ -1,1 +1,1 @@
-hello world
+hello universe
*** End Patch"""

    written = {}
    deleted = []

    def write_fn(p: str, c: str) -> None:
        written[p] = c

    def remove_fn(p: str) -> None:
        written.pop(p, None)
        deleted.append(p)

    patch, _ = text_to_patch(patch_text, orig)
    commit = patch_to_commit(patch, orig)
    apply_commit(commit, write_fn, remove_fn)

    assert written == {"a.txt": "hello universe"}
    assert "a.txt" not in deleted


def test_delete_before_move_destination_order_independent():
    orig = {"old.txt": "old file\n", "src.txt": "source file\n"}
    patch_text = """*** Begin Patch
*** Update File: src.txt
*** Move to: old.txt
@@ -1,1 +1,1 @@
-source file
+source file updated
*** Delete File: old.txt
*** End Patch"""

    written = {}
    deleted = []

    def write_fn(p: str, c: str) -> None:
        written[p] = c

    def remove_fn(p: str) -> None:
        written.pop(p, None)
        deleted.append(p)

    patch, _ = text_to_patch(patch_text, orig)
    commit = patch_to_commit(patch, orig)
    apply_commit(commit, write_fn, remove_fn)

    assert written == {"old.txt": "source file updated\n"}
    assert "src.txt" in deleted


def test_chain_rename_preserves_intermediate_destination():
    orig = {"a.txt": "content A\n", "b.txt": "content B\n"}
    patch_text = """*** Begin Patch
*** Update File: a.txt
*** Move to: b.txt
@@ -1,1 +1,1 @@
-content A
+content A new
*** Update File: b.txt
*** Move to: c.txt
@@ -1,1 +1,1 @@
-content B
+content B new
*** End Patch"""

    written = {}
    deleted = []

    def write_fn(p: str, c: str) -> None:
        written[p] = c

    def remove_fn(p: str) -> None:
        written.pop(p, None)
        deleted.append(p)

    patch, _ = text_to_patch(patch_text, orig)
    commit = patch_to_commit(patch, orig)
    apply_commit(commit, write_fn, remove_fn)

    assert written == {"b.txt": "content A new\n", "c.txt": "content B new\n"}
    assert "a.txt" in deleted
    assert "b.txt" not in written.get("a.txt", "")


def test_apply_patch_valid_workflow():
    files = {
        "file1.txt": "line1\nline2\n",
        "file2.txt": "to be deleted\n",
    }

    def open_fn(p: str) -> str:
        return files[p]

    def write_fn(p: str, c: str) -> None:
        files[p] = c

    def remove_fn(p: str) -> None:
        files.pop(p, None)

    patch_text = """*** Begin Patch
*** Update File: file1.txt
*** Move to: file1_renamed.txt
@@ -1,2 +1,2 @@
 line1
-line2
+line2 modified
*** Delete File: file2.txt
*** Add File: file3.txt
+newly created file
*** End Patch"""

    result = apply_patch(
        patch_text, open_fn=open_fn, write_fn=write_fn, remove_fn=remove_fn
    )
    assert result == "Done!"
    assert "file1.txt" not in files
    assert "file2.txt" not in files
    assert files["file1_renamed.txt"] == "line1\nline2 modified\n"
    assert files["file3.txt"] == "newly created file"
