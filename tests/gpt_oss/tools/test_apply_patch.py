from gpt_oss.tools.apply_patch import apply_patch


def _apply(original: str, patch: str) -> str:
    written = {}
    apply_patch(
        patch,
        open_fn=lambda _path: original,
        write_fn=lambda path, content: written.__setitem__(path, content),
        remove_fn=lambda _path: None,
    )
    return written["example.py"]


def test_add_only_chunk_uses_preceding_stripped_context_indentation() -> None:
    original = 'def greet():\n    print("Hi")'
    patch = """*** Begin Patch
*** Update File: example.py
@@
 print("Hi")
+print("Bye")
*** End Patch"""

    assert _apply(original, patch) == (
        'def greet():\n    print("Hi")\n    print("Bye")'
    )


def test_add_only_chunk_uses_following_stripped_context_indentation() -> None:
    original = 'def greet():\n    print("Hi")'
    patch = """*** Begin Patch
*** Update File: example.py
@@
+print("Before")
 print("Hi")
*** End Patch"""

    assert _apply(original, patch) == (
        'def greet():\n    print("Before")\n    print("Hi")'
    )
