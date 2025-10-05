import re

def build_harmony_messages(system_prompt: str, user_query: str, retrieved_chunks: list[dict]) -> list[dict]:
    """
    Build Harmony-style messages for OpenAI-compatible chat completion.
    Each chunk is cited as [n] in CONTEXT and mapped to its source.
    """
    context_lines = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_lines.append(f"[{i}] {chunk['text']}")
    context = "\n".join(context_lines)
    user_content = f"QUESTION: {user_query}\nCONTEXT:\n{context}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return messages

def validate_harmony_response(text: str) -> bool:
    """
    Minimal checks: non-empty, not a tool-call JSON.
    """
    if not text or not text.strip():
        return False
    # Disallow tool-call JSON (e.g., starts with '{' and contains "tool_call")
    if text.strip().startswith('{') and 'tool_call' in text:
        return False
    return True
