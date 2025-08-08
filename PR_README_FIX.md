# Fix README grammar & wording

## Summary
Refines README wording for clarity and consistency. No functional or code changes—documentation only.

## Changes
- Remove stray comma in model list sentence.
- Capitalize “Harmony” consistently when referring to the response format.
- Clarify LM Studio download sentence.
- Clarify MoE tensor storage description (blocks/scales wording, FP4 capitalization).
- Improve wording for tool training descriptions ("were trained on").
- Clarify activations precision recommendation (BF16 wording).
- Minor grammar/punctuation cleanups (e.g. inference backend lines, article usage).

## Rationale
Improves professionalism, readability, and consistency for new contributors and users evaluating the project.

## Testing
- Manual review of rendered Markdown.
- No code paths touched; `pytest` not re-run (not required for docs-only change).

## Backward Compatibility
Safe: Documentation-only.

## Checklist
- [x] Conventional commit message (`docs(readme): ...`).
- [x] Narrow scope (single file updated: README.md).
- [x] No generated files or build artifacts committed.
- [x] No behavioral or API changes.

## Notes
If any additional style adjustments are preferred (e.g. sentence case vs title case for headings), happy to follow up.
