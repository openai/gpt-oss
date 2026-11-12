# BooleLOT Vintage AI — Architecture Proposal

## Goal

Build a usable historical language-model application whose generator is actually trained on period-bounded material. A period-flavoured system prompt alone is not a vintage model and must never be presented as one.

## Research inputs and their proper roles

| Project | Reusable contribution | Correct integration boundary |
|---|---|---|
| croqaz/vintage-LLM | Pre-1900 corpus preparation, tokenizer/training recipes, 340M Llama-3-based checkpoint | Primary small-footprint 19th-century generator candidate; its base checkpoint has limited chat ability |
| Living-with-machines/histLM | Historical embeddings and time-sliced BERT models trained on 1760–1900 books | Historical retrieval, semantic search, diachronic comparison; not a chat generator |
| talkie-lm/talkie | Inference library and 1930-cutoff 13B base/instruction-tuned models | High-quality optional historical generation backend; 1930 cutoff is not identical to a strict 19th-century cutoff |
| entanglr/awesome-vintage-llms | Curated model, dataset, benchmark and research catalogue | Model registry and ongoing evaluation discovery; not a runtime dependency |

## Runtime design

- VintageConfig: explicit model, backend, historical window and generation settings.
- VintageEngine: lazy model loading and backend adapter boundary.
- prompts: behavioral guidance and explicit temporal-horizon disclosures.
- MODEL_PROFILES: transparent profiles distinguishing generation from retrieval.
- cli: planned command-line entry point.

The initial implementation avoids copying source-project code. It calls published model identifiers and the Talkie public API.

## Model-selection caveats

- croqaz/vintage-LLM-340m-v1-base is the closest listed small model to a pre-1900 target, but its model card describes it as a base model with limited chat capabilities. A real product needs instruction tuning and evaluation before claiming dependable dialogue.
- talkie-lm/talkie-1930-13b-it is instruction tuned and more capable, but represents a pre-1931 corpus rather than a strict 1800–1899 corpus. Its documented inference setup expects a CUDA GPU with at least 28 GB VRAM and roughly 26–50 GB disk per model.
- histLM BERT and embedding artifacts are encoders, not autoregressive text generators. They should power retrieval and historical evidence, not be wired into generate().

## Next engineering phases

1. Runtime correctness: add smoke tests against each real model, model-specific chat templates, streaming, batching, CPU/quantized inference and deterministic error handling.
2. Period-grounded retrieval: ingest licensed/public-domain sources with publication dates and provenance; retrieve passages only inside the selected historical window; cite retrieved source metadata in answers.
3. Instruction tuning: build a clean, deduplicated period-appropriate instruction corpus. Keep synthetic dialogues separately labelled from authentic historical texts.
4. Evaluation: test period vocabulary, temporal leakage, factuality, instruction following, hallucinated citations, uncertainty and modern-concept handling. Compare against a modern control model where possible.
5. Release discipline: record model and dataset licenses, exact revisions, data provenance, training configuration, benchmark results, hardware requirements and known limitations.

## License and provenance

Do not assume all assets share one license. The source repositories report MIT for croqaz code, MIT for histLM code with released model weights under CC BY 4.0, and Apache-2.0 for Talkie. Verify each model card, dataset card, individual artifact and current upstream license before redistribution. The awesome list is a discovery index, not a blanket license grant.

## Current limitations

This package is an integration foundation, not a newly trained or fully validated model. Prompt framing cannot erase modern knowledge already present in model weights. A production release must not claim a perfectly sealed historical knowledge boundary without dedicated temporal-leakage tests.
