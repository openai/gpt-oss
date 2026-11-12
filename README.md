# BooleLOT

**BooleLOT is an experimental vintage-language-model project.** Its goal is to create an AI whose language, knowledge, and conversational behavior are shaped by historically bounded texts—not merely to make a modern chatbot imitate Victorian prose.

## Current status

This repository now contains an initial inference integration layer under `boolelot_vintage/`. It supports a documented Talkie chat adapter and a generic Hugging Face Transformers causal-language-model adapter. It is an engineering foundation, not yet a newly trained or fully evaluated BooleLOT model.

## Quick start

Python 3.11+ is recommended for the Talkie backend.

```bash
# Install core inference dependencies
python -m pip install -r requirements-vintage.txt

# Install Talkie from its upstream repository (recommended backend)
git clone https://github.com/talkie-lm/talkie.git
cd talkie
uv sync
cd ..

# Run BooleLOT
python -m boolelot_vintage.cli
```

Talkie requires a CUDA GPU with at least 28 GB VRAM for its documented bfloat16 inference path, plus approximately 26–50 GB disk per model. Model weights download on first load. Set `BOOLELOT_MODEL` and `BOOLELOT_BACKEND` to select another model/backend.

For the experimental smaller Transformers model:

```bash
BOOLELOT_BACKEND=transformers BOOLELOT_MODEL=croqaz/vintage-LLM-340m-v1-base python -m boolelot_vintage.cli
```

The small vintage checkpoint is a base model, not a dependable instruction-tuned chatbot. Expect weaker instruction following.

## Research foundations

- [croqaz/vintage-LLM](https://github.com/croqaz/vintage-LLM): pre-1900 text preparation, vocabulary, training recipes, and small-model experiments. Its README describes the data → vocabulary → base training → conversational fine-tuning pipeline.
- [Living-with-machines/histLM](https://github.com/Living-with-machines/histLM): historical word embeddings and contextual models trained on English books published 1760–1900 (about 5.1B tokens). Use these for retrieval, semantic search, and diachronic analysis—not as autoregressive chat generators.
- [talkie-lm/talkie](https://github.com/talkie-lm/talkie): documented Python inference API and instruction-tuned `talkie-1930-13b-it` model, trained on pre-1931 English text.
- [entanglr/awesome-vintage-llms](https://github.com/entanglr/awesome-vintage-llms): discovery index for models, datasets, papers, and evaluation ideas.

See [docs/VINTAGE_AI_ARCHITECTURE.md](docs/VINTAGE_AI_ARCHITECTURE.md) for design rationale, limitations, and the development roadmap.

## Important historical limitation

A system prompt cannot remove modern knowledge from a model's weights. Talkie is trained on pre-1931 text, not strictly 1800–1899. The small vintage-LLM checkpoint is closer to the desired period but is not instruction tuned. BooleLOT must disclose these distinctions and must not claim a perfect historical knowledge boundary until temporal-leakage benchmarks support that claim.

## License and attribution

The source projects have different licenses and asset conditions. Check each upstream model card, dataset card, and license before redistributing weights or training data. See the architecture document for the current provenance notes.
