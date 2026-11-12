# BooleLOT: upstream vintage-LLM integration

This integration preserves BooleLOT's existing 1899 chronology guard, reviewed classical corpus, programming exception, game evaluation, and LoRA training workflow. The new modules are an optional research and inference layer; they do not replace the current GPT-OSS service or silently change its default backend.

## Upstream projects and integration boundaries

| Upstream | Integrated capability | Deliberately not copied |
|---|---|---|
| croqaz/vintage-LLM | Date-filtered, deduplicating, source-grouped corpus preparation; model profile for its 340M pre-1900 base checkpoint; pretraining pipeline guidance | Its large training scripts and separate toolchain are not imported wholesale. Port tested concepts rather than introducing a second incompatible training stack. |
| Living-with-machines/histLM | Lazy HistLM BERT semantic retriever supporting full-period and time-sliced checkpoint identifiers | BERT, Flair, fastText and word2vec are encoders/representations, not autoregressive dialogue generators. Weights are downloaded separately, never committed. |
| talkie-lm/talkie | Direct adapter using the public Talkie, Message and chat API; base and instruction-tuned profiles | Talkie's custom 13B architecture, tokenizer and weights remain in its own package and are not duplicated here. |
| entanglr/awesome-vintage-llms | Research catalogue and discovery source recorded in the registry documentation | It is a curated list, not an executable model library; there is no runtime code to merge. |

## Model roles

- talkie-1930-it: instruction-tuned generator trained on pre-1931 English text. Best-supported dialogue adapter here, but not a strict 1800-1899 model.
- talkie-1930-base: pre-1931 base model for completion, not chat.
- vintage-llm-340m: small pre-1900 base-model experiment. It is not instruction-tuned.
- histlm-bert-1760-1900 and histlm-bert-1890-1900: historical encoders for semantic retrieval, not generation.

The registry explicitly records task, time range and instruction-tuning status. A published period is not automatically a guarantee of zero temporal leakage.

## Runtime

Install optional dependencies with pip install -r requirements-vintage.txt.

Talkie is maintained as its own package:
1. Clone https://github.com/talkie-lm/talkie
2. Run uv sync in that repository.
3. Run python -m gpt_oss.vintage.cli --model talkie-1930-it.

The small base model can be tried with python -m gpt_oss.vintage.cli --model vintage-llm-340m, but its instruction following is experimental.

Talkie's documented bfloat16 setup requires CUDA with at least 28 GB VRAM and approximately 26-50 GB disk per model.

## HistLM retrieval corpus

Create a JSONL corpus with one passage per line and fields passage_id, text, year, source. Example:
{"passage_id":"book-001-p12","text":"A passage transcribed from a reviewed source.","year":1884,"source":"Book title, edition and page"}

Use the retriever:
from gpt_oss.vintage.retrieval import HistLMRetriever
retriever = HistLMRetriever(
    "data/historical_passages.jsonl",
    model_id="Livingwithmachines/bert_1890_1900",
    start_year=1890,
    end_year=1900,
)
for hit in retriever.search("steam engines and manufacture", top_k=5):
    print(hit.passage.year, hit.score, hit.passage.source, hit.passage.text)

The caller must retain source/year metadata when adding retrieved passages to prompts. Similarity is not a truth score. Do not ingest unreviewed OCR or modern commentary into a bounded corpus.

## Pretraining corpus preparation

The gpt_oss.vintage.corpus utility accepts JSONL rows with text, integer year, and source. It rejects missing metadata and post-cutoff dates, normalizes whitespace, removes exact normalized duplicates, and assigns entire sources to train or validation to reduce source leakage.

Run: python -m gpt_oss.vintage.corpus raw-books.jsonl --cutoff-year 1899 --output data/vintage-corpus

It emits train.jsonl, valid.jsonl, and a SHA-256 manifest. This is only data preparation. A from-scratch run still needs a period-specific tokenizer, tokenization/sharding, model configuration, pretraining, supervised conversation tuning, checkpoints, and held-out evaluation. See the upstream Vintage-LLM training guide for its end-to-end pipeline.

## Validation and limitations

- The existing BooleLOT chronology firewall remains the application-level output guard.
- Existing reviewed classical/programming corpora and game/coding evaluations remain intact.
- This integration does not download model weights, train new weights, or claim a prompt removes modern knowledge.
- HistLM requires a reviewed local passage corpus; it does not automatically ingest the upstream historical book archive.
- The small Vintage-LLM model and Talkie have different time windows and capabilities. Evaluate them separately.
- Verify each model-card, dataset and repository license before redistributing assets. The awesome list itself grants no rights to linked models.
