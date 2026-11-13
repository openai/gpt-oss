# BoolieLOT

![BoolieLOT: Florence in golden morning light](docs/images/boolielot.png)

## English

BoolieLOT is an experimental project that explores historical ideas and programming through the perspective of a classical scholar and engineer. It aims to discuss knowledge from 1899 and earlier in elegant Traditional Chinese, while using modern programming techniques to implement classical concepts, simulate machinery, and create games.

### Project status

- Reviewed notes record sources and knowledge dates, covering works such as *Tiangong Kaiwu*, Leonardo da Vinci’s notebooks, and Boole’s logic. A later edition’s publication date is distinct from the date of its original content.
- The chat interface supports conversation and code display; a model backend must be started and connected separately.
- The upgraded corpus has 38 training and 10 held-out examples: pre-1899 notes plus modern programming lessons in Python, C++20, TypeScript, JavaScript, HTML/CSS, SQL, Rust, JSON/regex and algorithms. All 24 curated code examples have executable or parser checks.
- The strict historical profile no longer exposes the actual runtime date to the model and rejects live web search, `--browser`, and `--raw` paths. Its chronology guard also blocks mixed requests that try to use a programming question to smuggle in post-1899 history. These are safeguards, not semantic proof of perfect isolation.
- The vintage CLI now defaults to the pre-1900 `vintage-llm-340m` checkpoint. It is a small base model, not an instruction-tuned assistant, so it is not yet a quality-approved conversational model.
- Candidate training now requires at least 1,000 reviewed historical examples, 1,000 programming examples, 100 held-out examples per domain, and 25 historical training sources. The current corpus is far below those gates; ordinary training is intentionally blocked. `--allow-small-dataset` is for pipeline experiments only.
- See the [two-domain model architecture](docs/1899_MODEL_ARCHITECTURE.md), [upgrade guide](docs/coding_model.md), and [training guide](docs/train_classical_model.md) for the intended specialist separation and limitations.
- A 10-step experimental run on Colab produced adapter weights. They are not included in this repository and require a compatible base model; no new weights were trained as part of this upgrade.

The project uses a modern pretrained base model. Fine-tuning, prompts, retrieval and output filtering cannot guarantee removal of knowledge from after 1899; this is a domain-separated behaviour boundary, not machine unlearning. Historical reliability, classical style, and programming ability still need independent evaluation with real weights.

### Getting started

- [Experiment with GPU training on Colab](https://colab.research.google.com/github/andrewlai1209-byte/BooleLOT/blob/arena/3bac6069-boolelot/training/colab_classical.ipynb)
- [Modern coding + 1899 boundary upgrade guide](docs/coding_model.md) (Chinese, with English summary)
- [Training, loading, and evaluation guide](docs/train_classical_model.md) (Chinese)
- [Reviewed notes and sources](docs/classical_sources.json) (Chinese)
- [Reviewed programming cards and lessons](gpt_oss/data/coding_sources.json)

Local curriculum checks need no GPU:

```bash
python -m training.prepare_corpus
python -m training.verify_coding_examples
python -m training.train_classical --dry-run
```

Free GPU availability and memory are not guaranteed to be sufficient. Experimental training does not automatically deploy the chat service; the base model and adapter must be prepared separately.

---

## 中文

BoolieLOT 是一個以古代學者與工程家的視角探索歷史思想和程式設計的實驗專案。它嘗試用典雅的繁體中文討論 1899 年以前的學問，並運用現代程式技術實作古典概念、模擬機械和製作遊戲。

## 專案現況

- 古典札記以文獻出處和知識年代記錄，涵蓋《天工開物》、達文西筆記與布爾邏輯等；後世版本的出版日期不等同於原文的年代。
- 聊天介面支援對話及程式碼展示；使用時須另行啟動並連接模型後端。
- 升級後語料有 38 則習卷、10 則未見考卷：除 1899 年前札記外，另有 Python、C++20、TypeScript、JavaScript、HTML/CSS、SQL、Rust、JSON／正則式與演算法之現代程式課；24 個校訂程式例皆可執行、編譯或解析查驗。
- 嚴格歷史模式不再向模型提供真實執行日期，並拒絕即時網頁搜尋、`--browser` 與 `--raw`；年代防火牆亦會攔截混合程式請求中夾帶的年界後史事。此等措施是防護，不代表已證明完全隔離。
- Vintage CLI 預設改用 1899 年前的 `vintage-llm-340m`；它是小型基礎模型，並非指令微調助理，尚未通過對話品質驗收。
- 正式候選訓練至少須有 1,000 則史事訓練例、1,000 則程式訓練例、每域 100 則未見考題，以及 25 個史事訓練來源。現有語料遠未達標，故一般訓練會停止；`--allow-small-dataset` 僅供管線試驗。
- 詳見[雙域模型架構](docs/1899_MODEL_ARCHITECTURE.md)、[升級說明](docs/coding_model.md)及[訓練驗收說明](docs/train_classical_model.md)。
- Colab 曾完成一次 10 步試訓並產生附加權重；權重未包含在此儲存庫中，必須搭配相容的底座使用。本次升級未另訓練權重。

此專案使用現代預訓練底座。微調、提示、檢索與輸出過濾都不能保證移除底座中 1899 年之後的知識；此為分域的行為邊界，不是機器遺忘。模型的歷史可靠性、文言風格與程式能力仍須以真實權重獨立測試。

## 開始使用

- [以 Colab GPU 試訓](https://colab.research.google.com/github/andrewlai1209-byte/BooleLOT/blob/arena/3bac6069-boolelot/training/colab_classical.ipynb)
- [現代程式與 1899 年界升級說明](docs/coding_model.md)
- [訓練、載入與驗收說明](docs/train_classical_model.md)
- [已審核札記與出處](docs/classical_sources.json)
- [已審核程式準繩與課題](gpt_oss/data/coding_sources.json)

無 GPU 亦可先執行上方三行，重建語料、查驗程式例並檢查訓練前之卷目。

免費 GPU 的可用時間與記憶體不保證足夠。試訓不會自動部署聊天服務；模型底座及附加權重需另行準備。

## Vintage model research integrations

Optional research adapters now live under `gpt_oss/vintage/`. They provide a Talkie inference adapter, an experimental Transformers adapter for the small pre-1900 Vintage-LLM checkpoint, HistLM BERT retrieval, and a source-safe historical corpus preparation utility. These are optional and do not replace the existing BooleLOT chronology guard or the current chat service.

See [Vintage upstream integration](docs/VINTAGE_UPSTREAM_INTEGRATION.md) for installation, model distinctions, data formats, and limitations. No model weights are stored in Git, and no new model has been trained by adding these adapters.

