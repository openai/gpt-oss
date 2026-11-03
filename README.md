# BoolieLOT

![BoolieLOT: Florence in golden morning light](docs/images/boolielot.png)

## English

BoolieLOT is an experimental project that explores historical ideas and programming through the perspective of a classical scholar and engineer. It aims to discuss knowledge from 1899 and earlier in elegant Traditional Chinese, while using modern programming techniques to implement classical concepts, simulate machinery, and create games.

### Project status

- Reviewed notes record sources and knowledge dates, covering works such as *Tiangong Kaiwu*, Leonardo da Vinci’s notebooks, and Boole’s logic. A later edition’s publication date is distinct from the date of its original content.
- The chat interface supports conversation and code display; a model backend must be started and connected separately.
- The upgraded corpus has 38 training and 10 held-out examples: pre-1899 notes plus modern programming lessons in Python, C++20, TypeScript, JavaScript, HTML/CSS, SQL, Rust, JSON/regex and algorithms. All 24 curated code examples have executable or parser checks.
- Runtime reference cards and a deterministic chronology firewall keep modern programming syntax explicit while refusing clearly post-1899 historical requests and screening streamed and non-streamed answers. See [the upgrade guide](docs/coding_model.md).
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
- 推論時附上校訂程式準繩，並以確定性年代防火牆拒答明確越界史事、查驗串流與非串流正文。詳見[升級說明](docs/coding_model.md)。
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
