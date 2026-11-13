# 古典學者權重：訓練與驗收

本分支的訓練管線已加入現代程式教材與年代分域，**目前仍未產生新的模型權重**。升級工作環境沒有 GPU、底座權重及訓練依賴；訓練必須在可使用 CUDA GPU 的主機上進行，所需記憶體取決於硬體與配置，請先參考 `examples/reinforcement-fine-tuning.ipynb`。設計與限制詳見 [coding_model.md](coding_model.md)。

管線以 `unsloth/gpt-oss-20b` 作底座，產生 LoRA 附加權重；此附加權重不是可獨立執行的完整模型。該底座已接受現代預訓練，故本法不可能證明它沒有 1899 年後的知識。底座與附加權重必須一起載入。底座本身是現代預訓練模型，微調及提示**不能保證移除 1899 年後的知識**。若要求訓練資料與知識嚴格限於該年代，必須從已審核的資料重新訓練並進行大量驗證，此流程尚不提供這項保證。

## 建立資料與檢查

```bash
cd /workspace/BooleLOT
pip install '.[coding]'          # 可選：TypeScript / Rust 語法樹查驗
python -m training.prepare_corpus
python -m training.verify_coding_examples
python -m training.train_classical --dry-run
```

`prepare_corpus` 會讀兩份校訂目錄：`docs/classical_sources.json` 的八則史事札記，
以及 `gpt_oss/data/coding_sources.json` 的三十二則程式習題與八則未見程式考題。
合計 38 則習卷、10 則考卷，其中史事 6／2、程式 32／8；依書源或題源分卷，同一來源不兼入兩卷。這是流程示範量，遠不足以訓練合格候選模型。
史事列 `knowledge_year`，程式列 `knowledge_domain: programming` 且年代為 `null`，
以免把現代語法冒充古籍。`manifest.json` 記錄兩份目錄與 24 個程式例的 SHA-256。

`verify_coding_examples` 會執行 Python、編譯並執行 C++、檢查並執行 JavaScript、
執行 SQL、解析 HTML/CSS 與 Node 腳本；TypeScript 以 `tsc --strict` 或 tree-sitter
解析，Rust 以 `rustc` 或 tree-sitter 解析。缺工具記為 `skipped`，不冒充通過；
本次工作環境 24 例全數通過。報告存於 `data/classical/coding-examples-report.json`。
《授時曆故》與《世界通史》仍須校讀，錯亂 OCR 不入卷。

正式訓練前仍應擴充具出處的古典問答、文言說明與可執行程式範例；範例應涵蓋完整遊戲、
除錯、邊界條件、不能確認的歷史資訊，以及古代學說與現代程式技術的區分。
以獨立書源與未見過的編程題驗收，不能以訓練用例充當能力測試。

## GPU 訓練

在 GPU 環境先依現有 `examples/reinforcement-fine-tuning.ipynb` 的 Unsloth、PyTorch、Transformers、TRL 與 Triton 相容版本安裝依賴，再執行：

```bash
python -m training.train_classical --data data/classical/training --output data/classical/model-adapter
```

候選模型另設資料量門檻：至少 1,000 則史事訓練例、1,000 則程式訓練例、各 100 則互相獨立的史事與程式考題，且史事訓練材料須來自至少 25 個來源。現有 38／10 則語料遠未達標；`--dry-run` 會明示不足，正式訓練會停止。僅為檢查 GPU 與程式管線時可加 `--allow-small-dataset`；此時產生的附加權重只屬實驗樣本，不是候選成品。程式會逐位元比對 `train.jsonl`、`eval.jsonl` 與 `manifest.json`，核對程式例雜湊，拒絕覆寫非空輸出目錄、要求訓練與驗證來源不同，並在模型對話格式與預期不符時停止。預設步數提高為 2,000；LoRA rank 為 16，並增加分步評估，但增加步數不能彌補資料不足。此管線尚未在 GPU 上實測；版本相容性、記憶體需求和訓練結果仍待驗證。

## 載入與驗收

```bash
python -m gpt_oss.responses_api.serve \
  --inference-backend transformers --checkpoint /path/to/compatible/base-model \
  --adapter /path/to/model-adapter --knowledge-cutoff-year 1899 \
  --classical-library data/classical/siku.sqlite
```

`--adapter` 明確只支援 Transformers 後端；其他後端會報錯而不會靜默忽略。需安裝 `.[adapter]`，並使用與訓練時相容的底座。原生 MXFP4 與 Unsloth LoRA 的載入相容性需在目標 GPU 環境實測；本工作環境尚無法驗證。

模型服務啟動後，可執行：

```bash
python -m gpt_oss.evals.coding_scope_eval \
  --endpoint http://127.0.0.1:8000/v1/responses --model gpt-oss-20b
python -m gpt_oss.evals.classical_game_eval --repair
```

前者查現代語言標記、Python 語法、年代拒答與年界內題目；後者產生九宮移數遊戲。
另須以未見過的題目檢查文言是否自然、知識是否有據、遊戲能否開局、移動、判勝與重開，
並記錄模型原始回答、實際試玩與人工審核。自動檢查不執行模型生成的程式，
也不判定史事真偽；沒有完成這些驗證前，不應宣稱模型能可靠編程或只知道 1899 年前的事。

權重與訓練資料位於被 Git 忽略的 `data/classical/`，不會自動推送至 GitHub；若日後要將大型權重放到儲存庫，須另外選定支援大型檔案的儲存方式。

## Colab 免費 GPU 試訓

開啟 [Colab 筆記本](https://colab.research.google.com/github/andrewlai1209-byte/BooleLOT/blob/arena/3bac6069-boolelot/training/colab_classical.ipynb)，選擇含本次更新且已推送的分支、GPU 執行階段並依序執行。筆記本會下載底座、查驗 38／10 則語料，試訓 10 步後下載 LoRA 附加權重。免費 GPU 資源可能不足或中斷，且尚未實測此流程；訓練不會自動部署到聊天網站。十步試訓只供流程試驗，不是能力驗收。

## 年代限制與審核

訓練程式會依目前兩份校訂目錄重建習卷、考卷與清單，並逐位元比對輸入；不符即停止，避免混入未審資料。史事札記的 `knowledge_year` 必須是 1 至 1899 的整數，且應代表所引內容的年代，而非數位版本出版年代；程式課不得附年代，須明標 `cutoff_exception`。修改任一目錄或程式例後須重新編卷。年代欄位是人工判定，不是自動考證；訓練前仍須校對內容和出處。

系統指示包含現代程式能力的例外，因此並非所有訓練文字都出自 1899 年前；此例外只及語法與驗法，不及人物、史事、機構或後世學說。推論另有 `ChronologyGuard` 輸出閘門，但現代底座的預訓練知識不會因此被清除；先前訓練的附加權重也不會因程式變更而自動更新。若要保證底座從未見過後世資料，必須另以受限語料從頭訓練並大量驗收，本流程不提供該保證。
