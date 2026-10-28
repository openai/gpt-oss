# 古典學者權重：訓練與驗收

本分支新增了訓練管線，**目前尚未產生模型權重**。當前工作環境沒有 GPU、模型權重及訓練依賴，只有約 9.7 GiB 記憶體。訓練必須在可使用 CUDA GPU 的主機上進行；所需記憶體取決於硬體與訓練配置，請先參考專案現有的 `examples/reinforcement-fine-tuning.ipynb`。

管線以 `unsloth/gpt-oss-20b` 作底座，產生 LoRA 附加權重；此附加權重不是可獨立執行的完整模型。底座與附加權重必須一起載入。底座本身是現代預訓練模型，微調及提示**不能保證移除 1899 年後的知識**。若要求訓練資料與知識嚴格限於該年代，必須從已審核的資料重新訓練並進行大量驗證，此流程尚不提供這項保證。

## 建立資料與檢查

```bash
cd /workspace/BooleLOT
python -m training.prepare_corpus
python -m training.train_classical --dry-run
```

只會採用 `docs/classical_sources.json` 中已核對的古典札記，並依書源分開習卷與考卷。現有八則札記僅形成六則訓練資料、兩則驗證資料，遠不足以訓練可靠模型，尤其沒有充分的程式能力訓練材料。《授時曆故》與《世界通史》仍須校讀，不能將其錯亂 OCR 加入訓練。`data/classical/training/manifest.json` 記錄來源與摘要。

正式訓練前應擴充具出處的古典問答、文言說明與可執行程式範例；範例應涵蓋要求完整遊戲、除錯、邊界條件、不能確認的歷史資訊，以及古代學說與現代程式技術的區分。以獨立書源與未見過的編程題驗收，不能以訓練用例充當能力測試。

## GPU 訓練

在 GPU 環境先依現有 `examples/reinforcement-fine-tuning.ipynb` 的 Unsloth、PyTorch、Transformers、TRL 與 Triton 相容版本安裝依賴，再執行：

```bash
python -m training.train_classical --data data/classical/training --output data/classical/model-adapter
```

預設資料少於 32 則會停止，以免將微小樣本誤認為可用模型。僅為檢查管線時可加 `--allow-small-dataset`；此時產生的權重不是經驗證的成品。程式拒絕覆寫非空輸出目錄、要求訓練與驗證書源不同，並在模型對話格式與預期不符時停止。此管線尚未在 GPU 上執行；實際版本相容性、記憶體需求和訓練結果仍待驗證。

## 載入與驗收

```bash
python -m gpt_oss.responses_api.serve \
  --inference-backend transformers --checkpoint /path/to/compatible/base-model \
  --adapter /path/to/model-adapter --knowledge-cutoff-year 1899 \
  --classical-library data/classical/siku.sqlite
```

`--adapter` 明確只支援 Transformers 後端；其他後端會報錯而不會靜默忽略。需安裝 `.[adapter]`，並使用與訓練時相容的底座。原生 MXFP4 與 Unsloth LoRA 的載入相容性需在目標 GPU 環境實測；本工作環境尚無法驗證。

使用 `python -m gpt_oss.evals.classical_game_eval --repair` 產生九宮移數遊戲，另以未見過的題目檢查文言是否自然、知識是否有據、遊戲能否開局、移動、判勝與重開。必須記錄模型原始回答並實際試玩。沒有完成這些驗證前，不應宣稱模型能可靠編程或只知道 1899 年前的事。

權重與訓練資料位於被 Git 忽略的 `data/classical/`，不會自動推送至 GitHub；若日後要將大型權重放到儲存庫，須另外選定支援大型檔案的儲存方式。

## Colab 免費 GPU 試訓

開啟 [Colab 筆記本](https://colab.research.google.com/github/andrewlai1209-byte/BooleLOT/blob/work/training/colab_classical.ipynb)，選擇 GPU 執行階段並依序執行。筆記本會下載 `work` 分支與底座、試訓 10 步，完成後下載 LoRA 附加權重。免費 GPU 資源可能不足或中斷，且尚未實測此流程；訓練不會自動部署到聊天網站。現有六筆資料只供流程試驗。

## 年代限制與審核

訓練程式會依目前的 `docs/classical_sources.json` 重建習卷和考卷，並逐位元比對輸入；不符即停止，避免直接混入未經此流程審核的資料。札記的 `knowledge_year` 必須是 1 至 1899 的整數，且應代表所引用內容的年代，而非數位版本出版年代。修改札記後須重新編卷。年代欄位是人工判定，不是自動考證；訓練前仍須校對每則內容和出處。系統指示包含現代程式能力的例外，因此這並非所有訓練文字都出自 1899 年前。現代底座的預訓練知識亦不會因此被清除；先前訓練的附加權重也不會因程式變更而自動更新。
