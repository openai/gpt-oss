# 現代程式能力與一八九九年界：升級說明

> English summary: BooleLOT now treats modern programming syntax as an explicit,
> narrow exception to its 1899 historical boundary. Curated lessons, runtime
> reference cards, a deterministic response firewall, and reproducible checks are
> included. This is **not machine unlearning** and no new model weights are
> produced here.

## 要旨

「具現代程式能力」與「不知 1899 年後世事」在邏輯上須分域，不能只靠一句提示。
本升級採用三層，並在嚴格模式下限制即時資料來源：

1. **教材分域**：史事札記保留 `knowledge_year <= 1899`；程式課標
   `knowledge_domain = "programming"`、`knowledge_year = null`，只收語法、
   標準函式庫、型別、演算法、測試、除錯與執行方法。語言創製年代、人物、
   機構、戰事政事與今世城邑現況皆不收。
2. **推論準繩**：`CodingLibrary` 只在程式問題時附上至多三則校訂參考卡，
   明標為「符術準繩」，不是古籍書證，也不是命令。`ClassicalCanon` 明示此特許
   不得旁推其他後世知識。
3. **輸出邊界**：`ChronologyGuard` 對明確越界的請求不呼叫底座；一般回覆則於
   非串流與串流路徑緩衝完整正文後查驗日期、紀元與後世事件詞，命中即以文言
   告「非余所當知」。程式碼區塊與行內碼中的年份數值視為資料，不當史事。

三層皆為可審、可測的軟體邊界。**現代預訓練底座原有的後世知識並未被刪去**；
微調、提示、檢索或輸出過濾都不能保證「遺忘」。本次工作環境沒有 GPU、底座權重
或新訓練出的 LoRA；若要求訓練資料與知識嚴格限於 1899 年以前，必須另以受限
語料從頭訓練並大量驗收，本儲存庫不提供該保證。

## 新增內容

| 檔案 | 功用 |
| --- | --- |
| `gpt_oss/data/coding_sources.json` | 12 則程式準繩、32 則習題、8 則未見考題；皆標為程式特例 |
| `training/coding_examples/` | 24 個可驗程式例，涵蓋 Python、C++20、TypeScript、JavaScript、HTML/CSS、SQL、Rust、JSON/正則式與演算法 |
| `gpt_oss/coding_library.py` | 確定性關鍵詞檢索；載入時即拒絕混入後世史事的卡片 |
| `gpt_oss/chronology.py` | 知識域判別、日期與事件詞查核、拒答文；提供 `ScopeVerdict` 供稽核 |
| `training/prepare_corpus.py` | 由史事與程式兩份目錄重建 38 筆習卷、10 筆考卷（史事 6／2、程式 32／8），記錄 SHA-256 與分卷來源；僅供流程開發 |
| `training/train_classical.py` | 逐位元核對已校語料，檢查程式例雜湊，然後才允許 GPU 微調 |
| `training/verify_coding_examples.py` | 執行／編譯／解析所有程式例，並檢查沒有後世日期或事件詞 |
| `gpt_oss/evals/coding_scope_eval.py` | 對已啟動的模型服務查現代語言、年代拒答與年界內問答，存原始回覆 |
| `tests/test_coding_upgrade.py` | 驗教材、準繩、請求閘門、串流輸出閘門與分卷隔離 |

程式例刻意避開年界後日期與具名事件；碼中數值仍可作資料。本次工作環境的 24 個程式例
皆通過執行、編譯或解析檢查，TypeScript 另以 `tsc --strict` 查驗。

## 使用

內建準繩會隨 `gpt_oss.responses_api.serve`、`gpt_oss.chat` 與
`gpt_oss.responses_api.openai_chat` 預設載入。亦可明示路徑：

```bash
python -m gpt_oss.prepare_classical \
  docs/classical_sources.json data/classical/siku.sqlite

python -m gpt_oss.responses_api.serve \
  --checkpoint /path/to/base-model --inference-backend transformers \
  --adapter /path/to/adapter \
  --knowledge-cutoff-year 1899 \
  --classical-library data/classical/siku.sqlite \
  --coding-library gpt_oss/data/coding_sources.json
```

`gpt_oss.chat` 亦接受 `--coding-library`。嚴格模式固定對話日期為所設年界末日，拒絕 `--raw`、`--browser` 與 Responses API 即時網頁搜尋。Python 程式執行仍可供驗證程式，但不得用作取得年界後世事的通道。
遠端代轉的 `openai_chat` 亦會在送出前擋明確越界請求、附上程式準繩，並於回傳前
查驗遠端正文。

## 可重複查驗

```bash
python -m training.prepare_corpus
python -m training.verify_coding_examples
python -m training.train_classical --dry-run
python -m pytest -q
```

`verify_coding_examples` 會以 Python 執行、C++ 編譯執行、Node 檢查或執行、
SQLite 執行 SQL、HTML 解析與腳本語法檢查等方式驗教材。安裝 `.[coding]` 可讓
TypeScript 與 Rust 在沒有 `tsc`/`rustc` 時仍由 tree-sitter 解析；沒有工具時報告
記為 `skipped`，決不冒充通過。完整型別檢查可另裝 TypeScript 編譯器。

模型服務啟動後可執行：

```bash
python -m gpt_oss.evals.coding_scope_eval \
  --endpoint http://127.0.0.1:8000/v1/responses \
  --model gpt-oss-20b
```

結果在 `data/classical/coding-scope-eval/`。自動檢查只驗語言標記、Python 語法、
年代標記與拒答；不執行模型生成的程式，也不判定文言優劣、史事真偽或遊戲可玩性。
請逐題填寫 `human_review`，並保留原始 JSON。

## 邊界與已知限制

- 防火牆以可審詞表、日期式與字串比對為本，不是完整語義理解；未收錄的後世事件、
  隱喻或外語轉寫可能越界。
- 嚴格模式封鎖即時瀏覽，但使用者仍可能自行貼入後世資料，底座也可能從原有權重回想未列入詞表的事件；詞表與日期規則不是完整語義理解。
- `metadata.chronology_scope.reasons` 可能含來客自己提供的年份；這是稽核資料，
  不是模型新知的證明。
- 使用者當次提供、附時日的「今世城邑」特例仍由 `ClassicalCanon` 規範；若答案中
  出現明確後世年份，防火牆可能攔截，須人工複核。
- 不啟用工具、不載入未審語料、不宣稱底座已遺忘，是本專案一貫的誠實界線。
- 候選訓練現設 1,000 則史事訓練例、1,000 則程式訓練例、各 100 則獨立考題及至少 25 個史事訓練來源門檻。現有 38／10 筆遠低於門檻；`--dry-run` 會顯示不足，正式訓練須停止。

## 增修教材

1. 在 `training/coding_examples/` 加一個可執行或可解析的小例，避免後世日期、
   人物、機構與事件詞。
2. 在 `gpt_oss/data/coding_sources.json` 的 `lessons` 增一則；考題須用新的
   `heldout-*` 題源與未見程式檔，且 `split: "eval"`。
3. 重跑 `prepare_corpus`、`verify_coding_examples`、`train_classical --dry-run` 與測試。
4. 若要加新語言，先擴充 `ALLOWED_LANGUAGES`、準繩卡與驗法，再入卷；不可只改提示。
5. 微調後另以未見題目、固定題目重複問答與人工試玩驗收，勿把教材本身當作能力證明。
