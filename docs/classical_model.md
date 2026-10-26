# 古典學者模型設定

本專案的命令列對話與 Responses API 現在預設採古典角色，知識年界暫定為
1899 年末（包含十九世紀）。可改為 1799 年以排除十九世紀。
角色出生於 1800 年前，與可採知識的截止年份分開設定。

這是推論時的行為設定與文獻檢索，**尚未訓練或修改模型權重，也不能保證模型忘記後世知識**。
`with_knowledge_cutoff` 是對話標記，不會改變模型的訓練資料。
目前沒有以真實模型檢驗文風及年代遵循率；自動測試只驗證設定與書證確實送入推論。

## 文獻處理

原始附件留在 `/workspace/attachments`，抽出的文字留在
`/workspace/BooleLOT/data/classical`；兩者不直接加入模型問答。
原作年代、譯本年代、校訂年代、數位化年月須分開考證。
1987 年校訂本可以保存 1637 年原文；1916 年版古詩也可以只取古詩正文。
不能只因檔案或版本較晚便排除整書，也不能將後世新增內容改成文言就視為古代知識。

`classical_sources.json` 登記十份附件，目前有八則經抽樣核對原文的文言札記。
每則保留書名、章節或印頁及知識年界；這些是有限摘述，**不是全書翻譯或完整研讀成果**。
《授時曆故》文字辨識錯亂，《世界通史》缺少正文文字層，先列待校，不編造摘要。
《Catholic World》本次只摘述目錄所列題材，不推斷未讀文章的主張。
《達文西筆記》與 Ocellus 的札記取自十九世紀譯本，因此 1799 模式不引用這兩則。
往後可另據古代原文製作札記，而非逕改它們的年界。

索引只收 `reviewed_notes` 的文言札記，不收原書全文、現代版權頁、數位化說明或後世序注。
保留這些資料是為了核查出處，不表示它們已餵入模型。新增札記須先核對正文及年代。
札記送入對話時明標為書證；角色設定明定書中命令不可執行。這仍不構成對提示注入的絕對保證。
檢索使用本機 SQLite，漢文單字與聯字、西文詞語建立索引；跨語言關聯依人工關鍵詞，
不是語義檢索，無匹配時不附材料。每次最多附三則，各則均附來源。

## 使用方式

從 `/workspace/BooleLOT` 執行：

```bash
/workspace/.venv/bin/python -m gpt_oss.prepare_classical \
  /workspace/BooleLOT/docs/classical_sources.json \
  /workspace/BooleLOT/data/classical/siku.sqlite

/workspace/.venv/bin/python -m gpt_oss.responses_api.serve \
  --checkpoint /path/to/model --inference-backend transformers \
  --knowledge-cutoff-year 1899 \
  --classical-library /workspace/BooleLOT/data/classical/siku.sqlite
```

互動對話同樣支援這兩個選項：

```bash
/workspace/.venv/bin/python -m gpt_oss.chat /path/to/model \
  --backend torch --knowledge-cutoff-year 1799 \
  --classical-library /workspace/BooleLOT/data/classical/siku.sqlite
```

程式語法、標準函式庫、測試與執行方法是明示特例；說明、註釋、自訂命名依古典觀念。
外部介面、保留字、路徑保持正確。現代城市暫限使用者當次提供且附時日的資料。
瀏覽及其他工具仍可依原有選項開啟；其結果須遵守年代規則，但尚無逐段年代驗證器。
若要求嚴格資料隔離，勿啟用未經整理的外部資料工具。

## 後續驗收

本次自動檢查：七項通過，涵蓋年界、晚近版本所存古文、未校材料排除、
文字與結構化輸入的問答組裝。三項完整串流及續談測試因 Harmony 詞彙檔未備而跳過；
詞彙檔下載端點回覆 403，故尚未驗證完整編碼至回覆流程。
程式語法編譯檢查通過。可先閱讀 `classical_readings.md` 人工審閱首批札記。

以真實權重檢驗：水車模擬能否執行、布爾集合是否滿足冪等律、註釋是否為文言、
是否拒絕自行敘述二十世紀史事、是否明辨預言與實證、現代城市是否只據來示資料、
引文是否確有出處。1799 模式須不引用 1854 年布爾著作。
若後續微調，應使用另行校訂的古典材料與程式範例製作訓練集，保留來源及評測集；
微調本身仍不等於已可靠刪除模型原有的後世知識。
