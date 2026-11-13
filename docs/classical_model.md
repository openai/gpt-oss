# 古典學者模型設定

本專案的命令列對話與 Responses API 現在預設採古典角色，知識年界暫定為
1899 年末（包含十九世紀）。可改為 1799 年以排除十九世紀。
角色出生於 1800 年前，與可採知識的截止年份分開設定。嚴格模式把模型上下文中的對話日期固定為年界末日，不向模型提供真實執行日期；即使知識年界設為 1899，也不代表底座權重已忘記其後世事。

這是推論時的行為設定、文獻檢索與輸出邊界，**尚未產生或修改模型權重，也不能保證模型忘記後世知識**。
`with_knowledge_cutoff` 是對話標記，不會改變模型的訓練資料。
現已加入分工明確的現代程式教材、準繩檢索與年代防火牆；自動測試驗證資料、提示及串流輸出之攔截，
但尚未以真實權重檢驗文風及年代遵循率。

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

若要先看聊天介面，無須下載模型，可執行：

```bash
cd /workspace/BooleLOT
/workspace/.venv/bin/python -m gpt_oss.responses_api.chat_page
```

開啟 `http://127.0.0.1:8000/`。此模式會標明「介面預覽」，呈問按鈕待模型載入後才啟用。
欲開始真實問答，停止此預覽服務，改用下方的模型服務啟動指令。
若程式運行於遠端主機，請使用該主機的連接埠轉送網址；自己電腦的 localhost 不會指向遠端主機。

啟動下列服務後，在瀏覽器開啟 `http://127.0.0.1:8000/`，即可使用「格物書齋」聊天框。
頁面支援連續問答、程式碼分段顯示、範例提問與清空對話；重新整理會清除本頁對話。
每次問答會帶上本頁先前的往來文字。模型仍須依後端的安裝方式載入；此頁沒有內建模型權重。
聊天頁與 API 由同一個服務提供，無須另裝前端依賴。

從 `/workspace/BooleLOT` 執行：

```bash
/workspace/.venv/bin/python -m gpt_oss.prepare_classical \
  /workspace/BooleLOT/docs/classical_sources.json \
  /workspace/BooleLOT/data/classical/siku.sqlite

/workspace/.venv/bin/python -m gpt_oss.responses_api.serve \
  --checkpoint /path/to/model --inference-backend transformers \
  --knowledge-cutoff-year 1899 \
  --classical-library /workspace/BooleLOT/data/classical/siku.sqlite \
  --coding-library /workspace/BooleLOT/gpt_oss/data/coding_sources.json
```

`--coding-library` 可省略；服務與交談程式預設載入隨套件提供的已校程式準繩。

互動對話同樣支援這兩個選項：

```bash
/workspace/.venv/bin/python -m gpt_oss.chat /path/to/model \
  --backend torch --knowledge-cutoff-year 1799 \
  --classical-library /workspace/BooleLOT/data/classical/siku.sqlite
```

程式語法、標準函式庫、測試與執行方法是明示特例；說明、註釋、自訂命名依古典觀念。
外部介面、保留字、路徑保持正確。現代城市暫限使用者當次提供且附時日的資料。
嚴格年界模式拒絕即時網頁搜尋，CLI 亦拒絕 --browser 與 --raw；否則外部資料或未經篩選的 token 輸出可能越界。Python 程式執行仍可按需開啟，但不得用以取得或回答年界後的世事。
若要求嚴格資料隔離，勿啟用未經整理的外部資料工具。

## 現代程式與年代防火牆

本年度升級把「學問」分為兩域：史事、人物、世事與自然學說仍以 1899 年十二月三十一日為界；
Python、C++、TypeScript、JavaScript、HTML/CSS、SQL、Rust、JSON、正則式與命令列之語法、
標準函式庫、型別、演算法、測試、除錯及執行方法，則屬明示的程式特例。
此特例不得旁及語言創製年代、人物、機構、版本沿革、戰事政事或今世城邑現況。
詳見 [程式與年代升級說明](coding_model.md)。

`gpt_oss/data/coding_sources.json` 收十二則校訂準繩與四十則問答課；
推論時僅按關鍵詞附上相關準繩，並明標為技術參考而非書證或命令。
`ChronologyGuard` 對明顯越界之請求直接回覆不知，且在非串流與串流路徑皆先緩衝正文、
再查驗完整正文；程式碼區塊中的年份數值不當作史事。每次答覆附 `metadata.chronology_guard`
與 `metadata.chronology_scope`，便於稽核。防火牆以可審詞表與日期式為本，
不是語義理解，也不是機器遺忘：未列之後世事件、工具回傳內容、`--raw` 診斷模式及底座原有知識
仍可能越界；若要求嚴格資料隔離，勿啟用未經整理的外部工具，且須另以受限語料從頭訓練。

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
另可以 `python -m gpt_oss.evals.coding_scope_eval` 對已啟動之模型服務查程式能力與年代拒答，
並以 `python -m training.verify_coding_examples` 驗校訂教材本身；兩者皆須保留原始回答供人工審核。
