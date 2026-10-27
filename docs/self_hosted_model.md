# 自行運行古典學者

這個專案提供 gpt-oss 推論程式、古典角色提示及文獻札記；目前並未產生經訓練或微調的獨立模型權重。使用下列方式時，實際回答來自你自己運行的 gpt-oss，套用專案的古典設定，而非呼叫 OpenAI 託管的模型 API。

需先準備 gpt-oss-20b 模型權重、相容的推論環境及足夠記憶體。官方參考配置約需 16 GB 記憶體；依後端及環境會有所不同。本工作環境沒有發現權重或 GPU，且只有約 9.7 GiB 記憶體，因此無法在此直接載入並驗證模型。

在具備模型及 Transformers 後端的主機上，從專案根目錄執行：

```bash
python -m gpt_oss.prepare_classical docs/classical_sources.json data/classical/siku.sqlite
python -m gpt_oss.responses_api.serve \
  --checkpoint /path/to/gpt-oss-20b \
  --inference-backend transformers \
  --knowledge-cutoff-year 1899 \
  --classical-library data/classical/siku.sqlite
```

該主機的瀏覽器可開啟 `http://127.0.0.1:8000/` 使用聊天頁。安裝及其他推論後端請參考主 README；需先準備模型及相應依賴。首次啟動可能需要 Harmony 詞彙檔，無法取得時需先解決其來源，不可用固定假回覆代替模型。

私人 Sites 網站需要額外綁定私人 HTTP 連線，別名為 `model`，將網站的 `CUSTOMER_HTTP_MODEL` 指向上述模型服務。該服務必須能在該連線內被安全存取，且提供 `/api/chat-config` 與 `/v1/responses`。未設定連線時網站會顯示「模型服務尚未連線」，不會呼叫 OpenAI 模型 API。不要直接將未經保護的模型服務公開至網際網路。

模型權重若要真正具備新的行為，需另建及驗收微調流程；現有提示與札記不會改寫模型既有知識，也不保證它忘記 1899 年後的知識。
