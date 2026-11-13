# 格物書齋 API 與網站

## 本機 OpenAI 後端

在執行環境的 Secret 管理中設定 `OPENAI_API_KEY`，可另設 `OPENAI_MODEL`（預設 `gpt-4.1`）。
金鑰不應放進聊天訊息、前端或版本庫。

```bash
python -m gpt_oss.responses_api.openai_chat --classical-library data/classical/siku.sqlite
```

開啟 `http://127.0.0.1:8000/`。此後端不需本機模型或 Harmony 詞彙檔。
使用者提問與札記將送至 OpenAI Responses API，會依帳戶的 API 用量計費。
設定存在只代表可嘗試連線，並不證明憑證有效、模型可用或帳戶仍有額度。

## 託管網站

網站 ID 為 `appgprj_6aded7af578481919bca55592ebd8e57`，來源位於
`/workspace/sites/boolelot-shuzhai`，私人網站，尚待 Secret 設定與發布。
`build_site.py` 將聊天頁、角色設定、八則已核對札記編入 Worker；只輸出文字，不執行模型產生的伺服器程式。
網站保留古典角色及全部八則札記，本機後端則使用指定的 SQLite 檢索索引。

```bash
PYTHONPATH=. python web/build_site.py /workspace/sites/boolelot-shuzhai/worker/index.js
```

複製／下載程式可供 Python、JavaScript 等使用。HTML 試玩使用 `sandbox="allow-scripts"`，
並以內容政策禁止外部網路與表單提交；不授予同源存取權。其餘語言須在合適環境另行執行。
較長回覆可能仍遭模型輸出上限截斷，畫面會標記未完；不能宣稱任意程式皆正確。
真實 API 對話、古典文風與遊戲可玩性，必須在 Secret 設定後另行驗證。
