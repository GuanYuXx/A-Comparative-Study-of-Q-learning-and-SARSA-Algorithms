# AI 對話紀錄 — DRL 作業 2：Q-Learning vs SARSA

> 本文件紀錄 Antigravity AI 與使用者在完成本次強化學習作業過程中的完整對話紀錄。
> 對話時間：2026-04-21 ~ 2026-04-22
> 對話模型：Antigravity (Gemini / Claude Sonnet 4.6 Thinking)

---

## 對話紀錄

### [使用者]
進去conda 的py3.8 這個專案之後都億澳在這個環境解決

### [AI]
沒問題！確認系統中有 `py3.8` conda 環境，接下來所有操作都會在這個環境進行。

---

### [使用者]
`C:\Users\absol\MainProject\Q-learning&SARSA` 這是我這次的專案位置，之後產出的任何文件都要放這裡

### [AI]
收到！確認資料夾存在且有 `rule` 子資料夾，將以此為專案根目錄。

---

### [使用者]
你這裡可以看到我的git嗎

### [AI]
可以透過終端機指令查看 git 狀態，但發現當前資料夾還不是 git 儲存庫（執行 `git status` 返回 fatal: not a git repository）。詢問是否要初始化或連結現有的 GitHub 倉庫。

---

### [使用者]
是的沒錯，那你接到我的github了嗎

### [AI]
說明 AI 是在本地電腦系統上運作，透過 Git Credential Manager 可以操作 GitHub。發現帳號設定為 `absol`，詢問 GitHub 倉庫 URL。

---

### [使用者]
好了 `A-Comparative-Study-of-Q-learning-and-SARSA-Algorithms`

### [AI]
執行 `git init` 初始化本地倉庫，並嘗試連結 `https://github.com/absol/...`，但返回 repository not found（帳號不對），詢問正確的完整 URL。

---

### [使用者]
`https://github.com/GuanYuXx/A-Comparative-Study-of-Q-learning-and-SARSA-Algorithms.git`

### [AI]
執行：
```bash
git remote add origin https://github.com/GuanYuXx/A-Comparative-Study-of-Q-learning-and-SARSA-Algorithms.git
git branch -M main
git fetch origin
```
確認遠端為全新空倉庫，本地有 `rule/` 資料夾。

---

### [使用者]
先幫我做1.（將目前內容作為 Initial commit 推送）

### [AI]
執行：
```bash
git add .
git commit -m "Initial commit"
git push -u origin main
```
**結果：** 成功推送 3 個檔案（Graph.png, style.png, 強化學習作業說明.md）到 GitHub `main` 分支。

---

### [使用者]
這是一個來自DRL課程的作業2，幫我根據強化學習作業說明的MD檔來完成這份作業，附圖兩個是參考圖，並且希望會使用到openspec（附圖）。交付文件：程式碼上傳 GitHub + ai_record.md（本地）

### [AI]
閱讀 `rule/強化學習作業說明.md`，確認任務為 Cliff Walking + Q-Learning vs SARSA 比較，並整合 Openspec 流程。建立 Implementation Plan。

---

### [使用者]
因為Episodes要>=500回合的關係可能要使用到cuda cudnn等功能，這些也幫我安裝上去。另外會希望結果都要是可視化的圖如Graph、style這兩個。最後如果有崩潰的話停止該專案執行，並回報我你下的指令及崩潰原因

### [AI]
更新計畫，加入：
1. PyTorch + CUDA 安裝
2. matplotlib + seaborn 高質感可視化
3. try-except 崩潰防呆機制

---

### [使用者]
好的幫我撰寫

### [AI]

**Phase 1 — 套件安裝：**
```bash
# 指令 1: 安裝基礎套件
conda run -n py3.8 pip install numpy matplotlib seaborn
# 結果: 成功安裝 matplotlib-3.7.5, seaborn-0.13.2

# 指令 2: 安裝 PyTorch CUDA
conda install -n py3.8 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
# 結果: 成功安裝 pytorch-2.4.1 (CUDA 12.1, cuDNN 9)

# 指令 3: 確認 CUDA 狀態
conda run -n py3.8 python -c "import torch; print(torch.cuda.is_available())"
# 結果: True — NVIDIA GeForce RTX 4060 Laptop GPU
```

**Phase 2 — Openspec 整合：**
```bash
# 指令 4: 初始化 Openspec
openspec init ./ --tools antigravity
# 結果: 成功，建立 .agent/ 目錄，配置 4 skills + 4 commands

# 指令 5: 建立變更提案
openspec new change "cliff-walking-implementation"
# 結果: 成功建立 openspec/changes/cliff-walking-implementation/
```

**Phase 3 — 程式撰寫（建立以下檔案）：**
- `cliff_walking_env.py` — 4×12 懸崖行走環境
- `agents.py` — QLearningAgent + SarsaAgent (PyTorch CUDA Tensor)
- `main.py` — 主訓練腳本 (500 Episodes + 崩潰防呆 + 可視化)
- `startup.sh` — 專案啟動自動化腳本
- `ending.sh` — 結束與提交自動化腳本

**Phase 4 — 執行訓練：**
```bash
# 指令 6: 執行主程式（首次，編碼錯誤）
conda run -n py3.8 python main.py
# 崩潰原因: UnicodeEncodeError: 'cp950' codec 無法編碼 PyTorch Unicode 字元
# 修正方案: 直接使用 py3.8 的 Python 執行檔並設定 UTF-8 編碼

# 指令 7: 修正後執行
$env:PYTHONIOENCODING="utf-8"
& "C:\Users\absol\conda\envs\py3.8\python.exe" main.py
# 結果: 訓練成功，但有 emoji 字型警告（⚡ Glyph 9889）

# 指令 8: 修正 emoji 後再次執行（最終版）
# 結果: 完全成功 (exit code 0)
# Q-Learning Final 50-ep avg reward: -54.62
# SARSA     Final 50-ep avg reward: -27.08
# [✓] Saved Graph.png
# [✓] Saved style.png
```

**Phase 5 — 報告產出：**
- `Report.md` — 完整分析報告
- `ai_record.md` — 本對話紀錄（此文件）

---

### [使用者]
（設定：這裡能用conda install就用conda install 不行再用pip）

### [AI]
確認並調整策略，優先使用 conda install，fallback 才用 pip。

---

### [使用者]
圖表很快就上去了，y軸幫我取到 -500~0 應該會看得更清晰。另外我覺得 style 的可視化乍看之下無法明顯看出 Q-Learning 跟 SARSA 的差異，希望能改的顯著一點（如箭頭明顯一點等等）。最後要基於 style 圖上畫一個最佳路徑 from S to G。

### [AI]
確認三點修改需求：
1. **Graph.png** — Y 軸固定 -500 ~ 0（聚焦學習行為的範圍）
2. **style.png** — 重新設計：Q-Learning 用暖色系（YlOrRd）、SARSA 用冷色系（YlGnBu），以 `ax.annotate` 繪製較大的方向箭頭取代文字符號，視覺對比更顯著
3. **style.png** — 貪婪式 rollout 取得最佳路徑（S → G），以帶圓點的路徑線疊加在熱力圖上

執行指令：
```bash
$env:PYTHONIOENCODING="utf-8"
& "C:\Users\absol\conda\envs\py3.8\python.exe" main.py
# 結果: 完全成功 (exit code 0)
# Q-Learning Final 50-ep avg reward: -60.12
# SARSA     Final 50-ep avg reward: -25.48
# [✓] Saved Graph.png
# [✓] Saved style.png

git add main.py Graph.png style.png
git commit -m "feat: improved visualizations - Y-axis clip, distinct policy arrows, optimal path overlay"
git push origin main
# 結果: 成功推送 (91e5460..d79352d  main -> main)
```

Report.md 同步更新，新增策略地圖圖例說明表格與最佳路徑比較敘述。

---

## 最終交付清單（更新版）

| 文件 | 位置 | 說明 | 狀態 |
|------|------|------|------|
| cliff_walking_env.py | 專案根目錄 | 4×12 Cliff Walking 環境 | ✅ |
| agents.py | 專案根目錄 | Q-Learning + SARSA (PyTorch CUDA) | ✅ |
| main.py | 專案根目錄 | 主訓練腳本（崩潰防呆 + 可視化）| ✅ |
| startup.sh | 專案根目錄 | Openspec 啟動自動化腳本 | ✅ |
| ending.sh | 專案根目錄 | Openspec 結束自動化腳本 | ✅ |
| Graph.png | 專案根目錄 | Learning Curve（Y 軸 -500~0）| ✅ |
| style.png | 專案根目錄 | Policy Map + 最佳路徑疊加 | ✅ |
| Report.md | 專案根目錄 | 完整分析報告（含圖表說明）| ✅ |
| ai_record.md | 專案根目錄 | 本 AI 對話紀錄（供手動上傳）| ✅ |
| GitHub Push | GuanYuXx/A-Comparative-Study-... | 所有程式碼與圖表 | ✅ |
