#!/bin/bash
# ============================================================
# ending.sh — 專案結束與自動提交腳本
# Antigravity × Openspec 整合專案
# ============================================================

echo "=================================================="
echo " 🏁 Q-Learning & SARSA 專案收尾流程啟動..."
echo "=================================================="

# 1. Openspec validate — 驗證所有任務是否符合規範
echo "[1/4] openspec validate..."
openspec validate
if [ $? -ne 0 ]; then
    echo "⚠️  Openspec 驗證有警告，請檢查 tasks.md 後再繼續。"
fi

# 2. 顯示 Git 狀態 (查看哪些檔案有變動)
echo ""
echo "[2/4] 目前 Git 狀態："
git status

# 3. 自動 add + commit + push
echo ""
echo "[3/4] 自動提交所有變更到 GitHub..."
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
git add .
git commit -m "chore: project results & report — ${TIMESTAMP}"
git push origin main
if [ $? -ne 0 ]; then
    echo "❌ Push 失敗，請確認遠端倉庫權限。"
    exit 1
fi

# 4. 提醒手動上傳 ai_record.md
echo ""
echo "[4/4] ✅ 程式碼已成功推送至 GitHub！"
echo ""
echo "=================================================="
echo "📋 交付提醒："
echo "   ✅ 程式碼、圖表 → 已推送至 GitHub"
echo "   📁 ai_record.md → 請手動複製後上傳至繳交系統"
echo "=================================================="
