#!/bin/bash
# ============================================================
# startup.sh — 專案啟動腳本
# Antigravity × Openspec 整合專案
# ============================================================

echo "=================================================="
echo " 📦 Q-Learning & SARSA 專案啟動中..."
echo "=================================================="

# 1. 拉取最新遠端程式碼
echo "[1/3] Git pull..."
git pull origin main
if [ $? -ne 0 ]; then
    echo "❌ Git pull 失敗，請確認網路連線與遠端倉庫狀態。"
    exit 1
fi

# 2. 顯示目前 Openspec 變更清單
echo ""
echo "[2/3] Openspec 目前追蹤的變更："
openspec list

# 3. 提示工作交接配置
echo ""
echo "[3/3] 讀取目前任務進度..."
if [ -f "task.md" ]; then
    echo "--- task.md ---"
    cat task.md
else
    echo "（尚無 task.md）"
fi

echo ""
echo "=================================================="
echo "✅ 環境已就緒！請在 conda py3.8 環境下執行主程式："
echo "   conda activate py3.8"
echo "   python main.py"
echo "=================================================="
