# Design: Cliff Walking 技術設計文件

## 架構設計

```
Q-learning&SARSA/
├── cliff_walking_env.py   # 4x12 網格環境
├── agents.py              # QLearningAgent + SarsaAgent (PyTorch CUDA Tensor)
├── main.py                # 訓練主程式 + 可視化輸出 + 崩潰防呆
├── Graph.png              # 學習曲線輸出
├── style.png              # 策略地圖輸出
├── Report.md              # 分析報告
├── startup.sh             # 啟動腳本
└── ending.sh              # 收尾腳本
```

## 技術選型
- **語言**：Python 3.8 (conda py3.8)
- **Q-table 儲存**：PyTorch Tensor on CUDA (若可用)，降低後續擴展 DRL 的成本
- **視覺化**：matplotlib + seaborn（深色主題，符合 Graph/style 參考圖風格）

## 崩潰防呆設計
- main.py 以 try-except 包裝全程
- 任何 crash 立即輸出：執行指令、錯誤類型、完整 Traceback
