# 图神经网络（GCN）节点分类 - Cora 数据集

## 项目简介
使用 PyTorch Geometric 实现 GCN（图卷积网络），在 Cora 论文引用数据集上进行节点分类。

## 数据集
- 节点数：2708（每篇论文）
- 边数：10556（引用关系）
- 节点特征维度：1433（词袋模型）
- 类别数：7（论文研究领域）

## 模型结构
- GCNConv(1433 → 16) + ReLU
- GCNConv(16 → 7)

## 运行方法
```bash
python gcn_cora.py
