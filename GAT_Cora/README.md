# 图注意力网络（GAT）节点分类 - Cora 数据集

## 项目简介
使用 PyTorch Geometric 实现 GAT（图注意力网络），在 Cora 论文引用数据集上进行节点分类。采用多头注意力机制，为不同邻居分配动态权重。

## 数据集
- 节点数：2708（每篇论文）
- 边数：10556（引用关系）
- 节点特征维度：1433（词袋模型）
- 类别数：7（论文研究领域）

## 模型结构
- GATConv(1433 → 8, heads=8) + ELU + Dropout(0.6)
- GATConv(64 → 7, heads=1) + Dropout(0.6)

## 运行方法
```bash
python gat_cora.py
