# 图注意力网络（GAT）节点分类 - Cora 数据集

## 项目简介
使用 PyTorch Geometric 实现 GAT（图注意力网络），在 Cora 数据集上进行节点分类，并与 GCN 进行对比。

## 模型结构
- GATConv(1433 → 8, heads=8) + ELU + Dropout(0.6)
- GATConv(64 → 7, heads=1) + Dropout(0.6)

## 训练结果
- 测试准确率：**79.5%**
- 验证准确率：最高 78.6%
- 相比 GCN（77.7%）提升约 1.8%

## 技术点
- 图注意力机制（动态邻居权重）
- 多头注意力（8个头）
- Dropout 正则化
