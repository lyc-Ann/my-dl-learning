import torch
import torch.nn as nn
import torch.nn.functional as f
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.conv import GCNConv


# 加载数据
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]


print(f"节点数: {data.num_nodes}")
print(f"边数: {data.num_edges}")
print(f"节点特征维度: {data.num_node_features}")
print(f"类别数: {dataset.num_classes}")


# 定义GCN模型
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = f.relu(x)
        x = self.conv2(x, edge_index)
        return x


# 创建模型
model = GCN(
    input_dim=data.num_node_features,
    hidden_dim=16,  # 也可以32 64 数字越大，模型能记住的特征信息越多，容易过拟合
    output_dim=dataset.num_classes
)

# 定义损失函数和优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 训练模型
model.train()
for epoch in range(100):
    optimizer.zero_grad()  # 梯度清零
    out = model(data.x, data.edge_index)  # 前向传播
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 参数更新

    if epoch % 10 == 0:
        print(f"Epoch{epoch}, Loss{loss.item():.4f}")  # 每19轮打印一次损失


# 测试
model.eval()
pred = model(data.x, data.edge_index).argmax(dim=1)
acc = ((pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item())
print(f'测试准确率：{acc:.4f}')
