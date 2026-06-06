# 猫狗分类（Cat vs Dog）

## 文件说明
- `export_data.py`：从 CIFAR-10 导出猫狗图片
- `train.py`：训练 CNN 分类模型
- `requirements.txt`：依赖包列表

## 运行步骤
1. `pip install -r requirements.txt`
2. `python export_data.py`（首次运行）
3. `python train.py`

## 技术点
- 自定义 Dataset
- CNN 卷积神经网络
- 训练/测试集划分（80%/20%）
