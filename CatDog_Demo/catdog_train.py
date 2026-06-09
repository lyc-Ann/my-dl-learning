"""
猫狗图像分类（Cat vs Dog）
从 CIFAR-10 提取猫狗图片，构建 CNN 进行二分类
包含自定义 Dataset、数据预处理、训练/测试划分、Dropout 防过拟合
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from PIL import Image


# ==================== 第1步：导出猫狗图片（从CIFAR-10）====================
def export_cat_dog_data():
    """从CIFAR-10导出猫和狗的图片到 ./train/cat 和 ./train/dog"""
    # 如果数据已存在，跳过导出
    if os.path.exists("./train/cat") and os.path.exists("./train/dog"):
        cat_count = len(os.listdir("./train/cat"))
        dog_count = len(os.listdir("./train/dog"))
        print(f"数据已存在: 猫 {cat_count} 张, 狗 {dog_count} 张")
        return cat_count, dog_count

    os.makedirs("./train/cat", exist_ok=True)
    os.makedirs("./train/dog", exist_ok=True)

    print("正在下载/加载CIFAR-10数据集...")
    cifar = datasets.CIFAR10(root="./cifar10", train=True, download=True)

    cat_count = 0
    dog_count = 0

    for idx, (img, label) in enumerate(cifar):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        if label == 3:
            img.save(f"./train/cat/{cat_count}.jpg")
            cat_count += 1
        elif label == 5:
            img.save(f"./train/dog/{dog_count}.jpg")
            dog_count += 1

        if (cat_count + dog_count) % 1000 == 0 and (cat_count + dog_count) > 0:
            print(f"已导出: {cat_count} 张猫, {dog_count} 张狗")

    print(f"导出完成！猫: {cat_count} 张, 狗: {dog_count} 张")
    return cat_count, dog_count


# ==================== 第2步：自定义Dataset ====================
class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for label, class_name in enumerate(['cat', 'dog']):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"警告: {class_dir} 不存在")
                continue
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)

        print(f"加载了 {len(self.images)} 张图片")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


# ==================== 第3步：定义CNN模型 ====================
class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # 30% 的神经元随机失活
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # 在全连接层后加 Dropout
        x = self.fc2(x)
        return x


# ==================== 第4步：训练函数 ====================
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}, Accuracy = {accuracy:.2f}%")

    return model


# ==================== 第5步：测试函数 ====================
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():    # 关闭梯度计算
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"测试准确率: {accuracy:.2f}%")
    return accuracy


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 50)
    print("第1步：导出猫狗图片")
    print("=" * 50)
    export_cat_dog_data()

    print("\n" + "=" * 50)
    print("第2步：加载数据")
    print("=" * 50)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    full_dataset = CatDogDataset(root_dir="./train", transform=transform)

    # 划分训练集和测试集
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    print(f"训练集: {train_size} 张, 测试集: {test_size} 张")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("\n" + "=" * 50)
    print("第3步：创建模型")
    print("=" * 50)
    model = CatDogCNN()
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n" + "=" * 50)
    print("第4步：训练模型")
    print("=" * 50)
    model = train_model(model, train_loader, criterion, optimizer, epochs=5)

    print("\n" + "=" * 50)
    print("第5步：测试模型")
    print("=" * 50)
    test_model(model, test_loader)

    print("\n项目完成！")
