import os
from torchvision import datasets
from PIL import Image

# 自动创建文件夹
os.makedirs("./train/cat", exist_ok=True)
os.makedirs("./train/dog", exist_ok=True)

# 下载 CIFAR10
cifar = datasets.CIFAR10(root="./cifar10", train=True, download=True)

# 导出猫狗
for idx, (img, label) in enumerate(cifar):
    # 关键：不管 img 是啥，统一转成可以保存的 PIL 图片
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    # 猫=3，狗=5
    if label == 3:
        img.save(f"./train/cat/{idx}.jpg")
    elif label == 5:
        img.save(f"./train/dog/{idx}.jpg")

print("导出完成！")
print(f"猫数量：{len(os.listdir('./train/cat'))}")
print(f"狗数量：{len(os.listdir('./train/dog'))}")