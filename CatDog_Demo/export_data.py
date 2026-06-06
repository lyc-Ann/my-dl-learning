import os
from torchvision import datasets
from PIL import Image

def export_cat_dog_data():
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

if __name__ == "__main__":
    export_cat_dog_data()
