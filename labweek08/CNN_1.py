import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from PIL import Image


# 自定义训练数据集
class PreloadedDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = {label: idx for idx, label in enumerate(os.listdir(data_dir))}

        for label in self.label_map:
            class_dir = os.path.join(data_dir, label)
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.jpg'):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.label_map[label])

        self.data = []
        for img_path, label in zip(self.image_paths, self.labels):
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            self.data.append((image, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 自定义测试数据集
class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = {'baihe': 0, 'dangshen': 1, 'gouqi': 2, 'huaihua': 3, 'jinyinhua': 4}

        for img_name in os.listdir(test_dir):
            if img_name.endswith('.jpg'):
                self.image_paths.append(os.path.join(test_dir, img_name))
                label_name = img_name.split('0')[0]  # 根据图像名称获取标签
                self.labels.append(self.label_map[label_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    # 数据预处理，包括调整图像大小和标准化
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((128, 128)),  # 调整图像大小为128x128
            transforms.ToTensor(),  # 转换为Tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
        ]),
        'test': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 数据集路径
    data_dir = 'F:/PythonProjects/AI_Labcodes/labweek08'
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # 获取标签
    class_names = os.listdir(train_dir)
    class_names.sort()  # 保证标签顺序一致

    # 加载数据集
    train_dataset = PreloadedDataset(train_dir, transform=data_transforms['train'])
    test_dataset = TestDataset(test_dir, transform=data_transforms['test'])

    # 创建数据加载器
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0),  # 不使用额外的worker
        'test': DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    }

    # 获取数据集的大小
    dataset_sizes = {'train': len(train_dataset), 'test': len(test_dataset)}

    # 选择设备（GPU或CPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载预训练的ResNet模型，并调整最后的全连接层
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 训练和评估模型的函数
    def train_model(model, criterion, optimizer, num_epochs=25):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        train_loss_history = []
        train_acc_history = []

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # 每个epoch都有一个训练阶段
            model.train()  # 设置模型为训练模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders['train']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes['train']
            epoch_acc = running_corrects.double() / dataset_sizes['train']

            train_loss_history.append(epoch_loss)
            train_acc_history.append(epoch_acc.cpu().numpy())  # 将 Tensor 移动到 CPU 然后转换为 numpy

            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 保存最佳模型权重
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            print()

        print(f'Best train Acc: {best_acc:4f}')

        # 加载最佳模型权重
        model.load_state_dict(best_model_wts)
        return model, train_loss_history, train_acc_history

    # 训练模型
    model, train_loss, train_acc = train_model(model, criterion, optimizer, num_epochs=25)

    # 绘制训练过程中的损失曲线和准确率曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')

    plt.show()

    # 冻结模型参数并测试模型
    model.eval()
    test_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)

    test_acc = test_corrects.double() / dataset_sizes['test']
    print(f'Test Acc: {test_acc:.4f}')

    # 测试函数，处理单张图像
    def predict_image(image_path, model, class_names):
        model.eval()
        image = Image.open(image_path)
        image = data_transforms['test'](image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            predicted_class = class_names[preds.item()]
            return predicted_class

    # 遍历测试文件夹，预测每张图像的类别
    test_images = os.listdir(test_dir)
    test_images = [os.path.join(test_dir, img) for img in test_images if img.endswith('.jpg')]

    for image_path in test_images:
        predicted_class = predict_image(image_path, model, class_names)
        print(f'Image: {os.path.basename(image_path)}, Predicted Class: {predicted_class}')


if __name__ == '__main__':
    main()
