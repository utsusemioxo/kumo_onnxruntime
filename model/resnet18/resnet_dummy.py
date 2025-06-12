import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

def main():
    # 1. 准备数据变换魔法
    transform = transforms.Compose([
        transforms.Resize(224),  # ResNet默认输入224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. 创建训练数据集（用CIFAR10小兵训练）
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)

    # 3. 召唤ResNet-18战士
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)  # CIFAR10有10类
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 4. 设定损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 5. 开始战斗：简单训练1个epoch
    model.train()
    for epoch in range(1):  # 只跑一轮示范
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # 每100个batch打印一次
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('训练结束，开始导出ONNX！')

    # 6. 准备dummy输入（魔法阵启动器）
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # 7. 导出ONNX模型卷轴
    torch.onnx.export(model, dummy_input, "resnet18_dummy.onnx",
                      input_names=['input'], output_names=['output'],
                      opset_version=13)
    print('ONNX模型已保存为 resnet18_dummy.onnx')

if __name__ == '__main__':
    main()
