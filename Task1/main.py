import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse
import time

# =========================
# 定义网络结构
# =========================
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x

# =========================
# 训练函数
# =========================
def train(model, device, batch_size=64, num_workers=4, epochs=10, lr=0.001, momentum=0.9):
    all_losses = []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_function = nn.CrossEntropyLoss()

    global_step = 0
    total_time =0
    for epoch in range(epochs):
        running_loss = 0.0
        epoch_start_time = time.time()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if (i + 1) % 200 == 0:
                avg_loss = running_loss / 200
                print(f"[Epoch {epoch+1}, Batch {i+1:5d}] Loss: {avg_loss:.3f}")
                all_losses.append(avg_loss)
                running_loss = 0.0
        epoch_time = time.time() - epoch_start_time
        total_time += epoch_time
    print(f"Average epoch training time {total_time / epochs}s")

    PATH = "./cifar10_mymodel.pth"
    torch.save(model.state_dict(), PATH)
    print(f"Model saved to {PATH}")

    plt.plot(all_losses, marker='o')
    plt.xlabel("Steps (x200 batches)")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.show()

# =========================
# 测试函数
# =========================
def test(model, device, batch_size=64, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    print(f"Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %")

# =========================
# 主函数
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="运行模式: train 或 test")
    parser.add_argument("--batch_size", type=int, default=64, help="批量大小")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD 动量")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LeNet()
    model.to(device)

    if args.mode == "train":
        train(model, device, batch_size=args.batch_size, num_workers=args.num_workers,
              epochs=args.epochs, lr=args.lr, momentum=args.momentum)
    elif args.mode == "test":
        model.load_state_dict(torch.load('./cifar10_mymodel.pth', map_location=device))
        test(model, device, batch_size=args.batch_size, num_workers=args.num_workers)
