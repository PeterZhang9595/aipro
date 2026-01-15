import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt  

# =========================
# LeNet with BatchNorm
# =========================
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
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
        return self.fc3(x)

# =========================
# Train
# =========================
def train(rank, world_size, args):
    #初始化分布式训练环境，使用英伟达NCCL库进行GPU通信
    dist.init_process_group(backend="nccl")

    #把每个进程都绑定到指定的GPU上
    torch.cuda.set_device(rank)

    #把后续的训练都绑定到device上面
    device = torch.device(f"cuda:{rank}")


    model = LeNet().to(device)

    #由于我在模型中使用了BatchNormlization，因此要同步各个GPU上的均值和方差
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    #最终使用pytorch接口把模型和device包装成最终的model
    model = DDP(model, device_ids=[rank])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    #分布式采样，保证各个GPU不重复处理相同数据
    sampler = DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, sampler=sampler,
                             num_workers=4, pin_memory=True)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    epoch_times = []   # 用来记录每个 epoch 时间
    epoch_losses = []  # 用来记录每个 epoch 的平均 loss

    for epoch in range(args.epochs):
        # 每个epoch开始的时候重新打乱数据集
        sampler.set_epoch(epoch)
        model.train()
        start = time.time()
        running_loss = 0.0
        batch_count = 0

        for images, labels in trainloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

        epoch_time = time.time() - start
        epoch_times.append(epoch_time)

        avg_epoch_loss = running_loss / batch_count
        epoch_losses.append(avg_epoch_loss)

        if rank == 0:
            print(f"[Epoch {epoch+1}] Loss: {avg_epoch_loss:.3f}, Time: {epoch_time:.2f}s")

    if rank == 0:
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        print(f"Average epoch time: {avg_epoch_time:.2f}s")

        torch.save(model.module.state_dict(), "cifar10_ddp_lenet.pth")
        print("Model saved.")

        # 绘制 loss 曲线并保存
        plt.figure()
        plt.plot(range(1, args.epochs + 1), epoch_losses, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.savefig("loss_curve.png")
        print("Loss curve saved as loss_curve.png")


def test(device, args):
    device = torch.device(device)
    model = LeNet().to(device)
    # 加载 DDP 保存的权重
    state_dict = torch.load("cifar10_ddp_lenet.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# =========================
# 主函数
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)  
    parser.add_argument("--device", type=str, default="cuda:0") 
    args = parser.parse_args()

    if args.mode == "train":
        world_size = torch.cuda.device_count()
        train(rank=int(os.environ["LOCAL_RANK"]), world_size=world_size, args=args)
    elif args.mode == "test":
        test(device=args.device, args=args)