import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. 模型定义 (完全对齐你的架构)
# =========================
class torch_network(nn.Module):
    def __init__(self):
        super(torch_network, self).__init__()
        # Conv1: 3->8, Pool
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv2: 8->16, Pool
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv3: 16->32
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # FC Layers (32*8*8 = 2048)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        
        self.relu = nn.ReLU()

        # 使用 He 初始化 (对齐你的 np.sqrt(2/n_in))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# =========================
# 2. 数据准备
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# =========================
# 3. 训练与评估
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch_network().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2) # 对齐你的 lr=1e-2

loss_history = []
epochs = 5

for epoch in range(epochs):
    model.train()
    start_time = time.time()
    running_loss = 0.0
    
    for step, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if step % 100 == 99:
            loss_history.append(loss.item())
            if step % 500 == 499:
                print(f"Epoch {epoch+1} | Step {step+1:04d} | Loss: {loss.item():.4f}")

    epoch_duration = time.time() - start_time
    print(f">>> Epoch {epoch+1} Duration: {epoch_duration:.2f}s")

# =========================
# 4. 测试准确率
# =========================
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nFinal Test Accuracy: {100 * correct / total:.2f}%")

# =========================
# 5. 绘制 Loss 曲线
# =========================
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.title("Training Loss (PyTorch)")
plt.xlabel("Steps (x100)")
plt.ylabel("Loss")
plt.grid(True)
plt.show()