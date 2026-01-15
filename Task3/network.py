import numpy as np
import torch
import example as e
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.transforms import ToTensor
import random
import time
import matplotlib.pyplot as plt

# =========================
# Conv Layer (CUDA)
# =========================
class MyConv:
    def __init__(self, cin, cout):
        self.kernel_size = 3
        self.padding = 1
        self.stride = 1

        n_in = cin * self.kernel_size * self.kernel_size
        std = np.sqrt(2.0/n_in)
        w = (np.random.randn(cout, cin, 3, 3)*std).astype(np.float32) 
        b = np.zeros(cout, dtype=np.float32)

        self.weights = e.from_numpy(w).gpu()
        self.bias = e.from_numpy(b).gpu()
        self.cache = None

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = e.from_numpy(x)
        if not isinstance(x, e.Tensor):
            raise TypeError("Conv 输入必须是 Tensor")
        if x.device() != e.DeviceType.CUDA:
            x = x.gpu()

        self.cache = x
        y = e.convolution_forward(x, self.weights, self.bias,
                                  self.kernel_size, self.stride, self.padding)
        return y

    def backward(self, grad_output, input=None):
        if input is None:
            input = self.cache
        if input is None:
            raise RuntimeError("MyConv backward 缺少 forward 输入")

        if isinstance(grad_output, np.ndarray):
            grad_output = e.from_numpy(grad_output)
        if grad_output.device() != e.DeviceType.CUDA:
            grad_output = grad_output.gpu()

        grad_input, grad_weights, grad_bias = e.convolution_backward(
            input, self.weights, self.bias, grad_output
        )
        return grad_input, grad_weights, grad_bias

# =========================
# ReLU Layer (CUDA)
# =========================
class MY_RELU:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = e.from_numpy(x)
        if not isinstance(x, e.Tensor):
            raise TypeError("ReLU 输入必须是 Tensor")
        if x.device() != e.DeviceType.CUDA:
            x = x.gpu()

        self.cache = x
        return e.relu(x)

    def backward(self, grad_output):
        if isinstance(grad_output, np.ndarray):
            grad_output = e.from_numpy(grad_output)
        if not isinstance(grad_output, e.Tensor):
            raise TypeError("ReLU grad 必须是 Tensor")
        if grad_output.device() != e.DeviceType.CUDA:
            grad_output = grad_output.gpu()

        return e.relu_backward(self.cache, grad_output)

# =========================
# Fully Connected Layer (CUDA)
# =========================
class MY_FC:
    def __init__(self, in_dim, out_dim):
        std = np.sqrt(2.0/in_dim)
        self.weights = e.from_numpy((np.random.randn(out_dim,in_dim)*std).astype(np.float32)).gpu()
        self.bias = e.from_numpy(np.zeros(out_dim, dtype=np.float32)).gpu()
        self.cache = None

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = e.from_numpy(x)
        if not isinstance(x, e.Tensor):
            raise TypeError("FC 输入必须是 Tensor")
        if x.device() != e.DeviceType.CUDA:
            x = x.gpu()
        self.cache = x
        y = e.fully_connected_forward(x, self.weights, self.bias)
        return y

    def backward(self, grad_output):
        if isinstance(grad_output, np.ndarray):
            grad_output = e.from_numpy(grad_output)
        if not isinstance(grad_output, e.Tensor):
            raise TypeError("FC grad 输入必须是 Tensor")
        if grad_output.device() != e.DeviceType.CUDA:
            grad_output = grad_output.gpu()
        grad_input, grad_weights, grad_bias = e.fully_connected_backward(self.cache, self.weights, self.bias, grad_output)
        return grad_input, grad_weights, grad_bias

# =========================
# Softmax + CrossEntropy (CUDA)
# =========================
class My_softmax_cross_entropy:
    def __init__(self):
        self.cache = None
        self.probs = None
        self.labels = None

    def forward(self, x, labels):
        if isinstance(x, np.ndarray):
            x = e.from_numpy(x)
        if not isinstance(x, e.Tensor):
            raise TypeError("softmax 输入必须是 Tensor")
        if x.device() != e.DeviceType.CUDA:
            x = x.gpu()
        self.cache = x
        # ----------------------------
        # labels 必须 float32 + GPU Tensor
        # ----------------------------
        if isinstance(labels, np.ndarray):
            labels = labels.astype(np.float32).reshape(-1).copy()
            labels = e.from_numpy(labels).gpu()
        elif isinstance(labels, e.Tensor):
            if labels.dtype() != e.DType.Float32:
                labels = labels.astype(e.DType.Float32)
            if labels.device() != e.DeviceType.CUDA:
                labels = labels.gpu()
        else:
            raise TypeError("labels 必须是 np.ndarray 或 Tensor")

        self.labels = labels
        self.probs = e.softmax(x)
        loss = e.cross_entropy_loss(self.probs, labels)
        return loss

    def backward(self):
        grad = e.softmax_cross_entropy_backward(self.probs, self.labels)
        return grad
    
class My_maxpooling:
    def __init__(self):
        self.cache = None
        self.mask = None

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = e.from_numpy(x)
        if not isinstance(x, e.Tensor):
            raise TypeError("MaxPooling 输入必须是 Tensor")
        if x.device() != e.DeviceType.CUDA:
            x = x.gpu()

        self.cache = x
        # 默认 kernel_size=2, stride=2, padding=0
        output, mask = e.max_pooling_forward(x)
        self.mask = mask
        return output

    def backward(self, grad_out, input=None):
        if input is None:
            input = self.cache
        if input is None:
            raise RuntimeError("My_maxpooling.backward() 没有输入缓存，请先 forward 一次或显式传入 input")
        if isinstance(grad_out, np.ndarray):
            grad_out = e.from_numpy(grad_out)
        if not isinstance(grad_out, e.Tensor):
            raise TypeError("maxpooling梯度输入必须是 Tensor")
        if grad_out.device() != e.DeviceType.CUDA:
            grad_out = grad_out.gpu()

        B, C, H, W = input.shape()
        grad_input = e.max_pooling_backward(grad_out, self.mask, H, W)
        return grad_input
# =========================
# CIFAR-10 Data Loader
# =========================
def load_cifar10_traindataset(batch_size=1,train=True):
    # -----------------------------
    # 标准 CIFAR-10 Transform
    # -----------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0,255] -> [0,1], shape: [C,H,W]
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    dataset = CIFAR10(root="./data", train=train, download=True, transform=transform)

    # 将 dataset 转成 numpy list
    data = [(np.asarray(img, dtype=np.float32), label) for img, label in dataset]
    random.shuffle(data)

    # 按 batch 返回
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        xs = np.stack([b[0] for b in batch], axis=0)  # [N,C,H,W]
        ys = np.array([b[1] for b in batch], dtype=np.float32)
        yield xs, ys

def load_cifar10_testdataset(batch_size=1,train=False):
    # -----------------------------
    # 标准 CIFAR-10 Transform
    # -----------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0,255] -> [0,1], shape: [C,H,W]
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    dataset = CIFAR10(root="./data", train=train, download=True, transform=transform)

    # 将 dataset 转成 numpy list
    data = [(np.asarray(img, dtype=np.float32), label) for img, label in dataset]

    # 按 batch 返回
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        xs = np.stack([b[0] for b in batch], axis=0)  # [N,C,H,W]
        ys = np.array([b[1] for b in batch], dtype=np.int64)
        yield xs, ys
# =========================
# flatten helper
# =========================
def tensor_flatten(x):
    shape = x.shape()
    batch_size = shape[0]
    numel_per_sample = x.numel() // batch_size
    return x.reshape([batch_size, numel_per_sample])

class MyNetWork:
    def __init__(self,lr=1e-2):
        self.conv1 = MyConv(3,8)
        self.relu1 = MY_RELU()
        self.pool1 = My_maxpooling()
        self.conv2 = MyConv(8,16)
        self.relu2 = MY_RELU()
        self.pool2 = My_maxpooling()
        self.conv3 = MyConv(16,32)
        self.relu3 = MY_RELU()
        self.fc1 = MY_FC(32*8*8,256)
        self.relu4 = MY_RELU()
        self.fc2 = MY_FC(256,10)
        self.criterion =My_softmax_cross_entropy()
        self.lr = lr
        self.correct_match = 0
        self.total = 0

    def forward(self,x_np:np.ndarray,y_np:np.ndarray):
        x = self.conv1.forward(x_np)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = self.conv3.forward(x)
        x = self.relu3.forward(x)
        self.transfer_shape = x.shape()
        # Flatten
        x_flat = tensor_flatten(x)

        # --------------------
        # labels 转 GPU Tensor
        # --------------------
        ys = y_np.astype(np.float32).reshape(-1).copy()
        labels = e.from_numpy(ys).gpu()

        x = self.fc1.forward(x_flat)
        x = self.relu4.forward(x)
        logits = self.fc2.forward(x)
        #logits.cpu().print()
        loss = self.criterion.forward(logits, labels)
        return loss

    def backward(self):
        grad = self.criterion.backward()
        g,self.dW_fc2,self.db_fc2 = self.fc2.backward(grad)
        g = self.relu4.backward(g)
        g,self.dW_fc1,self.db_fc1 = self.fc1.backward(g)
        
        g = g.reshape(self.transfer_shape)
        g = self.relu3.backward(g)
        g,self.dW3,self.db3 = self.conv3.backward(g)
        g = self.pool2.backward(g)
        g = self.relu2.backward(g)
        g,self.dW2,self.db2 = self.conv2.backward(g)
        g = self.pool1.backward(g)
        g = self.relu1.backward(g)
        g,self.dW1,self.db1 = self.conv1.backward(g)



    def update(self):
        #temp = self.conv1.weights.clone()
        #temp.cpu().print()
        self.conv1.weights -= self.lr * self.dW1
        #tt = self.conv1.weights.clone()
        #tt.cpu().print()
        self.conv1.bias -= self.lr * self.db1
        self.conv2.weights -= self.lr * self.dW2
        self.conv2.bias -= self.lr * self.db2
        self.conv3.weights -= self.lr * self.dW3
        self.conv3.bias -= self.lr * self.db3
        self.fc1.weights -= self.lr * self.dW_fc1 
        self.fc1.bias -= self.lr * self.db_fc1
        self.fc2.weights -= self.lr * self.dW_fc2
        self.fc2.bias -= self.lr * self.db_fc2

    def test(self,x_np:np.ndarray,y_np:np.ndarray,batch_size):
        x = self.conv1.forward(x_np)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = self.conv3.forward(x)
        x = self.relu3.forward(x)
        self.transfer_shape = x.shape()
        x_flat = tensor_flatten(x)

        x = self.fc1.forward(x_flat)
        x = self.relu4.forward(x)
        logits = self.fc2.forward(x)
        #logits.cpu().print()
        #e.softmax(logits).cpu().print()
        probs = e.softmax(logits).to_numpy().reshape(y_np.shape[0],-1)#shape[batch,classes]
        #print(probs)
        
        #print(probs.shape)
        prediction = np.argmax(probs,axis=1)
        #print(prediction)
        self.correct_match += np.sum(prediction == y_np)
        self.total += y_np.shape[0]

# =========================
# Train and test in one function
# =========================
def train():

    batch_size = 16  
    model = MyNetWork()
    # train
    loss_list = [] 
    epochs = 5

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        start = time.time()
        for step, (x_np, y_np) in enumerate(load_cifar10_traindataset(batch_size)):
            loss = model.forward(x_np,y_np)
            model.backward()
            model.update()

            if step % 100 == 0:
                print(f"step {step:04d} | loss {loss:.4f}")
                loss_list.append(loss)
        print(f"Epoch time: {time.time() - start:.2f}s")

    plt.figure(figsize=(8,5))
    plt.plot(loss_list, label='Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    # test
    for _,(x_np,y_np) in enumerate(load_cifar10_testdataset(batch_size=64)):
        model.test(x_np,y_np,64)
    print(f"test acc: {model.correct_match / model.total}")



if __name__ == "__main__":
    train()
