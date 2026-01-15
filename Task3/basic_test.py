import torch
import numpy as np
import unittest
import example as e
import torch.nn.functional as F

class MY_RELU:
    def __init__(self):
        self.cache = None

    def forward(self,x):
        if isinstance(x,np.ndarray):
            x = e.from_numpy(x)
        if not isinstance(x,e.Tensor):
            raise TypeError("RELU 输入必须是 Tensor")
        if x.device() !=e.DeviceType.CUDA:
            x = x.gpu()

        self.cache = x.clone()

        output = e.relu(x)

        return output
    
    def backward(self,grad):
        if isinstance(grad,np.ndarray):
            grad = e.from_numpy(grad)
        if not isinstance(grad,e.Tensor):
            raise TypeError("RELU 输入必须是 Tensor")
        if grad.device() !=e.DeviceType.CUDA:
            grad = grad.gpu()
        
        output = e.relu_backward(self.cache,grad)

        return output

class TestMYRELU(unittest.TestCase):
    def setUp(self):
        # 每次测试都生成相同随机数据
        np.random.seed(42)
        self.batch_size = 32
        self.features = 16

        self.x_np = np.random.randn(self.batch_size, self.features).astype(np.float32)
        self.grad_np = np.random.randn(self.batch_size, self.features).astype(np.float32)

        # PyTorch 对应张量
        self.x_torch = torch.from_numpy(self.x_np).cuda()
        self.grad_torch = torch.from_numpy(self.grad_np).cuda()

        # mytorch Tensor
        self.x_mytorch = e.from_numpy(self.x_np).gpu()
        self.grad_mytorch = e.from_numpy(self.grad_np).gpu()

        # 创建 relu 算子
        self.relu = MY_RELU()

    def test_forward(self):
        print("relu forward test begin")
        out_mytorch = self.relu.forward(self.x_mytorch).to_numpy()
        out_torch = torch.relu(self.x_torch).cpu().numpy().flatten()

        diff = np.abs(out_mytorch - out_torch)
        print(f"mean diff:{diff.mean()},max diff:{diff.max()}")
        self.assertTrue(diff.max() < 1e-4, f"Forward max diff too large: {diff.max()}")
        print("relu forward test finished")

    def test_backward(self):
        # 前向缓存
        print("relu backward test begin")
        _ = self.relu.forward(self.x_mytorch)

        grad_out_mytorch = self.relu.backward(self.grad_mytorch).to_numpy()
        grad_out_torch = (torch.relu(self.x_torch) > 0).float() * self.grad_torch
        grad_out_torch = grad_out_torch.cpu().numpy().flatten()

        diff = np.abs(grad_out_mytorch - grad_out_torch)
        print(f"mean diff:{diff.mean()},max diff:{diff.max()}")
        self.assertTrue(diff.max() < 1e-4, f"Backward max diff too large: {diff.max()}")

        print("relu backward test finished")

class MY_SIGMOID:
    def __init__(self):
        self.cache = None

    def forward(self,x):
        if isinstance(x,np.ndarray):
            x = e.from_numpy(x)
        if not isinstance(x,e.Tensor):
            raise TypeError("SIGMOID 输入必须是 Tensor")
        if x.device() !=e.DeviceType.CUDA:
            x = x.gpu()

        self.cache = x.clone()

        output = e.sigmoid(x)

        return output
    
    def backward(self,grad):
        if isinstance(grad,np.ndarray):
            grad = e.from_numpy(grad)
        if not isinstance(grad,e.Tensor):
            raise TypeError("SIGMOID 输入必须是 Tensor")
        if grad.device() !=e.DeviceType.CUDA:
            grad = grad.gpu()
        
        output = e.sigmoid_backward(self.cache,grad)

        return output


class TestMYSIGMOID(unittest.TestCase):
    def setUp(self):
        # 每次测试都生成相同随机数据
        np.random.seed(42)
        self.batch_size = 320
        self.features = 160

        self.x_np = np.random.randn(self.batch_size, self.features).astype(np.float32)
        self.grad_np = np.random.randn(self.batch_size, self.features).astype(np.float32)

        # PyTorch 对应张量
        self.x_torch = torch.from_numpy(self.x_np).cuda()
        self.grad_torch = torch.from_numpy(self.grad_np).cuda()

        # mytorch Tensor
        self.x_mytorch = e.from_numpy(self.x_np).gpu()
        self.grad_mytorch = e.from_numpy(self.grad_np).gpu()

        # 创建 relu 算子
        self.sigmoid = MY_SIGMOID()

    def test_forward(self):
        print("sigmoid forward test begin")
        out_mytorch = self.sigmoid.forward(self.x_mytorch).to_numpy()
        out_torch = torch.sigmoid(self.x_torch).cpu().numpy().flatten()

        diff = np.abs(out_mytorch - out_torch)
        print(f"mean diff:{diff.mean()},max diff:{diff.max()}")
        self.assertTrue(diff.max() < 1e-4, f"Forward max diff too large: {diff.max()}")
        print("sigmoid forward test finished")

    def test_backward(self):
        # 前向缓存
        print("sigmoid backward test begin")
        _ = self.sigmoid.forward(self.x_mytorch)

        grad_out_mytorch = self.sigmoid.backward(self.grad_mytorch).to_numpy()

            

        y_torch = torch.sigmoid(self.x_torch)
        grad_out_torch = self.grad_torch * y_torch * (1 - y_torch)
        grad_out_torch = grad_out_torch.cpu().numpy().flatten()


        diff = np.abs(grad_out_mytorch - grad_out_torch)
        print(f"mean diff:{diff.mean()},max diff:{diff.max()}")
        self.assertTrue(diff.max() < 1e-4, f"Backward max diff too large: {diff.max()}")
        print("simoid backward test finished")

class MY_FC:
    def __init__(self):
        self.cache = None
        self.weights = None
        self.bias = None
    def forward(self,x):
        if isinstance(x,np.ndarray):
            x = e.from_numpy(x)
        if not isinstance(x,e.Tensor):
            raise TypeError("FC 输入必须是 Tensor")
        if x.device() !=e.DeviceType.CUDA:
            x = x.gpu()
        self.cache = x.clone()
        y = e.fully_connected_forward(x,self.weights,self.bias)
        return y
    def backward(self,x,input=None):
        if input == None:
            input = self.cache
        if input is None:
            raise RuntimeError("MY_FC.backward() 没有输入缓存，请先 forward 一次或显式传入 input")
        if isinstance(x,np.ndarray):
            x = e.from_numpy(x)
        if not isinstance(x,e.Tensor):
            raise TypeError("FC 输入必须是 Tensor")
        if x.device() !=e.DeviceType.CUDA:
            x = x.gpu()
        
        grad_input,grad_weights,grad_bias = e.fully_connected_backward(input,self.weights,self.bias,x)

        return (grad_input,grad_weights,grad_bias)



class TestFullyConnectedForward(unittest.TestCase):
    def setUp(self):
        # 测试参数
        self.batch_size = 400
        self.in_features = 784  
        self.out_features = 256

        # 随机输入
        self.x_np = np.random.randn(self.batch_size, self.in_features).astype(np.float32)
        self.w_np = np.random.randn(self.out_features, self.in_features).astype(np.float32)
        self.b_np = np.random.randn(self.out_features).astype(np.float32)
        self.grad_np = np.random.randn(self.batch_size, self.out_features).astype(np.float32)
        # PyTorch reference
        x_torch = torch.from_numpy(self.x_np)
        w_torch = torch.from_numpy(self.w_np)
        b_torch = torch.from_numpy(self.b_np)
        grad_torch = torch.from_numpy(self.grad_np)
        self.ref = (x_torch @ w_torch.T + b_torch).numpy().flatten()

        # mytorch Tensor
        self.fc = MY_FC()
        self.tensor_x = e.from_numpy(self.x_np).gpu()
        self.tensor_w = e.from_numpy(self.w_np).gpu()
        self.tensor_b = e.from_numpy(self.b_np).gpu()
        self.tensor_grad = e.from_numpy(self.grad_np).gpu()
        self.fc.weights = self.tensor_w
        self.fc.bias = self.tensor_b
        self.fc.forward(self.tensor_x)

    def test_forward(self):
        print("fc forward test begin")
        out = self.fc.forward(self.tensor_x)

        # 转回 NumPy
        out_np = out.to_numpy()

        # 检查误差
        diff = np.abs(out_np - self.ref)
        max_diff = diff.max()
        mean_diff = diff.mean()

        print(f"max diff: {max_diff}")
        print(f"mean diff: {mean_diff}")

        self.assertTrue(max_diff < 1e-4, "Forward pass 与 PyTorch 结果太一致了孩子")
        print("fc backward test finished")

    def test_backward(self):
        print("fc backward test begin")
                # ===== 1. PyTorch Reference =====
        x_torch = torch.tensor(self.x_np, requires_grad=True)
        w_torch = torch.tensor(self.w_np, requires_grad=True)
        b_torch = torch.tensor(self.b_np, requires_grad=True)
        grad_torch = torch.tensor(self.grad_np)

        # Forward
        y_torch = x_torch @ w_torch.T + b_torch
        # Backward
        y_torch.backward(grad_torch)

        ref_grad_input = x_torch.grad.detach().numpy()
        ref_grad_weights = w_torch.grad.detach().numpy()
        ref_grad_bias = b_torch.grad.detach().numpy()

        # ===== 2. MyTorch CUDA implementation =====
        grad_input, grad_weights, grad_bias = self.fc.backward(self.tensor_grad)

        grad_input_np = grad_input.to_numpy()
        grad_weights_np = grad_weights.to_numpy()
        grad_bias_np = grad_bias.to_numpy()

        # ===== 3. Compare =====
        diff_in = np.abs(grad_input_np - ref_grad_input.flatten())
        diff_w = np.abs(grad_weights_np - ref_grad_weights.flatten())
        diff_b = np.abs(grad_bias_np - ref_grad_bias.flatten())

        print(f"grad_input  max diff: {diff_in.max():.6f}, mean diff: {diff_in.mean():.6f}")
        print(f"grad_weights max diff: {diff_w.max():.6f}, mean diff: {diff_w.mean():.6f}")
        print(f"grad_bias    max diff: {diff_b.max():.6f}, mean diff: {diff_b.mean():.6f}")

        # ===== 4. Check tolerance =====
        self.assertTrue(diff_in.max() < 1e-3, "grad_input 与 PyTorch 不一致")
        self.assertTrue(diff_w.max() < 1e-3, "grad_weights 与 PyTorch 不一致")
        self.assertTrue(diff_b.max() < 1e-3, "grad_bias 与 PyTorch 不一致")
        print("fc backward test finished,hoo-ray!")

class My_conv:
    def __init__(self):
        self.cache = None
        self.weights = None
        self.bias = None
        self.kernel_size = 3
        self.padding = 1
        self.stride =1
    def forward(self,x):
        if isinstance(x,np.ndarray):
            x = e.from_numpy(x)
        if not isinstance(x,e.Tensor):
            raise TypeError("Conv 输入必须是 Tensor")
        if x.device() !=e.DeviceType.CUDA:
            x = x.gpu()
        self.cache = x.clone()

        y = e.convolution_forward(x,self.weights,self.bias,self.kernel_size,self.stride,self.padding)
        return y
    def backward(self,x,input=None):
        if input == None:
            input = self.cache
        if input is None:
            raise RuntimeError("MY_conv 没有输入缓存，请先 forward 一次或显式传入 input")
        if isinstance(x,np.ndarray):
            x = e.from_numpy(x)
        if not isinstance(x,e.Tensor):
            raise TypeError("conv 输入必须是 Tensor")
        if x.device() !=e.DeviceType.CUDA:
            x = x.gpu()
        
        grad_input,grad_weights,grad_bias = e.convolution_backward(input,self.weights,self.bias,x)

        return (grad_input,grad_weights,grad_bias)

class TestMyConv(unittest.TestCase):
    def setUp(self):
        # ====== 基本参数 ======
        self.batch_size = 32
        self.in_channels = 3
        self.out_channels = 6
        self.kernel_size = 3
        self.height = 32
        self.width = 32
        self.stride = 1
        self.padding = 1

        # ====== 随机初始化输入和参数 ======
        self.x_np = np.random.randn(self.batch_size, self.in_channels, self.height, self.width).astype(np.float32)
        self.w_np = np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size).astype(np.float32)
        self.b_np = np.random.randn(self.out_channels).astype(np.float32)
        self.grad_np = np.random.randn(self.batch_size, self.out_channels, self.height, self.width).astype(np.float32)

        # ====== PyTorch 参考实现 ======
        x_torch = torch.tensor(self.x_np, requires_grad=True)
        w_torch = torch.tensor(self.w_np, requires_grad=True)
        b_torch = torch.tensor(self.b_np, requires_grad=True)
        grad_torch = torch.tensor(self.grad_np)

        y_torch = F.conv2d(x_torch, w_torch, b_torch, stride=self.stride, padding=self.padding)
        self.ref_forward = y_torch.detach().numpy()

        # 反向传播
        y_torch.backward(grad_torch)
        self.ref_grad_input = x_torch.grad.detach().numpy()
        self.ref_grad_weights = w_torch.grad.detach().numpy()
        self.ref_grad_bias = b_torch.grad.detach().numpy()

        # ====== MyTorch 实现 ======
        self.conv = My_conv()
        self.conv.weights = e.from_numpy(self.w_np).gpu()
        self.conv.bias = e.from_numpy(self.b_np).gpu()
        self.tensor_x = e.from_numpy(self.x_np).gpu()
        self.tensor_grad = e.from_numpy(self.grad_np).gpu()

    # ===== 测试 forward =====
    def test_forward(self):
        print("conv forward test begin")
        out = self.conv.forward(self.tensor_x)
        out_np = out.to_numpy()

        diff = np.abs(out_np - self.ref_forward.flatten())
        print(f"Forward max diff: {diff.max():.6f}, mean diff: {diff.mean():.6f}")

        self.assertTrue(diff.max() < 1e-3, "My_conv.forward() 与 PyTorch 不一致")
        print("conv forward test finished")

    # ===== 测试 backward =====
    def test_backward(self):
        print("conv backward test begin")
        grad_input, grad_weights, grad_bias = self.conv.backward(self.tensor_grad,self.tensor_x)

        grad_input_np = grad_input.to_numpy()
        grad_weights_np = grad_weights.to_numpy()
        grad_bias_np = grad_bias.to_numpy()

        diff_in = np.abs(grad_input_np - self.ref_grad_input.flatten())
        diff_w = np.abs(grad_weights_np - self.ref_grad_weights.flatten())
        diff_b = np.abs(grad_bias_np - self.ref_grad_bias.flatten())

        print(f"grad_input  max diff: {diff_in.max():.6f}, mean diff: {diff_in.mean():.6f}")
        print(f"grad_weights max diff: {diff_w.max():.6f}, mean diff: {diff_w.mean():.6f}")
        print(f"grad_bias    max diff: {diff_b.max():.6f}, mean diff: {diff_b.mean():.6f}")

        self.assertTrue(diff_in.max() < 1e-3, "grad_input 不一致")
        self.assertTrue(diff_w.max() < 1e-3, "grad_weights 不一致")
        self.assertTrue(diff_b.max() < 1e-3, "grad_bias 不一致")
        print("conv backward test finished")


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

        self.cache = x.clone()
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


class TestMaxPoolingForwardBackward(unittest.TestCase):
    def setUp(self):
        # 测试输入
        self.batch_size = 2
        self.channels = 12
        self.height = 80
        self.width = 80

        np.random.seed(42)
        self.x_np = np.random.randn(self.batch_size, self.channels, self.height, self.width).astype(np.float32)
        self.grad_np = np.random.randn(self.batch_size, self.channels, self.height // 2, self.width // 2).astype(np.float32)

        # PyTorch reference
        x_torch = torch.tensor(self.x_np, requires_grad=True)
        y_torch = F.max_pool2d(x_torch, kernel_size=2, stride=2, padding=0)
        grad_torch = torch.tensor(self.grad_np)
        y_torch.backward(grad_torch)

        self.ref_forward = y_torch.detach().numpy()
        self.ref_grad_input = x_torch.grad.detach().numpy()

        # MyTorch implementation
        self.pool = My_maxpooling()
        self.tensor_x = e.from_numpy(self.x_np).gpu()
        self.tensor_grad = e.from_numpy(self.grad_np).gpu()
        self.pool.forward(self.tensor_x)

    def test_forward(self):
        print("maxpooling forward test begin")
        out = self.pool.forward(self.tensor_x)
        out_np = out.to_numpy()

        diff = np.abs(out_np - self.ref_forward.flatten())
        print(f"max diff: {diff.max():.6f}")
        print(f"mean diff: {diff.mean():.6f}")

        self.assertTrue(diff.max() < 1e-4, "MaxPooling forward 与 PyTorch 不一致")
        print("maxpooling forward test finished")

    def test_backward(self):
        print("maxpooling backward test begin")
        grad_in = self.pool.backward(self.tensor_grad)
        grad_in_np = grad_in.to_numpy()

        diff = np.abs(grad_in_np - self.ref_grad_input.flatten())
        print(f"grad_input max diff: {diff.max():.6f}, mean diff: {diff.mean():.6f}")

        self.assertTrue(diff.max() < 1e-3, "MaxPooling backward 与 PyTorch 不一致")
        print("maxpooling backward test finished")
    

class My_softmax_cross_entropy:
    def __init__(self):
        self.cache = None
        self.probs = None
        self.labels = None
        self.prediction = None
    def forward(self,x):
        if isinstance(x, np.ndarray):
            x = e.from_numpy(x)
        if not isinstance(x, e.Tensor):
            raise TypeError("softmax 输入必须是 Tensor")
        if x.device() != e.DeviceType.CUDA:
            x = x.gpu()
        self.cache = x.clone()
        self.probs = e.softmax(x)
        loss = e.cross_entropy_loss(self.probs,self.labels)
        return loss
    def backward(self):
        grad = e.softmax_cross_entropy_backward(self.probs,self.labels)
        return grad
    
class TestSoftmaxCrossEntropy(unittest.TestCase):
    def setUp(self):
        # --------------------
        # 测试参数
        # --------------------
        self.batch_size = 120
        self.num_classes = 20

        np.random.seed(42)
        self.x_np = np.random.randn(self.batch_size, self.num_classes).astype(np.float32)
        self.labels_np = np.random.randint(0, self.num_classes, size=(self.batch_size,), dtype=np.int64)

        # --------------------
        # PyTorch 作为参考结果
        # --------------------
        x_torch = torch.tensor(self.x_np, dtype=torch.float32, requires_grad=True)
        labels_torch = torch.tensor(self.labels_np, dtype=torch.long)

        # torch CrossEntropyLoss 内部会自动执行 log_softmax + NLLLoss
        loss_torch = F.cross_entropy(x_torch, labels_torch)
        loss_torch.backward()

        self.ref_loss = loss_torch.item()
        self.ref_grad_input = x_torch.grad.detach().numpy()

        # --------------------
        # MyTorch
        # --------------------
        self.ce = My_softmax_cross_entropy()

        # 先把 label 放进去
        
        labels_np = self.labels_np.astype(np.float32).reshape(-1).copy()
        self.ce.labels = e.from_numpy(labels_np).gpu()

        self.tensor_x = e.from_numpy(self.x_np).gpu()
        self.tensor_labels = e.from_numpy(labels_np).gpu()

        
        

    # ------------------------
    # 测试 forward
    # ------------------------
    def test_forward(self):
        print("softmax forward test begin")
        out = self.ce.forward(self.tensor_x)
        out_np = float(out)

        diff = abs(out_np - self.ref_loss)
        print(f"[Forward] loss diff: {diff:.6f}")

        self.assertTrue(diff < 1e-4, "softmax+cross_entropy forward 与 PyTorch 不一致")
        print("softmax forward test finished")

    # ------------------------
    # 测试 backward
    # ------------------------
    def test_backward(self):
        # 前向计算
        print("softmax backward test begin")
        _ = self.ce.forward(self.tensor_x)

        # backward 输入通常是标量 loss 对应的梯度（1.0）
        #grad_out = e.from_numpy(np.ones((1,), dtype=np.float32)).gpu()
        grad_in = self.ce.backward()
        grad_in_np = grad_in.to_numpy().reshape(self.batch_size, self.num_classes)

        diff = np.abs(grad_in_np - self.ref_grad_input)
        print(f"[Backward] max diff: {diff.max():.6f}, mean diff: {diff.mean():.6f}")

        self.assertTrue(diff.max() < 1e-3, "softmax+cross_entropy backward 与 PyTorch 不一致")
        print("softmax backward test finished")


x = e.from_numpy(np.random.randn(1,16,32,32).astype(np.float32)).gpu()
y = x.clone()  # 看这里会不会报错
