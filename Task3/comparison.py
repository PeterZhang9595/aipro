import torch
import example as e
import torch.nn as nn
import numpy as np
import argparse
import time
from network import MyConv

def test_torch_conv(
    batch_size,
    cin,
    cout,
    h,
    w,
    device,
    kernel_size=3,
    stride=1,
    padding=1,
    warmup=10,
    iters=50,
):
    assert device.type == "cuda", "PyTorch benchmark 只测 CUDA"

    x = torch.randn(
        batch_size, cin, h, w,
        device=device,
        requires_grad=True
    )
    conv = nn.Conv2d(
        cin, cout,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    ).to(device)

    grad = torch.randn(
        batch_size, cout, h, w,
        device=device
    )

    # --------------------
    # warmup
    # --------------------
    for _ in range(warmup):
        y = conv(x)
        y.backward(grad)
        x.grad.zero_()

    torch.cuda.synchronize()

    # --------------------
    # benchmark
    # --------------------
    t0 = time.time()
    for _ in range(iters):
        y = conv(x)
        y.backward(grad)
        x.grad.zero_()
    torch.cuda.synchronize()
    t1 = time.time()

    avg_ms = (t1 - t0) * 1000 / iters
    print(f"[PyTorch] avg forward+backward: {avg_ms:.3f} ms")

    return avg_ms

def test_mytorch_conv(
    batch_size,
    cin,
    cout,
    h,
    w,
    kernel_size=3,
    stride=1,
    padding=1,
    warmup=10,
    iters=50,
):

    x_np = np.random.randn(batch_size, cin, h, w).astype(np.float32)
    x = e.from_numpy(x_np).gpu()

    conv = MyConv(cin, cout)

    grad_np = np.random.randn(batch_size, cout, h, w).astype(np.float32)
    grad = e.from_numpy(grad_np).gpu()

    # --------------------
    # warmup
    # --------------------
    for _ in range(warmup):
        y = conv.forward(x)
        conv.backward(grad)

    torch.cuda.synchronize()

    # --------------------
    # benchmark
    # --------------------
    t0 = time.time()
    for _ in range(iters):
        y = conv.forward(x)
        conv.backward(grad)
    torch.cuda.synchronize()
    t1 = time.time()

    avg_ms = (t1 - t0) * 1000 / iters
    print(f"[MyTorch] avg forward+backward: {avg_ms:.3f} ms")

    return avg_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",type=int,default=8)
    parser.add_argument("--input_channel",type=int,default=3)
    parser.add_argument("--output_channel",type=int,default=8)
    parser.add_argument("--img_size",type=int,default=64)
    args = parser.parse_args()

    device = torch.device("cuda:0"if torch.cuda.is_available()else"cpu")
    print(f"using device: {device}")
    print(f"batch_size: {args.batch_size}")
    print(f"cin: {args.input_channel}")
    print(f"cout: {args.output_channel}")
    print(f"h/w: {args.img_size}")

    torch_ms = test_torch_conv(
        args.batch_size,args.input_channel,args.output_channel,args.img_size,args.img_size,device
    )

    mytorch_ms = test_mytorch_conv(
        args.batch_size,args.input_channel,args.output_channel,args.img_size,args.img_size,device
    )

    print(f"speedup (MyTorch / PyTorch): {torch_ms / mytorch_ms:.2f}x")