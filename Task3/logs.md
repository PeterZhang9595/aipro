using device: cuda:0
batch_size: 16
cin: 3
cout: 8
h/w: 32
[PyTorch] avg forward+backward: 0.275 ms
[MyTorch] avg forward+backward: 2.016 ms
speedup (MyTorch / PyTorch): 0.14x

using device: cuda:0
batch_size: 32
cin: 3
cout: 8
h/w: 32
[PyTorch] avg forward+backward: 0.313 ms
[MyTorch] avg forward+backward: 3.946 ms
speedup (MyTorch / PyTorch): 0.08x

using device: cuda:0
batch_size: 1
cin: 3
cout: 8
h/w: 32
[PyTorch] avg forward+backward: 0.316 ms
[MyTorch] avg forward+backward: 0.154 ms
speedup (MyTorch / PyTorch): 2.06x

using device: cuda:0
batch_size: 4
cin: 3
cout: 8
h/w: 32
[PyTorch] avg forward+backward: 0.315 ms
[MyTorch] avg forward+backward: 0.313 ms
speedup (MyTorch / PyTorch): 1.01x

using device: cuda:0
batch_size: 8
cin: 3
cout: 8
h/w: 32
[PyTorch] avg forward+backward: 0.275 ms
[MyTorch] avg forward+backward: 1.161 ms
speedup (MyTorch / PyTorch): 0.24x

using device: cuda:0
batch_size: 8
cin: 8
cout: 8
h/w: 32
[PyTorch] avg forward+backward: 0.241 ms
[MyTorch] avg forward+backward: 1.894 ms
speedup (MyTorch / PyTorch): 0.13x

using device: cuda:0
batch_size: 8
cin: 8
cout: 16
h/w: 32
[PyTorch] avg forward+backward: 0.308 ms
[MyTorch] avg forward+backward: 1.867 ms
speedup (MyTorch / PyTorch): 0.17x

using device: cuda:0
batch_size: 8
cin: 3
cout: 8
h/w: 16
[PyTorch] avg forward+backward: 0.314 ms
[MyTorch] avg forward+backward: 0.684 ms
speedup (MyTorch / PyTorch): 0.46x

using device: cuda:0
batch_size: 8
cin: 3
cout: 8
h/w: 8
[PyTorch] avg forward+backward: 0.148 ms
[MyTorch] avg forward+backward: 0.313 ms
speedup (MyTorch / PyTorch): 0.47x

using device: cuda:0
batch_size: 8
cin: 3
cout: 8
h/w: 64
[PyTorch] avg forward+backward: 0.256 ms
[MyTorch] avg forward+backward: 2.743 ms
speedup (MyTorch / PyTorch): 0.09x

NVIDIA GeForce RTX 4070

Epoch 1 | Step 0500 | Loss: 1.6464
Epoch 1 | Step 1000 | Loss: 1.3511
Epoch 1 | Step 1500 | Loss: 1.4405
Epoch 1 | Step 2000 | Loss: 1.6181
Epoch 1 | Step 2500 | Loss: 1.8588
Epoch 1 | Step 3000 | Loss: 1.0808
>>> Epoch 1 Duration: 9.37s
Epoch 2 | Step 0500 | Loss: 0.8827
Epoch 2 | Step 1000 | Loss: 0.8357
Epoch 2 | Step 1500 | Loss: 1.3531
Epoch 2 | Step 2000 | Loss: 1.2056
Epoch 2 | Step 2500 | Loss: 1.0299
Epoch 2 | Step 3000 | Loss: 0.9376
>>> Epoch 2 Duration: 9.76s
Epoch 3 | Step 0500 | Loss: 1.1944
Epoch 3 | Step 1000 | Loss: 0.9085
Epoch 3 | Step 1500 | Loss: 1.0537
Epoch 3 | Step 2000 | Loss: 0.8964
Epoch 3 | Step 2500 | Loss: 0.8745
Epoch 3 | Step 3000 | Loss: 0.8874
>>> Epoch 3 Duration: 9.94s
Epoch 4 | Step 0500 | Loss: 0.6012
Epoch 4 | Step 1000 | Loss: 1.0483
Epoch 4 | Step 1500 | Loss: 0.8843
Epoch 4 | Step 2000 | Loss: 0.9863
Epoch 4 | Step 2500 | Loss: 0.5579
Epoch 4 | Step 3000 | Loss: 0.8154
>>> Epoch 4 Duration: 9.91s
Epoch 5 | Step 0500 | Loss: 0.8392
Epoch 5 | Step 1000 | Loss: 0.4896
Epoch 5 | Step 1500 | Loss: 0.8147
Epoch 5 | Step 2000 | Loss: 0.5958
Epoch 5 | Step 2500 | Loss: 0.9217
Epoch 5 | Step 3000 | Loss: 0.9558
>>> Epoch 5 Duration: 10.15s

Final Test Accuracy: 65.77%
