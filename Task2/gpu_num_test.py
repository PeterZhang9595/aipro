import torch
    
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for data parallel training")