### Anonymous GitHub repository for MemTorch: A Simulation Framework for Deep Memristive Cross-bar Architectures (ISCAS 2020 Paper ID: 2540)
Our C++/CUDA PyTorch quantization extension is used within MemTorch to model the finite number of conductance states for non-ideal memristive devices. To install from source (requires the torch and numpy packages and CUDA Toolkit 10.1) execute the following commands within a terminal:

```
git clone https://github.com/UNCBJKNQHR/ZWJACXXSTD
cd ZWJACXXSTD
python setup.py install
```

Example usage:

```
import torch
import quantization

torch.manual_seed(1)
num_quantization_states = 5
tensor = torch.zeros(1, 5, 4).uniform_(-1, 1).cuda()
print(tensor)
quantization.quantize(tensor, num_quantization_states, tensor.min(), tensor.max())
print(tensor)
```

Corresponding output:

```
# First print(tensor)
tensor([[[ 0.5153, -0.4414, -0.1939,  0.4694],
         [-0.9414,  0.5997, -0.2057,  0.5087],
         [ 0.1390, -0.1224,  0.2774,  0.0493],
         [ 0.3652, -0.3897, -0.0729, -0.0900],
         [ 0.1449, -0.0040,  0.8742,  0.3112]]], device='cuda:0')
```
```
# Second print(tensor)
tensor([[[ 0.4203, -0.4875, -0.0336,  0.4203],
         [-0.9414,  0.4203, -0.0336,  0.4203],
         [-0.0336, -0.0336,  0.4203, -0.0336],
         [ 0.4203, -0.4875, -0.0336, -0.0336],
         [-0.0336, -0.0336,  0.8742,  0.4203]]], device='cuda:0')
```
