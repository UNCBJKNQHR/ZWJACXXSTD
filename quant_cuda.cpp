#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// CUDA kernel
torch::Tensor quant_cuda(torch::Tensor tensor, int num_quant_levels, float min_value, float max_value);

// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor quant(torch::Tensor tensor, int num_quant_levels, float min_value, float max_value) {
  CHECK_INPUT(tensor);
  int num_elements = tensor.numel();
  return quant_cuda(tensor, num_quant_levels, min_value, max_value);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize", &quant, "tbd");
}
