#include <torch/types.h>
#include <iostream>
#include <math.h>
#include <limits>
#include "cuda_runtime.h"
#include "gpu.cuh"

__global__ void quantize(int num_quant_levels,
                         float* quant_levels,
                         int num_elements,
                         float* tensor,
                         float* result) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < num_elements; i += stride) {
		int middle_point; // Middle point
		int optimal_point; // Optimal point
		int l = 0; // Lower bound
		int h = num_quant_levels; // Higher bound
		float difference = 1.0f; // Difference between a given point and the current middle point
		while (l <= h) {
			middle_point = l + (h - l) / 2;
			if (abs(tensor[i] - quant_levels[middle_point]) < difference) {
				// If the distance between the new point is smaller than the current distance
				difference = abs(tensor[i] - quant_levels[middle_point]);
				optimal_point = middle_point;
			}
			if (quant_levels[middle_point] < tensor[i]) {
				l = middle_point + 1;
			}
			else {
				h = middle_point - 1;
			}
		}
		result[i] = quant_levels[optimal_point];
	}
	return;
}

torch::Tensor quant_cuda(torch::Tensor tensor, int num_quant_levels, float min_value, float max_value) {
  // Determine quantization levels
  torch::Tensor quant_levels = at::linspace(min_value, max_value, num_quant_levels);
  torch::Tensor result = torch::zeros(tensor.sizes());
  float *quant_levels_gpu;
  torch::Tensor *result_gpu;
  cudaMalloc(&quant_levels_gpu, sizeof(float) * quant_levels.numel());
  cudaMalloc(&result_gpu, sizeof(float) * tensor.numel());
  cudaMemcpy(quant_levels_gpu, quant_levels.data<float>(), sizeof(float) * quant_levels.numel(), cudaMemcpyHostToDevice);
  // cudaMemcpy(result_gpu, result.data<float>(), sizeof(float) * tensor.numel(), cudaMemcpyHostToDevice);
  quantize<<<GET_BLOCKS(tensor.numel()), CUDA_NUM_THREADS>>>(num_quant_levels, quant_levels_gpu, tensor.numel(), tensor.data<float>(), result_gpu.data<float>());
  cudaDeviceSynchronize();
  cudaFree(quant_levels_gpu);
  return result;
}
