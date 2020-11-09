#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include <stdlib.h>

__global__ void rayleighKernel(double sigma, unsigned int n, double* devPointer) {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		devPointer[tid] = sigma * sqrt(-2 * log(devPointer[tid]));
	}
}

unsigned int initializeRayleigh(size_t n, double* returnArray, double sigma, unsigned long int seed) {

	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32);
	//unsigned long long seed;

	//seed = 12345;
	curandSetPseudoRandomGeneratorSeed(generator, seed);
	curandGenerateUniformDouble(generator, returnArray, n);
	cudaDeviceSynchronize();

	dim3 blockSize(512);
	dim3 gridSize(n / 512 + 1);
	rayleighKernel << <gridSize, blockSize >> > (sigma, n, returnArray);
	cudaDeviceSynchronize();

	curandDestroyGenerator(generator);

	return 0;

}
