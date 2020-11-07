#pragma once

int assignString(char *str, char assignstr[], int n) {
	for (int i = 0; i < n; i++) {
		str[i] = assignstr[i];
	}
	return 0;
}

int setString(char *str, char value, int n) {
	for (int i = 0; i < n; i++) {
		str[i] = value;
	}
	return 0;
}


__global__ void userMemsetKernel(double *devPtr, double value, unsigned int size) {
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < size) {
		devPtr[tid] = value;
	}
}

int userMemset(double *devPtr, double value, unsigned int size) {
	dim3 blockSize(512);
	dim3 gridSize(size / 512 + 1);
	userMemsetKernel << <gridSize, blockSize >> > (devPtr, value, size);
	return 0;
}