#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include <stdlib.h>
#include <initialization.cuh>


__global__ void scaleRandomNumber(double *deviceOutputPtr, unsigned int n, double init, double domainSize) {
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < n) {
		deviceOutputPtr[tid] = init + domainSize * deviceOutputPtr[tid];
	}
}

int particlesGeneratorInitialize(double **velocityX, double **velocityY, double **velocityZ, 
	double **positionX, double **positionY) {

	dim3 blockSize(512);
	dim3 gridSize(totalSimulatedParticles / 512 + 1);

	*velocityX = (double*)malloc(totalSimulatedParticles * sizeof(double));
	*velocityY = (double*)malloc(totalSimulatedParticles * sizeof(double));
	*velocityZ = (double*)malloc(totalSimulatedParticles * sizeof(double));
	*positionX = (double*)malloc(totalSimulatedParticles * sizeof(double));
	*positionY = (double*)malloc(totalSimulatedParticles * sizeof(double));

	double *deviceOutputPtr;
	cudaMalloc(&deviceOutputPtr, totalSimulatedParticles * sizeof(double));

	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32);
	unsigned long long seed;

	seed = 12345;
	curandSetPseudoRandomGeneratorSeed(generator, seed);
	curandStatus_t posGeneratorStatusX = curandGenerateUniformDouble(generator, deviceOutputPtr, totalSimulatedParticles);
	cudaDeviceSynchronize();

	scaleRandomNumber << <gridSize, blockSize >> > (deviceOutputPtr, totalSimulatedParticles, 0, domainLength);
	cudaDeviceSynchronize();

	cudaMemcpy(*positionX, deviceOutputPtr, totalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	seed = 23456;
	curandSetPseudoRandomGeneratorSeed(generator, seed);
	curandStatus_t posGeneratorStatusY = curandGenerateUniformDouble(generator, deviceOutputPtr, totalSimulatedParticles);
	cudaDeviceSynchronize();

	scaleRandomNumber << <gridSize, blockSize >> > (deviceOutputPtr, totalSimulatedParticles, 0, domainLength);
	cudaDeviceSynchronize();

	cudaMemcpy(*positionY, deviceOutputPtr, totalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	seed = 34567;
	curandSetPseudoRandomGeneratorSeed(generator, seed);
	curandStatus_t velGeneratorStatusX = curandGenerateNormalDouble(generator, deviceOutputPtr, totalSimulatedParticles, UFree[0], stdDev);
	cudaDeviceSynchronize();

	cudaMemcpy(*velocityX, deviceOutputPtr, totalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	seed = 45678;
	curandSetPseudoRandomGeneratorSeed(generator, seed);
	curandStatus_t velGeneratorStatusY = curandGenerateNormalDouble(generator, deviceOutputPtr, totalSimulatedParticles, UFree[1], stdDev);
	cudaDeviceSynchronize();

	cudaMemcpy(*velocityY, deviceOutputPtr, totalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	seed = 56789;
	curandSetPseudoRandomGeneratorSeed(generator, seed);
	curandStatus_t velGeneratorStatusZ = curandGenerateNormalDouble(generator, deviceOutputPtr, totalSimulatedParticles, UFree[2], stdDev);
	cudaDeviceSynchronize();

	cudaMemcpy(*velocityZ, deviceOutputPtr, totalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	curandDestroyGenerator(generator);
	cudaFree(deviceOutputPtr);
	cudaDeviceReset();

	return 0;
}