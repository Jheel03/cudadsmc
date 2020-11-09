#include <stdio.h>
#include <string.h>
#include <initialization.h>
#include <translation_boundaryCollisions.h>
#include <indexing.h>
#include <binaryCollisions.h>
#include <memoryOps.cuh>

void posHostToDevice();
void posDeviceToHost();
void velHostToDevice();
void velDeviceToHost();

int main() {

	Setup();
	plot(1);

	for (unsigned int j = 2; j <= 200; j++) {

		translationInitialize();
		velDeviceToHost();
		plot(j);
		indexing();
		initializeBinaryCollisions();
		posHostToDevice();
		velHostToDevice();
		printf("\nIteration %u Completed.", j - 1);
	}
	

	return 0;

}





void posHostToDevice() {
	cudaMemcpy(DevPosX, PosX, TotalSimulatedParticles * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(DevPosY, PosY, TotalSimulatedParticles * sizeof(double), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
}

void velHostToDevice() {
	cudaMemcpy(DevVelX, VelX, TotalSimulatedParticles * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(DevVelY, VelY, TotalSimulatedParticles * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(DevVelZ, VelZ, TotalSimulatedParticles * sizeof(double), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
}

void posDeviceToHost() {
	cudaMemcpy(PosX, DevPosX, TotalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(PosY, DevPosY, TotalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
}

void velDeviceToHost() {
	cudaMemcpy(VelX, DevVelX, TotalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(VelY, DevVelY, TotalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(VelZ, DevVelZ, TotalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
}