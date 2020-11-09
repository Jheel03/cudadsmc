#pragma once
#include <initialization.h>

void indexing() {

	unsigned int mainAcc = 0;
	for (unsigned int i = 0; i < NY; i++) {
		for (unsigned int j = 0; j < NX; j++) {
			unsigned int cellAcc = 0;
			for (unsigned int k = 0; k < TotalSimulatedParticles; k++) {
				if (PosY[k] > i * DY && PosY[k] <= (i + 1) * DY && PosX[k] > j * DX && PosX[k] <= (j + 1) * DX) {
					NewPosX[mainAcc + cellAcc] = PosX[k];
					NewPosY[mainAcc + cellAcc] = PosY[k];
					NewVelX[mainAcc + cellAcc] = VelX[k];
					NewVelY[mainAcc + cellAcc] = VelY[k];
					NewVelZ[mainAcc + cellAcc] = VelZ[k];
					cellAcc++;
				}
			}
			IndexArray[i * NX + j] = cellAcc;
			mainAcc += cellAcc;
		}
	}

	cudaMemcpy(PosX, NewPosX, TotalSimulatedParticles * sizeof(double), cudaMemcpyHostToHost);
	cudaMemcpy(PosY, NewPosY, TotalSimulatedParticles * sizeof(double), cudaMemcpyHostToHost);
	cudaMemcpy(VelX, NewVelX, TotalSimulatedParticles * sizeof(double), cudaMemcpyHostToHost);
	cudaMemcpy(VelY, NewVelY, TotalSimulatedParticles * sizeof(double), cudaMemcpyHostToHost);
	cudaMemcpy(VelZ, NewVelZ, TotalSimulatedParticles * sizeof(double), cudaMemcpyHostToHost);

}