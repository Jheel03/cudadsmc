#pragma once

void sorting(unsigned int n, unsigned int NX, unsigned int NY, double cellHeight, double cellLength, 
	double *positionX, double *positionY, double *velocityX, double *velocityY, double *newPositionX, 
	double *newPositionY, unsigned int *indexArray, double *newVelocityX, double *newVelocityY) {
	unsigned int mainAcc = 0;
	//printf("\nSorting");
	for (unsigned int i = 0; i < NY; i++) {
		for (unsigned int j = 0; j < NX; j++) {
			unsigned int cellAcc = 0;
			for (unsigned int k = 0; k < n; k++) {
				if (positionY[k] > i * cellHeight && positionY[k] <= (i + 1) * cellHeight && positionX[k] > j * cellLength && positionX[k] <= (j + 1) * cellLength) {
					newPositionX[mainAcc + cellAcc] = positionX[k];
					newPositionY[mainAcc + cellAcc] = positionY[k];
					newVelocityX[mainAcc + cellAcc] = velocityX[k];
					newVelocityY[mainAcc + cellAcc] = velocityY[k];
					cellAcc++;
				}
			}
			indexArray[i * NX + j] = cellAcc;
			//printf("\n%u", cellAcc);
			mainAcc += cellAcc;
		}
		//printf("\n%u", mainAcc);
	}
}


void indexingInitialization(unsigned int* totalSimulatedParticles, unsigned int NX, unsigned int NY, double cellHeight, double cellLength,
	double* positionX, double* positionY, double* velocityX, double *velocityY, double* newPositionX, double* newPositionY, 
	unsigned int* indexArray, double* newVelocityX, double* newVelocityY) {

	sorting(*totalSimulatedParticles, NX, NY, cellHeight, cellLength, positionX, positionY, velocityX, velocityY,
		newPositionX, newPositionY, indexArray, newVelocityX, newVelocityY);

}