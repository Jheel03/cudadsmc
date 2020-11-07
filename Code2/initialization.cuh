#pragma once

#include <constants.h>
#include <stdlib.h>
#include <math.h>


unsigned int initializeGasState() {
	
	numberDensity = Avogadro * PFree / (gasConstant * TFree);
	//meanFreePath = 5;
	MMass = Avogadro * moleculeMass;
	MMass_g = MMass * 1000;
	//dt = 1e-2;
	dRef = pow(5 * (alpha + 1) * (alpha + 2) * pow(48.1 * Boltzmann * TFree / PI, 0.5) / (4 * alpha * (5 - 2 * ohmega) * (7 - 2 * ohmega) * muRef), 0.5);
	sigmaRef = PI * dRef * dRef;

	molecularCS = sigmaRef;
	molecularDia = dRef;
	//meanFreePath = 1 / (sqrt(2) * numberDensity * molecularCS);
	meanFreePath = (muRef / PFree) * sqrt(PI * Boltzmann * TFree / (2 * moleculeMass));
	dt = meanFreePath / stdDev;

	//printf("Total Particles = %d, DTime = %8e, Weight = %llu, Cell Length = %8e\n", totalSimulatedParticles, dt, totalParticles / totalSimulatedParticles, cellLength);
	printf("MMass = %8e, Molecular CS = %8e, D Ref = %8e, Mean Free Path = %8e\n", MMass, sigmaRef, dRef, meanFreePath);
	fflush(stdin); // option ONE to clean stdin
	getchar();

	return 0;
}

unsigned int domainCoords(double *boundaryNodesX, double *boundaryNodesY, double *boundaryNormalsX,
	double *boundaryNormalsY, double *boundaryVectorsX, double *boundaryVectorsY) {

	for (unsigned int i = 0; i <= numberCellsX; i++) {
		boundaryNodesX[i] = i * cellLength;
		boundaryNodesY[i] = 0;
	}
	for (unsigned int i = 1; i <= numberCellsY; i++) {
		boundaryNodesX[numberCellsX + i] = domainLength;
		boundaryNodesY[numberCellsX + i] = i * cellHeight;
	}
	for (unsigned int i = 1; i <= numberCellsX; i++) {
		boundaryNodesX[numberCellsX + numberCellsY + i] = domainLength - i * cellLength;
		boundaryNodesY[numberCellsX + numberCellsX + i] = domainHeight;
	}
	for (unsigned int i = 1; i < numberCellsY; i++) {
		boundaryNodesX[numberCellsX + numberCellsX + numberCellsY + i] = 0;
		boundaryNodesY[numberCellsX + numberCellsX + numberCellsY + i] = domainHeight - i * cellHeight;
	}

	for (unsigned int i = 0; i < 2 * (numberCellsX + numberCellsY) - 1; i++) {
		boundaryVectorsX[i] = boundaryNodesX[i + 1] - boundaryNodesX[i];
		boundaryVectorsY[i] = boundaryNodesY[i + 1] - boundaryNodesY[i];
		boundaryNormalsX[i] = -(boundaryNodesY[i + 1] - boundaryNodesY[i]) / cellLength;
		boundaryNormalsY[i] = (boundaryNodesX[i + 1] - boundaryNodesX[i]) / cellLength;
	}
	boundaryVectorsX[2 * (numberCellsX + numberCellsY) - 1] = boundaryNodesX[0] - boundaryNodesX[2 * (numberCellsX + numberCellsY) - 1];
	boundaryVectorsY[2 * (numberCellsX + numberCellsY) - 1] = boundaryNodesY[0] - boundaryNodesY[2 * (numberCellsX + numberCellsY) - 1];
	boundaryNormalsX[2 * (numberCellsX + numberCellsY) - 1] = -(boundaryNodesY[0] - boundaryNodesY[2 * (numberCellsX + numberCellsY) - 1]) / cellLength;
	boundaryNormalsY[2 * (numberCellsX + numberCellsY) - 1] = (boundaryNodesX[0] - boundaryNodesX[2 * (numberCellsX + numberCellsY) - 1]) / cellLength;

	return 0;
}

unsigned int initializeDomain(double **boundaryNodesX, double **boundaryNodesY, double **boundaryNormalsX,
	double **boundaryNormalsY, double **boundaryVectorsX, double **boundaryVectorsY) {

	numberCellsX = domainLength / (0.7*meanFreePath);
	numberCellsY = domainHeight / (0.7*meanFreePath);

	//numberCellsX = domainLength / (meanFreePath);
	//numberCellsY = domainHeight / (meanFreePath);
	
	cellHeight = domainHeight / numberCellsY;
	cellLength = domainLength / numberCellsX;

	totalParticles = numberDensity * domainLength * domainHeight * domainWidth;
	totalSimulatedParticles = numberCellsX * numberCellsY * simulatedParticlesPerCell;
	similarityWeight = totalParticles / totalSimulatedParticles;

	printf("Total Particles = %d, DTime = %8e, Weight = %llu, Cell Length = %8e\n", totalSimulatedParticles, dt, similarityWeight, cellLength);
	fflush(stdin); // option ONE to clean stdin
	getchar();

	*boundaryNodesX = (double*)malloc(2 * (numberCellsX + numberCellsY) * sizeof(double));
	*boundaryNodesY = (double*)malloc(2 * (numberCellsX + numberCellsY) * sizeof(double));
	*boundaryNormalsX = (double*)malloc(2 * (numberCellsX + numberCellsY) * sizeof(double));
	*boundaryNormalsY = (double*)malloc(2 * (numberCellsX + numberCellsY) * sizeof(double));
	*boundaryVectorsX = (double*)malloc(2 * (numberCellsX + numberCellsY) * sizeof(double));
	*boundaryVectorsY = (double*)malloc(2 * (numberCellsX + numberCellsY) * sizeof(double));

	domainCoords(*boundaryNodesX, *boundaryNodesY, *boundaryNormalsX,
		*boundaryNormalsY, *boundaryVectorsX, *boundaryVectorsY);

	return 0;
}