#pragma once
#include <math.h>
#include <constants.h>
#include <random>



void binaryCollisions(unsigned int nParticles, unsigned int mainAcc, double *velocityX, double *velocityY) {
	double CRef = sqrt(4 * gasConstant * TFree) / static_cast <double> (pow(tgamma(5 / 2 - ohmega), 1 / (2 * ohmega - 1)));
	//double sigmaRef = 1;
	double r, PColl, CR, cRX, cRY, cRXTemp, cRYTemp, vCX, vCY;
	unsigned int m, n;
	unsigned int loopON = 1;
	unsigned int Zmax = (nParticles * (nParticles - 1) / 2) * ((totalParticles / totalSimulatedParticles) / (cellHeight*cellLength*domainWidth)) * CRef * sigmaRef * dt;
	printf("%u, %u, %u, %8e, %8e, %8e", Zmax, totalParticles / totalSimulatedParticles, nParticles, CRef, sigmaRef, dt);
	fflush(stdin); // option ONE to clean stdin
	getchar();
	for (unsigned int i = 0; i < Zmax; i++) {
		while (loopON == 1) {
			r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
			if (r != 1) {
				loopON = 0;
			}
		}
		m = r * nParticles;
		while (loopON == 1) {
			r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
			if (r != 1) {
				loopON = 0;
			}
		}
		n = r * nParticles;
		CR = sqrt(pow(velocityX[mainAcc + m] - velocityX[mainAcc + n], 2) + pow(velocityY[mainAcc + m] - velocityY[mainAcc + n], 2));
		PColl = CR * pow((CRef / CR), 2 * ohmega - 1) / CRef;
		if (PColl > static_cast <double> (rand()) / static_cast <double> (RAND_MAX)) {
			cRXTemp = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
			cRYTemp = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
			cRX = cRXTemp / sqrt(cRXTemp*cRXTemp + cRYTemp*cRYTemp);
			cRY = cRYTemp / sqrt(cRXTemp*cRXTemp + cRYTemp*cRYTemp);
			vCX = (velocityX[mainAcc + m] + velocityX[mainAcc + n]) / 2;
			vCY = (velocityY[mainAcc + m] + velocityY[mainAcc + n]) / 2;
			velocityX[mainAcc + m] = vCX - (1 / 2) * CR * cRX;
			velocityY[mainAcc + m] = vCY - (1 / 2) * CR * cRY;
			velocityX[mainAcc + n] = vCX + (1 / 2) * CR * cRX;
			velocityY[mainAcc + n] = vCY + (1 / 2) * CR * cRY;
		}
	}
}


void initializeBinaryCollisions(unsigned int *indexArray, double* velocityX, double* velocityY) {
	unsigned int nCells = numberCellsX * numberCellsY;
	unsigned int mainAcc = 0;
	for (unsigned int i = 0; i < nCells; i++) {
		binaryCollisions(indexArray[i], mainAcc, velocityX, velocityY);
		mainAcc += indexArray[i];
	}
	printf("\n >>");
}