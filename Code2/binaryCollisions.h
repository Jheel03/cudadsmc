#pragma once
#include <initialization.h>

void binaryCollisions(unsigned int nParticles, unsigned int mainAcc);

void initializeBinaryCollisions() {
	unsigned int nCells = NX * NY;
	unsigned int mainAcc = 0;
	for (unsigned int i = 0; i < nCells; i++) {
		binaryCollisions(IndexArray[i], mainAcc);
		mainAcc += IndexArray[i];
	}
	//printf(" >>");
}

void binaryCollisions(unsigned int nParticles, unsigned int mainAcc) {
	double r, PColl, CR, cRX, cRY, cRXTemp, cRYTemp, vCX, vCY;
	unsigned int m, n;
	unsigned int loopON = 1;
	unsigned int Zmax = (nParticles * (nParticles - 1) / 2) * ((Weight) / (DY * DX * DZ)) * RelVelRef * SigmaRef * Dt;
	//printf("\n%u, %u, %u, %8e, %8e, %8e", Zmax, Weight, nParticles, RelVelRef, SigmaRef, Dt);
	//fflush(stdin); // option ONE to clean stdin
	//getchar();
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
		CR = sqrt(pow(VelX[mainAcc + m] - VelX[mainAcc + n], 2) + pow(VelY[mainAcc + m] - VelY[mainAcc + n], 2));
		PColl = CR * pow((RelVelRef / CR), 2 * Ohmega - 1) / RelVelRef;
		if (PColl > static_cast <double> (rand()) / static_cast <double> (RAND_MAX)) {
			cRXTemp = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
			cRYTemp = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
			cRX = cRXTemp / sqrt(cRXTemp * cRXTemp + cRYTemp * cRYTemp);
			cRY = cRYTemp / sqrt(cRXTemp * cRXTemp + cRYTemp * cRYTemp);
			vCX = (VelX[mainAcc + m] + VelX[mainAcc + n]) / 2;
			vCY = (VelY[mainAcc + m] + VelY[mainAcc + n]) / 2;
			VelX[mainAcc + m] = vCX - (1 / 2) * CR * cRX;
			VelY[mainAcc + m] = vCY - (1 / 2) * CR * cRY;
			VelX[mainAcc + n] = vCX + (1 / 2) * CR * cRX;
			VelY[mainAcc + n] = vCY + (1 / 2) * CR * cRY;
		}
	}
}
