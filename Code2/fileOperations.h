#pragma once

#include <stdio.h>


int fileWriting(int nWalls, char fileName[30], double *boundaryNodesX, double *boundaryNodesY) {
	FILE *filePointer;
	filePointer = fopen(fileName, "w+");
	for (int i = 0; i < nWalls; i++) {
		fprintf(filePointer, "%6e, %6e\n", boundaryNodesX[i], boundaryNodesY[i]);
	}
	fclose(filePointer);
	return 0;
}
