#include <stdio.h>
#include <particleGenerator.cuh>
#include <fileOperations.h>
#include <memoryOps.cuh>
#include <string.h>
#include <boundaryCollisions.cuh>
#include <indexing.h>
#include <binaryCollisions.h>


int main() {

	cudaError_t error;

	//////////////Initialization/////////////////
	initializeGasState();
	double *boundaryNodesX, *boundaryNodesY, *boundaryNormalsX, *boundaryNormalsY, *boundaryVectorsX, *boundaryVectorsY;
	unsigned int* indexArray;
	initializeDomain(&boundaryNodesX, &boundaryNodesY, &boundaryNormalsX, &boundaryNormalsY, &boundaryVectorsX, &boundaryVectorsY);
	unsigned int nWalls = 2 * (numberCellsX + numberCellsX);
	indexArray = (unsigned int*)malloc(numberCellsX * numberCellsY * sizeof(unsigned int));

	/////////////////////////////////////////////
	fileWriting(2 * (numberCellsX + numberCellsY), "plots/boundaryNodes.txt", boundaryNodesX, boundaryNodesY);
	fileWriting(2 * (numberCellsX + numberCellsY), "plots/boundaryVectors.txt", boundaryVectorsX, boundaryVectorsY);
	fileWriting(2 * (numberCellsX + numberCellsY), "plots/boundaryNormals.txt", boundaryNormalsX, boundaryNormalsY);
	
	/////////////Particles Generation//////////////
	double *velocityX, *velocityY, *velocityZ, *positionX, *positionY, *newPositionX, *newPositionY, *newVelocityX, *newVelocityY;
	particlesGeneratorInitialize(&velocityX, &velocityY, &velocityZ, &positionX, &positionY);
	///////////////////////////////////////////////
	fileWriting(totalSimulatedParticles, "plots/1.txt", positionX, positionY);

	/*for (unsigned int i = 0; i < totalSimulatedParticles; i++) {
		printf("\n%d: (%4e, %4e, %4e, %4e, %4e)", i, positionX[i], positionX[i], velocityX[i], velocityY[i], velocityZ[i]);
	}*/

	///////////////////Initializing Device Arrays//////////////////
	double *devPositionX, *devPositionY, *devVelocityX, *devVelocityY, *devVelocityZ;
	double *dPositionX, *dPositionY, *devDPositionX, *devDPositionY;
	double *devNewPositionX, *devNewPositionY;
	double *newDt;
	dPositionX = (double*)malloc(totalSimulatedParticles * sizeof(double));
	dPositionY = (double*)malloc(totalSimulatedParticles * sizeof(double));
	newPositionX = (double*)malloc(totalSimulatedParticles * sizeof(double));
	newPositionY = (double*)malloc(totalSimulatedParticles * sizeof(double));
	newVelocityX = (double*)malloc(totalSimulatedParticles * sizeof(double));
	newVelocityY = (double*)malloc(totalSimulatedParticles * sizeof(double));
	cudaMalloc(&devPositionX, totalSimulatedParticles * sizeof(double));
	cudaMalloc(&devPositionY, totalSimulatedParticles * sizeof(double));
	cudaMalloc(&devDPositionX, totalSimulatedParticles * sizeof(double));
	cudaMalloc(&devDPositionY, totalSimulatedParticles * sizeof(double));
	cudaMalloc(&devVelocityX, totalSimulatedParticles * sizeof(double));
	cudaMalloc(&devVelocityY, totalSimulatedParticles * sizeof(double));
	cudaMalloc(&devVelocityZ, totalSimulatedParticles * sizeof(double));
	cudaMalloc(&devNewPositionX, totalSimulatedParticles * sizeof(double));
	cudaMalloc(&devNewPositionY, totalSimulatedParticles * sizeof(double));
	cudaMalloc(&newDt, totalSimulatedParticles * sizeof(double));

	cudaMemcpy(devPositionX, positionX, totalSimulatedParticles * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devPositionY, positionY, totalSimulatedParticles * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devVelocityX, velocityX, totalSimulatedParticles * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devVelocityY, velocityY, totalSimulatedParticles * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devVelocityZ, velocityZ, totalSimulatedParticles * sizeof(double), cudaMemcpyHostToDevice);

	double *devBoundaryVectorsX, *devBoundaryVectorsY, *devBoundaryNodesX,
		*devBoundaryNodesY, *devBoundaryNormalsX, *devBoundaryNormalsY;
	cudaMalloc(&devBoundaryNodesX, nWalls * sizeof(double));
	cudaMalloc(&devBoundaryNodesY, nWalls * sizeof(double));
	cudaMalloc(&devBoundaryVectorsX, nWalls * sizeof(double));
	cudaMalloc(&devBoundaryVectorsY, nWalls * sizeof(double));
	cudaMalloc(&devBoundaryNormalsX, nWalls * sizeof(double));
	cudaMalloc(&devBoundaryNormalsY, nWalls * sizeof(double));

	cudaMemcpy(devBoundaryNodesX, boundaryNodesX, nWalls * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devBoundaryNodesY, boundaryNodesY, nWalls * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devBoundaryVectorsX, boundaryVectorsX, nWalls * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devBoundaryVectorsY, boundaryVectorsY, nWalls * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devBoundaryNormalsX, boundaryNormalsX, nWalls * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devBoundaryNormalsY, boundaryNormalsY, nWalls * sizeof(double), cudaMemcpyHostToDevice);
	////////////////////////////////////////////////////////////////


	///////////////////Initializing BC Variables/////////////////////
	double *intersectionParameter, *minimumIntersectionParameter;
	unsigned int *intersectionFlag, *intersectionWallId, *tempIntersectionFlag, *outFlag, *devOutFlag;
	outFlag = (unsigned int*)malloc(totalSimulatedParticles * sizeof(unsigned int));
	cudaMalloc(&intersectionParameter, nWalls * totalSimulatedParticles * sizeof(double));
	cudaMalloc(&minimumIntersectionParameter, totalSimulatedParticles * sizeof(double));
	cudaMalloc(&intersectionFlag, totalSimulatedParticles * sizeof(unsigned int));
	cudaMalloc(&tempIntersectionFlag, totalSimulatedParticles * sizeof(unsigned int));
	cudaMalloc(&intersectionWallId, totalSimulatedParticles * sizeof(unsigned int));
	cudaMalloc(&devOutFlag, totalSimulatedParticles * sizeof(unsigned int));
	unsigned int *hostIntersectionFlag;
	unsigned int *devCounter;								//Here atomicAdd doesnot support unsigned long
	cudaMalloc(&devCounter, sizeof(unsigned int));
	unsigned int *hostCollisionCounter;
	hostCollisionCounter = (unsigned int*)malloc(sizeof(unsigned int));
	hostIntersectionFlag = (unsigned int*)malloc(totalSimulatedParticles * sizeof(unsigned int));
	double *newVelocitiesN, *newVelocitiesNX, *newVelocitiesNY;
	////////////////////////////////////////////////////////////////


	char str1[50];
	char str3[10];
	char str2[10];

	dim3 blockSize;
	dim3 gridSize;

	////////////////Main Loop Starts///////////////
	for (unsigned int j = 2; j <= 200; j++) {

		boundaryCollisionsInitialize(&totalSimulatedParticles, positionX, positionY, velocityX, velocityY, velocityZ,
			devPositionX, devPositionY, devVelocityX, devVelocityY, devVelocityZ, intersectionFlag, tempIntersectionFlag,
			intersectionParameter, minimumIntersectionParameter, nWalls, devCounter, newDt, dt, &newVelocitiesN, &newVelocitiesNX, 
			&newVelocitiesNY, devDPositionX, devDPositionY, devBoundaryNodesX, devBoundaryNodesY, devBoundaryVectorsX, 
			devBoundaryVectorsY, devBoundaryNormalsX, devBoundaryNormalsY, intersectionWallId, j, hostCollisionCounter);

		indexingInitialization(&totalSimulatedParticles, numberCellsX, numberCellsY, cellHeight, cellLength, positionX, positionY, 
			velocityX, velocityY, newPositionX, newPositionY, indexArray, newVelocityX, newVelocityY);

		cudaMemcpy(positionX, newPositionX, totalSimulatedParticles * sizeof(double), cudaMemcpyHostToHost);
		cudaMemcpy(positionY, newPositionY, totalSimulatedParticles * sizeof(double), cudaMemcpyHostToHost);
		cudaMemcpy(velocityX, newVelocityX, totalSimulatedParticles * sizeof(double), cudaMemcpyHostToHost);
		cudaMemcpy(velocityY, newVelocityY, totalSimulatedParticles * sizeof(double), cudaMemcpyHostToHost);

		initializeBinaryCollisions(indexArray, velocityX, velocityY);
		cudaMemcpy(velocityX, newVelocityX, totalSimulatedParticles * sizeof(double), cudaMemcpyHostToHost);
		cudaMemcpy(velocityY, newVelocityY, totalSimulatedParticles * sizeof(double), cudaMemcpyHostToHost);


		setString(str1, '\0', 50);
		setString(str2, '\0', 10);
		setString(str3, '\0', 10);
		assignString(str1, "plots/", 50);
		assignString(str3, ".txt", 10);
		_itoa_s(j, str2, 10);
		strcat(str1, str2);
		strcat(str1, str3);
		//printf("\n%s", str1);
		fileWriting(totalSimulatedParticles, str1, positionX, positionY);
		
	}

	/*cudaFree(devPositionX);
	cudaFree(devPositionY);
	cudaFree(devDPositionX);
	cudaFree(devDPositionY);
	cudaFree(devVelocityX);
	cudaFree(devVelocityY);
	cudaFree(devVelocityZ);*/
	cudaDeviceReset();

	free(positionX);
	free(positionY);
	free(velocityX);
	free(velocityY);
	free(velocityZ);
	free(dPositionX);
	free(dPositionY);
	free(boundaryNodesX);
	free(boundaryNodesY);
	free(boundaryNormalsX);
	free(boundaryNormalsY);
	free(boundaryVectorsX);
	free(boundaryVectorsY);
	free(newPositionX);
	free(newPositionY);
	free(newVelocityX);
	free(newVelocityY);
	free(indexArray);

	return 0;
}




