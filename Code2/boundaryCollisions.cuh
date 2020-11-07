#include <boundaryCollisionsKernels.cuh>
#include <memoryOps.cuh>
#include <rayleighDistribution.cuh>
#include <math.h>
#include <constants.h>


int boundaryCollisionsInitialize(unsigned int *totalSimulatedParticles, double *positionX, double *positionY, double *velocityX, 
	double *velocityY, double *velocityZ, double *devPositionX, double *devPositionY, double *devVelocityX, double *devVelocityY, 
	double *devVelocityZ, unsigned int *intersectionFlag, unsigned int *tempIntersectionFlag, double *intersectionParameter, 
	double *minimumIntersectionParameter, unsigned int nWalls, unsigned int *devCounter, double *newDt, double dt, 
	double **newVelocitiesN, double **newVelocitiesNX, double **newVelocitiesNY, double *devDPositionX, double *devDPositionY, 
	double *devBoundaryNodesX, double *devBoundaryNodesY, double *devBoundaryVectorsX, double *devBoundaryVectorsY, 
	double *devBoundaryNormalsX, double *devBoundaryNormalsY, unsigned int *intersectionWallId, unsigned int j, 
	unsigned int *hostCollisionCounter) {

	dim3 blockSize;
	dim3 gridSize;
	
	blockSize.x = 512;
	gridSize.x = *totalSimulatedParticles / 512 + 1;
	differetialVector << <gridSize, blockSize >> > (*totalSimulatedParticles, devDPositionX, devDPositionY,
		devVelocityX, devVelocityY, dt);
	cudaDeviceSynchronize();

	cudaMemset(intersectionFlag, 0, *totalSimulatedParticles * sizeof(unsigned int));
	userMemset(intersectionParameter, 3.0, nWalls * *totalSimulatedParticles);
	cudaDeviceSynchronize();

	blockSize.x = 32;
	blockSize.y = 32;
	gridSize.x = *totalSimulatedParticles / 32 + 1;
	gridSize.y = nWalls / 32 + 1;
	findVectorIntersection << <gridSize, blockSize >> > (devDPositionX, devDPositionY, devBoundaryVectorsX,
		devBoundaryVectorsY, devBoundaryNodesX, devBoundaryNodesY, devPositionX, devPositionY, intersectionParameter,
		intersectionFlag, *totalSimulatedParticles, nWalls);
	cudaDeviceSynchronize();

	blockSize.x = 512;
	blockSize.y = 1;
	gridSize.x = *totalSimulatedParticles / 512 + 1;
	gridSize.y = 1;
	findIntersectionWall << <gridSize, blockSize >> > (*totalSimulatedParticles, intersectionFlag, intersectionParameter, 2 * (numberCellsX + numberCellsY),
		intersectionWallId, minimumIntersectionParameter, devPositionX, devPositionY, devDPositionX, devDPositionY);
	cudaDeviceSynchronize();

	vectorUpdate << <gridSize, blockSize >> > (*totalSimulatedParticles, intersectionFlag, devPositionX, devPositionY,
		devDPositionX, devDPositionY, minimumIntersectionParameter);
	cudaDeviceSynchronize();

	cudaMemset(devCounter, 0, sizeof(unsigned int));
	devCollisionCounter << <gridSize, blockSize >> > (*totalSimulatedParticles, intersectionFlag, devCounter);
	cudaDeviceSynchronize();

	cudaMemcpy(hostCollisionCounter, devCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	//printf("\nNo. of Collisions = %u", *hostCollisionCounter);

	userMemset(newDt, dt, *totalSimulatedParticles);
	cudaDeviceSynchronize();

	///////////////////////////New Velocities//////////////////////////////
	while (*hostCollisionCounter > 0) {

		cudaMalloc(newVelocitiesN, *hostCollisionCounter * sizeof(double));
		cudaMalloc(newVelocitiesNX, *hostCollisionCounter * sizeof(double));
		cudaMalloc(newVelocitiesNY, *hostCollisionCounter * sizeof(double));

		unsigned long long seed;
		seed = j * 12345;
		initializeRayleigh(*hostCollisionCounter, *newVelocitiesN, sqrt(gasConstant*wallTemp), seed);

		curandGenerator_t generator;
		curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32);

		seed = j * 23456;
		curandSetPseudoRandomGeneratorSeed(generator, seed);
		curandGenerateNormalDouble(generator, *newVelocitiesNX, *hostCollisionCounter, 0, sqrt(gasConstant*wallTemp));
		cudaDeviceSynchronize();

		seed = j * 34567;
		curandSetPseudoRandomGeneratorSeed(generator, seed);
		curandGenerateNormalDouble(generator, *newVelocitiesNY, *hostCollisionCounter, 0, sqrt(gasConstant*wallTemp));
		cudaDeviceSynchronize();

		curandDestroyGenerator(generator);

		cudaMemset(devCounter, 0, sizeof(unsigned int));
		velocityUpdate << <gridSize, blockSize >> > (*totalSimulatedParticles, intersectionFlag, devCounter,
			devVelocityX, devVelocityY, devVelocityZ, *newVelocitiesN, *newVelocitiesNX,
			*newVelocitiesNY, devBoundaryVectorsX, devBoundaryVectorsY, devBoundaryNormalsX,
			devBoundaryNormalsY, intersectionWallId, cellLength);
		cudaDeviceSynchronize();

		cudaFree(*newVelocitiesN);
		cudaFree(*newVelocitiesNX);
		cudaFree(*newVelocitiesNY);

		//slightVectorUpdate << <gridSize, blockSize >> >(*totalSimulatedParticles, intersectionFlag, devPositionX,
		//	devPositionY, devVelocityX, devVelocityY);
		//cudaDeviceSynchronize();

		cudaMemcpy(velocityX, devVelocityX, *totalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(velocityY, devVelocityY, *totalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(velocityZ, devVelocityZ, *totalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		

		differentialVectorItr << <gridSize, blockSize >> > (*totalSimulatedParticles, devDPositionX, devDPositionY, devVelocityX,
			devVelocityY, minimumIntersectionParameter, intersectionFlag, dt, newDt);
		cudaDeviceSynchronize();

		userMemset(intersectionParameter, 3.0, nWalls * *totalSimulatedParticles);
		userMemset(minimumIntersectionParameter, 3.0, *totalSimulatedParticles);
		cudaDeviceSynchronize();

		cudaMemset(tempIntersectionFlag, 0, *totalSimulatedParticles * sizeof(unsigned int));

		blockSize.x = 32;
		blockSize.y = 32;
		gridSize.x = *totalSimulatedParticles / 32 + 1;
		gridSize.y = nWalls / 32 + 1;
		findVectorIntersectionItr << <gridSize, blockSize >> > (devDPositionX, devDPositionY, devBoundaryVectorsX,
			devBoundaryVectorsY, devBoundaryNodesX, devBoundaryNodesY, devPositionX, devPositionY,
			intersectionParameter, intersectionFlag, tempIntersectionFlag, *totalSimulatedParticles, nWalls);
		cudaDeviceSynchronize();

		cudaMemset(intersectionFlag, 0, *totalSimulatedParticles * sizeof(unsigned int));
		cudaMemcpy(intersectionFlag, tempIntersectionFlag, *totalSimulatedParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

		cudaMemset(devCounter, 0, sizeof(unsigned int));
		blockSize.x = 512;
		blockSize.y = 1;
		gridSize.x = *totalSimulatedParticles / 512 + 1;
		gridSize.y = 1;
		devCollisionCounter << <gridSize, blockSize >> > (*totalSimulatedParticles, intersectionFlag, devCounter);
		cudaDeviceSynchronize();

		cudaMemcpy(hostCollisionCounter, devCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		//printf(" >> %u", *hostCollisionCounter);

		findIntersectionWallItr << <gridSize, blockSize >> > (*totalSimulatedParticles, intersectionFlag, intersectionParameter, 
			nWalls, intersectionWallId, minimumIntersectionParameter, devPositionX, devPositionY, devDPositionX, devDPositionY);
		cudaDeviceSynchronize();

		vectorUpdate << <gridSize, blockSize >> > (*totalSimulatedParticles, intersectionFlag, devPositionX, devPositionY,
			devDPositionX, devDPositionY, minimumIntersectionParameter);
		cudaDeviceSynchronize();

		/*if (*hostCollisionCounter > 0) {
			int ans;
			printf("\nAnother Collision");
			scanf("%d", &ans);
		}*/

	}

	cudaMemcpy(positionX, devPositionX, *totalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(positionY, devPositionY, *totalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToHost);
	bringBackParticle(*totalSimulatedParticles, positionX, positionY, domainLength, domainHeight);
	cudaMemcpy(devPositionX, positionX, *totalSimulatedParticles * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devPositionY, positionY, *totalSimulatedParticles * sizeof(double), cudaMemcpyHostToDevice);


	return 0;
}