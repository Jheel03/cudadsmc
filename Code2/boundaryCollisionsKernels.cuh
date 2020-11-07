#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include <stdio.h>


__global__ void differetialVector(unsigned int n, double *dPositionX, double *dPositionY,
	double *velocityX, double *velocityY, double timeStep) {
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid < n) {
		dPositionX[tid] = velocityX[tid] * timeStep;
		dPositionY[tid] = velocityY[tid] * timeStep;
	}

}

__global__ void findVectorIntersection(double *dPositionX, double *dPositionY, double *boundaryVectorsX,
	double *boundaryVectorsY, double *boundaryNodesX, double *boundaryNodesY, double *positionX,
	double *positionY, double *intersectionParameter, unsigned int *intersectionFlag, unsigned int nx, unsigned int ny) {

	unsigned int tidx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int tidy = blockIdx.y*blockDim.y + threadIdx.y;

	if (tidx < nx && tidy < ny) {
		if (((dPositionX[tidx] * boundaryVectorsY[tidy]) - (boundaryVectorsX[tidy] * dPositionY[tidx])) != 0) {
			double tempX = boundaryNodesX[tidy] - positionX[tidx];
			double tempY = boundaryNodesY[tidy] - positionY[tidx];
			double tempCross = (dPositionX[tidx] * boundaryVectorsY[tidy]) - (boundaryVectorsX[tidy] * dPositionY[tidx]);
			double u = (tempX * dPositionY[tidx] - tempY * dPositionX[tidx]) / tempCross;
			double t = (tempX * boundaryVectorsY[tidy] - tempY * boundaryVectorsX[tidy]) / tempCross;
			if ((t >= 0) && (t <= 1) && (u >= 0) && (u <= 1)) {
				if (intersectionFlag[tidx] != 1) {
					intersectionFlag[tidx] = 1;
				}
				intersectionParameter[tidy*nx + tidx] = t;
			}
		}
	}
}

__global__ void findIntersectionWall(unsigned int n, unsigned int *intersectionFlag, double *intersectionParameter, 
	unsigned int nWalls, unsigned int *intersectionWallId, double *minimumIntersectionParameter, double *positionX, 
	double *positionY, double *dPositionX, double *dPositionY) {

	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < n) {
		if (intersectionFlag[tid] == 1) {
			double tempT = 1;
			for (unsigned int i = 0; i < nWalls; i++) {
				if (intersectionParameter[i*n + tid] < tempT) {
					tempT = intersectionParameter[i*n + tid];
					intersectionWallId[tid] = i;
				}
			}
			minimumIntersectionParameter[tid] = tempT;
		}
	}
}

__global__ void devCollisionCounter(unsigned int n, unsigned int *intersectionFlag, unsigned int *counter) {
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < n) {
		if (intersectionFlag[tid] == 1) {
			atomicAdd(counter, 1);
		}
	}
}


__global__ void vectorUpdate(unsigned int n, unsigned int *intersectionFlag, double *positionX,
	double *positionY, double *dPositionX, double *dPositionY, double *minimumIntersectionParameter) {
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid < n) {
		if (intersectionFlag[tid] == 1) {
			positionX[tid] = positionX[tid] + dPositionX[tid] * minimumIntersectionParameter[tid];
			positionY[tid] = positionY[tid] + dPositionY[tid] * minimumIntersectionParameter[tid];
		}
		else {
			positionX[tid] = positionX[tid] + dPositionX[tid];
			positionY[tid] = positionY[tid] + dPositionY[tid];
		}
	}
}

__global__ void velocityUpdate(unsigned int n, unsigned int *intersectionFlag, unsigned int *i,
	double *velocityX, double *velocityY, double *velocityZ, double *newVelocityN, double *newVelocityNX,
	double *newVelocityNY, double *boundaryVectorsX, double *boundaryVectorsY, double *boundaryNormalsX,
	double *boundaryNormalsY, unsigned int *intersectionWallId, double cellLength) {

	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < n) {
		if (intersectionFlag[tid] == 1) {
			unsigned int tempIdx = atomicAdd(i, 1);
			velocityX[tid] = fabsf(newVelocityN[tempIdx]) * boundaryNormalsX[intersectionWallId[tid]]
				+ newVelocityNX[tempIdx] * boundaryVectorsX[intersectionWallId[tid]] / cellLength;
			velocityY[tid] = fabsf(newVelocityN[tempIdx]) * boundaryNormalsY[intersectionWallId[tid]]
				+ newVelocityNX[tempIdx] * boundaryVectorsY[intersectionWallId[tid]] / cellLength;
			velocityZ[tid] = newVelocityNY[tempIdx];
		}
	}
}

__global__ void differentialVectorItr(unsigned long int n, double *dPositionX, double *dPositionY, double *velocityX,
	double *velocityY, double *minimumIntersectionParameter, unsigned int *intersectionFlag, double dt, double *newDt) {
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < n) {
		if (intersectionFlag[tid] == 1) {
			newDt[tid] = newDt[tid] * (1 - minimumIntersectionParameter[tid]);
			dPositionX[tid] = velocityX[tid] * newDt[tid];
			dPositionY[tid] = velocityY[tid] * newDt[tid];
		}
		else {
			dPositionX[tid] = 0;
			dPositionY[tid] = 0;
		}
	}
}

__global__ void findVectorIntersectionItr(double *dPositionX, double *dPositionY, double *boundaryVectorsX,
	double *boundaryVectorsY, double *boundaryNodesX, double *boundaryNodesY, double *positionX,
	double *positionY, double *intersectionParameter, unsigned int *intersectionFlag, unsigned int *tempIntersectionFlag, 
	unsigned int nx, unsigned int ny) {

	unsigned int tidx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int tidy = blockIdx.x*blockDim.x + threadIdx.y;

	if (tidx < nx && tidy < ny) {
		if (intersectionFlag[tidx] == 1) {
			if (((dPositionX[tidx] * boundaryVectorsY[tidy]) - (boundaryVectorsX[tidy] * dPositionY[tidx])) != 0) {
				double tempX = boundaryNodesX[tidy] - positionX[tidx];
				double tempY = boundaryNodesY[tidy] - positionY[tidx];
				double tempCross = (dPositionX[tidx] * boundaryVectorsY[tidy]) - (boundaryVectorsX[tidy] * dPositionY[tidx]);
				double u = (tempX * dPositionY[tidx] - tempY * dPositionX[tidx]) / tempCross;
				double t = (tempX * boundaryVectorsY[tidy] - tempY * boundaryVectorsX[tidy]) / tempCross;
				if (t > 0 && t <= 1 && u >= 0 && u <= 1) {
					if (tempIntersectionFlag[tidx] == 0) {
						tempIntersectionFlag[tidx] = 1;
					}
					intersectionParameter[tidy*nx + tidx] = t;
				}
			}
		}
	}
}

__global__ void findIntersectionWallItr(unsigned int n, unsigned int *intersectionFlag, double *intersectionParameter, 
	unsigned int nWalls, unsigned int *intersectionWallId, double *minimumIntersectionParameter, double *positionX, 
	double *positionY, double *dPositionX, double *dPositionY) {

	unsigned long int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < n) {
		if (intersectionFlag[tid] == 1) {
			double tempT = 1;
			for (unsigned int i = 0; i < nWalls; i++) {
				if (intersectionParameter[i*n + tid] < tempT) {
					tempT = intersectionParameter[i*n + tid];
					intersectionWallId[tid] = i;
				}
			}
			minimumIntersectionParameter[tid] = tempT;
		}
	}
}

__global__ void slightVectorUpdate(unsigned int n, unsigned int *intersectionFlag, double *positionX, double *positionY, 
	double *velocityX, double *velocityY) {
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < n) {
		if (intersectionFlag[tid] == 1) {
			positionX[tid] = positionX[tid] + velocityX[tid] * 1e-20;
			positionY[tid] = positionY[tid] + velocityY[tid] * 1e-20;
		}
	}
}

void bringBackParticle(unsigned int n, double *positionX, double *positionY, double domainLength, double domainHeight) {
	unsigned int loopON = 1;
	double r;
	for (unsigned int p = 0; p < n; p++) {
		if (positionX[p] <= 0 || positionX[p] >= domainLength || positionY[p] <= 0 || positionY[p] >= domainHeight) {
			//printf("\n%u", p);
			while (loopON == 1) {
				r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
				if (r != 0 && r != 1) {
					positionX[p] = domainLength * r;
					loopON = 0;
				}
			}
			loopON = 1;
			while (loopON == 1) {
				r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
				if (r != 0 && r != 1) {
					positionY[p] = domainHeight * r;
					loopON = 0;
				}
			}
		}
	}
}
