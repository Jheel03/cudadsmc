#pragma once
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include <fileOperations.h>
#include <memoryOps.cuh>

constexpr auto BOLTZMANN_CONSTANT = 1.3806e-23;
constexpr auto AVOGADRO_CONSTANT = 6.022e23;
constexpr auto PI = 3.14159265359;
constexpr auto GasConstant = 8.1345;
constexpr auto MAX_PTL_CELL = 10;				//Max. particles in a cell;
constexpr auto MEAN_FREE_PATH_FACTOR = 1;
unsigned long TotalSimulatedParticles;

double X_LEN = 1e-5;
double Y_LEN = 1e-5;
double Z_LEN = 1e-6;							//Domain dimensions (m)
int NX;
int NY;											//No. of cells
int NWalls;
double DX, DY, DV;
double DZ = 0.01;								//Cell dimensions (m)

double Dt;
int NStep = 100000;
int FirstSamplingStep = 10000;
int SamplingPeriod = 1;
int PrintPeriod = 10000;

double UFree[3] = { 0, 0, 0 };

double Weight;
double MoleculeMass;
double MolarMass = 0.02897;						//kg/mole
double PFree = 101325;							//Pascal
double TFree = 273.0;							//Free stream temp. (K)
double NFree;									//Free stream number density
double TWall = 300.0;							//Wall temp. (K)

double OhmegaBar;
double Ohmega = 0.77;							//Temp. Viscosity Coeff.
double MuRef = 1.719e-5;						//Viscosity Coeff.
double DRef = 0;
double Alpha = 1;
double SigmaRef = 0;
double MeanFreePath = 0;
double StdDev;
double RelVelRef;
double MeanChaoticVel;

double *VelX, *VelY, *VelZ, *PosX, *PosY;
double *NewVelX, *NewVelY, *NewVelZ, *NewPosX, *NewPosY;
double *DevVelX, *DevVelY, *DevVelZ, *DevPosX, *DevPosY;
double *DevNewVelX, *DevNewVelY, *DevNewVelZ, *DevNewPosX, *DevNewPosY;

double* BoundaryNodesX, * BoundaryNodesY, * BoundaryNormalsX, * BoundaryNormalsY, * BoundaryVectorsX, * BoundaryVectorsY;
double* DevBoundaryVectorsX, * DevBoundaryVectorsY, * DevBoundaryNodesX, * DevBoundaryNodesY, * DevBoundaryNormalsX, * DevBoundaryNormalsY;

double* DevIntersectionParameter, * DevMinimumIntersectionParameter;
unsigned int* DevIntersectionFlag, * DevIntersectionWallId, * DevTempIntersectionFlag, * OutFlag, * DevOutFlag;
unsigned int* IntersectionFlag, *CollisionCounter;
unsigned int* DevCounter;
double* NewVelocitiesN, * NewVelocitiesNX, * NewVelocitiesNY;
double* DevDPositionX, * DevDPositionY;
double* NewDt;

char str1[50];
char str3[10];
char str2[10];

unsigned int* IndexArray;

void domainDefinition();
void generateParticles();
__global__ void scaleRandomNumber(double* deviceOutputPtr, unsigned int n, double init, double domainSize);
void freeEverything();
void plot(unsigned int j);


void Setup() {

	MoleculeMass = MolarMass / AVOGADRO_CONSTANT;
	NFree = PFree / (BOLTZMANN_CONSTANT * TFree);
	
	StdDev = sqrt(BOLTZMANN_CONSTANT * TFree / MoleculeMass);
	MeanChaoticVel = sqrt(8 * GasConstant * TFree / PI);
	OhmegaBar = 2 * Ohmega - 1;
	SigmaRef = 15 * MoleculeMass * sqrt(PI * GasConstant * TFree) / (2 * (5 - 2 * Ohmega) * (7 - 2 * Ohmega) * MuRef);
	RelVelRef = sqrt(4 * GasConstant * TFree) / pow(tgamma(5 / 2 - Ohmega), 1 / OhmegaBar);
	MeanFreePath = (1 / (sqrt(2) * SigmaRef * NFree * tgamma((4 - OhmegaBar) / 2))) * pow(sqrt(4 * GasConstant * TFree) / RelVelRef, OhmegaBar);
	Dt = MeanFreePath / MeanChaoticVel;
	
	printf("\nMolecule Mass = %6e, Mean Free Path = %6e, DT = %6e", MoleculeMass, MeanFreePath, Dt);

	NX = static_cast<int> (X_LEN / (MEAN_FREE_PATH_FACTOR * MeanFreePath));
	NY = static_cast<int> (Y_LEN / (MEAN_FREE_PATH_FACTOR * MeanFreePath));
	NWalls = 2 * (NX + NY);
	DX = static_cast<double> (X_LEN / NX);
	DY = static_cast<double> (Y_LEN / NY);
	DV = DX * DY * DZ;

	Weight = NFree * DV / MAX_PTL_CELL;
	TotalSimulatedParticles = static_cast<unsigned long> (MAX_PTL_CELL * NX * NY);

	printf("\nNX = %d, NY = %d, Weight = %6e, Total Sim. Particles = %lu", NX, NY, Weight, TotalSimulatedParticles);

	domainDefinition();
	generateParticles();

	IndexArray = (unsigned int*)malloc(NX * NY * sizeof(unsigned int));
	cudaMalloc(&NewDt, TotalSimulatedParticles * sizeof(double));

}


void domainDefinition() {

	BoundaryNodesX = (double*)malloc(NWalls * sizeof(double));
	BoundaryNodesY = (double*)malloc(NWalls * sizeof(double));
	BoundaryNormalsX = (double*)malloc(NWalls * sizeof(double));
	BoundaryNormalsY = (double*)malloc(NWalls * sizeof(double));
	BoundaryVectorsX = (double*)malloc(NWalls * sizeof(double));
	BoundaryVectorsY = (double*)malloc(NWalls * sizeof(double));
	cudaMalloc(&DevBoundaryNodesX, NWalls * sizeof(double));
	cudaMalloc(&DevBoundaryNodesY, NWalls * sizeof(double));
	cudaMalloc(&DevBoundaryVectorsX, NWalls * sizeof(double));
	cudaMalloc(&DevBoundaryVectorsY, NWalls * sizeof(double));
	cudaMalloc(&DevBoundaryNormalsX, NWalls * sizeof(double));
	cudaMalloc(&DevBoundaryNormalsY, NWalls * sizeof(double));

	for (int i = 0; i <= NX; i++) {
		BoundaryNodesX[i] = i * DX;
		BoundaryNodesY[i] = 0;
	}
	for (int i = 1; i <= NY; i++) {
		BoundaryNodesX[NX + i] = X_LEN;
		BoundaryNodesY[NX + i] = i * DY;
	}
	for (int i = 1; i <= NX; i++) {
		BoundaryNodesX[NX + NY + i] = X_LEN - i * DX;
		BoundaryNodesY[NX + NY + i] = Y_LEN;
	}
	for (int i = 1; i < NY; i++) {
		BoundaryNodesX[NX + NX + NY + i] = 0;
		BoundaryNodesY[NX + NX + NY + i] = Y_LEN - i * DY;
	}

	for (int i = 0; i < 2 * (NX + NY) - 1; i++) {
		BoundaryVectorsX[i] = BoundaryNodesX[i + 1] - BoundaryNodesX[i];
		BoundaryVectorsY[i] = BoundaryNodesY[i + 1] - BoundaryNodesY[i];
		BoundaryNormalsX[i] = -(BoundaryNodesY[i + 1] - BoundaryNodesY[i]) / DX;
		BoundaryNormalsY[i] = (BoundaryNodesX[i + 1] - BoundaryNodesX[i]) / DX;
	}
	BoundaryVectorsX[2 * (NX + NY) - 1] = BoundaryNodesX[0] - BoundaryNodesX[2 * (NX + NY) - 1];
	BoundaryVectorsY[2 * (NX + NY) - 1] = BoundaryNodesY[0] - BoundaryNodesY[2 * (NX + NY) - 1];
	BoundaryNormalsX[2 * (NX + NY) - 1] = -(BoundaryNodesY[0] - BoundaryNodesY[2 * (NX + NY) - 1]) / DX;
	BoundaryNormalsY[2 * (NX + NY) - 1] = (BoundaryNodesX[0] - BoundaryNodesX[2 * (NX + NY) - 1]) / DX;

	cudaMemcpy(DevBoundaryNodesX, BoundaryNodesX, NWalls * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(DevBoundaryNodesY, BoundaryNodesY, NWalls * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(DevBoundaryVectorsX, BoundaryVectorsX, NWalls * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(DevBoundaryVectorsY, BoundaryVectorsY, NWalls * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(DevBoundaryNormalsX, BoundaryNormalsX, NWalls * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(DevBoundaryNormalsY, BoundaryNormalsY, NWalls * sizeof(double), cudaMemcpyHostToDevice);

	fileWriting(NWalls, "plots/boundaryNodes.txt", BoundaryNodesX, BoundaryNodesY);
	fileWriting(NWalls, "plots/boundaryVectors.txt", BoundaryVectorsX, BoundaryVectorsY);
	fileWriting(NWalls, "plots/boundaryNormals.txt", BoundaryNormalsX, BoundaryNormalsY);

	OutFlag = (unsigned int*)malloc(TotalSimulatedParticles * sizeof(unsigned int));
	cudaMalloc(&DevIntersectionParameter, NWalls * TotalSimulatedParticles * sizeof(double));
	cudaMalloc(&DevMinimumIntersectionParameter, TotalSimulatedParticles * sizeof(double));
	cudaMalloc(&DevIntersectionFlag, TotalSimulatedParticles * sizeof(unsigned int));
	cudaMalloc(&DevTempIntersectionFlag, TotalSimulatedParticles * sizeof(unsigned int));
	cudaMalloc(&DevIntersectionWallId, TotalSimulatedParticles * sizeof(unsigned int));
	cudaMalloc(&DevOutFlag, TotalSimulatedParticles * sizeof(unsigned int));
	cudaMalloc(&DevCounter, sizeof(unsigned int));
	CollisionCounter = (unsigned int*)malloc(sizeof(unsigned int));
	IntersectionFlag = (unsigned int*)malloc(TotalSimulatedParticles * sizeof(unsigned int));

}


void generateParticles() {

	VelX = (double*)malloc(TotalSimulatedParticles * sizeof(double));
	VelY = (double*)malloc(TotalSimulatedParticles * sizeof(double));
	VelZ = (double*)malloc(TotalSimulatedParticles * sizeof(double));
	PosX = (double*)malloc(TotalSimulatedParticles * sizeof(double));
	PosY = (double*)malloc(TotalSimulatedParticles * sizeof(double));
	NewVelX = (double*)malloc(TotalSimulatedParticles * sizeof(double));
	NewVelY = (double*)malloc(TotalSimulatedParticles * sizeof(double));
	NewVelZ = (double*)malloc(TotalSimulatedParticles * sizeof(double));
	NewPosX = (double*)malloc(TotalSimulatedParticles * sizeof(double));
	NewPosY = (double*)malloc(TotalSimulatedParticles * sizeof(double));

	cudaMalloc(&DevVelX, TotalSimulatedParticles * sizeof(double));
	cudaMalloc(&DevVelY, TotalSimulatedParticles * sizeof(double));
	cudaMalloc(&DevVelZ, TotalSimulatedParticles * sizeof(double));
	cudaMalloc(&DevPosX, TotalSimulatedParticles * sizeof(double));
	cudaMalloc(&DevPosY, TotalSimulatedParticles * sizeof(double));
	cudaMalloc(&DevDPositionX, TotalSimulatedParticles * sizeof(double));
	cudaMalloc(&DevDPositionY, TotalSimulatedParticles * sizeof(double));
	cudaMalloc(&DevNewVelX, TotalSimulatedParticles * sizeof(double));
	cudaMalloc(&DevNewVelY, TotalSimulatedParticles * sizeof(double));
	cudaMalloc(&DevNewVelZ, TotalSimulatedParticles * sizeof(double));
	cudaMalloc(&DevNewPosY, TotalSimulatedParticles * sizeof(double));
	cudaMalloc(&DevNewPosY, TotalSimulatedParticles * sizeof(double));

	dim3 blockSize(512);
	dim3 gridSize(TotalSimulatedParticles / 512 + 1);

	double* deviceOutputPtr;
	cudaMalloc(&deviceOutputPtr, TotalSimulatedParticles * sizeof(double));
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32);
	unsigned long long seed;

	seed = 12345;
	curandSetPseudoRandomGeneratorSeed(generator, seed);
	curandStatus_t posGeneratorStatusX = curandGenerateUniformDouble(generator, deviceOutputPtr, TotalSimulatedParticles);
	cudaDeviceSynchronize();
	scaleRandomNumber << <gridSize, blockSize >> > (deviceOutputPtr, TotalSimulatedParticles, 0, X_LEN);
	cudaDeviceSynchronize();
	cudaMemcpy(PosX, deviceOutputPtr, TotalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(DevPosX, deviceOutputPtr, TotalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();

	seed = 23456;
	curandSetPseudoRandomGeneratorSeed(generator, seed);
	curandStatus_t posGeneratorStatusY = curandGenerateUniformDouble(generator, deviceOutputPtr, TotalSimulatedParticles);
	cudaDeviceSynchronize();
	scaleRandomNumber << <gridSize, blockSize >> > (deviceOutputPtr, TotalSimulatedParticles, 0, Y_LEN);
	cudaDeviceSynchronize();
	cudaMemcpy(PosY, deviceOutputPtr, TotalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(DevPosY, deviceOutputPtr, TotalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();

	seed = 34567;
	curandSetPseudoRandomGeneratorSeed(generator, seed);
	curandStatus_t velGeneratorStatusX = curandGenerateNormalDouble(generator, deviceOutputPtr, TotalSimulatedParticles, UFree[0], StdDev);
	cudaDeviceSynchronize();
	cudaMemcpy(VelX, deviceOutputPtr, TotalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(DevVelX, deviceOutputPtr, TotalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();

	seed = 45678;
	curandSetPseudoRandomGeneratorSeed(generator, seed);
	curandStatus_t velGeneratorStatusY = curandGenerateNormalDouble(generator, deviceOutputPtr, TotalSimulatedParticles, UFree[1], StdDev);
	cudaDeviceSynchronize();
	cudaMemcpy(VelY, deviceOutputPtr, TotalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(DevVelY, deviceOutputPtr, TotalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();

	seed = 56789;
	curandSetPseudoRandomGeneratorSeed(generator, seed);
	curandStatus_t velGeneratorStatusZ = curandGenerateNormalDouble(generator, deviceOutputPtr, TotalSimulatedParticles, UFree[2], StdDev);
	cudaDeviceSynchronize();
	cudaMemcpy(VelZ, deviceOutputPtr, TotalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(DevVelZ, deviceOutputPtr, TotalSimulatedParticles * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();

	curandDestroyGenerator(generator);
	cudaFree(deviceOutputPtr);

}


__global__ void scaleRandomNumber(double* deviceOutputPtr, unsigned int n, double init, double domainSize) {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		deviceOutputPtr[tid] = init + domainSize * deviceOutputPtr[tid];
	}
}

void plot(unsigned int j) {
	setString(str1, '\0', 50);
	setString(str2, '\0', 10);
	setString(str3, '\0', 10);
	assignString(str1, "plots/", 50);
	assignString(str3, ".txt", 10);
	_itoa_s(j, str2, 10);
	strcat(str1, str2);
	strcat(str1, str3);
	fileWriting(TotalSimulatedParticles, str1, PosX, PosY);
}

void freeEverything() {
	cudaDeviceReset();
	free(VelX);
	free(VelY);
	free(VelZ);
	free(PosX);
	free(PosY);
	free(NewPosX);
	free(NewPosY);
	free(NewVelX);
	free(NewVelY);
	free(NewVelZ);
	free(BoundaryNodesX);
	free(BoundaryNodesY);
	free(BoundaryNormalsX);
	free(BoundaryNormalsY);
	free(BoundaryVectorsX);
	free(BoundaryVectorsY);
}
