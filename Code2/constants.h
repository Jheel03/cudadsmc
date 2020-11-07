#pragma once

///////////////////Natural Values/////////////////////

constexpr auto PI = 3.14159265359;
double TFree = 293.15;				//Free Stream Temperature (K)
double PFree = 101.325e3;			//Free Stream Pressure (Pa)
double UFree[3] = { 0, 0, 0 };		//Free Stream Velocity {x, y, z}
double MMass = 0;			//Gas Molecular Mass kg/mole (air)
double MMass_g = 0;
double gasConstant = 8.314;			//Per gram mole
double Avogadro = 6.022e23;			//Per gram mole
double Boltzmann = 1.3806e-23;
double stdDev = 0;					//Standard Deviation of Velocity Distribution
double moleculeMass = 48.1e-27;		//kg per molecule
double molecularDia = 0;
double molecularCS = 0;			//Molecular Cross-Sectional Area (air)
double meanFreePath = 0;
double numberDensity = 0;
unsigned int totalParticles = 0;
unsigned int simulatedParticlesPerCell = 20;
unsigned int totalSimulatedParticles = 0;
double similarityWeight = 0;
double dt;							//Time Step

//double domainWidth = 1e-6;
//double domainHeight = 1e-5;
//double domainLength = 1e-5;
double domainWidth = 100;
double domainHeight = 200;
double domainLength = 200;
unsigned int numberCellsX = 0;				//No. of cells along length
unsigned int numberCellsY = 0;				//No. of cells along height
double cellLength = 0;
double cellHeight = 0;
double wallTemp = 500;


/////////Collision Parameters///////////
double ohmega = 0.77;
double muRef = 1.719; //x1e5
double dRef = 0;
double alpha = 1;
double sigmaRef = 0;

/////////////////Artificial Values///////////////////

//constexpr auto PI = 3.14159265359;
//double TFree = 293.15;				//Free Stream Temperature (K)
//double PFree = 101.325e3;			//Free Stream Pressure (Pa)
//double UFree[3] = { 0, 0, 0 };		//Free Stream Velocity {x, y, z}
//double MMass = 28.97e-3;			//Gas Molecular Mass kg/mole (air)
//double gasConstant = 8.314;			//Per gram mole
//double Avogadro = 6.022e23;			//Per gram mole
//double Boltzmann = 1.3806e-23;
//double stdDev = 0;					//Standard Deviation of Velocity Distribution
//double moleculeMass = 0;
//double molecularDia = 0;
//double molecularCS = 1e-19;			//Molecular Cross-Sectional Area (air)
//double meanFreePath = 0;
//double numberDensity = 0;
//unsigned int totalParticles = 0;
//unsigned int simulatedParticlesPerCell = 10;
//unsigned int totalSimulatedParticles = 0;
//double dt;							//Time Step
//
//double domainWidth = 1;
//double domainHeight = 50;
//double domainLength = 50;
//unsigned int numberCellsX = 0;				//No. of cells along length
//unsigned int numberCellsY = 0;				//No. of cells along height
//double cellLength = 0;
//double cellHeight = 0;
//double wallTemp = 300;
