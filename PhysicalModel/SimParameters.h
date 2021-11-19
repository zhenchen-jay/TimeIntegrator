#ifndef SIMPARAMETERS_H
#define SIMPARAMETERS_H

#include <cmath>
#include <vector>

struct SimParameters
{
    SimParameters()
    {
        timeStep = 5e-3;
        integrator = TI_NEWMARK;
        NewtonMaxIters = 20;
        NewtonTolerance = 1e-8;

        gravityEnabled = false;
        gravityG = -0.98;
        springsEnabled = true;
        springStiffness = 1e3;
        maxSpringStrain = 0.2;
        dampingEnabled = true;
        dampingStiffness = 1.0;
        floorEnabled = false;
		frictionEnabled = false;

        youngs = 1000;
        poisson = 0.3;
        elasticEnabled = true;
        internalContactEnabled = false;
        

        particleMass = 1.0;
        maxSpringDist = 1.0;

        ceil = 0;
        TRBDF2_gamma = 1 - std::sqrt(2) / 2;
        NM_gamma = 0.5;
        NM_beta = 0.25;

        modelType = MT_POGO_STICK;
        materialType = MT_NEOHOOKEAN;
        barrierStiffness = youngs;
        barrierEps = 0.01;

        totalTime = 10;
        totalNumIter = totalTime / timeStep;

        numSegs = 2;
       /* topLine = 20;
        barLen = 10;
        barHeight = 5;*/
        topLine = 10;
        barLen = 2;
        barHeight = 1;

        numSpectra = 10;
        isSaveInfo = true;
        
        youngsType = YT_CONSTANT;
        youngsList.resize(youngs, youngs);

        impulsePow = 10;
        impulseMag = 0;

        splitRatio = 0.5;
        rho = 1.0;
    }

    enum TimeIntegrator {TI_EXPLICIT_EULER, TI_VELOCITY_VERLET, TI_RUNGE_KUTTA, TI_EXP_ROSENBROCK_EULER, TI_IMPLICIT_EULER, TI_IMPLICIT_MIDPOINT, TI_TRAPEZOID, TI_TR_BDF2, TI_BDF2, TI_NEWMARK, TI_ADDITIVE, TI_SPLIT, TI_PROJECTION, TI_COMPOSITE};
    enum ModelType {MT_HARMONIC_1D, MT_POGO_STICK};
    enum MaterialType {MT_LINEAR, MT_StVK, MT_NEOHOOKEAN};
    enum YoungsType {YT_CONSTANT, YT_LINEAR, YT_RANDOM};

    double timeStep;
    TimeIntegrator integrator;
    double NewtonTolerance;
    int NewtonMaxIters;
    
    bool gravityEnabled;
    double gravityG;
    bool springsEnabled;
    double springStiffness;
    double maxSpringStrain;
    bool floorEnabled;
	bool frictionEnabled;
    bool dampingEnabled;
    double dampingStiffness;

    double particleMass;
    double maxSpringDist;
    double sawRadius;

    double ceil;
    double TRBDF2_gamma;
    double NM_gamma;
    double NM_beta;
    double rho;

    double barrierStiffness;
    double barrierEps;

    ModelType modelType;
    MaterialType materialType;
    YoungsType youngsType;

    double totalTime;   // total simulation time
    int totalNumIter;   // total simulation steps

    int numSegs;    // number of segments

    double youngs;
    std::vector<double> youngsList;
    double poisson;

    bool elasticEnabled;
    bool internalContactEnabled;

    double topLine;
    double barLen;
    double barHeight;

    int numSpectra;
    bool isSaveInfo;

    int impulsePow;
    double impulseMag;

    double splitRatio;
};

#endif
