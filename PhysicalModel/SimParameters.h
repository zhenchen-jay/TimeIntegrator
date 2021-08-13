#ifndef SIMPARAMETERS_H
#define SIMPARAMETERS_H

struct SimParameters
{
    SimParameters()
    {
        timeStep = 0.005;
        integrator = TI_IMPLICIT_EULER;
        NewtonMaxIters = 20;
        NewtonTolerance = 1e-8;

        gravityEnabled = true;
        gravityG = -9.8;
        springsEnabled = true;
        springStiffness = 1e3;
        maxSpringStrain = 0.2;
        dampingEnabled = true;
        dampingStiffness = 1.0;
        floorEnabled = true;
		frictionEnabled = false;

        particleMass = 1.0;
        maxSpringDist = 1.0;
        particleFixed = false;

        ceil = 0;
        TRBDF2_gamma = 1 - sqrt(2) / 2;
        NM_gamma = 0.5;
        NM_beta = 0.25;

        modelType = MT_POGO_STICK;
        barrierStiffness = 1e10;
        barrierEps = 1e-2;

        totalTime = 5;
        totalNumIter = totalTime / timeStep;
    }

    enum TimeIntegrator {TI_EXPLICIT_EULER, TI_VELOCITY_VERLET, TI_RUNGE_KUTTA, TI_EXP_ROSENBROCK_EULER, TI_IMPLICIT_EULER, TI_IMPLICIT_MIDPOINT, TI_TRAPEZOID, TI_TR_BDF2, TI_BDF2, TI_NEWMARK};
    enum ModelType {MT_HARMONIC_1D, MT_POGO_STICK};

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
    bool particleFixed;
    double sawRadius;

    double ceil;
    double TRBDF2_gamma;
    double NM_gamma;
    double NM_beta;

    double barrierStiffness;
    double barrierEps;

    ModelType modelType; 

    double totalTime;   // total simulation time
    int totalNumIter;   // total simulation steps
};

#endif
