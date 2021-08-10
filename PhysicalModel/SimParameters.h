#ifndef SIMPARAMETERS_H
#define SIMPARAMETERS_H

struct SimParameters
{
    SimParameters()
    {
        timeStep = 0.001;
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

        clickMode = CM_ADDPARTICLE;
        particleMass = 1.0;
        maxSpringDist = 1.0;
        particleFixed = false;

        ceil = 0;

        sawRadius= 0.1;
        TRBDF2_gamma = 1 - sqrt(2) / 2;
        NM_gamma = 0.5;
        NM_beta = 0.25;

        modelType = MT_MASS_SPRING;
        barrierStiffness = springStiffness;
        barrierEps = 1e-2;

    }

    enum ClickMode {CM_ADDPARTICLE, CM_ADDSAW};
    enum TimeIntegrator {TI_EXPLICIT_EULER, TI_VELOCITY_VERLET, TI_RUNGE_KUTTA, TI_EXP_ROSENBROCK_EULER, TI_IMPLICIT_EULER, TI_IMPLICIT_MIDPOINT, TI_TRAPEZOID, TI_TR_BDF2, TI_NEWMARK};
    enum ModelType {MT_HARMONIC_1D, MT_MASS_SPRING};

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

    ClickMode clickMode;
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
};

#endif
