#ifndef SIMPARAMETERS_H
#define SIMPARAMETERS_H

struct SimParameters
{
    SimParameters()
    {
        timeStep = 0.001;
        integrator = TI_EXPLICIT_EULER;
        NewtonMaxIters = 20;
        NewtonTolerance = 1e-8;

        gravityEnabled = false;
        gravityG = -9.8;
        springsEnabled = true;
        springStiffness = 100;
        maxSpringStrain = 0.2;
        dampingEnabled = true;
        dampingStiffness = 1.0;
        floorEnabled = true;
		frictionEnabled = false;

        clickMode = CM_ADDPARTICLE;
        particleMass = 1.0;
        maxSpringDist = 0.25;
        particleFixed = false;

        ceil = 0.0;

        sawRadius= 0.1;
    }

    enum ClickMode {CM_ADDPARTICLE, CM_ADDSAW};
    enum TimeIntegrator {TI_EXPLICIT_EULER, TI_VELOCITY_VERLET, TI_RUNGE_KUTTA, TI_EXP_ROSENBROCK_EULER, TI_IMPLICIT_EULER, TI_IMPLICIT_MIDPOINT, TI_TRAPEZOID, TI_TR_BDF2};

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
};

#endif
