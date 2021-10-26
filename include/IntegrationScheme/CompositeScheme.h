#pragma once
#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "../optimization/NewtonDescent.h"
#include "../../PhysicalModel/SimParameters.h"
#include "BDF2.h"
#include "ImplicitEuler.h"
#include "Newmark.h"
#include "TrapezoidBDF2.h"
#include "GeneralizedTrapezoid.h"

namespace TimeIntegrator
{
/*
* For the system we aimed to solve:
* dx / dt = v
* M dv / dt = F(x)
* where F(x) = -\nabla E(x),
* E(x) = 1/2 x^T M x + potential_energy(x).
*
*
We use implemented the scheme mentioned in the paper "An unconditionally stable time integration methodwith controllable dissipation for second-order nonlineardynamics", which is claimed to be second-order accurate the unconditionally stable. 
*/
    void computeCoefficients(double rho, double &c1, double &c2, double &b1, double &b2, double &alpha)
    {
        if(rho == 1.0)
        {
            c1 = 1.0 / 4;
            c2 = 3.0 / 4;
            b1 = 1.0 / 2;
            b2 = 1.0 / 2;
            alpha = 1.0 / 3;
        }
        else
        {
            c1 = (2 - std::sqrt(2 * (1 + rho))) / (2 * (1 - rho));
            c2 = (std::sqrt(2) * (1 + rho) - 2 * rho * std::sqrt(1 + rho)) / (2 * (1 - rho)* std::sqrt(1 + rho));
            b1 = 1.0 / 2;
            b2 = 1.0 / 2;
            alpha = c1 / c2;
        }
    }

	template<typename Problem>
	void compositeScheme(const Eigen::VectorXd& xcur, const Eigen::VectorXd& vcur, const double h, const Eigen::VectorXd& M, Problem energyModel, Eigen::VectorXd& xnext, Eigen::VectorXd& vnext, double rho = 1.0)
	{
		double c1, c2, b1, b2, alpha;
        computeCoefficients(rho, c1, c2, b1, b2, alpha);
        std::cout << "c1: " << c1 << ", c2: " << c2 << ", b1: " << b1 << ", b2 " << b2 << ", alpha: " << alpha << std::endl;

        Eigen::VectorXd x1, v1, a1, f1, x2, v2, a2, f2;

        // IE for the first step, with step size c1 * h
        implicitEuler(xcur, vcur, c1 * h, M, energyModel, x1, v1);

        // generalized trapezoid
        generalizedTrapezoid(x1, v1, c2 * h, M, energyModel, x2, v2, alpha);

        // average x1, x2, v1, v2
        xnext = xcur + h * (b1 * v1 + (1 - b1) * v2);

        energyModel.computeGradient(x1, f1);
        energyModel.computeGradient(x2, f2);

        f1 *= -1;
        f2 *= -1;
        vnext = vcur;
        for(int i = 0; i < vnext.size(); i++)
            vnext(i) += h * (b1 * f1(i) + (1 - b1) * f2(i)) / M(i);
		
	}
}