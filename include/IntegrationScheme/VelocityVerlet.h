#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

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
VelocityVerlet Euler:
x_{n+1} = x_n + h * v_n
v_{n+1} = v_n + h M^{-1} F(x_{n+1})
*/

    template <typename Problem>
    void velocityVerlet(const Eigen::VectorXd& xcur, const Eigen::VectorXd& vcur, const double h, const Eigen::VectorXd& M, Problem energyModel, Eigen::VectorXd& xnext, Eigen::VectorXd& vnext)
    {
        xnext = xcur + h * vcur;
        Eigen::VectorXd force;
        energyModel.computeGradient(xnext, force);
        force *= -1;
        vnext = vcur + h * force.cwiseQuotient(M);
    }
}
