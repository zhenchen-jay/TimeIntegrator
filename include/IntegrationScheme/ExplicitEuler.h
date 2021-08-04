#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

/*
* For the system we aimed to solve:
* dx / dt = v
* M dv / dt = F(x)
* where F(x) = -\nabla E(x), 
* E(x) = 1/2 x^T M x + potential_energy(x).
* 
* 
Explicit (Forward) Euler:
x_{n+1} = x_n + h * v_n
v_{n+1} = v_n + h M^{-1} F(x_n)
*/

template <typename Problem>
void explicitEuler(const Eigen::VectorXd& xcur, const Eigen::VectorXd& vcur, const double h, const Eigen::VectorXd& M, Problem energyModel, Eigen::VectorXd& xnext, Eigen::VectorXd& vnext)
{
    Eigen::VectorXd force; 
    energyModel.gradient(xcur, force);
    force *= -1;
    xnext = xcur + h * vcur;
    vnext = vcur + h * force.cwiseQuotient(M);
}