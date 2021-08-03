#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

/*
This script implement the implicit Euler scheme for solving dx / dt = f(x, t) 
x_{n+1} = x_n + h f(t_n, x_n), or in other words
*/

template <typename Problem>
void explicitEuler(const Eigen::VectorXd& xcur, Problem f, const double tcur, const double h, Eigen::VectorXd& xnext)
{
    xnext = xcurr + h * f.eval(xcur);
}