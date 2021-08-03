#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

/*
This script implement the implicit Euler scheme for solving dx / dt = f(x, t) 
x_{n+1} = x_n + h f(t_{n+1}, x_{n+1}), or in other words
*/

template <typename Problem>
void implicitEuler(const Eigen::VectorXd& xcurr, Problem f, const double tcur, const double h, Eigen::VectorXd& xnext)
{

}