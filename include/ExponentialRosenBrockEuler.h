#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

/*
This script implement the explicit Euler scheme:
x_{n+1} = x_n + c v_n
v_{n+1} = v_n + c a_n
a_n = M^{-1} (F_ext - \nabla E(x_n)), where F_ext is external force, E(x_n) is the potential energy at state x_n
*/

template <typename Problem>
void exponetialRosenBrockEuler(const Eigen::VectorXd& xcur, Problem f, const double tcur, const double h, Eigen::VectorXd& xnext)
{
    Eigen::SparseMatrix<double> Jn;
    f.gradient(xcur, Jn);

}