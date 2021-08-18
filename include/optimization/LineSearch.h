#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace LineSearch
{
    double backtrackingArmijo(const Eigen::VectorXd& x, const Eigen::VectorXd& grad, const Eigen::VectorXd& dir, std::function<double(Eigen::VectorXd, Eigen::VectorXd*, Eigen::SparseMatrix<double>*)> objFunc, const double alphaInit = 1.0);
}
