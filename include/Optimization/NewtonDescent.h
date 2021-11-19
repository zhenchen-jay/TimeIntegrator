#pragma once

#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "LineSearch.h"


namespace OptSolver
{
	void newtonSolver(std::function<double(Eigen::VectorXd, Eigen::VectorXd*, Eigen::SparseMatrix<double>*)> objFunc, std::function<double(Eigen::VectorXd, Eigen::VectorXd)> findMaxStep, std::function<void(Eigen::VectorXd)> postIteration, Eigen::VectorXd& x0, int numIter = 1000, double gradTol = 1e-14, double xTol = 0, double fTol = 0, bool isPrintInfo = false);
	void testFuncGradHessian(std::function<double(Eigen::VectorXd, Eigen::VectorXd*, Eigen::SparseMatrix<double>*)> objFunc, const Eigen::VectorXd& x0);
}


