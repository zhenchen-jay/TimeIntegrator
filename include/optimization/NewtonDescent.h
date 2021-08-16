#pragma once

#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "LineSearch.h"


void newtonSolver(std::function<double (Eigen::VectorXd, Eigen::VectorXd*, Eigen::SparseMatrix<double>* )> objFunc, std::function<double (Eigen::VectorXd, Eigen::VectorXd)> findMaxStep, std::function<void (Eigen::VectorXd)> postIteration, Eigen::VectorXd& x0, int numIter = 1000, double gradTol = 1e-7, double xTol = 1e-15, double fTol = 1e-15)
{
	const int DIM = x0.rows();
	Eigen::VectorXd grad = Eigen::VectorXd::Zero(DIM);
	Eigen::SparseMatrix<double> hessian;

	Eigen::VectorXd neggrad, delta_x;
	double maxStepSize = 1.0;
	double reg = 1e-6;


	for (int i = 0; i < numIter; i++)
	{
		//std::cout << "\niter: " << i << std::endl;
		double f = objFunc(x0, &grad, &hessian);
		
		if (grad.norm() < gradTol)
			return;

		Eigen::SparseMatrix<double> H = hessian;
		Eigen::SparseMatrix<double> I(DIM, DIM);
		I.setIdentity();
		hessian = H + reg * I;

		Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > solver(hessian);

		while (solver.info() != Eigen::Success)
		{
			reg = std::max(2 * reg, 1e-16);
			std::cout << "Matrix is not positive definite, current reg = " << reg << std::endl;
			hessian = H + reg * I;
			solver.compute(hessian);
		}

		neggrad = -grad;
		delta_x = solver.solve(neggrad);

		maxStepSize = findMaxStep(x0, delta_x);

		double rate = backtrackingArmijo(x0, grad, delta_x, objFunc, maxStepSize);

		reg *= 0.5;
		reg = std::max(reg, 1e-16);

		x0 = x0 + rate * delta_x;

		double fnew = objFunc(x0, &grad, NULL);

		/*std::cout << "f_old: " << f << ", f_new: " << fnew << ", grad norm: " << grad.norm() << ", variable update: " << rate * delta_x.norm() << std::endl;*/

		postIteration(x0);

		if (grad.norm() < gradTol)
			return;
		if (rate * delta_x.norm() < xTol)
			return;
		if (f - fnew < fTol)
			return;
	}
}
