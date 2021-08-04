#pragma once

#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "LineSearch.h"

template<typename Problem>
class NewtonDescentSolver
{
	public:
		void minimize(Problem& objFunc, Eigen::VectorXd& x0, int numIter = 1000, double graTol = 1e-7, double xTol = 1e-10, double fTol = 1e-15)
		{
			const int DIM = x0.rows();
			Eigen::VectorXd grad = Eigen::VectorXd::Zero(DIM);
			Eigen::SparseMatrix<double> hessian;

			Eigen::VectorXd neggrad, delta_x;
			double maxStepSize = 1.0;
			double reg = 1e-6;


			for (int i = 0; i < numIter; i++)
			{
				double f = objFunc.value(x0);
				objFunc.gradient(x0, grad);

				if (grad.norm() < gradTol)
					return;

				objFunc.hessian(x0, hessian);

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

				maxStepSize = objFunc.getMaxStepSize(x0);

				double rate = backtrackingArmijo<Problem>(x0, grad, delta_x, objFunc, maxStepSize);

				reg *= 0.5;
				reg = std::max(reg, 1e-16);

				double

					x0 = x0 + rate * delta_x;

				double fnew = objFunc.value(x0);

				objFunc.gradient(x0, grad);

				std::cout << "iter: " << i << ", f_old: " << f << ", f_new: " << fnew << ", grad norm: " << grad.norm() << ", variable update: " << rate * delta_x.norm() << std::endl;

				if (grad.norm() < gradTol)
					return;
				if (rate * delta_x.norm() < xTol)
					return;
				if (f - fnew < fTol)
					return;
			}
		}
};