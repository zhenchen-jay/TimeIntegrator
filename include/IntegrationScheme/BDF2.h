#pragma once
#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "../optimization/NewtonDescent.h"

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
	second-order backward-difference method:
	x_{n+2} = 4/3 x_{n+1} - 1 / 3 x_n + 2/3 h v_{n+2}
	v_{n+2} = 4/3 v_{n+1} - 1 / 3 v_n + 2/3 h M^{-1}F(x_{n+2})

	=>
	xtilde = 4/3(x_{n+1} + 2/3 h v_{n+1}) - 1/3(x_n + 2/3 h v_n)
	x_{n+2} = min_y (y - xtilde)^T M (y - xtilde) + 4/9 h^2 E(x)
	*/

	template <typename Problem>
	void BDF2(const Eigen::VectorXd& xcur, const Eigen::VectorXd& vcur, const Eigen::VectorXd& xprev, const Eigen::VectorXd& vprev, const double h, const Eigen::VectorXd& M, Problem energyModel, Eigen::VectorXd& xnext, Eigen::VectorXd& vnext)
	{
	    if(h == 0)
	    {
	        xnext = xcur;
	        vnext = vcur;
	    }
		std::vector<Eigen::Triplet<double>> massTrip;
		Eigen::SparseMatrix<double> massMat(M.size(), M.size());

		for (int i = 0; i < M.size(); i++)
			massTrip.push_back({ i, i , M(i) });
		massMat.setFromTriplets(massTrip.begin(), massTrip.end());

		auto BDF2Energy = [&](Eigen::VectorXd x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess)
		{
			Eigen::VectorXd xtilde = 4.0 / 3.0 * (xcur + 2.0 / 3.0 * h * vcur) - 1.0 / 3.0 * (xprev + 2.0 / 3.0 * h * vprev);
			double E = 1.0 / 2.0 * (x - xtilde).transpose() * massMat * (x - xtilde) + 4.0 / 9.0 * h * h * energyModel.computeEnergy(x);

			if (grad)
			{
				energyModel.computeGradient(x, (*grad));
				(*grad) = massMat * (x - xtilde) + 4.0 / 9.0 * h * h * (*grad);
			}

			if (hess)
			{
				energyModel.computeHessian(x, (*hess));
				(*hess) = massMat + 4.0 / 9.0 * h * h * (*hess);
			}

			return E;
		};

		auto findMaxStep = [&](Eigen::VectorXd x, Eigen::VectorXd dir)
		{
			return energyModel.getMaxStepSize(x, dir);
		};

		auto postIteration = [&](Eigen::VectorXd x)
		{
			energyModel.postIteration(x);
		};

		// newton step to find the optimal
		xnext = xcur;
		energyModel.preTimeStep(xnext);
		OptSolver::newtonSolver(BDF2Energy, findMaxStep, postIteration, xnext);

		vnext = (xnext - 4.0 / 3.0 * xcur + 1.0 / 3.0 * xprev) / (2.0 / 3.0 * h);
	}

}
