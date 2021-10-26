#pragma once
#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "../optimization/NewtonDescent.h"
#include "ImplicitEuler.h"
#include "ExplicitEuler.h"

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
Trapezoid rule:
x_{n+1} = x_n + h * ((1 - alpha) * v_{n+1} + alpha * v_n)
v_{n+1} = v_n + h M^{-1} ((1 - alpha) * F(x_{n+1}) + alpha * F(x_n))

=>
	x_{n+1} = x_n + h v_n + h^2 (1 - alpha) * M^{-1} ((1 - alpha) * F(x_{n+1}) + alpha * F(x_n))
=>
	xtilde = x_n + h v_n + h^2 alpha * (1 - alpha) M^{-1} F(x_n)
	x_{n+1} = min_y 1/2 (y - xtilde)^T M (y - xtilde) + h^2 (1 - alpha)^2 E(y)  (*)
*/

	template <typename Problem>
	void generalizedTrapezoid(const Eigen::VectorXd& xcur, const Eigen::VectorXd& vcur, const double h, const Eigen::VectorXd& M, Problem energyModel, Eigen::VectorXd& xnext, Eigen::VectorXd& vnext, double alpha = 0.5)
	{
	    if(h == 0)
	    {
	        xnext = xcur;
	        vnext = vcur;
	    }
        if(alpha == 0)
        {
            implicitEuler(xcur, vcur, h, M, energyModel, xnext, vnext);
            return;
        }
        if(alpha == 1)
        {
            explicitEuler(xcur, vcur, h, M, energyModel, xnext, vnext);
            return;
        }
		std::vector<Eigen::Triplet<double>> massTrip;
		Eigen::SparseMatrix<double> massMat(M.size(), M.size());

		for (int i = 0; i < M.size(); i++)
			massTrip.push_back({ i, i , M(i) });
		massMat.setFromTriplets(massTrip.begin(), massTrip.end());

		massTrip.clear();
		Eigen::SparseMatrix<double> massMatInv(M.size(), M.size());

		for (int i = 0; i < M.size(); i++)
			massTrip.push_back({ i, i , 1.0 / M(i) });
		massMatInv.setFromTriplets(massTrip.begin(), massTrip.end());

		Eigen::VectorXd fcur;
		energyModel.computeGradient(xcur, fcur);
		fcur *= -1;

		auto trapezoidEnergy = [&](Eigen::VectorXd x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess)
		{
			Eigen::VectorXd xtilde = xcur + h * vcur + h * h * alpha * (1 - alpha) * massMatInv * fcur;
			double E = 1.0 / 2.0 * (x - xtilde).transpose() * massMat * (x - xtilde) + (1 - alpha) * (1 - alpha) * h * h * energyModel.computeEnergy(x);

			if (grad)
			{
				energyModel.computeGradient(x, (*grad));
				(*grad) = massMat * (x - xtilde) + (1 - alpha) * (1 - alpha) * h * h * (*grad);
			}

			if (hess)
			{
				energyModel.computeHessian(x, (*hess));
				(*hess) = massMat + (1 - alpha) * (1 - alpha) * h * h * (*hess);
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
		OptSolver::newtonSolver(trapezoidEnergy, findMaxStep, postIteration, xnext);

		vnext = (xnext - xcur - h * alpha * vcur) / h / (1 - alpha);
	}
}

