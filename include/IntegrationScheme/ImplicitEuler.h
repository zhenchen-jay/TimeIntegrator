#pragma once
#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "../optimization/NewtonDescent.h"


/*
* For the system we aimed to solve:
* dx / dt = v
* M dv / dt = F(x)
* where F(x) = -\nabla E(x),
* E(x) = 1/2 x^T M x + potential_energy(x).
*
*
Implicit (Backward) Euler:
x_{n+1} = x_n + h * v_{n+1}
v_{n+1} = v_n + h M^{-1} F(x_{n+1})

=>
	x_{n+1} = x_n + h v_n + h^2 M^{-1} F(x_{n+1})
=>
	x_{n+1} = min_y 1/2 (y - x_n - hv_n)^T M (y - x_n - hv_n) + h^2 E(y)  (*)
*/

template <typename Problem>
void implicitEuler(const Eigen::VectorXd& xcur, const Eigen::VectorXd& vcur, const double h, const Eigen::VectorXd& M, Problem energyModel, Eigen::VectorXd& xnext, Eigen::VectorXd& vnext)
{
	std::vector<Eigen::Triplet<double>> massTrip;
	Eigen::SparseMatrix<double> massMat(M.size(), M.size());

	for (int i = 0; i < M.size(); i++)
		massTrip.push_back({ i, i , M(i) });
	massMat.setFromTriplets(massTrip.begin(), massTrip.end());

	auto implicitEulerEnergy = [&](Eigen::VectorXd x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess)
	{
		double E = 0.5 * (x - xcur - h * vcur).transpose() * massMat * (x - xcur - h * vcur) + h * h * energyModel.computeEnergy(x);

		if (grad)
		{
			energyModel.computeGradient(x, (*grad));
			(*grad) = massMat * (x - xcur - h * vcur) + h * h * (*grad);
		}

		if (hess)
		{
			energyModel.computeHessian(x, (*hess));
			(*hess) = massMat + h * h * (*hess);
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
	newtonSolver(implicitEulerEnergy, findMaxStep, postIteration, xnext);

	vnext = (xnext - xcur) / h;
}
