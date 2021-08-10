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
Implicit midpoint rule:
x_{n+1} = x_n + h * (v_{n+1} + v_n) / 2
v_{n+1} = v_n + h M^{-1} F((x_{n+1} + x_n) / 2)

=>
	x_{n+1} = x_n + h v_n + h^2 / 2 M^{-1} F((x_{n+1} + x_n) / 2)
=>
	x_{n+1} = min_y 1/2 (y - x_n - hv_n)^T M (y - x_n - hv_n) + h^2 E((y + x_n) / 2)  (*)
*/

template <typename Problem>
void implicitMidPoint(const Eigen::VectorXd& xcur, const Eigen::VectorXd& vcur, const double h, const Eigen::VectorXd& M, Problem energyModel, Eigen::VectorXd& xnext, Eigen::VectorXd& vnext)
{
	std::vector<Eigen::Triplet<double>> massTrip;
	SparseMatrix<double> massMat(M.size(), M.size());

	for (int i = 0; i < M.size(); i++)
		massTrip.push_back({ i, i , M(i) });
	massMat.setFromTriplets(massTrip.begin(), massTrip.end());

    Eigen::VectorXd fcur;
    energyModel.computeGradient(xcur, fcur);
    fcur *= -1;

	auto implicitMidPointEnergy = [&](Eigen::VectorXd x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess)
	{
        Eigen::VectorXd midPoint = (x + xcur) / 2.0;
		double E = 0.5 * (x - xcur - h * vcur).transpose() * massMat * (x - xcur - h * vcur) + h * h * energyModel.computeEnergy(midPoint);

		if (grad)
		{
			energyModel.computeGradient(midPoint, (*grad));
			(*grad) = massMat * (x - xcur - h * vcur) + h * h / 2.0 * (*grad);
		}

		if (hess)
		{
			energyModel.computeHessian(midPoint, (*hess));
			(*hess) = massMat + h * h / 4.0 * (*hess);
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
	newtonSolver(implicitMidPointEnergy, findMaxStep, postIteration, xnext);

	vnext = 2 * (xnext - xcur) / h - vcur;
}