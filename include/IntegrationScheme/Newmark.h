#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

/*
* For the system we aimed to solve:
* dx / dt = v
* M dv / dt = F(x)
* where F(x) = -\nabla E(x),
* E(x) = 1/2 x^T M x + potential_energy(x).
*
*
Newmark method ("A Method of Computation for Structural Dynamics"):
x_{n+1} = x_n + h * v_n + (1/2 - \beta) h^2 M^{-1} F(x_n) + h^2 \beta M^{-1} F(x_{n+1}) 
v_{n+1} = v_n + (1 - \gamma) h M^{-1} F(x_n) + \gamma h M^{-1} F(x_{n+1})

=>
x_{n+1} =  min_y 1/2 (y - x_n - hv_n)^T M (y - x_n - hv_n) + h^2 \beta E(y) - h^2 (1/2 - \beta) * y^T F(x_n)
*/

template <typename Problem>
void Newmark(const Eigen::VectorXd& xcur, const Eigen::VectorXd& vcur, const double h, const Eigen::VectorXd& M, Problem energyModel, Eigen::VectorXd& xnext, Eigen::VectorXd& vnext, double gamma = 0.5, double beta = 0.25)
{
	std::vector<Eigen::Triplet<double>> massTrip;
	SparseMatrix<double> massMat(M.size(), M.size());

	for (int i = 0; i < M.size(); i++)
		massTrip.push_back({ i, i , M(i) });
	massMat.setFromTriplets(massTrip.begin(), massTrip.end());

	Eigen::VectorXd fcur;
	energyModel.computeGradient(xcur, fcur);
	fcur *= -1;

	auto newmarkEnergy = [&](Eigen::VectorXd x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess)
	{
		double E = 0.5 * (x - xcur - h * vcur).transpose() * massMat * (x - xcur - h * vcur) + h * h * beta * energyModel.computeEnergy(x) - h * h * (0.5 - beta) * x.dot(fcur);

		if (grad)
		{
			energyModel.computeGradient(x, (*grad));
			(*grad) = massMat * (x - xcur - h * vcur) + h * h * beta * (*grad) - h * h * (0.5 - beta) * fcur;
		}

		if (hess)
		{
			energyModel.computeHessian(x, (*hess));
			(*hess) = massMat + h * h * beta * (*hess);
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
	newtonSolver(newmarkEnergy, findMaxStep, postIteration, xnext);

	Eigen::VectorXd forceNext;
	energyModel.computeGradient(xnext, forceNext);
	forceNext *= -1;

	massTrip.clear();
	Eigen::SparseMatrix<double> massMatInv(M.size(), M.size());

	for (int i = 0; i < M.size(); i++)
		massTrip.push_back({ i, i , 1.0 / M(i) });
	massMatInv.setFromTriplets(massTrip.begin(), massTrip.end());

	vnext = vcur + h * massMatInv * ((1 - gamma) * fcur + gamma * forceNext);
}