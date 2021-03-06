#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

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
Newmark method ("A Method of Computation for Structural Dynamics"):
x_{n+1} = x_n + h * v_n + (1/2 - \beta) h^2 M^{-1} F(x_n) + h^2 \beta M^{-1} F(x_{n+1}) 
v_{n+1} = v_n + (1 - \gamma) h M^{-1} F(x_n) + \gamma h M^{-1} F(x_{n+1})

=>
xtilde = x_n + h v_n + h^2 (1/2 - \beta) M^{-1} F(x_n)
x_{n+1} =  min_y 1/2 (y - xtilde)^T M (y - xtilde) + h^2 \beta E(y)
*/

	template <typename Problem>
	void Newmark(const Eigen::VectorXd& xcur, const Eigen::VectorXd& vcur, const double h, const Eigen::VectorXd& M, Problem energyModel, Eigen::VectorXd& xnext, Eigen::VectorXd& vnext, double gamma = 0.5, double beta = 0.25)
	{
	    if(h == 0)
	    {
	        xnext = xcur;
	        vnext = vcur;
	    }
		bool isPrintInfo = false;
		/*if (std::abs(xcur(2) - 0.0119639999999786) < 1e-8)
			isPrintInfo = true;*/

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

		auto newmarkEnergy = [&](Eigen::VectorXd x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess)
		{
			Eigen::VectorXd xtilde = xcur + h * vcur + h * h * (0.5 - beta) * massMatInv * fcur;
			double E = 1.0 / 2.0 * (x - xtilde).transpose() * massMat * (x - xtilde) + beta * h * h * energyModel.computeEnergy(x);

			if (grad)
			{
				energyModel.computeGradient(x, (*grad));
				(*grad) = massMat * (x - xtilde) + beta * h * h * (*grad);
			}

			if (hess)
			{
				energyModel.computeHessian(x, (*hess));
				(*hess) = massMat + beta * h * h * (*hess);
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
		

		OptSolver::newtonSolver(newmarkEnergy, findMaxStep, postIteration, xnext, 1000,1e-14, 0, 0, isPrintInfo);

		if (isPrintInfo)
			system("pause");

		Eigen::VectorXd forceNext;
		energyModel.computeGradient(xnext, forceNext);
		forceNext *= -1;



		vnext = vcur + h * massMatInv * ((1 - gamma) * fcur + gamma * forceNext);

		auto totalEnergy = [&](Eigen::VectorXd x, Eigen::VectorXd v){
		    return 0.5 * v.dot(massMat * v) + energyModel.computeEnergy(x);
		};

		std::cout << "energy before update: " << totalEnergy(xcur, vcur) << ", energy after update: " << totalEnergy(xnext, vnext) << std::endl;
	}

}