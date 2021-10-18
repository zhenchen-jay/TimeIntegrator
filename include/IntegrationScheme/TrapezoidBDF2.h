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
Trapezoid-second-order backward differncetiation formula (TR-BDF2) rule ("Composite backward differentiation formula:
an extension of the TR-BDF2 scheme"):
\gamma = 1 - sqrt(2) / 2 by default (if \gamma = 1/ 2 => we get just trapezoid formula)
\gamma_2 = (1 - 2 \gamma) / (2 * (1 - \gamma))
\gamma_3 = (1 - \gamma_2) / (2 * \gamma)

x_{n + 2\gamma} - \gamma h v_{n + 2\gamma} = x_n + \gamma h v_n
x_{n+1} - \gamma_2 h v_{n + 1} = \gamma_3 x_{n + 2\gamma} + (1 - \gamma_3) x_n

v_{n + 2\gamma} - \gamma h M^{-1} F(x_{n + 2\gamma}) = v_n + \gamma h M^{-1} F(x_n)
v_{n+1} - \gamma_2 h M^{-1} F(x_{n + 1}) = \gamma_3 v_{n + 2\gamma} + (1 - \gamma_3) v_n

(for x_{n + 2\gamma} and v_{n + 2 \gamma})

	x_{n + 2\gamma} = x_n +  \gamma h v_n + \gamma h (v_n + \gamma h M^{-1} F(x_n) + \gamma h M^{-1} F(x_{n + 2\gamma}))
	x_{n + 2\gamma} = x_n +  2 * \gamma h v_n + (\gamma h)^2 M^{-1} (F(x_n) + F(x_{n + 2\gamma}))
=>
	xtilde = x_n + 2 * \gamma h v_n + (\gamma h)^2 M^{-1} F(x_n)
	x_{n + 2\gamma} = min_y 1/2 (y - xtilde)^T M (y - xtilde) + (\gamma h)^2 E(y)
	v_{n + 2\gamma} = v_n + \gamma h M^{-1} (F(x_n) + F(x_{n + 2\gamma}))

(for x_{n+1} and v_{n+1})
	x_{n + 1} = \gamma_3 x_{n + 2\gamma} + (1 - \gamma_3) x_n + \gamma_2 h ( \gamma_3 v_{n + 2\gamma} + (1 -  \gamma_3) v_n) + (\gamma_2 h)^2 M^{-1} F(x_{n + 1})
=>
	x_{n + 1} = min_y [y - (\gamma_3 x_{n + 2\gamma} + (1 - \gamma_3) x_n + \gamma_2 h ( \gamma_3 v_{n + 2\gamma} + (1 -  \gamma_3) v_n))]^T M [y - (\gamma_3 x_{n + 2\gamma} + (1 - \gamma_3) x_n + \gamma_2 h ( \gamma_3 v_{n + 2\gamma} + (1 -  \gamma_3) v_n))] + (\gamma_2 h)^2  E(y)
	v_{n + 1} = \gamma_3 v_{n + 2\gamma} + (1 - \gamma_3) v_n + \gamma_2 h M^{-1} F(x_{n + 1})
*/

	template <typename Problem>
	void trapezoidBDF2(const Eigen::VectorXd& xcur, const Eigen::VectorXd& vcur, const double h, const Eigen::VectorXd& M, Problem energyModel, Eigen::VectorXd& xnext, Eigen::VectorXd& vnext, double gamma = 1 - std::sqrt(2) / 2)
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

		Eigen::VectorXd fcur;
		energyModel.computeGradient(xcur, fcur);
		fcur *= -1;

		massTrip.clear();
		Eigen::SparseMatrix<double> massMatInv(M.size(), M.size());

		for (int i = 0; i < M.size(); i++)
			massTrip.push_back({ i, i , 1.0 / M(i) });
		massMatInv.setFromTriplets(massTrip.begin(), massTrip.end());

		double gamma2 = (1 - 2 * gamma) / (2 * (1 - gamma));
		double gamma3 = (1 - gamma2) / (2 * gamma);

		// step 1: compute x_gamma and v_gamma
		auto trapezoidBDF2EnergyPart1 = [&](Eigen::VectorXd x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess)
		{
			Eigen::VectorXd xtilde = xcur + 2 * gamma * h * vcur + std::pow(gamma * h, 2.0) * massMatInv * fcur;

			double E = 1.0 / 2.0 * (x - xtilde).transpose() * massMat * (x - xtilde) + std::pow(gamma * h, 2.0) * energyModel.computeEnergy(x);

			if (grad)
			{
				energyModel.computeGradient(x, (*grad));
				(*grad) = massMat * (x - xtilde) + std::pow(gamma * h, 2.0) * (*grad);
			}

			if (hess)
			{
				energyModel.computeHessian(x, (*hess));
				(*hess) = massMat + std::pow(gamma * h, 2.0) * (*hess);
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
		Eigen::VectorXd xGamma = xcur;
		energyModel.preTimeStep(xGamma);
		OptSolver::newtonSolver(trapezoidBDF2EnergyPart1, findMaxStep, postIteration, xGamma);

		Eigen::VectorXd vGamma = (xGamma - xcur) / (gamma * h) - vcur;

		// step 2: compute x_{n+1} and v_{n+1}
		auto trapezoidBDF2EnergyPart2 = [&](Eigen::VectorXd x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess)
		{
			Eigen::VectorXd xpredict = gamma3 * xGamma + (1 - gamma3) * xcur + gamma2 * h * (gamma3 * vGamma + (1 - gamma3) * vcur);

			double E = 1.0 / 2.0 * (x - xpredict).transpose() * massMat * (x - xpredict) + std::pow(gamma2 * h, 2.0) * energyModel.computeEnergy(x);

			if (grad)
			{
				energyModel.computeGradient(x, (*grad));
				(*grad) = massMat * (x - xpredict) + std::pow(gamma2 * h, 2.0) * (*grad);
			}

			if (hess)
			{
				energyModel.computeHessian(x, (*hess));
				(*hess) = massMat + std::pow(gamma2 * h, 2.0) * (*hess);
			}

			return E;
		};

		xnext = xGamma;
		OptSolver::newtonSolver(trapezoidBDF2EnergyPart2, findMaxStep, postIteration, xnext);
		if (std::abs(gamma2) > 1e-6)
		{
			vnext = (xnext - gamma3 * xGamma - (1 - gamma3) * xcur) / (gamma2 * h);
		}
		else
		{
			Eigen::VectorXd forceNext;
			energyModel.computeGradient(xnext, forceNext);
			forceNext *= -1;

			vnext = gamma3 * vGamma - (1 - gamma3) * vcur + (gamma2 * h) * massMatInv * forceNext;
		}

	}
}

