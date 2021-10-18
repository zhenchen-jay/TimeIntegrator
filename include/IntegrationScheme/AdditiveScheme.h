#pragma once
#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "../optimization/NewtonDescent.h"
#include "../../PhysicalModel/SimParameters.h"
#include "BDF2.h"
#include "ImplicitEuler.h"
#include "Newmark.h"
#include "TrapezoidBDF2.h"

namespace TimeIntegrator
{
	template<typename Problem>
	void additiveScheme(const Eigen::VectorXd& xcur, const Eigen::VectorXd& vcur, const double h, const Eigen::VectorXd& M, Problem energyModel, Eigen::VectorXd& xnext, Eigen::VectorXd& vnext, SimParameters::TimeIntegrator firstTI = SimParameters::TimeIntegrator::TI_NEWMARK, SimParameters::TimeIntegrator secondTI = SimParameters::TimeIntegrator::TI_IMPLICIT_EULER, double gamma = 0.5, double beta = 0.25)
	{
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


		auto findMaxStep = [&](Eigen::VectorXd x, Eigen::VectorXd dir)
		{
			return energyModel.getMaxStepSize(x, dir);
		};

		auto postIteration = [&](Eigen::VectorXd x)
		{
			energyModel.postIteration(x);
		};

		Eigen::VectorXd fcur;
		energyModel.computeGradient(xcur, fcur);
		fcur *= -1;

		auto firstPartEnergy = [&](Eigen::VectorXd x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess)
		{
			Eigen::VectorXd xtilde = xcur + h * vcur + h * h * (0.5 - beta) * massMatInv * fcur;
			double E = 1.0 / 2.0 * (x - xtilde).transpose() * massMat * (x - xtilde) + beta * h * h * energyModel.computeEnergyPart1(x);

			if (grad)
			{
				energyModel.computeGradientPart1(x, (*grad));
				(*grad) = massMat * (x - xtilde) + beta * h * h * (*grad);
			}

			if (hess)
			{
				energyModel.computeHessianPart1(x, (*hess));
				(*hess) = massMat + beta * h * h * (*hess);
			}

			return E;
		};

		// newton step to find the optimal
		Eigen::VectorXd xtemp = xcur;
		energyModel.preTimeStep(xtemp);
		//OptSolver::testFuncGradHessian(firstPartEnergy, xcur);
		OptSolver::newtonSolver(firstPartEnergy, findMaxStep, postIteration, xtemp);

		Eigen::VectorXd forcetemp;
		energyModel.computeGradientPart1(xtemp, forcetemp);
		forcetemp *= -1;

		Eigen::VectorXd vtemp = vcur + h * massMatInv * ((1 - gamma) * fcur + gamma * forcetemp);

		std::cout << "first part finished!" << std::endl;

		auto secondPartEnergy = [&](Eigen::VectorXd x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess)
		{
			Eigen::VectorXd xtilde = xtemp + h * vtemp;
			double E = 1.0 / 2.0 * (x - xtilde).transpose() * massMat * (x - xtilde) + h * h * energyModel.computeEnergyPart2(x);

			if (grad)
			{
				energyModel.computeGradientPart2(x, (*grad));
				(*grad) = massMat * (x - xtilde) + h * h * (*grad);
			}

			if (hess)
			{
				energyModel.computeHessianPart2(x, (*hess));
				(*hess) = massMat + h * h * (*hess);
			}

			return E;
		};

		// newton step to find the optimal
		xnext = xtemp;
		energyModel.preTimeStep(xnext);
		//OptSolver::testFuncGradHessian(firstPartEnergy, xnext);
		OptSolver::newtonSolver(secondPartEnergy, findMaxStep, postIteration, xnext);

		vnext = (xnext - xtemp) / h;
		std::cout << "second part finished!" << std::endl;
		
	}
}