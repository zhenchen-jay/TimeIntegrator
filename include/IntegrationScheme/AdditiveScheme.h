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
    /*
* For the system we aimed to solve:
* dx / dt = v
* M dv / dt = F(x)
* where F(x) = -\nabla E(x),
* E(x) = 1/2 x^T M x + potential_energy(x).
*
*
Newmark method ("A Method of Computation for Structural Dynamics"):
x_{n+1} = x_n + h * v_n + (1/2 - \beta) h^2 M^{-1} Fe(x_n) + h^2 \beta M^{-1} Fe(x_{n+1}) + h^2 / 2 M^{-1} Fc(x_{n+1})
v_{n+1} = v_n + (1 - \gamma) h M^{-1} Fe(x_n) + \gamma h M^{-1} Fe(x_{n+1}) + (1 - \gamm_1) h M^{-1} Fc(x_n) + \gamma_1 h M^{-1} Fc(x_{n+1})

where F = Fe + Fc, if gamma = gamma1 = 1/2, then the integration is second order accurate, otherwise the second order error is (1 / 2 - gamma1) h^2 v_k^T \nabla^2 V(x_k) v_k (assuming gamma = 1 / 2)
*/
	template<typename Problem>
	void additiveScheme(const Eigen::VectorXd& xcur, const Eigen::VectorXd& vcur, const double h, const Eigen::VectorXd& M, Problem energyModel, Eigen::VectorXd& xnext, Eigen::VectorXd& vnext, 
	double initialE = 0, double gamma = 0.5, double beta = 0.25)
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
		Eigen::VectorXd fcur, fcurPart1, fcurPart2;
		energyModel.computeGradient(xcur, fcur);
		energyModel.computeGradientPart1(xcur, fcurPart1);
		energyModel.computeGradientPart2(xcur, fcurPart2);

		fcur *= -1;
        fcurPart1 *= -1;
        fcurPart2 *= -1;

		auto newmarkEnergy = [&](Eigen::VectorXd x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess){
		    Eigen::VectorXd xtilde = xcur + h * vcur + h * h * (0.5 - beta) * massMatInv * fcur;
		    double E = 1.0 / 2.0 * (x - xtilde).transpose() * massMat * (x - xtilde) + beta * h * h * energyModel.computeEnergyPart1(x) + 0.5 * h * h * energyModel.computeEnergyPart2(x);

		    if (grad)
		    {
				Eigen::VectorXd g1, g2;
		        energyModel.computeGradientPart1(x, g1);
				energyModel.computeGradientPart2(x, g2);

		        (*grad) = massMat * (x - xtilde) + h * h * (beta * g1 + 0.5 * g2);
		    }

		    if (hess)
		    {
				Eigen::SparseMatrix<double> h1, h2;
		        energyModel.computeHessianPart1(x, h1);
				energyModel.computeHessianPart2(x, h2);

		        (*hess) = massMat + h * h * (beta * h1 + 0.5 * h2);
		    }

		    return E;
		};

		auto findMaxStep = [&](Eigen::VectorXd x, Eigen::VectorXd dir){
		    return energyModel.getMaxStepSize(x, dir);
		};

		auto postIteration = [&](Eigen::VectorXd x){
		    energyModel.postIteration(x);
		};


		// newton step to find the optimal
		xnext = xcur;
		energyModel.preTimeStep(xnext);
		OptSolver::newtonSolver(newmarkEnergy, findMaxStep, postIteration, xnext);

		Eigen::VectorXd fnext,fnextPart1, fnextPart2;
		energyModel.computeGradient(xnext, fnext);
		energyModel.computeGradientPart1(xnext, fnextPart1);
		energyModel.computeGradientPart2(xnext, fnextPart2);

		fnext *= -1;
		fnextPart1 *= -1;
		fnextPart2 *= -1;

		// the energy error coming from gamma1 is roughly (1 / 2 - gamma1) h^2 v_k^T \nabla^2 V(x_k) v_k

		Eigen::VectorXd vnext0 = vcur + h * massMatInv * ((1 - gamma) * fcurPart1 + gamma * fnextPart1 + fcurPart2);
		Eigen::VectorXd vnext1 = vcur + h * massMatInv * ((1 - gamma) * fcurPart1 + gamma * fnextPart1 + fnextPart2);
		Eigen::VectorXd vnextHalf = vcur + h * massMatInv * ((1 - gamma) * fcurPart1 + gamma * fnextPart1 + 0.5 * fcurPart2 + 0.5 * fnextPart2);

		auto totalEnergy = [&](Eigen::VectorXd x, Eigen::VectorXd v){
		    return 0.5 * v.dot(massMat * v) + energyModel.computeEnergy(x);
		};

		double E0 = totalEnergy(xnext, vnext0);
		double E1 = totalEnergy(xnext, vnext1);
		double EHalf = totalEnergy(xnext, vnextHalf);


		double Ecur = totalEnergy(xcur, vcur);
		double gamma1 = 0.5;

		auto binarySearch = [&](double begin, double end)
		{
			double mid = (begin + end) / 2;
		    while(end - begin > 1e-6)
		    {
		       
		        Eigen::VectorXd vtmp = vcur + h * massMatInv * ((1 - gamma) * fcurPart1 + gamma * fnextPart1 + (1 - mid) * fcurPart2 + mid * fnextPart2);
		        double Etmp = totalEnergy(xnext, vtmp);
				 std::cout << "current begin: " << begin << ", end: " << end << ", gamma1: " << mid << ", E_gamma: " << Etmp << std::endl;
		        if(std::abs(Etmp - initialE) < std::min(1e-4 * initialE, 1e-8))
		        {
		            vnext = vtmp;
		            std::cout << "optimal gamma 1 = " << mid << std::endl;
		            break;
		        }

		        if((Etmp - initialE) * (E0 - initialE) <= 0)
		            end = mid;
		        else
		            begin = mid;
		        mid = (begin + end) / 2;
		    }
			return mid;
		};

		if((E0 >= initialE && E1 >= initialE))
		{
			if(EHalf <= initialE)
				gamma1 = binarySearch(0, 0.5);
			else
			{
				if(EHalf <= E0 && EHalf <= E1)
					gamma1 = 0.5;
				if (E0 <= E1 && E0 <= EHalf)
					gamma1 = 0;
				if (E1 <= E0 && E1 <= EHalf)
					gamma1 = 1;
			}
		}
		else if((E0 <= initialE && E1 <= initialE))
		{
			if(EHalf >= initialE)
				gamma1 = binarySearch(0.5, 1);
			else
			{
				if(EHalf <= E0 && EHalf <= E1)
					gamma1 = 0.5;
				if (E0 <= E1 && E0 <= EHalf)
					gamma1 = 0;
				if (E1 <= E0 && E1 <= EHalf)
					gamma1 = 1;
			}
		}
		    
		else                  // do the binary search to find the optimal gamma 1
		{
		    gamma1 = binarySearch(0, 1);
		}
		std::cout << "reasonable gamma 1 = " << gamma1 << std::endl;
		    vnext = vcur + h * massMatInv * ((1 - gamma) * fcurPart1 + gamma * fnextPart1 + (1 - gamma1) * fcurPart2 + gamma1 * fnextPart2);
		std::cout << "energy before update: " << Ecur << ", energy after update: " << totalEnergy(xnext, vnext) << ", initial energy: " << initialE << std::endl;

		
	}
}