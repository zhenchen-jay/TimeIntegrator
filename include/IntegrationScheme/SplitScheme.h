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
	void splitScheme(const Eigen::VectorXd& xcur, const Eigen::VectorXd& vcur, const double h, const Eigen::VectorXd& M, Problem energyModel, Eigen::VectorXd& xnext, Eigen::VectorXd& vnext, double alpha = 0.5, double desireEnergy = 0, SimParameters::TimeIntegrator firstTI = SimParameters::TimeIntegrator::TI_NEWMARK, SimParameters::TimeIntegrator secondTI = SimParameters::TimeIntegrator::TI_IMPLICIT_EULER, double gamma = 0.5, double beta = 0.25)
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

	    massTrip.clear();
	    Eigen::SparseMatrix<double> massMatInv(M.size(), M.size());

	    for (int i = 0; i < M.size(); i++)
	        massTrip.push_back({ i, i , 1.0 / M(i) });
	    massMatInv.setFromTriplets(massTrip.begin(), massTrip.end());

	    Eigen::VectorXd xtemp, vtemp;
	    Newmark<Problem>(xcur, vcur, h, M, energyModel, xtemp, vtemp, gamma, beta);
	    double energy0 = 0.5 * vcur.dot(massMat*vcur) + energyModel.computeEnergy(xcur);    // total energy
	    if(desireEnergy)
	        energy0 = desireEnergy;
	    double energy1 = 0.5 * vtemp.dot(massMat*vtemp) + energyModel.computeEnergy(xtemp);    // total energy
	    Eigen::VectorXd xtemp1, vtemp1;
	    implicitEuler<Problem>(xcur, vcur, h, M, energyModel, xtemp1, vtemp1);
	    double energy2 = 0.5 * vtemp1.dot(massMat*vtemp1) + energyModel.computeEnergy(xtemp1);    // total energy

	    std::cout << "initial energy: " << energy0 << ", NM energy: " << energy1 << ", IE energy: " << energy2 << std::endl;

        xnext= xtemp;
        vnext = vtemp;

	    if(energy1 > energy0 && energy2 < energy0)
	    {
	        //	        if(alpha == 0)
	        //	            implicitEuler<Problem>(xcur, vcur, (1 - alpha) * h, M, energyModel, xnext, vnext);
	        //	        else if(alpha == 1)
	        //	            Newmark<Problem>(xcur, vcur, alpha * h, M, energyModel, xtemp, vtemp, gamma, beta);
	        double tempAlpha = 0.5;
	        double alphaBegin = 0;
	        double alphaEnd = 1;
	        double alphaUpdate = 1;

	        double energyDiff = std::min(std::abs(energy1 - energy0), std::abs(energy2 - energy0));

	        while(alphaUpdate > 1e-6 || energyDiff > 1e-3 * energy0)
	        {
	            Newmark<Problem>(xcur, vcur, tempAlpha * h, M, energyModel, xtemp, vtemp, gamma, beta);
	            implicitEuler<Problem>(xtemp, vtemp, (1 - tempAlpha) * h, M, energyModel, xnext, vnext);
	            energy1 = 0.5 * vnext.dot(massMat*vnext) + energyModel.computeEnergy(xnext);    // total energy
	            if(energy1 > energy0)
	                alphaEnd = tempAlpha;
	            else
	                alphaBegin = tempAlpha;
	            alphaUpdate = alphaEnd - alphaBegin;
	            energyDiff = std::abs(energy1 - energy0);
	            tempAlpha = (alphaBegin + alphaEnd) / 2;
	        }
	        std::cout << "optimal alpha = " << tempAlpha << ", energy difference: " << energyDiff << std::endl;

	    }

		
	}
}