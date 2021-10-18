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
	void splitScheme(const Eigen::VectorXd& xcur, const Eigen::VectorXd& vcur, const double h, const Eigen::VectorXd& M, Problem energyModel, Eigen::VectorXd& xnext, Eigen::VectorXd& vnext, double alpha = 0.5, SimParameters::TimeIntegrator firstTI = SimParameters::TimeIntegrator::TI_NEWMARK, SimParameters::TimeIntegrator secondTI = SimParameters::TimeIntegrator::TI_IMPLICIT_EULER, double gamma = 0.5, double beta = 0.25)
	{
		Eigen::VectorXd xtemp, vtemp;
		if(alpha == 0)
			implicitEuler<Problem>(xtemp, vtemp, (1 - alpha) * h, M, energyModel, xnext, vnext);
		else if(alpha == 1)
			Newmark<Problem>(xcur, vcur, alpha * h, M, energyModel, xtemp, vtemp, gamma, beta);
		else
		{
			Newmark<Problem>(xcur, vcur, alpha * h, M, energyModel, xtemp, vtemp, gamma, beta);
			implicitEuler<Problem>(xtemp, vtemp, (1 - alpha) * h, M, energyModel, xnext, vnext);
		}
		
	}
}