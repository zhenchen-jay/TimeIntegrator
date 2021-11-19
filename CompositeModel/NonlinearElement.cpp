#include <iomanip>
#include <iostream>
#include <fstream>
#include "NonlinearElement.h"


double NonlinearElement::computeElementPotential(double P0, double P1, Eigen::Vector2d *grad, Eigen::Matrix2d *hess)
{
	double mu, lambda;
	getLambdaMu(lambda, mu);
	double energy = 0;

	double drRest = restP1_ - restP0_;

 	double dr = P1 - P0;

    double d = drRest * nonlinearRatio_;

	double strain = (drRest - d) / 2 * (dr - drRest) * std::log((dr - d) / (drRest - d));
	energy = youngs_ * strain * std::abs(drRest);

	if(grad || hess)
	{
		Eigen::Vector2d gradDr = Eigen::Vector2d::Zero();
		gradDr << -1, 1;

		Eigen::Vector2d gradStrain = gradDr  * ((drRest - d) / 2 * std::log((dr - d) / (drRest - d)) + (dr - drRest) * (drRest - d) / (2 * (dr - d)) );

		if(grad)
		{
			(*grad) = youngs_ * gradStrain * std::abs(drRest);
		}
		if(hess)
		{
			double H = (drRest - d) / (dr - d) - (dr - drRest) * (drRest - d) / (2 * (dr - d) * (dr - d));

			Eigen::Matrix2d hessStrain = gradDr * gradDr.transpose() * H;
			
			(*hess) = youngs_ * hessStrain * std::abs(drRest);
		}
	}

	return energy;
}
