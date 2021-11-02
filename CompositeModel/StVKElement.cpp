#include <iomanip>
#include <iostream>
#include <fstream>
#include "StVKElement.h"


double StVKElement::computeElementPotential(double P0, double P1, Eigen::Vector2d *grad, Eigen::Matrix2d *hess)
{
	double mu, lambda;
	getLambdaMu(lambda, mu);
	double energy = 0;

	double drRest = restP1_ - restP0_;

 	double dr = P1 - P0;

	double strain = 0.5 * (dr * dr / (drRest * drRest) - 1);
	energy = (mu + lambda / 2) * strain * strain * std::abs(drRest);

	if(grad || hess)
	{
		Eigen::Vector2d gradDr = Eigen::Vector2d::Zero();
		gradDr << -1, 1;

		Eigen::Vector2d gradStrain = gradDr  * dr / (drRest * drRest);

		if(grad)
		{
			(*grad) = 2.0 * (mu + lambda / 2) * strain * gradStrain * std::abs(drRest);
		}
		if(hess)
		{
			
			Eigen::Matrix2d hessStrain = gradDr * gradDr.transpose() / (drRest * drRest);
			
			(*hess) = 2.0 * (mu + lambda / 2) * (gradStrain * gradStrain.transpose() + strain * hessStrain) * std::abs(drRest);
		}
	}

	return energy;
}
