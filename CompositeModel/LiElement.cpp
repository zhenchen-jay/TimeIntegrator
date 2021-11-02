#include <iomanip>
#include <iostream>
#include <fstream>
#include "LiElement.h"


double LiElement::computeElementPotential(double P0, double P1, Eigen::Vector2d *grad, Eigen::Matrix2d *hess)
{
	double mu, lambda;
	getLambdaMu(lambda, mu);
	double energy = 0;

	double drRest = restP1_ - restP0_;

 	double dr = P1 - P0;

	// for linear elements: strain = 1/2 (d u + d u^T), where assuming that r = r_rest + u
	double strain = (dr - drRest) / drRest;

	energy = (mu + lambda / 2) * strain * strain * std::abs(drRest);

	if(grad || hess)
	{
		Eigen::Vector2d gradDr;
		gradDr << -1, 1;
		Eigen::Vector2d gradStrain = gradDr / drRest;

		if(grad)
		{
			(*grad) = 2.0 * (mu + lambda / 2) * strain * gradStrain * std::abs(drRest);
		}
		if(hess)
		{
			(*hess) = 2.0 * (mu + lambda / 2) * gradStrain * gradStrain.transpose() * std::abs(drRest);
		}
	}

	return energy;
}
