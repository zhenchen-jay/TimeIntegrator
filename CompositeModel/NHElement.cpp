#include <iomanip>
#include <iostream>
#include <fstream>
#include "NHElement.h"


double NHElement::computeElementPotential(double P0, double P1, Eigen::Vector2d *grad, Eigen::Matrix2d *hess)
{
	double mu, lambda;
	getLambdaMu(lambda, mu);
	double energy = 0;

	double drRest = restP1_ - restP0_;

 	double dr = P1 - P0;

	double J = dr / drRest;
	double s = J * J;

	if (J < 0)
	{
		std::cerr << "invert happens in the NHElement model. J = " << J << std::endl;
		exit(1);
	}
	double lnJ = std::log(J);

	energy = 0.5 * (mu * (s - 1 - 2 * lnJ) + lambda * lnJ * lnJ) * std::abs(drRest);

	if(grad || hess)
	{
		Eigen::Vector2d gradDr = Eigen::Vector2d::Zero();
		gradDr << -1, 1;

		Eigen::Vector2d gradJ = gradDr / drRest;
		Eigen::Vector2d grads = 2 * J * gradJ;
		Eigen::Vector2d gradlnJ = gradJ / J;

		if(grad)
		{
			(*grad) = 0.5 * (mu * (grads - 2 * gradlnJ) + 2 * lambda * lnJ * gradlnJ) * std::abs(drRest);
		}
		if(hess)
		{
			Eigen::Matrix2d hesss = 2 * gradJ * gradJ.transpose();
			Eigen::Matrix2d hesslnJ = -gradJ * gradJ.transpose() / (J * J);

			(*hess) = 0.5 * (mu * (hesss - 2 * hesslnJ) + 2 * lambda * (lnJ * hesslnJ + gradlnJ * gradlnJ.transpose())) * std::abs(drRest);
		}

	}

	return energy;
}
