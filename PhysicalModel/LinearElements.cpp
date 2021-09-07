#include <iomanip>
#include <iostream>
#include <fstream>
#include "LinearElements.h"


double LinearElements::computeElasticPotentialPerface(Eigen::VectorXd q, int faceId)
{
	double energy = 0;
	int v0 = restF_(faceId, 0);
	int v1 = restF_(faceId, 1);

	int reducedV0 = indexMap_[v0];
	int reducedV1 = indexMap_[v1];

	double drRest = restPos_(v1) - restPos_(v0);

 	double dr = 0;
	if (reducedV0 == -1)
	{
		if (reducedV1 == -1)
			return 0;
		else
			dr = q(reducedV1) - restPos_(v0);
	}
	else
	{
		if (reducedV1 == -1)
			dr = restPos_(v1) - q(reducedV0);
		else
			dr = q(reducedV1) - q(reducedV0);
	}

	// for linear elements: strain = 1/2 (d u + d u^T), where assuming that r = r_rest + u
	double strain = (dr - drRest) / drRest;

	energy = (mu_ + lambda_ / 2) * strain * strain * std::abs(drRest);

	return energy;
}

void LinearElements::computeElasticGradientPerface(Eigen::VectorXd q, int faceId, Eigen::Vector2d& grad)
{
	grad.setZero();

	int v0 = restF_(faceId, 0);
	int v1 = restF_(faceId, 1);

	int reducedV0 = indexMap_[v0];
	int reducedV1 = indexMap_[v1];

	double drRest = restPos_(v1) - restPos_(v0);
	double dr = 0;


	Eigen::Vector2d gradDr = Eigen::Vector2d::Zero();

	if (reducedV0 == -1)
	{
		if (reducedV1 == -1)
			return;
		else
		{
			dr = q(reducedV1) - restPos_(v0);
			gradDr << 0, 1;
		}
	}
	else
	{
		if (reducedV1 == -1)
		{
			dr = restPos_(v1); - q(reducedV0);
			gradDr << -1, 0;
		}

		else
		{
			dr = q(reducedV1) - q(reducedV0);
			gradDr << -1, 1;
		}

	}

	// for linear elements: strain = 1/2 (d u + d u^T), where assuming that r = r_rest + u
	double strain = (dr - drRest) / drRest;
	Eigen::Vector2d gradStrain = gradDr / drRest;

	grad = 2.0 * (mu_ + lambda_ / 2) * strain * gradStrain * std::abs(drRest);
}

void LinearElements::computeElasticHessianPerface(Eigen::VectorXd q, int faceId, Eigen::Matrix2d& hess)
{
	hess.setZero();

	int v0 = restF_(faceId, 0);
	int v1 = restF_(faceId, 1);

	int reducedV0 = indexMap_[v0];
	int reducedV1 = indexMap_[v1];

	double drRest = restPos_(v1) - restPos_(v0);
	double dr = 0;


	Eigen::Vector2d gradDr = Eigen::Vector2d::Zero();

	if (reducedV0 == -1)
	{
		if (reducedV1 == -1)
			return;
		else
			gradDr << 0, 1;
	}
	else
	{
		if (reducedV1 == -1)
			gradDr << -1, 0;

		else
			gradDr << -1, 1;

	}
	Eigen::Vector2d gradStrain = gradDr / drRest;
	hess = 2.0 * (mu_ + lambda_ / 2) * gradStrain * gradStrain.transpose() * std::abs(drRest);
}
