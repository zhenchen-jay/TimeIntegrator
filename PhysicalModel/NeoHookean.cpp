#include <iomanip>
#include <iostream>
#include <fstream>
#include "NeoHookean.h"


double NeoHookean::computeElasticPotentialPerface(Eigen::VectorXd q, int faceId)
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

	double s = dr * dr / (drRest * drRest);
	double lnJ = std::log(s) / 2;

	energy = 0.25 * (mu_ * (s - 1 - 2 * lnJ) + lambda_ * lnJ * lnJ) * std::abs(drRest);

	return energy;
}

void NeoHookean::computeElasticGradientPerface(Eigen::VectorXd q, int faceId, Eigen::Vector2d& grad)
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
			dr = restPos_(v1); -q(reducedV0);
			gradDr << -1, 0;
		}

		else
		{
			dr = q(reducedV1) - q(reducedV0);
			gradDr << -1, 1;
		}

	}

	double s = dr * dr / (drRest * drRest);
	double lnJ = std::log(s) / 2;

	Eigen::Vector2d grads = 2 * gradDr * dr / (drRest * drRest);
	Eigen::Vector2d gradlnJ = grads / (2 * s);

	grad = 0.25 * (mu_ * (grads - 2 * gradlnJ) + 2 * lambda_ * lnJ * gradlnJ) * std::abs(drRest);
}

void NeoHookean::computeElasticHessianPerface(Eigen::VectorXd q, int faceId, Eigen::Matrix2d& hess)
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
	    {
	        dr = q(reducedV1) - restPos_(v0);
	        gradDr << 0, 1;
	    }
	}
	else
	{
	    if (reducedV1 == -1)
	    {
	        dr = restPos_(v1); -q(reducedV0);
	        gradDr << -1, 0;
	    }

	    else
	    {
	        dr = q(reducedV1) - q(reducedV0);
	        gradDr << -1, 1;
	    }

	}
	double s = dr * dr / (drRest * drRest);
	double lnJ = std::log(s) / 2;

	Eigen::Vector2d grads = 2 * gradDr * dr / (drRest * drRest);
	Eigen::Vector2d gradlnJ = grads / (2 * s);

	Eigen::Matrix2d hesss = 2 * gradDr * gradDr.transpose() / (drRest * drRest);
	Eigen::Matrix2d hesslnJ = hesss / (2 * s) - grads * grads.transpose() / (2 * s * s);

	hess = 0.25 * (mu_ * (hesss - 2 * hesslnJ) + 2 * lambda_ * (lnJ * hesslnJ + gradlnJ * gradlnJ.transpose())) * std::abs(drRest);
}
