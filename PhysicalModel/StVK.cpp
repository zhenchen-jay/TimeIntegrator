#include <iomanip>
#include <iostream>
#include <fstream>
#include "StVK.h"


double StVK::computeElasticPotentialPerface(Eigen::VectorXd q, int faceId)
{
	double mu = params_.youngsList[faceId] / (2 * (1 + params_.poisson));
	double lambda = params_.youngsList[faceId] * params_.poisson / (1 + params_.poisson);
	// for 3d lambda = Y * nu / (1 + nv) / (1 - 2 nv), for 2d  lambda = Y * nu / (1 + nv) / (1 - nv), and for 1d lambda = Y * nu / (1 + nv). In one dimensional case, the stiffness is 2 * mu + lambda = Y

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

	double strain = 0.5 * (dr * dr / (drRest * drRest) - 1);
	energy = (mu + lambda / 2) * strain * strain * std::abs(drRest);

	return energy;
}

void StVK::computeElasticGradientPerface(Eigen::VectorXd q, int faceId, Eigen::Vector2d& grad)
{
	double mu = params_.youngsList[faceId] / (2 * (1 + params_.poisson));
	double lambda = params_.youngsList[faceId] * params_.poisson / (1 + params_.poisson);
	// for 3d lambda = Y * nu / (1 + nv) / (1 - 2 nv), for 2d  lambda = Y * nu / (1 + nv) / (1 - nv), and for 1d lambda = Y * nu / (1 + nv). In one dimensional case, the stiffness is 2 * mu + lambda = Y

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

	double strain = 0.5 * (dr * dr / (drRest * drRest) - 1);
	Eigen::Vector2d gradStrain = gradDr  * dr / (drRest * drRest);

	grad = 2.0 * (mu + lambda / 2) * strain * gradStrain * std::abs(drRest);
}

void StVK::computeElasticHessianPerface(Eigen::VectorXd q, int faceId, Eigen::Matrix2d& hess)
{
	double mu = params_.youngsList[faceId] / (2 * (1 + params_.poisson));
	double lambda = params_.youngsList[faceId] * params_.poisson / (1 + params_.poisson);
	// for 3d lambda = Y * nu / (1 + nv) / (1 - 2 nv), for 2d  lambda = Y * nu / (1 + nv) / (1 - nv), and for 1d lambda = Y * nu / (1 + nv). In one dimensional case, the stiffness is 2 * mu + lambda = Y

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

	double strain = 0.5 * (dr * dr / (drRest * drRest) - 1);
	Eigen::Vector2d gradStrain = gradDr * dr / (drRest * drRest);
	Eigen::Matrix2d hessStrain = gradDr * gradDr.transpose() / (drRest * drRest);
	
	hess = 2.0 * (mu + lambda / 2) * (gradStrain * gradStrain.transpose() + strain * hessStrain) * std::abs(drRest);
}
