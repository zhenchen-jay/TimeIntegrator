#include <iomanip>
#include <iostream>
#include <fstream>
#include "NeoHookean.h"


double NeoHookean::computeElasticPotentialPerface(Eigen::VectorXd q, int faceId)
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

	double J = dr / drRest;
	double s = J * J;

	if (J < 0)
	{
		std::cerr << "invert happens in the neohookean model. J = " << J << std::endl;
		exit(1);
	}
	double lnJ = std::log(J);

	energy = 0.5 * (mu * (s - 1 - 2 * lnJ) + lambda * lnJ * lnJ) * std::abs(drRest);

	return energy;
}

void NeoHookean::computeElasticGradientPerface(Eigen::VectorXd q, int faceId, Eigen::Vector2d& grad)
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

	double J = dr / drRest;
	if (J < 0)
	{
		std::cerr << "invert happens in the neohookean model. J = " << J << std::endl;
		exit(1);
	}
	double s = J * J;
	double lnJ = std::log(J);

	Eigen::Vector2d gradJ = gradDr / drRest;
	Eigen::Vector2d grads = 2 * J * gradJ;
	Eigen::Vector2d gradlnJ = gradJ / J;

	grad = 0.5 * (mu * (grads - 2 * gradlnJ) + 2 * lambda * lnJ * gradlnJ) * std::abs(drRest);
}

void NeoHookean::computeElasticHessianPerface(Eigen::VectorXd q, int faceId, Eigen::Matrix2d& hess)
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

	double J = dr / drRest;
	if (J < 0)
	{
		std::cerr << "invert happens in the neohookean model. J = " << J << std::endl;
		exit(1);
	}
	double s = J * J;
	double lnJ = std::log(J);

	Eigen::Vector2d gradJ = gradDr / drRest;
	Eigen::Vector2d grads = 2 * J * gradJ;
	Eigen::Vector2d gradlnJ = gradJ / J;

	Eigen::Matrix2d hesss = 2 * gradJ * gradJ.transpose();
	Eigen::Matrix2d hesslnJ = -gradJ * gradJ.transpose() / (J * J);

	hess = 0.5 * (mu * (hesss - 2 * hesslnJ) + 2 * lambda * (lnJ * hesslnJ + gradlnJ * gradlnJ.transpose())) * std::abs(drRest);
}
