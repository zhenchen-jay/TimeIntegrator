#pragma once
#include "PhysicalModel.h"

class StVK : public PhysicalModel
{
public:
	StVK() {}
	StVK(SimParameters simParams) : PhysicalModel(simParams)
	{}

	virtual double computeElasticPotentialPerface(Eigen::VectorXd q, int faceId) override;
	virtual void computeElasticGradientPerface(Eigen::VectorXd q, int faceId, Eigen::Vector2d& grad) override;
	virtual void computeElasticHessianPerface(Eigen::VectorXd q, int faceId, Eigen::Matrix2d& hess) override;
};