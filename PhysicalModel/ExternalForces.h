#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>

namespace ExternalForces
{
	Eigen::VectorXd externalForce(const Eigen::VectorXd pos, double time, double mag = 5000, int pow = 20);
}