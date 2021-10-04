#include "ExternalForces.h"

Eigen::VectorXd ExternalForces::externalForce(const Eigen::VectorXd pos, double time, double mag, int pow)
{
	double M_PI = 3.1415926;
	Eigen::VectorXd forces = Eigen::VectorXd::Zero(pos.size());
	double f = mag * std::pow(std::sin(M_PI * time), pow);
	forces(forces.size() - 1) = f;
	return forces;
}