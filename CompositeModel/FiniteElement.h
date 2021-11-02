#pragma once
#include <Eigen/Dense>

class FiniteElement
{
public:
	FiniteElement(){}
	FiniteElement(double youngs, double poisson, double restP0, double restP1, int vid0, int vid1) : 
	youngs_(youngs),
	poisson_(poisson),
	restP0_(restP0),
	restP1_(restP1),
	vid0_(vid0),
	vid1_(vid1)
	{}

	virtual double computeElementPotential(double P0, double P1, Eigen::Vector2d *grad = NULL, Eigen::Matrix2d *hess = NULL) = 0;

protected:
	void getLambdaMu(double& lambda, double& mu)
	{
		mu = youngs_ / (2 * (1 + poisson_));
		lambda = youngs_ * poisson_ / (1 + poisson_);
	// for 3d lambda = Y * nu / (1 + nv) / (1 - 2 nv), for 2d  lambda = Y * nu / (1 + nv) / (1 - nv), and for 1d lambda = Y * nu / (1 + nv). In one dimensional case, the stiffness is 2 * mu + lambda = Y

	}


public:
	double youngs_;
	double poisson_;

	int vid0_;
	int vid1_;


	double restP0_;
	double restP1_;
};