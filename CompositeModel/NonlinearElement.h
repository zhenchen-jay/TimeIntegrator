#pragma once
#include "FiniteElement.h"

class NonlinearElement : public FiniteElement
{
public:
	NonlinearElement() {}
	NonlinearElement(double youngs, double poisson, double restP0, double restP1, int vid0, int vid1) : FiniteElement(youngs, poisson, restP0, restP1, vid0, vid1)
	{}

	virtual double computeElementPotential(double P0, double P1, Eigen::Vector2d *grad = NULL, Eigen::Matrix2d *hess = NULL) override;
};