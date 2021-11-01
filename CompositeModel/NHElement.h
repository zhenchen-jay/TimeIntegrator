#pragma once
#include "FiniteElement.h"

class NHElement : public FiniteElement
{
public:
	NHElement() {}
	NHElement(double youngs, double poisson, double restP0, double restP1) : FiniteElement(simParams, restP0, restP1)
	{}

	virtual double computeElementPotential(double P0, double P1, Eigen::Vector2d *grad = NULL, Eigen::Matrix2d *hess = NULL) override;
};