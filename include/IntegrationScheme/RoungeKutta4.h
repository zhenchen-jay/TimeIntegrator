#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

namespace TimeIntegrator
{
	/*
* For the system we aimed to solve:
* dx / dt = v
* M dv / dt = F(x)
* where F(x) = -\nabla E(x),
* E(x) = 1/2 x^T M x + potential_energy(x).
*
*
Rounge-Kutta-4 Euler:
x_{n+1} = x_n + h * v_n / 6 (a1 + 2 * b1 + 2 * c1 + d1)
v_{n+1} = v_n + h M^{-1} / 6 (a2 + 2 * b2 + 2 * c2 + d2)

with
a1 = v_n
a2 = M^{-1} F(x_n)

b1 = v_n + h / 2 * a2
b2 = M^{-1} F(x_n + h / 2 * a1)

c1 = v_n + h / 2 * b2
c2 = M^{-1} F(x_n + h / 2 * b1)

d1 = v_n + h / 2 * c2
d2 = M^{-1} F(x_n + h / 2 * c1)
*/

	template <typename Problem>
	void RoungeKutta4(const Eigen::VectorXd& xcur, const Eigen::VectorXd& vcur, const double h, const Eigen::VectorXd& M, Problem energyModel, Eigen::VectorXd& xnext, Eigen::VectorXd& vnext)
	{
		std::vector<Eigen::Triplet<double>> massTrip;
		Eigen::SparseMatrix<double> massMatInv(M.size(), M.size());

		for (int i = 0; i < M.size(); i++)
			massTrip.push_back({ i, i , 1.0 / M(i) });

		massMatInv.setFromTriplets(massTrip.begin(), massTrip.end());


		Eigen::VectorXd a1, a2, b1, b2, c1, c2, d1, d2;
		a1 = vcur;
		energyModel.computeGradient(xcur, a2);
		a2 = -massMatInv * a2;

		b1 = vcur + h / 2 * a2;
		energyModel.computeGradient(xcur + h / 2 * a1, b2);
		b2 = -massMatInv * b2;

		c1 = vcur + h / 2 * b2;
		energyModel.computeGradient(xcur + h / 2 * b1, c2);
		c2 = -massMatInv * c2;

		d1 = vcur + h * c2;
		energyModel.computeGradient(xcur + h / 2 * c1, d2);
		d2 = -massMatInv * d2;

		xnext = xcur + h / 6 * (a1 + 2 * b1 + 2 * c1 + d1);
		vnext = vcur + h / 6 * (a2 + 2 * b2 + 2 * c2 + d2);
	}

}
