#pragma once
#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "../optimization/NewtonDescent.h"


/*
* For the system we aimed to solve:
* dx / dt = v
* M dv / dt = F(x)
* where F(x) = -\nabla E(x),
* E(x) = 1/2 x^T M x + potential_energy(x).
*
*
Implicit (Backward) Euler:
x_{n+1} = x_n + h * v_{n+1}
v_{n+1} = v_n + h M^{-1} F(x_{n+1})

=>
	x_{n+1} = x_n + h v_n + h^2 M^{-1} F(x_{n+1})
=>
	x_{n+1} = min_y 1/2 (y - x_n - hv_n)^T M (y - x_n - hv_n) - h^2 E(y)  (*)
*/

template <typename Problem> 
class implicitEulerEnergy
{
public:
	implicitEulerEnergy(const Eigen::VectorXd& xcur, const Eigen::VectorXd& vcur, const double h, const Eigen::SparseMatrix<double>& massMat, Problem energyModel)
	{
		_xcur = xcur;
		_vcur = vcur;
		_h = h;
		_massMat = massMat;
		_energyModel = energyModel;
	}

	double energy(const Eigen::VectorXd& x)
	{
		double E = 0.5 * (x - _xcur - _h * _vcur).transpose() * _massMat * (x - _xcur - _h * _vcur) - _h * _h * _energyModel.value(x);
		return E;
	}

	void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& grad)
	{
		_energyModel.gradient(x, grad);
		grad = _massMat * (x - _xcur - _h * _vcur) - _h * _h * grad;
	}

	void hessian(const Eigen::VectorXd& x, Eigen::SparseMatrix<double>& H)
	{
		_energyModel.hessian(x, H);
		H = _massMat - _h * _h * H;
	}

private:
	Eigen::VectorXd _xcur;
	Eigen::VectorXd _vcur;
	double _h;
	Eigen::SparseMatrix<double> _massMat;
	Problem _energyModel;

};

template <typename Problem>
void implicitEuler(const Eigen::VectorXd& xcur, const Eigen::VectorXd& vcur, const double h, const Eigen::VectorXd& M, Problem energyModel, Eigen::VectorXd& xnext, Eigen::VectorXd& vnext)
{
	using namespace Eigen;
	std::vector<Triplet> massTrip;
	SparseMatrix<double> massMat(M.size(), M.size());

	for (int i = 0; i < M.size(); i++)
		massTrip.push_back({ i, i , M(i) });
	massMat.setFromTriplets(massTrip.begin(), massTrip.end());

	implicitEulerEnergy<Problem> model(xcur, vcur, h, massMat, energyModel);
	
	// newton step to find the optimal
	NewtonDescentSolver<implicitEulerEnergy> newtonSolver;
	xnext = xcur;
	newtonSolver.minimize(model, xnext);

	vnext = (xnext - xcur) / h;


}