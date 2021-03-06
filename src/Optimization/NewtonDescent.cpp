
#include "../../include/Optimization/LineSearch.h"
#include "../../include/Optimization/NewtonDescent.h"
#include <iomanip>


void OptSolver::newtonSolver(std::function<double(Eigen::VectorXd, Eigen::VectorXd*, Eigen::SparseMatrix<double>*)> objFunc, std::function<double(Eigen::VectorXd, Eigen::VectorXd)> findMaxStep, std::function<void(Eigen::VectorXd)> postIteration, Eigen::VectorXd& x0, int numIter, double gradTol, double xTol, double fTol, bool isPrintInfo)
{
	const int DIM = x0.rows();
	Eigen::VectorXd grad = Eigen::VectorXd::Zero(DIM);
	Eigen::SparseMatrix<double> hessian;

	Eigen::VectorXd neggrad, delta_x;
	double maxStepSize = 1.0;
	double reg = 1e-6;

	int i = 0;
	for (; i < numIter; i++)
	{
		if(isPrintInfo)
			std::cout << "\niter: " << i << std::endl;
		double f = objFunc(x0, &grad, &hessian);

		Eigen::SparseMatrix<double> H = hessian;
		Eigen::SparseMatrix<double> I(DIM, DIM);
		I.setIdentity();
		hessian = H;

		Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > solver(hessian);

		while (solver.info() != Eigen::Success)
		{
			std::cout << "Matrix is not positive definite, current reg = " << reg << std::endl;
			hessian = H + reg * I;
			solver.compute(hessian);
			reg = std::max(2 * reg, 1e-16);
		}

		neggrad = -grad;
		delta_x = solver.solve(neggrad);

		maxStepSize = findMaxStep(x0, delta_x);

		double rate = LineSearch::backtrackingArmijo(x0, grad, delta_x, objFunc, maxStepSize);

		reg *= 0.5;
		reg = std::max(reg, 1e-16);

		if (isPrintInfo)
		{
			std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << "x0: " << x0.transpose() << ", after update: " << (x0 + rate * delta_x).transpose() << std::endl;
			std::cout << "hessian: \n" << H.toDense() << std::endl;
			std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << "neg grad: " << neggrad.transpose() << ", dir: " << delta_x.transpose() << std::endl;
		}

		x0 = x0 + rate * delta_x;

		double fnew = objFunc(x0, &grad, NULL);

		if (isPrintInfo)
		{
			std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << "f_old: " << f << ", f_new: " << fnew << ", rate: " << rate << ", max step size: " << maxStepSize << ", grad norm: " << grad.norm() << ", variable update: " << rate * delta_x.norm() << std::endl;
			
		}
		
		/*if (i == 1)
			system("pause");*/

		postIteration(x0);

		if (grad.norm() < gradTol)
			return;
		if (rate * delta_x.norm() < xTol)
			return;
		if (f - fnew < fTol)
			return;
	}
	if (i >= numIter)
		std::cout << "terminate with reaching the maximum iteration, with gradient L2-norm = " << grad.norm() << std::endl;
}


void OptSolver::testFuncGradHessian(std::function<double(Eigen::VectorXd, Eigen::VectorXd*, Eigen::SparseMatrix<double>*)> objFunc, const Eigen::VectorXd& x0)
{
	Eigen::VectorXd dir = x0;
	dir(0) = 0;
	dir.setRandom();

	Eigen::VectorXd grad;
	Eigen::SparseMatrix<double> H;

	double f = objFunc(x0, &grad, &H);

	for (int i = 3; i < 10; i++)
	{
		double eps = std::pow(0.1, i);
		Eigen::VectorXd x = x0 + eps * dir;
		Eigen::VectorXd grad1;
		double f1 = objFunc(x, &grad1, NULL);

		std::cout << "\neps: " << eps << std::endl;
		std::cout << "energy-gradient: " << (f1 - f) / eps - grad.dot(dir) << std::endl;
		std::cout << "gradient-hessian: " << ((grad1 - grad) / eps - H * dir).norm() << std::endl;

		std::cout << "gradient-difference: \n" << (grad1 - grad) / eps << std::endl;
		std::cout << "direction-hessian: \n" << H * dir << std::endl;
	}
}

