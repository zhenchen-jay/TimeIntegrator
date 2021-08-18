
#include "../../include/Optimization/LineSearch.h"
#include "../../include/Optimization/NewtonDescent.h"

void OptSolver::newtonSolver(std::function<double(Eigen::VectorXd, Eigen::VectorXd*, Eigen::SparseMatrix<double>*)> objFunc, std::function<double(Eigen::VectorXd, Eigen::VectorXd)> findMaxStep, std::function<void(Eigen::VectorXd)> postIteration, Eigen::VectorXd& x0, int numIter, double gradTol, double xTol, double fTol)
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
		//std::cout << "\niter: " << i << std::endl;
		double f = objFunc(x0, &grad, &hessian);

		Eigen::SparseMatrix<double> H = hessian;
		Eigen::SparseMatrix<double> I(DIM, DIM);
		I.setIdentity();
		hessian = H + reg * I;

		Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > solver(hessian);

		while (solver.info() != Eigen::Success)
		{
			reg = std::max(2 * reg, 1e-16);
			std::cout << "Matrix is not positive definite, current reg = " << reg << std::endl;
			hessian = H + reg * I;
			solver.compute(hessian);
		}

		neggrad = -grad;
		delta_x = solver.solve(neggrad);

		maxStepSize = findMaxStep(x0, delta_x);

		double rate = LineSearch::backtrackingArmijo(x0, grad, delta_x, objFunc, maxStepSize);

		reg *= 0.5;
		reg = std::max(reg, 1e-16);

		x0 = x0 + rate * delta_x;

		double fnew = objFunc(x0, &grad, NULL);

		/*std::cout << "f_old: " << f << ", f_new: " << fnew << ", grad norm: " << grad.norm() << ", variable update: " << rate * delta_x.norm() << std::endl;*/

		postIteration(x0);

		if (grad.norm() < gradTol)
			return;
		if (rate * delta_x.norm() < xTol)
			return;
		if (f - fnew < fTol)
			return;
	}
	if (i >= numIter)
		std::cout << "terminate with reaching the maximum iteration." << std::endl;
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

