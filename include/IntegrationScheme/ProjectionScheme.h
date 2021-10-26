#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "../optimization/NewtonDescent.h"
#include "../../PhysicalModel/SimParameters.h"
#include "BDF2.h"
#include "ImplicitEuler.h"
#include "Newmark.h"
#include "TrapezoidBDF2.h"

namespace TimeIntegrator
{
	/*
	* The implementation of "FEPR: Fast Energy Projection for Real-Time Simulation of Deformable Objects"
	*/

	template <typename Problem>
	void FEPR(const Eigen::VectorXd& xcur, const Eigen::VectorXd& vcur, const double h, const Eigen::VectorXd& M, Problem energyModel, Eigen::VectorXd& xnext, Eigen::VectorXd& vnext, SimParameters::TimeIntegrator baseTI = SimParameters::TimeIntegrator::TI_NEWMARK)
	{
		if (h == 0)
		{
			xnext = xcur;
			vnext = vcur;
		}
		std::vector<Eigen::Triplet<double>> massTrip;
		Eigen::SparseMatrix<double> massMat(M.size(), M.size());

		for (int i = 0; i < M.size(); i++)
			massTrip.push_back({ i, i , M(i) });
		massMat.setFromTriplets(massTrip.begin(), massTrip.end());

		massTrip.clear();
		Eigen::SparseMatrix<double> massMatInv(M.size(), M.size());

		for (int i = 0; i < M.size(); i++)
			massTrip.push_back({ i, i , 1.0 / M(i) });
		massMatInv.setFromTriplets(massTrip.begin(), massTrip.end());

		Eigen::VectorXd xPred, vPred;
		if (baseTI == SimParameters::TimeIntegrator::TI_NEWMARK)
			Newmark(xcur, vcur, h, M, energyModel, xPred, vPred, 0.5, 0.25);
		else
			implicitEuler(xcur, vcur, h, M, energyModel, xPred, vPred);
		

		// do the projection:
		/*
		*    min_{x, v, s} 1/2 (x - x_{n+1})^T M (x - x_{n+1}) + h^2 / 2 (v - v_{n+1})^T M (v - v_{n+1}) + eps / 2 s^2, (eps = 1e-4)
		* s.t.
		*    H(x, v) = 1/2 v^T M v + V(x) = H(x_n, v_n)
		*    P(v) = \sum m^(i) v^(i) = s P(v_{n}) + (1 - s) P(v_{n+1})
		* 
		* Let q = (x, v, s), q_{n+1} = (x_{n+1}, v_{n+1}, 0), D = diag(M, h^2 M, eps), 
		*      min_q = 1/2 (q - q_{n+1})^T D (q - q_{n+1}) = 1/2 ||q - q_{n+1}||_D
		* s.t.
		*     c_1(q) = 0, (\nabla c_1(q) = [\nabla V(x), M v, 0])
		*     c_2(q) = 0, (\nabla c_2(q) = [0, M_vec, P(v_n) - P(v_{n+1})], M_vec = [m0, ...., m_N])
		* 
		* As suggested by the FEPR paper, we use SQP, start with q^0 = q_{n+1}, solve the following
		* approximate problem:
		*     min_q = 1/2 ||q - q^k||_D
		* s.t.
		*     c_1(q^k) + \nabla c_1(q^k)^T (q - q^k) = 0
		*     c_2(q^k) + \nabla c_2(q^k)^T (q - q^k) = 0
		* This can be solved by Lagrange multiplier. (stop with max(|c_1|, |c_2|) < 1e-7)
		*/
		
		auto totalEnergy = [&](Eigen::VectorXd x, Eigen::VectorXd v, Eigen::VectorXd* grad = NULL, std::vector<Eigen::Triplet<double>>* hess = NULL)
		{
			double E = 1.0 / 2.0 * v.transpose() * massMat * v + energyModel.computeEnergy(x);
			int nverts = x.rows();
			if (grad)
			{
				Eigen::VectorXd g1;
				energyModel.computeGradient(x, g1);
				grad->setZero(2 * nverts);

				grad->segment(0, nverts) = g1;
				grad->segment(nverts, nverts) = massMat * v;

			}

			if (hess)
			{
				Eigen::SparseMatrix<double> H;
				energyModel.computeHessian(x, H);

				for (int k = 0; k < H.outerSize(); ++k)
					for (Eigen::SparseMatrix<double>::InnerIterator it(H, k); it; ++it)
						hess->push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));

				for (int i = 0; i < nverts; i++)
					hess->push_back({ nverts + i , nverts + i,  M(i) });
			}

			return E;
		};

		auto totalLinearMomentum = [&](Eigen::VectorXd v, Eigen::VectorXd* grad = NULL)
		{
			double P = 0;
			int nverts = v.rows();
			for (int i = 0; i < nverts; i++)
				P += v(i) * M(i);

			if (grad)
				(*grad) = M;
			return P;
		};

		int nverts = xcur.size();
		double Hn = totalEnergy(xcur, vcur);
		double Hnplus1 = totalEnergy(xPred, vPred);
		double Pn = totalLinearMomentum(vcur);
		double Pnplus1 = totalLinearMomentum(vPred);

		auto constraints = [&](Eigen::VectorXd x, Eigen::VectorXd v, double s, Eigen::MatrixXd* grad)
		{
			Eigen::Vector2d c(0, 0);

			c(0) = totalEnergy(x, v) - Hn;
			c(1) = totalLinearMomentum(v) - s * Pn - (1 - s) * Pnplus1;

			if (grad)
			{
				grad->setZero(2, 2 * xcur.size() + 1);

				Eigen::VectorXd energyGrad, momentumGrad;
				double E = totalEnergy(x, v, &energyGrad);
				double P = totalLinearMomentum(v, &momentumGrad);

				grad->row(0).segment(0, 2 * xcur.size()) = energyGrad;
				grad->row(1).segment(xcur.size(), xcur.size()) = momentumGrad;
				(*grad)(1, 2 * xcur.size()) = Pn - Pnplus1;
			}
			return c;
		};

		// form the matrix D and D^{-1}
		std::vector<Eigen::Triplet<double>> T, Tinv;
		for (int i = 0 ; i < nverts; i++)
		{
			T.push_back({ i, i, M(i) });
			T.push_back({ i + nverts, i + nverts, M(i) * h * h });

			Tinv.push_back({ i, i, 1.0 / M(i) });
			Tinv.push_back({ i + nverts, i + nverts, 1.0 / (M(i) * h * h) });
		}
			
		T.push_back({ 2 * nverts, 2 * nverts, 1e-4 });
		Tinv.push_back({ 2 * nverts, 2 * nverts, 1e4 });

		Eigen::SparseMatrix<double> D, Dinv;
		D.resize(2 * nverts + 1, 2 * nverts + 1);
		Dinv.resize(2 * nverts + 1, 2 * nverts + 1);

		D.setFromTriplets(T.begin(), T.end());
		Dinv.setFromTriplets(T.begin(), T.end());

		Eigen::VectorXd xk = xPred;
		Eigen::VectorXd vk = vPred;
		double sk = 0;

		for (int iter = 0; iter < 1000; iter++)
		{
			Eigen::Vector2d c;
			Eigen::MatrixXd gradc;
			
			c = constraints(xk, vk, sk, &gradc);
			std::cout << "inner iter: " << iter << ", const norm: " << c.lpNorm<1>() << std::endl;
			

			Eigen::Matrix2d A = gradc * Dinv * gradc.transpose();
			Eigen::Vector2d lambda = A.colPivHouseholderQr().solve(c);

			Eigen::VectorXd update = Dinv * gradc.transpose() * lambda;

			std::cout << "(pos, update): "<< std::endl;
			for (int i = 0; i < nverts; i++)
			{
				std::cout << "(" << xk(i) << ", " << update(i) << ")" << std::endl;
			}

			xk = xk - update.segment(0, nverts);
			vk = vk - update.segment(nverts, nverts);
			sk = sk - update(2 * nverts);
			
			if (c.lpNorm<1>() < 1e-7)
				break;
		}
		xnext = xk;
		vnext = vk;
		std::cout << "energy before update: " << Hn << ", energy got by base TI: " << Hnplus1 << ", energy after projection: " << totalEnergy(xnext, vnext) << std::endl;
	}

}