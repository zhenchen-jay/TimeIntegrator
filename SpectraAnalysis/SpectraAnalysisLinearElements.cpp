#include <iostream>
#include "SpectraAnalysisLinearElements.h"

SpectraAnalysisLinearElements::SpectraAnalysisLinearElements(const SimParameters& params, Eigen::VectorXd& q0, Eigen::VectorXd& v0, const LinearElements& model, int numSpectras)
{
	params_ = params;
	q0_ = q0;
	v0_ = v0;
	numSpectras_ = numSpectras;

	model_ = model;

	initialization();
}

void SpectraAnalysisLinearElements::initialization()
{
	// form mass matrix and its inverse
	std::vector<Eigen::Triplet<double>> T, T1;
	for (int i = 0; i < model_.massVec_.size(); i++)
	{
		T.push_back({ i, i, model_.massVec_(i) });
		T1.push_back({ i, i, 1.0 / model_.massVec_(i) });
	}

	massMat_.resize(q0_.size(), q0_.size());
	massInvMat_.resize(q0_.size(), q0_.size());

	massMat_.setFromTriplets(T.begin(), T.end());
	massInvMat_.setFromTriplets(T1.begin(), T.end());

	// compute the eigen modes
	T.clear();
	model_.computeElasticHessian(q0_, T);
	K_.resize(q0_.size(), q0_.size());
	K_.setFromTriplets(T.begin(), T.end());

	Eigen::VectorXd g;
	model_.computeElasticGradient(q0_, g);
	b_ = g - K_ * q0_;

	Spectra::SparseSymMatProd<double> opK(K_);
	Spectra::SparseCholesky<double> opM(massMat_);

	Spectra::SymGEigsSolver<Spectra::SparseSymMatProd<double>, Spectra::SparseCholesky<double>, Spectra::GEigsMode::Cholesky> eigs(opK, opM, numSpectras_, 2 * numSpectras_);

	eigs.init();
	int nconv = eigs.compute(Spectra::SortRule::LargestAlge);
	if (eigs.info() == Spectra::CompInfo::Successful)
	{
		eigenValues_ = eigs.eigenvalues();
		eigenVecs_ = eigs.eigenvectors();
	}
	else
	{
		std::cerr << "error in t computing the eigen values of the M^{-1} K " << std::endl;
		exit(1);
	}

	// check the othonormality
	Eigen::MatrixXd idMat = massMat_;
	idMat.setIdentity();
	std::cout << "error: " << (eigenVecs_.transpose() * massMat_ * eigenValues_ - idMat).norm() << std::endl;
	
	// compute initial alphas and betas
	curAlphaBeta_.resize(numSpectras_);
	preAlphaBeta_.resize(numSpectras_);

	for (int i = 0; i < numSpectras_; i++)
	{
		double alpha = eigenVecs_.col(i).dot(massMat_ * q0_);
		double beta = eigenVecs_.col(i).dot(massMat_ * v0_);
		curAlphaBeta_[i] << alpha, beta;
		preAlphaBeta_[i] = curAlphaBeta_[i];
	}

	// compute the constant part
	cis_.resize(numSpectras_);

	for (int i = 0; i < numSpectras_; i++)
	{
		double value = eigenVecs_.col(i).dot(b_);
		cis_[i] = value;
	}

	// current time
	curTime_ = 0;
}

void SpectraAnalysisLinearElements::updateAlphasBetas()
{
	double h = params_.timeStep;
	for (int i = 0; i < numSpectras_; i++)
	{
		if (params_.integrator == SimParameters::TI_IMPLICIT_EULER)
		{
			Eigen::Matrix2d A;
			A << 1, -h, eigenValues_[i] * h, 1;
			Eigen::Vector2d consVec;
			consVec << 0, -h * cis_[i];

			Eigen::Matrix2d Ainv = A.inverse();

			preAlphaBeta_[i] = curAlphaBeta_[i];
			curAlphaBeta_[i] = Ainv * (preAlphaBeta_[i] + consVec);
		}
		else if (params_.integrator == SimParameters::TI_NEWMARK)
		{
			Eigen::Matrix2d A;
			double NM_beta = params_.NM_beta;
			A << 1 + h * h * eigenValues_[i] * NM_beta, 0, 
				eigenValues_[i] * h / 2.0, 1;
			Eigen::Vector2d consVec;

			Eigen::Matrix2d A1;
			A1 << 1 - h * h * eigenValues_[i] * (0.5 - NM_beta), h,
				-eigenValues_[i] * h / 2.0, 1;

			consVec << -h * h * cis_[i] / 2, -params_.timeStep * cis_[i];

			Eigen::Matrix2d Ainv = A.inverse();

			preAlphaBeta_[i] = curAlphaBeta_[i];
			curAlphaBeta_[i] = Ainv * (A1 * preAlphaBeta_[i] + consVec);
		}
		else if (params_.integrator == SimParameters::TI_TR_BDF2)
		{
			Eigen::Vector2d alphabeta;
			double gamma = params_.TRBDF2_gamma;
			double gamma2 = (1 - 2 * gamma) / (2 - 2 * gamma);
			double gamma3 = (1 - gamma2) / (2 * gamma);

			Eigen::Matrix2d A, A1, A2, Ainv;
			Eigen::Vector2d consVec;

			A << 1, -gamma * h, eigenValues_[i] * gamma* h, 1;
			Ainv = A.inverse();

			A << 1, gamma * h, -eigenValues_[i] * gamma* h, 1;

			consVec << 0, -gamma * h * cis_[i];
			alphabeta = Ainv * (A1 * curAlphaBeta_[i] + consVec);

			A << 1, -gamma2 * h, eigenValues_[i] * gamma2 * h, 1;
			Ainv = A.inverse();
			consVec << 0, -gamma2 * h * cis_[i];

			A1 << gamma3, 0, 0, gamma3;
			A2 << 1 - gamma3, 0, 0, 1 - gamma3;

			preAlphaBeta_[i] = curAlphaBeta_[i];
			curAlphaBeta_[i] = Ainv * (A1 * alphabeta + A2 * preAlphaBeta_[i] + consVec);
		}
		else if (params_.integrator == SimParameters::TI_BDF2)
		{
			if (curTime_ == 0)	// use IE for the first step
			{
				Eigen::Matrix2d A;
				A << 1, -h, eigenValues_[i] * h, 1;
				Eigen::Vector2d consVec;
				consVec << 0, -h * cis_[i];

				Eigen::Matrix2d Ainv = A.inverse();

				preAlphaBeta_[i] = curAlphaBeta_[i];
				curAlphaBeta_[i] = Ainv * (preAlphaBeta_[i] + consVec);
			}
			else
			{
				Eigen::Vector2d alphabeta;

				Eigen::Matrix2d A, A1, A2, Ainv;
				Eigen::Vector2d consVec;

				A << 1, -2.0 / 3.0 * h, eigenValues_[i] * 2.0 / 3.0 * h, 1;
				Ainv = A.inverse();
				consVec << 0, -2.0 / 3.0 * h * cis_[i];

				A1 << 4.0 / 3.0, 0, 0, 4.0 / 3.0;
				A2 << -1.0 / 3.0, 0, 0, -1.0 / 3.0;

				alphabeta = Ainv * (A1 * curAlphaBeta_[i] + A2 * preAlphaBeta_[i] + consVec);

				preAlphaBeta_[i] = curAlphaBeta_[i];
				curAlphaBeta_[i] = alphabeta;
			}
		}
	}
	curTime_ += h;
}