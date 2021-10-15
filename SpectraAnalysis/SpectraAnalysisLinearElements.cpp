#include <iostream>
#include <filesystem>
#include <Eigen/Eigenvalues>
#include <fstream>
#include "SpectraAnalysisLinearElements.h"
#include "../PhysicalModel/ExternalForces.h"

SpectraAnalysisLinearElements::SpectraAnalysisLinearElements(const SimParameters& params, Eigen::VectorXd& q0, Eigen::VectorXd& v0, const LinearElements& model)
{
	params_ = params;
	q0_ = q0;
	v0_ = v0;
	numSpectras_ = params.numSpectra;

	if (numSpectras_ > q0.size())
		numSpectras_ = q0.size();

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
	massInvMat_.setFromTriplets(T1.begin(), T1.end());

	// compute the eigen modes
	model_.computeHessian(q0_, K_);

	Eigen::VectorXd g;
	model_.computeGradient(q0_, g);
	b_ = g - K_ * q0_;


	// check the energy
	Eigen::VectorXd testQ = q0_;
	testQ.setZero();
	double E0 = model_.computeEnergy(testQ);
	double E = model_.computeEnergy(q0_);
	double E1 = 0.5 * q0_.dot(K_ * q0_) + q0_.dot(b_) + E0;


	std::cout << "E = " << E << ", E1 = " << E1 << ", error = " << E1 - E << std::endl;

	
	testQ.setRandom();
	E = model_.computeEnergy(testQ);
	E1 = 0.5 * testQ.dot(K_ * testQ) + testQ.dot(b_) + E0;
	std::cout << "E = " << E << ", E1 = " << E1 << ", error = " << E1 - E << std::endl;

	if (numSpectras_ < q0_.size())
	{
		Spectra::SparseSymMatProd<double> opK(K_);
		Spectra::SparseCholesky<double> opM(massMat_);

		Spectra::SymGEigsSolver<Spectra::SparseSymMatProd<double>, Spectra::SparseCholesky<double>, Spectra::GEigsMode::Cholesky> eigs(opK, opM, numSpectras_, (2 * numSpectras_ > q0_.size()) ? q0_.size() : 2 * numSpectras_);

		eigs.init();
		int nconv = eigs.compute(Spectra::SortRule::LargestMagn);
		if (eigs.info() == Spectra::CompInfo::Successful)
		{
			std::cout << eigs.eigenvalues() << std::endl;
			eigenValues_ = eigs.eigenvalues();
			eigenVecs_ = eigs.eigenvectors();
		}
		else
		{
			std::cerr << "error in t computing the eigen values of the M^{-1} K " << std::endl;
			exit(1);
		}
	}
	
	else
	{
		int halfNum = numSpectras_ / 2;
		Spectra::SparseSymMatProd<double> opK(K_);
		Spectra::SparseCholesky<double> opM(massMat_);

		Spectra::SymGEigsSolver<Spectra::SparseSymMatProd<double>, Spectra::SparseCholesky<double>, Spectra::GEigsMode::Cholesky> eigs(opK, opM, halfNum, numSpectras_);

		eigs.init();
		eigs.compute(Spectra::SortRule::LargestMagn);
		
		int leftNum = numSpectras_ - halfNum;

		using OpType = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
		using BOpType = Spectra::SparseSymMatProd<double>;
		OpType op(K_, massMat_);
		BOpType Bop(massMat_);

		Spectra::SymGEigsShiftSolver<OpType, BOpType, Spectra::GEigsMode::ShiftInvert> eigs1(op, Bop, leftNum, numSpectras_, 0.0);
		eigs1.init();
		eigs1.compute(Spectra::SortRule::LargestMagn);

		if (eigs.info() == Spectra::CompInfo::Successful && eigs1.info() == Spectra::CompInfo::Successful)
		{
			std::cout << eigs.eigenvalues() << std::endl;
			std::cout << eigs1.eigenvalues() << std::endl;

			eigenValues_.resize(numSpectras_);
			eigenValues_.segment(0, halfNum) = eigs.eigenvalues();
			eigenValues_.segment(halfNum, leftNum) = eigs1.eigenvalues();

			eigenVecs_.resize(q0_.size(), numSpectras_);
			eigenVecs_.block(0, 0, q0_.size(), halfNum) = eigs.eigenvectors();
			eigenVecs_.block(0, halfNum, q0_.size(), leftNum) = eigs1.eigenvectors();
		}
		else
		{
			// super low when size of q is large
			Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> ges;
			ges.compute(K_.toDense(), massMat_.toDense());
			eigenValues_ = ges.eigenvalues().real();
			eigenVecs_ = ges.eigenvectors().real();
		}
		
	}
	

	// check the othonormality
	Eigen::MatrixXd idMat = Eigen::MatrixXd::Identity(numSpectras_, numSpectras_);
	std::cout << "error: " << (eigenVecs_.transpose() * massMat_ * eigenVecs_ - idMat).norm() << std::endl;
	
	// compute initial alphas and betas
	curAlphaBeta_.resize(numSpectras_);
	preAlphaBeta_.resize(numSpectras_);
	initialAlphaBeta_.resize(numSpectras_);
	curAlphaBetaTheo_.resize(numSpectras_);

	for (int i = 0; i < numSpectras_; i++)
	{
		double alpha = eigenVecs_.col(i).dot(massMat_ * q0_);
		double beta = eigenVecs_.col(i).dot(massMat_ * v0_);
		initialAlphaBeta_[i] << alpha, beta;
		preAlphaBeta_[i] = initialAlphaBeta_[i];
		curAlphaBeta_[i] = initialAlphaBeta_[i];
		curAlphaBetaTheo_[i] = initialAlphaBeta_[i];
	}

	// compute the constant part
	/*cis_.resize(numSpectras_);

	for (int i = 0; i < numSpectras_; i++)
	{
		double value = eigenVecs_.col(i).dot(b_);
		cis_[i] = value;
	}*/

	// current time
	curTime_ = 0;

	updateCis();
}

void SpectraAnalysisLinearElements::updateCis()
{
	Eigen::VectorXd extForce = ExternalForces::externalForce(q0_, curTime_, params_.impulseMag, params_.impulsePow);
	Eigen::VectorXd c = b_ - extForce;

	// compute the constant part
	cis_.resize(numSpectras_);

	for (int i = 0; i < numSpectras_; i++)
	{
		double value = eigenVecs_.col(i).dot(c);
		cis_[i] = value;
	}
}

void SpectraAnalysisLinearElements::updateAlphasBetas()
{
	double h = params_.timeStep;
	updateCis();
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

			A1 << 1, gamma * h, -eigenValues_[i] * gamma* h, 1;

			consVec << 0, -2 * gamma * h * cis_[i];
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

	// compute the theoretical alpha beta
	for (int i = 0; i < numSpectras_; i++)
	{
		curAlphaBetaTheo_[i](0) = (initialAlphaBeta_[i](0) + cis_[i] / eigenValues_[i]) * std::cos(std::sqrt(eigenValues_[i]) * curTime_) - cis_[i] / eigenValues_[i];
		curAlphaBetaTheo_[i](1) = (initialAlphaBeta_[i](0) + cis_[i] / eigenValues_[i]) * std::sin(std::sqrt(eigenValues_[i]) * curTime_) * std::sqrt(eigenValues_[i]);
	}
}

void SpectraAnalysisLinearElements::getCurPosVel(Eigen::VectorXd& pos, Eigen::VectorXd& vel)
{
	if (curAlphaBeta_.size() != numSpectras_)
	{
		std::cerr << "mismatch in alpha beta vector size and number of spectras." << std::endl;
		exit(1);
	}

	pos.setZero(eigenVecs_.rows());
	vel.setZero(eigenVecs_.rows());

	for (int i = 0; i < numSpectras_; i++)
	{
		pos += curAlphaBeta_[i](0) * eigenVecs_.col(i);
		vel += curAlphaBeta_[i](1) * eigenVecs_.col(i);
	}
}

void SpectraAnalysisLinearElements::getTheoPosVel(Eigen::VectorXd& pos, Eigen::VectorXd& vel)
{
	if (curAlphaBetaTheo_.size() != numSpectras_)
	{
		std::cerr << "mismatch in alpha beta vector size and number of spectras." << std::endl;
		exit(1);
	}

	pos.setZero(eigenVecs_.rows());
	vel.setZero(eigenVecs_.rows());

	for (int i = 0; i < numSpectras_; i++)
	{
		pos += curAlphaBetaTheo_[i](0) * eigenVecs_.col(i);
		vel += curAlphaBetaTheo_[i](1) * eigenVecs_.col(i);
	}
}


void SpectraAnalysisLinearElements::saveInfo(std::string outputFolder)
{
	if (!std::filesystem::exists(outputFolder))
	{
		std::cout << "create directory: " << outputFolder << std::endl;
		if (!std::filesystem::create_directories(outputFolder))
		{
			std::cout << "create folder failed." << outputFolder << std::endl;
			exit(1);
		}
	}

	std::string evalsFileName = outputFolder + "evals.txt";
    if(curTime_ == 0)
    {
        std::ofstream  efs;
        efs.open(evalsFileName, std::ofstream::out);
        for (int i = 0; i < numSpectras_; i++)
        {
            efs << eigenValues_[i] << std::endl;
        }
    }

	for (int i = 0; i < numSpectras_; i++)
	{
		std::string alphafileName = outputFolder + "alpha_" + std::to_string(i) + ".txt";
		std::string theoAlphafileName = outputFolder + "alpha_theo_" + std::to_string(i) + ".txt";

		std::string betafileName = outputFolder + "beta_" + std::to_string(i) + ".txt";
		std::string theoBetafileName = outputFolder + "beta_theo_" + std::to_string(i) + ".txt";

		std::string extFfileName = outputFolder + "c_" + std::to_string(i) + ".txt";

		std::ofstream afs, atfs, bfs, btfs, cfs;

		if (curTime_ == 0)
		{
			afs.open(alphafileName, std::ofstream::out);
			atfs.open(theoAlphafileName, std::ofstream::out);

			bfs.open(betafileName, std::ofstream::out);
			btfs.open(theoBetafileName, std::ofstream::out);

			cfs.open(extFfileName, std::ofstream::out);
		}
		else
		{
			afs.open(alphafileName, std::ofstream::out | std::ofstream::app);
			atfs.open(theoAlphafileName, std::ofstream::out | std::ofstream::app);

			bfs.open(betafileName, std::ofstream::out | std::ofstream::app);
			btfs.open(theoBetafileName, std::ofstream::out | std::ofstream::app);

			cfs.open(extFfileName, std::ofstream::out | std::ofstream::app);
		}

		afs << curAlphaBeta_[i](0) << std::endl;
		atfs << curAlphaBetaTheo_[i](0) << std::endl;

		bfs << curAlphaBeta_[i](1) << std::endl;
		btfs << curAlphaBetaTheo_[i](1) << std::endl;

		cfs << cis_[i] << std::endl;
	}
}