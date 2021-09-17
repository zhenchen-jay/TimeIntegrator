#pragma once

#include <MatOp/SparseCholesky.h>
#include <MatOp/SparseRegularInverse.h>
#include <MatOp/SymShiftInvert.h>
#include <MatOp/SparseSymMatProd.h>
#include <SymGEigsSolver.h>
#include <SymGEigsShiftSolver.h>

#include "../PhysicalModel/SimParameters.h"
#include "../PhysicalModel/PhysicalModel.h"
#include "../PhysicalModel/LinearElements.h"

class SpectraAnalysisLinearElements
{
public:
	SpectraAnalysisLinearElements() {}
	~SpectraAnalysisLinearElements() {}

	SpectraAnalysisLinearElements(const SimParameters& params, Eigen::VectorXd& q0, Eigen::VectorXd& v0, const LinearElements& model);
	
	void initialization();
	void updateAlphasBetas();
	
	void getCurPosVel(Eigen::VectorXd &pos, Eigen::VectorXd &vel);
	void getTheoPosVel(Eigen::VectorXd& pos, Eigen::VectorXd& vel);

	void saveInfo(std::string outputFolder);


public:
	SimParameters params_;
	Eigen::VectorXd q0_;
	Eigen::VectorXd v0_;
	int numSpectras_;
	std::vector<Eigen::Vector2d> curAlphaBeta_;
	std::vector<Eigen::Vector2d> preAlphaBeta_;
	std::vector<Eigen::Vector2d> initialAlphaBeta_;
	std::vector<Eigen::Vector2d> curAlphaBetaTheo_;

	std::vector<double> cis_;
	Eigen::MatrixXd eigenVecs_;
	Eigen::VectorXd eigenValues_;

	/*
	* For linear Elasticity, we can also express the elastic energy as 0.5 * q^T K q + b^T q, where K and b are the constant matrix (vector)
	*/
	Eigen::SparseMatrix<double> K_;	// stiffness matrix
	Eigen::VectorXd b_;
	Eigen::SparseMatrix<double> massMat_;
	Eigen::SparseMatrix<double> massInvMat_;

	LinearElements model_;
	double curTime_;
};
