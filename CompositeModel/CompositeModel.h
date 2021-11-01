#pragma once
#include <map>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "FiniteElement.h"

class CompositeModel
{
public:
	CompositeModel() {}
	virtual ~CompositeModel() = default;

	void initialize(Eigen::VectorXd resPos, std::vector<FiniteElement> elements, Eigen::VectorXd massVec, std::map<int, double>* clampedPoints);
	void updateProjM(std::map<int, double>* clampedPoints);
	
	// Implement potential computation
	double computeEnergy(Eigen::VectorXd q);
	void computeGradient(Eigen::VectorXd q, Eigen::VectorXd& grad);
	void computeHessian(Eigen::VectorXd q, Eigen::SparseMatrix<double>& hessian);


	double computeEnergyPart1(Eigen::VectorXd q);
	void computeGradientPart1(Eigen::VectorXd q, Eigen::VectorXd& grad);
	void computeHessianPart1(Eigen::VectorXd q, Eigen::SparseMatrix<double>& hessian);

	double computeEnergyPart2(Eigen::VectorXd q);
	void computeGradientPart2(Eigen::VectorXd q, Eigen::VectorXd& grad);
	void computeHessianPart2(Eigen::VectorXd q, Eigen::SparseMatrix<double>& hessian);

	// convert variable to current pos state
	void convertVar2Pos(Eigen::VectorXd q, Eigen::VectorXd &pos);
	void convertPos2Var(Eigen::VectorXd pos, Eigen::VectorXd &q);

	// Gravity does not have Hessian (zero matrix)
	double computeGravityPotential(Eigen::VectorXd q);
	void computeGravityGradient(Eigen::VectorXd q, Eigen::VectorXd& grad);

	// Elastic potential
	double computeElasticPotential(Eigen::VectorXd pos);
	void computeElasticGradient(Eigen::VectorXd pos, Eigen::VectorXd& grad);
	void computeElasticHessian(Eigen::VectorXd pos, std::vector<Eigen::Triplet<double> >& hessian);

	// compression
	void computeCompression(Eigen::VectorXd pos, Eigen::VectorXd& compression);


	// floor barrier function
	double computeFloorBarrier(Eigen::VectorXd pos);
	void computeFloorGradeint(Eigen::VectorXd pos, Eigen::VectorXd& grad);
	void computeFloorHessian(Eigen::VectorXd pos, std::vector<Eigen::Triplet<double>>& hessian);

	// internal contact barrier
	double computeInternalBarrier(Eigen::VectorXd pos);
	void computeInternalGradient(Eigen::VectorXd pos, Eigen::VectorXd& grad);
	void computeInternalHessian(Eigen::VectorXd pos, std::vector<Eigen::Triplet<double>>& hessian);


	double getMaxStepSize(Eigen::VectorXd q, Eigen::VectorXd dir);

	Eigen::VectorXd assembleMass(Eigen::VectorXd massVec);

	// right now, do nothing
	void preTimeStep(Eigen::VectorXd q) {}
	void postIteration(Eigen::VectorXd q) {}

	void testPotentialDifferential(Eigen::VectorXd q);
	void testGradientDifferential(Eigen::VectorXd q);

	// void testFloorBarrierEnergy(Eigen::VectorXd q);
	// void testFloorBarrierGradient(Eigen::VectorXd q);

	// void testInternalBarrierEnergy(Eigen::VectorXd q);
    // void testInternalBarrierGradient(Eigen::VectorXd q);

	// void testPotentialDifferentialPerface(Eigen::VectorXd q, int faceId);
	// void testGradientDifferentialPerface(Eigen::VectorXd q, int faceId);

	// void testElasticEnergy(Eigen::VectorXd q);
	// void testElasticGradient(Eigen::VectorXd q);
	

public:
	Eigen::VectorXd restPos_;
	std::vector<FiniteElement> elements_;
	Eigen::SparseMatrix<double> projM_;
	Eigen::SparseMatrix<double> unProjM_;
	std::vector<bool> fixedPos_;
	Eigen::VectorXd massVec_;
};