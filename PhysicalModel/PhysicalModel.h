#pragma once
#include "SimParameters.h"
#include <map>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

class PhysicalModel
{
public:
	PhysicalModel() {}
	PhysicalModel(SimParameters simParams)
	{
		params_ = simParams;
		mu_ = params_.youngs / (2 * (1 + params_.poisson));
		lambda_ = params_.youngs * params_.poisson / ((1 + params_.poisson) * (1 - 2 * params_.poisson));
	}
	virtual ~PhysicalModel() = default;

	void initialize(Eigen::VectorXd restPos, Eigen::MatrixXi restF, Eigen::VectorXd massVec, std::map<int, double>* clampedPoints);
	void updateProjM(std::map<int, double> clampedPoints);

	// Implement potential computation
	double computeEnergy(Eigen::VectorXd q);
	void computeGradient(Eigen::VectorXd q, Eigen::VectorXd& grad);
	void computeHessian(Eigen::VectorXd q, Eigen::SparseMatrix<double>& hessian);

	// Gravity does not have Hessian (zero matrix)
	double computeGravityPotential(Eigen::VectorXd q);
	void computeGravityGradient(Eigen::VectorXd q, Eigen::VectorXd& grad);

	// Elastic potential
	double computeElasticPotential(Eigen::VectorXd q);
	void computeElasticGradient(Eigen::VectorXd q, Eigen::VectorXd& grad);
	void computeElasticHessian(Eigen::VectorXd q, std::vector<Eigen::Triplet<double> >& hessian);

	virtual double computeElasticPotentialPerface(Eigen::VectorXd q, int faceId) = 0 ;
	virtual void computeElasticGradientPerface(Eigen::VectorXd q, int faceId, Eigen::Vector2d& grad) = 0;
	virtual void computeElasticHessianPerface(Eigen::VectorXd q, int faceId, Eigen::Matrix2d& hess) = 0;

	// floor barrier function
	double computeFloorBarrier(Eigen::VectorXd q);
	void computeFloorGradeint(Eigen::VectorXd q, Eigen::VectorXd& grad);
	void computeFloorHessian(Eigen::VectorXd q, std::vector<Eigen::Triplet<double>>& hessian);

	// internal contact barrier
	double computeInternalBarrier(Eigen::VectorXd q);
	void computeInternalGradient(Eigen::VectorXd q, Eigen::VectorXd& grad);
	void computeInternalHessian(Eigen::VectorXd q, std::vector<Eigen::Triplet<double>>& hessian);


	double getMaxStepSize(Eigen::VectorXd q, Eigen::VectorXd dir);

	void assembleMass(Eigen::VectorXd massVec);

	// right now, do nothing
	void preTimeStep(Eigen::VectorXd q) {}
	void postIteration(Eigen::VectorXd q) {}

	void PhysicalModel::testPotentialDifferential(Eigen::VectorXd q);
	void PhysicalModel::testGradientDifferential(Eigen::VectorXd q);

public:
	SimParameters params_;
	Eigen::VectorXd restPos_;
	Eigen::MatrixXi restF_;
	Eigen::SparseMatrix<double> projM_;
	Eigen::SparseMatrix<double> unProjM_;
	std::vector<int> indexMap_;
	std::vector<int> indexInvMap_;

	Eigen::VectorXd massVec_;
	double mu_;
	double lambda_;
};