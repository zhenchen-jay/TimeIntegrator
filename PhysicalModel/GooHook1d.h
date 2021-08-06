#include "SceneObjects.h"
#include <deque>
#include "SimParameters.h"
#include <Eigen/Sparse>
#include <Eigen/StdVector>
#include <Eigen/SparseCholesky>

// We fixed the x coordinate 

class GooHook1d
{
public:
	GooHook1d() {}
	GooHook1d(SimParameters simParams)
	{
		particles_.clear();
		for (std::vector<Connector1d*>::iterator it = connectors_.begin(); it != connectors_.end(); ++it)
			delete* it;
		connectors_.clear();
		params_ = simParams;
	}

	void addParticle(double x, double y);
	void removeSnappedSprings();

	void generateConfiguration(Eigen::VectorXd& pos, Eigen::VectorXd& vel, Eigen::VectorXd& prevPos);
	void degenerateConfiguration(Eigen::VectorXd pos, Eigen::VectorXd vel, Eigen::VectorXd prevPos);


	// Implement potential computation
	double computeEnergy(Eigen::VectorXd q);
	void computeGradient(Eigen::VectorXd q, Eigen::VectorXd& grad);
	void computeHessian(Eigen::VectorXd q, Eigen::SparseMatrix<double>& hessian);

	// Gravity does not have Hessian (zero matrix)
	double computeGravityPotential(Eigen::VectorXd q);
	void computeGravityGradient(Eigen::VectorXd q, Eigen::VectorXd& grad);

	// Spring potential
	double computeSpringPotential(Eigen::VectorXd q);
	void computeSpringGradient(Eigen::VectorXd q, Eigen::VectorXd& grad);
	void computeSpringHessian(Eigen::VectorXd q, std::vector<Eigen::Triplet<double> >& hessian);

	double getMaxStepSize(Eigen::VectorXd q)
	{
		return 1.0;
	}


	// mass vector
	Eigen::VectorXd massVec_;
	void assembleMassVec();

	void updatebyExplicitEuler(Eigen::VectorXd& q, Eigen::VectorXd& qDot, Eigen::VectorXd& qPrev);
	void updatebyVelocityVerlet(Eigen::VectorXd& q, Eigen::VectorXd& qDot, Eigen::VectorXd& qPrev);
	void updatebyImplicitEuler(Eigen::VectorXd& q, Eigen::VectorXd& qDot, Eigen::VectorXd& qPrev);
	void updatebyImplicitMidpoint(Eigen::VectorXd& q, Eigen::VectorXd& qDot, Eigen::VectorXd& qPrev);
	void updatebyRK4(Eigen::VectorXd& q, Eigen::VectorXd& qDot, Eigen::VectorXd& qPrev);

public:
	std::vector<Particle, Eigen::aligned_allocator<Particle> > particles_;
	std::vector<Connector1d*> connectors_;
	SimParameters params_;

public:
	// Helper function that test if we compute correct differential
	void testPotentialDifferential();
	void testForceDifferential();

	void saveConfiguration(std::string filePath);
	void loadConfiguration(std::string filePath);



};
