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
		for (std::vector<Connector*>::iterator it = fullConnectors_.begin(); it != fullConnectors_.end(); ++it)
			delete* it;
		connectors_.clear();
		params_ = simParams;
	}

	void addParticle(double x, double y);
	void removeSnappedSprings();

	void generateConfiguration(Eigen::VectorXd& pos, Eigen::VectorXd& vel, Eigen::VectorXd& prevPos, Eigen::VectorXd& preVel);
	void degenerateConfiguration(Eigen::VectorXd pos, Eigen::VectorXd vel, Eigen::VectorXd prevPos, Eigen::VectorXd preVel);


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

	// particle-floor barrier function
	double computeParticleFloorBarrier(Eigen::VectorXd q);
	void computeParticleFloorGradeint(Eigen::VectorXd q, Eigen::VectorXd& grad);
	void computeParticleFloorHessian(Eigen::VectorXd q, std::vector<Eigen::Triplet<double>>& hessian);

	double getMaxStepSize(Eigen::VectorXd q, Eigen::VectorXd dir);


	// mass vector
	Eigen::VectorXd massVec_;
	void assembleMassVec();
	void preTimeStep(Eigen::VectorXd q);	// update the stiffness
	void postIteration(Eigen::VectorXd q);	// update the stiffness

public:
	std::vector<Particle, Eigen::aligned_allocator<Particle> > particles_;
	std::vector<Connector1d*> connectors_;
	std::vector<Connector* > fullConnectors_;
	SimParameters params_;

public:
	// Helper function that test if we compute correct differential
	void testPotentialDifferential();
	void testGradientDifferential();

	void saveConfiguration(std::string filePath);
	void loadConfiguration(std::string filePath);

private:
	void updateCloseParticles(Eigen::VectorXd q, double d_eps);
	std::vector<std::pair<int, double>> closeParticles_;
};
