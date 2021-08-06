#include "GooHook1d.h"
#include<Eigen/SparseLU>
#include<Eigen/IterativeLinearSolvers>
#include <iomanip>
#include <iostream>
#include <fstream>

using namespace Eigen;

//////////////////////////////////////////////////////////////////////////////////////
///                         Add objects
//////////////////////////////////////////////////////////////////////////////////////

void GooHook1d::addParticle(double x, double y)
{
	Vector2d newpos(x,y);
	double mass = params_.particleMass;
	if(params_.particleFixed)
		mass = std::numeric_limits<double>::infinity();

	particles_.push_back(Particle(newpos, mass, params_.particleFixed, false));

	int nParticles = particles_.size();
	for(int i = 0; i < nParticles; i++)
	{
		double dist = particles_[i].pos(1) - params_.ceil;
		dist = dist > 0 ? 1.5 * dist : dist / 1.5;
		auto spring = new Spring1d(i, 0, params_.springStiffness, dist, true);
		connectors_.push_back(spring);
	}
}

//////////////////////////////////////////////////////////////////////////////////////
///             Generate/Degenerate Configuration
//////////////////////////////////////////////////////////////////////////////////////

void GooHook1d::generateConfiguration(Eigen::VectorXd &pos, Eigen::VectorXd &vel, Eigen::VectorXd &prevPos)
{
	int nParticles =  particles_.size();
	pos.resize(nParticles);
	vel.resize(nParticles);
	prevPos.resize(nParticles);
	
	for(int i = 0; i < nParticles; i++)
	{
		prevPos(i) = particles_[i].prevpos(1);
		pos(i) = particles_[i].pos(1);
		vel(i) = particles_[i].vel(1);
	}
}

void GooHook1d::degenerateConfiguration(Eigen::VectorXd pos, Eigen::VectorXd vel, Eigen::VectorXd prevPos)
{
	assert(pos.size() == particles_.size());
	int nParticles = particles_.size();
	for(int i = 0; i < nParticles; i++)
	{
		// Fixed point will have no configuration change
		if (particles_[i].fixed)
			continue;

		// TODO: check if we need to update previous position based on value of the configuration
		particles_[i].prevpos = particles_[i].pos;
		particles_[i].pos(1) = pos(i);
		particles_[i].vel(1) = vel(i);
	}
}


//////////////////////////////////////////////////////////////////////////////////////
///         Compute potential energy, gradient and hessian
//////////////////////////////////////////////////////////////////////////////////////

// Potential
double GooHook1d::computeEnergy(Eigen::VectorXd q)
{
	double energy = 0.0;
	if (params_.gravityEnabled)
		energy += computeGravityPotential(q);
	if (params_.springsEnabled)
		energy += computeSpringPotential(q);
	//TODO: implement other forces.
	return energy;
}

double GooHook1d::computeGravityPotential(Eigen::VectorXd q)
{
	int nParticles = particles_.size();
	double gpotential = 0.0;
	
	for (size_t i = 0; i < nParticles; i++)
	{
		if (particles_[i].fixed)
			continue;
		gpotential -= params_.gravityG * particles_[i].mass * q(i);
	}
	
	return gpotential;
}

double GooHook1d::computeSpringPotential(Eigen::VectorXd q)
{
	double spotential = 0.0;
	
	for (std::vector<Connector1d *>::iterator it = connectors_.begin(); it != connectors_.end(); ++it)
	{
		double restDis = static_cast<Spring1d *>(*it)->restDis;
		int p = (*it)->p;
		
		double stiffness = params_.springStiffness / std::abs(restDis);
		double presentDis = q(p) - params_.ceil;
		
		spotential = spotential + 0.5*stiffness*(presentDis - restDis)*(presentDis - restDis);
	}
	
	return spotential;
}

// gradient
void GooHook1d::computeGradient(Eigen::VectorXd q, Eigen::VectorXd& grad)
{
	grad = Eigen::VectorXd::Zero(q.size());
	if (params_.gravityEnabled)
	{
		Eigen::VectorXd gravityGrad;
		computeGravityGradient(q, gravityGrad);
		grad += gravityGrad;
	}
		
	if (params_.springsEnabled)
	{
		Eigen::VectorXd springGrad;
		computeSpringGradient(q, springGrad);
		grad += springGrad;
	}
}

void GooHook1d::computeGravityGradient(Eigen::VectorXd q, Eigen::VectorXd& grad)
{

	int nParticles = particles_.size();

	grad = Eigen::VectorXd::Zero(q.size());

	for (size_t i = 0; i < nParticles; i++)
	{
		if (particles_[i].fixed)
			continue;
		grad(i) = -params_.gravityG * particles_[i].mass;
	}
}


void GooHook1d::computeSpringGradient(Eigen::VectorXd q, Eigen::VectorXd& grad)
{
	grad = Eigen::VectorXd::Zero(q.size());
	assert(q.size() == particles_.size());

	for (std::vector<Connector1d*>::iterator it = connectors_.begin(); it != connectors_.end(); ++it)
	{
		double restDis = static_cast<Spring1d*>(*it)->restDis;
		int p = (*it)->p;

		double stiffness = params_.springStiffness / std::abs(restDis);
		double presentDis = q(p) - params_.ceil;

		grad(p) += stiffness * (presentDis - restDis);
	}
}


// hessian
void GooHook1d::computeHessian(Eigen::VectorXd q, Eigen::SparseMatrix<double>& hessian)
{
	//Hessian of gravity is zero
	std::vector<Eigen::Triplet<double>> hessianT;
	if (params_.springsEnabled)
		computeSpringHessian(q, hessianT);

	hessian.resize(q.size(), q.size());
	hessian.setFromTriplets(hessianT.begin(), hessianT.end());
}

void GooHook1d::computeSpringHessian(Eigen::VectorXd q, std::vector<Eigen::Triplet<double> >& hessian)
{
	assert(q.size() == particles_.size());

	for (std::vector<Connector1d *>::iterator it = connectors_.begin(); it != connectors_.end(); ++it)
	{
		double restDis = static_cast<Spring1d *>(*it)->restDis;
		int p = (*it)->p;

		double stiffness = params_.springStiffness / std::abs(restDis);
		double presentDis = q(p) - params_.ceil;

		hessian.push_back(Eigen::Triplet<double>(p, p, stiffness));
	}
}

void GooHook1d::assembleMassVec()
{
	int nParticles = particles_.size();

	massVec_.setZero(nParticles);

	std::vector<Eigen::Triplet<double>> coef;

	for (size_t i = 0; i < nParticles; i++)
	{
		massVec_(i) = particles_[i].mass;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
///                             Time integration
//////////////////////////////////////////////////////////////////////////////////////////////////////////
void GooHook1d::updatebyExplicitEuler(Eigen::VectorXd & q, Eigen::VectorXd & qDot, Eigen::VectorXd & qPrev)
{
	Eigen::VectorXd oddq= q;
	Eigen::VectorXd force;
	computeGradient(oddq, force);
	force *= -1;

	q += params_.timeStep * qDot;
	qDot += params_.timeStep  * force;
	// qPrev = q;
}

//void GooHook1d::updatebyVelocityVerlet(Eigen::VectorXd & q, Eigen::VectorXd & qDot, Eigen::VectorXd & qPrev)
//{
//	Eigen::VectorXd  oddq = q;
//	q += params_.timeStep * qDot;
//	qDot += params_.timeStep * massMatrixInv_ * computeTotalForce(q, oddq, NULL);
//}
//
//void GooHook1d::updatebyRK4(Eigen::VectorXd &q, Eigen::VectorXd &qDot, Eigen::VectorXd &qPrev)
//{
//	Eigen::VectorXd a1, a2, b1, b2, c1, c2, d1, d2;
//	a1 = qDot;
//	a2 = massMatrixInv_ * computeTotalForce(q, qPrev, &qDot);
//	b1 = qDot + params_.timeStep / 2 * a2;
//	b2 = massMatrixInv_ * computeTotalForce(q + params_.timeStep / 2 * a1, qPrev, &b1);
//	c1 = qDot + params_.timeStep / 2 * b2;
//	c2 = massMatrixInv_ * computeTotalForce(q + params_.timeStep / 2 * b1, qPrev, &c1);
//	d1 = qDot + params_.timeStep * c2;
//	d2 = massMatrixInv_ * computeTotalForce(q + params_.timeStep * c1, qPrev, &d1);
//	
//	q = q + params_.timeStep / 6 * (a1 + 2 * b1 + 2 * c1 + d1);
//	qDot = qDot + params_.timeStep / 6 * (a2 + 2 * b2 + 2 * c2 + d2);
//}
//
//void GooHook1d::updatebyImplicitEuler(Eigen::VectorXd & q, Eigen::VectorXd & qDot, Eigen::VectorXd &qPrev)
//{
//	Eigen::VectorXd dq(q.size());
//	double h = params_.timeStep;
//
//	// Using the result from Explicit Euler as an initial guess
//	qPrev = q;
//	q = q + h * qDot;
//	
//	Eigen::SparseMatrix<double> idMat(q.size(),q.size());
//	idMat.setIdentity();
//
//	for (int iterations = 0; iterations < params_.NewtonMaxIters; iterations++)
//	{
//		Eigen::VectorXd fval = (q - qPrev - h * qDot - h * h * massMatrixInv_ * computeTotalForce(q, qPrev, NULL));
//		
//		if (fval.norm() < params_.NewtonTolerance)
//			break;
//		
//		Eigen::SparseMatrix<double> gradF(q.size(), q.size());
//		std::vector<Eigen::Triplet<double> > triplet;
//		computeTotalHessian(q, qPrev, &triplet);
//		gradF.setFromTriplets(triplet.begin(), triplet.end());
//
//		// No need for predecomposing the matrix since it is always changing
////        Eigen::SparseLU<SparseMatrix<double>> solver;
//		Eigen::BiCGSTAB<SparseMatrix<double>, Eigen::IncompleteLUT<double>> solver; // Faster than LU for large matrix
//		solver.compute( idMat - h * h * massMatrixInv_ * gradF); // Multiply by mass matrix to make it symmetric
//		
//		dq = solver.solve(-fval);
//		q += dq;
//	}
//
//	// update velocity
//	qDot += h * massMatrixInv_ * computeTotalForce(q, qPrev, NULL);
//}
//
//void GooHook1d::updatebyImplicitMidpoint(Eigen::VectorXd & q, Eigen::VectorXd & qDot, Eigen::VectorXd &qPrev)
//{
//	Eigen::VectorXd currentQ = q;
//	Eigen::SparseMatrix<double> idMat(q.size(),q.size());
//	idMat.setIdentity();
//	
//	for(int i = 0; i < params_.NewtonMaxIters; i++)
//	{
//		Eigen::VectorXd force = computeTotalForce(0.5*(currentQ + q), 0.5*(q + qPrev), NULL);
//		
//		std::vector<Eigen::Triplet<double>> gradFTriplet;
//		computeTotalHessian(0.5*(currentQ + q), 0.5*(q + qPrev), &gradFTriplet);
//		
//		Eigen::SparseMatrix<double> gradF(q.size(),q.size());
//		gradF.setFromTriplets(gradFTriplet.begin(), gradFTriplet.end());
//		
//		Eigen::VectorXd fVal = currentQ - q - params_.timeStep*qDot - 0.5*params_.timeStep*params_.timeStep*massMatrixInv_*force;
//		
//		if(fVal.norm() < params_.NewtonTolerance)
//			break;
//		
//		Eigen::SparseMatrix<double> gradFVal = idMat - params_.timeStep * params_.timeStep * massMatrixInv_ * gradF / 4.0;
//		
//		//        Eigen::SparseLU<SparseMatrix<double>> solver;
//		Eigen::BiCGSTAB<SparseMatrix<double>, Eigen::IncompleteLUT<double>> solver; // Faster than LU for large matrix
//		solver.compute(gradFVal);
//		
//		Eigen::VectorXd deltaQ = solver.solve(-fVal);
//		
//		currentQ = currentQ + deltaQ;
//	}
//	
//	qDot = 2.0 * (currentQ - q)/params_.timeStep - qDot;
//	q = currentQ;
//}

//////////////////////////////////////////////////////////////////////////////////////
///  Removing broken springs
//////////////////////////////////////////////////////////////////////////////////////

void GooHook1d::removeSnappedSprings()
{
	std::vector<int> remainingSpring;
	int nConnectors = connectors_.size();
	
	for(int i = 0; i < nConnectors; i++)
	{
		Spring1d & curSpring = *(static_cast<Spring1d *> (connectors_[i]));
		if(curSpring.canSnap)
		{
			double curLen = std::abs( particles_[curSpring.p].pos(1) - params_.ceil );
			double strainTerm = (curLen - std::abs(curSpring.restDis))/ std::abs(curSpring.restDis);
			if(strainTerm <= params_.maxSpringStrain)
			{
				remainingSpring.push_back(i);
			}
		}
	}
	
	int nRemainingSpring = remainingSpring.size();
	if(nRemainingSpring == nConnectors)
		return;
	std::vector<Connector1d* > remainingConnectors;
	for(int i = 0; i<nRemainingSpring; i++)
	{
		remainingConnectors.push_back(connectors_[remainingSpring[i]]);
	}
	std::swap(connectors_, remainingConnectors);
	
}

//////////////////////////////////////////////////////////////////////////////////////
///                   Test Part
//////////////////////////////////////////////////////////////////////////////////////
void GooHook1d::testPotentialDifferential()
{
	Eigen::VectorXd q, qPrev, vel;
	generateConfiguration(q, vel, qPrev);
	
	Eigen::VectorXd direction = Eigen::VectorXd::Random(q.size());
	direction.normalize();
	
	double V = computeEnergy(q);
	Eigen::VectorXd g;
	computeGradient(q, g);
	
	for (int k = 4; k <= 12; k++)
	{
		double eps = pow(10, -k);
		double  epsV = computeEnergy(q + eps*direction);
		std::cout << "Epsilon = " << eps << std::endl;
		std::cout << "Finite difference: " << (epsV - V)/eps << std::endl;
		std::cout << "-Force dot directional vector: " << g.dot(direction) << std::endl;
		std::cout << "The difference between above two is: " << abs((epsV - V)/eps + g.dot(direction))<<std::endl<<std::endl;
	}
}

void GooHook1d::testForceDifferential()
{
	Eigen::VectorXd q, qPrev, vel;
	generateConfiguration(q, vel, qPrev);
	
	
	Eigen::VectorXd direction = Eigen::VectorXd::Random(q.size());
	direction.normalize();
	Eigen::VectorXd g;
	computeGradient(q, g);

	Eigen::SparseMatrix<double> H(q.size(), q.size());
	
	
	computeHessian(q, H);
	
	for(int k = 1; k<=12; k++)
	{
		double eps = pow(10, -k);
		
		VectorXd epsF;
		computeGradient(q + eps * direction, epsF);
		
		std::cout<<"EPS is: "<<eps<<std::endl;
		std::cout<<"Norm of Finite Difference is: "<< (epsF - g).norm() / eps <<std::endl;
		std::cout<<"Norm of Directinal Gradient is: "<< (H * direction).norm()<< std::endl;
		std::cout<<"The difference between above two is: "<< ((epsF - g)/eps - H * direction ).norm()<<std::endl<<std::endl;
		
	}
}

void GooHook1d::saveConfiguration(std::string filePath)
{
	int nParticles = particles_.size();
	int nConnectors = connectors_.size();
	std::ofstream outfile(filePath, std::ios::trunc);
	
	outfile<<nParticles<<"\n";
	outfile<<nConnectors<<"\n";
	
	for(int i=0;i<nParticles;i++)
	{
		outfile<<std::setprecision(16)<<particles_[i].pos(0)<<"\n";
		outfile<<std::setprecision(16)<<particles_[i].pos(1)<<"\n";
		outfile<<std::setprecision(16)<<particles_[i].prevpos(0)<<"\n";
		outfile<<std::setprecision(16)<<particles_[i].prevpos(1)<<"\n";
		outfile<<std::setprecision(16)<<particles_[i].vel(0)<<"\n";
		outfile<<std::setprecision(16)<<particles_[i].vel(1)<<"\n";
		
		outfile<<particles_[i].fixed<<"\n";
		if(!particles_[i].fixed)
			outfile<<std::setprecision(16)<<particles_[i].mass<<"\n";
		
	}
	
	for(int i=0;i<nConnectors;i++)
	{
		outfile<<connectors_[i]->p<<"\n";
		outfile<<connectors_[i]->mass<<"\n";
	}
	outfile.close();
}

void GooHook1d::loadConfiguration(std::string filePath)
{
	std::ifstream infile(filePath);
	if(!infile)
		return;
	int nParticles;
	int nConnectors;
	
	infile >> nParticles;
	infile >> nConnectors;
	
	particles_.clear();
	connectors_.clear();
	
	for(int i = 0; i < nParticles; i++)
	{
		Eigen::Vector2d pos,prevpos, vel;
		infile >> pos(0);
		infile >> pos(1);
		infile >> prevpos(0);
		infile >> prevpos(1);
		infile >> vel(0);
		infile >> vel(1);
		double mass;
		bool isFixed;
		
		infile >> isFixed;
		if(isFixed)
			mass = std::numeric_limits<double>::infinity();
		else
			infile >> mass;
		Particle newParticle = Particle(pos, mass, isFixed, false);
		newParticle.prevpos = prevpos;
		particles_.push_back(newParticle);
	}
	
	for(int i = 0; i < nConnectors; i++)
	{
		int p;
		double mass;
		infile >> p;
		infile >> mass;
		double dist = particles_[p].pos(1) - params_.ceil;
		auto spring = new Spring1d(p, mass, params_.springStiffness, dist, true);
		connectors_.push_back(spring);
	}
}
