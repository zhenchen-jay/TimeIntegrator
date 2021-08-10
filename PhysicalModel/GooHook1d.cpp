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

	int newid = particles_.size();
	particles_.push_back(Particle(newpos, mass, params_.particleFixed, false));

	int nParticles = particles_.size();
	if (params_.modelType == SimParameters::MT_HARMONIC_1D)
	{
		double dist = std::abs(particles_[newid].pos(1) - params_.ceil);
		dist *= 0.5;
		auto spring = new Spring1d(newid, 0, params_.springStiffness, dist, true);
		connectors_.push_back(spring);
	}
	else 
	{
		for (int i = 0; i < nParticles - 1; i++)
		{
			double dist = (newpos - particles_[i].pos).norm();
			if (std::abs(particles_[i].pos(0) - particles_[newid].pos(0)) < 1e-6)
			{
				//dist *= 0.5;
				auto spring = new Spring(newid, i, 0, params_.springStiffness, dist, true);
				fullConnectors_.push_back(spring);
			}
		}
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
	if (params_.floorEnabled)
	{
		double floorE = computeParticleFloorBarrier(q);;
		energy += floorE * params_.barrierStiffness;
	}
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
	
	if (params_.modelType == SimParameters::MT_HARMONIC_1D)
	{

		for (std::vector<Connector1d*>::iterator it = connectors_.begin(); it != connectors_.end(); ++it)
		{
			double restDis = static_cast<Spring1d*>(*it)->restDis;
			int p = (*it)->p;

			double stiffness = params_.springStiffness / std::abs(restDis);
			double presentDis = std::abs(q(p) - params_.ceil);

			spotential = spotential + 0.5 * stiffness * (presentDis - restDis) * (presentDis - restDis);
		}
	}
	else
	{
		for (std::vector<Connector*>::iterator it = fullConnectors_.begin(); it != fullConnectors_.end(); ++it)
		{
			double restlen = static_cast<Spring*>(*it)->restlen;
			int a = (*it)->p1;
			int b = (*it)->p2;

			double stiffness = params_.springStiffness / restlen;
			double presentlen = std::abs(q(b) - q(a));

			spotential = spotential + 0.5 * stiffness * (presentlen - restlen) * (presentlen - restlen);
		}
	}

	return spotential;
}

double GooHook1d::computeParticleFloorBarrier(Eigen::VectorXd q)
{
	double barrier = 0.0;
	int nParticles = particles_.size();
	for (int i = 0; i < nParticles; i++)
	{
		double radius = 0.02 * std::sqrt(particles_[i].mass);
		if (q(i) <= -0.5 + radius + params_.barrierEps || q(i) >= 0.5 - radius - params_.barrierEps)
		{
			double pos = q(i);
			double dist = params_.barrierEps;
			if (q(i) <= -0.5 + radius + params_.barrierEps)
				dist = q(i) + 0.5 - radius;
			else
				dist = 0.5 - radius - q(i);
			barrier += -(dist - params_.barrierEps) * (dist - params_.barrierEps) * std::log(dist / params_.barrierEps);
		}
	}
	return barrier;
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

	if (params_.floorEnabled)
	{
		Eigen::VectorXd floorGrad;
		computeParticleFloorGradeint(q, floorGrad);
		grad += params_.barrierStiffness * floorGrad;
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

	if (params_.modelType == SimParameters::MT_HARMONIC_1D)
	{
		for (std::vector<Connector1d*>::iterator it = connectors_.begin(); it != connectors_.end(); ++it)
		{
			double restDis = static_cast<Spring1d*>(*it)->restDis;
			int p = (*it)->p;

			double stiffness = params_.springStiffness / std::abs(restDis);
			double displacement = q(p) - params_.ceil;
			double presentlen = std::abs(displacement);

			grad(p) += stiffness * (presentlen - restDis) * displacement / presentlen;
		}
	}
	else
	{
		for (std::vector<Connector*>::iterator it = fullConnectors_.begin(); it != fullConnectors_.end(); ++it)
		{
			double restlen = static_cast<Spring*>(*it)->restlen;
			int a = (*it)->p1;
			int b = (*it)->p2;

			double stiffness = params_.springStiffness / restlen;
			double displacement = q(b) - q(a);
			double presentlen = std::abs(displacement);

			grad(b) += stiffness * (presentlen - restlen) / presentlen * displacement;
			grad(a) -= stiffness * (presentlen - restlen) / presentlen * displacement;
		}
	}
}


void GooHook1d::computeParticleFloorGradeint(Eigen::VectorXd q, Eigen::VectorXd& grad)
{
	int nParticles = particles_.size();
	grad.setZero(nParticles);
	for (int i = 0; i < nParticles; i++)
	{
		double radius = 0.02 * std::sqrt(particles_[i].mass);
		if (q(i) <= -0.5 + radius + params_.barrierEps || q(i) >= 0.5 - radius - params_.barrierEps)
		{
			double dist = params_.barrierEps;
			if (q(i) <= -0.5 + radius + params_.barrierEps)
				dist = q(i) + 0.5 - radius;
			else
				dist = 0.5 - radius - q(i);
			grad(i) += -(dist - params_.barrierEps) * (2 * std::log(dist / params_.barrierEps) - params_.barrierEps / dist + 1);
			if (q(i) >= 0.5 - radius - params_.barrierEps)
				grad(i) *= -1;
		}
	}
}

// hessian
void GooHook1d::computeHessian(Eigen::VectorXd q, Eigen::SparseMatrix<double>& hessian)
{
	//Hessian of gravity is zero
	std::vector<Eigen::Triplet<double>> hessianT;
	if (params_.springsEnabled)
		computeSpringHessian(q, hessianT);
	if (params_.floorEnabled)
	{
		std::vector<Eigen::Triplet<double>> floorT;
		computeParticleFloorHessian(q, floorT);
		for (int i = 0; i < floorT.size(); i++)
		{
			hessianT.push_back(Eigen::Triplet<double>(floorT[i].row(), floorT[i].col(), params_.barrierStiffness * floorT[i].value()));
		}
	}
		

	hessian.resize(q.size(), q.size());
	hessian.setFromTriplets(hessianT.begin(), hessianT.end());
}

void GooHook1d::computeSpringHessian(Eigen::VectorXd q, std::vector<Eigen::Triplet<double> >& hessian)
{
	assert(q.size() == particles_.size());

	if (params_.modelType == SimParameters::MT_HARMONIC_1D)
	{
		for (std::vector<Connector1d*>::iterator it = connectors_.begin(); it != connectors_.end(); ++it)
		{
			double restDis = static_cast<Spring1d*>(*it)->restDis;
			int p = (*it)->p;

			double stiffness = params_.springStiffness / std::abs(restDis);

			hessian.push_back(Eigen::Triplet<double>(p, p, stiffness));
		}
	}
	else
	{
		for (std::vector<Connector*>::iterator it = fullConnectors_.begin(); it != fullConnectors_.end(); ++it)
		{
			double restlen = static_cast<Spring*>(*it)->restlen;
			int a = (*it)->p1;
			int b = (*it)->p2;
			assert(a != b);

			double stiffness = params_.springStiffness / restlen;

			hessian.push_back(Eigen::Triplet<double>(a, a, stiffness));
			hessian.push_back(Eigen::Triplet<double>(a, b, -stiffness));

			hessian.push_back(Eigen::Triplet<double>(b, a, -stiffness));
			hessian.push_back(Eigen::Triplet<double>(b, b, stiffness));

		}
	}
}

void GooHook1d::computeParticleFloorHessian(Eigen::VectorXd q, std::vector<Eigen::Triplet<double>>& hessian)
{
	int nParticles = particles_.size();

	for (int i = 0; i < nParticles; i++)
	{
		double radius = 0.02 * std::sqrt(particles_[i].mass);
		if (q(i) <= -0.5 + radius + params_.barrierEps || q(i) >= 0.5 - radius - params_.barrierEps)
		{
			double dist = params_.barrierEps;
			if (q(i) <= -0.5 + radius + params_.barrierEps)
				dist = q(i) + 0.5 - radius;
			else
				dist = 0.5 - radius - q(i);
			double value = -2 * std::log(dist / params_.barrierEps) + (params_.barrierEps - dist) * (params_.barrierEps + 3 * dist) / (dist * dist);
			hessian.push_back(Eigen::Triplet<double>(i, i, value));
		}
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
///                   find maximum step size
//////////////////////////////////////////////////////////////////////////////////////
double GooHook1d::getMaxStepSize(Eigen::VectorXd q, Eigen::VectorXd dir)
{
	if (!params_.floorEnabled)
		return 1.0;
	else
	{
		int nParticles = q.size();
		double maxStep = 1.0;
		for (int i = 0; i < nParticles; i++)
		{
			double radius = 0.02 * std::sqrt(particles_[i].mass);
			double upperStep = (0.5 - radius - q(i)) / dir(i) > 0 ? (0.5 - radius - q(i)) / dir(i) : 1.0;
			double lowerStep = (-0.5 + radius - q(i)) / dir(i) > 0 ? (-0.5 + radius - q(i)) / dir(i) : 1.0;
			double qiStep = 0.8 * std::min(upperStep, lowerStep);
			/*std::wcout << "particle: " << i << ", pos: " << q(i) << ", dir: " << dir(i) << ", lower step: " << lowerStep << ", upper step: " << upperStep << std::endl;*/
			maxStep = std::min(maxStep, qiStep);
		}
		return maxStep;
	}
}

void GooHook1d::updateCloseParticles(Eigen::VectorXd q, double d_eps)
{
	closeParticles_.clear();

	for (int i = 0; i < particles_.size(); i++)
	{
		double l = 0.02 * std::sqrt(params_.particleMass);
		if (q(i) <= -0.5 + l + d_eps || q(i) >= 0.5 - l - d_eps)
		{
			double pos = q(i);
			double dist = d_eps;
			if (q(i) <= -0.5 + l + d_eps)
				dist = q(i) + 0.5 - l;
			else
				dist = 0.5 - l - q(i);
			closeParticles_.push_back(std::pair<int, double>(i, dist));
		}
	}
}

void GooHook1d::preTimeStep(Eigen::VectorXd q)
{
	if (params_.floorEnabled)
	{
		Eigen::VectorXd floorGrad;
		computeParticleFloorGradeint(q, floorGrad);

		Eigen::VectorXd springGrad, gravityGrad, gradE;
		computeSpringGradient(q, springGrad);
		gradE = springGrad;
		double l = 0.02 * std::sqrt(params_.particleMass);


		double kappa_g = floorGrad.norm() > 0 ? -floorGrad.dot(gradE) / floorGrad.squaredNorm() : 0;

		// suggested kappa by IPC paper:
		double d = 1e-8 * l; // 0.02 is the radius of the point in gui
		double Hb = -2 * std::log(d / params_.barrierEps) + (params_.barrierEps - d) * (params_.barrierEps + 3 * d) / (d * d);
		double kappa_min = 1e11 * params_.particleMass / (4e-16 * l * Hb);
		double kappa_max = 100 * kappa_min;

		params_.barrierStiffness = std::min(kappa_max, std::max(kappa_min, kappa_g));

		double d_eps = 1e-9 * l;
		updateCloseParticles(q, d_eps);

	}
}

void GooHook1d::postIteration(Eigen::VectorXd q)
{
	if (params_.floorEnabled)
	{
		double l = 0.02 * std::sqrt(params_.particleMass);
		double d_eps = 1e-9 * l;

		double d = 1e-8 * l; // 0.02 is the radius of the point in gui
		double Hb = -2 * std::log(d / params_.barrierEps) + (params_.barrierEps - d) * (params_.barrierEps + 3 * d) / (d * d);
		double kappa_min = 1e11 * params_.particleMass / (4e-16 * l * Hb);
		double kappa_max = 100 * kappa_min;

		for (int i = 0; i < closeParticles_.size(); i++)
		{
			int pid = closeParticles_[i].first;
			if (q(i) <= -0.5 + l + d_eps || q(i) >= 0.5 - l - d_eps)
			{
				double pos = q(i);
				double dist = d_eps;
				if (q(i) <= -0.5 + l + d_eps)
					dist = q(i) + 0.5 - l;
				else
					dist = 0.5 - l - q(i);
				if (dist < closeParticles_[i].second)
					params_.barrierStiffness = std::min(kappa_max, 2 * params_.barrierStiffness);
			}
		}

		updateCloseParticles(q, d_eps);
		std::cout << "After this iteration, barrier stiffness is: " << params_.barrierStiffness << std::endl;
	}
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
		std::cout << "directional derivative: " << g.dot(direction) << std::endl;
		std::cout << "The difference between above two is: " << abs((epsV - V)/eps - g.dot(direction))<<std::endl<<std::endl;
	}
}

void GooHook1d::testGradientDifferential()
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
