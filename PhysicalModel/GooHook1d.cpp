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

void GooHook1d::addParticle(double x, double y, bool isFixed, double maxEffectDist)
{
	Vector2d newpos(x,y);
	double mass = params_.particleMass;
	if(isFixed)
		mass = std::numeric_limits<double>::infinity();

	int newid = particles_.size();
	particles_.push_back(Particle(newpos, mass, isFixed, false));

	int nParticles = particles_.size();
	/*if (params_.modelType == SimParameters::MT_HARMONIC_1D)
	{
		double dist = std::abs(particles_[newid].pos(1) - params_.ceil);
		dist *= 0.5;
		auto spring = new Spring1d(newid, 0, params_.springStiffness, dist, true);
		connectors_.push_back(spring);
	}
	else 
	{*/
		for (int i = 0; i < nParticles - 1; i++)
		{
			double dist = (newpos - particles_[i].pos).norm();
			if (std::abs(particles_[i].pos(0) - particles_[newid].pos(0)) < 1e-6)
			{
				if (maxEffectDist > 0)
				{
					if (dist < maxEffectDist)
					{
						if (params_.modelType == SimParameters::MT_HARMONIC_1D)
							dist *= 0.8;
						auto spring = new Spring(newid, i, 0, params_.springStiffness, dist, true);
						fullConnectors_.push_back(spring);
					}

				}
				else
				{
					if (params_.modelType == SimParameters::MT_HARMONIC_1D)
						dist *= 0.5;
					auto spring = new Spring(newid, i, 0, params_.springStiffness, dist, true);
					fullConnectors_.push_back(spring);
				}
				
			}
		}
	//}
	
}

//////////////////////////////////////////////////////////////////////////////////////
///                         Projection matrix
//////////////////////////////////////////////////////////////////////////////////////
void GooHook1d::updateProjM()
{
	int nParticles = particles_.size();
	int row = 0;
	
	std::vector<Eigen::Triplet<double>> T;
	indexMap_.resize(nParticles, -1);

	for (int i = 0; i < nParticles; i++)
	{
		if (particles_[i].fixed)
			continue;
		else
		{
			T.push_back(Eigen::Triplet<double>(row, i, 1.0));
			indexMap_[i] = row;
			indexInvMap_.push_back(i);
			row++;
		}
	}

	projM_.resize(row, nParticles);
	projM_.setFromTriplets(T.begin(), T.end());

	unProjM_ = projM_.transpose();
}


//////////////////////////////////////////////////////////////////////////////////////
///             Generate/Degenerate Configuration
//////////////////////////////////////////////////////////////////////////////////////

void GooHook1d::generateConfiguration(Eigen::VectorXd &pos, Eigen::VectorXd &vel, Eigen::VectorXd &prevPos, Eigen::VectorXd& preVel)
{
	int nParticles =  particles_.size();
	int nVals = projM_.rows();
	pos.resize(nVals);
	vel.resize(nVals);
	prevPos.resize(nVals);
	preVel.resize(nVals);
	
	for(int i = 0; i < nParticles; i++)
	{
		if (particles_[i].fixed)
			continue;
		prevPos(indexMap_[i]) = particles_[i].prevpos(1);
		preVel(indexMap_[i]) = particles_[i].preVel(1);

		pos(indexMap_[i]) = particles_[i].pos(1);
		vel(indexMap_[i]) = particles_[i].vel(1);
	}
}

void GooHook1d::degenerateConfiguration(Eigen::VectorXd pos, Eigen::VectorXd vel, Eigen::VectorXd prevPos, Eigen::VectorXd preVel)
{
	int nParticles = particles_.size();
	for(int i = 0; i < nParticles; i++)
	{
		// Fixed point will have no configuration change
		if (particles_[i].fixed)
			continue;

		// TODO: check if we need to update previous position based on value of the configuration
		particles_[i].prevpos = particles_[i].pos;
		particles_[i].preVel = particles_[i].vel;

		particles_[i].pos(1) = pos(indexMap_[i]);
		particles_[i].vel(1) = vel(indexMap_[i]);
	}
}


double GooHook1d::getCurrentConnectorLen(Eigen::VectorXd q, int cid)
{
	double len = 0;
	if (cid < 0 || cid >= fullConnectors_.size())
	{
		std::cerr << "connector index is out of range." << std::endl;
		exit(1);
	}


	Spring* connector = static_cast<Spring*> (fullConnectors_[cid]);
	int a = indexMap_[connector->p1];
	int b = indexMap_[connector->p2];

	double va = (a == -1) ? particles_[connector->p1].pos(1) : q(a);
	double vb = (b == -1) ? particles_[connector->p2].pos(1) : q(b);

	double presentlen = std::abs(vb - va);
	len = presentlen;
	return len;

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

	Eigen::VectorXd unProjq = unProjM_ * q;
	
	for (size_t i = 0; i < nParticles; i++)
	{
		if (particles_[i].fixed)
			continue;
		gpotential -= params_.gravityG * particles_[i].mass * q(indexMap_[i]);
	}
	
	return gpotential;
}

double GooHook1d::computeSpringPotential(Eigen::VectorXd q)
{
	double spotential = 0.0;
	for (std::vector<Connector*>::iterator it = fullConnectors_.begin(); it != fullConnectors_.end(); ++it)
	{
		double restlen = static_cast<Spring*>(*it)->restlen;
		int a = (*it)->p1;
		int b = (*it)->p2;

		double stiffness = params_.springStiffness / restlen;
		double va = indexMap_[a] != -1 ? q(indexMap_[a]) : particles_[a].pos(1);
		double vb = indexMap_[b] != -1 ? q(indexMap_[b]) : particles_[b].pos(1);
		double presentlen = std::abs(vb - va);

		spotential = spotential + 0.5 * stiffness * (presentlen - restlen) * (presentlen - restlen);
	}

	return spotential;
}

double GooHook1d::computeParticleFloorBarrier(Eigen::VectorXd q)
{
	double barrier = 0.0;
	int nParticles = particles_.size();
	for (int i = 0; i < nParticles; i++)
	{
		if (particles_[i].fixed)
			continue;
		double radius = 0.02 * std::sqrt(particles_[i].mass);
		if (q(indexMap_[i]) <= -0.5 + radius + params_.barrierEps || q(indexMap_[i]) >= 0.5 - radius - params_.barrierEps)
		{
			double pos = q(indexMap_[i]);
			double dist = params_.barrierEps;
			if (q(indexMap_[i]) <= -0.5 + radius + params_.barrierEps)
				dist = q(indexMap_[i]) + 0.5 - radius;
			else
				dist = 0.5 - radius - q(indexMap_[i]);
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
		grad(indexMap_[i]) = -params_.gravityG * particles_[i].mass;
	}
}


void GooHook1d::computeSpringGradient(Eigen::VectorXd q, Eigen::VectorXd& grad)
{
	grad = Eigen::VectorXd::Zero(q.size());
	for (std::vector<Connector*>::iterator it = fullConnectors_.begin(); it != fullConnectors_.end(); ++it)
	{
		double restlen = static_cast<Spring*>(*it)->restlen;
		int a = indexMap_[(*it)->p1];
		int b = indexMap_[(*it)->p2];

		double stiffness = params_.springStiffness / restlen;

		double va = (a == -1) ? particles_[(*it)->p1].pos(1) : q(a);
		double vb = (b == -1) ? particles_[(*it)->p2].pos(1) : q(b);
	
		double displacement = vb - va;
		double presentlen = std::abs(displacement);

		double sign = displacement > 0 ? 1.0 : -1.0;

		if(b != -1)
			grad(b) += stiffness * (presentlen - restlen) * sign;
		if(a != -1)
			grad(a) -= stiffness * (presentlen - restlen) * sign;
	}
}


void GooHook1d::computeParticleFloorGradeint(Eigen::VectorXd q, Eigen::VectorXd& grad)
{
	int nParticles = particles_.size();
	grad.setZero(q.size());
	for (int i = 0; i < nParticles; i++)
	{
		if (particles_[i].fixed)
			continue;
		double radius = 0.02 * std::sqrt(particles_[i].mass);
		int id = indexMap_[i];
		if (q(id) <= -0.5 + radius + params_.barrierEps || q(id) >= 0.5 - radius - params_.barrierEps)
		{
			double dist = params_.barrierEps;
			if (q(id) <= -0.5 + radius + params_.barrierEps)
				dist = q(id) + 0.5 - radius;
			else
				dist = 0.5 - radius - q(id);
			grad(id) += -(dist - params_.barrierEps) * (2 * std::log(dist / params_.barrierEps) - params_.barrierEps / dist + 1);
			if (q(id) >= 0.5 - radius - params_.barrierEps)
				grad(id) *= -1;
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
	for (std::vector<Connector*>::iterator it = fullConnectors_.begin(); it != fullConnectors_.end(); ++it)
	{
		double restlen = static_cast<Spring*>(*it)->restlen;
		int a = indexMap_[(*it)->p1];
		int b = indexMap_[(*it)->p2];
		

		double stiffness = params_.springStiffness / restlen;

		if (a != -1)
		{
			hessian.push_back(Eigen::Triplet<double>(a, a, stiffness));
			if (b != -1)
			{
				hessian.push_back(Eigen::Triplet<double>(a, b, -stiffness));
				hessian.push_back(Eigen::Triplet<double>(b, a, -stiffness));
				hessian.push_back(Eigen::Triplet<double>(b, b, stiffness));
			}
		}
		else
			hessian.push_back(Eigen::Triplet<double>(b, b, stiffness));
	}
}

void GooHook1d::computeParticleFloorHessian(Eigen::VectorXd q, std::vector<Eigen::Triplet<double>>& hessian)
{
	int nParticles = particles_.size();

	for (int i = 0; i < nParticles; i++)
	{
		if (particles_[i].fixed)
			continue;
		double radius = 0.02 * std::sqrt(particles_[i].mass);
		int id = indexMap_[i];
		if (q(id) <= -0.5 + radius + params_.barrierEps || q(id) >= 0.5 - radius - params_.barrierEps)
		{
			double dist = params_.barrierEps;
			if (q(id) <= -0.5 + radius + params_.barrierEps)
				dist = q(id) + 0.5 - radius;
			else
				dist = 0.5 - radius - q(id);
			double value = -2 * std::log(dist / params_.barrierEps) + (params_.barrierEps - dist) * (params_.barrierEps + 3 * dist) / (dist * dist);
			hessian.push_back(Eigen::Triplet<double>(id, id, value));
		}
	}
}

void GooHook1d::assembleMassVec()
{
	int nParticles = particles_.size();
	int nVals = projM_.rows();

	massVec_.setZero(nVals);

	std::vector<Eigen::Triplet<double>> coef;

	for (size_t i = 0; i < nParticles; i++)
	{
		if (particles_[i].fixed)
			continue;
		massVec_(indexMap_[i]) = particles_[i].mass;
	}
}

//////////////////////////////////////////////////////////////////////////////////////
///                   find maximum step size
//////////////////////////////////////////////////////////////////////////////////////
double GooHook1d::getMaxStepSize(Eigen::VectorXd q, Eigen::VectorXd dir)
{
	double maxStep = 1.0;
	for (std::vector<Connector*>::iterator it = fullConnectors_.begin(); it != fullConnectors_.end(); ++it)
	{
		double restlen = static_cast<Spring*>(*it)->restlen;
		int a = (*it)->p1;
		int b = (*it)->p2;

		double va = indexMap_[a] != -1 ? q(indexMap_[a]) : particles_[a].pos(1);
		double vb = indexMap_[b] != -1 ? q(indexMap_[b]) : particles_[b].pos(1);

		double deltaa = indexMap_[a] != -1 ? dir(indexMap_[a]) : 0;
		double deltab = indexMap_[b] != -1 ? dir(indexMap_[b]) : 0;
		
		double presentDis = vb - va;
		double deltaDis = deltab - deltaa;

		double step = -presentDis / deltaDis;
		std::cout << step << std::endl;
		if (step > 0)
			maxStep = std::min(maxStep, 0.8 * step);
	}


	if (params_.floorEnabled)
	{
		int nParticles = q.size();
		
		for (int i = 0; i < nParticles; i++)
		{
			if (particles_[i].fixed)
				continue;
			double radius = 0.02 * std::sqrt(particles_[i].mass);
			double upperStep = (0.5 - radius - q(indexMap_[i])) / dir(indexMap_[i]) > 0 ? (0.5 - radius - q(indexMap_[i])) / dir(indexMap_[i]) : 1.0;
			double lowerStep = (-0.5 + radius - q(indexMap_[i])) / dir(indexMap_[i]) > 0 ? (-0.5 + radius - q(indexMap_[i])) / dir(indexMap_[i]) : 1.0;
			double qiStep = 0.8 * std::min(upperStep, lowerStep);
			maxStep = std::min(maxStep, qiStep);
		}

	}
	return maxStep;
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
//	if (params_.floorEnabled)
//	{
//		Eigen::VectorXd floorGrad;
//		computeParticleFloorGradeint(q, floorGrad);
//
//		Eigen::VectorXd springGrad, gravityGrad, gradE;
//		computeSpringGradient(q, springGrad);
//		gradE = springGrad;
//		double l = 0.02 * std::sqrt(params_.particleMass);
//
//
//		double kappa_g = floorGrad.norm() > 0 ? -floorGrad.dot(gradE) / floorGrad.squaredNorm() : 0;
//
//		// suggested kappa by IPC paper:
//		double d = 1e-8 * l; // 0.02 is the radius of the point in gui
//		double Hb = -2 * std::log(d / params_.barrierEps) + (params_.barrierEps - d) * (params_.barrierEps + 3 * d) / (d * d);
//		double kappa_min = 1e11 * params_.particleMass / (4e-16 * l * Hb);
//		double kappa_max = 100 * kappa_min;
//
//		params_.barrierStiffness = std::min(kappa_max, std::max(kappa_min, kappa_g));
//
//		double d_eps = 1e-9 * l;
//		updateCloseParticles(q, d_eps);
//
//	}
}

void GooHook1d::postIteration(Eigen::VectorXd q)
{
//	if (params_.floorEnabled)
//	{
//		double l = 0.02 * std::sqrt(params_.particleMass);
//		double d_eps = 1e-9 * l;
//
//		double d = 1e-8 * l; // 0.02 is the radius of the point in gui
//		double Hb = -2 * std::log(d / params_.barrierEps) + (params_.barrierEps - d) * (params_.barrierEps + 3 * d) / (d * d);
//		double kappa_min = 1e11 * params_.particleMass / (4e-16 * l * Hb);
//		double kappa_max = 100 * kappa_min;
//
//		for (int i = 0; i < closeParticles_.size(); i++)
//		{
//			int pid = closeParticles_[i].first;
//			if (q(i) <= -0.5 + l + d_eps || q(i) >= 0.5 - l - d_eps)
//			{
//				double pos = q(i);
//				double dist = d_eps;
//				if (q(i) <= -0.5 + l + d_eps)
//					dist = q(i) + 0.5 - l;
//				else
//					dist = 0.5 - l - q(i);
//				if (dist < closeParticles_[i].second)
//					params_.barrierStiffness = std::min(kappa_max, 2 * params_.barrierStiffness);
//			}
//		}
//
//		updateCloseParticles(q, d_eps);
//		std::cout << "After this iteration, barrier stiffness is: " << params_.barrierStiffness << std::endl;
//	}
}


//////////////////////////////////////////////////////////////////////////////////////
///                   Test Part
//////////////////////////////////////////////////////////////////////////////////////
void GooHook1d::testPotentialDifferential()
{
	Eigen::VectorXd q, qPrev, vel, preVel;
	generateConfiguration(q, vel, qPrev, preVel);
	
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
	Eigen::VectorXd q, qPrev, vel, preVel;
	generateConfiguration(q, vel, qPrev, preVel);
	
	
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
	int nConnectors = fullConnectors_.size();
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
		outfile << fullConnectors_[i]->p1 << "\n";
		outfile << fullConnectors_[i]->p2 << "\n";
		outfile << fullConnectors_[i]->mass << "\n";
		outfile << static_cast<Spring*>(fullConnectors_[i])->restlen << "\n";
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
	fullConnectors_.clear();
	
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
		int p1;
		int p2;
		double mass;
		double restLen;
		infile >> p1;
		infile >> p2;
		infile >> mass;
		infile >> restLen;
		auto spring = new Spring(p1, p2, mass, params_.springStiffness, restLen, true);
		fullConnectors_.push_back(spring);
	}
}
