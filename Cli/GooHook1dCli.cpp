#include<Eigen/SparseLU>
#include<Eigen/IterativeLinearSolvers>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <filesystem>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/colormap.h>
#include <igl/png/writePNG.h>

#include "GooHook1dCli.h"

#include "../include/IntegrationScheme/ExplicitEuler.h"
#include "../include/IntegrationScheme/RoungeKutta4.h"
#include "../include/IntegrationScheme/VelocityVerlet.h"
#include "../include/IntegrationScheme/ExponentialRosenBrockEuler.h"

#include "../include/IntegrationScheme/ImplicitEuler.h"
#include "../include/IntegrationScheme/Trapezoid.h"
#include "../include/IntegrationScheme/ImplicitMidPoint.h"
#include "../include/IntegrationScheme/TrapezoidBDF2.h"
#include "../include/IntegrationScheme/BDF2.h"
#include "../include/IntegrationScheme/Newmark.h"



#ifndef MASS_TOLERANCE
#define MASS_TOLERANCE 100.0
#endif

using namespace Eigen;

void GooHook1dCli::getOutputFolderPath()
{
	if (params_.modelType == SimParameters::MT_HARMONIC_1D)
	{
		outputFolderPath_ = baseFolder_ + "Harmonic1d/" + std::to_string(params_.timeStep) + "_" + std::to_string(params_.barrierStiffness) + "/";
	}

	else
	{
	    outputFolderPath_ = baseFolder_ + "Pogo_Stick/" + std::to_string(params_.timeStep) + "_" + std::to_string(params_.barrierStiffness) + "/";
	}
	switch (params_.integrator)
	{
	case SimParameters::TI_EXPLICIT_EULER:
		outputFolderPath_ = outputFolderPath_ + "Explicit Euler/";
		break;
	case SimParameters::TI_RUNGE_KUTTA:
		outputFolderPath_ = outputFolderPath_ + "RK4/";
		break;
	case SimParameters::TI_VELOCITY_VERLET:
		outputFolderPath_ = outputFolderPath_ + "Velocity Verlet/";
		break;
	case SimParameters::TI_EXP_ROSENBROCK_EULER:
		outputFolderPath_ = outputFolderPath_ + "Exponential Euler/";
		break;
	case SimParameters::TI_IMPLICIT_EULER:
		outputFolderPath_ = outputFolderPath_ + "Implicit Euler/";
		break;
	case SimParameters::TI_IMPLICIT_MIDPOINT:
		outputFolderPath_ = outputFolderPath_ + "Implicit midpoint/";
		break;
	case SimParameters::TI_TRAPEZOID:
		outputFolderPath_ = outputFolderPath_ + "Trapezoid/";
		break;
	case SimParameters::TI_TR_BDF2:
		outputFolderPath_ = outputFolderPath_ + "TRBDF2/";
		break;
	case SimParameters::TI_BDF2:
		outputFolderPath_ = outputFolderPath_ + "BDF2/";
		break;
	case SimParameters::TI_NEWMARK:
		outputFolderPath_ = outputFolderPath_ + "Newmark/";
		break;
	}

	if (!std::filesystem::exists(outputFolderPath_))
	{
		std::filesystem::create_directories(outputFolderPath_);
	}
}

bool GooHook1dCli::computePeriod(Eigen::VectorXd q, std::vector<double>& periods)
{

	periods.resize(model_.fullConnectors_.size(), -1);
	for (int i = 0; i < model_.fullConnectors_.size(); i++)
	{
		double curLen = model_.getCurrentConnectorLen(q, i);
		double restLen = static_cast<Spring*>(model_.fullConnectors_[i])->restlen;
		restLen = std::abs(restLen);

		bool isPassRestPoint = false;
		double zeroTime = time_;
		if (lastLens_[i] > 0)
		{
			if ((curLen - restLen) * (lastLens_[i] - restLen) <= 0)
			{
				isPassRestPoint = true;

				double a1 = std::abs(curLen - restLen);
				double a2 = std::abs(lastLens_[i] - restLen);

				zeroTime = a2 / (a1 + a2) * time_ + a1 / (a1 + a2) * (time_ - params_.timeStep);

			}

		}
		lastLens_[i] = curLen;

		if (isPassRestPoint)
		{
			if (lastPassRestPointTime_[i] < 0)
			{
				lastPassRestPointTime_[i] = zeroTime;
				periods[i] = -1;
			}
			else
			{
				double period = 2 * (zeroTime - lastPassRestPointTime_[i]);
				lastPassRestPointTime_[i] = zeroTime;
				periods[i] = period;
			}
		}
		else
			periods[i] = -1;
	}
	return true;
}

void GooHook1dCli::initSimulation(std::string outputFolder)
{
	time_ = 0;
	iterNum_ = 0;
	model_ = GooHook1d(params_);
	baseFolder_ = outputFolder;

	if (params_.modelType == SimParameters::MT_HARMONIC_1D)
	{
		model_.addParticle(0, -0.3);
		lastPassRestPointTime_.resize(1);
		lastLens_.resize(1);

		for (int i = 0; i < lastLens_.size(); i++)
		{
			lastLens_[i] = std::abs(model_.particles_[i].pos(1) - params_.ceil);
			lastPassRestPointTime_[i] = -1;
		}

	}

	else
	{
		model_.addParticle(0, 0.3);
		model_.addParticle(0, 0);
	}

	getOutputFolderPath();

	totalTime_ = params_.totalTime;
	totalIterNum_ = params_.totalNumIter;

	saveperiodFile_ = false;
	model_.updateProjM();
}

bool GooHook1dCli::simulateOneStep()
{
 	VectorXd pos, vel, prevPos, preVel;
	model_.generateConfiguration(pos, vel, prevPos, preVel);  
	model_.assembleMassVec();
	//std::cout<<"Generate Done"<<std::endl;
	Eigen::VectorXd posNew, velNew;

	switch (params_.integrator)
	{
	case SimParameters::TI_EXPLICIT_EULER:
		TimeIntegrator::explicitEuler<GooHook1d>(pos, vel, params_.timeStep, model_.massVec_, model_, posNew, velNew);
		break;
	case SimParameters::TI_RUNGE_KUTTA:
		TimeIntegrator::RoungeKutta4<GooHook1d>(pos, vel, params_.timeStep, model_.massVec_, model_, posNew, velNew);
		break;
	case SimParameters::TI_VELOCITY_VERLET:
		TimeIntegrator::velocityVerlet<GooHook1d>(pos, vel, params_.timeStep, model_.massVec_, model_, posNew, velNew);
		break;
	case SimParameters::TI_EXP_ROSENBROCK_EULER:
		TimeIntegrator::exponentialRosenBrockEuler<GooHook1d>(pos, vel, params_.timeStep, model_.massVec_, model_, posNew, velNew);
		break;
	case SimParameters::TI_IMPLICIT_EULER:
		TimeIntegrator::implicitEuler<GooHook1d>(pos, vel, params_.timeStep, model_.massVec_, model_, posNew, velNew);
		break;
	case SimParameters::TI_IMPLICIT_MIDPOINT:
		TimeIntegrator::implicitMidPoint<GooHook1d>(pos, vel, params_.timeStep, model_.massVec_, model_, posNew, velNew);
		break;
	case SimParameters::TI_TRAPEZOID:
		TimeIntegrator::trapezoid<GooHook1d>(pos, vel, params_.timeStep, model_.massVec_, model_, posNew, velNew);
		break;
	case SimParameters::TI_TR_BDF2:
		TimeIntegrator::trapezoidBDF2<GooHook1d>(pos, vel, params_.timeStep, model_.massVec_, model_, posNew, velNew, params_.TRBDF2_gamma);
		break;
	case SimParameters::TI_BDF2:
		if(time_ == 0)
			TimeIntegrator::implicitEuler<GooHook1d>(pos, vel, params_.timeStep, model_.massVec_, model_, posNew, velNew);
		else
			TimeIntegrator::BDF2<GooHook1d>(pos, vel, prevPos, preVel, params_.timeStep, model_.massVec_, model_, posNew, velNew);
		break;
	case SimParameters::TI_NEWMARK:
		TimeIntegrator::Newmark<GooHook1d>(pos, vel, params_.timeStep, model_.massVec_, model_, posNew, velNew, params_.NM_gamma, params_.NM_beta);
		break;
	}
	//update configuration into particle data structure
	prevPos = pos;
	preVel = vel;
	pos = posNew;
	vel = velNew;
	//std::cout << "prePos: " << pos.norm() << ", current Pos: " << pos.norm() << ", vel: " << vel.norm() << std::endl;
	model_.degenerateConfiguration(pos, vel, prevPos, preVel);
	//std::cout<<"Degenerate Done"<<std::endl;
	double gp = model_.computeGravityPotential(pos);
	std::vector<Eigen::Triplet<double>> massTrip;
	Eigen::SparseMatrix<double> massMat(model_.massVec_.size(), model_.massVec_.size());

	for (int i = 0; i < model_.massVec_.size(); i++)
		massTrip.push_back({ i, i , model_.massVec_(i) });
	massMat.setFromTriplets(massTrip.begin(), massTrip.end());

	double keneticEnergy = 0.5 * vel.transpose() * massMat * vel;

	//std::cout << "g + k= " << gp + keneticEnergy << std::endl;

	time_ += params_.timeStep;
	iterNum_ += 1;

	return false;
}

void GooHook1dCli::saveInfo()
{
	std::string statFileName = outputFolderPath_ + std::string("simulation_status.txt");
	std::ofstream sfs;
	
	if (time_)
		sfs.open(statFileName, std::ofstream::out | std::ofstream::app);
	else
		sfs.open(statFileName, std::ofstream::out);

	VectorXd pos, vel, prevPos, preVel;
	model_.generateConfiguration(pos, vel, prevPos, preVel);
	model_.assembleMassVec();

	double springPotential = model_.computeSpringPotential(pos);
	double gravityPotential = model_.computeGravityPotential(pos);
	double IPCbarier = model_.computeParticleFloorBarrier(pos);

	std::vector<Eigen::Triplet<double>> massTrip;
	Eigen::SparseMatrix<double> massMat(model_.massVec_.size(), model_.massVec_.size());

	for (int i = 0; i < model_.massVec_.size(); i++)
		massTrip.push_back({ i, i , model_.massVec_(i) });
	massMat.setFromTriplets(massTrip.begin(), massTrip.end());

	double kineticEnergy = 0.5 * vel.transpose() * massMat * vel;

	double center = pos.sum() / pos.size();

	sfs << time_ << " " << springPotential << " " << gravityPotential << " " << model_.params_.barrierStiffness * IPCbarier << " " << kineticEnergy << " " << springPotential + gravityPotential + model_.params_.barrierStiffness * IPCbarier + kineticEnergy  << " " << center << std::endl;
	std::cout << time_ << ", spring: " << springPotential << ", gravity: " << gravityPotential << ", IPC barrier: " << model_.params_.barrierStiffness * IPCbarier << ", kinetic: " << kineticEnergy << ", total: " << springPotential + gravityPotential + model_.params_.barrierStiffness * IPCbarier + kineticEnergy << ", com: " << center << std::endl;

	if (params_.modelType == SimParameters::MT_HARMONIC_1D)
	{
		std::string periodFileName = outputFolderPath_ + std::string("simulation_period.txt");
		std::ofstream pfs;

		std::vector<double> periods;
		computePeriod(pos, periods);
		bool worth2Save = false;
		for (int i = 0; i < periods.size(); i++)
			if (periods[i] > 0)
				worth2Save = true;

		if (worth2Save)
		{
			if (saveperiodFile_)
				pfs.open(periodFileName, std::ofstream::out | std::ofstream::app);
			else
				pfs.open(periodFileName, std::ofstream::out);
			saveperiodFile_ = true;
			pfs << time_ << " ";
			for (int i = 0; i < periods.size(); i++)
				pfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << periods[i] << " ";
			pfs << "\n";
		}
	}
}