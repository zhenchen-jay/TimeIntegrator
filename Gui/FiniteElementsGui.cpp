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

#include "FiniteElementsGui.h"

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


using namespace Eigen;

void FiniteElementsGui::drawGUI(igl::opengl::glfw::imgui::ImGuiMenu& menu)
{
	if (ImGui::CollapsingHeader("Simulation Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::InputDouble("Timestep", &params_.timeStep))
			updateParams();
		if (ImGui::Combo("Model Type", (int*)&params_.modelType, "Harmonic Ossocilation\0Pogo Stick\0\0"))
		{
			initSimulation();
		}
		if (ImGui::InputInt("Number of Segments", &params_.numSegs))
		{
			initSimulation();
		}
		if (ImGui::Combo("Integrator", (int*)&params_.integrator, "Explicit Euler\0Velocity Verlet\0Runge Kutta 4\0Exp Euler\0Implicit Euler\0Implicit Midpoint\0Trapzoid\0TRBDF2\0BDF2\0Newmark\0\0"))
			updateParams();
		if (ImGui::InputDouble("Newmark Gamma", &params_.NM_gamma))
			updateParams();
		if (ImGui::InputDouble("Newmark Beta", &params_.NM_beta))
			updateParams();
		if (ImGui::InputDouble("TRBDF2 Gamma", &params_.TRBDF2_gamma))
			updateParams();
		if (ImGui::InputDouble("Newton Tolerance", &params_.NewtonTolerance))
			updateParams();
		if (ImGui::InputInt("Newton Max Iters", &params_.NewtonMaxIters))
			updateParams();
		if (ImGui::InputDouble("Spring Stiffness", &params_.springStiffness))
			updateParams();
		if (ImGui::InputDouble("Total Time", &params_.totalTime))
			updateParams();
		if (ImGui::InputDouble("IPC Barrier stiffness", &params_.barrierStiffness))
			updateParams();
	}
	if (ImGui::CollapsingHeader("Forces", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::Checkbox("Gravity Enabled", &params_.gravityEnabled))
			updateParams();
		if (ImGui::InputDouble("  Gravity g", &params_.gravityG))
			updateParams();
		if (ImGui::Checkbox("Springs Enabled", &params_.springsEnabled))
			updateParams();
		if (ImGui::InputDouble("  Max Strain", &params_.maxSpringStrain))
			updateParams();
		if (ImGui::Checkbox("Damping Enabled", &params_.dampingEnabled))
			updateParams();
		if (ImGui::InputDouble("  Viscosity", &params_.dampingStiffness))
			updateParams();
		if (ImGui::Checkbox("Floor Enabled", &params_.floorEnabled))
			updateParams();
		if (ImGui::Checkbox("Floor Friction Enabled", &params_.frictionEnabled))
			updateParams();
	}


	if (ImGui::CollapsingHeader("Particles Params", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::InputDouble("Mass", &params_.particleMass))
			updateParams();
	}

	if (ImGui::CollapsingHeader("Springs Info", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::InputDouble("Max Spring Dist", &params_.maxSpringDist))
			updateParams();
		if (ImGui::InputDouble("Base Stiffness", &params_.springStiffness))
			updateParams();
	}

	if (ImGui::Button("Test Gradient and Hessian", ImVec2(-1, 0)))
	{
		model_->testPotentialDifferential(curQ_);
		model_->testGradientDifferential(curQ_);
	}

}


void FiniteElementsGui::reset()
{
	initSimulation();
	updateRenderGeometry();
}



void FiniteElementsGui::updateRenderGeometry()
{
	double baseradius = 0.02;
	double pulsefactor = 0.1;
	double pulsespeed = 50.0;

	int sawteeth = 20;
	double sawdepth = 0.1;
	double sawangspeed = 10.0;

	double baselinewidth = 0.005;

	int numcirclewedges = 20;

	// this is terrible. But, easiest to get up and running

	std::vector<Eigen::Vector3d> verts;
	std::vector<Eigen::Vector3i> faces;

	int idx = 0;

	for (int i = 0; i < curQ_.size(); i++)
	{
		verts.push_back(Eigen::Vector3d(-0.01, curQ_(i), 0));
		verts.push_back(Eigen::Vector3d(0.01, curQ_(i), 0));

		if (i != curQ_.size() - 1)
		{
			faces.push_back(Eigen::Vector3i(idx, idx + 1, idx + 3));
			faces.push_back(Eigen::Vector3i(idx, idx + 3, idx + 2));
		}
		idx += 2;
	}

	renderQ.resize(verts.size(), 3);
	for (int i = 0; i < verts.size(); i++)
	{
		renderQ.row(i) = verts[i];
	}
	renderF.resize(faces.size(), 3);
	for (int i = 0; i < faces.size(); i++)
		renderF.row(i) = faces[i];
}

void FiniteElementsGui::getOutputFolderPath()
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
		std::cout << "create directory: " << outputFolderPath_ << std::endl;
		if (!std::filesystem::create_directories(outputFolderPath_))
		{
			std::cout << "create folder failed." << outputFolderPath_ << std::endl;
		}
	}
	std::cout << outputFolderPath_ << std::endl;
}

bool FiniteElementsGui::computePeriod(Eigen::VectorXd q, std::vector<double>& periods)
{
	periods.resize(curF_.rows(), -1);
	return true;
}

void FiniteElementsGui::initSimulation()
{
	time_ = 0;
	iterNum_ = 0;
	GIFScale_ = 0.6;
	model_ = std::make_shared<LinearElements>(params_);
	baseFolder_ = "../output/";

	curQ_.resize(params_.numSegs + 1);
	curF_.resize(params_.numSegs, 2);

	if (params_.modelType == SimParameters::MT_HARMONIC_1D)
	{
		curQ_(0) = 0;
		for (int i = 1; i <= params_.numSegs; i++)
		{
			curQ_(i) = -0.3 * i / params_.numSegs;
			curF_(i - 1) << i - 1, i;
		}

		Eigen::VectorXd restQ = curQ_ * 0.5;
		Eigen::VectorXd massVec = restQ;
		massVec.setConstant(1.0);

		std::map<int, double> clampedPoints;
		clampedPoints[0] = 0;

		model_->initialize(restQ, curF_, massVec, &clampedPoints);
	}

	else
	{
		curQ_(0) = 0;
		for (int i = 1; i <= params_.numSegs; i++)
		{
			curQ_(i) = 0.3 * i / params_.numSegs;
			curF_(i - 1) << i - 1, i;
		}

		Eigen::VectorXd restQ = curQ_ * 0.5;
		Eigen::VectorXd massVec = restQ;
		massVec.setConstant(1.0);
		model_->initialize(restQ, curF_, massVec, NULL);

	}

	getOutputFolderPath();

	isPaused_ = true;
	totalTime_ = params_.totalTime;
	totalIterNum_ = params_.totalNumIter;

	saveperiodFile_ = false;

	preQ_ = curQ_;
	curVel_.setZero(curQ_.size());
	preVel_ = curVel_;
}

bool FiniteElementsGui::simulateOneStep()
{
	Eigen::VectorXd posNew, velNew;
	switch (params_.integrator)
	{
	case SimParameters::TI_EXPLICIT_EULER:
		TimeIntegrator::explicitEuler<LinearElements>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<LinearElements*>(model_.get())), posNew, velNew);
		break;
	case SimParameters::TI_RUNGE_KUTTA:
		TimeIntegrator::RoungeKutta4<LinearElements>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<LinearElements*>(model_.get())), posNew, velNew);
		break;
	case SimParameters::TI_VELOCITY_VERLET:
		TimeIntegrator::velocityVerlet<LinearElements>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<LinearElements*>(model_.get())), posNew, velNew);
		break;
	case SimParameters::TI_EXP_ROSENBROCK_EULER:
		TimeIntegrator::exponentialRosenBrockEuler<LinearElements>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<LinearElements*>(model_.get())), posNew, velNew);
		break;
	case SimParameters::TI_IMPLICIT_EULER:
		TimeIntegrator::implicitEuler<LinearElements>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<LinearElements*>(model_.get())), posNew, velNew);
		break;
	case SimParameters::TI_IMPLICIT_MIDPOINT:
		TimeIntegrator::implicitMidPoint<LinearElements>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<LinearElements*>(model_.get())), posNew, velNew);
		break;
	case SimParameters::TI_TRAPEZOID:
		TimeIntegrator::trapezoid<LinearElements>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<LinearElements*>(model_.get())), posNew, velNew);
		break;
	case SimParameters::TI_TR_BDF2:
		TimeIntegrator::trapezoidBDF2<LinearElements>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<LinearElements*>(model_.get())), posNew, velNew, params_.TRBDF2_gamma);
		break;
	case SimParameters::TI_BDF2:
		if (time_ == 0)
			TimeIntegrator::implicitEuler<LinearElements>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<LinearElements*>(model_.get())), posNew, velNew);
		else
			TimeIntegrator::BDF2<LinearElements>(curQ_, curVel_, preQ_, preVel_, params_.timeStep, model_->massVec_, *(static_cast<LinearElements*>(model_.get())), posNew, velNew);
		break;
	case SimParameters::TI_NEWMARK:
		TimeIntegrator::Newmark<LinearElements>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<LinearElements*>(model_.get())), posNew, velNew, params_.NM_gamma, params_.NM_beta);
		break;
	}
	//update configuration into particle data structure
	preQ_ = curQ_;
	preVel_ = curVel_;
	curQ_ = posNew;
	curVel_ = velNew;

	time_ += params_.timeStep;
	iterNum_ += 1;

	return false;
}

void FiniteElementsGui::saveInfo()
{
	std::string statFileName = outputFolderPath_ + std::string("simulation_status.txt");
	std::ofstream sfs;

	if (time_)
		sfs.open(statFileName, std::ofstream::out | std::ofstream::app);
	else
		sfs.open(statFileName, std::ofstream::out);

	double springPotential = model_->computeElasticPotential(curQ_);
	double gravityPotential = model_->computeGravityPotential(curQ_);
	double IPCbarier = model_->computeFloorBarrier(curQ_);

	std::vector<Eigen::Triplet<double>> massTrip;
	Eigen::SparseMatrix<double> massMat(model_->massVec_.size(), model_->massVec_.size());

	for (int i = 0; i < model_->massVec_.size(); i++)
		massTrip.push_back({ i, i , model_->massVec_(i) });
	massMat.setFromTriplets(massTrip.begin(), massTrip.end());

	double kineticEnergy = 0.5 * curVel_.transpose() * massMat * curVel_;

	double center = curQ_.sum() / curQ_.size();

	sfs << time_ << " " << springPotential << " " << gravityPotential << " " << model_->params_.barrierStiffness * IPCbarier << " " << kineticEnergy << " " << springPotential + gravityPotential + model_->params_.barrierStiffness * IPCbarier + kineticEnergy << " " << center << std::endl;
	std::cout << time_ << ", spring: " << springPotential << ", gravity: " << gravityPotential << ", IPC barrier: " << model_->params_.barrierStiffness * IPCbarier << ", kinetic: " << kineticEnergy << ", total: " << springPotential + gravityPotential + model_->params_.barrierStiffness * IPCbarier + kineticEnergy << ", com: " << center << std::endl;

}