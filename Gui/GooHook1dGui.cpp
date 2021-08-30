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

#include "GooHook1dGui.h"

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

void GooHook1dGui::drawGUI(igl::opengl::glfw::imgui::ImGuiMenu &menu)
{
	if (ImGui::CollapsingHeader("Configuration", ImGuiTreeNodeFlags_DefaultOpen))
	{
		float w = ImGui::GetContentRegionAvailWidth();
		float p = ImGui::GetStyle().FramePadding.x;
		if (ImGui::Button("Load", ImVec2((w - p) / 2.f, 0)))
		{
			std::string filePath = igl::file_dialog_open();
			model_.loadConfiguration(filePath);
		}
		ImGui::SameLine(0, p);
		if (ImGui::Button("Save", ImVec2((w - p) / 2.f, 0)))
		{
			std::string filePath = igl::file_dialog_save();
			model_.saveConfiguration(filePath);
		}
	}
	if (ImGui::CollapsingHeader("Simulation Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::InputDouble("Timestep", &params_.timeStep))
			updateParams();
		if(ImGui::Combo("Model Type", (int*)&params_.modelType, "Harmonic Ossocilation\0Pogo Stick\0\0"))
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
		if(ImGui::InputDouble("  Viscosity", &params_.dampingStiffness))
			updateParams();
		if(ImGui::Checkbox("Floor Enabled", &params_.floorEnabled))
			updateParams();
		if(ImGui::Checkbox("Floor Friction Enabled", &params_.frictionEnabled))
			updateParams();
	}


	if (ImGui::CollapsingHeader("New Particles", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::Checkbox("Is Fixed", &params_.particleFixed))
			updateParams();
		if (ImGui::InputDouble("Mass", &params_.particleMass))
			updateParams();
	}

	if (ImGui::CollapsingHeader("New Springs", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::InputDouble("Max Spring Dist", &params_.maxSpringDist))
			updateParams();
		if (ImGui::InputDouble("Base Stiffness", &params_.springStiffness))
			updateParams();
	}

	if (ImGui::Button("Test Gradient and Hessian", ImVec2(-1, 0)))
	{
		model_.testPotentialDifferential();
		model_.testGradientDifferential();
	}
	
}


void GooHook1dGui::reset()
{
	initSimulation();
	updateRenderGeometry();
}



void GooHook1dGui::updateRenderGeometry()
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
	std::vector<Eigen::Vector3d> vertexColors;
	std::vector<Eigen::Vector3i> faces;

	int idx = 0;

	double eps = 1e-4;


	if(params_.floorEnabled)
	{
		for (int i = 0; i < 6; i++)
		{
			vertexColors.push_back(Eigen::Vector3d(0, 0, 0));
		}

		verts.push_back(Eigen::Vector3d(-1, -0.5, eps));
		verts.push_back(Eigen::Vector3d(1, -0.5, eps));
		verts.push_back(Eigen::Vector3d(-1, -0.5 - 0.01, eps));

		faces.push_back(Eigen::Vector3i(idx, idx + 1, idx + 2));

		verts.push_back(Eigen::Vector3d(-1, -0.5 - 0.01, eps));
		verts.push_back(Eigen::Vector3d(1, -0.5, eps));
		verts.push_back(Eigen::Vector3d(1, -0.5 - 0.01, eps));
		faces.push_back(Eigen::Vector3i(idx + 3, idx + 4, idx + 5));
		idx += 6;

		for (int i = 0; i < 6; i++)
		{
			vertexColors.push_back(Eigen::Vector3d(0, 0, 0));
		}

		verts.push_back(Eigen::Vector3d(-1, 0.5, eps));
		verts.push_back(Eigen::Vector3d(1, 0.5, eps));
		verts.push_back(Eigen::Vector3d(-1, 0.5 + 0.01, eps));

		faces.push_back(Eigen::Vector3i(idx, idx + 1, idx + 2));

		verts.push_back(Eigen::Vector3d(-1, 0.5 + 0.01, eps));
		verts.push_back(Eigen::Vector3d(1, 0.5, eps));
		verts.push_back(Eigen::Vector3d(1, 0.5 + 0.01, eps));
		faces.push_back(Eigen::Vector3i(idx + 3, idx + 4, idx + 5));
		idx += 6;
	}

	if (params_.modelType == SimParameters::MT_HARMONIC_1D)
	{
		for (std::vector<Connector1d*>::iterator it = model_.connectors_.begin(); it != model_.connectors_.end(); ++it)
		{
			Eigen::Vector3d color;
			if ((*it)->associatedBendingStencils.empty())
				color << 0.0, 0.0, 1.0;
			else
				color << 0.75, 0.5, 0.75;
			Vector2d sourcepos = model_.particles_[(*it)->p].pos;
			sourcepos(1) = params_.ceil;
			Vector2d destpos = model_.particles_[(*it)->p].pos;

			Vector2d vec = destpos - sourcepos;
			Vector2d perp(-vec[1], vec[0]);
			perp /= perp.norm();

			double dist = (sourcepos - destpos).norm();

			double width = baselinewidth / (1.0 + 20.0 * dist * dist);

			for (int i = 0; i < 4; i++)
				vertexColors.push_back(color);

			verts.push_back(Eigen::Vector3d(sourcepos[0] + width * perp[0], sourcepos[1] + width * perp[1], -eps));
			verts.push_back(Eigen::Vector3d(sourcepos[0] - width * perp[0], sourcepos[1] - width * perp[1], -eps));
			verts.push_back(Eigen::Vector3d(destpos[0] + width * perp[0], destpos[1] + width * perp[1], -eps));
			verts.push_back(Eigen::Vector3d(destpos[0] - width * perp[0], destpos[1] - width * perp[1], -eps));

			faces.push_back(Eigen::Vector3i(idx, idx + 1, idx + 2));
			faces.push_back(Eigen::Vector3i(idx + 2, idx + 1, idx + 3));
			idx += 4;
		}
	}
	else
	{
		for (std::vector<Connector*>::iterator it = model_.fullConnectors_.begin(); it != model_.fullConnectors_.end(); ++it)
		{
			Eigen::Vector3d color;
			if ((*it)->associatedBendingStencils.empty())
				color << 0.0, 0.0, 1.0;
			else
				color << 0.75, 0.5, 0.75;
			Vector2d sourcepos = model_.particles_[(*it)->p1].pos;
			Vector2d destpos = model_.particles_[(*it)->p2].pos;

			Vector2d vec = destpos - sourcepos;
			Vector2d perp(-vec[1], vec[0]);
			perp /= perp.norm();

			double dist = (sourcepos - destpos).norm();

			double width = baselinewidth / (1.0 + 20.0 * dist * dist);

			for (int i = 0; i < 4; i++)
				vertexColors.push_back(color);

			verts.push_back(Eigen::Vector3d(sourcepos[0] + width * perp[0], sourcepos[1] + width * perp[1], -eps));
			verts.push_back(Eigen::Vector3d(sourcepos[0] - width * perp[0], sourcepos[1] - width * perp[1], -eps));
			verts.push_back(Eigen::Vector3d(destpos[0] + width * perp[0], destpos[1] + width * perp[1], -eps));
			verts.push_back(Eigen::Vector3d(destpos[0] - width * perp[0], destpos[1] - width * perp[1], -eps));

			faces.push_back(Eigen::Vector3i(idx, idx + 1, idx + 2));
			faces.push_back(Eigen::Vector3i(idx + 2, idx + 1, idx + 3));
			idx += 4;
		}
	}
	

	int nParticles = model_.particles_.size();

	for(int i=0; i<nParticles; i++)
	{
		double radius = baseradius*sqrt(model_.particles_[i].mass);

		Eigen::Vector3d color(0,0,0);

		if(model_.particles_[i].fixed)
		{
			radius = baseradius;
			color << 1.0, 0, 0;
		}

		for (int j = 0; j < numcirclewedges + 2; j++)
		{
			vertexColors.push_back(color);
		}


		verts.push_back(Eigen::Vector3d(model_.particles_[i].pos[0], model_.particles_[i].pos[1], 0));

		const double PI = 3.1415926535898;
		for (int j = 0; j <= numcirclewedges; j++)
		{
			verts.push_back(Eigen::Vector3d(model_.particles_[i].pos[0] + radius * cos(2 * PI*j / numcirclewedges),
				model_.particles_[i].pos[1] + radius * sin(2 * PI*j / numcirclewedges), 0));
		}

		for (int j = 0; j <= numcirclewedges; j++)
		{
			faces.push_back(Eigen::Vector3i(idx, idx + j + 1, idx + 1 + ((j + 1) % (numcirclewedges + 1))));
		}

		idx += numcirclewedges + 2;
	}


	renderQ.resize(verts.size(),3);
	renderC.resize(vertexColors.size(), 3);
	for (int i = 0; i < verts.size(); i++)
	{
		renderQ.row(i) = verts[i];
		renderC.row(i) = vertexColors[i];
	}
	renderF.resize(faces.size(), 3);
	for (int i = 0; i < faces.size(); i++)
		renderF.row(i) = faces[i];
}

void GooHook1dGui::getOutputFolderPath()
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

bool GooHook1dGui::computePeriod(Eigen::VectorXd q, std::vector<double>& periods)
{
	if (params_.modelType == SimParameters::MT_HARMONIC_1D)
	{
		periods.resize(model_.connectors_.size(), -1);
		for (int i = 0; i < model_.connectors_.size(); i++)
		{
			double curLen = model_.getCurrentConnectorLen(q, i);
			double restLen = static_cast<Spring1d*>(model_.connectors_[i])->restDis;
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
	}
	else
		return false;
}

void GooHook1dGui::initSimulation()
{
	time_ = 0;
	iterNum_ = 0;
	GIFScale_ = 0.6;
	model_ = GooHook1d(params_);
	baseFolder_ = "../output/";

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
	double delay_10ms = std::min(10.0, params_.timeStep * 100.0);
	GIFStep_ = static_cast<int>(std::ceil(3.0 / delay_10ms));
	GIFDelay_ = static_cast<int>(delay_10ms * GIFStep_); // always about 3x10ms, around 33FPS

	getOutputFolderPath();

	isPaused_ = true;
	totalTime_ = params_.totalTime;
	totalIterNum_ = params_.totalNumIter;

	saveperiodFile_ = false;
}

bool GooHook1dGui::simulateOneStep()
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

void GooHook1dGui::saveInfo()
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