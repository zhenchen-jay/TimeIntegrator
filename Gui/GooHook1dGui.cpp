#include<Eigen/SparseLU>
#include<Eigen/IterativeLinearSolvers>
#include <iomanip>
#include <iostream>
#include <fstream>

#include "GooHook1dGui.h"
#include "../include/IntegrationScheme/ExplicitEuler.h"
#include "../include/IntegrationScheme/RoungeKutta4.h"
#include "../include/IntegrationScheme/VelocityVerlet.h"
#include "../include/IntegrationScheme/ExponentialRosenBrockEuler.h"

#include "../include/IntegrationScheme/ImplicitEuler.h"
#include "../include/IntegrationScheme/Trapezoid.h"
#include "../include/IntegrationScheme/ImplicitMidPoint.h"
#include "../include/IntegrationScheme/TrapezoidBDF2.h"
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

	if (ImGui::CollapsingHeader("UI Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::Combo("Click Adds", (int*)&params_.clickMode, "Particles\0Saws\0\0"))
		{
			updateParams();
		}
	}
	if (ImGui::CollapsingHeader("Simulation Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::InputDouble("Timestep", &params_.timeStep))
			updateParams();
		if(ImGui::Combo("Model Type", (int*)&params_.modelType, "Harmonic Ossocilation\0Mass Spring\0\0"))
		{
			initSimulation();
		}
		if (ImGui::Combo("Integrator", (int*)&params_.integrator, "Explicit Euler\0Velocity Verlet\0Runge Kutta 4\0Exp Euler\0Implicit Euler\0Implicit Midpoint\0Trapzoid\0TRBDF2\0Newmark\0\0"))
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
			vertexColors.push_back(Eigen::Vector3d(0.3, 1.0, 0.3));
		}

		verts.push_back(Eigen::Vector3d(-1, -0.5, eps));
		verts.push_back(Eigen::Vector3d(1, -0.5, eps));
		verts.push_back(Eigen::Vector3d(-1, -1, eps));

		faces.push_back(Eigen::Vector3i(idx, idx + 1, idx + 2));

		verts.push_back(Eigen::Vector3d(-1, -1, eps));
		verts.push_back(Eigen::Vector3d(1, -0.5, eps));
		verts.push_back(Eigen::Vector3d(1, -1, eps));
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


void GooHook1dGui::initSimulation()
{
	time_ = 0;
	model_ = GooHook1d(params_);
	if(params_.modelType == SimParameters::MT_HARMONIC_1D)
		model_.addParticle(0, -0.3);
	else
	{
		model_.addParticle(0, 0.3);
		model_.addParticle(0, 0);
	}
}

void GooHook1dGui::tick()
{
	/*message_mutex.lock();
	{
		while (!mouseClicks_.empty())
		{
			MouseClick mc = mouseClicks_.front();
			mouseClicks_.pop_front();
			switch (mc.mode)
			{
			case SimParameters::ClickMode::CM_ADDPARTICLE:
			{
				model_.addParticle(mc.x, mc.y);
				break;
			}
			}
		}
	}
	message_mutex.unlock();*/
}

bool GooHook1dGui::simulateOneStep()
{
	VectorXd pos, vel, prevPos;
	model_.generateConfiguration(pos, vel, prevPos);  
	model_.assembleMassVec();
	//std::cout<<"Generate Done"<<std::endl;
	Eigen::VectorXd posNew, velNew;

	switch (params_.integrator)
	{
	case SimParameters::TI_EXPLICIT_EULER:
		explicitEuler<GooHook1d>(pos, vel, params_.timeStep, model_.massVec_, model_, posNew, velNew);
		break;
	case SimParameters::TI_RUNGE_KUTTA:
		RoungeKutta4<GooHook1d>(pos, vel, params_.timeStep, model_.massVec_, model_, posNew, velNew);
		break;
	case SimParameters::TI_VELOCITY_VERLET:
		velocityVerlet<GooHook1d>(pos, vel, params_.timeStep, model_.massVec_, model_, posNew, velNew);
		break;
	case SimParameters::TI_EXP_ROSENBROCK_EULER:
		exponentialRosenBrockEuler<GooHook1d>(pos, vel, params_.timeStep, model_.massVec_, model_, posNew, velNew);
		break;
	case SimParameters::TI_IMPLICIT_EULER:
		implicitEuler<GooHook1d>(pos, vel, params_.timeStep, model_.massVec_, model_, posNew, velNew);
		break;
	case SimParameters::TI_IMPLICIT_MIDPOINT:
		implicitMidPoint<GooHook1d>(pos, vel, params_.timeStep, model_.massVec_, model_, posNew, velNew);
		break;
	case SimParameters::TI_TRAPEZOID:
		trapezoid<GooHook1d>(pos, vel, params_.timeStep, model_.massVec_, model_, posNew, velNew);
		break;
	case SimParameters::TI_TR_BDF2:
		trapezoidBDF2<GooHook1d>(pos, vel, params_.timeStep, model_.massVec_, model_, posNew, velNew, params_.TRBDF2_gamma);
		break;
	case SimParameters::TI_NEWMARK:
		Newmark<GooHook1d>(pos, vel, params_.timeStep, model_.massVec_, model_, posNew, velNew, params_.NM_gamma, params_.NM_beta);
		break;
	}
	//update configuration into particle data structure
	prevPos = pos;
	pos = posNew;
	vel = velNew;
	std::cout << "prePos: " << pos.norm() << ", current Pos: " << pos.norm() << ", vel: " << vel.norm() << std::endl;
	model_.degenerateConfiguration(pos, vel, prevPos);
	//std::cout<<"Degenerate Done"<<std::endl;

	time_ += params_.timeStep;

	return false;
}
