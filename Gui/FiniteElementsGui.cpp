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
    if (ImGui::CollapsingHeader("Problem Setup Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::InputDouble("Bar Length", &params_.barLen))
        {
            if(params_.barLen < 0)
            {
                std::cerr << "wrong input for bar len." << std::endl;
                exit(1);
            }
            updateParams();
            reset();
        }
        if(ImGui::InputDouble("Bar Height", &params_.barHeight))
        {
            if(params_.barHeight < 0)
            {
                std::cerr << "wrong input for barHeight." << std::endl;
                exit(1);
            }

            updateParams();
            reset();
        }
        if(ImGui::InputDouble("Top Ceil", &params_.topLine))
        {
            if(params_.topLine < 0)
            {
                std::cerr << "wrong input for topLine." << std::endl;
                exit(1);
            }
            updateParams();
            reset();
        }
    }
	if (ImGui::CollapsingHeader("Simulation Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::InputDouble("Timestep", &params_.timeStep))
			updateParams();
		if (ImGui::Combo("Model Type", (int*)&params_.modelType, "Harmonic Ossocilation\0Pogo Stick\0\0"))
		{
		    updateParams();
		    reset();
		}
		if (ImGui::InputInt("Number of Segments", &params_.numSegs))
		{
		    updateParams();
		    reset();
		}
		if (ImGui::Combo("Material", (int*)&params_.materialType, "Linear\0NeoHookean\0\0"))
		{
		    updateParams();
		    reset();
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
			reset();
		if (ImGui::InputDouble("IPC Barrier stiffness", &params_.barrierStiffness))
			reset();
		if (ImGui::InputInt("Number of Spectra", &params_.numSpectra))
			reset();
		if (ImGui::Checkbox("Save Info", &params_.isSaveInfo))
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
		if (ImGui::Checkbox("Internal Contact Enabled", &params_.internalContactEnabled))
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
		std::cout << "test floor barrier: " << std::endl;
		std::cout << "f-g: " << std::endl;
		model_->testFloorBarrierEnergy(curQ_);

		std::cout << "g-h: " << std::endl;
		model_->testFloorBarrierGradient(curQ_);

		std::cout << "test internal barrier: " << std::endl;
		std::cout << "f-g: " << std::endl;
		model_->testInternalBarrierEnergy(curQ_);
		std::cout << "g-h: " << std::endl;
		model_->testInternalBarrierGradient(curQ_);

		std::cout << "test Elastic potential: " << std::endl;
		model_->testElasticEnergy(curQ_);
		model_->testElasticGradient(curQ_);
		//
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
	std::vector<Eigen::Vector3d> verts;
	std::vector<Eigen::Vector3i> faces;
	std::vector<Eigen::Vector3d> vertsColors;

	int idx = 0;

	double eps = 1e-4;


	if (params_.floorEnabled)
	{
	    for (int i = 0; i < 6; i++)
	    {
	        vertsColors.push_back(Eigen::Vector3d(0, 0, 0));
	    }

	    verts.push_back(Eigen::Vector3d(-params_.topLine, 0, eps));
	    verts.push_back(Eigen::Vector3d(params_.topLine, 0, eps));
	    verts.push_back(Eigen::Vector3d(-params_.topLine, 0 - 0.01 * params_.topLine, eps));

	    faces.push_back(Eigen::Vector3i(idx, idx + 1, idx + 2));

	    verts.push_back(Eigen::Vector3d(-params_.topLine, 0 - 0.01 * params_.topLine, eps));
	    verts.push_back(Eigen::Vector3d(params_.topLine, 0, eps));
	    verts.push_back(Eigen::Vector3d(params_.topLine, 0 - 0.01 * params_.topLine, eps));
	    faces.push_back(Eigen::Vector3i(idx + 3, idx + 4, idx + 5));
	    idx += 6;

	   /* for (int i = 0; i < 6; i++)
	    {
	        vertsColors.push_back(Eigen::Vector3d(0, 0, 0));
	    }

	    verts.push_back(Eigen::Vector3d(-params_.topLine, params_.topLine, eps));
	    verts.push_back(Eigen::Vector3d(params_.topLine, params_.topLine, eps));
	    verts.push_back(Eigen::Vector3d(-params_.topLine, params_.topLine + 0.01 * params_.topLine, eps));

	    faces.push_back(Eigen::Vector3i(idx, idx + 1, idx + 2));

	    verts.push_back(Eigen::Vector3d(-params_.topLine, params_.topLine + 0.01 * params_.topLine, eps));
	    verts.push_back(Eigen::Vector3d(params_.topLine, params_.topLine, eps));
	    verts.push_back(Eigen::Vector3d(params_.topLine, params_.topLine + 0.01 * params_.topLine, eps));
	    faces.push_back(Eigen::Vector3i(idx + 3, idx + 4, idx + 5));
	    idx += 6;*/
	}

	for (int i = 0; i < curPos_.size(); i++)
	{
	    verts.push_back(Eigen::Vector3d(-0.03 * params_.topLine, curPos_(i), 0));
	    verts.push_back(Eigen::Vector3d(-0.01 * params_.topLine, curPos_(i), 0));

		Eigen::RowVector3d color;

		igl::colormap(igl::COLOR_MAP_TYPE_VIRIDIS, 2.0 / 9.0, color.data());

		//color << 255, 175, 255;
		vertsColors.push_back(color);
		vertsColors.push_back(color);

		if (i != curPos_.size() - 1)
		{
			faces.push_back(Eigen::Vector3i(idx + 1, idx, idx + 3));
			faces.push_back(Eigen::Vector3i(idx + 3, idx, idx + 2));
		}
		idx += 2;
	}

	if (isTheoretical_)
	{
		for (int i = 0; i < curPosTheo_.size(); i++)
		{
			verts.push_back(Eigen::Vector3d(0.01 * params_.topLine, curPosTheo_(i), 0));
			verts.push_back(Eigen::Vector3d(0.03 * params_.topLine, curPosTheo_(i), 0));

			Eigen::RowVector3d color;

			igl::colormap(igl::COLOR_MAP_TYPE_VIRIDIS, 4.0 / 9.0, color.data());

			//color << 255, 175, 255;
			vertsColors.push_back(color);
			vertsColors.push_back(color);

			if (i != curPosTheo_.size() - 1)
			{
				faces.push_back(Eigen::Vector3i(idx + 1, idx, idx + 3));
				faces.push_back(Eigen::Vector3i(idx + 3, idx, idx + 2));
			}
			idx += 2;
		}
	}
	

	renderQ.resize(verts.size(), 3);
	for (int i = 0; i < verts.size(); i++)
	{
		renderQ.row(i) = verts[i];
	}
	renderF.resize(faces.size(), 3);
	for (int i = 0; i < faces.size(); i++)
		renderF.row(i) = faces[i];

	renderC.resize(vertsColors.size(), 3);
	for(int i = 0; i < vertsColors.size(); i++)
	{
	    renderC.row(i) = vertsColors[i];
	}

	if (isTheoretical_)
		edgeStart.setZero(2 * (curPos_.size() * 3 - 2), 3);
	else
		edgeStart.setZero(curPos_.size() * 3 - 2, 3);
	edgeEnd = edgeStart;

	for(int i = 0; i < curPos_.size(); i++)
	{
	    edgeStart.row(i) = Eigen::Vector3d(-0.03 * params_.topLine, curPos_(i), 0);
	    edgeEnd.row(i) = Eigen::Vector3d(-0.01 * params_.topLine, curPos_(i), 0);

		if (i < curPos_.size() - 1)
		{
		    edgeStart.row(2 * i + curPos_.size()) = Eigen::Vector3d(-0.03 * params_.topLine, curPos_(i), 0);
		    edgeEnd.row(2 * i + curPos_.size()) = Eigen::Vector3d(-0.03 * params_.topLine, curPos_(i + 1), 0);

		    edgeStart.row(2 * i + curPos_.size() + 1) = Eigen::Vector3d(-0.01 * params_.topLine, curPos_(i), 0);
		    edgeEnd.row(2 * i + curPos_.size() + 1) = Eigen::Vector3d(-0.01 * params_.topLine, curPos_(i + 1), 0);
		}
	}

	if (isTheoretical_)
	{
		for (int i = 0; i < curPosTheo_.size(); i++)
		{
			edgeStart.row(i + curPos_.size() * 3 - 2) = Eigen::Vector3d(0.01 * params_.topLine, curPosTheo_(i), 0);
			edgeEnd.row(i + curPos_.size() * 3 - 2) = Eigen::Vector3d(0.03 * params_.topLine, curPosTheo_(i), 0);

			if (i < curPos_.size() - 1)
			{
				edgeStart.row(2 * i + curPos_.size() + curPos_.size() * 3 - 2) = Eigen::Vector3d(0.01 * params_.topLine, curPosTheo_(i), 0);
				edgeEnd.row(2 * i + curPos_.size() + curPos_.size() * 3 - 2) = Eigen::Vector3d(0.01 * params_.topLine, curPosTheo_(i + 1), 0);

				edgeStart.row(2 * i + curPos_.size() + 1 + curPos_.size() * 3 - 2) = Eigen::Vector3d(0.03 * params_.topLine, curPosTheo_(i), 0);
				edgeEnd.row(2 * i + curPos_.size() + 1 + curPos_.size() * 3 - 2) = Eigen::Vector3d(0.03 * params_.topLine, curPosTheo_(i + 1), 0);
			}
		}
	}
}

void FiniteElementsGui::getOutputFolderPath()
{
	if (params_.modelType == SimParameters::MT_HARMONIC_1D)
	{
		outputFolderPath_ = baseFolder_ + "Harmonic1d/" + std::to_string(params_.timeStep) + "_" + std::to_string(params_.barrierStiffness) + "_numSeg_" + std::to_string(params_.numSegs) + "/";
	}

	else
	{
		outputFolderPath_ = baseFolder_ + "Pogo_Stick/" + std::to_string(params_.timeStep) + "_" + std::to_string(params_.barrierStiffness) + "_numSeg_" + std::to_string(params_.numSegs) + "/";
	}

	if (params_.materialType == SimParameters::MT_LINEAR)
		outputFolderPath_ = outputFolderPath_ + "Linear/";
	else
		outputFolderPath_ = outputFolderPath_ + "NeoHookean/";

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
	double deltax = params_.barLen / params_.numSegs;
	std::cout << "max time step size according CFL is: " << deltax / std::sqrt(params_.youngs / params_.particleMass) << std::endl;

	if (params_.modelType == SimParameters::MT_HARMONIC_1D && params_.materialType == SimParameters::MT_LINEAR && (params_.integrator == SimParameters::TI_IMPLICIT_EULER || params_.integrator == SimParameters::TI_BDF2 || params_.integrator == SimParameters::TI_NEWMARK || params_.integrator == SimParameters::TI_TR_BDF2) && params_.gravityEnabled == false)
	{
		isTheoretical_ = true;
	}


    switch (params_.materialType)
    {
        case SimParameters::MT_LINEAR:
            model_ = std::make_shared<LinearElements>(params_);
            break;
        case SimParameters::MT_NEOHOOKEAN:
            model_ = std::make_shared<NeoHookean>(params_);
    }

	baseFolder_ = "../output/";

	curPos_.resize(params_.numSegs + 1);
	curF_.resize(params_.numSegs, 2);

	if (params_.modelType == SimParameters::MT_HARMONIC_1D)
	{
	    Eigen::VectorXd restPos = curPos_;
		Eigen::VectorXd massVec = restPos;

		curPos_(0) = params_.barHeight + params_.barLen;
		restPos(0) = params_.barHeight + params_.barLen;
		for (int i = 1; i <= params_.numSegs; i++)
		{
			curPos_(i) = curPos_(0) - params_.barLen * i / params_.numSegs;
			restPos(i) = curPos_(0) - 0.8 * params_.barLen * i / params_.numSegs;
			curF_.row(i - 1) << i - 1, i;

			massVec(i) = params_.particleMass * params_.barLen / params_.numSegs;
		}
		
		std::map<int, double> clampedPoints;
		clampedPoints[0] = curPos_(0);
		model_->initialize(restPos, curF_, massVec, &clampedPoints);
	}

	else
	{
	    curPos_(0) = params_.barHeight + params_.barLen;
		for (int i = 1; i <= params_.numSegs; i++)
		{
		    curPos_(i) = curPos_(0) - params_.barLen * i / params_.numSegs;
			curF_.row(i - 1) << i - 1, i;
		}

		Eigen::VectorXd restPos = curPos_;
		Eigen::VectorXd massVec = restPos;
		massVec.setConstant(1.0);
		model_->initialize(restPos, curF_, massVec, NULL);

	}
    model_->convertPos2Var(curPos_, curQ_);
	getOutputFolderPath();

	isPaused_ = true;
	totalTime_ = params_.totalTime;
	totalIterNum_ = params_.totalNumIter;

	saveperiodFile_ = false;

	preQ_ = curQ_;
	curVel_.setZero(curQ_.size());
	preVel_ = curVel_;

	if (isTheoretical_)
	{
		curQTheo_ = curQ_;
		curVelTheo_ = curVel_;

		preQTheo_ = preQ_;
		preVelTheo_ = preVel_;

		curPosTheo_ = curPos_;

		theoModel_ = SpectraAnalysisLinearElements(params_, curQTheo_, curVelTheo_, *(static_cast<LinearElements*>(model_.get())));
	}
}

bool FiniteElementsGui::simulateOneStep()
{
    std::cout << "simulate one step." << std::endl;

	if (isTheoretical_)
	{
		preQTheo_ = curQTheo_;
		preVelTheo_ = curVel_;

		theoModel_.updateAlphasBetas();
		theoModel_.getCurPosVel(curQTheo_, curVelTheo_);
		model_->convertVar2Pos(curQTheo_, curPosTheo_);
	}

	Eigen::VectorXd posNew, velNew;
	switch (params_.integrator)
	{
	case SimParameters::TI_EXPLICIT_EULER:
	    if(params_.materialType == SimParameters::MT_LINEAR)
	    {
	        std::cout << "model type: Linear Elasticity. Time integration: Explicit Euler." << std::endl;
	        TimeIntegrator::explicitEuler<LinearElements>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<LinearElements*>(model_.get())), posNew, velNew);
	    }

	    else if(params_.materialType == SimParameters::MT_NEOHOOKEAN)
	    {
	        std::cout << "model type: NeoHookean. Time integration: Explicit Euler." << std::endl;
	        TimeIntegrator::explicitEuler<NeoHookean>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<NeoHookean*>(model_.get())), posNew, velNew);
	    }
		break;
	case SimParameters::TI_RUNGE_KUTTA:
	    if(params_.materialType == SimParameters::MT_LINEAR)
	    {
	        std::cout << "model type: Linear Elasticity. Time integration: RK4." << std::endl;
	        TimeIntegrator::RoungeKutta4<LinearElements>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<LinearElements*>(model_.get())), posNew, velNew);
	    }
	    else if(params_.materialType == SimParameters::MT_NEOHOOKEAN)
	    {
	        std::cout << "model type: NoeHookean. Time integration: RK4." << std::endl;
	        TimeIntegrator::RoungeKutta4<NeoHookean>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<NeoHookean*>(model_.get())), posNew, velNew);
	    }
		break;
	case SimParameters::TI_VELOCITY_VERLET:
	    if(params_.materialType == SimParameters::MT_LINEAR)
	    {
	        std::cout << "model type: Linear Elasticity. Time integration: Velocity Verlet." << std::endl;
	        TimeIntegrator::velocityVerlet<LinearElements>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<LinearElements*>(model_.get())), posNew, velNew);
	    }
	    else if(params_.materialType == SimParameters::MT_NEOHOOKEAN)
	    {
	        std::cout << "model type: NeoHookean. Time integration: Velocity Verlet." << std::endl;
	        TimeIntegrator::velocityVerlet<NeoHookean>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<NeoHookean*>(model_.get())), posNew, velNew);
	    }
		break;
	case SimParameters::TI_EXP_ROSENBROCK_EULER:
	    if(params_.materialType == SimParameters::MT_LINEAR)
	    {
	        std::cout << "model type: Linear Elasticity. Time integration: Exp Euler." << std::endl;
	        TimeIntegrator::exponentialRosenBrockEuler<LinearElements>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<LinearElements*>(model_.get())), posNew, velNew);
	    }
	    else if(params_.materialType == SimParameters::MT_NEOHOOKEAN)
	    {
	        std::cout << "model type: NeoHookean. Time integration: Exp Euler." << std::endl;
	        TimeIntegrator::exponentialRosenBrockEuler<NeoHookean>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<NeoHookean*>(model_.get())), posNew, velNew);
	    }
		break;
	case SimParameters::TI_IMPLICIT_EULER:
	    if(params_.materialType == SimParameters::MT_LINEAR)
	    {
	        std::cout << "model type: Linear Elasticity. Time integration: Implicit Euler." << std::endl;
	        TimeIntegrator::implicitEuler<LinearElements>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<LinearElements*>(model_.get())), posNew, velNew);
	    }
	    else if(params_.materialType == SimParameters::MT_NEOHOOKEAN)
	    {
	        std::cout << "model type: NeoHookean. Time integration: Implicit Euler." << std::endl;
	        TimeIntegrator::implicitEuler<NeoHookean>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<NeoHookean*>(model_.get())), posNew, velNew);
	    }
		break;
	case SimParameters::TI_IMPLICIT_MIDPOINT:
	    if(params_.materialType == SimParameters::MT_LINEAR)
	    {
	        std::cout << "model type: Linear Elasticity. Time integration: Midpoint." << std::endl;
	        TimeIntegrator::implicitMidPoint<LinearElements>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<LinearElements*>(model_.get())), posNew, velNew);
	    }
	    else if(params_.materialType == SimParameters::MT_NEOHOOKEAN)
	    {
	        std::cout << "model type: NeoHookean. Time integration: Midpoint." << std::endl;
	        TimeIntegrator::implicitMidPoint<NeoHookean>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<NeoHookean*>(model_.get())), posNew, velNew);
	    }
		break;
	case SimParameters::TI_TRAPEZOID:
	    if(params_.materialType == SimParameters::MT_LINEAR)
	    {
	        std::cout << "model type: Linear Elasticity. Time integration: Trapezoid." << std::endl;
	        TimeIntegrator::trapezoid<LinearElements>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<LinearElements*>(model_.get())), posNew, velNew);
	    }
	    else if(params_.materialType == SimParameters::MT_NEOHOOKEAN)
	    {
	        std::cout << "model type: NeoHookean. Time integration: Trapezoid." << std::endl;
	        TimeIntegrator::trapezoid<NeoHookean>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<NeoHookean*>(model_.get())), posNew, velNew);
	    }
		break;
	case SimParameters::TI_TR_BDF2:
	    if(params_.materialType == SimParameters::MT_LINEAR)
	    {
	        std::cout << "model type: Linear Elasticity. Time integration: TRBDF2." << std::endl;
	        TimeIntegrator::trapezoidBDF2<LinearElements>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<LinearElements*>(model_.get())), posNew, velNew, params_.TRBDF2_gamma);
	    }
	    else if(params_.materialType == SimParameters::MT_NEOHOOKEAN)
	    {
	        std::cout << "model type: NeoHookean. Time integration: TRBDF2." << std::endl;
	        TimeIntegrator::trapezoidBDF2<NeoHookean>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<NeoHookean*>(model_.get())), posNew, velNew);
	    }
		break;
	case SimParameters::TI_BDF2:
		if (time_ == 0)
		{
		    if(params_.materialType == SimParameters::MT_LINEAR)
		    {
		        std::cout << "model type: Linear Elasticity. Time integration: BDF2." << std::endl;
		        TimeIntegrator::implicitEuler<LinearElements>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<LinearElements*>(model_.get())), posNew, velNew);
		    }
		    else if(params_.materialType == SimParameters::MT_NEOHOOKEAN)
		    {
		        std::cout << "model type: NeoHookean. Time integration: BDF2." << std::endl;
		        TimeIntegrator::implicitEuler<NeoHookean>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<NeoHookean*>(model_.get())), posNew, velNew);
		    }
		}

		else
		{
		    if(params_.materialType == SimParameters::MT_LINEAR)
		    {
		        std::cout << "model type: Linear Elasticity. Time integration: BDF2." << std::endl;
		        TimeIntegrator::BDF2<LinearElements>(curQ_, curVel_, preQ_, preVel_, params_.timeStep, model_->massVec_, *(static_cast<LinearElements*>(model_.get())), posNew, velNew);
		    }
		    else if(params_.materialType == SimParameters::MT_NEOHOOKEAN)
		    {
		        std::cout << "model type: NeoHookean. Time integration: BDF2." << std::endl;
		        TimeIntegrator::BDF2<NeoHookean>(curQ_, curVel_, preQ_, preVel_, params_.timeStep, model_->massVec_, *(static_cast<NeoHookean*>(model_.get())), posNew, velNew);
		    }
		}

		break;
	case SimParameters::TI_NEWMARK:
	    if(params_.materialType == SimParameters::MT_LINEAR)
	    {
	        std::cout << "model type: Linear Elasticity. Time integration: Newmark." << std::endl;
	        TimeIntegrator::Newmark<LinearElements>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<LinearElements*>(model_.get())), posNew, velNew, params_.NM_gamma, params_.NM_beta);
	    }
	    else if(params_.materialType == SimParameters::MT_NEOHOOKEAN)
	    {
	        std::cout << "model type: NeoHookean. Time integration: Newmark." << std::endl;
	        TimeIntegrator::Newmark<NeoHookean>(curQ_, curVel_, params_.timeStep, model_->massVec_, *(static_cast<NeoHookean*>(model_.get())), posNew, velNew, params_.NM_gamma, params_.NM_beta);
	    }
		break;
	}
	//update configuration into particle data structure
	preQ_ = curQ_;
	preVel_ = curVel_;
	curQ_ = posNew;
	curVel_ = velNew;
	model_->convertVar2Pos(curQ_, curPos_);

	time_ += params_.timeStep;
	iterNum_ += 1;

	return false;
}

void FiniteElementsGui::saveInfo()
{
	std::string statFileName = outputFolderPath_ + std::string("simulation_status.txt");
	std::ofstream sfs;

	if (params_.isSaveInfo)
	{
		if (time_)
			sfs.open(statFileName, std::ofstream::out | std::ofstream::app);
		else
			sfs.open(statFileName, std::ofstream::out);
	}
	
	double springPotential = model_->computeElasticPotential(curQ_);
	double gravityPotential = model_->computeGravityPotential(curQ_);
	double IPCbarier = model_->computeFloorBarrier(curQ_);

	std::vector<Eigen::Triplet<double>> massTrip;
	Eigen::SparseMatrix<double> massMat(model_->massVec_.size(), model_->massVec_.size());

	for (int i = 0; i < model_->massVec_.size(); i++)
		massTrip.push_back({ i, i , model_->massVec_(i) });
	massMat.setFromTriplets(massTrip.begin(), massTrip.end());

	double kineticEnergy = 0.5 * curVel_.transpose() * massMat * curVel_;
	double internalContact = 0;
	if(params_.internalContactEnabled)
	    internalContact = model_->computeInternalBarrier(curQ_);

	double bottom = curQ_.minCoeff();
	if(params_.isSaveInfo)
		sfs << time_ << " " << springPotential << " " << gravityPotential << " " << model_->params_.barrierStiffness * IPCbarier << " " << internalContact * model_->params_.barrierStiffness << " " << kineticEnergy << " " << springPotential + gravityPotential + model_->params_.barrierStiffness * IPCbarier + kineticEnergy << " " << bottom;
	std::cout << time_ << ", spring: " << springPotential << ", gravity: " << gravityPotential << ", IPC barrier: " << model_->params_.barrierStiffness * IPCbarier << " , Internal Barrier: " << internalContact * model_->params_.barrierStiffness << ", kinetic: " << kineticEnergy << ", total: " << springPotential + gravityPotential + model_->params_.barrierStiffness * IPCbarier + kineticEnergy << ", bottom pos: " << bottom;
	if (isTheoretical_)
	{
		double theoBot = curQTheo_.minCoeff();
		if (params_.isSaveInfo)
			sfs << " " << theoBot << std::endl;
		std::cout << ", theoretical bottom: " << theoBot << ", error: " << theoBot - bottom << std::endl;
	}
	if (params_.isSaveInfo)
		sfs << std::endl;
	std::cout << std::endl;

}