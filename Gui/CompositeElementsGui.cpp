#include<Eigen/SparseLU>
#include<Eigen/IterativeLinearSolvers>
#include <iomanip>
#include <iostream>
#include <fstream>

#include <MatOp/SparseCholesky.h>
#include <MatOp/SparseRegularInverse.h>
#include <MatOp/SymShiftInvert.h>
#include <MatOp/SparseSymMatProd.h>
#include <SymGEigsSolver.h>
#include <SymGEigsShiftSolver.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/colormap.h>
#include <igl/png/writePNG.h>
#include <igl/jet.h>

#include "CompositeElementsGui.h"

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
#include "../include/IntegrationScheme/AdditiveScheme.h"
#include "../include/IntegrationScheme/SplitScheme.h"
#include "../include/IntegrationScheme/CompositeScheme.h"
#include "../include/IntegrationScheme/ProjectionScheme.h"

using namespace Eigen;

void CompositeElementsGui::drawGUI(igl::opengl::glfw::imgui::ImGuiMenu& menu)
{
	if (ImGui::CollapsingHeader("Visualization Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::Checkbox("Plot compression", &plotCompression_);
	}
	if (ImGui::CollapsingHeader("Simulation Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::InputDouble("Timestep", &params_.timeStep))
			updateParams();
		if (ImGui::Combo("Integrator", (int*)&params_.integrator, "Explicit Euler\0Velocity Verlet\0Runge Kutta 4\0Exp Euler\0Implicit Euler\0Implicit Midpoint\0Trapzoid\0TRBDF2\0BDF2\0Newmark\0Additive\0Split\0FEPR\0Composite\0\0"))
			updateParams();
		if (ImGui::InputDouble("Split ratio", &params_.splitRatio))
			updateParams();
		if (ImGui::InputDouble("Newmark Gamma", &params_.NM_gamma))
			updateParams();
		if (ImGui::InputDouble("Newmark Beta", &params_.NM_beta))
			updateParams();
		if (ImGui::InputDouble("Composte rho", &params_.rho))
			updateParams();
		if (ImGui::InputDouble("TRBDF2 Gamma", &params_.TRBDF2_gamma))
			updateParams();
		if (ImGui::InputDouble("Newton Tolerance", &params_.NewtonTolerance))
			updateParams();
		if (ImGui::InputInt("Newton Max Iters", &params_.NewtonMaxIters))
			updateParams();
		if (ImGui::InputDouble("Total Time", &params_.totalTime))
			reset();
		if (ImGui::InputDouble("IPC Barrier stiffness", &params_.barrierStiffness))
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
		if (ImGui::Checkbox("Damping Enabled", &params_.dampingEnabled))
			updateParams();
		if (ImGui::Checkbox("Floor Enabled", &params_.floorEnabled))
			updateParams();
		if (ImGui::Checkbox("Internal Contact Enabled", &params_.internalContactEnabled))
			updateParams();
	}


	if (ImGui::CollapsingHeader("Material Params", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::InputDouble("Mass", &params_.particleMass))
			updateParams();
		ImGui::InputDouble("Stiff Youngs", &stiffYoungs_);
		ImGui::InputInt("stiff seg num", &numStiffLi_);
		ImGui::InputDouble("stiff stretch ratio", &stiffEnlargeRatio_);

		ImGui::InputDouble("soft Youngs", &softYoungs_);
		ImGui::InputInt("soft seg num", &numSoftLi_);
		ImGui::InputDouble("soft stretch ratio", &softEnlargeRatio_);

		ImGui::InputDouble("Neo Youngs", &nonlinearYoungs_);
		ImGui::InputInt("Neo seg num", &numNH_);
		ImGui::InputDouble("NH stretch ratio", &NHEnlargeRatio_);
	}

	if (ImGui::Button("Test Gradient and Hessian", ImVec2(-1, 0)))
	{
		model_->testPotentialDifferential(curQ_);
		model_->testGradientDifferential(curQ_);
	}

}


void CompositeElementsGui::reset()
{
	initSimulation();
	updateParams();
	updateRenderGeometry();
}



void CompositeElementsGui::updateRenderGeometry()
{
	std::vector<Eigen::Vector3d> verts;
	std::vector<Eigen::Vector3i> faces;
	std::vector<Eigen::Vector3d> colorList;

	int idx = 0;

	double eps = 1e-4;


	if (params_.floorEnabled)
	{
		if (plotCompression_) // face based
		{
			for (int i = 0; i < 2; i++)
			{
				colorList.push_back(Eigen::Vector3d(0, 0, 0));
			}
		}
		else
		{
			for (int i = 0; i < 6; i++)
			{
				colorList.push_back(Eigen::Vector3d(0, 0, 0));
			}
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

	Eigen::MatrixXd faceColor;
	if (plotCompression_)
	{
		Eigen::VectorXd compression;
		model_->computeCompression(curPos_, compression);
		igl::jet(compression, false, faceColor);
	}


	int faceId = 0;
	for (int i = 0; i < curPos_.size(); i++)
	{
		verts.push_back(Eigen::Vector3d(-0.05 * params_.topLine, curPos_(i), 0));
		verts.push_back(Eigen::Vector3d(0.05 * params_.topLine, curPos_(i), 0));

		Eigen::RowVector3d color;

		igl::colormap(igl::COLOR_MAP_TYPE_VIRIDIS, 2.0 / 9.0, color.data());

		if (!plotCompression_)
		{
			colorList.push_back(color);
			colorList.push_back(color);
		}

		if (i != curPos_.size() - 1)
		{
			faces.push_back(Eigen::Vector3i(idx, idx + 1, idx + 3));
			faces.push_back(Eigen::Vector3i(idx, idx + 3, idx + 2));

			if (plotCompression_)
			{
				colorList.push_back(faceColor.row(faceId));
				colorList.push_back(faceColor.row(faceId));
				faceId++;
			}
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


	renderC.resize(colorList.size(), 3);
	for (int i = 0; i < colorList.size(); i++)
	{
		renderC.row(i) = colorList[i];
	}



	edgeStart.setZero(curPos_.size() * 3 - 2, 3);

	edgeEnd = edgeStart;

	idx = 0;


	for (int i = 0; i < curPos_.size(); i++)
	{
		edgeStart.row(i) = Eigen::Vector3d(-0.05 * params_.topLine, curPos_(i), 0);
		edgeEnd.row(i) = Eigen::Vector3d(0.05 * params_.topLine, curPos_(i), 0);

		if (i < curPos_.size() - 1)
		{
			edgeStart.row(2 * i + curPos_.size()) = Eigen::Vector3d(-0.05 * params_.topLine, curPos_(i), 0);
			edgeEnd.row(2 * i + curPos_.size()) = Eigen::Vector3d(-0.05 * params_.topLine, curPos_(i + 1), 0);

			edgeStart.row(2 * i + curPos_.size() + 1) = Eigen::Vector3d(0.05 * params_.topLine, curPos_(i), 0);
			edgeEnd.row(2 * i + curPos_.size() + 1) = Eigen::Vector3d(0.05 * params_.topLine, curPos_(i + 1), 0);
		}
	}
}

void CompositeElementsGui::getOutputFolderPath()
{
	outputFolderPath_ = baseFolder_ + "Compoiste/" + std::to_string(params_.timeStep) + "_" + std::to_string(params_.barrierStiffness) + "_numSeg_" + std::to_string(numNH_) + "_" + std::to_string(numStiffLi_) + "_" + std::to_string(numSoftLi_) + "/";

	outputFolderPath_ = outputFolderPath_ + "stretch_ratio_" + std::to_string(NHEnlargeRatio_) + "_" + std::to_string(stiffEnlargeRatio_) + "_" + std::to_string(softEnlargeRatio_) + "/";
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
	case SimParameters::TI_ADDITIVE:
		outputFolderPath_ = outputFolderPath_ + "Additive/";
		break;
	case SimParameters::TI_SPLIT:
		outputFolderPath_ = outputFolderPath_ + "Split_" + std::to_string(params_.splitRatio) + "/";
		break;
	case SimParameters::TI_COMPOSITE:
		outputFolderPath_ = outputFolderPath_ + "Composite/";
	case SimParameters::TI_PROJECTION:
		outputFolderPath_ = outputFolderPath_ + "Projection/";
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

void CompositeElementsGui::initSimulation()
{
	time_ = 0;
	iterNum_ = 0;

	int numSegs = numNH_ + numSoftLi_ + numStiffLi_;
	baseFolder_ = "../output/";

	curPos_.resize(numSegs + 1);
	elements_.resize(numSegs);

	Eigen::VectorXd restPos = curPos_;
	Eigen::VectorXd massVec = restPos;
	
	curPos_(0) = 0;
	restPos(0) = 0;
	massVec(0) = params_.particleMass * 1.0 / numNH_;
	
	std::map<int, double> clampedPos = { {0, 0} };

	for (int i = 0; i < numSegs; i++)
	{
		double deltax = 0;
		double ratio = 0;
		if (i < numNH_)
		{
			deltax = 1.0 / numNH_;
			ratio = NHEnlargeRatio_;
		}
			
		else if (i < numNH_ + numStiffLi_)
		{
			deltax = 1.0 / numStiffLi_;
			ratio = stiffEnlargeRatio_;
		}
			
		else
		{
			deltax = 1.0 / numSoftLi_;
			ratio = softEnlargeRatio_;
		}
		
		restPos(i + 1) = deltax + restPos(i);
		curPos_(i + 1) = ratio * deltax + curPos_(i);
	}

	for (int i = 0; i <= numSegs; i++)
	{
		double len = 0;
		if (i > 0 && i < numSegs)
		{
			len = (restPos(i + 1) - restPos(i)) + (restPos(i) - restPos(i - 1));
			len /= 2;
		}
		else if (i == 0)
			len = 1.0 / numNH_;
		else
			len = 1.0 / numSoftLi_;
		massVec(i) = len * params_.particleMass;
	}

	std::cout << restPos << std::endl;
	for (int i = 0; i < numNH_; i++)
	{
		elements_[i] = std::make_shared<NHElement>(nonlinearYoungs_, params_.poisson, restPos(i), restPos(i+1), i, i + 1);
	}

	for (int i = 0; i < numStiffLi_ + numSoftLi_; i++)
	{
		double tmpY = softYoungs_;
		if (i < numStiffLi_)
			tmpY = stiffYoungs_;
		elements_[i + numNH_] = std::make_shared<LiElement>(tmpY, params_.poisson, restPos(i + numNH_), restPos(i + numNH_ + 1), i + numNH_, i + numNH_ + 1);
	}

	model_ = std::make_shared<CompositeModel>();
	model_->initialize(params_, restPos, elements_, massVec, &clampedPos);

	model_->convertPos2Var(curPos_, curQ_);
	getOutputFolderPath();

	isPaused_ = true;
	totalTime_ = params_.totalTime;
	totalIterNum_ = params_.totalNumIter;

	preQ_ = curQ_;
	curVel_.setZero(curQ_.size());
	preVel_ = curVel_;
	
	initialEnergy_ = model_->computeEnergy(curQ_);

	for (int i = 0; i < model_->elements_.size(); i++)
	{
		int v0 = model_->elements_[i]->vid0_;
		int v1 = model_->elements_[i]->vid1_;
		double e = model_->elements_[i]->computeElementPotential(curPos_(v0), curPos_(v1), NULL, NULL);
		std::cout << "element " << i << ", energy: " << e << std::endl;
	}

}

bool CompositeElementsGui::simulateOneStep()
{
	std::cout << "simulate one step." << std::endl;


	Eigen::VectorXd posNew, velNew;
	Eigen::VectorXd massVec = model_->assembleMass(model_->massVec_);

	switch (params_.integrator)
	{
	case SimParameters::TI_EXPLICIT_EULER:
		std::cout << "Time integration: Explicit Euler." << std::endl;
		TimeIntegrator::explicitEuler<CompositeModel>(curQ_, curVel_, params_.timeStep, massVec, *(static_cast<CompositeModel*>(model_.get())), posNew, velNew);
		break;

	case SimParameters::TI_RUNGE_KUTTA:
		std::cout << "Time integration: RK4." << std::endl;
		TimeIntegrator::RoungeKutta4<CompositeModel>(curQ_, curVel_, params_.timeStep, massVec, *(static_cast<CompositeModel*>(model_.get())), posNew, velNew);
		break;

	case SimParameters::TI_VELOCITY_VERLET:
		std::cout << "Time integration: Velocity Verlet." << std::endl;
		TimeIntegrator::velocityVerlet<CompositeModel>(curQ_, curVel_, params_.timeStep, massVec, *(static_cast<CompositeModel*>(model_.get())), posNew, velNew);
		break;

	case SimParameters::TI_EXP_ROSENBROCK_EULER:
		std::cout << "Time integration: Exp Euler." << std::endl;
		TimeIntegrator::exponentialRosenBrockEuler<CompositeModel>(curQ_, curVel_, params_.timeStep, massVec, *(static_cast<CompositeModel*>(model_.get())), posNew, velNew);
		break;

	case SimParameters::TI_IMPLICIT_EULER:
		std::cout << "Time integration: Implicit Euler." << std::endl;
		TimeIntegrator::implicitEuler<CompositeModel>(curQ_, curVel_, params_.timeStep, massVec, *(static_cast<CompositeModel*>(model_.get())), posNew, velNew);
		break;

	case SimParameters::TI_IMPLICIT_MIDPOINT:
		std::cout << "Time integration: Midpoint." << std::endl;
		TimeIntegrator::implicitMidPoint<CompositeModel>(curQ_, curVel_, params_.timeStep, massVec, *(static_cast<CompositeModel*>(model_.get())), posNew, velNew);
		break;

	case SimParameters::TI_TRAPEZOID:
		std::cout << "Time integration: Trapezoid." << std::endl;
		TimeIntegrator::trapezoid<CompositeModel>(curQ_, curVel_, params_.timeStep, massVec, *(static_cast<CompositeModel*>(model_.get())), posNew, velNew);
		break;

	case SimParameters::TI_TR_BDF2:
		std::cout << "Time integration: TRBDF2." << std::endl;
		TimeIntegrator::trapezoidBDF2<CompositeModel>(curQ_, curVel_, params_.timeStep, massVec, *(static_cast<CompositeModel*>(model_.get())), posNew, velNew);
		break;

	case SimParameters::TI_BDF2:
		if (time_ == 0)
		{
			std::cout << "Time integration: BDF2." << std::endl;
			TimeIntegrator::implicitEuler<CompositeModel>(curQ_, curVel_, params_.timeStep, massVec, *(static_cast<CompositeModel*>(model_.get())), posNew, velNew);
		}

		else
		{
			std::cout << "Time integration: BDF2." << std::endl;
			TimeIntegrator::BDF2<CompositeModel>(curQ_, curVel_, preQ_, preVel_, params_.timeStep, massVec, *(static_cast<CompositeModel*>(model_.get())), posNew, velNew);
		}

		break;

	case SimParameters::TI_NEWMARK:
		std::cout << "Time integration: Newmark." << std::endl;
		TimeIntegrator::Newmark<CompositeModel>(curQ_, curVel_, params_.timeStep, massVec, *(static_cast<CompositeModel*>(model_.get())), posNew, velNew, params_.NM_gamma, params_.NM_beta);
		break;

	case SimParameters::TI_ADDITIVE:
		std::cout << "Time integration: Additive." << std::endl;
		TimeIntegrator::additiveScheme<CompositeModel>(curQ_, curVel_, params_.timeStep, massVec, *(static_cast<CompositeModel*>(model_.get())), posNew, velNew, initialEnergy_);
		break;
	case SimParameters::TI_SPLIT:
		std::cout << "Time integration: Split, split ratio = " << params_.splitRatio << std::endl;
		TimeIntegrator::splitScheme<CompositeModel>(curQ_, curVel_, params_.timeStep, massVec, *(static_cast<CompositeModel*>(model_.get())), posNew, velNew, params_.splitRatio, initialEnergy_);
		break;
	case SimParameters::TI_COMPOSITE:
		std::cout << "Time integration: Composite " << std::endl;
		TimeIntegrator::compositeScheme<CompositeModel>(curQ_, curVel_, params_.timeStep, massVec, *(static_cast<CompositeModel*>(model_.get())), posNew, velNew, params_.rho);
		break;
	case SimParameters::TI_PROJECTION:
		std::cout << "Time integration: FEPR" << std::endl;
		TimeIntegrator::FEPR<CompositeModel>(curQ_, curVel_, params_.timeStep, massVec, *(static_cast<CompositeModel*>(model_.get())), posNew, velNew);
		break;
	}
	//update configuration into particle data structure
	preQ_ = curQ_;
	preVel_ = curVel_;
	curQ_ = posNew;
	curVel_ = velNew;
	model_->convertVar2Pos(curQ_, curPos_);
	std::cout << "current Pos: " << curPos_.transpose() << std::endl;
	time_ += params_.timeStep;
	iterNum_ += 1;

	return false;
}

void CompositeElementsGui::computeEvecsEvalue(const Eigen::SparseMatrix<double>& K, const Eigen::SparseMatrix<double>& M, Eigen::MatrixXd& eigenVecs, Eigen::VectorXd& eigenValues)
{
	int numSpectras = K.rows();
	int halfNum = numSpectras / 2;
	Spectra::SparseSymMatProd<double> opK(K);
	Spectra::SparseCholesky<double> opM(M);

	Spectra::SymGEigsSolver<Spectra::SparseSymMatProd<double>, Spectra::SparseCholesky<double>, Spectra::GEigsMode::Cholesky> eigs(opK, opM, halfNum, numSpectras);

	eigs.init();
	eigs.compute(Spectra::SortRule::LargestMagn);

	int leftNum = numSpectras - halfNum;

	using OpType = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
	using BOpType = Spectra::SparseSymMatProd<double>;
	OpType op(K, M);
	BOpType Bop(M);

	Spectra::SymGEigsShiftSolver<OpType, BOpType, Spectra::GEigsMode::ShiftInvert> eigs1(op, Bop, leftNum, numSpectras, 0.0);
	eigs1.init();
	eigs1.compute(Spectra::SortRule::LargestMagn);

	if (eigs.info() == Spectra::CompInfo::Successful && eigs1.info() == Spectra::CompInfo::Successful)
	{
		eigenValues.resize(numSpectras);
		eigenValues.segment(0, halfNum) = eigs.eigenvalues();
		eigenValues.segment(halfNum, leftNum) = eigs1.eigenvalues();

		eigenVecs.resize(curQ_.size(), numSpectras);
		eigenVecs.block(0, 0, curQ_.size(), halfNum) = eigs.eigenvectors();
		eigenVecs.block(0, halfNum, curQ_.size(), leftNum) = eigs1.eigenvectors();
	}
	else
	{
		// super slow when size of q is large
		Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> ges;
		ges.compute(K.toDense(), M.toDense());
		eigenValues = ges.eigenvalues().real();
		eigenVecs = ges.eigenvectors().real();
	}
}

void CompositeElementsGui::computeDecompositeCoeff(const Eigen::SparseMatrix<double>& M, const Eigen::MatrixXd& eigenVecs, const Eigen::VectorXd& v, Eigen::VectorXd& coeff)
{
	// compute initial alphas and betas
	int numSpectras = eigenVecs.cols();

	coeff.resize(numSpectras);
	
	for (int i = 0; i < numSpectras; i++)
	{
		coeff[i] = eigenVecs.col(i).dot(M * v);
	}
}

void CompositeElementsGui::saveInfo()
{
	if (iterNum_ % saveFrame_ != 0)
		return;

	std::string statFileName = outputFolderPath_ + std::string("simulation_status.txt");
	std::ofstream sfs;

	if (params_.isSaveInfo)
	{
		if (time_)
			sfs.open(statFileName, std::ofstream::out | std::ofstream::app);
		else
			sfs.open(statFileName, std::ofstream::out);
	}

	double springPotential = model_->computeElasticPotential(curPos_);
	double gravityPotential = model_->computeGravityPotential(curPos_);
	if (!params_.gravityEnabled)
		gravityPotential = 0;
	double IPCbarier = model_->computeFloorBarrier(curPos_);
	if (!params_.floorEnabled)
		IPCbarier = 0;

	std::vector<Eigen::Triplet<double>> massTrip;
	Eigen::SparseMatrix<double> massMat(curVel_.size(), curVel_.size());

	Eigen::VectorXd massVec = model_->assembleMass(model_->massVec_);

	for (int i = 0; i < massVec.size(); i++)
		massTrip.push_back({ i, i , massVec(i) });
	massMat.setFromTriplets(massTrip.begin(), massTrip.end());

	double kineticEnergy = 0.5 * curVel_.transpose() * massMat * curVel_;
	double internalContact = 0;
	if (params_.internalContactEnabled)
		internalContact = model_->computeInternalBarrier(curPos_);

	if (params_.isSaveInfo)
		sfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << time_ << " " << springPotential << " " << gravityPotential << " " << model_->params_.barrierStiffness * IPCbarier << " " << internalContact * model_->params_.barrierStiffness << " " << kineticEnergy << " " << springPotential + gravityPotential + model_->params_.barrierStiffness * IPCbarier + kineticEnergy;
	if (params_.isSaveInfo)
		sfs << std::endl;


	std::string posFileName = outputFolderPath_ + std::string("pos_status.txt");
	std::ofstream pfs;

	if (params_.isSaveInfo)
	{
		if (time_)
			pfs.open(posFileName, std::ofstream::out | std::ofstream::app);
		else
			pfs.open(posFileName, std::ofstream::out);

		for (int i = 0; i < curPos_.rows(); i++)
		{
			if (i == 0)
				pfs << time_ << " ";
			pfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << curPos_(i);
			if (i != curPos_.rows() - 1)
				pfs << " ";
			else
				pfs << "\n";
		}
	}

	std::string vecFileName = outputFolderPath_ + std::string("velocity_status.txt");
	std::ofstream vfs;

	if (params_.isSaveInfo)
	{
		if (time_)
			vfs.open(vecFileName, std::ofstream::out | std::ofstream::app);
		else
			vfs.open(vecFileName, std::ofstream::out);

		for (int i = 0; i < curVel_.rows(); i++)
		{
			vfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << curVel_(i);
			if (i != curVel_.rows() - 1)
				vfs << " ";
			else
				vfs << "\n";
		}
	}

	if (params_.isSaveInfo)
	{
		Eigen::SparseMatrix<double> H;
		model_->computeHessian(curQ_, H);

		Eigen::VectorXd g, c, MinvC;
		model_->computeGradient(curQ_, g);
		c = g - H * curQ_;
		MinvC = c;

		massTrip.clear();
		for (int i = 0; i < massVec.size(); i++)
			MinvC(i) /= massVec(i);

		Eigen::VectorXd evals;
		Eigen::MatrixXd evecs;

		computeEvecsEvalue(H, massMat, evecs, evals);
		Eigen::VectorXd alpha, beta, cis;

		computeDecompositeCoeff(massMat, evecs, curQ_, alpha);
		computeDecompositeCoeff(massMat, evecs, curVel_, beta);
		computeDecompositeCoeff(massMat, evecs, c, cis);

		if (!std::filesystem::exists(outputFolderPath_ + "eigenmodes/"))
		{
			std::cout << "create directory: " << outputFolderPath_ + "eigenmodes/" << std::endl;
			if (!std::filesystem::create_directories(outputFolderPath_ + "eigenmodes/"))
			{
				std::cout << "create folder failed." << outputFolderPath_ + "eigenmodes/" << std::endl;
			}
		}

		std::string evecFileName = outputFolderPath_ + "eigenmodes/eigvec_" + std::to_string(iterNum_ / saveFrame_) + ".txt";
		std::ofstream evecfs(evecFileName);
		evecfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << evecs << std::endl;

		std::string evalFileName = outputFolderPath_ + "eigenmodes/eigval_" + std::to_string(iterNum_ / saveFrame_) + ".txt";
		std::ofstream evalfs(evalFileName);
		evalfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << evals << std::endl;


		std::string alphaFileName = outputFolderPath_ + "alpha_state.txt";
		std::ofstream alphafs;

		if (time_)
			alphafs.open(alphaFileName, std::ofstream::out | std::ofstream::app);
		else
			alphafs.open(alphaFileName, std::ofstream::out);
		for (int i = 0; i < alpha.rows(); i++)
		{
			if (i == 0)
				alphafs << time_ << " ";
			alphafs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << alpha(i);
			if (i != alpha.rows() - 1)
				alphafs << " ";
			else
				alphafs << "\n";
		}

		std::string betaFileName = outputFolderPath_ + "beta_state.txt";
		std::ofstream betafs;

		if (time_)
			betafs.open(betaFileName, std::ofstream::out | std::ofstream::app);
		else
			betafs.open(betaFileName, std::ofstream::out);
		for (int i = 0; i < beta.rows(); i++)
		{
			if (i == 0)
				betafs << time_ << " ";
			betafs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << beta(i);
			if (i != beta.rows() - 1)
				betafs << " ";
			else
				betafs << "\n";
		}


		std::string cvalsFileName = outputFolderPath_ + "cvalue_state.txt";
		std::ofstream cvalsfs;

		if (time_)
			cvalsfs.open(cvalsFileName, std::ofstream::out | std::ofstream::app);
		else
			cvalsfs.open(cvalsFileName, std::ofstream::out);
		for (int i = 0; i < cis.rows(); i++)
		{
			if (i == 0)
				cvalsfs << time_ << " ";
			cvalsfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << cis(i);
			if (i != cis.rows() - 1)
				cvalsfs << " ";
			else
				cvalsfs << "\n";
		}

		std::string modeEnergyFileName = outputFolderPath_ + "mode_energy_state.txt";
		std::ofstream modeEnergyfs;

		if (time_)
			modeEnergyfs.open(modeEnergyFileName, std::ofstream::out | std::ofstream::app);
		else
			modeEnergyfs.open(modeEnergyFileName, std::ofstream::out);

		for (int i = 0; i < cis.rows(); i++)
		{
			if (i == 0)
				modeEnergyfs << time_ << " ";
			double ei = evals(i) / 2 * alpha(i) * alpha(i) + alpha(i) * cis(i) + beta(i) * beta(i) / 2.0;
			modeEnergyfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << ei;
			if (i != cis.rows() - 1)
				modeEnergyfs << " ";
			else
				modeEnergyfs << "\n";
		}

	}
	

	std::cout << std::endl;

}