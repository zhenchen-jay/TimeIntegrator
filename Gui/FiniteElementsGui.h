#pragma once

#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/png/writePNG.h>

#include <deque>
#include <Eigen/Sparse>
#include <Eigen/StdVector>
#include <Eigen/SparseCholesky>

#include "../PhysicalModel/PhysicalModel.h"
#include "../PhysicalModel/LinearElements.h"
#include "../PhysicalModel/NeoHookean.h"
#include "../PhysicalModel/SimParameters.h"
#include "../PhysicalModel/SceneObjects.h"
#include "../SpectraAnalysis/SpectraAnalysisLinearElements.h"

#include "../include/Utils/GIF.h"

// We fixed the x coordinate 

class FiniteElementsGui
{
public:
	FiniteElementsGui()
	{
		time_ = 0;
		GIFScale_ = 0.6;
		GIFDelay_ = 10;
		GIFStep_ = 1;
		iterNum_ = 0;
	}

	void drawGUI(igl::opengl::glfw::imgui::ImGuiMenu& menu);
	void reset();

	void initSimulation();

	void updateRenderGeometry();

	bool simulateOneStep();

	void renderRenderGeometry(igl::opengl::glfw::Viewer& viewer)
	{
		viewer.data().clear();
		viewer.data().set_mesh(renderQ, renderF);
		viewer.data().set_colors(renderC);
		Eigen::RowVector3d edgeColor(0, 0, 0);
		viewer.data().add_edges(edgeStart, edgeEnd, edgeColor);
	}

	void updateParams()
	{
		model_->params_ = params_;
		getOutputFolderPath();
	}

	void getOutputFolderPath();

	bool reachTheTermination()
	{
		return time_ >= totalTime_;
	}

	void printTime()
	{
		std::cout << "current time: " << time_ << std::endl;
	}

	bool computePeriod(Eigen::VectorXd q, std::vector<double>& periods);

	void saveInfo();

private:
	Eigen::VectorXd curQ_;
	Eigen::VectorXd curVel_;

	Eigen::VectorXd curQTheo_;
	Eigen::VectorXd curVelTheo_;

	Eigen::VectorXd curQExact_;
	Eigen::VectorXd curVelExact_;

	Eigen::VectorXd preQ_;
	Eigen::VectorXd preVel_;

	Eigen::VectorXd preQTheo_;
	Eigen::VectorXd preVelTheo_;

	Eigen::VectorXd curPos_;
	Eigen::VectorXd curPosTheo_;
	Eigen::VectorXd exactPos_;
	Eigen::MatrixXi curF_;

	Eigen::MatrixXd renderQ;
	Eigen::MatrixXi renderF;
	Eigen::MatrixXd renderC;

	Eigen::MatrixXd edgeStart;
	Eigen::MatrixXd edgeEnd;


	std::shared_ptr<PhysicalModel> model_;
	SpectraAnalysisLinearElements theoModel_;
	double GIFScale_;
	GifWriter GIFWriter_;
	uint32_t GIFDelay_;
	int GIFStep_;

	double totalTime_;
	int totalIterNum_;

	std::vector<double> lastLens_;
	std::vector<double> lastPassRestPointTime_;

	bool saveperiodFile_;

public:
	bool isPaused_;
	bool isTheoretical_;
	std::string outputFolderPath_, baseFolder_;
	double time_;
	int iterNum_;
	SimParameters params_;
};



