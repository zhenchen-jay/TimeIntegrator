#pragma once
#include <filesystem>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/png/writePNG.h>

#include <deque>
#include <Eigen/Sparse>
#include <Eigen/StdVector>
#include <Eigen/SparseCholesky>

#include "../CompositeModel/CompositeModel.h"
#include "../CompositeModel/LiElement.h"
#include "../CompositeModel/StVKElement.h"
#include "../CompositeModel/NonlinearElement.h"
#include "../PhysicalModel/SimParameters.h"
#include "../PhysicalModel/SceneObjects.h"

#include "../include/Utils/GIF.h"

// We fixed the x coordinate 

class CompositeElementsGui
{
public:
	CompositeElementsGui()
	{
		time_ = 0;
		iterNum_ = 0;
		plotCompression_ = true;

		stiffYoungs_ = 1000;
		softYoungs_ = 1;
		nonlinearYoungs_ = 1;

		numNL_ = 1;
		numSoftLi_ = 1;
		numStiffLi_ = 1;

		NLEnlargeRatio_ = 0.5001;
		stiffEnlargeRatio_ = 1.2;
		softEnlargeRatio_ = 1.2;
		NLStiffRatio_ = 0.5;
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

		if (time_ > 0 && time_ < params_.totalTime)
		{
			std::string imagePath = outputFolderPath_ + "/imgs/";
			if (!std::filesystem::exists(imagePath))
			{
				std::cout << "create directory: " << imagePath << std::endl;
				if (!std::filesystem::create_directories(imagePath))
				{
					std::cout << "create folder failed." << imagePath << std::endl;
				}
			}

			// Allocate temporary buffers
			Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R(1280, 800);
			Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> G(1280, 800);
			Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> B(1280, 800);
			Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> A(1280, 800);

			// Draw the scene in the buffers
			viewer.core().draw_buffer(
				viewer.data(), false, R, G, B, A);

			// Save it to a PNG
			if (iterNum_ % saveFrame_ == 0)
				igl::png::writePNG(R, G, B, A, imagePath + "out_" + std::to_string(iterNum_ / saveFrame_) + ".png");
		}
	}

	void updateParams()
	{
		model_->params_ = params_;
		saveFrame_ = std::max(1, int(5e-3 / params_.timeStep));
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

	void saveInfo();
	void computeEvecsEvalue(const Eigen::SparseMatrix<double>& K, const Eigen::SparseMatrix<double>& M, Eigen::MatrixXd& eigenVecs, Eigen::VectorXd& eigenValues);
	void computeDecompositeCoeff(const Eigen::SparseMatrix<double>& M, const Eigen::MatrixXd& eigenVecs, const Eigen::VectorXd& v, Eigen::VectorXd& coeff);

private:
	Eigen::VectorXd curQ_;
	Eigen::VectorXd curVel_;

	Eigen::VectorXd preQ_;
	Eigen::VectorXd preVel_;

	Eigen::VectorXd curPos_;
	std::vector<std::shared_ptr<FiniteElement>> elements_;

	Eigen::MatrixXd renderQ;
	Eigen::MatrixXi renderF;
	Eigen::MatrixXd renderC;

	Eigen::MatrixXd edgeStart;
	Eigen::MatrixXd edgeEnd;


	std::shared_ptr<CompositeModel> model_;

	double totalTime_;
	int totalIterNum_;

	double stiffYoungs_;
	double softYoungs_;
	double nonlinearYoungs_;

	int numNL_;
	int numStiffLi_;
	int numSoftLi_;
	double initialEnergy_;
	Eigen::VectorXd initModeEnergy_;

	int saveFrame_;
	double NLEnlargeRatio_;
	double stiffEnlargeRatio_;
	double softEnlargeRatio_;
	double NLStiffRatio_;

public:
	bool isPaused_;
	std::string outputFolderPath_, baseFolder_;
	double time_;
	int iterNum_;
	SimParameters params_;
	bool plotCompression_;
};



