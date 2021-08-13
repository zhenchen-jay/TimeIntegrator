#pragma once

#include <deque>
#include <Eigen/Sparse>
#include <Eigen/StdVector>
#include <Eigen/SparseCholesky>

#include "../PhysicalModel/GooHook1d.h"
#include "../PhysicalModel/SimParameters.h"
#include "../PhysicalModel/SceneObjects.h"

#include "../include/Utils/GIF.h"

// We fixed the x coordinate 

class GooHook1dGui
{
public:
	GooHook1dGui()
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
	}

	void save(igl::opengl::glfw::Viewer viewer)
	{
		saveScreenshot(viewer, outputFolderPath_ + std::to_string(iterNum_) + ".png", 1.0, false, true); // save png
//		saveScreenshot(viewer, outputFolderPath_ + std::to_string(iterNum_) + ".gif", 0.5, true, false); // save gif
	}

	void updateParams()
	{
		model_.params_ = params_;
	}

	bool reachTheTermination()
	{
		return time_ >= totalTime_;
	}

	void printTime()
	{
		std::cout << "current time: " << time_ << std::endl;
	}

	void saveInfo(igl::opengl::glfw::Viewer& viewer, bool writePNG = true, bool writeGIF = true, int writeMesh = 1, double save_dt = 1e-2);
	void saveScreenshot(igl::opengl::glfw::Viewer& viewer, const std::string& filePath, double scale = 1.0, bool writeGIF = false, bool writePNG = true);
	void saveInfoForPresent(igl::opengl::glfw::Viewer& viewer, const std::string fileName = "info.txt", double save_dt = 1e-2);
	
private:
	SimParameters params_;

	Eigen::MatrixXd renderQ;
	Eigen::MatrixXi renderF;
	Eigen::MatrixXd renderC;

	GooHook1d model_;
	double GIFScale_;
	GifWriter GIFWriter_;
	uint32_t GIFDelay_;
	int GIFStep_;

	double totalTime_;
	int totalIterNum_;

public:
	bool isPaused_;
	std::string outputFolderPath_;
	double time_;
	int iterNum_;
};



