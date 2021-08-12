#pragma once

#include <deque>
#include <Eigen/Sparse>
#include <Eigen/StdVector>
#include <Eigen/SparseCholesky>

#include "PhysicsHookGui.h"
#include "../PhysicalModel/GooHook1d.h"
#include "../PhysicalModel/SimParameters.h"
#include "../PhysicalModel/SceneObjects.h"

// We fixed the x coordinate 

class GooHook1dGui : public PhysicsHookGui
{
public:
	struct MouseClick
	{
		double x;
		double y;
		SimParameters::ClickMode mode;
	};
public:
	GooHook1dGui() : PhysicsHookGui()
	{ 
		time_ = 0;
		GIFScale_ = 0.6;
		GIFDelay_ = 10;
		GIFStep_ = 1;
		iterNum_ = 0;
	}

	virtual void drawGUI(igl::opengl::glfw::imgui::ImGuiMenu& menu);

	virtual void initSimulation();

	virtual void mouseClicked(double x, double y, int button)
	{
		message_mutex.lock();
		{
			MouseClick mc;
			mc.x = x;
			mc.y = y;
			mc.mode = params_.clickMode;
			mouseClicks_.push_back(mc);
		}
		message_mutex.unlock();
	}

	virtual void updateRenderGeometry();

	virtual void tick();

	virtual bool simulateOneStep();

	virtual void renderRenderGeometry(igl::opengl::glfw::Viewer& viewer)
	{
		viewer.data().clear();
		viewer.data().set_mesh(renderQ, renderF);
		viewer.data().set_colors(renderC);
		saveScreenshot(viewer, outputFolderPath_ + std::to_string(iterNum_) + ".png", 1.0, false, true); // save png
		saveScreenshot(viewer, outputFolderPath_ + std::to_string(iterNum_) + ".png", 0.5, true, false); // save gif
	}

	void updateParams()
	{
		model_.params_ = params_;
	}

	void saveInfo(igl::opengl::glfw::Viewer& viewer, bool writePNG = true, bool writeGIF = true, int writeMesh = 1, double save_dt = 1e-2);
	void saveScreenshot(igl::opengl::glfw::Viewer& viewer, const std::string& filePath, double scale = 1.0, bool writeGIF = false, bool writePNG = true);
	void saveInfoForPresent(igl::opengl::glfw::Viewer& viewer, const std::string fileName = "info.txt", double save_dt = 1e-2);
	
private:
	SimParameters params_;
	double time_;
	int iterNum_;

	std::mutex message_mutex;
	std::deque<MouseClick> mouseClicks_;

	Eigen::MatrixXd renderQ;
	Eigen::MatrixXi renderF;
	Eigen::MatrixXd renderC;

	GooHook1d model_;
	double GIFScale_;
	GifWriter GIFWriter_;
	uint32_t GIFDelay_;
	int GIFStep_;
	std::string outputFolderPath_;
};



