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
	GooHook1dGui() : PhysicsHookGui() {}

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
	}

private:
	SimParameters params_;
	double time_;

	std::mutex message_mutex;
	std::deque<MouseClick> mouseClicks_;

	Eigen::MatrixXd renderQ;
	Eigen::MatrixXi renderF;
	Eigen::MatrixXd renderC;

	GooHook1d model_;
};



