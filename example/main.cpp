#include <igl/opengl/glfw/Viewer.h>
#include <thread>
#include <igl/unproject.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/colormap.h>
#include <igl/png/writePNG.h>

#include "../Gui/GooHook1dGui.h"
#include "../Gui/FiniteElementsGui.h"
#include "../Cli/GooHook1dCli.h"

std::shared_ptr<GooHook1dGui> hook = NULL;
std::shared_ptr<FiniteElementsGui> FEM = NULL;
std::shared_ptr<GooHook1dCli> hookCli = NULL;

bool isHookModel = false;

void toggleSimulation(igl::opengl::glfw::Viewer& viewer)
{
	if (!hook && !FEM)
		return;
	if (isHookModel)
	{
		if (!hook)
		{
			std::cerr << "error, hook pointer uninitialized" << std::endl;
			exit(1);
		}
		else
			hook->isPaused_ = !(hook->isPaused_);
	}
	else if (!FEM)
	{
		std::cerr << "error, FEM pointer uninitialized" << std::endl;
		exit(1);
	}
	else
	{
		FEM->isPaused_ = !(FEM->isPaused_);
	}

}

void resetSimulation(igl::opengl::glfw::Viewer& viewer)
{
	if (!hook && !FEM)
		return;
	if (isHookModel)
	{
		if (!hook)
		{
			std::cerr << "error, hook pointer uninitialized" << std::endl;
			exit(1);
		}
		else
		{
			hook->reset();
			hook->renderRenderGeometry(viewer);
		}
	}
	else if (!FEM)
	{
		std::cerr << "error, FEM pointer uninitialized" << std::endl;
		exit(1);
	}
	else
	{
		FEM->reset();
		FEM->renderRenderGeometry(viewer);
	}
}

void saveScreenshot(igl::opengl::glfw::Viewer& viewer, const std::string& filePath, double scale = 1.0)   // super slow
{
	viewer.data().point_size *= scale;

	int width = static_cast<int>(scale * (viewer.core().viewport[2] - viewer.core().viewport[0]));
	int height = static_cast<int>(scale * (viewer.core().viewport[3] - viewer.core().viewport[1]));

	// Allocate temporary buffers for image
	Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R(width, height);
	Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> G(width, height);
	Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> B(width, height);
	Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> A(width, height);

	// Draw the scene in the buffers
	viewer.core().draw_buffer(viewer.data(), false, R, G, B, A);

	igl::png::writePNG(R, G, B, A, filePath);
	viewer.data().point_size /= scale;
}

bool drawCallback(igl::opengl::glfw::Viewer &viewer)
{
	if (!hook && !FEM)
		return false;

	if (isHookModel)
	{
		if (!hook)
		{
			std::cerr << "error, hook pointer uninitialized" << std::endl;
			exit(1);
		}
		else
		{
			if (!hook->reachTheTermination() && !(hook->isPaused_))
			{
				hook->printTime();
				hook->saveInfo();
				hook->simulateOneStep();
				hook->updateRenderGeometry();
				hook->renderRenderGeometry(viewer);
				//      saveScreenshot(viewer, hook->outputFolderPath_, 1.0); // super slow (be careful)
			}
		}
		
	}
	else if (!FEM)
	{
		std::cerr << "error, FEM pointer uninitialized" << std::endl;
		exit(1);
	}
	else
	{
		if (!FEM->reachTheTermination() && !(FEM->isPaused_))
		{
			FEM->printTime();
			FEM->saveInfo();
			FEM->simulateOneStep();
			FEM->updateRenderGeometry();
			FEM->renderRenderGeometry(viewer);
		}
	}
	
	return false;
}

bool keyCallback(igl::opengl::glfw::Viewer& viewer, unsigned int key, int modifiers)
{
	if (key == ' ')
	{
		toggleSimulation(viewer);
		return true;
	}
	return false;
}


int main(int argc, char* argv[])
{
	bool offlineSimulation = false;
	std::cout << argc << std::endl;
	if (argc >= 2)
	{
		std::string offlineString = argv[1];
		if (offlineString == "offline")
			offlineSimulation = true;
		std::cout << argv[1] << std::endl;
	}

	if (offlineSimulation)
	{
		if (argc != 7 && argc != 8)
		{
			std::cerr << "Usage: ./test_integrator_bin offline (problem model) (integrator scheme) (time step) (total time) (IPC barrier stiffness) [output folder, default = ../output]\nproblem model: harmonic, pogo\nintegrator scheme: EE(Explcit Euler), RK4(Runge Kutta), VV(Velocity Verlet), EXP(Exponetial Rosenbrock Euler), IE(Implicity Euler), IM(Implicit Midpoint), TR(Trapezoid), TRBDF2(TRBDF2), BDF2(BDF2), NM(Newmark)" << std::endl;
			return 1;
		}
		std::string model = argv[2];
		std::string integrator = argv[3];

		double timeStep = std::strtod(argv[4], NULL);
		double totalTime = std::strtod(argv[5], NULL);
		double stiffness = std::strtod(argv[6], NULL);

		hookCli = std::make_shared<GooHook1dCli>();

		if (model == "harmonic")
			hookCli->params_.modelType = SimParameters::MT_HARMONIC_1D;
		else if (model == "pogo")
			hookCli->params_.modelType = SimParameters::MT_POGO_STICK;
		else
		{
			std::cerr << "incorrect model type." << std::endl;
			return 1;
		}

		if (integrator == "EE")
			hookCli->params_.integrator = SimParameters::TI_EXPLICIT_EULER;
		else if (integrator == "RK4")
			hookCli->params_.integrator = SimParameters::TI_RUNGE_KUTTA;
		else if (integrator == "VV")
			hookCli->params_.integrator = SimParameters::TI_VELOCITY_VERLET;
		else if (integrator == "EXP")
			hookCli->params_.integrator = SimParameters::TI_EXP_ROSENBROCK_EULER;
		else if (integrator == "IE")
			hookCli->params_.integrator = SimParameters::TI_IMPLICIT_EULER;
		else if (integrator == "IM")
			hookCli->params_.integrator = SimParameters::TI_IMPLICIT_MIDPOINT;
		else if (integrator == "TR")
			hookCli->params_.integrator = SimParameters::TI_TRAPEZOID;
		else if (integrator == "TRBDF2")
			hookCli->params_.integrator = SimParameters::TI_TR_BDF2;
		else if (integrator == "BDF2")
			hookCli->params_.integrator = SimParameters::TI_BDF2;
		else if (integrator == "NM")
			hookCli->params_.integrator = SimParameters::TI_NEWMARK;
		else
		{
			std::cerr << "incorrect integrator scheme." << std::endl;
			return 1;
		}
			

		hookCli->params_.timeStep = timeStep;
		hookCli->params_.barrierStiffness = stiffness;
		hookCli->params_.totalTime = totalTime;
		hookCli->params_.totalNumIter = totalTime / timeStep;

		if (argc == 8)
			hookCli->initSimulation(argv[7]);
		else
			hookCli->initSimulation();

		while(!hookCli->reachTheTermination())
		{
			std::cout << "current time: " << hookCli->time_ << std::endl;
			hookCli->saveInfo();
			hookCli->simulateOneStep();
		}

	}
	else
	{
		igl::opengl::glfw::Viewer viewer;
		if (isHookModel)
		{
			hook = std::make_shared<GooHook1dGui>();
			hook->reset();
			hook->renderRenderGeometry(viewer);
			viewer.data().set_face_based(false);
			viewer.core().camera_zoom = 2.10;

		}
		else
		{
			FEM = std::make_shared<FiniteElementsGui>();
			FEM->reset();
			FEM->renderRenderGeometry(viewer);
			viewer.data().set_face_based(false);
			viewer.core().camera_zoom = 0.5;
		}
		viewer.data().show_lines = false;
		viewer.core().background_color << 1.0f, 1.0f, 1.0f, 1.0f;
		viewer.core().orthographic = true;
		viewer.core().animation_max_fps = 60.0;


		viewer.core().is_animating = true;
		viewer.callback_key_pressed = keyCallback;
		viewer.callback_pre_draw = drawCallback;

		igl::opengl::glfw::imgui::ImGuiMenu menu;
		viewer.plugins.push_back(&menu);

		menu.callback_draw_viewer_menu = [&]()
		{
			if (ImGui::CollapsingHeader("Simulation Control", ImGuiTreeNodeFlags_DefaultOpen))
			{
				if (ImGui::Button("Run Sim", ImVec2(-1, 0)))
				{
					toggleSimulation(viewer);
				}
				if (ImGui::Button("Reset Sim", ImVec2(-1, 0)))
				{
					resetSimulation(viewer);
				}

			}
			if(isHookModel)
			    hook->drawGUI(menu);
			else
			    FEM->drawGUI(menu);
			return false;
		};

		viewer.launch();
	}
}


