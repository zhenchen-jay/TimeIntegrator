
#include <igl/opengl/glfw/Viewer.h>
#include <thread>
#include <igl/unproject.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/colormap.h>
#include <igl/png/writePNG.h>

#include "../Gui/GooHook1dGui.h"

static GooHook1dGui *hook = NULL;

void toggleSimulation(igl::opengl::glfw::Viewer& viewer)
{
    if (!hook)
        return;
    hook->isPaused_ = !(hook->isPaused_);

}

void resetSimulation(igl::opengl::glfw::Viewer& viewer)
{
    if (!hook)
        return;
//    static_cast<GooHook *> (hook)->reset();
    hook->reset();
    hook->renderRenderGeometry(viewer);
}

bool drawCallback(igl::opengl::glfw::Viewer &viewer)
{
    if (!hook)
        return false;

    if (!hook->reachTheTermination() && !(hook->isPaused_))
    {
        hook->printTime();
        hook->simulateOneStep();
        hook->updateRenderGeometry();
        //
        //std::cout << "before set: \n" << viewer.data().V << std::endl;
        hook->renderRenderGeometry(viewer); 
        hook->save(viewer);
        //std::cout << "after set: \n" << viewer.data().V << std::endl;
        /*double scale = 1;
        int width = static_cast<int>(scale * (viewer.core().viewport[2] - viewer.core().viewport[0]));
        int height = static_cast<int>(scale * (viewer.core().viewport[3] - viewer.core().viewport[1]));*/

        // Allocate temporary buffers for image
       /* Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R(width, height);
        Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> G(width, height);
        Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> B(width, height);
        Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> A(width, height);*/

        // Draw the scene in the buffers
        /*viewer.core().draw_buffer(viewer.data(), false, R, G, B, A);
        igl::png::writePNG(R, G, B, A, hook->outputFolderPath_ + std::to_string(hook->iterNum_) + ".png");*/
        //std::cout << "after save: \n" << viewer.data().V << std::endl;
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


int main(int argc, char *argv[])
{
  igl::opengl::glfw::Viewer viewer;

  hook = new GooHook1dGui();
  hook->reset();
  hook->renderRenderGeometry(viewer);
  viewer.core().background_color << 1.0f, 1.0f, 1.0f, 1.0f;
  viewer.core().orthographic = true;
  viewer.core().camera_zoom = 2.15;
  viewer.core().animation_max_fps = 60.0;
  viewer.data().show_lines = false;
  viewer.data().set_face_based(false);
  viewer.core().is_animating = true;
  viewer.callback_key_pressed = keyCallback;
  viewer.callback_pre_draw = drawCallback;

  igl::opengl::glfw::imgui::ImGuiMenu menu;
  igl::opengl::glfw::imgui::ImGuiMenu cusMenu;
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
      hook->drawGUI(menu);
      return false;
  };
  viewer.launch();
}


