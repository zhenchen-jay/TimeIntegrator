
#include <igl/opengl/glfw/Viewer.h>
#include <thread>
#include <igl/unproject.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>

#include "../Gui/PhysicsHookGui.h"
#include "../Gui/GooHook1dGui.h"

static PhysicsHookGui *hook = NULL;

void toggleSimulation()
{
    if (!hook)
        return;

    if (hook->isPaused())
        hook->run();
    else
        hook->pause();
}

void resetSimulation()
{
    if (!hook)
        return;
//    static_cast<GooHook *> (hook)->reset();
    hook->reset();
}

bool drawCallback(igl::opengl::glfw::Viewer &viewer)
{
    if (!hook)
        return false;

    hook->render(viewer);
    return false;
}

bool keyCallback(igl::opengl::glfw::Viewer& viewer, unsigned int key, int modifiers)
{
    if (key == ' ')
    {
        toggleSimulation();
        return true;
    }
    return false;
}

bool mouseCallback(igl::opengl::glfw::Viewer& viewer, int button, int modifier)
{
    Eigen::Vector3f pos(viewer.down_mouse_x, viewer.down_mouse_y, 0);
    Eigen::Matrix4f model = viewer.core().view;
    Eigen::Vector3f unproj = igl::unproject(pos, model, viewer.core().proj, viewer.core().viewport);
    hook->mouseClicked(unproj[0], -unproj[1], button);
    
    return true;
}

bool mouseScroll(igl::opengl::glfw::Viewer& viewer, float delta)
{
    return true;
}


bool drawGUI(igl::opengl::glfw::imgui::ImGuiMenu &menu)
{
    if (ImGui::CollapsingHeader("Simulation Control", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::Button("Run/Pause Sim", ImVec2(-1, 0)))
        {
            toggleSimulation();
        }
        if (ImGui::Button("Reset Sim", ImVec2(-1, 0)))
        {
            resetSimulation();
        }
        
    }
    hook->drawGUI(menu);
    return false;
}

int main(int argc, char *argv[])
{
  igl::opengl::glfw::Viewer viewer;

  hook = new GooHook1dGui();
  hook->reset();
  viewer.core().background_color << 1.0f, 1.0f, 1.0f, 0.0f;
  viewer.core().orthographic = true;
  viewer.core().camera_zoom = 4.0;
  viewer.core().animation_max_fps = 60.0;
  viewer.data().show_lines = false;
  viewer.data().set_face_based(false);
  viewer.core().is_animating = true;
  viewer.callback_key_pressed = keyCallback;
  viewer.callback_pre_draw = drawCallback;
  viewer.callback_mouse_down = mouseCallback;
  viewer.callback_mouse_scroll = mouseScroll;

  igl::opengl::glfw::imgui::ImGuiMenu menu;
  igl::opengl::glfw::imgui::ImGuiMenu cusMenu;
  viewer.plugins.push_back(&menu);

  menu.callback_draw_viewer_menu = [&]() {drawGUI(menu); };
  //menu.callback_draw_custom_window = [&]() {drawGUI(cusMenu); };

  viewer.launch();
}


