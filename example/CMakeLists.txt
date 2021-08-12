project(test_Integrator)
file(GLOB SRCFILES *.cpp *.h ../include/*.h ../include/IntegrationScheme/*.h ../include/Optimization/*.h ../PhysicalModel/*.h ../PhysicalModel/*.cpp ../Gui/*.h ../Gui/*.cpp ../include/Utils/*.h ../include/Utils/*.hpp)
add_executable(${PROJECT_NAME}_bin ${SRCFILES})
target_link_libraries(${PROJECT_NAME}_bin igl::core igl::opengl_glfw igl::opengl_glfw_imgui igl::png)
