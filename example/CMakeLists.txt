project(test_Integrator)
file(GLOB SRCFILES  
    *.cpp 
    ../PhysicalModel/*.cpp 
    ../Gui/*.cpp 
    ../src/Utils/*.cpp
    *.h 
    ../PhysicalModel/*.h
    ../Gui/*.h 
    ../include/IntegrationScheme/*.h 
    ../include/Optimization/*.h 
    ../include/Utils/*.h)

add_executable(${PROJECT_NAME}_bin ${SRCFILES})
target_link_libraries(${PROJECT_NAME}_bin igl::core igl::opengl_glfw igl::opengl_glfw_imgui igl::png)
