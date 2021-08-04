project(test_Integrator)
file(GLOB SRCFILES *.cpp *.h ../include/*.h ../include/IntegrationScheme/*.h ../include/optimization/*.h)
add_executable(${PROJECT_NAME}_bin ${SRCFILES})