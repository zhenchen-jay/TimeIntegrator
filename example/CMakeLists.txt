project(test_ERE)
file(GLOB SRCFILES *.cpp *.h ../include()/*.h)
add_executable(${PROJECT_NAME}_bin ${SRCFILES})