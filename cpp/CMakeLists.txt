cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(omigamax_core omigamax_core.cpp board.cpp mcts_engine.cpp)

set_target_properties(omigamax_core PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_compile_definitions(omigamax_core PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES=1)

install(TARGETS omigamax_core LIBRARY DESTINATION .)
