cmake_minimum_required(VERSION 2.8.12)
project(meshoptimizer)

add_subdirectory(pybind11)
pybind11_add_module(meshoptimizer
    "src/meshoptimizer.h"
    "src/indexgenerator.cpp"
    "src/overdrawoptimizer.cpp"
    "src/vcacheoptimizer.cpp"
    "src/vfetchoptimizer.cpp"
    "src/module.cpp"
)
