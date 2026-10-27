cmake_minimum_required(VERSION 3.20)

project(quant_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
  quant_native
  src/binning.cpp
  src/bindings.cpp
  src/rank.cpp
  src/ols.cpp
  src/pearson.cpp
  src/standardize.cpp
  src/spearman.cpp
  src/stability.cpp
)

target_include_directories(
  quant_native
  PRIVATE
    include
    third_party/eigen
)

target_compile_definitions(
  quant_native
  PRIVATE
    EIGEN_MPL2_ONLY
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(
    quant_native
    PRIVATE
      -Wall
      -Wextra
      -Wpedantic
  )
endif()

install(
  TARGETS quant_native
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .
)
