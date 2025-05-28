cmake_minimum_required(VERSION 3.10)

project(PArUco VERSION 0.9.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(TBB REQUIRED)

add_library(paruco STATIC paruco.cpp)
target_include_directories(paruco INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(paruco PUBLIC ${OpenCV_LIBS} ${TBB_LIBS})
