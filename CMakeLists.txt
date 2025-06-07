cmake_minimum_required(VERSION 3.10)
project(IncliArray)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

file(GLOB SOURCES "src/*.cpp")

add_executable(IncliArray main.cpp ${SOURCES})
