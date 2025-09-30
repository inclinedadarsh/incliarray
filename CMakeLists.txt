cmake_minimum_required(VERSION 3.10)
project(IncliArray VERSION 2.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to build examples
option(BUILD_EXAMPLES "Build example programs" ON)

# Create the library
add_library(NDArray 
    src/NDArray.cpp
    src/utils.cpp
)

# Public include directory (only include/)
target_include_directories(NDArray
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src  # For utils.h
)

# Build examples if requested
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation rules
install(TARGETS NDArray
    EXPORT NDArrayTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ 
    DESTINATION include
)

install(EXPORT NDArrayTargets
    FILE NDArrayConfig.cmake
    DESTINATION lib/cmake/NDArray
)
