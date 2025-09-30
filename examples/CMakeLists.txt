add_executable(basic_scalars basic_scalars.cpp)
target_link_libraries(basic_scalars PRIVATE NDArray)

add_executable(basic_vectors basic_vectors.cpp)
target_link_libraries(basic_vectors PRIVATE NDArray)
