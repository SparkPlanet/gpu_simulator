add_library(eda_task1_heterogeneous STATIC
    partition.cpp
)

target_include_directories(eda_task1_heterogeneous
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(eda_task1_heterogeneous PUBLIC eda_task1)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(eda_task1_heterogeneous PRIVATE
        -Wall -Wextra -Wpedantic)
endif()
