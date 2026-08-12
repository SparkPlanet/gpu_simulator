cmake_minimum_required(VERSION 3.24)

project(eda_gpu_task1 VERSION 1.0.0 LANGUAGES C CXX)

option(EDA_GPU_BUILD_TESTS "Build Task 1 infrastructure tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CPU baseline: use the same KLU sources that ship with the vendored ngspice.
# KLU is an implementation detail of the cpu-klu backend, not a public API.
set(KLU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ngspice/src/maths/KLU")
add_library(task1_klu STATIC
    "${KLU_DIR}/amd_1.c"
    "${KLU_DIR}/amd_2.c"
    "${KLU_DIR}/amd_aat.c"
    "${KLU_DIR}/amd_control.c"
    "${KLU_DIR}/amd_defaults.c"
    "${KLU_DIR}/amd_dump.c"
    "${KLU_DIR}/amd_global.c"
    "${KLU_DIR}/amd_info.c"
    "${KLU_DIR}/amd_order.c"
    "${KLU_DIR}/amd_postorder.c"
    "${KLU_DIR}/amd_post_tree.c"
    "${KLU_DIR}/amd_preprocess.c"
    "${KLU_DIR}/amd_valid.c"
    "${KLU_DIR}/btf_maxtrans.c"
    "${KLU_DIR}/btf_order.c"
    "${KLU_DIR}/btf_strongcomp.c"
    "${KLU_DIR}/colamd.c"
    "${KLU_DIR}/colamd_global.c"
    "${KLU_DIR}/klu.c"
    "${KLU_DIR}/klu_analyze.c"
    "${KLU_DIR}/klu_analyze_given.c"
    "${KLU_DIR}/klu_defaults.c"
    "${KLU_DIR}/klu_diagnostics.c"
    "${KLU_DIR}/klu_dump.c"
    "${KLU_DIR}/klu_extract.c"
    "${KLU_DIR}/klu_factor.c"
    "${KLU_DIR}/klu_free_numeric.c"
    "${KLU_DIR}/klu_free_symbolic.c"
    "${KLU_DIR}/klu_kernel.c"
    "${KLU_DIR}/klu_memory.c"
    "${KLU_DIR}/klu_multiply.c"
    "${KLU_DIR}/klu_scale.c"
    "${KLU_DIR}/klu_solve.c"
    "${KLU_DIR}/klu_sort.c"
    "${KLU_DIR}/klu_tsolve.c"
)
target_include_directories(task1_klu
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ngspice/src/include"
    PRIVATE "${KLU_DIR}"
)
target_compile_definitions(task1_klu PRIVATE NDEBUG)
set_target_properties(task1_klu PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED YES)

add_library(eda_task1 STATIC
    src/matrix.cpp
    src/analysis.cpp
    src/profiler.cpp
    src/executor.cpp
    src/task1.cpp
    src/report.cpp
    src/backend_registry.cpp
    backends/cpu_klu.cpp
    backends/reference_lu.cpp
)
target_include_directories(eda_task1
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(eda_task1 PRIVATE task1_klu m)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(eda_task1 PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(task1_benchmark benchmark/task1_benchmark.cpp)
target_link_libraries(task1_benchmark PRIVATE eda_task1)
target_compile_definitions(task1_benchmark PRIVATE
    EDA_GPU_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

if(EDA_GPU_BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_executable(task1_tests tests/task1_tests.cpp)
    target_link_libraries(task1_tests PRIVATE eda_task1)
    add_test(NAME task1_infrastructure COMMAND task1_tests)
    add_test(NAME task1_benchmark_smoke COMMAND task1_benchmark --backend cpu-klu --grid 8)
    add_test(NAME task1_reference_smoke
        COMMAND task1_benchmark --backend reference-lu --grid 8)
    set_tests_properties(task1_infrastructure PROPERTIES LABELS "task1;cpu")
    set_tests_properties(task1_benchmark_smoke PROPERTIES LABELS "task1;cpu;klu")
    set_tests_properties(task1_reference_smoke PROPERTIES LABELS "task1;cpu;reference")
endif()
