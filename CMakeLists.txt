cmake_minimum_required(VERSION 3.24)

project(eda_gpu_spice VERSION 0.3.0 LANGUAGES C CXX)

option(EDA_GPU_BUILD_TESTS "Build tests" ON)
option(EDA_GPU_ENABLE_NGSPICE "Build the ngspice interface" ON)
option(EDA_GPU_ENABLE_CUDA "Build CUDA operators and reference adapters" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(NGSPICE_EXECUTABLE
    "${CMAKE_CURRENT_SOURCE_DIR}/build/tools/ngspice/bin/ngspice"
    CACHE FILEPATH "Project-local ngspice executable")

if(EDA_GPU_ENABLE_NGSPICE)
    if(NOT EXISTS "${NGSPICE_EXECUTABLE}")
        message(FATAL_ERROR "ngspice is missing; run ./tools/build_ngspice.sh")
    endif()
    execute_process(
        COMMAND "${NGSPICE_EXECUTABLE}" --version
        RESULT_VARIABLE NGSPICE_VERSION_RESULT
        OUTPUT_VARIABLE NGSPICE_VERSION_TEXT
        ERROR_VARIABLE NGSPICE_VERSION_ERROR
    )
    string(APPEND NGSPICE_VERSION_TEXT "${NGSPICE_VERSION_ERROR}")
    if(NOT NGSPICE_VERSION_RESULT EQUAL 0 OR
       NOT NGSPICE_VERSION_TEXT MATCHES "KLU Direct Linear Solver")
        message(FATAL_ERROR "project ngspice must be built with KLU")
    endif()
endif()

set(KLU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ngspice/src/maths/KLU")
add_library(ngspice_klu STATIC
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
    "${KLU_DIR}/klu_refactor.c"
    "${KLU_DIR}/klu_scale.c"
    "${KLU_DIR}/klu_solve.c"
    "${KLU_DIR}/klu_sort.c"
    "${KLU_DIR}/klu_tsolve.c"
)
target_include_directories(ngspice_klu PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ngspice/src/include"
    PRIVATE "${KLU_DIR}"
)
target_compile_definitions(ngspice_klu PRIVATE NDEBUG)
set_target_properties(ngspice_klu PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED YES)

add_library(eda_core STATIC
    core/matrix.cpp
    core/registry.cpp
    operators/cpu/klu_solver.cpp
)
target_include_directories(eda_core
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/core"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(eda_core PRIVATE ngspice_klu m)

if(EDA_GPU_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "EDA_GPU_ENABLE_CUDA=ON but no CUDA compiler was found")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86" CACHE STRING
            "CUDA architectures for project operators")
    endif()
    target_sources(eda_core PRIVATE
        bridge/gpu_hybrid_solver.cpp
        bridge/cuda_custom_adapter.cpp
        bridge/cuda_reference_adapter.cpp
        operators/cpu/klu_initialization.cpp
        operators/cuda/fixed_pattern_lu.cu
        operators/cuda/fixed_pattern_lu_kernels.cuh
        operators/cuda/fixed_pattern_lu_runtime.cuh
        operators/cuda/cusolverrf_reference.cu
    )
    target_compile_definitions(eda_core PRIVATE EDA_GPU_HAS_CUDA=1)
    target_link_libraries(eda_core PRIVATE CUDA::cudart CUDA::cusolver)
    set_target_properties(eda_core PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
    )
endif()

add_executable(solver_benchmark interface/solver_benchmark.cpp)
target_link_libraries(solver_benchmark PRIVATE eda_core)
target_compile_definitions(solver_benchmark PRIVATE
    EDA_GPU_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

if(EDA_GPU_ENABLE_NGSPICE)
    find_package(Threads REQUIRED)
    add_executable(ngspice_runner interface/ngspice_runner.cpp)
    target_link_libraries(ngspice_runner PRIVATE Threads::Threads)
    target_compile_definitions(ngspice_runner PRIVATE
        EDA_GPU_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
        EDA_GPU_NGSPICE="${NGSPICE_EXECUTABLE}"
    )
endif()

if(EDA_GPU_BUILD_TESTS)
    include(CTest)
    enable_testing()

    add_test(NAME solver_generated
        COMMAND solver_benchmark --solver klu --grid 8)
    add_test(NAME solver_circuit_case
        COMMAND solver_benchmark --solver klu --case asic_320k)
    set_tests_properties(solver_generated solver_circuit_case
        PROPERTIES LABELS "core;cpu;klu")

    if(EDA_GPU_ENABLE_CUDA)
        add_test(NAME solver_cuda_custom_generated
            COMMAND solver_benchmark --solver cuda
                    --grid 8)
        add_test(NAME solver_cuda_reference_generated
            COMMAND solver_benchmark --solver cuda-cusolverrf-reference
                    --grid 8)
        set_tests_properties(solver_cuda_custom_generated solver_cuda_reference_generated
            PROPERTIES LABELS "core;cuda;gpu")
    endif()

    if(EDA_GPU_ENABLE_NGSPICE)
        add_test(NAME ngspice_interface
            COMMAND ngspice_runner --case rc --timeout 10)
        set_tests_properties(ngspice_interface PROPERTIES LABELS "interface;ngspice")
    endif()
endif()
