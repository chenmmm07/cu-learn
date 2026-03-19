cmake_minimum_required(VERSION 3.18)
project(cu-learn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# CUDA 架构配置
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)
# 查找依赖
find_package(CUDAToolkit REQUIRED)

# 包含目录
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/home/chenmin/3rdparty/cutlass/include)

# 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# 自动编译 src/optimizations/ 下的所有 .cu 文件
file(GLOB_RECURSE opt_files "src/optimizations/*.cu")
foreach(cu_file ${opt_files})
    # 获取文件名（不含扩展名）
    get_filename_component(file_name ${cu_file} NAME_WE)
    get_filename_component(parent_dir ${cu_file} DIRECTORY)

    # 从目录名提取数字 (如 00_naive -> 00)
    get_filename_component(dir_name ${parent_dir} NAME)
    string(REGEX MATCH "^([0-9]+)" dir_num ${dir_name})

    # 使用 optXX 格式命名 (如 opt00, opt01, opt02...)
    set(target_name "opt${dir_num}")

    add_executable(${target_name} ${cu_file})
    target_link_libraries(${target_name} cublas)

    # 收集需要安装的目标
    list(APPEND install_targets ${target_name})
endforeach()

# 自动编译 src/benchmarks/ 下的所有 .cu 文件
file(GLOB_RECURSE bench_files "src/benchmarks/*.cu")
foreach(cu_file ${bench_files})
    get_filename_component(file_name ${cu_file} NAME_WE)
    get_filename_component(parent_dir ${cu_file} DIRECTORY)

    string(REPLACE "${PROJECT_SOURCE_DIR}/src/" "" rel_dir ${parent_dir})
    string(REPLACE "/" "_" rel_dir ${rel_dir})
    set(target_name "${rel_dir}_${file_name}")

    add_executable(${target_name} ${cu_file})
    target_link_libraries(${target_name} cublas)

    list(APPEND install_targets ${target_name})
endforeach()

# 自动编译 src/demo/ 下的所有 .cu 文件
file(GLOB_RECURSE demo_files "src/demo/*.cu")
foreach(cu_file ${demo_files})
    get_filename_component(file_name ${cu_file} NAME_WE)
    get_filename_component(parent_dir ${cu_file} DIRECTORY)

    string(REPLACE "${PROJECT_SOURCE_DIR}/src/" "" rel_dir ${parent_dir})
    string(REPLACE "/" "_" rel_dir ${rel_dir})
    set(target_name "${rel_dir}_${file_name}")

    add_executable(${target_name} ${cu_file})
    target_link_libraries(${target_name} cublas)

    list(APPEND install_targets ${target_name})
endforeach()

# 自动编译其他目录下的 .cu 文件
file(GLOB_RECURSE other_files "src/*/*.cu")
list(REMOVE_ITEM other_files ${opt_files} ${bench_files} ${demo_files})

foreach(cu_file ${other_files})
    get_filename_component(file_name ${cu_file} NAME_WE)
    get_filename_component(parent_dir ${cu_file} DIRECTORY)

    string(REPLACE "${PROJECT_SOURCE_DIR}/src/" "" rel_dir ${parent_dir})
    string(REPLACE "/" "_" rel_dir ${rel_dir})
    set(target_name "${rel_dir}_${file_name}")

    add_executable(${target_name} ${cu_file})
    target_link_libraries(${target_name} cublas)

    list(APPEND install_targets ${target_name})
endforeach()

# 安装所有编译的目标
install(TARGETS ${install_targets} DESTINATION bin)
install(FILES src/plot/plot.py DESTINATION bin/plot)
