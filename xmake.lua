add_rules("plugin.compile_commands.autoupdate", {outputdir = ".vscode"})
add_rules("mode.debug", "mode.release")


add_requires("nlohmann_json")
add_requires("gtest")
add_requires("openblas")
add_requires("openmp")

add_includedirs("src/include")



target("main")
    set_kind("binary")
    set_languages("c++20") 
    add_files("main.cpp")  -- 你的源代码文件
    add_packages("nlohmann_json")  -- 引入 nlohmann/json 库
    add_packages("openblas")  -- 引入 OpenBLAS 库
    add_cxxflags("-mavx512f", "-mavx512bw", "-mavx512dq")  -- 添加AVX512编译选项
    add_packages("openmp")  -- 引入 OpenMP 库



target("test_gemv")
    set_kind("binary")
    set_languages("c++20")
    add_files("test/test_gemv.cpp")
    add_packages("gtest")
    add_packages("openblas")  -- 引入 OpenBLAS 库
    add_deps("main")
    add_cxxflags("-mavx512f", "-mavx512bw", "-mavx512dq")  -- 添加AVX512编译选项

target("test_gevm")
    set_kind("binary")
    set_languages("c++20")
    add_files("test/test_gevm.cpp")
    add_packages("gtest")
    add_packages("openblas")  -- 引入 OpenBLAS 库
    add_deps("main")
    add_cxxflags("-mavx512f", "-mavx512bw", "-mavx512dq")  -- 添加AVX512编译选项