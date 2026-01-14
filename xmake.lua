add_rules("plugin.compile_commands.autoupdate", {outputdir = "build"})
add_rules("mode.debug", "mode.release")


add_requires("nlohmann_json")
add_requires("gtest")
add_requires("openblas")

add_includedirs("src")


target("main")
    set_kind("binary")
    set_languages("c++20") 
    add_files("main.cpp")  -- 你的源代码文件
    add_files("src/**.cpp")
    add_packages("nlohmann_json")  -- 引入 nlohmann/json 库
    add_cxxflags("-mavx512f", "-mavx512bw", "-mavx512dq")  -- 添加AVX512编译选项

-- 添加测试目标
target("test_add")
    set_kind("binary")
    set_languages("c++20")
    add_files("test/test_add.cpp")
    add_files("src/**.cpp")
    add_packages("gtest", "nlohmann_json", "openblas")
    add_cxxflags("-mavx512f", "-mavx512bw", "-mavx512dq")

target("test_linear")
    set_kind("binary")
    set_languages("c++20")
    add_files("test/test_linear.cpp")
    add_files("src/**.cpp")
    add_packages("gtest", "nlohmann_json", "openblas")
    add_cxxflags("-mavx512f", "-mavx512bw", "-mavx512dq")

target("test_swiglu")
    set_kind("binary")
    set_languages("c++20")
    add_files("test/test_swiglu.cpp")
    add_files("src/**.cpp")
    add_packages("gtest", "nlohmann_json", "openblas")
    add_cxxflags("-mavx512f", "-mavx512bw", "-mavx512dq")