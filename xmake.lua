add_rules("plugin.compile_commands.autoupdate", {outputdir = "build"})
add_rules("mode.debug", "mode.release")



add_requires("nlohmann_json")
add_requires("gtest")
add_requires("openblas")
add_includedirs("src")

set_languages("c++20") 


option("backend")
    set_showmenu(true)
    set_default('cpu')
    set_description("Enable backend support")
option_end()

if is_config("backend", "cuda") then
    add_requires("cuda", {system=true, configs={utils={"cublas", ...}}})
end



target("main")
    set_kind("binary")
    add_files("main.cpp")  -- 你的源代码文件
    if is_config("backend", "cuda") then
        add_defines("USE_CUDA")
        add_files("src/backends/cuda/**.cpp")
        add_files("src/backends/cuda/**.cu")
        add_packages("cuda",{public=true})
        add_cugencodes("native")
        add_ldflags("-lcublas",{force = true})
        add_ldflags("-lcublasLt",{force = true})
    end
    add_files("src/backends/cpu/**.cpp")
    add_files("src/core/**.cpp")
    add_files("src/engine/**.cpp")
    add_files("src/utils/**.cpp")
    add_packages("nlohmann_json")  -- 引入 nlohmann/json 库
    add_cxxflags("-mavx512f", "-mavx512bw", "-mavx512dq")  -- 添加AVX512编译选项

-- 添加测试目标


target("test_add")
    set_kind("binary")
    set_languages("c++20")
    add_files("test/test_add.cpp")
    add_files("src/**.cpp|backends/cuda/**.cpp")
    add_packages("gtest", "nlohmann_json", "openblas")
    add_cxxflags("-mavx512f", "-mavx512bw", "-mavx512dq")

target("test_linear")
    set_kind("binary")
    set_languages("c++20")
    add_files("test/test_linear.cpp")
    add_files("src/**.cpp|backends/cuda/**.cpp")
    add_packages("gtest", "nlohmann_json", "openblas")
    add_cxxflags("-mavx512f", "-mavx512bw", "-mavx512dq")

target("test_swiglu")
    set_kind("binary")
    set_languages("c++20")
    add_files("test/test_swiglu.cpp")
    add_files("src/**.cpp|backends/cuda/**.cpp")
    add_packages("gtest", "nlohmann_json", "openblas")
    add_cxxflags("-mavx512f", "-mavx512bw", "-mavx512dq")


if is_config("backend", "cuda") then
    target("test_add_cuda")
        set_kind("binary")
        set_languages("c++20")
        add_files("test/test_add_cuda.cu")
        add_files("src/**.cpp")
        add_files("src/**.cu")
        add_defines("USE_CUDA")
        add_cugencodes("native")
        add_packages("gtest", "nlohmann_json")
        add_cxxflags("-mavx512f", "-mavx512bw", "-mavx512dq")
        add_ldflags("-lcublas",{force = true})
        add_ldflags("-lcublasLt",{force = true})


    target("test_linear_cuda")
        set_kind("binary")
        set_languages("c++20")
        add_files("test/test_linear_cuda.cu")
        add_files("src/**.cpp")
        add_files("src/**.cu")
        add_defines("USE_CUDA")
        add_cugencodes("native")
        add_packages("gtest", "nlohmann_json")
        add_cxxflags("-mavx512f", "-mavx512bw", "-mavx512dq")
        add_ldflags("-lcublas",{force = true})
        add_ldflags("-lcublasLt",{force = true})


    target("test_performance_linear_kernel")
        set_kind("binary")
        set_languages("c++20")
        add_files("test/performance/test_linear_kernel.cu")
        add_files("src/**.cu")
        add_defines("USE_CUDA")
        add_cugencodes("native")
        add_ldflags("-lcublas",{force = true})
        add_ldflags("-lcublasLt",{force = true})

    target("test_performance_norm_kernel")
        set_kind("binary")
        set_languages("c++20")
        add_files("test/performance/test_rmsnorm_kernel.cu")
        add_files("src/**.cpp")
        add_files("src/**.cu")
        add_defines("USE_CUDA")
        add_cugencodes("native")
        add_packages("gtest", "nlohmann_json")
        add_cxxflags("-mavx512f", "-mavx512bw", "-mavx512dq")
        add_cuflags('-G')
        add_cuflags('-g')
        add_ldflags("-lcublas",{force = true})
        add_ldflags("-lcublasLt",{force = true})

    target("test_rms_norm_kernel")
        set_kind("binary")
        set_languages("c++20")
        add_files("test/test_rms_norm_cuda.cu")
        add_files("src/**.cpp")
        add_files("src/**.cu")
        add_defines("USE_CUDA")
        add_cugencodes("native")
        add_packages("gtest", "nlohmann_json")
        add_cxxflags("-mavx512f", "-mavx512bw", "-mavx512dq")
        add_ldflags("-lcublas",{force = true})
        add_ldflags("-lcublasLt",{force = true})
end

