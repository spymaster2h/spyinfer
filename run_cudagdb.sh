export CUDA_VISIBLE_DEVICES=1
/usr/local/cuda-12.2/bin/cuda-gdb --args ./build/linux/x86_64/debug/main -m /data/hangzhou6/dev/model/Qwen3-0.6B -b cuda
#/usr/local/cuda-12.2/bin/cuda-gdb  ./build/linux/x86_64/debug/test_performance_norm_kernel