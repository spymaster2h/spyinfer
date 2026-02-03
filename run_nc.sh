export CUDA_VISIBLE_DEVICES=0

sudo /usr/local/cuda/bin/ncu  --target-processes all  -o report /data/hangzhou6/dev/engine/spyinfer/build/linux/x86_64/release/test_performance_linear_kernel