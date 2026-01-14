#pragma once

#include <vector>
#include <thread>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <iostream>

#if defined(__linux__)
#include <pthread.h>
#endif


class Barrier
{
public:
    explicit Barrier(size_t count) : thread_count_(count), count_(count), generation_(0) {}

    ~Barrier() = default;

    void wait()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        size_t gen = generation_;
        if (--count_ == 0)
        {
            generation_++;
            count_ = thread_count_;
            condition_.notify_all();
        }
        else
        {
            condition_.wait(lock, [this, gen] { return gen != generation_; });
        }
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    const size_t thread_count_;
    size_t count_;
    size_t generation_;
};

class ParallelExecutor
{
public:
    explicit ParallelExecutor(size_t num_threads, size_t thread_buffer_size);
    ~ParallelExecutor();

    template <typename Func>
    void execute(Func&& func)
    {
        this->task_ = func;
        start_barrier_.wait();
        finish_barrier_.wait();
    }

    void execute(const std::vector<std::function<void()>>& functions);

    size_t get_thread_count() const { return workers_.size(); }

    ParallelExecutor(const ParallelExecutor&) = delete;
    ParallelExecutor& operator=(const ParallelExecutor&) = delete;

private:
    void worker_loop(size_t thread_index);
    std::vector<std::thread> workers_;
    std::function<void(size_t, std::vector<std::byte>&)> task_;

    Barrier start_barrier_;
    Barrier finish_barrier_;
    std::atomic<bool> stop_ = false;

    size_t thread_buffer_size_;
};
