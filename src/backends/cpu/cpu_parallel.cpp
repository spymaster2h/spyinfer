#include "cpu_parallel.hpp"

ParallelExecutor::ParallelExecutor(size_t num_threads, size_t thread_buffer_size)
    : start_barrier_(num_threads + 1),
      finish_barrier_(num_threads + 1),
      thread_buffer_size_(thread_buffer_size)
{
    size_t threads_to_create = num_threads;
    const auto core_count = std::thread::hardware_concurrency();

    if (threads_to_create == 0)
    {
        threads_to_create = core_count;
        if (threads_to_create == 0)
        {
            threads_to_create = 2; // Fallback
        }
    }

    std::cout << "Creating ParallelExecutor with " << threads_to_create << " threads on a system with " << core_count << " cores.\n";

    workers_.reserve(threads_to_create);
    for (size_t i = 0; i < threads_to_create; ++i)
    {
        workers_.emplace_back(&ParallelExecutor::worker_loop, this, i);

        // --- Set CPU Affinity ---
#if defined(__linux__)
        if (core_count > 0)
        {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i % core_count, &cpuset);

            pthread_t native_handle = workers_.back().native_handle();
            int rc = pthread_setaffinity_np(native_handle, sizeof(cpu_set_t), &cpuset);
            if (rc != 0)
            {
                std::cerr << "Warning: Error setting affinity for thread " << i << ". Code: " << rc << "\n";
            }
            else
            {
                std::cout << "  - Worker thread " << i << " pinned to CPU core " << (i % core_count) << ".\n";
            }
        }
#else
        if (i == 0)
        {
            std::cout << "Warning: CPU affinity is not implemented/supported on this platform.\n";
        }
#endif
    }
}

ParallelExecutor::~ParallelExecutor()
{
    std::cout << "Shutting down ParallelExecutor...\n";
    stop_ = true;

    // Wake up all threads one last time so they can check the stop_ flag and exit.
    // An empty task is fine since the loop will terminate upon seeing stop_ = true.
    task_ = [](size_t, std::vector<std::byte>&) {};
    start_barrier_.wait(); // Unblock workers from their wait

    for (std::thread& worker : workers_)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }
    std::cout << "ParallelExecutor shut down.\n";
}

void ParallelExecutor::worker_loop(size_t thread_index)
{
    thread_local std::vector<std::byte> thread_buffer;
    thread_buffer.reserve(thread_buffer_size_);
    while (!stop_)
    {
        // 1. Worker waits at the start barrier for the main thread to dispatch a task.
        start_barrier_.wait();

        if (stop_) break;

        // 2. Execute the task assigned by the main thread.
        try
        {
            task_(thread_index, thread_buffer);
        }
        catch (const std::exception& e)
        {
            std::cerr << "Exception in worker thread " << thread_index << ": " << e.what() << '\n';
        }
        catch (...)
        {
            std::cerr << "Unknown exception in worker thread " << thread_index << '\n';
        }

        // 3. Signal the finish barrier to indicate completion.
        finish_barrier_.wait();
    }
}

void ParallelExecutor::execute(const std::vector<std::function<void()>>& functions)
{
    if (functions.size() != get_thread_count())
    {
        throw std::invalid_argument("The number of functions must match the thread count.");
    }
    this->execute([&functions](size_t thread_index, std::vector<std::byte>& thread_buffer) { functions[thread_index](); });
}
