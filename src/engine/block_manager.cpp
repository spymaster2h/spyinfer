#include "block_manager.hpp"

namespace spyinfer {

BlockManager::BlockManager(int num_blocks)
{
    if (num_blocks > 0)
    {
        free_blocks_.reserve(num_blocks);
        for (int i = 0; i < num_blocks; ++i)
        {
            free_blocks_.push_back(i);
        }
    }
}

void BlockManager::allocate(std::vector<int>& block_table)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_blocks_.empty())
    {
        throw std::runtime_error("Out of memory: No free blocks available.");
    }
    int block_id = free_blocks_.back();
    free_blocks_.pop_back();
    block_table.push_back(block_id);
}

void BlockManager::free(const std::vector<int>& block_table)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (int block_id : block_table)
    {
        free_blocks_.push_back(block_id);
    }
}

int BlockManager::get_num_free_blocks() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return free_blocks_.size();
}

} // namespace spyinfer
