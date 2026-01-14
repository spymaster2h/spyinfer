#pragma once

#include <vector>
#include <mutex>
#include <stdexcept>

namespace spyinfer {

class BlockManager
{
public:
    BlockManager(int num_blocks);
    ~BlockManager() = default;

    void allocate(std::vector<int>& block_table);
    void free(const std::vector<int>& block_table);

    int get_num_free_blocks() const;

private:
    std::vector<int> free_blocks_;
    mutable std::mutex mutex_;
};

} // namespace spyinfer
