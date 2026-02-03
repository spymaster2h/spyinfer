#include "cpu_device.hpp"

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cstring>
#include <unistd.h>

namespace spyinfer {

void* CPUDevice::allocate(size_t size_bytes)
{
    if (size_bytes == 0) return nullptr;

    void* ptr = malloc(size_bytes);
    if (!ptr) throw std::bad_alloc();

    return ptr;
}

void CPUDevice::deallocate(void* ptr)
{
    if (ptr) free(ptr);
}

size_t CPUDevice::get_total_memory() const
{
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return static_cast<size_t>(pages) * static_cast<size_t>(page_size);
}

size_t CPUDevice::get_free_memory() const
{
    FILE* file = fopen("/proc/meminfo", "r");
    if (!file) throw std::runtime_error("Failed to open /proc/meminfo");

    char buffer[256];
    size_t free_mem = 0;

    while (fgets(buffer, sizeof(buffer), file))
    {
        if (strncmp(buffer, "MemFree:", 8) == 0)
        {
            sscanf(buffer, "MemFree: %lu kB", &free_mem);
            free_mem *= 1024;
            break;
        }
    }

    fclose(file);
    return free_mem;
}

} // namespace spyinfer
