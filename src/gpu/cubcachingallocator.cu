#include <gpu/cubcachingallocator.cuh>
#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>

namespace care{
namespace gpu{

    // Caching allocator for device memory
    cub::CachingDeviceAllocator cubCachingAllocator(
                8,                                                  ///< Geometric growth factor for bin-sizes
                3,                                                  ///< Minimum bin (default is bin_growth ^ 1)
                cub::CachingDeviceAllocator::INVALID_BIN,           ///< Maximum bin (default is no max bin)
                cub::CachingDeviceAllocator::INVALID_SIZE,          ///< Maximum aggregate cached bytes per device (default is no limit)
                true,                                               ///< Whether or not to skip a call to \p FreeAllCached() when the destructor is called (default is to deallocate)
                false);                                             ///< Whether or not to print (de)allocation events to stdout (default is no stderr output)

}
}
