#ifndef READ_EXTENDER_GPU_HPP
#define READ_EXTENDER_GPU_HPP

#include <config.hpp>
#include <hpc_helpers.cuh>

#include <gpu/gpumsa.cuh>
#include <gpu/gpumsamanaged.cuh>
#include <gpu/kernels.hpp>
#include <gpu/kernellaunch.hpp>
#include <gpu/gpuminhasher.cuh>
#include <gpu/segmented_set_operations.cuh>
#include <gpu/cachingallocator.cuh>
#include <sequencehelpers.hpp>
#include <hostdevicefunctions.cuh>
#include <util.hpp>
#include <gpu/gpucpureadstorageadapter.cuh>
#include <gpu/gpucpuminhasheradapter.cuh>
#include <readextender_cpu.hpp>
#include <util_iterator.hpp>
#include <readextender_common.hpp>
#include <gpu/cubvector.cuh>
#include <gpu/cuda_block_select.cuh>
#include <mystringview.hpp>
#include <gpu/gpustringglueing.cuh>
#include <gpu/memcpykernel.cuh>
#include <gpu/cubwrappers.cuh>
#include <gpu/readextender_gpu_kernels.cuh>

#include <algorithm>
#include <vector>
#include <numeric>

#include <cub/cub.cuh>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>

#include <thrust/execution_policy.h>


#define DO_ONLY_REMOVE_MATE_IDS



#if 0
    #define DEBUGDEVICESYNC { \
        cudaDeviceSynchronize(); CUERR; \
    }

#else 
    #define DEBUGDEVICESYNC {}

#endif

namespace care{
namespace gpu{




struct GpuReadExtender{
    template<class T>
    using DeviceBuffer = helpers::SimpleAllocationDevice<T>;
    //using DeviceBuffer = helpers::SimpleAllocationPinnedHost<T>;

    template<class T>
    using PinnedBuffer = helpers::SimpleAllocationPinnedHost<T>;

    struct AnchorData{
        AnchorData(cub::CachingDeviceAllocator& cubAlloc_)
            : d_anchorSequencesLength(cubAlloc_),
            d_accumExtensionsLengths(cubAlloc_),
            d_subjectSequencesDataDecoded(cubAlloc_),
            d_anchorQualityScores(cubAlloc_),
            d_subjectSequencesData(cubAlloc_){

        }

        std::size_t encodedSequencePitchInInts = 0;
        std::size_t decodedSequencePitchInBytes = 0;
        std::size_t qualityPitchInBytes = 0;

        CachedDeviceUVector<int> d_anchorSequencesLength{};
        CachedDeviceUVector<int> d_accumExtensionsLengths{};
        CachedDeviceUVector<char> d_subjectSequencesDataDecoded{};
        CachedDeviceUVector<char> d_anchorQualityScores{};
        CachedDeviceUVector<unsigned int> d_subjectSequencesData{};
    };

    struct AnchorHashResult{
        AnchorHashResult(cub::CachingDeviceAllocator& cubAlloc_)
            : d_candidateReadIds(cubAlloc_),
            d_numCandidatesPerAnchor(cubAlloc_),
            d_numCandidatesPerAnchorPrefixSum(cubAlloc_){

        }

        void resizeUninitialized(std::size_t newsize, cudaStream_t stream){
            d_numCandidatesPerAnchor.resizeUninitialized(newsize, stream);
            d_numCandidatesPerAnchorPrefixSum.resizeUninitialized(newsize + 1, stream);
        }

        CachedDeviceUVector<read_number> d_candidateReadIds{};
        CachedDeviceUVector<int> d_numCandidatesPerAnchor{};
        CachedDeviceUVector<int> d_numCandidatesPerAnchorPrefixSum{};
    };

    struct RawExtendResult{
        bool noCandidates{};
        int numResults{};
        std::size_t outputpitch{};
        std::size_t decodedSequencePitchInBytes{};
        CudaEvent event{cudaEventDisableTiming};
        PinnedBuffer<int> h_tmp{1};
        PinnedBuffer<char> h_inputAnchorsDecoded{};
        PinnedBuffer<extension::AbortReason> h_gpuabortReasons{};
        PinnedBuffer<extension::ExtensionDirection> h_gpudirections{};
        PinnedBuffer<int> h_gpuiterations{};
        PinnedBuffer<read_number> h_gpuReadIds{};
        PinnedBuffer<read_number> h_gpuMateReadIds{};
        PinnedBuffer<int> h_gpuAnchorLengths{};
        PinnedBuffer<int> h_gpuMateLengths{};
        PinnedBuffer<float> h_gpugoodscores{};
        PinnedBuffer<bool> h_gpuMateHasBeenFound{};
        PinnedBuffer<bool> h_pairResultAnchorIsLR{};
        PinnedBuffer<char> h_pairResultSequences{};
        PinnedBuffer<char> h_pairResultQualities{};
        PinnedBuffer<int> h_pairResultLengths{};
        PinnedBuffer<int> h_pairResultRead1Begins{};
        PinnedBuffer<int> h_pairResultRead2Begins{};
        PinnedBuffer<bool> h_pairResultMateHasBeenFound{};
        PinnedBuffer<bool> h_pairResultMergedDifferentStrands{};
    };

    struct IterationConfig{
        int maxextensionPerStep{1};
        int minCoverageForExtension{1};
    };

    struct TaskData{
        template<class T>
        using HostVector = std::vector<T>;

        int deviceId = 0;
        std::size_t entries = 0;
        std::size_t reservedEntries = 0;
        std::size_t encodedSequencePitchInInts = 0;
        std::size_t decodedSequencePitchInBytes = 0;
        std::size_t qualityPitchInBytes = 0;
        cub::CachingDeviceAllocator* cubAllocator{};
        CachedDeviceUVector<bool> pairedEnd{};
        CachedDeviceUVector<bool> mateHasBeenFound{};
        CachedDeviceUVector<int> id{};
        CachedDeviceUVector<int> pairId{};
        CachedDeviceUVector<int> iteration{};
        CachedDeviceUVector<float> goodscore{};
        CachedDeviceUVector<read_number> myReadId{};
        CachedDeviceUVector<read_number> mateReadId{};
        CachedDeviceUVector<extension::AbortReason> abortReason{};
        CachedDeviceUVector<extension::ExtensionDirection> direction{};
        CachedDeviceUVector<unsigned int> inputEncodedMate{};
        CachedDeviceUVector<unsigned int> inputAnchorsEncoded{};
        CachedDeviceUVector<char> soainputdecodedMateRevC{};
        CachedDeviceUVector<char> soainputmateQualityScoresReversed{};
        CachedDeviceUVector<int> soainputmateLengths{};
        CachedDeviceUVector<char> soainputAnchorsDecoded{};
        CachedDeviceUVector<char> soainputAnchorQualities{};
        CachedDeviceUVector<int> soainputAnchorLengths{};
        CachedDeviceUVector<int> soatotalDecodedAnchorsLengths{};
        CachedDeviceUVector<char> soatotalDecodedAnchorsFlat{};
        CachedDeviceUVector<char> soatotalAnchorQualityScoresFlat{};
        CachedDeviceUVector<int> soatotalAnchorBeginInExtendedRead{};
        CachedDeviceUVector<int> soaNumIterationResultsPerTask{};
        CachedDeviceUVector<int> soaNumIterationResultsPerTaskPrefixSum{};

        CachedDeviceUVector<read_number> d_usedReadIds{};
        CachedDeviceUVector<int> d_numUsedReadIdsPerTask{};
        CachedDeviceUVector<int> d_numUsedReadIdsPerTaskPrefixSum{};

        CachedDeviceUVector<read_number> d_fullyUsedReadIds{};
        CachedDeviceUVector<int> d_numFullyUsedReadIdsPerTask{};
        CachedDeviceUVector<int> d_numFullyUsedReadIdsPerTaskPrefixSum{};




        void consistencyCheck(bool verbose = false) const{
            assert(size() == entries);
            assert(pairedEnd.size() == size());
            assert(mateHasBeenFound.size() == size());
            assert(id.size() == size());
            assert(pairId.size() == size());
            assert(iteration.size() == size());
            assert(goodscore.size() == size());
            assert(myReadId.size() == size());
            assert(mateReadId.size() == size());
            assert(abortReason.size() == size());
            assert(direction.size() == size());
            assert(soainputmateLengths.size() == size());
            assert(soainputAnchorLengths.size() == size());
            assert(soaNumIterationResultsPerTask.size() == size());
            assert(soaNumIterationResultsPerTaskPrefixSum.size() == size() + 1);

            assert(d_numUsedReadIdsPerTask.size() == size());
            assert(d_numUsedReadIdsPerTaskPrefixSum.size() == size() + 1);
            assert(d_numFullyUsedReadIdsPerTask.size() == size());
            assert(d_numFullyUsedReadIdsPerTaskPrefixSum.size() == size() + 1);

            if(verbose){
                cudaDeviceSynchronize(); CUERR;

                std::vector<int> nums(size());
                std::vector<int> numsPS(size()+1);
                cudaMemcpyAsync(nums.data(), d_numUsedReadIdsPerTask.data(), sizeof(int) * size(), D2H, 0); CUERR;
                cudaMemcpyAsync(numsPS.data(), d_numUsedReadIdsPerTaskPrefixSum.data(), sizeof(int) * (size()+1), D2H, 0); CUERR;
                cudaDeviceSynchronize(); CUERR;

                std::cerr << "used nums\n";
                std::copy(nums.begin(), nums.end(), std::ostream_iterator<int>(std::cerr, ","));
                std::cerr << "\n";

                std::cerr << "used numsPS\n";
                std::copy(numsPS.begin(), numsPS.end(), std::ostream_iterator<int>(std::cerr, ","));
                std::cerr << "\n";

                cudaMemcpyAsync(nums.data(), d_numFullyUsedReadIdsPerTask.data(), sizeof(int) * size(), D2H, 0); CUERR;
                cudaMemcpyAsync(numsPS.data(), d_numFullyUsedReadIdsPerTaskPrefixSum.data(), sizeof(int) * (size()+1), D2H, 0); CUERR;
                cudaDeviceSynchronize(); CUERR;

                std::cerr << "fully used nums\n";
                std::copy(nums.begin(), nums.end(), std::ostream_iterator<int>(std::cerr, ","));
                std::cerr << "\n";

                std::cerr << "fully used numsPS\n";
                std::copy(numsPS.begin(), numsPS.end(), std::ostream_iterator<int>(std::cerr, ","));
                std::cerr << "\n";

                cudaMemcpyAsync(nums.data(), soaNumIterationResultsPerTask.data(), sizeof(int) * size(), D2H, 0); CUERR;
                cudaMemcpyAsync(numsPS.data(), soaNumIterationResultsPerTaskPrefixSum.data(), sizeof(int) * (size()+1), D2H, 0); CUERR;
                cudaDeviceSynchronize(); CUERR;

                std::cerr << "iteration result nums\n";
                std::copy(nums.begin(), nums.end(), std::ostream_iterator<int>(std::cerr, ","));
                std::cerr << "\n";

                std::cerr << "iteration result numsPS\n";
                std::copy(numsPS.begin(), numsPS.end(), std::ostream_iterator<int>(std::cerr, ","));
                std::cerr << "\n";
            }

            #if 0
                cudaDeviceSynchronize(); CUERR;

                int numUsedIds = 0;
                int numFullyUsedIds = 0;
                cudaMemcpyAsync(&numUsedIds, d_numUsedReadIdsPerTaskPrefixSum.data() + size(), sizeof(int), D2H, 0); CUERR;
                cudaMemcpyAsync(&numFullyUsedIds, d_numFullyUsedReadIdsPerTaskPrefixSum.data() + size(), sizeof(int), D2H, 0); CUERR;
                cudaDeviceSynchronize(); CUERR;

                if(numUsedIds != int(d_usedReadIds.size())){
                    std::cerr << "numUsedIds " << numUsedIds << ", d_usedReadIds.size() " << d_usedReadIds.size() << "\n";
                }

                if(numFullyUsedIds != int(d_fullyUsedReadIds.size())){
                    std::cerr << "numFullyUsedIds " << numFullyUsedIds << ", d_fullyUsedReadIds.size() " << d_fullyUsedReadIds.size() << "\n";
                }

                assert(numUsedIds == int(d_usedReadIds.size()));
                assert(numFullyUsedIds == int(d_fullyUsedReadIds.size()));
            #endif
        }

        TaskData(cub::CachingDeviceAllocator& cubAlloc_) : TaskData(cubAlloc_, 0,0,0,0, (cudaStream_t)0) {}

        TaskData(
            cub::CachingDeviceAllocator& cubAlloc_, 
            int size, 
            std::size_t encodedSequencePitchInInts_, 
            std::size_t decodedSequencePitchInBytes_, 
            std::size_t qualityPitchInBytes_,
            cudaStream_t stream
        ) 
            : encodedSequencePitchInInts(encodedSequencePitchInInts_), 
                decodedSequencePitchInBytes(decodedSequencePitchInBytes_), 
                qualityPitchInBytes(qualityPitchInBytes_),
                cubAllocator(&cubAlloc_),
                pairedEnd{cubAlloc_},
                mateHasBeenFound{cubAlloc_},
                id{cubAlloc_},
                pairId{cubAlloc_},
                iteration{cubAlloc_},
                goodscore{cubAlloc_},
                myReadId{cubAlloc_},
                mateReadId{cubAlloc_},
                abortReason{cubAlloc_},
                direction{cubAlloc_},
                inputEncodedMate{cubAlloc_},
                inputAnchorsEncoded{cubAlloc_},
                soainputdecodedMateRevC{cubAlloc_},
                soainputmateQualityScoresReversed{cubAlloc_},
                soainputmateLengths{cubAlloc_},
                soainputAnchorsDecoded{cubAlloc_},
                soainputAnchorQualities{cubAlloc_},
                soainputAnchorLengths{cubAlloc_},
                soatotalDecodedAnchorsLengths{cubAlloc_},
                soatotalDecodedAnchorsFlat{cubAlloc_},
                soatotalAnchorQualityScoresFlat{cubAlloc_},
                soatotalAnchorBeginInExtendedRead{cubAlloc_},
                soaNumIterationResultsPerTask{cubAlloc_},
                soaNumIterationResultsPerTaskPrefixSum{cubAlloc_},
                d_usedReadIds{cubAlloc_},
                d_numUsedReadIdsPerTask{cubAlloc_},
                d_numUsedReadIdsPerTaskPrefixSum{cubAlloc_},
                d_fullyUsedReadIds{cubAlloc_},
                d_numFullyUsedReadIdsPerTask{cubAlloc_},
                d_numFullyUsedReadIdsPerTaskPrefixSum{cubAlloc_}
        {
            cudaGetDevice(&deviceId); CUERR;
            resize(size, stream);

            //std::cerr << "after construct\n";
            consistencyCheck();
        }

        std::size_t size() const noexcept{
            return entries;
        }

        std::size_t capacity() const noexcept{
            return reservedEntries;
        }

        void clear(cudaStream_t stream){
            pairedEnd.clear();
            mateHasBeenFound.clear();
            id.clear();
            pairId.clear();
            iteration.clear();
            goodscore.clear();
            myReadId.clear();
            mateReadId.clear();
            abortReason.clear();
            direction.clear();
            inputEncodedMate.clear();
            inputAnchorsEncoded.clear();
            soainputdecodedMateRevC.clear();
            soainputmateQualityScoresReversed.clear();
            soainputmateLengths.clear();
            soainputAnchorsDecoded.clear();
            soainputAnchorQualities.clear();
            soainputAnchorLengths.clear();
            soatotalDecodedAnchorsLengths.destroy();
            soatotalDecodedAnchorsFlat.destroy();
            soatotalAnchorQualityScoresFlat.destroy();
            soatotalAnchorBeginInExtendedRead.destroy();
            soaNumIterationResultsPerTask.clear();
            soaNumIterationResultsPerTaskPrefixSum.resizeUninitialized(1, stream);
            cudaMemsetAsync(soaNumIterationResultsPerTaskPrefixSum.data(), 0, sizeof(int), stream); CUERR;

            d_usedReadIds.destroy();
            d_numUsedReadIdsPerTask.clear();
            d_numUsedReadIdsPerTaskPrefixSum.resizeUninitialized(1, stream);
            cudaMemsetAsync(d_numUsedReadIdsPerTaskPrefixSum.data(), 0, sizeof(int), stream); CUERR;

            d_fullyUsedReadIds.destroy();
            d_numFullyUsedReadIdsPerTask.clear();
            d_numFullyUsedReadIdsPerTaskPrefixSum.resizeUninitialized(1, stream);
            cudaMemsetAsync(d_numFullyUsedReadIdsPerTaskPrefixSum.data(), 0, sizeof(int), stream); CUERR;

            entries = 0;
            //std::cerr << "after clear\n";
            consistencyCheck();
        }

        void reserve(std::size_t newsize, cudaStream_t stream){
            pairedEnd.reserve(newsize, stream);
            mateHasBeenFound.reserve(newsize, stream);
            id.reserve(newsize, stream);
            pairId.reserve(newsize, stream);
            iteration.reserve(newsize, stream);
            goodscore.reserve(newsize, stream);
            myReadId.reserve(newsize, stream);
            mateReadId.reserve(newsize, stream);
            abortReason.reserve(newsize, stream);
            direction.reserve(newsize, stream);
            inputEncodedMate.reserve(newsize * encodedSequencePitchInInts, stream);
            inputAnchorsEncoded.reserve(newsize * encodedSequencePitchInInts, stream);
            soainputdecodedMateRevC.reserve(newsize * decodedSequencePitchInBytes, stream);
            soainputmateQualityScoresReversed.reserve(newsize * qualityPitchInBytes, stream);
            soainputmateLengths.reserve(newsize, stream);
            soainputAnchorsDecoded.reserve(newsize * decodedSequencePitchInBytes, stream);
            soainputAnchorQualities.reserve(newsize * qualityPitchInBytes, stream);
            soainputAnchorLengths.reserve(newsize, stream);
            soaNumIterationResultsPerTask.reserve(newsize, stream);
            soaNumIterationResultsPerTaskPrefixSum.reserve(newsize + 1, stream);

            d_numUsedReadIdsPerTask.reserve(newsize, stream);
            d_numUsedReadIdsPerTaskPrefixSum.reserve(newsize + 1, stream);

            d_numFullyUsedReadIdsPerTask.reserve(newsize, stream);
            d_numFullyUsedReadIdsPerTaskPrefixSum.reserve(newsize + 1, stream);

            reservedEntries = newsize;

            //std::cerr << "after reserve\n";
            consistencyCheck();
        }

        void resize(std::size_t newsize, cudaStream_t stream){
            pairedEnd.resize(newsize, stream);
            mateHasBeenFound.resize(newsize, stream);
            id.resize(newsize, stream);
            pairId.resize(newsize, stream);
            iteration.resize(newsize, stream);
            goodscore.resize(newsize, stream);
            myReadId.resize(newsize, stream);
            mateReadId.resize(newsize, stream);
            abortReason.resize(newsize, stream);
            direction.resize(newsize, stream);
            inputEncodedMate.resize(newsize * encodedSequencePitchInInts, stream);
            inputAnchorsEncoded.resize(newsize * encodedSequencePitchInInts, stream);
            soainputdecodedMateRevC.resize(newsize * decodedSequencePitchInBytes, stream);
            soainputmateQualityScoresReversed.resize(newsize * qualityPitchInBytes, stream);
            soainputmateLengths.resize(newsize, stream);
            soainputAnchorsDecoded.resize(newsize * decodedSequencePitchInBytes, stream);
            soainputAnchorQualities.resize(newsize * qualityPitchInBytes, stream);
            soainputAnchorLengths.resize(newsize, stream);
            soaNumIterationResultsPerTask.resize(newsize, stream);
            soaNumIterationResultsPerTaskPrefixSum.resize(newsize + 1, stream);

            d_numUsedReadIdsPerTask.resize(newsize, stream);
            d_numUsedReadIdsPerTaskPrefixSum.resize(newsize + 1, stream);

            d_numFullyUsedReadIdsPerTask.resize(newsize, stream);
            d_numFullyUsedReadIdsPerTaskPrefixSum.resize(newsize + 1, stream);

            if(size() > 0){
                if(newsize > size()){

                    // readextendergpukernels::fillKernel<<<SDIV(newsize - size(), 128), 128, 0,stream>>>(
                    //     thrust::make_zip_iterator(thrust::make_tuple(
                    //         soaNumIterationResultsPerTask.data() + size(),
                    //         d_numUsedReadIdsPerTask.data() + size(),
                    //         d_numFullyUsedReadIdsPerTask.data() + size()
                    //     )),
                    //     newsize - size(),
                    //     thrust::make_tuple(
                    //         0,
                    //         0,
                    //         0
                    //     )
                    // ); CUERR;

                    //repeat last element of prefix sum in newly added elements. fill numbers with 0

                    helpers::lambda_kernel<<<SDIV(newsize - size(), 128), 128, 0, stream>>>(
                        [
                            soaNumIterationResultsPerTaskPrefixSum = soaNumIterationResultsPerTaskPrefixSum.data(),
                            d_numUsedReadIdsPerTaskPrefixSum = d_numUsedReadIdsPerTaskPrefixSum.data(),
                            d_numFullyUsedReadIdsPerTaskPrefixSum = d_numFullyUsedReadIdsPerTaskPrefixSum.data(),
                            soaNumIterationResultsPerTask = soaNumIterationResultsPerTask.data(),
                            d_numUsedReadIdsPerTask = d_numUsedReadIdsPerTask.data(),
                            d_numFullyUsedReadIdsPerTask = d_numFullyUsedReadIdsPerTask.data(),
                            size = size(),
                            newsize = newsize
                        ] __device__ (){
                            const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                            const int stride = blockDim.x * gridDim.x;

                            for(int i = tid; i < newsize - size; i += stride){
                                soaNumIterationResultsPerTaskPrefixSum[size + 1 + i] = soaNumIterationResultsPerTaskPrefixSum[size];
                                d_numUsedReadIdsPerTaskPrefixSum[size + 1 + i] = d_numUsedReadIdsPerTaskPrefixSum[size];
                                d_numFullyUsedReadIdsPerTaskPrefixSum[size + 1 + i] = d_numFullyUsedReadIdsPerTaskPrefixSum[size];

                                soaNumIterationResultsPerTask[size + i] = 0;
                                d_numUsedReadIdsPerTask[size + i] = 0;
                                d_numFullyUsedReadIdsPerTask[size + i] = 0;
                            }
                        }
                    ); CUERR;
                }
            }else{
                if(newsize > 0){
                    readextendergpukernels::fillKernel<<<SDIV(newsize, 128), 128, 0, stream>>>(
                        thrust::make_zip_iterator(thrust::make_tuple(
                            soaNumIterationResultsPerTask.data(),
                            d_numUsedReadIdsPerTask.data(),
                            d_numFullyUsedReadIdsPerTask.data()
                        )),
                        newsize,
                        thrust::make_tuple(
                            0,
                            0,
                            0
                        )
                    ); CUERR;
                }

                readextendergpukernels::fillKernel<<<SDIV(newsize+1, 128), 128, 0, stream>>>(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        soaNumIterationResultsPerTaskPrefixSum.data(),
                        d_numUsedReadIdsPerTaskPrefixSum.data(),
                        d_numFullyUsedReadIdsPerTaskPrefixSum.data()
                    )),
                    newsize+1,
                    thrust::make_tuple(
                        0,
                        0,
                        0
                    )
                ); CUERR;

            }

            entries = newsize;
            reservedEntries = std::max(entries, reservedEntries);

            //std::cerr << "after resize\n";
            consistencyCheck();
        }

        bool checkPitch(const TaskData& rhs) const noexcept{
            if(encodedSequencePitchInInts != rhs.encodedSequencePitchInInts) return false;
            if(decodedSequencePitchInBytes != rhs.decodedSequencePitchInBytes) return false;
            if(qualityPitchInBytes != rhs.qualityPitchInBytes) return false;
            return true;
        }

        void aggregateAnchorData(AnchorData& anchorData, cudaStream_t stream){

            anchorData.d_anchorSequencesLength.resizeUninitialized(size(), stream);
            anchorData.d_accumExtensionsLengths.resizeUninitialized(size(), stream);
            anchorData.d_subjectSequencesDataDecoded.resizeUninitialized(size() * decodedSequencePitchInBytes, stream);
            anchorData.d_anchorQualityScores.resizeUninitialized(size() * qualityPitchInBytes, stream);
            anchorData.d_subjectSequencesData.resizeUninitialized(size() * encodedSequencePitchInInts, stream);

            anchorData.encodedSequencePitchInInts = encodedSequencePitchInInts;
            anchorData.decodedSequencePitchInBytes = decodedSequencePitchInBytes;
            anchorData.qualityPitchInBytes = qualityPitchInBytes;

            //compact some data of tasks into contiguous buffers 
            const int threads = size() * 32;
            readextendergpukernels::updateWorkingSetFromTasksKernel<256,32><<<SDIV(threads, 256), 256, 0, stream>>>(
                size(),
                qualityPitchInBytes,
                decodedSequencePitchInBytes,
                soaNumIterationResultsPerTask.data(),
                soaNumIterationResultsPerTaskPrefixSum.data(),
                anchorData.d_accumExtensionsLengths.data(),
                anchorData.d_anchorSequencesLength.data(),
                anchorData.d_anchorQualityScores.data(),
                anchorData.d_subjectSequencesDataDecoded.data(),
                soatotalAnchorBeginInExtendedRead.data(),
                soatotalDecodedAnchorsLengths.data(),
                soainputAnchorLengths.data(),
                soatotalAnchorQualityScoresFlat.data(),
                soainputAnchorQualities.data(),
                soatotalDecodedAnchorsFlat.data(),
                soainputAnchorsDecoded.data()
            ); CUERR;

            readextendergpukernels::encodeSequencesTo2BitKernel<8>
            <<<SDIV(size(), (128 / 8)), 128, 0, stream>>>(
                anchorData.d_subjectSequencesData.data(),
                anchorData.d_subjectSequencesDataDecoded.data(),
                anchorData.d_anchorSequencesLength.data(),
                decodedSequencePitchInBytes,
                encodedSequencePitchInInts,
                size()
            ); CUERR;
        }

        void addTasks(
            int numReadPairs,
            // for the arrays, two consecutive numbers / sequences belong to same read pair
            const read_number* d_readpair_readIds,
            const int* d_readpair_readLengths,
            const unsigned int * d_readpair_sequences,
            const char* d_readpair_qualities,
            cudaStream_t stream
        ){
            if(numReadPairs == 0) return;

            const int numAdditionalTasks = 4 * numReadPairs;

            TaskData newGpuSoaTaskData(*cubAllocator, numAdditionalTasks, encodedSequencePitchInInts, decodedSequencePitchInBytes, qualityPitchInBytes, stream);

            readextendergpukernels::createGpuTaskData<128,8>
                <<<SDIV(numAdditionalTasks, (128 / 8)), 128, 0, stream>>>(
                numReadPairs,
                d_readpair_readIds,
                d_readpair_readLengths,
                d_readpair_sequences,
                d_readpair_qualities,
                newGpuSoaTaskData.pairedEnd.data(),
                newGpuSoaTaskData.mateHasBeenFound.data(),
                newGpuSoaTaskData.id.data(),
                newGpuSoaTaskData.pairId.data(),
                newGpuSoaTaskData.iteration.data(),
                newGpuSoaTaskData.goodscore.data(),
                newGpuSoaTaskData.myReadId.data(),
                newGpuSoaTaskData.mateReadId.data(),
                newGpuSoaTaskData.abortReason.data(),
                newGpuSoaTaskData.direction.data(),
                newGpuSoaTaskData.inputEncodedMate.data(),
                newGpuSoaTaskData.soainputdecodedMateRevC.data(),
                newGpuSoaTaskData.soainputmateQualityScoresReversed.data(),
                newGpuSoaTaskData.soainputmateLengths.data(),
                newGpuSoaTaskData.inputAnchorsEncoded.data(),
                newGpuSoaTaskData.soainputAnchorsDecoded.data(),
                newGpuSoaTaskData.soainputAnchorQualities.data(),
                newGpuSoaTaskData.soainputAnchorLengths.data(),
                newGpuSoaTaskData.soaNumIterationResultsPerTask.data(),
                newGpuSoaTaskData.soaNumIterationResultsPerTaskPrefixSum.data(),
                decodedSequencePitchInBytes,
                qualityPitchInBytes,
                encodedSequencePitchInInts
            ); CUERR;

            append(newGpuSoaTaskData, stream);
        }

        void append(const TaskData& rhs, cudaStream_t stream){
            assert(checkPitch(rhs));

            nvtx::push_range("soa append", 7);

            //std::cerr << "append check self\n";
            consistencyCheck();

            //std::cerr << "append check rhs\n";
            rhs.consistencyCheck();

            //create new arrays, copy both old arrays into it, then swap            
            if(rhs.size() > 0){
                const int newsize = size() + rhs.size();
                
                CachedDeviceUVector<bool> newpairedEnd(newsize, stream, *cubAllocator);
                CachedDeviceUVector<bool> newmateHasBeenFound(newsize, stream, *cubAllocator);
                CachedDeviceUVector<int> newid(newsize, stream, *cubAllocator);
                CachedDeviceUVector<int> newpairId(newsize, stream, *cubAllocator);
                CachedDeviceUVector<int> newiteration(newsize, stream, *cubAllocator);
                CachedDeviceUVector<float> newgoodscore(newsize, stream, *cubAllocator);
                CachedDeviceUVector<read_number> newmyReadId(newsize, stream, *cubAllocator);
                CachedDeviceUVector<read_number> newmateReadId(newsize, stream, *cubAllocator);
                CachedDeviceUVector<extension::AbortReason> newabortReason(newsize, stream, *cubAllocator);
                CachedDeviceUVector<extension::ExtensionDirection> newdirection(newsize, stream, *cubAllocator);
                CachedDeviceUVector<int> newsoainputmateLengths(newsize, stream, *cubAllocator);
                CachedDeviceUVector<int> newsoainputAnchorLengths(newsize, stream, *cubAllocator);
                CachedDeviceUVector<int> newsoaNumIterationResultsPerTask(newsize, stream, *cubAllocator);
                CachedDeviceUVector<int> newnumUsedReadidsPerTask(newsize, stream, *cubAllocator);
                CachedDeviceUVector<int> newnumFullyUsedReadidsPerTask(newsize, stream, *cubAllocator);


                helpers::call_copy_n_kernel(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        pairedEnd.data(),
                        mateHasBeenFound.data(),
                        id.data(),
                        pairId.data(),
                        iteration.data(),
                        goodscore.data(),
                        myReadId.data(),
                        mateReadId.data()
                    )),
                    size(),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        newpairedEnd.data(),
                        newmateHasBeenFound.data(),
                        newid.data(),
                        newpairId.data(),
                        newiteration.data(),
                        newgoodscore.data(),
                        newmyReadId.data(),
                        newmateReadId.data()
                    )),
                    stream
                );

                helpers::call_copy_n_kernel(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        rhs.pairedEnd.data(),
                        rhs.mateHasBeenFound.data(),
                        rhs.id.data(),
                        rhs.pairId.data(),
                        rhs.iteration.data(),
                        rhs.goodscore.data(),
                        rhs.myReadId.data(),
                        rhs.mateReadId.data()
                    )),
                    rhs.size(),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        newpairedEnd.data() + size(),
                        newmateHasBeenFound.data() + size(),
                        newid.data() + size(),
                        newpairId.data() + size(),
                        newiteration.data() + size(),
                        newgoodscore.data() + size(),
                        newmyReadId.data() + size(),
                        newmateReadId.data() + size()
                    )),
                    stream
                );

                helpers::call_copy_n_kernel(
                    thrust::make_zip_iterator(thrust::make_tuple(                    
                        abortReason.data(),
                        direction.data(),
                        soainputmateLengths.data(),
                        soainputAnchorLengths.data(),
                        soaNumIterationResultsPerTask.data(),
                        d_numUsedReadIdsPerTask.data(),
                        d_numFullyUsedReadIdsPerTask.data()
                    )),
                    size(),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        newabortReason.data(),
                        newdirection.data(),
                        newsoainputmateLengths.data(),
                        newsoainputAnchorLengths.data(),
                        newsoaNumIterationResultsPerTask.data(),
                        newnumUsedReadidsPerTask.data(),
                        newnumFullyUsedReadidsPerTask.data()
                    )),
                    stream
                );                

                helpers::call_copy_n_kernel(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        rhs.abortReason.data(),
                        rhs.direction.data(),
                        rhs.soainputmateLengths.data(),
                        rhs.soainputAnchorLengths.data(),
                        rhs.soaNumIterationResultsPerTask.data(),
                        rhs.d_numUsedReadIdsPerTask.data(),
                        rhs.d_numFullyUsedReadIdsPerTask.data()
                    )),
                    rhs.size(),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        newabortReason.data() + size(),
                        newdirection.data() + size(),
                        newsoainputmateLengths.data() + size(),
                        newsoainputAnchorLengths.data() + size(),
                        newsoaNumIterationResultsPerTask.data() + size(),
                        newnumUsedReadidsPerTask.data() + size(),
                        newnumFullyUsedReadidsPerTask.data() + size()
                    )),
                    stream
                );

                std::swap(pairedEnd, newpairedEnd);
                std::swap(mateHasBeenFound, newmateHasBeenFound);
                std::swap(id, newid);
                std::swap(pairId, newpairId);
                std::swap(iteration, newiteration);
                std::swap(goodscore, newgoodscore);
                std::swap(myReadId, newmyReadId);
                std::swap(mateReadId, newmateReadId);
                std::swap(abortReason, newabortReason);
                std::swap(direction, newdirection);
                std::swap(soainputmateLengths, newsoainputmateLengths);
                std::swap(soainputAnchorLengths, newsoainputAnchorLengths);
                std::swap(soaNumIterationResultsPerTask, newsoaNumIterationResultsPerTask);
                std::swap(d_numUsedReadIdsPerTask, newnumUsedReadidsPerTask);
                std::swap(d_numFullyUsedReadIdsPerTask, newnumFullyUsedReadidsPerTask);

                newpairedEnd.destroy();
                newmateHasBeenFound.destroy();
                newid.destroy();
                newpairId.destroy();
                newiteration.destroy();
                newgoodscore.destroy();
                newmyReadId.destroy();
                newmateReadId.destroy();
                newabortReason.destroy();
                newdirection.destroy();
                newsoainputmateLengths.destroy();
                newsoainputAnchorLengths.destroy();
                newsoaNumIterationResultsPerTask.destroy();
                newnumUsedReadidsPerTask.destroy();
                newnumFullyUsedReadidsPerTask.destroy();

                CachedDeviceUVector<int> newsoaNumIterationResultsPerTaskPrefixSum(newsize + 1, stream, *cubAllocator);
                CachedDeviceUVector<int> newnumUsedReadidsPerTaskPrefixSum(newsize + 1, stream, *cubAllocator);
                CachedDeviceUVector<int> newnumFullyUsedReadidsPerTaskPrefixSum(newsize + 1, stream, *cubAllocator);

                helpers::call_copy_n_kernel(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        soaNumIterationResultsPerTaskPrefixSum.data(),
                        d_numUsedReadIdsPerTaskPrefixSum.data(),
                        d_numFullyUsedReadIdsPerTaskPrefixSum.data()
                    )),
                    size(),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        newsoaNumIterationResultsPerTaskPrefixSum.data(),
                        newnumUsedReadidsPerTaskPrefixSum.data(),
                        newnumFullyUsedReadidsPerTaskPrefixSum.data()
                    )),
                    stream
                );                

                helpers::call_copy_n_kernel(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        rhs.soaNumIterationResultsPerTaskPrefixSum.data(),
                        rhs.d_numUsedReadIdsPerTaskPrefixSum.data(),
                        rhs.d_numFullyUsedReadIdsPerTaskPrefixSum.data()
                    )),
                    rhs.size() + 1,
                    thrust::make_zip_iterator(thrust::make_tuple(
                        newsoaNumIterationResultsPerTaskPrefixSum.data() + size(),
                        newnumUsedReadidsPerTaskPrefixSum.data() + size(),
                        newnumFullyUsedReadidsPerTaskPrefixSum.data() + size()
                    )),
                    stream
                );

                std::swap(soaNumIterationResultsPerTaskPrefixSum, newsoaNumIterationResultsPerTaskPrefixSum);
                std::swap(d_numUsedReadIdsPerTaskPrefixSum, newnumUsedReadidsPerTaskPrefixSum);
                std::swap(d_numFullyUsedReadIdsPerTaskPrefixSum, newnumFullyUsedReadidsPerTaskPrefixSum);

                newnumUsedReadidsPerTaskPrefixSum.destroy();
                newsoaNumIterationResultsPerTaskPrefixSum.destroy();
                newnumFullyUsedReadidsPerTaskPrefixSum.destroy();


                CachedDeviceUVector<unsigned int> newinputEncodedMate(newsize * encodedSequencePitchInInts, stream, *cubAllocator);
                CachedDeviceUVector<unsigned int> newinputAnchorsEncoded(newsize * encodedSequencePitchInInts, stream, *cubAllocator);

                helpers::call_copy_n_kernel(
                    thrust::make_zip_iterator(thrust::make_tuple(                    
                        inputEncodedMate.data(),
                        inputAnchorsEncoded.data()
                    )),
                    size() * encodedSequencePitchInInts,
                    thrust::make_zip_iterator(thrust::make_tuple(
                        newinputEncodedMate.data(),
                        newinputAnchorsEncoded.data()
                    )),
                    stream
                );

                helpers::call_copy_n_kernel(
                    thrust::make_zip_iterator(thrust::make_tuple(                    
                        rhs.inputEncodedMate.data(),
                        rhs.inputAnchorsEncoded.data()
                    )),
                    rhs.size() * encodedSequencePitchInInts,
                    thrust::make_zip_iterator(thrust::make_tuple(
                        newinputEncodedMate.data() + size() * encodedSequencePitchInInts,
                        newinputAnchorsEncoded.data() + size() * encodedSequencePitchInInts
                    )),
                    stream
                );

                std::swap(inputEncodedMate, newinputEncodedMate);
                std::swap(inputAnchorsEncoded, newinputAnchorsEncoded);

                newinputEncodedMate.destroy();
                newinputAnchorsEncoded.destroy(); 

                assert(decodedSequencePitchInBytes % sizeof(int) == 0);

                CachedDeviceUVector<char> newsoainputdecodedMateRevC(newsize * decodedSequencePitchInBytes, stream, *cubAllocator);
                CachedDeviceUVector<char> newsoainputAnchorsDecoded(newsize * decodedSequencePitchInBytes, stream, *cubAllocator);

                helpers::call_copy_n_kernel(
                    thrust::make_zip_iterator(thrust::make_tuple(                    
                        (int*)soainputdecodedMateRevC.data(),
                        (int*)soainputAnchorsDecoded.data()
                    )),
                    size() * (decodedSequencePitchInBytes / sizeof(int)),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        (int*)newsoainputdecodedMateRevC.data(),
                        (int*)newsoainputAnchorsDecoded.data()
                    )),
                    stream
                );

                helpers::call_copy_n_kernel(
                    thrust::make_zip_iterator(thrust::make_tuple(                    
                        (int*)rhs.soainputdecodedMateRevC.data(),
                        (int*)rhs.soainputAnchorsDecoded.data()
                    )),
                    rhs.size() * (decodedSequencePitchInBytes / sizeof(int)),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        ((int*)newsoainputdecodedMateRevC.data()) + size() * (decodedSequencePitchInBytes / sizeof(int)),
                        ((int*)newsoainputAnchorsDecoded.data()) + size() * (decodedSequencePitchInBytes / sizeof(int))
                    )),
                    stream
                );

                std::swap(soainputdecodedMateRevC, newsoainputdecodedMateRevC);
                std::swap(soainputAnchorsDecoded, newsoainputAnchorsDecoded);

                newsoainputdecodedMateRevC.destroy();
                newsoainputAnchorsDecoded.destroy(); 

                assert(qualityPitchInBytes % sizeof(int) == 0);

                CachedDeviceUVector<char> newsoainputmateQualityScoresReversed(newsize * qualityPitchInBytes, stream, *cubAllocator);
                CachedDeviceUVector<char> newsoainputAnchorQualities(newsize * qualityPitchInBytes, stream, *cubAllocator);

                helpers::call_copy_n_kernel(
                    thrust::make_zip_iterator(thrust::make_tuple(                    
                        (int*)soainputmateQualityScoresReversed.data(),
                        (int*)soainputAnchorQualities.data()
                    )),
                    size() * (qualityPitchInBytes / sizeof(int)),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        (int*)newsoainputmateQualityScoresReversed.data(),
                        (int*)newsoainputAnchorQualities.data()
                    )),
                    stream
                );

                helpers::call_copy_n_kernel(
                    thrust::make_zip_iterator(thrust::make_tuple(                    
                        (int*)rhs.soainputmateQualityScoresReversed.data(),
                        (int*)rhs.soainputAnchorQualities.data()
                    )),
                    rhs.size() * (qualityPitchInBytes / sizeof(int)),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        ((int*)newsoainputmateQualityScoresReversed.data()) + size() * (qualityPitchInBytes / sizeof(int)),
                        ((int*)newsoainputAnchorQualities.data()) + size() * (qualityPitchInBytes / sizeof(int))
                    )),
                    stream
                );

                std::swap(soainputmateQualityScoresReversed, newsoainputmateQualityScoresReversed);
                std::swap(soainputAnchorQualities, newsoainputAnchorQualities);

                newsoainputmateQualityScoresReversed.destroy();
                newsoainputAnchorQualities.destroy(); 

                #if 0

                soatotalDecodedAnchorsLengths.append(rhs.soatotalDecodedAnchorsLengths.data(), rhs.soatotalDecodedAnchorsLengths.data() + rhs.soatotalDecodedAnchorsLengths.size(), stream);
                soatotalAnchorBeginInExtendedRead.append(rhs.soatotalAnchorBeginInExtendedRead.data(), rhs.soatotalAnchorBeginInExtendedRead.data() + rhs.soatotalAnchorBeginInExtendedRead.size(), stream);

                soatotalDecodedAnchorsFlat.append(rhs.soatotalDecodedAnchorsFlat.data(), rhs.soatotalDecodedAnchorsFlat.data() + rhs.soatotalDecodedAnchorsFlat.size(), stream);
                soatotalAnchorQualityScoresFlat.append(rhs.soatotalAnchorQualityScoresFlat.data(), rhs.soatotalAnchorQualityScoresFlat.data() + rhs.soatotalAnchorQualityScoresFlat.size(), stream);
            
                d_usedReadIds.append(rhs.d_usedReadIds.data(), rhs.d_usedReadIds.data() + rhs.d_usedReadIds.size(), stream);
                d_fullyUsedReadIds.append(rhs.d_fullyUsedReadIds.data(), rhs.d_fullyUsedReadIds.data() + rhs.d_fullyUsedReadIds.size(), stream);

                #else 

                CachedDeviceUVector<int> newsoatotalDecodedAnchorsLengths(soatotalDecodedAnchorsLengths.size() + rhs.soatotalDecodedAnchorsLengths.size(), stream, *cubAllocator);
                CachedDeviceUVector<int> newsoatotalAnchorBeginInExtendedRead(soatotalAnchorBeginInExtendedRead.size() + rhs.soatotalAnchorBeginInExtendedRead.size(), stream, *cubAllocator);

                CachedDeviceUVector<read_number> newd_usedReadIds(d_usedReadIds.size() + rhs.d_usedReadIds.size(), stream, *cubAllocator);
                CachedDeviceUVector<read_number> newd_fullyUsedReadIds(d_fullyUsedReadIds.size() + rhs.d_fullyUsedReadIds.size(), stream, *cubAllocator);

                std::size_t elements = 0;
                elements = std::max(elements, soatotalDecodedAnchorsLengths.size());
                elements = std::max(elements, rhs.soatotalDecodedAnchorsLengths.size());
                elements = std::max(elements, soatotalAnchorBeginInExtendedRead.size());
                elements = std::max(elements, rhs.soatotalAnchorBeginInExtendedRead.size());
                elements = std::max(elements, d_usedReadIds.size());
                elements = std::max(elements, rhs.d_usedReadIds.size());
                elements = std::max(elements, d_fullyUsedReadIds.size());
                elements = std::max(elements, rhs.d_fullyUsedReadIds.size());

                if(elements > 0){

                    using care::gpu::MemcpyParams;

                    auto memcpyParams1 = cuda::std::tuple_cat(
                        cuda::std::make_tuple(MemcpyParams(newsoatotalDecodedAnchorsLengths.data(), soatotalDecodedAnchorsLengths.data(), sizeof(int) * soatotalDecodedAnchorsLengths.size())),
                        cuda::std::make_tuple(MemcpyParams(newsoatotalDecodedAnchorsLengths.data() + soatotalDecodedAnchorsLengths.size(), rhs.soatotalDecodedAnchorsLengths.data(), sizeof(int) * rhs.soatotalDecodedAnchorsLengths.size())),
                        cuda::std::make_tuple(MemcpyParams(newsoatotalAnchorBeginInExtendedRead.data(), soatotalAnchorBeginInExtendedRead.data(), sizeof(int) * soatotalAnchorBeginInExtendedRead.size())),
                        cuda::std::make_tuple(MemcpyParams(newsoatotalAnchorBeginInExtendedRead.data() + soatotalAnchorBeginInExtendedRead.size(), rhs.soatotalAnchorBeginInExtendedRead.data(), sizeof(int) * rhs.soatotalAnchorBeginInExtendedRead.size())),
                        cuda::std::make_tuple(MemcpyParams(newd_usedReadIds.data(), d_usedReadIds.data(), sizeof(read_number) * d_usedReadIds.size())),
                        cuda::std::make_tuple(MemcpyParams(newd_usedReadIds.data() + d_usedReadIds.size(), rhs.d_usedReadIds.data(), sizeof(read_number) * rhs.d_usedReadIds.size())),
                        cuda::std::make_tuple(MemcpyParams(newd_fullyUsedReadIds.data(), d_fullyUsedReadIds.data(), sizeof(read_number) * d_fullyUsedReadIds.size())),
                        cuda::std::make_tuple(MemcpyParams(newd_fullyUsedReadIds.data() + d_fullyUsedReadIds.size(), rhs.d_fullyUsedReadIds.data(), sizeof(read_number) * rhs.d_fullyUsedReadIds.size()))
                    );

                    care::gpu::memcpyKernel<int><<<SDIV(elements, 256), 256, 0, stream>>>(memcpyParams1); CUERR;
                }

                std::swap(soatotalDecodedAnchorsLengths, newsoatotalDecodedAnchorsLengths);
                std::swap(soatotalAnchorBeginInExtendedRead, newsoatotalAnchorBeginInExtendedRead);
                std::swap(d_usedReadIds, newd_usedReadIds);
                std::swap(d_fullyUsedReadIds, newd_fullyUsedReadIds);

                newsoatotalDecodedAnchorsLengths.destroy();
                newsoatotalAnchorBeginInExtendedRead.destroy();
                newd_usedReadIds.destroy();
                newd_fullyUsedReadIds.destroy();

                soatotalDecodedAnchorsFlat.append(rhs.soatotalDecodedAnchorsFlat.data(), rhs.soatotalDecodedAnchorsFlat.data() + rhs.soatotalDecodedAnchorsFlat.size(), stream);
                soatotalAnchorQualityScoresFlat.append(rhs.soatotalAnchorQualityScoresFlat.data(), rhs.soatotalAnchorQualityScoresFlat.data() + rhs.soatotalAnchorQualityScoresFlat.size(), stream);

                

                #endif
            }


            if(rhs.size() > 0){
                //fix appended prefixsums
                readextendergpukernels::taskFixAppendedPrefixSumsKernel<128><<<SDIV(rhs.size()+1, 128), 128, 0, stream>>>(
                    soaNumIterationResultsPerTaskPrefixSum.data(),
                    d_numUsedReadIdsPerTaskPrefixSum.data(),
                    d_numFullyUsedReadIdsPerTaskPrefixSum.data(),
                    soaNumIterationResultsPerTask.data(),
                    d_numUsedReadIdsPerTask.data(),
                    d_numFullyUsedReadIdsPerTask.data(),
                    size(),
                    rhs.size()
                ); CUERR;

            }

            entries += rhs.size();
            reservedEntries = std::max(entries, reservedEntries);

            //std::cerr << "after append\n";
            consistencyCheck();

            nvtx::pop_range();
        }

        void iterationIsFinished(cudaStream_t stream){
            readextendergpukernels::taskIncrementIterationKernel<128><<<SDIV(size(), 128), 128, 0, stream>>>(
                size(),
                direction.data(),
                pairedEnd.data(),
                mateHasBeenFound.data(),
                pairId.data(),
                id.data(),
                abortReason.data(), 
                iteration.data()
            ); CUERR;
        }

        template<class FlagIter>
        TaskData select(FlagIter d_selectionFlags, cudaStream_t stream){

            nvtx::push_range("soa_select", 1);
            CachedDeviceUVector<int> positions(entries, stream, *cubAllocator);
            CachedDeviceUScalar<int> d_numSelected(1, stream, *cubAllocator);

            CubCallWrapper(*cubAllocator).cubSelectFlagged(
                thrust::make_counting_iterator(0),
                d_selectionFlags,
                positions.begin(),
                d_numSelected.data(),
                entries,
                stream
            );

            int numSelected = 0;
            cudaMemcpyAsync(&numSelected, d_numSelected.data(), sizeof(int), D2H, stream); CUERR;
            cudaStreamSynchronize(stream); CUERR;

            TaskData selection = gather(positions.begin(), positions.begin() + numSelected, stream);
            nvtx::pop_range();

            //std::cerr << "check selected\n";
            selection.consistencyCheck();

            return selection;
        }

        template<class MapIter>
        TaskData gather(MapIter d_mapBegin, MapIter d_mapEnd, cudaStream_t stream){
            ThrustCachingAllocator<char> thrustCachingAllocator1(deviceId, cubAllocator, stream);

            nvtx::push_range("soa_gather", 2);

            auto gathersize = thrust::distance(d_mapBegin, d_mapEnd);

            TaskData selection(*cubAllocator, gathersize, encodedSequencePitchInInts, decodedSequencePitchInBytes, qualityPitchInBytes, stream);

            
            auto inputScalars1Begin = thrust::make_zip_iterator(thrust::make_tuple(
                pairedEnd.begin(),
                mateHasBeenFound.begin(),
                id.begin(),
                pairId.begin(),
                iteration.begin(),
                goodscore.begin(),
                myReadId.begin()
            ));

            auto outputScalars1Begin = thrust::make_zip_iterator(thrust::make_tuple(
                selection.pairedEnd.begin(),
                selection.mateHasBeenFound.begin(),
                selection.id.begin(),
                selection.pairId.begin(),
                selection.iteration.begin(),
                selection.goodscore.begin(),
                selection.myReadId.begin()
            ));            

            helpers::call_compact_kernel_async(
                outputScalars1Begin,
                inputScalars1Begin,
                d_mapBegin,
                gathersize,
                stream
            );

            auto inputScalars2Begin = thrust::make_zip_iterator(thrust::make_tuple(
                mateReadId.begin(),
                abortReason.begin(),
                direction.begin(),
                soainputmateLengths.begin(),
                soainputAnchorLengths.begin()
            ));

            auto outputScalars2Begin = thrust::make_zip_iterator(thrust::make_tuple(
                selection.mateReadId.begin(),
                selection.abortReason.begin(),
                selection.direction.begin(),
                selection.soainputmateLengths.begin(),
                selection.soainputAnchorLengths.begin()
            ));

            helpers::call_compact_kernel_async(
                outputScalars2Begin,
                inputScalars2Begin,
                d_mapBegin,
                gathersize,
                stream
            );
     

            gatherSoaData(selection, d_mapBegin, d_mapEnd, stream);   

            nvtx::pop_range();

            //std::cerr << "check gather\n";

            selection.consistencyCheck();

            return selection;
        }

        template<class MapIter>
        void gatherSoaData(TaskData& selection, MapIter d_mapBegin, MapIter d_mapEnd, cudaStream_t stream){
            assert(checkPitch(selection));

            auto gathersize = thrust::distance(d_mapBegin, d_mapEnd);

            selection.soaNumIterationResultsPerTask.resize(gathersize, stream);
            selection.soaNumIterationResultsPerTaskPrefixSum.resize(gathersize + 1, stream);

            selection.d_numUsedReadIdsPerTask.resize(gathersize, stream);
            selection.d_numUsedReadIdsPerTaskPrefixSum.resize(gathersize + 1, stream);

            selection.d_numFullyUsedReadIdsPerTask.resize(gathersize, stream);
            selection.d_numFullyUsedReadIdsPerTaskPrefixSum.resize(gathersize + 1, stream);

            selection.soainputdecodedMateRevC.resize(gathersize * decodedSequencePitchInBytes, stream);
            selection.soainputmateQualityScoresReversed.resize(gathersize * qualityPitchInBytes, stream);
            selection.soainputAnchorsDecoded.resize(gathersize * decodedSequencePitchInBytes, stream);
            selection.soainputAnchorQualities.resize(gathersize * qualityPitchInBytes, stream);

            selection.inputEncodedMate.resize(gathersize * encodedSequencePitchInInts, stream);
            selection.inputAnchorsEncoded.resize(gathersize * encodedSequencePitchInInts, stream);

            helpers::call_compact_kernel_async(
                thrust::make_zip_iterator(thrust::make_tuple(
                    selection.soaNumIterationResultsPerTask.begin(),
                    selection.d_numUsedReadIdsPerTask.begin(),
                    selection.d_numFullyUsedReadIdsPerTask.begin()
                )),
                thrust::make_zip_iterator(thrust::make_tuple(
                    soaNumIterationResultsPerTask.begin(),
                    d_numUsedReadIdsPerTask.begin(),
                    d_numFullyUsedReadIdsPerTask.begin()
                )),
                d_mapBegin,
                gathersize,
                stream
            ); CUERR;

            CubCallWrapper(*cubAllocator).cubInclusiveSum(
                selection.soaNumIterationResultsPerTask.begin(),
                selection.soaNumIterationResultsPerTaskPrefixSum.begin() + 1,
                gathersize,
                stream
            );

            CubCallWrapper(*cubAllocator).cubInclusiveSum(
                selection.d_numUsedReadIdsPerTask.begin(),
                selection.d_numUsedReadIdsPerTaskPrefixSum.begin() + 1,
                gathersize,
                stream
            );

            CubCallWrapper(*cubAllocator).cubInclusiveSum(
                selection.d_numFullyUsedReadIdsPerTask.begin(),
                selection.d_numFullyUsedReadIdsPerTaskPrefixSum.begin() + 1,
                gathersize,
                stream
            );

            cudaMemsetAsync(
                selection.soaNumIterationResultsPerTaskPrefixSum.data(),
                0, 
                sizeof(int),
                stream
            ); CUERR;
            cudaMemsetAsync(
                selection.d_numUsedReadIdsPerTaskPrefixSum.data(),
                0, 
                sizeof(int),
                stream
            ); CUERR;
            cudaMemsetAsync(
                selection.d_numFullyUsedReadIdsPerTaskPrefixSum.data(),
                0, 
                sizeof(int),
                stream
            ); CUERR;

            if(gathersize > 0){

                int selectedNumIterationResults = 0;
                int selectedNumUsedIds = 0;
                int selectedNumFullyUsedIds = 0;
                cudaMemcpyAsync(&selectedNumIterationResults, selection.soaNumIterationResultsPerTaskPrefixSum.data() + gathersize, sizeof(int), D2H, stream); CUERR;
                cudaMemcpyAsync(&selectedNumUsedIds, selection.d_numUsedReadIdsPerTaskPrefixSum.data() + gathersize, sizeof(int), D2H, stream); CUERR;
                cudaMemcpyAsync(&selectedNumFullyUsedIds, selection.d_numFullyUsedReadIdsPerTaskPrefixSum.data() + gathersize, sizeof(int), D2H, stream); CUERR;
                cudaStreamSynchronize(stream); CUERR;


                selection.soatotalDecodedAnchorsLengths.resizeUninitialized(selectedNumIterationResults, stream);
                selection.soatotalAnchorBeginInExtendedRead.resizeUninitialized(selectedNumIterationResults, stream);
                selection.soatotalDecodedAnchorsFlat.resizeUninitialized(selectedNumIterationResults * decodedSequencePitchInBytes, stream);
                selection.soatotalAnchorQualityScoresFlat.resizeUninitialized(selectedNumIterationResults * qualityPitchInBytes, stream);

                selection.d_usedReadIds.resizeUninitialized(selectedNumUsedIds, stream);
                selection.d_fullyUsedReadIds.resizeUninitialized(selectedNumFullyUsedIds, stream);
            }else{
                selection.soatotalDecodedAnchorsLengths.resizeUninitialized(0, stream);
                selection.soatotalAnchorBeginInExtendedRead.resizeUninitialized(0, stream);
                selection.soatotalDecodedAnchorsFlat.resizeUninitialized(0 * decodedSequencePitchInBytes, stream);
                selection.soatotalAnchorQualityScoresFlat.resizeUninitialized(0 * qualityPitchInBytes, stream);

                selection.d_usedReadIds.resizeUninitialized(0, stream);
                selection.d_fullyUsedReadIds.resizeUninitialized(0, stream);

                return;
            }

            readextendergpukernels::taskGatherKernel1<128, 32><<<gathersize, 128, 0, stream>>>(
                d_mapBegin,
                d_mapEnd,
                gathersize,
                decodedSequencePitchInBytes,
                qualityPitchInBytes,
                encodedSequencePitchInInts,
                selection.soainputAnchorLengths.data(),
                soainputAnchorLengths.data(),
                selection.soainputAnchorQualities.data(),
                soainputAnchorQualities.data(),
                selection.soainputmateQualityScoresReversed.data(),
                soainputmateQualityScoresReversed.data(),
                selection.soainputAnchorsDecoded.data(),
                soainputAnchorsDecoded.data(),
                selection.soainputdecodedMateRevC.data(),
                soainputdecodedMateRevC.data(),
                selection.inputEncodedMate.data(),
                inputEncodedMate.data(),
                selection.inputAnchorsEncoded.data(),
                inputAnchorsEncoded.data()
            ); CUERR;


            readextendergpukernels::taskGatherKernel2<128,32><<<gathersize, 128, 0, stream>>>(
                d_mapBegin,
                d_mapEnd,
                gathersize,
                decodedSequencePitchInBytes,
                qualityPitchInBytes,
                selection.soaNumIterationResultsPerTaskPrefixSum.data(),
                soaNumIterationResultsPerTaskPrefixSum.data(),
                soaNumIterationResultsPerTask.data(),
                selection.soatotalDecodedAnchorsLengths.data(),
                soatotalDecodedAnchorsLengths.data(),
                selection.soatotalAnchorBeginInExtendedRead.data(),
                soatotalAnchorBeginInExtendedRead.data(),
                selection.soatotalDecodedAnchorsFlat.data(),
                soatotalDecodedAnchorsFlat.data(),
                selection.soatotalAnchorQualityScoresFlat.data(),
                soatotalAnchorQualityScoresFlat.data(),
                selection.d_numUsedReadIdsPerTaskPrefixSum.data(),
                d_numUsedReadIdsPerTaskPrefixSum.data(),
                d_numUsedReadIdsPerTask.data(),
                selection.d_usedReadIds.data(),
                d_usedReadIds.data(),
                selection.d_numFullyUsedReadIdsPerTaskPrefixSum.data(),
                d_numFullyUsedReadIdsPerTaskPrefixSum.data(),
                d_numFullyUsedReadIdsPerTask.data(),
                selection.d_fullyUsedReadIds.data(),
                d_fullyUsedReadIds.data()
            ); CUERR;

        }

        void addScalarIterationResultData(
            const float* d_goodscores,
            const extension::AbortReason* d_abortReasons,
            const bool* d_mateHasBeenFound,
            cudaStream_t stream
        ){
            readextendergpukernels::taskUpdateScalarIterationResultsKernel<128><<<SDIV(size(), 128), 128, 0, stream>>>(
                size(),
                goodscore.data(),
                abortReason.data(),
                mateHasBeenFound.data(),
                d_goodscores,
                d_abortReasons,
                d_mateHasBeenFound
            ); CUERR;
        }

        
        void addSoAIterationResultData(
            const int* d_addNumEntriesPerTask,
            const int* d_addNumEntriesPerTaskPrefixSum,
            const int* d_addTotalDecodedAnchorsLengths,
            const int* d_addTotalAnchorBeginInExtendedRead,
            const char* d_addTotalDecodedAnchorsFlat,
            const char* d_addTotalAnchorQualityScoresFlat,
            std::size_t addSequencesPitchInBytes,
            std::size_t addQualityPitchInBytes,
            cudaStream_t stream
        ){

            CachedDeviceUVector<int> newNumEntriesPerTask(size(), stream, *cubAllocator);
            CachedDeviceUVector<int> newNumEntriesPerTaskPrefixSum(size() + 1, stream, *cubAllocator);

            readextendergpukernels::vectorAddKernel<<<SDIV(size(), 128), 128, 0, stream>>>(
                d_addNumEntriesPerTask, 
                soaNumIterationResultsPerTask.begin(), 
                newNumEntriesPerTask.begin(), 
                size()
            ); CUERR;

            CubCallWrapper(*cubAllocator).cubInclusiveSum(
                newNumEntriesPerTask.begin(),
                newNumEntriesPerTaskPrefixSum.begin() + 1,
                size(),
                stream
            );

            cudaMemsetAsync(
                newNumEntriesPerTaskPrefixSum.data(),
                0, 
                sizeof(int),
                stream
            ); CUERR;

            std::size_t newirregularsize = 0;
            if(size() > 0){
                int result = 0;
                cudaMemcpyAsync(&result, newNumEntriesPerTaskPrefixSum.data() + size(), sizeof(int), D2H, stream); CUERR;
                cudaStreamSynchronize(stream); CUERR;

                newirregularsize = result;
            }

            CachedDeviceUVector<int> newsoatotalDecodedAnchorsLengths(newirregularsize, stream, *cubAllocator);
            CachedDeviceUVector<int> newsoatotalAnchorBeginInExtendedRead(newirregularsize, stream, *cubAllocator);
            CachedDeviceUVector<char> newsoatotalDecodedAnchorsFlat(newirregularsize * decodedSequencePitchInBytes, stream, *cubAllocator);
            CachedDeviceUVector<char> newsoatotalAnchorQualityScoresFlat(newirregularsize * qualityPitchInBytes, stream, *cubAllocator);

            readextendergpukernels::taskAddIterationResultsKernel<128,32><<<size(), 128, 0, stream>>>(
                size(),
                decodedSequencePitchInBytes,
                qualityPitchInBytes,
                addSequencesPitchInBytes,
                addQualityPitchInBytes,
                newNumEntriesPerTaskPrefixSum.data(),
                newsoatotalDecodedAnchorsFlat.data(),
                newsoatotalAnchorQualityScoresFlat.data(),
                newsoatotalDecodedAnchorsLengths.data(),
                newsoatotalAnchorBeginInExtendedRead.data(),
                soaNumIterationResultsPerTask.data(),
                soaNumIterationResultsPerTaskPrefixSum.data(),
                soatotalDecodedAnchorsLengths.data(),
                soatotalAnchorBeginInExtendedRead.data(),
                soatotalDecodedAnchorsFlat.data(),
                soatotalAnchorQualityScoresFlat.data(),
                d_addNumEntriesPerTask,
                d_addNumEntriesPerTaskPrefixSum,
                d_addTotalDecodedAnchorsLengths,
                d_addTotalAnchorBeginInExtendedRead,
                d_addTotalDecodedAnchorsFlat,
                d_addTotalAnchorQualityScoresFlat
            ); CUERR;

            std::swap(soaNumIterationResultsPerTask, newNumEntriesPerTask);
            std::swap(soaNumIterationResultsPerTaskPrefixSum, newNumEntriesPerTaskPrefixSum);
            std::swap(soatotalDecodedAnchorsLengths, newsoatotalDecodedAnchorsLengths);
            std::swap(soatotalAnchorBeginInExtendedRead, newsoatotalAnchorBeginInExtendedRead);
            std::swap(soatotalDecodedAnchorsFlat, newsoatotalDecodedAnchorsFlat);
            std::swap(soatotalAnchorQualityScoresFlat, newsoatotalAnchorQualityScoresFlat);

            consistencyCheck();

        }

        void updateUsedReadIdsAndFullyUsedReadIds(
            const read_number* d_candidateReadIds,
            const int* d_numCandidatesPerAnchor,
            const int* d_numCandidatesPerAnchorPrefixSum,
            const bool* d_isFullyUsedId,
            int numCandidateIds,
            cudaStream_t stream
        ){
            int numUsedIds = 0;
            int numFullyUsedIds = 0;
            cudaMemcpyAsync(&numUsedIds, d_numUsedReadIdsPerTaskPrefixSum.data() + size(), sizeof(int), D2H, stream); CUERR;
            cudaMemcpyAsync(&numFullyUsedIds, d_numFullyUsedReadIdsPerTaskPrefixSum.data() + size(), sizeof(int), D2H, stream); CUERR;
            cudaStreamSynchronize(stream); CUERR;

            if(numUsedIds != int(d_usedReadIds.size())){
                std::cerr << "numUsedIds " << numUsedIds << ", d_usedReadIds.size() " << d_usedReadIds.size() << "\n";
            }

            if(numFullyUsedIds != int(d_fullyUsedReadIds.size())){
                std::cerr << "numFullyUsedIds " << numFullyUsedIds << ", d_fullyUsedReadIds.size() " << d_fullyUsedReadIds.size() << "\n";
            }

            assert(numUsedIds == int(d_usedReadIds.size()));
            assert(numFullyUsedIds == int(d_fullyUsedReadIds.size()));

            const int maxoutputsize = numCandidateIds + numUsedIds;

            CachedDeviceUVector<read_number> d_newUsedReadIds(maxoutputsize, stream, *cubAllocator);
            CachedDeviceUVector<int> d_newnumUsedReadIdsPerTask(size(), stream, *cubAllocator);

            ThrustCachingAllocator<char> thrustCachingAllocator1(deviceId, cubAllocator, stream);

            auto d_newUsedReadIds_end = GpuSegmentedSetOperation::set_union(
                thrustCachingAllocator1,
                d_candidateReadIds,
                d_numCandidatesPerAnchor,
                d_numCandidatesPerAnchorPrefixSum,
                numCandidateIds,
                size(),
                d_usedReadIds.data(),
                d_numUsedReadIdsPerTask.data(),
                d_numUsedReadIdsPerTaskPrefixSum.data(),
                numUsedIds,
                size(),   
                d_newUsedReadIds.data(),
                d_newnumUsedReadIdsPerTask.data(),
                size(),
                stream
            );

            const int newNumUsedIds = std::distance(d_newUsedReadIds.data(), d_newUsedReadIds_end);

            d_newUsedReadIds.erase(d_newUsedReadIds.begin() + newNumUsedIds, d_newUsedReadIds.end(), stream);
            std::swap(d_usedReadIds, d_newUsedReadIds);
            std::swap(d_numUsedReadIdsPerTask, d_newnumUsedReadIdsPerTask);

            d_newUsedReadIds.destroy();
            d_newnumUsedReadIdsPerTask.destroy();

            CachedDeviceUVector<read_number> d_currentFullyUsedReadIds(numCandidateIds, stream, *cubAllocator);
            CachedDeviceUVector<int> d_currentNumFullyUsedreadIdsPerAnchor(size(), stream, *cubAllocator);
            CachedDeviceUVector<int> d_currentNumFullyUsedreadIdsPerAnchorPS(size(), stream, *cubAllocator);
            CachedDeviceUScalar<int> d_addNumFullyUsed(1, stream, *cubAllocator);
            
            //make compact list of current fully used candidates
            CubCallWrapper(*cubAllocator).cubSelectFlagged(
                d_candidateReadIds,
                d_isFullyUsedId,
                d_currentFullyUsedReadIds.data(),
                d_addNumFullyUsed.data(),
                numCandidateIds,
                stream
            );

            //compute current number of fully used candidates per segment
            CubCallWrapper(*cubAllocator).cubSegmentedReduceSum(
                d_isFullyUsedId,
                d_currentNumFullyUsedreadIdsPerAnchor.data(),
                size(),
                d_numCandidatesPerAnchorPrefixSum,
                d_numCandidatesPerAnchorPrefixSum + 1,
                stream
            );

            //compute prefix sum of current number of fully used candidates per segment

            CubCallWrapper(*cubAllocator).cubExclusiveSum(
                d_currentNumFullyUsedreadIdsPerAnchor.data(), 
                d_currentNumFullyUsedreadIdsPerAnchorPS.data(), 
                size(),
                stream
            );

            int addNumFullyUsed = 0;
            cudaMemcpyAsync(&addNumFullyUsed, d_addNumFullyUsed.data(), sizeof(int), D2H, stream); CUERR;
            cudaStreamSynchronize(stream); CUERR;

            const int maxoutputsize2 = addNumFullyUsed + numFullyUsedIds;

            CachedDeviceUVector<read_number> d_newFullyUsedReadIds(maxoutputsize2, stream, *cubAllocator);
            CachedDeviceUVector<int> d_newNumFullyUsedreadIdsPerAnchor(size(), stream, *cubAllocator);

            auto d_newFullyUsedReadIds_end = GpuSegmentedSetOperation::set_union(
                thrustCachingAllocator1,
                d_currentFullyUsedReadIds.data(),
                d_currentNumFullyUsedreadIdsPerAnchor.data(),
                d_currentNumFullyUsedreadIdsPerAnchorPS.data(),
                addNumFullyUsed,
                size(),
                d_fullyUsedReadIds.data(),
                d_numFullyUsedReadIdsPerTask.data(),
                d_numFullyUsedReadIdsPerTaskPrefixSum.data(),
                numFullyUsedIds,
                size(),       
                d_newFullyUsedReadIds.data(),
                d_newNumFullyUsedreadIdsPerAnchor.data(),
                size(),
                stream
            );

            const int newNumFullyUsedIds = std::distance(d_newFullyUsedReadIds.data(), d_newFullyUsedReadIds_end);

            d_newFullyUsedReadIds.erase(d_newFullyUsedReadIds.begin() + newNumFullyUsedIds, d_newFullyUsedReadIds.end(), stream);

            std::swap(d_fullyUsedReadIds, d_newFullyUsedReadIds);
            std::swap(d_numFullyUsedReadIdsPerTask, d_newNumFullyUsedreadIdsPerAnchor);

            d_newFullyUsedReadIds.destroy();
            d_newNumFullyUsedreadIdsPerAnchor.destroy();

            CubCallWrapper(*cubAllocator).cubInclusiveSum(
                d_numFullyUsedReadIdsPerTask.data(), 
                d_numFullyUsedReadIdsPerTaskPrefixSum.data() + 1, 
                size(),
                stream
            );

            CubCallWrapper(*cubAllocator).cubInclusiveSum(
                d_numUsedReadIdsPerTask.data(), 
                d_numUsedReadIdsPerTaskPrefixSum.data() + 1, 
                size(),
                stream
            );

            //std::cerr << "after update used\n";
            consistencyCheck();
        }

  
        std::size_t sizeInBytes() const{
            std::size_t result = 0;
            for(std::size_t i = 0; i < entries; i++){
                result += 1;
                result += 1;
                result += 4;
                result += 4;
                result += 4;
                result += 4;
                result += 4;
                result += 4;
                result += 4;
                result += 4;
                result += 4;
                result += 4;
                result += 4;
                result += 4;
                // result += 4 + sizeof(char) * decodedMateRevC[i].size();
                // result += 4 + sizeof(char) * mateQualityScoresReversed[i].size();
                // result += 4 + sizeof(int) * totalDecodedAnchorsLengths[i].size();
                // result += 4 + sizeof(char) * totalDecodedAnchorsFlat[i].size();
                // result += 4 + sizeof(char) * totalAnchorQualityScoresFlat[i].size();
                // result += 4 + sizeof(int) * totalAnchorBeginInExtendedRead[i].size();
            }
            //TODO soa
            return result;
        }

        void getActiveFlags(bool* d_flags, int insertSize, int insertSizeStddev, cudaStream_t stream) const{
            readextendergpukernels::taskComputeActiveFlagsKernel<128><<<SDIV(size(), 128), 128, 0, stream>>>(
                size(),
                insertSize,
                insertSizeStddev,
                d_flags,
                iteration.data(),
                soaNumIterationResultsPerTask.data(),
                soaNumIterationResultsPerTaskPrefixSum.data(),
                soatotalAnchorBeginInExtendedRead.data(),
                soainputmateLengths.data(),
                abortReason.data(),
                mateHasBeenFound.data()
            ); CUERR;

            consistencyCheck();
        }

    };




    struct Hasher{
        const gpu::GpuMinhasher* gpuMinhasher{};
        mutable MinhasherHandle minhashHandle{};
        helpers::SimpleAllocationPinnedHost<int> h_numCandidates{};

        Hasher(const gpu::GpuMinhasher& gpuMinhasher_)
            : gpuMinhasher(&gpuMinhasher_),
            minhashHandle(gpuMinhasher_.makeMinhasherHandle()){

            h_numCandidates.resize(1);
        }

        ~Hasher(){
            gpuMinhasher->destroyHandle(minhashHandle);
        }

        void getCandidateReadIds(const AnchorData& anchorData, AnchorHashResult& results, cudaStream_t stream){
            const int numAnchors = anchorData.d_anchorSequencesLength.size();

            results.d_numCandidatesPerAnchor.resizeUninitialized(numAnchors, stream);
            results.d_numCandidatesPerAnchorPrefixSum.resizeUninitialized(numAnchors + 1, stream);

            int totalNumValues = 0;

            gpuMinhasher->determineNumValues(
                minhashHandle,
                anchorData.d_subjectSequencesData.data(),
                anchorData.encodedSequencePitchInInts,
                anchorData.d_anchorSequencesLength.data(),
                numAnchors,
                results.d_numCandidatesPerAnchor.data(),
                totalNumValues,
                stream
            );

            cudaStreamSynchronize(stream); CUERR;

            results.d_candidateReadIds.resizeUninitialized(totalNumValues, stream);    

            if(totalNumValues == 0){
                cudaMemsetAsync(results.d_numCandidatesPerAnchor.data(), 0, sizeof(int) * numAnchors , stream); CUERR;
                cudaMemsetAsync(results.d_numCandidatesPerAnchorPrefixSum.data(), 0, sizeof(int) * (1 + numAnchors), stream); CUERR;
            }else{

                gpuMinhasher->retrieveValues(
                    minhashHandle,
                    nullptr,
                    numAnchors,              
                    totalNumValues,
                    results.d_candidateReadIds.data(),
                    results.d_numCandidatesPerAnchor.data(),
                    results.d_numCandidatesPerAnchorPrefixSum.data(),
                    stream
                );

                cudaMemcpyAsync(
                    h_numCandidates.data(),
                    results.d_numCandidatesPerAnchorPrefixSum.data() + numAnchors,
                    sizeof(int),
                    D2H,
                    stream
                ); CUERR;

                cudaStreamSynchronize(stream); CUERR;

                results.d_candidateReadIds.erase(results.d_candidateReadIds.begin() + (*h_numCandidates), results.d_candidateReadIds.end(), stream);
            }
        }
    };


    enum class State{
        UpdateWorkingSet,
        BeforeHash,
        BeforeRemoveIds,
        BeforeComputePairFlags,
        BeforeLoadCandidates,
        BeforeEraseData,
        BeforeAlignment,
        BeforeAlignmentFilter,
        BeforeMSA,
        BeforeExtend,
        BeforePrepareNextIteration,
        Finished,
        None
    };

    static std::string to_string(State s){
        switch(s){
            case State::UpdateWorkingSet: return "UpdateWorkingSet";
            case State::BeforeHash: return "BeforeHash";
            case State::BeforeRemoveIds: return "BeforeRemoveIds";
            case State::BeforeComputePairFlags: return "BeforeComputePairFlags";
            case State::BeforeLoadCandidates: return "BeforeLoadCandidates";
            case State::BeforeEraseData: return "BeforeEraseData";
            case State::BeforeAlignment: return "BeforeAlignment";
            case State::BeforeAlignmentFilter: return "BeforeAlignmentFilter";
            case State::BeforeMSA: return "BeforeMSA";
            case State::BeforeExtend: return "BeforeExtend";
            case State::BeforePrepareNextIteration: return "BeforePrepareNextIteration";
            case State::Finished: return "Finished";
            case State::None: return "None";
            default: return "Missing case GpuReadExtender::to_string(State)\n";
        };
    }

    bool isEmpty() const noexcept{
        return tasks->size() == 0;
    }

    void setState(State newstate){      
        if(false){
            std::cerr << "batchdata " << someId << " statechange " << to_string(state) << " -> " << to_string(newstate);
            std::cerr << "\n";
        }

        state = newstate;
    }

    GpuReadExtender(
        std::size_t encodedSequencePitchInInts_,
        std::size_t decodedSequencePitchInBytes_,
        std::size_t qualityPitchInBytes_,
        std::size_t msaColumnPitchInElements_,
        bool isPairedEnd_,
        const gpu::GpuReadStorage& rs, 
        const CorrectionOptions& coropts,
        const GoodAlignmentProperties& gap,
        const cpu::QualityScoreConversion& qualityConversion_,
        int insertSize_,
        int insertSizeStddev_,
        int maxextensionPerStep_,
        std::array<cudaStream_t, 4> streams_,
        cub::CachingDeviceAllocator& cubAllocator_
    ) : 
        pairedEnd(isPairedEnd_),
        insertSize(insertSize_),
        insertSizeStddev(insertSizeStddev_),
        cubAllocator(&cubAllocator_),
        gpuReadStorage(&rs),
        correctionOptions(&coropts),
        goodAlignmentProperties(&gap),
        qualityConversion(&qualityConversion_),
        readStorageHandle(gpuReadStorage->makeHandle()),
        d_numAnchors(cubAllocator_),
        d_numCandidates(cubAllocator_),
        d_mateIdHasBeenRemoved(cubAllocator_),
        anchorHashResult(cubAllocator_),
        d_candidateSequencesData(cubAllocator_),
        d_candidateSequencesLength(cubAllocator_),    
        d_candidateReadIds(cubAllocator_),
        d_isPairedCandidate(cubAllocator_),
        d_alignment_overlaps(cubAllocator_),
        d_alignment_shifts(cubAllocator_),
        d_alignment_nOps(cubAllocator_),
        d_alignment_best_alignment_flags(cubAllocator_),
        d_numCandidatesPerAnchor(cubAllocator_),
        d_numCandidatesPerAnchorPrefixSum(cubAllocator_),
        anchorData(cubAllocator_),
        d_subjectSequencesDataDecoded(cubAllocator_),
        d_anchorQualityScores(cubAllocator_),
        d_anchorSequencesLength(cubAllocator_),
        d_subjectSequencesData(cubAllocator_),
        d_accumExtensionsLengths(cubAllocator_),
        multiMSA(cubAllocator_),
        streams(streams_)
    {

        cudaGetDevice(&deviceId); CUERR;

        kernelLaunchHandle = gpu::make_kernel_launch_handle(deviceId);

        h_numAnchors.resize(1);
        h_numCandidates.resize(1);
        h_numAnchorsWithRemovedMates.resize(1);

        h_minmax.resize(2);
        h_numPositions.resize(2);

        d_numAnchors.resize(1, cudaStreamDefault);
        d_numCandidates.resize(1, cudaStreamDefault);

        *h_numAnchors = 0;
        *h_numCandidates = 0;
        *h_numAnchorsWithRemovedMates = 0;

        encodedSequencePitchInInts = encodedSequencePitchInInts_;
        decodedSequencePitchInBytes = decodedSequencePitchInBytes_;
        qualityPitchInBytes = qualityPitchInBytes_;
        msaColumnPitchInElements = msaColumnPitchInElements_;

        cudaDeviceSynchronize(); CUERR;
    }

    static constexpr int getNumRefinementIterations() noexcept{
        return 5;
    }

    void processOneIteration(
        TaskData& inputTasks, 
        AnchorData& currentAnchorData, 
        AnchorHashResult& currentHashResults, 
        TaskData& outputFinishedTasks, 
        const IterationConfig& iterationConfig_,
        cudaStream_t stream
    ){
        std::lock_guard<std::mutex> lockguard(mutex);

        if(inputTasks.size() > 0){

            cudaEventRecord(events[0], stream); CUERR;
            cudaStreamWaitEvent(streams[0], events[0], 0); CUERR;
            cudaStreamWaitEvent(streams[1], events[0], 0); CUERR;

            tasks = &inputTasks;
            initialNumCandidates = currentHashResults.d_candidateReadIds.size();
            iterationConfig = &iterationConfig_;

            copyAnchorDataFrom(currentAnchorData, streams[0]);
            copyHashResultsFrom(currentHashResults, streams[0]);

            state = GpuReadExtender::State::BeforeRemoveIds;

            nvtx::push_range("removeUsedIdsAndMateIds", 0);
            removeUsedIdsAndMateIds();
            nvtx::pop_range();

            nvtx::push_range("computePairFlagsGpu", 1);
            computePairFlagsGpu();
            nvtx::pop_range();

            nvtx::push_range("loadCandidateSequenceData", 2);
            loadCandidateSequenceData();
            nvtx::pop_range();

            nvtx::push_range("eraseDataOfRemovedMates", 3);
            eraseDataOfRemovedMates();
            nvtx::pop_range();

            nvtx::push_range("calculateAlignments", 4);
            calculateAlignments();
            nvtx::pop_range();

            nvtx::push_range("filterAlignments", 5);
            filterAlignments();
            nvtx::pop_range();

            nvtx::push_range("computeMSAs", 6);
            computeMSAs();
            nvtx::pop_range();

            nvtx::push_range("computeExtendedSequencesFromMSAs", 7);
            computeExtendedSequencesFromMSAs();
            nvtx::pop_range();

            nvtx::push_range("prepareNextIteration", 8);
            prepareNextIteration(inputTasks, outputFinishedTasks);
            nvtx::pop_range();


            cudaEventRecord(events[0], streams[0]); CUERR;
            cudaStreamWaitEvent(stream, events[0], 0); CUERR;
            cudaEventRecord(events[0], streams[1]); CUERR;
            cudaStreamWaitEvent(stream, events[0], 0); CUERR;

        }else{
            addFinishedGpuSoaTasks(inputTasks, outputFinishedTasks, stream);
            inputTasks.clear(stream);
        }


    }


    void removeUsedIdsAndMateIds(){
        assert(state == GpuReadExtender::State::BeforeRemoveIds);

        cudaStream_t firstStream = streams[0];
             
        CachedDeviceUVector<bool> d_shouldBeKept(initialNumCandidates, firstStream, *cubAllocator);
        CachedDeviceUVector<int> d_numCandidatesPerAnchor2(tasks->size(), firstStream, *cubAllocator);        

        d_mateIdHasBeenRemoved.resizeUninitialized(tasks->size(), firstStream);

        helpers::call_fill_kernel_async(d_shouldBeKept.data(), initialNumCandidates, false, firstStream);

        
        //flag candidates ids to remove because they are equal to anchor id or equal to mate id
        readextendergpukernels::flagCandidateIdsWhichAreEqualToAnchorOrMateKernel<<<tasks->size(), 128, 0, firstStream>>>(
            d_candidateReadIds.data(),
            tasks->myReadId.data(),
            tasks->mateReadId.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_numCandidatesPerAnchor.data(),
            d_shouldBeKept.data(),
            d_mateIdHasBeenRemoved.data(),
            d_numCandidatesPerAnchor2.data(),
            tasks->size(),
            pairedEnd
        );
        CUERR;  

        //copy selected candidate ids

        CachedDeviceUVector<read_number> d_candidateReadIds2(initialNumCandidates, firstStream, *cubAllocator);
        assert(h_numCandidates.data() != nullptr);

        CubCallWrapper(*cubAllocator).cubSelectFlagged(
            d_candidateReadIds.data(),
            d_shouldBeKept.data(),
            d_candidateReadIds2.data(),
            h_numCandidates.data(),
            initialNumCandidates,
            firstStream
        );

        cudaEventRecord(h_numCandidatesEvent, firstStream); CUERR;

        d_shouldBeKept.destroy();

        //d_candidateReadIds2.erase(d_candidateReadIds2.begin() + *h_numCandidates, d_candidateReadIds2.end(), firstStream);

        CachedDeviceUVector<int> d_numCandidatesPerAnchorPrefixSum2(tasks->size() + 1, firstStream, *cubAllocator);

        //compute prefix sum of number of candidates per anchor
        cudaMemsetAsync(d_numCandidatesPerAnchorPrefixSum2.data(), 0, sizeof(int), firstStream); CUERR;

        CubCallWrapper(*cubAllocator).cubInclusiveSum(
            d_numCandidatesPerAnchor2.data(), 
            d_numCandidatesPerAnchorPrefixSum2.data() + 1, 
            tasks->size(),
            firstStream
        );

        cudaEventSynchronize(h_numCandidatesEvent); CUERR; //wait for h_numCandidates
   
        #ifdef DO_ONLY_REMOVE_MATE_IDS
            std::swap(d_candidateReadIds, d_candidateReadIds2);
            std::swap(d_numCandidatesPerAnchor, d_numCandidatesPerAnchor2);
        #else

            ThrustCachingAllocator<char> thrustCachingAllocator1(deviceId, cubAllocator, firstStream);
            

            //compute segmented set difference.  d_candidateReadIds = d_candidateReadIds2 \ d_usedReadIds
            auto d_candidateReadIds_end = GpuSegmentedSetOperation::set_difference(
                thrustCachingAllocator1,
                d_candidateReadIds2.data(),
                d_numCandidatesPerAnchor2.data(),
                d_numCandidatesPerAnchorPrefixSum2.data(),
                *h_numCandidates,
                tasks->size(),
                tasks->d_fullyUsedReadIds.data(),
                tasks->d_numFullyUsedReadIdsPerTask.data(),
                tasks->d_numFullyUsedReadIdsPerTaskPrefixSum.data(),
                tasks->d_fullyUsedReadIds.size(),
                tasks->size(),        
                d_candidateReadIds.data(),
                d_numCandidatesPerAnchor.data(),
                tasks->size(),
                firstStream
            );

            *h_numCandidates = std::distance(d_candidateReadIds.data(), d_candidateReadIds_end);

        #endif

        h_candidateReadIds.resize(*h_numCandidates);
        cudaEventRecord(events[0], firstStream);
        cudaStreamWaitEvent(hostOutputStream, events[0], 0); CUERR;

        cudaMemcpyAsync(
            h_candidateReadIds.data(),
            d_candidateReadIds.data(),
            sizeof(read_number) * (*h_numCandidates),
            D2H,
            hostOutputStream
        ); CUERR;

        cudaEventRecord(h_candidateReadIdsEvent, hostOutputStream); CUERR;        

        d_numCandidatesPerAnchor2.destroy();
        d_numCandidatesPerAnchorPrefixSum2.destroy();        
        
        cudaMemsetAsync(d_numCandidatesPerAnchorPrefixSum.data(), 0, sizeof(int), firstStream); CUERR;
        //compute prefix sum of new segment sizes
        CubCallWrapper(*cubAllocator).cubInclusiveSum(
            d_numCandidatesPerAnchor.data(), 
            d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            tasks->size(),
            firstStream
        );

        setState(GpuReadExtender::State::BeforeComputePairFlags);
    }

    void computePairFlagsGpu() {
        assert(state == GpuReadExtender::State::BeforeComputePairFlags);

        cudaStream_t stream = streams[0];

        d_isPairedCandidate.resizeUninitialized(initialNumCandidates, stream);
        CachedDeviceUVector<int> d_firstTasksOfPairsToCheck(tasks->size(), stream, *cubAllocator);
        CachedDeviceUVector<bool> d_flags(tasks->size(), stream, *cubAllocator);
        CachedDeviceUScalar<int> d_numChecks(1, stream, *cubAllocator);

        helpers::call_fill_kernel_async(d_isPairedCandidate.data(), initialNumCandidates, false, stream);
        helpers::call_fill_kernel_async(d_flags.data(), tasks->size(), false, stream);

        readextendergpukernels::flagFirstTasksOfConsecutivePairedTasks<128><<<SDIV(tasks->size(), 128), 128, 0, stream>>>(
            tasks->size(),
            d_flags.data(),
            tasks->id.data()
        ); CUERR;

        CubCallWrapper(*cubAllocator).cubSelectFlagged(
            thrust::make_counting_iterator(0),
            d_flags.data(),
            d_firstTasksOfPairsToCheck.data(),
            d_numChecks.data(),
            tasks->size(),
            stream
        );

        d_flags.destroy();

        readextendergpukernels::flagPairedCandidatesKernel<128,4096><<<tasks->size(), 128, 0, stream>>>(
            d_numChecks.data(),
            d_firstTasksOfPairsToCheck.data(),
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_candidateReadIds.data(),
            tasks->d_numUsedReadIdsPerTask.data(),
            tasks->d_numUsedReadIdsPerTaskPrefixSum.data(),
            tasks->d_usedReadIds.data(),
            d_isPairedCandidate.data()
        );

        setState(GpuReadExtender::State::BeforeLoadCandidates);

    }

    void loadCandidateSequenceData() {
        assert(state == GpuReadExtender::State::BeforeLoadCandidates);

        cudaStream_t stream = streams[0];

        d_candidateSequencesLength.resizeUninitialized(initialNumCandidates, stream);
        d_candidateSequencesData.resizeUninitialized(encodedSequencePitchInInts * initialNumCandidates, stream);

        //cudaEventSynchronize(h_candidateReadIdsEvent); CUERR;
        cudaEventSynchronize(h_numCandidatesEvent); CUERR;

        gpuReadStorage->gatherSequences(
            readStorageHandle,
            d_candidateSequencesData.data(),
            encodedSequencePitchInInts,
            makeAsyncConstBufferWrapper(h_candidateReadIds.data(), h_candidateReadIdsEvent),
            d_candidateReadIds.data(), //device accessible
            *h_numCandidates,
            stream
        );

        gpuReadStorage->gatherSequenceLengths(
            readStorageHandle,
            d_candidateSequencesLength.data(),
            d_candidateReadIds.data(),
            *h_numCandidates,
            stream
        );

        setState(GpuReadExtender::State::BeforeEraseData);
    }

    void eraseDataOfRemovedMates(){
        assert(state == GpuReadExtender::State::BeforeEraseData);

        cudaStream_t stream = streams[0];

        CachedDeviceUVector<bool> d_keepflags(initialNumCandidates, stream, *cubAllocator);

        //compute flags of candidates which should not be removed. Candidates which should be removed are identical to mate sequence
        helpers::call_fill_kernel_async(d_keepflags.data(), initialNumCandidates, true, stream);

        const int* d_currentNumCandidates = d_numCandidatesPerAnchorPrefixSum.data() + tasks->size();

        constexpr int groupsize = 32;
        constexpr int blocksize = 128;
        constexpr int groupsperblock = blocksize / groupsize;
        dim3 block(blocksize,1,1);
        dim3 grid(SDIV(tasks->size() * groupsize, blocksize), 1, 1);
        const std::size_t smembytes = sizeof(unsigned int) * groupsperblock * encodedSequencePitchInInts;

        readextendergpukernels::filtermatekernel<blocksize,groupsize><<<grid, block, smembytes, stream>>>(
            tasks->inputEncodedMate.data(),
            d_candidateSequencesData.data(),
            encodedSequencePitchInInts,
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_mateIdHasBeenRemoved.data(),
            tasks->size(),
            d_keepflags.data(),
            initialNumCandidates,
            d_currentNumCandidates
        ); CUERR;

        //cudaEventSynchronize(h_numAnchorsWithRemovedMatesEvent); CUERR;

        //if(*h_numAnchorsWithRemovedMates > 0){
        compactCandidateDataByFlagsExcludingAlignments(
            d_keepflags.data(),
            false,
            stream
        );
        //}

        setState(GpuReadExtender::State::BeforeAlignment);
    }

    void calculateAlignments(){
        assert(state == GpuReadExtender::State::BeforeAlignment);

        cudaStream_t stream = streams[0];

        CachedDeviceUVector<int> d_segmentIdsOfCandidates(initialNumCandidates, stream, *cubAllocator);
        setGpuSegmentIds(
            d_segmentIdsOfCandidates.data(),
            tasks->size(),
            initialNumCandidates,
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            stream
        );

        d_alignment_overlaps.resizeUninitialized(initialNumCandidates, stream);
        d_alignment_shifts.resizeUninitialized(initialNumCandidates, stream);
        d_alignment_nOps.resizeUninitialized(initialNumCandidates, stream);
        d_alignment_best_alignment_flags.resizeUninitialized(initialNumCandidates, stream);

        CachedDeviceUVector<bool> d_alignment_isValid(initialNumCandidates, stream, *cubAllocator);

        h_numAnchors[0] = tasks->size();

        const bool* const d_anchorContainsN = nullptr;
        const bool* const d_candidateContainsN = nullptr;
        const bool removeAmbiguousAnchors = false;
        const bool removeAmbiguousCandidates = false;
        const int maxNumAnchors = tasks->size();
        const int maxNumCandidates = initialNumCandidates; //this does not need to be exact, but it must be >= d_numCandidatesPerAnchorPrefixSum[tasks->size()]
        const int maximumSequenceLength = encodedSequencePitchInInts * 16;
        const int encodedSequencePitchInInts2Bit = encodedSequencePitchInInts;
        const int min_overlap = goodAlignmentProperties->min_overlap;
        const float maxErrorRate = goodAlignmentProperties->maxErrorRate;
        const float min_overlap_ratio = goodAlignmentProperties->min_overlap_ratio;
        const float estimatedNucleotideErrorRate = correctionOptions->estimatedErrorrate;

        auto callAlignmentKernel = [&](void* d_tempstorage, size_t& tempstoragebytes){

            call_popcount_rightshifted_hamming_distance_kernel_async(
                d_tempstorage,
                tempstoragebytes,
                d_alignment_overlaps.data(),
                d_alignment_shifts.data(),
                d_alignment_nOps.data(),
                d_alignment_isValid.data(),
                d_alignment_best_alignment_flags.data(),
                d_subjectSequencesData.data(),
                d_candidateSequencesData.data(),
                d_anchorSequencesLength.data(),
                d_candidateSequencesLength.data(),
                d_numCandidatesPerAnchorPrefixSum.data(),
                d_numCandidatesPerAnchor.data(),
                d_segmentIdsOfCandidates.data(),
                h_numAnchors.data(),
                &d_numCandidatesPerAnchorPrefixSum[tasks->size()],
                d_anchorContainsN,
                removeAmbiguousAnchors,
                d_candidateContainsN,
                removeAmbiguousCandidates,
                maxNumAnchors,
                maxNumCandidates,
                maximumSequenceLength,
                encodedSequencePitchInInts2Bit,
                min_overlap,
                maxErrorRate,
                min_overlap_ratio,
                estimatedNucleotideErrorRate,
                stream,
                kernelLaunchHandle
            );
        };

        size_t tempstoragebytes = 0;
        callAlignmentKernel(nullptr, tempstoragebytes);

        CachedDeviceUVector<char> d_tempstorage(tempstoragebytes, stream, *cubAllocator);

        callAlignmentKernel(d_tempstorage.data(), tempstoragebytes);

        setState(GpuReadExtender::State::BeforeAlignmentFilter);
    }

    void filterAlignments(){
        assert(state == GpuReadExtender::State::BeforeAlignmentFilter);

        cudaStream_t stream = streams[0];

        CachedDeviceUVector<bool> d_keepflags(initialNumCandidates, stream, *cubAllocator);

        const int* const d_currentNumCandidates = d_numCandidatesPerAnchorPrefixSum.data() + tasks->size();

        readextendergpukernels::flagGoodAlignmentsKernel<128><<<tasks->size(), 128, 0, stream>>>(
            d_alignment_best_alignment_flags.data(),
            d_alignment_shifts.data(),
            d_alignment_overlaps.data(),
            d_anchorSequencesLength.data(),
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_isPairedCandidate.data(),
            d_keepflags.data(),
            goodAlignmentProperties->min_overlap_ratio,
            tasks->size(),
            d_currentNumCandidates,
            initialNumCandidates
        ); CUERR;

        compactCandidateDataByFlags(
            d_keepflags.data(),
            true, //copy candidate read ids to host because they might be needed to load quality scores
            stream
        );

        setState(GpuReadExtender::State::BeforeMSA);
    }

    void computeMSAs(){
        assert(state == GpuReadExtender::State::BeforeMSA);

        cudaStream_t stream = streams[0];

        CachedDeviceUVector<char> d_candidateQualityScores(qualityPitchInBytes * initialNumCandidates, stream, *cubAllocator);

        loadCandidateQualityScores(stream, d_candidateQualityScores.data());

        CachedDeviceUVector<int> d_numCandidatesPerAnchor2(tasks->size(), stream, *cubAllocator);

        CachedDeviceUVector<int> indices1(initialNumCandidates, stream, *cubAllocator);
        CachedDeviceUVector<int> indices2(initialNumCandidates, stream, *cubAllocator);
        CachedDeviceUScalar<int> d_numCandidates2(1, stream, *cubAllocator);

        const int threads = 32 * tasks->size();
        readextendergpukernels::segmentedIotaKernel<32><<<SDIV(threads, 128), 128, 0, stream>>>(
            indices1.data(),
            tasks->size(),
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data()
        ); CUERR;

        *h_numAnchors = tasks->size();

        const bool useQualityScoresForMSA = true;

        multiMSA.construct(
            d_alignment_overlaps.data(),
            d_alignment_shifts.data(),
            d_alignment_nOps.data(),
            d_alignment_best_alignment_flags.data(),
            indices1.data(),
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_anchorSequencesLength.data(),
            d_subjectSequencesData.data(),
            d_anchorQualityScores.data(),
            tasks->size(),
            d_candidateSequencesLength.data(),
            d_candidateSequencesData.data(),
            d_candidateQualityScores.data(),
            d_isPairedCandidate.data(),
            initialNumCandidates,
            h_numAnchors.data(), //d_numAnchors
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            useQualityScoresForMSA,
            goodAlignmentProperties->maxErrorRate,
            gpu::MSAColumnCount(msaColumnPitchInElements),
            stream
        );

        multiMSA.refine(
            indices2.data(),
            d_numCandidatesPerAnchor2.data(),
            d_numCandidates2.data(),
            d_alignment_overlaps.data(),
            d_alignment_shifts.data(),
            d_alignment_nOps.data(),
            d_alignment_best_alignment_flags.data(),
            indices1.data(),
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_anchorSequencesLength.data(),
            d_subjectSequencesData.data(),
            d_anchorQualityScores.data(),
            tasks->size(),
            d_candidateSequencesLength.data(),
            d_candidateSequencesData.data(),
            d_candidateQualityScores.data(),
            d_isPairedCandidate.data(),
            initialNumCandidates,
            h_numAnchors.data(), //d_numAnchors
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            useQualityScoresForMSA,
            goodAlignmentProperties->maxErrorRate,
            correctionOptions->estimatedCoverage,
            getNumRefinementIterations(),
            stream
        );
 
        CachedDeviceUVector<bool> d_shouldBeKept(initialNumCandidates, stream, *cubAllocator);

        helpers::call_fill_kernel_async(d_shouldBeKept.data(), initialNumCandidates, false, stream); CUERR;

        const int numThreads2 = tasks->size() * 32;
        readextendergpukernels::convertLocalIndicesInSegmentsToGlobalFlags<128,32>
        <<<SDIV(numThreads2, 128), 128, 0, stream>>>(
            d_shouldBeKept.data(),
            indices2.data(),
            d_numCandidatesPerAnchor2.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            tasks->size()
        ); CUERR
     
        indices1.destroy();
        indices2.destroy();
        d_numCandidatesPerAnchor2.destroy();

        d_candidateQualityScores.destroy();

        compactCandidateDataByFlags(
            d_shouldBeKept.data(),
            false,
            stream
        );
        
        setState(GpuReadExtender::State::BeforeExtend);
    }


    void computeExtendedSequencesFromMSAs(){
        assert(state == GpuReadExtender::State::BeforeExtend);

        cudaStream_t stream = streams[0];

        outputAnchorPitchInBytes = SDIV(decodedSequencePitchInBytes, 128) * 128;
        outputAnchorQualityPitchInBytes = SDIV(qualityPitchInBytes, 128) * 128;
        decodedMatesRevCPitchInBytes = SDIV(decodedSequencePitchInBytes, 128) * 128;

        CachedDeviceUVector<int> d_accumExtensionsLengthsOUT(tasks->size(), stream, *cubAllocator);
        CachedDeviceUVector<int> d_sizeOfGapToMate(tasks->size(), stream, *cubAllocator);
        CachedDeviceUVector<float> d_goodscores(tasks->size(), stream, *cubAllocator);

        CachedDeviceUVector<char> d_outputAnchors(tasks->size() * outputAnchorPitchInBytes, stream, *cubAllocator);
        CachedDeviceUVector<char> d_outputAnchorQualities(tasks->size() * outputAnchorQualityPitchInBytes, stream, *cubAllocator);
        CachedDeviceUVector<bool> d_outputMateHasBeenFound(tasks->size(), stream, *cubAllocator);
        CachedDeviceUVector<extension::AbortReason> d_abortReasons(tasks->size(), stream, *cubAllocator);
        CachedDeviceUVector<int> d_outputAnchorLengths(tasks->size(), stream, *cubAllocator);
        CachedDeviceUVector<bool> d_isFullyUsedCandidate(initialNumCandidates, stream, *cubAllocator);
  
        readextendergpukernels::fillKernel<<<SDIV(128, tasks->size()), 128, 0, stream>>>(
            thrust::make_zip_iterator(thrust::make_tuple(
                d_outputMateHasBeenFound.data(),
                d_abortReasons.data(),
                d_goodscores.data()
            )),
            tasks->size(),
            thrust::make_tuple(
                false, 
                extension::AbortReason::None,
                0.0f
            )
        ); CUERR;

        helpers::call_fill_kernel_async(d_isFullyUsedCandidate.data(), initialNumCandidates, false, stream); CUERR;
      
        //compute extensions

        readextendergpukernels::computeExtensionStepFromMsaKernel<128><<<tasks->size(), 128, 0, stream>>>(
            insertSize,
            insertSizeStddev,
            multiMSA.multiMSAView(),
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_anchorSequencesLength.data(),
            d_accumExtensionsLengths.data(),
            tasks->soainputmateLengths.data(),
            d_abortReasons.data(),
            d_accumExtensionsLengthsOUT.data(),
            d_outputAnchors.data(),
            outputAnchorPitchInBytes,
            d_outputAnchorQualities.data(),
            outputAnchorQualityPitchInBytes,
            d_outputAnchorLengths.data(),
            tasks->pairedEnd.data(),
            tasks->inputEncodedMate.data(),
            encodedSequencePitchInInts,
            decodedMatesRevCPitchInBytes,
            d_outputMateHasBeenFound.data(),
            d_sizeOfGapToMate.data(),
            iterationConfig->minCoverageForExtension,
            iterationConfig->maxextensionPerStep
        ); CUERR;

        readextendergpukernels::computeExtensionStepQualityKernel<128><<<tasks->size(), 128, 0, stream>>>(
            d_goodscores.data(),
            multiMSA.multiMSAView(),
            d_abortReasons.data(),
            d_outputMateHasBeenFound.data(),
            d_accumExtensionsLengths.data(),
            d_accumExtensionsLengthsOUT.data(),
            d_anchorSequencesLength.data(),
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_candidateSequencesLength.data(),
            d_alignment_shifts.data(),
            d_alignment_best_alignment_flags.data(),
            d_candidateSequencesData.data(),
            multiMSA.d_columnProperties.data(),
            encodedSequencePitchInInts
        ); CUERR;

        readextendergpukernels::flagFullyUsedCandidatesKernel<128>
        <<<tasks->size(), 128, 0, stream>>>(
            tasks->size(),
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_candidateSequencesLength.data(),
            d_alignment_shifts.data(),
            d_anchorSequencesLength.data(),
            d_accumExtensionsLengths.data(),
            d_accumExtensionsLengthsOUT.data(),
            d_abortReasons.data(),
            d_outputMateHasBeenFound.data(),
            d_isFullyUsedCandidate.data()
        ); CUERR;

        nvtx::push_range("gpuunpack", 3);

        tasks->addScalarIterationResultData(
            d_goodscores.data(),
            d_abortReasons.data(),
            d_outputMateHasBeenFound.data(),
            stream
        );

        CachedDeviceUVector<int> d_addNumEntriesPerTask(tasks->size(), stream, *cubAllocator);
        CachedDeviceUVector<int> d_addNumEntriesPerTaskPrefixSum(tasks->size()+1, stream, *cubAllocator);

        //init first prefixsum element with 0.
        //compute number of soa iteration results per task
        readextendergpukernels::computeNumberOfSoaIterationResultsPerTaskKernel<128>
        <<<SDIV(tasks->size(), 128), 128, 0, stream>>>(
            tasks->size(),
            d_addNumEntriesPerTask.data(),
            d_addNumEntriesPerTaskPrefixSum.data(),
            d_abortReasons.data(),
            d_outputMateHasBeenFound.data(),
            d_sizeOfGapToMate.data()
        ); CUERR;
 
        CubCallWrapper(*cubAllocator).cubInclusiveSum(
            d_addNumEntriesPerTask.data(),
            d_addNumEntriesPerTaskPrefixSum.data() + 1,
            tasks->size(),
            stream
        );

        int numAdd = 0;
        cudaMemcpyAsync(
            &numAdd,
            d_addNumEntriesPerTaskPrefixSum.data() + tasks->size(),
            sizeof(int),
            D2H,
            stream
        ); CUERR;
        cudaStreamSynchronize(stream); CUERR;

        CachedDeviceUVector<char> d_addTotalDecodedAnchorsFlat(numAdd * outputAnchorPitchInBytes, stream, *cubAllocator);
        CachedDeviceUVector<char> d_addTotalAnchorQualityScoresFlat(numAdd * outputAnchorQualityPitchInBytes, stream, *cubAllocator);
        CachedDeviceUVector<int> d_addAnchorLengths(numAdd, stream, *cubAllocator);
        CachedDeviceUVector<int> d_addAnchorBeginsInExtendedRead(numAdd, stream, *cubAllocator);

        assert(tasks->decodedSequencePitchInBytes >= outputAnchorPitchInBytes);
        assert(tasks->qualityPitchInBytes >= outputAnchorQualityPitchInBytes);

        readextendergpukernels::makeSoAIterationResultsKernel<128><<<tasks->size(), 128, 0, stream>>>(
                tasks->size(),
                outputAnchorPitchInBytes,
                outputAnchorQualityPitchInBytes,
                d_addNumEntriesPerTask.data(),
                d_addNumEntriesPerTaskPrefixSum.data(),
                d_addTotalDecodedAnchorsFlat.data(),
                d_addTotalAnchorQualityScoresFlat.data(),
                d_addAnchorLengths.data(),
                d_addAnchorBeginsInExtendedRead.data(),
                tasks->decodedSequencePitchInBytes,
                tasks->qualityPitchInBytes,
                tasks->abortReason.data(),
                tasks->mateHasBeenFound.data(),
                tasks->soainputdecodedMateRevC.data(),
                tasks->soainputmateQualityScoresReversed.data(),
                tasks->soainputmateLengths.data(),
                d_sizeOfGapToMate.data(),
                d_outputAnchorLengths.data(),
                d_outputAnchors.data(),
                d_outputAnchorQualities.data(),
                d_accumExtensionsLengthsOUT.data()
        ); CUERR;

        tasks->addSoAIterationResultData(
            d_addNumEntriesPerTask.data(),
            d_addNumEntriesPerTaskPrefixSum.data(),
            d_addAnchorLengths.data(),
            d_addAnchorBeginsInExtendedRead.data(),
            d_addTotalDecodedAnchorsFlat.data(),
            d_addTotalAnchorQualityScoresFlat.data(),
            outputAnchorPitchInBytes,
            outputAnchorQualityPitchInBytes,
            stream
        );

        cudaEventSynchronize(h_numCandidatesEvent); CUERR;

        tasks->updateUsedReadIdsAndFullyUsedReadIds(
            d_candidateReadIds.data(),
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_isFullyUsedCandidate.data(),
            *h_numCandidates,
            stream
        );

        //increment iteration and check early exit of tasks
        tasks->iterationIsFinished(stream);

        nvtx::pop_range();

        std::swap(d_accumExtensionsLengths, d_accumExtensionsLengthsOUT);

        setState(GpuReadExtender::State::BeforePrepareNextIteration);
    }
    
    void prepareNextIteration(TaskData& outputActiveTasks, TaskData& outputFinishedTasks){
        assert(state == GpuReadExtender::State::BeforePrepareNextIteration);

        const int totalTasksBefore = tasks->size() + outputFinishedTasks.size();

        CachedDeviceUVector<bool> d_activeFlags(tasks->size(), streams[0], *cubAllocator);
        tasks->getActiveFlags(d_activeFlags.data(), insertSize, insertSizeStddev, streams[0]);

        TaskData newgpuSoaActiveTasks = tasks->select(
            d_activeFlags.data(),
            streams[0]
        );

        auto inactiveFlags = thrust::make_transform_iterator(
            d_activeFlags.data(),
            thrust::logical_not<bool>{}
        );

        TaskData newlygpuSoaFinishedTasks = tasks->select(
            inactiveFlags,
            streams[0]
        );

        addFinishedGpuSoaTasks(newlygpuSoaFinishedTasks, outputFinishedTasks, streams[0]);
        std::swap(outputActiveTasks, newgpuSoaActiveTasks);

        const int totalTasksAfter = tasks->size() + outputFinishedTasks.size();
        assert(totalTasksAfter == totalTasksBefore);

        if(!isEmpty()){
            setState(GpuReadExtender::State::UpdateWorkingSet);
        }else{
            setStateToFinished();
        }
        
    }


    TaskData getFinishedGpuSoaTasksOfFinishedPairsAndRemoveThem(TaskData& finishedTasks){
        //determine tasks in groups of 4

        cudaStream_t stream = streams[0];

        if(finishedTasks.size() > 0){
            CachedDeviceUVector<int> d_positions4(finishedTasks.size(), stream, *cubAllocator);
            CachedDeviceUVector<int> d_positionsNot4(finishedTasks.size(), stream, *cubAllocator);
            CachedDeviceUVector<int> d_numPositions(2, stream, *cubAllocator);

            helpers::call_fill_kernel_async(d_numPositions.data(), 2, 0, stream);

            if(computeTaskSplitGatherIndicesSmallInput.computationPossible(finishedTasks.size())){
                computeSplitGatherIndicesOfFinishedTasksSmall(
                    finishedTasks,
                    d_positions4.data(), 
                    d_positionsNot4.data(), 
                    d_numPositions.data(), 
                    d_numPositions.data() + 1,
                    stream
                );
            }else{
                computeSplitGatherIndicesOfFinishedTasksDefault(
                    finishedTasks,
                    d_positions4.data(), 
                    d_positionsNot4.data(), 
                    d_numPositions.data(), 
                    d_numPositions.data() + 1,
                    stream
                );
            }

            cudaMemcpyAsync(
                h_numPositions.data(),
                d_numPositions.data(),
                sizeof(int) * 2,
                D2H,
                stream
            ); CUERR;

            cudaStreamSynchronize(stream); CUERR;

            if(h_numPositions[0] > 0){

                TaskData gpufinishedTasks4 = finishedTasks.gather(
                    d_positions4.data(), 
                    d_positions4.data() + h_numPositions[0],
                    streams[0]
                );

                TaskData gpufinishedTasksNot4 = finishedTasks.gather(
                    d_positionsNot4.data(), 
                    d_positionsNot4.data() + h_numPositions[1],
                    streams[0]
                );

                std::swap(finishedTasks, gpufinishedTasksNot4);

                return gpufinishedTasks4;
            }else{
                return TaskData(*cubAllocator); //empty. no finished tasks to process
            }
        }else{
            return TaskData(*cubAllocator); //empty. no finished tasks to process
        }
    }

    void computeSplitGatherIndicesOfFinishedTasksSmall(
        const TaskData& finishedTasks,
        int* d_positions4, 
        int* d_positionsNot4, 
        int* d_numPositions4, 
        int* d_numPositionsNot4,
        cudaStream_t stream
    ){
        assert(computeTaskSplitGatherIndicesSmallInput.computationPossible(finishedTasks.size()));

        if(finishedTasks.size() == 0){
            cudaMemsetAsync(d_numPositions4, 0, sizeof(int), stream); CUERR;
            helpers::call_fill_kernel_async(d_numPositionsNot4, 1, int(finishedTasks.size()), stream); CUERR;
            return;
        }

        CachedDeviceUVector<int> d_minmax(2, stream, *cubAllocator);

        readextendergpukernels::minmaxSingleBlockKernel<512><<<1, 512, 0, stream>>>(
        //readextendergpukernels::minmaxSingleBlockKernel<128><<<1, 128, 0, stream>>>(
            finishedTasks.pairId.data(),
            finishedTasks.size(),
            d_minmax.data()
        ); CUERR;       

        computeTaskSplitGatherIndicesSmallInput.compute(
            finishedTasks.size(),
            d_positions4,
            d_positionsNot4,
            d_numPositions4,
            d_numPositionsNot4,
            finishedTasks.pairId.data(),
            finishedTasks.id.data(),
            d_minmax.data(),
            stream
        );
    }



    void computeSplitGatherIndicesOfFinishedTasksDefault(
        const TaskData& finishedTasks,
        int* d_positions4, 
        int* d_positionsNot4, 
        int* d_numPositions4, 
        int* d_numPositionsNot4,
        cudaStream_t stream
    ){
        if(finishedTasks.size() == 0){
            cudaMemsetAsync(d_numPositions4, 0, sizeof(int), stream);
            helpers::call_fill_kernel_async(d_numPositionsNot4, 1, int(finishedTasks.size()), stream);
            return;
        }

        readextendergpukernels::minmaxSingleBlockKernel<512><<<1, 512, 0, stream>>>(
        //readextendergpukernels::minmaxSingleBlockKernel<128><<<1, 128, 0, stream>>>(
            finishedTasks.pairId.data(),
            finishedTasks.size(),
            h_minmax.data()
        ); CUERR;

        cudaStreamSynchronize(stream); CUERR;

        CachedDeviceUVector<int> d_pairIds1(finishedTasks.size(), stream, *cubAllocator);
        CachedDeviceUVector<int> d_pairIds2(finishedTasks.size(), stream, *cubAllocator);
        CachedDeviceUVector<int> d_indices1(finishedTasks.size(), stream, *cubAllocator);
        CachedDeviceUVector<int> d_incices2(finishedTasks.size(), stream, *cubAllocator);

        //decrease pair ids by smallest pair id to improve radix sort performance
        readextendergpukernels::vectorAddKernel<<<SDIV(finishedTasks.size(), 128), 128, 0, stream>>>(
            finishedTasks.pairId.data(),
            thrust::make_constant_iterator(-h_minmax[0]),
            d_pairIds1.data(),
            finishedTasks.size()
        ); CUERR;

        readextendergpukernels::iotaKernel<<<SDIV(finishedTasks.size(), 128), 128, 0, stream>>>(
            d_indices1.begin(), 
            d_indices1.end(), 
            0
        ); CUERR;
       
        cub::DoubleBuffer<int> d_keys(d_pairIds1.data(), d_pairIds2.data());
        cub::DoubleBuffer<int> d_values(d_indices1.data(), d_incices2.data());

        const int begin_bit = 0;
        const int end_bit = std::ceil(std::log2(h_minmax[1] - h_minmax[0]));

        cudaError_t status = cudaSuccess;
        std::size_t tempbytes = 0;
        status = cub::DeviceRadixSort::SortPairs(
            nullptr,
            tempbytes,
            d_keys,
            d_values,
            finishedTasks.size(), 
            begin_bit, 
            end_bit, 
            stream
        );
        assert(cudaSuccess == status);

        CachedDeviceUVector<char> d_temp(tempbytes, stream, *cubAllocator);

        status = cub::DeviceRadixSort::SortPairs(
            d_temp.data(),
            tempbytes,
            d_keys,
            d_values,
            finishedTasks.size(), 
            begin_bit, 
            end_bit, 
            stream
        );
        assert(cudaSuccess == status);
        d_temp.destroy();       

        const int* d_theSortedPairIds = d_keys.Current();
        const int* d_theSortedIndices = d_values.Current();

        CachedDeviceUVector<int> d_counts_out(finishedTasks.size(), stream, *cubAllocator);
        CachedDeviceUScalar<int> d_num_runs_out(1, stream, *cubAllocator);

        CubCallWrapper(*cubAllocator).cubReduceByKey(
            d_theSortedPairIds, 
            cub::DiscardOutputIterator<>{},
            thrust::make_constant_iterator(1),
            d_counts_out.data(),
            d_num_runs_out.data(),
            thrust::plus<int>{},
            finishedTasks.size(),
            stream
        );

        d_pairIds1.destroy();
        d_pairIds2.destroy();

        //compute exclusive prefix sums + total to have stable output
        CachedDeviceUVector<int> d_outputoffsetsPos4(finishedTasks.size(), stream, *cubAllocator);
        CachedDeviceUVector<int> d_outputoffsetsNotPos4(finishedTasks.size(), stream, *cubAllocator);

        CubCallWrapper(*cubAllocator).cubExclusiveSum(
            thrust::make_transform_iterator(
                d_counts_out.data(),
                [] __host__ __device__ (int count){
                    if(count == 4){
                        return count;
                    }else{
                        return 0;
                    }
                }
            ),
            d_outputoffsetsPos4.data(),
            finishedTasks.size(),
            stream
        );

        CubCallWrapper(*cubAllocator).cubExclusiveSum(
            thrust::make_transform_iterator(
                d_counts_out.data(),
                [] __host__ __device__ (int count){
                    if(count != 4){
                        return count;
                    }else{
                        return 0;
                    }
                }
            ),
            d_outputoffsetsNotPos4.data(),
            finishedTasks.size(),
            stream
        );

        CubCallWrapper(*cubAllocator).cubInclusiveSum(
            d_counts_out.data(),
            d_counts_out.data(),
            finishedTasks.size(),
            stream
        );

        readextendergpukernels::computeTaskSplitGatherIndicesDefaultKernel<256><<<SDIV(finishedTasks.size(), 256), 256, 0, stream>>>(
            finishedTasks.size(),
            d_positions4,
            d_positionsNot4,
            d_numPositions4,
            d_numPositionsNot4,
            d_counts_out.data(),
            d_num_runs_out.data(),
            d_theSortedIndices,
            finishedTasks.id.data(),
            d_outputoffsetsPos4.data(),
            d_outputoffsetsNotPos4.data()
        ); CUERR;
    }

    void constructRawResults(TaskData& finishedTasks, RawExtendResult& rawResults, cudaStream_t callerstream){

        nvtx::push_range("constructRawResults", 5);
        std::lock_guard<std::mutex> lockguard(mutex);

        auto finishedTasks4 = getFinishedGpuSoaTasksOfFinishedPairsAndRemoveThem(finishedTasks);

        const int numFinishedTasks = finishedTasks4.size();
        rawResults.noCandidates = false;
        rawResults.decodedSequencePitchInBytes = decodedSequencePitchInBytes;
        rawResults.numResults = numFinishedTasks / 4;

        if(numFinishedTasks == 0){            
            return;
        }

        int resultMSAColumnPitchInElements = 1024;

        cudaStream_t stream = streams[0];
        cudaStream_t stream2 = streams[1];

        cudaEventRecord(events[0], callerstream); CUERR;
        cudaStreamWaitEvent(stream, events[0], 0); CUERR;
        cudaStreamWaitEvent(stream2, events[0], 0); CUERR;

        cudaMemcpyAsync(
            rawResults.h_tmp.data(),
            finishedTasks4.soaNumIterationResultsPerTaskPrefixSum.data() + numFinishedTasks,
            sizeof(int),
            D2H,
            stream
        ); CUERR;
        cudaEventRecord(rawResults.event, stream); CUERR;

        //copy data from device to host in second stream
        
        rawResults.h_gpuabortReasons.resize(numFinishedTasks);
        rawResults.h_gpudirections.resize(numFinishedTasks);
        rawResults.h_gpuiterations.resize(numFinishedTasks);
        rawResults.h_gpuReadIds.resize(numFinishedTasks);
        rawResults.h_gpuMateReadIds.resize(numFinishedTasks);
        rawResults.h_gpuAnchorLengths.resize(numFinishedTasks);
        rawResults.h_gpuMateLengths.resize(numFinishedTasks);
        rawResults.h_gpugoodscores.resize(numFinishedTasks);
        rawResults.h_gpuMateHasBeenFound.resize(numFinishedTasks);

        using care::gpu::MemcpyParams;

        auto memcpyParams1 = cuda::std::tuple_cat(
            cuda::std::make_tuple(MemcpyParams(rawResults.h_gpuabortReasons.data(), finishedTasks4.abortReason.data(), sizeof(extension::AbortReason) * numFinishedTasks)),
            cuda::std::make_tuple(MemcpyParams(rawResults.h_gpudirections.data(), finishedTasks4.direction.data(), sizeof(extension::ExtensionDirection) * numFinishedTasks)),
            cuda::std::make_tuple(MemcpyParams(rawResults.h_gpuiterations.data(), finishedTasks4.iteration.data(), sizeof(int) * numFinishedTasks)),
            cuda::std::make_tuple(MemcpyParams(rawResults.h_gpuReadIds.data(), finishedTasks4.myReadId.data(), sizeof(read_number) * numFinishedTasks)),
            cuda::std::make_tuple(MemcpyParams(rawResults.h_gpuMateReadIds.data(), finishedTasks4.mateReadId.data(), sizeof(read_number) * numFinishedTasks)),
            cuda::std::make_tuple(MemcpyParams(rawResults.h_gpuAnchorLengths.data(), finishedTasks4.soainputAnchorLengths.data(), sizeof(int) * numFinishedTasks)),
            cuda::std::make_tuple(MemcpyParams(rawResults.h_gpuMateLengths.data(), finishedTasks4.soainputmateLengths.data(), sizeof(int) * numFinishedTasks)),
            cuda::std::make_tuple(MemcpyParams(rawResults.h_gpugoodscores.data(), finishedTasks4.goodscore.data(), sizeof(float) * numFinishedTasks)),
            cuda::std::make_tuple(MemcpyParams(rawResults.h_gpuMateHasBeenFound.data(), finishedTasks4.mateHasBeenFound.data(), sizeof(bool) * numFinishedTasks))
        );

        care::gpu::memcpyKernel<int><<<SDIV(numFinishedTasks, 256), 256, 0, callerstream>>>(memcpyParams1); CUERR;
       
        cudaEventSynchronize(rawResults.event); CUERR;

        const int numCandidates = *rawResults.h_tmp;

        assert(numCandidates >= 0);

        //if there are no candidates, the resulting sequences will be identical to the input anchors. no computing required
        if(numCandidates == 0){
            //finishedTasks4.consistencyCheck(true);

            rawResults.h_inputAnchorsDecoded.resize(finishedTasks4.size() * decodedSequencePitchInBytes);

            assert(finishedTasks4.soainputAnchorsDecoded.size() >= sizeof(char) * finishedTasks4.size() * decodedSequencePitchInBytes);
            assert(rawResults.h_inputAnchorsDecoded.size() >= sizeof(char) * finishedTasks4.size() * decodedSequencePitchInBytes);

            cudaMemcpyAsync(
                rawResults.h_inputAnchorsDecoded.data(),
                finishedTasks4.soainputAnchorsDecoded.data(),
                sizeof(char) * finishedTasks4.size() * decodedSequencePitchInBytes,
                D2H,
                callerstream
            ); CUERR;

            rawResults.noCandidates = true;
            rawResults.decodedSequencePitchInBytes = decodedSequencePitchInBytes;

            return;
        }

        d_candidateSequencesData.resizeUninitialized(numCandidates * encodedSequencePitchInInts, stream);

        readextendergpukernels::encodeSequencesTo2BitKernel<8>
        <<<SDIV(numCandidates, (128 / 8)), 128, 0, stream>>>(
            d_candidateSequencesData.data(),
            finishedTasks4.soatotalDecodedAnchorsFlat.data(),
            finishedTasks4.soatotalDecodedAnchorsLengths.data(),
            decodedSequencePitchInBytes,
            encodedSequencePitchInInts,
            numCandidates
        ); CUERR;


        //sequence data has been transfered to gpu. now set up remaining msa input data

        d_alignment_overlaps.resizeUninitialized(numCandidates, stream);
        d_alignment_nOps.resizeUninitialized(numCandidates, stream);
        d_alignment_best_alignment_flags.resizeUninitialized(numCandidates, stream);
        d_isPairedCandidate.resizeUninitialized(numCandidates, stream);

        //fill the arrays such that msa will have good quality without pairedness
        readextendergpukernels::fillKernel<<<SDIV(numCandidates, 128), 128, 0, stream>>>(
            thrust::make_zip_iterator(thrust::make_tuple(
                d_alignment_overlaps.begin(),
                d_alignment_nOps.begin(),
                d_alignment_best_alignment_flags.begin(),
                d_isPairedCandidate.begin()
            )), 
            numCandidates, 
            thrust::make_tuple(
                100,
                0,
                BestAlignment_t::Forward,
                false
            )
        ); CUERR;
        
        //all input data ready. now set up msa

        CachedDeviceUVector<int> indices1(numCandidates, stream, *cubAllocator);

        const int threads = 32 * numFinishedTasks;

        readextendergpukernels::segmentedIotaKernel<32><<<SDIV(threads, 128), 128, 0, stream>>>(
            indices1.data(),
            numFinishedTasks,
            finishedTasks4.soaNumIterationResultsPerTask.data(),
            finishedTasks4.soaNumIterationResultsPerTaskPrefixSum.data()
        ); CUERR;

        *h_numAnchors = numFinishedTasks;

        multiMSA.construct(
            d_alignment_overlaps.data(),
            finishedTasks4.soatotalAnchorBeginInExtendedRead.data(),
            d_alignment_nOps.data(),
            d_alignment_best_alignment_flags.data(),
            indices1.data(),
            finishedTasks4.soaNumIterationResultsPerTask.data(),
            finishedTasks4.soaNumIterationResultsPerTaskPrefixSum.data(),
            finishedTasks4.soainputAnchorLengths.data(),
            finishedTasks4.inputAnchorsEncoded.data(),
            nullptr, //anchor qualities
            numFinishedTasks,
            finishedTasks4.soatotalDecodedAnchorsLengths.data(),
            d_candidateSequencesData.data(),
            nullptr, //candidate qualities
            d_isPairedCandidate.data(),
            numCandidates,
            h_numAnchors.data(), //d_numAnchors
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            false, //useQualityScores
            goodAlignmentProperties->maxErrorRate,
            gpu::MSAColumnCount::unknown(),
            stream
        );

        assert(multiMSA.numMSAs == numFinishedTasks);

        indices1.destroy();

        resultMSAColumnPitchInElements = multiMSA.getMaximumMsaWidth();

        //compute quality of consensus
        CachedDeviceUVector<char> d_consensusQuality(numFinishedTasks * resultMSAColumnPitchInElements, stream, *cubAllocator);
        CachedDeviceUVector<char> d_decodedConsensus(numFinishedTasks * resultMSAColumnPitchInElements, stream, *cubAllocator);
        CachedDeviceUVector<int> d_resultLengths(numFinishedTasks, stream, *cubAllocator);
        
        multiMSA.computeConsensusQuality(
            d_consensusQuality.data(),
            resultMSAColumnPitchInElements,
            stream
        );

        multiMSA.computeConsensus(
            d_decodedConsensus.data(),
            resultMSAColumnPitchInElements,
            stream
        );

        multiMSA.computeMsaSizes(
            d_resultLengths.data(),
            stream
        );

        cudaEventRecord(events[0], stream); CUERR;
        cudaStreamWaitEvent(stream2, events[0], 0); CUERR;

        const int numResults = numFinishedTasks / 4;

        CachedDeviceUVector<int> d_pairResultLengths(numResults, stream2, *cubAllocator);

        //in stream2, compute pair result output sizes and use them to determine required memory
        readextendergpukernels::makePairResultsFromFinishedTasksDryRunKernel<128><<<numResults, 128, 0, stream2>>>(
            numResults,
            d_pairResultLengths.data(),
            finishedTasks4.soainputAnchorLengths.data(), 
            d_resultLengths.data(),
            d_decodedConsensus.data(),
            d_consensusQuality.data(),
            finishedTasks4.mateHasBeenFound.data(),
            finishedTasks4.goodscore.data(),
            resultMSAColumnPitchInElements,
            insertSize,
            insertSizeStddev
        ); CUERR;

        readextendergpukernels::minmaxSingleBlockKernel<512><<<1, 512, 0, stream2>>>(
        //readextendergpukernels::minmaxSingleBlockKernel<128><<<1, 128, 0, stream2>>>(
            d_pairResultLengths.data(),
            numResults,
            h_minmax.data()
        ); CUERR;

        cudaEventRecord(h_numAnchorsEvent, stream2); CUERR;

        //replace positions which are covered by anchor and mate with the original data
        readextendergpukernels::applyOriginalReadsToExtendedReads<128,32>
        <<<SDIV(numFinishedTasks, 4), 128, 0, stream>>>(
            resultMSAColumnPitchInElements,
            numFinishedTasks,
            d_decodedConsensus.data(),
            d_consensusQuality.data(),
            d_resultLengths.data(),
            finishedTasks4.inputAnchorsEncoded.data(),
            finishedTasks4.soainputAnchorLengths.data(),
            finishedTasks4.soainputAnchorQualities.data(),
            finishedTasks4.mateHasBeenFound.data(),
            encodedSequencePitchInInts,
            qualityPitchInBytes
        ); CUERR;

        cudaEventSynchronize(h_numAnchorsEvent); CUERR;

        const int outputPitch = SDIV(h_minmax[1], 4) * 4; //round up maximum output size to 4 bytes

        CachedDeviceUVector<bool> d_pairResultAnchorIsLR(numResults, stream, *cubAllocator);
        CachedDeviceUVector<char> d_pairResultSequences(numResults * outputPitch, stream, *cubAllocator);
        CachedDeviceUVector<char> d_pairResultQualities(numResults * outputPitch, stream, *cubAllocator);
        CachedDeviceUVector<int> d_pairResultRead1Begins(numResults, stream, *cubAllocator);
        CachedDeviceUVector<int> d_pairResultRead2Begins(numResults, stream, *cubAllocator);
        CachedDeviceUVector<bool> d_pairResultMateHasBeenFound(numResults, stream, *cubAllocator);
        CachedDeviceUVector<bool> d_pairResultMergedDifferentStrands(numResults, stream, *cubAllocator);
        
        const std::size_t smem = 2 * outputPitch;

        readextendergpukernels::makePairResultsFromFinishedTasksKernel<128><<<numResults, 128, smem, stream>>>(
            numResults,
            d_pairResultAnchorIsLR.data(),
            d_pairResultSequences.data(),
            d_pairResultQualities.data(),
            d_pairResultLengths.data(),
            d_pairResultRead1Begins.data(),
            d_pairResultRead2Begins.data(),
            d_pairResultMateHasBeenFound.data(),
            d_pairResultMergedDifferentStrands.data(),
            outputPitch,
            finishedTasks4.soainputAnchorLengths.data(), 
            d_resultLengths.data(),
            d_decodedConsensus.data(),
            d_consensusQuality.data(),
            finishedTasks4.mateHasBeenFound.data(),
            finishedTasks4.goodscore.data(),
            resultMSAColumnPitchInElements,
            insertSize,
            insertSizeStddev
        ); CUERR;

        cudaEventRecord(events[0], stream); CUERR;
        cudaStreamWaitEvent(callerstream, events[0], 0); CUERR;

        rawResults.h_pairResultAnchorIsLR.resize(numResults);
        rawResults.h_pairResultSequences.resize(numResults * outputPitch);
        rawResults.h_pairResultQualities.resize(numResults * outputPitch);
        rawResults.h_pairResultLengths.resize(numResults);
        rawResults.h_pairResultRead1Begins.resize(numResults);
        rawResults.h_pairResultRead2Begins.resize(numResults);
        rawResults.h_pairResultMateHasBeenFound.resize(numResults);
        rawResults.h_pairResultMergedDifferentStrands.resize(numResults);

        auto memcpyParams2 = cuda::std::tuple_cat(
            cuda::std::make_tuple(MemcpyParams(rawResults.h_pairResultMateHasBeenFound.data(), d_pairResultMateHasBeenFound.data(), sizeof(bool) * numResults)),
            cuda::std::make_tuple(MemcpyParams(rawResults.h_pairResultMergedDifferentStrands.data(), d_pairResultMergedDifferentStrands.data(), sizeof(bool) * numResults)),
            cuda::std::make_tuple(MemcpyParams(rawResults.h_pairResultAnchorIsLR.data(), d_pairResultAnchorIsLR.data(), sizeof(bool) * numResults)),
            cuda::std::make_tuple(MemcpyParams(rawResults.h_pairResultSequences.data(), d_pairResultSequences.data(), sizeof(char) * outputPitch * numResults)),
            cuda::std::make_tuple(MemcpyParams(rawResults.h_pairResultQualities.data(), d_pairResultQualities.data(), sizeof(char) * outputPitch * numResults)),
            cuda::std::make_tuple(MemcpyParams(rawResults.h_pairResultLengths.data(), d_pairResultLengths.data(), sizeof(int) * numResults)),
            cuda::std::make_tuple(MemcpyParams(rawResults.h_pairResultRead1Begins.data(), d_pairResultRead1Begins.data(), sizeof(int) * numResults)),
            cuda::std::make_tuple(MemcpyParams(rawResults.h_pairResultRead2Begins.data(), d_pairResultRead2Begins.data(), sizeof(int) * numResults))
        );

        const int memcpyThreads = std::min(65536ul, (sizeof(char) * outputPitch * numResults) / sizeof(int));
        care::gpu::memcpyKernel<int><<<SDIV(memcpyThreads, 256), 256, 0, callerstream>>>(memcpyParams2); CUERR;

        rawResults.outputpitch = outputPitch;

        nvtx::pop_range();
    }

    std::vector<extension::ExtendResult> convertRawExtendResults(const RawExtendResult& rawResults) const{
        nvtx::push_range("convertRawExtendResults", 7);

        std::vector<extension::ExtendResult> gpuResultVector(rawResults.numResults);

        if(!rawResults.noCandidates){

            for(int k = 0; k < rawResults.numResults; k++){
                auto& gpuResult = gpuResultVector[k];

                const int index = k;

                const char* gpuSeq = &rawResults.h_pairResultSequences[k * rawResults.outputpitch];
                const char* gpuQual = &rawResults.h_pairResultQualities[k * rawResults.outputpitch];
                const int gpuLength = rawResults.h_pairResultLengths[k];
                const int read1begin = rawResults.h_pairResultRead1Begins[k];
                const int read2begin = rawResults.h_pairResultRead2Begins[k];
                const bool anchorIsLR = rawResults.h_pairResultAnchorIsLR[k]; 
                const bool mateHasBeenFound = rawResults.h_pairResultMateHasBeenFound[k];
                const bool mergedDifferentStrands = rawResults.h_pairResultMergedDifferentStrands[k];

                std::string s1(gpuSeq, gpuLength);
                std::string s2(gpuQual, gpuLength);

                const int i0 = 4 * index + 0;
                const int i2 = 4 * index + 2;

                int srcindex = i0;
                if(!anchorIsLR){
                    srcindex = i2;
                }

                if(mateHasBeenFound){
                    gpuResult.abortReason = extension::AbortReason::None;
                }else{
                    gpuResult.abortReason = rawResults.h_gpuabortReasons[srcindex];
                }

                gpuResult.direction = anchorIsLR ? extension::ExtensionDirection::LR : extension::ExtensionDirection::RL;
                gpuResult.numIterations = rawResults.h_gpuiterations[srcindex];
                gpuResult.aborted = gpuResult.abortReason != extension::AbortReason::None;
                gpuResult.readId1 = rawResults.h_gpuReadIds[srcindex];
                gpuResult.readId2 = rawResults.h_gpuMateReadIds[srcindex];
                gpuResult.originalLength = rawResults.h_gpuAnchorLengths[srcindex];
                gpuResult.originalMateLength = rawResults.h_gpuMateLengths[srcindex];
                gpuResult.read1begin = read1begin;
                gpuResult.goodscore = rawResults.h_gpugoodscores[srcindex];
                gpuResult.read2begin = read2begin;
                gpuResult.mateHasBeenFound = mateHasBeenFound;
                gpuResult.extendedRead = std::move(s1);
                gpuResult.qualityScores = std::move(s2);
                gpuResult.mergedFromReadsWithoutMate = mergedDifferentStrands;
            }
        }else{
            for(int p = 0; p < rawResults.numResults; p++){
                //LR search
                const int i0 = 4 * p + 0;

                auto& result = gpuResultVector[p];

                result.direction = extension::ExtensionDirection::LR;
                result.numIterations = rawResults.h_gpuiterations[i0];
                result.aborted = rawResults.h_gpuabortReasons[i0] != extension::AbortReason::None;
                result.abortReason = rawResults.h_gpuabortReasons[i0];
                result.readId1 = rawResults.h_gpuReadIds[i0];
                result.readId2 = rawResults.h_gpuMateReadIds[i0];
                result.originalLength = rawResults.h_gpuAnchorLengths[i0];
                result.originalMateLength = rawResults.h_gpuMateLengths[i0];
                result.read1begin = 0;
                result.goodscore = rawResults.h_gpugoodscores[i0];
                result.read2begin = -1;
                result.mateHasBeenFound = false;
                result.extendedRead.assign(
                    rawResults.h_inputAnchorsDecoded.begin() + i0 * rawResults.decodedSequencePitchInBytes,
                    rawResults.h_inputAnchorsDecoded.begin() + i0 * rawResults.decodedSequencePitchInBytes + rawResults.h_gpuAnchorLengths[i0]
                );
                result.qualityScores.resize(rawResults.h_gpuAnchorLengths[i0]);
                std::fill(result.qualityScores.begin(), result.qualityScores.end(), 'I');
            }
        }

        nvtx::pop_range();

        return gpuResultVector;
    }




    //helpers

    void setGpuSegmentIds(
        int* d_segmentIds, //size >= maxNumElements
        int numSegments,
        int maxNumElements,
        const int* d_numElementsPerSegment,
        const int* d_numElementsPerSegmentPrefixSum,
        cudaStream_t stream
    ) const {
        cudaMemsetAsync(d_segmentIds, 0, sizeof(int) * maxNumElements, stream); CUERR;
        
        readextendergpukernels::setFirstSegmentIdsKernel<<<SDIV(numSegments, 256), 256, 0, stream>>>(
            d_numElementsPerSegment,
            d_segmentIds,
            d_numElementsPerSegmentPrefixSum,
            numSegments
        ); CUERR;

        CubCallWrapper(*cubAllocator).cubInclusiveScan(
            d_segmentIds, 
            d_segmentIds, 
            cub::Max{},
            maxNumElements,
            stream
        );
    }

    void loadCandidateQualityScores(cudaStream_t stream, char* d_qualityscores){
        char* outputQualityScores = d_qualityscores;

        if(correctionOptions->useQualityScores){

            //cudaEventSynchronize(h_candidateReadIdsEvent); CUERR;
            cudaEventSynchronize(h_numCandidatesEvent); CUERR;

            gpuReadStorage->gatherQualities(
                readStorageHandle,
                outputQualityScores,
                qualityPitchInBytes,
                makeAsyncConstBufferWrapper(h_candidateReadIds.data(), h_candidateReadIdsEvent),
                d_candidateReadIds.data(),
                *h_numCandidates,
                stream
            );

        }else{
            helpers::call_fill_kernel_async(
                outputQualityScores,
                qualityPitchInBytes * initialNumCandidates,
                'I',
                stream
            ); CUERR;
        }        
    }

    void compactCandidateDataByFlagsExcludingAlignments(
        const bool* d_keepFlags,
        bool updateHostCandidateReadIds,
        cudaStream_t stream
    ){
        CachedDeviceUVector<int> d_numCandidatesPerAnchor2(tasks->size(), stream, *cubAllocator);

        CubCallWrapper(*cubAllocator).cubSegmentedReduceSum(
            d_keepFlags,
            d_numCandidatesPerAnchor2.data(),
            tasks->size(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_numCandidatesPerAnchorPrefixSum.data() + 1,
            stream
        );

        auto d_zip_data = thrust::make_zip_iterator(
            thrust::make_tuple(
                d_candidateReadIds.data(),
                d_candidateSequencesLength.data(),
                d_isPairedCandidate.data()
            )
        );

        CachedDeviceUVector<int> d_candidateSequencesLength2(initialNumCandidates, stream, *cubAllocator);
        CachedDeviceUVector<read_number> d_candidateReadIds2(initialNumCandidates, stream, *cubAllocator);
        CachedDeviceUVector<bool> d_isPairedCandidate2(initialNumCandidates, stream, *cubAllocator);
  
        auto d_zip_data_tmp = thrust::make_zip_iterator(
            thrust::make_tuple(
                d_candidateReadIds2.data(),
                d_candidateSequencesLength2.data(),
                d_isPairedCandidate2.data()
            )
        );

        cudaEventSynchronize(h_numCandidatesEvent); CUERR;
        const int currentNumCandidates = *h_numCandidates;

        //compact 1d arrays

        CubCallWrapper(*cubAllocator).cubSelectFlagged(
            d_zip_data, 
            d_keepFlags, 
            d_zip_data_tmp, 
            h_numCandidates.data(), 
            initialNumCandidates, 
            stream
        );

        cudaEventRecord(h_numCandidatesEvent, stream); CUERR;

        if(updateHostCandidateReadIds){
            cudaStreamWaitEvent(hostOutputStream, h_numCandidatesEvent, 0); CUERR;           

            cudaMemcpyAsync(
                h_candidateReadIds.data(),
                d_candidateReadIds2.data(),
                sizeof(read_number) * currentNumCandidates,
                D2H,
                hostOutputStream
            ); CUERR;

            cudaEventRecord(h_candidateReadIdsEvent, hostOutputStream); CUERR;  
        }

        cudaMemsetAsync(d_numCandidatesPerAnchorPrefixSum.data(), 0, sizeof(int), stream); CUERR;
        CubCallWrapper(*cubAllocator).cubInclusiveSum(
            d_numCandidatesPerAnchor2.data(), 
            d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            tasks->size(), 
            stream
        );
        std::swap(d_numCandidatesPerAnchor, d_numCandidatesPerAnchor2); 

        d_numCandidatesPerAnchor2.destroy();

        std::swap(d_candidateReadIds, d_candidateReadIds2);
        std::swap(d_candidateSequencesLength, d_candidateSequencesLength2);
        std::swap(d_isPairedCandidate, d_isPairedCandidate2);

        d_candidateSequencesLength2.destroy();
        d_candidateReadIds2.destroy();
        d_isPairedCandidate2.destroy();
        
        //update candidate sequences data
        CachedDeviceUVector<unsigned int> d_candidateSequencesData2(encodedSequencePitchInInts * initialNumCandidates, stream, *cubAllocator);

        CubCallWrapper(*cubAllocator).cubSelectFlagged(
            d_candidateSequencesData.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                make_iterator_multiplier(d_keepFlags, encodedSequencePitchInInts)
            ),
            d_candidateSequencesData2.data(),
            thrust::make_discard_iterator(),
            initialNumCandidates * encodedSequencePitchInInts,
            stream
        );

        std::swap(d_candidateSequencesData, d_candidateSequencesData2);
        d_candidateSequencesData2.destroy();
    }


    void compactCandidateDataByFlags(
        const bool* d_keepFlags,
        bool updateHostCandidateReadIds,
        cudaStream_t stream
    ){
        CachedDeviceUVector<int> d_numCandidatesPerAnchor2(tasks->size(), stream, *cubAllocator);

        CubCallWrapper(*cubAllocator).cubSegmentedReduceSum(
            d_keepFlags,
            d_numCandidatesPerAnchor2.data(),
            tasks->size(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_numCandidatesPerAnchorPrefixSum.data() + 1,
            stream
        );

        auto d_zip_data = thrust::make_zip_iterator(
            thrust::make_tuple(
                d_alignment_nOps.data(),
                d_alignment_overlaps.data(),
                d_alignment_shifts.data(),
                d_alignment_best_alignment_flags.data(),
                d_candidateReadIds.data(),
                d_candidateSequencesLength.data(),
                d_isPairedCandidate.data()
            )
        );

        CachedDeviceUVector<int> d_alignment_overlaps2(initialNumCandidates, stream, *cubAllocator);
        CachedDeviceUVector<int> d_alignment_shifts2(initialNumCandidates, stream, *cubAllocator);
        CachedDeviceUVector<int> d_alignment_nOps2(initialNumCandidates, stream, *cubAllocator);
        CachedDeviceUVector<BestAlignment_t> d_alignment_best_alignment_flags2(initialNumCandidates, stream, *cubAllocator);
        CachedDeviceUVector<int> d_candidateSequencesLength2(initialNumCandidates, stream, *cubAllocator);
        CachedDeviceUVector<read_number> d_candidateReadIds2(initialNumCandidates, stream, *cubAllocator);
        CachedDeviceUVector<bool> d_isPairedCandidate2(initialNumCandidates, stream, *cubAllocator);
  
        auto d_zip_data_tmp = thrust::make_zip_iterator(
            thrust::make_tuple(
                d_alignment_nOps2.data(),
                d_alignment_overlaps2.data(),
                d_alignment_shifts2.data(),
                d_alignment_best_alignment_flags2.data(),
                d_candidateReadIds2.data(),
                d_candidateSequencesLength2.data(),
                d_isPairedCandidate2.data()
            )
        );

        cudaEventSynchronize(h_numCandidatesEvent); CUERR;
        const int currentNumCandidates = *h_numCandidates;

        //compact 1d arrays

        CubCallWrapper(*cubAllocator).cubSelectFlagged(
            d_zip_data, 
            d_keepFlags, 
            d_zip_data_tmp, 
            h_numCandidates.data(), 
            initialNumCandidates, 
            stream
        );

        cudaEventRecord(h_numCandidatesEvent, stream); CUERR;

        if(updateHostCandidateReadIds){
            cudaStreamWaitEvent(hostOutputStream, h_numCandidatesEvent, 0); CUERR;           

            cudaMemcpyAsync(
                h_candidateReadIds.data(),
                d_candidateReadIds2.data(),
                sizeof(read_number) * currentNumCandidates,
                D2H,
                hostOutputStream
            ); CUERR;

            cudaEventRecord(h_candidateReadIdsEvent, hostOutputStream); CUERR;  
        }

        cudaMemsetAsync(d_numCandidatesPerAnchorPrefixSum.data(), 0, sizeof(int), stream); CUERR;
        CubCallWrapper(*cubAllocator).cubInclusiveSum(
            d_numCandidatesPerAnchor2.data(), 
            d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            tasks->size(), 
            stream
        );
        std::swap(d_numCandidatesPerAnchor, d_numCandidatesPerAnchor2); 

        d_numCandidatesPerAnchor2.destroy();

        std::swap(d_alignment_nOps, d_alignment_nOps2);
        std::swap(d_alignment_overlaps, d_alignment_overlaps2);
        std::swap(d_alignment_shifts, d_alignment_shifts2);
        std::swap(d_alignment_best_alignment_flags, d_alignment_best_alignment_flags2);
        std::swap(d_candidateReadIds, d_candidateReadIds2);
        std::swap(d_candidateSequencesLength, d_candidateSequencesLength2);
        std::swap(d_isPairedCandidate, d_isPairedCandidate2);

        d_alignment_overlaps2.destroy();
        d_alignment_shifts2.destroy();
        d_alignment_nOps2.destroy();
        d_alignment_best_alignment_flags2.destroy();
        d_candidateSequencesLength2.destroy();
        d_candidateReadIds2.destroy();
        d_isPairedCandidate2.destroy();
        
        //update candidate sequences data
        CachedDeviceUVector<unsigned int> d_candidateSequencesData2(encodedSequencePitchInInts * initialNumCandidates, stream, *cubAllocator);

        CubCallWrapper(*cubAllocator).cubSelectFlagged(
            d_candidateSequencesData.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                make_iterator_multiplier(d_keepFlags, encodedSequencePitchInInts)
            ),
            d_candidateSequencesData2.data(),
            thrust::make_discard_iterator(),
            initialNumCandidates * encodedSequencePitchInInts,
            stream
        );

        std::swap(d_candidateSequencesData, d_candidateSequencesData2);
        d_candidateSequencesData2.destroy();

        //update candidate quality scores
        // assert(qualityPitchInBytes % sizeof(int) == 0);
        // CachedDeviceUVector<char> d_candidateQualities2(qualityPitchInBytes * initialNumCandidates, stream, *cubAllocator);

        // cubSelectFlagged(
        //     (const int*)d_candidateQualityScores.data(),
        //     thrust::make_transform_iterator(
        //         thrust::make_counting_iterator(0),
        //         make_iterator_multiplier(d_keepFlags, qualityPitchInBytes / sizeof(int))
        //     ),
        //     (int*)d_candidateQualities2.data(),
        //     thrust::make_discard_iterator(),
        //     initialNumCandidates * qualityPitchInBytes / sizeof(int),
        //     firstStream
        // );

        // std::swap(d_candidateQualityScores, d_candidateQualities2);
    }

    void setStateToFinished(){
        //addFinishedGpuSoaTasks(tasks, streams[0]);
        tasks->clear(streams[0]);

        cudaStreamSynchronize(streams[0]); CUERR;

        setState(GpuReadExtender::State::Finished);
    }
    
    void addFinishedGpuSoaTasks(TaskData& tasksToAdd, TaskData& finishedTasks, cudaStream_t stream){
        finishedTasks.append(tasksToAdd, stream);
        //std::cerr << "addFinishedSoaTasks. soaFinishedTasks size " << soaFinishedTasks.entries << "\n";
    }

    void copyHashResultsFrom(const AnchorHashResult& results, cudaStream_t stream){
        d_candidateReadIds.resizeUninitialized(results.d_candidateReadIds.size(), stream);
        d_numCandidatesPerAnchor.resizeUninitialized(results.d_numCandidatesPerAnchor.size(), stream);
        d_numCandidatesPerAnchorPrefixSum.resizeUninitialized(results.d_numCandidatesPerAnchorPrefixSum.size(), stream);

        cudaMemcpyAsync(
            d_candidateReadIds.data(),
            results.d_candidateReadIds.data(),
            sizeof(read_number) * d_candidateReadIds.size(),
            D2D,
            stream
        ); CUERR;

        cudaMemcpyAsync(
            d_numCandidatesPerAnchor.data(),
            results.d_numCandidatesPerAnchor.data(),
            sizeof(int) * d_numCandidatesPerAnchor.size(),
            D2D,
            stream
        ); CUERR;

        cudaMemcpyAsync(
            d_numCandidatesPerAnchorPrefixSum.data(),
            results.d_numCandidatesPerAnchorPrefixSum.data(),
            sizeof(int) * d_numCandidatesPerAnchorPrefixSum.size(),
            D2D,
            stream
        ); CUERR;
    }

    void copyAnchorDataFrom(const AnchorData& results, cudaStream_t stream){
        d_subjectSequencesDataDecoded.resizeUninitialized(results.d_subjectSequencesDataDecoded.size(), stream);
        d_anchorQualityScores.resizeUninitialized(results.d_anchorQualityScores.size(), stream);
        d_anchorSequencesLength.resizeUninitialized(results.d_anchorSequencesLength.size(), stream);
        d_subjectSequencesData.resizeUninitialized(results.d_subjectSequencesData.size(), stream);
        d_accumExtensionsLengths.resizeUninitialized(results.d_accumExtensionsLengths.size(), stream);

        cudaMemcpyAsync(
            d_subjectSequencesDataDecoded.data(),
            results.d_subjectSequencesDataDecoded.data(),
            sizeof(char) * d_subjectSequencesDataDecoded.size(),
            D2D,
            stream
        ); CUERR;

        cudaMemcpyAsync(
            d_anchorQualityScores.data(),
            results.d_anchorQualityScores.data(),
            sizeof(char) * d_anchorQualityScores.size(),
            D2D,
            stream
        ); CUERR;

        cudaMemcpyAsync(
            d_anchorSequencesLength.data(),
            results.d_anchorSequencesLength.data(),
            sizeof(int) * d_anchorSequencesLength.size(),
            D2D,
            stream
        ); CUERR;

        cudaMemcpyAsync(
            d_subjectSequencesData.data(),
            results.d_subjectSequencesData.data(),
            sizeof(unsigned int) * d_subjectSequencesData.size(),
            D2D,
            stream
        ); CUERR;

        cudaMemcpyAsync(
            d_accumExtensionsLengths.data(),
            results.d_accumExtensionsLengths.data(),
            sizeof(int) * d_accumExtensionsLengths.size(),
            D2D,
            stream
        ); CUERR;
    }


    bool pairedEnd = false;
    State state = State::None;
    int someId = 0;
    int alltimeMaximumNumberOfTasks = 0;
    std::size_t alltimetotalTaskBytes = 0;

    int initialNumCandidates = 0;

    int deviceId{};
    int insertSize{};
    int insertSizeStddev{};

    cub::CachingDeviceAllocator* cubAllocator{};
    const gpu::GpuReadStorage* gpuReadStorage{};
    const CorrectionOptions* correctionOptions{};
    const GoodAlignmentProperties* goodAlignmentProperties{};
    const cpu::QualityScoreConversion* qualityConversion{};
    mutable ReadStorageHandle readStorageHandle{};
    mutable gpu::KernelLaunchHandle kernelLaunchHandle{};

    std::size_t encodedSequencePitchInInts = 0;
    std::size_t decodedSequencePitchInBytes = 0;
    std::size_t msaColumnPitchInElements = 0;
    std::size_t qualityPitchInBytes = 0;

    std::size_t outputAnchorPitchInBytes = 0;
    std::size_t outputAnchorQualityPitchInBytes = 0;
    std::size_t decodedMatesRevCPitchInBytes = 0;

    const IterationConfig* iterationConfig{};

    
    PinnedBuffer<read_number> h_candidateReadIds{};
    PinnedBuffer<int> h_numCandidatesPerAnchor{};
    PinnedBuffer<int> h_numCandidatesPerAnchorPrefixSum{};
    PinnedBuffer<int> h_numAnchors{};
    PinnedBuffer<int> h_numCandidates{};
    PinnedBuffer<int> h_numAnchorsWithRemovedMates{};
    PinnedBuffer<int> h_minmax{};
    PinnedBuffer<int> h_numPositions{};

    CachedDeviceUScalar<int> d_numAnchors{};
    CachedDeviceUScalar<int> d_numCandidates{};
    CachedDeviceUVector<bool> d_mateIdHasBeenRemoved{};



    // ----- candidate data
    AnchorHashResult anchorHashResult;
    CachedDeviceUVector<unsigned int> d_candidateSequencesData{};
    CachedDeviceUVector<int> d_candidateSequencesLength{};    
    CachedDeviceUVector<read_number> d_candidateReadIds{};
    CachedDeviceUVector<bool> d_isPairedCandidate{};
    CachedDeviceUVector<int> d_alignment_overlaps{};
    CachedDeviceUVector<int> d_alignment_shifts{};
    CachedDeviceUVector<int> d_alignment_nOps{};
    CachedDeviceUVector<BestAlignment_t> d_alignment_best_alignment_flags{};

    CachedDeviceUVector<int> d_numCandidatesPerAnchor{};
    CachedDeviceUVector<int> d_numCandidatesPerAnchorPrefixSum{};
    // ----- 
    
    // ----- input data
    AnchorData anchorData;
    CachedDeviceUVector<char> d_subjectSequencesDataDecoded{};
    CachedDeviceUVector<char> d_anchorQualityScores{};
    CachedDeviceUVector<int> d_anchorSequencesLength{};
    CachedDeviceUVector<unsigned int> d_subjectSequencesData{};
    CachedDeviceUVector<int> d_accumExtensionsLengths{};
    // -----

    
    // ----- MSA data
    gpu::ManagedGPUMultiMSA multiMSA;
    // -----


    // ----- Ready-events for pinned outputs
    CudaEvent h_numAnchorsEvent{};
    CudaEvent h_numCandidatesEvent{};
    CudaEvent h_numAnchorsWithRemovedMatesEvent{};
    CudaEvent h_numUsedReadIdsEvent{};
    CudaEvent h_numFullyUsedReadIdsEvent{};
    CudaEvent h_numFullyUsedReadIds2Event{};
    CudaEvent h_candidateReadIdsEvent{};

    // -----

    CudaStream hostOutputStream{};

    readextendergpukernels::ComputeTaskSplitGatherIndicesSmallInput computeTaskSplitGatherIndicesSmallInput{};
    
    std::array<CudaEvent, 1> events{};
    std::array<cudaStream_t, 4> streams{};

    TaskData* tasks;
    
    std::mutex mutex{};
};


} //namespace gpu
} //namespace care


#endif