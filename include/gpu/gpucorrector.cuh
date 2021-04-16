#ifndef CARE_GPUCORRECTOR_CUH
#define CARE_GPUCORRECTOR_CUH


#include <hpc_helpers.cuh>
#include <hpc_helpers/include/nvtx_markers.cuh>

#include <gpu/gpuminhasher.cuh>
#include <gpu/kernels.hpp>
#include <gpu/kernellaunch.hpp>
#include <gpu/gpucorrectorkernels.cuh>
#include <gpu/cudagraphhelpers.cuh>
#include <gpu/gpureadstorage.cuh>

#include <config.hpp>
#include <util.hpp>
#include <corrector_common.hpp>
#include <threadpool.hpp>

#include <options.hpp>
#include <correctionresultprocessing.hpp>
#include <memorymanagement.hpp>
#include <msa.hpp>
#include <classification.hpp>

#include <forest.hpp>
#include <gpu/forest_gpu.cuh>

#include <algorithm>
#include <array>
#include <map>

#include <cub/cub.cuh>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>


namespace care{
namespace gpu{

    struct SequenceFlagMultiplier{
        int pitch{};
        const bool* flags{};

        __host__ __device__
        SequenceFlagMultiplier(const bool* flags_, int pitch_)
            :pitch(pitch_), flags(flags_){

        }

        __host__ __device__
        bool operator()(int i) const{
            return flags[i / pitch];
        }
    };

    template<class Iter>
    struct IteratorMultiplier{
        using value_type = typename std::iterator_traits<Iter>::value_type;

        int factor{};
        Iter data{};

        __host__ __device__
        IteratorMultiplier(Iter data_, int factor_)
            : factor(factor_), data(data_){

        }

        __host__ __device__
        value_type operator()(int i) const{
            return *(data + (i / factor));
        }
    };

    template<class Iter>
    IteratorMultiplier<Iter> make_iterator_multiplier(Iter data, int factor){
        return IteratorMultiplier<Iter>{data, factor};
    }

    struct ReplaceNumberOp{
        int doNotUseEditsValue;
        int decodedSequencePitchInBytes;

        ReplaceNumberOp(int val, int pitch) 
            : doNotUseEditsValue(val), decodedSequencePitchInBytes(pitch)
        {}

        __host__ __device__
        int operator()(const int num) const noexcept{
            return num == doNotUseEditsValue ? decodedSequencePitchInBytes : 0;
        }
    };


    class GpuErrorCorrectorInput{
    public:
        template<class T>
        using PinnedBuffer = helpers::SimpleAllocationPinnedHost<T>;

        template<class T>
        using DeviceBuffer = helpers::SimpleAllocationDevice<T>;
        //using DeviceBuffer = helpers::SimpleAllocationPinnedHost<T>;


        CudaEvent event{cudaEventDisableTiming};

        PinnedBuffer<int> h_numAnchors;
        PinnedBuffer<int> h_numCandidates;
        PinnedBuffer<read_number> h_anchorReadIds;
        PinnedBuffer<read_number> h_candidate_read_ids;

        DeviceBuffer<int> d_numAnchors;
        DeviceBuffer<int> d_numCandidates;
        DeviceBuffer<read_number> d_anchorReadIds;
        DeviceBuffer<unsigned int> d_anchor_sequences_data;
        DeviceBuffer<int> d_anchor_sequences_lengths;
        DeviceBuffer<read_number> d_candidate_read_ids;
        DeviceBuffer<int> d_candidates_per_anchor;
        DeviceBuffer<int> d_candidates_per_anchor_prefixsum;  
        DeviceBuffer<int> d_candidatesBeginOffsets;

        MemoryUsage getMemoryInfo() const{
            MemoryUsage info{};
            auto handleHost = [&](const auto& h){
                info.host += h.sizeInBytes();
            };
            auto handleDevice = [&](const auto& d){
                info.device[event.getDeviceId()] += d.sizeInBytes();
            };

            handleHost(h_numAnchors);
            handleHost(h_numCandidates);
            handleHost(h_anchorReadIds);
            handleHost(h_candidate_read_ids);

            handleDevice(d_numAnchors);
            handleDevice(d_numCandidates);
            handleDevice(d_anchorReadIds);
            handleDevice(d_anchor_sequences_data);
            handleDevice(d_anchor_sequences_lengths);
            handleDevice(d_candidate_read_ids);
            handleDevice(d_candidates_per_anchor);
            handleDevice(d_candidates_per_anchor_prefixsum);

            return info;
        }  
    };

    class GpuErrorCorrectorRawOutput{
    public:
        template<class T>
        using PinnedBuffer = helpers::SimpleAllocationPinnedHost<T>;

        template<class T>
        using DeviceBuffer = helpers::SimpleAllocationDevice<T>;

        bool nothingToDo;
        int numAnchors;
        int numCandidates;
        int doNotUseEditsValue;
        std::size_t editsPitchInBytes;
        std::size_t decodedSequencePitchInBytes;
        CudaEvent event{cudaEventDisableTiming};
        PinnedBuffer<read_number> h_anchorReadIds;
        PinnedBuffer<read_number> h_candidate_read_ids;
        PinnedBuffer<bool> h_anchor_is_corrected;
        PinnedBuffer<AnchorHighQualityFlag> h_is_high_quality_anchor;
        PinnedBuffer<int> h_num_corrected_candidates_per_anchor;
        PinnedBuffer<int> h_num_corrected_candidates_per_anchor_prefixsum;
        PinnedBuffer<int> h_indices_of_corrected_candidates;

        PinnedBuffer<int> h_candidate_sequences_lengths;
        PinnedBuffer<int> h_numEditsPerCorrectedanchor;
        PinnedBuffer<TempCorrectedSequence::EncodedEdit> h_editsPerCorrectedanchor;
        PinnedBuffer<char> h_corrected_anchors;
        PinnedBuffer<int> h_anchor_sequences_lengths;
        PinnedBuffer<char> h_corrected_candidates;
        PinnedBuffer<int> h_alignment_shifts;
        PinnedBuffer<int> h_numEditsPerCorrectedCandidate;
        PinnedBuffer<TempCorrectedSequence::EncodedEdit> h_editsPerCorrectedCandidate;
        PinnedBuffer<int> h_anchorEditOffsets;
        PinnedBuffer<int> h_correctedAnchorsOffsets;
        PinnedBuffer<int> h_candidateEditOffsets;
        PinnedBuffer<int> h_correctedCandidatesOffsets;

        MemoryUsage getMemoryInfo() const{
            MemoryUsage info{};
            auto handleHost = [&](const auto& h){
                info.host += h.sizeInBytes();
            };

            handleHost(h_anchorReadIds);
            handleHost(h_candidate_read_ids);
            handleHost(h_anchor_is_corrected);
            handleHost(h_is_high_quality_anchor);
            handleHost(h_num_corrected_candidates_per_anchor);
            handleHost(h_num_corrected_candidates_per_anchor_prefixsum);
            handleHost(h_indices_of_corrected_candidates);
            handleHost(h_candidate_sequences_lengths);
            handleHost(h_numEditsPerCorrectedanchor);
            handleHost(h_editsPerCorrectedanchor);
            handleHost(h_corrected_anchors);
            handleHost(h_anchor_sequences_lengths);
            handleHost(h_corrected_candidates);
            handleHost(h_alignment_shifts);
            handleHost(h_numEditsPerCorrectedCandidate);
            handleHost(h_editsPerCorrectedCandidate);
            handleHost(h_anchorEditOffsets);
            handleHost(h_correctedAnchorsOffsets);
            handleHost(h_candidateEditOffsets);
            handleHost(h_correctedCandidatesOffsets);

            return info;
        }  
    };



    class GpuAnchorHasher{
    public:

        GpuAnchorHasher() = default;

        GpuAnchorHasher(
            const GpuReadStorage& gpuReadStorage_,
            const GpuMinhasher& gpuMinhasher_,
            ThreadPool* threadPool_
        ) : 
            gpuReadStorage{&gpuReadStorage_},
            gpuMinhasher{&gpuMinhasher_},
            threadPool{threadPool_},
            minhashHandle{gpuMinhasher->makeQueryHandle()},
            readstorageHandle{gpuReadStorage->makeHandle()}
        {
            cudaGetDevice(&deviceId); CUERR;            

            maxCandidatesPerRead = gpuMinhasher->getNumResultsPerMapThreshold() * gpuMinhasher->getNumberOfMaps();

            backgroundStream = CudaStream{};
            previousBatchFinishedEvent = CudaEvent{};

            encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(gpuReadStorage->getSequenceLengthUpperBound());
        }

        ~GpuAnchorHasher(){
            // std::cerr << "GpuAnchorHasher::~GpuAnchorHasher(). Memory of minhash handle: ";
            // auto memoryUsage = gpuMinhasher->getMemoryInfo(minhashHandle);
            // std::cerr << memoryUsage.host;
            // for(auto pair : memoryUsage.device){
            //     std::cerr << ", [" << pair.first << "] " << pair.second;
            // }
            // std::cerr << "\n";

            gpuReadStorage->destroyHandle(readstorageHandle);
            gpuMinhasher->destroyHandle(minhashHandle);
        }

        void makeErrorCorrectorInput(
            const read_number* anchorIds,
            int numIds,
            GpuErrorCorrectorInput& ecinput,
            cudaStream_t stream
        ){
            int curId = 0;
            cudaGetDevice(&curId); CUERR;
            cudaSetDevice(deviceId); CUERR;

            assert(cudaSuccess == ecinput.event.query());
            previousBatchFinishedEvent.synchronize();

            resizeBuffers(ecinput, numIds);
    
            //copy input to pinned memory
            *ecinput.h_numAnchors.get() = numIds;
            std::copy_n(anchorIds, numIds, ecinput.h_anchorReadIds.get());

            cudaMemcpyAsync(
                ecinput.d_numAnchors.get(),
                ecinput.h_numAnchors.get(),
                sizeof(int),
                H2D,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                ecinput.d_anchorReadIds.get(),
                ecinput.h_anchorReadIds.get(),
                sizeof(read_number) * (*ecinput.h_numAnchors.get()),
                H2D,
                stream
            ); CUERR;

            if(numIds > 0){
                nvtx::push_range("getAnchorReads", 0);
                getAnchorReads(ecinput, stream);
                nvtx::pop_range();

                nvtx::push_range("getCandidateReadIdsWithMinhashing", 1);
                getCandidateReadIdsWithMinhashing(ecinput, stream);
                nvtx::pop_range();
            }            

            ecinput.event.record(stream);
            previousBatchFinishedEvent.record(stream);

            cudaSetDevice(curId); CUERR;
        }

        MemoryUsage getMemoryInfo() const{
            MemoryUsage info{};
       
            info += gpuMinhasher->getMemoryInfo(minhashHandle);
          
            info += gpuReadStorage->getMemoryInfo(readstorageHandle);
            return info;
        } 

    public: //private:
        void resizeBuffers(GpuErrorCorrectorInput& ecinput, int numAnchors){
            const std::size_t maxCandidates = maxCandidatesPerRead * numAnchors;
            // large enough to store all minhash results
            ecinput.h_candidate_read_ids.resize(maxCandidates);
            ecinput.d_candidate_read_ids.resize(maxCandidates); 

            ecinput.h_numAnchors.resize(1);
            ecinput.h_numCandidates.resize(1);
            ecinput.h_anchorReadIds.resize(numAnchors);

            ecinput.d_numAnchors.resize(1);
            ecinput.d_numCandidates.resize(1);
            ecinput.d_anchorReadIds.resize(numAnchors);
            ecinput.d_anchor_sequences_data.resize(encodedSequencePitchInInts * numAnchors);
            ecinput.d_anchor_sequences_lengths.resize(numAnchors);
            ecinput.d_candidates_per_anchor.resize(numAnchors);
            ecinput.d_candidates_per_anchor_prefixsum.resize(numAnchors + 1);
            ecinput.d_candidatesBeginOffsets.resize(numAnchors);
        }
        
        void getAnchorReads(GpuErrorCorrectorInput& ecinput, cudaStream_t stream){
            gpuReadStorage->gatherSequences(
                readstorageHandle,
                ecinput.d_anchor_sequences_data.get(),
                encodedSequencePitchInInts,
                ecinput.h_anchorReadIds.get(),
                ecinput.d_anchorReadIds.get(),
                (*ecinput.h_numAnchors.get()),
                stream
            );

            gpuReadStorage->gatherSequenceLengths(
                readstorageHandle,
                ecinput.d_anchor_sequences_lengths.get(),
                ecinput.d_anchorReadIds.get(),
                (*ecinput.h_numAnchors.get()),
                stream
            );
        }

        void getCandidateReadIdsWithMinhashing(GpuErrorCorrectorInput& ecinput, cudaStream_t stream){
            int totalNumValues = 0;

            gpuMinhasher->determineNumValues(
                minhashHandle,
                ecinput.d_anchor_sequences_data.get(),
                encodedSequencePitchInInts,
                ecinput.d_anchor_sequences_lengths.get(),
                (*ecinput.h_numAnchors.get()),
                ecinput.d_candidates_per_anchor.get(),
                totalNumValues,
                stream
            );

            cudaStreamSynchronize(stream); CUERR;

            ecinput.d_candidate_read_ids.resize(totalNumValues);
            ecinput.h_candidate_read_ids.resize(totalNumValues);

            if(totalNumValues == 0){
                cudaMemsetAsync(ecinput.d_candidates_per_anchor.get(), 0, sizeof(int) * (*ecinput.h_numAnchors), stream); CUERR;
                cudaMemsetAsync(ecinput.d_candidates_per_anchor_prefixsum.get(), 0, sizeof(int) * (1 + (*ecinput.h_numAnchors)), stream); CUERR;
                return;
            }

            gpuMinhasher->retrieveValues(
                minhashHandle,
                ecinput.d_anchorReadIds.get(),
                (*ecinput.h_numAnchors.get()),                
                totalNumValues,
                ecinput.d_candidate_read_ids.get(),
                ecinput.d_candidates_per_anchor.get(),
                ecinput.d_candidates_per_anchor_prefixsum.get(),
                stream
            );

            gpucorrectorkernels::copyMinhashResultsKernel<<<640, 256, 0, stream>>>(
                ecinput.d_numCandidates.get(),
                ecinput.h_numCandidates.get(),
                ecinput.h_candidate_read_ids.get(),
                ecinput.d_candidates_per_anchor_prefixsum.get(),
                ecinput.d_candidate_read_ids.get(),
                *ecinput.h_numAnchors.get()
            ); CUERR;

            //  cudaStreamSynchronize(stream); CUERR;
            //  std::vector<int> vec((1 + *ecinput.h_numAnchors));
            //  cudaMemcpyAsync(vec.data(), ecinput.d_candidates_per_anchor_prefixsum, sizeof(int) * (1 + *ecinput.h_numAnchors), D2H, stream);
            //  std::vector<int> vec2((*ecinput.h_numAnchors));
            //  cudaMemcpyAsync(vec2.data(), ecinput.d_candidates_per_anchor, sizeof(int) * (*ecinput.h_numAnchors), D2H, stream);

            // std::cerr << *ecinput.h_numCandidates << "\n";
            //  for(int i = 0; i < (1 + *ecinput.h_numAnchors); i++){
            //      std::cerr << vec[i] << " ";
            //  }
            //  std::cerr << "\n";

            //  for(int i = 0; i < (*ecinput.h_numAnchors); i++){
            //      std::cerr << vec2[i] << " ";
            //  }
            //  std::cerr << "\n";
           

            
        }
    
        int deviceId;
        int maxCandidatesPerRead;
        std::size_t encodedSequencePitchInInts;
        CudaStream backgroundStream;
        CudaEvent previousBatchFinishedEvent;
        const GpuReadStorage* gpuReadStorage;
        const GpuMinhasher* gpuMinhasher;
        ThreadPool* threadPool;
        ThreadPool::ParallelForHandle pforHandle;
        DistributedReadStorage::GatherHandleSequences anchorSequenceGatherHandle;
        GpuMinhasher::QueryHandle minhashHandle;
        ReadStorageHandle readstorageHandle;
    };


    class OutputConstructor{
    public:

        OutputConstructor() = default;

        OutputConstructor(
            ReadCorrectionFlags& correctionFlags_,
            const CorrectionOptions& correctionOptions_
        ) :
            correctionFlags{&correctionFlags_},
            correctionOptions{&correctionOptions_}
        {

        }

        template<class ForLoop>
        CorrectionOutput constructResults(const GpuErrorCorrectorRawOutput& currentOutput, ForLoop loopExecutor) const{
            //assert(cudaSuccess == currentOutput.event.query());

            if(currentOutput.nothingToDo){
                return CorrectionOutput{};
            }

            std::vector<int> anchorIndicesToProcess;
            std::vector<std::pair<int,int>> candidateIndicesToProcess;

            anchorIndicesToProcess.reserve(currentOutput.numAnchors);
            if(correctionOptions->correctCandidates){
                candidateIndicesToProcess.reserve(16 * currentOutput.numAnchors);
            }

            nvtx::push_range("preprocess anchor results",0);

            for(int anchor_index = 0; anchor_index < currentOutput.numAnchors; anchor_index++){
                const read_number readId = currentOutput.h_anchorReadIds[anchor_index];
                const bool isCorrected = currentOutput.h_anchor_is_corrected[anchor_index];
                const bool isHQ = currentOutput.h_is_high_quality_anchor[anchor_index].hq();

                if(isHQ){
                    correctionFlags->setCorrectedAsHqAnchor(readId);
                }

                if(isCorrected){
                    anchorIndicesToProcess.emplace_back(anchor_index);
                }else{
                    correctionFlags->setCouldNotBeCorrectedAsAnchor(readId);
                }

                assert(!(isHQ && !isCorrected));
            }

            nvtx::pop_range();

            if(correctionOptions->correctCandidates){

                nvtx::push_range("preprocess candidate results",0);

                for(int anchor_index = 0; anchor_index < currentOutput.numAnchors; anchor_index++){

                    const int globalOffset = currentOutput.h_num_corrected_candidates_per_anchor_prefixsum[anchor_index];
                    const int n_corrected_candidates = currentOutput.h_num_corrected_candidates_per_anchor[anchor_index];

                    // const int* const my_indices_of_corrected_candidates = currentOutput.h_indices_of_corrected_candidates
                    //                                     + globalOffset;

                    for(int i = 0; i < n_corrected_candidates; ++i) {
                        //const int global_candidate_index = my_indices_of_corrected_candidates[i];
                        //const read_number candidate_read_id = currentOutput.h_candidate_read_ids[global_candidate_index];
                        const read_number candidate_read_id = currentOutput.h_candidate_read_ids[globalOffset + i];

                        if (!correctionFlags->isCorrectedAsHQAnchor(candidate_read_id)) {
                            candidateIndicesToProcess.emplace_back(std::make_pair(anchor_index, i));
                        }
                    }
                }

                nvtx::pop_range();

            }

            const int numCorrectedAnchors = anchorIndicesToProcess.size();
            const int numCorrectedCandidates = candidateIndicesToProcess.size();

            // std::cerr << "numCorrectedAnchors: " << numCorrectedAnchors << 
            //     ", numCorrectedCandidates: " << numCorrectedCandidates << "\n";

            CorrectionOutput correctionOutput;
            correctionOutput.anchorCorrections.resize(numCorrectedAnchors);

            if(correctionOptions->correctCandidates){
                correctionOutput.candidateCorrections.resize(numCorrectedCandidates);
            }

            auto unpackAnchors = [&](int begin, int end){
                nvtx::push_range("Anchor unpacking", 3);

                //Edits and numEdits are stored compact, only for corrected anchors.
                //they are indexed by positionInVector instead of anchor_index
                            
                for(int positionInVector = begin; positionInVector < end; ++positionInVector) {
                    const int anchor_index = anchorIndicesToProcess[positionInVector];

                    auto& tmp = correctionOutput.anchorCorrections[positionInVector];
                    
                    const read_number readId = currentOutput.h_anchorReadIds[anchor_index];

                    tmp.hq = currentOutput.h_is_high_quality_anchor[anchor_index].hq();                    
                    tmp.type = TempCorrectedSequence::Type::Anchor;
                    tmp.readId = readId;
                    
                    const int numEdits = currentOutput.h_numEditsPerCorrectedanchor[positionInVector];
                    if(numEdits != currentOutput.doNotUseEditsValue){
                        const int editOffset = currentOutput.h_anchorEditOffsets[positionInVector];
                        tmp.edits.resize(numEdits);
                        // const TempCorrectedSequence::EncodedEdit* const gpuedits 
                        //     = (const TempCorrectedSequence::EncodedEdit*)(((const char*)currentOutput.h_editsPerCorrectedanchor.get()) 
                        //         + positionInVector * currentOutput.editsPitchInBytes);
                        const auto* myedits = currentOutput.h_editsPerCorrectedanchor + editOffset;
                        std::copy_n(myedits, numEdits, tmp.edits.begin());
                        tmp.useEdits = true;
                    }else{
                        tmp.edits.clear();
                        tmp.useEdits = false;

                        const int sequenceOffset = currentOutput.h_correctedAnchorsOffsets[positionInVector];

                        const char* const my_corrected_anchor_data = currentOutput.h_corrected_anchors + sequenceOffset;
                        const int anchor_length = currentOutput.h_anchor_sequences_lengths[anchor_index];
                        tmp.sequence.assign(my_corrected_anchor_data, anchor_length);
                    }

                    // if(tmp.readId == 9273463){
                    //     std::cerr << tmp << "\n";
                    // }
                }

                nvtx::pop_range();
            };

            auto unpackcandidates = [&](int begin, int end){
                nvtx::push_range("candidate unpacking", 3);

                //buffers are stored compact. offsets for each anchor are given by h_num_corrected_candidates_per_anchor_prefixsum
                //Edits, numEdits, h_candidate_read_ids, h_candidate_sequences_lengths, h_alignment_shifts are stored compact, only for corrected candidates.
                //edits are only present for candidates which use edits and have numEdits > 0
                //offsets to the edits of candidates are stored in h_candidateEditOffsets
                

                for(int positionInVector = begin; positionInVector < end; ++positionInVector) {
                    

                    //TIMERSTARTCPU(setup);
                    const int anchor_index = candidateIndicesToProcess[positionInVector].first;
                    const int candidateIndex = candidateIndicesToProcess[positionInVector].second;
                    const read_number anchorReadId = currentOutput.h_anchorReadIds[anchor_index];

                    auto& tmp = correctionOutput.candidateCorrections[positionInVector];

                    const size_t offsetForCorrectedCandidateData = currentOutput.h_num_corrected_candidates_per_anchor_prefixsum[anchor_index];

                    const char* const my_corrected_candidates_data = currentOutput.h_corrected_candidates
                                                    + offsetForCorrectedCandidateData * currentOutput.decodedSequencePitchInBytes;

                    const read_number candidate_read_id = currentOutput.h_candidate_read_ids[offsetForCorrectedCandidateData + candidateIndex];
                    const int candidate_shift = currentOutput.h_alignment_shifts[offsetForCorrectedCandidateData + candidateIndex];

                    if(correctionOptions->new_columns_to_correct < candidate_shift){
                        std::cerr << "readid " << anchorReadId << " candidate readid " << candidate_read_id << " : "
                        << candidate_shift << " " << correctionOptions->new_columns_to_correct <<"\n";

                        assert(correctionOptions->new_columns_to_correct >= candidate_shift);
                    }                
                    
                    tmp.type = TempCorrectedSequence::Type::Candidate;
                    tmp.shift = candidate_shift;
                    tmp.readId = candidate_read_id;
                    
                    const int numEdits = currentOutput.h_numEditsPerCorrectedCandidate[offsetForCorrectedCandidateData + candidateIndex];
                    const int editsOffset = currentOutput.h_candidateEditOffsets[offsetForCorrectedCandidateData + candidateIndex];

                    if(numEdits != currentOutput.doNotUseEditsValue){
                        tmp.edits.resize(numEdits);
                        const auto* myEdits = &currentOutput.h_editsPerCorrectedCandidate[editsOffset];
                        std::copy_n(myEdits, numEdits, tmp.edits.begin());
                        tmp.useEdits = true;
                    }else{
                        const int correctionOffset = currentOutput.h_correctedCandidatesOffsets[offsetForCorrectedCandidateData + candidateIndex];
                        const int candidate_length = currentOutput.h_candidate_sequences_lengths[offsetForCorrectedCandidateData + candidateIndex];
                        const char* const candidate_data = my_corrected_candidates_data + correctionOffset * currentOutput.decodedSequencePitchInBytes;
                        tmp.sequence.assign(candidate_data, candidate_length);
                        tmp.edits.clear();
                        tmp.useEdits = false;
                    }

                    // if(tmp.readId == 9273463){
                    //     std::cerr << tmp << " with anchorid " << anchorReadId << "\n";
                    // }
                }

                nvtx::pop_range();
            };


            if(!correctionOptions->correctCandidates){
                loopExecutor(0, numCorrectedAnchors, [=](auto begin, auto end, auto /*threadId*/){
                    unpackAnchors(begin, end);
                });
            }else{
        
  
                loopExecutor(0, numCorrectedAnchors, [=](auto begin, auto end, auto /*threadId*/){
                    unpackAnchors(begin, end);
                });
           

                loopExecutor(0, numCorrectedCandidates, [=](auto begin, auto end, auto /*threadId*/){
                        unpackcandidates(begin, end);
                    } //,  threadPool->getConcurrency() * 4
                );         
            }

            std::sort(correctionOutput.candidateCorrections.begin(), correctionOutput.candidateCorrections.end(), [](const auto& l, const auto& r){
                return l.readId < r.readId;
            });

            return correctionOutput;
        }

        MemoryUsage getMemoryInfo() const{
            MemoryUsage info{};
            return info;
        }
    public: //private:
        ReadCorrectionFlags* correctionFlags;
        const CorrectionOptions* correctionOptions;
    };

    class GpuErrorCorrector{
        static constexpr bool useGraph() noexcept{
            return false;
        }

    public:

        template<class T>
        using PinnedBuffer = helpers::SimpleAllocationPinnedHost<T>;

        template<class T>
        using DeviceBuffer = helpers::SimpleAllocationDevice<T>;
        //using DeviceBuffer = helpers::SimpleAllocationPinnedHost<T>;


        static constexpr int getNumRefinementIterations() noexcept{
            return 5;
        }

        static constexpr bool useMsaRefinement() noexcept{
            return getNumRefinementIterations() > 0;
        }

        GpuErrorCorrector() = default;

        GpuErrorCorrector(
            const GpuReadStorage& gpuReadStorage_,
            const ReadCorrectionFlags& correctionFlags_,
            const CorrectionOptions& correctionOptions_,
            const GoodAlignmentProperties& goodAlignmentProperties_,
            int maxAnchorsPerCall,
            ThreadPool* threadPool_,
            const GpuForest* gpuForestAnchor_,
            const GpuForest* gpuForestCandidate_
        ) : 
            maxAnchors{maxAnchorsPerCall},
            maxCandidates{0},
            correctionFlags{&correctionFlags_},
            gpuReadStorage{&gpuReadStorage_},
            correctionOptions{&correctionOptions_},
            goodAlignmentProperties{&goodAlignmentProperties_},
            threadPool{threadPool_},
            gpuForestAnchor{gpuForestAnchor_},
            gpuForestCandidate{gpuForestCandidate_},
            readstorageHandle{gpuReadStorage->makeHandle()}
        {
            if(correctionOptions->correctionType != CorrectionType::Classic){
                assert(gpuForestAnchor != nullptr);
            }

            if(correctionOptions->correctionTypeCands != CorrectionType::Classic){
                assert(gpuForestCandidate != nullptr);
            }

            cudaGetDevice(&deviceId); CUERR;

            kernelLaunchHandle = make_kernel_launch_handle(deviceId);

            for(auto& event: events){
                event = std::move(CudaEvent{cudaEventDisableTiming});
            }
            backgroundStream = CudaStream{};
            previousBatchFinishedEvent = CudaEvent{};

            encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(gpuReadStorage->getSequenceLengthUpperBound());
            decodedSequencePitchInBytes = SDIV(gpuReadStorage->getSequenceLengthUpperBound(), 4) * 4;
            qualityPitchInBytes = SDIV(gpuReadStorage->getSequenceLengthUpperBound(), 32) * 32;
            maxNumEditsPerSequence = std::max(1,gpuReadStorage->getSequenceLengthUpperBound() / 7);
            //pad to multiple of 128 bytes
            editsPitchInBytes = SDIV(maxNumEditsPerSequence * sizeof(TempCorrectedSequence::EncodedEdit), 128) * 128;

            const std::size_t min_overlap = std::max(
                1, 
                std::max(
                    goodAlignmentProperties->min_overlap, 
                    int(gpuReadStorage->getSequenceLengthUpperBound() * goodAlignmentProperties->min_overlap_ratio)
                )
            );
            const std::size_t msa_max_column_count = (3*gpuReadStorage->getSequenceLengthUpperBound() - 2*min_overlap);
            //round up to 32 elements
            msaColumnPitchInElements = SDIV(msa_max_column_count, 32) * 32;

            initFixedSizeBuffers();
        }

        ~GpuErrorCorrector(){
            gpuReadStorage->destroyHandle(readstorageHandle);

            for(auto pair : numCandidatesPerReadMap){
                //std::cerr << pair.first << " " << pair.second << "\n";
            }
        }

        void correct(GpuErrorCorrectorInput& input, GpuErrorCorrectorRawOutput& output, cudaStream_t stream){
            previousBatchFinishedEvent.synchronize();
            cudaStreamSynchronize(stream); CUERR;

            //assert(cudaSuccess == input.event.query());
            //assert(cudaSuccess == output.event.query());

            currentInput = &input;
            currentOutput = &output;

            assert(*currentInput->h_numAnchors.get() <= maxAnchors);

            currentNumAnchors = *currentInput->h_numAnchors.get();
            currentNumCandidates = *currentInput->h_numCandidates.get();

            assert(currentNumAnchors % 2 == 0);

            currentOutput->nothingToDo = false;
            currentOutput->numAnchors = currentNumAnchors;
            currentOutput->numCandidates = currentNumCandidates;
            currentOutput->doNotUseEditsValue = getDoNotUseEditsValue();
            currentOutput->editsPitchInBytes = editsPitchInBytes;
            currentOutput->decodedSequencePitchInBytes = decodedSequencePitchInBytes;

            if(currentNumCandidates == 0){
                currentOutput->nothingToDo = true;
                return;
            }
            

            int curId = 0;
            cudaGetDevice(&curId); CUERR;
            cudaSetDevice(deviceId); CUERR;

            resizeBuffers(currentNumAnchors, currentNumCandidates);

            gpucorrectorkernels::copyCorrectionInputDeviceData<<<32768,256, 0, stream>>>(
                d_numAnchors,
                d_numCandidates,
                d_anchorReadIds,
                d_anchor_sequences_data,
                d_anchor_sequences_lengths,
                d_candidate_read_ids,
                d_candidates_per_anchor,
                d_candidates_per_anchor_prefixsum,
                encodedSequencePitchInInts,
                currentInput->d_numAnchors,
                currentInput->d_numCandidates,
                currentInput->d_anchorReadIds,
                currentInput->d_anchor_sequences_data,
                currentInput->d_anchor_sequences_lengths,
                currentInput->d_candidate_read_ids,
                currentInput->d_candidates_per_anchor,
                currentInput->d_candidates_per_anchor_prefixsum
            ); CUERR;

            //after gpu data has been copied to local working set, the gpu data of currentInput can be reused
            currentInput->event.record(stream);

            gpucorrectorkernels::setAnchorIndicesOfCandidateskernel
                    <<<1024, 128, 0, stream>>>(
                d_anchorIndicesOfCandidates.get(),
                d_numAnchors.get(),
                d_candidates_per_anchor.get(),
                d_candidates_per_anchor_prefixsum.get()
            ); CUERR;

            flagPairedCandidates(stream);

            getAmbiguousFlagsOfAnchors(stream);
            getAmbiguousFlagsOfCandidates(stream);


            nvtx::push_range("getCandidateSequenceData", 3);
            getCandidateSequenceData(stream); 
            nvtx::pop_range();


            // if(useGraph()){
            //     //std::cerr << "Launching graph for output " << currentOutput << "\n";
            //     graphMap[currentOutput].execute(stream);
            //     //cudaStreamSynchronize(stream); CUERR;
            // }else{
                execute(stream);
            //}

            nvtx::push_range("copyAnchorResultsFromDeviceToHost", 3);
            copyAnchorResultsFromDeviceToHost(stream);
            nvtx::pop_range();

            if(correctionOptions->correctCandidates){
                nvtx::push_range("copyCandidateResultsFromDeviceToHost", 4);
                copyCandidateResultsFromDeviceToHost(stream);
                nvtx::pop_range();
            }

            std::copy_n(currentInput->h_anchorReadIds.get(), currentNumAnchors, currentOutput->h_anchorReadIds.get());            
            //std::copy_n(currentInput->h_candidate_read_ids.get(), currentNumCandidates, currentOutput->h_candidate_read_ids.get()); //remove if candidates are compacted


            //after the current work in stream is completed, all results in currentOutput are ready to use.
            cudaEventRecord(currentOutput->event, stream); CUERR;

            cudaEventRecord(previousBatchFinishedEvent, stream); CUERR;
        }

        MemoryUsage getMemoryInfo() const{
            MemoryUsage info{};
            auto handleHost = [&](const auto& h){
                info.host += h.sizeInBytes();
            };
            auto handleDevice = [&](const auto& d){
                info.device[deviceId] += d.sizeInBytes();
            };

            info += gpuReadStorage->getMemoryInfo(readstorageHandle);

            handleHost(h_num_total_corrected_candidates);

            handleDevice(d_candidates_per_anchor_tmp);
            handleDevice(d_anchorContainsN);
            handleDevice(d_candidateContainsN);
            handleDevice(d_candidate_sequences_lengths);
            handleDevice(d_anchor_sequences_lengths);
            handleDevice(d_candidate_sequences_data);
            handleDevice(d_transposedCandidateSequencesData);
            handleDevice(d_anchor_qualities);
            handleDevice(d_candidate_qualities);
            handleDevice(d_anchorIndicesOfCandidates);
            handleDevice(d_tempstorage);
            handleDevice(d_alignment_overlaps);
            handleDevice(d_alignment_shifts);
            handleDevice(d_alignment_nOps);
            handleDevice(d_alignment_isValid);
            handleDevice(d_alignment_best_alignment_flags);
            handleDevice(d_indices);
            handleDevice(d_indices_per_anchor);
            handleDevice(d_num_indices);
            handleDevice(d_indices_tmp);
            handleDevice(d_indices_per_anchor_tmp);
            handleDevice(d_num_indices_tmp);
            handleDevice(d_consensus);
            handleDevice(d_support);
            handleDevice(d_coverage);
            handleDevice(d_origWeights);
            handleDevice(d_origCoverages);
            handleDevice(d_msa_column_properties);
            handleDevice(d_counts);
            handleDevice(d_weights);
            handleDevice(d_corrected_anchors);
            handleDevice(d_corrected_candidates);
            handleDevice(d_num_corrected_candidates_per_anchor);
            handleDevice(d_num_corrected_candidates_per_anchor_prefixsum);
            handleDevice(d_num_total_corrected_candidates);
            handleDevice(d_anchor_is_corrected);
            handleDevice(d_is_high_quality_anchor);
            handleDevice(d_high_quality_anchor_indices);
            handleDevice(d_num_high_quality_anchor_indices);
            handleDevice(d_editsPerCorrectedanchor);
            handleDevice(d_numEditsPerCorrectedanchor);
            handleDevice(d_editsPerCorrectedCandidate);
            handleDevice(d_numEditsPerCorrectedCandidate);
            handleDevice(d_indices_of_corrected_anchors);
            handleDevice(d_num_indices_of_corrected_anchors);
            handleDevice(d_indices_of_corrected_candidates);
            handleDevice(d_numEditsPerCorrectedanchor);
            handleDevice(d_numAnchors);
            handleDevice(d_numCandidates);
            handleDevice(d_anchorReadIds);
            handleDevice(d_anchor_sequences_data);
            handleDevice(d_anchor_sequences_lengths);
            handleDevice(d_candidate_read_ids);
            handleDevice(d_candidates_per_anchor);
            handleDevice(d_candidates_per_anchor_prefixsum);
            handleDevice(d_candidatesBeginOffsets);

            return info;
        } 

        


    public: //private:

        void gpuMemsetZero(cudaStream_t stream){
            auto zero = [&](auto& devicebuffer){
                cudaMemsetAsync(devicebuffer.get(), 0, devicebuffer.sizeInBytes(), stream);
            };

            zero(d_candidates_per_anchor_tmp);
            zero(d_anchorContainsN);
            zero(d_anchor_qualities);
 
            zero(d_indices_per_anchor);
            zero(d_num_indices);
            zero(d_indices_per_anchor_tmp);
            zero(d_num_indices_tmp);
            zero(d_consensus);
            zero(d_support);
            zero(d_coverage);
            zero(d_origWeights);
            zero(d_origCoverages);
            zero(d_msa_column_properties);
            zero(d_counts);
            zero(d_weights);
            zero(d_corrected_anchors);
            zero(d_num_corrected_candidates_per_anchor);
            zero(d_num_corrected_candidates_per_anchor_prefixsum);
            zero(d_num_total_corrected_candidates);
            zero(d_anchor_is_corrected);
            zero(d_is_high_quality_anchor);
            zero(d_high_quality_anchor_indices);
            zero(d_num_high_quality_anchor_indices);
            zero(d_editsPerCorrectedanchor);
            zero(d_numEditsPerCorrectedanchor);
            zero(d_indices_of_corrected_anchors);
            zero(d_num_indices_of_corrected_anchors);
            zero(d_numAnchors);
            zero(d_numCandidates);
            zero(d_anchorReadIds);
            zero(d_anchor_sequences_data);
            zero(d_anchor_sequences_lengths);
            zero(d_candidates_per_anchor);
            zero(d_candidates_per_anchor_prefixsum);
            zero(d_candidatesBeginOffsets);
            zero(d_anchorIndicesOfCandidates);
            zero(d_candidateContainsN);
            zero(d_candidate_read_ids);
            zero(d_candidate_sequences_lengths);
            zero(d_candidate_sequences_data);
            zero(d_transposedCandidateSequencesData);            
            zero(d_candidate_qualities);
            zero(d_alignment_overlaps);
            zero(d_alignment_shifts);
            zero(d_alignment_nOps);
            zero(d_alignment_isValid);
            zero(d_alignment_best_alignment_flags);
            zero(d_indices);
            zero(d_indices_tmp);
            zero(d_corrected_candidates);
            zero(d_editsPerCorrectedCandidate);
            zero(d_numEditsPerCorrectedCandidate);
            zero(d_indices_of_corrected_candidates);
            
        }

        void initFixedSizeBuffers(){
            const std::size_t numEditsAnchors = SDIV(editsPitchInBytes * maxAnchors, sizeof(TempCorrectedSequence::EncodedEdit));          

            //does not depend on number of candidates
            h_num_total_corrected_candidates.resize(1);
            h_num_indices.resize(1);
            h_numSelected.resize(1);
            h_numRemainingCandidatesAfterAlignment.resize(1);

            //does not depend on number of candidates
            d_candidates_per_anchor_tmp.resize(maxAnchors);
            d_anchorContainsN.resize(maxAnchors);

            if(correctionOptions->useQualityScores){
                d_anchor_qualities.resize(maxAnchors * qualityPitchInBytes);
            }

            d_indices_per_anchor.resize(maxAnchors);
            d_num_indices.resize(1);
            d_indices_per_anchor_tmp.resize(maxAnchors);
            d_num_indices_tmp.resize(1);
            d_indices_per_anchor_prefixsum.resize(maxAnchors);
            d_consensus.resize(maxAnchors * msaColumnPitchInElements);
            d_support.resize(maxAnchors * msaColumnPitchInElements);
            d_coverage.resize(maxAnchors * msaColumnPitchInElements);
            d_origWeights.resize(maxAnchors * msaColumnPitchInElements);
            d_origCoverages.resize(maxAnchors * msaColumnPitchInElements);
            d_msa_column_properties.resize(maxAnchors);
            d_counts.resize(maxAnchors * 4 * msaColumnPitchInElements);
            d_weights.resize(maxAnchors * 4 * msaColumnPitchInElements);
            d_corrected_anchors.resize(maxAnchors * decodedSequencePitchInBytes);
            d_num_corrected_candidates_per_anchor.resize(maxAnchors);
            d_num_corrected_candidates_per_anchor_prefixsum.resize(maxAnchors);
            d_num_total_corrected_candidates.resize(1);
            d_anchor_is_corrected.resize(maxAnchors);
            d_is_high_quality_anchor.resize(maxAnchors);
            d_high_quality_anchor_indices.resize(maxAnchors);
            d_num_high_quality_anchor_indices.resize(1); 
            d_editsPerCorrectedanchor.resize(numEditsAnchors);
            d_numEditsPerCorrectedanchor.resize(maxAnchors);
            d_indices_of_corrected_anchors.resize(maxAnchors);
            d_num_indices_of_corrected_anchors.resize(1);

            d_numAnchors.resize(1);
            d_numCandidates.resize(1);
            d_anchorReadIds.resize(maxAnchors);
            d_anchor_sequences_data.resize(encodedSequencePitchInInts * maxAnchors);
            d_anchor_sequences_lengths.resize(maxAnchors);
            d_candidates_per_anchor.resize(maxAnchors);
            h_candidates_per_anchor_prefixsum.resize(maxAnchors + 1);
            d_candidates_per_anchor_prefixsum.resize(maxAnchors + 1);
            d_candidatesBeginOffsets.resize(maxAnchors);
            d_totalNumEdits.resize(1);
        }
 
        void resizeBuffers(int numReads, int numCandidates){  
            assert(numReads <= maxAnchors);

            //bool maxCandidatesDidChange = false;
            constexpr int stepsizeForMaxCandidates = 10000;
            if(numCandidates > maxCandidates){
                //round up numCandidates to next multiple of stepsize
                maxCandidates = SDIV(numCandidates, stepsizeForMaxCandidates) * stepsizeForMaxCandidates;
                //maxCandidatesDidChange = true;

                if(useGraph()){
                    //reallocation will occure. invalidate all graphs and recapture them.
                    for(auto& pair : graphMap){
                        pair.second.valid = false;
                    }
                }
            }

            std::size_t numEditsCandidates = SDIV(editsPitchInBytes * maxCandidates, sizeof(TempCorrectedSequence::EncodedEdit));

            const std::size_t numEditsAnchors = SDIV(editsPitchInBytes * maxAnchors, sizeof(TempCorrectedSequence::EncodedEdit));          

            //does not depend on number of candidates
            bool outputBuffersReallocated = false;
            outputBuffersReallocated |= currentOutput->h_anchor_sequences_lengths.resize(maxAnchors);
            outputBuffersReallocated |= currentOutput->h_corrected_anchors.resize(maxAnchors * decodedSequencePitchInBytes);            
            outputBuffersReallocated |= currentOutput->h_anchor_is_corrected.resize(maxAnchors);
            outputBuffersReallocated |= currentOutput->h_is_high_quality_anchor.resize(maxAnchors);
            outputBuffersReallocated |= currentOutput->h_editsPerCorrectedanchor.resize(numEditsAnchors);
            outputBuffersReallocated |= currentOutput->h_numEditsPerCorrectedanchor.resize(maxAnchors);            
            outputBuffersReallocated |= currentOutput->h_num_corrected_candidates_per_anchor.resize(maxAnchors);
            outputBuffersReallocated |= currentOutput->h_num_corrected_candidates_per_anchor_prefixsum.resize(maxAnchors);
            outputBuffersReallocated |= currentOutput->h_candidate_sequences_lengths.resize(maxCandidates);
            outputBuffersReallocated |= currentOutput->h_corrected_candidates.resize(maxCandidates * decodedSequencePitchInBytes);
            outputBuffersReallocated |= currentOutput->h_editsPerCorrectedCandidate.resize(numEditsCandidates);
            outputBuffersReallocated |= currentOutput->h_numEditsPerCorrectedCandidate.resize(maxCandidates);
            outputBuffersReallocated |= currentOutput->h_candidateEditOffsets.resize(maxCandidates);
            outputBuffersReallocated |= currentOutput->h_indices_of_corrected_candidates.resize(maxCandidates);
            outputBuffersReallocated |= currentOutput->h_alignment_shifts.resize(maxCandidates);
            outputBuffersReallocated |= currentOutput->h_candidate_read_ids.resize(maxCandidates);
            outputBuffersReallocated |= currentOutput->h_anchorReadIds.resize(maxAnchors);
            outputBuffersReallocated |= currentOutput->h_correctedCandidatesOffsets.resize(maxCandidates * decodedSequencePitchInBytes);
            outputBuffersReallocated |= currentOutput->h_anchorEditOffsets.resize(maxAnchors);
            outputBuffersReallocated |= currentOutput->h_correctedAnchorsOffsets.resize(maxAnchors * decodedSequencePitchInBytes);
            
            
            d_anchorIndicesOfCandidates.resize(maxCandidates);
            d_candidateContainsN.resize(maxCandidates);
            d_candidate_read_ids.resize(maxCandidates);
            d_candidate_read_ids2.resize(maxCandidates);
            d_candidate_sequences_lengths.resize(maxCandidates);
            d_candidate_sequences_data.resize(maxCandidates * encodedSequencePitchInInts);
            d_transposedCandidateSequencesData.resize(maxCandidates * encodedSequencePitchInInts);
            d_isPairedCandidate.resize(maxCandidates);
            h_isPairedCandidate.resize(maxCandidates);

            d_flagsCandidates.resize(maxCandidates);
            h_flagsCandidates.resize(maxCandidates);
            
            if(correctionOptions->useQualityScores){
                d_candidate_qualities.resize(maxCandidates * qualityPitchInBytes);

                d_candidate_qualities_compact.resize(maxCandidates * qualityPitchInBytes);
            }

            h_indicesForGather.resize(maxCandidates);
            d_indicesForGather.resize(maxCandidates);
            
            d_alignment_overlaps.resize(maxCandidates);
            d_alignment_shifts.resize(maxCandidates);
            d_alignment_nOps.resize(maxCandidates);
            d_alignment_isValid.resize(maxCandidates);
            d_alignment_best_alignment_flags.resize(maxCandidates);
            d_indices.resize(maxCandidates);
            d_indices_tmp.resize(maxCandidates);
            d_corrected_candidates.resize(maxCandidates * decodedSequencePitchInBytes);
            d_corrected_candidates2.resize(maxCandidates * decodedSequencePitchInBytes);
            d_editsPerCorrectedCandidate.resize(numEditsCandidates);
            d_editsPerCorrectedCandidate2.resize(numEditsCandidates);
            d_numEditsPerCorrectedCandidate.resize(maxCandidates);
            d_indices_of_corrected_candidates.resize(maxCandidates);

            d_alignment_shifts2.resize(maxCandidates);
            d_alignment_overlaps2.resize(maxCandidates);
            d_alignment_nOps2.resize(maxCandidates);
            d_anchorIndicesOfCandidates2.resize(maxCandidates);
            d_candidateContainsN2.resize(maxCandidates);
            d_candidate_sequences_lengths2.resize(maxCandidates);
            d_candidate_sequences_data2.resize(maxCandidates * decodedSequencePitchInBytes);
            d_alignment_best_alignment_flags2.resize(maxCandidates);

            std::size_t flagTemp = sizeof(bool) * maxCandidates;
            std::size_t popcountShdTempBytes = 0; 
            
            const bool removeAmbiguousAnchors = correctionOptions->excludeAmbiguousReads;
            const bool removeAmbiguousCandidates = correctionOptions->excludeAmbiguousReads;
    
            call_popcount_shifted_hamming_distance_kernel_async(
                nullptr,
                popcountShdTempBytes,
                d_alignment_overlaps.get(),
                d_alignment_shifts.get(),
                d_alignment_nOps.get(),
                d_alignment_isValid.get(),
                d_alignment_best_alignment_flags.get(),
                d_anchor_sequences_data.get(),
                d_candidate_sequences_data.get(),
                d_anchor_sequences_lengths.get(),
                d_candidate_sequences_lengths.get(),
                d_candidates_per_anchor_prefixsum.get(),
                d_candidates_per_anchor.get(),
                d_anchorIndicesOfCandidates.get(),
                d_numAnchors.get(),
                d_numCandidates.get(),
                d_anchorContainsN.get(),
                removeAmbiguousAnchors,
                d_candidateContainsN.get(),
                removeAmbiguousCandidates,
                maxAnchors,
                maxCandidates,
                gpuReadStorage->getSequenceLengthUpperBound(),
                encodedSequencePitchInInts,
                goodAlignmentProperties->min_overlap,
                goodAlignmentProperties->maxErrorRate,
                goodAlignmentProperties->min_overlap_ratio,
                correctionOptions->estimatedErrorrate,                
                (cudaStream_t)0,
                kernelLaunchHandle
            );

            std::size_t cubtemp = 0;

            cub::DeviceSelect::Flagged(
                nullptr,
                cubtemp,
                cub::CountingInputIterator<int>(0),
                (bool*) nullptr,
                (int*) nullptr,
                (int*) nullptr,
                maxCandidates,
                (cudaStream_t)0
            ); CUERR;
            
            std::size_t tmpsize = std::max(cubtemp, flagTemp);
            tmpsize = std::max(tmpsize, popcountShdTempBytes);
            d_tempstorage.resize(tmpsize);

            // if(maxCandidatesDidChange){
            //     std::cerr << "maxCandidates changed to " << maxCandidates << "\n";
            // }

            if(useGraph()){
                if(!graphMap[currentOutput].valid){
                    if(outputBuffersReallocated){
                        std::cerr << "outputBuffersReallocated " << currentOutput << "\n";
                    }
                    //std::cerr << "Capture graph for output " << currentOutput << "\n";
                    graphMap[currentOutput].capture(
                        [&](cudaStream_t capstream){
                            execute(capstream);
                        }
                    );
                }
            }
        }

        void flagPairedCandidates(cudaStream_t stream){

            if(gpuReadStorage->isPairedEnd()){

                cudaMemcpyAsync(
                    h_candidates_per_anchor_prefixsum.data(),
                    d_candidates_per_anchor_prefixsum.data(),
                    sizeof(int) * (currentNumAnchors + 1),
                    D2H,
                    stream
                );

                std::fill(h_isPairedCandidate.begin(), h_isPairedCandidate.end(), false);

                cudaStreamSynchronize(stream); CUERR;

                for(int i = 0; i < currentNumAnchors; i++){
                    const int num = h_candidates_per_anchor_prefixsum[i+1] - h_candidates_per_anchor_prefixsum[i];
                    numCandidatesPerReadMap[num]++;
                }


                std::vector<int> numPairedPerAnchor(currentNumAnchors, 0);            

                for(int ap = 0; ap < currentNumAnchors / 2; ap++){
                    const int begin1 = h_candidates_per_anchor_prefixsum[2*ap + 0];
                    const int end1 = h_candidates_per_anchor_prefixsum[2*ap + 1];
                    const int begin2 = h_candidates_per_anchor_prefixsum[2*ap + 1];
                    const int end2 = h_candidates_per_anchor_prefixsum[2*ap + 2];

                    // assert(std::is_sorted(pairIds + begin1, pairIds + end1));
                    // assert(std::is_sorted(pairIds + begin2, pairIds + end2));

                    std::vector<int> pairedPositions(std::min(end1-begin1, end2-begin2));
                    std::vector<int> pairedPositions2(std::min(end1-begin1, end2-begin2));

                    auto endIters = findPositionsOfPairedReadIds(
                        currentInput->h_candidate_read_ids.data() + begin1,
                        currentInput->h_candidate_read_ids.data() + end1,
                        currentInput->h_candidate_read_ids.data() + begin2,
                        currentInput->h_candidate_read_ids.data() + end2,
                        pairedPositions.begin(),
                        pairedPositions2.begin()
                    );

                    pairedPositions.erase(endIters.first, pairedPositions.end());
                    pairedPositions2.erase(endIters.second, pairedPositions2.end());
                    for(auto i : pairedPositions){
                        h_isPairedCandidate[begin1 + i] = true;
                    }
                    for(auto i : pairedPositions2){
                        h_isPairedCandidate[begin2 + i] = true;
                    }

                    numPairedPerAnchor[2*ap + 0] = pairedPositions.size();
                    numPairedPerAnchor[2*ap + 1] = pairedPositions2.size();                
                }

                cudaMemcpyAsync(
                    d_isPairedCandidate.data(),
                    h_isPairedCandidate.data(),
                    sizeof(bool) * currentNumCandidates,
                    H2D,
                    stream
                ); CUERR;


                #if 0
                    // cudaDeviceSynchronize(); CUERR;

                    // std::cerr << "currentNumCandidates = " << currentNumCandidates << "\n";
                    // std::cerr << "isPairedCandidate\n";
                    // for(int i = 0; i < currentNumCandidates; i++){
                    //     std::cerr << d_isPairedCandidate[i] << " ";
                    // }
                    // std::cerr << "\n";
                    // std::cerr << "d_anchorIndicesOfCandidates\n";
                    // for(int i = 0; i < currentNumCandidates; i++){
                    //     std::cerr << d_anchorIndicesOfCandidates[i] << " ";
                    // }
                    // std::cerr << "\n";
                    // std::cerr << "d_candidate_read_ids\n";
                    // for(int i = 0; i < currentNumCandidates; i++){
                    //     std::cerr << d_candidate_read_ids[i] << " ";
                    // }
                    // std::cerr << "\n";


                    //remove candidates which are not paired
                    std::size_t cubTempSize = d_tempstorage.sizeInBytes();
                    cudaError_t cubstatus = cudaSuccess;
                    cubstatus = cub::DeviceSelect::Flagged(
                        d_tempstorage.get(),
                        cubTempSize,
                        thrust::make_zip_iterator(thrust::make_tuple(
                            d_candidate_read_ids.data(),
                            d_anchorIndicesOfCandidates.data()
                        )),                        
                        thrust::make_transform_iterator(d_isPairedCandidate.data(), thrust::identity<bool>()),
                        thrust::make_zip_iterator(thrust::make_tuple(
                            d_candidate_read_ids2.data(),
                            d_anchorIndicesOfCandidates2.data()
                        )),
                        d_numCandidates.get(),
                        currentNumCandidates,
                        stream
                    );
                    assert(cubstatus == cudaSuccess);

                    // cudaDeviceSynchronize(); CUERR;

                    // std::cerr << "currentNumCandidates2 = " << *d_numCandidates << "\n";

                    // std::cerr << "d_anchorIndicesOfCandidates2\n";
                    // for(int i = 0; i < *d_numCandidates; i++){
                    //     std::cerr << d_anchorIndicesOfCandidates2[i] << " ";
                    // }
                    // std::cerr << "\n";
                    // std::cerr << "d_candidate_read_ids2\n";
                    // for(int i = 0; i < *d_numCandidates; i++){
                    //     std::cerr << d_candidate_read_ids2[i] << " ";
                    // }
                    // std::cerr << "\n";

                    cudaMemcpyAsync(
                        h_num_indices.data(),
                        d_numCandidates.data(),
                        sizeof(int),
                        D2H,
                        stream
                    ); CUERR;
                    cudaStreamSynchronize(stream); CUERR;

                    auto oldNumCandidates = currentNumCandidates;
                    currentNumCandidates = *h_num_indices;

                    cudaMemcpyAsync(
                        currentInput->h_candidate_read_ids.data(),
                        d_candidate_read_ids2.data(),
                        sizeof(int) * currentNumCandidates,
                        D2H,
                        stream
                    ); CUERR;
                    cudaEventRecord(events[1], stream); CUERR;

                    //cudaDeviceSynchronize(); CUERR;

                    std::swap(d_candidate_read_ids, d_candidate_read_ids2);
                    std::swap(d_anchorIndicesOfCandidates, d_anchorIndicesOfCandidates2);

                    cudaMemsetAsync(
                        d_candidates_per_anchor.data(),
                        0,
                        sizeof(int) * currentNumAnchors,
                        stream
                    ); CUERR;

                    //cudaDeviceSynchronize(); CUERR;

                    if(currentNumCandidates > 0){

                        int* d_uniqueAnchorIndices = d_anchorIndicesOfCandidates2.data();
                        int* d_aggregates_out = d_indices.data();

                        cubstatus = cub::DeviceReduce::ReduceByKey(
                            d_tempstorage.get(),
                            cubTempSize,
                            d_anchorIndicesOfCandidates.data(), 
                            d_uniqueAnchorIndices, 
                            thrust::make_constant_iterator(1), 
                            d_aggregates_out, 
                            d_num_indices.data(), 
                            cub::Sum(), 
                            currentNumCandidates, 
                            stream
                        );
                        assert(cubstatus == cudaSuccess);

                        //cudaDeviceSynchronize(); CUERR;

                        helpers::lambda_kernel<<<4, 256, 0, stream>>>(
                            [
                                d_uniqueAnchorIndices,
                                d_aggregates_out,
                                d_candidates_per_anchor = d_candidates_per_anchor.data(),
                                d_num_indices = this->d_num_indices.data()
                            ] __device__ (){
                                
                                const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                                const int stride = blockDim.x * gridDim.x;

                                for(int i = tid; i < *d_num_indices; i += stride){
                                    d_candidates_per_anchor[d_uniqueAnchorIndices[i]]
                                        = d_aggregates_out[i];
                                }
                            }
                        ); CUERR;

                        //cudaDeviceSynchronize(); CUERR;

                        cubstatus = cub::DeviceScan::InclusiveSum(
                            d_tempstorage.get(),
                            cubTempSize,
                            d_candidates_per_anchor.data(),
                            d_candidates_per_anchor_prefixsum.data() + 1,
                            currentNumAnchors,
                            stream
                        );
                        assert(cubstatus == cudaSuccess);

                        // cudaDeviceSynchronize(); CUERR;
                        // std::cerr << "d_candidates_per_anchor\n";
                        // for(int i = 0; i < currentNumAnchors; i++){
                        //     std::cerr << d_candidates_per_anchor[i] << " ";
                        // }
                        // std::cerr << "\n";
                        // std::cerr << "d_candidates_per_anchor_prefixsum\n";
                        // for(int i = 0; i < currentNumAnchors + 1; i++){
                        //     std::cerr << d_candidates_per_anchor_prefixsum[i] << " ";
                        // }
                        // std::cerr << "\n";

                        //update host candidate read ids

                        // std::remove_if(
                        //     currentInput->h_candidate_read_ids.begin(),
                        //     currentInput->h_candidate_read_ids.begin() + oldNumCandidates,
                        //     [&](const read_number& id){
                        //         const std::size_t i = std::distance((const read_number*)&(*currentInput->h_candidate_read_ids.begin()), &id);
                        //         return h_isPairedCandidate[i];
                        //     }
                        // );

                    }

                    cudaEventSynchronize(events[1]); CUERR; //wait for currentInput->h_candidateReadIds

                    

                #endif
            }else{
                cudaMemsetAsync(
                    d_isPairedCandidate.data(),
                    0,
                    sizeof(bool) * currentNumCandidates,
                    stream
                );
            }
        }

        void execute(cudaStream_t stream){

            nvtx::push_range("getCandidateAlignments", 5);
            getCandidateAlignments(stream); 
            nvtx::pop_range();

            if(correctionOptions->useQualityScores) {
                events[0].record(stream);
                backgroundStream.waitEvent(events[0], 0);
                
                nvtx::push_range("getQualities", 4);

                getQualities(backgroundStream);

                nvtx::pop_range();

                events[0].record(backgroundStream);
                cudaStreamWaitEvent(stream, events[0], 0); CUERR;
            }

            nvtx::push_range("buildMultipleSequenceAlignment", 6);
            buildMultipleSequenceAlignment(stream);
            nvtx::pop_range();

            if(useMsaRefinement()){

                nvtx::push_range("refineMultipleSequenceAlignment", 7);
                refineMultipleSequenceAlignment(stream);
                nvtx::pop_range();

            }

            nvtx::push_range("correctanchors", 8);
            correctAnchors(stream);
            nvtx::pop_range();

            if(correctionOptions->correctCandidates) {                        

                nvtx::push_range("correctCandidates", 9);
                correctCandidates(stream);
                nvtx::pop_range();
                
            }
        }

        void copyAnchorResultsFromDeviceToHost(cudaStream_t stream){
            if(correctionOptions->correctionType == CorrectionType::Classic){
                copyAnchorResultsFromDeviceToHostClassic(stream);
            }else if(correctionOptions->correctionType == CorrectionType::Forest){
                copyAnchorResultsFromDeviceToHostForestGpu(stream);
            }else{
                throw std::runtime_error("copyAnchorResultsFromDeviceToHost not implemented for this correctionType");
            }
        }

        void copyAnchorResultsFromDeviceToHostClassic(cudaStream_t stream){

            if(int(d_indices.capacity()) < currentNumAnchors + 1){
                cudaStreamSynchronize(stream); CUERR;
                d_indices.resize(currentNumAnchors+1);
            }

            int* const d_editsOffsetsTmp = d_indices.data();
            int* const d_totalNumberOfEdits = d_editsOffsetsTmp + currentNumAnchors;
            helpers::call_fill_kernel_async(d_editsOffsetsTmp, 1, 0, stream); CUERR;

            cudaError_t cubstatus = cudaSuccess;
            std::size_t cubTempSize = d_tempstorage.sizeInBytes();

            //num edits per anchor prefixsum
            cubstatus = cub::DeviceScan::InclusiveSum(
                d_tempstorage.get(), 
                cubTempSize, 
                thrust::make_transform_iterator(
                    d_numEditsPerCorrectedanchor.data(),
                    [doNotUseEditsValue = getDoNotUseEditsValue()] __device__ (const auto& num){ return num == doNotUseEditsValue ? 0 : num;}
                ), 
                d_editsOffsetsTmp + 1, 
                currentNumAnchors,
                stream
            );
            assert(cubstatus == cudaSuccess);

            //compact edits
            helpers::lambda_kernel<<<SDIV(currentNumAnchors, 128), 128, 0, stream>>>(
                [
                    d_editsPerCorrectedanchor = d_editsPerCorrectedanchor.data(),
                    d_editsPerCorrectedanchor2 = d_editsPerCorrectedCandidate2.data(),
                    d_editsOffsetsTmp,
                    d_numEditsPerCorrectedanchor = d_numEditsPerCorrectedanchor.data(),
                    doNotUseEditsValue = getDoNotUseEditsValue(),
                    editsPitchInBytes = editsPitchInBytes,
                    currentNumAnchors = currentNumAnchors
                ] __device__ (){
                    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                    const int stride = blockDim.x * gridDim.x;

                    const int N = currentNumAnchors;

                    for(int c = tid; c < N; c += stride){
                        const int numEdits = d_numEditsPerCorrectedanchor[c];

                        if(numEdits != doNotUseEditsValue && numEdits > 0){
                            const int outputOffset = d_editsOffsetsTmp[c];

                            auto* outputPtr = d_editsPerCorrectedanchor2 + outputOffset;
                            const auto* inputPtr = (const TempCorrectedSequence::EncodedEdit*)(((const char*)d_editsPerCorrectedanchor) 
                                + c * editsPitchInBytes);
                            for(int e = 0; e < numEdits; e++){
                                outputPtr[e] = inputPtr[e];
                            }
                        }
                    }
                }
            ); CUERR;

            //copy compacted edits to host
            helpers::call_copy_n_kernel(
                (const int*)d_editsPerCorrectedCandidate2.data(), 
                thrust::make_transform_iterator(d_totalNumberOfEdits, [] __device__ (const int num){return SDIV(num * sizeof(TempCorrectedSequence::EncodedEdit), sizeof(int));}),//d_totalNumberOfEdits, 
                (int*)currentOutput->h_editsPerCorrectedanchor.data(), 
                currentNumAnchors, 
                stream
            );

            //copy other buffers to host
            helpers::call_copy_n_kernel(
                thrust::make_zip_iterator(thrust::make_tuple(
                    d_anchor_sequences_lengths.data(), 
                    d_anchor_is_corrected.data(),
                    d_is_high_quality_anchor.data(),
                    d_numEditsPerCorrectedanchor.data(),
                    d_editsOffsetsTmp
                )), 
                currentNumAnchors, 
                thrust::make_zip_iterator(thrust::make_tuple(
                    currentOutput->h_anchor_sequences_lengths.data(), 
                    currentOutput->h_anchor_is_corrected.data(),
                    currentOutput->h_is_high_quality_anchor.data(),
                    currentOutput->h_numEditsPerCorrectedanchor.data(),
                    currentOutput->h_anchorEditOffsets.data()
                )), 
                stream
            );

            
            //compact corrected anchor sequences with numEdits == getDoNotUseEditsValue
            auto correctedAnchorsPitches = thrust::make_transform_iterator(
                d_numEditsPerCorrectedanchor.data(),
                ReplaceNumberOp(getDoNotUseEditsValue(), decodedSequencePitchInBytes)
            );

            int* const d_correctedAnchorOffsetsTmp = d_indices.data();
            int* const d_totalCorrectedSequencesBytes = d_correctedAnchorOffsetsTmp + currentNumAnchors;
            helpers::call_fill_kernel_async(d_correctedAnchorOffsetsTmp, 1, 0, stream); CUERR;

            cubstatus = cub::DeviceScan::InclusiveSum(
                d_tempstorage.get(), 
                cubTempSize, 
                correctedAnchorsPitches, 
                d_correctedAnchorOffsetsTmp + 1, 
                currentNumAnchors,
                stream
            );
            assert(cubstatus == cudaSuccess);

            helpers::call_copy_n_kernel(
                d_correctedAnchorOffsetsTmp, 
                currentNumAnchors, 
                currentOutput->h_correctedAnchorsOffsets.data(), 
                stream
            );

            helpers::lambda_kernel<<<SDIV(currentNumAnchors, 128), 128, 0, stream>>>(
                [
                    d_corrected_anchors = d_corrected_anchors.data(),
                    d_corrected_anchors2 = d_corrected_candidates2.data(),
                    decodedSequencePitchInBytes = this->decodedSequencePitchInBytes,
                    d_numEditsPerCorrectedanchor = d_numEditsPerCorrectedanchor.data(),
                    doNotUseEditsValue = getDoNotUseEditsValue(),
                    d_correctedAnchorOffsetsTmp,
                    currentNumAnchors = currentNumAnchors
                ] __device__ (){

                    const int N = currentNumAnchors;

                    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
                    const int warpId = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
                    const int numWarps = (blockDim.x * gridDim.x) / 32;

                    for(int c = warpId; c < N; c += numWarps){
                        const int numEdits = d_numEditsPerCorrectedanchor[c];

                        if(numEdits == doNotUseEditsValue){
                            const int outputOffset = d_correctedAnchorOffsetsTmp[c];

                            char* outputPtr = d_corrected_anchors2 + outputOffset;
                            const char* inputPtr = d_corrected_anchors + c * decodedSequencePitchInBytes;

                            const int copyInts = (decodedSequencePitchInBytes) / sizeof(int);
                            const int remainingBytes = (decodedSequencePitchInBytes) - copyInts * sizeof(int);
                            for(int i = warp.thread_rank(); i < copyInts; i += warp.size()){
                                ((int*)outputPtr)[i] = ((const int*)inputPtr)[i];
                            }
                
                            if(warp.thread_rank() < remainingBytes){
                                ((char*)(((int*)outputPtr) + copyInts))[warp.thread_rank()]
                                    = ((const char*)(((const int*)inputPtr) + copyInts))[warp.thread_rank()];
                            }
                        }
                    }
                }
            ); CUERR;

            assert(decodedSequencePitchInBytes % sizeof(int) == 0);

            //copy compacted anchor corrections to host
            helpers::call_copy_n_kernel(
                (const int*)d_corrected_candidates2.data(), 
                thrust::make_transform_iterator(d_totalCorrectedSequencesBytes, [] __device__ (const int num){return SDIV(num, sizeof(int));}),
                (int*)currentOutput->h_corrected_anchors.data(), 
                currentNumAnchors * decodedSequencePitchInBytes / sizeof(int),
                stream
            );            

        }

        void copyAnchorResultsFromDeviceToHostForestGpu(cudaStream_t stream){
            copyAnchorResultsFromDeviceToHostClassic(stream);
        }

        void copyCandidateResultsFromDeviceToHost(cudaStream_t stream){
            if(correctionOptions->correctionTypeCands == CorrectionType::Classic){
                copyCandidateResultsFromDeviceToHostClassic(stream);
            }else if(correctionOptions->correctionTypeCands == CorrectionType::Forest){
                copyCandidateResultsFromDeviceToHostForestGpu(stream);
            }else{
                throw std::runtime_error("copyCandidateResultsFromDeviceToHost not implemented for this correctionTypeCands");
            }
        }

        void copyCandidateResultsFromDeviceToHostClassic(cudaStream_t stream){
            helpers::call_copy_n_kernel(
                d_num_total_corrected_candidates.data(),
                1,
                h_num_total_corrected_candidates.data(),
                stream
            );

            cudaError_t cubstatus = cudaSuccess;
            size_t cubTempSize = d_tempstorage.sizeInBytes();

            cubstatus = cub::DeviceScan::ExclusiveSum(
                d_tempstorage.get(), 
                cubTempSize, 
                d_num_corrected_candidates_per_anchor.get(), 
                d_num_corrected_candidates_per_anchor_prefixsum.get(), 
                maxAnchors, 
                stream
            );
            assert(cubstatus == cudaSuccess);

            helpers::call_copy_n_kernel(
                thrust::make_zip_iterator(thrust::make_tuple(
                    d_num_corrected_candidates_per_anchor_prefixsum.data(), 
                    d_num_corrected_candidates_per_anchor.data()
                )),
                currentNumAnchors,
                thrust::make_zip_iterator(thrust::make_tuple(
                    currentOutput->h_num_corrected_candidates_per_anchor_prefixsum.data(), 
                    currentOutput->h_num_corrected_candidates_per_anchor.data()
                )),
                stream
            );

            

            cudaEventSynchronize(events[1]); CUERR; //wait for h_numRemainingCandidatesAfterAlignment
            if(*h_numRemainingCandidatesAfterAlignment <= 0) return;

            helpers::call_compact_kernel_async(
                thrust::make_zip_iterator(thrust::make_tuple(
                    d_alignment_shifts2.data(), 
                    d_candidate_read_ids2.data(),
                    d_candidate_sequences_lengths2.data()
                )),
                thrust::make_zip_iterator(thrust::make_tuple(
                    d_alignment_shifts.data(), 
                    d_candidate_read_ids.data(),
                    d_candidate_sequences_lengths.data()
                )),
                d_indices_of_corrected_candidates.data(), 
                d_num_total_corrected_candidates.data(), 
                *h_numRemainingCandidatesAfterAlignment,
                stream
            );

            //compute edit offsets for compacted edits
            auto inputIter2 = thrust::make_transform_iterator(
                d_numEditsPerCorrectedCandidate.data(),
                [doNotUseEditsValue = getDoNotUseEditsValue()] __device__ (const auto& num){ return num == doNotUseEditsValue ? 0 : num;}
            );

            if(int(d_indices.capacity()) < (*h_numRemainingCandidatesAfterAlignment) + 1){
                cudaStreamSynchronize(stream); CUERR;
                d_indices.resize((*h_numRemainingCandidatesAfterAlignment)+1);
            }

            int* const d_editsOffsetsTmp = d_indices.data();
            int* const d_totalNumberOfEdits = d_editsOffsetsTmp + (*h_numRemainingCandidatesAfterAlignment);
            helpers::call_fill_kernel_async(d_editsOffsetsTmp, 1, 0, stream); CUERR;

            cubstatus = cub::DeviceScan::InclusiveSum(
                d_tempstorage.get(), 
                cubTempSize, 
                inputIter2, 
                d_editsOffsetsTmp + 1, 
                *h_numRemainingCandidatesAfterAlignment,
                stream
            );
            assert(cubstatus == cudaSuccess);

            //compact edits
            helpers::lambda_kernel<<<SDIV(*h_numRemainingCandidatesAfterAlignment, 128), 128, 0, stream>>>(
                [
                    d_editsPerCorrectedCandidate = d_editsPerCorrectedCandidate.data(),
                    d_editsPerCorrectedCandidate2 = d_editsPerCorrectedCandidate2.data(),
                    d_editsOffsetsTmp,
                    d_num_total_corrected_candidates = d_num_total_corrected_candidates.data(),
                    d_numEditsPerCorrectedCandidate = d_numEditsPerCorrectedCandidate.data(),
                    doNotUseEditsValue = getDoNotUseEditsValue(),
                    editsPitchInBytes = editsPitchInBytes
                ] __device__ (){
                    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                    const int stride = blockDim.x * gridDim.x;

                    const int N = *d_num_total_corrected_candidates;

                    for(int c = tid; c < N; c += stride){
                        const int numEdits = d_numEditsPerCorrectedCandidate[c];

                        if(numEdits != doNotUseEditsValue && numEdits > 0){
                            const int outputOffset = d_editsOffsetsTmp[c];

                            auto* outputPtr = d_editsPerCorrectedCandidate2 + outputOffset;
                            const auto* inputPtr = (const TempCorrectedSequence::EncodedEdit*)(((const char*)d_editsPerCorrectedCandidate) 
                                + c * editsPitchInBytes);
                            for(int e = 0; e < numEdits; e++){
                                outputPtr[e] = inputPtr[e];
                            }
                        }
                    }
                }
            ); CUERR;

            //copy compact edits to host
            helpers::call_copy_n_kernel(
                (const int*)d_editsPerCorrectedCandidate2.data(), 
                thrust::make_transform_iterator(d_totalNumberOfEdits, [] __device__ (const int num){return SDIV(num * sizeof(TempCorrectedSequence::EncodedEdit), sizeof(int));}),//d_totalNumberOfEdits, 
                (int*)currentOutput->h_editsPerCorrectedCandidate.data(), 
                (currentNumCandidates / 10), 
                stream
            );

            helpers::call_copy_n_kernel(
                thrust::make_zip_iterator(thrust::make_tuple(
                    d_alignment_shifts2.data(), 
                    d_candidate_read_ids2.data(),
                    d_candidate_sequences_lengths2.data(),
                    d_indices_of_corrected_candidates.data(),
                    d_editsOffsetsTmp,
                    d_numEditsPerCorrectedCandidate.data()
                )), 
                d_num_total_corrected_candidates.data(), 
                thrust::make_zip_iterator(thrust::make_tuple(
                    currentOutput->h_alignment_shifts.data(), 
                    currentOutput->h_candidate_read_ids.data(),
                    currentOutput->h_candidate_sequences_lengths.data(),
                    currentOutput->h_indices_of_corrected_candidates.data(),
                    currentOutput->h_candidateEditOffsets.data(),
                    currentOutput->h_numEditsPerCorrectedCandidate.data()
                )), 
                *h_numRemainingCandidatesAfterAlignment,
                stream
            );

            //compact corrected candidate sequences
            auto correctedCandidatesPitches = thrust::make_transform_iterator(
                d_numEditsPerCorrectedCandidate.data(),
                ReplaceNumberOp(getDoNotUseEditsValue(), decodedSequencePitchInBytes)
            );

            int* const d_correctedCandidatesOffsetsTmp = d_indices.data();
            int* const d_totalCorrectedSequencesBytes = d_correctedCandidatesOffsetsTmp + (*h_numRemainingCandidatesAfterAlignment);
            helpers::call_fill_kernel_async(d_correctedCandidatesOffsetsTmp, 1, 0, stream); CUERR;

            cubstatus = cub::DeviceScan::InclusiveSum(
                d_tempstorage.get(), 
                cubTempSize, 
                correctedCandidatesPitches, 
                d_correctedCandidatesOffsetsTmp + 1, 
                *h_numRemainingCandidatesAfterAlignment,
                stream
            );
            assert(cubstatus == cudaSuccess);

            helpers::call_copy_n_kernel(
                d_correctedCandidatesOffsetsTmp, 
                d_num_total_corrected_candidates.data(), 
                currentOutput->h_correctedCandidatesOffsets.data(), 
                *h_numRemainingCandidatesAfterAlignment, //currentNumCandidates, 
                stream
            );

            helpers::lambda_kernel<<<SDIV(*h_numRemainingCandidatesAfterAlignment, 128), 128, 0, stream>>>(
                [
                    d_corrected_candidates = d_corrected_candidates.data(),
                    d_corrected_candidates2 = d_corrected_candidates2.data(),
                    decodedSequencePitchInBytes = this->decodedSequencePitchInBytes,
                    d_num_total_corrected_candidates = d_num_total_corrected_candidates.data(),
                    d_numEditsPerCorrectedCandidate = d_numEditsPerCorrectedCandidate.data(),
                    doNotUseEditsValue = getDoNotUseEditsValue(),
                    d_correctedCandidatesOffsetsTmp
                ] __device__ (){

                    const int N = *d_num_total_corrected_candidates;

                    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
                    const int warpId = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
                    const int numWarps = (blockDim.x * gridDim.x) / 32;

                    for(int c = warpId; c < N; c += numWarps){
                        const int numEdits = d_numEditsPerCorrectedCandidate[c];

                        if(numEdits == doNotUseEditsValue){
                            const int outputOffset = d_correctedCandidatesOffsetsTmp[c];

                            char* outputPtr = d_corrected_candidates2 + outputOffset;
                            const char* inputPtr = d_corrected_candidates + c * decodedSequencePitchInBytes;

                            const int copyInts = (decodedSequencePitchInBytes) / sizeof(int);
                            const int remainingBytes = (decodedSequencePitchInBytes) - copyInts * sizeof(int);
                            for(int i = warp.thread_rank(); i < copyInts; i += warp.size()){
                                ((int*)outputPtr)[i] = ((const int*)inputPtr)[i];
                            }
                
                            if(warp.thread_rank() < remainingBytes){
                                ((char*)(((int*)outputPtr) + copyInts))[warp.thread_rank()]
                                    = ((const char*)(((const int*)inputPtr) + copyInts))[warp.thread_rank()];
                            }
                        }
                    }
                }
            ); CUERR;

            assert(decodedSequencePitchInBytes % sizeof(int) == 0);

            helpers::call_copy_n_kernel(
                (const int*)d_corrected_candidates2.data(), 
                thrust::make_transform_iterator(d_totalCorrectedSequencesBytes, [] __device__ (const int num){return SDIV(num, sizeof(int));}),
                (int*)currentOutput->h_corrected_candidates.data(), 
                (*h_numRemainingCandidatesAfterAlignment) * decodedSequencePitchInBytes / sizeof(int),
                stream
            );

        }

        void copyCandidateResultsFromDeviceToHostForestGpu(cudaStream_t stream){
            copyCandidateResultsFromDeviceToHostClassic(stream);
        }

        void getAmbiguousFlagsOfAnchors(cudaStream_t stream){

            gpuReadStorage->areSequencesAmbiguous(
                readstorageHandle,
                d_anchorContainsN.get(), 
                d_anchorReadIds.get(), 
                currentNumAnchors,
                stream
            );
        }

        void getAmbiguousFlagsOfCandidates(cudaStream_t stream){
            gpuReadStorage->areSequencesAmbiguous(
                readstorageHandle,
                d_candidateContainsN.get(), 
                d_candidate_read_ids.get(), 
                currentNumCandidates,
                stream
            ); 
        }

        void getCandidateSequenceData(cudaStream_t stream){

            gpuReadStorage->gatherSequenceLengths(
                readstorageHandle,
                d_candidate_sequences_lengths.get(),
                d_candidate_read_ids.get(),
                currentNumCandidates,            
                stream
            );

            gpuReadStorage->gatherSequences(
                readstorageHandle,
                d_candidate_sequences_data.get(),
                encodedSequencePitchInInts,
                currentInput->h_candidate_read_ids,
                d_candidate_read_ids,
                currentNumCandidates,
                stream
            );

            helpers::call_transpose_kernel(
                d_transposedCandidateSequencesData.get(), 
                d_candidate_sequences_data.get(), 
                currentNumCandidates, 
                encodedSequencePitchInInts, 
                encodedSequencePitchInInts, 
                stream
            );
        }

        void getQualities(cudaStream_t stream){

            if(correctionOptions->useQualityScores) {

//#define COMPACT_GATHER

#ifndef COMPACT_GATHER

                gpuReadStorage->gatherQualities(
                    readstorageHandle,
                    d_anchor_qualities,
                    qualityPitchInBytes,
                    currentInput->h_anchorReadIds,
                    d_anchorReadIds,
                    maxAnchors,
                    stream
                );

                gpuReadStorage->gatherQualities(
                    readstorageHandle,
                    d_candidate_qualities,
                    qualityPitchInBytes,
                    currentInput->h_candidate_read_ids.get(),
                    d_candidate_read_ids.get(),
                    currentNumCandidates,
                    stream
                );

#else 

                std::size_t cubTempSize = d_tempstorage.sizeInBytes();
                cudaError_t cubstatus = cub::DeviceScan::ExclusiveSum(
                    d_tempstorage.data(),
                    cubTempSize,
                    d_indices_per_anchor.data(),
                    d_indices_per_anchor_prefixsum.data(),
                    maxAnchors,
                    stream
                );
                assert(cubstatus == cudaSuccess);
                
                //from the list of remaining candidates per anchor, compact the corresponding candidate read ids
                helpers::lambda_kernel<<<maxAnchors, 128, 0, stream>>>(
                    [
                        h_indicesForGather = h_indicesForGather.data(),
                        d_indicesForGather = d_indicesForGather.data(),
                        d_indices = d_indices.data(),
                        d_indices_per_anchor = d_indices_per_anchor.data(),
                        d_indices_per_anchor_prefixsum = d_indices_per_anchor_prefixsum.data(),
                        d_num_indices = d_num_indices.data(),
                        d_candidates_per_anchor_prefixsum = d_candidates_per_anchor_prefixsum.data(),
                        d_candidate_read_ids = d_candidate_read_ids.data(),
                        currentNumAnchors = currentNumAnchors
                    ] __device__ (){

                        for(int anchor = blockIdx.x; anchor < currentNumAnchors; anchor += gridDim.x){

                            const int globalCandidateOffset = d_candidates_per_anchor_prefixsum[anchor];
                            const int* const myIndices = d_indices + globalCandidateOffset;
                            const int numIndices = d_indices_per_anchor[anchor];
                            const int offset = d_indices_per_anchor_prefixsum[anchor];

                            for(int i = threadIdx.x; i < numIndices; i += blockDim.x){
                                const int inputpos = myIndices[i];
                                d_indicesForGather[offset + i] = d_candidate_read_ids[globalCandidateOffset + inputpos];
                                h_indicesForGather[offset + i] = d_candidate_read_ids[globalCandidateOffset + inputpos];
                            }
                        }                   
                    }
                ); CUERR;

                cudaEventRecord(events[1], stream); CUERR;

                gpuReadStorage->gatherQualities(
                    readstorageHandle,
                    d_anchor_qualities,
                    qualityPitchInBytes,
                    currentInput->h_anchorReadIds,
                    d_anchorReadIds,
                    maxAnchors,
                    stream
                );

                //cudaStreamSynchronize(stream); CUERR; //wait for h_indicesForGather and h_numRemainingCandidatesAfterAlignment
                cudaEventSynchronize(events[1]); CUERR;
                const int hNumIndices = *h_numRemainingCandidatesAfterAlignment;

                nvtx::push_range("get compact qscores " + std::to_string(hNumIndices) + " " + std::to_string(currentNumCandidates), 6);
                gpuReadStorage->gatherQualities(
                    readstorageHandle,
                    d_candidate_qualities_compact,
                    qualityPitchInBytes,
                    h_indicesForGather.data(),
                    d_indicesForGather.data(),
                    currentNumCandidates,
                    stream
                );
                nvtx::pop_range();

                //scatter compact quality scores to correct positions
                helpers::lambda_kernel<<<maxAnchors, 256, 0, stream>>>(
                    [
                        d_candidate_qualities_compact = d_candidate_qualities_compact.data(),
                        d_candidate_qualities = d_candidate_qualities.data(),
                        d_candidate_sequences_lengths = d_candidate_sequences_lengths.data(),
                        qualityPitchInBytes = qualityPitchInBytes,
                        d_indices = d_indices.data(),
                        d_indices_per_anchor = d_indices_per_anchor.data(),
                        d_indices_per_anchor_prefixsum = d_indices_per_anchor_prefixsum.data(),
                        d_num_indices = d_num_indices.data(),
                        d_candidates_per_anchor_prefixsum = d_candidates_per_anchor_prefixsum.data(),
                        currentNumAnchors = currentNumAnchors
                    ] __device__ (){
                        constexpr int groupsize = 32;
                        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());

                        const int groupId = threadIdx.x / groupsize;
                        const int numgroups = blockDim.x / groupsize;

                        assert(qualityPitchInBytes % sizeof(int) == 0);

                        for(int anchor = blockIdx.x; anchor < currentNumAnchors; anchor += gridDim.x){

                            const int globalCandidateOffset = d_candidates_per_anchor_prefixsum[anchor];
                            const int* const myIndices = d_indices + globalCandidateOffset;
                            const int numIndices = d_indices_per_anchor[anchor];
                            const int offset = d_indices_per_anchor_prefixsum[anchor];

                            for(int c = groupId; c < numIndices; c += numgroups){
                                const int outputpos = globalCandidateOffset + myIndices[c];
                                const int inputpos = offset + c;
                                const int length = d_candidate_sequences_lengths[outputpos];

                                const int iters = SDIV(length, sizeof(int));

                                const int* const input = (const int*)(d_candidate_qualities_compact + size_t(inputpos) * qualityPitchInBytes);
                                int* const output = (int*)(d_candidate_qualities + size_t(outputpos) * qualityPitchInBytes);

                                for(int k = group.thread_rank(); k < iters; k += group.size()){
                                    output[k] = input[k];
                                }
                            }
                        }
                    }
                );

                // cudaStreamSynchronize(stream); CUERR; //wait for candidateQualitiesGatherHandle

                // // std::cerr << "gather candidate qual\n";
                // gpuReadStorage->gatherQualitiesToGpuBufferAsync(
                //     threadPool,
                //     candidateQualitiesGatherHandle,
                //     d_candidate_qualities,
                //     qualityPitchInBytes,
                //     currentInput->h_candidate_read_ids.get(),
                //     d_candidate_read_ids.get(),
                //     currentNumCandidates,
                //     deviceId,
                //     stream
                // );
#undef COMPACT_GATHER                
#endif                

            }
        }

        void getCandidateAlignments(cudaStream_t stream){


            std::size_t tempBytes = d_tempstorage.sizeInBytes();

            const bool removeAmbiguousAnchors = correctionOptions->excludeAmbiguousReads;
            const bool removeAmbiguousCandidates = correctionOptions->excludeAmbiguousReads;

            call_popcount_shifted_hamming_distance_kernel_async(
                d_tempstorage.get(),
                tempBytes,
                d_alignment_overlaps.get(),
                d_alignment_shifts.get(),
                d_alignment_nOps.get(),
                d_alignment_isValid.get(),
                d_alignment_best_alignment_flags.get(),
                d_anchor_sequences_data.get(),
                d_candidate_sequences_data.get(),
                d_anchor_sequences_lengths.get(),
                d_candidate_sequences_lengths.get(),
                d_candidates_per_anchor_prefixsum.get(),
                d_candidates_per_anchor.get(),
                d_anchorIndicesOfCandidates.get(),
                d_numAnchors.get(),
                d_numCandidates.get(),
                d_anchorContainsN.get(),
                removeAmbiguousAnchors,
                d_candidateContainsN.get(),
                removeAmbiguousCandidates,
                maxAnchors,
                maxCandidates,
                gpuReadStorage->getSequenceLengthUpperBound(),
                encodedSequencePitchInInts,
                goodAlignmentProperties->min_overlap,
                goodAlignmentProperties->maxErrorRate,
                goodAlignmentProperties->min_overlap_ratio,
                correctionOptions->estimatedErrorrate,
                stream,
                kernelLaunchHandle
            );

            #if 1
            if(false && !gpuReadStorage->isPairedEnd()){
                //default kernel
                call_cuda_filter_alignments_by_mismatchratio_kernel_async(
                    d_alignment_best_alignment_flags.get(),
                    d_alignment_nOps.get(),
                    d_alignment_overlaps.get(),
                    d_candidates_per_anchor_prefixsum.get(),
                    d_numAnchors.get(),
                    d_numCandidates.get(),
                    maxAnchors,
                    maxCandidates,
                    correctionOptions->estimatedErrorrate,
                    correctionOptions->estimatedCoverage * correctionOptions->m_coverage,
                    stream,
                    kernelLaunchHandle
                );
            }else{
                helpers::lambda_kernel<<<currentNumAnchors, 128, 0, stream>>>(
                    [
                        bestAlignmentFlags = d_alignment_best_alignment_flags.data(),
                        nOps = d_alignment_nOps.data(),
                        overlaps = d_alignment_overlaps.data(),
                        d_candidates_per_subject_prefixsum = d_candidates_per_anchor_prefixsum.data(),
                        n_subjects = currentNumAnchors,
                        mismatchratioBaseFactor = correctionOptions->estimatedErrorrate,
                        goodAlignmentsCountThreshold = correctionOptions->estimatedCoverage * correctionOptions->m_coverage,
                        d_isPairedCandidate = d_isPairedCandidate.data(),
                        pairedthreshold1 = correctionOptions->pairedthreshold1
                    ] __device__(){
                        using BlockReduceInt = cub::BlockReduce<int, 128>;

                        __shared__ union {
                            typename BlockReduceInt::TempStorage intreduce;
                            int broadcast[3];
                        } temp_storage;

                        for(int subjectindex = blockIdx.x; subjectindex < n_subjects; subjectindex += gridDim.x) {

                            const int candidatesForSubject = d_candidates_per_subject_prefixsum[subjectindex+1]
                                                            - d_candidates_per_subject_prefixsum[subjectindex];

                            const int firstIndex = d_candidates_per_subject_prefixsum[subjectindex];

                            //printf("subjectindex %d\n", subjectindex);

                            int counts[3]{0,0,0};

                            //if(threadIdx.x == 0){
                            //    printf("my_n_indices %d\n", my_n_indices);
                            //}

                            for(int index = threadIdx.x; index < candidatesForSubject; index += blockDim.x) {

                                const int candidate_index = firstIndex + index;
                                if(!d_isPairedCandidate[candidate_index]){
                                    if(bestAlignmentFlags[candidate_index] != BestAlignment_t::None) {

                                        const int alignment_overlap = overlaps[candidate_index];
                                        const int alignment_nops = nOps[candidate_index];

                                        const float mismatchratio = float(alignment_nops) / alignment_overlap;

                                        //if(mismatchratio >= 1 * mismatchratioBaseFactor) {
                                        if(mismatchratio >= pairedthreshold1) {
                                            bestAlignmentFlags[candidate_index] = BestAlignment_t::None;
                                        }else{

                                            // #pragma unroll
                                            // for(int i = 2; i <= 2; i++) {
                                            //     counts[i-2] += (mismatchratio < i * mismatchratioBaseFactor);
                                            // }
                                        }

                                    }
                                }
                            }

                            // //accumulate counts over block
                            //     #pragma unroll
                            // for(int i = 0; i < 3; i++) {
                            //     counts[i] = BlockReduceInt(temp_storage.intreduce).Sum(counts[i]);
                            //     __syncthreads();
                            // }

                            // //broadcast accumulated counts to block
                            // if(threadIdx.x == 0) {
                            //     #pragma unroll
                            //     for(int i = 0; i < 3; i++) {
                            //         temp_storage.broadcast[i] = counts[i];
                            //         //printf("count[%d] = %d\n", i, counts[i]);
                            //     }
                            //     //printf("mismatchratioBaseFactor %f, goodAlignmentsCountThreshold %f\n", mismatchratioBaseFactor, goodAlignmentsCountThreshold);
                            // }

                            // __syncthreads();

                            // #pragma unroll
                            // for(int i = 0; i < 3; i++) {
                            //     counts[i] = temp_storage.broadcast[i];
                            // }

                            // float mismatchratioThreshold = 0;
                            // if (counts[0] >= goodAlignmentsCountThreshold) {
                            //     mismatchratioThreshold = 2 * mismatchratioBaseFactor;
                            // } else if (counts[1] >= goodAlignmentsCountThreshold) {
                            //     mismatchratioThreshold = 3 * mismatchratioBaseFactor;
                            // } else if (counts[2] >= goodAlignmentsCountThreshold) {
                            //     mismatchratioThreshold = 4 * mismatchratioBaseFactor;
                            // } else {
                            //     mismatchratioThreshold = -1.0f;                         //this will invalidate all alignments for subject
                            //     //mismatchratioThreshold = 4 * mismatchratioBaseFactor; //use alignments from every bin
                            //     //mismatchratioThreshold = 1.1f;
                            // }

                            // // Invalidate all alignments for subject with mismatchratio >= mismatchratioThreshold which are not paired end
                            // for(int index = threadIdx.x; index < candidatesForSubject; index += blockDim.x) {
                            //     const int candidate_index = firstIndex + index;

                            //     if(!d_isPairedCandidate[candidate_index]){
                            //         if(bestAlignmentFlags[candidate_index] != BestAlignment_t::None) {

                            //             const int alignment_overlap = overlaps[candidate_index];
                            //             const int alignment_nops = nOps[candidate_index];

                            //             const float mismatchratio = float(alignment_nops) / alignment_overlap;

                            //             const bool doRemove = mismatchratio >= mismatchratioThreshold;
                            //             if(doRemove){
                            //                 bestAlignmentFlags[candidate_index] = BestAlignment_t::None;
                            //             }
                            //         }
                            //     }
                            // }
                        }
                    }
                );
            }
            #else
                //default kernel
                call_cuda_filter_alignments_by_mismatchratio_kernel_async(
                    d_alignment_best_alignment_flags.get(),
                    d_alignment_nOps.get(),
                    d_alignment_overlaps.get(),
                    d_candidates_per_anchor_prefixsum.get(),
                    d_numAnchors.get(),
                    d_numCandidates.get(),
                    maxAnchors,
                    maxCandidates,
                    correctionOptions->estimatedErrorrate,
                    correctionOptions->estimatedCoverage * correctionOptions->m_coverage,
                    stream,
                    kernelLaunchHandle
                );
            #endif

            callSelectIndicesOfGoodCandidatesKernelAsync(
                d_indices.get(),
                d_indices_per_anchor.get(),
                d_num_indices.get(),
                d_alignment_best_alignment_flags.get(),
                d_candidates_per_anchor.get(),
                d_candidates_per_anchor_prefixsum.get(),
                d_anchorIndicesOfCandidates.get(),
                d_numAnchors.get(),
                d_numCandidates.get(),
                maxAnchors,
                maxCandidates,
                stream,
                kernelLaunchHandle
            );

            cudaMemcpyAsync(
                h_numRemainingCandidatesAfterAlignment.get(),
                d_num_indices.get(),
                sizeof(int),
                D2H,
                stream
            );

            cudaEventRecord(events[1], stream); CUERR;

        }

        void buildMultipleSequenceAlignment(cudaStream_t stream){

            GPUMultiMSA multiMSA;

            multiMSA.numMSAs = maxAnchors;
            multiMSA.columnPitchInElements = msaColumnPitchInElements;
            multiMSA.counts = d_counts.get();
            multiMSA.weights = d_weights.get();
            multiMSA.coverages = d_coverage.get();
            multiMSA.consensus = d_consensus.get();
            multiMSA.support = d_support.get();
            multiMSA.origWeights = d_origWeights.get();
            multiMSA.origCoverages = d_origCoverages.get();
            multiMSA.columnProperties = d_msa_column_properties.get();

            callConstructMultipleSequenceAlignmentsKernel_async(
                multiMSA,
                d_alignment_overlaps.get(),
                d_alignment_shifts.get(),
                d_alignment_nOps.get(),
                d_alignment_best_alignment_flags.get(),
                d_anchor_sequences_lengths.get(),
                d_candidate_sequences_lengths.get(),
                d_indices.get(),
                d_indices_per_anchor.get(),
                d_candidates_per_anchor_prefixsum,
                d_anchor_sequences_data.get(),
                d_candidate_sequences_data.get(),
                d_isPairedCandidate.get(),
                d_anchor_qualities.get(),
                d_candidate_qualities.get(),
                d_numAnchors.get(),
                goodAlignmentProperties->maxErrorRate,
                maxAnchors,
                maxCandidates,
                correctionOptions->useQualityScores,
                encodedSequencePitchInInts,
                qualityPitchInBytes,
                stream,
                kernelLaunchHandle
            );
        }

        void refineMultipleSequenceAlignment(cudaStream_t stream){

            GPUMultiMSA multiMSA;

            multiMSA.numMSAs = maxAnchors;
            multiMSA.columnPitchInElements = msaColumnPitchInElements;
            multiMSA.counts = d_counts.get();
            multiMSA.weights = d_weights.get();
            multiMSA.coverages = d_coverage.get();
            multiMSA.consensus = d_consensus.get();
            multiMSA.support = d_support.get();
            multiMSA.origWeights = d_origWeights.get();
            multiMSA.origCoverages = d_origCoverages.get();
            multiMSA.columnProperties = d_msa_column_properties.get();

            const std::size_t requiredTempStorageBytes = sizeof(bool) * maxCandidates; // d_shouldBeKept
                
            assert(d_tempstorage.sizeInBytes() >= requiredTempStorageBytes);

            bool* d_shouldBeKept = (bool*)d_tempstorage.get();

            callMsaCandidateRefinementKernel_multiiter_async(
                d_indices_tmp.get(),
                d_indices_per_anchor_tmp.get(),
                d_num_indices_tmp.get(),
                multiMSA,
                d_alignment_best_alignment_flags.get(),
                d_alignment_shifts.get(),
                d_alignment_nOps.get(),
                d_alignment_overlaps.get(),
                d_anchor_sequences_data.get(),
                d_candidate_sequences_data.get(),
                d_isPairedCandidate.get(),
                d_anchor_sequences_lengths.get(),
                d_candidate_sequences_lengths.get(),
                d_anchor_qualities.get(),
                d_candidate_qualities.get(),
                d_shouldBeKept,
                d_candidates_per_anchor_prefixsum,
                d_numAnchors.get(),
                goodAlignmentProperties->maxErrorRate,
                maxAnchors,
                maxCandidates,
                correctionOptions->useQualityScores,
                encodedSequencePitchInInts,
                qualityPitchInBytes,
                d_indices.get(),
                d_indices_per_anchor.get(),
                correctionOptions->estimatedCoverage,
                getNumRefinementIterations(),
                stream,
                kernelLaunchHandle
            );

            std::swap(d_indices_tmp, d_indices);
            std::swap(d_indices_per_anchor_tmp, d_indices_per_anchor);
            std::swap(d_num_indices_tmp, d_num_indices);

        }

        void correctAnchors(cudaStream_t stream){
            if(correctionOptions->correctionType == CorrectionType::Classic){
                correctAnchorsClassic(stream);
            }else if(correctionOptions->correctionType == CorrectionType::Forest){
                correctAnchorsForestGpu(stream);
            }else{
                throw std::runtime_error("correctAnchors not implemented for this correctionType");
            }
        }

        void correctAnchorsClassic(cudaStream_t stream){

            const float avg_support_threshold = 1.0f - 1.0f * correctionOptions->estimatedErrorrate;
            const float min_support_threshold = 1.0f - 3.0f * correctionOptions->estimatedErrorrate;
            // coverage is always >= 1
            const float min_coverage_threshold = std::max(1.0f,
                correctionOptions->m_coverage / 6.0f * correctionOptions->estimatedCoverage);
            const float max_coverage_threshold = 0.5 * correctionOptions->estimatedCoverage;

            // correct anchors

            GPUMultiMSA multiMSA;

            multiMSA.numMSAs = maxAnchors;
            multiMSA.columnPitchInElements = msaColumnPitchInElements;
            multiMSA.counts = d_counts.get();
            multiMSA.weights = d_weights.get();
            multiMSA.coverages = d_coverage.get();
            multiMSA.consensus = d_consensus.get();
            multiMSA.support = d_support.get();
            multiMSA.origWeights = d_origWeights.get();
            multiMSA.origCoverages = d_origCoverages.get();
            multiMSA.columnProperties = d_msa_column_properties.get();


            call_msaCorrectAnchorsKernel_async(
                d_corrected_anchors.get(),
                d_anchor_is_corrected.get(),
                d_is_high_quality_anchor.get(),
                multiMSA,
                d_anchor_sequences_data.get(),
                d_candidate_sequences_data.get(),
                d_candidate_sequences_lengths.get(),
                d_indices_per_anchor.get(),
                d_numAnchors.get(),
                maxAnchors,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                gpuReadStorage->getSequenceLengthUpperBound(),
                correctionOptions->estimatedErrorrate,
                goodAlignmentProperties->maxErrorRate,
                avg_support_threshold,
                min_support_threshold,
                min_coverage_threshold,
                max_coverage_threshold,
                correctionOptions->kmerlength,
                gpuReadStorage->getSequenceLengthUpperBound(),
                stream,
                kernelLaunchHandle
            );

            gpucorrectorkernels::selectIndicesOfFlagsOneBlock<256><<<1,256,0, stream>>>(
                d_indices_of_corrected_anchors.get(),
                d_num_indices_of_corrected_anchors.get(),
                d_anchor_is_corrected.get(),
                d_numAnchors.get()
            ); CUERR;

            helpers::call_fill_kernel_async(d_numEditsPerCorrectedanchor.data(), currentNumAnchors, 0, stream);

            callConstructSequenceCorrectionResultsKernel(
                d_editsPerCorrectedanchor.get(),
                d_numEditsPerCorrectedanchor.get(),
                getDoNotUseEditsValue(),
                d_indices_of_corrected_anchors.get(),
                d_num_indices_of_corrected_anchors.get(),
                d_anchorContainsN.get(),
                d_anchor_sequences_data.get(),
                d_anchor_sequences_lengths.get(),
                d_corrected_anchors.get(),
                currentNumAnchors,
                false,
                maxNumEditsPerSequence,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                editsPitchInBytes,      
                stream,
                kernelLaunchHandle
            );
            
        }

        void correctAnchorsForestGpu(cudaStream_t stream){

            const float avg_support_threshold = 1.0f - 1.0f * correctionOptions->estimatedErrorrate;
            const float min_support_threshold = 1.0f - 3.0f * correctionOptions->estimatedErrorrate;
            // coverage is always >= 1
            const float min_coverage_threshold = std::max(1.0f,
                correctionOptions->m_coverage / 6.0f * correctionOptions->estimatedCoverage);
            const float max_coverage_threshold = 0.5 * correctionOptions->estimatedCoverage;

            // correct anchors

            GPUMultiMSA multiMSA;

            multiMSA.numMSAs = maxAnchors;
            multiMSA.columnPitchInElements = msaColumnPitchInElements;
            multiMSA.counts = d_counts.get();
            multiMSA.weights = d_weights.get();
            multiMSA.coverages = d_coverage.get();
            multiMSA.consensus = d_consensus.get();
            multiMSA.support = d_support.get();
            multiMSA.origWeights = d_origWeights.get();
            multiMSA.origCoverages = d_origCoverages.get();
            multiMSA.columnProperties = d_msa_column_properties.get();


            callMsaCorrectAnchorsWithForestKernel(
                d_corrected_anchors.get(),
                d_anchor_is_corrected.get(),
                d_is_high_quality_anchor.get(),
                multiMSA,
                *gpuForestAnchor,
                correctionOptions->thresholdAnchor,
                d_anchor_sequences_data.get(),
                d_indices_per_anchor.get(),
                currentNumAnchors,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                gpuReadStorage->getSequenceLengthUpperBound(),
                correctionOptions->estimatedErrorrate,
                goodAlignmentProperties->maxErrorRate,
                correctionOptions->estimatedCoverage,
                avg_support_threshold,
                min_support_threshold,
                min_coverage_threshold,
                max_coverage_threshold,
                stream,
                kernelLaunchHandle
            );

            gpucorrectorkernels::selectIndicesOfFlagsOneBlock<256><<<1,256,0, stream>>>(
                d_indices_of_corrected_anchors.get(),
                d_num_indices_of_corrected_anchors.get(),
                d_anchor_is_corrected.get(),
                d_numAnchors.get()
            ); CUERR;

            helpers::call_fill_kernel_async(d_numEditsPerCorrectedanchor.data(), currentNumAnchors, 0, stream);

            callConstructSequenceCorrectionResultsKernel(
                d_editsPerCorrectedanchor.get(),
                d_numEditsPerCorrectedanchor.get(),
                getDoNotUseEditsValue(),
                d_indices_of_corrected_anchors.get(),
                d_num_indices_of_corrected_anchors.get(),
                d_anchorContainsN.get(),
                d_anchor_sequences_data.get(),
                d_anchor_sequences_lengths.get(),
                d_corrected_anchors.get(),
                currentNumAnchors,
                false,
                maxNumEditsPerSequence,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                editsPitchInBytes,      
                stream,
                kernelLaunchHandle
            );

        }
        
        void correctCandidates(cudaStream_t stream){
            if(correctionOptions->correctionTypeCands == CorrectionType::Classic){
                correctCandidatesClassic(stream);
            }else if(correctionOptions->correctionTypeCands == CorrectionType::Forest){
                correctCandidatesForestGpu(stream);
            }else{
                throw std::runtime_error("correctCandidates not implemented for this correctionTypeCands");
            }
        }

        void correctCandidatesClassic(cudaStream_t stream){

            const float min_support_threshold = 1.0f-3.0f*correctionOptions->estimatedErrorrate;
            // coverage is always >= 1
            const float min_coverage_threshold = std::max(1.0f,
                correctionOptions->m_coverage / 6.0f * correctionOptions->estimatedCoverage);
            const int new_columns_to_correct = correctionOptions->new_columns_to_correct;

            bool* const d_candidateCanBeCorrected = d_alignment_isValid.get(); //repurpose

            cub::TransformInputIterator<bool, IsHqAnchor, AnchorHighQualityFlag*>
                d_isHqanchor(d_is_high_quality_anchor, IsHqAnchor{});

            gpucorrectorkernels::selectIndicesOfFlagsOneBlock<256><<<1,256,0, stream>>>(
                d_high_quality_anchor_indices.get(),
                d_num_high_quality_anchor_indices.get(),
                d_isHqanchor,
                d_numAnchors.get()
            ); CUERR;

            gpucorrectorkernels::initArraysBeforeCandidateCorrectionKernel<<<640, 128, 0, stream>>>(
                maxCandidates,
                d_numAnchors.get(),
                d_num_corrected_candidates_per_anchor.get(),
                d_candidateCanBeCorrected
            ); CUERR;

   
            GPUMultiMSA multiMSA;

            multiMSA.numMSAs = maxAnchors;
            multiMSA.columnPitchInElements = msaColumnPitchInElements;
            multiMSA.counts = d_counts.get();
            multiMSA.weights = d_weights.get();
            multiMSA.coverages = d_coverage.get();
            multiMSA.consensus = d_consensus.get();
            multiMSA.support = d_support.get();
            multiMSA.origWeights = d_origWeights.get();
            multiMSA.origCoverages = d_origCoverages.get();
            multiMSA.columnProperties = d_msa_column_properties.get();

            #if 1
                bool* d_excludeFlags = d_flagsCandidates.data();
                bool* h_excludeFlags = h_flagsCandidates.data();

                //corrections of candidates for which a high quality anchor correction exists will not be used
                //-> don't compute them
                for(int i = 0; i < currentNumCandidates; i++){
                    const read_number candidateReadId = currentInput->h_candidate_read_ids[i];
                    h_excludeFlags[i] = correctionFlags->isCorrectedAsHQAnchor(candidateReadId);
                }

                helpers::call_copy_n_kernel(
                    (const int*)h_excludeFlags,
                    SDIV(currentNumCandidates, sizeof(int)),
                    (int*)d_excludeFlags,
                    stream
                );

                callFlagCandidatesToBeCorrectedWithExcludeFlagsKernel(
                    d_candidateCanBeCorrected,
                    d_num_corrected_candidates_per_anchor.get(),
                    multiMSA,
                    d_excludeFlags,
                    d_alignment_shifts.get(),
                    d_candidate_sequences_lengths.get(),
                    d_anchorIndicesOfCandidates.get(),
                    d_is_high_quality_anchor.get(),
                    d_candidates_per_anchor_prefixsum,
                    d_indices.get(),
                    d_indices_per_anchor.get(),
                    d_numAnchors,
                    d_numCandidates,
                    min_support_threshold,
                    min_coverage_threshold,
                    new_columns_to_correct,
                    stream,
                    kernelLaunchHandle
                );
            #else
            callFlagCandidatesToBeCorrectedKernel_async(
                d_candidateCanBeCorrected,
                d_num_corrected_candidates_per_anchor.get(),
                multiMSA,
                d_alignment_shifts.get(),
                d_candidate_sequences_lengths.get(),
                d_anchorIndicesOfCandidates.get(),
                d_is_high_quality_anchor.get(),
                d_candidates_per_anchor_prefixsum,
                d_indices.get(),
                d_indices_per_anchor.get(),
                d_numAnchors,
                d_numCandidates,
                min_support_threshold,
                min_coverage_threshold,
                new_columns_to_correct,
                stream,
                kernelLaunchHandle
            );
            #endif

            size_t cubTempSize = d_tempstorage.sizeInBytes();

            cub::DeviceSelect::Flagged(
                d_tempstorage.get(),
                cubTempSize,
                cub::CountingInputIterator<int>(0),
                d_candidateCanBeCorrected,
                d_indices_of_corrected_candidates.get(),
                d_num_total_corrected_candidates.get(),
                maxCandidates,
                stream
            ); CUERR;

            cudaMemsetAsync(
                d_numEditsPerCorrectedCandidate.data(),
                0,
                sizeof(int) * currentNumCandidates,
                stream
            ); CUERR;

            callCorrectCandidatesKernel_async(
                d_corrected_candidates.get(),
                d_editsPerCorrectedCandidate.get(),
                d_numEditsPerCorrectedCandidate.get(),              
                multiMSA,
                d_alignment_shifts.get(),
                d_alignment_best_alignment_flags.get(),
                d_candidate_sequences_data.get(),
                d_candidate_sequences_lengths.get(),
                d_candidateContainsN.get(),
                d_indices_of_corrected_candidates.get(),
                d_num_total_corrected_candidates.get(),
                d_anchorIndicesOfCandidates.get(),
                d_numAnchors,
                d_numCandidates,
                getDoNotUseEditsValue(),
                maxNumEditsPerSequence,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                editsPitchInBytes,
                gpuReadStorage->getSequenceLengthUpperBound(),
                stream,
                kernelLaunchHandle
            );    
  
        }

        void correctCandidatesForestGpu(cudaStream_t stream){

            const float min_support_threshold = 1.0f-3.0f*correctionOptions->estimatedErrorrate;
            // coverage is always >= 1
            const float min_coverage_threshold = std::max(1.0f,
                correctionOptions->m_coverage / 6.0f * correctionOptions->estimatedCoverage);
            const int new_columns_to_correct = correctionOptions->new_columns_to_correct;

            bool* const d_candidateCanBeCorrected = d_alignment_isValid.get(); //repurpose

            cub::TransformInputIterator<bool, IsHqAnchor, AnchorHighQualityFlag*>
                d_isHqanchor(d_is_high_quality_anchor, IsHqAnchor{});

            gpucorrectorkernels::selectIndicesOfFlagsOneBlock<256><<<1,256,0, stream>>>(
                d_high_quality_anchor_indices.get(),
                d_num_high_quality_anchor_indices.get(),
                d_isHqanchor,
                d_numAnchors.get()
            ); CUERR;

            gpucorrectorkernels::initArraysBeforeCandidateCorrectionKernel<<<640, 128, 0, stream>>>(
                maxCandidates,
                d_numAnchors.get(),
                d_num_corrected_candidates_per_anchor.get(),
                d_candidateCanBeCorrected
            ); CUERR;

   
            GPUMultiMSA multiMSA;

            multiMSA.numMSAs = maxAnchors;
            multiMSA.columnPitchInElements = msaColumnPitchInElements;
            multiMSA.counts = d_counts.get();
            multiMSA.weights = d_weights.get();
            multiMSA.coverages = d_coverage.get();
            multiMSA.consensus = d_consensus.get();
            multiMSA.support = d_support.get();
            multiMSA.origWeights = d_origWeights.get();
            multiMSA.origCoverages = d_origCoverages.get();
            multiMSA.columnProperties = d_msa_column_properties.get();

            #if 1
                bool* d_excludeFlags = d_flagsCandidates.data();
                bool* h_excludeFlags = h_flagsCandidates.data();

                //corrections of candidates for which a high quality anchor correction exists will not be used
                //-> don't compute them
                for(int i = 0; i < currentNumCandidates; i++){
                    const read_number candidateReadId = currentInput->h_candidate_read_ids[i];
                    h_excludeFlags[i] = correctionFlags->isCorrectedAsHQAnchor(candidateReadId);
                }

                cudaMemcpyAsync(
                    d_excludeFlags,
                    h_excludeFlags,
                    sizeof(bool) * currentNumCandidates,
                    H2D,
                    stream
                );

                callFlagCandidatesToBeCorrectedWithExcludeFlagsKernel(
                    d_candidateCanBeCorrected,
                    d_num_corrected_candidates_per_anchor.get(),
                    multiMSA,
                    d_excludeFlags,
                    d_alignment_shifts.get(),
                    d_candidate_sequences_lengths.get(),
                    d_anchorIndicesOfCandidates.get(),
                    d_is_high_quality_anchor.get(),
                    d_candidates_per_anchor_prefixsum,
                    d_indices.get(),
                    d_indices_per_anchor.get(),
                    d_numAnchors,
                    d_numCandidates,
                    min_support_threshold,
                    min_coverage_threshold,
                    new_columns_to_correct,
                    stream,
                    kernelLaunchHandle
                );
            #else
            callFlagCandidatesToBeCorrectedKernel_async(
                d_candidateCanBeCorrected,
                d_num_corrected_candidates_per_anchor.get(),
                multiMSA,
                d_alignment_shifts.get(),
                d_candidate_sequences_lengths.get(),
                d_anchorIndicesOfCandidates.get(),
                d_is_high_quality_anchor.get(),
                d_candidates_per_anchor_prefixsum,
                d_indices.get(),
                d_indices_per_anchor.get(),
                d_numAnchors,
                d_numCandidates,
                min_support_threshold,
                min_coverage_threshold,
                new_columns_to_correct,
                stream,
                kernelLaunchHandle
            );
            #endif

            size_t cubTempSize = d_tempstorage.sizeInBytes();

            cub::DeviceSelect::Flagged(
                d_tempstorage.get(),
                cubTempSize,
                cub::CountingInputIterator<int>(0),
                d_candidateCanBeCorrected,
                d_indices_of_corrected_candidates.get(),
                d_num_total_corrected_candidates.get(),
                maxCandidates,
                stream
            ); CUERR;

            // cudaMemcpyAsync(
            //     h_num_total_corrected_candidates.get(),
            //     d_num_total_corrected_candidates.get(),
            //     sizeof(int),
            //     D2H,
            //     stream
            // ); CUERR;

            // cudaStreamSynchronize(stream); CUERR; //DEBUG

            int* d_forestOpCandidateIndices = nullptr;
            int* d_forestOpPositionsInCandidates = nullptr;
            int* d_numForestOperationsPerCandidate = d_indices_tmp.get();
            int* d_numForestOperations = d_num_indices_tmp.get();


            callMsaCorrectCandidatesWithForestKernel(
                d_forestOpCandidateIndices,
                d_forestOpPositionsInCandidates,
                d_numForestOperationsPerCandidate,
                d_numForestOperations,
                d_corrected_candidates.get(),
                d_editsPerCorrectedCandidate.get(),
                d_numEditsPerCorrectedCandidate.get(),              
                multiMSA,
                *gpuForestCandidate,
                correctionOptions->thresholdCands,
                correctionOptions->estimatedCoverage,
                d_alignment_shifts.get(),
                d_alignment_best_alignment_flags.get(),
                d_candidate_sequences_data.get(),
                d_candidate_sequences_lengths.get(),
                d_candidateContainsN.get(),
                d_indices_of_corrected_candidates.get(),
                d_num_total_corrected_candidates.get(),
                d_anchorIndicesOfCandidates.get(),
                currentNumCandidates, //*h_num_total_corrected_candidates,
                getDoNotUseEditsValue(),
                maxNumEditsPerSequence,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                editsPitchInBytes,
                gpuReadStorage->getSequenceLengthUpperBound(),
                stream,
                kernelLaunchHandle,
                d_candidate_read_ids.data()
            );  

            helpers::call_fill_kernel_async(d_numEditsPerCorrectedCandidate.data(), currentNumCandidates, 0, stream);

            callConstructSequenceCorrectionResultsKernel(
                d_editsPerCorrectedCandidate.get(),
                d_numEditsPerCorrectedCandidate.get(),
                getDoNotUseEditsValue(),
                d_indices_of_corrected_candidates.get(),
                d_num_total_corrected_candidates.get(),
                d_candidateContainsN.get(),
                d_candidate_sequences_data.get(),
                d_candidate_sequences_lengths.get(),
                d_corrected_candidates.get(),
                currentNumCandidates,
                true,
                maxNumEditsPerSequence,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                editsPitchInBytes,      
                stream,
                kernelLaunchHandle
            );
            
        }

        static constexpr int getDoNotUseEditsValue() noexcept{
            return -1;
        }

    public: //private:

        int deviceId;
        std::array<CudaEvent, 2> events;
        CudaStream backgroundStream;
        CudaEvent previousBatchFinishedEvent;

        std::size_t msaColumnPitchInElements;
        std::size_t encodedSequencePitchInInts;
        std::size_t decodedSequencePitchInBytes;
        std::size_t qualityPitchInBytes;
        std::size_t editsPitchInBytes;

        int maxAnchors;
        int maxCandidates;
        int maxNumEditsPerSequence;
        int currentNumAnchors;
        int currentNumCandidates;

        std::map<int, int> numCandidatesPerReadMap{};

        const ReadCorrectionFlags* correctionFlags;

        const GpuReadStorage* gpuReadStorage;

        const CorrectionOptions* correctionOptions;
        const GoodAlignmentProperties* goodAlignmentProperties;

        GpuErrorCorrectorInput* currentInput;
        GpuErrorCorrectorRawOutput* currentOutput;
        GpuErrorCorrectorRawOutput currentOutputData;

        ThreadPool* threadPool;
        ThreadPool::ParallelForHandle pforHandle;
        KernelLaunchHandle kernelLaunchHandle; 

        const GpuForest* gpuForestAnchor{};
        const GpuForest* gpuForestCandidate{};

        ReadStorageHandle readstorageHandle;

        PinnedBuffer<int> h_num_total_corrected_candidates;
        PinnedBuffer<int> h_num_indices;
        PinnedBuffer<int> h_numSelected;
        PinnedBuffer<int> h_numRemainingCandidatesAfterAlignment;

        PinnedBuffer<read_number> h_indicesForGather;
        DeviceBuffer<read_number> d_indicesForGather;

        DeviceBuffer<char> d_candidate_qualities_compact;

        DeviceBuffer<int> d_candidates_per_anchor_tmp;
        DeviceBuffer<bool> d_anchorContainsN;
        DeviceBuffer<bool> d_candidateContainsN;
        DeviceBuffer<int> d_candidate_sequences_lengths;
        DeviceBuffer<unsigned int> d_candidate_sequences_data;
        DeviceBuffer<unsigned int> d_transposedCandidateSequencesData;
        DeviceBuffer<char> d_anchor_qualities;
        DeviceBuffer<char> d_candidate_qualities;
        DeviceBuffer<int> d_anchorIndicesOfCandidates;
        DeviceBuffer<char> d_tempstorage;
        DeviceBuffer<int> d_alignment_overlaps;
        DeviceBuffer<int> d_alignment_shifts;
        DeviceBuffer<int> d_alignment_nOps;
        DeviceBuffer<bool> d_alignment_isValid;
        DeviceBuffer<BestAlignment_t> d_alignment_best_alignment_flags; 
        DeviceBuffer<int> d_indices;
        DeviceBuffer<int> d_indices_per_anchor;
        DeviceBuffer<int> d_indices_per_anchor_prefixsum;
        DeviceBuffer<int> d_num_indices;
        DeviceBuffer<int> d_indices_tmp;
        DeviceBuffer<int> d_indices_per_anchor_tmp;
        DeviceBuffer<int> d_num_indices_tmp;
        DeviceBuffer<std::uint8_t> d_consensus;
        DeviceBuffer<float> d_support;
        DeviceBuffer<int> d_coverage;
        DeviceBuffer<float> d_origWeights;
        DeviceBuffer<int> d_origCoverages;
        DeviceBuffer<MSAColumnProperties> d_msa_column_properties;
        DeviceBuffer<int> d_counts;
        DeviceBuffer<float> d_weights;
        DeviceBuffer<char> d_corrected_anchors;
        DeviceBuffer<char> d_corrected_candidates;
        DeviceBuffer<char> d_corrected_candidates2;
        DeviceBuffer<int> d_num_corrected_candidates_per_anchor;
        DeviceBuffer<int> d_num_corrected_candidates_per_anchor_prefixsum;
        DeviceBuffer<int> d_num_total_corrected_candidates;
        DeviceBuffer<bool> d_anchor_is_corrected;
        DeviceBuffer<AnchorHighQualityFlag> d_is_high_quality_anchor;
        DeviceBuffer<int> d_high_quality_anchor_indices;
        DeviceBuffer<int> d_num_high_quality_anchor_indices; 
        DeviceBuffer<TempCorrectedSequence::EncodedEdit> d_editsPerCorrectedanchor;
        DeviceBuffer<int> d_numEditsPerCorrectedanchor;
        DeviceBuffer<TempCorrectedSequence::EncodedEdit> d_editsPerCorrectedCandidate;
        DeviceBuffer<TempCorrectedSequence::EncodedEdit> d_editsPerCorrectedCandidate2;
        DeviceBuffer<int> d_numEditsPerCorrectedCandidate;
        DeviceBuffer<int> d_indices_of_corrected_anchors;
        DeviceBuffer<int> d_num_indices_of_corrected_anchors;
        DeviceBuffer<int> d_indices_of_corrected_candidates;
        DeviceBuffer<int> d_totalNumEdits;
        DeviceBuffer<bool> d_isPairedCandidate;
        PinnedBuffer<bool> h_isPairedCandidate;

        DeviceBuffer<int> d_numAnchors;
        DeviceBuffer<int> d_numCandidates;
        DeviceBuffer<read_number> d_anchorReadIds;
        DeviceBuffer<unsigned int> d_anchor_sequences_data;
        DeviceBuffer<int> d_anchor_sequences_lengths;
        DeviceBuffer<read_number> d_candidate_read_ids;
        DeviceBuffer<read_number> d_candidate_read_ids2;
        DeviceBuffer<int> d_candidates_per_anchor;
        DeviceBuffer<int> d_candidates_per_anchor_prefixsum; 
        DeviceBuffer<int> d_candidatesBeginOffsets;

        //host buffers for random forest correction
        PinnedBuffer<char> h_consensus;
        PinnedBuffer<int> h_coverage;
        PinnedBuffer<int> h_counts;
        PinnedBuffer<float> h_weights;
        PinnedBuffer<float> h_support;
        PinnedBuffer<MSAColumnProperties> h_msa_column_properties;
        PinnedBuffer<unsigned int> h_anchor_sequences_data;
        PinnedBuffer<unsigned int> h_candidate_sequences_data;
        PinnedBuffer<int> h_segmentSizes;
        PinnedBuffer<int> h_segmentOffsets;
        PinnedBuffer<BestAlignment_t> h_alignment_best_alignment_flags;
        PinnedBuffer<bool> h_anchorContainsN;
        PinnedBuffer<bool> h_candidateContainsN;
        PinnedBuffer<int> h_candidates_per_anchor_prefixsum; 
        PinnedBuffer<int> h_indices;

        PinnedBuffer<char> h_decoded_candidates;

        DeviceBuffer<char> d_decoded_candidates;

        PinnedBuffer<bool> h_flagsCandidates;
        DeviceBuffer<bool> d_flagsCandidates;
        DeviceBuffer<int> d_alignment_shifts2;
        DeviceBuffer<int> d_alignment_overlaps2;
        DeviceBuffer<int> d_alignment_nOps2;
        DeviceBuffer<int> d_anchorIndicesOfCandidates2;
        DeviceBuffer<bool> d_candidateContainsN2;
        DeviceBuffer<int> d_candidate_sequences_lengths2;
        DeviceBuffer<unsigned int> d_candidate_sequences_data2;
        DeviceBuffer<BestAlignment_t> d_alignment_best_alignment_flags2;
        DeviceBuffer<char> d_cubtemp2;




        
        std::map<GpuErrorCorrectorRawOutput*, CudaGraph> graphMap;
    };



}
}






#endif