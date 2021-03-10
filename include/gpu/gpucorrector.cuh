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



#define GPUANCHORFOREST

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

                    const int* const my_indices_of_corrected_candidates = currentOutput.h_indices_of_corrected_candidates
                                                        + globalOffset;

                    for(int i = 0; i < n_corrected_candidates; ++i) {
                        const int global_candidate_index = my_indices_of_corrected_candidates[i];
                        const read_number candidate_read_id = currentOutput.h_candidate_read_ids[global_candidate_index];

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
                        tmp.edits.resize(numEdits);
                        const TempCorrectedSequence::EncodedEdit* const gpuedits 
                            = (const TempCorrectedSequence::EncodedEdit*)(((const char*)currentOutput.h_editsPerCorrectedanchor.get()) 
                                + positionInVector * currentOutput.editsPitchInBytes);
                        std::copy_n(gpuedits, numEdits, tmp.edits.begin());
                        tmp.useEdits = true;
                    }else{
                        tmp.edits.clear();
                        tmp.useEdits = false;

                        const char* const my_corrected_anchor_data = currentOutput.h_corrected_anchors + anchor_index * currentOutput.decodedSequencePitchInBytes;
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

                //most candidate buffers are stored compact. offsets for each anchor are given by h_num_corrected_candidates_per_anchor_prefixsum
                //Edits and numEdits are stored compact, only for corrected candidates.

                //h_candidate_read_ids, h_candidate_sequences_lengths, h_alignment_shifts are not compact
                

                for(int positionInVector = begin; positionInVector < end; ++positionInVector) {
                    

                    //TIMERSTARTCPU(setup);
                    const int anchor_index = candidateIndicesToProcess[positionInVector].first;
                    const int candidateIndex = candidateIndicesToProcess[positionInVector].second;
                    const read_number anchorReadId = currentOutput.h_anchorReadIds[anchor_index];

                    auto& tmp = correctionOutput.candidateCorrections[positionInVector];

                    const size_t offsetForCorrectedCandidateData = currentOutput.h_num_corrected_candidates_per_anchor_prefixsum[anchor_index];

                    const char* const my_corrected_candidates_data = currentOutput.h_corrected_candidates
                                                    + offsetForCorrectedCandidateData * currentOutput.decodedSequencePitchInBytes;
                    const int* const my_indices_of_corrected_candidates = currentOutput.h_indices_of_corrected_candidates
                                                    + offsetForCorrectedCandidateData;

                    const TempCorrectedSequence::EncodedEdit* const my_editsPerCorrectedCandidate 
                        = (const TempCorrectedSequence::EncodedEdit*)(((const char*)currentOutput.h_editsPerCorrectedCandidate.get()) 
                            + offsetForCorrectedCandidateData * currentOutput.editsPitchInBytes);


                    const int global_candidate_index = my_indices_of_corrected_candidates[candidateIndex];
                    const read_number candidate_read_id = currentOutput.h_candidate_read_ids[global_candidate_index];

                    const int candidate_shift = currentOutput.h_alignment_shifts[global_candidate_index];

                    if(correctionOptions->new_columns_to_correct < candidate_shift){
                        std::cerr << "readid " << anchorReadId << " candidate readid " << candidate_read_id << " : "
                        << candidate_shift << " " << correctionOptions->new_columns_to_correct <<"\n";

                        assert(correctionOptions->new_columns_to_correct >= candidate_shift);
                    }                
                    
                    tmp.type = TempCorrectedSequence::Type::Candidate;
                    tmp.shift = candidate_shift;
                    tmp.readId = candidate_read_id;
                    
                    const int numEdits = currentOutput.h_numEditsPerCorrectedCandidate[offsetForCorrectedCandidateData + candidateIndex];

                    if(numEdits != currentOutput.doNotUseEditsValue){
                        tmp.edits.resize(numEdits);
                        const TempCorrectedSequence::EncodedEdit* gpuedits 
                            = (const TempCorrectedSequence::EncodedEdit*)(((const char*)my_editsPerCorrectedCandidate) 
                                + candidateIndex * currentOutput.editsPitchInBytes);
                        std::copy_n(gpuedits, numEdits, tmp.edits.begin());
                        tmp.useEdits = true;
                    }else{
                        const int candidate_length = currentOutput.h_candidate_sequences_lengths[global_candidate_index];
                        const char* const candidate_data = my_corrected_candidates_data + candidateIndex * currentOutput.decodedSequencePitchInBytes;
                        //tmp.sequence = std::string{candidate_data, candidate_data + candidate_length};
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
            ClfAgent* clfAgent_,
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
            clfAgent{clfAgent_},
            gpuForestAnchor{gpuForestAnchor_},
            gpuForestCandidate{gpuForestCandidate_},
            readstorageHandle{gpuReadStorage->makeHandle()}
        {
            if(correctionOptions->correctionType != CorrectionType::Classic){
                assert(clfAgent != nullptr);
                assert(gpuForestAnchor != nullptr);
            }

            if(correctionOptions->correctionTypeCands != CorrectionType::Classic){
                assert(clfAgent != nullptr);
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

            //gpuMemsetZero(stream);

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

            copyAnchorResultsFromDeviceToHost(stream);

            if(correctionOptions->correctCandidates){
                copyCandidateResultsFromDeviceToHost(stream);
            }


            //fill missing output arrays. This may overlap with gpu execution
            std::copy_n(currentInput->h_anchorReadIds.get(), currentNumAnchors, currentOutput->h_anchorReadIds.get());            
            std::copy_n(currentInput->h_candidate_read_ids.get(), currentNumCandidates, currentOutput->h_candidate_read_ids.get()); //remove if candidates are compacted

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
            d_candidates_per_anchor_prefixsum.resize(maxAnchors + 1);
            d_candidatesBeginOffsets.resize(maxAnchors);
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
            outputBuffersReallocated |= currentOutput->h_indices_of_corrected_candidates.resize(maxCandidates);
            outputBuffersReallocated |= currentOutput->h_alignment_shifts.resize(maxCandidates);
            outputBuffersReallocated |= currentOutput->h_candidate_read_ids.resize(maxCandidates);
            outputBuffersReallocated |= currentOutput->h_anchorReadIds.resize(maxAnchors);
            
            d_anchorIndicesOfCandidates.resize(maxCandidates);
            d_candidateContainsN.resize(maxCandidates);
            d_candidate_read_ids.resize(maxCandidates);
            d_candidate_read_ids2.resize(maxCandidates);
            d_candidate_sequences_lengths.resize(maxCandidates);
            d_candidate_sequences_data.resize(maxCandidates * encodedSequencePitchInInts);
            d_transposedCandidateSequencesData.resize(maxCandidates * encodedSequencePitchInInts);
            
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
            d_editsPerCorrectedCandidate.resize(numEditsCandidates);
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

            #ifndef GPUANCHORFOREST
            if(correctionOptions->correctionType == CorrectionType::Forest
                || correctionOptions->correctionTypeCands == CorrectionType::Forest){

                copyDataForForestCorrectionToHost(stream);
            }
            #endif

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
            //copyAnchorResultsFromDeviceToHostForest(stream);
            if(correctionOptions->correctionType == CorrectionType::Classic){
                copyAnchorResultsFromDeviceToHostClassic(stream);
            }else if(correctionOptions->correctionType == CorrectionType::Forest){
                #ifdef GPUANCHORFOREST
                copyAnchorResultsFromDeviceToHostForestGpu(stream);
                #else
                copyAnchorResultsFromDeviceToHostForest(stream);
                #endif
            }else{
                std::cerr << "copyAnchorResultsFromDeviceToHost not implemented for correctionType\n";
            }
        }

        void copyAnchorResultsFromDeviceToHostClassic(cudaStream_t stream){
            cudaMemcpyAsync(
                currentOutput->h_anchor_sequences_lengths.get(),
                d_anchor_sequences_lengths.get(),
                sizeof(int) * maxAnchors,
                D2H,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                currentOutput->h_corrected_anchors,
                d_corrected_anchors,
                decodedSequencePitchInBytes * maxAnchors,
                D2H,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                currentOutput->h_anchor_is_corrected,
                d_anchor_is_corrected,
                sizeof(bool) * maxAnchors,
                D2H,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                currentOutput->h_is_high_quality_anchor,
                d_is_high_quality_anchor,
                sizeof(AnchorHighQualityFlag) * maxAnchors,
                D2H,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                currentOutput->h_editsPerCorrectedanchor,
                d_editsPerCorrectedanchor,
                editsPitchInBytes * maxAnchors,
                D2H,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                currentOutput->h_numEditsPerCorrectedanchor,
                d_numEditsPerCorrectedanchor,
                sizeof(int) * maxAnchors,
                D2H,
                stream
            ); CUERR;

        }

        void copyAnchorResultsFromDeviceToHostForest(cudaStream_t stream){
            //filled by copying data for forest to host
            //currentOutput->h_anchor_sequences_lengths 

            //filled by forest anchor correction
            //currentOutput->h_corrected_anchors
            //currentOutput->h_anchor_is_corrected
            //currentOutput->h_is_high_quality_anchor
            //currentOutput->h_editsPerCorrectedanchor
            //currentOutput->h_numEditsPerCorrectedanchor
        }

        void copyAnchorResultsFromDeviceToHostForestGpu(cudaStream_t stream){
            //same as default correction
            copyAnchorResultsFromDeviceToHostClassic(stream);
        }


        void copyCandidateResultsFromDeviceToHost(cudaStream_t stream){
            if(correctionOptions->correctionTypeCands == CorrectionType::Classic){
                copyCandidateResultsFromDeviceToHostClassic(stream);
            }
        }

        void copyCandidateResultsFromDeviceToHostClassic(cudaStream_t stream){
            cudaMemcpyAsync(
                currentOutput->h_candidate_sequences_lengths,
                d_candidate_sequences_lengths,
                sizeof(int) * currentNumCandidates,
                D2H,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                h_num_total_corrected_candidates.get(),
                d_num_total_corrected_candidates.get(),
                sizeof(int),
                D2H,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                currentOutput->h_num_corrected_candidates_per_anchor,
                d_num_corrected_candidates_per_anchor,
                sizeof(int) * maxAnchors,
                D2H,
                stream
            ); CUERR;

            size_t cubTempSize = d_tempstorage.sizeInBytes();

            cub::DeviceScan::ExclusiveSum(
                d_tempstorage.get(), 
                cubTempSize, 
                d_num_corrected_candidates_per_anchor.get(), 
                d_num_corrected_candidates_per_anchor_prefixsum.get(), 
                maxAnchors, 
                stream
            );

            cudaMemcpyAsync(
                currentOutput->h_num_corrected_candidates_per_anchor_prefixsum.get(),
                d_num_corrected_candidates_per_anchor_prefixsum.get(),
                sizeof(int) * maxAnchors,
                D2H,
                stream
            ); CUERR;

            gpucorrectorkernels::copyCandidateCorrectionResultsKernel<<<4096, 256, 0, stream>>>(
                currentOutput->h_corrected_candidates.get(),
                currentOutput->h_editsPerCorrectedCandidate.get(),
                currentOutput->h_numEditsPerCorrectedCandidate.get(),
                decodedSequencePitchInBytes,
                editsPitchInBytes,
                d_num_total_corrected_candidates.get(),
                d_corrected_candidates.get(),
                d_editsPerCorrectedCandidate.get(),
                d_numEditsPerCorrectedCandidate.get()
            );


            //copy alignment shifts and indices of corrected candidates from device to host

            gpucorrectorkernels::copyShiftsAndCorrectedCandidateIndices<<<320, 256, 0, stream>>>(
                currentOutput->h_alignment_shifts.get(),
                currentOutput->h_indices_of_corrected_candidates.get(),
                d_numCandidates.get(),
                d_alignment_shifts.get(),
                d_indices_of_corrected_candidates.get()
            ); CUERR;
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

                cudaMemcpyAsync(h_num_indices, d_num_indices, sizeof(int), D2H, stream); CUERR;

                gpuReadStorage->gatherQualities(
                    readstorageHandle,
                    d_anchor_qualities,
                    qualityPitchInBytes,
                    currentInput->h_anchorReadIds,
                    d_anchorReadIds,
                    maxAnchors,
                    stream
                );

                cudaStreamSynchronize(stream); CUERR; //wait for h_indicesForGather and h_num_indices
                const int hNumIndices = h_num_indices[0];

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

            {
                
                gpucorrectorkernels::setAnchorIndicesOfCandidateskernel
                        <<<1024, 128, 0, stream>>>(
                    d_anchorIndicesOfCandidates.get(),
                    d_numAnchors.get(),
                    d_candidates_per_anchor.get(),
                    d_candidates_per_anchor_prefixsum.get()
                ); CUERR;
            }

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

            #if 1

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

            #else

            cudaMemsetAsync(
                d_indices_per_anchor.get(), 0, sizeof(int) * currentNumAnchors, stream
            );

            helpers::lambda_kernel<<<SDIV(currentNumAnchors, (128 / 32)), 128, 0, stream>>>(
                [
                    d_indicesOfGoodCandidates = this->d_indices.get(),
                    d_numIndicesPerAnchor = this->d_indices_per_anchor.get(),
                    d_alignmentFlags = this->d_alignment_best_alignment_flags.get(),
                    d_candidates_per_subject = this->d_candidates_per_anchor.get(),
                    d_candidates_per_subject_prefixsum = this->d_candidates_per_anchor_prefixsum.get(),
                    currentNumAnchors = this->currentNumAnchors
                ] __device__ ()
            {
                constexpr int blocksize = 128;
                constexpr int tilesize = 32;

                static_assert(blocksize % tilesize == 0);
                static_assert(tilesize == 32);

                constexpr int numTilesPerBlock = blocksize / tilesize;

                const int numAnchors = currentNumAnchors;

                const int numTiles = (gridDim.x * blocksize) / tilesize;
                const int tileId = (threadIdx.x + blockIdx.x * blocksize) / tilesize;
                const int tileIdInBlock = threadIdx.x / tilesize;

                __shared__ int totalIndices;
                __shared__ int counts[numTilesPerBlock];

                if(threadIdx.x == 0){
                    totalIndices = 0;
                }
                __syncthreads();

                auto tile = cg::tiled_partition<tilesize>(cg::this_thread_block());

                for(int anchorIndex = tileId; anchorIndex < numAnchors; anchorIndex += numTiles){

                    const int offset = d_candidates_per_subject_prefixsum[anchorIndex];
                    int* const indicesPtr = d_indicesOfGoodCandidates + offset;
                    int* const numIndicesPtr = d_numIndicesPerAnchor + anchorIndex;
                    const BestAlignment_t* const myAlignmentFlagsPtr = d_alignmentFlags + offset;

                    const int numCandidatesForAnchor = d_candidates_per_subject[anchorIndex];

                    if(tile.thread_rank() == 0){
                        counts[tileIdInBlock] = 0;
                    }
                    tile.sync();

                    for(int localCandidateIndex = tile.thread_rank(); 
                            localCandidateIndex < numCandidatesForAnchor; 
                            localCandidateIndex += tile.size()){
                        
                        const BestAlignment_t alignmentflag = myAlignmentFlagsPtr[localCandidateIndex];

                        if(alignmentflag != BestAlignment_t::None){
                            cg::coalesced_group g = cg::coalesced_threads();
                            int outputPos;
                            if (g.thread_rank() == 0) {
                                outputPos = atomicAdd(&counts[tileIdInBlock], g.size());
                                atomicAdd(&totalIndices, g.size());
                            }
                            outputPos = g.thread_rank() + g.shfl(outputPos, 0);
                            //indicesPtr[outputPos] = localCandidateIndex;
                            indicesPtr[outputPos] = outputPos;
                        }
                    }

                    tile.sync();
                    if(tile.thread_rank() == 0){
                        atomicAdd(numIndicesPtr, counts[tileIdInBlock]);
                    }

                }
            }
            );


            std::size_t cubTempSize = d_tempstorage.sizeInBytes();
            cudaError_t cubstatus = cub::DeviceScan::ExclusiveSum(
                d_tempstorage.data(),
                cubTempSize,
                d_indices_per_anchor.data(),
                d_indices_per_anchor_prefixsum.data(),
                currentNumAnchors,
                stream
            );
            assert(cubstatus == cudaSuccess);

            //copy indices into contiguous locations
            helpers::lambda_kernel<<<currentNumAnchors, 128, 0, stream>>>(
                [
                    d_indices_tmp = this->d_indices_tmp.data(),
                    d_indices = this->d_indices.data(),
                    d_indices_per_anchor = this->d_indices_per_anchor.data(),
                    d_indices_per_anchor_prefixsum = this->d_indices_per_anchor_prefixsum.data(),
                    d_candidates_per_anchor_prefixsum = this->d_candidates_per_anchor_prefixsum.data(),
                    currentNumAnchors = this->currentNumAnchors
                ] __device__ ()
            {
                for(int a = blockIdx.x; a < currentNumAnchors; a += gridDim.x){

                    const int outputOffset = d_indices_per_anchor_prefixsum[a];
                    const int inputOffset = d_candidates_per_anchor_prefixsum[a];
                    const int segmentsize = d_indices_per_anchor[a];

                    for(int c = threadIdx.x; c < segmentsize; c += blockDim.x){
                        d_indices_tmp[outputOffset + c] = d_indices[inputOffset + c];
                    }
                }
            }
            );
            std::swap(d_indices_tmp, d_indices);

            //compact
            auto flagsIter = thrust::make_transform_iterator(
                d_alignment_best_alignment_flags.data(), 
                [] __device__ (const auto& flag){ return flag != BestAlignment_t::None;}
            );
            auto srcIter = thrust::make_zip_iterator(thrust::make_tuple(
                d_alignment_overlaps.get(),
                d_alignment_shifts.get(),
                d_alignment_nOps.get(),
                d_alignment_best_alignment_flags.get(),
                d_candidate_sequences_lengths.get(),
                d_anchorIndicesOfCandidates.get(),
                d_candidateContainsN.get(),
                d_candidate_read_ids.get()
            ));
            auto destIter = thrust::make_zip_iterator(thrust::make_tuple(
                d_alignment_overlaps2.get(),
                d_alignment_shifts2.get(),
                d_alignment_nOps2.get(),
                d_alignment_best_alignment_flags2.get(),
                d_candidate_sequences_lengths2.get(),
                d_anchorIndicesOfCandidates2.get(),
                d_candidateContainsN2.get(),
                d_candidate_read_ids2.get()
            ));
            cubstatus = cub::DeviceSelect::Flagged(
                d_tempstorage.data(),
                cubTempSize,
                srcIter, 
                flagsIter, 
                destIter, 
                d_num_indices.data(), 
                currentNumCandidates
            );

            assert(cubstatus == cudaSuccess);

            cudaMemcpyAsync(
                h_numSelected.data(),
                d_num_indices.data(),
                sizeof(int),
                D2H,
                stream
            ); CUERR;

            helpers::lambda_kernel<<<SDIV(maxCandidates, 256), 256, 0, stream>>>(
                [
                    dest = currentOutput->h_candidate_read_ids.get(),
                    src = d_candidate_read_ids2.get(),
                    numPtr = d_num_indices.get()
                ] __device__ (){
                    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                    const int stride = blockDim.x * gridDim.x;
                    const int num = *numPtr;

                    for(int i = tid; i < num; i += stride){
                        dest[i] = src[i];
                    }
                }
            );

            cudaEventRecord(events[0], stream); CUERR;

            auto flagsIter2 = thrust::make_transform_iterator(
                thrust::make_transform_iterator(
                    thrust::make_counting_iterator(0),
                    make_iterator_multiplier(d_alignment_best_alignment_flags.data(), int(encodedSequencePitchInInts))
                ),
                [] __device__ (const auto& flag){ return flag != BestAlignment_t::None;}
            );

            cubstatus = cub::DeviceSelect::Flagged(
                d_tempstorage.data(),
                cubTempSize,
                d_candidate_sequences_data.data(),
                flagsIter2,
                d_candidate_sequences_data2.data(),
                cub::DiscardOutputIterator<int>{},
                currentNumCandidates * encodedSequencePitchInInts,
                stream
            );
            assert(cubstatus == cudaSuccess);

            cudaMemcpyAsync(
                d_numCandidates.data(),
                d_num_indices.data(),
                sizeof(int),
                D2D,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                d_candidates_per_anchor_prefixsum.data() + currentNumAnchors,
                d_num_indices.data(),
                sizeof(int),
                D2D,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                d_candidates_per_anchor_prefixsum.data(),
                d_indices_per_anchor_prefixsum.data(),
                sizeof(int) * currentNumAnchors,
                D2D,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                d_candidates_per_anchor.data(),
                d_indices_per_anchor.data(),
                sizeof(int) * currentNumAnchors,
                D2D,
                stream
            ); CUERR;

            std::swap(d_alignment_overlaps2, d_alignment_overlaps);
            std::swap(d_alignment_shifts2, d_alignment_shifts);
            std::swap(d_alignment_nOps2, d_alignment_nOps);
            std::swap(d_alignment_best_alignment_flags2, d_alignment_best_alignment_flags);
            std::swap(d_candidate_sequences_lengths2, d_candidate_sequences_lengths);
            std::swap(d_anchorIndicesOfCandidates2, d_anchorIndicesOfCandidates);
            std::swap(d_candidateContainsN2, d_candidateContainsN);
            std::swap(d_candidate_sequences_data2, d_candidate_sequences_data);  
            std::swap(d_candidate_read_ids2, d_candidate_read_ids);



            cudaEventSynchronize(events[0]); CUERR; //wait for h_numSelected
            currentNumCandidates = *h_numSelected;

            #endif


            //testing 

            // int numSelected = 0;
            // cudaMemcpyAsync(&numSelected, d_num_indices, sizeof(int), D2H, stream); CUERR;
            // cudaStreamSynchronize(stream); CUERR;
            //std::cerr << numSelected << " / " << currentNumCandidates << "\n";
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

        void copyDataForForestCorrectionToHost(cudaStream_t stream){
            h_consensus.resize(d_consensus.size());
            h_msa_column_properties.resize(d_msa_column_properties.size());
            h_coverage.resize(d_coverage.size());
            h_counts.resize(d_counts.size());
            h_weights.resize(d_weights.size());
            h_support.resize(d_support.size());
            //h_alignment_shifts.resize(d_alignment_shifts.size());
            //h_anchor_sequences_lengths.resize(d_anchor_sequences_lengths.size());
            //h_candidate_sequences_lengths.resize(d_candidate_sequences_lengths.size());
            h_anchor_sequences_data.resize(d_anchor_sequences_data.size());
            h_candidate_sequences_data.resize(d_candidate_sequences_data.size());
            h_alignment_best_alignment_flags.resize(d_alignment_best_alignment_flags.size());
            h_segmentSizes.resize(d_indices_per_anchor.size());
            h_segmentOffsets.resize(d_candidates_per_anchor_prefixsum.size());
            h_candidates_per_anchor_prefixsum.resize(d_candidates_per_anchor_prefixsum.size());

            h_anchorContainsN.resize(d_anchorContainsN.size());
            h_candidateContainsN.resize(d_candidateContainsN.size());

            d_alignment_shifts2.resize(d_alignment_shifts.size());
            d_candidate_sequences_lengths2.resize(d_candidate_sequences_lengths.size());
            d_candidate_sequences_data2.resize(d_candidate_sequences_data.size());
            d_alignment_best_alignment_flags2.resize(d_alignment_best_alignment_flags.size());

            h_numSelected.resize(1);
            h_indices.resize(d_indices.size());

            h_decoded_candidates.resize(maxCandidates * decodedSequencePitchInBytes);
            d_decoded_candidates.resize(maxCandidates * decodedSequencePitchInBytes);
            #if 0
            //decode active candidates
            helpers::lambda_kernel<<<currentNumAnchors, 128, 0, stream>>>(
                [
                    h_decoded_candidates = this->h_decoded_candidates.data(),
                    d_candidate_sequences_data = this->d_candidate_sequences_data.data(),
                    d_alignment_best_alignment_flags = this->d_alignment_best_alignment_flags.data(),
                    d_candidates_per_anchor_prefixsum = this->d_candidates_per_anchor_prefixsum.data(),
                    d_indices = this->d_indices.data(),
                    d_indices_per_anchor = this->d_indices_per_anchor.data(),
                    d_num_indices = this->d_num_indices.data(),
                    d_anchorIndicesOfCandidates = this->d_anchorIndicesOfCandidates.data(),
                    d_candidate_sequences_lengths = this->d_candidate_sequences_lengths.data(),
                    encodedSequencePitchInInts = this->encodedSequencePitchInInts,
                    decodedSequencePitchInBytes = this->decodedSequencePitchInBytes
                ] __device__ (){

                        auto decode = [](const std::uint8_t encoded){
                            char decoded = 'F';
                            if(encoded == std::uint8_t{0}){
                                decoded = 'A';
                            }else if(encoded == std::uint8_t{1}){
                                decoded = 'C';
                            }else if(encoded == std::uint8_t{2}){
                                decoded = 'G';
                            }else if(encoded == std::uint8_t{3}){
                                decoded = 'T';
                            }
                            return decoded;
                        };

                        const int anchorIndex = blockIdx.x;
                        const int globalOffset = d_candidates_per_anchor_prefixsum[anchorIndex];
                        const int numCandidates = d_indices_per_anchor[anchorIndex];

                        for(int c = 0; c < numCandidates; c++){
                            const int localCandidateIndex = d_indices[globalOffset + c];
                            const int globalCandidateIndex = globalOffset + localCandidateIndex;
                            const int length = d_candidate_sequences_lengths[globalCandidateIndex];
                            const BestAlignment_t alignmentFlag = d_alignment_best_alignment_flags[globalCandidateIndex];

                            const unsigned int* encodedPtr = d_candidate_sequences_data
                                + globalCandidateIndex * encodedSequencePitchInInts;

                            char* decodedPtr = h_decoded_candidates + globalCandidateIndex * decodedSequencePitchInBytes;


                            constexpr int basesPerInt = SequenceHelpers::basesPerInt2Bit();
                            const int fullInts = length / basesPerInt;   
                            
                            for(int i = 0; i < fullInts; i++){
                                const unsigned int encodedDataInt = encodedPtr[i];

                                for(int k = threadIdx.x; k < basesPerInt; k += blockDim.x){
                                    const int posInInt = k;
                                    const int posInSequence = i * basesPerInt + posInInt;
                                    const std::uint8_t encodedNuc = SequenceHelpers::getEncodedNucFromInt2Bit(encodedDataInt, posInInt);
                                    const char decodedNuc = decode(encodedNuc);

                                    if(alignmentFlag == BestAlignment_t::Forward){
                                        decodedPtr[posInSequence] = decodedNuc;
                                    }else{
                                        decodedPtr[length - posInSequence - 1] = decodedNuc;
                                    }
                                }
                            }

                            const int remainingPositions = length - basesPerInt * fullInts;

                            if(remainingPositions > 0){
                                const unsigned int encodedDataInt = encodedPtr[fullInts];
                                for(int posInInt = threadIdx.x; posInInt < remainingPositions; posInInt += blockDim.x){
                                    const int posInSequence = fullInts * basesPerInt + posInInt;
                                    const std::uint8_t encodedNuc = SequenceHelpers::getEncodedNucFromInt2Bit(encodedDataInt, posInInt);
                                    const char decodedNuc = decode(encodedNuc);

                                    if(alignmentFlag == BestAlignment_t::Forward){
                                        decodedPtr[posInSequence] = decodedNuc;
                                    }else{
                                        decodedPtr[length - posInSequence - 1] = decodedNuc;
                                    }
                                }
                            }
                        }
                    }
            ); CUERR;
            #endif
            //decode encoded device consensus and store on host. copy column properties to host
            helpers::lambda_kernel<<<currentNumAnchors, 128, 0, stream>>>(
                [
                    msaColumnPitchInElements = this->msaColumnPitchInElements,
                    d_consensus = this->d_consensus.data(),
                    h_consensus = this->h_consensus.data(),
                    h_msa_column_properties = this->h_msa_column_properties.data(),
                    d_msa_column_properties = this->d_msa_column_properties.data(),
                    currentNumAnchors = this->currentNumAnchors
                ] __device__ (){

                    auto decode = [](const std::uint8_t encoded){
                        char decoded = 'F';
                        if(encoded == std::uint8_t{0}){
                            decoded = 'A';
                        }else if(encoded == std::uint8_t{1}){
                            decoded = 'C';
                        }else if(encoded == std::uint8_t{2}){
                            decoded = 'G';
                        }else if(encoded == std::uint8_t{3}){
                            decoded = 'T';
                        }
                        return decoded;
                    };

                    const std::uint8_t* inputConsensus = &d_consensus[blockIdx.x * msaColumnPitchInElements];
                    char* outputConsensus = &h_consensus[blockIdx.x * msaColumnPitchInElements];

                    const int numVecIters = msaColumnPitchInElements / sizeof(char4);

                    for(int i = threadIdx.x; i < numVecIters; i += blockDim.x){
                        char4 data = ((const char4*)inputConsensus)[i];

                        char4 result;
                        result.x = decode(data.x);
                        result.y = decode(data.y);
                        result.z = decode(data.z);
                        result.w = decode(data.w);

                        ((char4*)outputConsensus)[i] = result;
                    }

                    const int remaining = msaColumnPitchInElements - (numVecIters * sizeof(char4));
                    
                    for(int i = threadIdx.x; i < remaining; i += blockDim.x){
                        std::uint8_t encoded = d_consensus[blockIdx.x * msaColumnPitchInElements + numVecIters * sizeof(char4) + i];

                        char decoded = decode(encoded);

                        h_consensus[blockIdx.x * msaColumnPitchInElements + numVecIters * sizeof(char4) + i] = decoded;
                    }

                    //copy column properties
                    for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < currentNumAnchors; i += blockDim.x * gridDim.x){
                        h_msa_column_properties[i] = d_msa_column_properties[i];
                    }
                }
            ); CUERR;


            d_flagsCandidates.resize(currentNumCandidates);

            cudaMemsetAsync(
                d_flagsCandidates.data(),
                0,
                d_flagsCandidates.sizeInBytes(),
                stream
            ); CUERR;

            //compute flags of active candidates
            helpers::lambda_kernel<<<currentNumAnchors, 128, 0, stream>>>(
                [
                    d_flagscandidates = this->d_flagsCandidates.data(),
                    indices = this->d_indices.data(),
                    d_indices_per_anchor = this->d_indices_per_anchor.data(),
                    d_candidates_per_anchor_prefixsum = this->d_candidates_per_anchor_prefixsum.data()
                ] __device__ (){

                    const int num = d_indices_per_anchor[blockIdx.x];
                    const int offset = d_candidates_per_anchor_prefixsum[blockIdx.x];
                    
                    for(int i = threadIdx.x; i < num; i += blockDim.x){
                        const int globalIndex = indices[offset + i] + offset;
                        d_flagscandidates[globalIndex] = true;
                    }
                }
            ); CUERR;

            //set up cub

            auto d_zip = thrust::make_zip_iterator(
                thrust::make_tuple(
                    d_alignment_shifts.data(),
                    d_candidate_sequences_lengths.data(),
                    d_alignment_best_alignment_flags.data()
                )
            );

            auto d_zip_out = thrust::make_zip_iterator(
                thrust::make_tuple(
                    d_alignment_shifts2.data(),
                    d_candidate_sequences_lengths2.data(),
                    d_alignment_best_alignment_flags2.data()
                )
            );

            std::size_t cubBytes = 0;
            std::size_t cubBytes2 = 0;

            cudaError_t cubstatus = cudaSuccess;

            cubstatus = cub::DeviceSelect::Flagged(
                nullptr, 
                cubBytes2, 
                d_zip, 
                d_flagsCandidates.data(), 
                d_zip_out, 
                h_numSelected.data(), 
                currentNumCandidates,
                stream
            );
            assert(cubstatus == cudaSuccess);
            cubBytes = std::max(cubBytes, cubBytes2);

            cubstatus = cub::DeviceSelect::Flagged(
                nullptr,
                cubBytes2,
                d_candidate_sequences_data.data(),
                thrust::make_transform_iterator(
                    thrust::make_counting_iterator(0),
                    SequenceFlagMultiplier{d_flagsCandidates.data(), int(encodedSequencePitchInInts)}
                ),
                d_candidate_sequences_data2.data(),
                cub::DiscardOutputIterator<int>{},
                currentNumCandidates * encodedSequencePitchInInts,
                stream
            );
            assert(cubstatus == cudaSuccess);
            cubBytes = std::max(cubBytes, cubBytes2);

            cubstatus = cub::DeviceScan::ExclusiveSum(
                nullptr,
                cubBytes2,
                d_indices_per_anchor.data(),
                h_segmentOffsets.data(),
                currentNumAnchors,
                stream
            );
            assert(cubstatus == cudaSuccess);
            cubBytes = std::max(cubBytes, cubBytes2);

            d_cubtemp2.resize(cubBytes);

            //run cub

            cubstatus = cub::DeviceScan::ExclusiveSum(
                d_cubtemp2.data(), 
                cubBytes,
                d_indices_per_anchor.data(),
                h_segmentOffsets.data(),
                currentNumAnchors,
                stream
            );
            assert(cubstatus == cudaSuccess);

            #if 1 // 1 full copy, 0 compact by active candidates and copy

                cudaMemcpyAsync(
                    h_candidate_sequences_data.data(),
                    d_candidate_sequences_data.data(),
                    sizeof(unsigned int) * encodedSequencePitchInInts * currentNumCandidates,
                    D2H,
                    stream
                ); CUERR;

                cudaMemcpyAsync(
                    currentOutput->h_alignment_shifts.data(),
                    d_alignment_shifts.data(),
                    sizeof(int) * currentNumCandidates,
                    D2H,
                    stream
                ); CUERR;

                cudaMemcpyAsync(
                    currentOutput->h_candidate_sequences_lengths.data(),
                    d_candidate_sequences_lengths.data(),
                    sizeof(int) * currentNumCandidates,
                    D2H,
                    stream
                ); CUERR;

                cudaMemcpyAsync(
                    h_alignment_best_alignment_flags.data(),
                    d_alignment_best_alignment_flags.data(),
                    sizeof(BestAlignment_t) * currentNumCandidates,
                    D2H,
                    stream
                ); CUERR;

            #else



            cubstatus = cub::DeviceSelect::Flagged(
                d_cubtemp2.data(), 
                cubBytes, 
                d_zip, 
                d_flagsCandidates.data(), 
                d_zip_out, 
                h_numSelected.data(), 
                currentNumCandidates,
                stream
            );
            assert(cubstatus == cudaSuccess);

            cubstatus = cub::DeviceSelect::Flagged(
                d_cubtemp2.data(), 
                cubBytes,
                d_candidate_sequences_data.data(),
                thrust::make_transform_iterator(
                    thrust::make_counting_iterator(0),
                    SequenceFlagMultiplier{d_flagsCandidates.data(), int(encodedSequencePitchInInts)}
                ),
                d_candidate_sequences_data2.data(),
                cub::DiscardOutputIterator<int>{},
                currentNumCandidates * encodedSequencePitchInInts,
                stream
            );
            assert(cubstatus == cudaSuccess);

            cudaStreamSynchronize(stream); //wait for h_numSelected

            cudaMemcpyAsync(
                h_candidate_sequences_data.data(),
                d_candidate_sequences_data2.data(),
                sizeof(unsigned int) * encodedSequencePitchInInts * (*h_numSelected),
                D2H,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                currentOutput->h_alignment_shifts.data(),
                d_alignment_shifts2.data(),
                sizeof(int) * (*h_numSelected),
                D2H,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                currentOutput->h_candidate_sequences_lengths.data(),
                d_candidate_sequences_lengths2.data(),
                sizeof(int) * (*h_numSelected),
                D2H,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                h_alignment_best_alignment_flags.data(),
                d_alignment_best_alignment_flags2.data(),
                sizeof(BestAlignment_t) * (*h_numSelected),
                D2H,
                stream
            ); CUERR;

            #endif

            //copy remaining data 

            cudaMemcpyAsync(
                h_candidates_per_anchor_prefixsum.data(),
                d_candidates_per_anchor_prefixsum.data(),
                d_candidates_per_anchor_prefixsum.sizeInBytes(),
                D2H,
                stream
            ); CUERR;  

            cudaMemcpyAsync(
                h_anchorContainsN.data(),
                d_anchorContainsN.data(),
                d_anchorContainsN.sizeInBytes(),
                D2H,
                stream
            ); CUERR;     

            cudaMemcpyAsync(
                h_candidateContainsN.data(),
                d_candidateContainsN.data(),
                d_candidateContainsN.sizeInBytes(),
                D2H,
                stream
            ); CUERR;       

            cudaMemcpyAsync(
                h_coverage.data(),
                d_coverage.data(),
                d_coverage.sizeInBytes(),
                D2H,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                h_support.data(),
                d_support.data(),
                d_support.sizeInBytes(),
                D2H,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                h_counts.data(),
                d_counts.data(),
                d_counts.sizeInBytes(),
                D2H,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                h_weights.data(),
                d_weights.data(),
                d_weights.sizeInBytes(),
                D2H,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                h_anchor_sequences_data.data(),
                d_anchor_sequences_data.data(),
                d_anchor_sequences_data.sizeInBytes(),
                D2H,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                currentOutput->h_anchor_sequences_lengths.data(),
                d_anchor_sequences_lengths.data(),
                d_anchor_sequences_lengths.sizeInBytes(),
                D2H,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                h_segmentSizes.data(),
                d_indices_per_anchor.data(),
                d_indices_per_anchor.sizeInBytes(),
                D2H,
                stream
            ); CUERR;   

            cudaMemcpyAsync(
                h_indices.data(),
                d_indices.data(),
                d_indices.sizeInBytes(),
                D2H,
                stream
            ); CUERR;      
        };

        void correctAnchors(cudaStream_t stream){
            //correctAnchorsForest(stream);
            if(correctionOptions->correctionType == CorrectionType::Classic){
                correctAnchorsClassic(stream);
            }else if(correctionOptions->correctionType == CorrectionType::Forest){
                #ifdef GPUANCHORFOREST
                correctAnchorsForestGpu(stream);
                #else
                correctAnchorsForest(stream);
                #endif
            }else{
                std::cerr << "correctAnchors not implemented for this type\n";
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

            callConstructAnchorResultsKernelAsync(
                d_editsPerCorrectedanchor.get(),
                d_numEditsPerCorrectedanchor.get(),
                getDoNotUseEditsValue(),
                d_indices_of_corrected_anchors.get(),
                d_num_indices_of_corrected_anchors.get(),
                d_anchorContainsN.get(),
                d_anchor_sequences_data.get(),
                d_anchor_sequences_lengths.get(),
                d_corrected_anchors.get(),
                maxNumEditsPerSequence,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                editsPitchInBytes,
                d_numAnchors.get(),
                maxAnchors,
                stream,
                kernelLaunchHandle
            );
            
        }

        void correctAnchorsForest(cudaStream_t stream){

            cudaStreamSynchronize(stream); CUERR;

            int numCorrectedAnchors = 0; //edit information is only stored for corrected anchors, and stored compact

            for(int a = 0; a < currentNumAnchors; a++){
                const int numCandidates = h_segmentSizes[a];

                if(numCandidates == 0){
                    //without candidates, no correction can be made
                    currentOutput->h_anchor_is_corrected[a] = false;
                    currentOutput->h_is_high_quality_anchor[a].hq(false);
                }else{
                    char* const myCorrection = currentOutput->h_corrected_anchors.data() + decodedSequencePitchInBytes * a;
                    const char* const myConsensus = h_consensus.data() + msaColumnPitchInElements * a;

                    const int anchorLength = currentOutput->h_anchor_sequences_lengths[a];

                    const int subjectColumnsBegin_incl = h_msa_column_properties[a].subjectColumnsBegin_incl;
                    const int subjectColumnsEnd_excl = h_msa_column_properties[a].subjectColumnsEnd_excl;

                    assert(anchorLength == subjectColumnsEnd_excl - subjectColumnsBegin_incl);

                    std::copy(
                        myConsensus + subjectColumnsBegin_incl,
                        myConsensus + subjectColumnsEnd_excl,
                        myCorrection
                    );

                    const int* const myCoverages = h_coverage.data() + msaColumnPitchInElements * a;
                    const float* const mySupport = h_support.data() + msaColumnPitchInElements * a;

                    MSAProperties anchorMsaProperties = getMSAProperties(
                        mySupport,
                        myCoverages,
                        subjectColumnsBegin_incl,
                        subjectColumnsEnd_excl, //exclusive
                        correctionOptions->estimatedErrorrate, 
                        correctionOptions->estimatedCoverage, 
                        correctionOptions->m_coverage
                    );

                    std::vector<char> decodedAnchor(decodedSequencePitchInBytes);
                    const unsigned int* const encodedAnchor = h_anchor_sequences_data.data() + a * encodedSequencePitchInInts;
                    

                    SequenceHelpers::decode2BitSequence(decodedAnchor.data(), encodedAnchor, anchorLength);

                    if(!anchorMsaProperties.isHQ){

                        const int* const myCounts = h_counts.data() + 4 * msaColumnPitchInElements * a;
                        const float* const myWeights = h_weights.data() + 4 * msaColumnPitchInElements * a;

                        ClfAgentDecisionInputData clfInput{};

                        clfInput.decodedAnchor = decodedAnchor.data();
                        clfInput.subjectColumnsBegin_incl = subjectColumnsBegin_incl;
                        clfInput.subjectColumnsEnd_excl = subjectColumnsEnd_excl;
                        clfInput.alignmentShifts = nullptr; //unused for anchor correction
                        clfInput.candidateSequencesLengths = nullptr; //unused for anchor correction
                        clfInput.decodedCandidateSequences = nullptr; //unused for anchor correction
                        clfInput.countsA = myCounts + 0 * msaColumnPitchInElements;
                        clfInput.countsC = myCounts + 1 * msaColumnPitchInElements;
                        clfInput.countsG = myCounts + 2 * msaColumnPitchInElements;
                        clfInput.countsT = myCounts + 3 * msaColumnPitchInElements;
                        clfInput.weightsA = myWeights + 0 * msaColumnPitchInElements;
                        clfInput.weightsC = myWeights + 1 * msaColumnPitchInElements;
                        clfInput.weightsG = myWeights + 2 * msaColumnPitchInElements;
                        clfInput.weightsT = myWeights + 3 * msaColumnPitchInElements;
                        clfInput.coverages = h_coverage.data() + msaColumnPitchInElements * a;
                        clfInput.consensus = myConsensus;
                        clfInput.getMSAProperties = nullptr; // unused for anchor correction
                        clfInput.anchorMsaProperties = anchorMsaProperties;

                        for (int i = 0; i < anchorLength; ++i) {
                            if (decodedAnchor[i] != myConsensus[subjectColumnsBegin_incl+i] 
                                && !clfAgent->decide_anchor(clfInput, i, *correctionOptions))
                            {
                                myCorrection[i] = decodedAnchor[i];
                            }
                        }
                    }else{
                        ;
                    }

                    currentOutput->h_anchor_is_corrected[a] = true;
                    currentOutput->h_is_high_quality_anchor[a].hq(anchorMsaProperties.isHQ);

                    auto* const edits = (TempCorrectedSequence::EncodedEdit*)(((char*)currentOutput->h_editsPerCorrectedanchor.data())
                        + editsPitchInBytes * numCorrectedAnchors);
                    int* const numEditsPtr = currentOutput->h_numEditsPerCorrectedanchor.data() + numCorrectedAnchors;

                    const bool ambiguous = h_anchorContainsN[a];
                    if(!ambiguous){
                        const int maxEdits = anchorLength / 7;
                        int numedits = 0;
                        for(int i = 0; i < anchorLength && numedits <= maxEdits; i++){
                            if(myCorrection[i] != decodedAnchor[i]){
                                edits[numedits] = TempCorrectedSequence::EncodedEdit{i, myCorrection[i]};
                                numedits++;
                            }
                        }
                        if(numedits <= maxEdits){
                            *numEditsPtr = numedits;
                        }else{
                            *numEditsPtr = getDoNotUseEditsValue();
                        }
                    }else{
                        *numEditsPtr = getDoNotUseEditsValue();
                    }

                    numCorrectedAnchors++;
                }
            }
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
                stream
            );

            gpucorrectorkernels::selectIndicesOfFlagsOneBlock<256><<<1,256,0, stream>>>(
                d_indices_of_corrected_anchors.get(),
                d_num_indices_of_corrected_anchors.get(),
                d_anchor_is_corrected.get(),
                d_numAnchors.get()
            ); CUERR;

            callConstructAnchorResultsKernelAsync(
                d_editsPerCorrectedanchor.get(),
                d_numEditsPerCorrectedanchor.get(),
                getDoNotUseEditsValue(),
                d_indices_of_corrected_anchors.get(),
                d_num_indices_of_corrected_anchors.get(),
                d_anchorContainsN.get(),
                d_anchor_sequences_data.get(),
                d_anchor_sequences_lengths.get(),
                d_corrected_anchors.get(),
                maxNumEditsPerSequence,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                editsPitchInBytes,
                d_numAnchors.get(),
                maxAnchors,
                stream,
                kernelLaunchHandle
            );
        }
        

        void correctCandidates(cudaStream_t stream){
            if(correctionOptions->correctionTypeCands == CorrectionType::Classic){
                correctCandidatesClassic(stream);
            }else if(correctionOptions->correctionTypeCands == CorrectionType::Forest){
                correctionCandidatesForest(stream);
            }else{
                std::cerr << "correctCandidates not implemented for this type\n";
            }
        }


        void correctCandidatesClassic(cudaStream_t stream){

            const float min_support_threshold = 1.0f-3.0f*correctionOptions->estimatedErrorrate;
            // coverage is always >= 1
            const float min_coverage_threshold = std::max(1.0f,
                correctionOptions->m_coverage / 6.0f * correctionOptions->estimatedCoverage);
            const int new_columns_to_correct = correctionOptions->new_columns_to_correct;

            bool* const d_candidateCanBeCorrected = d_alignment_isValid.get(); //repurpose

            if(correctionOptions->correctionType == CorrectionType::Forest){
                //transfer flags of cpu forest anchor correction to gpu
                cudaMemcpyAsync(
                    d_is_high_quality_anchor.data(),
                    currentOutput->h_is_high_quality_anchor.data(),
                    sizeof(AnchorHighQualityFlag) * currentNumAnchors,
                    H2D,
                    stream
                );
            }

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
            
#if 0            
            //debug: sanity check kernel
            helpers::lambda_kernel<<<1,1, 0, stream>>>([
                encodedSequencePitchInInts = encodedSequencePitchInInts,
                decodedSequencePitchInBytes = decodedSequencePitchInBytes,
                d_num_total_corrected_candidates = d_num_total_corrected_candidates.get(),
                noEdits = getDoNotUseEditsValue(),
                maxNumEditsPerSequence,
                d_numEditsPerCorrectedCandidate = d_numEditsPerCorrectedCandidate.get(),
                d_indices_of_corrected_candidates = d_indices_of_corrected_candidates.get(),
                d_candidate_sequences_lengths = d_candidate_sequences_lengths.get(),
                d_candidate_sequences_data = d_candidate_sequences_data.get(),
                d_corrected_candidates = d_corrected_candidates.data(),
                d_alignment_best_alignment_flags = d_alignment_best_alignment_flags.data()
            ] __device__ (){
                const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                const int stride = blockDim.x * gridDim.x;

                const int totalNumCorrected = d_num_total_corrected_candidates[0];

                for(int i = tid; i < totalNumCorrected; i += stride){
                    const int numEdits = d_numEditsPerCorrectedCandidate[i];
                    if(numEdits != noEdits && numEdits > maxNumEditsPerSequence){
                        printf("corrected candidate %d, numEdits = %d\n", i, numEdits);
                        printf("read numEdits of corrected %d from address %p\n", i, (d_numEditsPerCorrectedCandidate + i));

                        const int candidateIndex = d_indices_of_corrected_candidates[i];

                        printf("candidateIndex %d\n", candidateIndex);

                        const int len = d_candidate_sequences_lengths[candidateIndex];
                        const BestAlignment_t bestAlignmentFlag = d_alignment_best_alignment_flags[candidateIndex];

                        printf("len %d, bestAlignmentFlag %d\n", len, int(bestAlignmentFlag));
                        printf("encodedSequencePitchInInts %lu\n", encodedSequencePitchInInts);



                        printf("%lu\n", sizeof(d_candidate_sequences_data[0]));

                        for(int k = 0; k < len; k++){
                            const char corChar =  d_corrected_candidates[decodedSequencePitchInBytes * i + k];
                            const std::uint8_t a = SequenceHelpers::getEncodedNuc2Bit((const unsigned int*)(d_candidate_sequences_data) + encodedSequencePitchInInts * candidateIndex, len, k);
                            const std::uint8_t b = SequenceHelpers::getEncodedNuc2Bit((const unsigned int*)(d_candidate_sequences_data) + encodedSequencePitchInInts * candidateIndex, len, len - 1 - k);
                            const char uncorFwChar = SequenceHelpers::decodeBase(a);
                            const char uncorRcChar = SequenceHelpers::decodeBase(b);

                            printf("%d %c %c %c, %d %d %d %d\n", k, corChar, uncorFwChar, uncorRcChar, int(corChar), int(uncorFwChar), int(uncorRcChar), (uncorFwChar == corChar) ? 0 : 1);
                        }

                        assert(false);
                    }
                    
                }
            });

            cudaStreamSynchronize(stream); CUERR; //DEBUG
#endif
        }

        void correctionCandidatesForest(cudaStream_t stream){

            nvtx::push_range("correctionCandidatesForest", 4);
            cudaStreamSynchronize(stream); CUERR;

            int numCorrectedCandidates = 0; //edit information is only stored for corrected anchors, and stored compact
            currentOutput->h_num_corrected_candidates_per_anchor_prefixsum[0] = 0;

            for(int a = 0; a < currentNumAnchors; a++){
                const int numCandidates = h_segmentSizes[a];
                int numCorrectedCandidatesForAnchor = 0;

                if(numCandidates > 0){
                    const int globalOffset = h_candidates_per_anchor_prefixsum[a];
                    const char* const myConsensus = h_consensus.data() + msaColumnPitchInElements * a;

                    const int subjectColumnsBegin_incl = h_msa_column_properties[a].subjectColumnsBegin_incl;
                    const int subjectColumnsEnd_excl = h_msa_column_properties[a].subjectColumnsEnd_excl;

                    const int* const myCoverages = h_coverage.data() + msaColumnPitchInElements * a;
                    const float* const mySupport = h_support.data() + msaColumnPitchInElements * a;
                    
                    MSAProperties anchorMsaProperties = getMSAProperties(
                        mySupport,
                        myCoverages,
                        subjectColumnsBegin_incl,
                        subjectColumnsEnd_excl, //exclusive
                        correctionOptions->estimatedErrorrate, 
                        correctionOptions->estimatedCoverage, 
                        correctionOptions->m_coverage
                    );

                    // auto it = std::find(&currentInput->h_candidate_read_ids[globalOffset], &currentInput->h_candidate_read_ids[globalOffset + numCandidates], 37);
                    // if(it != &currentInput->h_candidate_read_ids[globalOffset + numCandidates]){
                    //     std::cerr << "found 37 at local position " << std::distance(&currentInput->h_candidate_read_ids[globalOffset], it) << "\n";
                    //     std::cerr << "anchorReadId " << currentInput->h_anchorReadIds[a] << "\n";
                    // }

                    if(anchorMsaProperties.isHQ){

                        const int* const myShifts = currentOutput->h_alignment_shifts.data() + globalOffset;

                        for(int c = 0; c < numCandidates; c++){
                            const int candidateIndex = h_indices[globalOffset + c];
                            const int cand_begin = subjectColumnsBegin_incl + myShifts[candidateIndex];
                            const int cand_length = currentOutput->h_candidate_sequences_lengths[globalOffset + candidateIndex];
                            const int cand_end = cand_begin + cand_length;
                            const read_number candidateReadId = currentInput->h_candidate_read_ids[globalOffset + candidateIndex];

                            if(correctionFlags->isCorrectedAsHQAnchor(candidateReadId)){
                                continue;
                            }

                            // if(candidateReadId == 37){
                            //     std::cerr <<  "candidateIndex " << candidateIndex << "\n";
                            //     std::cerr <<  "cand_begin " << cand_begin << "\n";
                            //     std::cerr <<  "cand_end " << cand_end << "\n";
                            //     std::cerr <<  "cand_length " << cand_length << "\n";
                            //     std::cerr <<  "candidateReadId " << candidateReadId << "\n";
                            //     std::cerr << "anchorReadId " << currentInput->h_anchorReadIds[a] << "\n";
                            //     std::cerr << "shift " << myShifts[candidateIndex] << "\n";
                            // }

                            if(cand_begin >= subjectColumnsBegin_incl - correctionOptions->new_columns_to_correct
                                && cand_end <= subjectColumnsEnd_excl + correctionOptions->new_columns_to_correct){

                                currentOutput->h_indices_of_corrected_candidates[numCorrectedCandidates] = globalOffset + candidateIndex;

                                char* const myCorrection = currentOutput->h_corrected_candidates.data() 
                                    + decodedSequencePitchInBytes * numCorrectedCandidates;
                                std::copy(myConsensus + cand_begin, myConsensus + cand_end, myCorrection);

                                //decode encoded candidate with proper orientation
                                #if 1
                                const unsigned int* const encodedCandidate = h_candidate_sequences_data.data() 
                                    + (globalOffset + candidateIndex) * encodedSequencePitchInInts;
                                

                                std::vector<char> decodedCandidate(decodedSequencePitchInBytes);
                                SequenceHelpers::decode2BitSequence(
                                    decodedCandidate.data(), 
                                    encodedCandidate, 
                                    cand_length
                                );

                                const BestAlignment_t alignmentDirection = h_alignment_best_alignment_flags[globalOffset + candidateIndex];

                                if(alignmentDirection == BestAlignment_t::ReverseComplement){
                                    SequenceHelpers::reverseComplementSequenceDecodedInplace(decodedCandidate.data(), cand_length);
                                }else{
                                    assert(alignmentDirection == BestAlignment_t::Forward);
                                }

                                const char* const decodedCandidatePtr = decodedCandidate.data();

                                #else 

                                const char* const decodedCandidatePtr = h_decoded_candidates.data()
                                    + (globalOffset + candidateIndex) * decodedSequencePitchInBytes;

                                #endif

                                // if(candidateReadId == 37){
                                //     std::cerr <<  "candidateIndex " << candidateIndex << "\n";
                                //     std::cerr <<  "cand_begin " << cand_begin << "\n";
                                //     std::cerr <<  "cand_end " << cand_end << "\n";
                                //     std::cerr <<  "cand_length " << cand_length << "\n";
                                //     std::cerr <<  "candidateReadId " << candidateReadId << "\n";
                                //     std::cerr <<  "alignmentDirection " << int(alignmentDirection) << "\n";

                                //     std::cerr << "decodedCandidate:\n";
                                //     for(int k = 0; k < cand_length; k++){
                                //         std::cerr << decodedCandidatePtr[k];
                                //     }
                                //     std::cerr << "\n";

                                //     std::cerr << "consensusCandidate:\n";
                                //     for(int k = 0; k < cand_length; k++){
                                //         std::cerr << myCorrection[k];
                                //     }
                                //     std::cerr << "\n";

                                // }

                                const int* const myCounts = h_counts.data() + 4 * msaColumnPitchInElements * a;
                                const float* const myWeights = h_weights.data() + 4 * msaColumnPitchInElements * a;

                                ClfAgentDecisionInputData clfInput{};

                                clfInput.decodedAnchor = nullptr; //unused for candidate correction
                                clfInput.subjectColumnsBegin_incl = subjectColumnsBegin_incl;
                                clfInput.subjectColumnsEnd_excl = subjectColumnsEnd_excl;
                                clfInput.alignmentShifts = &myShifts[candidateIndex]; //treat as if only 1 candidate exists
                                clfInput.candidateSequencesLengths = &cand_length; //treat as if only 1 candidate exists
                                clfInput.decodedCandidateSequences = decodedCandidatePtr; //treat as if only 1 candidate exists
                                clfInput.countsA = myCounts + 0 * msaColumnPitchInElements;
                                clfInput.countsC = myCounts + 1 * msaColumnPitchInElements;
                                clfInput.countsG = myCounts + 2 * msaColumnPitchInElements;
                                clfInput.countsT = myCounts + 3 * msaColumnPitchInElements;
                                clfInput.weightsA = myWeights + 0 * msaColumnPitchInElements;
                                clfInput.weightsC = myWeights + 1 * msaColumnPitchInElements;
                                clfInput.weightsG = myWeights + 2 * msaColumnPitchInElements;
                                clfInput.weightsT = myWeights + 3 * msaColumnPitchInElements;
                                clfInput.coverages = myCoverages;
                                clfInput.consensus = myConsensus;
                                clfInput.anchorMsaProperties = anchorMsaProperties;

                                MSAProperties candidateMsaProperties = getMSAProperties(
                                    mySupport,
                                    myCoverages,
                                    cand_begin,
                                    cand_end, //exclusive
                                    correctionOptions->estimatedErrorrate, 
                                    correctionOptions->estimatedCoverage, 
                                    correctionOptions->m_coverage
                                );
                                
                                clfInput.getMSAProperties = [&](
                                        int begin, int end, float est_err, float est_cov, float m_cov
                                ){
                                    return candidateMsaProperties; //cached for multiple calls for same candidate
                                };


                                for(int i = 0; i < cand_length; i++){
                                    if(decodedCandidatePtr[i] != myConsensus[cand_begin + i]){
                                        //if(a == 5 && candidateReadId == 633) std::cerr << "checking position " << i << ". ";
                                        if(!clfAgent->decide_cand(clfInput, i, *correctionOptions, 0, 0)){
                                            myCorrection[i] = decodedCandidatePtr[i];
                                            //if(a == 5 && candidateReadId == 633) std::cerr << "revert consensus\n";
                                        }else{
                                            //if(a == 5 && candidateReadId == 633) std::cerr << "keep consensus\n";
                                        }
                                    }
                                }

                                auto* const edits = (TempCorrectedSequence::EncodedEdit*)(((char*)currentOutput->h_editsPerCorrectedCandidate.data())
                                    + editsPitchInBytes * numCorrectedCandidates);
                                int* const numEditsPtr = currentOutput->h_numEditsPerCorrectedCandidate.data() + numCorrectedCandidates;

                                const bool ambiguous = h_anchorContainsN[a];
                                if(!ambiguous){
                                    const int maxEdits = cand_length / 7;
                                    int numedits = 0;
                                    for(int i = 0; i < cand_length && numedits <= maxEdits; i++){
                                        if(myCorrection[i] != decodedCandidatePtr[i]){
                                            //edits are applied to the forward sequence
                                            if(alignmentDirection == BestAlignment_t::ReverseComplement){
                                                edits[numedits] = TempCorrectedSequence::EncodedEdit{cand_length - i - 1, SequenceHelpers::reverseComplementBaseDecoded(myCorrection[i])};
                                            }else{
                                                edits[numedits] = TempCorrectedSequence::EncodedEdit{i, myCorrection[i]};
                                            }
                                            numedits++;
                                        }
                                    }
                                    if(numedits <= maxEdits){
                                        std::sort(edits, edits + numedits, [](auto l, auto r){return l.pos() < r.pos();});
                                        *numEditsPtr = numedits;
                                    }else{
                                        *numEditsPtr = getDoNotUseEditsValue();
                                    }
                                }else{
                                    *numEditsPtr = getDoNotUseEditsValue();
                                }

                                numCorrectedCandidates++;
                                numCorrectedCandidatesForAnchor++;
                            }else{
                                // if(candidateReadId == 37){
                                //     std::cerr << "not in range with shift " << myShifts[candidateIndex] << "\n";
                                // }
                            }
                        }
                    }else{
                        // if(it != &currentInput->h_candidate_read_ids[globalOffset + numCandidates]){
                        //     std::cerr << "not hq anchor for candidate id 37\n";
                        // }
                    }
                }

                currentOutput->h_num_corrected_candidates_per_anchor[a] = numCorrectedCandidatesForAnchor;
            
                if(a < currentNumAnchors - 1){
                    currentOutput->h_num_corrected_candidates_per_anchor_prefixsum[a+1]
                        = currentOutput->h_num_corrected_candidates_per_anchor_prefixsum[a]
                            + numCorrectedCandidatesForAnchor;
                }
            }
        
            nvtx::pop_range();
        }

        

        static constexpr int getDoNotUseEditsValue() noexcept{
            return -1;
        }

    public: //private:

        int deviceId;
        std::array<CudaEvent, 1> events;
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

        ClfAgent* clfAgent{};
        const GpuForest* gpuForestAnchor{};
        const GpuForest* gpuForestCandidate{};

        ReadStorageHandle readstorageHandle;

        PinnedBuffer<int> h_num_total_corrected_candidates;
        PinnedBuffer<int> h_num_indices;
        PinnedBuffer<int> h_numSelected;

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
        DeviceBuffer<int> d_numEditsPerCorrectedCandidate;
        DeviceBuffer<int> d_indices_of_corrected_anchors;
        DeviceBuffer<int> d_num_indices_of_corrected_anchors;
        DeviceBuffer<int> d_indices_of_corrected_candidates;

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