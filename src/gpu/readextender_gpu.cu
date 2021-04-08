#include <gpu/readextender_gpu.hpp>
#include <readextenderbase.hpp>

#include <vector>
#include <algorithm>
#include <sequencehelpers.hpp>
#include <string>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <gpu/segmented_set_operations.cuh>
#include <gpu/cachingallocator.cuh>
#include <hostdevicefunctions.cuh>

//#define SINGLEBLOCKPREFIXSUM

namespace care{

    template<int blocksize, int itemsPerThread, bool inclusive, class T>
    __global__
    void prefixSumSingleBlockKernel(
        const T* input,
        T* output,
        int N
    ){
        struct BlockPrefixCallbackOp{
            // Running prefix
            int running_total;

            __device__
            BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
            // Callback operator to be entered by the first warp of threads in the block.
            // Thread-0 is responsible for returning a value for seeding the block-wide scan.
            __device__
            int operator()(int block_aggregate){
                int old_prefix = running_total;
                running_total += block_aggregate;
                return old_prefix;
            }
        };

        assert(blocksize == blockDim.x);

        using BlockScan = cub::BlockScan<T, blocksize>;
        using BlockLoad = cub::BlockLoad<T, blocksize, itemsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
        using BlockStore = cub::BlockStore<T, blocksize, itemsPerThread, cub::BLOCK_STORE_WARP_TRANSPOSE>;

        __shared__ typename BlockScan::TempStorage blockscantemp;
        __shared__ union{
            typename BlockLoad::TempStorage load;
            typename BlockStore::TempStorage store;
        } temp;

        T items[itemsPerThread];

        BlockPrefixCallbackOp prefix_op(0);

        const int iterations = SDIV(N, blocksize);

        int remaining = N;

        const T* currentInput = input;
        T* currentOutput = output;

        for(int iteration = 0; iteration < iterations; iteration++){
            const int valid_items = min(itemsPerThread * blocksize, remaining);

            BlockLoad(temp.load).Load(currentInput, items, valid_items, 0);

            if(inclusive){
                BlockScan(blockscantemp).InclusiveSum(
                    items, items, prefix_op
                );
            }else{
                BlockScan(blockscantemp).ExclusiveSum(
                    items, items, prefix_op
                );
            }
            __syncthreads();

            BlockStore(temp.store).Store(currentOutput, items, valid_items);
            __syncthreads();

            remaining -= valid_items;
            currentInput += valid_items;
            currentOutput += valid_items;
        }
    }

namespace readextendergpukernels{
    

    //flag candidates to remove because they are equal to anchor id or equal to mate id
    __global__
    void flagCandidateIdsWhichAreEqualToAnchorOrMateKernel(
        const read_number* __restrict__ candidateReadIds,
        const read_number* __restrict__ anchorReadIds,
        const read_number* __restrict__ mateReadIds,
        const int* __restrict__ numCandidatesPerAnchorPrefixSum,
        const int* __restrict__ numCandidatesPerAnchor,
        bool* __restrict__ keepflags, // size numCandidates
        bool* __restrict__ mateRemovedFlags, //size numTasks
        int* __restrict__ numCandidatesPerAnchorOut,
        int numTasks,
        bool isPairedEnd
    ){

        using BlockReduceInt = cub::BlockReduce<int, 128>;

        __shared__ typename BlockReduceInt::TempStorage intreduce1;
        __shared__ typename BlockReduceInt::TempStorage intreduce2;

        for(int a = blockIdx.x; a < numTasks; a += gridDim.x){
            const int size = numCandidatesPerAnchor[a];
            const int offset = numCandidatesPerAnchorPrefixSum[a];
            const read_number anchorId = anchorReadIds[a];
            read_number mateId = 0;
            if(isPairedEnd){
                mateId = mateReadIds[a];
            }

            int mateIsRemoved = 0;
            int numRemoved = 0;

            // if(threadIdx.x == 0){
            //     printf("looking for anchor %u, mate %u\n", anchorId, mateId);
            // }
            __syncthreads();

            for(int i = threadIdx.x; i < size; i+= blockDim.x){
                bool keep = true;

                const read_number candidateId = candidateReadIds[offset + i];
                //printf("tid %d, comp %u at position %d\n", threadIdx.x, candidateId, offset + i);

                if(candidateId == anchorId){
                    keep = false;
                    numRemoved++;
                }

                if(isPairedEnd && candidateId == mateId){
                    if(keep){
                        keep = false;
                        numRemoved++;
                    }
                    mateIsRemoved++;
                    //printf("mate removed. i = %d\n", i);
                }

                keepflags[offset + i] = keep;
            }
            //printf("tid = %d, mateIsRemoved = %d\n", threadIdx.x, mateIsRemoved);
            int numRemovedBlock = BlockReduceInt(intreduce1).Sum(numRemoved);
            int numMateRemovedBlock = BlockReduceInt(intreduce2).Sum(mateIsRemoved);
            if(threadIdx.x == 0){
                numCandidatesPerAnchorOut[a] = size - numRemovedBlock;
                //printf("numMateRemovedBlock %d\n", numMateRemovedBlock);
                if(numMateRemovedBlock > 0){
                    mateRemovedFlags[a] = true;
                }else{
                    mateRemovedFlags[a] = false;
                }
            }
        }
    }

}

    //output[map[i]] = input[i];
    template<class T, class U>
    __global__ 
    void setFirstSegmentIdsKernel(
        const T* __restrict__ segmentSizes,
        int* __restrict__ segmentIds,
        const U* __restrict__ segmentOffsets,
        int N
    ){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        for(int i = tid; i < N; i += stride){
            if(segmentSizes[i] > 0){
                segmentIds[segmentOffsets[i]] = i;
            }
        }
    }

    __global__
    void setSegmentIndicesKernel(
        int* __restrict__ d_indices,
        int N,
        const int* __restrict__ d_segment_sizes,
        const int* __restrict__ d_segment_sizes_prefixsum
    ){
        for(int segmentIndex = blockIdx.x; segmentIndex < N; segmentIndex += gridDim.x){
            const int offset = d_segment_sizes_prefixsum[segmentIndex];
            const int size = d_segment_sizes[segmentIndex];
            int* const beginptr = &d_indices[offset];

            for(int localindex = threadIdx.x; localindex < size; localindex += blockDim.x){
                beginptr[localindex] = segmentIndex;
            }
        }
    }







#if 1
std::vector<ExtendResult> ReadExtenderGpu::processPairedEndTasks(
    std::vector<ReadExtenderBase::Task> tasks_
) {

    if(tasks_.empty()) return {};

    batchData.init(
        std::move(tasks_), 
        encodedSequencePitchInInts, 
        decodedSequencePitchInBytes, 
        msaColumnPitchInElements
    );

    do{
        gpuExtensionStepper.prepareStep(batchData);

        gpuReadHasher.getCandidateReadIds(batchData, batchData.streams[0]);

        gpuExtensionStepper.step(batchData);

        gpuExtensionStepper.extendAfterStep(batchData);
    } while (!batchData.isEmpty());

    //construct results

    std::vector<ExtendResult> extendResults;

    for(const auto& task : batchData.tasks){

        ExtendResult extendResult;
        extendResult.direction = task.direction;
        extendResult.numIterations = task.iteration;
        extendResult.aborted = task.abort;
        extendResult.abortReason = task.abortReason;
        extendResult.readId1 = task.myReadId;
        extendResult.readId2 = task.mateReadId;
        extendResult.originalLength = task.myLength;

#if 0
        //extendResult.extendedRead = std::move(task.resultsequence);
        extendResult.success = true;
        extendResult.mateHasBeenFound = task.mateHasBeenFound;

#else
        // if(abort){
        //     ; //no read extension possible
        // }else
        {
            //if(mateHasBeenFound){
            {
                //construct extended read
                //build msa of all saved totalDecodedAnchors[0]

                const int numsteps = task.totalDecodedAnchors.size();

                // if(task.myReadId == 90 || task.mateReadId == 90){
                //     std::cerr << "task.totalDecodedAnchors\n";
                // }

                int maxlen = 0;
                for(const auto& s: task.totalDecodedAnchors){
                    const int len = s.length();
                    if(len > maxlen){
                        maxlen = len;
                    }

                    // if(task.myReadId == 90 || task.mateReadId == 90){
                    //     std::cerr << s << "\n";
                    // }
                }

                // if(task.myReadId == 90 || task.mateReadId == 90){
                //     std::cerr << "\n";
                // }

                const std::string& decodedAnchor = task.totalDecodedAnchors[0];

                const std::vector<int> shifts(task.totalAnchorBeginInExtendedRead.begin() + 1, task.totalAnchorBeginInExtendedRead.end());
                std::vector<float> initialWeights(numsteps-1, 1.0f);


                std::vector<char> stepstrings(maxlen * (numsteps-1), '\0');
                std::vector<int> stepstringlengths(numsteps-1);
                for(int c = 1; c < numsteps; c++){
                    std::copy(
                        task.totalDecodedAnchors[c].begin(),
                        task.totalDecodedAnchors[c].end(),
                        stepstrings.begin() + (c-1) * maxlen
                    );
                    stepstringlengths[c-1] = task.totalDecodedAnchors[c].size();
                }

                MultipleSequenceAlignment::InputData msaInput;
                msaInput.useQualityScores = false;
                msaInput.subjectLength = decodedAnchor.length();
                msaInput.nCandidates = numsteps-1;
                msaInput.candidatesPitch = maxlen;
                msaInput.candidateQualitiesPitch = 0;
                msaInput.subject = decodedAnchor.c_str();
                msaInput.candidates = stepstrings.data();
                msaInput.subjectQualities = nullptr;
                msaInput.candidateQualities = nullptr;
                msaInput.candidateLengths = stepstringlengths.data();
                msaInput.candidateShifts = shifts.data();
                msaInput.candidateDefaultWeightFactors = initialWeights.data();

                MultipleSequenceAlignment msa;

                msa.build(msaInput);

                // if(task.myReadId == 90 || task.mateReadId == 90){
                //     std::cerr << "Id " << task.myReadId << ", Final\n";
                //     msa.print(std::cerr);
                //     std::cerr << "\n";
                // }

                extendResult.success = true;

                std::string extendedRead(msa.consensus.begin(), msa.consensus.end());
                //std::cerr << "before: " << extendedRead << "\n";
                std::copy(decodedAnchor.begin(), decodedAnchor.end(), extendedRead.begin());
                if(task.mateHasBeenFound){
                    std::copy(
                        task.decodedMateRevC.begin(),
                        task.decodedMateRevC.end(),
                        extendedRead.begin() + extendedRead.length() - task.decodedMateRevC.length()
                    );
                }
                // extendedRead.replace(extendedRead.begin(), extendedRead.begin() + decodedAnchor, decodedAnchor.begin(), decodedAnchor.end());
                // std::cerr << "after : " << extendedRead << "\n";
                
                // msa.print(std::cerr);
                // std::cerr << "msa cons:\n";
                // std::cerr << extendedRead << "\n";
                // std::cerr << "new cons:\n";
                // std::cerr << task.resultsequence << "\n";


                extendResult.extendedRead = std::move(extendedRead);

                extendResult.mateHasBeenFound = task.mateHasBeenFound;
            }
            // else{
            //     ; //no read extension possible
            // }
        }

        // if(extendResult.extendedRead.length() != task.resultsequence.length()){
        //     std::cerr << task.myReadId << "\n";
        //     std::cerr << extendResult.extendedRead << "\n";
        //     std::cerr << task.resultsequence << "\n";
        //     std::exit(0);
        // }
#endif
        extendResults.emplace_back(std::move(extendResult));

    }

    return extendResults;
}

#else 



    std::vector<ReadExtenderBase::ExtendResult> ReadExtenderGpu::processPairedEndTasks(
        std::vector<ReadExtenderBase::Task> tasks_
    ) {


        batchData.init(
            std::move(tasks_), 
            encodedSequencePitchInInts, 
            decodedSequencePitchInBytes, 
            msaColumnPitchInElements
        );

#if 1
        //undo: replace vecAccess\(([a-zA-z]+), ([a-zA-z]+)\) by $1[$2]
        auto vecAccess = [](auto& vec, auto index) -> decltype(vec[index]){
            return vec[index];
        };
#else 
        auto vecAccess = [](auto& vec, auto index) -> decltype(vec.at(index)){
            return vec.at(index);
        };
#endif 

        batchData.h_numAnchors.resize(1);
        batchData.h_numCandidates.resize(1);
        batchData.d_numAnchors.resize(1);
        batchData.d_numCandidates.resize(1);


        while(batchData.indicesOfActiveTasks.size() > 0){

            //perform one extension iteration for active tasks

            //setup batchdata for active tasks
            const int numActiveTasks = batchData.indicesOfActiveTasks.size();
            batchData.numTasks = numActiveTasks;

            batchData.h_numAnchors.resize(1);
            batchData.d_numAnchors.resize(1);
            batchData.h_numCandidates.resize(1);
            batchData.d_numCandidates.resize(1);
            batchData.d_numCandidates2.resize(1);
            batchData.h_numAnchorsWithRemovedMates.resize(1);

            batchData.h_anchorReadIds.resize(numActiveTasks);
            batchData.d_anchorReadIds.resize(numActiveTasks);
            batchData.h_mateReadIds.resize(numActiveTasks);
            batchData.d_mateReadIds.resize(numActiveTasks);
            
            batchData.h_subjectSequencesData.resize(encodedSequencePitchInInts * numActiveTasks);
            batchData.d_subjectSequencesData.resize(encodedSequencePitchInInts * numActiveTasks);
            batchData.h_anchorSequencesLength.resize(numActiveTasks);
            batchData.d_anchorSequencesLength.resize(numActiveTasks);

            batchData.h_anchormatedata.resize(numActiveTasks * encodedSequencePitchInInts);
            batchData.d_anchormatedata.resize(numActiveTasks * encodedSequencePitchInInts);

            batchData.h_inputanchormatedata.resize(numActiveTasks * encodedSequencePitchInInts);
            batchData.d_inputanchormatedata.resize(numActiveTasks * encodedSequencePitchInInts);

            batchData.h_numCandidatesPerAnchor.resize(numActiveTasks);
            batchData.d_numCandidatesPerAnchor.resize(numActiveTasks);
            batchData.h_numCandidatesPerAnchor2.resize(numActiveTasks);
            batchData.d_numCandidatesPerAnchor2.resize(numActiveTasks);
            batchData.h_numCandidatesPerAnchorPrefixSum.resize(numActiveTasks+1);
            batchData.d_numCandidatesPerAnchorPrefixSum.resize(numActiveTasks+1);
            batchData.d_numCandidatesPerAnchorPrefixSum2.resize(numActiveTasks+1);

            batchData.d_anchorIndicesWithRemovedMates.resize(numActiveTasks);

            batchData.h_numCandidatesPerAnchorPrefixSum[0] = 0;

            batchData.totalNumberOfUsedIds = 0;

            for(int t = 0; t < numActiveTasks; t++){
                auto& task = batchData.tasks[batchData.indicesOfActiveTasks[t]];
                task.dataIsAvailable = false;

                batchData.h_anchorReadIds[t] = task.myReadId;
                batchData.h_mateReadIds[t] = task.mateReadId;
                batchData.totalNumberOfUsedIds += task.allUsedCandidateReadIdPairs.size();

                std::copy(
                    task.encodedMate.begin(),
                    task.encodedMate.end(),
                    batchData.h_inputanchormatedata.begin() + t * encodedSequencePitchInInts
                );
    
                if(task.iteration >= 0){
    
                    batchData.h_anchorSequencesLength[t] = task.currentAnchorLength;
    
                    std::copy(
                        task.currentAnchor.begin(),
                        task.currentAnchor.end(),
                        batchData.h_subjectSequencesData.begin() + t * encodedSequencePitchInInts
                    );
                }else{
                    //only hash kmers which include extended positions
    
                    const int extendedPositionsPreviousIteration 
                        = task.totalAnchorBeginInExtendedRead.at(task.iteration) - task.totalAnchorBeginInExtendedRead.at(task.iteration-1);
    
                    const int lengthToHash = std::min(task.currentAnchorLength, kmerLength + extendedPositionsPreviousIteration - 1);
                    batchData.h_anchorSequencesLength[t] = lengthToHash;
    
                    //std::cerr << "lengthToHash = " << lengthToHash << "\n";
    
                    std::vector<char> buf(task.currentAnchorLength);
                    SequenceHelpers::decode2BitSequence(buf.data(), task.currentAnchor.data(), task.currentAnchorLength);
                    SequenceHelpers::encodeSequence2Bit(
                        batchData.h_subjectSequencesData.get() + t * encodedSequencePitchInInts, 
                        buf.data() + task.currentAnchorLength - lengthToHash, 
                        lengthToHash
                    );
                }
            }

            batchData.h_usedReadIds.resize(batchData.totalNumberOfUsedIds);
            batchData.h_numUsedReadIdsPerAnchor.resize(batchData.numTasks);
            batchData.h_numUsedReadIdsPerAnchorPrefixSum.resize(batchData.numTasks);
            batchData.d_usedReadIds.resize(batchData.totalNumberOfUsedIds);
            batchData.d_numUsedReadIdsPerAnchor.resize(batchData.numTasks);
            batchData.d_numUsedReadIdsPerAnchorPrefixSum.resize(batchData.numTasks);

            batchData.d_segmentIdsOfUsedReadIds.resize(batchData.totalNumberOfUsedIds);
            batchData.h_segmentIdsOfUsedReadIds.resize(batchData.totalNumberOfUsedIds);

            helpers::call_copy_n_kernel(
                thrust::make_zip_iterator(thrust::make_tuple(
                    batchData.h_inputanchormatedata.data(),
                    batchData.h_subjectSequencesData.data()
                )),
                batchData.numTasks * encodedSequencePitchInInts,
                thrust::make_zip_iterator(thrust::make_tuple(
                    batchData.d_inputanchormatedata.data(),
                    batchData.d_subjectSequencesData.data()
                )),
                batchData.streams[0]
            );

            helpers::call_copy_n_kernel(
                thrust::make_zip_iterator(thrust::make_tuple(
                    batchData.h_anchorSequencesLength.data(),
                    batchData.h_anchorReadIds.data(),
                    batchData.h_mateReadIds.data()
                )),
                batchData.numTasks,
                thrust::make_zip_iterator(thrust::make_tuple(
                    batchData.d_anchorSequencesLength.data(),
                    batchData.d_anchorReadIds.data(),
                    batchData.d_mateReadIds.data()
                )),
                batchData.streams[0]
            );

            {
                batchData.h_numUsedReadIdsPerAnchorPrefixSum[0] = 0;

                auto segmentIdsIter = batchData.h_segmentIdsOfUsedReadIds.begin();
                auto h_usedReadIdsIter = batchData.h_usedReadIds.begin();

                for(int i = 0; i < batchData.numTasks; i++){
                    auto& task = vecAccess(batchData.tasks, batchData.indicesOfActiveTasks[i]);

                    const int numUsedIds = task.allUsedCandidateReadIdPairs.size();

                    std::fill(segmentIdsIter, segmentIdsIter + numUsedIds, i);
                    segmentIdsIter += numUsedIds;

                    h_usedReadIdsIter = std::copy(
                        task.allUsedCandidateReadIdPairs.begin(),
                        task.allUsedCandidateReadIdPairs.end(),
                        h_usedReadIdsIter
                    );
                    batchData.h_numUsedReadIdsPerAnchor[i] = numUsedIds;

                    if(i < batchData.numTasks - 1){
                        batchData.h_numUsedReadIdsPerAnchorPrefixSum[i+1] 
                            = batchData.h_numUsedReadIdsPerAnchorPrefixSum[i] + batchData.h_numUsedReadIdsPerAnchor[i];
                    }
                }

                assert(std::distance(batchData.h_usedReadIds.data(), h_usedReadIdsIter) == batchData.totalNumberOfUsedIds);

                // cudaMemcpyAsync(
                //     batchData.d_segmentIdsOfUsedReadIds.data(),
                //     batchData.h_segmentIdsOfUsedReadIds.data(),
                //     sizeof(int) * batchData.totalNumberOfUsedIds,
                //     H2D,
                //     batchData.streams[1]
                // ); CUERR;

                // cudaMemcpyAsync(
                //     batchData.d_usedReadIds.data(),
                //     batchData.h_usedReadIds.data(),
                //     sizeof(read_number) * batchData.totalNumberOfUsedIds,
                //     H2D,
                //     batchData.streams[1]
                // ); CUERR;

                helpers::call_copy_n_kernel(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        batchData.h_segmentIdsOfUsedReadIds.data(),
                        batchData.h_usedReadIds.data()
                    )),
                    batchData.totalNumberOfUsedIds,
                    thrust::make_zip_iterator(thrust::make_tuple(
                        batchData.d_segmentIdsOfUsedReadIds.data(),
                        batchData.d_usedReadIds.data()
                    )),
                    batchData.streams[1]
                );

                helpers::call_copy_n_kernel(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        batchData.h_numUsedReadIdsPerAnchorPrefixSum.data(),
                        batchData.h_numUsedReadIdsPerAnchor.data()
                    )),
                    batchData.numTasks,
                    thrust::make_zip_iterator(thrust::make_tuple(
                        batchData.d_numUsedReadIdsPerAnchorPrefixSum.data(),
                        batchData.d_numUsedReadIdsPerAnchor.data()
                    )),
                    batchData.streams[1]
                );
            }


            hashTimer.start();

            //computes the candidate read ids for current tasks.
            //Sets batchData.totalNumCandidates to the sum of number of candidates for all tasks
            nvtx::push_range("getCandidateReadIds", 0);

            getCandidateReadIds(batchData, batchData.streams[0]);

            nvtx::pop_range();

            // nvtx::push_range("computesegmentids", 4);

            // batchData.h_segmentIdsOfReadIds.resize(batchData.totalNumCandidates);

            // batchData.h_numCandidatesPerAnchor.resize(batchData.numTasks);

            // cudaMemcpyAsync(
            //     batchData.h_numCandidatesPerAnchor.data(),
            //     batchData.d_numCandidatesPerAnchor.data(),
            //     sizeof(int) * batchData.numTasks,
            //     D2H,
            //     batchData.streams[0]
            // ); 

            // cudaStreamSynchronize(batchData.streams[0]); CUERR;


            // for(int i = 0, sum = 0; i < batchData.numTasks; i++){
            //     std::fill(
            //         batchData.h_segmentIdsOfReadIds.data() + sum,
            //         batchData.h_segmentIdsOfReadIds.data() + sum + batchData.h_numCandidatesPerAnchor[i],
            //         i
            //     );
            //     sum += batchData.h_numCandidatesPerAnchor[i];
            // }

            // nvtx::pop_range();

            hashTimer.stop();           

            //remove candidate read ids with are equal to the anchors mate read id, or which have already been used in a previous iteration.
            //Sets batchData.totalNumCandidates to the sum of number of candidates for all tasks.
            //Sets batchData.numTasksWithMateRemoved to the number of tasks whose mate read id appeared in its candidate ids.
            nvtx::push_range("removeUsedIdsAndMateIds", 1);

            removeUsedIdsAndMateIds(batchData, batchData.streams[0], batchData.streams[1]);        

            nvtx::pop_range();


            collectTimer.start();

            nvtx::push_range("loadCandidateSequenceData", 2);

            loadCandidateSequenceData(batchData, batchData.streams[0]);

            nvtx::pop_range();
            
            if(batchData.numTasksWithMateRemoved > 0){

                //for those tasks where a mate id has been removed, remove candidates whose sequence is equal to the mate sequence.
                //Sets batchData.totalNumCandidates to the sum of number of candidates for all tasks.

                nvtx::push_range("eraseDataOfRemovedMates", 3);

                eraseDataOfRemovedMates(batchData, batchData.streams[0]);

                nvtx::pop_range();

            }

            collectTimer.stop();

            /*
                Compute alignments
            */

            alignmentTimer.start();

            nvtx::push_range("calculateAlignments", 4);

            calculateAlignments(batchData, batchData.streams[0]);

            nvtx::pop_range();

            alignmentTimer.stop();
            
            nvtx::push_range("filterAlignments", 5);
       
            //Sets batchData.totalNumCandidates to the sum of number of candidates for all tasks.
            filterAlignments(batchData, batchData.streams[0]);

            nvtx::pop_range();

            nvtx::push_range("computeMSAs", 6);

            //Sets batchData.totalNumCandidates to the sum of number of candidates for all tasks. (msa refinement can remove candidates)
            computeMSAs(batchData, batchData.streams[0], batchData.streams[1]);

            nvtx::pop_range();

            //copy all necessary buffers to host
                
            nvtx::push_range("copyBuffersToHost", 7);

            copyBuffersToHost(batchData, batchData.streams[0], batchData.streams[1]);

            nvtx::pop_range();
            

            cudaStreamSynchronize(batchData.streams[0]); CUERR;
            cudaStreamSynchronize(batchData.streams[1]); CUERR;

            for(int i = 0; i < numActiveTasks; i++){
                auto& task = vecAccess(batchData.tasks, batchData.indicesOfActiveTasks[i]);

                task.numRemainingCandidates = batchData.h_numCandidatesPerAnchor[i];

                if(task.numRemainingCandidates == 0){
                    task.abort = true;
                    task.abortReason = AbortReason::NoPairedCandidatesAfterAlignment;
                }
            }


            std::vector<Task> newTasksFromSplit;
            std::vector<int> newTaskIndices;

            auto constructMsa = [&](auto& task, int taskindex){
                assert(task.dataIsAvailable);
                return constructMsaWithDataFromTask(task);
            };

            auto extendWithMsa = [&](auto& task, const char* consensus, int consensusLength, int taskIndex){

                //can extend by at most maxextensionPerStep bps
                int extendBy = std::min(
                    consensusLength - task.currentAnchorLength, 
                    maxextensionPerStep
                );
                //cannot extend over fragment 
                extendBy = std::min(extendBy, (insertSize + insertSizeStddev - task.mateLength) - task.accumExtensionLengths);

                auto makeAnchorForNextIteration = [&](){
                    if(extendBy == 0){
                        task.abort = true;
                        task.abortReason = AbortReason::MsaNotExtended;
                    }else{
                        task.accumExtensionLengths += extendBy;

                        //update data for next iteration of outer while loop                           

                        std::string decodedAnchor(consensus + extendBy, task.currentAnchorLength);

                        const int numInts = SequenceHelpers::getEncodedNumInts2Bit(task.currentAnchorLength);

                        task.currentAnchor.resize(numInts);

                        SequenceHelpers::encodeSequence2Bit(
                            task.currentAnchor.data(), 
                            decodedAnchor.data(), 
                            task.currentAnchorLength
                        );

                        task.totalDecodedAnchors.emplace_back(std::move(decodedAnchor));
                        task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);

                        // task.resultsequence.insert(
                        //     task.resultsequence.end(), 
                        //     consensus + task.currentAnchorLength, 
                        //     consensus + task.currentAnchorLength + extendBy
                        // );


                        // std::string tmp(task.currentAnchorLength, '\0');

                        // decode2BitSequence(
                        //     &tmp[0],
                        //     task.currentAnchor.data(),
                        //     task.currentAnchorLength
                        // );

                        // auto sub = task.resultsequence.substr(task.resultsequence.length() - task.currentAnchorLength);

                        // assert(sub == tmp);
                    }
                };

                constexpr int requiredOverlapMate = 70; //TODO relative overlap 
                constexpr int numMismatchesUpperBound = 2;

                if(task.pairedEnd && task.accumExtensionLengths + consensusLength - requiredOverlapMate + task.mateLength >= insertSize - insertSizeStddev){
                    //check if mate can be overlapped with consensus 

                    //hamMap[i] stores possible starting positions of overlaps which would have hamming distance i
                    std::map<int, std::vector<int>> hamMap;

                    //longmatchMap[i] stores possible starting positions of overlaps which would have a longest match of length i between mate and msa consensus
                    //std::map<int, std::vector<int>> longmatchMap; //map length of longest match to list start positions

                    //for each possibility to overlap the mate and consensus such that the merged sequence would end in the desired range [insertSize - insertSizeStddev, insertSize + insertSizeStddev]

                    const int firstStartpos = std::max(0, insertSize - insertSizeStddev - task.accumExtensionLengths - task.mateLength);
                    const int lastStartposExcl = std::min(
                        std::max(0, insertSize + insertSizeStddev - task.accumExtensionLengths - task.mateLength) + 1,
                        consensusLength - requiredOverlapMate
                    );

                    for(int startpos = firstStartpos; startpos < lastStartposExcl; startpos++){
                        //compute metrics of overlap
                            
                        const int ham = cpu::hammingDistanceOverlap(
                            consensus + startpos, consensus + consensusLength, 
                            task.decodedMateRevC.begin(), task.decodedMateRevC.end()
                        );

                        hamMap[ham].emplace_back(startpos);

                        // const int longest = cpu::longestMatch(
                        //     consensus + startpos, consensus + consensusLength, 
                        //     task.decodedMateRevC.begin(), task.decodedMateRevC.end()
                        // );

                        // longmatchMap[longest].emplace_back(startpos);
                    }
                    
                    std::vector<std::pair<int, std::vector<int>>> flatMap(hamMap.begin(), hamMap.end());
                    //sort by hamming distance, ascending
                    std::sort(flatMap.begin(), flatMap.end(), [](const auto& p1, const auto& p2){return p1.first < p2.first;});

                    //std::vector<std::pair<int, std::vector<int>>> flatMap2(longmatchMap.begin(), longmatchMap.end());
                    //sort by length of longest match, descending
                    //std::sort(flatMap2.begin(), flatMap2.end(), [](const auto& p1, const auto& p2){return p2.first < p1.first;});

                    //if there exists an overlap between msa consensus and mate which would end the merge, use the best one
                    if(flatMap.size() > 0 && flatMap[0].first <= numMismatchesUpperBound){
                    //if(flatMap2.size() > 0 && flatMap2[0].first >= 40){
                        const int mateStartposInConsensus = flatMap[0].second.front();
                        const int missingPositionsBetweenAnchorEndAndMateBegin = std::max(0, mateStartposInConsensus - task.currentAnchorLength);

                        if(missingPositionsBetweenAnchorEndAndMateBegin > 0){
                            //bridge the gap between current anchor and mate
                            task.totalDecodedAnchors.emplace_back(
                                consensus + missingPositionsBetweenAnchorEndAndMateBegin,
                                consensus + missingPositionsBetweenAnchorEndAndMateBegin + mateStartposInConsensus
                            );
                            task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths + missingPositionsBetweenAnchorEndAndMateBegin);
                        }


                        task.mateHasBeenFound = true;

                        //const int currentAccumExtensionLengths = task.accumExtensionLengths;
                        
                        task.accumExtensionLengths += mateStartposInConsensus;
                        std::string decodedAnchor(task.decodedMateRevC);

                        task.totalDecodedAnchors.emplace_back(std::move(decodedAnchor));
                        task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);

                        // const int startpos = mateStartposInConsensus;
                        // task.resultsequence.resize(currentAccumExtensionLengths + startpos + task.decodedMateRevC.length());
                        // const auto replaceBegin = task.resultsequence.begin() + currentAccumExtensionLengths + startpos;
                        // task.resultsequence.replace(
                        //     replaceBegin, 
                        //     replaceBegin + task.decodedMateRevC.length(), 
                        //     task.decodedMateRevC.begin(), 
                        //     task.decodedMateRevC.end()
                        // );

                    }else{
                        makeAnchorForNextIteration();
                    }
                }else{
                    makeAnchorForNextIteration();
                }
            };

            auto keepSelectedCandidates = [&](auto& task, const auto& selectedCandidateIndices, int taskIndex){
                assert(task.dataIsAvailable);

                const int numCandidateIndices = selectedCandidateIndices.size();
                assert(numCandidateIndices <= task.numRemainingCandidates);

                for(int i = 0; i < numCandidateIndices; i++){
                    const int c = vecAccess(selectedCandidateIndices, i);
                    // if(!(0 <= c && c < task.candidateReadIds.size())){
                    //     std::cerr << "c = " << c << ", candidateReadIds.size() = " << task.candidateReadIds.size() << "\n";
                    // }

                    // assert(0 <= c && c < task.candidateReadIds.size());
                    // assert(0 <= c && c < task.candidateSequenceLengths.size());
                    // assert(0 <= c && c < task.alignments.size());
                    // assert(0 <= c && c < task.alignmentFlags.size());

                    // assert(0 <= c && c*encodedSequencePitchInInts < task.candidateSequencesFwdData.size());
                    // assert(0 <= c && c*encodedSequencePitchInInts < task.candidateSequencesRevcData.size());
                    // assert(0 <= c && c*encodedSequencePitchInInts < task.candidateSequenceData.size());

                    vecAccess(task.candidateReadIds, i) = vecAccess(task.candidateReadIds, c);
                    vecAccess(task.candidateSequenceLengths , i) = vecAccess(task.candidateSequenceLengths, c);
                    vecAccess(task.alignments, i) = vecAccess(task.alignments, c);
                    vecAccess(task.alignmentFlags, i) = vecAccess(task.alignmentFlags, c);
                    vecAccess(task.candidateShifts, i) = vecAccess(task.candidateShifts, c);
                    vecAccess(task.candidateOverlapWeights, i) = vecAccess(task.candidateOverlapWeights, c);

                    // std::copy_n(
                    //     task.candidateSequencesFwdData.begin() + c * encodedSequencePitchInInts,
                    //     encodedSequencePitchInInts,
                    //     task.candidateSequencesFwdData.begin() + i * encodedSequencePitchInInts
                    // );

                    // std::copy_n(
                    //     task.candidateSequencesRevcData.begin() + c * encodedSequencePitchInInts,
                    //     encodedSequencePitchInInts,
                    //     task.candidateSequencesRevcData.begin() + i * encodedSequencePitchInInts
                    // );

                    std::copy_n(
                        task.candidateSequenceData.begin() + c * encodedSequencePitchInInts,
                        encodedSequencePitchInInts,
                        task.candidateSequenceData.begin() + i * encodedSequencePitchInInts
                    );

                    std::copy_n(
                        task.candidateStrings.begin() + c * decodedSequencePitchInBytes,
                        decodedSequencePitchInBytes,
                        task.candidateStrings.begin() + i * decodedSequencePitchInBytes
                    );
                }

                task.candidateReadIds.erase(
                    task.candidateReadIds.begin() + numCandidateIndices,
                    task.candidateReadIds.end()
                );
                task.candidateSequenceLengths.erase(
                    task.candidateSequenceLengths.begin() + numCandidateIndices,
                    task.candidateSequenceLengths.end()
                );
                task.alignments.erase(
                    task.alignments.begin() + numCandidateIndices,
                    task.alignments.end()
                );
                task.alignmentFlags.erase(
                    task.alignmentFlags.begin() + numCandidateIndices,
                    task.alignmentFlags.end()
                );
                task.candidateSequenceData.erase(
                    task.candidateSequenceData.begin() + numCandidateIndices * encodedSequencePitchInInts,
                    task.candidateSequenceData.end()
                );
                if(task.pairedEnd){
                    task.mateIdLocationIter = std::lower_bound(
                        task.candidateReadIds.begin(),
                        task.candidateReadIds.end(),
                        task.mateReadId
                    );

                    task.mateHasBeenFound = (task.mateIdLocationIter != task.candidateReadIds.end() 
                        && *task.mateIdLocationIter == task.mateReadId);
                }
                task.numRemainingCandidates = numCandidateIndices;
            };

            nvtx::push_range("MSA", 6);
            msaTimer.start();

            for(int i = 0; i < numActiveTasks; i++){ 
                const int indexOfActiveTask = batchData.indicesOfActiveTasks[i];
                auto& task = vecAccess(batchData.tasks, indexOfActiveTask);

                if(task.numRemainingCandidates == 0){
                    continue;
                }
                assert(task.numRemainingCandidates > 0);

                const std::size_t splitcolumnsPitchElements = 32;
                const auto* const possibleSplitColumns = batchData.h_possibleSplitColumns.data() + splitcolumnsPitchElements * i;
                const int numPossibleSplitColumns = batchData.h_numPossibleSplitColumnsPerAnchor[i];
                const gpu::MSAColumnProperties msaProps = batchData.h_msa_column_properties[i];

                const int consensusLength = msaProps.lastColumn_excl - msaProps.firstColumn_incl;
                const char* const consensus = batchData.h_consensus.data() + i * msaColumnPitchInElements;

                
#if 1
                //if(task.splitDepth == 0){
                if(batchData.splitTracker[task.myReadId] <= 4 && batchData.h_numPossibleSplitColumnsPerAnchor[i] > 0){
                    
                    const int numCandidates = task.numRemainingCandidates;
                    const int offset = batchData.h_numCandidatesPerAnchorPrefixSum[i];
                    const int* const candidateShifts = batchData.h_alignment_shifts.data() + offset;
                    const int* const candidateLengths = batchData.h_candidateSequencesLength.data() + offset;
                    const auto* const alignmentFlags = batchData.h_alignment_best_alignment_flags.data() + offset;

                    const unsigned int* const myCandidateSequencesData = &batchData.h_candidateSequencesData[offset * encodedSequencePitchInInts];

                    task.candidateStrings.resize(decodedSequencePitchInBytes * numCandidates, '\0');

                    //decode the candidates for msa
                    for(int c = 0; c < task.numRemainingCandidates; c++){
                        SequenceHelpers::decode2BitSequence(
                            task.candidateStrings.data() + c * decodedSequencePitchInBytes,
                            myCandidateSequencesData + c * encodedSequencePitchInInts,
                            candidateLengths[c]
                        );

                        if(alignmentFlags[c] == BestAlignment_t::ReverseComplement){
                            SequenceHelpers::reverseComplementSequenceDecodedInplace(
                                task.candidateStrings.data() + c * decodedSequencePitchInBytes, 
                                candidateLengths[c]
                            );
                        }
                    }

                    auto possibleSplits = inspectColumnsRegionSplit(
                        possibleSplitColumns,
                        numPossibleSplitColumns,
                        task.currentAnchorLength, 
                        msaProps.lastColumn_excl - msaProps.firstColumn_incl,
                        msaProps.subjectColumnsBegin_incl,
                        numCandidates,
                        task.candidateStrings.data(),
                        decodedSequencePitchInBytes,
                        candidateShifts,
                        candidateLengths
                    );

                    if(possibleSplits.splits.size() > 1){
                        //nvtx::push_range("split msa", 8);
                        //auto& task = batchData.tasks[indexOfActiveTask];
                        #if 1
                        std::sort(
                            possibleSplits.splits.begin(), 
                            possibleSplits.splits.end(),
                            [](const auto& split1, const auto& split2){
                                //sort by size, descending
                                return split2.listOfCandidates.size() < split1.listOfCandidates.size();
                            }
                        );
                        #else
                        std::nth_element(
                            possibleSplits.splits.begin(), 
                            possibleSplits.splits.begin() + 1, 
                            possibleSplits.splits.end(),
                            [](const auto& split1, const auto& split2){
                                //sort by size, descending
                                return split2.listOfCandidates.size() < split1.listOfCandidates.size();
                            }
                        );
                        #endif

                        // std::cerr << "split[0] = ";
                        // for(auto x : possibleSplits.splits[0].listOfCandidates) std::cerr << x << " ";
                        // std::cerr << "\nsplit[1] = ";
                        // for(auto x : possibleSplits.splits[1].listOfCandidates) std::cerr << x << " ";
                        // std::cerr << "\n";

                        // auto printColumnInfo = [](const auto& x){
                        //     std::cerr << "(" << x.column << ", " << x.letter << ", " << x.ratio << ") ";
                        // };

                        // std::cerr << "columns[0] = ";
                        // for(auto x : possibleSplits.splits[0].columnInfo) printColumnInfo(x);
                        // std::cerr << "\ncolumns[1] = ";
                        // for(auto x : possibleSplits.splits[1].columnInfo) printColumnInfo(x);
                        // std::cerr << "\n";


                        //copy task's data from batchData into task

                        copyBatchDataIntoTask(task, i, batchData);

                        //create the separate shifts array
                        //create defaultweights, which is split in keepSelectedCandidates

                        task.candidateShifts.resize(task.alignments.size());
                        task.candidateOverlapWeights.resize(task.numRemainingCandidates);

                        for(int c = 0; c < task.numRemainingCandidates; c++){
                            task.candidateShifts[c] = task.alignments[c].shift;

                            task.candidateOverlapWeights[c] = calculateOverlapWeight(
                                task.currentAnchorLength, 
                                task.alignments[c].nOps,
                                task.alignments[c].overlap,
                                goodAlignmentProperties.maxErrorRate
                            );
                        }                       

                        task.dataIsAvailable = true;
                        //create a copy of task, and only keep candidates of first split
                        
                        Task taskCopy = task;
                        taskCopy.splitDepth++;

                        // std::cerr << "split\n";
                        // msa.print(std::cerr); 
                        // std::cerr << "\n into \n";

                        keepSelectedCandidates(taskCopy, possibleSplits.splits[0].listOfCandidates, -1);
                        const MultipleSequenceAlignment msaOfCopy = constructMsa(taskCopy, -1);

                        // msaOfCopy.print(std::cerr); 
                        // std::cerr << "\n and \n";

                        extendWithMsa(taskCopy, msaOfCopy.consensus.data(), msaOfCopy.consensus.size(), -1);

                        //only keep canddiates of second split
                        keepSelectedCandidates(task, possibleSplits.splits[1].listOfCandidates, -1); 
                        const MultipleSequenceAlignment newMsa = constructMsa(task, -1);

                        // newMsa.print(std::cerr); 
                        // std::cerr << "\n";

                        extendWithMsa(task, newMsa.consensus.data(), newMsa.consensus.size(), -1);

                        //if extension was not possible in task, replace task by task copy
                        if(task.abort && task.abortReason == AbortReason::MsaNotExtended){
                            //replace task by taskCopy
                            task = std::move(taskCopy);
                        }else if(!taskCopy.abort){
                            //if extension was possible in both task and taskCopy, taskCopy will be added to tasks and list of active tasks
                            newTaskIndices.emplace_back(batchData.tasks.size() + newTasksFromSplit.size());
                            newTasksFromSplit.emplace_back(std::move(taskCopy));

                            batchData.splitTracker[task.myReadId]++;


                        }
                        //nvtx::pop_range();                     
                    }else{
                        extendWithMsa(task, consensus, consensusLength, indexOfActiveTask);
                    }
                }else{
                    extendWithMsa(task, consensus, consensusLength, indexOfActiveTask);
                }
#else 
                extendWithMsa(task, consensus, consensusLength, indexOfActiveTask);
#endif

            }

            msaTimer.stop();

            nvtx::pop_range();

            if(newTasksFromSplit.size() > 0){
                //std::cerr << "Added " << newTasksFromSplit.size() << " tasks\n";
                batchData.tasks.insert(
                    batchData.tasks.end(), 
                    std::make_move_iterator(newTasksFromSplit.begin()), 
                    std::make_move_iterator(newTasksFromSplit.end())
                );
                batchData.indicesOfActiveTasks.insert(
                    batchData.indicesOfActiveTasks.end(), 
                    newTaskIndices.begin(), 
                    newTaskIndices.end()
                );
            }           

            /*
                update book-keeping of used candidates
            */  

            for(int i = 0; i < numActiveTasks; i++){
                auto& task = vecAccess(batchData.tasks, batchData.indicesOfActiveTasks[i]);

                                      
                {
                    if(task.dataIsAvailable){
                        std::vector<read_number> tmp(task.allUsedCandidateReadIdPairs.size() + task.candidateReadIds.size());
                        auto tmp_end = std::merge(
                            task.allUsedCandidateReadIdPairs.begin(),
                            task.allUsedCandidateReadIdPairs.end(),
                            task.candidateReadIds.begin(),
                            task.candidateReadIds.end(),
                            tmp.begin()
                        );

                        tmp.erase(tmp_end, tmp.end());

                        std::swap(task.allUsedCandidateReadIdPairs, tmp);
                    }else{
                        const int numCandidates = batchData.h_numCandidatesPerAnchor[i];
                        const int offset = batchData.h_numCandidatesPerAnchorPrefixSum[i];
                        const read_number* ids = &batchData.h_candidateReadIds[offset];

                        std::vector<read_number> tmp(task.allUsedCandidateReadIdPairs.size() + numCandidates);
                        auto tmp_end = std::merge(
                            task.allUsedCandidateReadIdPairs.begin(),
                            task.allUsedCandidateReadIdPairs.end(),
                            ids,
                            ids + numCandidates,
                            tmp.begin()
                        );

                        tmp.erase(tmp_end, tmp.end());

                        std::swap(task.allUsedCandidateReadIdPairs, tmp);
                    }
                }

                // task.usedCandidateReadIdsPerIteration.emplace_back(std::move(task.candidateReadIds));
                // task.usedAlignmentsPerIteration.emplace_back(std::move(task.alignments));
                // task.usedAlignmentFlagsPerIteration.emplace_back(std::move(task.alignmentFlags));

                task.iteration++;
            }
            
            //update list of active task indices

            batchData.indicesOfActiveTasks.erase(
                std::remove_if(
                    batchData.indicesOfActiveTasks.begin(), 
                    batchData.indicesOfActiveTasks.end(),
                    [&](int index){
                        return !batchData.tasks[index].isActive(insertSize, insertSizeStddev);
                    }
                ),
                batchData.indicesOfActiveTasks.end()
            );
        }

        //construct results

        std::vector<ExtendResult> extendResults;

        for(const auto& task : batchData.tasks){

            ExtendResult extendResult;
            extendResult.direction = task.direction;
            extendResult.numIterations = task.iteration;
            extendResult.aborted = task.abort;
            extendResult.abortReason = task.abortReason;
            extendResult.readId1 = task.myReadId;
            extendResult.readId2 = task.mateReadId;
            extendResult.originalLength = task.myLength;

#if 0
            //extendResult.extendedRead = std::move(task.resultsequence);
            extendResult.success = true;
            extendResult.mateHasBeenFound = task.mateHasBeenFound;

#else
            // if(abort){
            //     ; //no read extension possible
            // }else
            {
                //if(mateHasBeenFound){
                {
                    //construct extended read
                    //build msa of all saved totalDecodedAnchors[0]

                    const int numsteps = task.totalDecodedAnchors.size();

                    // if(task.myReadId == 90 || task.mateReadId == 90){
                    //     std::cerr << "task.totalDecodedAnchors\n";
                    // }

                    int maxlen = 0;
                    for(const auto& s: task.totalDecodedAnchors){
                        const int len = s.length();
                        if(len > maxlen){
                            maxlen = len;
                        }

                        // if(task.myReadId == 90 || task.mateReadId == 90){
                        //     std::cerr << s << "\n";
                        // }
                    }

                    // if(task.myReadId == 90 || task.mateReadId == 90){
                    //     std::cerr << "\n";
                    // }

                    const std::string& decodedAnchor = vecAccess(task.totalDecodedAnchors, 0);

                    const std::vector<int> shifts(task.totalAnchorBeginInExtendedRead.begin() + 1, task.totalAnchorBeginInExtendedRead.end());
                    std::vector<float> initialWeights(numsteps-1, 1.0f);


                    std::vector<char> stepstrings(maxlen * (numsteps-1), '\0');
                    std::vector<int> stepstringlengths(numsteps-1);
                    for(int c = 1; c < numsteps; c++){
                        std::copy(
                            vecAccess(task.totalDecodedAnchors, c).begin(),
                            vecAccess(task.totalDecodedAnchors, c).end(),
                            stepstrings.begin() + (c-1) * maxlen
                        );
                        vecAccess(stepstringlengths, c-1) = vecAccess(task.totalDecodedAnchors, c).size();
                    }

                    MultipleSequenceAlignment::InputData msaInput;
                    msaInput.useQualityScores = false;
                    msaInput.subjectLength = decodedAnchor.length();
                    msaInput.nCandidates = numsteps-1;
                    msaInput.candidatesPitch = maxlen;
                    msaInput.candidateQualitiesPitch = 0;
                    msaInput.subject = decodedAnchor.c_str();
                    msaInput.candidates = stepstrings.data();
                    msaInput.subjectQualities = nullptr;
                    msaInput.candidateQualities = nullptr;
                    msaInput.candidateLengths = stepstringlengths.data();
                    msaInput.candidateShifts = shifts.data();
                    msaInput.candidateDefaultWeightFactors = initialWeights.data();

                    MultipleSequenceAlignment msa;

                    msa.build(msaInput);

                    // if(task.myReadId == 90 || task.mateReadId == 90){
                    //     std::cerr << "Id " << task.myReadId << ", Final\n";
                    //     msa.print(std::cerr);
                    //     std::cerr << "\n";
                    // }

                    extendResult.success = true;

                    std::string extendedRead(msa.consensus.begin(), msa.consensus.end());
                    //std::cerr << "before: " << extendedRead << "\n";
                    std::copy(decodedAnchor.begin(), decodedAnchor.end(), extendedRead.begin());
                    if(task.mateHasBeenFound){
                        std::copy(
                            task.decodedMateRevC.begin(),
                            task.decodedMateRevC.end(),
                            extendedRead.begin() + extendedRead.length() - task.decodedMateRevC.length()
                        );
                    }
                    // extendedRead.replace(extendedRead.begin(), extendedRead.begin() + decodedAnchor, decodedAnchor.begin(), decodedAnchor.end());
                    // std::cerr << "after : " << extendedRead << "\n";
                    
                    // msa.print(std::cerr);
                    // std::cerr << "msa cons:\n";
                    // std::cerr << extendedRead << "\n";
                    // std::cerr << "new cons:\n";
                    // std::cerr << task.resultsequence << "\n";


                    extendResult.extendedRead = std::move(extendedRead);

                    extendResult.mateHasBeenFound = task.mateHasBeenFound;
                }
                // else{
                //     ; //no read extension possible
                // }
            }

            // if(extendResult.extendedRead.length() != task.resultsequence.length()){
            //     std::cerr << task.myReadId << "\n";
            //     std::cerr << extendResult.extendedRead << "\n";
            //     std::cerr << task.resultsequence << "\n";
            //     std::exit(0);
            // }
#endif
            extendResults.emplace_back(std::move(extendResult));

        }

        return extendResults;
    }
#endif

    std::vector<ExtendResult> ReadExtenderGpu::processSingleEndTasks(
        std::vector<ReadExtenderBase::Task> tasks
    ){
        return processPairedEndTasks(std::move(tasks));
    }


    




}