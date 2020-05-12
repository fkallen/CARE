#ifndef CARE_GPUMINHASHER_CUH
#define CARE_GPUMINHASHER_CUH

#include <config.hpp>
#include <gpu/gpuhashtable.cuh>

#include <vector>
#include <memory>
#include <limits>

namespace care{
namespace gpu{

    class GpuMinhasher{
    public:
        using Key_t = kmer_type;
        using Value_t = read_number;

        using Range_t = std::pair<const Value_t*, const Value_t*>;

        void queryPrecalculatedSignatures(
            const std::uint64_t* signatures, //getNumberOfMaps() elements per sequence
            GpuMinhasher::Range_t* ranges, //getNumberOfMaps() elements per sequence
            int* totalNumResultsInRanges, 
            int numSequences) const{ 
            
            int numResults = 0;
    
            for(int i = 0; i < numSequences; i++){
                const std::uint64_t* const signature = &signatures[i * getNumberOfMaps()];
                GpuMinhasher::Range_t* const range = &ranges[i * getNumberOfMaps()];            
    
                for(int map = 0; map < getNumberOfMaps(); ++map){
                    Key_t key = signature[map] & key_mask;
                    auto entries_range = queryMap(map, key);
                    numResults += std::distance(entries_range.first, entries_range.second);
                    range[map] = entries_range;
                }
            }   
    
            *totalNumResultsInRanges = numResults;   
        }

        int getNumberOfMaps() const{
            return minhashTables.size();
        }

        int getKmerSize() const{
            return kmerSize;
        }

        int getNumResultsPerMapThreshold() const{

        }

        void addHashMap(HashMap&& hm){
            minhashTables.emplace(std::make_unique<Hashmap>(std::move(hm)));
        }

        int calculateResultsPerMapThreshold(int coverage){
            int result = int(coverage * 2.5f);
            result = std::min(result, int(std::numeric_limits<BucketSize>::max()));
            result = std::max(10, result);
            return result;
        }

        void computeReadHashesOnGpu(
            std::uint64_t* d_hashValues,
            std::size_t hashValuesPitchInElements,
            const unsigned int* d_encodedSequenceData,
            std::size_t encodedSequencePitchInInts,
            int numSequences,
            const int* d_sequenceLengths,
            cudaStream_t stream
        ){
            callMinhashSignaturesKernel_async(
                d_hashValues,
                hashValuesPitchInElements,
                d_encodedSequenceData,
                encodedSequencePitchInInts,
                numSequences,
                d_sequenceLengths,
                getKmerSize,
                getNumberOfMaps(),
                nextData.stream
            );
        }

        void computeReadHashesOnGpu(
            std::uint64_t* d_hashValues,
            std::size_t hashValuesPitchInElements,
            const unsigned int* d_encodedSequenceData,
            std::size_t encodedSequencePitchInInts,
            int numSequences,
            const int* d_sequenceLengths,
            int firstHashFunc,
            cudaStream_t stream
        ){
            callMinhashSignaturesKernel_async(
                d_hashValues,
                hashValuesPitchInElements,
                d_encodedSequenceData,
                encodedSequencePitchInInts,
                numSequences,
                d_sequenceLengths,
                getKmerSize,
                getNumberOfMaps(),
                firstHashFunc,
                nextData.stream
            );
        }

        



    private:
        using HashMap = CpuReadOnlyMultiValueHashTable<kmer_type, read_number>;

        Range_t queryMap(int id, const Key_t& key) const{
            HashMap::QueryResult qr = minhashTables[id].query(key);

            return std::make_pair(qr.valuesBegin, qr.valuesBegin + qr.numValues);
        }



        int kmerSize;
        int resultsPerMapThreshold;
        std::vector<std::unique_ptr<HashMap>> minhashTables;
    };




    
}
}



#endif
