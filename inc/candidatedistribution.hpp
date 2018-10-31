#ifndef CARE_CANDIDATE_DISTRIBUTION_HPP
#define CARE_CANDIDATE_DISTRIBUTION_HPP

#include <map>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdint>
#include <future>
#include <string>

namespace care{
namespace cpu{
    
    template<class T, class Count>
    struct Dist{
        T max;
        T average;
        T stddev;
        Count maxCount;
        Count averageCount;
    };

    template<class T, class Count>
    Dist<T,Count> estimateDist(const std::map<T,Count>& map){
        Dist<T, Count> distribution;

        Count sum = 0;
        std::vector<std::pair<T, Count>> vec(map.begin(), map.end());
        std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b){
            return a.second < b.second;
        });

    // AVG
        sum = 0;
        for(const auto& pair : map){
            sum += pair.second;
        }
        distribution.averageCount = sum / vec.size();

        auto it = std::lower_bound(vec.begin(),
                                    vec.end(),
                                    std::make_pair(T{}, distribution.averageCount),
                                    [](const auto& a, const auto& b){
                                        return a.second < b.second;
                                    });
        if(it == vec.end())
            it = vec.end() - 1;

        distribution.average = it->first;
    // MAX
        it = std::max_element(vec.begin(), vec.end(), [](const auto& a, const auto& b){
            return a.second < b.second;
        });

        distribution.max = it->first;
        distribution.maxCount = it->second;
    // STDDEV
        T sum2 = 0;
        distribution.stddev = 0;
        for(const auto& pair : map){
            sum2 += pair.first - distribution.average;
        }

        distribution.stddev = std::sqrt(1.0/vec.size() * sum2);

        return distribution;
    }

    template<class minhasher_t, class readStorage_t>
    std::map<std::int64_t, std::int64_t> getCandidateCountHistogram(const minhasher_t& minhasher,
                                                                    const readStorage_t& readStorage,
                                                                    std::uint64_t candidatesToCheck,
                                                                    int threads){

        using ReadStorage_t = readStorage_t;
        using Sequence_t = typename ReadStorage_t::Sequence_t;
        using ReadId_t = typename ReadStorage_t::ReadId_t;

        std::vector<std::future<std::map<std::int64_t, std::int64_t>>> candidateCounterFutures;
        const ReadId_t sampleCount = candidatesToCheck;
        for(int i = 0; i < threads; i++){
            candidateCounterFutures.push_back(std::async(std::launch::async, [&,i]{
                std::map<std::int64_t, std::int64_t> candidateMap;
                std::vector<std::pair<ReadId_t, const Sequence_t*>> numseqpairs;
                typename minhasher_t::Handle handle;

                for(ReadId_t readId = i; readId < sampleCount; readId += threads){
                    std::string sequencestring = readStorage.fetchSequence_ptr(readId)->toString();
                    //auto candidateList = minhasher.getCandidates(sequencestring, std::numeric_limits<std::uint64_t>::max());
                    //std::int64_t count = std::int64_t(candidateList.size()) - 1;

                    std::int64_t count = minhasher.getNumberOfCandidates(sequencestring, handle);
                    if(count > 0)
                        --count;

                    //std::int64_t count = minhasher.getNumberOfCandidatesUpperBound(sequencestring);

                    candidateMap[count]++;
                }

                return candidateMap;
            }));
        }

        std::map<std::int64_t, std::int64_t> allncandidates;

        for(auto& future : candidateCounterFutures){
            const auto& tmpresult = future.get();
            for(const auto& pair : tmpresult){
                allncandidates[pair.first] += pair.second;
            }
        }

        return allncandidates;
    }
}
}




#endif
