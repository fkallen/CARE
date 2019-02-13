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
    struct Dist2{
        std::vector<std::pair<T, Count>> percentRanges;
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

	template<class T, class Count>
    Dist2<T,Count> estimateDist2(const std::map<T,Count>& map){

        Dist2<T, Count> distribution;
        distribution.percentRanges.resize(101);

        std::vector<std::pair<T, Count>> vec(map.begin(), map.end());
        std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b){
            return a.first < b.first;
        });

		Count totalCandidates = 0;

        for(const auto& pair : vec){
			totalCandidates += pair.second;
		}

		std::size_t boundary = 0;
		Count candidatesUntilBoundary = 0;

		for(int i = 0; i <= 100; i++){
			double fac = i / 100.0;
			while(candidatesUntilBoundary < fac * totalCandidates && boundary < vec.size()){
				candidatesUntilBoundary += vec[boundary].second;
				boundary++;
			}

			std::size_t b = boundary > 0 ? boundary-1 : boundary;
			std::cout << (i) << " % boundary element: " << vec[b].first << ", " << vec[b].second << std::endl;
            distribution.percentRanges[i] = vec[b];
		}

        return distribution;
    }

    template<class minhasher_t, class readStorage_t>
    std::map<std::int64_t, std::int64_t> getCandidateCountHistogram(const minhasher_t& minhasher,
                                                                    const readStorage_t& readStorage,
                                                                    std::uint64_t candidatesToCheck,
                                                                    int num_hits,
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
                    //std::string sequencestring = readStorage.fetchSequence_ptr(readId)->toString();
					const std::uint8_t* sequenceptr = (const std::uint8_t*)readStorage.fetchSequenceData_ptr(readId);
					const int sequencelength = readStorage.fetchSequenceLength(readId);
					const std::string sequencestring = Sequence_t::Impl_t::toString(sequenceptr, sequencelength);
					//if(sequencestring != sequencestring2){
					//	std::cout << sequencestring << '\n' << sequencestring2 << std::endl;
					//	assert(false);
					//}

                    //auto candidateList = minhasher.getCandidates(sequencestring, std::numeric_limits<std::uint64_t>::max());
                    //std::int64_t count = std::int64_t(candidateList.size()) - 1;

                    std::int64_t count = minhasher.getNumberOfCandidates(sequencestring, num_hits);
                    //std::int64_t count = minhasher.getNumberOfCandidates(sequencestring, 2);
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

        //for(auto p : allncandidates){
        //    std::cerr << p.first << " " << p.second << '\n';
        //}

        return allncandidates;
    }
}
}




#endif