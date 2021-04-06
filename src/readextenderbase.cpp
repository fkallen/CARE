#include <readextenderbase.hpp>
#include <cpu_alignment.hpp>

#include <stringglueing.hpp>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>
#include <cassert>
#include <iterator>

namespace care{

    std::vector<ReadExtenderBase::ExtendResult> ReadExtenderBase::combinePairedEndDirectionResults(
        std::vector<ReadExtenderBase::ExtendResult>& pairedEndDirectionResults
    ){
        auto idcomp = [](const auto& l, const auto& r){ return l.getReadPairId() < r.getReadPairId();};
        auto lengthcomp = [](const auto& l, const auto& r){ return l.extendedRead.length() < r.extendedRead.length();};

        std::vector<ReadExtenderBase::ExtendResult>& combinedResults = pairedEndDirectionResults;

        std::sort(
            combinedResults.begin(), 
            combinedResults.end(),
            idcomp
        );

        auto glue = [&](const std::string& lrString, const std::string& rlString, const auto& decider, const auto& gluer){
            std::vector<std::string> possibleResults;

            const int maxNumberOfPossibilities = 2*insertSizeStddev + 1;

            for(int p = 0; p < maxNumberOfPossibilities; p++){
                auto decision = decider(
                    lrString, 
                    rlString, 
                    insertSize - insertSizeStddev + p
                );

                if(decision.has_value()){
                    // auto aaa = gluer(*decision);
                    // auto bbb = wgluer(*decision);
                    // if(aaa != bbb){
                    //     std::cerr << "glue\n" << lrString << " and\n" << rlString << "\n";
                    //     std::cerr << aaa << "\n" << bbb << "\n\n";
                    // }
                    possibleResults.emplace_back(gluer(*decision));
                }
            }

            return possibleResults;
        };

        auto combineWithSameIdNoMate = [&](auto begin, auto end, auto func){
            assert(std::distance(begin, end) > 0);

            //TODO optimization: store pairs of indices to results
            //std::vector<std::pair<ReadExtenderBase::ExtendResult, ReadExtenderBase::ExtendResult>> pairsToCheck;
            std::vector<std::pair<int, int>> pairPositionsToCheck;

            constexpr int minimumOverlap = 40;

            //try to find a pair of extensions with opposite directions which could be overlapped to produce an extension which reached the mate
            for(auto x = begin; x != end; ++x){
                for(auto y = std::next(x); y != end; ++y){
                    const int xl = x->extendedRead.length();
                    const int yl = y->extendedRead.length();

                    if((x->direction == ExtensionDirection::LR && y->direction == ExtensionDirection::RL)
                            || (x->direction == ExtensionDirection::RL && y->direction == ExtensionDirection::LR)){
                        if(xl + yl >= insertSize - insertSizeStddev + minimumOverlap){

                            //put direction LR first
                            if(x->direction == ExtensionDirection::LR){
                                //pairsToCheck.emplace_back(*x, *y);
                                pairPositionsToCheck.emplace_back(std::distance(begin, x), std::distance(begin,y));
                            }else{
                                //pairsToCheck.emplace_back(*y, *x);
                                pairPositionsToCheck.emplace_back(std::distance(begin, y), std::distance(begin,x));
                            }
                        }
                    }
                }
            }

            std::vector<std::string> possibleResults;

            //for(const auto& pair : pairsToCheck){
            for(const auto& pair : pairPositionsToCheck){
                auto iteratorLR = std::next(begin, pair.first);
                auto iteratorRL = std::next(begin, pair.second);

                //const auto& lr = pair.first;
                //const auto& rl = pair.second;

                const auto& lr = *iteratorLR;
                const auto& rl = *iteratorRL;
                assert(lr.direction == ExtensionDirection::LR);
                assert(rl.direction == ExtensionDirection::RL);

                std::string revcRLSeq(rl.extendedRead.begin(), rl.extendedRead.end());
                SequenceHelpers::reverseComplementSequenceDecodedInplace(revcRLSeq.data(), revcRLSeq.size());

                //  std::stringstream sstream;

                //  sstream << to_string(lr.abortReason) << " " << to_string(rl.abortReason) << " - " << lr.readId1 << "\n";
                //  sstream << lr.extendedRead << "\n";
                //  sstream << revcRLSeq << "\n\n";

                // std::cerr << sstream.rdbuf();

                auto strings = func(lr.extendedRead, revcRLSeq);
                possibleResults.insert(possibleResults.end(), std::make_move_iterator(strings.begin()), std::make_move_iterator(strings.end()));
            }

            if(possibleResults.size() > 0){
                
                std::map<std::string, int> histogram;
                for(const auto& r : possibleResults){
                    histogram[r]++;
                }

                //find sequence with highest frequency and return it;
                auto maxIter = std::max_element(
                    histogram.begin(), histogram.end(),
                    [](const auto& p1, const auto& p2){
                        return p1.second < p2.second;
                    }
                );

                // if(histogram.size() >= 1){
                //     std::cerr << "Possible results:\n";

                //     for(const auto& pair : histogram){
                //         std::cerr << pair.second << " : " << pair.first << "\n";
                //     }
                // }

                ExtendResult er;
                er.mateHasBeenFound = true;
                er.success = true;
                er.aborted = false;
                er.numIterations = -1;

                er.direction = ExtensionDirection::LR;
                er.abortReason = AbortReason::None;

                auto iteratorLR = std::next(begin, pairPositionsToCheck[0].first);
                
                //er.readId1 = pairsToCheck[0].first.readId1;
                //er.readId2 = pairsToCheck[0].first.readId2;
                er.readId1 = iteratorLR->readId1;
                er.readId2 = iteratorLR->readId2;
                er.extendedRead = std::move(maxIter->first);

                return er;
            }else{
                //from results which did not find mate, choose longest
                // std::cerr << "Could not merge the following extensions:\n";

                // for(auto it = begin; it != end; ++it){
                //     std::cerr << "id: " << it->readId1;
                //     std::cerr << ", aborted: " << it->aborted;
                //     std::cerr << ", reason: " << to_string(it->abortReason);
                //     std::cerr << ", direction: " << to_string(it->direction) << "\n";
                //     std::cerr << it->extendedRead << "\n";
                // }
                // std::cerr << "\n";
                return *std::max_element(begin, end, lengthcomp);    
            }
        };

        auto combineWithSameIdFoundMate = [&](auto begin, auto end){
            assert(std::distance(begin, end) > 0);

            //return longest read
            return *std::max_element(begin, end, lengthcomp);
        };

        auto combineWithSameId = [&](auto begin, auto end){
            assert(std::distance(begin, end) > 0);

            auto partitionPoint = std::partition(begin, end, [](const auto& x){ return x.mateHasBeenFound;});

            if(std::distance(begin, partitionPoint) > 0){

                return combineWithSameIdFoundMate(begin, partitionPoint);
                //MismatchRatioGlueDecider decider(40, 0.01f);
                // MatchLengthGlueDecider decider(insertSize - insertSizeStddev, 50);
                // WeightedGapGluer gluer(begin->originalLength);
                // auto func = [&](const auto& lr, const auto& rl){
                //     return glue(lr, rl, decider, gluer);
                // };

                // return combineWithSameIdNoMate(begin, partitionPoint, func);

            }else{
#if 1                
                MismatchRatioGlueDecider decider(40, 0.05f);
                //NaiveGluer gluer{};
                WeightedGapGluer gluer(begin->originalLength);
                auto func = [&](const auto& lr, const auto& rl){
                    return glue(lr, rl, decider, gluer);
                };

                return combineWithSameIdNoMate(partitionPoint, end, func);
#else
                //from results which did not find mate, choose longest
                return *std::max_element(partitionPoint, end, lengthcomp);
#endif                
            }
        };

        auto iter1 = combinedResults.begin();
        auto iter2 = combinedResults.begin();
        auto dest = combinedResults.begin();

        while(iter1 != combinedResults.end()){
            while(iter2 != combinedResults.end() && iter1->getReadPairId() == iter2->getReadPairId()){
                ++iter2;
            }

            //elements in range [iter1, iter2) have same read pair id
            *dest = combineWithSameId(iter1, iter2);

            ++dest;
            iter1 = iter2;
        }

        combinedResults.erase(dest, combinedResults.end());

        return combinedResults;
    }


    std::vector<ReadExtenderBase::ExtendResult> ReadExtenderBase::combinePairedEndDirectionResults(
        std::vector<ReadExtenderBase::ExtendResult>& resultsLR,
        std::vector<ReadExtenderBase::ExtendResult>& resultsRL
    ){
        auto idcomp = [](const auto& l, const auto& r){ return l.getReadPairId() < r.getReadPairId();};
        auto lengthcomp = [](const auto& l, const auto& r){ return l.extendedRead.length() < r.extendedRead.length();};

        #if 1

        std::sort(resultsLR.begin(), resultsLR.end(), idcomp);

        std::sort(resultsRL.begin(), resultsRL.end(), idcomp);

        std::vector<ReadExtenderBase::ExtendResult> combinedResults(resultsLR.size() +  resultsRL.size());

        std::merge(
            resultsLR.begin(), resultsLR.end(), 
            resultsRL.begin(), resultsRL.end(), 
            combinedResults.begin(),
            idcomp
        );

        #else
        std::vector<ReadExtenderBase::ExtendResult> combinedResults(resultsLR.size() +  resultsRL.size());
        auto itertmp = std::copy(
            std::make_move_iterator(resultsLR.begin()), std::make_move_iterator(resultsLR.end()), 
            combinedResults.begin()
        );
        std::copy(
            std::make_move_iterator(resultsRL.begin()), std::make_move_iterator(resultsRL.end()), 
            itertmp
        );
        std::sort(
            combinedResults.begin(), 
            combinedResults.end(),
            idcomp
        );
        #endif

        auto glue = [&](const std::string& lrString, const std::string& rlString, const auto& decider, const auto& gluer){
            std::vector<std::string> possibleResults;

            const int maxNumberOfPossibilities = 2*insertSizeStddev + 1;

            for(int p = 0; p < maxNumberOfPossibilities; p++){
                auto decision = decider(
                    lrString, 
                    rlString, 
                    insertSize - insertSizeStddev + p
                );

                if(decision.has_value()){
                    // auto aaa = gluer(*decision);
                    // auto bbb = wgluer(*decision);
                    // if(aaa != bbb){
                    //     std::cerr << "glue\n" << lrString << " and\n" << rlString << "\n";
                    //     std::cerr << aaa << "\n" << bbb << "\n\n";
                    // }
                    possibleResults.emplace_back(gluer(*decision));
                }
            }

            return possibleResults;
        };

        auto combineWithSameIdNoMate = [&](auto begin, auto end, auto func){
            assert(std::distance(begin, end) > 0);

            //TODO optimization: store pairs of indices to results
            //std::vector<std::pair<ReadExtenderBase::ExtendResult, ReadExtenderBase::ExtendResult>> pairsToCheck;
            std::vector<std::pair<int, int>> pairPositionsToCheck;

            constexpr int minimumOverlap = 40;

            //try to find a pair of extensions with opposite directions which could be overlapped to produce an extension which reached the mate
            for(auto x = begin; x != end; ++x){
                for(auto y = std::next(x); y != end; ++y){
                    const int xl = x->extendedRead.length();
                    const int yl = y->extendedRead.length();

                    if((x->direction == ExtensionDirection::LR && y->direction == ExtensionDirection::RL)
                            || (x->direction == ExtensionDirection::RL && y->direction == ExtensionDirection::LR)){
                        if(xl + yl >= insertSize - insertSizeStddev + minimumOverlap){

                            //put direction LR first
                            if(x->direction == ExtensionDirection::LR){
                                //pairsToCheck.emplace_back(*x, *y);
                                pairPositionsToCheck.emplace_back(std::distance(begin, x), std::distance(begin,y));
                            }else{
                                //pairsToCheck.emplace_back(*y, *x);
                                pairPositionsToCheck.emplace_back(std::distance(begin, y), std::distance(begin,x));
                            }
                        }
                    }
                }
            }

            std::vector<std::string> possibleResults;

            //for(const auto& pair : pairsToCheck){
            for(const auto& pair : pairPositionsToCheck){
                auto iteratorLR = std::next(begin, pair.first);
                auto iteratorRL = std::next(begin, pair.second);

                //const auto& lr = pair.first;
                //const auto& rl = pair.second;

                const auto& lr = *iteratorLR;
                const auto& rl = *iteratorRL;
                assert(lr.direction == ExtensionDirection::LR);
                assert(rl.direction == ExtensionDirection::RL);

                std::string revcRLSeq(rl.extendedRead.begin(), rl.extendedRead.end());
                SequenceHelpers::reverseComplementSequenceDecodedInplace(revcRLSeq.data(), revcRLSeq.size());

                //  std::stringstream sstream;

                //  sstream << to_string(lr.abortReason) << " " << to_string(rl.abortReason) << " - " << lr.readId1 << "\n";
                //  sstream << lr.extendedRead << "\n";
                //  sstream << revcRLSeq << "\n\n";

                // std::cerr << sstream.rdbuf();

                auto strings = func(lr.extendedRead, revcRLSeq);
                possibleResults.insert(possibleResults.end(), std::make_move_iterator(strings.begin()), std::make_move_iterator(strings.end()));
            }

            if(possibleResults.size() > 0){
                
                std::map<std::string, int> histogram;
                for(const auto& r : possibleResults){
                    histogram[r]++;
                }

                //find sequence with highest frequency and return it;
                auto maxIter = std::max_element(
                    histogram.begin(), histogram.end(),
                    [](const auto& p1, const auto& p2){
                        return p1.second < p2.second;
                    }
                );

                // if(histogram.size() >= 1){
                //     std::cerr << "Possible results:\n";

                //     for(const auto& pair : histogram){
                //         std::cerr << pair.second << " : " << pair.first << "\n";
                //     }
                // }

                ExtendResult er;
                er.mateHasBeenFound = true;
                er.success = true;
                er.aborted = false;
                er.numIterations = -1;

                er.direction = ExtensionDirection::LR;
                er.abortReason = AbortReason::None;

                auto iteratorLR = std::next(begin, pairPositionsToCheck[0].first);
                
                //er.readId1 = pairsToCheck[0].first.readId1;
                //er.readId2 = pairsToCheck[0].first.readId2;
                er.readId1 = iteratorLR->readId1;
                er.readId2 = iteratorLR->readId2;
                er.extendedRead = std::move(maxIter->first);

                return er;
            }else{
                //from results which did not find mate, choose longest
                // std::cerr << "Could not merge the following extensions:\n";

                // for(auto it = begin; it != end; ++it){
                //     std::cerr << "id: " << it->readId1;
                //     std::cerr << ", aborted: " << it->aborted;
                //     std::cerr << ", reason: " << to_string(it->abortReason);
                //     std::cerr << ", direction: " << to_string(it->direction) << "\n";
                //     std::cerr << it->extendedRead << "\n";
                // }
                // std::cerr << "\n";
                return *std::max_element(begin, end, lengthcomp);    
            }
        };

        auto combineWithSameIdFoundMate = [&](auto begin, auto end){
            assert(std::distance(begin, end) > 0);

            //return longest read
            return *std::max_element(begin, end, lengthcomp);
        };

        auto combineWithSameId = [&](auto begin, auto end){
            assert(std::distance(begin, end) > 0);

            auto partitionPoint = std::partition(begin, end, [](const auto& x){ return x.mateHasBeenFound;});

            if(std::distance(begin, partitionPoint) > 0){

                return combineWithSameIdFoundMate(begin, partitionPoint);
                //MismatchRatioGlueDecider decider(40, 0.01f);
                // MatchLengthGlueDecider decider(insertSize - insertSizeStddev, 50);
                // WeightedGapGluer gluer(begin->originalLength);
                // auto func = [&](const auto& lr, const auto& rl){
                //     return glue(lr, rl, decider, gluer);
                // };

                // return combineWithSameIdNoMate(begin, partitionPoint, func);

            }else{
#if 1                
                MismatchRatioGlueDecider decider(40, 0.05f);
                //NaiveGluer gluer{};
                WeightedGapGluer gluer(begin->originalLength);
                auto func = [&](const auto& lr, const auto& rl){
                    return glue(lr, rl, decider, gluer);
                };

                return combineWithSameIdNoMate(partitionPoint, end, func);
#else
                //from results which did not find mate, choose longest
                return *std::max_element(partitionPoint, end, lengthcomp);
#endif                
            }
        };

        auto iter1 = combinedResults.begin();
        auto iter2 = combinedResults.begin();
        auto dest = combinedResults.begin();

        while(iter1 != combinedResults.end()){
            while(iter2 != combinedResults.end() && iter1->getReadPairId() == iter2->getReadPairId()){
                ++iter2;
            }

            //elements in range [iter1, iter2) have same read pair id
            *dest = combineWithSameId(iter1, iter2);

            ++dest;
            iter1 = iter2;
        }

        combinedResults.erase(dest, combinedResults.end());

        return combinedResults;
    }

    //int batchId = 0;

    std::vector<ReadExtenderBase::ExtendResult> ReadExtenderBase::extendPairedReadBatch(
        const std::vector<ExtendInput>& inputs
    ){

        #if 0
        std::vector<Task> tasks(inputs.size());

        //std::cerr << "Transform LR " << batchId << "\n";
        std::transform(inputs.begin(), inputs.end(), tasks.begin(), 
            [this](const auto& i){return makePairedEndTask(i, ExtensionDirection::LR);});

        //std::cerr << "Process LR " << batchId << "\n";
        std::vector<ExtendResult> extendResultsLR = processPairedEndTasks(std::move(tasks));

        std::vector<Task> tasks2(inputs.size());

        //std::cerr << "Transform RL " << batchId << "\n";
        std::transform(inputs.begin(), inputs.end(), tasks2.begin(), 
            [this](const auto& i){return makePairedEndTask(i, ExtensionDirection::RL);});

        //std::cerr << "Process RL " << batchId << "\n";
        std::vector<ExtendResult> extendResultsRL = processPairedEndTasks(std::move(tasks2));

        //std::cerr << "Combine " << batchId << "\n";
        std::vector<ExtendResult> extendResultsCombined = combinePairedEndDirectionResults(
            extendResultsLR,
            extendResultsRL
        );

        #else
        std::vector<Task> tasks(inputs.size() * 2);

        //std::cerr << "Transform LR " << batchId << "\n";
        auto itertmp = std::transform(inputs.begin(), inputs.end(), tasks.begin(), 
            [this](const auto& i){return makePairedEndTask(i, ExtensionDirection::LR);});

        std::transform(inputs.begin(), inputs.end(), itertmp, 
            [this](const auto& i){return makePairedEndTask(i, ExtensionDirection::RL);});

        //std::cerr << "Process LR " << batchId << "\n";
        std::vector<ExtendResult> extendResults = processPairedEndTasks(std::move(tasks));

        //std::cerr << "Combine " << batchId << "\n";
        std::vector<ExtendResult> extendResultsCombined = combinePairedEndDirectionResults(
            extendResults
        );
        #endif

        //std::cerr << "replace " << batchId << "\n";
        //replace original positions in extend read by original sequences
        // for(std::size_t i = 0; i < inputs.size(); i++){
        //     auto& comb = extendResultsCombined[i];
        //     const auto& input = inputs[i];

        //     if(comb.direction == ExtensionDirection::LR){
        //         decode2BitSequence(
        //             comb.extendedRead.data(),
        //             input.encodedRead1,
        //             input.readLength1
        //         );

        //         if(comb.mateHasBeenFound){
        //             std::vector<char> buf(input.readLength2);
        //             decode2BitSequence(
        //                 buf.data(),
        //                 input.encodedRead2,
        //                 input.readLength2
        //             );
        //             reverseComplementStringInplace(buf.data(), buf.size());
        //             std::copy(
        //                 buf.begin(),
        //                 buf.end(),
        //                 comb.extendedRead.begin() + comb.extendedRead.length() - input.readLength2
        //             );
        //         }
        //     }else{
        //         decode2BitSequence(
        //             comb.extendedRead.data(),
        //             input.encodedRead2,
        //             input.readLength2
        //         );

        //         if(comb.mateHasBeenFound){
        //             std::vector<char> buf(input.readLength1);
        //             decode2BitSequence(
        //                 buf.data(),
        //                 input.encodedRead1,
        //                 input.readLength1
        //             );
        //             reverseComplementStringInplace(buf.data(), buf.size());
        //             std::copy(
        //                 buf.begin(),
        //                 buf.end(),
        //                 comb.extendedRead.begin() + comb.extendedRead.length() - input.readLength1
        //             );
        //         }
        //     }
        // }

        //std::cerr << "done " << batchId << "\n";

        //batchId++;

        return extendResultsCombined;
    }


    /*
        SINGLE END
    */



    std::vector<ReadExtenderBase::ExtendResult> ReadExtenderBase::combineSingleEndDirectionResults(
        std::vector<ReadExtenderBase::ExtendResult>& resultsLR,
        std::vector<ReadExtenderBase::ExtendResult>& resultsRL,
        const std::vector<ReadExtenderBase::Task>& tasks
    ){
        auto idcomp = [](const auto& l, const auto& r){ return l.readId1 < r.readId1;};
        auto lengthcomp = [](const auto& l, const auto& r){ return l.extendedRead.length() < r.extendedRead.length();};

        //for each consecutive range with same readId, keep the longest sequence
        auto keepLongest = [&](auto& vec){
            auto iter1 = vec.begin();
            auto iter2 = vec.begin();
            auto dest = vec.begin();

            while(iter1 != vec.end()){
                while(iter2 != vec.end() && iter1->readId1 == iter2->readId1){
                    ++iter2;
                }

                //range [iter1, iter2) has same read id
                *dest =  *std::max_element(iter1, iter2, lengthcomp);

                ++dest;
                iter1 = iter2;
            }

            return dest;
        };

        std::sort(resultsLR.begin(), resultsLR.end(), idcomp);
        
        auto resultsLR_end = keepLongest(resultsLR);

        std::sort(resultsRL.begin(), resultsRL.end(), idcomp);

        auto resultsRL_end = keepLongest(resultsRL);

        const int remainingLR = std::distance(resultsLR.begin(), resultsLR_end);
        const int remainingRL = std::distance(resultsRL.begin(), resultsRL_end);

        assert(remainingLR == remainingRL);

        std::vector<ReadExtenderBase::ExtendResult> combinedResults(remainingRL);

        for(int i = 0; i < remainingRL; i++){
            auto& comb = combinedResults[i];
            auto& res1 = resultsLR[i];
            auto& res2 = resultsRL[i];
            const auto& task = tasks[i];

            assert(res1.readId1 == res2.readId1);
            assert(task.myReadId == res1.readId1);

            comb.success = true;
            comb.numIterations = res1.numIterations + res2.numIterations;
            comb.readId1 = res1.readId1;
            comb.readId2 = res1.readId2;

            //get reverse complement of RL extension. overlap it with LR extension
            const int newbasesRL = res2.extendedRead.length() - task.myLength;
            if(newbasesRL > 0){
                SequenceHelpers::reverseComplementSequenceDecodedInplace(res2.extendedRead.data() + task.myLength, newbasesRL);
                comb.extendedRead.append(res2.extendedRead.data() + task.myLength, newbasesRL);
            }

            comb.extendedRead.append(res1.extendedRead);

            
        }

        return combinedResults;
    }

    std::vector<ReadExtenderBase::ExtendResult> ReadExtenderBase::extendSingleEndReadBatch(
        const std::vector<ExtendInput>& inputs
    ){

        std::vector<Task> tasks(inputs.size());

        std::transform(inputs.begin(), inputs.end(), tasks.begin(), 
            [this](const auto& i){return makeSingleEndTask(i, ExtensionDirection::LR);});

        std::vector<ExtendResult> extendResultsLR = processSingleEndTasks(tasks);

        std::vector<Task> tasks2(inputs.size());
        std::transform(inputs.begin(), inputs.end(), tasks2.begin(), 
            [this](const auto& i){return makeSingleEndTask(i, ExtensionDirection::RL);});

        //make sure candidates which were used in LR direction cannot be used again in RL direction

        for(std::size_t i = 0; i < inputs.size(); i++){
            tasks2[i].allUsedCandidateReadIdPairs = std::move(tasks[i].allUsedCandidateReadIdPairs);
        }

        std::vector<ExtendResult> extendResultsRL = processSingleEndTasks(std::move(tasks2));

        std::vector<ExtendResult> extendResultsCombined = combineSingleEndDirectionResults(
            extendResultsLR,
            extendResultsRL,
            tasks
        );

        return extendResultsCombined;
    }

}