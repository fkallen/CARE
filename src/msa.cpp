#include <msa2.hpp>

#include <qualityscoreweights.hpp>

namespace care{


cpu::QualityScoreConversion qualityConversion{};


std::pair<int,int> find_good_consensus_region_of_subject(const View<char>& subject,
                                                    const View<char>& consensus,
                                                    const View<int>& shifts,
                                                    const View<int>& candidateLengths){
    const int min_clip = 10;
    constexpr int max_clip = 20;
    constexpr int mismatches_required_for_clipping = 5;

    const int subjectLength = subject.size();

    const int negativeShifts = std::count_if(shifts.begin(), shifts.end(), [](int s){return s < 0;});
    const int positiveShifts = std::count_if(shifts.begin(), shifts.end(), [](int s){return s > 0;});


    int remainingRegionBegin = 0;
    int remainingRegionEnd = subjectLength; //exclusive

    auto getRemainingRegionBegin = [&](){
        //look for mismatches on the left end
        int nMismatches = 0;
        int lastMismatchPos = -1;
        for(int localIndex = 0; localIndex < max_clip && localIndex < subjectLength; localIndex++){
            if(consensus[localIndex] != subject[localIndex]){
                nMismatches++;
                lastMismatchPos = localIndex;
            }
        }
        if(nMismatches >= mismatches_required_for_clipping){
            //clip after position of last mismatch in max_clip region
            return std::min(subjectLength, lastMismatchPos+1);
        }else{
            //everything is fine
            return 0;
        }
    };

    auto getRemainingRegionEnd = [&](){
        //look for mismatches on the right end
        int nMismatches = 0;
        int firstMismatchPos = subjectLength;
        const int begin = std::max(subjectLength - max_clip, 0);

        for(int localIndex = begin; localIndex < max_clip && localIndex < subjectLength; localIndex++){
            if(consensus[localIndex] != subject[localIndex]){
                nMismatches++;
                firstMismatchPos = localIndex;
            }
        }
        if(nMismatches >= mismatches_required_for_clipping){
            //clip after position of last mismatch in max_clip region
            return firstMismatchPos;
        }else{
            //everything is fine
            return subjectLength;
        }
    };

    //every shift is zero
    if(negativeShifts == 0 && positiveShifts == 0){
        //check both ends
        remainingRegionBegin = getRemainingRegionBegin();
        remainingRegionEnd = getRemainingRegionEnd();
    }else{

        if(negativeShifts == 0){
            remainingRegionBegin = 0;
            for(int i = 0; i < shifts.size(); i++){
                if(shifts[i] <= max_clip){
                    remainingRegionBegin = std::max(shifts[i], remainingRegionBegin);
                }
            }
            remainingRegionBegin = std::max(min_clip, remainingRegionBegin);
        }else if(positiveShifts == 0){
            remainingRegionEnd = subjectLength;
            for(int i = 0; i < shifts.size(); i++){
                const int candidateEndsAt = shifts[i] + candidateLengths[i];
                if(candidateEndsAt < subjectLength && candidateEndsAt >= subjectLength-max_clip){
                    remainingRegionEnd = std::min(candidateEndsAt, remainingRegionEnd);
                }
            }
            remainingRegionEnd = std::min(subjectLength - min_clip, remainingRegionEnd);
        }else{
            ;//do nothing
        }
    }

    return {remainingRegionBegin, remainingRegionEnd};
}





template<int dummy=0>
std::pair<int,int> find_good_consensus_region_of_subject2(const View<char>& subject,
                                                    const View<int>& coverage){
    constexpr int max_clip = 10;
    constexpr int coverage_threshold = 4;

    const int subjectLength = subject.size();

    int remainingRegionBegin = 0;
    int remainingRegionEnd = subjectLength; //exclusive

    for(int i = 0; i < std::min(max_clip, subjectLength); i++){
        if(coverage[i] < coverage_threshold){
            remainingRegionBegin = i+1;
        }else{
            break;
        }
    }

    for(int i = subjectLength - 1; i >= std::max(0, subjectLength - max_clip); i--){
        if(coverage[i] < coverage_threshold){
            remainingRegionEnd = i;
        }else{
            break;
        }
    }

    return {remainingRegionBegin, remainingRegionEnd};

}



void MultipleSequenceAlignment::build(const char* subject,
            int subjectLength,
            const char* candidates,
            const int* candidateLengths,
            int nCandidates_,
            const int* candidateShifts,
            const float* candidateDefaultWeightFactors,
            const char* subjectQualities,
            const char* candidateQualities,
            size_t candidatesPitch,
            size_t candidateQualitiesPitch,
            bool useQualityScores){

    assert(subjectLength > 0);
    assert(subject != nullptr);

    nCandidates = nCandidates_;

    //determine number of columns in pileup image
    int startindex = 0;
    int endindex = subjectLength;

    for(int i = 0; i < nCandidates; ++i){
        const int shift = candidateShifts[i];
        const int candidateEndsAt = candidateLengths[i] + shift;
        startindex = std::min(shift, startindex);
        endindex = std::max(candidateEndsAt, endindex);
    }

    nColumns = endindex - startindex;

    subjectColumnsBegin_incl = std::max(-startindex,0);
    subjectColumnsEnd_excl = subjectColumnsBegin_incl + subjectLength;

    resize(nColumns);

    fillzero();

    addSequence(useQualityScores, subject, subjectQualities, subjectLength, 0, 1.0f);

    for(int candidateIndex = 0; candidateIndex < nCandidates; candidateIndex++){
        const char* ptr = candidates + candidateIndex * candidatesPitch;
        const char* qptr = candidateQualities + candidateIndex * candidateQualitiesPitch;
        const int candidateLength = candidateLengths[candidateIndex];
        const int shift = candidateShifts[candidateIndex];
        const float defaultWeightFactor = candidateDefaultWeightFactors[candidateIndex];

        addSequence(useQualityScores, ptr, qptr, candidateLength, shift, defaultWeightFactor);
    }

    find_consensus();

    findOrigWeightAndCoverage(subject);
}

void MultipleSequenceAlignment::resize(int cols){

    consensus.resize(cols);
    support.resize(cols);
    coverage.resize(cols);
    origWeights.resize(cols);
    origCoverages.resize(cols);
    countsA.resize(cols);
    countsC.resize(cols);
    countsG.resize(cols);
    countsT.resize(cols);
    weightsA.resize(cols);
    weightsC.resize(cols);
    weightsG.resize(cols);
    weightsT.resize(cols);
}

void MultipleSequenceAlignment::fillzero(){
    auto zero = [](auto& vec){
        std::fill(vec.begin(), vec.end(), 0);
    };

    zero(consensus);
    zero(support);
    zero(coverage);
    zero(origWeights);
    zero(origCoverages);
    zero(countsA);
    zero(countsC);
    zero(countsG);
    zero(countsT);
    zero(weightsA);
    zero(weightsC);
    zero(weightsG);
    zero(weightsT);
}

void MultipleSequenceAlignment::addSequence(bool useQualityScores, const char* sequence, const char* quality, int length, int shift, float defaultWeightFactor){
    assert(sequence != nullptr);
    assert(!useQualityScores || quality != nullptr);

    for(int i = 0; i < length; i++){
        const int globalIndex = subjectColumnsBegin_incl + shift + i;
        const char base = sequence[i];
        const float weight = defaultWeightFactor * (useQualityScores ? qualityConversion.getWeight(quality[i]) : 1.0f);
        switch(base){
            case 'A': countsA[globalIndex]++; weightsA[globalIndex] += weight;break;
            case 'C': countsC[globalIndex]++; weightsC[globalIndex] += weight;break;
            case 'G': countsG[globalIndex]++; weightsG[globalIndex] += weight;break;
            case 'T': countsT[globalIndex]++; weightsT[globalIndex] += weight;break;
            default: assert(false); break;
        }
        coverage[globalIndex]++;
    }
}

void MultipleSequenceAlignment::removeSequence(bool useQualityScores, const char* sequence, const char* quality, int length, int shift, float defaultWeightFactor){
    assert(sequence != nullptr);
    assert(!useQualityScores || quality != nullptr);

    for(int i = 0; i < length; i++){
        const int globalIndex = subjectColumnsBegin_incl + shift + i;
        const char base = sequence[i];
        const float weight = defaultWeightFactor * (useQualityScores ? qualityConversion.getWeight(quality[i]) : 1.0f);
        switch(base){
            case 'A': countsA[globalIndex]--; weightsA[globalIndex] -= weight;break;
            case 'C': countsC[globalIndex]--; weightsC[globalIndex] -= weight;break;
            case 'G': countsG[globalIndex]--; weightsG[globalIndex] -= weight;break;
            case 'T': countsT[globalIndex]--; weightsT[globalIndex] -= weight;break;
            default: assert(false); break;
        }
        coverage[globalIndex]--;
    }
}

void MultipleSequenceAlignment::find_consensus(){
    for(int column = 0; column < nColumns; ++column){
        char cons = 'A';
        float consWeight = weightsA[column];
        if(weightsC[column] > consWeight){
            cons = 'C';
            consWeight = weightsC[column];
        }
        if(weightsG[column] > consWeight){
            cons = 'G';
            consWeight = weightsG[column];
        }
        if(weightsT[column] > consWeight){
            cons = 'T';
            consWeight = weightsT[column];
        }
        consensus[column] = cons;

        const float columnWeight = weightsA[column] + weightsC[column] + weightsG[column] + weightsT[column];
        support[column] = consWeight / columnWeight;
    }
}

void MultipleSequenceAlignment::findOrigWeightAndCoverage(const char* subject){
    for(int column = subjectColumnsBegin_incl; column < subjectColumnsEnd_excl; ++column){

        const int localIndex = column - subjectColumnsBegin_incl;
        const char subjectBase = subject[localIndex];
        switch(subjectBase){
            case 'A':origWeights[column] = weightsA[column]; origCoverages[column] = countsA[column]; break;
            case 'C':origWeights[column] = weightsG[column]; origCoverages[column] = countsC[column]; break;
            case 'G':origWeights[column] = weightsC[column]; origCoverages[column] = countsG[column]; break;
            case 'T':origWeights[column] = weightsT[column]; origCoverages[column] = countsT[column]; break;
            default: assert(false); break;
        }

    }
}



MSAProperties getMSAProperties(const float* support,
                            const int* coverage,
                            int nColumns,
                            float estimatedErrorrate,
                            float estimatedCoverage,
                            float m_coverage){

    const float avg_support_threshold = 1.0f-1.0f*estimatedErrorrate;
    const float min_support_threshold = 1.0f-3.0f*estimatedErrorrate;
    const float min_coverage_threshold = m_coverage / 6.0f * estimatedCoverage;

    MSAProperties msaProperties;

    msaProperties.min_support = *std::min_element(support, support + nColumns);

    const float supportsum = std::accumulate(support, support + nColumns, 0.0f);
    msaProperties.avg_support = supportsum / nColumns;

    auto minmax = std::minmax_element(coverage, coverage + nColumns);

    msaProperties.min_coverage = *minmax.second;
    msaProperties.max_coverage = *minmax.first;

    auto isGoodAvgSupport = [=](float avgsupport){
        return avgsupport >= avg_support_threshold;
    };
    auto isGoodMinSupport = [=](float minsupport){
        return minsupport >= min_support_threshold;
    };
    auto isGoodMinCoverage = [=](float mincoverage){
        return mincoverage >= min_coverage_threshold;
    };

    msaProperties.isHQ = isGoodAvgSupport(msaProperties.avg_support)
                        && isGoodMinSupport(msaProperties.min_support)
                        && isGoodMinCoverage(msaProperties.min_coverage);

    msaProperties.failedAvgSupport = !isGoodAvgSupport(msaProperties.avg_support);
    msaProperties.failedMinSupport = !isGoodMinSupport(msaProperties.min_support);
    msaProperties.failedMinCoverage = !isGoodMinCoverage(msaProperties.min_coverage);

    return msaProperties;
}


CorrectionResult getCorrectedSubject(const char* consensus,
                                    const int* support,
                                    const int* coverage,
                                    const int* originalCoverage,
                                    int nColumns,
                                    const char* subject,
                                    bool isHQ,
                                    float estimatedErrorrate,
                                    float estimatedCoverage,
                                    float m_coverage,
                                    int neighborRegionSize){

    //const float avg_support_threshold = 1.0f-1.0f*estimatedErrorrate;
    //const float min_support_threshold = 1.0f-3.0f*estimatedErrorrate;
    const float min_coverage_threshold = m_coverage / 6.0f * estimatedCoverage;

    CorrectionResult result;
    result.isCorrected = false;
    result.correctedSequence.resize(nColumns);

    if(isHQ){
        //corrected sequence = consensus;

        std::copy(consensus,
                  consensus + nColumns,
                  result.correctedSequence.begin());
        result.isCorrected = true;
    }else{
        //set corrected sequence to original subject. then search for positions with good properties. correct these positions
        std::copy(subject,
                  subject + nColumns,
                  result.correctedSequence.begin());

        bool foundAColumn = false;
        for(int column = 0; column < nColumns; column++){

            if(support[column] > 0.5f && originalCoverage[column] < min_coverage_threshold){
                float avgsupportkregion = 0;
                int c = 0;
                bool neighborregioncoverageisgood = true;

                for(int neighborcolumn = column - neighborRegionSize/2; neighborcolumn <= column + neighborRegionSize/2 && neighborregioncoverageisgood; neighborcolumn++){
                    if(neighborcolumn != column && neighborcolumn >= 0 && neighborcolumn < nColumns){
                        avgsupportkregion += support[neighborcolumn];
                        neighborregioncoverageisgood &= (coverage[neighborcolumn] >= min_coverage_threshold);
                        c++;
                    }
                }

                avgsupportkregion /= c;
                if(neighborregioncoverageisgood && avgsupportkregion >= 1.0f-estimatedErrorrate){
                    result.correctedSequence[column] = consensus[column];
                    foundAColumn = true;
                }
            }
        }

        result.isCorrected = foundAColumn;
    }

    return result;
}




std::vector<CorrectedCandidate> getCorrectedCandidates(const char* consensus,
                                    const int* support,
                                    const int* coverage,
                                    int nColumns,
                                    int subjectColumnsBegin_incl,
                                    int subjectColumnsEnd_excl,
                                    const int* candidateShifts,
                                    const int* candidateLengths,
                                    int nCandidates,
                                    float estimatedErrorrate,
                                    float estimatedCoverage,
                                    float m_coverage,
                                    int new_columns_to_correct){

    //const float avg_support_threshold = 1.0f-1.0f*estimatedErrorrate;
    const float min_support_threshold = 1.0f-3.0f*estimatedErrorrate;
    const float min_coverage_threshold = m_coverage / 6.0f * estimatedCoverage;

    std::vector<CorrectedCandidate> result;
    result.reserve(nCandidates);

    for(int candidate_index = 0; candidate_index < nCandidates; ++candidate_index){

        const int queryColumnsBegin_incl = subjectColumnsBegin_incl + candidateShifts[candidate_index];
        const int candidateLength = candidateLengths[candidate_index];
        const int queryColumnsEnd_excl = queryColumnsBegin_incl + candidateLength;

        //check range condition and length condition
        if(subjectColumnsBegin_incl - new_columns_to_correct <= queryColumnsBegin_incl
            && queryColumnsBegin_incl <= subjectColumnsBegin_incl + new_columns_to_correct
            && queryColumnsEnd_excl <= subjectColumnsEnd_excl + new_columns_to_correct){

            float newColMinSupport = 1.0f;
            int newColMinCov = std::numeric_limits<int>::max();

            //check new columns left of subject
            for(int columnindex = subjectColumnsBegin_incl - new_columns_to_correct;
                columnindex < subjectColumnsBegin_incl;
                columnindex++){

                assert(columnindex < nColumns);

                if(queryColumnsBegin_incl <= columnindex){
                    newColMinSupport = support[columnindex] < newColMinSupport ? support[columnindex] : newColMinSupport;
                    newColMinCov = coverage[columnindex] < newColMinCov ? coverage[columnindex] : newColMinCov;
                }
            }
            //check new columns right of subject
            for(int columnindex = subjectColumnsEnd_excl;
                columnindex < subjectColumnsEnd_excl + new_columns_to_correct
                && columnindex < nColumns;
                columnindex++){

                newColMinSupport = support[columnindex] < newColMinSupport ? support[columnindex] : newColMinSupport;
                newColMinCov = coverage[columnindex] < newColMinCov ? coverage[columnindex] : newColMinCov;
            }

            if(newColMinSupport >= min_support_threshold
                && newColMinCov >= min_coverage_threshold){

                std::string correctedString(&consensus[queryColumnsBegin_incl], &consensus[queryColumnsEnd_excl]);

                result.emplace_back(candidate_index, std::move(correctedString));
            }
        }
    }

    return result;
}





//remove all candidate reads from alignment which are assumed to originate from a different genomic region
//the indices of remaining candidates are returned in MinimizationResult::remaining_candidates
//candidates in vector must be in the same order as they were inserted into the msa!!!

RegionSelectionResult findCandidatesOfDifferentRegion(const char* subject,
                                                    int subjectLength,
                                                    const char* candidates,
                                                    const int* candidateLengths,
                                                    int nCandidates,
                                                    size_t candidatesPitch,
                                                    const char* consensus,
                                                    const int* countsA,
                                                    const int* countsC,
                                                    const int* countsG,
                                                    const int* countsT,
                                                    const float* weightsA,
                                                    const float* weightsC,
                                                    const float* weightsG,
                                                    const float* weightsT,
                                                    int subjectColumnsBegin_incl,
                                                    int subjectColumnsEnd_excl,
                                                    const int* candidateShifts,
                                                    int dataset_coverage){

    auto is_significant_count = [&](int count, int dataset_coverage){
        if(int(dataset_coverage * 0.3f) <= count)
            return true;
        return false;
    };

    constexpr std::array<char, 4> index_to_base{'A','C','G','T'};

    //find column with a non-consensus base with significant coverage
    int col = 0;
    bool foundColumn = false;
    char foundBase = 'F';
    int foundBaseIndex = 0;
    int consindex = 0;

    //if anchor has no mismatch to consensus, don't minimize
    auto pair = std::mismatch(subject,
                                subject + subjectLength,
                                consensus + subjectColumnsBegin_incl);

    if(pair.first == subject + subjectLength){
        RegionSelectionResult result;
        result.performedMinimization = false;
        return result;
    }

    for(int columnindex = subjectColumnsBegin_incl; columnindex < subjectColumnsEnd_excl && !foundColumn; columnindex++){
        std::array<int,4> counts;
        //std::array<float,4> weights;

        counts[0] = countsA[columnindex];
        counts[1] = countsC[columnindex];
        counts[2] = countsG[columnindex];
        counts[3] = countsT[columnindex];

        /*weights[0] = weightsA[columnindex];
        weights[1] = weightsC[columnindex];
        weights[2] = weightsG[columnindex];
        weights[3] = weightsT[columnindex];*/

        char cons = consensus[columnindex];
        consindex = -1;

        switch(cons){
            case 'A': consindex = 0;break;
            case 'C': consindex = 1;break;
            case 'G': consindex = 2;break;
            case 'T': consindex = 3;break;
        }

        //const char originalbase = subject[columnindex - columnProperties.subjectColumnsBegin_incl];

        //find out if there is a non-consensus base with significant coverage
        int significantBaseIndex = -1;
        //int maxcount = 0;
        for(int i = 0; i < 4; i++){
            if(i != consindex){
                bool significant = is_significant_count(counts[i], dataset_coverage);

                bool process = significant; //maxcount < counts[i] && significant && (cons == originalbase || index_to_base[i] == originalbase);

                significantBaseIndex = process ? i : significantBaseIndex;

                //maxcount = process ? std::max(maxcount, counts[i]) : maxcount;
            }
        }

        if(significantBaseIndex != -1){
            foundColumn = true;
            col = columnindex;
            foundBase = index_to_base[significantBaseIndex];
            foundBaseIndex = significantBaseIndex;
        }
    }



    RegionSelectionResult result;
    result.performedMinimization = foundColumn;
    result.column = col;

    if(foundColumn){

        result.differentRegionCandidate.resize(nCandidates);

        auto discard_rows = [&](bool keepMatching){

            for(int candidateIndex = 0; candidateIndex < nCandidates; candidateIndex++){
                //check if row is affected by column col
                const int row_begin_incl = subjectColumnsBegin_incl + candidateShifts[candidateIndex];
                const int row_end_excl = row_begin_incl + candidateLengths[candidateIndex];
                const bool notAffected = (col < row_begin_incl || row_end_excl <= col);
                const char base = notAffected ? 'F' : candidates[candidateIndex * candidatesPitch + (col - row_begin_incl)];

                if(notAffected || (!(keepMatching ^ (base == foundBase)))){
                    result.differentRegionCandidate[candidateIndex] = false;
                }else{
                    result.differentRegionCandidate[candidateIndex] = true;
                }
            }
        };

        //compare found base to original base
        const char originalbase = subject[col - subjectColumnsBegin_incl];

        result.significantBase = foundBase;
        result.originalBase = originalbase;
        result.consensusBase = consensus[col];

        std::array<int,4> counts;

        counts[0] = countsA[col];
        counts[1] = countsC[col];
        counts[2] = countsG[col];
        counts[3] = countsT[col];

        result.significantCount = counts[foundBaseIndex];
        result.consensuscount = counts[consindex];

        if(originalbase == foundBase){
            //discard all candidates whose base in column col differs from foundBase
            discard_rows(true);
        }else{
            //discard all candidates whose base in column col matches foundBase
            discard_rows(false);
        }

        //if(result.num_discarded_candidates > 0){
        //    find_consensus();
        //}

        return result;
    }else{

        return result;
    }
}


std::pair<int,int> findGoodConsensusRegionOfSubject(const char* subject,
                                                    int subjectLength,
                                                    const char* consensus,
                                                    const int* candidateShifts,
                                                    const int* candidateLengths,
                                                    int nCandidates){

    View<char> subjectview{subject, subjectLength};
    View<char> consensusview{consensus, subjectLength};
    View<int> shiftview{candidateShifts, nCandidates}; //starting at index 1 because index 0 is subject
    const View<int> lengthview{candidateLengths, nCandidates}; //starting at index 1 because index 0 is subject

    auto result = find_good_consensus_region_of_subject(subjectview, consensusview, shiftview, lengthview);

    return result;
}

std::pair<int,int> findGoodConsensusRegionOfSubject2(const char* subject,
                                                    int subjectLength,
                                                    const int* coverage,
                                                    int nColumns,
                                                    int subjectColumnsEnd_excl){

    if(nColumns - subjectColumnsEnd_excl <= 3){
        View<char> subjectview{subject, subjectLength};

        const View<int> coverageview{coverage, subjectLength};

        auto result = find_good_consensus_region_of_subject2(subjectview, coverageview);

        return result;
    }else{
        return std::make_pair(0, subjectLength);
    }
}


}
