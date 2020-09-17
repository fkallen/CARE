#include <msa.hpp>
#include <hostdevicefunctions.cuh>

#include <qualityscoreweights.hpp>
#include <bestalignment.hpp>

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



void MultipleSequenceAlignment::build(const InputData& args){

    assert(args.subjectLength > 0);
    assert(args.subject != nullptr);

    inputData = args;

    nCandidates = args.nCandidates;

    //determine number of columns in pileup image
    int startindex = 0;
    int endindex = args.subjectLength;

    for(int i = 0; i < nCandidates; ++i){
        const int shift = args.candidateShifts[i];
        const int candidateEndsAt = args.candidateLengths[i] + shift;
        startindex = std::min(shift, startindex);
        endindex = std::max(candidateEndsAt, endindex);
    }

    nColumns = endindex - startindex;

    subjectColumnsBegin_incl = std::max(-startindex,0);
    subjectColumnsEnd_excl = subjectColumnsBegin_incl + args.subjectLength;

    resize(nColumns);

    fillzero();

    addSequence(args.useQualityScores, args.subject, args.subjectQualities, args.subjectLength, 0, 1.0f);

    for(int candidateIndex = 0; candidateIndex < nCandidates; candidateIndex++){
        const char* ptr = args.candidates + candidateIndex * args.candidatesPitch;
        const char* qptr = args.candidateQualities + candidateIndex * args.candidateQualitiesPitch;
        const int candidateLength = args.candidateLengths[candidateIndex];
        const int shift = args.candidateShifts[candidateIndex];
        const float defaultWeightFactor = args.candidateDefaultWeightFactors[candidateIndex];

        addSequence(args.useQualityScores, ptr, qptr, candidateLength, shift, defaultWeightFactor);
    }

    findConsensus();

    findOrigWeightAndCoverage(args.subject);
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

/*
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
*/

void MultipleSequenceAlignment::findConsensus(){
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
        assert(!std::isnan(support[column]));
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


void MultipleSequenceAlignment::print(std::ostream& os) const{
    std::vector<int> indices(nCandidates+1);
    std::iota(indices.begin(), indices.end(), 0);

    auto get_shift_of_row = [&](int k){
        if(k == 0) return 0;
        else return inputData.candidateShifts[k-1];
    };

    std::sort(indices.begin(), indices.end(),
              [&](int l, int r){return get_shift_of_row(l) < get_shift_of_row(r);});

    for(int row = 0; row < nCandidates+1; row++) {
        int sortedrow = indices[row];

        if(sortedrow == 0){
            os << ">> ";

            for(int i = 0; i < subjectColumnsBegin_incl; i++){
                os << "0";
            }

            for(int i = 0; i < inputData.subjectLength; i++){
                os << inputData.subject[i];
            }

            for(int i = subjectColumnsEnd_excl; i < nColumns; i++){
                os << "0";
            }

            os << " <<";
        }else{
            os << "   ";
            int written = 0;
            for(int i = 0; i < subjectColumnsBegin_incl + get_shift_of_row(sortedrow); i++){
                os << "0";
                written++;
            }

            for(int i = 0; i < inputData.candidateLengths[sortedrow-1]; i++){
                os << inputData.candidates[(sortedrow-1) * inputData.candidatesPitch + i];
                written++;
            }

            for(int i = subjectColumnsBegin_incl + get_shift_of_row(sortedrow) 
                        + inputData.candidateLengths[sortedrow-1]; 
                    i < nColumns; i++){
                os << "0";
                written++;
            }

            assert(written == nColumns);

            os << "   " << inputData.candidateLengths[sortedrow-1] << " " << get_shift_of_row(sortedrow);
        }

        os << '\n';
    }
}


void MultipleSequenceAlignment::printWithDiffToConsensus(std::ostream& os) const{
    std::vector<int> indices(nCandidates+1);
    std::iota(indices.begin(), indices.end(), 0);

    auto get_shift_of_row = [&](int k){
        if(k == 0) return 0;
        else return inputData.candidateShifts[k-1];
    };

    std::sort(indices.begin(), indices.end(),
              [&](int l, int r){return get_shift_of_row(l) < get_shift_of_row(r);});

    for(int row = 0; row < nCandidates+1; row++) {
        int sortedrow = indices[row];

        if(sortedrow == 0){
            os << ">> ";

            for(int i = 0; i < subjectColumnsBegin_incl; i++){
                os << "0";
            }

            for(int i = 0; i < inputData.subjectLength; i++){
                const int globalIndex = subjectColumnsBegin_incl + i;
                const char c = consensus[globalIndex] == inputData.subject[i] ? '=' : inputData.subject[i];
                os << c;
            }

            for(int i = subjectColumnsEnd_excl; i < nColumns; i++){
                os << "0";
            }

            os << " <<";
        }else{
            os << "   ";
            int written = 0;
            for(int i = 0; i < subjectColumnsBegin_incl + get_shift_of_row(sortedrow); i++){
                os << "0";
                written++;
            }

            for(int i = 0; i < inputData.candidateLengths[sortedrow-1]; i++){
                const int globalIndex = subjectColumnsBegin_incl + get_shift_of_row(sortedrow) + i;
                const char base = inputData.candidates[(sortedrow-1) * inputData.candidatesPitch + i];
                const char c = consensus[globalIndex] == base ? '=' : base;

                os << c;
                written++;
            }

            for(int i = subjectColumnsBegin_incl + get_shift_of_row(sortedrow) 
                        + inputData.candidateLengths[sortedrow-1]; 
                    i < nColumns; i++){
                os << "0";
                written++;
            }

            assert(written == nColumns);

            os << "   " << inputData.candidateLengths[sortedrow-1] << " " << get_shift_of_row(sortedrow);
        }

        os << '\n';
    }
}





MSAProperties getMSAProperties(const float* support,
                            const int* coverage,
                            int nColumns,
                            float estimatedErrorrate,
                            float estimatedCoverage,
                            float m_coverage){

    return getMSAProperties2(support,
                            coverage,
                            0,
                            nColumns,
                            estimatedErrorrate,
                            estimatedCoverage,
                            m_coverage);
}

MSAProperties getMSAProperties2(const float* support,
                            const int* coverage,
                            int firstCol,
                            int lastCol, //exclusive
                            float estimatedErrorrate,
                            float estimatedCoverage,
                            float m_coverage){

    assert(firstCol <= lastCol);

    const float avg_support_threshold = 1.0f-1.0f*estimatedErrorrate;
    const float min_support_threshold = 1.0f-3.0f*estimatedErrorrate;
    const float min_coverage_threshold = m_coverage / 6.0f * estimatedCoverage;

    //const int firstCol = subjectColumnsBegin_incl; //0;
    //const int lastCol = subjectColumnsEnd_excl; //nColumns; //exclusive
    const int distance = lastCol - firstCol;

    MSAProperties msaProperties;

    msaProperties.min_support = *std::min_element(support + firstCol, support + lastCol);

    const float supportsum = std::accumulate(support + firstCol, support + lastCol, 0.0f);
    msaProperties.avg_support = supportsum / distance;

    auto minmax = std::minmax_element(coverage + firstCol, coverage + lastCol);

    msaProperties.min_coverage = *minmax.first;
    msaProperties.max_coverage = *minmax.second;

    auto isGoodAvgSupport = [=](float avgsupport){
        return fgeq(avgsupport, avg_support_threshold);
    };
    auto isGoodMinSupport = [=](float minsupport){
        return fgeq(minsupport, min_support_threshold);
    };
    auto isGoodMinCoverage = [=](float mincoverage){
        return fgeq(mincoverage, min_coverage_threshold);
    };

    msaProperties.failedAvgSupport = !isGoodAvgSupport(msaProperties.avg_support);
    msaProperties.failedMinSupport = !isGoodMinSupport(msaProperties.min_support);
    msaProperties.failedMinCoverage = !isGoodMinCoverage(msaProperties.min_coverage);


    const float avg_support = msaProperties.avg_support;
    const float min_support = msaProperties.min_support;
    const int min_coverage = msaProperties.min_coverage;

    msaProperties.isHQ = false;

    const bool allGood = isGoodAvgSupport(avg_support) 
                                        && isGoodMinSupport(min_support) 
                                        && isGoodMinCoverage(min_coverage);
    if(allGood){
        int smallestErrorrateThatWouldMakeHQ = 100;

        const int estimatedErrorratePercent = ceil(estimatedErrorrate * 100.0f);
        for(int percent = estimatedErrorratePercent; percent >= 0; percent--){
            const float factor = percent / 100.0f;
            const float avg_threshold = 1.0f - 1.0f * factor;
            const float min_threshold = 1.0f - 3.0f * factor;
            if(fgeq(avg_support, avg_threshold) && fgeq(min_support, min_threshold)){
                smallestErrorrateThatWouldMakeHQ = percent;
            }
        }

        msaProperties.isHQ = isGoodMinCoverage(min_coverage)
                            && fleq(smallestErrorrateThatWouldMakeHQ, estimatedErrorratePercent * 0.5f);
    }

    return msaProperties;
}


CorrectionResult getCorrectedSubject(const char* consensus,
                                    const float* support,
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
    result.uncorrectedPositionsNoConsensus.reserve(nColumns);

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
                        neighborregioncoverageisgood &= (fgeq(coverage[neighborcolumn], min_coverage_threshold));
                        c++;
                    }
                }

                avgsupportkregion /= c;
                if(neighborregioncoverageisgood && fgeq(avgsupportkregion, 1.0f-estimatedErrorrate)){
                    result.correctedSequence[column] = consensus[column];
                    foundAColumn = true;
                }else{
                    if(subject[column] != consensus[column]){
                        result.uncorrectedPositionsNoConsensus.emplace_back(column);
                    }
                }
            }else{
                if(subject[column] != consensus[column]){
                    result.uncorrectedPositionsNoConsensus.emplace_back(column);
                }
            }
        }

        result.isCorrected = foundAColumn;
    }

    return result;
}

#if 0
CorrectionResult getCorrectedSubject(const char* consensus,
                                    const float* support,
                                    const int* coverage,
                                    const int* originalCoverage,
                                    int nColumns,
                                    const char* subject,
                                    int subjectColumnsBegin_incl,
                                    const char* candidates,
                                    int nCandidates,
                                    const float* candidateAlignmentWeights,
                                    const int* candidateLengths,
                                    const int* candidateShifts,
                                    size_t candidatesPitch,
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
    result.uncorrectedPositionsNoConsensus.reserve(nColumns);
    result.bestAlignmentWeightOfConsensusBase.resize(nColumns);
    result.bestAlignmentWeightOfAnchorBase.resize(nColumns);

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
            const int origCoverage = originalCoverage[column];
            const char origBase = subject[column];
            const char cons = consensus[column];

            const int globalIndex = subjectColumnsBegin_incl + column;

            if(origBase != cons
                    && support[column] > 0.5f
                    //&& origCoverage <= 7){
                ){
                bool canCorrect = true;
                if(canCorrect && origCoverage > 1){
                    int numFoundCandidates = 0;

                    for(int candidatenr = 0; candidatenr < nCandidates/* && numFoundCandidates < origCoverage*/; candidatenr++){

                        const char* candidateptr = candidates + candidatenr * candidatesPitch;
                        const int candidateLength = candidateLengths[candidatenr];
                        const int candidateShift = candidateShifts[candidatenr];
                        const int candidateBasePosition = globalIndex - (subjectColumnsBegin_incl + candidateShift);
                        if(candidateBasePosition >= 0 && candidateBasePosition < candidateLength){
                            //char candidateBase = 'F';

                            //if(bestAlignmentFlags[candidatenr] == cpu::BestAlignment_t::ReverseComplement){
                            //    candidateBase = candidateptr[candidateLength - candidateBasePosition-1];
                            //}else{
                            const char candidateBase = candidateptr[candidateBasePosition];
                            //}

                            const float overlapweight = candidateAlignmentWeights[candidatenr];
                            assert(overlapweight <= 1.0f);
                            assert(overlapweight >= 0.0f);

                            if(origBase == candidateBase){
                                numFoundCandidates++;

                                if(fgeq(overlapweight, 0.90f)){
                                    canCorrect = false;
                                    //break;
                                }
                            }else{
                                ;
                            }
                        }
                    }
                    assert(numFoundCandidates+1 == origCoverage);

                }

#if 1
                float maxweightOrig = 0;
                float maxweightCons = 0;

                for(int candidatenr = 0; candidatenr < nCandidates/* && numFoundCandidates < origCoverage*/; candidatenr++){

                    const char* candidateptr = candidates + candidatenr * candidatesPitch;
                    const int candidateLength = candidateLengths[candidatenr];
                    const int candidateShift = candidateShifts[candidatenr];
                    const int candidateBasePosition = globalIndex - (subjectColumnsBegin_incl + candidateShift);
                    if(candidateBasePosition >= 0 && candidateBasePosition < candidateLength){
                        //char candidateBase = 'F';

                        //if(bestAlignmentFlags[candidatenr] == cpu::BestAlignment_t::ReverseComplement){
                        //    candidateBase = candidateptr[candidateLength - candidateBasePosition-1];
                        //}else{
                        const char candidateBase = candidateptr[candidateBasePosition];
                        //}

                        const float overlapweight = candidateAlignmentWeights[candidatenr];
                        assert(overlapweight <= 1.0f);
                        assert(overlapweight > 0.0f);

                        if(origBase == candidateBase){
                            maxweightOrig = std::max(maxweightOrig, overlapweight);
                        }else{
                            if(cons == candidateBase){
                                maxweightCons = std::max(maxweightCons, overlapweight);
                            }
                        }
                    }
                }

                result.bestAlignmentWeightOfConsensusBase[column] = maxweightCons;
                result.bestAlignmentWeightOfAnchorBase[column] = maxweightOrig;
#endif

                if(canCorrect){

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
                    if(neighborregioncoverageisgood && fgeq(avgsupportkregion, 1.0f-4*estimatedErrorrate)){
                        result.correctedSequence[column] = consensus[column];
                        foundAColumn = true;
                    }
                }else{
                    result.uncorrectedPositionsNoConsensus.emplace_back(column);
                }
            }
        }

        result.isCorrected = foundAColumn;
    }

    return result;
}



#else 


CorrectionResult getCorrectedSubjectNew(const char* consensus,
                                    const float* support,
                                    const int* coverage,
                                    const int* originalCoverage,
                                    int nColumns,
                                    const char* subject,
                                    int subjectColumnsBegin_incl,
                                    const char* candidates,
                                    int nCandidates,
                                    const float* candidateAlignmentWeights,
                                    const int* candidateLengths,
                                    const int* candidateShifts,
                                    size_t candidatesPitch,
                                    MSAProperties msaProperties,
                                    float estimatedErrorrate,
                                    float estimatedCoverage,
                                    float m_coverage,
                                    int neighborRegionSize,
                                    read_number readId){

    if(nCandidates == 0){
        //cannot be corrected without candidates

        CorrectionResult result{};
        result.isCorrected = false;
        return result;
    }

    const float avg_support_threshold = 1.0f-1.0f*estimatedErrorrate;
    const float min_support_threshold = 1.0f-3.0f*estimatedErrorrate;
    const float min_coverage_threshold = m_coverage / 6.0f * estimatedCoverage;

    auto isGoodAvgSupport = [=](float avgsupport){
        return fgeq(avgsupport, avg_support_threshold);
    };
    auto isGoodMinSupport = [=](float minsupport){
        return fgeq(minsupport, min_support_threshold);
    };
    auto isGoodMinCoverage = [=](float mincoverage){
        return fgeq(mincoverage, min_coverage_threshold);
    };

    const float avg_support = msaProperties.avg_support;
    const float min_support = msaProperties.min_support;
    const int min_coverage = msaProperties.min_coverage;

    CorrectionResult result{};
    result.isCorrected = false;
    result.correctedSequence.resize(nColumns);
    result.uncorrectedPositionsNoConsensus.reserve(nColumns);
    result.bestAlignmentWeightOfConsensusBase.resize(nColumns);
    result.bestAlignmentWeightOfAnchorBase.resize(nColumns);

    result.isCorrected = true;
    result.isHQ = false;

    const bool canBeCorrectedByConsensus = isGoodAvgSupport(avg_support) 
                                        && isGoodMinSupport(min_support) 
                                        && isGoodMinCoverage(min_coverage);
    int flag = 0;    

    if(canBeCorrectedByConsensus){
        int smallestErrorrateThatWouldMakeHQ = 100;


        const int estimatedErrorratePercent = ceil(estimatedErrorrate * 100.0f);
        for(int percent = estimatedErrorratePercent; percent >= 0; percent--){
            const float factor = percent / 100.0f;
            const float avg_threshold = 1.0f - 1.0f * factor;
            const float min_threshold = 1.0f - 3.0f * factor;
            // if(readId == 134){
            //     printf("avg_support %f, avg_threshold %f, min_support %f, min_threshold %f\n", 
            //         avg_support, avg_threshold, min_support, min_threshold);
            // }
            if(fgeq(avg_support, avg_threshold) && fgeq(min_support, min_threshold)){
                smallestErrorrateThatWouldMakeHQ = percent;
            }
        }

        const bool isHQ = isGoodMinCoverage(min_coverage)
                            && fleq(smallestErrorrateThatWouldMakeHQ, estimatedErrorratePercent * 0.5f);

        // if(readId == 134){
        //     printf("read 134 isHQ %d, min_coverage %d, avg_support %f, min_support %f, smallestErrorrateThatWouldMakeHQ %d, min_coverage_threshold %f\n", 
        //         isHQ, min_coverage, avg_support, min_support, smallestErrorrateThatWouldMakeHQ, min_coverage_threshold);
        // }

        //broadcastbuffer = isHQ;
        result.isHQ = isHQ;

        flag = isHQ ? 2 : 1;
    }

    if(flag > 0){
        std::copy(consensus,
                  consensus + nColumns,
                  result.correctedSequence.begin());
    }else{
        //correct only positions with high support to consensus, else leave position unchanged.
        for(int i = 0; i < nColumns; i += 1){
            //assert(consensus[i] == 'A' || consensus[i] == 'C' || consensus[i] == 'G' || consensus[i] == 'T');
            if(support[i] > 0.90f && originalCoverage[i] <= 2){
                result.correctedSequence[i] = consensus[i];
            }else{
                result.correctedSequence[i] = subject[i];
            }
        }
    }

    return result;
}






#endif



std::vector<CorrectedCandidate> getCorrectedCandidates(const char* consensus,
                                    const float* support,
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

    // const float avg_support_threshold = 1.0f-1.0f*estimatedErrorrate;
    // const float min_support_threshold = 1.0f-3.0f*estimatedErrorrate;
    // const float min_coverage_threshold = m_coverage / 6.0f * estimatedCoverage;

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

            // float newColMinSupport = 1.0f;
            // int newColMinCov = std::numeric_limits<int>::max();

            // //check new columns left of subject
            // for(int columnindex = subjectColumnsBegin_incl - new_columns_to_correct;
            //     columnindex < subjectColumnsBegin_incl;
            //     columnindex++){

            //     assert(columnindex < nColumns);

            //     if(queryColumnsBegin_incl <= columnindex){
            //         newColMinSupport = support[columnindex] < newColMinSupport ? support[columnindex] : newColMinSupport;
            //         newColMinCov = coverage[columnindex] < newColMinCov ? coverage[columnindex] : newColMinCov;
            //     }
            // }
            // //check new columns right of subject
            // for(int columnindex = subjectColumnsEnd_excl;
            //     columnindex < subjectColumnsEnd_excl + new_columns_to_correct
            //     && columnindex < nColumns;
            //     columnindex++){

            //     newColMinSupport = support[columnindex] < newColMinSupport ? support[columnindex] : newColMinSupport;
            //     newColMinCov = coverage[columnindex] < newColMinCov ? coverage[columnindex] : newColMinCov;
            // }

            // if(fgeq(newColMinSupport, min_support_threshold)
            //     && fgeq(newColMinCov, min_coverage_threshold)){

                std::string correctedString(&consensus[queryColumnsBegin_incl], &consensus[queryColumnsEnd_excl]);

                result.emplace_back(candidate_index, candidateShifts[candidate_index], std::move(correctedString));
            //}
        }
    }

    return result;
}




std::vector<CorrectedCandidate> getCorrectedCandidatesNew(const char* consensus,
                                    const float* support,
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

        bool candidateShouldBeCorrected = false;

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

            candidateShouldBeCorrected = fgeq(newColMinSupport, min_support_threshold)
                            && fgeq(newColMinCov, min_coverage_threshold);

            candidateShouldBeCorrected = true;
        }

        if(candidateShouldBeCorrected){
            std::string correctedString(&consensus[queryColumnsBegin_incl], &consensus[queryColumnsEnd_excl]);
            result.emplace_back(candidate_index, candidateShifts[candidate_index], std::move(correctedString));
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
                                                    const int* alignments_nOps,
                                                    const int* alignments_overlaps,
                                                    int subjectColumnsBegin_incl,
                                                    int subjectColumnsEnd_excl,
                                                    const int* candidateShifts,
                                                    int dataset_coverage,
                                                    float desiredAlignmentMaxErrorRate){

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

            //printf("found col %d, baseIndex %d\n", col, foundBaseIndex);
        }
    }



    RegionSelectionResult result;
    result.performedMinimization = foundColumn;
    result.column = col;

    if(foundColumn){

        result.differentRegionCandidate.resize(nCandidates);

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

        auto discard_rows = [&](bool keepMatching){

            std::array<int, 4> seenCounts{0,0,0,0};

            for(int candidateIndex = 0; candidateIndex < nCandidates; candidateIndex++){
                //check if row is affected by column col
                const int row_begin_incl = subjectColumnsBegin_incl + candidateShifts[candidateIndex];
                const int row_end_excl = row_begin_incl + candidateLengths[candidateIndex];
                const bool notAffected = (col < row_begin_incl || row_end_excl <= col);
                const char base = notAffected ? 'F' : candidates[candidateIndex * candidatesPitch + (col - row_begin_incl)];

                /*printf("k %d, candidateIndex %d, row_begin_incl %d, row_end_excl %d, notAffected %d, base %c\n", candidateIndex, candidateIndex,
                row_begin_incl, row_end_excl, notAffected, base);
                for(int i = 0; i < row_end_excl - row_begin_incl; i++){
                    if(i == (col - row_begin_incl))
                        printf("_");
                    printf("%c", candidates[candidateIndex * candidatesPitch + i]);
                    if(i == (col - row_begin_incl))
                        printf("_");
                }
                printf("\n");*/

                if(base == 'A') seenCounts[0]++;
                if(base == 'C') seenCounts[1]++;
                if(base == 'G') seenCounts[2]++;
                if(base == 'T') seenCounts[3]++;

                if(notAffected || (!(keepMatching ^ (base == foundBase)))){
                    result.differentRegionCandidate[candidateIndex] = false;
                }else{
                    result.differentRegionCandidate[candidateIndex] = true;
                }
            }

            if(originalbase == 'A') seenCounts[0]++;
            if(originalbase == 'C') seenCounts[1]++;
            if(originalbase == 'G') seenCounts[2]++;
            if(originalbase == 'T') seenCounts[3]++;

            assert(seenCounts[0] == countsA[col]);
            assert(seenCounts[1] == countsC[col]);
            assert(seenCounts[2] == countsG[col]);
            assert(seenCounts[3] == countsT[col]);


#if 1
            //check that no candidate which should be removed has very good alignment.
            //if there is such a candidate, none of the candidates will be removed.
            bool veryGoodAlignment = false;
            for(int candidateIndex = 0; candidateIndex < nCandidates; candidateIndex++){
                if(result.differentRegionCandidate[candidateIndex]){
                    const int nOps = alignments_nOps[candidateIndex];
                    const int overlapsize = alignments_overlaps[candidateIndex];
                    const float overlapweight = calculateOverlapWeight(
                        subjectLength, 
                        nOps, 
                        overlapsize, 
                        desiredAlignmentMaxErrorRate
                    );
                    assert(overlapweight <= 1.0f);
                    assert(overlapweight >= 0.0f);

                    if(overlapweight >= 0.90f){
                        veryGoodAlignment = true;
                    }
                }
            }

            if(veryGoodAlignment){
                for(int candidateIndex = 0; candidateIndex < nCandidates; candidateIndex++){
                    result.differentRegionCandidate[candidateIndex] = false;
                }
            }
#endif
        };



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




void printSequencesInMSA(std::ostream& out,
                         const char* subject,
                         int subjectLength,
                         const char* candidates,
                         const int* candidateLengths,
                         int nCandidates,
                         const int* candidateShifts,
                         int subjectColumnsBegin_incl,
                         int subjectColumnsEnd_excl,
                         int nColumns,
                         size_t candidatesPitch){

    std::vector<int> indices(nCandidates+1);
    std::iota(indices.begin(), indices.end(), 0);

    auto get_shift_of_row = [&](int k){
        if(k == 0) return 0;
        else return candidateShifts[k-1];
    };

    std::sort(indices.begin(), indices.end(),
              [&](int l, int r){return get_shift_of_row(l) < get_shift_of_row(r);});

    for(int row = 0; row < nCandidates+1; row++) {
        int sortedrow = indices[row];

        if(sortedrow == 0){
            out << ">> ";

            for(int i = 0; i < subjectColumnsBegin_incl; i++){
                std::cout << "0";
            }

            for(int i = 0; i < subjectLength; i++){
                std::cout << subject[i];
            }

            for(int i = subjectColumnsEnd_excl; i < nColumns; i++){
                std::cout << "0";
            }

            out << " <<";
        }else{
            out << "   ";
            int written = 0;
            for(int i = 0; i < subjectColumnsBegin_incl + get_shift_of_row(sortedrow); i++){
                std::cout << "0";
                written++;
            }

            for(int i = 0; i < candidateLengths[sortedrow-1]; i++){
                std::cout << candidates[(sortedrow-1) * candidatesPitch + i];
                written++;
            }

            for(int i = subjectColumnsBegin_incl + get_shift_of_row(sortedrow) + candidateLengths[sortedrow-1]; i < nColumns; i++){
                std::cout << "0";
                written++;
            }

            assert(written == nColumns);

            out << "   " << candidateLengths[sortedrow-1] << " " << get_shift_of_row(sortedrow);
        }

        out << '\n';
    }
}

void printSequencesInMSAConsEq(std::ostream& out,
                         const char* subject,
                         int subjectLength,
                         const char* candidates,
                         const int* candidateLengths,
                         int nCandidates,
                         const int* candidateShifts,
                         const char* consensus,
                         int subjectColumnsBegin_incl,
                         int subjectColumnsEnd_excl,
                         int nColumns,
                         size_t candidatesPitch){

    std::vector<int> indices(nCandidates+1);
    std::iota(indices.begin(), indices.end(), 0);

    auto get_shift_of_row = [&](int k){
        if(k == 0) return 0;
        else return candidateShifts[k-1];
    };

    std::sort(indices.begin(), indices.end(),
              [&](int l, int r){return get_shift_of_row(l) < get_shift_of_row(r);});

    for(int row = 0; row < nCandidates+1; row++) {
        int sortedrow = indices[row];

        if(sortedrow == 0){
            out << ">> ";

            for(int i = 0; i < subjectColumnsBegin_incl; i++){
                std::cout << "0";
            }

            for(int i = 0; i < subjectLength; i++){
                const int globalIndex = subjectColumnsBegin_incl + i;
                const char c = consensus[globalIndex] == subject[i] ? '=' : subject[i];
                std::cout << c;
            }

            for(int i = subjectColumnsEnd_excl; i < nColumns; i++){
                std::cout << "0";
            }

            out << " <<";
        }else{
            out << "   ";

            for(int i = 0; i < subjectColumnsBegin_incl + get_shift_of_row(sortedrow); i++){
                std::cout << "0";
            }

            for(int i = 0; i < candidateLengths[sortedrow-1]; i++){
                const int globalIndex = subjectColumnsBegin_incl + get_shift_of_row(sortedrow) + i;
                const char base = candidates[(sortedrow-1) * candidatesPitch + i];
                const char c = consensus[globalIndex] == base ? '=' : base;
                std::cout << c;
            }

            for(int i = subjectColumnsBegin_incl + get_shift_of_row(sortedrow) + candidateLengths[sortedrow-1]; i < nColumns; i++){
                std::cout << "0";
            }

            out << "   ";
        }

        out << '\n';
    }
}


}
