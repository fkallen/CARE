#ifndef CARE_CORRECTION_WRAPPER_HPP
#define CARE_CORRECTION_WRAPPER_HPP

#include "graph.hpp"
#include "pileup.hpp"
#include "tasktiming.hpp"

namespace care{

    //correct batchElem using Pileup
    using PileupCorrectionResult = pileup::PileupImage::CorrectionResult;

    template<class BatchElem_t>
    std::pair<PileupCorrectionResult, TaskTimings> correct(pileup::PileupImage& pileup,
                                                            const BatchElem_t& b,
                                                            double desiredAlignmentMaxErrorRate,
                                                            double estimatedErrorrate,
                                                            double estimatedCoverage,
                                                            bool correctCandidates,
                                                            int new_columns_to_correct) noexcept{
            std::pair<PileupCorrectionResult, TaskTimings> result;

            auto& cor = result.first;
            auto& tt = result.second;

            tt.preprocessingBegin();

            pileup.init(b.fwdSequenceString,
                        b.fwdQuality,
                        b.bestAlignments.begin(),
                        b.bestAlignments.end(),
                        b.bestSequenceStrings.begin(),
                        b.bestSequenceStrings.end(),
                        b.bestQualities.begin(),
                        b.bestQualities.end());

            pileup.cpu_add_candidates(b.fwdSequenceString,
                                    b.bestAlignments.begin(),
                                    b.bestAlignments.end(),
                                    desiredAlignmentMaxErrorRate,
                                    b.bestSequenceStrings.begin(),
                                    b.bestSequenceStrings.end(),
                                    b.candidateCounts.begin(),
                                    b.candidateCounts.end(),
                                    b.bestQualities.begin(),
                                    b.bestQualities.end());

            tt.preprocessingEnd();

            tt.executionBegin();

            cor = pileup.cpu_correct(b.fwdSequenceString,
                                    b.bestAlignments.begin(),
                                    b.bestAlignments.end(),
                                    b.bestSequenceStrings.begin(),
                                    b.bestSequenceStrings.end(),
                                    estimatedErrorrate,
                                    estimatedCoverage,
                                    correctCandidates,
                                    new_columns_to_correct);

            tt.executionEnd();

            return result;
    }

    //correct batchElem using Error Graph
    using GraphCorrectionResult = errorgraph::ErrorGraph::CorrectionResult;

    template<class BatchElem_t>
    std::pair<GraphCorrectionResult, TaskTimings> correct(errorgraph::ErrorGraph& graph,
                                                            const BatchElem_t& b,
                                                            double desiredAlignmentMaxErrorRate,
                                                            double alpha,
                                                            double x) noexcept{
            std::pair<GraphCorrectionResult, TaskTimings> result;

            auto& cor = result.first;
            auto& tt = result.second;

            tt.preprocessingBegin();

            graph.init(b.fwdSequenceString, b.fwdQuality);

            graph.add_candidates(b.fwdSequenceString,
                                    b.fwdQuality,
                                    b.bestAlignments.begin(),
                                    b.bestAlignments.end(),
                                    desiredAlignmentMaxErrorRate,
                                    b.bestSequenceStrings.begin(),
                                    b.bestSequenceStrings.end(),
                                    b.candidateCounts.begin(),
                                    b.candidateCounts.end(),
                                    b.bestQualities.begin(),
                                    b.bestQualities.end());

            tt.preprocessingEnd();

            tt.executionBegin();

            cor = graph.extractCorrectedSequence(alpha, x);

            tt.executionEnd();

            return result;
    }



}




#endif
