#include <iostream>
#include <vector>
#include <utility>
#include <cassert>
#include <string>

#include "../../include/sequence.hpp"
#include "../../include/cpu_alignment.hpp"
#include "../../include/bestalignment.hpp"

#ifdef __NVCC__

#include "../../include/gpu/kernels.hpp"
#endif

std::ostream& operator<<(std::ostream& os, const care::cpu::shd::AlignmentResult& a){
    os << "AlignmentResult{"
        << " score: " << a.score
        << ", overlap: " << a.overlap
        << ", shift: " << a.shift
        << ", nOps: " << a.nOps
        << ", isValid: " << a.isValid
        << " }";
    return os;
}

std::ostream& operator<<(std::ostream& os, const care::cpu::BestAlignment_t& flag){
    switch(flag){
        case care::cpu::BestAlignment_t::Forward:
            os << "Forward"; break;
        case care::cpu::BestAlignment_t::ReverseComplement:
            os << "ReverseComplement"; break;
        case care::cpu::BestAlignment_t::None:
            os << "None"; break;
        default: os << "This should not happen\n";
    }
    return os;
}

template<class Sequence_t>
std::tuple<care::cpu::shd::AlignmentResult, care::cpu::shd::AlignmentResult, care::cpu::BestAlignment_t>
align(const std::string& s1, const std::string& s2){
    const int min_overlap = 30;
    const float maxErrorRate = 0.2;
    const float min_overlap_ratio = 0.3;

    Sequence_t subject(s1);
    Sequence_t candidate(s2);
    Sequence_t reverse_complement_candidate = candidate.reverseComplement();

    const int subjectLength = subject.length();
    const int candidateLength = candidate.length();

    care::cpu::shd::AlignmentResult forwardAlignment =
        care::cpu::CPUShiftedHammingDistanceChooser<Sequence_t>::cpu_shifted_hamming_distance((char*)subject.begin(),
                                            subjectLength,
                                            (char*)candidate.begin(),
                                            candidateLength,
                                            min_overlap,
                                            maxErrorRate,
                                            min_overlap_ratio);

    care::cpu::shd::AlignmentResult reverseComplementAlignment =
        care::cpu::CPUShiftedHammingDistanceChooser<Sequence_t>::cpu_shifted_hamming_distance((char*)subject.begin(),
                                            subjectLength,
                                            (char*)reverse_complement_candidate.begin(),
                                            candidateLength,
                                            min_overlap,
                                            maxErrorRate,
                                            min_overlap_ratio);

    care::cpu::BestAlignment_t bestAlignmentFlag = care::cpu::choose_best_alignment(forwardAlignment,
                                                                  reverseComplementAlignment,
                                                                  subjectLength,
                                                                  candidateLength,
                                                                  min_overlap_ratio,
                                                                  min_overlap,
                                                                  maxErrorRate);

    return {forwardAlignment, reverseComplementAlignment, bestAlignmentFlag};
}

struct TestResult{
    std::vector<care::cpu::shd::AlignmentResult> allfwdAlignmentResults;
    std::vector<care::cpu::shd::AlignmentResult> allrevcAlignmentResults;
    std::vector<care::cpu::BestAlignment_t> allbestAlignmentFlags;

    bool operator==(const TestResult& rhs) const{
        return allfwdAlignmentResults == rhs.allfwdAlignmentResults
                && allrevcAlignmentResults == rhs.allrevcAlignmentResults
                && allbestAlignmentFlags == rhs.allbestAlignmentFlags;
    }

    bool operator!=(const TestResult& rhs) const{
        return !(operator==(rhs));
    }
};

template<class Sequence_t>
TestResult getResults(bool checkValidity = true){
    using care::cpu::shd::AlignmentResult;

    const std::vector<std::string> s1{
        "ATCGATCGACTATCATCATCGACTGAGCAGCGATCTACGAGCACTATCGACGACTCG",
        "AGCNGTTAACCGGGTCAGGATGCCGATAATGCCGATGGGAATGGCAATCCATAGCTGCCACGGACGAAAGCGCCGGAATACGGACAGATTTAGGATGCAGC",
        "AGCNGTTAACCGGGTCAGGATGCCGATAATGCCGATGGGAATGGCAATCCATAGCTGCCACGGACGAAAGCGCCGGAATACGGACAGATTTAGGATGCAGC",
        "AGCNGTTAACCGGGTCAGGATGCCGATAATGCCGATGGGAATGGCAATCCATAGCTGCCACGGACGAAAGCGCCGGAATACGGACAGATTTAGGATGCAGC",
    };

    const std::vector<std::string> s2{
        "ATCGATCGACTATCATCATCGACTGAGCAGCGATCTACGAGCACTATCGACGACTCG",
        "GGCGATTTTTACTCCCATGCTGGCATCTGGCACGGTGAACGTCAGGATGCCGATAATGCCGATGGGAATGGCAATCCATAGCTGCCACGGACGAAAGCGCC",
        "CCGATGGGAATGGCAATCCATAGCTGCCACGGACGAA",
        "TCGATGGGAATGGCAATCCATAGCTGCCACGGACGAA",
    };

    const std::vector<AlignmentResult> expectedFwdResults{
        AlignmentResult{0,57,0,0,true},
        AlignmentResult{66,74,-27,12,true},
        AlignmentResult{64,37,31,0,true},
        AlignmentResult{65,37,31,1,true},
    };

    const std::vector<AlignmentResult> expectedRevcResults{
        AlignmentResult{114,0,-57,0,false},
        AlignmentResult{202,0,-101,0,false},
        AlignmentResult{138,0,-37,0,false},
        AlignmentResult{138,0,-37,0,false},
    };

    assert(s1.size() == s2.size());
    assert(s1.size() == expectedFwdResults.size());
    assert(s1.size() == expectedRevcResults.size());

    TestResult result;

    for(size_t i = 0; i < s1.size(); i++){
        auto tuple = align<Sequence_t>(s1[i], s2[i]);
        if(checkValidity){
            if(std::get<0>(tuple) != expectedFwdResults[i]){
                std::cout << "fwd error " << care::getSequenceType<Sequence_t>() << '\n';
                std::cout << s1[i] << '\n';
                std::cout << s2[i] << '\n';
                std::cout << "Expected " << expectedFwdResults[i] << '\n';
                std::cout << "Got " << std::get<0>(tuple) << '\n';
            }
            if(std::get<1>(tuple) != expectedRevcResults[i]){
                std::cout << "revc error " << care::getSequenceType<Sequence_t>() << '\n';
                std::cout << s1[i] << '\n';
                std::cout << s2[i] << '\n';
                std::cout << "Expected " << expectedRevcResults[i] << '\n';
                std::cout << "Got " << std::get<1>(tuple) << '\n';
            }

            assert(std::get<0>(tuple) == expectedFwdResults[i]);
            assert(std::get<1>(tuple) == expectedRevcResults[i]);
        }
        result.allfwdAlignmentResults.emplace_back(std::get<0>(tuple));
        result.allrevcAlignmentResults.emplace_back(std::get<1>(tuple));
        result.allbestAlignmentFlags.emplace_back(std::get<2>(tuple));
    }


#ifdef __NVCC__

    const int shd_tilesize = 32;
    const int min_overlap = 30;
    const float maxErrorRate = 0.2;
    const float min_overlap_ratio = 0.3;

    std::vector<int> candidates_per_subject;

    for(size_t i = 0; i < s1.size(); i++){
        candidates_per_subject.emplace_back((i+1)* 50);
    }

    std::vector<int> h_candidates_per_subject_prefixsum;
    std::vector<int> h_tiles_per_subject_prefixsum;

    h_candidates_per_subject_prefixsum.resize(1+candidates_per_subject.size());
    h_tiles_per_subject_prefixsum.resize(1+candidates_per_subject.size());
    h_candidates_per_subject_prefixsum[0] = 0;
    h_tiles_per_subject_prefixsum[0] = 0;
    for(size_t i = 0; i < candidates_per_subject.size(); i++){
        h_candidates_per_subject_prefixsum[i+1] = h_candidates_per_subject_prefixsum[i] + candidates_per_subject[i];
        h_tiles_per_subject_prefixsum[i+1] = h_tiles_per_subject_prefixsum[i] + SDIV(candidates_per_subject[i], shd_tilesize);
    }

    int* d_candidates_per_subject_prefixsum;
    int* d_tiles_per_subject_prefixsum;

    cudaMalloc(&d_candidates_per_subject_prefixsum, sizeof(int) * (1+candidates_per_subject.size())); CUERR;
    cudaMalloc(&d_tiles_per_subject_prefixsum, sizeof(int) * (1+candidates_per_subject.size())); CUERR;

    cudaMemcpy(d_candidates_per_subject_prefixsum,h_candidates_per_subject_prefixsum.data(), sizeof(int) * (1+candidates_per_subject.size()), H2D); CUERR;
    cudaMemcpy(d_tiles_per_subject_prefixsum,h_tiles_per_subject_prefixsum.data(), sizeof(int) * (1+candidates_per_subject.size()), H2D); CUERR;


    std::vector<int> h_alignment_scores;
    std::vector<int> h_alignment_overlaps;
    std::vector<int> h_alignment_shifts;
    std::vector<int> h_alignment_nOps;
    std::vector<char> h_alignment_isValid;

    h_alignment_scores.resize(h_candidates_per_subject_prefixsum.back() * 2);
    h_alignment_overlaps.resize(h_candidates_per_subject_prefixsum.back() * 2);
    h_alignment_shifts.resize(h_candidates_per_subject_prefixsum.back() * 2);
    h_alignment_nOps.resize(h_candidates_per_subject_prefixsum.back() * 2);
    h_alignment_isValid.resize(h_candidates_per_subject_prefixsum.back() * 2);

    int* d_alignment_scores;
    int* d_alignment_overlaps;
    int* d_alignment_shifts;
    int* d_alignment_nOps;
    bool* d_alignment_isValid;

    cudaMalloc(&d_alignment_scores, sizeof(int) * h_candidates_per_subject_prefixsum.back() * 2); CUERR;
    cudaMalloc(&d_alignment_overlaps, sizeof(int) * h_candidates_per_subject_prefixsum.back() * 2); CUERR;
    cudaMalloc(&d_alignment_shifts, sizeof(int) * h_candidates_per_subject_prefixsum.back() * 2); CUERR;
    cudaMalloc(&d_alignment_nOps, sizeof(int) * h_candidates_per_subject_prefixsum.back() * 2); CUERR;
    cudaMalloc(&d_alignment_isValid, sizeof(bool) * h_candidates_per_subject_prefixsum.back() * 2); CUERR;





    int max_sequence_bytes = 0;
    for(size_t i = 0; i < s1.size(); i++){
        max_sequence_bytes = std::max(max_sequence_bytes, care::Sequence2BitHiLo::getNumBytes(int(s1[i].size())));
    }

    std::vector<char> subjectData(s1.size() * max_sequence_bytes, 0);
    std::vector<char> candidateData(max_sequence_bytes * h_candidates_per_subject_prefixsum.back(), 0);
    std::vector<int> subjectLengths(s1.size(), 0);
    std::vector<int> candidateLengths(h_candidates_per_subject_prefixsum.back(), 0);

    for(size_t i = 0; i < s1.size(); i++){
        care::Sequence2BitHiLo seq(s1[i]);
        std::copy(seq.begin(), seq.end(), &subjectData[i * max_sequence_bytes]);
        subjectLengths[i] = int(s1[i].size());
    }

    for(size_t i = 0; i < s2.size(); i++){
        care::Sequence2BitHiLo seq(s2[i]);
        int begin = h_candidates_per_subject_prefixsum[i];
        int count = candidates_per_subject[i];
        for(int k = begin; k < begin+count; k++){
            std::copy(seq.begin(), seq.end(), &candidateData[k * max_sequence_bytes]);
            candidateLengths[k] = int(s2[i].size());
        }
    }

    char* d_subject_data; cudaMalloc(&d_subject_data, sizeof(char) * subjectData.size()); CUERR;
    char* d_candidate_data; cudaMalloc(&d_candidate_data, sizeof(char) * candidateData.size()); CUERR;
    int* d_subject_lengths; cudaMalloc(&d_subject_lengths, sizeof(int) * subjectLengths.size()); CUERR;
    int* d_candidate_lengths; cudaMalloc(&d_candidate_lengths, sizeof(int) * candidateLengths.size()); CUERR;

    cudaMemcpy(d_subject_data, subjectData.data(), sizeof(char) * subjectData.size(), H2D); CUERR;
    cudaMemcpy(d_candidate_data, candidateData.data(), sizeof(char) * candidateData.size(), H2D); CUERR;
    cudaMemcpy(d_subject_lengths, subjectLengths.data(), sizeof(int) * subjectLengths.size(), H2D); CUERR;
    cudaMemcpy(d_candidate_lengths, candidateLengths.data(), sizeof(int) * candidateLengths.size(), H2D); CUERR;

    auto getNumBytes = [] __device__ (int sequencelength){
        return care::Sequence2BitHiLo::getNumBytes(sequencelength);
    };

    auto getSubjectPtr_dense = [=] __device__ (int subjectIndex){
        const char* result = d_subject_data + std::size_t(subjectIndex) * max_sequence_bytes;
        return result;
    };

    auto getCandidatePtr_dense = [=] __device__ (int candidateIndex){
        const char* result = d_candidate_data + std::size_t(candidateIndex) * max_sequence_bytes;
        return result;
    };

    auto getSubjectLength_dense = [=] __device__ (int subjectIndex){
        const int length = d_subject_lengths[subjectIndex];
        return length;
    };

    auto getCandidateLength_dense = [=] __device__ (int candidateIndex){
        const int length = d_candidate_lengths[candidateIndex];
        return length;
    };

    auto kernellaunchhandle = care::gpu::make_kernel_launch_handle(0);


    call_cuda_popcount_shifted_hamming_distance_with_revcompl_tiled_kernel_async(
			d_alignment_scores,
			d_alignment_overlaps,
			d_alignment_shifts,
			d_alignment_nOps,
			d_alignment_isValid,
			d_candidates_per_subject_prefixsum,
            h_tiles_per_subject_prefixsum.data(),
            d_tiles_per_subject_prefixsum,
            shd_tilesize,
			int(s1.size()),
			h_candidates_per_subject_prefixsum.back(),
			max_sequence_bytes,
			min_overlap,
			maxErrorRate,
			min_overlap_ratio,
			getNumBytes,
			getSubjectPtr_dense,
			getCandidatePtr_dense,
			getSubjectLength_dense,
			getCandidateLength_dense,
			0,
			kernellaunchhandle);

        cudaDeviceSynchronize(); CUERR;

        cudaMemcpy(h_alignment_scores.data(), d_alignment_scores, sizeof(int) * h_candidates_per_subject_prefixsum.back() * 2, D2H);
        cudaMemcpy(h_alignment_overlaps.data(), d_alignment_overlaps, sizeof(int) * h_candidates_per_subject_prefixsum.back() * 2, D2H);
        cudaMemcpy(h_alignment_shifts.data(), d_alignment_shifts, sizeof(int) * h_candidates_per_subject_prefixsum.back() * 2, D2H);
        cudaMemcpy(h_alignment_nOps.data(), d_alignment_nOps, sizeof(int) * h_candidates_per_subject_prefixsum.back() * 2, D2H);
        cudaMemcpy(h_alignment_isValid.data(), d_alignment_isValid, sizeof(bool) * h_candidates_per_subject_prefixsum.back() * 2, D2H);

        cudaDeviceSynchronize(); CUERR;

        if(checkValidity){

            for(size_t i = 0; i < s1.size(); i++){
                int begin = h_candidates_per_subject_prefixsum[i];
                int count = candidates_per_subject[i];

                for(int j = begin; j < begin+count; j++){
                    int rcindex = j + h_candidates_per_subject_prefixsum.back();

                    AlignmentResult fwd{h_alignment_scores[j], h_alignment_overlaps[j],
                                    h_alignment_shifts[j], h_alignment_nOps[j], h_alignment_isValid[j]};
                    AlignmentResult revc{h_alignment_scores[rcindex], h_alignment_overlaps[rcindex],
                                        h_alignment_shifts[rcindex], h_alignment_nOps[rcindex], h_alignment_isValid[rcindex]};

                    if(fwd != expectedFwdResults[i]){
                        std::cout << "fwd gpu error \n";
                        std::cout << s1[i] << '\n';
                        std::cout << s2[i] << '\n';
                        std::cout << "Expected " << expectedFwdResults[i] << '\n';
                        std::cout << "Got " << fwd << '\n';
                    }
                    if(revc != expectedRevcResults[i]){
                        std::cout << "revc gpu error\n";
                        std::cout << s1[i] << '\n';
                        std::cout << s2[i] << '\n';
                        std::cout << "Expected " << expectedRevcResults[i] << '\n';
                        std::cout << "Got " << revc << '\n';
                    }

                    assert(fwd == expectedFwdResults[i]);
                    assert(revc == expectedRevcResults[i]);
                }
            }

        }

        cudaDeviceReset(); CUERR;
#endif


    return result;
}

int main(){

    auto testresult1 = getResults<care::Sequence2BitHiLo>();
    auto testresult2 = getResults<care::Sequence2Bit>();
    auto testresult3 = getResults<care::SequenceString>();

    assert(testresult1 == testresult2);
    assert(testresult1 == testresult3);
    assert(testresult2 == testresult3);

    for(size_t i = 0; i < testresult1.allfwdAlignmentResults.size(); i++){
        const auto& fwdResult = testresult1.allfwdAlignmentResults[i];
        const auto& revcResult = testresult1.allrevcAlignmentResults[i];
        const auto& flag = testresult1.allbestAlignmentFlags[i];

        std::cout << fwdResult << " - " << revcResult << " - " << flag << std::endl;
    }
}
