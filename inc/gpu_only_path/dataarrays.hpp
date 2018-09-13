#ifndef CARE_GPU_DATA_ARRAYS_HPP
#define CARE_GPU_DATA_ARRAYS_HPP

namespace care{
namespace gpu{

    #ifdef __NVCC__

    template<class Sequence_t>
    struct DataArrays{
        static constexpr int padding_bytes = 512;

        void set_problem_dimensions(int n_sub, int n_quer, int max_seq_length, int min_overlap_, double min_overlap_ratio_);
        void set_n_indices(int n_indices_);
        void set_tmp_storage_size(std::size_t newsize);
        void reset();

        DataArrays(){}
        DataArrays(int deviceId) : deviceId(deviceId){

        };


        void set_problem_dimensions(int n_sub, int n_quer, int max_seq_length, int min_overlap_, double min_overlap_ratio_){

            encoded_sequence_pitch = SDIV(Sequence_t::getNumBytes(max_seq_length), padding_bytes) * padding_bytes;
            quality_pitch = SDIV(max_seq_length * sizeof(char), padding_bytes) * padding_bytes;
            sequence_pitch = SDIV(max_seq_length * sizeof(char), padding_bytes) * padding_bytes;

            //alignment input
            std::size_t memSubjects = n_sub * encoded_sequence_pitch;
            std::size_t memSubjectLengths = SDIV(n_sub * sizeof(int), padding_bytes) * padding_bytes;
            std::size_t memNqueriesPrefixSum = SDIV((n_sub+1) * sizeof(int), padding_bytes) * padding_bytes;
            std::size_t memQueries = n_quer * encoded_sequence_pitch;
            std::size_t memQueryLengths = SDIV(n_quer * sizeof(int), padding_bytes) * padding_bytes;

            std::size_t total_alignment_transfer_data_size = memSubjects
                                                            + memSubjectLengths
                                                            + memNqueriesPrefixSum
                                                            + memQueries
                                                            + memQueryLengths;

            if(total_alignment_transfer_data_size > alignment_transfer_data_size){
                cudaFree(alignment_transfer_data_device); CUERR;
                cudaMalloc(&alignment_transfer_data_device, total_alignment_transfer_data_size); CUERR;
                cudaFreeHost(alignment_transfer_data_host); CUERR;
                cudaMallocHost(&alignment_transfer_data_host, total_alignment_transfer_data_size); CUERR;

                alignment_transfer_data_size = total_alignment_transfer_data_size;
            }

            h_subject_sequences_data = (char*)alignment_transfer_data_host;
            h_candidate_sequences_data = (char*)(((char*)h_subject_sequences_data) + memSubjects);
            h_subject_sequences_lengths = (int*)(((char*)h_candidate_sequences_data) + memQueries);
            h_candidate_sequences_lengths = (int*)(((char*)h_subject_sequences_lengths) + memSubjectLengths);
            h_candidates_per_subject_prefixsum = (int*)(((char*)h_candidate_sequences_lengths) + memQueryLengths);

            d_subject_sequences_data = (char*)alignment_transfer_data_device;
            d_candidate_sequences_data = (char*)(((char*)d_subject_sequences_data) + memSubjects);
            d_subject_sequences_lengths = (int*)(((char*)d_candidate_sequences_data) + memQueries);
            d_candidate_sequences_lengths = (int*)(((char*)d_subject_sequences_lengths) + memSubjectLengths);
            d_candidates_per_subject_prefixsum = (int*)(((char*)d_candidate_sequences_lengths) + memQueryLengths);

            //alignment output
            std::size_t memAlignmentScores = SDIV((2*n_quer) * sizeof(int), padding_bytes) * padding_bytes;
            std::size_t memAlignmentOverlaps = SDIV((2*n_quer) * sizeof(int), padding_bytes) * padding_bytes;
            std::size_t memAlignmentShifts = SDIV((2*n_quer) * sizeof(int), padding_bytes) * padding_bytes;
            std::size_t memAlignmentnOps = SDIV((2*n_quer) * sizeof(int), padding_bytes) * padding_bytes;
            std::size_t memAlignmentisValid = SDIV((2*n_quer) * sizeof(bool), padding_bytes) * padding_bytes;
            std::size_t memAlignmentBestAlignmentFlags = SDIV((n_quer) * sizeof(BestAlignment_t), padding_bytes) * padding_bytes;

            std::size_t total_alignment_result_data_size = memAlignmentScores
                                                            + memAlignmentOverlaps
                                                            + memAlignmentShifts
                                                            + memAlignmentnOps
                                                            + memAlignmentisValid
                                                            + memAlignmentBestAlignmentFlags;

            if(total_alignment_result_data_size > alignment_result_data_size){
                cudaFree(alignment_result_data_device); CUERR;
                cudaMalloc(&alignment_result_data_device, total_alignment_result_data_size); CUERR;

                alignment_result_data_size = total_alignment_result_data_size;
            }

            d_alignment_scores = (int*)alignment_transfer_data_device;
            d_alignment_overlaps = (int*)(((char*)d_alignment_scores) + memAlignmentScores);
            d_alignment_shifts = (int*)(((char*)d_alignment_overlaps) + memAlignmentOverlaps);
            d_alignment_nOps = (int*)(((char*)d_alignment_shifts) + memAlignmentShifts);
            d_alignment_isValid = (bool*)(((char*)d_alignment_nOps) + memAlignmentnOps);
            d_alignment_best_alignment_flags = (BestAlignment_t*)(((char*)d_alignment_isValid) + memAlignmentisValid);

            if(n_quer > n_queries){
                cudaFree(d_indices_per_subject); CUERR;
                cudaMalloc(&d_indices_per_subject, sizeof(int) * n_quer); CUERR;
            }

            n_subjects = n_sub;
            n_queries = n_quer;
            maximum_sequence_length = max_seq_length;
            min_overlap = std::max(1, std::max(min_overlap_, int(maximum_sequence_length * min_overlap_ratio_)));
        }

        void set_n_indices(int n_indices_){
            n_indices = n_indices_;

            //indices

            std::size_t memIndices = SDIV(n_indices * sizeof(int), padding_bytes) * padding_bytes;
            std::size_t memIndicesPerSubjectPrefixSum = SDIV((n_subjects+1)* sizeof(int), padding_bytes) * padding_bytes;

            std::size_t total_indices_transfer_data_size = memIndices
                                                            + memIndicesPerSubjectPrefixSum;

            if(total_indices_transfer_data_size > indices_transfer_data_size){
                cudaFree(indices_transfer_data_device); CUERR;
                cudaMalloc(&indices_transfer_data_device, total_indices_transfer_data_size); CUERR;
                cudaFreeHost(indices_transfer_data_host); CUERR;
                cudaMallocHost(&indices_transfer_data_host, total_indices_transfer_data_size); CUERR;

                indices_transfer_data_size = total_indices_transfer_data_size;
            }

            h_indices = (int*)indices_transfer_data_host;
            h_indices_per_subject_prefixsum = (int*)(((char*)h_indices) + memIndices);

            d_indices = (int*)indices_transfer_data_device;
            d_indices_per_subject_prefixsum = (int*)(((char*)d_indices) + memIndices);

            //qualities

            std::size_t memCandidateQualities = n_indices * quality_pitch;
            std::size_t memSubjectQualities = n_subjects * quality_pitch;

            std::size_t total_qualities_transfer_data_size = memIndices
                                                            + memIndicesPerSubjectPrefixSum;

            if(total_qualities_transfer_data_size > qualities_transfer_data_size){
                cudaFree(qualities_transfer_data_device); CUERR;
                cudaMalloc(&qualities_transfer_data_device, total_qualities_transfer_data_size); CUERR;
                cudaFreeHost(qualities_transfer_data_host); CUERR;
                cudaMallocHost(&qualities_transfer_data_host, total_qualities_transfer_data_size); CUERR;

                qualities_transfer_data_size = total_qualities_transfer_data_size;
            }

            h_candidate_qualities = (char*)qualities_transfer_data_host;
            h_subject_qualities = (char*)(((char*)h_candidate_qualities) + memCandidateQualities);

            d_candidate_qualities = (char*)qualities_transfer_data_device;
            d_subject_qualities = (char*)(((char*)d_candidate_qualities) + memCandidateQualities);


            //correction results

            std::size_t memCorrectedSubjects = n_subjects * sequence_pitch;
            std::size_t memCorrectedCandidates = n_indices * sequence_pitch;
            std::size_t memNumCorrectedCandidates = SDIV(n_subjects * sizeof(int), padding_bytes) * padding_bytes;
            std::size_t memSubjectIsCorrected = SDIV(n_subjects * sizeof(int), padding_bytes) * padding_bytes;

            std::size_t total_correction_results_transfer_data_size = memCorrectedSubjects
                                                                        + memCorrectedCandidates
                                                                        + memNumCorrectedCandidates
                                                                        + memSubjectIsCorrected;

            if(total_correction_results_transfer_data_size > correction_results_transfer_data_size){
                cudaFree(correction_results_transfer_data_device); CUERR;
                cudaMalloc(&correction_results_transfer_data_device, total_correction_results_transfer_data_size); CUERR;
                cudaFreeHost(correction_results_transfer_data_host); CUERR;
                cudaMallocHost(&correction_results_transfer_data_host, total_correction_results_transfer_data_size); CUERR;

                correction_results_transfer_data_size = total_correction_results_transfer_data_size;
            }

            h_corrected_subjects = (char*)correction_results_transfer_data_host;
            h_corrected_candidates = (char*)(((char*)h_corrected_subjects) + memCorrectedSubjects);
            h_num_corrected_candidates = (int*)(((char*)h_corrected_candidates) + memCorrectedCandidates);
            h_subject_is_corrected = (int*)(((char*)h_num_corrected_candidates) + memNumCorrectedCandidates);

            d_corrected_subjects = (char*)correction_results_transfer_data_device;
            d_corrected_candidates = (char*)(((char*)d_corrected_subjects) + memCorrectedSubjects);
            d_num_corrected_candidates = (int*)(((char*)d_corrected_candidates) + memCorrectedCandidates);
            d_subject_is_corrected = (int*)(((char*)d_num_corrected_candidates) + memNumCorrectedCandidates);


            //multple sequence alignment

            int msa_max_column_count = (3*maximum_sequence_length - 2*min_overlap);
            msa_pitch = SDIV(sizeof(char)*msa_max_column_count, padding_bytes) * padding_bytes;
            msa_weights_pitch = SDIV(sizeof(float)*msa_max_column_count, padding_bytes) * padding_bytes;

            std::size_t memMultipleSequenceAlignment = (n_subjects + n_indices) * msa_pitch;
            std::size_t memMultipleSequenceAlignmentWeights = (n_subjects + n_indices) * msa_weights_pitch;
            std::size_t memConsensus = SDIV(msa_max_column_count * n_subjects * sizeof(char), padding_bytes) * padding_bytes;
            std::size_t memSupport = SDIV(msa_max_column_count * n_subjects * sizeof(float), padding_bytes) * padding_bytes;
            std::size_t memCoverage = SDIV(msa_max_column_count * n_subjects * sizeof(int), padding_bytes) * padding_bytes;
            std::size_t memOrigWeights = SDIV(msa_max_column_count * n_subjects * sizeof(float), padding_bytes) * padding_bytes;
            std::size_t memOrigCoverage = SDIV(msa_max_column_count * n_subjects * sizeof(int), padding_bytes) * padding_bytes;
            std::size_t memMSAColumnProperties = SDIV(n_subjects * sizeof(care::msa::MSAColumnProperties), padding_bytes) * padding_bytes;

            std::size_t total_msa_data_size = memMultipleSequenceAlignment
                                                + memMultipleSequenceAlignmentWeights
                                                + memConsensus
                                                + memSupport
                                                + memCoverage
                                                + memOrigWeights
                                                + memOrigCoverage
                                                + memMSAColumnProperties;

            if(total_msa_data_size > msa_data_size){
                cudaFree(msa_data_device); CUERR;
                cudaMalloc(msa_data_device); CUERR;
                msa_data_size = total_msa_data_size;
            }

            d_multiple_sequence_alignment = (char*)msa_data_device;
            d_multiple_sequence_alignment_weights = (float*)(((char*)d_multiple_sequence_alignment) + memMultipleSequenceAlignment);
            d_consensus = (char*)(((char*)d_multiple_sequence_alignment_weights) + memMultipleSequenceAlignmentWeights);
            d_support = (float*)(((char*)d_consensus) + memConsensus);
            d_coverage = (int*)(((char*)d_support) + memSupport);
            d_origWeights = (float*)(((char*)d_coverage) + memCoverage);
            d_origCoverages = (int*)(((char*)d_origWeights) + memOrigWeights);
            d_msa_column_properties = (care::msa::MSAColumnProperties*)(((char*)d_origCoverages) + memOrigCoverage);
        }

        void set_tmp_storage_size(std::size_t newsize){
            if(newsize > tmp_storage_size){
                cudaFree(d_temp_storage); CUERR;
                cudaMalloc(&d_temp_storage, newsize); CUERR;
                tmp_storage_size = newsize;
            }
        }

        void reset(){
            auto& a = *this;

            cudaFree(a.d_temp_storage); CUERR;
            cudaFree(a.msa_data_device); CUERR;
            cudaFree(a.correction_results_transfer_data_device); CUERR;
            cudaFree(a.qualities_transfer_data_device); CUERR;
            cudaFree(a.indices_transfer_data_device); CUERR;
            cudaFree(a.d_indices_per_subject); CUERR;
            cudaFree(a.alignment_result_data_device); CUERR;
            cudaFree(a.alignment_transfer_data_device); CUERR;

            cudaFreeHost(a.correction_results_transfer_data_host); CUERR;
            cudaFreeHost(a.qualities_transfer_data_host); CUERR;
            cudaFreeHost(a.indices_transfer_data_host); CUERR;
            cudaFreeHost(a.alignment_transfer_data_host); CUERR;

            a.d_temp_storage = nullptr;
            a.msa_data_device = nullptr;
            a.correction_results_transfer_data_device = nullptr;
            a.qualities_transfer_data_device = nullptr;
            a.indices_transfer_data_device = nullptr;
            a.d_indices_per_subject = nullptr;
            a.alignment_result_data_device = nullptr;
            a.alignment_transfer_data_device = nullptr;
            a.correction_results_transfer_data_host = nullptr;
            a.qualities_transfer_data_host = nullptr;
            a.indices_transfer_data_host = nullptr;
            a.alignment_transfer_data_host = nullptr;

            a.alignment_transfer_data_size = 0;
            a.alignment_result_data_size = 0;
            a.indices_transfer_data_size = 0;
            a.qualities_transfer_data_size = 0;
            a.correction_results_transfer_data_size = 0;
            a.msa_data_size = 0;
            a.tmp_storage_size = 0;
        }

        int deviceId;

        int n_subjects = 0;
        int n_queries = 0;
        int n_indices = 0;
        int maximum_sequence_length = 0;
        int min_overlap = 1;

        // alignment input
        void* alignment_transfer_data_host = nullptr;
        void* alignment_transfer_data_device = nullptr;

        std::size_t alignment_transfer_data_size = 0;
        std::size_t encoded_sequence_pitch = 0;

        char* h_subject_sequences_data = nullptr;
        char* h_candidate_sequences_data = nullptr;
        int* h_subject_sequences_lengths = nullptr;
        int* h_candidate_sequences_lengths = nullptr;
        int* h_candidates_per_subject_prefixsum = nullptr;

        char* d_subject_sequences_data = nullptr;
        char* d_candidate_sequences_data = nullptr;
        int* d_subject_sequences_lengths = nullptr;
        int* d_candidate_sequences_lengths = nullptr;
        int* d_candidates_per_subject_prefixsum = nullptr;

        //indices output
        void* indices_transfer_data_host = nullptr;
        void* indices_transfer_data_device = nullptr;
        std::size_t indices_transfer_data_size = 0;

        int* h_indices = nullptr;
        int* h_indices_per_subject_prefixsum = nullptr;

        int* d_indices = nullptr;
        int* d_indices_per_subject_prefixsum = nullptr;

        //qualities input
        void* qualities_transfer_data_host = nullptr;
        void* qualities_transfer_data_device = nullptr;
        std::size_t qualities_transfer_data_size = 0;
        std::size_t quality_pitch = 0;

        char* h_candidate_qualities = nullptr;
        char* h_subject_qualities = nullptr;

        char* d_candidate_qualities = nullptr;
        char* d_subject_qualities = nullptr;

        //correction results output

        void* correction_results_transfer_data_host = nullptr;
        void* correction_results_transfer_data_device = nullptr;
        std::size_t correction_results_transfer_data_size = 0;
        std::size_t sequence_pitch = 0;

        char* h_corrected_subjects = nullptr;
        char* h_corrected_candidates = nullptr;
        int* h_num_corrected_candidates = nullptr;
        int* h_subject_is_corrected = nullptr;

        char* d_corrected_subjects = nullptr;
        char* d_corrected_candidates = nullptr;
        char* d_num_corrected_candidates = nullptr;
        int* d_subject_is_corrected = nullptr;


        //alignment results
        void* alignment_result_data_device = nullptr;
        std::size_t alignment_result_data_size = 0;

        int* d_alignment_scores = nullptr;
        int* d_alignment_overlaps = nullptr;
        int* d_alignment_shifts = nullptr;
        int* d_alignment_nOps = nullptr;
        bool* d_alignment_isValid = nullptr;
        BestAlignment_t* d_alignment_best_alignment_flags = nullptr;

        //tmp storage
        std::size_t tmp_storage_size = 0;
        char* d_temp_storage = nullptr;

        //indices per subject
        int* d_indices_per_subject = nullptr;

        // multiple sequence alignment
        void* msa_data_device = nullptr;
        std::size_t msa_data_size = 0;

        char* d_multiple_sequence_alignment = nullptr;
        float* d_multiple_sequence_alignment_weights = nullptr;
        char* d_consensus = nullptr;
        float* d_support = nullptr;
        int* d_coverage = nullptr;
        float* d_origWeights = nullptr;
        int* d_origCoverages = nullptr;
        care::msa::MSAColumnProperties* d_msa_column_properties = nullptr;

    };





    #endif

}
}




#endif
