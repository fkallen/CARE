#include <correctionresultprocessing.hpp>
#include <programoutputprocessing.hpp>

#include <config.hpp>
#include <hpc_helpers.cuh>
#include <memoryfile.hpp>
#include <readlibraryio.hpp>
#include <threadpool.hpp>
#include <concurrencyhelpers.hpp>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <queue>

#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <future>

namespace care{


struct CombinedCorrectionResult{
    bool corrected = false;
    bool hqCorrection = false;
    bool lqCorrectionOnlyAnchor = false;
    bool lqCorrectionWithCandidates = false;
    std::string correctedSequence{};
};

CombinedCorrectionResult combineMultipleCorrectionResults1(
    std::vector<TempCorrectedSequence>& tmpresults, 
    std::string& originalSequence
){
    if(tmpresults.empty()){
        CombinedCorrectionResult result;
        result.corrected = false;
        std::swap(result.correctedSequence, originalSequence);
        
        return result;
    }

    constexpr bool outputHQ = true;
    constexpr bool outputLQWithCandidates = true;
    constexpr bool outputLQOnlyAnchor = true;
    // constexpr bool outputOnlyCand = false;

    

    auto isAnchor = [](const auto& tcs){
        return tcs.type == TempCorrectedSequence::Type::Anchor;
    };

    auto anchorIter = std::find_if(tmpresults.begin(), tmpresults.end(), isAnchor);

    if(anchorIter != tmpresults.end()){
        //if there is a correction using a high quality alignment, use it
        if(anchorIter->hq){
            if(outputHQ){

                assert(anchorIter->sequence.size() == originalSequence.size());
                CombinedCorrectionResult result;

                result.corrected = true;
                result.hqCorrection = true;
                std::swap(result.correctedSequence, anchorIter->sequence);
                
                return result;
            }else{
                CombinedCorrectionResult result;

                result.corrected = false;
                result.correctedSequence = originalSequence;
                return result;
            }
        }else{

            TempCorrectedSequence& anchor = *anchorIter;

            if(tmpresults.size() >= 3){

                const bool sizelimitok = true; //tmpresults.size() > 3;

                const bool sameCorrections = std::all_of(tmpresults.begin()+1,
                                                        tmpresults.end(),
                                                        [&](const auto& tcs){
                                                            return tmpresults[0].sequence == tcs.sequence;
                                                        });

                if(sameCorrections && sizelimitok){
                    if(outputLQWithCandidates){
                        CombinedCorrectionResult result;
                        result.corrected = true;
                        result.lqCorrectionWithCandidates = true;
                        std::swap(result.correctedSequence, tmpresults[0].sequence);
                        return result;
                    }else{
                        CombinedCorrectionResult result;
                        result.corrected = false;
                        result.correctedSequence = originalSequence;
                        return result;
                    }
                }else{
                    CombinedCorrectionResult result;
                    result.corrected = false;
                    result.correctedSequence = originalSequence;
                    return result;
                }
            }else{
                if(outputLQOnlyAnchor){
                    CombinedCorrectionResult result;
                    result.corrected = true;
                    result.lqCorrectionOnlyAnchor = true;
                    std::swap(result.correctedSequence, anchor.sequence);
                    return result;
                }else{
                    CombinedCorrectionResult result;
                    result.corrected = false;
                    result.correctedSequence = originalSequence;
                    return result;
                }
            }
        }
    }else{

        CombinedCorrectionResult result;
        result.corrected = false;
        result.correctedSequence = originalSequence;
        return result;

        // tmpresults.erase(std::remove_if(tmpresults.begin(),
        //                                 tmpresults.end(),
        //                                 [](const auto& tcs){
        //                                     return std::abs(tcs.shift) > 0;
        //                                 }),
        //                   tmpresults.end());
        //
        // if(tmpresults.size() >= 1){
        //
        //     const bool sameCorrections = std::all_of(tmpresults.begin()+1,
        //                                             tmpresults.end(),
        //                                             [&](const auto& tcs){
        //                                                 return tmpresults[0].sequence == tcs.sequence;
        //                                             });
        //
        //     if(sameCorrections){
        //         return std::make_pair(tmpresults[0].sequence, outputOnlyCand);
        //     }else{
        //         return std::make_pair(std::string{""}, false);
        //     }
        // }else{
        //     return std::make_pair(std::string{""}, false);
        // }

    }

};

//same as combineMultipleCorrectionResults1, but edits have not been applied in the tcs
CombinedCorrectionResult combineMultipleCorrectionResults1_rawtcs(
    std::vector<TempCorrectedSequence>& tmpresults, 
    std::string& originalSequence
){
    if(tmpresults.empty()){
        CombinedCorrectionResult result;
        result.corrected = false;
        std::swap(result.correctedSequence, originalSequence);
        
        return result;
    }

    constexpr bool outputHQ = true;
    constexpr bool outputLQWithCandidates = true;
    constexpr bool outputLQOnlyAnchor = true;
    // constexpr bool outputOnlyCand = false;

    auto isAnchor = [](const auto& tcs){
        return tcs.type == TempCorrectedSequence::Type::Anchor;
    };

    auto anchorIter = std::find_if(tmpresults.begin(), tmpresults.end(), isAnchor);

    if(anchorIter != tmpresults.end()){
        //if there is a correction using a high quality alignment, use it
        if(anchorIter->hq){
            if(outputHQ){
                if(anchorIter->useEdits){
                    for(const auto& edit : anchorIter->edits){
                        originalSequence[edit.pos] = edit.base;
                    }
                }

                //assert(anchorIter->sequence.size() == originalSequence.size());
                CombinedCorrectionResult result;

                result.corrected = true;
                result.hqCorrection = true;
                std::swap(result.correctedSequence, originalSequence);
                
                return result;
            }else{
                CombinedCorrectionResult result;

                result.corrected = false;
                std::swap(result.correctedSequence, originalSequence);
                return result;
            }
        }else{

            if(tmpresults.size() >= 3){

                for(auto& tmpres : tmpresults){
                    if(tmpres.useEdits){
                        tmpres.sequence = originalSequence;
                        for(const auto& edit : tmpres.edits){
                            tmpres.sequence[edit.pos] = edit.base;
                        }
                    }
                }

                const bool sizelimitok = true; //tmpresults.size() > 3;

                const bool sameCorrections = std::all_of(tmpresults.begin()+1,
                                                        tmpresults.end(),
                                                        [&](const auto& tcs){
                                                            return tmpresults[0].sequence == tcs.sequence;
                                                        });

                if(sameCorrections && sizelimitok){
                    if(outputLQWithCandidates){
                        CombinedCorrectionResult result;
                        result.corrected = true;
                        result.lqCorrectionWithCandidates = true;
                        std::swap(result.correctedSequence, tmpresults[0].sequence);
                        return result;
                    }else{
                        CombinedCorrectionResult result;
                        result.corrected = false;
                        result.correctedSequence = originalSequence;
                        return result;
                    }
                }else{
                    CombinedCorrectionResult result;
                    result.corrected = false;
                    std::swap(result.correctedSequence, originalSequence);
                    return result;
                }
            }else{
                if(outputLQOnlyAnchor){
                    if(anchorIter->useEdits){
                        for(const auto& edit : anchorIter->edits){
                            originalSequence[edit.pos] = edit.base;
                        }
                    }
                    
                    CombinedCorrectionResult result;
                    result.corrected = true;
                    result.lqCorrectionOnlyAnchor = true;
                    std::swap(result.correctedSequence, originalSequence);
                    return result;
                }else{
                    CombinedCorrectionResult result;
                    result.corrected = false;
                    std::swap(result.correctedSequence, originalSequence);
                    return result;
                }
            }
        }
    }else{

        CombinedCorrectionResult result;
        result.corrected = false;
        std::swap(result.correctedSequence, originalSequence);
        return result;

        // tmpresults.erase(std::remove_if(tmpresults.begin(),
        //                                 tmpresults.end(),
        //                                 [](const auto& tcs){
        //                                     return std::abs(tcs.shift) > 0;
        //                                 }),
        //                   tmpresults.end());
        //
        // if(tmpresults.size() >= 1){
        //
        //     const bool sameCorrections = std::all_of(tmpresults.begin()+1,
        //                                             tmpresults.end(),
        //                                             [&](const auto& tcs){
        //                                                 return tmpresults[0].sequence == tcs.sequence;
        //                                             });
        //
        //     if(sameCorrections){
        //         return std::make_pair(tmpresults[0].sequence, outputOnlyCand);
        //     }else{
        //         return std::make_pair(std::string{""}, false);
        //     }
        // }else{
        //     return std::make_pair(std::string{""}, false);
        // }

    }

};


CombinedCorrectionResult combineMultipleCorrectionResults1_rawtcs2(
    std::vector<TempCorrectedSequence>& tmpresults, 
    ReadWithId& readWithId
){
    if(tmpresults.empty()){
        CombinedCorrectionResult result;
        result.corrected = false;        
        return result;
    }

    constexpr bool outputHQ = true;
    constexpr bool outputLQWithCandidates = true;
    constexpr bool outputLQOnlyAnchor = true;
    // constexpr bool outputOnlyCand = false;

    auto isValidSequence = [](const std::string& s){
        return std::all_of(s.begin(), s.end(), [](char c){
            return (c == 'A' || c == 'C' || c == 'G' || c == 'T' || c == 'N');
        });
    };

    auto isAnchor = [](const auto& tcs){
        return tcs.type == TempCorrectedSequence::Type::Anchor;
    };

    auto anchorIter = std::find_if(tmpresults.begin(), tmpresults.end(), isAnchor);

    if(anchorIter != tmpresults.end()){
        //if there is a correction using a high quality alignment, use it
        if(anchorIter->hq){
            if(outputHQ){
                if(anchorIter->useEdits){
                    for(const auto& edit : anchorIter->edits){
                        readWithId.read.sequence[edit.pos] = edit.base;
                    }
                }else{
                    std::swap(readWithId.read.sequence, anchorIter->sequence);
                }

                if(!isValidSequence(readWithId.read.sequence)){
                    std::cerr << "Warning. Corrected read " << readWithId.globalReadId
                            << " with header " << readWithId.read.header
                            << "does contain an invalid DNA base!\n"
                            << "Corrected sequence is: "  << readWithId.read.sequence << '\n';
                }

                //assert(anchorIter->sequence.size() == originalSequence.size());
                CombinedCorrectionResult result;

                result.corrected = true;
                result.hqCorrection = true;
                
                return result;
            }else{
                CombinedCorrectionResult result;

                result.corrected = false;
                return result;
            }
        }else{

            auto isEqualSequenceOneHasEdits = [&](const auto& tcswithedit, const auto& tcs2){
                assert(tcswithedit.useEdits);
                assert(!tcs2.useEdits);

                const int len1 = readWithId.read.sequence.length();

                auto editIter = tcswithedit.edits.begin();
                int pos = 0;
                while(pos != len1 && editIter != tcswithedit.edits.end()){
                    if(pos != editIter->pos){
                        if(readWithId.read.sequence[pos] != tcs2.sequence[pos]){
                            return false;
                        }
                    }else{
                        if(editIter->base != tcs2.sequence[pos]){
                            return false;
                        }
                        ++editIter;
                    }

                    pos++;
                }
                while(pos != len1){
                    if(readWithId.read.sequence[pos] != tcs2.sequence[pos]){
                        return false;
                    }
                    pos++;
                }
                return true;
            };

            auto isEqualSequence = [&](const auto& tcs1, const auto& tcs2){
                
                const int len1 = tcs1.useEdits ? readWithId.read.sequence.length() : tcs1.sequence.length();
                const int len2 = tcs2.useEdits ? readWithId.read.sequence.length() : tcs2.sequence.length();
                if(len1 == len2){
                    if(tcs1.useEdits && tcs2.useEdits){
                        return tcs1.edits == tcs2.edits;
                    }else if(!tcs1.useEdits && !tcs2.useEdits){
                        return tcs1.sequence == tcs2.sequence;
                    }else if(tcs1.useEdits && !tcs2.useEdits){
                        if(tcs1.edits.empty()){
                            return readWithId.read.sequence == tcs2.sequence;
                        }else{
                            return isEqualSequenceOneHasEdits(tcs1, tcs2);
                        }
                    }else{
                        //if(!tcs1.useEdits && tcs2.useEdits)
                        if(tcs2.edits.empty()){
                            return readWithId.read.sequence == tcs1.sequence;
                        }else{
                            return isEqualSequenceOneHasEdits(tcs2, tcs1);
                        }
                    }
                }else{
                    return false;
                }
            };

            if(tmpresults.size() >= 3){

                // for(auto& tmpres : tmpresults){
                //     if(tmpres.useEdits){
                //         tmpres.sequence = readWithId.read.sequence;
                //         for(const auto& edit : tmpres.edits){
                //             tmpres.sequence[edit.pos] = edit.base;
                //         }
                //     }
                // }

                const bool sizelimitok = true; //tmpresults.size() > 3;

                // const bool sameCorrections2 = std::all_of(tmpresults.begin()+1,
                //                                         tmpresults.end(),
                //                                         [&](const auto& tcs){
                //                                             return tmpresults[0].sequence == tcs.sequence;
                //                                         });

                const bool sameCorrections = std::all_of(tmpresults.begin()+1,
                                                        tmpresults.end(),
                                                        [&](const auto& tcs){
                                                            return isEqualSequence(tmpresults[0], tcs);
                                                        });

                //assert(sameCorrections == sameCorrections2);                

                if(sameCorrections && sizelimitok){
                    if(outputLQWithCandidates){
                        CombinedCorrectionResult result;
                        result.corrected = true;
                        result.lqCorrectionWithCandidates = true;

                        if(tmpresults[0].useEdits){
                            for(const auto& edit : tmpresults[0].edits){
                                readWithId.read.sequence[edit.pos] = edit.base;
                            }
                        }else{
                            std::swap(readWithId.read.sequence, tmpresults[0].sequence);
                        }

                        if(!isValidSequence(readWithId.read.sequence)){
                            std::cerr << "Warning. Corrected read " << readWithId.globalReadId
                                    << " with header " << readWithId.read.header
                                    << "does contain an invalid DNA base!\n"
                                    << "Corrected sequence is: "  << readWithId.read.sequence << '\n';
                        }

                        
                        return result;
                    }else{
                        CombinedCorrectionResult result;
                        result.corrected = false;
                        return result;
                    }
                }else{
                    CombinedCorrectionResult result;
                    result.corrected = false;
                    return result;
                }
            }else{
                if(outputLQOnlyAnchor){
                    if(anchorIter->useEdits){
                        for(const auto& edit : anchorIter->edits){
                            readWithId.read.sequence[edit.pos] = edit.base;
                        }
                    }else{
                        std::swap(readWithId.read.sequence, anchorIter->sequence);
                    }

                    if(!isValidSequence(readWithId.read.sequence)){
                        std::cerr << "Warning. Corrected read " << readWithId.globalReadId
                                << " with header " << readWithId.read.header
                                << "does contain an invalid DNA base!\n"
                                << "Corrected sequence is: "  << readWithId.read.sequence << '\n';
                    }
                    
                    CombinedCorrectionResult result;
                    result.corrected = true;
                    result.lqCorrectionOnlyAnchor = true;
                    return result;
                }else{
                    CombinedCorrectionResult result;
                    result.corrected = false;
                    return result;
                }
            }
        }
    }else{

        CombinedCorrectionResult result;
        result.corrected = false;
        return result;

        // tmpresults.erase(std::remove_if(tmpresults.begin(),
        //                                 tmpresults.end(),
        //                                 [](const auto& tcs){
        //                                     return std::abs(tcs.shift) > 0;
        //                                 }),
        //                   tmpresults.end());
        //
        // if(tmpresults.size() >= 1){
        //
        //     const bool sameCorrections = std::all_of(tmpresults.begin()+1,
        //                                             tmpresults.end(),
        //                                             [&](const auto& tcs){
        //                                                 return tmpresults[0].sequence == tcs.sequence;
        //                                             });
        //
        //     if(sameCorrections){
        //         return std::make_pair(tmpresults[0].sequence, outputOnlyCand);
        //     }else{
        //         return std::make_pair(std::string{""}, false);
        //     }
        // }else{
        //     return std::make_pair(std::string{""}, false);
        // }

    }

};





template<class MemoryFile_t>
void constructOutputFileFromCorrectionResults_impl(
                    const std::string& tempdir,
                    const std::vector<std::string>& originalReadFiles,
                    MemoryFile_t& partialResults, 
                    std::size_t memoryForSorting,
                    FileFormat outputFormat,
                    const std::vector<std::string>& outputfiles,
                    bool isSorted){

    assert(outputfiles.size() == 1 || originalReadFiles.size() == outputfiles.size());


    if(!isSorted){
        auto ptrcomparator = [](const std::uint8_t* ptr1, const std::uint8_t* ptr2){
            read_number id1, id2;
            std::memcpy(&id1, ptr1, sizeof(read_number));
            std::memcpy(&id2, ptr2, sizeof(read_number));
            
            return id1 < id2;
        };

        auto elementcomparator = [](const auto& l, const auto& r){
            return l.readId < r.readId;
        };

        helpers::CpuTimer timer("sort_results_by_read_id");
        partialResults.sort(tempdir, memoryForSorting, ptrcomparator, elementcomparator);
        timer.print();
    }

    helpers::CpuTimer mergetimer("merging");

    auto isValidSequence = [](const std::string& s){
        return std::all_of(s.begin(), s.end(), [](char c){
            return (c == 'A' || c == 'C' || c == 'G' || c == 'T' || c == 'N');
        });
    };

    
    BackgroundThread outputThread(true);


    std::uint64_t currentReadId = 0;
    std::vector<TempCorrectedSequence> correctionVector;
    correctionVector.reserve(256);
    //bool hqSubject = false;

    std::uint64_t currentReadId_tmp = 0;
    std::vector<TempCorrectedSequence> correctionVector_tmp;
    correctionVector_tmp.reserve(256);
    //bool hqSubject_tmp = false;


    auto partialResultsReader = partialResults.makeReader();

    //no gz output
    if(outputFormat == FileFormat::FASTQGZ)
        outputFormat = FileFormat::FASTQ;
    if(outputFormat == FileFormat::FASTAGZ)
        outputFormat = FileFormat::FASTA;


    std::vector<std::unique_ptr<SequenceFileWriter>> writerVector;
    std::vector<kseqpp::KseqPP> inputReaderVector;

    for(const auto& outputfile : outputfiles){
        writerVector.emplace_back(makeSequenceWriter(outputfile, outputFormat));
    }   

    MultiInputReader multiInputReader(originalReadFiles);

    ReadWithId readWithId{};
    bool firstiter = true;

    std::chrono::time_point<std::chrono::system_clock> timebegin, timeend;

    std::chrono::duration<double> durationInOutputThread{0};
    std::chrono::duration<double> durationPrload{0};
    std::chrono::duration<double> durationCopyOrig{0};
    std::chrono::duration<double> durationConstruction{0};

    while(partialResultsReader.hasNext() || correctionVector.size() > 0){
        timebegin = std::chrono::system_clock::now();

        if(partialResultsReader.hasNext()){
            TempCorrectedSequence tcs = *(partialResultsReader.next());

            if(firstiter || tcs.readId == currentReadId){
                currentReadId = tcs.readId ;
                correctionVector.emplace_back(std::move(tcs));

                while(partialResultsReader.hasNext()){
                    TempCorrectedSequence tcs2 = *(partialResultsReader.next());
                    if(tcs2.readId == currentReadId){
                        correctionVector.emplace_back(std::move(tcs2));
                    }else{
                        currentReadId_tmp = tcs2.readId;
                        correctionVector_tmp.emplace_back(std::move(tcs2));
                        break;
                    }
                }
            }else{
                currentReadId_tmp = tcs.readId;
                correctionVector_tmp.emplace_back(std::move(tcs));
            }
        }else{
            //std::cerr << "partial results empty with currentReadId = " << currentReadId << "\n";
        }

        timeend = std::chrono::system_clock::now();

        durationPrload += timeend - timebegin;

        timebegin = std::chrono::system_clock::now();

        //copy preceding reads from original file

        if(currentReadId != 0){  //to search for the first read, no skipping is needed
            while(readWithId.globalReadId < currentReadId - 1){
                const int status = multiInputReader.next();
                bool hasNext = status >= 0;

                if(hasNext){
                    std::swap(readWithId, multiInputReader.getCurrent());

                    writerVector[readWithId.fileId]->writeRead(readWithId.read);
                }else{
                    throw std::runtime_error{"Cannot skip to read " + std::to_string(currentReadId)
                            + " during merge."};
                }
            }

            assert(readWithId.globalReadId + 1 == currentReadId);
        }

        timeend = std::chrono::system_clock::now();

        durationCopyOrig += timeend - timebegin;

        timebegin = std::chrono::system_clock::now();

        //get read with id currentReadId
        {
            const int status = multiInputReader.next();
            bool hasNext = status >= 0;

            if(hasNext){
                std::swap(readWithId, multiInputReader.getCurrent());
            }else{
                throw std::runtime_error{"Could not find read " + std::to_string(currentReadId)
                        + " during merge."};
            }
        }

        assert(readWithId.globalReadId == currentReadId);

        //replace sequence of next read with corrected sequence

        for(auto& tmpres : correctionVector){
            if(tmpres.useEdits){
                tmpres.sequence = readWithId.read.sequence;
                for(const auto& edit : tmpres.edits){
                    tmpres.sequence[edit.pos] = edit.base;
                }
            }
        }

        CombinedCorrectionResult combinedresult = combineMultipleCorrectionResults1(
            correctionVector, 
            readWithId.read.sequence
        );


        if(combinedresult.corrected){
            if(!isValidSequence(combinedresult.correctedSequence)){
                std::cerr << "Warning. Corrected read " << readWithId.globalReadId
                        << " with header " << readWithId.read.header
                        << "does contain an invalid DNA base!\n"
                        << "Corrected sequence is: "  << combinedresult.correctedSequence << '\n';
            }
            //readWithId.read.sequence = std::move(correctedSequence.first);
        }      

        std::swap(readWithId.read.sequence, combinedresult.correctedSequence);        

        writerVector[readWithId.fileId]->writeRead(readWithId.read.header, readWithId.read.sequence, readWithId.read.quality);

        timeend = std::chrono::system_clock::now();
        durationConstruction += timeend - timebegin;

        correctionVector.clear();
        std::swap(correctionVector, correctionVector_tmp);
        std::swap(currentReadId, currentReadId_tmp);


        firstiter = false;
    }


    // std::cerr << "# elapsed time (durationInOutputThread): " << durationInOutputThread.count()  << " s" << std::endl;
    // std::cerr << "# elapsed time (durationPrload): " << durationPrload.count()  << " s" << std::endl;
    // std::cerr << "# elapsed time (durationCopyOrig): " << durationCopyOrig.count()  << " s" << std::endl;
    // std::cerr << "# elapsed time (durationConstruction): " << durationConstruction.count()  << " s" << std::endl;


    //copy remaining reads from original files
    while(multiInputReader.next() >= 0){
        std::swap(readWithId, multiInputReader.getCurrent());
        writerVector[readWithId.fileId]->writeRead(readWithId.read);
    }

    outputThread.stopThread(BackgroundThread::StopType::FinishAndStop);

    mergetimer.print();
}






template<class MemoryFile_t>
void constructOutputFileFromCorrectionResults2_impl(
                    const std::string& tempdir,
                    const std::vector<std::string>& originalReadFiles,
                    MemoryFile_t& partialResults, 
                    std::size_t memoryForSorting,
                    FileFormat outputFormat,
                    const std::vector<std::string>& outputfiles,
                    bool isSorted){

    assert(outputfiles.size() == 1 || originalReadFiles.size() == outputfiles.size());


    if(!isSorted){
        auto ptrcomparator = [](const std::uint8_t* ptr1, const std::uint8_t* ptr2){
            read_number id1, id2;
            std::memcpy(&id1, ptr1, sizeof(read_number));
            std::memcpy(&id2, ptr2, sizeof(read_number));
            
            return id1 < id2;
        };

        auto elementcomparator = [](const auto& l, const auto& r){
            return l.readId < r.readId;
        };

        helpers::CpuTimer timer("sort_results_by_read_id");
        partialResults.sort(tempdir, memoryForSorting, ptrcomparator, elementcomparator);
        timer.print();
    }

    helpers::CpuTimer mergetimer("merging");

    auto isValidSequence = [](const std::string& s){
        return std::all_of(s.begin(), s.end(), [](char c){
            return (c == 'A' || c == 'C' || c == 'G' || c == 'T' || c == 'N');
        });
    };


    auto partialResultsReader = partialResults.makeReader();

    //no gz output
    if(outputFormat == FileFormat::FASTQGZ)
        outputFormat = FileFormat::FASTQ;
    if(outputFormat == FileFormat::FASTAGZ)
        outputFormat = FileFormat::FASTA;


    std::vector<std::unique_ptr<SequenceFileWriter>> writerVector;
    std::vector<kseqpp::KseqPP> inputReaderVector;

    for(const auto& outputfile : outputfiles){
        writerVector.emplace_back(makeSequenceWriter(outputfile, outputFormat));
    }   

    MultiInputReader multiInputReader(originalReadFiles);

    ReadWithId readWithId{};


    if(partialResultsReader.hasNext()){
        partialResultsReader.next();
    }

    while(multiInputReader.next() >= 0){
        std::swap(readWithId, multiInputReader.getCurrent());

        if(!partialResultsReader.hasNext()){
            //correction results are processed
            //copy remaining input reads to output file

            writerVector[readWithId.fileId]->writeRead(readWithId.read);
            while(multiInputReader.next() >= 0){
                std::swap(readWithId, multiInputReader.getCurrent());
                writerVector[readWithId.fileId]->writeRead(readWithId.read);
            }
            break;      
        }

        if(partialResultsReader.current()->readId < readWithId.globalReadId){
            assert(false); //for each tcs with readId X there must be an input read with globalreadId X
        }else{
            std::vector<TempCorrectedSequence> buffer;

            // fill buffer with tcs of same read id as input read
            bool loop = false;

            do{       
                loop = false;

                if(!(readWithId.globalReadId < partialResultsReader.current()->readId)){
                    buffer.emplace_back(*partialResultsReader.current());

                    if(partialResultsReader.hasNext()){
                        partialResultsReader.next();
                        loop = true;
                    }
                }
            } while(loop);

            for(auto& tmpres : buffer){
                if(tmpres.useEdits){
                    tmpres.sequence = readWithId.read.sequence;
                    for(const auto& edit : tmpres.edits){
                        tmpres.sequence[edit.pos] = edit.base;
                    }
                }
            }

            CombinedCorrectionResult combinedresult = combineMultipleCorrectionResults1(
                buffer, 
                readWithId.read.sequence
            );

            if(combinedresult.corrected){
                if(!isValidSequence(combinedresult.correctedSequence)){
                    std::cerr << "Warning. Corrected read " << readWithId.globalReadId
                            << " with header " << readWithId.read.header
                            << "does contain an invalid DNA base!\n"
                            << "Corrected sequence is: "  << combinedresult.correctedSequence << '\n';
                }
            }      

            std::swap(readWithId.read.sequence, combinedresult.correctedSequence);
            writerVector[readWithId.fileId]->writeRead(readWithId.read);
        }
    }

    mergetimer.print();
}





template<class MemoryFile_t>
void constructOutputFileFromCorrectionResults_multithreading_impl(
                    const std::string& tempdir,
                    const std::vector<std::string>& originalReadFiles,
                    MemoryFile_t& partialResults, 
                    std::size_t memoryForSorting,
                    FileFormat outputFormat,
                    const std::vector<std::string>& outputfiles,
                    bool isSorted){

    assert(outputfiles.size() == 1 || originalReadFiles.size() == outputfiles.size());


    if(!isSorted){
        auto ptrcomparator = [](const std::uint8_t* ptr1, const std::uint8_t* ptr2){
            read_number id1, id2;
            std::memcpy(&id1, ptr1, sizeof(read_number));
            std::memcpy(&id2, ptr2, sizeof(read_number));
            
            return id1 < id2;
        };

        auto elementcomparator = [](const auto& l, const auto& r){
            return l.readId < r.readId;
        };

        helpers::CpuTimer timer("sort_results_by_read_id");
        partialResults.sort(tempdir, memoryForSorting, ptrcomparator, elementcomparator);
        timer.print();
    }

    helpers::CpuTimer mergetimer("merging");

    auto isValidSequence = [](const std::string& s){
        return std::all_of(s.begin(), s.end(), [](char c){
            return (c == 'A' || c == 'C' || c == 'G' || c == 'T' || c == 'N');
        });
    };

    struct TempCorrectedSequenceBatch{
        int validItems = 0;
        int processedItems = 0;
        std::vector<TempCorrectedSequence> items;
    };

    std::array<TempCorrectedSequenceBatch, 4> tcsBatches;

    SimpleSingleProducerSingleConsumerQueue<TempCorrectedSequenceBatch*> freeTcsBatches;
    SimpleSingleProducerSingleConsumerQueue<TempCorrectedSequenceBatch*> unprocessedTcsBatches;

    std::atomic<bool> noMoreTcsBatches{false};

    for(auto& batch : tcsBatches){
        freeTcsBatches.push(&batch);
    }

    constexpr int decoder_maxbatchsize = 100000;

    auto decoderFuture = std::async(std::launch::async,
        [&](){
            

            auto partialResultsReader = partialResults.makeReader();

            // std::chrono::time_point<std::chrono::system_clock> abegin, aend;
            // std::chrono::duration<double> adelta{0};

            // TIMERSTARTCPU(tcsparsing);

            while(partialResultsReader.hasNext()){
                TempCorrectedSequenceBatch* batch = freeTcsBatches.pop();

                //abegin = std::chrono::system_clock::now();

                batch->items.resize(decoder_maxbatchsize);

                int batchsize = 0;
                while(batchsize < decoder_maxbatchsize && partialResultsReader.hasNext()){
                    batch->items[batchsize] = *(partialResultsReader.next());
                    batchsize++;
                }

                // aend = std::chrono::system_clock::now();
                // adelta += aend - abegin;

                batch->processedItems = 0;
                batch->validItems = batchsize;

                unprocessedTcsBatches.push(batch);
            }

            //std::cout << "# elapsed time ("<< "tcsparsing without queues" <<"): " << adelta.count()  << " s" << std::endl;

            // TIMERSTOPCPU(tcsparsing);

            noMoreTcsBatches = true;
        }
    );


    struct ReadBatch{
        int validItems = 0;
        int processedItems = 0;
        std::vector<ReadWithId> items;
    };

    std::array<ReadBatch, 4> readBatches;

    // free -> unprocessed input -> unprocessed output -> free -> ...
    SimpleSingleProducerSingleConsumerQueue<ReadBatch*> freeReadBatches;
    SimpleSingleProducerSingleConsumerQueue<ReadBatch*> unprocessedInputreadBatches;
    SimpleSingleProducerSingleConsumerQueue<ReadBatch*> unprocessedOutputreadBatches;

    std::atomic<bool> noMoreInputreadBatches{false};

    constexpr int inputreader_maxbatchsize = 200000;

    for(auto& batch : readBatches){
        freeReadBatches.push(&batch);
    }

    auto inputReaderFuture = std::async(std::launch::async,
        [&](){

            MultiInputReader multiInputReader(originalReadFiles);

            // TIMERSTARTCPU(inputparsing);

            // std::chrono::time_point<std::chrono::system_clock> abegin, aend;
            // std::chrono::duration<double> adelta{0};

            while(multiInputReader.next() >= 0){

                ReadBatch* batch = freeReadBatches.pop();

                // abegin = std::chrono::system_clock::now();

                batch->items.resize(inputreader_maxbatchsize);

                std::swap(batch->items[0], multiInputReader.getCurrent()); //process element from outer loop next() call
                int batchsize = 1;

                while(batchsize < inputreader_maxbatchsize && multiInputReader.next() >= 0){
                    std::swap(batch->items[batchsize], multiInputReader.getCurrent());
                    batchsize++;
                }

                // aend = std::chrono::system_clock::now();
                // adelta += aend - abegin;

                batch->processedItems = 0;
                batch->validItems = batchsize;

                unprocessedInputreadBatches.push(batch);                
            }

            // std::cout << "# elapsed time ("<< "inputparsing without queues" <<"): " << adelta.count()  << " s" << std::endl;

            // TIMERSTOPCPU(inputparsing);

            noMoreInputreadBatches = true;
        }
    );

    std::atomic<bool> noMoreOutputreadBatches{false};

    auto outputWriterFuture = std::async(std::launch::async,
        [&](){
            //no gz output
            auto format = outputFormat;

            if(format == FileFormat::FASTQGZ)
                format = FileFormat::FASTQ;
            if(format == FileFormat::FASTAGZ)
                format = FileFormat::FASTA;

            std::vector<std::unique_ptr<SequenceFileWriter>> writerVector;


            for(const auto& outputfile : outputfiles){
                writerVector.emplace_back(makeSequenceWriter(outputfile, format));
            }

            // TIMERSTARTCPU(outputwriting);

            // std::chrono::time_point<std::chrono::system_clock> abegin, aend;
            // std::chrono::duration<double> adelta{0};

            ReadBatch* outputBatch = unprocessedOutputreadBatches.pop();

            while(outputBatch != nullptr){                

                // abegin = std::chrono::system_clock::now();
                
                int processed = outputBatch->processedItems;
                const int valid = outputBatch->validItems;
                while(processed < valid){
                    const auto& readWithId = outputBatch->items[processed];

                    writerVector[readWithId.fileId]->writeRead(readWithId.read);

                    processed++;
                }

                // aend = std::chrono::system_clock::now();
                // adelta += aend - abegin;

                freeReadBatches.push(outputBatch);     

                outputBatch = unprocessedOutputreadBatches.popOrDefault(
                    [&](){
                        return !noMoreOutputreadBatches;  //its possible that there are no corrections. In that case, return nullptr
                    },
                    nullptr
                );            
            }

            // std::cout << "# elapsed time ("<< "outputwriting without queues" <<"): " << adelta.count()  << " s" << std::endl;

            // TIMERSTOPCPU(outputwriting);
        }
    );

    TempCorrectedSequenceBatch* tcsBatch = unprocessedTcsBatches.popOrDefault(
        [&](){
            return !noMoreTcsBatches;  //its possible that there are no corrections. In that case, return nullptr
        },
        nullptr
    ); 

    ReadBatch* inputBatch = unprocessedInputreadBatches.pop();

    assert(!(inputBatch == nullptr && tcsBatch != nullptr)); //there must be at least one batch of input reads

    // std::chrono::time_point<std::chrono::system_clock> abegin, aend;
    // std::chrono::duration<double> adelta{0};

    // std::chrono::time_point<std::chrono::system_clock> bbegin, bend;
    // std::chrono::duration<double> bdelta{0};

    // std::chrono::time_point<std::chrono::system_clock> cbegin, cend;
    // std::chrono::duration<double> cdelta{0};

    std::vector<TempCorrectedSequence> buffer;

    while(!(inputBatch == nullptr && tcsBatch == nullptr)){
        // bbegin = std::chrono::system_clock::now();
        
        // bend = std::chrono::system_clock::now();
        // bdelta += bend - bbegin;

        //modify reads in inputbatch, applying corrections.
        //then place inputbatch in outputqueue

        if(tcsBatch == nullptr){
            //all correction results are processed
            //copy remaining input reads to output file
            // abegin = std::chrono::system_clock::now();
            
            ; //nothing to do

            // aend = std::chrono::system_clock::now();
            // adelta += aend - abegin;
        }else{

            // abegin = std::chrono::system_clock::now();

            auto last1 = inputBatch->items.begin() + inputBatch->validItems;
            auto last2 = tcsBatch->items.begin() + tcsBatch->validItems;

            auto first1 = inputBatch->items.begin()+ inputBatch->processedItems;
            auto first2 = tcsBatch->items.begin()+ tcsBatch->processedItems;      

            while(first1 != last1) {
                if(first2 == last2){
                    //all correction results are processed
                    //copy remaining input reads to output file

                    ; //nothing to do

                    // aend = std::chrono::system_clock::now();
                    // adelta += aend - abegin;
                    break;
                }
                assert(first2->readId >= first1->globalReadId);
                {

                    ReadWithId& readWithId = *first1;
                    
                    buffer.clear();
                    while(first2 != last2 && !(first1->globalReadId < first2->readId)){
                        buffer.push_back(*first2);
                        ++first2;
                        tcsBatch->processedItems++;

                        if(first2 == last2){
                            // aend = std::chrono::system_clock::now();
                            // adelta += aend - abegin;

                            // bbegin = std::chrono::system_clock::now();
                            
                            freeTcsBatches.push(tcsBatch);

                            tcsBatch = unprocessedTcsBatches.popOrDefault(
                                [&](){
                                    return !noMoreTcsBatches;  //its possible that there are no more corrections. In that case, return nullptr
                                },
                                nullptr
                            );

                            // bend = std::chrono::system_clock::now();
                            // bdelta += bend - bbegin;

                            // abegin = std::chrono::system_clock::now();

                            if(tcsBatch != nullptr){
                                //new batch could be fetched. update begin and end accordingly
                                last2 = tcsBatch->items.begin() + tcsBatch->validItems;
                                first2 = tcsBatch->items.begin()+ tcsBatch->processedItems;
                            }
                        }
                    }                    

                    CombinedCorrectionResult combinedresult = combineMultipleCorrectionResults1_rawtcs2(
                        buffer, 
                        readWithId
                    );

                    if(combinedresult.corrected){
                        if(!isValidSequence(readWithId.read.sequence)){
                            std::cerr << "Warning. Corrected read " << readWithId.globalReadId
                                    << " with header " << readWithId.read.header
                                    << "does contain an invalid DNA base!\n"
                                    << "Corrected sequence is: "  << readWithId.read.sequence << '\n';
                        }
                    }      

                    ++first1;
                }
                
            }

            // aend = std::chrono::system_clock::now();
            // adelta += aend - abegin;
        }

        // bbegin = std::chrono::system_clock::now();

        assert(inputBatch != nullptr);

        unprocessedOutputreadBatches.push(inputBatch);

        inputBatch = unprocessedInputreadBatches.popOrDefault(
            [&](){
                return !noMoreInputreadBatches;
            },
            nullptr
        );  

        // bend = std::chrono::system_clock::now();
        // bdelta += bend - bbegin;
        
        assert(!(inputBatch == nullptr && tcsBatch != nullptr));

    }

    noMoreOutputreadBatches = true;

    // std::cout << "# elapsed time ("<< "a" <<"): " << adelta.count()  << " s" << std::endl;
    // std::cout << "# elapsed time ("<< "b" <<"): " << bdelta.count()  << " s" << std::endl;
    // std::cout << "# elapsed time ("<< "c" <<"): " << cdelta.count()  << " s" << std::endl;

    decoderFuture.wait();
    inputReaderFuture.wait();
    outputWriterFuture.wait();


    mergetimer.print();
}

void constructOutputFileFromCorrectionResults(
    const std::string& tempdir,
    const std::vector<std::string>& originalReadFiles,
    MemoryFileFixedSize<EncodedTempCorrectedSequence>& partialResults, 
    std::size_t memoryForSorting,
    FileFormat outputFormat,
    const std::vector<std::string>& outputfiles,
    bool isSorted
){
                        
    //constructOutputFileFromCorrectionResults2_impl(
    // constructOutputFileFromCorrectionResults_multithreading_impl(
    //     tempdir, 
    //     originalReadFiles, 
    //     partialResults, 
    //     memoryForSorting, 
    //     outputFormat,
    //     outputfiles, 
    //     isSorted
    // );

    mergeResultsWithOriginalReads_multithreaded<TempCorrectedSequence>(
        tempdir,
        originalReadFiles,
        partialResults, 
        memoryForSorting,
        outputFormat,
        outputfiles,
        isSorted,
        combineMultipleCorrectionResults1_rawtcs2
    );
}




    EncodedTempCorrectedSequence& EncodedTempCorrectedSequence::operator=(const TempCorrectedSequence& rhs){
        rhs.encodeInto(*this);

        return *this;
    }

    bool EncodedTempCorrectedSequence::writeToBinaryStream(std::ostream& os) const{
        //assert(bool(os)); 
        os.write(reinterpret_cast<const char*>(&readId), sizeof(read_number));
        //assert(bool(os));
        os.write(reinterpret_cast<const char*>(&encodedflags), sizeof(std::uint32_t));
        //assert(bool(os));
        const int numBytes = getNumBytes();
        os.write(reinterpret_cast<const char*>(data.get()), sizeof(std::uint8_t) * numBytes);
        //assert(bool(os));
        return bool(os);
    }

    bool EncodedTempCorrectedSequence::readFromBinaryStream(std::istream& is){
        is.read(reinterpret_cast<char*>(&readId), sizeof(read_number));
        is.read(reinterpret_cast<char*>(&encodedflags), sizeof(std::uint32_t));
        const int numBytes = getNumBytes();

        data = std::make_unique<std::uint8_t[]>(numBytes);

        is.read(reinterpret_cast<char*>(data.get()), sizeof(std::uint8_t) * numBytes);

        return bool(is);
    }

    std::uint8_t* EncodedTempCorrectedSequence::copyToContiguousMemory(std::uint8_t* ptr, std::uint8_t* endPtr) const{
        const int dataBytes = getNumBytes();

        const std::size_t availableBytes = std::distance(ptr, endPtr);
        const std::size_t requiredBytes = sizeof(read_number) + sizeof(std::uint32_t) + dataBytes;
        if(requiredBytes <= availableBytes){
            std::memcpy(ptr, &readId, sizeof(read_number));
            ptr += sizeof(read_number);
            std::memcpy(ptr, &encodedflags, sizeof(std::uint32_t));
            ptr += sizeof(std::uint32_t);
            std::memcpy(ptr, data.get(), dataBytes);
            ptr += dataBytes;
            return ptr;
        }else{
            return nullptr;
        }        
    }

    void EncodedTempCorrectedSequence::copyFromContiguousMemory(const std::uint8_t* ptr){
        std::memcpy(&readId, ptr, sizeof(read_number));
        ptr += sizeof(read_number);
        std::memcpy(&encodedflags, ptr, sizeof(std::uint32_t));
        ptr += sizeof(read_number);

        const int numBytes = getNumBytes();
        data = std::make_unique<std::uint8_t[]>(numBytes);

        std::memcpy(data.get(), ptr, numBytes);
        //ptr += numBytes;
    }


    TempCorrectedSequence::TempCorrectedSequence(const EncodedTempCorrectedSequence& encoded){
        decode(encoded);
    }

    TempCorrectedSequence& TempCorrectedSequence::operator=(const EncodedTempCorrectedSequence& encoded){
        decode(encoded);
        return *this;
    }

    void TempCorrectedSequence::encodeInto(EncodedTempCorrectedSequence& target) const{
        const std::uint32_t oldNumBytes = target.getNumBytes(); 

        target.readId = readId;

        target.encodedflags = (std::uint32_t(hq) << 31);
        target.encodedflags |= (std::uint32_t(useEdits) << 30);
        target.encodedflags |= (std::uint32_t(int(type)) << 29);

        constexpr std::uint32_t maxNumBytes = (std::uint32_t(1) << 29)-1;

        std::uint32_t numBytes = 0;
        if(useEdits){
            const int numEdits = edits.size();
            numBytes += sizeof(int);
            numBytes += numEdits * (sizeof(int) + sizeof(char));
        }else{
            numBytes += sizeof(int);
            numBytes += sequence.length();
        }

        if(type == TempCorrectedSequence::Type::Anchor){
            ; //nothing
        }else{
            numBytes += sizeof(int);
        }

        assert(numBytes <= maxNumBytes);
        target.encodedflags |= numBytes;

        if(numBytes > oldNumBytes){
            target.data = std::make_unique<std::uint8_t[]>(numBytes);
        }else{
            ; //reuse buffer
        }

        //fill buffer

        std::uint8_t* ptr = target.data.get();

        if(useEdits){
            const int numEdits = edits.size();
            std::memcpy(ptr, &numEdits, sizeof(int));
            ptr += sizeof(int);
            for(const auto& edit : edits){
                std::memcpy(ptr, &edit.pos, sizeof(int));
                ptr += sizeof(int);
            }
            for(const auto& edit : edits){
                std::memcpy(ptr, &edit.base, sizeof(char));
                ptr += sizeof(char);
            }
        }else{
            const int length = sequence.length();
            std::memcpy(ptr, &length, sizeof(int));
            ptr += sizeof(int);
            std::memcpy(ptr, sequence.c_str(), sizeof(char) * length);
            ptr += sizeof(char) * length;
        }

        if(type == TempCorrectedSequence::Type::Anchor){
            // const auto& vec = uncorrectedPositionsNoConsensus;
            // sstream << vec.size();
            // if(!vec.empty()){
            //     sstream << ' ';
            //     std::copy(vec.begin(), vec.end(), std::ostream_iterator<int>(sstream, " "));
            // }
        }else{
            std::memcpy(ptr, &shift, sizeof(int));
            ptr += sizeof(int);
        }
    }

    EncodedTempCorrectedSequence TempCorrectedSequence::encode() const{
        EncodedTempCorrectedSequence encoded;
        encodeInto(encoded);

        return encoded;
    }

    void TempCorrectedSequence::decode(const EncodedTempCorrectedSequence& encoded){

        readId = encoded.readId;

        hq = (encoded.encodedflags >> 31) & std::uint32_t(1);
        useEdits = (encoded.encodedflags >> 30) & std::uint32_t(1);
        type = TempCorrectedSequence::Type((encoded.encodedflags >> 29) & std::uint32_t(1));

        const std::uint8_t* ptr = encoded.data.get();
    

        if(useEdits){
            int size;
            std::memcpy(&size, ptr, sizeof(int));
            ptr += sizeof(int);

            edits.resize(size);

            for(auto& edit : edits){
                std::memcpy(&edit.pos, ptr, sizeof(int));
                ptr += sizeof(int);
            }
            for(auto& edit : edits){
                std::memcpy(&edit.base, ptr, sizeof(char));
                ptr += sizeof(char);
            }
        }else{
            int length;
            std::memcpy(&length, ptr, sizeof(int));
            ptr += sizeof(int);

            sequence.resize(length);
            sequence.replace(0, length, (const char*)ptr, length);

            ptr += sizeof(char) * length;
        }

        if(type == TempCorrectedSequence::Type::Anchor){
            // size_t vecsize;
            // sstream >> vecsize;
            // if(vecsize > 0){
            //     auto& vec = uncorrectedPositionsNoConsensus;
            //     vec.resize(vecsize);
            //     for(size_t i = 0; i < vecsize; i++){
            //         sstream >> vec[i];
            //     }
            // }
        }else{
            std::memcpy(&shift, ptr, sizeof(int));
            ptr += sizeof(int);
        }
    }

    bool TempCorrectedSequence::writeToBinaryStream(std::ostream& os) const{
        if(tmpresultfileformat == 0){
            os << readId << ' ';
        }else if(tmpresultfileformat == 1){
            os.write(reinterpret_cast<const char*>(&readId), sizeof(read_number));
        }
        
        std::uint8_t data = bool(hq);
        data = (data << 1) | bool(useEdits);
        data = (data << 6) | std::uint8_t(int(type));
        
        if(tmpresultfileformat == 0){
            os << data << ' ';
        }else if(tmpresultfileformat == 1){
            os.write(reinterpret_cast<const char*>(&data), sizeof(std::uint8_t));
        }

        if(useEdits){
            os << edits.size() << ' ';
            for(const auto& edit : edits){
                os << edit.pos << ' ';
            }
            for(const auto& edit : edits){
                os << edit.base;
            }
            if(edits.size() > 0){
                os << ' ';
            }
        }else{
            os << sequence << ' ';
        }

        if(type == TempCorrectedSequence::Type::Anchor){
            const auto& vec = uncorrectedPositionsNoConsensus;
            os << vec.size();
            if(!vec.empty()){
                os << ' ';
                std::copy(vec.begin(), vec.end(), std::ostream_iterator<int>(os, " "));
            }
        }else{
            os << shift;
        }

        return bool(os);
    }

    bool TempCorrectedSequence::readFromBinaryStream(std::istream& is){
        std::uint8_t data = 0;

        if(tmpresultfileformat == 0){
            is >> readId;
        }else if(tmpresultfileformat == 1){
            is.read(reinterpret_cast<char*>(&readId), sizeof(read_number));
            is.read(reinterpret_cast<char*>(&data), sizeof(std::uint8_t));
        }

        std::string line;
        if(std::getline(is, line)){
            std::stringstream sstream(line);
            auto& stream = sstream;

            if(tmpresultfileformat == 0){
                stream >> data; 
            }
            
            hq = (data >> 7) & 1;
            useEdits = (data >> 6) & 1;
            type = TempCorrectedSequence::Type(int(data & 0x3F));

            if(useEdits){
                size_t size;
                stream >> size;
                int numEdits = size;
                edits.resize(size);
                for(int i = 0; i < numEdits; i++){
                    stream >> edits[i].pos;
                }
                for(int i = 0; i < numEdits; i++){
                    stream >> edits[i].base;
                }
            }else{
                stream >> sequence;
            }

            if(type == TempCorrectedSequence::Type::Anchor){
                size_t vecsize;
                stream >> vecsize;
                if(vecsize > 0){
                    auto& vec = uncorrectedPositionsNoConsensus;
                    vec.resize(vecsize);
                    for(size_t i = 0; i < vecsize; i++){
                        stream >> vec[i];
                    }
                }
            }else{
                stream >> shift;
                shift = std::abs(shift);
            }
        }

        return bool(is);
    }

    



    std::ostream& operator<<(std::ostream& os, const TempCorrectedSequence& tmp){
        //tmp.writeToBinaryStream(os);
        os << "readid = " << tmp.readId << ", type = " << int(tmp.type) << ", hq = " << tmp.hq 
            << ", useEdits = " << tmp.useEdits << ", numEdits = " << tmp.edits.size();
        if(tmp.edits.size() > 0){
            for(const auto& edit : tmp.edits){
                os << " , (" << edit.pos << "," << edit.base << ")";
            }
        }

        return os;
    }

    std::istream& operator>>(std::istream& is, TempCorrectedSequence& tmp){
        tmp.readFromBinaryStream(is);
        return is;
    }




}