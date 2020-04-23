#include <correctionresultprocessing.hpp>

#include <config.hpp>
#include <hpc_helpers.cuh>
#include <memoryfile.hpp>
#include <readlibraryio.hpp>
#include <threadpool.hpp>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>

namespace care{




struct SyncFlag{
    std::atomic<bool> busy{false};
    std::mutex m;
    std::condition_variable cv;

    void setBusy(){
        assert(busy == false);
        busy = true;
    }

    bool isBusy() const{
        return busy;
    }

    void wait(){
        if(isBusy()){
            std::unique_lock<std::mutex> l(m);
            while(isBusy()){
                cv.wait(l);
            }
        }
    }

    void signal(){
        std::unique_lock<std::mutex> l(m);
        busy = false;
        cv.notify_all();
    }        
};

template<class T>
struct WaitableData{
    T data;
    SyncFlag syncFlag;

    void setBusy(){
        syncFlag.setBusy();
    }

    bool isBusy() const{
        return syncFlag.isBusy();
    }

    void wait(){
        syncFlag.wait();
    }

    void signal(){
        syncFlag.signal();
    } 
};





template<class MemoryFile_t>
void mergeResultFiles2_impl(
                    const std::string& tempdir,
                    const std::vector<std::string>& originalReadFiles,
                    MemoryFile_t& partialResults, 
                    std::size_t memoryForSorting,
                    FileFormat outputFormat,
                    const std::vector<std::string>& outputfiles,
                    bool isSorted){

    assert(outputfiles.size() == 1 || originalReadFiles.size() == outputfiles.size());

    bool oldsyncflag = true;//std::ios::sync_with_stdio(false);

    //sort the result files and save sorted result file in tempfile.
    //Then, merge original file and tempfile, replacing the reads in
    //original file by the corresponding reads in the tempfile.

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

        TIMERSTARTCPU(sort_during_merge);
        partialResults.sort(tempdir, memoryForSorting, ptrcomparator, elementcomparator);
        TIMERSTOPCPU(sort_during_merge);
    }

    //loop over correction sequences
    TIMERSTARTCPU(actualmerging);

    auto isValidSequence = [](const std::string& s){
        return std::all_of(s.begin(), s.end(), [](char c){
            return (c == 'A' || c == 'C' || c == 'G' || c == 'T' || c == 'N');
        });
    };

    int numberOfHQCorrections = 0;
    int numberOfEqualHQCorrections = 0;
    int numberOfLQCorrections = 0;
    int numberOfUsableLQCorrectionsWithCandidates = 0;
    int numberOfUsableLQCorrectionsOnlyAnchor = 0;

    #if 0
    auto combineMultipleCorrectionResults2 = [](std::vector<TempCorrectedSequence>& tmpresults, const std::string& originalSequence){
        assert(!tmpresults.empty());
        assert(false && "uncorrectedpositions cannot be used");

        constexpr bool outputHQ = true;
        constexpr bool outputLQAnchorDifferentCand = false;
        constexpr bool outputLQAnchorSameCand = false;
        constexpr bool outputLQAnchorNoCand = false;
        constexpr bool outputLQOnlyCand = false;

        auto isHQ = [](const auto& tcs){
            return tcs.type == TempCorrectedSequence::Type::Anchor && tcs.hq;
        };

        //if there is a correction using a high quality alignment, use it
        auto firstHqSequence = std::find_if(tmpresults.begin(), tmpresults.end(), isHQ);
        if(firstHqSequence != tmpresults.end()){
            assert(firstHqSequence->sequence.size() == originalSequence.size());
            return std::make_pair(firstHqSequence->sequence, outputHQ);
        }

        auto equalsFirstSequence = [&](const auto& result){
            return result.sequence == tmpresults[0].sequence;
        };

        auto getSequence = [&](int index){
            return tmpresults[index].sequence;
        };

        auto anchorIter = std::find_if(tmpresults.begin(), tmpresults.end(), [](const auto& r){
            return r.type == TempCorrectedSequence::Type::Anchor;
        });

        if(!std::all_of(tmpresults.begin()+1, tmpresults.end(), equalsFirstSequence)){
            // std::copy(sequences.begin(), sequences.end(), std::ostream_iterator<std::string>(std::cerr, "\n"));
            // std::cerr << "\n";
            // std::exit(0);
            std::string consensus(getSequence(0).size(), 'F');
            std::vector<int> countsA(getSequence(0).size(), 0);
            std::vector<int> countsC(getSequence(0).size(), 0);
            std::vector<int> countsG(getSequence(0).size(), 0);
            std::vector<int> countsT(getSequence(0).size(), 0);

            auto countBases = [&](const auto& result){
                const auto& sequence = result.sequence;
                assert(sequence.size() == consensus.size());
                for(size_t i = 0; i < sequence.size();  i++){
                    const char c = sequence[i];
                    if(c == 'A') countsA[i]++;
                    else if(c == 'C') countsC[i]++;
                    else if(c == 'G') countsG[i]++;
                    else if(c == 'T') countsT[i]++;
                    else if(c == 'N'){
                        ;
                    }else{
                        std::cerr << result.readId << " : " << sequence << "\n"; assert(false);
                    }
                }
            };

            auto findConsensusOfPosition = [&](int i){
                int count = countsA[i];
                char c = 'A';
                if(countsC[i] > count){
                    count = countsC[i];
                    c = 'C';
                }
                if(countsG[i] > count){
                    count = countsG[i];
                    c = 'G';
                }
                if(countsT[i] > count){
                    count = countsT[i];
                    c = 'T';
                }
                return c;
            };

            auto setConsensusOfPosition = [&](int position){
                consensus[position] = findConsensusOfPosition(position);
            };



            if(anchorIter != tmpresults.end() && (anchorIter->uncorrectedPositionsNoConsensus.size() >= 5)){

                const int maxShiftInResult = std::max_element(tmpresults.begin(),
                                                            tmpresults.end(),
                                                            [](const auto& l, const auto& r){
                                                                return l.shift < r.shift;
                                                            })->shift;
                // if(maxShiftInResult > 3){
                //     return std::make_pair(std::string{""}, false);
                // }
                // //return std::make_pair(anchorIter->sequence, true);
                // for(const auto& t : tmpresults){
                //     if(t.shift <= 3){
                //         //const int iters = maxShiftInResult - t.shift + 1;
                //         //for(int k = 0; k < iters; k++){
                //             countBases(t);
                //         //}
                //     }
                // }
                std::for_each(tmpresults.begin(), tmpresults.end(), countBases);

                // if(!anchorIter->uncorrectedPositionsNoConsensus.empty()){
                //     std::copy(anchorIter->sequence.begin(), anchorIter->sequence.end(), consensus.begin());
                //     const auto& positions = anchorIter->uncorrectedPositionsNoConsensus;
                //
                //     std::for_each(positions.begin(), positions.end(), setConsensusOfPosition);
                //     // std::copy(positions.begin(), positions.end(), std::ostream_iterator<int>(std::cerr, " "));
                //     // std::cerr << '\n';
                //     // std::copy(consensus.begin(), consensus.end(), std::ostream_iterator<char>(std::cerr, ""));
                //     // std::cerr << '\n';
                //     return std::make_pair(consensus, false);
                // }else{
                //     for(size_t i = 0; i < consensus.size();  i++){
                //         setConsensusOfPosition(i);
                //     }
                //     return std::make_pair(consensus, false);
                // }

                for(size_t i = 0; i < consensus.size();  i++){
                    setConsensusOfPosition(i);
                }
                return std::make_pair(consensus, outputLQAnchorDifferentCand);

            }else{
                //only candidates available

                return std::make_pair(std::string{""}, false); //always false
            }

        }else{
            //return std::make_pair(tmpresults[0].sequence, false);

            if(anchorIter != tmpresults.end()){
                //return std::make_pair(anchorIter->sequence, true);
                auto checkshift = [](const auto& r){
                    return r.type == TempCorrectedSequence::Type::Candidate && std::abs(r.shift) <= 15;
                    //return true;
                };
                if(0 < std::count_if(tmpresults.begin(), tmpresults.end(), checkshift)){
                    return std::make_pair(tmpresults[0].sequence, outputLQAnchorSameCand);
                }else{
                    if(tmpresults.size() == 1){
                        if(tmpresults[0].uncorrectedPositionsNoConsensus.size() < 1){
                            return std::make_pair(tmpresults[0].sequence, outputLQAnchorNoCand);
                        }else{
                            return std::make_pair(std::string{""}, false); //always false
                        }

                    }else{
                        return std::make_pair(std::string{""}, false); //always false
                    }
                }
            }else{
                //no correction as anchor. all corrections as candidate are equal.
                //only use the correction if at least one correction as candidate was performed with 0 new columns
                auto checkshift = [](const auto& r){
                    return std::abs(r.shift) <= 1;
                };
                if(0 < std::count_if(tmpresults.begin(), tmpresults.end(), checkshift)){
                    return std::make_pair(tmpresults[0].sequence, outputLQOnlyCand);
                }else{
                    return std::make_pair(std::string{""}, false); //always false
                }

                //return std::make_pair(std::string{""}, false);

                //return std::make_pair(tmpresults[0].sequence, true);
            }
        }

        //return tmpresults[0].sequence;
    };
    #endif 

    #if 0
    auto combineMultipleCorrectionResults3 = [](std::vector<TempCorrectedSequence>& tmpresults, const std::string& originalSequence){
        assert(!tmpresults.empty());

        constexpr bool outputHQ = true;
        constexpr bool outputLQ = true;

        auto isAnchor = [](const auto& tcs){
            return tcs.type == TempCorrectedSequence::Type::Anchor;
        };

        auto anchorIter = std::find_if(tmpresults.begin(), tmpresults.end(), isAnchor);

        if(anchorIter != tmpresults.end()){
            //if there is a correction using a high quality alignment, use it
            if(anchorIter->hq){
                assert(anchorIter->sequence.size() == originalSequence.size());
                return std::make_pair(anchorIter->sequence, outputHQ);
            }else{

                const TempCorrectedSequence anchor = *anchorIter;
                tmpresults.erase(anchorIter);

                tmpresults.erase(std::remove_if(tmpresults.begin(),
                                                tmpresults.end(),
                                                [](const auto& tcs){
                                                    return std::abs(tcs.shift) > 5;
                                                }),
                                  tmpresults.end());

                if(tmpresults.size() > 3){

                    const bool sameCorrections = std::all_of(tmpresults.begin()+1,
                                                            tmpresults.end(),
                                                            [&](const auto& tcs){
                                                                return tmpresults[0].sequence == tcs.sequence;
                                                            });

                    if(sameCorrections){
                        return std::make_pair(tmpresults[0].sequence, outputLQ);
                    }else{
                        return std::make_pair(std::string{""}, false);
                    }
                }else{
                    return std::make_pair(std::string{""}, false);
                }
            }
        }else{
            return std::make_pair(std::string{""}, false);
        }

    };
    #endif 

    #if 1
    auto combineMultipleCorrectionResults4NewHQLQ = [&](std::vector<TempCorrectedSequence>& tmpresults, const std::string& originalSequence){
        assert(!tmpresults.empty());

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
                numberOfHQCorrections++;

                assert(anchorIter->sequence.size() == originalSequence.size());
                return std::make_pair(anchorIter->sequence, outputHQ);
            }else{
                numberOfLQCorrections++;

                const TempCorrectedSequence anchor = *anchorIter;
                //tmpresults.erase(anchorIter);

                // tmpresults.erase(std::remove_if(tmpresults.begin(),
                //                                 tmpresults.end(),
                //                                 [](const auto& tcs){
                //                                     return std::abs(tcs.shift) > 5;
                //                                 }),
                //                   tmpresults.end());

                if(tmpresults.size() >= 3){

                    const bool sizelimitok = true; //tmpresults.size() > 3;

                    const bool sameCorrections = std::all_of(tmpresults.begin()+1,
                                                            tmpresults.end(),
                                                            [&](const auto& tcs){
                                                                return tmpresults[0].sequence == tcs.sequence;
                                                            });

                    if(sameCorrections && sizelimitok){
                        numberOfUsableLQCorrectionsWithCandidates++;

                        return std::make_pair(tmpresults[0].sequence, outputLQWithCandidates);
                    }else{
                        return std::make_pair(std::string{""}, false);
                    }
                }else{
                    numberOfUsableLQCorrectionsOnlyAnchor++;
                    //return std::make_pair(std::string{""}, false);
                    return std::make_pair(anchor.sequence, outputLQOnlyAnchor);
                }
            }
        }else{

            return std::make_pair(std::string{""}, false);
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
    #endif 

    auto combineMultipleCorrectionResultsFunction = combineMultipleCorrectionResults4NewHQLQ;


    struct Task{
        read_number readId;
        SequenceFileWriter* writer;
        Read read;

        // Task() = default;
        // Task(read_number id, SequenceFileWriter* w, Read r){
        //     readId = id;
        //     writer = w;
        //     read = std::move(r);
        // }
        
    };

    auto processTask = [&](auto& task){
        task.writer->writeRead(task.read.name, task.read.comment, task.read.sequence, task.read.quality);
    };

    const std::size_t batchsizetasks = 10000;
    //std::vector<Task> writeTasks;

    std::array<WaitableData<std::vector<Task>>, 2> taskdblbuf;
    int dblbufindex = 0;

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
    for(const auto& inputfile : originalReadFiles){
        inputReaderVector.emplace_back(std::move(kseqpp::KseqPP{inputfile}));
    }

    bool inputReaderIsValid = true;
    int inputFileId = 0;
    int outputFileId = 0;

    const int numFiles = originalReadFiles.size();
    Read read;
    std::uint64_t originalReadId = 0;
    bool firstiter = true;

    auto updateRead = [&](Read& read){
        read.name = inputReaderVector[inputFileId].getCurrentName();
        read.comment = inputReaderVector[inputFileId].getCurrentComment();
        read.sequence = inputReaderVector[inputFileId].getCurrentSequence();
        read.quality = inputReaderVector[inputFileId].getCurrentQuality();
    };

    std::chrono::time_point<std::chrono::system_clock> timebegin, timeend;

    std::chrono::duration<double> durationInOutputThread{0};
    std::chrono::duration<double> durationPrload{0};
    std::chrono::duration<double> durationCopyOrig{0};
    std::chrono::duration<double> durationConstruction{0};

    int origLessThanCurrent = 0;

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
            std::cerr << "partial results empty with currentReadId = " << currentReadId << "\n";
        }

        timeend = std::chrono::system_clock::now();

        durationPrload += timeend - timebegin;

        timebegin = std::chrono::system_clock::now();

        //copy preceding reads from original file
        while(originalReadId < currentReadId){
            origLessThanCurrent++;
            const int status = inputReaderVector[inputFileId].next();
            inputReaderIsValid = status >= 0;

            if(inputReaderIsValid){
                // Task writeTask;
                // writeTask.readId = originalReadId;
                // writeTask.writer = writerVector[outputFileId].get();
                // updateRead(writeTask.read);
                // // writeTask.read = std::move(read);
                // //processTask(writeTask);

                // taskdblbuf[dblbufindex].data.emplace_back(std::move(writeTask));

                // if(taskdblbuf[dblbufindex].data.size() >= batchsizetasks){
                //     taskdblbuf[dblbufindex].setBusy();

                //     outputThread.enqueue(
                //         [&, index = dblbufindex](){
                //             auto a = std::chrono::system_clock::now();
                            
                //             for(const auto& task : taskdblbuf[index].data){
                //                 processTask(task);
                //             }
                //             taskdblbuf[index].data.clear();
                //             taskdblbuf[index].signal();
                //             auto b = std::chrono::system_clock::now();

                //             durationInOutputThread += b-a;
                //         }
                //     ); 

                //     dblbufindex = 1 - dblbufindex; 
                //     taskdblbuf[dblbufindex].wait(); //avoid overtaking output thread
                // }


                updateRead(read);
                writerVector[outputFileId]->writeRead(read);

                originalReadId++;
            }else{
                inputFileId++;
                if(inputFileId >= numFiles){
                    throw std::runtime_error{"Cannot skip to read " + std::to_string(currentReadId)
                        + " during merge."};
                }

                if(outputfiles.size() > 1){
                    outputFileId++;
                }
            }
        }

        timeend = std::chrono::system_clock::now();

        durationCopyOrig += timeend - timebegin;

        timebegin = std::chrono::system_clock::now();

        //get read with id currentReadId
        int status = 0;
        do{
            status = inputReaderVector[inputFileId].next();
            inputReaderIsValid = status >= 0;
            if(inputReaderIsValid){
                updateRead(read);
                originalReadId++;
                break;
            }else{
                inputFileId++;
                if(inputFileId >= numFiles){
                    throw std::runtime_error{"Could not find read " + std::to_string(currentReadId)
                        + " during merge."};
                }

                if(outputfiles.size() > 1){
                    outputFileId++;
                }
            }
        }while(true);

        assert(inputReaderIsValid);

        // Task constructionTask;
        // constructionTask.readId = currentReadId;
        // constructionTask.writer = writerVector[outputFileId].get();
        // constructionTask.correctionVector = std::move(correctionVector);
        // constructionTask.read = std::move(read);


        // constructionTasks.emplace_back(std::move(constructionTask));

        // if(constructionTasks.size() >= batchsizetasks){
        //     for(auto& task : constructionTasks){
        //         processTask(task);
        //     }
        //     constructionTasks.clear();
        // }

        //replace sequence of next read with corrected sequence
        

        

        for(auto& tmpres : correctionVector){
            if(tmpres.useEdits){
                tmpres.sequence = read.sequence;
                for(const auto& edit : tmpres.edits){
                    tmpres.sequence[edit.pos] = edit.base;
                }
            }
        }

        
        
        auto correctedSequence = combineMultipleCorrectionResultsFunction(
            correctionVector, 
            read.sequence
        );



        if(correctedSequence.second){
            if(!isValidSequence(correctedSequence.first)){
                std::cerr << "Warning. Corrected read " << currentReadId
                        << " with header " << read.name << " " << read.comment
                        << "does contain an invalid DNA base!\n"
                        << "Corrected sequence is: "  << correctedSequence.first << '\n';
            }
            read.sequence = std::move(correctedSequence.first);
        }      

        

        writerVector[outputFileId]->writeRead(read.name, read.comment, read.sequence, read.quality);

        // Task writeTask;
        // writeTask.readId = currentReadId;
        // writeTask.writer = writerVector[outputFileId].get();
        // writeTask.read = std::move(read);

        //processTask(writeTask);
        // writeTasks.emplace_back(std::move(writeTask));

        // if(writeTasks.size() >= batchsizetasks){
        //     for(const auto& task : writeTasks){
        //         processTask(task);
        //     }
        //     writeTasks.clear();
        // }



        //taskdblbuf[dblbufindex].data.emplace_back(std::move(writeTask));

        timeend = std::chrono::system_clock::now();
        durationConstruction += timeend - timebegin;


        // if(taskdblbuf[dblbufindex].data.size() >= batchsizetasks){
        //     taskdblbuf[dblbufindex].setBusy();

        //     outputThread.enqueue(
        //         [&, index = dblbufindex](){
        //             auto a = std::chrono::system_clock::now();
                    
        //             for(const auto& task : taskdblbuf[index].data){
        //                 processTask(task);
        //             }
        //             taskdblbuf[index].data.clear();
        //             taskdblbuf[index].signal();
        //             auto b = std::chrono::system_clock::now();

        //             durationInOutputThread += b-a;
        //         }
        //     ); 

        //     dblbufindex = 1 - dblbufindex; 
        //     taskdblbuf[dblbufindex].wait(); //avoid overtaking output thread
        // }

        // if(correctedSequence.second){
        //     //assert(isValidSequence(correctedSequence.first));
        //     if(!isValidSequence(correctedSequence.first)){
        //         std::cerr << "Warning. Corrected read " << currentReadId
        //                 << " with header " << read.name << " " << read.comment
        //                 << "does contain an invalid DNA base!\n"
        //                 << "Corrected sequence is: "  << correctedSequence.first << '\n';
        //     }
        //     writerVector[outputFileId]->writeRead(read.name, read.comment, correctedSequence.first, read.quality);
        // }else{
        //     writerVector[outputFileId]->writeRead(read.name, read.comment, read.sequence, read.quality);
        // }

        correctionVector.clear();
        std::swap(correctionVector, correctionVector_tmp);
        std::swap(currentReadId, currentReadId_tmp);


        firstiter = false;
    }

    TIMERSTARTCPU(waitforoutputthread);
    taskdblbuf[0].wait();
    taskdblbuf[1].wait();
    TIMERSTOPCPU(waitforoutputthread);

    std::cerr << "origLessThanCurrent = " << origLessThanCurrent << "\n";

    std::cout << "# elapsed time (durationInOutputThread): " << durationInOutputThread.count()  << " s" << std::endl;
    std::cout << "# elapsed time (durationPrload): " << durationPrload.count()  << " s" << std::endl;
    std::cout << "# elapsed time (durationCopyOrig): " << durationCopyOrig.count()  << " s" << std::endl;
    std::cout << "# elapsed time (durationConstruction): " << durationConstruction.count()  << " s" << std::endl;

    // TIMERSTARTCPU(processremainingtasks);
    // for(const auto& task : taskdblbuf[dblbufindex].data){
    //     processTask(task);
    // }
    // taskdblbuf[dblbufindex].data.clear();
    // TIMERSTOPCPU(processremainingtasks);

    // for(const auto& task : writeTasks){
    //     processTask(task);
    // }
    // writeTasks.clear();

    // for(auto& task : constructionTasks){
    //     processTask(task);
    // }
    // constructionTasks.clear();

    TIMERSTARTCPU(copyremainingreads);
    //copy remaining reads from original files
    while((inputReaderIsValid = (inputReaderVector[inputFileId].next() >= 0)) && inputFileId < numFiles){
        if(inputReaderIsValid){
            updateRead(read);
            writerVector[outputFileId]->writeRead(read);
            originalReadId++;
        }else{
            inputFileId++;

            if(outputfiles.size() > 1){
                outputFileId++;
            }
        }
    }
    TIMERSTOPCPU(copyremainingreads);

    outputThread.stopThread(BackgroundThread::StopType::FinishAndStop);

    TIMERSTOPCPU(actualmerging);

    std::cerr << "numberOfHQCorrections " << numberOfHQCorrections << "\n";
    std::cerr << "numberOfEqualHQCorrections " << numberOfEqualHQCorrections << "\n";
    std::cerr << "numberOfLQCorrections " << numberOfLQCorrections << "\n";
    std::cerr << "numberOfUsableLQCorrectionsWithCandidates " << numberOfUsableLQCorrectionsWithCandidates << "\n";
    std::cerr << "numberOfUsableLQCorrectionsOnlyAnchor " << numberOfUsableLQCorrectionsOnlyAnchor << "\n";

    //deleteFiles({tempfile});

    std::ios::sync_with_stdio(oldsyncflag);
}


void constructOutputFileFromResults2(
                    const std::string& tempdir,
                    const std::vector<std::string>& originalReadFiles,
                    MemoryFileFixedSize<EncodedTempCorrectedSequence>& partialResults, 
                    std::size_t memoryForSorting,
                    FileFormat outputFormat,
                    const std::vector<std::string>& outputfiles,
                    bool isSorted){
                        
    mergeResultFiles2_impl(
        tempdir, 
        originalReadFiles, 
        partialResults, 
        memoryForSorting, 
        outputFormat,
        outputfiles, 
        isSorted
    );
}



    TempCorrectedSequence::TempCorrectedSequence(const EncodedTempCorrectedSequence& encoded){
        decode(encoded);
    }

    TempCorrectedSequence& TempCorrectedSequence::operator=(const EncodedTempCorrectedSequence& encoded){
        decode(encoded);
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

    EncodedTempCorrectedSequence TempCorrectedSequence::encode() const{
        EncodedTempCorrectedSequence encoded;
        encoded.readId = readId;

        encoded.encodedflags = (std::uint32_t(hq) << 31);
        encoded.encodedflags |= (std::uint32_t(useEdits) << 30);
        encoded.encodedflags |= (std::uint32_t(int(type)) << 29);

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
        encoded.encodedflags |= numBytes;

        encoded.data = std::make_unique<std::uint8_t[]>(numBytes);

        //fill buffer

        std::uint8_t* ptr = encoded.data.get();

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