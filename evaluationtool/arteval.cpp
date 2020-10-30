
//#include "sequencereader.hpp"
#include "kseqpp.hpp"

#include <iostream>
#include <string>
#include <fstream>
#include <cstdint>
#include <cassert>
#include <sstream>
#include <vector>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cctype>
#include <map>
#include <algorithm>

using namespace kseqpp;

struct Read {
    std::int64_t readNumber = -1;
	std::string header = "";
	std::string sequence = "";
	std::string quality = "";

    bool operator==(const Read& other) const{
        return (readNumber == other.readNumber 
                && header == other.header 
                && sequence == other.sequence 
                && quality == other.quality);
    }

    bool operator!=(const Read& other) const{
        return !(*this == other);
    }

    void reset(){
        readNumber = -1;
        header.clear();
        sequence.clear();
        quality.clear();
    }
};

template<class Func>
void forEachReadVectorInFiles(const std::vector<std::string>& filenames, Func&& consumeReadVector){
    const int numFiles = filenames.size();
    std::vector<std::unique_ptr<KseqPP>> readers(numFiles);
    
    for(int i = 0; i < numFiles; i++){
        readers[i].reset(new KseqPP(filenames[i]));
    }
    

    std::vector<Read> reads(numFiles);
    std::int64_t readNumber = 0;
    
    std::uint64_t checkedNumReads = 0;
    std::uint64_t progressCounter = 0;

    auto getNextReads = [&](){
        bool success = true;

        for(int i = 0; i < numFiles && success; i++){
            int status = readers[i]->next();
            //std::cerr << "parser status = 0 in file " << filenames[i] << '\n';
            if(status >= 0){
                reads[i].readNumber = readNumber;
                std::swap(reads[i].header, readers[i]->getCurrentHeader());
                std::swap(reads[i].sequence, readers[i]->getCurrentSequence());
                std::swap(reads[i].quality, readers[i]->getCurrentQuality());
            }else if(status < -1){
                std::cerr << "parser error status " << status << " in file " << filenames[i] << '\n';
            }
            success &= (status >= 0);
        }
        readNumber++;
        
        checkedNumReads++;
        progressCounter++;
        
        if(progressCounter == 10000000){
            std::cerr << "checked " << checkedNumReads << " reads\n";
            progressCounter = 0;
        }

        return success;
    };

    bool success = getNextReads();

    while(success){

        consumeReadVector(reads);

        reads.resize(numFiles);

        success = getNextReads();
    }
}


std::uint64_t linecount(const std::string& filename){
	std::uint64_t count = 0;
	std::ifstream is(filename);
	if(is){
		std::string s;
		while(std::getline(is, s)){
			++count;
		}
	}
	return count;
}

std::vector<std::string> split(const std::string& str, char c){
	std::vector<std::string> result;

	std::stringstream ss(str);
	std::string s;

	while (std::getline(ss, s, c)) {
		result.emplace_back(s);
	}

	return result;
}

char my_toupper(char ch){
    return std::toupper(static_cast<unsigned char>(ch));
}

int main(int argc, char** argv){
	if(argc < 4){
		std::cout << "Usage: " << argv[0] << " originalReads perfectReads correctedReads [correctedReads...]" << std::endl;
		return 0;
	}

    std::vector<std::string> filenames;
    for(int i = 1; i < argc; i++){
        filenames.emplace_back(argv[i]);
        std::ifstream is(filenames.back());
        if(!is){
            std::cout << "Error opening " << filenames.back() << std::endl;
        }
    }

    const int nCorrectedFiles = filenames.size() - 2;

    std::uint64_t checkedNumReads = 0;

    std::string tmpline;

    std::vector<std::uint64_t> tp(filenames.size()-1, 0);
    std::vector<std::uint64_t> fp(filenames.size()-1, 0);
    std::vector<std::uint64_t> fn(filenames.size()-1, 0);
    std::vector<std::uint64_t> tn(filenames.size()-1, 0);

    std::vector<std::map<int,int>> remainingErrorsInReads(nCorrectedFiles);

    forEachReadVectorInFiles(filenames, [&](const auto& reads){

        const auto& originalSeq = reads[0].sequence;
        const auto& perfectSeq = reads[1].sequence;

        const bool allSameLength = std::all_of(reads.begin() + 1, reads.end(), [&](const auto& read){
            return read.sequence.length() == originalSeq.length();
        });

        if(!allSameLength){
            for(const auto& read : reads){
                std::cout << read.sequence.length() << std::endl;
            }
        }

        assert(allSameLength);

        std::vector<int> mismatchesToPerfectRead(nCorrectedFiles, 0);

        const int readLength = originalSeq.length();

        //make count tp fp
        for(int i = 0; i < readLength; i++){
    		const char r = my_toupper(originalSeq[i]);
			const char p = my_toupper(perfectSeq[i]);

            for(int corrId = 0; corrId < nCorrectedFiles; corrId++){
                const int index = corrId + 2;
                const char c = my_toupper(reads[index].sequence[i]);

                if(r != p && c == p){
				    tp[corrId] += 1;
                }else if(r == p && c == p){
                    tn[corrId] += 1;
                }else if(r == p && c != p){
                    fp[corrId] += 1;
                    mismatchesToPerfectRead[corrId]++;
                }else if(r != p && c != p){
                    fn[corrId] += 1;
                    mismatchesToPerfectRead[corrId]++;
                }
            }			
        }

        for(int corrId = 0; corrId < nCorrectedFiles; corrId++){
            remainingErrorsInReads[corrId][mismatchesToPerfectRead[corrId]]++;
        }

        checkedNumReads++;
    });
    
    std::cout << "Checked " << checkedNumReads << " reads." << std::endl;

    for(int corrId = 0; corrId < nCorrectedFiles; corrId++){
        const int index = corrId + 2;
        const std::uint64_t myTP = tp[corrId];
        const std::uint64_t myFP = fp[corrId];
        const std::uint64_t myFN = fn[corrId];
        const std::uint64_t myTN = tn[corrId];
        std::cout << filenames[index] << std::endl;
        std::cout << "TP: " << myTP << std::endl;
        std::cout << "FP: " << myFP << std::endl;
        std::cout << "FN: " << myFN << std::endl;
        std::cout << "TN: " << myTN << std::endl;
        std::cout << "Total: " << (myTP+myFP+myFN+myTN) << std::endl;

        float gain = (myTP+myFN) == 0 ? 0 : (float(myTP) - myFP) / (myTP + myFN);
        float sensitivity = (myTP+myFN) == 0 ? 0 : (float(myTP)) / (myTP + myFN);
        float specificity = (myFP+myTN) == 0 ? 0 : (float(myTN) / (myFP + myTN));

        std::cout << "Gain: " << gain << std::endl;
        std::cout << "Sensitivity: " << sensitivity << std::endl;
        std::cout << "Specificity: " << specificity << std::endl;

        std::cout << "Remaining errors per read | frequency | frequency accumulated\n";
        int threshold = 10;
        int overThreshold = 0;
        int accumulated = 0;
        for(auto pair : remainingErrorsInReads[corrId]){
            accumulated += pair.second;
            if(pair.first <= threshold){
                std::cout << pair.first << " | " << pair.second << " | " << accumulated << '\n';
            }else{
                overThreshold += pair.second;
            }            
        }
        std::cout << ">" << threshold << " | " << overThreshold << " | " << accumulated << '\n';

        if(corrId < nCorrectedFiles - 1){
            std::cout << "------------------------------------\n";
        }
    }

}
