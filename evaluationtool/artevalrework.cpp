
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
#include <array>

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

using NucStats = std::array<std::uint64_t, 4>;

constexpr std::size_t tp_index = 0;
constexpr std::size_t fp_index = 1;
constexpr std::size_t fn_index = 2;
constexpr std::size_t tn_index = 3;

struct StatsPerQuality{
    NucStats nucStats{};
    std::map<int,std::uint64_t> remainingErrorsInReads{};
};

struct StatsOfFile{
    std::map<int, StatsPerQuality> statsMap{};
};

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
    bool includeCorrectionQualityLabel = true;

    std::uint64_t checkedNumReads = 0;

    std::string tmpline;
    StatsOfFile stats{};

    forEachReadVectorInFiles(filenames, [&](const auto& reads){
        assert(reads.size() == 3);
        const auto& originalSeq = reads[0].sequence;
        const auto& perfectSeq = reads[1].sequence;

        const bool allSameLength = std::all_of(reads.begin() + 1, reads.end(), [&](const auto& read){
            return read.sequence.length() == originalSeq.length();
        });

        if(!allSameLength){
            for(const auto& read : reads){
                std::cout << read.header << " - " << read.sequence.length() << std::endl;
            }
            assert(allSameLength);
        }
        const int readLength = originalSeq.length();

        bool hasCorrectionQuality = false;
        int correctionQuality = 42;

        const std::string& header = reads[2].header;
        auto pos1 = header.find("care:");
        if(pos1 != std::string::npos){
            auto careflags = header.substr(pos1 + 5);
            auto kvstrings = split(careflags, ' ');
            for(const auto& s : kvstrings){
                auto tokens = split(s, '=');
                if(tokens[0] == "q"){
                    assert(tokens.size() == 2);
                    correctionQuality = std::stoi(tokens[1]);
                    hasCorrectionQuality = true;
                    break;
                }
            }
        }

        int mismatchesToPerfectRead = 0;
        for(int i = 0; i < readLength; i++){
            const char r = my_toupper(originalSeq[i]);
            const char p = my_toupper(perfectSeq[i]);
            const char c = my_toupper(reads[2].sequence[i]);

            if(r != p && c == p){
                stats.statsMap[correctionQuality].nucStats[tp_index]++;
            }else if(r == p && c == p){
                stats.statsMap[correctionQuality].nucStats[tn_index]++;
            }else if(r == p && c != p){
                stats.statsMap[correctionQuality].nucStats[fp_index]++;
                mismatchesToPerfectRead++;
            }else if(r != p && c != p){
                stats.statsMap[correctionQuality].nucStats[fn_index]++;
                mismatchesToPerfectRead++;
            }		
        }

        stats.statsMap[correctionQuality].remainingErrorsInReads[mismatchesToPerfectRead]++;

        checkedNumReads++;
    });
    
    std::cout << "Checked " << checkedNumReads << " reads." << std::endl;

    NucStats totalstats{};
    std::map<int, std::uint64_t> totalRemainingErrorsInReads{};
    for(auto& qualityStatsPair : stats.statsMap){
        std::cout << "Quality: " << qualityStatsPair.first << "\n";
        std::cout << "TP: " << qualityStatsPair.second.nucStats[tp_index] << "\n";
        std::cout << "FP: " << qualityStatsPair.second.nucStats[fp_index] << "\n";
        std::cout << "FN: " << qualityStatsPair.second.nucStats[fn_index] << "\n";
        std::cout << "TN: " << qualityStatsPair.second.nucStats[tn_index] << "\n";
        std::cout << "Error free reads: " << qualityStatsPair.second.remainingErrorsInReads[0] << "\n";

        for(int i = 0; i < 4; i++){
            totalstats[i] += qualityStatsPair.second.nucStats[i];
        }

        for(const auto& pair : qualityStatsPair.second.remainingErrorsInReads){
            totalRemainingErrorsInReads[pair.first] += pair.second;
        }
        std::cout << std::endl;
    }

    std::cout << "Total:\n";
    std::cout << "TP: " << totalstats[tp_index] << "\n";
    std::cout << "FP: " << totalstats[fp_index] << "\n";
    std::cout << "FN: " << totalstats[fn_index] << "\n";
    std::cout << "TN: " << totalstats[tn_index] << "\n";

    float gain = (totalstats[tp_index]+totalstats[fn_index]) == 0 ? 0 : (float(totalstats[tp_index]) - totalstats[fp_index]) / (totalstats[tp_index] + totalstats[fn_index]);
    float sensitivity = (totalstats[tp_index]+totalstats[fn_index]) == 0 ? 0 : (float(totalstats[tp_index])) / (totalstats[tp_index] + totalstats[fn_index]);
    float specificity = (totalstats[fp_index]+totalstats[tn_index]) == 0 ? 0 : (float(totalstats[tn_index]) / (totalstats[fp_index] + totalstats[tn_index]));

    std::cout << "Gain: " << gain << std::endl;
    std::cout << "Sensitivity: " << sensitivity << std::endl;
    std::cout << "Specificity: " << specificity << std::endl;

    std::cout << "Remaining errors per read | frequency | frequency accumulated\n";
    int threshold = 10;
    int overThreshold = 0;
    int accumulated = 0;
    for(auto pair : totalRemainingErrorsInReads){
        accumulated += pair.second;
        if(pair.first <= threshold){
            std::cout << pair.first << " | " << pair.second << " | " << accumulated << '\n';
        }else{
            overThreshold += pair.second;
        }            
    }
    std::cout << ">" << threshold << " | " << overThreshold << " | " << accumulated << '\n';



}
