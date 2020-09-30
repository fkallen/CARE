#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <numeric>
#include <map>

using kmer = std::uint64_t;

std::string get_file_contents(const char *filename){
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if(in){
        std::string contents;
        in.seekg(0, std::ios::end);
        contents.resize(in.tellg());
        in.seekg(0, std::ios::beg);
        in.read(&contents[0], contents.size());
        in.close();
        return contents;
    }else{
        return "";
    }
}

std::int64_t getNumberOfLines(std::string filename){
    std::ifstream is(filename);
    if(!is){
        return 0;
    }

    std::int64_t result = 0;

    std::string line;
    while(std::getline(is, line)){
        result++;
    }

    return result;
}

struct KmersAndCounts{
    std::vector<int> counts;
    std::vector<kmer> kmers;
};

KmersAndCounts parsekmers(std::string filename, int k, int countlimit = std::numeric_limits<int>::max()){
    std::int64_t numlines = getNumberOfLines(filename);
    std::int64_t numkmers = numlines / 2;

    std::cout << filename << " " << numkmers << " kmers." << std::endl;   

    KmersAndCounts result;
    result.kmers.reserve(numkmers);
    result.counts.reserve(numkmers);

    std::ifstream is(filename);

    std::string line;
    while(std::getline(is, line)){
        int count = std::stoi(line.substr(1));
        
        std::getline(is, line);
        if(count <= countlimit){
            kmer encoded = 0;
            for(int i = 0; i < k; i++){
                kmer base = 0;
                switch(line[i]){
                    case 'A': base = 0; break;
                    case 'C': base = 1; break;
                    case 'G': base = 2; break;
                    case 'T': base = 3; break;
                }
                encoded = (encoded << 2) | base;
            }
            result.kmers.emplace_back(encoded);
            result.counts.emplace_back(count);
        }
    }

    return result;
}

KmersAndCounts parsekmers2(std::string filename, int k, int countlimit = std::numeric_limits<int>::max()){
    std::int64_t numlines = getNumberOfLines(filename);
    std::int64_t numkmers = numlines / 2;

    std::cout << filename << " " << numkmers << " kmers." << std::endl;

    KmersAndCounts result;
    result.kmers.reserve(numkmers);
    result.counts.reserve(numkmers);

    std::ifstream is(filename);

    std::string line;
    while(std::getline(is, line)){
        auto space = line.find(' ');
        int count = std::stoi(line.substr(0, space));        

        if(count <= countlimit){
            line = line.substr(space+1);

            kmer encoded = 0;
            for(int i = 0; i < k; i++){
                kmer base = 0;
                switch(line[i]){
                    case 'A': base = 0; break;
                    case 'C': base = 1; break;
                    case 'G': base = 2; break;
                    case 'T': base = 3; break;
                }
                encoded = (encoded << 2) | base;
            }
            result.kmers.emplace_back(encoded);
            result.counts.emplace_back(count);
        }
    }

    return result;
}


int main(int argc, char** argv){

    if(argc < 5){
        std::cout << "Usage: " << argv[0] << " uncorrectedkmers.fasta correctedkmers.fasta k limit" << std::endl;
        std::cout << "Counts how many k-mers are present in both uncorrected file and corrected file." << std::endl;
        std::cout << "Counts how many k-mers are present in uncorrected file but not in corrected file (i.e. k-mers which were lost during correction)" << std::endl;
        std::cout << "Computes a histogram: (k-mer coverage , number of distinct lost true k-mers over coverage)" << std::endl;
        std::cout << "In the uncorrected file, only k-mers which occur at most limit times are considered." << std::endl;
        return 0;
    }

    std::string uncorrectedkmerfile = argv[1];
    std::string correctedkmerfile = argv[2];
    int k = std::atoi(argv[3]);
    int limit = std::atoi(argv[4]);

    if(k > 32){
        std::cout << "k must be <= 32" << std::endl;
        return 0;
    }

    //load kmers

    KmersAndCounts uncorrectedKmers = parsekmers(uncorrectedkmerfile, k, limit);

    KmersAndCounts correctedKmers = parsekmers(correctedkmerfile, k);

   
    //sort (indices to) kmers lexicographicaly
    std::vector<int> uncorrectedIndices(uncorrectedKmers.kmers.size());
    std::vector<int> correctedIndices(correctedKmers.kmers.size());

    std::iota(uncorrectedIndices.begin(), uncorrectedIndices.end(), 0);
    std::iota(correctedIndices.begin(), correctedIndices.end(), 0);

    std::sort(uncorrectedIndices.begin(), uncorrectedIndices.end(),
        [&](const auto l, const auto r){
            return uncorrectedKmers.kmers[l] < uncorrectedKmers.kmers[r];
        }
    );
    std::sort(correctedIndices.begin(), correctedIndices.end(),
        [&](const auto l, const auto r){
            return correctedKmers.kmers[l] < correctedKmers.kmers[r];
        }
    );


    //find kmers which occur in both files via set_intersection of sorted kmer lists
    std::int64_t uncorrectedkmersInCorrectedFile = 0;
    std::int64_t distinctUncorrectedkmersInCorrectedFile = 0;
    //set_intersection
    {
        auto first1 = uncorrectedIndices.begin();
        auto last1 = uncorrectedIndices.end();
        auto first2 = correctedIndices.begin();
        auto last2 = correctedIndices.end();

        while (first1 != last1 && first2 != last2) {
            if (uncorrectedKmers.kmers[*first1] < correctedKmers.kmers[*first2]) {
                ++first1;
            } else  {
                if (!(correctedKmers.kmers[*first2] < uncorrectedKmers.kmers[*first1])) {
                    distinctUncorrectedkmersInCorrectedFile++;
                    uncorrectedkmersInCorrectedFile += std::min(
                        uncorrectedKmers.counts[*first1],
                        correctedKmers.counts[*first2]
                    );
                    ++first1;
                }
                ++first2;
            }
        }
    }

    std::map<int, std::int64_t> histogram;

    //find kmers which occur in uncorrectedkmers file, but do not occur in correctedkmers file via set_difference of sorted kmer lists
    std::int64_t uncorrectedkmersMissingInCorrectedFile = 0;
    std::int64_t distinctUncorrectedkmersMissingInCorrectedFile = 0;
    //set_difference
    {
        auto first1 = uncorrectedIndices.begin();
        auto last1 = uncorrectedIndices.end();
        auto first2 = correctedIndices.begin();
        auto last2 = correctedIndices.end();

        auto handle_missing = [&](auto begin, auto end){
            distinctUncorrectedkmersMissingInCorrectedFile += std::distance(begin, end);
            for(auto it = begin; it != end; ++it){
                const int count = uncorrectedKmers.counts[*it];
                uncorrectedkmersMissingInCorrectedFile += count;
                histogram[count]++;
            }
        };

        while (first1 != last1) {
            if (first2 == last2){
                //all corrected kmers have been processed. remaining uncorrected kmers are missing
                handle_missing(first1, last1);
                break;
            }
    
            if (uncorrectedKmers.kmers[*first1] < correctedKmers.kmers[*first2]) {
                //kmer *first1 is missing in corrected kmers
                handle_missing(first1, std::next(first1));
                // distinctUncorrectedkmersMissingInCorrectedFile++;
                // const int count = uncorrectedKmers.counts[*first1];
                // uncorrectedkmersMissingInCorrectedFile += count;
                // histogram[count]++;

                ++first1;
            } else {
                if (!(correctedKmers.kmers[*first2] < uncorrectedKmers.kmers[*first1])) {
                    ++first1;
                }
                ++first2;
            }
        }
    }

    std::cout << "num uncorrected kmers: " << uncorrectedKmers.kmers.size() << std::endl;
    std::cout << "num corrected kmers: " << correctedKmers.kmers.size() << std::endl;
    std::cout << "num uncorrected kmers in corrected file: " << uncorrectedkmersInCorrectedFile << std::endl;
    std::cout << "num distinct uncorrected kmers in corrected file: " << distinctUncorrectedkmersInCorrectedFile << std::endl;
    std::cout << "num uncorrected kmers missing in corrected file: " << uncorrectedkmersMissingInCorrectedFile << std::endl;
    std::cout << "num distinct uncorrected kmers missing in corrected file: " << distinctUncorrectedkmersMissingInCorrectedFile << std::endl;


    std::cout << "histogram:\n";
    for(const auto& pair : histogram){
        std::cout << pair.first << " " << pair.second << "\n";
    }
    std::cout << std::endl;



    return 0;
}
