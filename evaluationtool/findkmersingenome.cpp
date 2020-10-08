#include <seqan3/search/fm_index/fm_index.hpp>
#include <seqan3/search/search.hpp>
#include <seqan3/search/configuration/max_error.hpp>
#include <seqan3/search/configuration/parallel.hpp>

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>

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


int main(int argc, char** argv){

    if(argc < 5){
        std::cout << "Usage: " << argv[0] << " inputfile.fasta k genomefile.fasta outputfile.fasta" << std::endl;
        std::cout << "For each k-mer in inputfile, write it to output file if it can be found in genome file" << std::endl;
        std::cout << "inputfile contains all k-mers produced by Jellyfish" << std::endl;
        return 0;
    }

    std::string inputfile = argv[1];
    int k = std::atoi(argv[2]);
    std::string genomefile = argv[3];
    std::string outputfile = argv[4];

    std::string genome = get_file_contents(genomefile.c_str());

    for(char& c : genome){
        switch(c){
            case 'a' : c = 'A'; break;
            case 'c' : c = 'C'; break;
            case 'g' : c = 'G'; break;
            case 't' : c = 'T'; break;
            case 'n' : c = 'N'; break;
        }
    }

    std::cerr << "make index\n";
    seqan3::fm_index fmindex{genome};


    std::ifstream input(inputfile);
    std::ofstream output(outputfile);

    std::string line;

    std::int64_t found = 0;
    std::int64_t notfound = 0;

    std::int64_t numLines = 0;
    while(std::getline(input, line)){
        numLines++;
    }
    std::int64_t numKmers = numLines / 2;
    std::cerr << "num kmers: " << numKmers << "\n";
    input = std::move(std::ifstream(inputfile));

    std::cerr << "loading kmers\n";
    std::vector<int> counts(numKmers);
    std::vector<char> kmers(numKmers * k);
    std::vector<std::int8_t> flags(numKmers);

    std::int64_t iter = 0;


    while(std::getline(input, line)){
        counts[iter] = std::stoi(line.substr(1));
        std::getline(input, line);
        std::copy(line.begin(), line.end(), kmers.begin() + iter * k);
        iter++;
    }

    auto revcomplinplace = [](auto& kmerstring){
        for(auto& c : kmerstring){
            switch(c){
                case 'A': c = 'T'; break;
                case 'C': c = 'G'; break;
                case 'G': c = 'C'; break;
                case 'T': c = 'A'; break;
            }
        }
        std::reverse(kmerstring.begin(), kmerstring.end());
    };


    std::cerr << "query\n";

    #pragma omp parallel for schedule(static, 1024)
    for(std::int64_t i = 0; i < numKmers; i++){
        flags[i] = 0;
    }

    #pragma omp parallel for schedule(static, 1024)
    for(std::int64_t i = 0; i < numKmers; i++){
        std::string kmer(kmers.begin() + i * k, kmers.begin() + (i+1) * k);
        auto resultrange = search(kmer, fmindex);
        if(resultrange.begin() != resultrange.end()){
            flags[i] = 1; //found kmer
        }else{
            revcomplinplace(kmer);

            auto resultrange2 = search(kmer, fmindex);
            if(resultrange2.begin() != resultrange2.end()){
                flags[i] = 1; // found kmer (reverse complement of kmer)
            }
        }
    }

    constexpr int outputbatchsize = 1024 * 1024;
    std::stringstream sstream;
    int sstreamElements = 0;

    auto reset_stringstream = [&](){
        sstream.str("");
        sstream.clear();
        sstreamElements = 0;
    };

    std::cerr << "output\n";
    std::int64_t iters = (numKmers + outputbatchsize - 1) / outputbatchsize;

    for(std::int64_t i = 0; i < iters; i++){
        std::int64_t b = i * outputbatchsize;
        std::int64_t e = std::min((i+1) * outputbatchsize, numKmers);
        for(std::int64_t j = b; j < e; j++){
            if(flags[j]){
                std::string kmer(kmers.begin() + j * k, kmers.begin() + (j+1) * k);
                sstream << '>' << counts[j] << '\n' << kmer << '\n';
                found++;
            }else{
                notfound++;
            }
        }

        output << sstream.rdbuf();
        reset_stringstream();
    }
    
    std::cerr << "num found " << found << "\n";
    std::cerr << "num not found " << notfound << "\n";

    return 0;
}
