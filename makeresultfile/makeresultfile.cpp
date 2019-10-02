#if 1
#include <sequencefileio.hpp>
#include <config.hpp>

#include <cxxopts/cxxopts.hpp>
#include <args.hpp>
#include <options.hpp>

#include <string>
#include <vector>
#include <cstdint>


int main(int argc, char** argv){

    bool help = false;

    cxxopts::Options options(argv[0], "Construct result file from corrected sequences.");

    options.add_options("Group")
        ("h", "Show this help message", cxxopts::value<bool>(help))
        ("inputfile", "The fastq file to correct", cxxopts::value<std::string>())
        ("fileformat", "Format of input file. Overrides automatic detection. Allowed values: {fasta, fastq, fastagz, fastqgz}",
            cxxopts::value<std::string>()->default_value("")->implicit_value(""))
        ("correctedsequences", "The file(s) with corrected sequences as produced by care", cxxopts::value<std::vector<std::string>>())
        ("outdir", "The output directory", cxxopts::value<std::string>()->default_value("."))
        ("outfile", "The output file", cxxopts::value<std::string>()->default_value("")->implicit_value(""))

        ("threads", "Maximum number of thread to use. Must be greater than 0", cxxopts::value<int>()->default_value("1"))

        ("progress", "If set, progress is displayed.",
        cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
    ;

    options.parse_positional({"correctedsequences"});

    auto parseresults = options.parse(argc, argv);

    if(help) {
        std::cout << options.help({"", "Group"}) << std::endl;
        exit(0);
    }

    std::string inputfile = parseresults["inputfile"].as<std::string>();
    std::string fileformat = parseresults["fileformat"].as<std::string>();
    std::string outdir = parseresults["outdir"].as<std::string>();
    std::string outfile = parseresults["outfile"].as<std::string>();
    std::string tempdir = outdir;
    //int threads = parseresults["threads"].as<int>();
    //bool progress = parseresults["progress"].as<bool>();

    auto filesToMerge = parseresults["correctedsequences"].as<std::vector<std::string>>();

    care::FileFormat originalFormat = care::FileFormat::NONE;

    if (fileformat == "fasta"){
        originalFormat = care::FileFormat::FASTA;
    }else if(fileformat == "fastq"){
        originalFormat = care::FileFormat::FASTQ;
    }else if(fileformat == "fastagz"){
        originalFormat = care::FileFormat::FASTAGZ;
    }else if(fileformat == "fastqgz"){
        originalFormat = care::FileFormat::FASTQGZ;
    };

    if(originalFormat == care::FileFormat::NONE){
        originalFormat = care::getFileFormat(inputfile);
    }

    care::mergeResultFiles(tempdir, 0, inputfile, originalFormat, filesToMerge, outdir + "/" + outfile, true);
}

#endif
