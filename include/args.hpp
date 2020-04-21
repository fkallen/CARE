#ifndef CARE_ARGS_HPP
#define CARE_ARGS_HPP

#include <config.hpp>

#include "cxxopts/cxxopts.hpp"
#include "options.hpp"



namespace care{

namespace args{

	template<class T>
	T to(const cxxopts::ParseResult& pr);

	template<>
	GoodAlignmentProperties to<GoodAlignmentProperties>(const cxxopts::ParseResult& pr);

	template<>
	CorrectionOptions to<CorrectionOptions>(const cxxopts::ParseResult& pr);

	template<>
	RuntimeOptions to<RuntimeOptions>(const cxxopts::ParseResult& pr);

	template<>
	MemoryOptions to<MemoryOptions>(const cxxopts::ParseResult& pr);

	template<>
	FileOptions to<FileOptions>(const cxxopts::ParseResult& pr);

    template<class T>
    bool isValid(const T& opt);

    template<>
    bool isValid<GoodAlignmentProperties>(const GoodAlignmentProperties& opt);

    template<>
    bool isValid<CorrectionOptions>(const CorrectionOptions& opt);

    template<>
    bool isValid<RuntimeOptions>(const RuntimeOptions& opt);

    template<>
    bool isValid<MemoryOptions>(const MemoryOptions& opt);

    template<>
    bool isValid<FileOptions>(const FileOptions& opt);

}

}


#endif
