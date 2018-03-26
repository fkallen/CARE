#ifndef CARE_ARGS_HPP
#define CARE_ARGS_HPP

#include "cxxopts/cxxopts.hpp"
#include "options.hpp"



namespace care{

namespace args{

	bool areValid(const cxxopts::ParseResult& pr);

	template<class T>
	T to(const cxxopts::ParseResult& pr);

    template<>
	MinhashOptions to<MinhashOptions>(const cxxopts::ParseResult& pr);

	template<>
	AlignmentOptions to<AlignmentOptions>(const cxxopts::ParseResult& pr);

	template<>
	GoodAlignmentProperties to<GoodAlignmentProperties>(const cxxopts::ParseResult& pr);

	template<>
	CorrectionOptions to<CorrectionOptions>(const cxxopts::ParseResult& pr);

	template<>
	RuntimeOptions to<RuntimeOptions>(const cxxopts::ParseResult& pr);

	template<>
	FileOptions to<FileOptions>(const cxxopts::ParseResult& pr);

}

}


#endif
