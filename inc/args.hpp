#ifndef CARE_ARGS_HPP
#define CARE_ARGS_HPP

#include "cxxopts/cxxopts.hpp"
#include "options.hpp"



namespace care{
	
namespace args{

	bool areValid(const cxxopts::ParseResult& pr);
	
	template<class T>
	T to(const cxxopts::ParseResult& pr);
	
}

}


#endif
