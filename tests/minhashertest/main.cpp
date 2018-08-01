#include <sequencefileio.hpp>
#include <minhasher.hpp>
#include <options.hpp>
#include <hpc_helpers.cuh>

#include <string>
#include <vector>
#include <cstdint>
#include <memory>


void benchmark(const std::string& filename, int iters){

	using Key_t = std::uint32_t; // asume minhashOptions.k <= 16
	using ReadId_t = std::uint32_t; // asume nReads <= std::numeric_limits<std::uint32_t>::max()
	using Minhasher_t = care::Minhasher<Key_t, ReadId_t>;

	auto properties = care::getSequenceFileProperties(filename, care::FileFormat::FASTQ);

	std::cout << "----------------------------------------" << std::endl;
	std::cout << "File: " << filename << std::endl;
	std::cout << "Reads: " << properties.nReads << std::endl;
	std::cout << "Minimum sequence length: " << properties.minSequenceLength << std::endl;
	std::cout << "Maximum sequence length: " << properties.maxSequenceLength << std::endl;
	std::cout << "----------------------------------------" << std::endl;

	int maps = 8;
        int k = 16;

	care::MinhashOptions minhashOptions{maps, k, 0.0};

	Minhasher_t minhasher(minhashOptions);

	minhasher.init(properties.nReads);

	std::vector<std::string> sequences(properties.nReads);

	std::unique_ptr<care::SequenceFileReader> reader = std::make_unique<care::FastqReader>(filename);
	
	care::Read read;

	//std::cout << "Enter" << std::endl;
	//std::cin >> inputline;

	
	while(reader->getNextRead(&read)){
		std::uint64_t readIndex = reader->getReadnum() - 1;
		//minhasher.insertSequence(read.sequence, readIndex);
		sequences[readIndex] = read.sequence;
	}

	TIMERSTARTCPU(insert);
	for(std::size_t i = 0; i < properties.nReads; i++){
		minhasher.insertSequence(sequences[i], i);
	}
	TIMERSTOPCPU(insert);


	TIMERSTARTCPU(transform);
	minhasher.transform();
	TIMERSTOPCPU(transform);

#if 0

	TIMERSTARTCPU(unlimited);
	#pragma omp parallel for
	for(int i = 0; i < iters; i++){
		auto result = minhasher.getCandidates(sequences[i], std::numeric_limits<std::uint64_t>::max());
	}
	TIMERSTOPCPU(unlimited);

	TIMERSTARTCPU(limited);
	#pragma omp parallel for
	for(int i = 0; i < iters; i++){
		auto result = minhasher.getCandidates(sequences[i], 200);
	}
	TIMERSTOPCPU(limited);

#else


	for(int i = 0; i < 100; i++){
		auto tuple = minhasher.getCandidatesTimed(sequences[i], std::numeric_limits<std::uint64_t>::max());
		auto& result = std::get<0>(tuple);
		auto& hashTime = std::get<1>(tuple);
		auto& getTime = std::get<2>(tuple);
		auto& mergeTime = std::get<3>(tuple);
		auto& uniqueTime = std::get<4>(tuple);
		auto& resizeTime = std::get<5>(tuple);		
		auto& totalTime = std::get<6>(tuple);

		printf("%6lu %.8e %.8e %.8e %.8e %.8e %.8e\n", 
				result.size(), hashTime.count(), getTime.count(), 
				mergeTime.count(), uniqueTime.count(), 
				resizeTime.count(), totalTime.count());
	}

#endif
}


int main(int argc, char** argv){

#if 0
	for(int i = 1; i < argc; i++){

		std::string filename = argv[i];

		benchmark(filename, 10000000);

	}


#else

	using Key_t = std::uint32_t; // asume minhashOptions.k <= 16
	using ReadId_t = std::uint32_t; // asume nReads <= std::numeric_limits<std::uint32_t>::max()
	using Minhasher_t = care::Minhasher<Key_t, ReadId_t>;

	std::string filename = argv[1];

	auto properties = care::getSequenceFileProperties(filename, care::FileFormat::FASTQ);

	std::cout << "----------------------------------------" << std::endl;
	std::cout << "File: " << filename << std::endl;
	std::cout << "Reads: " << properties.nReads << std::endl;
	std::cout << "Minimum sequence length: " << properties.minSequenceLength << std::endl;
	std::cout << "Maximum sequence length: " << properties.maxSequenceLength << std::endl;
	std::cout << "----------------------------------------" << std::endl;

	int maps = 8;
        int k = 16;

	care::MinhashOptions minhashOptions{maps, k, 0.0};

	Minhasher_t minhasher(minhashOptions);

	minhasher.init(properties.nReads);

	std::vector<std::string> sequences(properties.nReads);

	std::unique_ptr<care::SequenceFileReader> reader = std::make_unique<care::FastqReader>(filename);
	
	care::Read read;

	//std::cout << "Enter" << std::endl;
	//std::cin >> inputline;

	TIMERSTARTCPU(insert);
	while(reader->getNextRead(&read)){
		std::uint64_t readIndex = reader->getReadnum() - 1;
		minhasher.insertSequence(read.sequence, readIndex);
		sequences[readIndex] = read.sequence;
	}
	TIMERSTOPCPU(insert);
	//std::cout << "Enter" << std::endl;
	//std::cin >> inputline;

	TIMERSTARTCPU(transform);
	minhasher.transform();
	TIMERSTOPCPU(transform);


#if 0
	for(int i = 0; i < 100; i++){
		auto tuple = minhasher.getCandidatesTimed(sequences[i], std::numeric_limits<std::uint64_t>::max());
		auto& result = std::get<0>(tuple);
		auto& hashTime = std::get<1>(tuple);
		auto& getTime = std::get<2>(tuple);
		auto& mergeTime = std::get<3>(tuple);
		auto& uniqueTime = std::get<4>(tuple);
		auto& resizeTime = std::get<5>(tuple);		
		auto& totalTime = std::get<6>(tuple);

		printf("%6lu %.8e %.8e %.8e %.8e %.8e %.8e\n", 
				result.size(), hashTime.count(), getTime.count(), 
				mergeTime.count(), uniqueTime.count(), 
				resizeTime.count(), totalTime.count());
	}
#endif

	std::uint64_t sum = 0;
	int iters = 10000000; //properties.nReads;
	std::uint64_t limit = 20000; //std::numeric_limits<std::uint64_t>::max(); 

	int percentold = 0;
	int percent = 0;

#if 0
	//#pragma omp parallel for
	#pragma omp parallel for
	for(int i = 0; i < iters; i++){
		auto resultcomplex = minhasher.getCandidates(sequences[i], limit);
		auto result = minhasher.getCandidatesMergeUnique(sequences[i], limit);

		assert(std::is_sorted(resultcomplex.begin(), resultcomplex.end()));
		assert(std::is_sorted(result.begin(), result.end()));
		assert(resultcomplex.size() == result.size());
		//assert(resultcomplex == result);
	}
#endif
#if 1
	TIMERSTARTCPU(complex)
	#pragma omp parallel for
	for(int i = 0; i < iters; i++){
		auto result = minhasher.getCandidates(sequences[i], limit);
		/*sum += result.size();
		percent = int(double(i) / double(iters) * 100.0);
		if(percent > percentold + 9){
			std::cout << percent << "%" << std::endl;
			percentold = percent;
		}*/
	}
	TIMERSTOPCPU(complex)
#endif

#if 1
	TIMERSTARTCPU(complex_new)
	#pragma omp parallel for
	for(int i = 0; i < iters; i++){
		auto result = minhasher.getCandidatesMergeUnique(sequences[i], limit);
		/*sum += result.size();
		percent = int(double(i) / double(iters) * 100.0);
		if(percent > percentold + 9){
			std::cout << percent << "%" << std::endl;
			percentold = percent;
		}*/
	}
	TIMERSTOPCPU(complex_new)
#endif

#if 0
	percentold = 0;
	percent = 0;

	TIMERSTARTCPU(merge)
	#pragma omp parallel for
	for(int i = 0; i < iters; i++){
		auto result = minhasher.getCandidatesMerge(sequences[i], limit);
		/*sum += result.size();
		percent = int(double(i) / double(iters) * 100.0);
		if(percent > percentold + 9){
			std::cout << percent << "%" << std::endl;
			percentold = percent;
		}*/
	}
	TIMERSTOPCPU(merge)
#endif

#if 0
	percentold = 0;
	percent = 0;

	TIMERSTARTCPU(simple)
	#pragma omp parallel for
	for(int i = 0; i < iters; i++){
		auto result = minhasher.getCandidatesSimple(sequences[i], limit);
		/*sum += result.size();
		percent = int(double(i) / double(iters) * 100.0);
		if(percent > percentold + 9){
			std::cout << percent << "%" << std::endl;
			percentold = percent;
		}*/
	}
	TIMERSTOPCPU(simple)
#endif

#if 0  
	percentold = 0;
	percent = 0;

	TIMERSTARTCPU(set)
	#pragma omp parallel for
	for(int i = 0; i < iters; i++){
		auto result = minhasher.getCandidatesSet(sequences[i], limit);
		/*sum += result.size();
		percent = int(double(i) / double(iters) * 100.0);
		if(percent > percentold + 9){
			std::cout << percent << "%" << std::endl;
			percentold = percent;
		}*/
	}
	TIMERSTOPCPU(set)
#endif

	//std::cout << sum << std::endl;

#endif

}

