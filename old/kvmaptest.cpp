#include "inc/kvmapfixed.hpp"
#include "inc/ganja/hpc_helpers.cuh"
#include "inc/ganja/open_addressing_multi_hash_map.cuh"
#include "inc/ganja/hash_functions.cuh"

#include <random>

using ui = std::uint64_t;
using hash_func = mueller_hash_uint32_t;
using prob_func = linear_probing_scheme_t;
using oa_hash_t = OpenAddressingMultiHashMap<uint64_t, 32, 32,
				   hash_func,prob_func>;

int main(){
#if 0
	KVMapFixed<ui> map(4);

	map.print(std::cout);

	map.set(0, 42);
	map.set(1, 13);
	map.set(2, 17);
	map.set(3, 42);
	map.set(12, 42);

	map.print(std::cout);

	map.getByKey(42);

	map.freeze();

	map.print(std::cout);

	auto p = map.getByKey(42);
	std::cout << "42: ";
	for(ui i = 0; i < p.second; i++)
		std::cout << p.first[i] << ' ';
	std::cout << '\n';

	p = map.getByKey(13);
	std::cout << "13: ";
	for(ui i = 0; i < p.second; i++)
		std::cout << p.first[i] << ' ';
	std::cout << '\n';


	p = map.getByKey(17);
	std::cout << "17: ";
	for(ui i = 0; i < p.second; i++)
		std::cout << p.first[i] << ' ';
	std::cout << '\n';
#endif
	ui N = 1000000;
	ui maxkey = N;

	std::vector<ui> keys(N);
	#pragma omp parallel for num_threads(64)
	for (ui i = 0; i < N; i++)
		keys[i] = i % maxkey;

	std::mt19937 urng(42);
	std::shuffle(keys.begin(), keys.end(), urng);

	KVMapFixed<ui> fmap(N);
	oa_hash_t oamap{N, hash_func {}, prob_func {}};

	std::cout << "begin\n";
	TIMERSTARTCPU(fmap_insert);
	#pragma omp parallel for num_threads(8)
	for(ui i = 0; i < N; i++)
		fmap.add(keys[i], i);
	TIMERSTOPCPU(fmap_insert);

	TIMERSTARTCPU(fmap_transform);
	fmap.freeze();
	TIMERSTOPCPU(fmap_transform);

	TIMERSTARTCPU(oamap_insert);
	#pragma omp parallel for num_threads(8)
	for(ui i = 0; i < N; i++)
		oamap.add(keys[i], i);
	TIMERSTOPCPU(oamap_insert);

	TIMERSTARTCPU(oamap_transform);
	oamap.transform();
	TIMERSTOPCPU(oamap_transform);

	bool ok = true;
	for(ui i = 0; i < N && ok; i++){
		auto fmapvals = fmap.get(keys[i]);
		auto oavals = oamap.get(keys[i]);
		if(fmapvals.size() == oavals.size()){
			std::sort(fmapvals.begin(), fmapvals.end());
			std::sort(oavals.begin(), oavals.end());
			for(int j = 0; j < fmapvals.size() && ok; j++){
				if(fmapvals[j] != oavals[j]){
					std::cout << "error " << i << " " << j << std::endl;
					ok = false;
				}
			}
		}else{
			std::cout << "error " << i << std::endl;
			ok = false;
		}
	}
}
