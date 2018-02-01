#ifndef COUNT_MIN_SKETCH_HPP
#define COUNT_MIN_SKETCH_HPP

#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>
#include <stdexcept>



template<typename value_t = uint32_t, typename count_t = uint32_t>
struct CountMinSketch{

	static constexpr int PRIME = 777743;

	double epsilon;
	double delta;

	int d; // number of hash functions
	int w; // upper limit of hash values

	int totalcount;
	count_t globalMin;
	bool needMinUpdate;

	std::vector<count_t> counts;
	std::vector<value_t> hashes;


	CountMinSketch() : CountMinSketch(0.01, 0.1){}

	CountMinSketch(double epsilon_, double delta_): 
		epsilon(epsilon_), delta(delta_), totalcount(0), globalMin(0), needMinUpdate(false){

		if( !(0 < epsilon && epsilon <= 1) || !(0 < delta && delta <= 1))
			throw std::runtime_error("invalid CountMinSketch parameters");

		w = std::ceil(std::exp(1.0) / epsilon);
		d = std::ceil(std::log(1.0/delta));

		counts.resize(d * w);
		hashes.resize(2*d);

		srand(42);
		for(int i = 0; i < d; i++){
			hashes[2 * i + 0] = int(float(rand())*float(PRIME)/float(RAND_MAX) + 1);
			hashes[2 * i + 1] = int(float(rand())*float(PRIME)/float(RAND_MAX) + 1);
		}	

		//double mem = (w*d*sizeof(count_t) + 2*d*sizeof(value_t)) / 1024. / 1024.;
		//std::cout << "CountMinSketch memory usage : " << mem << "MB" << std::endl;
	}

	CountMinSketch(const CountMinSketch& other){
		*this = other;
	}

	CountMinSketch(CountMinSketch&& other){
		*this = std::move(other);
	}
	

	CountMinSketch& operator=(const CountMinSketch& other){
		epsilon = other.epsilon;
		delta = other.delta;
		d = other.d;
		w = other.w;
		counts = other.counts;
		hashes = other.hashes;
		needMinUpdate = other.needMinUpdate;
		globalMin = other.globalMin;
		return *this;
	}
	

	CountMinSketch& operator=(CountMinSketch&& other){
		epsilon = other.epsilon;
		delta = other.delta;
		d = other.d;
		w = other.w;
		counts = std::move(other.counts);
		hashes = std::move(other.hashes);
		needMinUpdate = other.needMinUpdate;
		globalMin = other.globalMin;
		return *this;
	}

	count_t operator[](value_t elem) const{
		return estimate(elem);
	}


	void update(value_t elem, count_t count){
		totalcount += count;
		for(int i = 0; i < d; i++){
			value_t wi = hash(elem, i);
			counts[i * w + wi] += count;
		}
		needMinUpdate = true;
	}

	// the actual count c' <= estimate.
	// with a probability of at least (1 - delta): estimate <= c' + epsilon * totalcount
	count_t estimate(value_t elem) const{
		count_t count = std::numeric_limits<count_t>::max();
		for(int i = 0; i < d; i++){
			value_t wi = hash(elem, i);
			count = std::min(count, counts[i * w + wi]);
		}
		return count;
	}

	value_t hash(value_t elem, int i) const{
		return (hashes[2*i + 0] * elem + hashes[2 * i + 1]) % w;
	}

	// O(d * w)
	count_t min(){
		if(needMinUpdate)
			globalMin = *std::min_element(counts.begin(), counts.end());
		needMinUpdate = false;
		return globalMin;
	}

	double getError() const{
		return totalcount * epsilon;
	}

	void clear(){
		counts.clear();
		totalcount = 0;
	}

};







#endif
