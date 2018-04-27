#ifndef GRAPH_HPP
#define GRAPH_HPP

#include "alignment.hpp"
#include "tasktiming.hpp"

#include "qualityscoreweights.hpp"

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define ASSERT_TOPOLOGIC_SORT

namespace care{

	namespace errorgraphdetail{
		// split substitutions in alignment into deletion + insertion
        template<class Op>
		int split_subs(std::vector<Op>& ops, const std::string& subject){
			int splitted_subs = 0;
			for(auto it = ops.begin(); it != ops.end(); it++){
				if(it->type == Op::Type::sub){
					Op del = *it;
					del.base = subject[it->position];
					del.type = Op::Type::match;

					Op ins = *it;
					ins.type = Op::Type::ins;

					it = ops.erase(it);
					it = ops.insert(it, del);
					it = ops.insert(it, ins);
					splitted_subs++;
				}
			}
			return splitted_subs;
		};
	}

struct GraphTimings{
    std::chrono::duration<double> buildtime{0};
    std::chrono::duration<double> correctiontime{0};
};

template<class BatchElem_t>
struct ErrorGraph{
	using BatchElem = BatchElem_t;

	struct Edge {
		int to;
		double weight;
		double prob;
		bool canBeUsed;

		Edge(int t, double w) : to(t), weight(w), canBeUsed(true){}
	};

	struct Vertex {
		char base;
		std::vector<Edge> edges;
		double outedgeweightsum = 0.0;
		int bestprevnode = -1;

		double bestPathProb = 0.0;
		double incomingOrigReadProb = 0.0;

		Vertex(char b) : base(b){
			bestPathProb = std::numeric_limits<double>::lowest();
		}
	};

	struct LinkOperation {
		int from;
		int to;
		bool isOriginal; // if true, graph uses chars from its query, if false, use chars from chs
		std::vector<char> chs;

		LinkOperation():from(0), to(0), isOriginal(false){}
		LinkOperation(int f, int t, bool o):from(f), to(t), isOriginal(o){}
	};

    GraphTimings timings;
    TaskTimings taskTimings;

	std::string read;

	bool useQscores = false;
    double max_mismatch_ratio;
    double alpha;
    double x;

    int endNode; // the last node which belongs to the read
	std::vector<Vertex> vertices;

	int insertCalls = 0;
	int totalInsertedAlignments = 0;

	std::vector<int> topoIndices; // the vertex ids topologically sorted
	std::vector<int> finalPath;

	ErrorGraph() : ErrorGraph(false, 1.0, 1.0, 1.0){}

	ErrorGraph(bool useQscores_, double max_mismatch_ratio_, double alpha_, double x_)
				: useQscores(useQscores_), max_mismatch_ratio(max_mismatch_ratio_),
				alpha(alpha_), x(x_){}


	void init(const std::string& seq){
		init(seq, nullptr);
	}

	void init(const std::string& seq, const std::string* qualityScores){
		assert(seq.length() > 0);
		assert(!BatchElem::canUseQualityScores || (BatchElem::canUseQualityScores && qualityScores != nullptr));

		clearVectors();

		read = seq;

		double weight;
		int newindex;
		char base;

		newindex = addNewNode('S');
		topoIndices.push_back(newindex);

		vertices[newindex].bestPathProb = 0.0;

		for (int i = 0; i < int(read.length()); ++i) {
			base = read[i];
			newindex = addNewNode(base);
			topoIndices.push_back(newindex);

			double initialWeight = BatchElem::canUseQualityScores ? qscore_to_weight[(unsigned char)(*qualityScores)[i]] : 1.0;

			weight = initialWeight;

			Edge edge(newindex, weight);
			vertices[newindex - 1].edges.push_back(edge);
			vertices[newindex - 1].outedgeweightsum += weight;
		}

		newindex = addNewNode('E');
		topoIndices.push_back(newindex);
		weight = 1.0;
		Edge edge(newindex, weight);
		vertices[newindex - 1].edges.push_back(edge);
		vertices[newindex - 1].outedgeweightsum += weight;

		endNode = vertices.size() - 1;
	}

	// add new node, but don't add edges yet
	// return the index of the new node in the vertices vector
	int addNewNode(const char base)
	{
		vertices.push_back(Vertex(base));
		return vertices.size() - 1;
	}

	void clearVectors(){
		vertices.clear();
		topoIndices.clear();
		finalPath.clear();
	}

    template<class Alignment>
	void insertAlignment(Alignment& alignment,
						 const int nTimes){
		insertAlignment(alignment, nullptr, nTimes);
	}

	// insert alignment nTimes into the graph
    template<class Alignment>
	void insertAlignment(Alignment& alignment,
							const std::string* qualityScores_, const int nTimes){

		assert(!(useQscores && !qualityScores_));
		const std::string& qualityScores = *qualityScores_;

		insertCalls++;
		totalInsertedAlignments += nTimes;

		const double weight = 1 - std::sqrt(alignment.get_nOps() / (alignment.get_overlap() * max_mismatch_ratio));

		int last_a = alignment.get_subject_begin_incl() + alignment.get_overlap();

		normalizeAlignment(alignment); //returns immediatly if already normalized (e.g. by previous insert)

		const auto& linkOps = makeLinkOperations(alignment);

		int prev_node = alignment.get_subject_begin_incl();

		if (prev_node != 0) {
			prev_node = -1;
		}

		int qindex = alignment.get_query_begin_incl();

		for (const auto& op : linkOps) {

			if (op.isOriginal) {
				for (int j = op.from; j < op.to; j++) {
					if (prev_node == -1) {
						prev_node = j + 1;
					}else{
						double qweight = weight;
						if(BatchElem::canUseQualityScores){
							if(qindex < alignment.get_query_begin_incl() + alignment.get_overlap()){
								qweight *= qscore_to_weight[(unsigned char)qualityScores[qindex]];
							}
						}
						prev_node = makeLink(prev_node, j + 1, read[j], qweight, nTimes);
					}
					qindex++;
				}
			}else {
				for(size_t j = 0; j < op.chs.size(); j++){
					const char& c = op.chs[j];
					if (prev_node == -1) {
						prev_node = addNewNode(c);
						topoIndices.insert(topoIndices.begin(), prev_node);
					}else{
						double qweight = weight;
						if(BatchElem::canUseQualityScores){
							if(qindex < alignment.get_query_begin_incl() + alignment.get_overlap()){
								qweight *= qscore_to_weight[(unsigned char)qualityScores[qindex]];
							}
						}
						prev_node = makeLink(prev_node, -1, c, qweight, nTimes);
					}
					qindex++;
				}
			}
		}

		if (last_a == int(read.length()) && prev_node != -1) {
			makeLink(prev_node, endNode, 'E', weight, 1);
		}
	}

	// insert edge from --weight--> to
	// to == -1 means it is unknown if to is already in the graph
	// returns the index of to in vertices
	int makeLink(const int from, const int to, const char base_of_to, const double weight, const int nTimes)
	{
	#if 0
		std::cout << "makeLink(" << from << ", " << to << ", " << base_of_to << ", " << weight << ")" << std::endl;
	#endif
		assert(to != 0); // there cannot be an edge to the start node
		assert(from != to); // no edge to itself
		assert(from >= 0);

		int ret_to = to;

		if (to == -1) {
			// for each neighbor of from
			for (Edge& edge : vertices[from].edges) {

				//if neighbor is not a node from the initial read, and neighbor has the wanted base
				if (edge.to > endNode && vertices[edge.to].base == base_of_to) {
					edge.weight += weight * nTimes;
					ret_to = edge.to;
					break;
				}
			}

			// node for base does not exist yet, make new node
			if (ret_to == -1) {
				ret_to = addNewNode(base_of_to);

				auto it = std::find(topoIndices.begin(), topoIndices.end(), from);
				if (it == topoIndices.end()){
					std::cout << "vertices : " << vertices.size() << "\n" << "topoIndices: " << topoIndices.size() << "\n";
					for(const auto& ind : topoIndices) std::cout << ind << " ";
					throw std::logic_error("cannot find node in topoIndices which must exist there");
				}
				++it; // insert at this position
				topoIndices.insert(it, ret_to);

				vertices[from].edges.push_back(Edge(ret_to, weight * nTimes));
			}
		}else { // to != -1
			bool found = false;
			// find the edge which ends in to and update the weight
			for (Edge& edge : vertices[from].edges) {
				if (edge.to == to) {
					if(vertices[to].base != base_of_to)
						assert(vertices[to].base == base_of_to);
					edge.weight += weight * nTimes;
					found = true;
					break;
				}
			}

			// there is no edge yet. create this edge

			if (!found){
				vertices[from].edges.push_back(Edge(to, weight * nTimes));
			}

		}
		vertices[from].outedgeweightsum += weight * nTimes;

		return ret_to;
	}


	void calculateEdgeProbabilities()
	{

		for (size_t i = 0; i < vertices.size(); ++i) {

			Vertex& v = vertices[i];

			double bestNonOriginalEdgeWeight = 0.0;
			double originalEdgeWeight = 0.0;
			//int bestNonOriginalEdge = -1;
			int originalEdge = -1;

			for(size_t j = 0; j < v.edges.size(); ++j){
				Edge& e = v.edges[j];

				//calculate normalized edge weight
				e.prob = e.weight / v.outedgeweightsum;

				// if original edge
				if(i < unsigned(endNode) && unsigned(e.to) == i+1){
					originalEdgeWeight = e.weight;
					originalEdge = j;
				}else{
					if(bestNonOriginalEdgeWeight < e.weight){
						bestNonOriginalEdgeWeight = e.weight;
						//bestNonOriginalEdge = j;
					}
				}
			}


			if(originalEdge != -1){

				// original edge is preferred
				// if (heighest weight - orig weight) < alpha*(x^orig weight)
				if((bestNonOriginalEdgeWeight - originalEdgeWeight) < alpha * std::pow(x, originalEdgeWeight) ){
					for(size_t j = 0; j < v.edges.size(); ++j){
						if(j != unsigned(originalEdge)){
							v.edges[j].canBeUsed = false;
						}
					}
				}

			}

		}
	}

	// extract corrected read from graph, which is the most reliable path
	std::string getCorrectedRead()
	{

		if(totalInsertedAlignments == 0){
			return {read};
		}

	#ifdef ASSERT_TOPOLOGIC_SORT
		// better be safe
		assertTopologicallySorted();
	#endif

		calculateEdgeProbabilities();

		//find most reliable path, which should be equivalent to the corrected read

		for (const int& i : topoIndices) {
			const Vertex& cur = vertices[i];

			for (const Edge& e : cur.edges) {
				if (e.canBeUsed) {

					double newBestLogPathProb = cur.bestPathProb + log(e.prob);
					if (newBestLogPathProb > vertices[e.to].bestPathProb) {
						vertices[e.to].bestPathProb = newBestLogPathProb;
						vertices[e.to].bestprevnode = i;
					}
				}
			}
		}

		std::string correctedRead;
		//double prob = std::exp(vertices[endNode].bestPathProb);

		// backtrack to extract corrected read
		std::string rcorrectedRead = "";

		Vertex cur('F');
		try{
			cur = vertices.at(vertices.at(endNode).bestprevnode);
		}catch(std::out_of_range ex){
			throw ex;
		}
		int currentVertexNumber = cur.bestprevnode;

		std::vector<int> rpath;
		rpath.push_back(endNode);
		rpath.push_back(vertices.at(endNode).bestprevnode);

		while (cur.bestprevnode != -1) {
			rpath.push_back(currentVertexNumber);
			rcorrectedRead += cur.base;
			try{
				cur = vertices.at(cur.bestprevnode);
			}catch(std::out_of_range ex){
				printf("bestprevnode %d\n", cur.bestprevnode);
				throw ex;
			}
			currentVertexNumber = cur.bestprevnode;
		}

		finalPath.resize(rpath.size());
		std::reverse_copy(rpath.begin(), rpath.end(), finalPath.begin());

		correctedRead.resize(rcorrectedRead.size());
		std::reverse_copy(rcorrectedRead.begin(), rcorrectedRead.end(), correctedRead.begin());
		return correctedRead;
	}




	void dumpGraph(std::string prefix) const
	{
		std::ofstream vertexout(prefix + "vertices.txt");
		std::ofstream edgeout(prefix + "edges.txt");
		std::ofstream pathout(prefix + "path.txt");
		//std::ofstream prevout(prefix + "prev.txt");



		for (size_t i = 0; i < vertices.size(); ++i) {
			vertexout << i << " " << vertices[i].base << '\n';
			//prevout << i << " " << vertices[i].bestprevnode << '\n';
		}
		for (size_t i = 0; i < vertices.size(); ++i) {
			for (const Edge& edge : vertices[i].edges) {
				edgeout << i << " " << edge.to << " " << edge.weight << " " << edge.prob << '\n';
			}
		}

		for(const auto& n : finalPath){
			pathout << n << " ";
		}
		pathout << '\n';


	}

    template<class Alignment>
	void normalizeAlignment(Alignment& alignment) const{
        using Op_t = typename Alignment::Op_t;
		if(alignment.get_isNormalized())
			return;

        auto& alignOps = alignment.operations;

		int na = int(read.length());
		int last_val = na;

		// delay operations as long as possible

		for (int i = alignOps.size() - 1; i >= 0; i--) {
			auto& op = alignOps[i];
			int position = op.position;
			const int base = op.base;

			if (op.type == Op_t::Type::del) {
				position++;
				while (position < last_val && read[position] == base)
					position++;
				position--;
				op.position = position;
				last_val = position;

				for (size_t j = i; j < alignOps.size() - 1; j++) {
					if (alignOps[j + 1].type == Op_t::Type::del)
						break;

					if (alignOps[j].position >= alignOps[j + 1].position) {
						std::swap(alignOps[j], alignOps[j + 1]);
						alignOps[j].position--;
					}else break;
				}
			}else if (op.type == Op_t::Type::ins) {

				if (position < na && read[position] == base) {
					position++;
					while (position < na && read[position] == base)
						position++;
					op.position = position;
				}

				for (size_t j = i; j < alignOps.size() - 1; j++) {
					if (alignOps[j + 1].type == Op_t::Type::del) {
						if (alignOps[j].position >= alignOps[j + 1].position) {

							// insertion and deletion of same base cancel each other. don't need this op
							if (alignOps[j].base == alignOps[j + 1].base) {
								alignOps.erase(alignOps.begin() + j + 1);
								alignOps.erase(alignOps.begin() + j);
								break;
							}

							std::swap(alignOps[j], alignOps[j+1]);

							if (alignOps[j + 1].position < na)
								alignOps[j + 1].position++;

							int position2 = alignOps[j + 1].position;
							const char base2 = alignOps[j + 1].base;

							if (position2 < na && read[position2] == base2) {
								position2++;
								while (position2 < na && read[position2] == base2)
									position2++;
								alignOps[j + 1].position = position2;
							}
						}else break;
					}else {
						if (alignOps[j].position > alignOps[j + 1].position) {
							std::swap(alignOps[j], alignOps[j + 1]);
							alignOps[j].position++;
						}else break;
					}
				}
			}
		}

		alignment.get_isNormalized() = true;
	}

    template<class Alignment>
	std::vector<LinkOperation> makeLinkOperations(const Alignment& alignment) const{
        using Op_t = typename Alignment::Op_t;

		int cur_a = alignment.get_subject_begin_incl();
		int last_a = cur_a + alignment.get_overlap();

        const auto& alignOps = alignment.operations;

		std::vector<LinkOperation> linkOps;

		for (size_t i = 0; i < alignOps.size(); i++) {
			const auto& alignop = alignOps[i];
			const int ca = alignop.position;
			const char ch = alignop.base;

			if (cur_a < ca) {
				linkOps.push_back( LinkOperation(cur_a, ca, true) );
				cur_a = ca;
				i--;
			}else {
				LinkOperation linkop(ca, ca, false);

				if (alignop.type == Op_t::Type::del) {
					linkop.to++;
					cur_a++;
				}else{
					linkop.chs.push_back(ch);
				}

				i++;
				for (; i < alignOps.size(); i++) {
					if (alignOps[i].position > cur_a) break;
					if (alignOps[i].type == Op_t::Type::del) {
						linkop.to++;
						cur_a++;
					}else{
						linkop.chs.push_back(alignOps[i].base);
					}
				}
				i--;
				linkOps.push_back(linkop);
			}
		}

		if (cur_a < last_a) {
			linkOps.push_back( LinkOperation(cur_a, last_a, true) );
		}

		return linkOps;
	}

	void assertTopologicallySorted() const
	{
		//check number of elements
		if (vertices.size() != topoIndices.size()) {
			throw std::logic_error("nodes are missing in topoIndices");
		}

		// check if unique
		std::vector<int> tmpvec(topoIndices);
		std::sort(tmpvec.begin(), tmpvec.end());

		for (size_t i = 0; i < tmpvec.size(); ++i){
			if (unsigned(tmpvec[i]) != i){
				for (size_t j = 0; j < tmpvec.size(); ++j){
					std::cout << tmpvec[j] << " ";
				}
				throw std::logic_error("duplicates in topoIndices");
			}
		}

		// for each edge check if topo[from] < topo[to]
		for (const auto& i : topoIndices) {
			for (const auto& edge : vertices[i].edges) {
				const auto itfrom = std::find(topoIndices.begin(), topoIndices.end(), i);
				const auto itto = std::find(topoIndices.begin(), topoIndices.end(), edge.to);
				if (itto < itfrom){
				//	dumpGraph("topoerror");
				//	for (const auto& i : topoIndices) {
						//std::cout << i << " ";
				//	}
					throw std::logic_error("no topological sorting");
				}
			}
		}
	}

	TaskTimings correct_batch_elem(BatchElem& b){
		if(BatchElem::canUseQualityScores){
			init(b.fwdSequenceString, b.fwdQuality);
		}else{
			init(b.fwdSequenceString);
		}

		TaskTimings tt;
		std::chrono::time_point<std::chrono::system_clock> tpa, tpb;

		tpa = std::chrono::system_clock::now();

		for(size_t i = 0; i < b.n_unique_candidates; i++){
			auto& alignmentptr = b.bestAlignments[i];
			const int freq = b.candidateCountsPrefixSum[i+1] - b.candidateCountsPrefixSum[i];

			errorgraphdetail::split_subs(alignmentptr->operations, b.fwdSequenceString);

			for(int f = 0; f < freq; f++){
				if(BatchElem::canUseQualityScores){
					const int qualindex = b.candidateCountsPrefixSum[i] + f;
					const std::string* qual = b.bestQualities[qualindex];
					insertAlignment(*alignmentptr, qual, 1);
				}else{
					insertAlignment(*alignmentptr, 1);
				}
			}
		}
		tpb = std::chrono::system_clock::now();

		timings.buildtime += tpb - tpa;
		taskTimings.preprocessingtime += tpb - tpa;
		tt.preprocessingtime = tpb - tpa;

		tpa = std::chrono::system_clock::now();
		// let the graph to its work
		b.correctedSequence = getCorrectedRead();

		tpb = std::chrono::system_clock::now();

		timings.correctiontime += tpb - tpa;
		taskTimings.executiontime += tpb - tpa;
		tt.executiontime = tpb - tpa;

		b.corrected = true;

		return tt;
	}

};


}
#endif
