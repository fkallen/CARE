#ifndef GRAPH_HPP
#define GRAPH_HPP

#include "qualityscoreweights.hpp"

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <numeric>
#include <cmath>

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
					del.type = Op::Type::del;

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

namespace errorgraph{


    struct ErrorGraph{

        struct CorrectionResult{
            std::string correctedSequence;
            double probability;
        };

    	struct Edge {
            double weight;
    		double prob;
    		int to;
    		bool canBeUsed;

    		Edge(int t, double w) noexcept : weight(w), to(t), canBeUsed(true){}
    	};

    	struct Vertex {
            std::vector<Edge> edges;
    		double outedgeweightsum = 0.0;
    		double bestPathProb = std::numeric_limits<double>::lowest();
    		double incomingOrigReadProb = 0.0;
    		int bestprevnode = -1;
    		char base;

    		Vertex(char b) noexcept : base(b){
                edges.reserve(2);
            }
    	};

    	struct LinkOperation {
            std::vector<char> chs;
    		int from;
    		int to;
    		bool isOriginal; // if true, graph uses chars from its query, if false, use chars from chs

    		LinkOperation():from(0), to(0), isOriginal(false){}
    		LinkOperation(int f, int t, bool o):from(f), to(t), isOriginal(o){}
    	};

        static constexpr int unknown_vertex_index = -1;
        static constexpr int unknown_destination_index = -1;

        int endNode; // the last node which belongs to the read
    	std::vector<Vertex> vertices;
    	std::vector<int> topoIndices; // the vertex ids topologically sorted
    	std::vector<int> finalPath;

        // add new node, but don't add edges yet
    	// return the index of the new node in the vertices vector
    	int addNewNode(const char base) noexcept
    	{
    		vertices.push_back(Vertex(base));
    		return vertices.size() - 1;
    	}

    	void clearVectors() noexcept{
    		vertices.clear();
    		topoIndices.clear();
    		finalPath.clear();
    	}

        void init(const std::string& sequence_to_correct,
                    const std::string* quality_of_sequence_to_correct) noexcept{

            assert(sequence_to_correct.length() > 0);

            clearVectors();

            vertices.reserve(2*sequence_to_correct.size() + 2);
            topoIndices.reserve(2*sequence_to_correct.size() + 2);

            vertices.emplace_back('S');
            topoIndices.emplace_back(0);

            vertices[0].bestPathProb = 0.0;

            for(std::size_t i = 0; i < sequence_to_correct.size(); i++){
                const char base = sequence_to_correct[i];
                const int newindex = addNewNode(base);
                topoIndices.emplace_back(newindex);

                const double initialWeight = quality_of_sequence_to_correct != nullptr
                                            ? qscore_to_weight[(unsigned char)(*quality_of_sequence_to_correct)[i]]
                                            : 1.0;

                vertices[newindex - 1].edges.emplace_back(newindex, initialWeight);
                vertices[newindex - 1].outedgeweightsum += initialWeight;
            }

            const int newindex = addNewNode('E');
            topoIndices.emplace_back(newindex);
            const double initialWeight = 1.0;

            vertices[newindex - 1].edges.emplace_back(newindex, initialWeight);
            vertices[newindex - 1].outedgeweightsum += initialWeight;

            endNode = vertices.size() - 1;
            assert(endNode == int(sequence_to_correct.size()) + 1);
        }

        /*
            AlignmentIter: Iterator to Alignment pointer
            SequenceIter: Iterator to std::string
            CountIter: Iterator to int
            QualityIter: Iter to pointer to std::string
        */
        template<class AlignmentIter, class SequenceIter, class CountIter, class QualityIter>
        void add_candidates(const std::string& sequence_to_correct,
                    const std::string* quality_of_sequence_to_correct,
                    AlignmentIter alignmentsBegin,
                    AlignmentIter alignmentsEnd,
                    double desiredAlignmentMaxErrorRate,
                    SequenceIter candidateSequencesBegin,
                    SequenceIter candidateSequencesEnd,
                    CountIter candidateCountsBegin,
                    CountIter candidateCountsEnd,
                    QualityIter candidateQualitiesBegin,
                    QualityIter candidateQualitiesEnd) noexcept{

            // loop over alignments and insert them to the graph
		auto alignmentiter = alignmentsBegin;
		auto countiter = candidateCountsBegin;
		auto candidateQualityiter = candidateQualitiesBegin;
#if 0
            for(auto t = std::make_tuple(alignmentsBegin, candidateSequencesBegin, candidateCountsBegin, candidateQualitiesBegin);
                std::get<0>(t) != alignmentsEnd;
                std::get<0>(t)++, std::get<1>(t)++, std::get<2>(t)++/*quality iter is incremented in loop body*/){

                auto& alignmentiter = std::get<0>(t);
                //auto& sequenceiter = std::get<1>(t);
                auto& countiter = std::get<2>(t);
                auto& candidateQualityiter = std::get<3>(t);
#endif
	    for(; alignmentiter != alignmentsEnd; alignmentiter++, countiter++){

                assert(*alignmentiter != nullptr);
                errorgraphdetail::split_subs((*alignmentiter)->operations, sequence_to_correct);

                const double weight = 1.0 - std::sqrt((*alignmentiter)->get_nOps() / ((*alignmentiter)->get_overlap() * desiredAlignmentMaxErrorRate));

                int last_a = (*alignmentiter)->get_subject_begin_incl() + (*alignmentiter)->get_overlap();

                normalizeAlignment(*(*alignmentiter), sequence_to_correct); //returns immediatly if already normalized (e.g. by previous insert)

        		const auto linkOps = makeLinkOperations(*(*alignmentiter));

        		const int default_prev_node = (*alignmentiter)->get_subject_begin_incl() == 0 ?
                                                (*alignmentiter)->get_subject_begin_incl()
                                                : unknown_vertex_index;

                const int count = *countiter; //number identical sequences with different quality scores

                for(int i = 0; i < count; i++, candidateQualityiter++){
                    const std::string* const qualitypointer = *candidateQualityiter;
                    int qindex = (*alignmentiter)->get_query_begin_incl();

                    int prev_node = default_prev_node;

                    for (const auto& op : linkOps) {

            			if (op.isOriginal) {
            				for (int j = op.from; j < op.to; j++, qindex++) {
            					if (prev_node == unknown_vertex_index) {
            						prev_node = j + 1;
            					}else{
            						double qweight = weight;
            						if(qualitypointer != nullptr){
            							if(qindex < (*alignmentiter)->get_query_begin_incl() + (*alignmentiter)->get_overlap()){
            								qweight *= qscore_to_weight[(unsigned char)(*qualitypointer)[qindex]];
            							}
            						}
            						prev_node = makeLink(prev_node, j + 1, sequence_to_correct[j], qweight);
            					}
            				}
            			}else {
            				for(size_t j = 0; j < op.chs.size(); j++, qindex++){
            					const char c = op.chs[j];
            					if (prev_node == unknown_vertex_index) {
            						prev_node = addNewNode(c);
            						topoIndices.insert(topoIndices.begin(), prev_node);
            					}else{
            						double qweight = weight;
            						if(qualitypointer != nullptr){
            							if(qindex < (*alignmentiter)->get_query_begin_incl() + (*alignmentiter)->get_overlap()){
            								qweight *= qscore_to_weight[(unsigned char)(*qualitypointer)[qindex]];
            							}
            						}
            						prev_node = makeLink(prev_node, unknown_destination_index, c, qweight);
            					}
            				}
            			}
            		}
                    if (last_a == int(sequence_to_correct.length()) && prev_node != unknown_vertex_index) {
                        makeLink(prev_node, endNode, 'E', weight);
                    }
                }
            }
        }

    	// insert edge from --weight--> to
    	// to == unknown_destination_index means it is unknown if to is already in the graph
    	// returns the index of to in vertices
    	int makeLink(const int from, const int to, const char base_of_to, const double weight)
    	{
    	#if 0
    		std::cout << "makeLink(" << from << ", " << to << ", " << base_of_to << ", " << weight << ")" << std::endl;
    	#endif
    		assert(to != 0); // there cannot be an edge to the start node
    		assert(from != to); // no edge to itself
    		assert(from >= 0);

    		int ret_to = to;

    		if (to == unknown_destination_index) {
    			// for each neighbor of from
    			for (Edge& edge : vertices[from].edges) {

    				//if neighbor is not a node from the initial read, and neighbor has the wanted base
    				if (edge.to > endNode && vertices[edge.to].base == base_of_to) {
    					edge.weight += weight;
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

    				vertices[from].edges.emplace_back(ret_to, weight);
    			}
    		}else { // to != unknown_destination_index
    			bool found = false;
    			// find the edge which ends in to and update the weight
    			for (Edge& edge : vertices[from].edges) {
    				if (edge.to == to) {
    					if(vertices[to].base != base_of_to)
    						assert(vertices[to].base == base_of_to);
    					edge.weight += weight;
    					found = true;
    					break;
    				}
    			}

    			// there is no edge yet. create this edge

    			if (!found){
    				vertices[from].edges.emplace_back(to, weight);
    			}

    		}
    		vertices[from].outedgeweightsum += weight;

    		return ret_to;
    	}


    	void calculateEdgeProbabilities(double alpha, double x) noexcept
    	{

            constexpr int no_original_edge = -1;
    		for (size_t i = 0; i < vertices.size(); ++i) {

    			Vertex& v = vertices[i];

    			double bestNonOriginalEdgeWeight = 0.0;
    			double originalEdgeWeight = 0.0;
    			//int bestNonOriginalEdge = -1;
    			int originalEdge = no_original_edge;

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


    			if(originalEdge != no_original_edge){

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

        CorrectionResult extractCorrectedSequence(double alpha, double x){
            #ifdef ASSERT_TOPOLOGIC_SORT
                // better be safe
                assertTopologicallySorted();
            #endif

            calculateEdgeProbabilities(alpha, x);

            //find most reliable path, which should be equivalent to the corrected read

            for (const int& i : topoIndices) {
                const Vertex& cur = vertices[i];

                for (const Edge& e : cur.edges) {
                    if (e.canBeUsed) {

                        //it is important that bestPathProb of the start node is set to 0 during init.
                        //otherwise, the condition will never be met if start node only has one outgoing edge with probability 1
                        double newBestLogPathProb = cur.bestPathProb + std::log(e.prob);
                        if (newBestLogPathProb > vertices[e.to].bestPathProb) {
                            vertices[e.to].bestPathProb = newBestLogPathProb;
                            vertices[e.to].bestprevnode = i;
                        }
                    }
                }
            }

            CorrectionResult result;
            result.probability = std::exp(vertices[endNode].bestPathProb);
            result.correctedSequence.reserve(vertices.size());

            std::string rcorrectedRead = "";

            // backtrack to extract corrected read

            Vertex cur('F');
            try{
                cur = vertices.at(vertices.at(endNode).bestprevnode);
            }catch(std::out_of_range ex){
                throw ex;
            }
            int currentVertexNumber = cur.bestprevnode;

            //std::vector<int> finalPath;
            finalPath.clear();
            finalPath.reserve(vertices.size());
            finalPath.push_back(endNode);
            finalPath.push_back(vertices.at(endNode).bestprevnode);

            while (cur.bestprevnode != -1) {
                finalPath.push_back(currentVertexNumber);
                result.correctedSequence += cur.base;
                try{
                    cur = vertices.at(cur.bestprevnode);
                }catch(std::out_of_range ex){
                    printf("bestprevnode %d\n", cur.bestprevnode);
                    throw ex;
                }
                currentVertexNumber = cur.bestprevnode;
            }

            //std::reverse(path.begin(), path.end());

            std::reverse(result.correctedSequence.begin(), result.correctedSequence.end());

            return result;
        }


        template<class AlignmentIter, class SequenceIter, class CountIter, class QualityIter>
        CorrectionResult correct(const std::string& sequence_to_correct,
                    const std::string* quality_of_sequence_to_correct,
                    AlignmentIter alignmentsBegin,
                    AlignmentIter alignmentsEnd,
                    double desiredAlignmentMaxErrorRate,
                    SequenceIter candidateSequencesBegin,
                    SequenceIter candidateSequencesEnd,
                    CountIter candidateCountsBegin,
                    CountIter candidateCountsEnd,
                    QualityIter candidateQualitiesBegin,
                    QualityIter candidateQualitiesEnd,
                    double alpha,
                    double x) noexcept{

            init(sequence_to_correct, quality_of_sequence_to_correct);

            add_candidates(sequence_to_correct,
                        quality_of_sequence_to_correct,
                        alignmentsBegin,
                        alignmentsEnd,
                        desiredAlignmentMaxErrorRate,
                        candidateSequencesBegin,
                        candidateSequencesEnd,
                        candidateCountsBegin,
                        candidateCountsEnd,
                        candidateQualitiesBegin,
                        candidateQualitiesEnd);

            CorrectionResult result = extractCorrectedSequence(alpha, x);

    		return result;
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
    	void normalizeAlignment(Alignment& alignment, const std::string& sequence_to_correct) const noexcept{
            using Op_t = typename Alignment::Op_t;
    		if(alignment.get_isNormalized())
    			return;

            auto& alignOps = alignment.operations;

    		int na = int(sequence_to_correct.length());
    		int last_val = na;

    		// delay operations as long as possible

    		for (int i = alignOps.size() - 1; i >= 0; i--) {
    			auto& op = alignOps[i];
    			int position = op.position;
    			const int base = op.base;

    			if (op.type == Op_t::Type::del) {
    				position++;
    				while (position < last_val && sequence_to_correct[position] == base)
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

    				if (position < na && sequence_to_correct[position] == base) {
    					position++;
    					while (position < na && sequence_to_correct[position] == base)
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

    							if (position2 < na && sequence_to_correct[position2] == base2) {
    								position2++;
    								while (position2 < na && sequence_to_correct[position2] == base2)
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
    	std::vector<LinkOperation> makeLinkOperations(const Alignment& alignment) const noexcept{
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
    				linkOps.emplace_back(cur_a, ca, true);
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
    			linkOps.emplace_back(cur_a, last_a, true);
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

    };
}

}
#endif
