#include "../inc/errorgraph.hpp"

#include <cmath>
#include <utility>

#define ASSERT_TOPOLOGIC_SORT

double	qscore_to_error_prob[256];
double	qscore_to_graph_weight[256];


void eg_global_init(){

	constexpr int ASCII_BASE = 33;
	constexpr double MIN_GRAPH_WEIGHT = 0.001;

	for(int i = 0; i < 256; i++){
		if(i < ASCII_BASE)
			qscore_to_error_prob[i] = 1.0;
		else
			qscore_to_error_prob[i] = std::pow(10.0, -(i-ASCII_BASE)/10.0);
	}

	for(int i = 0; i < 256; i++){
		qscore_to_graph_weight[i] = std::max(MIN_GRAPH_WEIGHT, 1.0 - qscore_to_error_prob[i]);
	}
}

ErrorGraph::ErrorGraph(const char* seq, int seqlength, const char* qualityScores, bool useQscores_, const int nTimes){
	init(seq, seqlength, qualityScores, useQscores_, nTimes);
}


void ErrorGraph::init(const char* seq, int seqlength, const char* qualityScores, bool useQscores_, const int nTimes)
{
	assert(seqlength > 0);

	useQscores = useQscores_;
	readqualityscores = qualityScores;
	readLength = seqlength;
	read = seq;

	vertices.clear();

	double weight;
	int newindex;
	char base;

	newindex = addNewNode('S');
	topoIndices.push_back(newindex);

#ifdef LOGBASEDPATH
	vertices[newindex].bestLogPathProb = 0.0;
#else
	vertices[newindex].bestPathProb = 1.0;
#endif

	for (int i = 0; i < readLength; ++i) {
		base = read[i];
		newindex = addNewNode(base);
		topoIndices.push_back(newindex);

		double initialWeight = useQscores ? qscore_to_graph_weight[(unsigned char)qualityScores[i]] : 1.0;

		weight = initialWeight * nTimes;

		Edge edge(newindex, weight);
		vertices[newindex - 1].edges.push_back(edge);
		vertices[newindex - 1].outedgeweightsum += weight;
	}

	newindex = addNewNode('E');
	topoIndices.push_back(newindex);
	weight = 1.0 * nTimes;
	Edge edge(newindex, weight);
	vertices[newindex - 1].edges.push_back(edge);
	vertices[newindex - 1].outedgeweightsum += weight;

	startNode = 0;
	endNode = nodes - 1;
}

// add new node, but don't add edges yet
// return the index of the new node in the vertices vector
int ErrorGraph::addNewNode(const char base)
{
	nodes++;
	vertices.push_back(Vertex(base));
	return nodes - 1;
}

// insert alignment nTimes into the graph
void ErrorGraph::insertAlignment(AlignResult& alignment, const char* qualityScores, double maxErrorRate, const int nTimes)
{

	assert(!(useQscores && !qualityScores));

	insertCalls++;
	totalInsertedAlignments += nTimes;

	const double weight = 1 - std::sqrt(alignment.arc.nOps / (alignment.arc.overlap * maxErrorRate));
	//std::cout << "overlapError : " << alignment.arc.nOps << " overlapSize : " << alignment.arc.overlap << " maxErrorRate : " << maxErrorRate << " weight : " << weight << std::endl;

	int last_a = alignment.arc.subject_begin_incl + alignment.arc.overlap;

	normalizeAlignment(alignment); //returns immediatly if already normalized (e.g. by previous insert)

	//if(&alignment != previousAlignResult){
		previousLinkOperations = makeLinkOperations(alignment);
	//}

	previousAlignResult = &alignment;

	const auto& linkOps = previousLinkOperations;

	int prev_node = alignment.arc.subject_begin_incl;

	if (prev_node != 0) {
		prev_node = -1;
	}

	int qindex = alignment.arc.query_begin_incl;

	for (const auto& op : linkOps) {

		if (op.isOriginal) {
			for (int j = op.from; j < op.to; j++) {
				if (prev_node == -1) {
					prev_node = j + 1;
				}else{					
					double qweight = weight;
					if(useQscores){
						if(qindex < alignment.arc.query_begin_incl + alignment.arc.overlap){
							qweight *= qscore_to_graph_weight[(unsigned char)qualityScores[qindex]];
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
					if(useQscores){
						if(qindex < alignment.arc.query_begin_incl + alignment.arc.overlap){
							qweight *= qscore_to_graph_weight[(unsigned char)qualityScores[qindex]];
						}
					}
					prev_node = makeLink(prev_node, -1, c, qweight, nTimes);
				}
				qindex++;
			}
		}
	}

	if (last_a == readLength && prev_node != -1) {
		makeLink(prev_node, endNode, 'E', weight);
	}
}

// insert edge from --weight--> to
// to == -1 means it is unknown if to is already in the graph
// returns the index of to in vertices
int ErrorGraph::makeLink(const int from, const int to, const char base_of_to, const double weight, const int nTimes)
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


void ErrorGraph::calculateEdgeProbabilities(double alpha, double x)
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
CorrectedRead ErrorGraph::getCorrectedRead(double alpha, double x)
{

	if(totalInsertedAlignments == 0){
		CorrectedRead cr;

		cr.probability = 1.0;
		cr.sequence = read;
		return cr;
	}

#ifdef ASSERT_TOPOLOGIC_SORT
	// better be safe
	assertTopologicallySorted();
#endif

	calculateEdgeProbabilities(alpha, x);

	//find most reliable path, which should be equivalent to the corrected read

	for (const int& i : topoIndices) {
		const Vertex& cur = vertices[i];

		for (const Edge& e : cur.edges) {

			//if((readid == 7232071 || readid == 4495925) && e.to == 102) 
			//	printf("a");

#ifdef LOGBASEDPATH
			if (e.canBeUsed) {

				double newBestLogPathProb = cur.bestLogPathProb + log(e.prob);
				if (newBestLogPathProb > vertices[e.to].bestLogPathProb) {
					vertices[e.to].bestLogPathProb = newBestLogPathProb;
					vertices[e.to].bestprevnode = i;
				}
			}
#else
			if (e.canBeUsed) {

				double npw = cur.bestPathProb * e.prob;
				if (npw > vertices[e.to].bestPathProb) {
					vertices[e.to].bestPathProb = npw;
					vertices[e.to].bestprevnode = i;
				}
			}
#endif
		}
	}

	CorrectedRead cr;

#ifdef LOGBASEDPATH
	cr.probability = std::exp(vertices[endNode].bestLogPathProb);
#else
	cr.probability = vertices[endNode].bestPathProb;
#endif

	/*cr.edgemax = edgemax;
	cr.edgemax = edgemin;
	cr.edgemax = origpos;*/

/*std::cout << "b\n";
std::cout << cr.origpos.size() << '\n';
		for(int i = 0; i < cr.origpos.size(); i++){
			std::cout << cr.origpos[i] << '\n';
			std::cout << cr.edgemin[i] << '\n';
			std::cout << cr.edgemax[i] << '\n';
		}*/


	// backtrack to extract corrected read
	std::string rcorrectedRead = "";

	Vertex cur('F');
	try{
		cur = vertices.at(vertices.at(endNode).bestprevnode);
	}catch(std::out_of_range ex){
		printf("endnode %d, vertices.at(endNode).bestprevnode %d, readid %d\n", endNode, vertices.at(endNode).bestprevnode, readid);
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

	cr.sequence.resize(rcorrectedRead.size());
	std::reverse_copy(rcorrectedRead.begin(), rcorrectedRead.end(), cr.sequence.begin());
	return cr;
}




void ErrorGraph::dumpGraph(std::string prefix) const
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

void ErrorGraph::normalizeAlignment(AlignResult& alignment) const{
	if(alignment.arc.isNormalized)
		return;
	
	int na = readLength;
	int last_val = na;

	// delay operations as long as possible

	for (int i = alignment.operations.size() - 1; i >= 0; i--) {
		AlignOp& op = alignment.operations[i];
		int position = op.position;
		const int base = op.base;

		if (op.type == ALIGNTYPE_DELETE) {
			position++;
			while (position < last_val && read[position] == base)
				position++;
			position--;
			op.position = position;
			last_val = position;

			for (size_t j = i; j < alignment.operations.size() - 1; j++) {
				if (alignment.operations[j + 1].type == ALIGNTYPE_DELETE)
					break;

				if (alignment.operations[j].position >= alignment.operations[j + 1].position) {
					std::swap(alignment.operations[j], alignment.operations[j + 1]);
					alignment.operations[j].position--;
				}else break;
			}
		}else if (op.type == ALIGNTYPE_INSERT) {

			if (position < na && read[position] == base) {
				position++;
				while (position < na && read[position] == base)
					position++;
				op.position = position;
			}

			for (size_t j = i; j < alignment.operations.size() - 1; j++) {
				if (alignment.operations[j + 1].type == ALIGNTYPE_DELETE) {
					if (alignment.operations[j].position >= alignment.operations[j + 1].position) {

						// insertion and deletion of same base cancel each other. don't need this op
						if (alignment.operations[j].base == alignment.operations[j + 1].base) {
							alignment.operations.erase(alignment.operations.begin() + j + 1);
							alignment.operations.erase(alignment.operations.begin() + j);
							break;
						}

						const AlignOp temp = alignment.operations[j];
						alignment.operations[j] = alignment.operations[j + 1];
						alignment.operations[j + 1] = temp;

						if (alignment.operations[j + 1].position < na)
							alignment.operations[j + 1].position++;

						int position2 = alignment.operations[j + 1].position;
						const char base2 = alignment.operations[j + 1].base;

						if (position2 < na && read[position2] == base2) {
							position2++;
							while (position2 < na && read[position2] == base2)
								position2++;
							alignment.operations[j + 1].position = position2;
						}
					}else break;
				}else {
					if (alignment.operations[j].position > alignment.operations[j + 1].position) {
						std::swap(alignment.operations[j], alignment.operations[j + 1]);
						alignment.operations[j].position++;
					}else break;
				}
			}
		}
	}

	alignment.arc.isNormalized = true;
}

std::vector<ErrorGraph::LinkOperation> ErrorGraph::makeLinkOperations(const AlignResult& alignment) const{
	int cur_a = alignment.arc.subject_begin_incl;
	int last_a = cur_a + alignment.arc.overlap; 

	std::vector<LinkOperation> linkOps;

	for (size_t i = 0; i < alignment.operations.size(); i++) {
		const AlignOp& alignop = alignment.operations[i];
		const int ca = alignop.position;
		const char ch = alignop.base;

		if (cur_a < ca) {
			linkOps.push_back( LinkOperation(cur_a, ca, true) );
			cur_a = ca;
			i--;
		}else {
			LinkOperation linkop(ca, ca, false);

			if (alignop.type == ALIGNTYPE_DELETE) {
				linkop.to++;
				cur_a++;
			}else{
				linkop.chs.push_back(ch);
			}

			i++;
			for (; i < alignment.operations.size(); i++) {
				if (alignment.operations[i].position > cur_a) break;
				if (alignment.operations[i].type == ALIGNTYPE_DELETE) {
					linkop.to++;
					cur_a++;
				}else{
					linkop.chs.push_back(alignment.operations[i].base);
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

void ErrorGraph::assertTopologicallySorted() const
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
