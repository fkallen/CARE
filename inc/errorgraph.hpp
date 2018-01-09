#ifndef ERROR_GRAPH_HPP
#define ERROR_GRAPH_HPP


#include <iostream>
#include <vector>
#include <string>
#include <assert.h>
#include <algorithm>
#include <fstream>
#include <functional>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <set>
#include <map>

#include "alignment.hpp"


void eg_global_init();


/*
	if defined, the relaxation to find the best path will add logarithms of probabilites
	instead of multiplying the probabilities. this may increase numerical accuracy
*/
#define LOGBASEDPATH

struct CorrectedRead {
	std::string sequence;
	double probability;
	/*std::vector<int> edgemax;
	std::vector<int> edgemin;
	std::vector<int> origpos;*/
};


struct ErrorGraph {

	struct Edge {
		int to;
		double weight;
		double prob;
		bool canBeUsed;

		Edge(int t, double w) : to(t), weight(w), canBeUsed(true)
		{
		}
	};

	struct Vertex {
		char base;
		std::vector<Edge> edges;
		double outedgeweightsum = 0.0;
		int bestprevnode = -1;

	#ifdef LOGBASEDPATH
		double bestLogPathProb = std::numeric_limits<double>::lowest();
	#else
		double bestPathProb = 0.0;
	#endif
		double incomingOrigReadProb = 0.0;

		Vertex(char b) : base(b)
		{
		}
	};

	struct LinkOperation {

		LinkOperation():from(0), to(0), isOriginal(false){}

		LinkOperation(int f, int t, bool o):from(f), to(t), isOriginal(o){}

		int from;
		int to;
		bool isOriginal; // if true, graph uses chars from its query, if false, use chars from chs
		std::vector<char> chs;
	};


	/*std::vector<int> edgemax;
	std::vector<int> edgemin;
	std::vector<int> origpos;*/

	std::vector<ErrorGraph::LinkOperation> previousLinkOperations;
	AlignResult* previousAlignResult;

	std::uint32_t readid = 0;

	int nodes = 0;
	int startNode;
	int endNode;
	std::vector<Vertex> vertices;

	const char* read;
	int readLength;
	const char* readqualityscores;
	bool useQscores = false;

	int insertCalls = 0;
	int totalInsertedAlignments = 0;


	std::vector<int> topoIndices; // the vertex ids topologically sorted

	std::vector<int> finalPath;

	ErrorGraph()
	{
	}

	ErrorGraph(const char* seq, int seqlength, const char* qualityScores, bool useQscores, const int nTimes = 1);

	// initialize the graph with the read
	void init(const char* seq, int seqlength, const char* qualityScores, bool useQscores_, const int nTimes = 1);

	// add new node, but don't add edges yet
	// return the index of the new node in the vertices vector
	int addNewNode(const char base);

	// insert alignment into graph
	void insertAlignment(AlignResult& alignment, const char* qualityScores, int overlapError, int overlapSize, double maxErrorRate, const int nTimes = 1);

	// insert edge (from --weight--> to)
	// to == -1 means it is unknown if to is already in the graph
	// returns the index of to in vertices
	int makeLink(const int from, const int to, const char base_of_to, const double weight, const int nTimes = 1);

	// for each edge calculate the probability of using this edge
	void calculateEdgeProbabilities(double alpha, double x);

	// extract corrected read from graph
	CorrectedRead getCorrectedRead(double alpha, double x);

	// print vertices and edges to files prefix+"vertices.txt" and prefix+"edges.txt"
	void dumpGraph(std::string prefix = "") const;

	// make sure the topologic sort is valid
	void assertTopologicallySorted() const;

	void normalizeAlignment(AlignResult& alignment) const;

	std::vector<LinkOperation> makeLinkOperations(const AlignResult& alignment) const;

};

#endif
