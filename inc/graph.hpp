#ifndef GRAPH_HPP
#define GRAPH_HPP

#include "alignment.hpp"
#include "batchelem.hpp"

#include <vector>
#include <string>

namespace care{
namespace graphtools{

	namespace correction{

        struct TaskTimings{
        	std::chrono::duration<double> preprocessingtime{0};
        	std::chrono::duration<double> h2dtime{0};
        	std::chrono::duration<double> executiontime{0};
        	std::chrono::duration<double> d2htime{0};
        	std::chrono::duration<double> postprocessingtime{0};
        };

        struct GraphTimings{
            std::chrono::duration<double> buildtime{0};
            std::chrono::duration<double> correctiontime{0};
        };

		struct ErrorGraph {

			struct Edge {
				int to;
				double weight;
				double prob;
				bool canBeUsed;

				Edge(int t, double w);
			};

			struct Vertex {
				char base;
				std::vector<Edge> edges;
				double outedgeweightsum = 0.0;
				int bestprevnode = -1;

				double bestPathProb = 0.0;
				double incomingOrigReadProb = 0.0;

				Vertex(char b);
			};

			struct LinkOperation {

				LinkOperation();
				LinkOperation(int f, int t, bool o);

				int from;
				int to;
				bool isOriginal; // if true, graph uses chars from its query, if false, use chars from chs
				std::vector<char> chs;
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

			ErrorGraph();

			ErrorGraph(bool useQscores, double max_mismatch_ratio, double alpha, double x);

			// initialize the graph with the read
			void init(const std::string& seq, const std::string& qualityScores);

			// add new node, but don't add edges yet
			// return the index of the new node in the vertices vector
			int addNewNode(const char base);

            void clearVectors();

			// insert alignment into graph
			void insertAlignment(AlignResultCompact& alignment, std::vector<AlignOp>& ops,
                                 const std::string* qualityScores, int nTimes = 1);

			// insert edge (from --weight--> to)
			// to == -1 means it is unknown if to is already in the graph
			// returns the index of to in vertices
			int makeLink(const int from, const int to, const char base_of_to, const double weight, const int nTimes = 1);

			// for each edge calculate the probability of using this edge
			void calculateEdgeProbabilities();

			// extract corrected read from graph
			std::string getCorrectedRead();

			// print vertices and edges to files prefix+"vertices.txt" and prefix+"edges.txt"
			void dumpGraph(std::string prefix = "") const;

			// make sure the topologic sort is valid
			void assertTopologicallySorted() const;

			void normalizeAlignment(AlignResultCompact& alignment, std::vector<AlignOp>& ops) const;

			std::vector<LinkOperation> makeLinkOperations(const AlignResultCompact& alignment, const std::vector<AlignOp>& ops) const;

            TaskTimings correct_batch_elem(BatchElem& batchElem);

		};

	}

}
}
#endif
