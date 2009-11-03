#ifndef __DEEP_BELIEF_NETWORK_H__
#define __DEEP_BELIEF_NETWORK_H__

#include <iostream>
#include <iterator>
#include <algorithm>

#include <boost/utility.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/visitors.hpp>

#include "types.h"
#include "neurons.h"
#include "rbm.h"

namespace thinkerbell {
using namespace std;
using namespace boost;

struct VertexProperties
{
  string name;
  bool active;
  Neurons *neurons;
  VertexProperties() : active(true), name("anonymous neurons") {}
};

struct EdgeProperties
{
  uint total_input_samples, count_input_samples;
  Rbm *rbm;
  EdgeProperties() : total_input_samples(1), count_input_samples(0) {}
};

typedef adjacency_list<
  vecS,               // out edge vector
  vecS,               // vertex vector
  bidirectionalS,     // bidirectional
  VertexProperties,   // vertex properties
  EdgeProperties      // edge properties
  > Graph;

typedef graph_traits<Graph>::vertex_descriptor Vertex;
typedef graph_traits<Graph>::edge_descriptor Edge;

class DeepBeliefNetwork
{
  public:
    DeepBeliefNetwork();
    ~DeepBeliefNetwork();
    Vertex add_neurons( uint num_neurons, const std::string name = "anonymous neurons" );
    Edge connect( const Vertex &va, const Vertex &vb );
    void debugify();
    void activate( const Vertex &v, const cuda::Stream &stream );
    void training_step( const Vertex &v, const cuda::Stream &stream );
  private:
    Graph m_graph;
};

}
#endif

