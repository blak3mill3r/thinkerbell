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
#include "trainers/base.h"
#include "training_example.h"

namespace thinkerbell {
using namespace std;
using namespace boost;

struct VertexProperties
{
  string name;
  bool active;
  Neurons *neurons;
  VertexProperties() : name("anonymous neurons"), active(true) {}
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
    void training_step( const cuda::Stream &stream );
    void set_example_factory( const AbstractTrainer *factory );
    float absolute_error( const Vertex &v );
  private:
    void activate_vertex( const Vertex &v, const cuda::Stream &stream );
    void training_step_vertex( const Vertex &v, const cuda::Stream &stream );
    // FIXME think of a better name for set_neurons_from_example
    void set_neurons_from_example( const Vertex &v, const TrainingExample &example );
    Graph m_graph;
    const AbstractTrainer * m_example_factory;
};

}
#endif

