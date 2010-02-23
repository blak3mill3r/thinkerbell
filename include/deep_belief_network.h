#ifndef __DEEP_BELIEF_NETWORK_H__
#define __DEEP_BELIEF_NETWORK_H__

#include <iostream>
#include <iterator>
#include <algorithm>

#include <boost/serialization/serialization.hpp>
#include <boost/utility.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/visitors.hpp>
#include <boost/graph/adj_list_serialize.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/serialization/scoped_ptr.hpp>

#include "types.h"
#include "neurons.h"
#include "rbm.h"

namespace thinkerbell {
using namespace std;
using namespace boost;
using namespace boost::lambda;

// stores all data unique to a vertex
struct VertexProperties
{
    string name;
    bool active;
    Neurons *neurons;
    VertexProperties() : name("anonymous neurons"), active(true) {}
  private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize( Archive & ar, const unsigned int version )
    {
      ar & name;
      ar & neurons;
    }
};

// stores all data unique to a edge
struct EdgeProperties
{
    uint total_input_samples, count_input_samples;
    Rbm *rbm;
    EdgeProperties() : total_input_samples(1), count_input_samples(0) {}
  private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize( Archive & ar, const unsigned int version )
    {
      ar & rbm;
    }
};

// our graph data type
//  a bidirectional adjacency list with vectors for both vertices and edges
typedef adjacency_list<
  vecS,               // out edge vector
  vecS,               // vertex vector
  bidirectionalS,     // bidirectional
  VertexProperties,   // vertex properties
  EdgeProperties      // edge properties
  > DeepBeliefNetworkGraph;

typedef graph_traits<DeepBeliefNetworkGraph>::vertex_descriptor Vertex;
typedef graph_traits<DeepBeliefNetworkGraph>::edge_descriptor Edge;

// a DeepBeliefNetwork is a graph with a Neurons instance at each vertex and an Rbm instance at each edge
// Edges represent connected Neurons pairs
// activating the target based on the source is "perceiving"
// activating the source based on the target is "fantasizing"
// a Node with no in-edges is an "input" Node
// a Node with no out-edges is a "output" Node
// Nodes with multiple in-edges or multiple out-edges is not implemented yet (in other words the graph might as well be a list at this point)
class DeepBeliefNetwork
{
  public:
    DeepBeliefNetwork();
    ~DeepBeliefNetwork();
    Vertex add_neurons( uint num_neurons, const std::string name = "anonymous neurons" );
    Edge connect( const Vertex &va, const Vertex &vb );
    list<Vertex> topological_order() const { return m_topological_order; }
    DeepBeliefNetworkGraph m_graph; // FIXME this should be protected, maybe make DeepBeliefNetworkScheduler a friend class

  protected:
    void update_topological_order();
    list<Vertex> m_topological_order;

  private:
    friend class boost::serialization::access;
    template<class Archive>
    void save( Archive & ar, const unsigned int version ) const
    {
      ar << m_graph;
    }
    template<class Archive>
    void load( Archive & ar, const unsigned int version )
    {
      ar >> m_graph;
      update_topological_order();
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

} // namespace thinkerbell

#endif

