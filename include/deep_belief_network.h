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
#include <boost/foreach.hpp>

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
  > DBNGraph;

typedef graph_traits<DBNGraph>::vertex_descriptor Vertex;
typedef graph_traits<DBNGraph>::edge_descriptor Edge;

// a DBN is a graph with a Neurons instance at each vertex and an Rbm instance at each edge
// Edges represent connected Neurons pairs
// activating the target based on the source is "perceiving"
// activating the source based on the target is "fantasizing"
// a Node with no in-edges is an "input" Node
// a Node with no out-edges is a "output" Node
// Nodes with multiple in-edges or multiple out-edges is not implemented yet (in other words the graph might as well be a list at this point)
class DBN
{
  public:
    DBN();
    ~DBN();
    Vertex add_neurons( uint num_neurons, const std::string name = "anonymous neurons" );
    Edge connect( const Vertex &va, const Vertex &vb );
    list<Vertex>::const_iterator topological_order_begin() const { return m_topological_order.begin(); }
    list<Vertex>::const_iterator topological_order_end() const { return m_topological_order.end(); }
    list<Vertex>::const_iterator input_vertices_begin() const { return m_input_vertices.begin(); }
    list<Vertex>::const_iterator input_vertices_end() const { return m_input_vertices.end(); }
    list<Edge>::const_iterator training_edges_begin() const { return m_training_edges.begin(); }
    list<Edge>::const_iterator training_edges_end() const { return m_training_edges.end(); }
    list<Edge>::const_iterator non_training_edges_begin() const { return m_non_training_edges.begin(); }
    list<Edge>::const_iterator non_training_edges_end() const { return m_non_training_edges.end(); }
    list<Edge>::const_iterator all_edges_begin() const { return m_all_edges.begin(); }
    list<Edge>::const_iterator all_edges_end() const { return m_all_edges.end(); }
    Vertex top_vertex() { return m_topological_order.back(); }
    DBNGraph m_graph;

    int neurons_size( Vertex v )
      { return (m_graph)[v].neurons->size(); }

    int weights_size( Edge e )
      { return (m_graph)[e].rbm->m_W.size(); }

    Neurons * neurons( Vertex v )
      { return (m_graph)[v].neurons; }

    std::string neurons_name( Vertex v )
      { return (m_graph)[v].name; }

    bool is_in_training( Vertex v );
    bool is_in_training( Edge e );
    bool is_input_vertex( Vertex v );

  protected:
    void update_graph_metadata();
    list<Vertex> m_topological_order;
    list<Vertex> m_input_vertices;
    list<Edge> m_training_edges;
    list<Edge> m_non_training_edges;
    list<Edge> m_all_edges;

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
      update_graph_metadata();
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

} // namespace thinkerbell

#endif

