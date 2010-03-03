#include <thinkerbell/deep_belief_network.h>

namespace thinkerbell {

using namespace std;
using namespace boost;

DBN::DBN()
{
}

DBN::~DBN()
{
}

void DBN::mask(Vertex v)
{
  m_graph[v].mask = true;
  update_graph_metadata();
}

void DBN::unmask(Vertex v)
{
  m_graph[v].mask = false;
  update_graph_metadata();
}

bool DBN::is_masked(Vertex v)
  { return m_graph[v].mask; }

// an edge is masked if either its source or target is masked
bool DBN::is_masked(Edge e)
  { return ( is_masked(source( e, m_graph )) | is_masked(target( e, m_graph )) ); }

// a vertex is the top vertex if it has no out edges
// OR if it's out edge is masked
bool DBN::is_top_vertex( Vertex v )
{
  graph_traits< DBNGraph >::out_edge_iterator out_i, out_end;
  tie(out_i, out_end) = out_edges( v, m_graph );
  bool no_out_edges = ( (out_end - out_i) == 0 );
  if(no_out_edges) return true;
  return is_masked(*out_i);
}

// a vertex is an input vertex iff it has no in-edges
bool DBN::is_input_vertex( Vertex v )
{
  graph_traits< DBNGraph >::in_edge_iterator in_i, in_end;
  tie(in_i, in_end) = in_edges( v, m_graph );
  return ((in_end - in_i) == 0 );
}

bool DBN::is_in_training( Vertex v )
{
  bool is_training_vertex = false;
  BOOST_FOREACH( Edge e, (in_edges(v, m_graph)) )
    {
      if( is_in_training( e ) ) { is_training_vertex = true; break; }
    }
  BOOST_FOREACH( Edge e, (out_edges(v, m_graph)) )
    {
      if( is_in_training( e ) ) { is_training_vertex = true; break; }
    }
  return is_training_vertex;
}

// an Edge is "in training" iff it's target is the top vertex
bool DBN::is_in_training( Edge e )
{
  return is_top_vertex( target( e, m_graph ) );
}


// create a connection from A to B
Edge DBN::connect( const Vertex &va, const Vertex &vb )
{
  Edge e = add_edge(va, vb, m_graph).first;
  m_graph[e].rbm = new Rbm( //FIXME problems serializing auto pointers
    m_graph[va].neurons,
    m_graph[vb].neurons
  );

  // the graph has changed
  update_graph_metadata();

  return e;
}

// create a neuron blob
Vertex DBN::add_neurons( uint num_neurons, const std::string name )
{
  // add a vertex:
  Vertex v = add_vertex(m_graph);
  // create a new Neurons and assign it to the new vertex
  m_graph[v].name = name;
  m_graph[v].neurons = new Neurons(num_neurons);  //FIXME problems serializing auto pointers

  // the graph has changed
  update_graph_metadata();

  return v;
}

void DBN::update_graph_metadata()
{
  // sort topologically:
  m_topological_order.clear();
  m_all_vertices.clear();
  m_input_vertices.clear();
  m_all_edges.clear();
  m_training_edges.clear();
  m_non_training_edges.clear();
  topological_sort(m_graph, front_inserter(m_topological_order));

  BOOST_FOREACH( Vertex currentv, m_topological_order )
  {
    if( !is_masked(currentv) )
    {
      m_all_vertices.push_back(currentv);
      if( is_input_vertex(currentv) )
        m_input_vertices.push_back(currentv);
    }
  }

  graph_traits< DBNGraph >::edge_iterator ei, eend;
  // build lists of training edges and non-training edges
  for(tie(ei, eend) = edges(m_graph); ei != eend; ei++)
  {
    Edge e = *ei;
    if( !is_masked(e) )
    {
      // is this a training edge or not?
      // it's a training edge iff it's target is the top
      if( is_top_vertex( target( e, m_graph ) ) )
        m_training_edges.push_back(e);
      else
        m_non_training_edges.push_back(e);
      m_all_edges.push_back(e);
    }
  }


}

} // namespace thinkerbell
