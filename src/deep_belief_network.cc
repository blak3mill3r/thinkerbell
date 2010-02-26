#include "deep_belief_network.h"

namespace thinkerbell {

using namespace std;
using namespace boost;

DeepBeliefNetwork::DeepBeliefNetwork()
{
}

DeepBeliefNetwork::~DeepBeliefNetwork()
{
}

// a vertex is an input vertex iff it has no in-edges
bool DeepBeliefNetwork::is_input_vertex( Vertex v )
{
  graph_traits< DeepBeliefNetworkGraph >::in_edge_iterator in_i, in_end;
  tie(in_i, in_end) = in_edges( v, m_graph );
  return ((in_end - in_i) == 0 );
}

bool DeepBeliefNetwork::is_in_training( Vertex v )
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

// an Edge is "in training" iff it's target has no out edges
bool DeepBeliefNetwork::is_in_training( Edge e )
{
  Vertex v = target( e, m_graph );
  graph_traits< DeepBeliefNetworkGraph >::out_edge_iterator out_i, out_end;
  tie(out_i, out_end) = out_edges( v, m_graph );
  return ((out_end - out_i)==0);
}


// create a connection from A to B
Edge DeepBeliefNetwork::connect( const Vertex &va, const Vertex &vb )
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
Vertex DeepBeliefNetwork::add_neurons( uint num_neurons, const std::string name )
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

void DeepBeliefNetwork::update_graph_metadata()
{
  // sort topologically:
  m_topological_order.clear();
  m_training_edges.clear();
  m_non_training_edges.clear();
  m_all_edges.clear();
  m_input_vertices.clear();
  topological_sort(m_graph, std::front_inserter(m_topological_order));

  BOOST_FOREACH( Vertex currentv, m_topological_order )
  {
    if( is_input_vertex(currentv) ) m_input_vertices.push_back(currentv);
  }

  graph_traits< DeepBeliefNetworkGraph >::edge_iterator ei, eend;
  // build lists of training edges and non-training edges
  for(tie(ei, eend) = edges(m_graph); ei != eend; ei++)
  {
    // is this a training edge or not?
    Edge e = *ei;
    Vertex tv = target( e, m_graph );
    graph_traits< DeepBeliefNetworkGraph >::out_edge_iterator out_i, out_end;
    tie(out_i, out_end) = out_edges( tv, m_graph );
    bool no_out_edges = ( (out_end - out_i) == 0 );
    //( no_out_edges ? m_training_edges : m_non_training_edges ).push_back(e);
    if( no_out_edges )
      m_training_edges.push_back(e);
    else
      m_non_training_edges.push_back(e);
    m_all_edges.push_back(e);
  }


}

} // namespace thinkerbell


/*
// call this after training step
// it returns the average absolute weight adjustment per connection
float DeepBeliefNetwork::average_weight_adjustment( const Vertex &v ) 
{
  graph_traits< DeepBeliefNetworkGraph >::in_edge_iterator in_i, in_end;
  tie( in_i, in_end ) = in_edges( v, m_graph );
  m_graph[*in_i].rbm->m_W_statistics.device_to_host();
  weight_type * f = m_graph[*in_i].rbm->m_W_statistics.weights();
  int size = m_graph[v].neurons->size();
  float sum = 0.0;
  for(int i =0; i< size; ++i) sum += f[i];

  int numweights = m_graph[*in_i].rbm->m_W.size();
  return (sum/numweights); 
}

// sets activation of a Vertex based on all of its inputs
void DeepBeliefNetwork::activate_vertex( const Vertex &v )
{
  // get a list of in-edges
  graph_traits< DeepBeliefNetworkGraph >::in_edge_iterator in_i, in_end;
  tie( in_i, in_end ) = in_edges( v, m_graph );
  uint num_inputs = (in_end - in_i);
  switch( num_inputs )
  {
    case 0:         // a perceptron blob
      set_neurons_from_example( v, m_example_trainer->get_example() );
      break;
    case 1:         // a blob which is activated by 1 input blob
      Edge edge = *in_i;
      //Vertex input_vertex = source( edge, m_graph );
      m_graph[edge].rbm->activate_b(*m_stream);
      break;
  }

}

// sets activation of a Vertex based on all of its outputs (fantasize)
void DeepBeliefNetwork::inverted_activate_vertex( const Vertex &v )
{
  // get a list of out-edges
  graph_traits< DeepBeliefNetworkGraph >::out_edge_iterator out_i, out_end;
  tie( out_i, out_end ) = out_edges( v, m_graph );
  uint num_inputs = (out_end - out_i);
  switch( num_inputs )
  {
    case 0:         // the highest-level-neurons ... no-op (it had better have meaningful activations before this function is called)
      break;
    case 1:         // a blob which is activated by 1 output blob
      Edge edge = *out_i;
      //Vertex input_vertex = source( edge, m_graph );
      m_graph[edge].rbm->activate_a(*m_stream);
      break;
  }

}

// performs a single training iteration (alternating Gibbs sampling and weight update)
void DeepBeliefNetwork::training_step_vertex( const Vertex &v )
{
  // get a list of in-edges
  graph_traits< DeepBeliefNetworkGraph >::in_edge_iterator in_i, in_end;
  tie( in_i, in_end ) = in_edges( v, m_graph );
  uint num_inputs = (in_end - in_i);
  switch( num_inputs )
  {
    case 0:         // a perceptron blob
      throw 69;     // makes no sense to "train" a perceptron blob, as it's activations are given
                    // execution ought not reach here anyway

    case 1:         // a blob which is activated by 1 input blob
      Edge edge = *in_i;
      m_graph[edge].rbm->training_step(*m_stream);
      break;
  }
}

void DeepBeliefNetwork::set_example_trainer( const AbstractTrainer *trainer )
{
  m_example_trainer = trainer;
}

void DeepBeliefNetwork::set_neurons_from_example( const Vertex &v, const TrainingExample &example )
{
  // device-to-device asynchronous copy
  cuda::memcpy( m_graph[v].neurons->m_device_memory.ptr(),
                example.get_device_ptr(),
                m_graph[v].neurons->m_device_memory.size() );
}

// perception through the whole graph
// then performs one training step on the highest-level-Vertex
void DeepBeliefNetwork::training_step( )
{
  perceive();

  // train the highest-level-Vertex
  training_step_vertex( topological_order.back() );
}

// activate forwards/upwards/perceptionwise through the whole graph
void DeepBeliefNetwork::perceive( )
{
  // activate in topological order
  for_each( topological_order.begin(),
            topological_order.end(),
            bind(&DeepBeliefNetwork::activate_vertex, this, _1 )
          );
}

// activate backwards/downwards/fantasywise through the whole graph
// (note: assumes that the highest level neurons has meaningful activations before this function is called, e.g. right after calling perceive())
void DeepBeliefNetwork::fantasize( )
{
  // activate in reverse topological order
  for_each( topological_order.rbegin(),
            topological_order.rend(),
            bind(&DeepBeliefNetwork::inverted_activate_vertex, this, _1 )
          );
}

// temporary ... grab the last training example and return a pointer to it
// FIXME get rid of it
activation_type * DeepBeliefNetwork::get_training_example()
{
  Neurons * n = m_graph[topological_order.front()].neurons;
  // copy back from device
  n->device_to_host();
  // return a pointer to the activations
  return n->activations();
}

*/


