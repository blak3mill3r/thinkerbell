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

void DeepBeliefNetwork::set_stream( const cuda::Stream &stream )
{ m_stream = &stream; }

// call this after training step
// it returns the average absolute weight adjustment per connection
float DeepBeliefNetwork::average_weight_adjustment( const Vertex &v ) 
{
  graph_traits< Graph >::in_edge_iterator in_i, in_end;
  tie( in_i, in_end ) = in_edges( v, m_graph );
  m_graph[*in_i].rbm->m_W_statistics.device_to_host();
  weight_type * f = m_graph[*in_i].rbm->m_W_statistics.weights();
  int size = m_graph[v].neurons->size();
  float sum = 0.0;
  for(int i =0; i< size; ++i) sum += f[i];

  int numweights = m_graph[*in_i].rbm->m_W.size();
  return (sum/numweights); 
}

// create a connection from A to B
Edge DeepBeliefNetwork::connect( const Vertex &va, const Vertex &vb )
{
  Edge e = add_edge(va, vb, m_graph).first;
  m_graph[e].rbm = new Rbm( //FIXME
    m_graph[va].neurons,
    m_graph[vb].neurons
  );

  // FIXME these next two lines are temporary
  // they belong somewhere else
  m_graph[e].rbm->randomize_weights();
  m_graph[e].rbm->m_W.host_to_device();

  return e;
}

// create a neuron blob
Vertex DeepBeliefNetwork::add_neurons( uint num_neurons, const std::string name )
{
  // add a vertex:
  Vertex v = add_vertex(m_graph);
  // create a new Neurons and assign it to the new vertex
  m_graph[v].name = name;
  m_graph[v].neurons = new Neurons(num_neurons);  //FIXME
  return v;
}

void DeepBeliefNetwork::debugify()
{
  list<Vertex> activation_order;
  list<Vertex>::iterator i;

  topological_sort(m_graph, front_inserter(activation_order));
  cout << "activation ordering: ";
  for (i = activation_order.begin(); i != activation_order.end(); ++i) 
    cout << *i << " ";
  
  cout << endl << endl;

}

// sets activation of a Vertex based on all of its inputs
void DeepBeliefNetwork::activate_vertex( const Vertex &v )
{
  //cout << "activating vertex " << v << "\t\"" << m_graph[v].name << "\"" << endl;
  // get a list of in-edges
  graph_traits< Graph >::in_edge_iterator in_i, in_end;
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
  //cout << "activating vertex " << v << "\t\"" << m_graph[v].name << "\"" << endl;
  // get a list of out-edges
  graph_traits< Graph >::out_edge_iterator out_i, out_end;
  tie( out_i, out_end ) = out_edges( v, m_graph );
  uint num_inputs = (out_end - out_i);
  switch( num_inputs )
  {
    case 0:         // the highest-level-neurons ... no outputs ... do nothing
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
  //cout << "training vertex " << v << "\t\"" << m_graph[v].name << "\"" << endl;
  // get a list of in-edges
  graph_traits< Graph >::in_edge_iterator in_i, in_end;
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

// activates in topological order up to the highest-level-Vertex (the one with no out edges)
// then performs one training step on that highest-level-Vertex
void DeepBeliefNetwork::training_step( )
{
  list<Vertex> activation_order;
  list<Vertex>::iterator i;

  topological_sort(m_graph, std::front_inserter(activation_order));

  // activate in topological order
  for (i = activation_order.begin(); i != activation_order.end(); ++i) 
    activate_vertex( *i );

  // train the highest-level-Vertex
  i = activation_order.end();
  i--;
  training_step_vertex( *i );
}

// activate forwards/upwards/perceptionwise through the whole graph
void DeepBeliefNetwork::perceive( )
{
  list<Vertex> activation_order;
  list<Vertex>::iterator i;

  topological_sort(m_graph, std::front_inserter(activation_order));

  // activate in topological order
  for (i = activation_order.begin(); i != activation_order.end(); ++i) 
    activate_vertex( *i );
}

// activate backwards/downwards/fantasywise through the whole graph
// (note: assumes that the highest level neurons has meaningful activations before this function is called, e.g. right after calling perceive())
void DeepBeliefNetwork::fantasize( )
{
  list<Vertex> activation_order;
  list<Vertex>::iterator i;

  topological_sort(m_graph, std::back_inserter(activation_order));

  // activate in reverse topological order
  for (i = activation_order.begin(); i != activation_order.end(); ++i) 
    inverted_activate_vertex( *i );
}

// temporary ... grab the last training example and copy it into foo
// FIXME get rid of it
activation_type * DeepBeliefNetwork::get_training_example()
{
  list<Vertex> activation_order;
  list<Vertex>::iterator i;

  topological_sort(m_graph, std::front_inserter(activation_order));
  i = activation_order.begin();
  Neurons * n = m_graph[*i].neurons;
  n->device_to_host();
  return n->activations();

}


}

