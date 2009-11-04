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

// create a connection from A to B
Edge DeepBeliefNetwork::connect( const Vertex &va, const Vertex &vb )
{
  Edge e = add_edge(va, vb, m_graph).first;
  m_graph[e].rbm = new Rbm( //FIXME
    m_graph[va].neurons,
    m_graph[vb].neurons
  );
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
  typedef list<Vertex> ActivationOrder;
  ActivationOrder::iterator i;
  ActivationOrder activation_order;

  topological_sort(m_graph, std::front_inserter(activation_order));
  cout << "activation ordering: ";
  for (i = activation_order.begin(); i != activation_order.end(); ++i) 
    cout << *i << " ";
  
  cout << endl << endl;

}

// sets activation of a Neurons blob based on all of its inputs
void DeepBeliefNetwork::activate( const Vertex &v, const cuda::Stream &stream )
{
  // get a list of in-edges
  graph_traits< Graph >::in_edge_iterator in_i, in_end;
  tie( in_i, in_end ) = in_edges( v, m_graph );
  uint num_inputs = (in_end - in_i);
  switch( num_inputs )
  {
    case 0:         // a perceptron blob
      return;
      break;
    case 1:         // a blob which is activated by 1 input blob
      Edge edge = *in_i;
      //Vertex input_vertex = source( edge, m_graph );
      m_graph[edge].rbm->activate_b(stream);
      break;
  }

}

// performs a single training iteration (alternating Gibbs sampling and weight update)
void DeepBeliefNetwork::training_step( const Vertex &v, const cuda::Stream &stream )
{
  // get a list of in-edges
  graph_traits< Graph >::in_edge_iterator in_i, in_end;
  tie( in_i, in_end ) = in_edges( v, m_graph );
  uint num_inputs = (in_end - in_i);
  switch( num_inputs )
  {
    case 0:         // a perceptron blob
      set_neurons_from_example( v, m_example_factory->get_example() );
      break;
    case 1:         // a blob which is activated by 1 input blob
      Edge edge = *in_i;
      m_graph[edge].rbm->inverted_training_step(stream);
      break;
  }
}

void DeepBeliefNetwork::set_example_factory( const AbstractExampleFactory *factory )
{
  m_example_factory = factory;
}

void DeepBeliefNetwork::set_neurons_from_example( const Vertex &v, const TrainingExample &example )
{
  // device-to-device asynchronous copy
  cuda::memcpy( m_graph[v].neurons->m_device_memory.ptr(),
                example.get_device_ptr(),
                m_graph[v].neurons->m_device_memory.size() );
}

}

