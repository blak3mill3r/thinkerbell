#include "deep_belief_network/scheduler.h"

namespace thinkerbell {
using namespace std;
using namespace boost::lambda;
using boost::lambda::bind;
using boost::lambda::_1;

DeepBeliefNetworkMemoryMapper::DeepBeliefNetworkMemoryMapper( DeepBeliefNetwork * dbn_, int batch_size_, int example_buffer_size_ )
  : dbn( dbn_ ),
    batch_size( batch_size_ ),
    example_buffer_size( example_buffer_size_ ),
    weights_memory_layout_map(),
    weights_ptr_map(),
    weights_memory( 0x10 ),
    example_memory( 0x10 ),
    temporary_memory( 0x10 )
    //weights_memory( weights_memory_size() ),
    //example_memory( example_memory_size() ),
    //temporary_memory( temporary_memory_size() )
{
  cout << "constructing memory mapper..." << endl;
  cout << "weights size: " << weights_memory_size() << endl;
  cout << "example size: " << example_memory_size() << endl;
  cout << "temporary size: " << temporary_memory_size() << endl;
  DevicePtr currentp = weights_memory.ptr();
  // iterate through weights_memory_layout_map creating pointers and inserting them in weights_ptr_map
  for_each( weights_memory_layout_map.begin()
          , weights_memory_layout_map.end()
          , var(currentp) = ret< DevicePtr >(boost::lambda::bind( &DeepBeliefNetworkMemoryMapper::map_weights_ptrs, this, _1, var(currentp)))
          );
}


// two temporary spaces (so operations requiring temporary space can alternate back and forth, reading/writing from these two alternately)
// times three buffers (triple buffering)
// so the answer is 6 times the size of the biggest operand (which is either a weight matrix or a batch of neuron values)
int DeepBeliefNetworkMemoryMapper::temporary_memory_size()
{
  cout << "calculating temporary memory size" << endl;
  int max_size = 0;
  // set max_size to the size of the biggest neuron batch operand
  for_each( dbn->topological_order_begin()
          , dbn->topological_order_end()
          , var(max_size) = bind(
                              &std::max<int>,
                              var(max_size),
                              bind(
                                &DeepBeliefNetwork::neurons_size,
                                dbn,
                                _1)
                            )
          );

  max_size *= batch_size;

  // set max_size to the size of the biggest weight matrix operand iff its bigger than max_size
  for_each( dbn->all_edges_begin()
          , dbn->all_edges_end()
          , var(max_size) = ret<int>(boost::lambda::bind(
                              &std::max<int>,
                              var(max_size),
                              boost::lambda::bind(
                                &DeepBeliefNetwork::weights_size,
                                dbn,
                                _1)
                            ))
          );

  return (sizeof(float) * max_size * 6);
}


// FIXME only accomodates one input vertex
int DeepBeliefNetworkMemoryMapper::example_memory_size()
{
  Vertex inputv = *dbn->topological_order_begin();
  int example_size = dbn->neurons_size(inputv);
  example_buffer_size = num_examples * example_size;
  return (example_buffer_size * 3);
}

DevicePtr DeepBeliefNetworkMemoryMapper::map_weights_ptrs( const pair<Edge,pair<int,bool> > &edge_and_layout, DevicePtr p )
{
  Edge e;
  pair<int,bool> layout;
  tie(e, layout) = edge_and_layout;
  int memory_requirement;
  bool triple_buffer;
  tie(memory_requirement, triple_buffer) = layout;
  if (triple_buffer) {
    int buffer_size = memory_requirement / 3;
    for( int i = 0; i < 3; ++i )
      (weights_ptr_map[e]).push_back( p + (i * buffer_size * sizeof(float)) );
  } else {
    for( int i = 0; i < 3; ++i )
      (weights_ptr_map[e]).push_back( p );
  }
  return( p + (sizeof(float) * memory_requirement) );
}

void DeepBeliefNetworkMemoryMapper::weights_memory_requirements( Edge e, bool triple_buffered )
{
  Vertex sourcev = source( e, dbn->m_graph );
  Vertex targetv = target( e, dbn->m_graph );
  // get source and target sizes
  int sourcevsize = dbn->neurons_size(sourcev) 
    , targetvsize = dbn->neurons_size(targetv)
    ;
  // for edges which will be trained, 3 weight buffers are required
  // for all other edges, just 1 buffer is fine
  int buffer_size = sourcevsize * targetvsize;
  int requirement = triple_buffered ?  (3 * buffer_size) : (buffer_size);
  weights_memory_layout_map[e] = make_pair(requirement, triple_buffered);
}

int DeepBeliefNetworkMemoryMapper::weights_memory_size()
{
  for_each( dbn->non_training_edges_begin()
          , dbn->non_training_edges_end()
          , boost::lambda::bind( &DeepBeliefNetworkMemoryMapper::weights_memory_requirements
                               , this
                               , _1
                               , constant(false) // not a training edge
                               )
          );
  for_each( dbn->training_edges_begin()
          , dbn->training_edges_end()
          , boost::lambda::bind( &DeepBeliefNetworkMemoryMapper::weights_memory_requirements
                               , this
                               , _1
                               , constant(true) // a training edge
                               )
          );

  int total_size = 0;

  for_each( weights_memory_layout_map.begin()
          , weights_memory_layout_map.end()
          , (var(total_size) =
               ret< int >(
                 boost::lambda::bind( &pair< int, bool >::first
                                    , ret< pair< int,bool > >(
                                        boost::lambda::bind( &pair< Edge, pair< int, bool > >::second
                                                           , boost::lambda::_1
                                                           )
                                    ))
                         )
            )
          );

  return (sizeof(float) * total_size);
}

}
