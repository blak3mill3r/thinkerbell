#include "deep_belief_network/scheduler.h"

namespace thinkerbell {
using namespace std;
using namespace boost::lambda;
using boost::lambda::bind;
using boost::lambda::_1;

DeepBeliefNetworkScheduler::DeepBeliefNetworkScheduler( DeepBeliefNetwork * dbn_, int batch_size_, int num_examples_ )
  : batch_size( batch_size_ ),
    dbn( dbn_ ),
    num_examples( num_examples_ ),
    time_to_stop( false )
{}

void DeepBeliefNetworkScheduler::operator()()
{
  Cuda context(0);
  Module module_test_kernels("src/test_kernels.cubin");
  Function mmul( module_test_kernels, "mmul" );
  Function mmultb( module_test_kernels, "mmul_transpose_b" );
  //Function madd( module_test_kernels, "madd" );
  vector<Stream *> streams;

  // allocate device memory for the dbn:
  DeepBeliefNetworkMemoryMapper dmemory( dbn, batch_size, num_examples );

  //////////////////////////////////////
  // get the triple buffering rolling...
  //////////////////////////////////////

  // 2 streams:
  streams.push_back(new Stream());
  streams.push_back(new Stream());

  // begin/end events for each buffer
  Event exec_begin[3];
  Event exec_end[3];

  // first 2 steps are different:
  // there is no batch finishing with the current buffer
  exec_end[3].record( *streams[0] );
  exec_end[0].record( *streams[1] );
  // FIXME foreachinputvertex alloc space in A B C
  // FIXME xfer examples into buffers B and C

  while(true)
  for(int i=0; i<2*3; ++i) // 6 is divisible by 2 and 3
  {
    // this gives us three phases
    int bufa = ((i+0)%3); // bufa weights will be written to because it will be bufc next iteration and will be used for activation steps
    int bufb = ((i+1)%3); // 
    int bufc = ((i+2)%3); // bufc weight buffers will be used for activation steps
    int streami = i%2;

    // FIXME foreachinputvertex transfer examples into A buffer

    // synch with the end of the execution of the last one using C buffers
    // FIXME add timing
    exec_end[bufc].synchronize();

    // FIXME up activation through the whole graph using C buffers

    // synch with B-execution done and time it (it should be 0)
    // FIXME add the timing
    exec_end[bufb].synchronize();

    // FIXME foreachtrainingedge: weight sample 1, sum with values in weights-B overwriting values in the weight-delta-C
    // FIXME foreachtrainingedge: AGS steps down&up
    // FIXME foreachtrainingedge: weight sample 2, subtracting from values in weight-delta-C writing to weight-B

    // we're done with a training step on a vertex

    if( time_to_stop )
      goto alldone;

  }

  alldone:
  streams[0]->synchronize();
  streams[1]->synchronize();
  delete streams[1];
  delete streams[0];
  
}


// FIXME todo for the mapper: weight_delta_memory should be zeroed before starting
DeepBeliefNetworkMemoryMapper::DeepBeliefNetworkMemoryMapper( DeepBeliefNetwork * dbn_, int batch_size_, int example_buffer_size_ )
  : dbn( dbn_ )
  , batch_size( batch_size_ )
  , example_buffer_size( example_buffer_size_ )
  , weights_memory_layout_map()
  , weights_ptr_map()
  , weights_memory( weights_memory_size() )
  , weights_delta_memory( weights_delta_memory_size() )
  , example_memory( example_memory_size() )
  , temporary_memory( temporary_memory_size() )
{
  cout << "constructing memory mapper..." << endl;
  cout << "weights size: " << weights_memory_size() << endl;
  cout << "example size: " << example_memory_size() << endl;
  cout << "temporary size: " << temporary_memory_size() << endl;
  cout << "weights delta size: " << weights_delta_memory_size() << endl;

  // FIXME wrong
  // assign temp_ptr :
  for(int f=0; f<3; ++f)
  for(int u=0; u<2; ++u)
    temp_ptr[f][u] = temporary_memory.ptr()
                   + (sizeof(float) * temporary_buffer_size * (u * 3 + f));

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

int DeepBeliefNetworkMemoryMapper::weight_matrix_size( Edge e )
{
  Vertex sourcev = source( e, dbn->m_graph );
  Vertex targetv = target( e, dbn->m_graph );
  int sourcevsize = dbn->neurons_size(sourcev) 
    , targetvsize = dbn->neurons_size(targetv)
    ;
  return (sourcevsize * targetvsize);
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

int DeepBeliefNetworkMemoryMapper::weights_delta_memory_size()
{
  int total_size = 0;
  for_each( dbn->training_edges_begin()
          , dbn->training_edges_end()
          , var(total_size) = var(total_size)
          + lambda::bind( &DeepBeliefNetworkMemoryMapper::weight_matrix_size
                        , this
                        , _1
                        )
          );

  for_each( weights_delta_memory_layout_map.begin()
          , weights_delta_memory_layout_map.end()
          , (var(total_size) = var(total_size) + 
                ret<int>( lambda::bind( &pair<Edge,int>::second
                                      , _1
                                      )
                        )
            )
          );

  return (sizeof(float) * total_size);
}

}
