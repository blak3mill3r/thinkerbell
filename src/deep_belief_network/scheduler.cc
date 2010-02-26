#include "deep_belief_network/scheduler.h"

namespace thinkerbell {
using namespace std;
using namespace boost::lambda;
using boost::lambda::bind;
using boost::lambda::_1;

DeepBeliefNetworkScheduler::DeepBeliefNetworkScheduler( DeepBeliefNetwork * dbn_, int batch_size_, int num_examples_ )
//, madd( module_test_kernels, "madd" )
  : batch_size( batch_size_ )
  , dbn( dbn_ )
  , num_examples( num_examples_ )
  , time_to_stop( false )
  , dmemory( new DeepBeliefNetworkMemoryMapper( this, dbn, batch_size, num_examples ) )
{
}

void DeepBeliefNetworkScheduler::operator()()
{
  Cuda context(0);
  Module module_test_kernels("src/test_kernels.cubin");
  Module module_rng_kernels("src/mersenne_twister_kernels.cubin");
  Function mmul( module_test_kernels, "mmul" )
         , mmultb( module_test_kernels, "mmul_transpose_b" )
         , activate_neurons( module_test_kernels, "activate_neurons" )
         //, rng( module_rng_kernels, "rng" )
         ;
  // begin/end events for each buffer
  Event exec_begin[3];
  Event exec_end[3];

  auto_ptr<DeviceMemory> weights_memory( new DeviceMemory( dmemory->weights_memory_size() ) );
  auto_ptr<DeviceMemory> example_memory( new DeviceMemory( dmemory->example_memory_size() ) );
  auto_ptr<DeviceMemory> temporary_memory( new DeviceMemory( dmemory->temporary_memory_size() ) );

  // allocate device memory for the dbn:
  dmemory->allocate_device_memory( weights_memory->ptr(), example_memory->ptr(), temporary_memory->ptr() );
  
  //////////////////////////////////////
  // get the triple buffering rolling...
  //////////////////////////////////////

  // 2 streams:
  vector<Stream *> streams;
  streams.push_back(new Stream());
  streams.push_back(new Stream());

  // first 2 steps are different:
  // there is no batch finishing with the current buffer
  exec_end[3].record( *streams[0] );
  exec_end[0].record( *streams[1] );
  // FIXME xfer examples into buffers A, B and C

  while(true)
  for(int i=0; i<2*3; ++i) // 6 is divisible by 2 and 3
  {
    // this gives us three phases
    int bufa = ((i+0)%3); // bufa weights will be written to because it will be bufc next iteration and will be used for activation steps
    int bufb = ((i+1)%3); // bufb weights will be used as the source of weights (because it was written to last step)
    int bufc = ((i+2)%3); // bufc weight buffers will be used for activation steps
    int streami = i%2;

    choose_random_example();

    // FIXME sometimes transfer examples into A buffer
    //BOOST_FOREACH( Vertex inputv, make_pair(dbn->input_vertices_begin(),dbn->input_vertices_end()))
    //{
    //  activate_from_example(inputv);
    //}

    // synch with the end of the execution of the last one using A buffers
    exec_end[bufa].synchronize();

    // for each vertex in topo order
    BOOST_FOREACH( Vertex v, make_pair(dbn->topological_order_begin(),dbn->topological_order_end()) )
    {
      //tmp_spaces_debug();
      if(dbn->is_input_vertex(v))
      { // v's activation amounts to setting neuron energies from a training example
        activate_neurons.setBlockShape( BLOCK_SIZE, BLOCK_SIZE, 1 );
        activate_neurons.go( dbn->neurons_size( v ) / BLOCK_SIZE
                           , batch_size / BLOCK_SIZE
                           , (*streams[streami])
                           , dmemory->vertex_ptr(v, bufa)//FIXME the address where to find the example
                           , dmemory->vertex_ptr(v, bufa)
                           , dmemory->vertex_ptr(v, bufa)//FIXME where to find randoms (which wont be used) .. can I pass NULL safely?
                           , dbn->neurons_size(v)
                           , false
                           );
      }
      else
      { // v gets activated by each of its in-edges:
        bool first_one = true;
        BOOST_FOREACH( Edge e, in_edges( v, dbn->m_graph ) )
        {
          Vertex sourcev = source( e, dbn->m_graph )
               , targetv = target( e, dbn->m_graph )
               ;
          int sourcev_size = dbn->neurons_size( sourcev )
            , targetv_size = dbn->neurons_size( targetv )
            ;
          mmul.setBlockShape( BLOCK_SIZE, BLOCK_SIZE, 1 );
          if(first_one)
          {
            mmul.go( targetv_size / BLOCK_SIZE
                   , batch_size / BLOCK_SIZE
                   , *(streams[streami])
                   , dmemory->vertex_ptr( targetv, bufa )
                   , dmemory->vertex_ptr( sourcev, bufa )
                   , dmemory->weights_ptr( e, bufa )
                   , NULL
                   , true
                   , sourcev_size
                   , targetv_size
                   );
            first_one = false;
          } else {
            mmul.go( targetv_size / BLOCK_SIZE
                   , batch_size / BLOCK_SIZE
                   , *(streams[streami])
                   , dmemory->vertex_ptr( targetv, bufa )
                   , dmemory->vertex_ptr( sourcev, bufa )
                   , dmemory->weights_ptr( e, bufa )
                   , dmemory->vertex_ptr( targetv, bufa )
                   , false
                   , sourcev_size
                   , targetv_size
                   );
          }
        }
      }
      // set v's activations based on energies
      activate_neurons.setBlockShape( BLOCK_SIZE, BLOCK_SIZE, 1 );
      activate_neurons.go( dbn->neurons_size( v ) / BLOCK_SIZE
                         , batch_size / BLOCK_SIZE
                         , (*streams[streami])
                         , dmemory->vertex_ptr(v, bufa)
                         , dmemory->vertex_ptr(v, bufa)
                         , dmemory->vertex_ptr(v, bufa)// FIXME address of randoms
                         , dbn->neurons_size(v)
                         , true
                         );

    }
    
    // synch with B-execution done
    // we're expecting not to wait here but better safe than sorry
    exec_end[bufc].synchronize();

    exec_end[bufa].record( *(streams[streami]) );

    if( time_to_stop )
      goto alldone;
  }

  alldone:
  Logger::log("wait for streams to finish");
  streams[0]->synchronize();
  streams[1]->synchronize();
  Logger::log("cleanup");
  delete streams[1];
  delete streams[0];
  Logger::log("done!");
  
}


// FIXME todo for the mapper: weight_delta_memory should be zeroed before starting
DeepBeliefNetworkMemoryMapper::DeepBeliefNetworkMemoryMapper( DeepBeliefNetworkScheduler * dbn_scheduler_
                                                            , DeepBeliefNetwork * dbn_
                                                            , int batch_size_
                                                            , int num_examples_
                                                            )
  : dbn( dbn_ )
  , dbn_scheduler( dbn_scheduler_ )
  , batch_size( batch_size_ )
  , num_examples( num_examples_ )
  , temporary_memory_minimum_size(0)
{
  cout << "Constructozoid!" << endl;
}

void DeepBeliefNetworkMemoryMapper::allocate_device_memory( DevicePtr weights_memory_ptr_
                                                          , DevicePtr example_memory_ptr_
                                                          , DevicePtr temporary_memory_ptr_
                                                          )
{

  weights_memory_ptr = weights_memory_ptr_;
  example_memory_ptr = example_memory_ptr_;
  temporary_memory_ptr = temporary_memory_ptr_; 

  map_temporary_ptrs();

  DevicePtr currentp = weights_memory_ptr;

  // iterate through weights_memory_layout_map creating pointers and inserting them in weights_ptr_map
  for_each( weights_memory_layout_map.begin()
          , weights_memory_layout_map.end()
          , var(currentp) = ret< DevicePtr >(lambda::bind( &DeepBeliefNetworkMemoryMapper::map_weights_ptrs, this, _1, var(currentp)))
          );
}

int DeepBeliefNetworkMemoryMapper::temporary_memory_size()
{
  BOOST_FOREACH( Vertex v, make_pair(dbn->topological_order_begin(),dbn->topological_order_end()) )
  {
    tmp_alloc( v );
    //tmp_spaces_debug();
    if(!dbn->is_input_vertex(v))
    { // v gets activated by each of its in-edges:
      bool first_one = true;
      BOOST_FOREACH( Edge e, in_edges( v, dbn->m_graph ) )
      {
        activate_edge( e, !first_one );
        first_one = false;
      }
    }

  }
  return temporary_memory_minimum_size*3;
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

int DeepBeliefNetworkMemoryMapper::neurons_batch_size( Vertex v )
{
  return ( dbn->neurons_size(v) * batch_size );
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
  Edge current;
  BOOST_FOREACH( current, make_pair(dbn->non_training_edges_begin(),dbn->non_training_edges_end()) )
  {
    weights_memory_requirements(current, false);
  }
  BOOST_FOREACH( current, make_pair(dbn->training_edges_begin(),dbn->training_edges_end()) )
  {
    weights_memory_requirements(current, true);
  }

  int total_size = 0;

  for_each( weights_memory_layout_map.begin()
          , weights_memory_layout_map.end()
          , (var(total_size) =
               ret< int >(
                 lambda::bind( &pair< int, bool >::first
                                    , ret< pair< int,bool > >(
                                        lambda::bind( &pair< Edge, pair< int, bool > >::second
                                                           , lambda::_1
                                                           )
                                    ))
                         )
            )
          );
  cout << "weights memory size = " << total_size << " floats " << endl;
  return (sizeof(float) * total_size);
}

}
