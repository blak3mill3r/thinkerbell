#include "deep_belief_network/memory_mapper.h"

namespace thinkerbell {

DBNMemoryMapper::DBNMemoryMapper( DBNScheduler * dbn_scheduler_
                                , DBN * dbn_
                                , int batch_size_
                                , int num_example_batches_
                                )
  : dbn( dbn_ )
  , dbn_scheduler( dbn_scheduler_ )
  , batch_size( batch_size_ )
  , num_example_batches( num_example_batches_ )
  , temporary_memory_minimum_size(0)
{
}

void DBNMemoryMapper::allocate_device_memory( DevicePtr weights_memory_ptr_
                                            , DevicePtr biases_memory_ptr_
                                            , DevicePtr example_memory_ptr_
                                            , DevicePtr temporary_memory_ptr_
                                            , DevicePtr randoms_memory_ptr_
                                            , DevicePtr random_configs_memory_ptr_
                                            )
{
  weights_memory_ptr    = weights_memory_ptr_;
  biases_memory_ptr     = biases_memory_ptr_;
  example_memory_ptr    = example_memory_ptr_;
  temporary_memory_ptr  = temporary_memory_ptr_; 
  randoms_ptr_          = randoms_memory_ptr_; 
  random_configs_ptr_   = random_configs_memory_ptr_; 

  // decide on a temporary memory layout
  map_temporary_ptrs();

  // iterate through weights_memory_layout_map creating pointers and inserting them in weights_ptr_map
  DevicePtr currentp = weights_memory_ptr;
  pair<Edge, pair<int,bool> > jj;
  BOOST_FOREACH( jj, make_pair(weights_memory_layout_map.begin(), weights_memory_layout_map.end()) )
  {
    currentp = map_weights_ptrs( jj, currentp );
  }

  // iterate through biases_memory_layout_map creating pointers and inserting them in biases_ptr_map
  currentp = biases_memory_ptr;
  pair<Vertex, pair<int,bool> > vv;
  BOOST_FOREACH( vv, make_pair(biases_memory_layout_map.begin(),biases_memory_layout_map.end()) )
  {
    currentp = map_biases_ptrs( vv, currentp );
  }

  // assign DevicePtrs for the start of each example buffer for each input vertex
  int offset = 0;
  BOOST_FOREACH( Vertex v, make_pair(dbn->input_vertices_begin(),dbn->input_vertices_end()))
  {
    for(int z=0; z<3; ++z) 
      example_memory_ptr_map[v].push_back( example_memory_ptr + sizeof(float) * (z * example_buffer_size + offset) );
    offset += dbn->neurons_size(v) * batch_size;
  }
}

int DBNMemoryMapper::randoms_memory_size()
  { return (5860*4096*sizeof(float)); }

int DBNMemoryMapper::random_configs_memory_size()
  { return (MT_RNG_COUNT * sizeof(mt_struct_stripped)); }

int DBNMemoryMapper::temporary_memory_size()
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


int DBNMemoryMapper::example_memory_size()
{
  int example_batch_size = 0;
  BOOST_FOREACH( Vertex v, make_pair(dbn->input_vertices_begin(),dbn->input_vertices_end()))
  {
    example_batch_size += dbn->neurons_size(v) * batch_size;
  }
  example_buffer_size = example_batch_size * num_example_batches;
  return (sizeof(float) * example_buffer_size * 3);
}

DevicePtr DBNMemoryMapper::map_weights_ptrs( const pair<Edge,pair<int,bool> > &edge_and_layout, DevicePtr p )
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

DevicePtr DBNMemoryMapper::map_biases_ptrs( const pair<Vertex,pair<int,bool> > &vertex_and_layout, DevicePtr p )
{
  Vertex v;
  pair<int,bool> layout;
  tie(v, layout) = vertex_and_layout;
  int memory_requirement;
  bool triple_buffer;
  tie(memory_requirement, triple_buffer) = layout;
  if (triple_buffer) {
    int buffer_size = memory_requirement / 3;
    for( int i = 0; i < 3; ++i )
      (biases_ptr_map[v]).push_back( p + (i * buffer_size * sizeof(float)) );
  } else {
    for( int i = 0; i < 3; ++i )
      (biases_ptr_map[v]).push_back( p );
  }
  return( p + (sizeof(float) * memory_requirement) );
}

int DBNMemoryMapper::weight_matrix_size( Edge e )
{
  Vertex sourcev = source( e, dbn->m_graph );
  Vertex targetv = target( e, dbn->m_graph );
  int sourcevsize = dbn->neurons_size(sourcev) 
    , targetvsize = dbn->neurons_size(targetv)
    ;
  return (sourcevsize * targetvsize);
}

int DBNMemoryMapper::neurons_batch_size( Vertex v )
{
  return ( dbn->neurons_size(v) * batch_size );
}

void DBNMemoryMapper::weights_memory_requirements( Edge e, bool triple_buffered )
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

int DBNMemoryMapper::weights_memory_size()
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
          , (var(total_size) = var(total_size) +
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
  return (sizeof(float) * total_size);
}

int DBNMemoryMapper::biases_memory_size()
{
  Vertex current;
  BOOST_FOREACH( current, make_pair(dbn->topological_order_begin(),dbn->topological_order_end()) )
  {
    int size = dbn->neurons_size(current);
    int requirement = dbn->is_in_training(current) ? 3*size : size ;
    biases_memory_layout_map[current] = make_pair(requirement, dbn->is_in_training(current));
  }

  int total_size = 0;

  pair<Vertex, pair<int,bool> > vv;
  BOOST_FOREACH(vv, biases_memory_layout_map)
  {
    total_size += vv.second.first;
  }

  return (sizeof(float) * total_size);
}

} // end namespace thinkerbell
