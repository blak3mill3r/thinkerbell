#ifndef __DBN_MEMORY_MAPPER_H__
#define __DBN_MEMORY_MAPPER_H__

#include <boost/foreach.hpp>
#include <cudamm/cuda.hpp>
#include <thinkerbell/deep_belief_network.h>
#include <thinkerbell/mersenne_twister.h>

namespace thinkerbell {

using namespace cuda;
using namespace std;

class DBNScheduler;

// class DBNMemoryMapper
//   manages the layout of device memory for the DBN algorithm
//   allocates space for weights, biases, temporary neuron activation values, and training examples
class DBNMemoryMapper : noncopyable
{
public:
  DBNMemoryMapper( DBNScheduler * dbn_scheduler_, DBN * dbn_, int batch_size_, int num_example_batches_ );
  void allocate_device_memory(DevicePtr, DevicePtr, DevicePtr, DevicePtr, DevicePtr, DevicePtr);

  DevicePtr weights_ptr( Edge e, int buffer_index )
    { return weights_ptr_map[e][buffer_index]; }

  DevicePtr example_ptr( Vertex v, int buffer_index )
    { return example_memory_ptr_map[v][buffer_index]; }

  DevicePtr neurons_ptr( Vertex v, int buffer_index )
    { return (temporary_vertex_memory_ptr[v][buffer_index]); }

  DevicePtr randoms_ptr()
    { return randoms_ptr_; }

  DevicePtr random_configs_ptr()
    { return random_configs_ptr_; }

  DevicePtr biases_ptr( Vertex v, int buffer_index )
    { return (biases_ptr_map[v][buffer_index]); }

  void upload_weights( Edge e, int buffer_index )
  {
    cuda::memcpy( weights_ptr(e, buffer_index)
                , dbn->m_graph[e].rbm->m_W.weights()
                , sizeof(float) * weight_matrix_size(e)
                );
  }

  void download_weights( Edge e, int buffer_index )
  {
    cuda::memcpy( dbn->m_graph[e].rbm->m_W.weights()
                , weights_ptr(e, buffer_index)
                , sizeof(float) * weight_matrix_size(e)
                );
  }

  void upload_biases( Vertex v, int buffer_index )
  {
    cuda::memcpy( biases_ptr(v, buffer_index)
                , dbn->m_graph[v].neurons->biases
                , sizeof(float) * dbn->neurons_size(v)
                );
  }

  void download_biases( Vertex v, int buffer_index )
  {
    cuda::memcpy( dbn->m_graph[v].neurons->biases
                , biases_ptr(v, buffer_index)
                , sizeof(float) * dbn->neurons_size(v)
                );
  }

  inline void activate_edge( Edge e, bool add_to_result )
  {
    // we can now free up some temporary memory, as the neuron batch operand that we just used to activate this vertex isn't needed anymore, unless it's a training vertex
    Vertex sourcev = source( e, dbn->m_graph );
    if( !dbn->is_in_training(sourcev) ) tmp_free(sourcev);
  }

  inline void tmp_alloc( Vertex v )
  {
    map<Vertex,int>::iterator found = temporary_vertex_memory_offsets.find(v);
    if(found == temporary_vertex_memory_offsets.end())
    { // no record of temp memory for this vertex yet, allocate some
      int size = sizeof(float) * neurons_batch_size(v);
      temporary_vertex_memory_offsets[v] = tmp_alloc(size);
    }
  }

  inline void tmp_free( Vertex v )
  {
    map<Vertex,int>::iterator foundi = temporary_vertex_memory_offsets.find(v);
    if(foundi == temporary_vertex_memory_offsets.end())
    { // no record of temp memory for this vertex yet, which is a bad sign since we were planning on freeing it
      cout << "Bad sign" << endl;
    }
    else
    {
      pair<Vertex,int> zzz = *foundi;
      tmp_free(zzz.second);
    }
  }

  // transform int offsets into DevicePtrs
  void map_temporary_ptrs()
  {
    pair<Vertex,int> current;
    BOOST_FOREACH( current, make_pair(temporary_vertex_memory_offsets.begin(), temporary_vertex_memory_offsets.end()))
    {
      DevicePtr p0 = temporary_memory_ptr + (0*temporary_memory_minimum_size) + current.second;
      DevicePtr p1 = temporary_memory_ptr + (1*temporary_memory_minimum_size) + current.second;
      DevicePtr p2 = temporary_memory_ptr + (2*temporary_memory_minimum_size) + current.second;
      temporary_vertex_memory_ptr[current.first].push_back(p0);
      temporary_vertex_memory_ptr[current.first].push_back(p1);
      temporary_vertex_memory_ptr[current.first].push_back(p2);
    } 
  }

  inline void tmp_spaces_debug()
  {
    cout << "Debug temp spaces:\n----------------------\nfree:\n-------------------\n";
    pair<int,int> current;
    BOOST_FOREACH( current, make_pair(temporary_memory_free.begin(),temporary_memory_free.end()))
    {
      cout << current.first << "\t-\t" << current.second << "\n";
    }
    cout << "allocated:\n--------------\n";
    BOOST_FOREACH( current, make_pair(temporary_memory_allocated.begin(),temporary_memory_allocated.end()))
    {
      cout << current.first << "\t-\t" << current.second << "\n";
    }
    pair<Vertex,int> vert;
    BOOST_FOREACH( vert, make_pair(temporary_vertex_memory_offsets.begin(),temporary_vertex_memory_offsets.end()))
    {
      cout << dbn->neurons_name(vert.first) << ": " << vert.second << "\n";
    }
    cout << endl;
  }
private:

  inline int tmp_alloc( int size )
  {
    int offset_begin, offset_end;
    //cout << "allocating " << size << " bytes of temp space" << endl;
    // look for an available free space
    pair<int,int> current, workable;
    BOOST_FOREACH( current, make_pair(temporary_memory_free.begin(),temporary_memory_free.end()) )
    {
      //cout << "examining " << current.first << " through " << current.second << endl;
      if(current.second - current.first >= size) workable = current;
    }
    list<pair<int,int> >::iterator foundi = 
    find( temporary_memory_free.begin()
        , temporary_memory_free.end()
        , workable
        );
    if( temporary_memory_free.end() == foundi )
    {
      //cout << "no suitable space found, expanding by " << size << "... ";
      // no suitable space was found, we'll need to expand
      offset_begin = temporary_memory_minimum_size;
      temporary_memory_minimum_size = max(temporary_memory_minimum_size, (offset_begin + size));
      //cout << "offset is " << offset_begin << endl;
      //cout << "new min size = " << temporary_memory_minimum_size << endl;
    }
    else
    {
      pair<int,int> temp = *foundi;
      //cout << "Found a suitable free space, using that: " << temp.first<< " through " << temp.second << endl;
      temporary_memory_free.erase(foundi);
      int leftover = (temp.second - temp.first) - size;
      if(leftover > 0)
      {
        //cout << "There's leftover space amounting to " << leftover << endl;
        temp.first += size;
        temporary_memory_free.push_back(temp);
        offset_begin = temporary_memory_free.back().first;
      }
      else
      {
        //cout << "There's no leftover space..." << endl;
        offset_begin = temp.first;
      }
    }
    offset_end = offset_begin + size;
    //cout << "Allocating in the suitable free space " << offset_begin << " to " << offset_end << endl;
    temporary_memory_allocated.push_back(make_pair(offset_begin, offset_end));
    return offset_begin;
    
  }

  inline void tmp_free( int offset )
  {
    //cout << "tmp_free(" << offset << ")" << endl;
    pair<int,int> range_to_free;
    pair<int,int> current;
    bool found = false;
    BOOST_FOREACH( current, make_pair(temporary_memory_allocated.begin(), temporary_memory_allocated.end()) )
    {
      if(current.first == offset) {
        found = true;
        range_to_free = current;
      }
    }
    if(!found) throw(69);
    //cout << "memory range to free = " << range_to_free.first << " through " << range_to_free.second << endl;
    list<pair<int,int> >::iterator foundi = 
    find( temporary_memory_allocated.begin()
        , temporary_memory_allocated.end()
        , range_to_free
        );
    if( foundi == temporary_memory_allocated.end() )
      cout << "trying to free at " << offset << " but there's nothing at that offset which is real bad" << endl;
    else
    {
      pair<int,int> zzz = *foundi;
      //cout << "Freeing temporary memory range: " << zzz.first  << " to " << zzz.second << endl;
      temporary_memory_allocated.erase(foundi);
      temporary_memory_free.push_back( zzz );
    }
  }

  int temporary_memory_minimum_size;
  map<Edge,int>         temporary_edge_memory_offsets;
  map<Vertex,int>       temporary_vertex_memory_offsets;
  map<Edge,DevicePtr>   temporary_edge_memory_ptr;
  map<Vertex,vector<DevicePtr> > temporary_vertex_memory_ptr;
  map<Vertex,vector<DevicePtr> > example_memory_ptr_map;
  list<pair<int,int> > temporary_memory_allocated;
  list<pair<int,int> > temporary_memory_free;
  DevicePtr             temporary_memory_ptr
          ,             example_memory_ptr
          ,             weights_memory_ptr
          ,             biases_memory_ptr
          ,             randoms_ptr_
          ,             random_configs_ptr_
          ;
public:
  int temporary_memory_size();
  int example_memory_size();
  int weights_memory_size();
  int biases_memory_size();
  int randoms_memory_size();
  int random_configs_memory_size();
  int neurons_batch_size( Vertex v );
private:
  void weights_memory_requirements( Edge e, bool triple_buffered );
  DevicePtr map_weights_ptrs( const pair<Edge,pair<int,bool> > &edge_and_layout, DevicePtr p );
  DevicePtr map_biases_ptrs( const pair<Vertex,pair<int,bool> > &vertex_and_layout, DevicePtr p );
  int weight_matrix_size( Edge e );

protected:
  DBN * dbn;
  DBNScheduler * dbn_scheduler;
  int batch_size
    , num_example_batches
    , example_buffer_size
    , temporary_buffer_size
    ;
 
   // the int is the size, the bool is triple-buffering
   map<Edge, pair< int, bool > > weights_memory_layout_map;
   map<Vertex, pair< int, bool > > biases_memory_layout_map;
 
   // for each Edge, a pointer for each of 3 phases
   // (they will all point to the same buffer iff the weights will not be written to)
   map<Edge, vector< DevicePtr > > weights_ptr_map;
 
   // for each Vertex, a pointer for each of 3 phases
   // (they will all point to the same buffer iff the vertex's biases will not be written to)
   map<Vertex, vector< DevicePtr > > biases_ptr_map;
 
}; // end class DBNMemoryMapper

} // end namespace thinkerbell


#endif


