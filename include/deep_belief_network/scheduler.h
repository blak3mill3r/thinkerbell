#ifndef __DEEP_BELIEF_NETWORK_SCHEDULER_H__
#define __DEEP_BELIEF_NETWORK_SCHEDULER_H__

#include <vector>
#include <list>
#include <iostream>
#include <algorithm>
#include "deep_belief_network.h"
#include <boost/thread/condition.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/lambda/algorithm.hpp>
#include <boost/foreach.hpp>
#include <cudamm/cuda.hpp>
#include "tmp.h"
#define WITH_LOGGING
#include "logger.h"

namespace thinkerbell {

using namespace cuda;
using namespace std;
using namespace boost::lambda;
using boost::lambda::_1;
using boost::lambda::bind;

// Manages cuda device memory for the DBN training process
// just enough triple-buffered-temp space for the largest operand
// 3 buffers for examples
// for each vertex
//   3temp: neuron space for temporary neuron values
// for training edges
//   3 buffers for weights
//   3temp: weight space for adjustments
// for all other edges
//   1 buffer for weights
// just enough space is allocated for temporaries to
// accommodate the result of the largest operation
//
// Random numbers
// random gaussian floats, enough for each neuron batch resulting from an activation (so not the first one, the example)
// enough random gaussian floats for batch_size*neurons_size(v) for each vertex other than the input one
// + batch_size*neurons_size(ags_va) + batch_size*neurons_size(ags_vb) for the two AGS steps

class DeepBeliefNetworkScheduler;

class DeepBeliefNetworkMemoryMapper : noncopyable
{
public:
  DeepBeliefNetworkMemoryMapper( DeepBeliefNetworkScheduler * dbn_scheduler_, DeepBeliefNetwork * dbn_, int batch_size_, int num_examples_ );
  void allocate_device_memory(DevicePtr, DevicePtr, DevicePtr);

  DevicePtr weights_ptr( Edge e, int buffer_index )
    { return weights_ptr_map[e][buffer_index]; }

  DevicePtr weights_delta_ptr( Edge e, int buffer_index )
    { return weights_delta_ptr_map[e][buffer_index]; }

  DevicePtr examples_ptr( int buffer_index )
    { return (example_memory_ptr + (sizeof(float) * buffer_index * example_buffer_size)); }

  // the training process concept:
  inline void wait_to_activate() {}
  inline void wait_to_train() {}
  inline void activate_vertex( Vertex v )
  {
  }

  inline void activate_edge( Edge e, bool add_to_result )
  {
    cout << "activate_edge " << e << ":\t";
    cout << "training edge: " << (dbn->is_in_training(e) ? "yes" : "no" ) << ":\t" << endl;
    
  }

  inline void tmp_alloc( Edge e )
  {
    
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

  inline void tmp_free( Edge e ) {}
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
      DevicePtr p = temporary_memory_ptr + current.second;
      temporary_vertex_memory_ptr[current.first] = p;
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
    cout << endl;
  }
private:

  inline int tmp_alloc( int size )
  {
    int offset_begin, offset_end;
    cout << "allocating " << size << " bytes of temp space" << endl;
    // look for an available free space
    pair<int,int> current, workable;
    BOOST_FOREACH( current, make_pair(temporary_memory_free.begin(),temporary_memory_free.end()) )
    {
      cout << "looking... examining " << current.first << " through " << current.second << endl;
      if(current.second - current.first >= size) workable = current;
    }
    list<pair<int,int> >::iterator foundi = 
    find( temporary_memory_free.begin()
        , temporary_memory_free.end()
        , workable
        );
    if( temporary_memory_free.end() == foundi )
    {
      cout << "no suitable space found, expanding by " << size << "... ";
      // no suitable space was found, we'll need to expand
      offset_begin = temporary_memory_free.empty()
                   ? (temporary_memory_allocated.empty() ? 0 : temporary_memory_allocated.back().second)
                   : temporary_memory_free.back().second;
      temporary_memory_minimum_size = max(temporary_memory_minimum_size, (offset_begin + size));
      cout << "offset is " << offset_begin << endl;
      cout << "new min size = " << temporary_memory_minimum_size << endl;
    }
    else
    {
      pair<int,int> temp = *foundi;
      cout << "Found a suitable free space, using that: " << temp.first<< " through " << temp.second << endl;
      temporary_memory_free.erase(foundi);
      int leftover = (temp.second - temp.first) - size;
      if(leftover > 0)
      {
        cout << "There's leftover space amounting to " << leftover << endl;
        temp.first += size;
        temporary_memory_free.push_back(temp);
        offset_begin = temporary_memory_free.back().first;
      }
      else
      {
        cout << "There's no leftover space..." << endl;
        offset_begin = temp.first;
      }
    }
    offset_end = offset_begin + size;
    cout << "Allocating in the suitable free space " << offset_begin << " to " << offset_end << endl;
    temporary_memory_allocated.push_back(make_pair(offset_begin, offset_end));
    return offset_begin;
    
  }

  inline void tmp_free( int offset )
  {
    cout << "tmp_free(" << offset << ")" << endl;
    pair<int,int> range_to_free;
    pair<int,int> current;
    bool found = false;
    BOOST_FOREACH( current, make_pair(temporary_memory_allocated.begin(), temporary_memory_allocated.end()) )
    {
      cout << "Examining free space starting at " << current.first << endl;
      cout << "We're looking for " << offset << endl;
      if(current.first == offset) {
        found = true;
        range_to_free = current;
      }
    }
    if(!found) throw("baderror");
    cout << "memory range to free = " << range_to_free.first << " through " << range_to_free.second << endl;
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
      cout << "Freeing temporary memory range: " << zzz.first  << " to " << zzz.second << endl;
      // FIXME sanity check...
      temporary_memory_allocated.erase(foundi);
      temporary_memory_free.push_back( zzz );
    }
  }

  int temporary_memory_minimum_size;
  map<Edge,int>         temporary_edge_memory_offsets;
  map<Vertex,int>       temporary_vertex_memory_offsets;
  map<Edge,DevicePtr>   temporary_edge_memory_ptr;
  map<Vertex,DevicePtr> temporary_vertex_memory_ptr;
  list<pair<int,int> > temporary_memory_allocated;
  list<pair<int,int> > temporary_memory_free;
  DevicePtr             temporary_memory_ptr
          ,             example_memory_ptr
          ,             weights_memory_ptr
          ;
public:
  int temporary_memory_size();
  int example_memory_size();
  int weights_memory_size();
private:
  void weights_memory_requirements( Edge e, bool triple_buffered );
  DevicePtr map_weights_ptrs( const pair<Edge,pair<int,bool> > &edge_and_layout, DevicePtr p );
  DevicePtr map_weights_delta_ptrs( const pair<Edge,int> &edge_and_size, DevicePtr p );
  int weight_matrix_size( Edge e );
  int neurons_batch_size( Vertex v );

protected:
  DeepBeliefNetwork * dbn;
  DeepBeliefNetworkScheduler * dbn_scheduler;
  int batch_size
    , num_examples
    , example_buffer_size
    , temporary_buffer_size
    ;
 
   // the int is the size, the bool is triple-buffering
   map<Edge, pair< int, bool > > weights_memory_layout_map;
 
   // for each Edge, a pointer for each of 3 phases
   // (they will all point to the same buffer iff the weights will not be written to)
   map<Edge, vector< DevicePtr > > weights_ptr_map
                                 , weights_delta_ptr_map;
 
};

class DeepBeliefNetworkScheduler : noncopyable
{
private:
  int batch_size;
  int num_examples;
  DeepBeliefNetwork * dbn;
  auto_ptr<DeepBeliefNetworkMemoryMapper> dmemory;
  volatile bool time_to_stop;

public:
  explicit
  DeepBeliefNetworkScheduler( DeepBeliefNetwork * dbn_, int batch_size_, int num_examples_ );

  void stop() { time_to_stop = true; }

  void operator()();

  template< class T >
  void training_process( T *impl
                       , int read_weights_buffer_index = 0
                       , int write_buffer_index = 0
                       );

};

}
#endif
