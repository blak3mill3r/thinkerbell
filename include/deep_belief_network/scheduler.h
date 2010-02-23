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
#include <cudamm/cuda.hpp>
#include "tmp.h"
#define WITH_LOGGING
#include "logger.h"

namespace thinkerbell {

// really must descend a class from DeepBeliefNetworkGraph
// class DbnGraph : adjacency_list<>
// give it methods returning iterators appropriate for the algorithm
// vertices (this one exists already no doubt)
// training_edges
// non_training_edges

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
// batch_size random integers between 0 and example_buffer_size to grab random samples
// random gaussian floats, enough for each neuron batch resulting from an activation (so not the first one, the example)
// enough random gaussian floats for batch_size*neurons_size(v) for each vertex other than the input one
// + batch_size*neurons_size(ags_va) + batch_size*neurons_size(ags_vb) for the two AGS steps

class DeepBeliefNetworkScheduler;

class DeepBeliefNetworkMemoryMapper
{
public:
  DeepBeliefNetworkMemoryMapper( DeepBeliefNetwork * dbn_, int batch_size_, int example_buffer_size_ );

  DevicePtr temporaries_ptr( int buffer_index, int temporary_space_index )
    { return temp_ptr[buffer_index][temporary_space_index]; }

  DevicePtr weights_ptr( Edge e, int buffer_index )
    { return weights_ptr_map[e][buffer_index]; }

  DevicePtr examples_ptr( int buffer_index )
    { return (example_memory.ptr() + (sizeof(float) * buffer_index * example_buffer_size)); }

public: // FIXME private

  int temporary_memory_size();

  int example_memory_size();

  void weights_memory_requirements( Edge e, bool triple_buffered );
  DevicePtr map_weights_ptrs( const pair<Edge,pair<int,bool> > &edge_and_layout, DevicePtr p );

  int weights_memory_size();

protected:
  DeepBeliefNetwork * dbn;
  int batch_size;
  int num_examples;
  int example_buffer_size;
  map<Edge, pair< int, bool > > weights_memory_layout_map; // the int is the size, the bool is triple-buffering
  map<Edge, vector< DevicePtr > > weights_ptr_map;
  DevicePtr temp_ptr[3][2]; // 3 x 2, 2 memory spaces in each of 3 phases of triple-buffering
  DeviceMemory weights_memory;
  DeviceMemory example_memory;
  DeviceMemory temporary_memory;
};

class DeepBeliefNetworkScheduler
{
private:
  int batch_size;
  int num_examples;
  DeepBeliefNetwork * dbn;
  volatile bool time_to_stop;
  DeepBeliefNetworkScheduler() {}
public:
  explicit
  DeepBeliefNetworkScheduler( DeepBeliefNetwork * dbn_, int batch_size_, int num_examples_ )
    : batch_size( batch_size_ ),
      dbn( dbn_ ),
      num_examples( num_examples_ ),
      time_to_stop( false )
  {}

  void stop()
  { time_to_stop = true; }

  void operator()()
  {
    cout << "here we go" << endl;
    Cuda context(0);
    Module module_test_kernels("src/test_kernels.cubin");
    Function mmul( module_test_kernels, "mmul" );
    Function mmultb( module_test_kernels, "mmul_transpose_b" );
    //Function madd( module_test_kernels, "madd" );
    vector<Stream *> streams;

    // allocate device memory for the dbn:
    cout << "init dmemory" << endl;
    DeepBeliefNetworkMemoryMapper dmemory( dbn, batch_size, num_examples );
    cout << "temp size = " << dmemory.temporary_memory_size() << endl;

    return;

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
    // FIXME xfer examples into buffers B and C

    while(true)
    for(int i=0; i<2*3; ++i) // 6 is divisible by 2 and 3
    {
      int bufa = ((i+0)%3); // this gives us three phases
      int bufb = ((i+1)%3); // weights will be copied from bufb (because bufb was bufc last iteration)
      int bufc = ((i+2)%3); // buffer C will be written to
      int streami = i%2;

      // transfer examples into A buffer
      // FIXME

      // synch with the end of the execution of the last one using C buffers
      // FIXME add timing
      exec_end[bufc].synchronize();

      // up activation through the whole graph using C buffers
      // for each vertex
      /*for( list<Vertex>::iterator vi = activation_order.begin(); vi != activation_order.end(); vi++ )
      {
        // find the out edge
        graph_traits< DeepBeliefNetworkGraph >::out_edge_iterator out_i, out_end;
        tie(out_i, out_end) = out_edges( *vi, dbn->m_graph );
        // if there's one out edge
        if( (out_end - out_i) == 1 )
        {
          Edge e = *out_i;
          list<Vertex>::iterator nvi = vi; nvi++;  // next vertex iterator and (current) vertex iterator
          mmul.setBlockShape( BLOCK_SIZE, BLOCK_SIZE, 1 );
          mmul.go(
            dmemory.neurons_size( *nvi ) / BLOCK_SIZE,
            batch_size / BLOCK_SIZE,
            *streams[streami],
            dmemory.neurons_ptr( *nvi, bufc ),
            dmemory.neurons_ptr( *vi, bufc ),
            dmemory.weights_ptr( e, bufc ),
            dmemory.neurons_size( *vi ),
            dmemory.neurons_size( *nvi )
          );
        }
      }

      // FIXME AGS steps

      // synch with B-execution done and time it (it should be 0)
      // FIXME add the timing
      exec_end[bufb].synchronize();

      // start weight adjustment kernel on results in C buffer
      // adding the results to the weights in A buffer
      // writing to B buffer
      // FIXME
          */

      if( time_to_stop )
        goto alldone;

    }

    alldone:
    streams[0]->synchronize();
    streams[1]->synchronize();
    delete streams[1];
    delete streams[0];
    
  }

};

}
#endif
